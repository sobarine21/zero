import streamlit as st
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker  # websocket ticker
import pandas as pd
import json
import threading
import time
from datetime import datetime, date
import altair as alt

st.set_page_config(page_title="Kite Connect - Full demo", layout="wide", initial_sidebar_state="expanded")
st.title("Kite Connect (Zerodha) — Full Streamlit demo")

# ---------------------------
# CONFIG / SECRETS
# ---------------------------
try:
    kite_conf = st.secrets["kite"]
    API_KEY = kite_conf.get("api_key")
    API_SECRET = kite_conf.get("api_secret")
    REDIRECT_URI = kite_conf.get("redirect_uri")
except Exception:
    API_KEY = None
    API_SECRET = None
    REDIRECT_URI = None

if not API_KEY or not API_SECRET or not REDIRECT_URI:
    st.error("Missing Kite credentials in Streamlit secrets. Add [kite] api_key, api_secret and redirect_uri.")
    st.stop()

# ---------------------------
# Helper: init unauth client (used for login URL & instruments download)
# ---------------------------
kite_client = KiteConnect(api_key=API_KEY)
login_url = kite_client.login_url()

st.markdown("### Step 1 — Login")
st.write("Click the link below to login to Kite. After login Zerodha will redirect to your configured redirect URI with `request_token` in query params.")
st.markdown(f"[🔗 Open Kite login]({login_url})")

# read request_token from URL (Streamlit >= 1.14)
query_params = st.query_params
request_token = None
if "request_token" in query_params:
    # st.query_params values are lists (or str depending on Streamlit version)
    rt = query_params.get("request_token")
    if isinstance(rt, list):
        request_token = rt[0]
    else:
        request_token = rt

# Exchange request_token for access_token (only once)
if request_token and "kite_access_token" not in st.session_state:
    st.info("Received request_token — exchanging for access token...")
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")
        st.session_state["kite_access_token"] = access_token
        st.session_state["kite_login_response"] = data
        st.success("Access token obtained and stored in session.")
        
        # Display login response in a structured way
        st.subheader("Login Success!")
        col1_login, col2_login = st.columns(2)
        with col1_login:
            st.metric("User ID", data.get("user_id"))
            st.metric("User Name", data.get("user_name"))
            st.metric("Broker", data.get("broker"))
        with col2_login:
            st.metric("Public Token", data.get("public_token"))
            st.metric("Access Token (first 5 chars)", data.get("access_token")[:5] + "****")
            st.metric("Login Time", datetime.fromisoformat(data["login_time"].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S"))
        st.download_button("⬇️ Download full token JSON", json.dumps(data, default=str, indent=2), file_name="kite_token.json", mime="application/json")
    except Exception as e:
        st.error(f"Failed to generate session: {e}")
        st.stop()

# ---------------------------
# Create authenticated kite client if we have access token
# ---------------------------
k = None
if "kite_access_token" in st.session_state:
    access_token = st.session_state["kite_access_token"]
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(access_token)

# ---------------------------
# Utility: instruments dump & lookup
# ---------------------------
@st.cache_data(show_spinner=False, ttl=3600*24) # Cache for 24 hours
def load_instruments(api_key_param, access_token_param, exchange=None):
    """
    Returns pandas.DataFrame of instrument dump.
    If exchange is None, tries to fetch all instruments (may be large).
    Requires api_key and access_token to instantiate KiteConnect.
    """
    try:
        # Re-instantiate KiteConnect inside the cached function
        # This makes the cached function depend on hashable parameters (strings)
        kite_instance_for_cache = KiteConnect(api_key=api_key_param)
        kite_instance_for_cache.set_access_token(access_token_param)

        if exchange:
            inst = kite_instance_for_cache.instruments(exchange)
        else:
            inst = kite_instance_for_cache.instruments()
        df = pd.DataFrame(inst)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        return df
    except Exception as e:
        st.warning(f"Could not fetch instruments for {exchange if exchange else 'all'}: {e}")
        return pd.DataFrame()

def find_instrument_token(df, tradingsymbol, exchange="NSE"):
    """Lookup instrument_token given exchange and tradingsymbol (case-insensitive)."""
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    if not hits.empty:
        return int(hits.iloc[0]["instrument_token"])
    return None

# Custom helper functions for robustness and adhering to API requirements
def get_ltp_price(kite_instance, symbol, exchange="NSE"):
    try:
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        ltp_data = kite_instance.ltp([exchange_symbol])
        return ltp_data
    except Exception as e:
        return {"error": str(e)}

def get_ohlc_quote(kite_instance, symbol, exchange="NSE"):
    try:
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        ohlc_data = kite_instance.ohlc([exchange_symbol])
        return ohlc_data
    except Exception as e:
        return {"error": str(e)}

def get_full_market_quote(kite_instance, symbol, exchange="NSE"):
    try:
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        quote = kite_instance.quote(exchange_symbol)
        return quote
    except Exception as e:
        return {"error": str(e)}

def get_historical(kite_instance, symbol, from_date, to_date, interval="day", exchange="NSE"):
    try:
        inst_df = st.session_state.get("instruments_df", pd.DataFrame())
        token = None
        if not inst_df.empty:
            token = find_instrument_token(inst_df, symbol, exchange)
        
        if token is None:
            st.info(f"Attempting to load instruments for {exchange} to find token for {symbol}...")
            # Call load_instruments with hashable parameters
            token_access = st.session_state.get("kite_access_token")
            if token_access:
                all_instruments = load_instruments(API_KEY, token_access, exchange)
                if not all_instruments.empty:
                    st.session_state["instruments_df"] = all_instruments
                    token = find_instrument_token(all_instruments, symbol, exchange)
            else:
                return {"error": "Access token not available to load instruments."}
        
        if not token:
            return {"error": f"Instrument token not found for {symbol} on {exchange}. Please ensure instruments are loaded or symbol/exchange is correct."}
        
        from_datetime = datetime.combine(from_date, datetime.min.time())
        to_datetime = datetime.combine(to_date, datetime.max.time())

        data = kite_instance.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval=interval)
        return data
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Sidebar quick actions / profile / logout
# ---------------------------
with st.sidebar:
    st.header("Account Information")
    if k:
        try:
            profile = k.profile()
            st.success("Authenticated")
            st.markdown(f"**User:** {profile.get('user_name') or profile.get('user_id')}")
            st.markdown(f"**User ID:** {profile.get('user_id')}")
            st.markdown(f"**Login time:** {datetime.fromisoformat(profile.get('login_time').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Additional profile details
            with st.expander("More Profile Details"):
                st.markdown(f"**Email:** {profile.get('email')}")
                st.markdown(f"**User Type:** {profile.get('user_type')}")
                st.markdown(f"**Broker:** {profile.get('broker')}")

        except Exception:
            st.warning("Authenticated, but profile fetch failed. Check API permissions.")

        if st.button("Logout (clear token)", help="Clear the stored access token and force re-login."):
            st.session_state.pop("kite_access_token", None)
            st.session_state.pop("kite_login_response", None)
            st.session_state.pop("kt_ticker", None)
            st.session_state.pop("kt_thread", None)
            st.session_state.pop("kt_running", False)
            st.session_state.pop("kt_ticks", [])
            st.session_state.pop("instruments_df", pd.DataFrame())
            st.success("Logged out. Please login again.")
            st.experimental_rerun()
    else:
        st.info("Not authenticated yet. Login using the link above.")

# ---------------------------
# Main UI - Tabs for modules
# ---------------------------
tabs = st.tabs(["Dashboard", "Portfolio", "Orders & Trades", "Market Data & Historical", "Websocket (Live Ticks)", "Instruments & Utilities"])
tab_dashboard, tab_portfolio, tab_orders, tab_market, tab_ws, tab_inst = tabs

# ---------------------------
# TAB: DASHBOARD (New Tab)
# ---------------------------
with tab_dashboard:
    st.header("📈 Dashboard Overview")
    if not k:
        st.info("Login first to see dashboard data.")
    else:
        st.markdown("Here's a quick overview of your account information. More sophisticated visualizations can be built based on your specific requirements and data.")

        col_margin, col_holding, col_position = st.columns(3)

        with col_margin:
            st.subheader("Account Margins")
            if st.button("Fetch Margins", key="dash_margins"):
                try:
                    margins = k.margins()
                    equity_margin = margins.get("equity", {})
                    commodity_margin = margins.get("commodity", {})

                    st.metric("Available Cash", f"₹ {equity_margin.get('available', {}).get('cash', 0):,.2f}")
                    st.metric("Used Margin", f"₹ {equity_margin.get('utilised', {}).get('overall', 0):,.2f}")
                    
                    with st.expander("Detailed Margins"):
                        # Convert to DataFrame for better display
                        margin_data = {
                            "Equity": {
                                "Available Cash": equity_margin.get('available', {}).get('cash', 0),
                                "Used Overall": equity_margin.get('utilised', {}).get('overall', 0),
                                "Intraday Pay CNC": equity_margin.get('available', {}).get('intraday_payin_cnc', 0),
                                "Delivery Margin": equity_margin.get('available', {}).get('delivery_margin', 0)
                            },
                            "Commodity": {
                                "Available Cash": commodity_margin.get('available', {}).get('cash', 0),
                                "Used Overall": commodity_margin.get('utilised', {}).get('overall', 0)
                            }
                        }
                        df_margin = pd.DataFrame.from_dict(margin_data, orient='index')
                        st.dataframe(df_margin.applymap(lambda x: f"₹ {x:,.2f}" if isinstance(x, (int, float)) else x), use_container_width=True)
                except Exception as e:
                    st.error(f"Error fetching margins: {e}")
        
        with col_holding:
            st.subheader("Holdings Summary")
            if st.button("Fetch Holdings", key="dash_holdings"):
                try:
                    holdings = k.holdings()
                    if holdings:
                        df_holdings = pd.DataFrame(holdings)
                        total_holdings_value = df_holdings["last_price"].mul(df_holdings["quantity"]).sum()
                        total_pnl = df_holdings["pnl"].sum()
                        st.metric("Total Holdings Value", f"₹ {total_holdings_value:,.2f}")
                        st.metric("Total P&L (Today)", f"₹ {total_pnl:,.2f}")

                        if not df_holdings.empty:
                            df_holdings_display = df_holdings[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl', 'value']]
                            st.dataframe(df_holdings_display.head(5), use_container_width=True) # Show top 5
                            with st.expander("All Holdings"):
                                st.dataframe(df_holdings, use_container_width=True)

                            # Basic pie chart for holdings distribution
                            holdings_chart_data = df_holdings.groupby('tradingsymbol')['value'].sum().reset_index()
                            # Limit to top 10 for readability, group others
                            if len(holdings_chart_data) > 10:
                                holdings_chart_data = holdings_chart_data.nlargest(10, 'value')
                                other_value = df_holdings['value'].sum() - holdings_chart_data['value'].sum()
                                holdings_chart_data = pd.concat([holdings_chart_data, pd.DataFrame([{'tradingsymbol': 'Others', 'value': other_value}])])

                            chart = alt.Chart(holdings_chart_data).mark_arc(outerRadius=120).encode(
                                theta=alt.Theta(field="value", type="quantitative"),
                                color=alt.Color(field="tradingsymbol", type="nominal", title="Symbol"),
                                order=alt.Order(field="value", sort="descending"),
                                tooltip=["tradingsymbol", alt.Tooltip("value", format=",.2f")]
                            ).properties(title="Holdings Value Distribution").interactive()
                            st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("No holdings found.")
                except Exception as e:
                    st.error(f"Error fetching holdings: {e}")

        with col_position:
            st.subheader("Positions Summary")
            if st.button("Fetch Positions", key="dash_positions"):
                try:
                    positions = k.positions()
                    net_positions = positions.get("net", [])
                    day_positions = positions.get("day", [])

                    df_net_positions = pd.DataFrame(net_positions)
                    df_day_positions = pd.DataFrame(day_positions)

                    st.metric("Open Net Positions", len(df_net_positions) if not df_net_positions.empty else 0)
                    st.metric("Open Day Positions", len(df_day_positions) if not df_day_positions.empty else 0)

                    if not df_net_positions.empty:
                        st.subheader("Net Positions")
                        st.dataframe(df_net_positions[['tradingsymbol', 'quantity', 'buy_price', 'sell_price', 'pnl']].head(5), use_container_width=True)
                        with st.expander("All Net Positions"):
                            st.dataframe(df_net_positions, use_container_width=True)

                    if not df_day_positions.empty:
                        st.subheader("Day Positions")
                        st.dataframe(df_day_positions[['tradingsymbol', 'quantity', 'buy_price', 'sell_price', 'pnl']].head(5), use_container_width=True)
                        with st.expander("All Day Positions"):
                            st.dataframe(df_day_positions, use_container_width=True)
                except Exception as e:
                    st.error(f"Error fetching positions: {e}")

# ---------------------------
# TAB: PORTFOLIO
# ---------------------------
with tab_portfolio:
    st.header("📦 Portfolio Details")
    if not k:
        st.info("Login first to fetch portfolio data.")
    else:
        st.markdown("Detailed views of your holdings, positions, and margins.")

        tab_h, tab_p, tab_m = st.tabs(["Holdings", "Positions", "Margins"])

        with tab_h:
            st.subheader("Your Stock Holdings")
            if st.button("Refresh Holdings", key="portfolio_holdings"):
                try:
                    holdings = k.holdings()
                    if holdings:
                        df = pd.DataFrame(holdings)
                        # Clean and display
                        df_display = df[['tradingsymbol', 'isin', 'quantity', 'average_price', 'last_price', 'close_price', 'pnl', 'day_change_percentage', 'value']].copy() # Use .copy() to avoid SettingWithCopyWarning
                        df_display['value'] = df_display['value'].map('₹ {:,.2f}'.format)
                        df_display['average_price'] = df_display['average_price'].map('₹ {:,.2f}'.format)
                        df_display['last_price'] = df_display['last_price'].map('₹ {:,.2f}'.format)
                        df_display['close_price'] = df_display['close_price'].map('₹ {:,.2f}'.format)
                        df_display['pnl'] = df_display['pnl'].map('₹ {:,.2f}'.format)
                        df_display['day_change_percentage'] = df_display['day_change_percentage'].map('{:.2f}%'.format)

                        st.dataframe(df_display, use_container_width=True)

                        st.subheader("Holdings Value Over Time (Simulated, requires historical data for actual tracking)")
                        # This part would require historical portfolio data or a more complex simulation
                        st.info("Actual holdings value over time would require tracking your portfolio daily and fetching historical prices for each holding. This is a placeholder.")
                        
                        # Example: simple bar chart of current P&L per stock
                        pnl_chart_data = df[['tradingsymbol', 'pnl']].copy()
                        pnl_chart_data['color_pnl'] = pnl_chart_data['pnl'].apply(lambda x: 'positive' if x >= 0 else 'negative')
                        
                        pnl_chart = alt.Chart(pnl_chart_data).mark_bar().encode(
                            x=alt.X('pnl', title='Profit/Loss (₹)'),
                            y=alt.Y('tradingsymbol', sort='-x', title='Symbol'),
                            color=alt.Color('color_pnl', scale=alt.Scale(domain=['positive', 'negative'], range=['green', 'red']), legend=None),
                            tooltip=['tradingsymbol', alt.Tooltip('pnl', format='f')]
                        ).properties(title='Current P&L per Holding').interactive()
                        st.altair_chart(pnl_chart, use_container_width=True)

                    else:
                        st.info("No holdings to display.")
                except Exception as e:
                    st.error(f"Error fetching holdings: {e}")

        with tab_p:
            st.subheader("Your Trading Positions")
            if st.button("Refresh Positions", key="portfolio_positions"):
                try:
                    positions = k.positions()
                    
                    st.markdown("#### Net Positions (Overall)")
                    df_net = pd.DataFrame(positions.get("net", []))
                    if not df_net.empty:
                        df_net_display = df_net[['tradingsymbol', 'quantity', 'buy_quantity', 'sell_quantity', 'buy_price', 'sell_price', 'last_price', 'pnl']]
                        st.dataframe(df_net_display, use_container_width=True)
                    else:
                        st.info("No net positions found.")

                    st.markdown("#### Day Positions (Intraday)")
                    df_day = pd.DataFrame(positions.get("day", []))
                    if not df_day.empty:
                        df_day_display = df_day[['tradingsymbol', 'quantity', 'buy_quantity', 'sell_quantity', 'buy_price', 'sell_price', 'last_price', 'pnl']]
                        st.dataframe(df_day_display, use_container_width=True)
                    else:
                        st.info("No day positions found.")
                except Exception as e:
                    st.error(f"Error fetching positions: {e}")
        
        with tab_m:
            st.subheader("Your Margin Details")
            if st.button("Refresh Margins", key="portfolio_margins"):
                try:
                    margins = k.margins()
                    st.write("#### Equity Segment")
                    df_equity_margin = pd.DataFrame([margins.get("equity", {})])
                    if not df_equity_margin.empty:
                        df_equity_margin_display = pd.DataFrame({
                            "Category": ["Available Cash", "Used Margin", "Intraday Pay CNC", "Delivery Margin"],
                            "Amount": [
                                df_equity_margin['available'].apply(lambda x: x.get('cash', 0)).iloc[0],
                                df_equity_margin['utilised'].apply(lambda x: x.get('overall', 0)).iloc[0],
                                df_equity_margin['available'].apply(lambda x: x.get('intraday_payin_cnc', 0)).iloc[0],
                                df_equity_margin['available'].apply(lambda x: x.get('delivery_margin', 0)).iloc[0]
                            ]
                        })
                        df_equity_margin_display['Amount'] = df_equity_margin_display['Amount'].map('₹ {:,.2f}'.format)
                        st.dataframe(df_equity_margin_display.set_index("Category"), use_container_width=True)
                    
                    st.write("#### Commodity Segment")
                    df_commodity_margin = pd.DataFrame([margins.get("commodity", {})])
                    if not df_commodity_margin.empty:
                        df_commodity_margin_display = pd.DataFrame({
                            "Category": ["Available Cash", "Used Margin"],
                            "Amount": [
                                df_commodity_margin['available'].apply(lambda x: x.get('cash', 0)).iloc[0],
                                df_commodity_margin['utilised'].apply(lambda x: x.get('overall', 0)).iloc[0]
                            ]
                        })
                        df_commodity_margin_display['Amount'] = df_commodity_margin_display['Amount'].map('₹ {:,.2f}'.format)
                        st.dataframe(df_commodity_margin_display.set_index("Category"), use_container_width=True)

                    with st.expander("Raw Margin Data"):
                        st.json(margins)

                except Exception as e:
                    st.error(f"Error fetching margins: {e}")

# ---------------------------
# TAB: ORDERS & TRADES
# ---------------------------
with tab_orders:
    st.header("🛍️ Orders & Trades Management")

    if not k:
        st.info("Login first to use orders API.")
    else:
        st.markdown("Place new orders, modify existing ones, or view your order and trade history.")
        
        tab_place, tab_manage, tab_history = st.tabs(["Place New Order", "Manage Orders", "Order & Trade History"])

        with tab_place:
            st.subheader("Place a New Order")
            with st.form("place_order_form", clear_on_submit=False):
                col_place1, col_place2, col_place3 = st.columns(3)
                with col_place1:
                    variety = st.selectbox("Variety", ["regular", "amo", "co", "iceberg"], index=0, help="Order type: Regular, After Market Order, Cover Order, Iceberg.")
                    exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0, help="Trading exchange.")
                    tradingsymbol = st.text_input("Tradingsymbol (e.g. INFY, NIFTY24FEBFUT)", value="INFY", help="Symbol of the instrument.")
                with col_place2:
                    transaction_type = st.radio("Transaction", ["BUY", "SELL"], index=0, horizontal=True, help="Buy or Sell.")
                    order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"], index=0, help="Market, Limit, Stop Loss, Stop Loss Market.")
                    quantity = st.number_input("Quantity", min_value=1, value=1, step=1, help="Number of units to trade.")
                with col_place3:
                    product = st.selectbox("Product", ["CNC", "MIS", "NRML", "CO", "MTF"], index=0, help="Product type: CNC (Cash & Carry), MIS (Margin Intraday Square off), NRML (Normal), CO (Cover Order), MTF (Margin Trading Facility).")
                    validity = st.selectbox("Validity", ["DAY", "IOC", "TTL"], index=0, help="DAY (Good for Day), IOC (Immediate or Cancel), TTL (Time to Live - for iceberg).")
                    price = st.text_input("Price (for LIMIT/SL)", value="", help="Specify if order type is LIMIT or SL. Leave empty for MARKET orders.")
                    trigger_price = st.text_input("Trigger Price (for SL/SL-M)", value="", help="Specify if order type is SL or SL-M.")
                tag = st.text_input("Tag (optional, max 20 chars)", value="", help="Custom tag to identify your order.")
                
                submit_place = st.form_submit_button("Place Order")

                if submit_place:
                    try:
                        params = dict(
                            variety=variety,
                            exchange=exchange,
                            tradingsymbol=tradingsymbol,
                            transaction_type=transaction_type,
                            order_type=order_type,
                            quantity=int(quantity),
                            product=product,
                            validity=validity,
                        )
                        if price:
                            params["price"] = float(price)
                        if trigger_price:
                            params["trigger_price"] = float(trigger_price)
                        if tag:
                            params["tag"] = tag[:20]

                        resp = k.place_order(**params)
                        st.success(f"Order placed successfully!")
                        st.markdown("#### Order Response:")
                        st.json(resp)
                    except Exception as e:
                        st.error(f"Place order failed: {e}")

        with tab_manage:
            st.subheader("Manage Live Orders")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                if st.button("Fetch All Live Orders", key="fetch_live_orders"):
                    try:
                        orders = k.orders()
                        if orders:
                            df_orders = pd.DataFrame(orders)
                            df_orders_display = df_orders[['order_id', 'tradingsymbol', 'exchange', 'transaction_type', 'order_type', 'quantity', 'filled_quantity', 'pending_quantity', 'price', 'status', 'status_message', 'order_timestamp']]
                            st.dataframe(df_orders_display, use_container_width=True)
                        else:
                            st.info("No live orders found.")
                    except Exception as e:
                        st.error(f"Error fetching orders: {e}")
            
            st.markdown("---")
            st.subheader("Modify / Cancel Order")
            mod_cancel_order_id = st.text_input("Enter Order ID to Modify/Cancel:", value="", key="mod_cancel_order_id_input")

            if mod_cancel_order_id:
                col_mod_cancel1, col_mod_cancel2 = st.columns(2)
                with col_mod_cancel1:
                    st.markdown("##### Modify Order")
                    mod_variety = st.selectbox("Variety for Modify", ["regular", "co", "amo", "iceberg"], index=0, key="mod_variety")
                    new_price = st.text_input("New Price (optional)", value="", key="mod_new_price")
                    new_qty = st.number_input("New Quantity (optional)", min_value=0, value=0, step=1, key="mod_new_qty")
                    if st.button("Modify Order", key="submit_modify_order"):
                        if not mod_cancel_order_id:
                            st.warning("Please provide an Order ID to modify.")
                        else:
                            try:
                                modify_args = {"variety": mod_variety}
                                if new_price:
                                    modify_args["price"] = float(new_price)
                                if new_qty > 0:
                                    modify_args["quantity"] = int(new_qty)
                                
                                if not new_price and new_qty == 0:
                                    st.warning("Please provide a new price or quantity to modify.")
                                else:
                                    res = k.modify_order(order_id=mod_cancel_order_id, **modify_args)
                                    st.success(f"Order modified: {res.get('order_id', 'Unknown')}")
                                    st.json(res)
                            except Exception as e:
                                st.error(f"Modify failed: {e}")

                with col_mod_cancel2:
                    st.markdown("##### Cancel Order")
                    cancel_variety = st.selectbox("Variety for Cancel", ["regular", "co", "amo", "iceberg"], index=0, key="cancel_variety")
                    if st.button("Cancel Order", key="submit_cancel_order"):
                        if not mod_cancel_order_id:
                            st.warning("Please provide an Order ID to cancel.")
                        else:
                            try:
                                res = k.cancel_order(variety=cancel_variety, order_id=mod_cancel_order_id)
                                st.success(f"Order cancelled: {res.get('order_id', 'Unknown')}")
                                st.json(res)
                            except Exception as e:
                                st.error(f"Cancel failed: {e}")
            else:
                st.info("Enter an Order ID above to enable modify/cancel options.")

        with tab_history:
            st.subheader("Order & Trade History")
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                if st.button("Fetch All Orders (Today)", key="fetch_today_orders"):
                    try:
                        orders = k.orders()
                        if orders:
                            df_orders_all = pd.DataFrame(orders)
                            st.dataframe(df_orders_all[['order_id', 'tradingsymbol', 'transaction_type', 'order_type', 'quantity', 'price', 'status', 'status_message', 'order_timestamp']], use_container_width=True)
                        else:
                            st.info("No orders found for today.")
                    except Exception as e:
                        st.error(f"Error fetching all orders: {e}")
            with col_h2:
                if st.button("Fetch All Trades (Today)", key="fetch_today_trades"):
                    try:
                        trades = k.trades()
                        if trades:
                            df_trades = pd.DataFrame(trades)
                            st.dataframe(df_trades[['trade_id', 'order_id', 'tradingsymbol', 'exchange', 'transaction_type', 'quantity', 'average_price', 'trade_timestamp']], use_container_width=True)
                        else:
                            st.info("No trades found for today.")
                    except Exception as e:
                        st.error(f"Error fetching trades: {e}")
            
            st.markdown("---")
            st.subheader("Specific Order History")
            order_id_for_history = st.text_input("Enter Order ID for detailed history:", value="", key="order_history_id")
            if st.button("Get Order History", key="get_specific_order_history"):
                if not order_id_for_history:
                    st.warning("Please provide an order_id.")
                else:
                    try:
                        history = k.order_history(order_id_for_history)
                        if history:
                            df_history = pd.DataFrame(history)
                            st.dataframe(df_history, use_container_width=True)
                        else:
                            st.info(f"No history found for Order ID: {order_id_for_history}")
                    except Exception as e:
                        st.error(f"Get order history failed: {e}")


# ---------------------------
# TAB: MARKET & HISTORICAL
# ---------------------------
with tab_market:
    st.header("📊 Market Data & Historical Analysis")

    if not k:
        st.info("Login first to fetch market data (quotes/historical).")
    else:
        st.markdown("Retrieve real-time market quotes and historical candlestick data for analysis.")

        tab_quotes, tab_historical = st.tabs(["Market Quotes", "Historical Data"])

        with tab_quotes:
            st.subheader("Market Data Snapshot")
            col_quote1, col_quote2 = st.columns(2)
            with col_quote1:
                q_exchange = st.selectbox("Exchange for market data", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0, key="market_exchange_tab")
                q_symbol = st.text_input("Tradingsymbol (e.g., INFY, NIFTY24FEBFUT)", value="INFY", key="market_symbol_tab")
            with col_quote2:
                market_data_type = st.radio("Choose data type:", 
                                            ("LTP (Last Traded Price)", "OHLC + LTP", "Full Market Quote (OHLC, Depth, OI)"), 
                                            index=0, key="market_data_type_radio_tab")

            if st.button("Get Market Data", key="get_market_data_btn"):
                market_data_response = {}
                if market_data_type == "LTP (Last Traded Price)":
                    market_data_response = get_ltp_price(k, q_symbol, q_exchange)
                    if not market_data_response.get("error"):
                        st.subheader(f"LTP for {q_symbol} ({q_exchange})")
                        key = f"{q_exchange.upper()}:{q_symbol.upper()}"
                        if key in market_data_response:
                            st.metric("Last Traded Price", f"₹ {market_data_response[key]['last_price']:,.2f}")
                        else:
                            st.warning("LTP data not found for the given symbol.")
                    
                elif market_data_type == "OHLC + LTP":
                    market_data_response = get_ohlc_quote(k, q_symbol, q_exchange)
                    if not market_data_response.get("error"):
                        st.subheader(f"OHLC + LTP for {q_symbol} ({q_exchange})")
                        key = f"{q_exchange.upper()}:{q_symbol.upper()}"
                        if key in market_data_response:
                            data = market_data_response[key]['ohlc']
                            ltp = market_data_response[key]['last_price']
                            col_ohlc1, col_ohlc2, col_ohlc3 = st.columns(3)
                            col_ohlc1.metric("Open", f"₹ {data['open']:,.2f}")
                            col_ohlc2.metric("High", f"₹ {data['high']:,.2f}")
                            col_ohlc3.metric("Low", f"₹ {data['low']:,.2f}")
                            col_ohlc1.metric("Close", f"₹ {data['close']:,.2f}")
                            col_ohlc2.metric("Last Traded Price", f"₹ {ltp:,.2f}")
                        else:
                            st.warning("OHLC data not found for the given symbol.")
                else: # Full Market Quote
                    market_data_response = get_full_market_quote(k, q_symbol, q_exchange)
                    if not market_data_response.get("error"):
                        st.subheader(f"Full Market Quote for {q_symbol} ({q_exchange})")
                        key = f"{q_exchange.upper()}:{q_symbol.upper()}"
                        if key in market_data_response:
                            quote_data = market_data_response[key]
                            
                            st.markdown("#### Price Information")
                            col_f1, col_f2, col_f3 = st.columns(3)
                            col_f1.metric("Last Price", f"₹ {quote_data.get('last_price', 0):,.2f}")
                            col_f2.metric("Open", f"₹ {quote_data.get('ohlc', {}).get('open', 0):,.2f}")
                            col_f3.metric("High", f"₹ {quote_data.get('ohlc', {}).get('high', 0):,.2f}")
                            col_f1.metric("Low", f"₹ {quote_data.get('ohlc', {}).get('low', 0):,.2f}")
                            col_f2.metric("Close", f"₹ {quote_data.get('ohlc', {}).get('close', 0):,.2f}")
                            col_f3.metric("Avg. Price", f"₹ {quote_data.get('average_price', 0):,.2f}")
                            
                            st.markdown("#### Market Depth (Top 5)")
                            depth_data = quote_data.get('depth', {})
                            buys = depth_data.get('buy', [])
                            sells = depth_data.get('sell', [])
                            
                            col_depth1, col_depth2 = st.columns(2)
                            with col_depth1:
                                st.markdown("##### Buy Orders")
                                if buys:
                                    df_buys = pd.DataFrame(buys).rename(columns={'quantity': 'Qty', 'price': 'Price', 'orders': 'Orders'})
                                    st.dataframe(df_buys, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No buy depth available.")
                            with col_depth2:
                                st.markdown("##### Sell Orders")
                                if sells:
                                    df_sells = pd.DataFrame(sells).rename(columns={'quantity': 'Qty', 'price': 'Price', 'orders': 'Orders'})
                                    st.dataframe(df_sells, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No sell depth available.")

                            st.markdown("#### Other Info")
                            col_other1, col_other2, col_other3 = st.columns(3)
                            col_other1.metric("Volume", f"{quote_data.get('volume', 0):,.0f}")
                            col_other2.metric("Total Buy Qty", f"{quote_data.get('total_buy_quantity', 0):,.0f}")
                            col_other3.metric("Total Sell Qty", f"{quote_data.get('total_sell_quantity', 0):,.0f}")
                            if 'oi' in quote_data:
                                col_other1.metric("Open Interest", f"{quote_data.get('oi', 0):,.0f}")
                                col_other2.metric("OI Day High", f"{quote_data.get('oi_day_high', 0):,.0f}")
                                col_other3.metric("OI Day Low", f"{quote_data.get('oi_day_low', 0):,.0f}")

                            with st.expander("Raw Full Market Quote"):
                                st.json(quote_data)
                        else:
                            st.warning("Full market quote data not found for the given symbol.")

                if "error" in market_data_response:
                    st.error(f"Market data fetch failed: {market_data_response['error']}")
                    if "Insufficient permission" in market_data_response['error']:
                        st.warning("For 'Full Market Quote' (especially depth/OI), you might need a paid subscription to the Kite Connect API. Try 'LTP' or 'OHLC + LTP' if you encounter permission errors.")

        with tab_historical:
            st.subheader("Historical Candlestick Data")
            st.info("To fetch historical data, you might need to load the instrument dump first to get the correct instrument token.")
            
            # Load instruments (cached)
            with st.expander("Step 1: Load Instrument Dump (if token lookup fails)"):
                exchange_for_dump = st.selectbox("Select Exchange to load instruments for lookup:", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0, key="hist_inst_exchange")
                if st.button("Load Instrument Dump for Lookup", key="load_inst_dump_btn"):
                    # Call load_instruments with hashable parameters
                    token_access = st.session_state.get("kite_access_token")
                    if token_access:
                        inst_df = load_instruments(API_KEY, token_access, exchange_for_dump)
                        st.session_state["instruments_df"] = inst_df
                        if not inst_df.empty:
                            st.success(f"Loaded {len(inst_df)} instruments for {exchange_for_dump}.")
                            st.dataframe(inst_df.head(), use_container_width=True) # Show a preview
                        else:
                            st.warning(f"Could not load instruments for {exchange_for_dump}.")
                    else:
                        st.error("Access token not available. Please log in.")

            col_hist1, col_hist2, col_hist3 = st.columns(3)
            with col_hist1:
                hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "MCX"], index=0, key="hist_ex_tab")
                hist_symbol = st.text_input("Tradingsymbol (e.g. INFY)", value="INFY", key="hist_sym_tab")
            with col_hist2:
                from_date = st.date_input("From Date", value=date.today() - pd.DateOffset(days=30), key="from_dt_tab")
                to_date = st.date_input("To Date", value=date.today(), key="to_dt_tab")
            with col_hist3:
                interval = st.selectbox("Interval", ["minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day", "week", "month"], index=7, key="hist_interval_tab")

            if st.button("Fetch Historical Data", key="fetch_hist_data_btn"):
                hist_data = get_historical(k, hist_symbol, from_date, to_date, interval, hist_exchange)
                
                if "error" in hist_data:
                    st.error(f"Historical fetch failed: {hist_data['error']}")
                    if "Insufficient permission" in hist_data['error']:
                        st.warning("This error often indicates that your Zerodha API key does not have an active subscription for historical data. Please check your Kite Connect developer console for subscription status.")
                else:
                    df = pd.DataFrame(hist_data)
                    if not df.empty:
                        df["date"] = pd.to_datetime(df["date"])
                        st.dataframe(df, use_container_width=True)

                        # Candlestick Chart
                        st.subheader("Candlestick Chart")
                        base = alt.Chart(df).encode(x=alt.X('date:T', axis=alt.Axis(title='Date', format='%Y-%m-%d %H:%M')))

                        # Candlestick body
                        candlesticks = base.mark_rule().encode(
                            y=alt.Y('low', title='Price'),
                            y2='high',
                            color=alt.condition(
                                alt.datum.open < alt.datum.close,
                                alt.value('green'),
                                alt.value('red')
                            ),
                            tooltip=[
                                alt.Tooltip('date:T', title='Date/Time'),
                                'open', 'high', 'low', 'close', 'volume'
                            ]
                        )
                        # Candlestick wicks
                        bars = base.mark_bar().encode(
                            y='open',
                            y2='close',
                            color=alt.condition(
                                alt.datum.open < alt.datum.close,
                                alt.value('green'),
                                alt.value('red')
                            ),
                            tooltip=[
                                alt.Tooltip('date:T', title='Date/Time'),
                                'open', 'high', 'low', 'close', 'volume'
                            ]
                        )

                        # Volume bar chart
                        volume = alt.Chart(df).mark_bar().encode(
                            x='date:T',
                            y=alt.Y('volume', title='Volume'),
                            color=alt.condition(
                                alt.datum.open < alt.datum.close,
                                alt.value('lightgreen'),
                                alt.value('salmon')
                            ),
                            tooltip=[
                                alt.Tooltip('date:T', title='Date/Time'),
                                'volume'
                            ]
                        ).properties(height=100)

                        chart = alt.vconcat(candlesticks + bars, volume).resolve_scale(x='shared').properties(
                            title=f"Candlestick Chart for {hist_symbol} ({hist_exchange}, {interval} interval)"
                        )
                        st.altair_chart(chart, use_container_width=True)

                    else:
                        st.info("No historical data returned for the selected criteria.")

# ---------------------------
# TAB: WEBSOCKET (Ticker)
# ---------------------------
# Global variable to hold the placeholder for live tick updates
tick_display_placeholder = st.empty()

def update_live_ticks_ui():
    """Function to be called by the background thread to update Streamlit UI."""
    with tick_display_placeholder.container():
        st.subheader("Latest Ticks (most recent 50)")
        ticks = st.session_state.get("kt_ticks", [])
        if ticks:
            df_ticks = pd.json_normalize(ticks[-50:][::-1]) # Last 50, reversed for latest first
            df_ticks_display = df_ticks.head(10).astype(str) # Display first 10 for brevity, convert to string to avoid complex types
            st.dataframe(df_ticks_display, use_container_width=True, height=300)
            
            # Simple line chart for LTP (if available)
            if 'last_price' in df_ticks.columns and 'instrument_token' in df_ticks.columns:
                df_ticks['last_price'] = pd.to_numeric(df_ticks['last_price'], errors='coerce')
                df_ticks['time'] = pd.to_datetime(df_ticks['_ts']) # Use internal timestamp
                
                # Filter out NaNs and take last few to avoid overcrowding
                chart_data = df_ticks[['time', 'last_price', 'instrument_token']].dropna().tail(100)
                if not chart_data.empty:
                    st.subheader("Live LTP Trend")
                    chart = alt.Chart(chart_data).mark_line().encode(
                        x=alt.X('time:T', title='Time', axis=alt.Axis(format='%H:%M:%S')),
                        y=alt.Y('last_price:Q', title='LTP'),
                        color='instrument_token:N',
                        tooltip=['time:T', 'instrument_token', alt.Tooltip('last_price', format='f')]
                    ).properties(
                        height=250
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No ticks yet. Start ticker and/or subscribe tokens.")
    time.sleep(1) # Refresh UI every second


def ticker_thread_target(kt_instance):
    """Target function for the background thread to run KiteTicker."""
    try:
        # It's important that connect() is called in the thread that's going to run the WebSocket loop.
        # threaded=True for KiteTicker means its internal loop runs in a daemon thread.
        # The external thread (this one) just needs to keep running to prevent Python from exiting.
        # Check KiteTicker documentation for exact behavior; sometimes connect() is blocking if not threaded.
        kt_instance.connect(threaded=True, disable_ssl_certs=True) 
        
        # Keep this thread alive while the ticker is running (by monitoring the session state flag)
        while st.session_state["kt_running"]:
            time.sleep(1) 
    except Exception as e:
        st.session_state["kt_ticks"].append({"event": "ticker_error", "error": str(e), "time": datetime.utcnow().isoformat()})
        st.session_state["kt_running"] = False
    finally:
        # Ensure ticker disconnects if the loop breaks
        if kt_instance and kt_instance.is_connected():
            kt_instance.disconnect()


with tab_ws:
    st.header("⚡ WebSocket Streaming — KiteTicker")
    st.write("Receive live market data ticks. This feature uses background threads to maintain the WebSocket connection. Click 'Start Ticker', then 'Subscribe' to tokens.")

    if not k:
        st.info("Login first to start the websocket ticker.")
    else:
        # session state for ticker & ticks
        if "kt_ticker" not in st.session_state:
            st.session_state["kt_ticker"] = None
        if "kt_thread" not in st.session_state:
            st.session_state["kt_thread"] = None
        if "kt_running" not in st.session_state:
            st.session_state["kt_running"] = False
        if "kt_ticks" not in st.session_state:
            st.session_state["kt_ticks"] = []
        
        st.info("Ensure the instrument dump is loaded in 'Instruments & Utilities' tab for token lookup, or manually enter tokens.")

        # Controls for starting/stopping the ticker
        col_ws_start, col_ws_stop = st.columns(2)
        with col_ws_start:
            if st.button("Start Ticker", type="primary") and not st.session_state["kt_running"]:
                try:
                    access_token = st.session_state["kite_access_token"]
                    user_id = st.session_state["kite_login_response"].get("user_id")
                    
                    try:
                        kt = KiteTicker(user_id, access_token, API_KEY)
                    except Exception:
                        kt = KiteTicker(API_KEY, access_token)

                    st.session_state["kt_ticker"] = kt
                    st.session_state["kt_running"] = True
                    st.session_state["kt_ticks"] = [] # Clear previous ticks

                    def on_connect(ws, response):
                        st.session_state["kt_ticks"].append({"event": "connected", "time": datetime.utcnow().isoformat()})
                        st.success("WebSocket Connected!")
                        # Subscribe to initial tokens if any (handled below with explicit subscribe button)

                    def on_ticks(ws, ticks):
                        for t in ticks:
                            t["_ts"] = datetime.utcnow().isoformat()
                            st.session_state["kt_ticks"].append(t)
                        # Trim ticks to avoid excessive memory usage
                        if len(st.session_state["kt_ticks"]) > 500:
                            st.session_state["kt_ticks"] = st.session_state["kt_ticks"][-500:]

                    def on_close(ws, code, reason):
                        st.session_state["kt_ticks"].append({"event": "closed", "code": code, "reason": reason, "time": datetime.utcnow().isoformat()})
                        st.session_state["kt_running"] = False
                        st.warning(f"WebSocket Closed: Code={code}, Reason={reason}")

                    def on_error(ws, code, reason):
                        st.session_state["kt_ticks"].append({"event": "error", "code": code, "reason": reason, "time": datetime.utcnow().isoformat()})
                        st.error(f"WebSocket Error: Code={code}, Reason={reason}")

                    kt.on_connect = on_connect
                    kt.on_ticks = on_ticks
                    kt.on_close = on_close
                    kt.on_error = on_error # Add error handler

                    th = threading.Thread(target=ticker_thread_target, args=(kt,), daemon=True)
                    st.session_state["kt_thread"] = th
                    th.start()
                    st.success("Ticker thread initiated. Connecting...")
                except Exception as e:
                    st.error(f"Failed to start ticker: {e}")
            elif st.session_state["kt_running"]:
                st.info("Ticker is already running.")

        with col_ws_stop:
            if st.button("Stop Ticker", type="secondary") and st.session_state.get("kt_running"):
                try:
                    kt = st.session_state.get("kt_ticker")
                    if kt:
                        kt.disconnect()
                    st.session_state["kt_running"] = False
                    if st.session_state["kt_thread"] and st.session_state["kt_thread"].is_alive():
                        # Give it a moment to properly shut down, then forcefully stop if needed
                        st.session_state["kt_thread"].join(timeout=2) 
                    st.success("Ticker stopped.")
                    st.session_state["kt_ticker"] = None
                    st.session_state["kt_thread"] = None
                except Exception as e:
                    st.error(f"Failed to stop ticker: {e}")
            elif not st.session_state.get("kt_running"):
                st.info("Ticker is not running.")


        # Subscription controls
        st.markdown("---")
        st.subheader("Subscribe / Unsubscribe Instruments")
        instrument_lookup_col, tokens_input_col = st.columns([1, 2])
        
        with instrument_lookup_col:
            st.markdown("##### Lookup Token")
            lookup_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0, key="ws_lookup_exchange")
            lookup_symbol = st.text_input("Symbol", value="INFY", key="ws_lookup_symbol")
            if st.button("Get Token", key="ws_get_token_btn"):
                inst_df = st.session_state.get("instruments_df", pd.DataFrame())
                token = find_instrument_token(inst_df, lookup_symbol, lookup_exchange)
                if token:
                    st.success(f"Token for {lookup_symbol}: `{token}`")
                    st.session_state['last_looked_up_token'] = str(token) # Store for easy copy-paste
                else:
                    st.warning(f"Token not found for {lookup_symbol}. Load instruments in 'Instruments & Utilities' tab first.")
        
        with tokens_input_col:
            st.markdown("##### Subscribe/Unsubscribe")
            symbol_for_ws = st.text_input("Instrument token(s) comma separated (e.g. 738561,3409)", 
                                        value=st.session_state.get('last_looked_up_token', ''), key="ws_symbol_input")
            
            mode_option = st.radio("Tick Mode", ["LTP (MODE_LTP)", "Quote (MODE_QUOTE)", "Full (MODE_FULL)"], index=2, horizontal=True)

            col_sub, col_unsub = st.columns(2)
            with col_sub:
                if st.button("Subscribe", type="primary") and st.session_state.get("kt_running"):
                    try:
                        tokens = [int(x.strip()) for x in symbol_for_ws.split(",") if x.strip()]
                        if tokens:
                            kt = st.session_state["kt_ticker"]
                            kt.subscribe(tokens)
                            
                            mode_map = {
                                "LTP (MODE_LTP)": kt.MODE_LTP,
                                "Quote (MODE_QUOTE)": kt.MODE_QUOTE,
                                "Full (MODE_FULL)": kt.MODE_FULL
                            }
                            selected_mode = mode_map.get(mode_option, kt.MODE_FULL)
                            kt.set_mode(selected_mode, tokens)
                            st.success(f"Subscribed to tokens: {tokens} in {mode_option} mode.")
                        else:
                            st.warning("Please enter instrument tokens to subscribe.")
                    except Exception as e:
                        st.error(f"Failed to subscribe: {e}")
                elif not st.session_state.get("kt_running"):
                    st.warning("Please start the ticker first.")

            with col_unsub:
                if st.button("Unsubscribe", type="secondary") and st.session_state.get("kt_running"):
                    try:
                        tokens = [int(x.strip()) for x in symbol_for_ws.split(",") if x.strip()]
                        if tokens:
                            kt = st.session_state["kt_ticker"]
                            kt.unsubscribe(tokens)
                            st.success(f"Unsubscribed from tokens: {tokens}.")
                        else:
                            st.warning("Please enter instrument tokens to unsubscribe.")
                    except Exception as e:
                        st.error(f"Failed to unsubscribe: {e}")
                elif not st.session_state.get("kt_running"):
                    st.warning("Please start the ticker first.")

        st.markdown("---")
        # Placeholder for live tick updates
        if st.session_state.get("kt_running"):
            update_live_ticks_ui() # Initial call
            # Streamlit reruns the script on widget interactions, so this will be called
            # repeatedly while the ticker is running. No need for explicit thread for UI updates.
        else:
            st.info("Start the ticker and subscribe to instruments to see live ticks here.")

# ---------------------------
# TAB: INSTRUMENTS DUMP & UTILS
# ---------------------------
with tab_inst:
    st.header("🔧 Instruments Dump & Utilities")
    st.markdown("Load and search through the complete list of tradable instruments on various exchanges.")

    inst_exchange = st.selectbox("Select Exchange to load instruments:", ["NSE", "BSE", "NFO", "CDS", "MCX", "BCD"], index=0, key="inst_dump_exchange")
    
    col_inst_load, col_inst_download = st.columns(2)
    with col_inst_load:
        if st.button(f"Load Instruments for {inst_exchange} (Cached)", key="load_inst_btn"):
            try:
                # Call load_instruments with hashable parameters
                token_access = st.session_state.get("kite_access_token")
                if token_access:
                    df = load_instruments(API_KEY, token_access, inst_exchange)
                    st.session_state["instruments_df"] = df
                    if not df.empty:
                        st.success(f"Loaded {len(df)} instruments for {inst_exchange}.")
                    else:
                        st.warning(f"Could not load instruments for {inst_exchange}. Check connection or exchange validity.")
                else:
                    st.error("Access token not available. Please log in first.")
            except Exception as e:
                st.error(f"Load instruments failed: {e}")
    
    df = st.session_state.get("instruments_df", pd.DataFrame())
    
    with col_inst_download:
        if not df.empty:
            st.download_button(
                label="Download Loaded Instruments as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"{inst_exchange}_instruments.csv",
                mime="text/csv",
            )

    if not df.empty:
        st.markdown("---")
        st.subheader("Search & Lookup Instruments")
        col_inst_search, col_inst_token = st.columns(2)
        with col_inst_search:
            search_query = st.text_input("Search instruments by Symbol, Name or ISIN:", value="", key="inst_search_query")
            
            if search_query:
                # Case-insensitive search across relevant columns
                df_filtered = df[
                    df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)
                ]
                st.dataframe(df_filtered.head(100), use_container_width=True)
                if len(df_filtered) > 100:
                    st.info(f"Showing first 100 of {len(df_filtered)} matching instruments.")
            else:
                st.markdown("Preview instruments (first 50 rows)")
                st.dataframe(df.head(50), use_container_width=True)
        
        with col_inst_token:
            st.subheader("Find Instrument Token")
            sy = st.text_input("Tradingsymbol (e.g. INFY)", value="INFY", key="inst_search_sym_lookup")
            ex = st.selectbox("Exchange for lookup", ["NSE", "BSE", "NFO", "CDS", "MCX", "BCD"], index=0, key="inst_lookup_exchange")
            
            if st.button("Find Instrument Token", key="find_token_btn"):
                token = find_instrument_token(df, sy, ex)
                if token:
                    st.success(f"Found instrument_token for `{sy}` on `{ex}`: `{token}`")
                else:
                    st.warning(f"Instrument token not found for `{sy}` on `{ex}`. Ensure the symbol and exchange are correct, and the instrument dump for this exchange has been loaded.")
    else:
        st.info("No instruments loaded yet. Click 'Load Instruments' to fetch them.")
