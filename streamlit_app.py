import streamlit as st
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker  # websocket ticker
import pandas as pd
import json
import threading
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import ta # Technical Analysis library
import yfinance as yf # For fetching benchmark data for comparison

st.set_page_config(page_title="Kite Connect - Advanced Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("Kite Connect (Zerodha) — Advanced Financial Analysis")
st.markdown("A comprehensive platform for fetching market data, performing ML-driven analysis, risk assessment, and live data streaming.")

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
    st.error("Missing Kite credentials in Streamlit secrets. Add [kite] api_key, api_secret and redirect_uri in `.streamlit/secrets.toml`.")
    st.info("Example `secrets.toml`:\n```toml\n[kite]\napi_key=\"YOUR_API_KEY\"\napi_secret=\"YOUR_KITE_SECRET\"\nredirect_uri=\"http://localhost:8501\"\n```")
    st.stop()

# ---------------------------
# Helper: init unauth client (used for login URL)
# ---------------------------
kite_client = KiteConnect(api_key=API_KEY)
login_url = kite_client.login_url()

st.sidebar.markdown("### Step 1 — Login to Kite")
st.sidebar.write("Click the link below to login to Kite. After login Zerodha will redirect to your configured redirect URI with `request_token` in query params.")
st.sidebar.markdown(f"[🔗 Open Kite login]({login_url})")

# read request_token from URL (Streamlit >= 1.14)
query_params = st.query_params
request_token = None
if "request_token" in query_params:
    rt = query_params.get("request_token")
    if isinstance(rt, list):
        request_token = rt[0]
    else:
        request_token = rt

# Exchange request_token for access_token (only once)
if request_token and "kite_access_token" not in st.session_state:
    st.sidebar.info("Received request_token — exchanging for access token...")
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")
        st.session_state["kite_access_token"] = access_token
        st.session_state["kite_login_response"] = data
        st.sidebar.success("Access token obtained and stored in session.")
        st.sidebar.download_button("⬇️ Download token JSON", json.dumps(data, default=str), file_name="kite_token.json")
        st.rerun() # FIX: Changed to st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to generate session: {e}")
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
# Utility: instruments lookup (kept as it's essential for fetching historical data)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_instruments(_kite_instance, exchange=None):
    """
    Returns pandas.DataFrame of instrument data.
    """
    try:
        if exchange:
            inst = _kite_instance.instruments(exchange)
        else:
            inst = _kite_instance.instruments()
        df = pd.DataFrame(inst)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        return df
    except Exception as e:
        st.warning(f"Could not fetch instruments: {e}")
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
        token = find_instrument_token(inst_df, symbol, exchange)
        
        if not token:
            st.info(f"Instrument token for {symbol} on {exchange} not found in cache. Attempting to fetch...")
            all_instruments = load_instruments(kite_instance, exchange) 
            if not all_instruments.empty:
                st.session_state["instruments_df"] = all_instruments
                token = find_instrument_token(all_instruments, symbol, exchange)
            
            if not token:
                return {"error": f"Instrument token not found for {symbol} on {exchange}. Please ensure instruments are loaded or symbol/exchange is correct."}
        
        from_datetime = datetime.combine(from_date, datetime.min.time())
        to_datetime = datetime.combine(to_date, datetime.max.time())

        data = kite_instance.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval=interval)
        return data
    except Exception as e:
        return {"error": str(e)}

# Function to add technical indicators using 'ta' library for robustness
def add_indicators(df, sma_short=5, sma_long=20, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_window=20, bb_std_dev=2):
    if df.empty:
        return df
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['close'], inplace=True)
    if df.empty:
        return pd.DataFrame()

    df['SMA_Short'] = ta.trend.sma_indicator(df['close'], window=sma_short)
    df['SMA_Long'] = ta.trend.sma_indicator(df['close'], window=sma_long)
    df['RSI'] = ta.momentum.rsi(df['close'], window=rsi_window)
    
    macd_obj = ta.trend.MACD(df['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df['MACD'] = macd_obj.macd()
    df['MACD_signal'] = macd_obj.macd_signal()
    # Adding MACD Histogram for visualization
    df['MACD_hist'] = macd_obj.macd_diff() 
    
    bollinger = ta.volatility.BollingerBands(df['close'], window=bb_window, window_dev=bb_std_dev)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Mid'] = bollinger.bollinger_mavg()
    df['Bollinger_Width'] = bollinger.bollinger_wband()
    
    df['Daily_Return'] = df['close'].pct_change() * 100
    df['Lag_1_Close'] = df['close'].shift(1)
    
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# Helper for performance metrics
def calculate_performance_metrics(returns_series, risk_free_rate=0.0):
    if returns_series.empty or len(returns_series) < 2:
        return {}

    cumulative_returns = (1 + returns_series / 100).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100

    annualized_return = (1 + returns_series / 100).prod()**(252/len(returns_series)) - 1 if len(returns_series) > 0 else 0
    annualized_return *= 100 # Convert to percentage

    daily_volatility = returns_series.std()
    annualized_volatility = daily_volatility * np.sqrt(252) # Assuming 252 trading days

    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

    # Max Drawdown
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / (peak + 1e-9) # Avoid division by zero for initial peak 0
    max_drawdown = drawdown.min() * 100

    # Sortino Ratio (requires negative returns)
    negative_returns = returns_series[returns_series < 0]
    downside_std_dev = negative_returns.std()
    sortino_ratio = (annualized_return - risk_free_rate) / (downside_std_dev * np.sqrt(252)) if downside_std_dev != 0 else np.nan

    return {
        "Total Return (%)": total_return,
        "Annualized Return (%)": annualized_return,
        "Annualized Volatility (%)": annualized_volatility * 100, # Convert to percentage
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Sortino Ratio": sortino_ratio
    }


# ---------------------------
# Sidebar for global actions
# ---------------------------
with st.sidebar:
    st.header("Account Info")
    if k:
        try:
            profile = k.profile()
            st.success("Authenticated ✅")
            st.write(f"**User:** {profile.get('user_name') or profile.get('user_id')}")
            st.write(f"**User ID:** {profile.get('user_id')}")
            st.write(f"**Login time:** {profile.get('login_time').strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception:
            st.warning("Authenticated, but profile fetch failed (check API permissions).")

        if st.button("Logout (clear token)", help="This will clear your access token from the session and require re-login."):
            st.session_state.pop("kite_access_token", None)
            st.session_state.pop("kite_login_response", None)
            for key in list(st.session_state.keys()): # Clear all session state for a clean re-run
                st.session_state.pop(key)
            st.success("Logged out. Please login again.")
            st.rerun() # FIX: Changed to st.rerun()
    else:
        st.info("Not authenticated yet. Please login using the link above.")

    st.markdown("---")
    st.header("Quick Data Access")
    if k:
        if st.button("Fetch Current Holdings"):
            try:
                holdings = k.holdings()
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success(f"Fetched {len(holdings)} holdings.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
            with st.expander("Show Holdings"):
                st.dataframe(st.session_state["holdings_data"])
    else:
        st.info("Login to access quick data.")


# ---------------------------
# Main UI - Tabs for modules
# ---------------------------
tabs = st.tabs(["Dashboard", "Portfolio", "Orders", "Market & Historical", "Machine Learning Analysis", "Risk & Stress Testing", "Performance Analysis", "Multi-Asset Analysis", "Websocket (stream)", "Instruments Utils"])
tab_dashboard, tab_portfolio, tab_orders, tab_market, tab_ml, tab_risk, tab_performance, tab_multi_asset, tab_ws, tab_inst = tabs

# ---------------------------
# TAB: DASHBOARD (New!)
# ---------------------------
with tab_dashboard:
    st.header("Personalized Dashboard")
    st.write("Welcome to your advanced financial analysis dashboard. Get a quick overview of your account and market insights.")

    if not k:
        st.info("Please login to Kite Connect to view your personalized dashboard.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Account Summary")
            try:
                profile = k.profile()
                margins = k.margins()
                st.metric("Account Holder", profile.get("user_name", "N/A"))
                st.metric("Available Equity Margin", f"₹{margins.get('equity', {}).get('available', {}).get('live_balance', 0):,.2f}")
                st.metric("Available Commodity Margin", f"₹{margins.get('commodity', {}).get('available', {}).get('live_balance', 0):,.2f}")
            except Exception as e:
                st.warning(f"Could not fetch full account summary: {e}")
                st.info("Check if your API key has sufficient permissions for profile and margins access.")

        with col2:
            st.subheader("Market Insight (NIFTY 50)")
            try:
                # Fetch NIFTY 50 (or a default benchmark) LTP
                nifty_ltp_data = get_ltp_price(k, "NIFTY 50", "NSE")
                if nifty_ltp_data and "NSE:NIFTY 50" in nifty_ltp_data:
                    nifty_ltp = nifty_ltp_data["NSE:NIFTY 50"]["last_price"]
                    nifty_change = nifty_ltp_data["NSE:NIFTY 50"]["change"]
                    st.metric("NIFTY 50 (LTP)", f"₹{nifty_ltp:,.2f}", delta=f"{nifty_change:.2f}%")
                else:
                    st.warning("Could not fetch NIFTY 50 LTP.")
            except Exception as e:
                st.warning(f"Error fetching NIFTY 50 data: {e}")

            # Optionally, show historical chart for NIFTY
            if st.session_state.get("historical_data_NIFTY") is None:
                if st.button("Load NIFTY 50 Historical for Chart"):
                    with st.spinner("Fetching NIFTY 50 historical data..."):
                        nifty_hist = get_historical(k, "NIFTY 50", datetime.now().date() - timedelta(days=180), datetime.now().date(), "day", "NSE")
                        if not nifty_hist.get("error"):
                            nifty_df = pd.DataFrame(nifty_hist)
                            nifty_df["date"] = pd.to_datetime(nifty_df["date"])
                            nifty_df.set_index("date", inplace=True)
                            nifty_df.sort_index(inplace=True)
                            st.session_state["historical_data_NIFTY"] = nifty_df
                            st.success("NIFTY 50 historical data loaded.")
                        else:
                            st.error(f"Error fetching NIFTY 50 historical: {nifty_hist['error']}")
            
            if st.session_state.get("historical_data_NIFTY") is not None:
                nifty_df = st.session_state["historical_data_NIFTY"]
                fig_nifty = go.Figure(data=[go.Candlestick(x=nifty_df.index,
                                                        open=nifty_df['open'],
                                                        high=nifty_df['high'],
                                                        low=nifty_df['low'],
                                                        close=nifty_df['close'],
                                                        name='NIFTY 50')])
                fig_nifty.update_layout(title_text="NIFTY 50 Last 6 Months",
                                        xaxis_rangeslider_visible=False,
                                        height=300, template="plotly_white")
                st.plotly_chart(fig_nifty, use_container_width=True)


        with col3:
            st.subheader("Quick Performance")
            if st.session_state.get("historical_data") is not None:
                last_symbol = st.session_state["last_fetched_symbol"]
                returns = st.session_state["historical_data"]["close"].pct_change().dropna() * 100
                if not returns.empty:
                    perf = calculate_performance_metrics(returns)
                    st.write(f"**{last_symbol}** (Last Fetched)")
                    st.metric("Total Return", f"{perf.get('Total Return (%)', 0):.2f}%")
                    st.metric("Annualized Volatility", f"{perf.get('Annualized Volatility (%)', 0):.2f}%")
                    st.metric("Sharpe Ratio", f"{perf.get('Sharpe Ratio', 0):.2f}")
                else:
                    st.info("No sufficient historical data for quick performance calculation.")
            else:
                st.info("Fetch some historical data in 'Market & Historical' tab to see quick performance here.")


# ---------------------------
# TAB: PORTFOLIO
# ---------------------------
with tab_portfolio:
    st.header("Your Portfolio Overview")
    st.markdown("View your current holdings, positions, and margin details.")

    if not k:
        st.info("Login first to fetch portfolio data.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Fetch Holdings"):
                try:
                    holdings = k.holdings()
                    st.session_state["holdings_data"] = pd.DataFrame(holdings)
                    st.success(f"Fetched {len(holdings)} holdings.")
                except Exception as e:
                    st.error(f"Error fetching holdings: {e}")
            if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
                st.subheader("Current Holdings")
                st.dataframe(st.session_state["holdings_data"], use_container_width=True)
            else:
                st.info("No holdings data available. Click 'Fetch Holdings'.")

        with col2:
            if st.button("Fetch Positions"):
                try:
                    positions = k.positions()
                    st.session_state["net_positions"] = pd.DataFrame(positions.get("net", []))
                    st.session_state["day_positions"] = pd.DataFrame(positions.get("day", []))
                    st.success(f"Fetched positions (Net: {len(positions.get('net', []))}, Day: {len(positions.get('day', []))}).")
                except Exception as e:
                    st.error(f"Error fetching positions: {e}")
            
            if st.session_state.get("net_positions") is not None and not st.session_state["net_positions"].empty:
                st.subheader("Net Positions")
                st.dataframe(st.session_state["net_positions"], use_container_width=True)
            else:
                st.info("No net positions data available. Click 'Fetch Positions'.")
            
            if st.session_state.get("day_positions") is not None and not st.session_state["day_positions"].empty:
                st.subheader("Day Positions")
                st.dataframe(st.session_state["day_positions"], use_container_width=True)
            else:
                st.info("No day positions data available.")

        with col3:
            if st.button("Fetch Margins"):
                try:
                    margins = k.margins()
                    st.session_state["margins_data"] = margins
                    st.success("Fetched margins data.")
                except Exception as e:
                    st.error(f"Error fetching margins: {e}")
            if st.session_state.get("margins_data") is not None:
                st.subheader("Available Margins")
                margins_df = pd.DataFrame([
                    {"Category": "Equity - Available", "Value": st.session_state["margins_data"].get('equity', {}).get('available', {}).get('live_balance', 0)},
                    {"Category": "Equity - Used", "Value": st.session_state["margins_data"].get('equity', {}).get('utilised', {}).get('overall', 0)},
                    {"Category": "Commodity - Available", "Value": st.session_state["margins_data"].get('commodity', {}).get('available', {}).get('live_balance', 0)},
                    {"Category": "Commodity - Used", "Value": st.session_state["margins_data'].get('commodity', {}).get('utilised', {}).get('overall', 0)},
                ])
                margins_df["Value"] = margins_df["Value"].apply(lambda x: f"₹{x:,.2f}")
                st.dataframe(margins_df, use_container_width=True)
            else:
                st.info("No margins data available. Click 'Fetch Margins'.")


# ---------------------------
# TAB: ORDERS
# ---------------------------
with tab_orders:
    st.header("Orders — Place, Modify, Cancel & View")
    st.markdown("Manage your trading orders directly from here.")

    if not k:
        st.info("Login first to use orders API.")
    else:
        st.subheader("Place New Order")
        with st.form("place_order_form", clear_on_submit=False):
            col_order1, col_order2 = st.columns(2)
            with col_order1:
                variety = st.selectbox("Variety", ["regular", "amo", "co", "iceberg"], index=0, help="Order variety: Regular, After Market Order (AMO), Cover Order (CO), Iceberg.")
                exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0)
                tradingsymbol = st.text_input("Tradingsymbol (e.g., INFY / NIFTY24FEBCALL19000)", value="INFY", help="Enter the exact trading symbol.")
                transaction_type = st.radio("Transaction Type", ["BUY", "SELL"], index=0, horizontal=True)
                quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            with col_order2:
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"], index=0, help="MARKET: Best available price. LIMIT: Specify a price. SL: Stop Loss. SL-M: Stop Loss Market.")
                product = st.selectbox("Product Type", ["CNC", "MIS", "NRML", "CO", "MTF"], index=0, help="CNC: Cash & Carry (Delivery). MIS: Margin Intraday Square-off. NRML: Normal (Futures & Options). CO: Cover Order. MTF: Margin Trade Funding.")
                price = st.text_input("Price (for LIMIT/SL orders)", value="", help="Required for LIMIT and SL orders. Leave blank for MARKET orders.")
                trigger_price = st.text_input("Trigger Price (for SL/SL-M orders)", value="", help="Required for SL and SL-M orders. This price triggers the order.")
                validity = st.selectbox("Validity", ["DAY", "IOC", "TTL"], index=0, help="DAY: Valid for the day. IOC: Immediate or Cancel. TTL: Time in Minutes (for bracket orders, etc.).")
                tag = st.text_input("Tag (optional, max 20 chars)", value="", help="A custom tag for identifying your order.")
            
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

                    with st.spinner("Placing order..."):
                        resp = k.place_order(**params)
                        st.success(f"Order placed successfully! Order ID: {resp.get('order_id')}")
                        st.json(resp)
                except Exception as e:
                    st.error(f"Failed to place order: {e}")

        st.markdown("---")
        st.subheader("Manage Existing Orders & Trades")
        col_view_orders, col_manage_single = st.columns(2)

        with col_view_orders:
            st.markdown("#### View All Orders and Trades")
            if st.button("Fetch All Orders (Today)"):
                try:
                    orders = k.orders()
                    st.session_state["all_orders"] = pd.DataFrame(orders)
                    st.success(f"Fetched {len(orders)} orders.")
                except Exception as e:
                    st.error(f"Error fetching orders: {e}")
            if st.session_state.get("all_orders") is not None:
                with st.expander("Show Orders"):
                    st.dataframe(st.session_state["all_orders"], use_container_width=True)

            if st.button("Fetch All Trades (Today)"):
                try:
                    trades = k.trades()
                    st.session_state["all_trades"] = pd.DataFrame(trades)
                    st.success(f"Fetched {len(trades)} trades.")
                except Exception as e:
                    st.error(f"Error fetching trades: {e}")
            if st.session_state.get("all_trades") is not None:
                with st.expander("Show Trades"):
                    st.dataframe(st.session_state["all_trades"], use_container_width=True)

        with col_manage_single:
            st.markdown("#### Get History, Modify, or Cancel Single Order")
            order_id_action = st.text_input("Enter Order ID for action", value="", help="The Order ID of the order you wish to manage.")
            
            st.markdown("---")
            st.markdown("##### Order History")
            if st.button("Get Order History", key="get_order_history"):
                if order_id_action:
                    try:
                        history = k.order_history(order_id_action)
                        st.success(f"History for Order ID: {order_id_action}")
                        st.json(history)
                    except Exception as e:
                        st.error(f"Failed to get order history: {e}")
                else:
                    st.warning("Please provide an Order ID.")

            st.markdown("##### Modify Order")
            with st.form("modify_order_form"):
                mod_variety = st.selectbox("Variety (for Modify)", ["regular", "amo", "co", "iceberg"], index=0, key="mod_variety")
                mod_new_price = st.text_input("New Price (optional)", value="", key="mod_new_price")
                mod_new_qty = st.number_input("New Quantity (optional)", min_value=0, value=0, step=1, key="mod_new_qty")
                mod_new_trigger_price = st.text_input("New Trigger Price (optional)", value="", key="mod_new_trigger_price")
                
                submit_modify = st.form_submit_button("Modify Order")
                if submit_modify:
                    if order_id_action:
                        try:
                            modify_args = {}
                            if mod_new_price:
                                modify_args["price"] = float(mod_new_price)
                            if mod_new_qty > 0:
                                modify_args["quantity"] = int(mod_new_qty)
                            if mod_new_trigger_price:
                                modify_args["trigger_price"] = float(mod_new_trigger_price)

                            if not modify_args:
                                st.warning("No new price or quantity provided for modification.")
                            else:
                                with st.spinner(f"Modifying order {order_id_action}..."):
                                    res = k.modify_order(variety=mod_variety, order_id=order_id_action, **modify_args)
                                    st.success(f"Order {order_id_action} modified successfully!")
                                    st.json(res)
                        except Exception as e:
                            st.error(f"Failed to modify order: {e}")
                    else:
                        st.warning("Please provide an Order ID to modify.")

            st.markdown("##### Cancel Order")
            if st.button("Cancel Order", key="cancel_order"):
                if order_id_action:
                    try:
                        with st.spinner(f"Cancelling order {order_id_action}..."):
                            # Assuming 'regular' variety for simplicity; in a real app, this would be dynamic
                            res = k.cancel_order(variety="regular", order_id=order_id_action)
                            st.success(f"Order {order_id_action} cancelled successfully!")
                            st.json(res)
                    except Exception as e:
                        st.error(f"Failed to cancel order: {e}")
                else:
                    st.warning("Please provide an Order ID to cancel.")

# ---------------------------
# TAB: MARKET & HISTORICAL
# ---------------------------
with tab_market:
    st.header("Market Data & Historical Candles")
    st.markdown("Fetch real-time quotes and extensive historical data for any instrument.")

    if not k:
        st.info("Login first to fetch market data (quotes/historical).")
    else:
        st.subheader("Current Market Data Snapshot")
        col_market_quote1, col_market_quote2 = st.columns([1, 2])
        with col_market_quote1:
            q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], index=0, key="market_exchange_tab")
            q_symbol = st.text_input("Tradingsymbol (e.g., INFY)", value="INFY", key="market_symbol_tab")
            market_data_type = st.radio("Choose data type:", 
                                         ("LTP (Last Traded Price)", "OHLC + LTP", "Full Market Quote (OHLC, Depth, OI)"), 
                                         index=0, key="market_data_type_radio_tab")
            if st.button("Get Market Data"):
                market_data_response = {}
                if market_data_type == "LTP (Last Traded Price)":
                    market_data_response = get_ltp_price(k, q_symbol, q_exchange)
                elif market_data_type == "OHLC + LTP":
                    market_data_response = get_ohlc_quote(k, q_symbol, q_exchange)
                else: # Full Market Quote
                    market_data_response = get_full_market_quote(k, q_symbol, q_exchange)
                
                if "error" in market_data_response:
                    st.error(f"Market data fetch failed: {market_data_response['error']}")
                    if "Insufficient permission" in market_data_response['error']:
                        st.warning("For 'Full Market Quote', you might need a paid subscription to the Kite Connect API. Try 'LTP' or 'OHLC + LTP' if you encounter permission errors.")
                    st.session_state["current_market_data"] = None # Clear previous data
                else:
                    st.session_state["current_market_data"] = market_data_response
                    st.success(f"Fetched {market_data_type} for {q_symbol}.")

        with col_market_quote2:
            if st.session_state.get("current_market_data"):
                st.markdown("##### Latest Quote Details")
                st.json(st.session_state["current_market_data"])
            else:
                st.info("Market data will appear here after fetching.")

        st.markdown("---")
        st.subheader("Historical Price Data (Candlestick Charts)")
        st.markdown("Retrieve and visualize historical OHLCV data for comprehensive analysis.")

        with st.expander("Load Instruments for Symbol Lookup (Required)"):
            exchange_for_lookup = st.selectbox("Exchange to load instruments for lookup", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0, key="hist_inst_load_exchange")
            if st.button("Load Instruments into Cache"):
                inst_df = load_instruments(k, exchange_for_lookup)
                st.session_state["instruments_df"] = inst_df
                if not inst_df.empty:
                    st.success(f"Loaded {len(inst_df)} instruments for {exchange_for_lookup}.")
                else:
                    st.warning(f"Could not load instruments for {exchange_for_lookup}. Check API key and permissions.")

        col_hist_controls, col_hist_plot = st.columns([1, 2])

        with col_hist_controls:
            hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], index=0, key="hist_ex_tab")
            hist_symbol = st.text_input("Tradingsymbol (e.g., INFY)", value="INFY", key="hist_sym_tab")
            
            default_to_date = datetime.now().date()
            default_from_date = default_to_date - timedelta(days=90) # Default to 3 months

            from_date = st.date_input("From Date", value=default_from_date, key="from_dt_tab")
            to_date = st.date_input("To Date", value=default_to_date, key="to_dt_tab")
            interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "day", "week", "month"], index=4)

            if st.button("Fetch Historical Data"):
                if "instruments_df" not in st.session_state or st.session_state["instruments_df"].empty:
                    st.error("Please load instruments first from the expander above to enable symbol lookup.")
                else:
                    with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                        hist_data = get_historical(k, hist_symbol, from_date, to_date, interval, hist_exchange)
                        
                        if "error" in hist_data:
                            st.error(f"Historical fetch failed: {hist_data['error']}")
                            if "Insufficient permission" in hist_data['error']:
                                st.warning("Your Zerodha API key might require an active subscription for historical data.")
                            st.session_state["historical_data"] = pd.DataFrame() # Clear previous data
                            st.session_state["last_fetched_symbol"] = None
                        else:
                            df = pd.DataFrame(hist_data)
                            if not df.empty:
                                df["date"] = pd.to_datetime(df["date"])
                                df.set_index("date", inplace=True)
                                df.sort_index(inplace=True)
                                st.session_state["historical_data"] = df # Store for ML analysis
                                st.session_state["last_fetched_symbol"] = hist_symbol
                                st.success(f"Successfully fetched {len(df)} records for {hist_symbol} ({interval}).")
                                st.dataframe(df.head()) # Show a preview
                            else:
                                st.info(f"No historical data returned for {hist_symbol} for the selected period.")
                                st.session_state["historical_data"] = pd.DataFrame()
                                st.session_state["last_fetched_symbol"] = None

        with col_hist_plot:
            if st.session_state.get("historical_data") is not None and not st.session_state["historical_data"].empty:
                df = st.session_state["historical_data"]
                symbol_name = st.session_state["last_fetched_symbol"]

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, 
                                    row_heights=[0.7, 0.3])

                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['open'],
                                            high=df['high'],
                                            low=df['low'],
                                            close=df['close'],
                                            name='Candlestick'), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue'), row=2, col=1)

                fig.update_layout(title_text=f"Historical Price & Volume for {symbol_name}",
                                  xaxis_rangeslider_visible=False,
                                  height=600,
                                  template="plotly_white",
                                  hovermode="x unified") # Improved hover
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Historical chart will appear here after fetching data.")

# ---------------------------
# TAB: MACHINE LEARNING ANALYSIS
# ---------------------------
with tab_ml:
    st.header("Machine Learning Driven Price Analysis")
    st.markdown("Apply technical indicators, train ML models, and visualize predictions for informed decision-making.")

    if not k:
        st.info("Login first to perform ML analysis.")
    else:
        historical_data = st.session_state.get("historical_data")
        last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

        if historical_data is None or historical_data.empty:
            st.warning("No historical data available. Please fetch data for an instrument from the 'Market & Historical' tab first.")
        else:
            st.subheader(f"1. Feature Engineering: Technical Indicators for {last_symbol}")
            st.write("Generate various technical indicators which will serve as features for machine learning models.")

            col_indicator_params, col_indicator_data = st.columns([1,2])
            with col_indicator_params:
                st.markdown("##### Indicator Parameters")
                sma_short_window = st.slider("SMA Short Window", min_value=5, max_value=50, value=10, step=1)
                sma_long_window = st.slider("SMA Long Window", min_value=20, max_value=200, value=50, step=5)
                rsi_window = st.slider("RSI Window", min_value=7, max_value=30, value=14, step=1)
                macd_fast = st.slider("MACD Fast Period", min_value=5, max_value=20, value=12, step=1)
                macd_slow = st.slider("MACD Slow Period", min_value=20, max_value=40, value=26, step=1)
                macd_signal = st.slider("MACD Signal Period", min_value=5, max_value=15, value=9, step=1)
                bb_window = st.slider("Bollinger Bands Window", min_value=10, max_value=50, value=20, step=1)
                bb_std_dev = st.slider("Bollinger Bands Std Dev", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
                
                if st.button("Apply Indicators"):
                    df_with_indicators = add_indicators(historical_data.copy(), 
                                                        sma_short_window, sma_long_window, 
                                                        rsi_window, macd_fast, macd_slow, macd_signal, 
                                                        bb_window, bb_std_dev)
                    if df_with_indicators.empty:
                        st.error("Failed to add indicators. Data might be too short or contains invalid values for the chosen parameters. Try adjusting the parameters or fetching more historical data.")
                        st.session_state["ml_data"] = None
                    else:
                        st.session_state["ml_data"] = df_with_indicators
                        st.success("Technical indicators applied successfully.")
                        
            with col_indicator_data:
                if st.session_state.get("ml_data") is not None and not st.session_state["ml_data"].empty:
                    st.markdown("##### Data with Indicators (Head)")
                    st.dataframe(st.session_state["ml_data"].head(), use_container_width=True)
                    st.markdown("##### Visualizations of Indicators")
                    df_plot = st.session_state["ml_data"]
                    
                    fig_indicators = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                                                vertical_spacing=0.05,
                                                row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
                                                subplot_titles=(f"{last_symbol} Price with Moving Averages & Bollinger Bands", "RSI", "MACD", "Bollinger Band Width", "Volume"))

                    fig_indicators.add_trace(go.Candlestick(x=df_plot.index,
                                                            open=df_plot['open'], high=df_plot['high'],
                                                            low=df_plot['low'], close=df_plot['close'],
                                                            name='Candlestick'), row=1, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_Short'], mode='lines', name=f'SMA {sma_short_window}', line=dict(color='orange', width=1)), row=1, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_Long'], mode='lines', name=f'SMA {sma_long_window}', line=dict(color='purple', width=1)), row=1, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Bollinger_High'], mode='lines', name='BB High', line=dict(color='gray', dash='dot', width=1)), row=1, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Bollinger_Low'], mode='lines', name='BB Low', line=dict(color='gray', dash='dot', width=1)), row=1, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Bollinger_Mid'], mode='lines', name='BB Mid', line=dict(color='blue', dash='dash', width=1)), row=1, col=1)
                    
                    fig_indicators.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], mode='lines', name='RSI', line=dict(color='green')), row=2, col=1)
                    fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top right", row=2, col=1)
                    fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right", row=2, col=1)

                    fig_indicators.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')), row=3, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_signal'], mode='lines', name='Signal Line', line=dict(color='red')), row=3, col=1)
                    # Use MACD_hist for histogram bars
                    fig_indicators.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_hist'], name='MACD Histogram', marker_color='gray'), row=3, col=1)

                    fig_indicators.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Bollinger_Width'], mode='lines', name='BB Width', line=dict(color='purple')), row=4, col=1)

                    fig_indicators.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], name='Volume', marker_color='lightgray'), row=5, col=1)

                    fig_indicators.update_layout(height=1200, xaxis_rangeslider_visible=False, template="plotly_white", hovermode="x unified")
                    fig_indicators.update_yaxes(title_text="Price", row=1, col=1)
                    fig_indicators.update_yaxes(title_text="RSI", row=2, col=1)
                    fig_indicators.update_yaxes(title_text="MACD", row=3, col=1)
                    fig_indicators.update_yaxes(title_text="BB Width", row=4, col=1)
                    fig_indicators.update_yaxes(title_text="Volume", row=5, col=1)
                    st.plotly_chart(fig_indicators, use_container_width=True)
                else:
                    st.info("Apply indicators to see the processed data and visualizations.")

            ml_data = st.session_state.get("ml_data")
            if ml_data is not None and not ml_data.empty:
                st.subheader(f"2. Machine Learning Model Training for {last_symbol}")
                st.write("Train a model to predict the next period's closing price based on the selected features.")

                col_ml_controls, col_ml_output = st.columns(2)
                with col_ml_controls:
                    model_type = st.selectbox("Select ML Model", ["Linear Regression", "Random Forest Regressor", "LightGBM Regressor"], key="ml_model_type")
                    target_column = st.selectbox("Select Target Variable", ["close"], help="Currently, only 'close' price is supported as target.", key="ml_target_col")
                    
                    # Shift target for prediction: predicting next period's close price
                    ml_data_processed = ml_data.copy()
                    ml_data_processed['target'] = ml_data_processed[target_column].shift(-1)
                    ml_data_processed.dropna(subset=['target'], inplace=True)
                    
                    features = [col for col in ml_data_processed.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target', 'MACD_hist']] # Exclude MACD_hist if not a direct feature
                    
                    # Allow user to select features
                    selected_features = st.multiselect("Select Features for Model (recommended: use all indicators)", 
                                                        options=features, 
                                                        default=features,
                                                        help="Choose which technical indicators and lagged prices to use as input features for the model.")
                    
                    if not selected_features:
                        st.warning("Please select at least one feature.")
                    else:
                        X = ml_data_processed[selected_features]
                        y = ml_data_processed['target']

                        if X.empty or y.empty:
                            st.error("Not enough clean data after preprocessing to train the model. Adjust parameters or fetch more data.")
                        else:
                            test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
                            
                            st.info(f"Training data: {len(X_train)} samples, Testing data: {len(X_test)} samples")

                            if st.button(f"Train {model_type} Model"):
                                if len(X_train) == 0 or len(X_test) == 0:
                                    st.error("Insufficient data for training or testing after split. Adjust test size or fetch more data.")
                                else:
                                    model = None
                                    if model_type == "Linear Regression":
                                        model = LinearRegression()
                                    elif model_type == "Random Forest Regressor":
                                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                                    elif model_type == "LightGBM Regressor":
                                        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)

                                    if model:
                                        with st.spinner(f"Training {model_type} model... this may take a moment."):
                                            model.fit(X_train, y_train)
                                            y_pred = model.predict(X_test)

                                        st.session_state["ml_model"] = model
                                        st.session_state["y_test"] = y_test
                                        st.session_state["y_pred"] = y_pred
                                        st.session_state["X_test_ml"] = X_test
                                        st.session_state["ml_features"] = selected_features
                                        st.session_state["ml_model_type"] = model_type

                                        st.success(f"{model_type} Model Trained Successfully!")
                                        
                with col_ml_output:
                    if st.session_state.get("ml_model") and st.session_state.get("y_test") is not None:
                        st.markdown("##### Model Performance Metrics")
                        st.write(f"**Model Type:** {st.session_state['ml_model_type']}")
                        st.metric("Mean Squared Error (MSE)", f"{mean_squared_error(st.session_state['y_test'], st.session_state['y_pred']):.4f}")
                        st.metric("R2 Score", f"{r2_score(st.session_state['y_test'], st.session_state['y_pred']):.4f}")

                        st.markdown("##### Actual vs. Predicted Prices")
                        pred_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['y_pred']}, index=st.session_state['y_test'].index)
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'], mode='lines', name='Actual Price', line=dict(color='blue')))
                        fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'], mode='lines', name='Predicted Price', line=dict(color='red', dash='dot')))
                        fig_pred.update_layout(title_text=f"{st.session_state['ml_model_type']} Actual vs. Predicted Prices on Test Set for {last_symbol}", 
                                            height=500, xaxis_title="Date", yaxis_title="Price",
                                            template="plotly_white", hovermode="x unified")
                        st.plotly_chart(fig_pred, use_container_width=True)

                        # Feature Importance for tree-based models
                        if st.session_state["ml_model_type"] in ["Random Forest Regressor", "LightGBM Regressor"]:
                            st.markdown("##### Feature Importance")
                            model = st.session_state["ml_model"]
                            features = st.session_state["ml_features"]
                            importance = model.feature_importances_
                            feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
                            
                            fig_feat_imp = go.Figure(go.Bar(
                                x=feature_importance_df['Importance'],
                                y=feature_importance_df['Feature'],
                                orientation='h',
                                marker_color='skyblue'
                            ))
                            fig_feat_imp.update_layout(title_text=f"Feature Importance for {st.session_state['ml_model_type']}",
                                                       yaxis_title="Feature", xaxis_title="Importance Score",
                                                       height=400, template="plotly_white")
                            st.plotly_chart(fig_feat_imp, use_container_width=True)


            st.subheader(f"3. Real-time Price Prediction (Simulated for {last_symbol})")
            st.markdown("This section simulates using the trained model for **next period** price prediction. For a live system, you would feed real-time aggregated data points.")

            if st.session_state.get("ml_model") and st.session_state.get("X_test_ml") is not None:
                model = st.session_state["ml_model"]
                X_test_ml = st.session_state["X_test_ml"]
                ml_features = st.session_state["ml_features"]

                st.write(f"**Model trained:** {st.session_state['ml_model_type']}")
                st.write(f"**Features used:** {', '.join(ml_features)}")

                if not X_test_ml.empty:
                    latest_features_df = X_test_ml.iloc[[-1]][ml_features] # Simulate the latest available features
                    if st.button("Simulate Next Period Prediction"):
                        with st.spinner("Generating simulated prediction..."):
                            simulated_prediction = model.predict(latest_features_df)[0]
                        st.success(f"Simulated **next period** close price prediction: **₹{simulated_prediction:.2f}**")
                        st.info("This is a simulation using the last available test data point's features. In a live trading system, these features would be derived from fresh, real-time market data (e.g., from the WebSocket feed aggregated into candles).")
                else:
                    st.warning("No test data available for simulation. Please train the model first.")
            else:
                st.info("Train a machine learning model first to see a real-time prediction simulation.")
            
            st.markdown("---")
            st.subheader("4. Basic Backtesting: SMA Crossover Strategy")
            st.write("Test a simple moving average crossover strategy on the historical data.")

            if st.session_state.get("ml_data") is not None and not st.session_state["ml_data"].empty:
                df_backtest = st.session_state["ml_data"].copy()
                short_ma = st.slider("Short MA Window", 5, 50, 10, key="bt_short_ma")
                long_ma = st.slider("Long MA Window", 20, 200, 50, key="bt_long_ma")

                if st.button("Run Backtest"):
                    df_backtest['SMA_Short_BT'] = ta.trend.sma_indicator(df_backtest['close'], window=short_ma)
                    df_backtest['SMA_Long_BT'] = ta.trend.sma_indicator(df_backtest['close'], window=long_ma)
                    df_backtest['Signal'] = 0.0
                    
                    # Ensure alignment and sufficient data for comparison
                    # Use .loc to avoid SettingWithCopyWarning
                    df_backtest.loc[df_backtest.index[short_ma:], 'Signal'] = np.where(
                        df_backtest['SMA_Short_BT'][short_ma:] > df_backtest['SMA_Long_BT'][short_ma:], 1.0, 0.0
                    )
                    df_backtest['Position'] = df_backtest['Signal'].diff()

                    # Calculate strategy returns
                    df_backtest['Strategy_Return'] = df_backtest['Daily_Return'] * df_backtest['Signal'].shift(1)
                    df_backtest['Cumulative_Strategy_Return'] = (1 + df_backtest['Strategy_Return'] / 100).cumprod() - 1
                    df_backtest['Cumulative_Buy_Hold_Return'] = (1 + df_backtest['Daily_Return'] / 100).cumprod() - 1

                    st.markdown("##### Strategy Performance")
                    
                    col_bt_metrics, col_bt_chart = st.columns(2)
                    with col_bt_metrics:
                        if not df_backtest['Strategy_Return'].dropna().empty:
                            strategy_metrics = calculate_performance_metrics(df_backtest['Strategy_Return'].dropna())
                            buy_hold_metrics = calculate_performance_metrics(df_backtest['Daily_Return'].dropna())

                            st.write("**Strategy Metrics**")
                            st.metric("Strategy Total Return", f"{strategy_metrics.get('Total Return (%)', 0):.2f}%")
                            st.metric("Strategy Annualized Volatility", f"{strategy_metrics.get('Annualized Volatility (%)', 0):.2f}%")
                            st.metric("Strategy Sharpe Ratio", f"{strategy_metrics.get('Sharpe Ratio', 0):.2f}")
                            st.metric("Strategy Max Drawdown", f"{strategy_metrics.get('Max Drawdown (%)', 0):.2f}%")
                            st.write("---")
                            st.write("**Buy & Hold Metrics**")
                            st.metric("Buy & Hold Total Return", f"{buy_hold_metrics.get('Total Return (%)', 0):.2f}%")
                            st.metric("Buy & Hold Annualized Volatility", f"{buy_hold_metrics.get('Annualized Volatility (%)', 0):.2f}%")

                        else:
                            st.warning("Not enough data to calculate strategy returns.")

                    with col_bt_chart:
                        fig_backtest = go.Figure()
                        fig_backtest.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Cumulative_Strategy_Return'] * 100, mode='lines', name='Strategy Return (%)', line=dict(color='green', width=2)))
                        fig_backtest.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Cumulative_Buy_Hold_Return'] * 100, mode='lines', name='Buy & Hold Return (%)', line=dict(color='blue', dash='dash', width=1)))
                        
                        fig_backtest.update_layout(title_text=f"SMA Crossover Strategy vs. Buy & Hold for {last_symbol}",
                                                   xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                                                   template="plotly_white", hovermode="x unified", height=450)
                        st.plotly_chart(fig_backtest, use_container_width=True)

                        # Visualize trades
                        fig_trades = go.Figure(data=[go.Candlestick(x=df_backtest.index,
                                                                    open=df_backtest['open'], high=df_backtest['high'],
                                                                    low=df_backtest['low'], close=df_backtest['close'],
                                                                    name='Candlestick')])
                        
                        fig_trades.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['SMA_Short_BT'], mode='lines', name=f'SMA {short_ma}', line=dict(color='orange', width=1)))
                        fig_trades.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['SMA_Long_BT'], mode='lines', name=f'SMA {long_ma}', line=dict(color='purple', width=1)))

                        # Plot buy signals
                        fig_trades.add_trace(go.Scatter(
                            x=df_backtest.index[df_backtest['Position'] == 1],
                            y=df_backtest['close'][df_backtest['Position'] == 1],
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=10, color='green'),
                            name='Buy Signal'
                        ))
                        # Plot sell signals
                        fig_trades.add_trace(go.Scatter(
                            x=df_backtest.index[df_backtest['Position'] == -1],
                            y=df_backtest['close'][df_backtest['Position'] == -1],
                            mode='markers',
                            marker=dict(symbol='triangle-down', size=10, color='red'),
                            name='Sell Signal'
                        ))
                        fig_trades.update_layout(title=f'SMA Crossover Trading Signals for {last_symbol}',
                                                  xaxis_rangeslider_visible=False,
                                                  template="plotly_white", height=500)
                        st.plotly_chart(fig_trades, use_container_width=True)

            else:
                st.info("Apply technical indicators first to enable backtesting.")


# ---------------------------
# TAB: RISK & STRESS TESTING
# ---------------------------
with tab_risk:
    st.header("Risk & Stress Testing Models")
    st.markdown("Analyze historical volatility, calculate Value at Risk (VaR), and simulate extreme market scenarios for robust risk management.")

    if not k:
        st.info("Login first to perform risk analysis.")
    else:
        historical_data = st.session_state.get("historical_data")
        last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

        if historical_data is None or historical_data.empty:
            st.warning("No historical data available. Please fetch data for an instrument from the 'Market & Historical' tab first.")
        else:
            if historical_data.index.duplicated().any():
                historical_data = historical_data.loc[~historical_data.index.duplicated(keep='first')]
            
            historical_data['close'] = pd.to_numeric(historical_data['close'], errors='coerce')
            historical_data.dropna(subset=['close'], inplace=True)

            daily_returns = historical_data['close'].pct_change().dropna() * 100 # Convert to percentage returns
            
            if daily_returns.empty or len(daily_returns) < 2:
                st.error("Not enough valid data to compute daily returns for risk analysis. Ensure historical data is fetched correctly.")
                st.stop()

            st.subheader(f"1. Volatility & Returns Analysis for {last_symbol}")
            st.write("Examine the historical distribution and volatility of returns.")
            
            col_vol_metrics, col_vol_dist = st.columns([1,2])
            with col_vol_metrics:
                st.markdown("##### Key Metrics")
                st.dataframe(daily_returns.describe().to_frame().T, use_container_width=True)

                trading_days_per_year = 252 
                annualized_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
                st.metric("Annualized Volatility", f"{annualized_volatility:.2f}%")
                st.metric("Mean Daily Return", f"{daily_returns.mean():.2f}%")

                st.markdown("---")
                st.markdown("##### Rolling Volatility")
                rolling_window = st.slider("Rolling Volatility Window (days)", min_value=10, max_value=252, value=30)
                if len(daily_returns) > rolling_window:
                    rolling_vol = daily_returns.rolling(window=rolling_window).std() * np.sqrt(trading_days_per_year)
                    fig_rolling_vol = go.Figure(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name='Rolling Volatility'))
                    fig_rolling_vol.update_layout(title_text=f"Rolling {rolling_window}-Day Annualized Volatility",
                                                  xaxis_title="Date", yaxis_title="Volatility (%)",
                                                  template="plotly_white", height=300)
                    st.plotly_chart(fig_rolling_vol, use_container_width=True)
                else:
                    st.info("Not enough data for rolling volatility calculation with selected window.")

            with col_vol_dist:
                st.markdown("##### Distribution of Daily Returns")
                fig_volatility = go.Figure()
                fig_volatility.add_trace(go.Histogram(x=daily_returns, nbinsx=50, name='Daily Returns', marker_color='skyblue'))
                fig_volatility.update_layout(title_text=f'Distribution of Daily Returns for {last_symbol}',
                                             xaxis_title='Daily Return (%)',
                                             yaxis_title='Frequency',
                                             height=500, template="plotly_white")
                st.plotly_chart(fig_volatility, use_container_width=True)

            st.subheader(f"2. Value at Risk (VaR) Calculation for {last_symbol}")
            st.write("Estimate the maximum potential loss over a specified period and confidence level using the historical method.")
            
            col_var_controls, col_var_plot = st.columns([1,2])
            with col_var_controls:
                confidence_level = st.slider("Confidence Level (%)", min_value=90, max_value=99, value=95, step=1)
                holding_period_var = st.number_input("Holding Period for VaR (days)", min_value=1, value=1, step=1)
                
                # Calculate VaR using the historical percentile method
                var_percentile_1day = np.percentile(daily_returns, 100 - confidence_level)
                
                # Scale 1-day VaR for multiple days (simplified assumption: returns are independent and identically distributed)
                var_percentile_multiday = var_percentile_1day * np.sqrt(holding_period_var)

                st.write(f"With **{confidence_level}% confidence**, the maximum expected loss over **{holding_period_var} day(s)** is:")
                st.metric(label=f"VaR ({confidence_level}%)", value=f"{abs(var_percentile_multiday):.2f}%")

                current_price = historical_data['close'].iloc[-1]
                potential_loss_value = (abs(var_percentile_multiday) / 100) * current_price
                st.metric(label=f"Potential Loss (based on current price ₹{current_price:.2f})", value=f"₹{potential_loss_value:,.2f}")
                st.info("This VaR calculation uses the historical method. Other methods (e.g., parametric, Monte Carlo) can also be used.")
            
            with col_var_plot:
                fig_var = go.Figure()
                fig_var.add_trace(go.Histogram(x=daily_returns, nbinsx=50, name='Daily Returns', marker_color='skyblue'))
                fig_var.add_vline(x=var_percentile_1day, line_dash="dash", line_color="red", 
                                  annotation_text=f"1-Day VaR {confidence_level}%: {var_percentile_1day:.2f}%", 
                                  annotation_position="top right")
                fig_var.update_layout(title_text=f'Daily Returns Distribution with {confidence_level}% VaR for {last_symbol}',
                                      xaxis_title='Daily Return (%)',
                                      yaxis_title='Frequency',
                                      height=400, template="plotly_white")
                st.plotly_chart(fig_var, use_container_width=True)

            st.subheader(f"3. Stress Testing (Scenario Analysis) for {last_symbol}")
            st.write("Simulate the impact of adverse (or favorable) market scenarios on the instrument's price.")
            
            col_stress_controls, col_stress_results = st.columns([1,2])
            with col_stress_controls:
                # Pre-defined scenarios
                scenarios = {
                    "Historical Worst Day Drop": {"type": "historical", "percent": daily_returns.min() if not daily_returns.empty else 0},
                    "Global Financial Crisis (-20%)": {"type": "fixed", "percent": -20.0},
                    "Flash Crash (-10%)": {"type": "fixed", "percent": -10.0},
                    "Moderate Correction (-5%)": {"type": "fixed", "percent": -5.0},
                    "Significant Rally (+10%)": {"type": "fixed", "percent": 10.0},
                    "Custom % Change": {"type": "custom", "percent": 0.0}
                }

                scenario_key = st.selectbox("Select Stress Scenario", list(scenarios.keys()))
                custom_change_percent = 0.0
                if scenario_key == "Custom % Change":
                    custom_change_percent = st.number_input("Enter Custom Percentage Change (%)", value=0.0, step=0.1)
                
                if st.button("Run Stress Test"):
                    current_price = historical_data['close'].iloc[-1]
                    scenario_data = scenarios[scenario_key]
                    stressed_price = 0
                    scenario_change_percent = 0

                    if scenario_data["type"] == "historical":
                        scenario_change_percent = scenario_data["percent"]
                    elif scenario_data["type"] == "fixed":
                        scenario_change_percent = scenario_data["percent"]
                    elif scenario_data["type"] == "custom":
                        scenario_change_percent = custom_change_percent
                    
                    stressed_price = current_price * (1 + scenario_change_percent / 100)

                    st.session_state["stress_test_results"] = {
                        "scenario_key": scenario_key,
                        "current_price": current_price,
                        "stressed_price": stressed_price,
                        "scenario_change_percent": scenario_change_percent
                    }
                    st.success("Stress test executed.")
            
            with col_stress_results:
                if st.session_state.get("stress_test_results"):
                    results = st.session_state["stress_test_results"]
                    st.markdown(f"##### Results for Scenario: **{results['scenario_key']}**")
                    st.metric("Current Price", f"₹{results['current_price']:.2f}")
                    st.metric("Stressed Price", f"₹{results['stressed_price']:.2f}")
                    st.metric("Potential Price Change", f"₹{(results['stressed_price'] - results['current_price']):.2f}")
                    st.metric("Percentage Change", f"{results['scenario_change_percent']:.2f}%")
                    st.info("This is a simplified stress test on a single instrument. Advanced stress testing would involve simulating multiple correlated factors across an entire portfolio.")
                else:
                    st.info("Run a stress test to see results here.")


# ---------------------------
# TAB: PERFORMANCE ANALYSIS (NEW!)
# ---------------------------
with tab_performance:
    st.header("Performance Analysis")
    st.markdown("Evaluate the historical performance of your selected instrument using standard financial metrics and compare it with a benchmark.")

    if not k:
        st.info("Login first to analyze performance.")
    else:
        historical_data = st.session_state.get("historical_data")
        last_symbol = st.session_state.get("last_fetched_symbol", "N/A")

        if historical_data is None or historical_data.empty:
            st.warning("No historical data available. Please fetch data for an instrument from the 'Market & Historical' tab first.")
        else:
            st.subheader(f"Performance Metrics for {last_symbol}")

            # Ensure data is ready
            historical_data['close'] = pd.to_numeric(historical_data['close'], errors='coerce')
            returns_series = historical_data['close'].pct_change().dropna() * 100
            
            if returns_series.empty or len(returns_series) < 2:
                st.error("Not enough valid data to compute performance metrics. Ensure historical data is fetched correctly.")
                st.stop()
            
            col_metrics, col_chart = st.columns([1,2])
            with col_metrics:
                risk_free_rate = st.number_input("Risk-Free Rate (Annualized %)", min_value=0.0, max_value=10.0, value=4.0, step=0.1, help="e.g., current interest rate on short-term government bonds.")
                performance_metrics = calculate_performance_metrics(returns_series, risk_free_rate)

                st.metric("Total Return", f"{performance_metrics.get('Total Return (%)', 0):.2f}%")
                st.metric("Annualized Return", f"{performance_metrics.get('Annualized Return (%)', 0):.2f}%")
                st.metric("Annualized Volatility", f"{performance_metrics.get('Annualized Volatility (%)', 0):.2f}%")
                st.metric("Sharpe Ratio", f"{performance_metrics.get('Sharpe Ratio', 0):.2f}")
                st.metric("Sortino Ratio", f"{performance_metrics.get('Sortino Ratio', 0):.2f}")
                st.metric("Max Drawdown", f"{performance_metrics.get('Max Drawdown (%)', 0):.2f}%")
            
            with col_chart:
                st.subheader("Cumulative Returns Comparison")
                fig_cum_returns = go.Figure()

                # Instrument Cumulative Returns
                cumulative_instrument_returns = (1 + returns_series / 100).cumprod() - 1
                fig_cum_returns.add_trace(go.Scatter(x=cumulative_instrument_returns.index, 
                                                     y=cumulative_instrument_returns * 100, 
                                                     mode='lines', 
                                                     name=f'{last_symbol} Cumulative Returns',
                                                     line=dict(color='blue', width=2)))
                
                # Benchmark comparison (using yfinance for NIFTY 50 as an example)
                st.markdown("---")
                st.subheader("Benchmark Comparison (e.g., NIFTY 50)")
                benchmark_symbol = st.text_input("Benchmark Symbol (e.g., ^NSEI for NIFTY 50 from Yahoo Finance)", "^NSEI")
                
                if st.button("Fetch & Compare Benchmark"):
                    with st.spinner(f"Fetching {benchmark_symbol} data..."):
                        try:
                            benchmark_data = yf.download(benchmark_symbol, start=historical_data.index.min().strftime('%Y-%m-%d'), end=historical_data.index.max().strftime('%Y-%m-%d'))
                            if not benchmark_data.empty:
                                benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna() * 100
                                # Align dates
                                common_dates = returns_series.index.intersection(benchmark_returns.index)
                                returns_series_aligned = returns_series.loc[common_dates]
                                benchmark_returns_aligned = benchmark_returns.loc[common_dates]

                                if not common_dates.empty and len(common_dates) > 1:
                                    cumulative_benchmark_returns = (1 + benchmark_returns_aligned / 100).cumprod() - 1
                                    fig_cum_returns.add_trace(go.Scatter(x=cumulative_benchmark_returns.index, 
                                                                         y=cumulative_benchmark_returns * 100, 
                                                                         mode='lines', 
                                                                         name=f'{benchmark_symbol} Cumulative Returns',
                                                                         line=dict(color='green', dash='dash', width=2)))
                                    
                                    # Calculate Alpha & Beta (simplified)
                                    # Need to re-align for regression
                                    df_for_alpha_beta = pd.DataFrame({'Asset': returns_series_aligned, 'Benchmark': benchmark_returns_aligned}).dropna()
                                    if len(df_for_alpha_beta) > 1:
                                        covariance = df_for_alpha_beta['Asset'].cov(df_for_alpha_beta['Benchmark'])
                                        benchmark_variance = df_for_alpha_beta['Benchmark'].var()
                                        beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan

                                        # Convert annualized returns to daily for Alpha calculation
                                        annual_asset_return = (1 + performance_metrics['Annualized Return (%)'] / 100)
                                        annual_benchmark_return = (1 + calculate_performance_metrics(benchmark_returns_aligned)['Annualized Return (%)'] / 100)

                                        # Simplified Jensen's Alpha (annualized)
                                        # Alpha = Asset_Return - [Risk_Free_Rate + Beta * (Benchmark_Return - Risk_Free_Rate)]
                                        alpha_annual = (annual_asset_return - (1 + risk_free_rate/100) - beta * (annual_benchmark_return - (1 + risk_free_rate/100))) * 100 if not np.isnan(beta) else np.nan

                                        st.markdown("##### Alpha and Beta")
                                        st.metric("Beta (vs. Benchmark)", f"{beta:.2f}" if not np.isnan(beta) else "N/A", help="Measures asset's volatility relative to the benchmark.")
                                        st.metric("Alpha (Annualized %)", f"{alpha_annual:.2f}%" if not np.isnan(alpha_annual) else "N/A", help="Measures excess return relative to the return of the benchmark asset.")
                                    else:
                                        st.warning("Not enough common data points between instrument and benchmark to calculate Alpha/Beta.")
                                else:
                                    st.warning("No common historical data points with benchmark to compare.")
                            else:
                                st.warning(f"Could not fetch benchmark data for {benchmark_symbol}. Check symbol or date range.")
                        except Exception as e:
                            st.error(f"Error fetching benchmark data: {e}")

                fig_cum_returns.update_layout(title_text=f"Cumulative Returns: {last_symbol} vs. Benchmark",
                                              xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                                              template="plotly_white", hovermode="x unified", height=500)
                st.plotly_chart(fig_cum_returns, use_container_width=True)

# ---------------------------
# TAB: MULTI-ASSET ANALYSIS (NEW!)
# ---------------------------
with tab_multi_asset:
    st.header("Multi-Asset Analysis: Correlation & Diversification")
    st.markdown("Analyze relationships between multiple financial instruments. Understanding correlations is key for portfolio diversification.")

    if not k:
        st.info("Login first to perform multi-asset analysis.")
    else:
        st.subheader("Select Instruments for Analysis")
        
        # User input for multiple symbols
        selected_symbols_str = st.text_area("Enter Trading Symbols (comma-separated, e.g., INFY,RELIANCE,TCS,NIFTY 50)", "INFY,RELIANCE,TCS,NIFTY 50", height=80)
        symbols_to_analyze = [s.strip().upper() for s in selected_symbols_str.split(',') if s.strip()]
        
        multi_asset_exchange = st.selectbox("Exchange for all symbols", ["NSE", "BSE", "NFO"], index=0, key="multi_asset_exchange")
        multi_asset_interval = st.selectbox("Interval for historical data", ["day", "week", "month"], index=0, key="multi_asset_interval")
        
        default_to_date_multi = datetime.now().date()
        default_from_date_multi = default_to_date_multi - timedelta(days=365) # Default to 1 year

        from_date_multi = st.date_input("From Date", value=default_from_date_multi, key="from_dt_multi")
        to_date_multi = st.date_input("To Date", value=default_to_date_multi, key="to_dt_multi")

        if st.button("Fetch Multi-Asset Data & Analyze"):
            if not st.session_state.get("instruments_df"):
                st.warning("Please load instruments in 'Market & Historical' or 'Instruments Utils' tab first.")
                st.stop()
            
            all_historical_data = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(symbols_to_analyze):
                status_text.text(f"Fetching historical data for {symbol} ({i+1}/{len(symbols_to_analyze)})...")
                hist_data = get_historical(k, symbol, from_date_multi, to_date_multi, multi_asset_interval, multi_asset_exchange)
                
                if "error" in hist_data:
                    st.error(f"Error fetching data for {symbol}: {hist_data['error']}")
                else:
                    df = pd.DataFrame(hist_data)
                    if not df.empty:
                        df["date"] = pd.to_datetime(df["date"])
                        df.set_index("date", inplace=True)
                        df.sort_index(inplace=True)
                        df['close'] = pd.to_numeric(df['close'], errors='coerce') # Ensure close is numeric
                        df.dropna(subset=['close'], inplace=True)
                        all_historical_data[symbol] = df['close']
                    else:
                        st.warning(f"No historical data found for {symbol}. Skipping.")
                progress_bar.progress((i + 1) / len(symbols_to_analyze))
            
            progress_bar.empty()
            status_text.empty()

            if len(all_historical_data) < 2:
                st.error("Please select at least two instruments with available historical data for correlation analysis.")
            else:
                # Combine close prices into a single DataFrame
                combined_df = pd.DataFrame(all_historical_data)
                combined_df.dropna(inplace=True)

                if combined_df.empty or len(combined_df) < 2:
                    st.error("No common historical data points for selected instruments. Try adjusting date range or symbols.")
                    st.session_state["multi_asset_returns"] = None
                    st.session_state["multi_asset_correlation"] = None
                else:
                    returns_df = combined_df.pct_change().dropna()
                    st.session_state["multi_asset_returns"] = returns_df
                    st.session_state["multi_asset_correlation"] = returns_df.corr()
                    st.success("Multi-asset data fetched and correlations calculated.")
                    st.dataframe(combined_df.head(), use_container_width=True)

        if st.session_state.get("multi_asset_correlation") is not None:
            st.subheader("Correlation Matrix (Daily Returns)")
            st.write("A correlation of +1 means assets move in the same direction, -1 means they move in opposite directions, and 0 means no linear relationship.")
            
            correlation_matrix = st.session_state["multi_asset_correlation"]
            st.dataframe(correlation_matrix.style.background_gradient(cmap='RdBu', axis=None).format(precision=2), use_container_width=True)

            fig_corr_heatmap = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1
                ))
            fig_corr_heatmap.update_layout(title_text='Correlation Heatmap',
                                          xaxis_title="Instrument", yaxis_title="Instrument",
                                          template="plotly_white", height=600)
            st.plotly_chart(fig_corr_heatmap, use_container_width=True)
            
            st.info("Assets with low or negative correlation are good candidates for diversification, as they may reduce overall portfolio risk.")

# ---------------------------
# TAB: WEBSOCKET (Ticker)
# ---------------------------
with tab_ws:
    st.header("WebSocket Streaming — Live Ticks")
    st.markdown("Receive real-time market data ticks via Zerodha KiteTicker. Visualize live price movements.")

    if not k:
        st.info("Login first to start websocket.")
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
        if "kt_live_prices" not in st.session_state: # For live plot
            st.session_state["kt_live_prices"] = pd.DataFrame(columns=['timestamp', 'last_price', 'instrument_token'])

        with st.expander("Lookup Instrument Token for WebSocket Subscription"):
            # Autoload instruments if not already loaded, for convenience in getting tokens
            if "instruments_df" not in st.session_state or st.session_state["instruments_df"].empty:
                st.info("Loading instruments for NSE to facilitate instrument token lookup for WebSocket.")
                nse_instruments = load_instruments(k, "NSE") 
                if not nse_instruments.empty:
                    st.session_state["instruments_df"] = nse_instruments
                    st.success(f"Loaded {len(nse_instruments)} instruments for NSE.")
                else:
                    st.warning("Could not load NSE instruments. WebSocket token lookup might be limited.")
            
            ws_exchange = st.selectbox("Exchange for Symbol Lookup", ["NSE", "BSE", "NFO"], index=0, key="ws_lookup_ex")
            ws_tradingsymbol = st.text_input("Tradingsymbol (e.g., INFY)", value="INFY", key="ws_lookup_sym")
            
            instrument_token_for_ws = None
            if st.button("Lookup Token"):
                if "instruments_df" in st.session_state and not st.session_state["instruments_df"].empty:
                    instrument_token_for_ws = find_instrument_token(st.session_state["instruments_df"], ws_tradingsymbol, ws_exchange)
                    if instrument_token_for_ws:
                        st.success(f"Found instrument_token for {ws_tradingsymbol}: **{instrument_token_for_ws}**")
                        st.session_state["ws_instrument_token_input"] = str(instrument_token_for_ws) # Store as string for input
                        st.session_state["ws_instrument_name"] = ws_tradingsymbol # Store name for plot title
                    else:
                        st.warning(f"Could not find instrument token for {ws_tradingsymbol} on {ws_exchange}.")
                else:
                    st.warning("Please load instruments first from 'Market & Historical' tab or 'Instruments Utils' tab.")

        symbol_for_ws = st.text_input("Instrument token(s) (comma separated, e.g., 738561,3409)", 
                                      value=st.session_state.get("ws_instrument_token_input", ""),
                                      key="ws_symbol_input")
        st.caption("Enter numeric instrument token(s) or use the lookup above. Leave blank to subscribe none initially.")

        col_ws_controls, col_ws_status = st.columns(2)
        with col_ws_controls:
            if st.button("Start Ticker (Subscribe)", help="Start the WebSocket connection and subscribe to selected tokens.") and not st.session_state["kt_running"]:
                try:
                    access_token = st.session_state["kite_access_token"]
                    user_id = st.session_state["kite_login_response"].get("user_id")
                    
                    try:
                        kt = KiteTicker(user_id, access_token, API_KEY)
                    except Exception:
                        kt = KiteTicker(API_KEY, access_token)

                    st.session_state["kt_ticker"] = kt
                    st.session_state["kt_running"] = True
                    st.session_state["kt_ticks"] = []
                    st.session_state["kt_live_prices"] = pd.DataFrame(columns=['timestamp', 'last_price', 'instrument_token'])
                    st.session_state["kt_status_message"] = "Ticker connecting..."

                    def on_connect(ws, response):
                        st.session_state["kt_ticks"].append({"event": "connected", "time": datetime.utcnow().isoformat()})
                        st.session_state["kt_status_message"] = "Ticker connected. Subscribing..."
                        if symbol_for_ws:
                            tokens = [int(x.strip()) for x in symbol_for_ws.split(",") if x.strip()]
                            if tokens:
                                try:
                                    ws.subscribe(tokens)
                                    ws.set_mode(ws.MODE_FULL, tokens)
                                    st.session_state["kt_status_message"] = f"Subscribed to {len(tokens)} tokens ({', '.join(map(str, tokens))})."
                                except Exception as e:
                                    st.session_state["kt_ticks"].append({"event": "subscribe_error", "error": str(e), "time": datetime.utcnow().isoformat()})
                                    st.session_state["kt_status_message"] = f"Subscription error: {e}"
                            else:
                                st.session_state["kt_status_message"] = "Connected, but no tokens provided for subscription."
                        else:
                            st.session_state["kt_status_message"] = "Connected, no tokens specified for initial subscription."

                    def on_ticks(ws, ticks):
                        for t in ticks:
                            t["_ts"] = datetime.utcnow().isoformat()
                            st.session_state["kt_ticks"].append(t)
                            
                            # Update live prices for plotting (only for full mode ticks with last_price)
                            if 'last_price' in t and 'instrument_token' in t:
                                new_row = pd.DataFrame([{'timestamp': datetime.now(), 'last_price': t['last_price'], 'instrument_token': t['instrument_token']}])
                                # Append efficiently without full dataframe re-creation if possible, or limit size
                                if len(st.session_state["kt_live_prices"]) > 500: # Limit history for plot
                                    st.session_state["kt_live_prices"] = st.session_state["kt_live_prices"].iloc[1:] # Drop oldest
                                st.session_state["kt_live_prices"] = pd.concat([st.session_state["kt_live_prices"], new_row], ignore_index=True)
                        
                        if len(st.session_state["kt_ticks"]) > 200:
                            st.session_state["kt_ticks"] = st.session_state["kt_ticks"][-200:]
                        st.session_state["_last_tick_update"] = time.time() # Trigger redraw

                    def on_close(ws, code, reason):
                        st.session_state["kt_ticks"].append({"event": "closed", "code": code, "reason": reason, "time": datetime.utcnow().isoformat()})
                        st.session_state["kt_running"] = False
                        st.session_state["kt_status_message"] = f"Ticker disconnected: {reason} (Code: {code})"

                    def on_error(ws, code, reason):
                        st.session_state["kt_ticks"].append({"event": "error", "code": code, "reason": reason, "time": datetime.utcnow().isoformat()})
                        st.session_state["kt_status_message"] = f"Ticker error: {reason} (Code: {code})"

                    kt.on_connect = on_connect
                    kt.on_ticks = on_ticks
                    kt.on_close = on_close
                    kt.on_error = on_error

                    def run_ticker():
                        try:
                            kt.connect(threaded=True)
                            while st.session_state["kt_running"]:
                                time.sleep(0.1) # Small sleep to yield to other threads/Streamlit
                        except Exception as e:
                            st.session_state["kt_ticks"].append({"event": "fatal_error", "error": str(e)})
                            st.session_state["kt_running"] = False
                            st.session_state["kt_status_message"] = f"Fatal ticker error: {e}"

                    th = threading.Thread(target=run_ticker, daemon=True)
                    st.session_state["kt_thread"] = th
                    th.start()
                    st.success("Ticker start attempt initiated. Check status below and 'Latest Ticks' table.")
                except Exception as e:
                    st.error(f"Failed to start ticker: {e}")
                    st.session_state["kt_status_message"] = f"Failed to start ticker: {e}"

        with col_ws_status:
            if st.button("Stop Ticker", help="Disconnect the WebSocket connection.") and st.session_state.get("kt_running"):
                try:
                    kt = st.session_state.get("kt_ticker")
                    if kt:
                        kt.disconnect()
                    st.session_state["kt_running"] = False
                    st.session_state["kt_status_message"] = "Ticker explicitly stopped."
                    st.success("Ticker stopped.")
                except Exception as e:
                    st.error(f"Failed to stop ticker: {e}")
                    st.session_state["kt_status_message"] = f"Failed to stop ticker: {e}"
            st.info(f"**Ticker Status:** {st.session_state.get('kt_status_message', 'Not started')}")
            if st.session_state.get("kt_running"):
                st.markdown("💡 *Ticker is running in a background thread. Keep this tab open to receive ticks.*")


        st.markdown("---")
        st.subheader("Live Price Chart (First Subscribed Token)")
        live_chart_placeholder = st.empty()
        
        # Continuously update the live chart
        if st.session_state.get("kt_running") and not st.session_state["kt_live_prices"].empty:
            df_live = st.session_state["kt_live_prices"]
            # Filter for the first token subscribed (or primary token if multiple)
            if not df_live.empty:
                first_token = df_live['instrument_token'].iloc[0]
                df_live_filtered = df_live[df_live['instrument_token'] == first_token]

                if not df_live_filtered.empty:
                    fig_live = go.Figure()
                    fig_live.add_trace(go.Scatter(x=df_live_filtered['timestamp'], y=df_live_filtered['last_price'], mode='lines+markers', name='Last Price', line=dict(color='blue')))
                    
                    # Try to get instrument name from session state
                    inst_name = st.session_state.get("ws_instrument_name", f"Token {first_token}")
                    fig_live.update_layout(title_text=f"Live LTP for {inst_name}",
                                        xaxis_title="Time",
                                        yaxis_title="Price",
                                        template="plotly_white",
                                        height=400,
                                        xaxis_rangeslider_visible=False)
                    live_chart_placeholder.plotly_chart(fig_live, use_container_width=True)
                else:
                    live_chart_placeholder.info("Waiting for live price data for plotting...")
            else:
                 live_chart_placeholder.info("Waiting for live price data for plotting...")
        else:
            live_chart_placeholder.info("Start the ticker to see live price updates.")

        st.markdown("---")
        st.subheader("Latest Ticks Data Table")
        tick_data_placeholder = st.empty()
        
        # This block will re-run automatically due to `_last_tick_update` in on_ticks
        ticks = st.session_state.get("kt_ticks", [])
        if ticks:
            df_ticks = pd.json_normalize(ticks[-100:][::-1]) # Show most recent 100 ticks
            display_cols = ['_ts', 'instrument_token', 'last_price', 'ohlc.open', 'ohlc.high', 'ohlc.low', 'ohlc.close', 'volume', 'change']
            available_cols = [col for col in display_cols if col in df_ticks.columns]
            tick_data_placeholder.dataframe(df_ticks[available_cols], use_container_width=True)
        else:
            tick_data_placeholder.write("No ticks yet. Start ticker and/or subscribe tokens.")

# ---------------------------
# TAB: INSTRUMENTS UTILS
# ---------------------------
with tab_inst:
    st.header("Instrument Lookup and Utilities")
    st.markdown("Find instrument tokens, which are essential for fetching historical data or subscribing to live market data.")
    
    inst_exchange = st.selectbox("Select Exchange to Load Instruments", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0)
    if st.button("Load Instruments for Selected Exchange (cached)", help="Fetching instruments can take a moment, especially for large exchanges. Data is cached."):
        try:
            df = load_instruments(k, inst_exchange)
            st.session_state["instruments_df"] = df
            if not df.empty:
                st.success(f"Loaded {len(df)} instruments for {inst_exchange}.")
            else:
                st.warning(f"Could not load instruments for {inst_exchange}. Check API permissions or if the exchange has instruments available.")
        except Exception as e:
            st.error(f"Failed to load instruments: {e}")

    df_instruments = st.session_state.get("instruments_df", pd.DataFrame())
    if not df_instruments.empty:
        st.subheader("Search Instrument Token by Symbol")
        col_search_inst, col_search_results = st.columns([1,2])
        with col_search_inst:
            search_symbol = st.text_input(f"Enter Tradingsymbol (e.g., INFY for {inst_exchange})", value="INFY", key="inst_search_sym")
            search_exchange = st.selectbox("Specify Exchange for Search", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0, key="inst_search_ex")
            
            if st.button("Find Token"):
                token = find_instrument_token(df_instruments, search_symbol, search_exchange)
                if token:
                    st.session_state["last_found_token"] = token
                    st.session_state["last_found_symbol"] = search_symbol
                    st.session_state["last_found_exchange"] = search_exchange
                    st.success(f"Found instrument_token for {search_symbol} on {search_exchange}: **{token}**")
                else:
                    st.warning(f"Instrument token not found for '{search_symbol}' on '{search_exchange}'. Ensure correct symbol/exchange and that instruments for this exchange are loaded.")
        
        with col_search_results:
            if st.session_state.get("last_found_token"):
                st.markdown("##### Details for Last Found Instrument")
                token_details = df_instruments[df_instruments['instrument_token'] == st.session_state["last_found_token"]]
                if not token_details.empty:
                    st.dataframe(token_details, use_container_width=True)
                else:
                    st.info("No detailed data for the last found token.")

        st.subheader("Preview Loaded Instruments (First 200 Rows)")
        st.dataframe(df_instruments.head(200), use_container_width=True)
    else:
        st.info("No instruments loaded. Click 'Load Instruments for Selected Exchange' above to fetch.")
