# streamlit_kite_app.py
# Full-featured Streamlit frontend for Zerodha Kite Connect.
# Requirements: streamlit, pykiteconnect, pandas
# Put your Kite credentials in Streamlit secrets under [kite] as api_key, api_secret, redirect_uri

import streamlit as st
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker  # websocket ticker
import pandas as pd
import json
import threading
import time
from datetime import datetime

st.set_page_config(page_title="Kite Connect - Full demo", layout="wide")
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
        st.download_button("⬇️ Download token JSON", json.dumps(data, default=str), file_name="kite_token.json")
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
@st.cache_data(show_spinner=False)
def load_instruments(exchange=None):
    """
    Returns pandas.DataFrame of instrument dump.
    If exchange is None, tries to fetch all instruments (may be large).
    """
    try:
        if exchange:
            inst = k.instruments(exchange)
        else:
            # call without exchange may return full dump
            inst = k.instruments()
        df = pd.DataFrame(inst)
        # keep token as int
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        return df
    except Exception as e:
        st.warning(f"Could not fetch instruments: {e}")
        return pd.DataFrame()

def find_instrument_token(df, exchange, tradingsymbol):
    """Lookup instrument_token given exchange and tradingsymbol (case-insensitive)."""
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    if not hits.empty:
        return int(hits.iloc[0]["instrument_token"])
    return None

# ---------------------------
# Sidebar quick actions / profile / logout
# ---------------------------
with st.sidebar:
    st.header("Account")
    if k:
        try:
            profile = k.profile()
            st.write("User:", profile.get("user_name") or profile.get("user_id"))
            st.write("User ID:", profile.get("user_id"))
            st.write("Login time:", profile.get("login_time"))
        except Exception:
            st.write("Authenticated (profile fetch failed)")

        if st.button("Logout (clear token)"):
            st.session_state.pop("kite_access_token", None)
            st.session_state.pop("kite_login_response", None)
            st.success("Logged out. Please login again.")
            st.experimental_rerun()
    else:
        st.info("Not authenticated yet. Login using the link above.")

# ---------------------------
# Main UI - Tabs for modules
# ---------------------------
tabs = st.tabs(["Portfolio", "Orders", "Market & Historical", "Websocket (stream)", "Mutual Funds", "Instruments Dump & Utils", "Admin/Debug"])
tab_portfolio, tab_orders, tab_market, tab_ws, tab_mf, tab_inst, tab_debug = tabs

# ---------------------------
# TAB: PORTFOLIO
# ---------------------------
with tab_portfolio:
    st.header("Portfolio")
    if not k:
        st.info("Login first to fetch portfolio data.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Fetch holdings"):
                try:
                    holdings = k.holdings()
                    st.dataframe(pd.DataFrame(holdings))
                except Exception as e:
                    st.error(f"Error fetching holdings: {e}")
        with col2:
            if st.button("Fetch positions"):
                try:
                    positions = k.positions()
                    # positions contains 'net' and 'day'
                    st.subheader("Net positions")
                    st.dataframe(pd.DataFrame(positions.get("net", [])))
                    st.subheader("Day positions")
                    st.dataframe(pd.DataFrame(positions.get("day", [])))
                except Exception as e:
                    st.error(f"Error fetching positions: {e}")
        with col3:
            if st.button("Fetch margins"):
                try:
                    margins = k.margins()
                    st.json(margins)
                except Exception as e:
                    st.error(f"Error fetching margins: {e}")

# ---------------------------
# TAB: ORDERS
# ---------------------------
with tab_orders:
    st.header("Orders — place / modify / cancel / list")

    if not k:
        st.info("Login first to use orders API.")
    else:
        st.subheader("Place order")
        with st.form("place_order_form", clear_on_submit=False):
            variety = st.selectbox("Variety", ["regular", "amo", "co", "iceberg"], index=0)
            exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0)
            tradingsymbol = st.text_input("Tradingsymbol (e.g. INFY / NIFTY21...)", value="INFY")
            transaction_type = st.selectbox("Transaction", ["BUY", "SELL"], index=0)
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"], index=0)
            quantity = st.number_input("Quantity", min_value=1, value=1)
            product = st.selectbox("Product", ["CNC", "MIS", "NRML", "CO", "MTF"], index=0)
            price = st.text_input("Price (for LIMIT/SL)", value="")
            trigger_price = st.text_input("Trigger Price (for SL/SL-M)", value="")
            validity = st.selectbox("Validity", ["DAY", "IOC", "TTL"], index=0)
            tag = st.text_input("Tag (optional, max 20 chars)", value="")
            submit_place = st.form_submit_button("Place order")

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

                    # place order
                    resp = k.place_order(**params)
                    st.success(f"Order placed: {resp}")
                    st.json(resp)
                except Exception as e:
                    st.error(f"Place order failed: {e}")

        st.markdown("---")
        st.subheader("Modify / Cancel / Fetch orders")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("Fetch all orders (today)"):
                try:
                    orders = k.orders()
                    st.dataframe(pd.DataFrame(orders))
                except Exception as e:
                    st.error(f"Error fetching orders: {e}")

            if st.button("Fetch all trades (today)"):
                try:
                    trades = k.trades()
                    st.dataframe(pd.DataFrame(trades))
                except Exception as e:
                    st.error(f"Error fetching trades: {e}")

        with col_b:
            order_id_for_history = st.text_input("Order ID (history / modify / cancel)", value="")
            if st.button("Get order history"):
                if not order_id_for_history:
                    st.warning("Provide order_id")
                else:
                    try:
                        history = k.order_history(order_id_for_history)
                        st.json(history)
                    except Exception as e:
                        st.error(f"Get order history failed: {e}")

        with col_c:
            mod_order_id = st.text_input("Modify order id", value="")
            new_price = st.text_input("New price", value="")
            new_qty = st.text_input("New qty", value="")
            if st.button("Modify order"):
                if not mod_order_id:
                    st.warning("Provide order id")
                else:
                    try:
                        modify_args = {}
                        if new_price:
                            modify_args["price"] = float(new_price)
                        if new_qty:
                            modify_args["quantity"] = int(new_qty)
                        # note: variety is required for modify; here we assume 'regular' but user can change
                        res = k.modify_order(variety="regular", order_id=mod_order_id, **modify_args)
                        st.success("Modify response")
                        st.json(res)
                    except Exception as e:
                        st.error(f"Modify failed: {e}")

            if st.button("Cancel order"):
                cid = st.text_input("Cancel order id (re-enter)", value="")
                if cid:
                    try:
                        res = k.cancel_order(variety="regular", order_id=cid)
                        st.success("Cancel response")
                        st.json(res)
                    except Exception as e:
                        st.error(f"Cancel failed: {e}")

# ---------------------------
# TAB: MARKET & HISTORICAL
# ---------------------------
with tab_market:
    st.header("Market: Quotes & Historical data")

    if not k:
        st.info("Login first to fetch market data (quotes/historical).")
    else:
        st.subheader("Quote / LTP")
        q_exchange = st.selectbox("Exchange for quote", ["NSE", "BSE", "NFO"], index=0)
        q_symbol = st.text_input("Tradingsymbol (eg INFY)", value="INFY")
        if st.button("Get quote / LTP"):
            try:
                # Quote typically expects keys like "NSE:INFY" or instrument tokens list
                key = f"{q_exchange}:{q_symbol}"
                quote = k.quote(key)
                st.json(quote)
            except Exception as e:
                st.error(f"Quote failed: {e}")

        st.markdown("---")
        st.subheader("Historical candles")
        # Load instruments (cached)
        with st.expander("Instrument dump (load)"):
            exchange_for_dump = st.selectbox("Instrument dump exchange (for lookup)", ["NSE", "BSE", "NFO", "BCD", "MCX"], index=0, key="inst_exchange")
            if st.button("Load instrument dump"):
                inst_df = load_instruments(exchange_for_dump)
                st.session_state["instruments_df"] = inst_df
                st.success(f"Loaded {len(inst_df)} instruments for {exchange_for_dump}")

        inst_df = st.session_state.get("instruments_df", pd.DataFrame())

        hist_exchange = st.selectbox("Exchange (for historical)", ["NSE", "BSE", "NFO"], index=0, key="hist_ex")
        hist_symbol = st.text_input("Historical tradingsymbol (eg INFY)", value="INFY", key="hist_sym")
        from_date = st.date_input("From date", key="from_dt")
        to_date = st.date_input("To date", key="to_dt")
        interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "day", "week", "month"], index=4)

        if st.button("Fetch historical data"):
            try:
                # Need instrument_token numeric. Try to find from inst_df if loaded.
                token = None
                if not inst_df.empty:
                    token = find_instrument_token(inst_df, hist_exchange, hist_symbol)
                if token is None:
                    st.warning("Instrument token not found in loaded dump. Please load instrument dump for that exchange or provide token manually.")
                    manual_token = st.text_input("Manual instrument_token (optional)", value="")
                    if manual_token:
                        token = int(manual_token)
                if token is None:
                    st.stop()

                # pykiteconnect historical_data expects iso/datetime strings
                start = datetime.combine(from_date, datetime.min.time()).isoformat()
                end = datetime.combine(to_date, datetime.max.time()).isoformat()
                candles = k.historical_data(token, from_date=start, to_date=end, interval=interval)
                df = pd.DataFrame(candles)
                if not df.empty:
                    # normalize datetime
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    st.dataframe(df)
                else:
                    st.write("No historical data returned.")
            except Exception as e:
                st.error(f"Historical fetch failed: {e}")

# ---------------------------
# TAB: WEBSOCKET (Ticker)
# ---------------------------
with tab_ws:
    st.header("WebSocket streaming — KiteTicker (authenticated)")
    st.write("Start the KiteTicker to receive live ticks. This component uses threads — click Start, then Stop to disconnect.")

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

        symbol_for_ws = st.text_input("Instrument token(s) comma separated (e.g. 738561,3409) OR use instrument dump lookup", value="")
        st.caption("Note: provide numeric instrument_token(s) or leave blank to subscribe none (you can subscribe later).")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start ticker") and not st.session_state["kt_running"]:
                try:
                    # create ticker
                    access_token = st.session_state["kite_access_token"]
                    user_id = st.session_state["kite_login_response"].get("user_id")
                    # KiteTicker signature: KiteTicker(api_key, access_token) or (user_id, access_token, api_key) depending on version
                    try:
                        kt = KiteTicker(user_id, access_token, API_KEY)  # newer signature
                    except Exception:
                        kt = KiteTicker(API_KEY, access_token)  # fallback older signature

                    st.session_state["kt_ticker"] = kt
                    st.session_state["kt_running"] = True
                    st.session_state["kt_ticks"] = []

                    # define callbacks
                    def on_connect(ws, response):
                        st.session_state["kt_ticks"].append({"event": "connected", "time": datetime.utcnow().isoformat()})
                        # subscribe if tokens provided
                        if symbol_for_ws:
                            tokens = [int(x.strip()) for x in symbol_for_ws.split(",") if x.strip()]
                            try:
                                ws.subscribe(tokens)
                            except Exception:
                                # some versions accept ws.subscribe or ws.subscribe(tokens)
                                pass
                        ws.set_mode(ws.MODE_FULL, [])  # attempt to set mode (may require tokens list)

                    def on_ticks(ws, ticks):
                        # append latest ticks (limit to 200)
                        for t in ticks:
                            t["_ts"] = datetime.utcnow().isoformat()
                            st.session_state["kt_ticks"].append(t)
                        if len(st.session_state["kt_ticks"]) > 200:
                            st.session_state["kt_ticks"] = st.session_state["kt_ticks"][-200:]

                    def on_close(ws, code, reason):
                        st.session_state["kt_ticks"].append({"event": "closed", "code": code, "reason": reason, "time": datetime.utcnow().isoformat()})
                        st.session_state["kt_running"] = False

                    # bind callbacks (function names depend on pykiteconnect version)
                    try:
                        kt.on_connect = on_connect
                        kt.on_ticks = on_ticks
                        kt.on_close = on_close
                    except Exception:
                        kt.on_connect = on_connect
                        kt.on_ticks = on_ticks
                        kt.on_close = on_close

                    # run ticker in a background thread
                    def run_ticker():
                        try:
                            kt.connect(threaded=True)
                            # connect(threaded=True) will start internal loop; keep this thread alive waiting for stop
                            while st.session_state["kt_running"]:
                                time.sleep(0.5)
                        except Exception as e:
                            st.session_state["kt_ticks"].append({"event": "error", "error": str(e)})
                            st.session_state["kt_running"] = False

                    th = threading.Thread(target=run_ticker, daemon=True)
                    st.session_state["kt_thread"] = th
                    th.start()
                    st.success("Ticker started (background thread).")
                except Exception as e:
                    st.error(f"Failed to start ticker: {e}")

        with col2:
            if st.button("Stop ticker") and st.session_state.get("kt_running"):
                try:
                    kt = st.session_state.get("kt_ticker")
                    if kt:
                        try:
                            kt.disconnect()
                        except Exception:
                            try:
                                kt.stop()
                            except Exception:
                                pass
                    st.session_state["kt_running"] = False
                    st.success("Ticker stopped.")
                except Exception as e:
                    st.error(f"Failed to stop ticker: {e}")

        st.markdown("#### Latest ticks (most recent 100)")
        ticks = st.session_state.get("kt_ticks", [])
        if ticks:
            # show last 50 in reverse order (most recent first)
            df_ticks = pd.json_normalize(ticks[-100:][::-1])
            st.dataframe(df_ticks)
        else:
            st.write("No ticks yet. Start ticker and/or subscribe tokens.")

# ---------------------------
# TAB: MUTUAL FUNDS
# ---------------------------
with tab_mf:
    st.header("Mutual funds")

    if not k:
        st.info("Login first to use mutual funds APIs.")
    else:
        st.subheader("MF Instruments")
        col1, col2 = st.columns([2,1])
        with col1:
            if st.button("Load MF instruments"):
                try:
                    mf_inst = k.get_mf_instruments()
                    st.session_state["mf_instruments"] = pd.DataFrame(mf_inst)
                    st.success(f"Loaded {len(mf_inst)} mutual fund instruments")
                except Exception as e:
                    st.error(f"Error loading MF instruments: {e}")
        with col2:
            if st.button("Show saved MF instruments"):
                df = st.session_state.get("mf_instruments", pd.DataFrame())
                if df.empty:
                    st.info("No MF instruments loaded yet.")
                else:
                    st.dataframe(df.head(200))

        st.markdown("---")
        st.subheader("Place MF order (SIP / Lumpsum)")
        with st.form("place_mf"):
            tradingsymbol = st.text_input("Tradingsymbol (scheme code or folio id)", value="")
            transaction_type = st.selectbox("Transaction type", ["BUY", "SELL"], index=0)
            quantity = st.number_input("Quantity (units)", min_value=0.0, format="%.3f", value=0.0)
            amount = st.text_input("Amount (for lumpsum)", value="")
            # If using place_mf_order, pass relevant args per API docs
            submit_mf = st.form_submit_button("Place MF order")
            if submit_mf:
                try:
                    mf_args = {
                        # example keys; actual required keys depend on Kite's MF API.
                        "tradingsymbol": tradingsymbol,
                        "transaction_type": transaction_type,
                    }
                    if quantity and quantity > 0:
                        mf_args["quantity"] = float(quantity)
                    if amount:
                        mf_args["amount"] = float(amount)
                    resp = k.place_mf_order(**mf_args)
                    st.success("MF order response")
                    st.json(resp)
                except Exception as e:
                    st.error(f"Place MF order failed (check required params): {e}")

        st.markdown("---")
        if st.button("Get MF orders"):
            try:
                mf_orders = k.get_mf_orders()
                st.dataframe(pd.DataFrame(mf_orders))
            except Exception as e:
                st.error(f"Get MF orders failed: {e}")

# ---------------------------
# TAB: INSTRUMENTS DUMP & UTILS
# ---------------------------
with tab_inst:
    st.header("Instruments dump & helper utilities")
    inst_exchange = st.selectbox("Load instruments for exchange", ["NSE", "BSE", "NFO", "BCD", "MCX"], index=0)
    if st.button("Load instruments for exchange (cached)"):
        try:
            df = load_instruments(inst_exchange)
            st.session_state["instruments_df"] = df
            st.success(f"Loaded {len(df)} instruments for {inst_exchange}")
        except Exception as e:
            st.error(f"Load instruments failed: {e}")

    df = st.session_state.get("instruments_df", pd.DataFrame())
    if not df.empty:
        st.write("Search by trading symbol & exchange")
        sy = st.text_input("Symbol to search (tradingsymbol)", value="INFY", key="inst_search_sym")
        if st.button("Find instrument token"):
            token = find_instrument_token(df, inst_exchange, sy)
            if token:
                st.success(f"Found instrument_token: {token}")
            else:
                st.warning("Not found. Try loading correct exchange dump or exact tradingsymbol.")

        st.markdown("Preview instruments (first 200 rows)")
        st.dataframe(df.head(200))
    else:
        st.info("No instruments loaded. Click Load instruments to fetch.")

# ---------------------------
# TAB: ADMIN / DEBUG
# ---------------------------
with tab_debug:
    st.header("Admin / debug")
    st.write("Session keys (sensitive values hidden):")
    safe_view = {k: (type(v).__name__ if k != "kite_login_response" else "login_response_present") for k, v in st.session_state.items()}
    st.json(safe_view)
    st.markdown("---")
    if st.button("Show raw login response (dangerous)"):
        lr = st.session_state.get("kite_login_response")
        st.write(json.dumps(lr, default=str, indent=2))
    st.markdown("---")
    st.write("Library versions:")
    try:
        import kiteconnect, pkg_resources
        st.write("pykiteconnect:", pkg_resources.get_distribution("kiteconnect").version)
    except Exception:
        st.write("pykiteconnect not found or version unknown")

    st.markdown("## Notes")
    st.write("""
    - request_token is single-use and short-lived. If you see `Token is invalid or has expired`, re-login and exchange immediately.
    - For production: exchange request_token on a secure server; don't keep api_secret in public client code.
    - If MF / some endpoints methods raise AttributeError, check your pykiteconnect version and refer to docs for exact method names (they map closely to the HTTP endpoints).
    """)

# End of file
