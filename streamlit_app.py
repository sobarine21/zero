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
def load_instruments(kite_instance, exchange=None):
    try:
        if exchange:
            inst = kite_instance.instruments(exchange)
        else:
            inst = kite_instance.instruments()
        df = pd.DataFrame(inst)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        return df
    except Exception as e:
        st.warning(f"Could not fetch instruments: {e}")
        return pd.DataFrame()

def find_instrument_token(df, tradingsymbol, exchange="NSE"):
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    if not hits.empty:
        return int(hits.iloc[0]["instrument_token"])
    return None

# ---------------------------
# Market Data Helpers
# ---------------------------
def get_ltp_price(kite_instance, symbol, exchange="NSE"):
    """Get Last Traded Price (LTP)"""
    try:
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        ltp_data = kite_instance.ltp([exchange_symbol])
        return ltp_data.get(exchange_symbol, ltp_data)
    except Exception as e:
        return {"error": str(e)}

def get_ohlc_quote(kite_instance, symbol, exchange="NSE"):
    """Get OHLC + LTP"""
    try:
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        ohlc_data = kite_instance.ohlc([exchange_symbol])
        return ohlc_data.get(exchange_symbol, ohlc_data)
    except Exception as e:
        return {"error": str(e)}

def get_full_market_quote(kite_instance, symbol, exchange="NSE"):
    """Full market quote (OHLC, depth, OI)"""
    try:
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        quote = kite_instance.quote([exchange_symbol])
        return quote.get(exchange_symbol, quote)
    except Exception as e:
        return {"error": str(e)}

def get_historical(kite_instance, symbol, from_date, to_date, interval="day", exchange="NSE"):
    try:
        inst_df = st.session_state.get("instruments_df", pd.DataFrame())
        token = find_instrument_token(inst_df, symbol, exchange) if not inst_df.empty else None

        if token is None:
            all_instruments = load_instruments(kite_instance, exchange)
            if not all_instruments.empty:
                st.session_state["instruments_df"] = all_instruments
                token = find_instrument_token(all_instruments, symbol, exchange)

        if not token:
            return {"error": f"Instrument token not found for {symbol} on {exchange}"}

        from_datetime = datetime.combine(from_date, datetime.min.time())
        to_datetime = datetime.combine(to_date, datetime.max.time())
        data = kite_instance.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval=interval)
        return data
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Account")
    if k:
        try:
            profile = k.profile()
            st.write("User:", profile.get("user_name") or profile.get("user_id"))
            st.write("User ID:", profile.get("user_id"))
        except Exception:
            st.write("Authenticated (profile fetch failed)")
        if st.button("Logout (clear token)"):
            st.session_state.pop("kite_access_token", None)
            st.session_state.pop("kite_login_response", None)
            st.success("Logged out.")
            st.experimental_rerun()
    else:
        st.info("Not authenticated yet.")

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["Portfolio", "Orders", "Market & Historical", "Websocket", "MFs", "Instruments", "Debug"])
tab_portfolio, tab_orders, tab_market, tab_ws, tab_mf, tab_inst, tab_debug = tabs

# ---------------------------
# Market Tab (fixed LTP / OHLC / Quote)
# ---------------------------
with tab_market:
    st.header("Market Data")

    if not k:
        st.info("Login first to fetch market data.")
    else:
        q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], index=0)
        q_symbol = st.text_input("Symbol", value="INFY")

        market_data_type = st.radio("Data type:", 
            ("LTP", "OHLC + LTP", "Full Quote"), index=0)

        if st.button("Get market data"):
            if market_data_type == "LTP":
                res = get_ltp_price(k, q_symbol, q_exchange)
            elif market_data_type == "OHLC + LTP":
                res = get_ohlc_quote(k, q_symbol, q_exchange)
            else:
                res = get_full_market_quote(k, q_symbol, q_exchange)

            if "error" in res:
                st.error(res["error"])
            else:
                st.json(res)
