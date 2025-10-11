import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import json
import time
from datetime import datetime, date
from typing import Optional, List, Dict
import re

# Supabase Python client
# pip install supabase
from supabase import create_client, Client

st.set_page_config(page_title="Kite + Supabase — Restricted Securities Checker", layout="wide")
st.title("Kite Connect (Zerodha) — Streamlit demo with Supabase restricted-check")

# ---------------------------
# CONFIG / SECRETS
# ---------------------------
# Put kite and supabase credentials in Streamlit secrets as shown above
try:
    kite_conf = st.secrets["kite"]
    API_KEY = kite_conf.get("api_key")
    API_SECRET = kite_conf.get("api_secret")
    REDIRECT_URI = kite_conf.get("redirect_uri")
except Exception:
    API_KEY = None
    API_SECRET = None
    REDIRECT_URI = None

try:
    supa_conf = st.secrets["supabase"]
    SUPABASE_URL = supa_conf.get("url")
    SUPABASE_KEY = supa_conf.get("key")
except Exception:
    SUPABASE_URL = None
    SUPABASE_KEY = None

if not API_KEY or not API_SECRET or not REDIRECT_URI:
    st.error("Missing Kite credentials in Streamlit secrets. Add [kite] api_key, api_secret and redirect_uri.")
    st.stop()

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing Supabase credentials in Streamlit secrets. Add [supabase] url and key.")
    st.stop()

# ---------------------------
# Supabase client init
# ---------------------------
# Uses supabase-py (create_client)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Kite init (unauth client)
# ---------------------------
kite_client = KiteConnect(api_key=API_KEY)
login_url = kite_client.login_url()

# Sidebar: show login link
with st.sidebar:
    st.markdown("### Kite Connect Login")
    st.write("Click link to login and obtain request_token (redirect will include `request_token` param):")
    st.markdown(f"[🔗 Open Kite login]({login_url})")
    st.caption("After login, Streamlit reads request_token from URL query params (Streamlit >=1.14).")

# read request_token from URL (Streamlit >= 1.14)
query_params = st.experimental_get_query_params()
request_token = None
if "request_token" in query_params:
    rt = query_params.get("request_token")
    if isinstance(rt, list):
        request_token = rt[0]
    else:
        request_token = rt

# Exchange request_token for access_token (store in session_state)
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

# Create authenticated kite client if we have access token
k = None
if "kite_access_token" in st.session_state:
    access_token = st.session_state["kite_access_token"]
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(access_token)

# ---------------------------
# Utilities: Instruments / Market helpers (kept from your original)
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
            st.info(f"Loading instruments for {exchange} to find token for {symbol}...")
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

# ---------------------------
# Supabase query helpers for rst_name table
# ---------------------------
@st.cache_data(show_spinner=False)
def query_by_isin(isin: str) -> List[Dict]:
    """Query rst_name table by exact ISIN."""
    res = supabase.table("rst_name").select("*").eq("isin", isin).execute()
    if res.status_code != 200:
        raise RuntimeError(f"Supabase error: {res.data}")
    return res.data or []

@st.cache_data(show_spinner=False)
def query_by_ticker(ticker: str) -> List[Dict]:
    """Case-insensitive exact ticker match."""
    # Use ilike for case-insensitive exact by wrapping
    res = supabase.table("rst_name").select("*").ilike("ticker", ticker).execute()
    if res.status_code != 200:
        raise RuntimeError(f"Supabase error: {res.data}")
    return res.data or []

@st.cache_data(show_spinner=False)
def query_by_name_fragment(fragment: str, limit: int = 20) -> List[Dict]:
    """Search name with ilike %fragment%."""
    pattern = f"%{fragment}%"
    res = supabase.table("rst_name").select("*").ilike("name", pattern).limit(limit).execute()
    if res.status_code != 200:
        raise RuntimeError(f"Supabase error: {res.data}")
    return res.data or []

def parse_daterange(range_str: str):
    """
    Very small parser for PostgreSQL daterange text form:
    Examples: '[2025-10-01,2025-12-31)' or '(,2026-01-01]' etc.
    Returns (lower_date_or_none, upper_date_or_none, lower_inclusive, upper_inclusive)
    """
    if range_str is None:
        return (None, None, False, False)
    # Pattern to capture e.g. [2025-10-01,2025-12-31)
    m = re.match(r'^([\[\(])\s*([^,]*?)\s*,\s*([^,\]]*?)\s*([\]\)])$', range_str.strip())
    if not m:
        # not in expected format, return None
        return (None, None, False, False)
    lower_sym, lower_val, upper_val, upper_sym = m.groups()
    lower_inc = lower_sym == "["
    upper_inc = upper_sym == "]"
    lower_date = None
    upper_date = None
    if lower_val:
        try:
            lower_date = datetime.fromisoformat(lower_val).date()
        except Exception:
            lower_date = None
    if upper_val:
        try:
            upper_date = datetime.fromisoformat(upper_val).date()
        except Exception:
            upper_date = None
    return (lower_date, upper_date, lower_inc, upper_inc)

def is_date_in_daterange(range_str: str, check_date: date) -> bool:
    lower, upper, lower_inc, upper_inc = parse_daterange(range_str)
    if lower is None and upper is None:
        return False
    # lower bound
    if lower:
        if lower_inc:
            if check_date < lower:
                return False
        else:
            if check_date <= lower:
                return False
    if upper:
        if upper_inc:
            if check_date > upper:
                return False
        else:
            if check_date >= upper:
                return False
    return True

# ---------------------------
# Tabs (removed: Websocket, Mutual Funds, Admin/Debug)
# ---------------------------
tabs = st.tabs(["Portfolio", "Orders", "Market & Historical", "Instruments Dump & Utils", "Check Restricted or Not"])
tab_portfolio, tab_orders, tab_market, tab_inst, tab_restrict = tabs

# ---------------------------
# TAB: PORTFOLIO
# ---------------------------
with tab_portfolio:
    st.header("Portfolio")
    if not k:
        st.info("Login first to fetch portfolio data (use Kite login link in sidebar).")
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
        st.subheader("Market Data Snapshot")
        q_exchange = st.selectbox("Exchange for market data", ["NSE", "BSE", "NFO"], index=0, key="market_exchange")
        q_symbol = st.text_input("Tradingsymbol (e.g., INFY)", value="INFY", key="market_symbol")
        market_data_type = st.radio("Choose data type:", 
                                     ("LTP (Last Traded Price)", "OHLC + LTP", "Full Market Quote (OHLC, Depth, OI)"), 
                                     index=0, key="market_data_type_radio")

        if st.button("Get market data"):
            market_data_response = {}
            if market_data_type == "LTP (Last Traded Price)":
                market_data_response = get_ltp_price(k, q_symbol, q_exchange)
            elif market_data_type == "OHLC + LTP":
                market_data_response = get_ohlc_quote(k, q_symbol, q_exchange)
            else:
                market_data_response = get_full_market_quote(k, q_symbol, q_exchange)
            if "error" in market_data_response:
                st.error(f"Market data fetch failed: {market_data_response['error']}")
            else:
                st.json(market_data_response)

        st.markdown("---")
        st.subheader("Historical candles")
        with st.expander("Instrument dump (load)"):
            exchange_for_dump = st.selectbox("Instrument dump exchange (for lookup)", ["NSE", "BSE", "NFO", "BCD", "MCX"], index=0, key="inst_exchange")
            if st.button("Load instrument dump"):
                inst_df = load_instruments(k, exchange_for_dump)
                st.session_state["instruments_df"] = inst_df
                if not inst_df.empty:
                    st.success(f"Loaded {len(inst_df)} instruments for {exchange_for_dump}")
                else:
                    st.warning(f"Could not load instruments for {exchange_for_dump}.")

        inst_df = st.session_state.get("instruments_df", pd.DataFrame())
        hist_exchange = st.selectbox("Exchange (for historical)", ["NSE", "BSE", "NFO"], index=0, key="hist_ex")
        hist_symbol = st.text_input("Historical tradingsymbol (eg INFY)", value="INFY", key="hist_sym")
        from_date = st.date_input("From date", key="from_dt")
        to_date = st.date_input("To date", key="to_dt")
        interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "day", "week", "month"], index=4)

        if st.button("Fetch historical data"):
            hist_data = get_historical(k, hist_symbol, from_date, to_date, interval, hist_exchange)
            if "error" in hist_data:
                st.error(f"Historical fetch failed: {hist_data['error']}")
            else:
                df = pd.DataFrame(hist_data)
                if not df.empty:
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    st.dataframe(df)
                else:
                    st.write("No historical data returned.")

# ---------------------------
# TAB: INSTRUMENTS DUMP & UTILS
# ---------------------------
with tab_inst:
    st.header("Instruments dump & helper utilities")
    inst_exchange = st.selectbox("Load instruments for exchange", ["NSE", "BSE", "NFO", "BCD", "MCX"], index=0)
    if st.button("Load instruments for exchange (cached)"):
        try:
            df = load_instruments(k, inst_exchange)
            st.session_state["instruments_df"] = df
            st.success(f"Loaded {len(df)} instruments for {inst_exchange}")
        except Exception as e:
            st.error(f"Load instruments failed: {e}")

    df = st.session_state.get("instruments_df", pd.DataFrame())
    if not df.empty:
        st.write("Search by trading symbol & exchange")
        sy = st.text_input("Symbol to search (tradingsymbol)", value="INFY", key="inst_search_sym")
        if st.button("Find instrument token"):
            token = find_instrument_token(df, sy, inst_exchange)
            if token:
                st.success(f"Found instrument_token: {token}")
            else:
                st.warning("Not found. Try loading correct exchange dump or exact tradingsymbol.")
        st.markdown("Preview instruments (first 200 rows)")
        st.dataframe(df.head(200))
    else:
        st.info("No instruments loaded. Click Load instruments to fetch.")

# ---------------------------
# TAB: Check Restricted or Not
# ---------------------------
with tab_restrict:
    st.header("Check restricted or not")
    st.write("Enter an ISIN, ticker or partial name — the system will search the `rst_name` table and tell you if the security is currently in a restricted period.")

    col1, col2 = st.columns([3,1])
    with col1:
        user_input = st.text_input("ISIN / Ticker / Name", value="", placeholder="e.g. INE009A01021 or INFY or Infosys")
    with col2:
        search_btn = st.button("Check")

    def decide_query_and_run(term: str):
        term_clean = term.strip()
        if not term_clean:
            return []

        # heuristic: ISIN is commonly 12 characters and alphanumeric (e.g. INE009A01021)
        if len(term_clean) >= 12 and len(term_clean) <= 12 and re.match(r'^[A-Za-z0-9]+$', term_clean):
            # treat as ISIN
            rows = query_by_isin(term_clean)
            source = "isin"
        elif len(term_clean) <= 10 and re.match(r'^[A-Za-z0-9\.\-]+$', term_clean):
            # likely ticker
            rows = query_by_ticker(term_clean)
            source = "ticker"
            # if no rows, fallback to name fragment
            if not rows:
                rows = query_by_name_fragment(term_clean)
                source = "name_fragment"
        else:
            # treat as name fragment
            rows = query_by_name_fragment(term_clean)
            source = "name_fragment"
        return rows, source

    if search_btn:
        try:
            rows, source = decide_query_and_run(user_input)
            if not rows:
                st.info("No matching rows found in `rst_name` for your query.")
            else:
                # Transform result to DataFrame for display
                df = pd.DataFrame(rows)
                # compute 'currently_restricted' column using restricted_period
                today = date.today()
                df["currently_restricted"] = df["restricted_period"].apply(lambda x: is_date_in_daterange(x, today) if isinstance(x, str) else False)
                # show top info summary
                restricted_any = df["currently_restricted"].any()
                if restricted_any:
                    st.error(f"At least one matching security is currently RESTRICTED (as of {today.isoformat()}).")
                else:
                    st.success(f"No matching security is currently restricted (as of {today.isoformat()}).")

                st.markdown("### Matches")
                # Show a clean table
                show_df = df[["isin", "ticker", "name", "restricted_period", "currently_restricted"]]
                st.dataframe(show_df)

                # If multiple rows, allow user to inspect the row and show parsed range details
                if st.checkbox("Show parsed restricted_period details for rows"):
                    parsed = []
                    for idx, row in df.iterrows():
                        r = row.get("restricted_period")
                        lower, upper, lower_inc, upper_inc = parse_daterange(r) if isinstance(r, str) else (None, None, False, False)
                        parsed.append({
                            "isin": row.get("isin"),
                            "ticker": row.get("ticker"),
                            "name": row.get("name"),
                            "restricted_period_raw": r,
                            "lower_date": lower,
                            "upper_date": upper,
                            "lower_inclusive": lower_inc,
                            "upper_inclusive": upper_inc,
                            "currently_restricted": row.get("currently_restricted")
                        })
                    st.dataframe(pd.DataFrame(parsed))
        except Exception as e:
            st.error(f"Check failed: {e}")

# ---------------------------
# Footer / Notes
# ---------------------------
st.markdown("---")
st.write("Notes & setup")
st.write("""
- Ensure your `rst_name` table exists in Supabase and has the RLS policies you posted (rows with `created_by IS NULL` are readable by anon requests because the policy allows `created_by IS NULL`).
- If you want to enforce user-level visibility (created_by = user uuid), use Supabase client-side auth and pass JWT to Supabase so `auth.uid()` works; then replace the anon key with user-scoped requests.
- This app uses the Supabase Python client to query the `rst_name` table and a tiny daterange parser to check if today's date falls inside `restricted_period`. The parser handles typical textual PostgreSQL daterange shapes like `[2025-10-01,2025-12-31)` and similar.
- If you use Postgres range types in a different textual shape or JSON, adjust `parse_daterange()` accordingly.
""")
