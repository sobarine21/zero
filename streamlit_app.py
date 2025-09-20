# streamlit_kite_app.py
# Streamlit app to authenticate a user with Zerodha Kite (Kite Connect) and fetch basic account data.

import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import json
from datetime import datetime

st.set_page_config(page_title="Kite Connect - Streamlit demo", layout="wide")
st.title("Kite Connect (Zerodha) — Streamlit demo")

# --- CONFIG ---
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
    st.error("❌ Missing Kite credentials in Streamlit secrets. Add [kite] api_key, api_secret and redirect_uri.")
    st.stop()

# --- INIT CLIENT ---
kite_client = KiteConnect(api_key=API_KEY)
login_url = kite_client.login_url()

# --- LOGIN STEP ---
st.markdown("### Step 1 — Login")
st.write("Click the link below to login to Kite. After successful login you will be redirected to the configured redirect URI with a `request_token` in query params.")
st.markdown(f"[🔗 Open Kite login]({login_url})")

query_params = st.query_params
request_token = query_params.get("request_token")

# --- SESSION HANDLING ---
if request_token and "kite_access_token" not in st.session_state:
    st.success("✅ Received request_token. Exchanging for access token...")

    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")

        st.session_state["kite_access_token"] = access_token
        st.session_state["kite_login_response"] = data

        st.success("🎉 Access token obtained and stored in session.")

        json_blob = json.dumps(data, default=str)
        st.download_button("⬇️ Download token JSON", json_blob, file_name="kite_token.json", mime="application/json")

    except Exception as e:
        st.error(f"Failed to generate session: {e}")
        st.stop()

# --- AUTHENTICATED CLIENT ---
if "kite_access_token" in st.session_state:
    access_token = st.session_state["kite_access_token"]
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(access_token)

    st.markdown("---")
    st.markdown("## 📊 Fetch account data")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("👤 Fetch profile"):
            try:
                profile = k.profile()
                st.json(profile)
            except Exception as e:
                st.error(f"Error fetching profile: {e}")

        if st.button("💰 Get margins"):
            try:
                margins = k.margins()
                st.json(margins)
            except Exception as e:
                st.error(f"Error fetching margins: {e}")

        if st.button("📑 Get orders"):
            try:
                orders = k.orders()
                st.dataframe(pd.DataFrame(orders))
            except Exception as e:
                st.error(f"Error fetching orders: {e}")

        if st.button("📈 Get positions"):
            try:
                positions = k.positions()
                st.write("Net positions")
                st.dataframe(pd.DataFrame(positions.get("net", [])))
                st.write("Day positions")
                st.dataframe(pd.DataFrame(positions.get("day", [])))
            except Exception as e:
                st.error(f"Error fetching positions: {e}")

        if st.button("📂 Get holdings"):
            try:
                holdings = k.holdings()
                st.dataframe(pd.DataFrame(holdings))
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")

        if st.button("🏦 Get portfolio (funds)"):
            try:
                funds = k.margins("equity")
                st.json(funds)
            except Exception as e:
                st.error(f"Error fetching funds: {e}")

        if st.button("🚪 Logout / clear token"):
            st.session_state.pop("kite_access_token", None)
            st.success("Cleared access token. Please login again.")
            st.rerun()

    with col2:
        st.markdown("### ⚡ Quotes (example)")
        symbol = st.text_input("Enter tradingsymbol (eg: INFY)", value="INFY")
        if st.button("Get quote for symbol"):
            try:
                quote = k.quote("NSE:" + symbol)
                st.json(quote)
            except Exception as e:
                st.error(f"Error fetching quote: {e}")

        st.markdown("### 📜 Historical data (demo)")

        # Load instrument dump once & cache
        @st.cache_data
        def load_instruments():
            return pd.DataFrame(k.instruments())

        instruments_df = load_instruments()

        hist_symbol = st.text_input("Symbol (tradingsymbol)", value="INFY")
        hist_exchange = st.selectbox("Exchange", ["NSE", "BSE"], index=0)
        from_date = st.date_input("From date", datetime(2024, 1, 1))
        to_date = st.date_input("To date", datetime.now().date())
        interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "day", "week", "month"], index=4)

        if st.button("Fetch historical"):
            try:
                row = instruments_df[
                    (instruments_df["tradingsymbol"] == hist_symbol.upper()) &
                    (instruments_df["exchange"] == hist_exchange)
                ]

                if row.empty:
                    st.error("❌ Symbol not found in instruments dump. Check tradingsymbol & exchange.")
                else:
                    instrument_token = int(row.iloc[0]["instrument_token"])
                    st.info(f"Using instrument_token: {instrument_token}")

                    hist_data = k.historical_data(
                        instrument_token=instrument_token,
                        from_date=from_date,
                        to_date=to_date,
                        interval=interval
                    )
                    df_hist = pd.DataFrame(hist_data)
                    st.dataframe(df_hist)

            except Exception as e:
                st.error(f"Error fetching historical data: {e}")

else:
    st.info("ℹ️ No access token yet. Login via the link above and ensure the redirect URI matches exactly in developer console.")
