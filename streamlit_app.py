# streamlit_kite_app.py
# Streamlit app to authenticate a user with Zerodha Kite (Kite Connect) and fetch basic account data.
# Requirements: streamlit, pykiteconnect, pandas
# Usage: provide kite_api_key, kite_api_secret, redirect_uri in Streamlit secrets.

import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import json

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

# Helper: init client with api_key only for login URL
kite_client = KiteConnect(api_key=API_KEY)

# Build login URL (redirect_uri is configured in Kite Developer Console, not passed here)
login_url = kite_client.login_url()

st.markdown("### Step 1 — Login")
st.write("Click the link below to login to Kite. After successful login you will be redirected to the configured redirect URI with a `request_token` in query params.")
st.markdown(f"[🔗 Open Kite login]({login_url})")

# Read params from URL after redirect
query_params = st.query_params
request_token = query_params.get("request_token")

if request_token:
    st.success("✅ Received request_token from redirect URL. Exchanging for access token...")

    # Exchange request token for an access token
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")

        st.session_state["kite_access_token"] = access_token
        st.session_state["kite_login_response"] = data

        st.success("🎉 Access token obtained and stored in session.")

        json_blob = json.dumps(data, default=str)  # handle datetime serialization
        st.download_button("⬇️ Download token JSON", json_blob, file_name="kite_token.json", mime="application/json")

    except Exception as e:
        st.error(f"Failed to generate session: {e}")
        st.stop()

# If we have an access token in session, create an authorized client
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
                df = pd.DataFrame(orders)
                st.dataframe(df)
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
            st.success("Cleared access token from session. You can login again.")
            st.rerun()

    with col2:
        st.markdown("### ⚡ Quick instruments & quotes (example)")
        symbol = st.text_input("Enter tradingsymbol (eg: INFY)", value="INFY")
        if st.button("Get quote for symbol"):
            try:
                quote = k.quote("NSE:" + symbol) if symbol else None
                st.json(quote)
            except Exception as e:
                st.error(f"Error fetching quote: {e}")

        st.markdown("### 📜 Historical data (example)")
        hist_symbol = st.text_input("Historical symbol (exchange:tradingsymbol)", value="NSE:INFY")
        from_date = st.date_input("From date")
        to_date = st.date_input("To date")
        interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "day", "week", "month"], index=4)

        if st.button("Fetch historical"):
            try:
                st.warning("⚠️ Historical data requires instrument_token (numeric). Replace demo with instrument dump lookup.")
                # Example: k.historical_data(token, from_date, to_date, interval)
            except Exception as e:
                st.error(f"Error fetching historical: {e}")

    st.markdown("---")
    st.caption("Notes: This demo stores the access token in Streamlit session only. Do NOT embed api_secret in frontend production apps. For production, perform session exchange on a secure server and persist tokens in an encrypted store.")

else:
    st.info("ℹ️ No access token in session. Login via the link above and ensure the redirect URI returns a `request_token` query parameter.")

st.write("---")
st.write("Built with [Kite Connect](https://kite.trade/docs/connect/v3/)")
