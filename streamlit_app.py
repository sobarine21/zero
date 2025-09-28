# streamlit_kite_compliance.py
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import json
from datetime import datetime
from supabase import create_client, Client
import io
from PyPDF2 import PdfReader

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Realtime Portfolio Compliance", layout="wide")
st.title("📊 Realtime Portfolio Compliance with Zerodha + Supabase")

# ---------- Supabase config ----------
try:
    supabase_conf = st.secrets["supabase"]
    SUPABASE_URL = supabase_conf["url"]
    SUPABASE_KEY = supabase_conf["anon_key"]
except Exception as e:
    st.error("Missing Supabase secrets under [supabase] in Streamlit secrets. Provide url and anon_key.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Kite config ----------
try:
    kite_conf = st.secrets["kite"]
    API_KEY = kite_conf["api_key"]
    API_SECRET = kite_conf["api_secret"]
    REDIRECT_URI = kite_conf["redirect_uri"]
except Exception as e:
    st.error("Missing Kite credentials under [kite] in Streamlit secrets.")
    st.stop()

kite_client = KiteConnect(api_key=API_KEY)
login_url = kite_client.login_url()

# ---------- Supabase Auth UI ----------
st.sidebar.title("🔐 Supabase Login")
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    try:
        # sign in
        supabase.auth.sign_in_with_password({"email": email, "password": password})
        # fetch user
        current = supabase.auth.get_user()
        # normalize shape
        user = None
        if isinstance(current, dict):
            # newer library often returns {'data': {'user': {...}}}
            user = current.get("data", {}).get("user") or current.get("user") or current.get("data")
        else:
            # older clients return object with .user
            try:
                user = current.user
            except Exception:
                user = None

        if not user:
            st.sidebar.error("Login: could not fetch user object. Check credentials or client version.")
        else:
            st.session_state["user"] = user
            uid = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
            st.sidebar.success(f"Logged in: {email} (uid={uid})")
    except Exception as e:
        st.sidebar.error(f"Login failed: {e}")

if "user" not in st.session_state:
    st.info("Please login via the sidebar (Supabase Auth) to proceed.")
    st.stop()

# helper for uid (not used for inserts; only for display)
def _uid_from_user(user_obj):
    if user_obj is None:
        return None
    if isinstance(user_obj, dict):
        return user_obj.get("id") or user_obj.get("user", {}).get("id")
    return getattr(user_obj, "id", None)

user = st.session_state["user"]
user_id = _uid_from_user(user)
if not user_id:
    st.error("Could not determine user id from Supabase user object.")
    st.stop()

# ---------- Kite login link ----------
st.markdown("### Step 1 — Login to Zerodha Kite")
st.write("Click the link below and complete login. You will be redirected to the configured redirect URI with a request_token.")
st.markdown(f"[🔗 Open Kite login]({login_url})")

query_params = st.experimental_get_query_params()
request_token = None
if "request_token" in query_params:
    vals = query_params.get("request_token")
    if isinstance(vals, list) and len(vals) > 0:
        request_token = vals[0]
    elif isinstance(vals, str):
        request_token = vals

# ---------- Exchange token ----------
if request_token and "kite_access_token" not in st.session_state:
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")
        if not access_token:
            st.error(f"Failed to get access token from Kite. Response: {data}")
        else:
            st.session_state["kite_access_token"] = access_token
            st.session_state["kite_login_response"] = data
            st.success("Kite access token obtained.")

            # Persist token in DB WITHOUT user_id (DB will set user_id = auth.uid())
            try:
                supabase.table("kite_tokens").insert({
                    "access_token": access_token,
                    "login_data": data,
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
            except Exception as e:
                st.warning(f"Could not persist kite token to DB: {e}")
    except Exception as e:
        st.error(f"Kite session exchange failed: {e}")

# ---------- If connected to Kite ----------
if "kite_access_token" in st.session_state:
    access_token = st.session_state["kite_access_token"]
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(access_token)

    st.markdown("## 🚀 Portfolio Data & Compliance Checks")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Broker Data")

        if st.button("📑 Fetch & Save Orders"):
            try:
                orders = k.orders()
                df = pd.DataFrame(orders)
                st.write("Orders fetched:")
                st.dataframe(df)

                # Persist: do NOT include user_id; DB default auth.uid() will apply
                supabase.table("orders").insert({
                    "data": df.to_dict(orient="records"),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()

                st.success("Orders saved to Supabase.")
            except Exception as e:
                st.error(f"Error fetching/saving orders: {e}")

        if st.button("📈 Fetch & Save Positions"):
            try:
                positions = k.positions()
                net = positions.get("net", []) if isinstance(positions, dict) else []
                df_net = pd.DataFrame(net)
                st.write("Net positions:")
                st.dataframe(df_net)

                supabase.table("positions").insert({
                    "data": df_net.to_dict(orient="records"),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.success("Positions saved to Supabase.")

                # demo compliance check
                if not df_net.empty and "quantity" in df_net.columns:
                    try:
                        qty_series = pd.to_numeric(df_net["quantity"], errors="coerce").fillna(0).astype(int)
                        over_limit = df_net[qty_series > 10000]
                        if not over_limit.empty:
                            st.error("⚠️ Demo Compliance: position(s) exceed 10,000 units.")
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Error fetching/saving positions: {e}")

        if st.button("📂 Fetch & Save Holdings"):
            try:
                holdings = k.holdings()
                df = pd.DataFrame(holdings)
                st.write("Holdings:")
                st.dataframe(df)

                supabase.table("holdings").insert({
                    "data": df.to_dict(orient="records"),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.success("Holdings saved to Supabase.")
            except Exception as e:
                st.error(f"Error fetching/saving holdings: {e}")

        if st.button("🚪 Logout (clear Kite token)"):
            st.session_state.pop("kite_access_token", None)
            st.success("Cleared Kite token. You can login to Kite again if needed.")
            st.experimental_rerun()

    with col2:
        st.subheader("Upload Fund Document (PDF / TXT)")
        st.info("Supported: PDF and plain TXT only.")

        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.read()
                fname = uploaded_file.name

                extracted_text = ""
                if fname.lower().endswith(".pdf"):
                    try:
                        reader = PdfReader(io.BytesIO(raw_bytes))
                        parts = []
                        for p in reader.pages:
                            parts.append(p.extract_text() or "")
                        extracted_text = "\n".join(parts)
                    except Exception as e:
                        st.error(f"PDF parse error: {e}")
                        extracted_text = ""
                elif fname.lower().endswith(".txt"):
                    try:
                        extracted_text = raw_bytes.decode("utf-8", errors="ignore")
                    except Exception:
                        extracted_text = raw_bytes.decode("latin-1", errors="ignore")
                else:
                    st.error("Only PDF or TXT allowed.")
                    extracted_text = ""

                # Persist extracted text WITHOUT user_id (DB will set it)
                if extracted_text is not None:
                    supabase.table("documents").insert({
                        "file_name": fname,
                        "extracted_text": extracted_text,
                        "uploaded_at": datetime.utcnow().isoformat()
                    }).execute()
                    st.success(f"Saved extracted text for {fname}")
                    st.text_area("Preview (first 2000 chars)", extracted_text[:2000], height=300)
            except Exception as e:
                st.error(f"Failed to process upload: {e}")

else:
    st.info("No Kite access token in session. Login to Kite and complete redirect to get request_token.")
