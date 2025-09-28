# streamlit_kite_compliance.py
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import json
from datetime import datetime
from supabase import create_client, Client
import io

# PDF parser
from PyPDF2 import PdfReader

# ---------------------------
# STREAMLIT CONFIG
# ---------------------------
st.set_page_config(page_title="Realtime Portfolio Compliance", layout="wide")
st.title("📊 Realtime Portfolio Compliance with Zerodha + Supabase")

# ---------------------------
# SUPABASE CONFIG
# ---------------------------
try:
    supabase_conf = st.secrets["supabase"]
    SUPABASE_URL = supabase_conf["url"]
    SUPABASE_KEY = supabase_conf["anon_key"]
except Exception:
    st.error("❌ Missing Supabase credentials in Streamlit secrets (under 'supabase').")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# KITE CONFIG
# ---------------------------
try:
    kite_conf = st.secrets["kite"]
    API_KEY = kite_conf["api_key"]
    API_SECRET = kite_conf["api_secret"]
    REDIRECT_URI = kite_conf["redirect_uri"]
except Exception:
    st.error("❌ Missing Kite credentials in Streamlit secrets (under 'kite').")
    st.stop()

# ---------------------------
# INIT KITE CLIENT
# ---------------------------
kite_client = KiteConnect(api_key=API_KEY)
login_url = kite_client.login_url()

# ---------------------------
# SUPABASE AUTH UI
# ---------------------------
st.sidebar.title("🔐 Login (Supabase Auth)")
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    try:
        # sign in
        supabase.auth.sign_in_with_password({"email": email, "password": password})
        # fetch current user
        current = supabase.auth.get_user()
        # current can be an object or dict depending on library version
        user = None
        try:
            # attempt attribute access
            user = current.user  # v1 style
        except Exception:
            # try dictionary shape
            if isinstance(current, dict):
                # common shapes: {'data': {'user': {...}}} or {'user': {...}}
                user = current.get("user") or current.get("data", {}).get("user")
            else:
                user = None

        if not user:
            st.sidebar.error("Login failed: unable to fetch user. Check credentials or supabase client version.")
        else:
            st.session_state["user"] = user
            # show minimal
            uid = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
            st.sidebar.success(f"Logged in: {email} (uid={uid})")
    except Exception as e:
        st.sidebar.error(f"Login failed: {e}")

# require login
if "user" not in st.session_state:
    st.warning("⚠️ Please log in via the sidebar to continue.")
    st.stop()

# helper to get user id robustly
def _get_user_id(user_obj):
    if user_obj is None:
        return None
    if isinstance(user_obj, dict):
        return user_obj.get("id") or user_obj.get("user", {}).get("id")
    return getattr(user_obj, "id", None)

user = st.session_state["user"]
user_id = _get_user_id(user)
if not user_id:
    st.error("Could not determine user id from Supabase user object. Check supabase client behavior.")
    st.stop()

# ---------------------------
# STEP 1: LOGIN TO KITE (open link)
# ---------------------------
st.markdown("### Step 1 — Login to Zerodha Kite")
st.write("Click the link below and complete login on Kite. You will be redirected to the configured redirect URI with a request_token.")
st.markdown(f"[🔗 Open Kite login]({login_url})")

query_params = st.experimental_get_query_params()
# request_token may be returned as list by streamlit query params
request_token = None
if "request_token" in query_params:
    vals = query_params.get("request_token")
    if isinstance(vals, list) and len(vals) > 0:
        request_token = vals[0]
    elif isinstance(vals, str):
        request_token = vals

# ---------------------------
# STEP 2: EXCHANGE REQUEST TOKEN (server-side exchange)
# ---------------------------
if request_token and "kite_access_token" not in st.session_state:
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")
        if not access_token:
            st.error(f"Failed to get access token. Response: {data}")
        else:
            st.session_state["kite_access_token"] = access_token
            st.session_state["kite_login_response"] = data
            st.success("🎉 Access token obtained and stored in session.")

            # insert into supabase kite_tokens table
            try:
                supabase.table("documents").insert({
    "file_name": uploaded_file.name,
    "extracted_text": extracted_text,
    "uploaded_at": datetime.utcnow().isoformat()
}).execute()

            except Exception as e:
                # log but continue
                st.warning(f"Could not persist kite token to DB: {e}")
    except Exception as e:
        st.error(f"Failed to generate session with Kite: {e}")

# ---------------------------
# STEP 3: AUTHENTICATED CLIENT UI
# ---------------------------
if "kite_access_token" in st.session_state:
    access_token = st.session_state["kite_access_token"]
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(access_token)

    st.markdown("## 🚀 Portfolio Data & Compliance Checks")
    left, right = st.columns([1, 1])

    # ---------------------------
    # LEFT: Broker Data Fetch & Save
    # ---------------------------
    with left:
        st.subheader("Broker Data")

        if st.button("📑 Fetch & Save Orders"):
            try:
                orders = k.orders()
                df = pd.DataFrame(orders)
                st.write("Fetched orders:")
                st.dataframe(df)

                # persist
                supabase.table("orders").insert({
                    "user_id": user_id,
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
                    "user_id": user_id,
                    "data": df_net.to_dict(orient="records"),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.success("Positions saved to Supabase.")

                # Example simple compliance check (demo only)
                if not df_net.empty and "quantity" in df_net.columns:
                    # cast safely
                    try:
                        qty_series = pd.to_numeric(df_net["quantity"], errors="coerce").fillna(0).astype(int)
                        over_limit = df_net[qty_series > 10000]
                        if not over_limit.empty:
                            st.error("⚠️ Demo Compliance: position(s) exceed 10,000 units.")
                    except Exception:
                        # ignore errors in demo check
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
                    "user_id": user_id,
                    "data": df.to_dict(orient="records"),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.success("Holdings saved to Supabase.")
            except Exception as e:
                st.error(f"Error fetching/saving holdings: {e}")

        if st.button("🚪 Logout (clear Kite token)"):
            st.session_state.pop("kite_access_token", None)
            st.success("Cleared Kite token. Please login again on Kite if you want to reconnect.")
            st.experimental_rerun()

    # ---------------------------
    # RIGHT: Document Upload (PDF / TXT) and simple extraction
    # ---------------------------
    with right:
        st.subheader("Upload Fund Document (PDF / TXT)")
        st.info("Supported: PDF and plain TXT. (DOCX disabled to avoid extra packages.)")

        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.read()  # read once
                file_name = uploaded_file.name.lower()

                extracted_text = ""
                if file_name.endswith(".pdf"):
                    try:
                        reader = PdfReader(io.BytesIO(raw_bytes))
                        text_parts = []
                        for page in reader.pages:
                            # extract_text can return None
                            page_text = page.extract_text() or ""
                            text_parts.append(page_text)
                        extracted_text = "\n".join(text_parts)
                    except Exception as e:
                        st.error(f"PDF parsing failed: {e}")
                        extracted_text = ""
                elif file_name.endswith(".txt"):
                    try:
                        extracted_text = raw_bytes.decode("utf-8", errors="ignore")
                    except Exception:
                        extracted_text = raw_bytes.decode("latin-1", errors="ignore")
                else:
                    st.error("Unsupported file type (only PDF and TXT allowed).")

                # persist extracted text into documents table
                if extracted_text is not None:
                    supabase.table("documents").insert({
                        "user_id": user_id,
                        "file_name": uploaded_file.name,
                        "extracted_text": extracted_text,
                        "uploaded_at": datetime.utcnow().isoformat()
                    }).execute()
                    st.success(f"Extracted text saved for {uploaded_file.name}")
                    # show preview (limit size)
                    st.text_area("Preview (first 2000 chars)", extracted_text[:2000], height=300)
            except Exception as e:
                st.error(f"Failed to process upload: {e}")

else:
    st.info("ℹ️ No Kite access token in session. After you login on Kite, you'll be redirected back with a request_token.")
