import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import json
from datetime import datetime
from supabase import create_client, Client
import io

# For doc parsing
from PyPDF2 import PdfReader
import docx

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
    st.error("❌ Missing Supabase credentials in Streamlit secrets")
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
    st.error("❌ Missing Kite credentials in Streamlit secrets")
    st.stop()

# ---------------------------
# INIT KITE CLIENT
# ---------------------------
kite_client = KiteConnect(api_key=API_KEY)
login_url = kite_client.login_url()

# ---------------------------
# SUPABASE AUTH
# ---------------------------
st.sidebar.title("🔐 User Login")
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state["user"] = res.user
        st.sidebar.success(f"Logged in as {email}")
    except Exception as e:
        st.sidebar.error(f"Login failed: {e}")

if "user" not in st.session_state:
    st.warning("⚠️ Please log in first.")
    st.stop()

user = st.session_state["user"]

# ---------------------------
# STEP 1: LOGIN TO KITE
# ---------------------------
st.markdown("### Step 1 — Login to Zerodha Kite")
st.markdown(f"[🔗 Open Kite login]({login_url})")

query_params = st.query_params
request_token = query_params.get("request_token")

# ---------------------------
# STEP 2: SESSION HANDLING
# ---------------------------
if request_token and "kite_access_token" not in st.session_state:
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]

        st.session_state["kite_access_token"] = access_token
        st.session_state["kite_login_response"] = data

        st.success("🎉 Access token obtained and stored.")

        supabase.table("kite_tokens").insert({
            "user_id": user.id,
            "access_token": access_token,
            "login_data": json.dumps(data),
            "created_at": datetime.utcnow().isoformat()
        }).execute()

    except Exception as e:
        st.error(f"Failed to generate session: {e}")
        st.stop()

# ---------------------------
# STEP 3: AUTHENTICATED CLIENT
# ---------------------------
if "kite_access_token" in st.session_state:
    access_token = st.session_state["kite_access_token"]
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(access_token)

    st.markdown("## 🚀 Portfolio Data & Compliance Checks")
    col1, col2 = st.columns([1, 2])

    # --- LEFT PANEL: DATA FETCH ---
    with col1:
        if st.button("📑 Get Orders"):
            try:
                orders = k.orders()
                df = pd.DataFrame(orders)
                st.dataframe(df)

                supabase.table("orders").insert({
                    "user_id": user.id,
                    "data": df.to_dict(orient="records"),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
            except Exception as e:
                st.error(f"Error fetching orders: {e}")

        if st.button("📈 Get Positions"):
            try:
                positions = k.positions()
                df_net = pd.DataFrame(positions.get("net", []))
                st.write("Net positions")
                st.dataframe(df_net)

                supabase.table("positions").insert({
                    "user_id": user.id,
                    "data": df_net.to_dict(orient="records"),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()

                if not df_net.empty and "quantity" in df_net:
                    over_limit = df_net[df_net["quantity"].astype(int) > 10000]
                    if not over_limit.empty:
                        st.error("⚠️ Compliance Breach: Position size exceeds 10,000 units")
            except Exception as e:
                st.error(f"Error fetching positions: {e}")

        if st.button("📂 Get Holdings"):
            try:
                holdings = k.holdings()
                df = pd.DataFrame(holdings)
                st.dataframe(df)

                supabase.table("holdings").insert({
                    "user_id": user.id,
                    "data": df.to_dict(orient="records"),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")

        if st.button("🚪 Logout from Kite"):
            st.session_state.pop("kite_access_token", None)
            st.success("Cleared Kite token. Please login again.")
            st.rerun()

    # --- RIGHT PANEL: DOCUMENT UPLOAD + EXTRACTION ---
    with col2:
        st.markdown("### 📤 Upload Fund Documents / Compliance Policies")
        uploaded_file = st.file_uploader("Upload PDF/Docx/TXT", type=["pdf", "docx", "txt"])

        if uploaded_file is not None:
            file_name = uploaded_file.name
            extracted_text = ""

            if file_name.endswith(".pdf"):
                pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                extracted_text = " ".join([page.extract_text() or "" for page in pdf_reader.pages])
            elif file_name.endswith(".docx"):
                doc = docx.Document(io.BytesIO(uploaded_file.read()))
                extracted_text = " ".join([para.text for para in doc.paragraphs])
            else:  # txt
                extracted_text = uploaded_file.read().decode("utf-8")

            supabase.table("documents").insert({
                "user_id": user.id,
                "file_name": file_name,
                "extracted_text": extracted_text,
                "uploaded_at": datetime.utcnow().isoformat()
            }).execute()

            st.success(f"✅ Extracted & saved {file_name}")
            st.text_area("Extracted Text", extracted_text[:2000])  # preview
