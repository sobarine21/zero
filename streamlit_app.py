# -------------------- streamlit_kite_compliance.py --------------------
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
import io
from PyPDF2 import PdfReader
import json
import re

# ---------- helper: safe JSON encoder ----------
def safe_json(obj):
    """Convert dicts/dataframes to JSON-safe python objects (str for datetime)."""
    return json.loads(json.dumps(obj, default=str))

# ---------- helper: compliance checks ----------
def run_compliance_checks(positions_df, documents_list):
    """
    positions_df: DataFrame of positions fetched from Kite
    documents_list: List of dicts from Supabase documents table [{'extracted_text':...}, ...]
    Returns issues list, positions_df with margin/compliance columns
    """
    issues = []
    positions_df["margin_required"] = 0
    positions_df["compliance_status"] = "OK"
    positions_df["compliance_issues"] = [[] for _ in range(len(positions_df))]

    # Parse fund limits from uploaded documents
    max_limit = 10000  # default demo
    for doc in documents_list:
        text = doc.get("extracted_text", "").lower()
        m = re.search(r"max position[:\s]+(\d+)", text)
        if m:
            max_limit = int(m.group(1))

    for idx, row in positions_df.iterrows():
        # Demo margin calculation (10% of value)
        price = float(row.get("average_price") or 0)
        quantity = int(row.get("quantity") or 0)
        margin = price * quantity * 0.1
        positions_df.at[idx, "margin_required"] = margin

        # Compliance check vs max limit
        if quantity > max_limit:
            positions_df.at[idx, "compliance_status"] = "VIOLATION"
            positions_df.at[idx, "compliance_issues"].append(
                f"Position {row['tradingsymbol']} exceeds limit ({quantity} > {max_limit})"
            )
            issues.append(f"Position {row['tradingsymbol']} exceeds limit ({quantity} > {max_limit})")

        # Demo loss check
        day_change = float(row.get("day_change") or 0)
        if day_change < -50000:
            positions_df.at[idx, "compliance_status"] = "VIOLATION"
            positions_df.at[idx, "compliance_issues"].append(
                f"Position {row['tradingsymbol']} daily loss exceeds 50k ({day_change})"
            )
            issues.append(f"Position {row['tradingsymbol']} daily loss exceeds 50k ({day_change})")
    return issues, positions_df

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Realtime Portfolio Compliance", layout="wide")
st.title("📊 Realtime Portfolio Compliance with Zerodha + Supabase")

# ---------- Supabase config ----------
try:
    supabase_conf = st.secrets["supabase"]
    SUPABASE_URL = supabase_conf["url"]
    SUPABASE_KEY = supabase_conf["anon_key"]
except Exception:
    st.error("Missing Supabase secrets under [supabase]. Provide url and anon_key.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Kite config ----------
try:
    kite_conf = st.secrets["kite"]
    API_KEY = kite_conf["api_key"]
    API_SECRET = kite_conf["api_secret"]
    REDIRECT_URI = kite_conf["redirect_uri"]
except Exception:
    st.error("Missing Kite credentials under [kite].")
    st.stop()

kite_client = KiteConnect(api_key=API_KEY)
login_url = kite_client.login_url()

# ---------- Supabase Auth ----------
st.sidebar.title("🔐 Supabase Login")
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    try:
        session = supabase.auth.sign_in_with_password({"email": email, "password": password})
        supabase.auth.set_session(session.session.access_token, session.session.refresh_token)
        st.session_state["supabase"] = supabase
        st.session_state["user"] = session.user
        st.sidebar.success(f"Logged in: {email} (uid={session.user.id})")
    except Exception as e:
        st.sidebar.error(f"Login failed: {e}")

if "user" not in st.session_state:
    st.info("Please login via the sidebar (Supabase Auth) to proceed.")
    st.stop()

user = st.session_state["user"]
user_id = user.id

# ---------- Kite login ----------
st.markdown("### Step 1 — Login to Zerodha Kite")
st.write("Click the link below and complete login. You will be redirected to the configured redirect URI with a request_token.")
st.markdown(f"[🔗 Open Kite login]({login_url})")

query_params = st.experimental_get_query_params()
request_token = query_params.get("request_token", [None])[0]

# ---------- Exchange token ----------
if request_token and "kite_access_token" not in st.session_state:
    try:
        data = kite_client.generate_session(request_token, api_secret=API_SECRET)
        access_token = data.get("access_token")
        st.session_state["kite_access_token"] = access_token
        st.session_state["kite_login_response"] = data
        st.success("Kite access token obtained.")
        # Persist token
        supabase.table("kite_tokens").insert({
            "user_id": user_id,
            "access_token": access_token,
            "login_data": safe_json(data),
            "created_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        st.error(f"Kite session exchange failed: {e}")

# ---------- Main App ----------
if "kite_access_token" in st.session_state:
    access_token = st.session_state["kite_access_token"]
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(access_token)

    st.markdown("## 🚀 Portfolio Data & Compliance Checks")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Broker Data")

        # ---------- Orders ----------
        if st.button("📑 Fetch & Save Orders"):
            try:
                orders = k.orders()
                df = pd.DataFrame(orders)
                st.dataframe(df)
                supabase.table("orders").insert({
                    "user_id": user_id,
                    "data": safe_json(df.to_dict(orient="records")),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.success("Orders saved to Supabase.")
            except Exception as e:
                st.error(f"Error fetching/saving orders: {e}")

        # ---------- Positions ----------
        if st.button("📈 Fetch & Save Positions & Compliance"):
            try:
                positions = k.positions()
                net = positions.get("net", []) if isinstance(positions, dict) else []
                df_net = pd.DataFrame(net)

                # fetch uploaded documents
                docs_res = supabase.table("documents").select("*").eq("user_id", user_id).execute()
                documents_list = docs_res.data or []

                # Run compliance
                issues, df_net = run_compliance_checks(df_net, documents_list)

                st.dataframe(df_net)

                # Save positions with margin/compliance columns
                supabase.table("positions").insert({
                    "user_id": user_id,
                    "data": safe_json(df_net.to_dict(orient="records")),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()

                if issues:
                    st.error("⚠️ Compliance Issues Found:")
                    for i in issues:
                        st.write(f"- {i}")
                else:
                    st.success("✅ All positions within compliance limits.")

            except Exception as e:
                st.error(f"Error fetching/saving positions: {e}")

        # ---------- Holdings ----------
        if st.button("📂 Fetch & Save Holdings"):
            try:
                holdings = k.holdings()
                df = pd.DataFrame(holdings)
                st.dataframe(df)
                supabase.table("holdings").insert({
                    "user_id": user_id,
                    "data": safe_json(df.to_dict(orient="records")),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.success("Holdings saved to Supabase.")
            except Exception as e:
                st.error(f"Error fetching/saving holdings: {e}")

        # ---------- Logout ----------
        if st.button("🚪 Logout (clear Kite token)"):
            st.session_state.pop("kite_access_token", None)
            st.success("Cleared Kite token.")
            st.experimental_rerun()

    # ---------- Document Upload & Compliance ----------
    with col2:
        st.subheader("Upload Fund Document (PDF / TXT)")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.read()
                fname = uploaded_file.name
                extracted_text = ""

                if fname.lower().endswith(".pdf"):
                    reader = PdfReader(io.BytesIO(raw_bytes))
                    extracted_text = "\n".join([p.extract_text() or "" for p in reader.pages])
                elif fname.lower().endswith(".txt"):
                    extracted_text = raw_bytes.decode("utf-8", errors="ignore")

                # Parse fund limits / mandate summary
                fund_limits = {}
                mandate_summary = ""
                max_match = re.search(r"max position[:\s]+(\d+)", extracted_text.lower())
                if max_match:
                    fund_limits["max_position"] = int(max_match.group(1))
                    mandate_summary = f"Max position: {fund_limits['max_position']}"

                if extracted_text:
                    supabase.table("documents").insert({
                        "user_id": user_id,
                        "file_name": fname,
                        "extracted_text": extracted_text,
                        "fund_limits": fund_limits,
                        "mandate_summary": mandate_summary,
                        "uploaded_at": datetime.utcnow().isoformat()
                    }).execute()
                    st.success(f"Saved extracted text for {fname}")
                    st.text_area("Preview (first 2000 chars)", extracted_text[:2000], height=300)

else:
    st.info("No Kite access token in session. Login to Kite first.")
