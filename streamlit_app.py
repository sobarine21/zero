# -------------------- streamlit_kite_compliance_pro.py --------------------
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
import io
from PyPDF2 import PdfReader
import json
import re

# ---------- Helper: safe JSON encoder ----------
def safe_json(obj):
    return json.loads(json.dumps(obj, default=str))

# ---------- Helper: run compliance checks ----------
def run_compliance_checks(positions_df, documents_list, margin_data):
    """
    positions_df: DataFrame of positions fetched from Kite
    documents_list: List of dicts from Supabase documents table [{'extracted_text':...}, ...]
    margin_data: dict from Kite API
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
        price = float(row.get("average_price") or 0)
        quantity = int(row.get("quantity") or 0)
        product = row.get("exchange") or "NSE"

        # Margin lookup by segment (demo: equity / commodity / currency)
        margin_pct = 0.1  # default fallback
        segment = row.get("product") or "MIS"
        if margin_data:
            if "equity" in margin_data and segment.upper() in margin_data["equity"]:
                margin_pct = float(margin_data["equity"][segment.upper()])
            elif "commodity" in margin_data and segment.upper() in margin_data["commodity"]:
                margin_pct = float(margin_data["commodity"][segment.upper()])
            elif "currency" in margin_data and segment.upper() in margin_data["currency"]:
                margin_pct = float(margin_data["currency"][segment.upper()])

        margin = price * quantity * margin_pct
        positions_df.at[idx, "margin_required"] = margin

        # Compliance check vs max limit
        if quantity > max_limit:
            positions_df.at[idx, "compliance_status"] = "VIOLATION"
            positions_df.at[idx, "compliance_issues"].append(
                f"Position {row['tradingsymbol']} exceeds limit ({quantity} > {max_limit})"
            )
            issues.append(f"Position {row['tradingsymbol']} exceeds limit ({quantity} > {max_limit})")

        # Demo daily loss check
        day_change = float(row.get("day_change") or 0)
        if day_change < -50000:
            positions_df.at[idx, "compliance_status"] = "VIOLATION"
            positions_df.at[idx, "compliance_issues"].append(
                f"Position {row['tradingsymbol']} daily loss exceeds 50k ({day_change})"
            )
            issues.append(f"Position {row['tradingsymbol']} daily loss exceeds 50k ({day_change})")

    return issues, positions_df

# ---------- Streamlit UI ----------
st.set_page_config(page_title="📊 Realtime Portfolio Compliance Pro", layout="wide")
st.title("📊 Realtime Portfolio Compliance with Zerodha + Supabase (Pro)")

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

    # ---------- Left Column: Broker Data ----------
    with col1:
        st.subheader("Broker Data & Compliance")

        # ---------- Orders ----------
        if st.button("📑 Fetch & Save Orders"):
            try:
                orders = k.orders()
                df_orders = pd.DataFrame(orders)
                st.dataframe(df_orders)
                supabase.table("orders").insert({
                    "user_id": user_id,
                    "data": safe_json(df_orders.to_dict(orient="records")),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.success("Orders saved to Supabase.")
            except Exception as e:
                st.error(f"Error fetching/saving orders: {e}")

        # ---------- Holdings ----------
        if st.button("📂 Fetch & Save Holdings"):
            try:
                holdings = k.holdings()
                df_holdings = pd.DataFrame(holdings)
                st.dataframe(df_holdings)
                supabase.table("holdings").insert({
                    "user_id": user_id,
                    "data": safe_json(df_holdings.to_dict(orient="records")),
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                st.success("Holdings saved to Supabase.")
            except Exception as e:
                st.error(f"Error fetching/saving holdings: {e}")

        # ---------- Positions with Margin & Compliance ----------
        if st.button("📈 Fetch & Save Positions & Compliance"):
            try:
                positions = k.positions()
                net_positions = positions.get("net", []) if isinstance(positions, dict) else []
                df_positions = pd.DataFrame(net_positions)

                # Fetch uploaded fund documents
                docs_res = supabase.table("documents").select("*").eq("user_id", user_id).execute()
                documents_list = docs_res.data or []

                # Fetch live margins per segment
                margin_data = {}
                try:
                    margin_info = k.margins(segment="equity")
                    margin_data["equity"] = {k: v.get("enabled", 0) for k, v in margin_info.items()}
                except:
                    pass
                try:
                    margin_info = k.margins(segment="commodity")
                    margin_data["commodity"] = {k: v.get("enabled", 0) for k, v in margin_info.items()}
                except:
                    pass
                try:
                    margin_info = k.margins(segment="currency")
                    margin_data["currency"] = {k: v.get("enabled", 0) for k, v in margin_info.items()}
                except:
                    pass

                # Run compliance checks
                issues, df_positions = run_compliance_checks(df_positions, documents_list, margin_data)
                st.dataframe(df_positions)

                # Save positions with compliance & margin
                supabase.table("positions").insert({
                    "user_id": user_id,
                    "data": safe_json(df_positions.to_dict(orient="records")),
                    "margin_required": None,  # can be aggregated or stored in each row's JSON
                    "compliance_status": None,
                    "compliance_issues": None,
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

        # ---------- Logout ----------
        if st.button("🚪 Logout"):
            st.session_state.pop("kite_access_token", None)
            st.success("Cleared Kite token.")
            st.experimental_rerun()

    # ---------- Right Column: Document Upload ----------
    with col2:
        st.subheader("Upload Fund Document (PDF / TXT)")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        if uploaded_file:
            try:
                raw_bytes = uploaded_file.read()
                fname = uploaded_file.name
                extracted_text = ""

                if fname.lower().endswith(".pdf"):
                    reader = PdfReader(io.BytesIO(raw_bytes))
                    extracted_text = "\n".join([p.extract_text() or "" for p in reader.pages])
                elif fname.lower().endswith(".txt"):
                    extracted_text = raw_bytes.decode("utf-8", errors="ignore")

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
    st.info("No Kite access token. Login first.")
