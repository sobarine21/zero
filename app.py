import streamlit as st
import requests
import json
import pandas as pd

# ---------------- UI Setup ----------------
st.set_page_config(page_title="SniffR 🐾 by Ever Tech", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🐾 SniffR by Ever Tech</h1>",
    unsafe_allow_html=True
)

# Sidebar for API settings
st.sidebar.header("⚙️ API Settings")
api_url = st.secrets.get("indiav1_api_url", "")
jwt_token = st.secrets.get("indiav1_jwt_token", "")
user_id = st.secrets.get("indiav1_user_id", "")

if not api_url or not jwt_token or not user_id:
    st.sidebar.error("❌ API URL, JWT Token or User ID not found in secrets.")
else:
    st.sidebar.success("✅ Secrets loaded successfully")

# ---------------- Intro Write-up ----------------
st.markdown("## 📖 Overview")
st.write("""
The **India-v1 Edge Function** is a focused enforcement screening service that searches across 
**29 key regulatory and enforcement databases** with emphasis on Indian markets and global sanctions.  
It provides **parallel search execution** with exact and partial matching options.
""")

st.markdown("## 🏗️ Architecture")
st.write("""
- **Function Name**: `indiav1`  
- **Search Method**: Parallel execution across all tables  
- **Authentication**: Required user validation via `invisionid` table  
- **Response Time**: Optimized for sub-second performance  
- **Coverage**: Indian enforcement + key global sanctions
""")

with st.expander("📚 Database Coverage (29 Tables)", expanded=False):
    st.markdown("""
### 🇮🇳 Indian Stock Exchanges & Trading (10 databases)
- **NSE Under Liquidations** (`nse_under_liquidations`)  
- **NSE Suspended Companies** (`nse_suspended`)  
- **NSE Banned/Debarred** (`nse_banned_debared`)  
- **Delisted Under Liquidations** (`delisted_under_liquidations_nse`)  
- **CRIP NSE Cases** (`crip_nse_cases`)  
- **NSE Defaulting Clients** (`defaulting_clients_nse`)  
- **NSE Defaulting Client Database** (`Defaulting_Client_Database nse_`)  
- **NCDEX Defaulting Clients** (`defaulting_clients_ncdex`)  
- **MCX Defaulting Clients** (`defaulting_clients_mcx`)  
- **BSE Defaulting Clients** (`defaulting_clients_bse`)  

### 🇮🇳 Indian Regulatory Bodies (8 databases)
- **SEBI Circulars** (`sebi_circulars`)  
- **SEBI Deactivated** (`SEBI_DEACTIVATED`)  
- **Archive SEBI Debarred** (`Archive SEBI DEBARRED entities`)  
- **Disqualified Directors** (`disqualified_directors`)  
- **Directors Struck Off** (`directors_struckoff`)  
- **Companies IBC Moratorium** (`Companies_IBC_Moratorium_Debt`)  
- **Consolidated Legacy** (`consolidatedLegacyByPRN`)  
- **Banned by Competent Authorities** (`banned by  Competent Authorities India`)  
- **UAPA Banned Organizations** (`banned _list_uapa`)  

### 🏛️ IBBI (Insolvency & Bankruptcy Board) (5 databases)
- **IBBI NCLT Orders** (`ibbi_nclt_orders`)  
- **IBBI Supreme Court Orders** (`ibbi_supreme_court_orders`)  
- **IBBI Orders** (`ibbi_orders`)  
- **IBBI NCLAT Orders** (`ibbi_nclat_orders`)  
- **IBBI High Court Orders** (`ibbi_high_courts_orders`)  

### 🌍 Global Sanctions & International (4 databases)
- **Global SDN (OFAC)** (`GLOBAL_SDN`)  
- **World Bank Sanctioned** (`world_bank_sanctioned`)  
- **Euro Sanctions** (`euro_sanction`)  
- **ESMA Sanctions** (`esma_sanctions`)  

### 🗳️ Political & Public Data (1 database)
- **Indian Local Politicians** (`indian_local_politicians`)  

**Total Coverage**: 29 Databases
""")

st.markdown("## 🔎 Search Features")
st.write("""
- **Exact Search**: `searchType: "exact"` (Perfect matches only)  
- **Partial Search**: `searchType: "partial"` (Contains-based, default)  
- **Multi-field Matching**: Company names, PAN numbers, Director names, Case IDs  
- **Authentication & Security**: Requires valid `userId` + Authorization header  
""")

# ---------------- Search Form ----------------
st.markdown("## 🚀 Run a Search")
with st.form(key="indiav1_search_form"):
    query = st.text_input("🔍 Enter company name, CIN, PAN, etc.")
    submit_btn = st.form_submit_button("Search")

# ---------------- API Call ----------------
if submit_btn:
    if not api_url or not jwt_token or not user_id:
        st.error("API URL, JWT token, or user ID missing. Please check your secrets.")
    elif not query.strip():
        st.error("Please enter a search query.")
    else:
        payload = {"query": query, "userId": user_id}
        headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}

        try:
            with st.spinner("Sniffing records... 🐕"):
                response = requests.post(api_url, headers=headers, data=json.dumps(payload))

            if response.ok:
                data = response.json()

                # Show quick stats
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                stats_col1.metric("Execution Time (ms)", data.get("executionTimeMs", "N/A"))
                stats_col2.metric("Total Matches", data.get("totalMatches", "N/A"))
                stats_col3.metric("Tables With Matches", len(data.get("tablesWithMatches", [])))

                st.markdown("---")

                results = data.get("results", [])
                results_with_matches = [t for t in results if t.get("matches")]

                if results_with_matches:
                    for table_result in results_with_matches:
                        table_name = table_result.get("table", "Unknown")
                        matches = table_result.get("matches", [])

                        st.subheader(f"📂 Table: {table_name}")

                        # Convert matches into dataframe
                        df = pd.DataFrame(matches)
                        st.dataframe(df, use_container_width=True)

                        # Add download button
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"⬇️ Download {table_name} Matches",
                            data=csv,
                            file_name=f"{table_name}_matches.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("⚠️ No enforcement matches found in any table.")
            else:
                st.error(f"Request failed: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"🚨 Error contacting API: {e}")
