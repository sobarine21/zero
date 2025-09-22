import streamlit as st
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker  # websocket ticker
import pandas as pd
import json
import threading
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import ta # Technical Analysis library


st.set_page_config(page_title="Kite Connect - ML & Risk Analysis", layout="wide")
st.title("Kite Connect (Zerodha) — Machine Learning & Risk Analysis Demo")

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
# Helper: init unauth client (used for login URL)
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
    # st.query_params values are lists (or str depending on Streamlit version)
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
# Utility: instruments lookup (kept as it's essential for fetching historical data)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_instruments(_kite_instance, exchange=None): # FIX: Added underscore to _kite_instance
    """
    Returns pandas.DataFrame of instrument dump.
    If exchange is None, tries to fetch all instruments (may be large).
    """
    try:
        if exchange:
            inst = _kite_instance.instruments(exchange) # FIX: Used _kite_instance
        else:
            # call without exchange may return full dump
            inst = _kite_instance.instruments() # FIX: Used _kite_instance
        df = pd.DataFrame(inst)
        # keep token as int
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        return df
    except Exception as e:
        st.warning(f"Could not fetch instruments: {e}")
        return pd.DataFrame()

def find_instrument_token(df, tradingsymbol, exchange="NSE"):
    """Lookup instrument_token given exchange and tradingsymbol (case-insensitive)."""
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    if not hits.empty:
        return int(hits.iloc[0]["instrument_token"])
    return None

# Custom helper functions for robustness and adhering to API requirements

# Helper for LTP quotes (uses kite.ltp)
def get_ltp_price(kite_instance, symbol, exchange="NSE"):
    try:
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        ltp_data = kite_instance.ltp([exchange_symbol]) # ltp expects a list of instrument keys
        return ltp_data
    except Exception as e:
        return {"error": str(e)}

# Helper for OHLC + LTP quotes (uses kite.ohlc)
def get_ohlc_quote(kite_instance, symbol, exchange="NSE"):
    try:
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        ohlc_data = kite_instance.ohlc([exchange_symbol]) # ohlc expects a list of instrument keys
        return ohlc_data
    except Exception as e:
        return {"error": str(e)}

# Helper for Full Market quotes (uses kite.quote)
def get_full_market_quote(kite_instance, symbol, exchange="NSE"):
    try:
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        quote = kite_instance.quote(exchange_symbol)
        return quote
    except Exception as e:
        return {"error": str(e)}


# Fix for historical data
def get_historical(kite_instance, symbol, from_date, to_date, interval="day", exchange="NSE"):
    try:
        inst_df = st.session_state.get("instruments_df", pd.DataFrame())
        token = find_instrument_token(inst_df, symbol, exchange)
        
        if not token:
            # If not found in loaded DF, try fetching directly from Kite and storing
            st.info(f"Instrument token for {symbol} on {exchange} not found in cache. Attempting to fetch...")
            # Pass kite_instance to load_instruments, which now expects _kite_instance
            all_instruments = load_instruments(kite_instance, exchange) 
            if not all_instruments.empty:
                st.session_state["instruments_df"] = all_instruments # Update cached DF
                token = find_instrument_token(all_instruments, symbol, exchange)
            
            if not token:
                return {"error": f"Instrument token not found for {symbol} on {exchange}. Please ensure instruments are loaded or symbol/exchange is correct."}
        
        from_datetime = datetime.combine(from_date, datetime.min.time())
        to_datetime = datetime.combine(to_date, datetime.max.time())

        data = kite_instance.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval=interval)
        return data
    except Exception as e:
        return {"error": str(e)}

# Function to add technical indicators using 'ta' library for robustness
def add_indicators(df):
    if df.empty:
        return df
    
    # Ensure columns are numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that became NaN due to coercion, or for indicators to calculate
    df.dropna(subset=['close'], inplace=True)
    if df.empty:
        return pd.DataFrame() # Return empty if no valid data remains

    # Moving Averages
    df['SMA_5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    
    # Daily Return
    df['Daily_Return'] = df['close'].pct_change() * 100
    
    # Lagged Close Price
    df['Lag_1_Close'] = df['close'].shift(1)
    
    df.fillna(method='bfill', inplace=True) # Fill NaNs at the beginning (for early indicator values)
    df.fillna(method='ffill', inplace=True) # Fill any remaining NaNs
    return df

# ---------------------------
# Sidebar quick actions / profile / logout
# ---------------------------
with st.sidebar:
    st.header("Account")
    if k:
        try:
            profile = k.profile()
            st.write("User:", profile.get("user_name") or profile.get("user_id"))
            st.write("User ID:", profile.get("user_id"))
            st.write("Login time:", profile.get("login_time"))
        except Exception:
            st.write("Authenticated (profile fetch failed)")

        if st.button("Logout (clear token)"):
            st.session_state.pop("kite_access_token", None)
            st.session_state.pop("kite_login_response", None)
            # Clear all session state for a clean re-run on logout
            for key in list(st.session_state.keys()):
                st.session_state.pop(key)
            st.success("Logged out. Please login again.")
            st.experimental_rerun()
    else:
        st.info("Not authenticated yet. Login using the link above.")

# ---------------------------
# Main UI - Tabs for modules
# ---------------------------
tabs = st.tabs(["Portfolio", "Orders", "Market & Historical", "Machine Learning Analysis", "Risk & Stress Testing", "Websocket (stream)", "Instruments Utils"])
tab_portfolio, tab_orders, tab_market, tab_ml, tab_risk, tab_ws, tab_inst = tabs

# ---------------------------
# TAB: PORTFOLIO
# ---------------------------
with tab_portfolio:
    st.header("Portfolio")
    if not k:
        st.info("Login first to fetch portfolio data.")
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
                    # positions contains 'net' and 'day'
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

                    # place order
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
                        # note: variety is required for modify; here we assume 'regular' but user can change
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

        # Option to choose between LTP, OHLC, and full Quote
        market_data_type = st.radio("Choose data type:", 
                                     ("LTP (Last Traded Price)", "OHLC + LTP", "Full Market Quote (OHLC, Depth, OI)"), 
                                     index=0, key="market_data_type_radio")

        if st.button("Get market data"):
            market_data_response = {}
            if market_data_type == "LTP (Last Traded Price)":
                market_data_response = get_ltp_price(k, q_symbol, q_exchange)
            elif market_data_type == "OHLC + LTP":
                market_data_response = get_ohlc_quote(k, q_symbol, q_exchange)
            else: # Full Market Quote
                market_data_response = get_full_market_quote(k, q_symbol, q_exchange)
            
            if "error" in market_data_response:
                st.error(f"Market data fetch failed: {market_data_response['error']}")
                if "Insufficient permission" in market_data_response['error']:
                    st.warning("For 'Full Market Quote', you might need a paid subscription to the Kite Connect API. Try 'LTP' or 'OHLC + LTP' if you encounter permission errors.")
            else:
                st.json(market_data_response)

        st.markdown("---")
        st.subheader("Historical candles")
        # Ensure instruments are loaded for lookup
        with st.expander("Load Instruments for Historical Data Lookup"):
            exchange_for_dump = st.selectbox("Exchange to load instruments for lookup", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0, key="inst_load_exchange")
            if st.button("Load Instruments Now"):
                # Pass k to load_instruments, which now expects _kite_instance
                inst_df = load_instruments(k, exchange_for_dump)
                st.session_state["instruments_df"] = inst_df
                if not inst_df.empty:
                    st.success(f"Loaded {len(inst_df)} instruments for {exchange_for_dump}")
                else:
                    st.warning(f"Could not load instruments for {exchange_for_dump}. Check API key and permissions.")

        hist_exchange = st.selectbox("Exchange (for historical data)", ["NSE", "BSE", "NFO"], index=0, key="hist_ex")
        hist_symbol = st.text_input("Tradingsymbol (e.g., INFY)", value="INFY", key="hist_sym")
        
        # Default dates for convenience (e.g., last 3 months)
        default_to_date = datetime.now().date()
        default_from_date = default_to_date - timedelta(days=90)

        from_date = st.date_input("From date", value=default_from_date, key="from_dt")
        to_date = st.date_input("To date", value=default_to_date, key="to_dt")
        interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "day", "week", "month"], index=4)

        if st.button("Fetch historical data"):
            if "instruments_df" not in st.session_state or st.session_state["instruments_df"].empty:
                st.error("Please load instruments first from the expander above to enable symbol lookup.")
            else:
                hist_data = get_historical(k, hist_symbol, from_date, to_date, interval, hist_exchange)
                
                if "error" in hist_data:
                    st.error(f"Historical fetch failed: {hist_data['error']}")
                    if "Insufficient permission" in hist_data['error']:
                        st.warning("This error often indicates that your Zerodha API key does not have an active subscription for historical data. Please check your Kite Connect developer console for subscription status.")
                else:
                    df = pd.DataFrame(hist_data)
                    if not df.empty:
                        # Ensure 'date' column is datetime and set as index
                        if "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            df.set_index("date", inplace=True)
                            df.sort_index(inplace=True) # Ensure chronological order

                        st.session_state["historical_data"] = df # Store for ML analysis
                        st.success(f"Successfully fetched {len(df)} records for {hist_symbol} ({interval}).")
                        st.dataframe(df.head()) # Show a preview

                        # Plotting historical data
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                            vertical_spacing=0.03, 
                                            row_heights=[0.7, 0.3])

                        # Candlestick chart
                        fig.add_trace(go.Candlestick(x=df.index,
                                                    open=df['open'],
                                                    high=df['high'],
                                                    low=df['low'],
                                                    close=df['close'],
                                                    name='Candlestick'), row=1, col=1)

                        # Volume chart
                        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue'), row=2, col=1)

                        fig.update_layout(title_text=f"{hist_symbol} Historical Data - {interval}",
                                          xaxis_rangeslider_visible=False,
                                          height=600,
                                          template="plotly_white") # A clean theme
                        fig.update_yaxes(title_text="Price", row=1, col=1)
                        fig.update_yaxes(title_text="Volume", row=2, col=1)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No historical data returned for the selected period and symbol.")

# ---------------------------
# TAB: MACHINE LEARNING ANALYSIS
# ---------------------------
with tab_ml:
    st.header("Machine Learning Based Analysis")

    if not k:
        st.info("Login first to perform ML analysis.")
    else:
        st.write("This module allows you to perform machine learning analysis on historical price data.")
        st.write("First, ensure you have fetched historical data from the 'Market & Historical' tab.")

        historical_data = st.session_state.get("historical_data")

        if historical_data is None or historical_data.empty:
            st.warning("No historical data available. Please fetch data from the 'Market & Historical' tab first.")
        else:
            st.subheader("1. Data Preprocessing & Feature Engineering")
            st.dataframe(historical_data.head())

            if st.button("Add Technical Indicators & Prepare Data"):
                df_with_indicators = add_indicators(historical_data.copy())
                if df_with_indicators.empty:
                    st.error("Failed to add indicators. Data might be too short or contains invalid values.")
                    st.session_state["ml_data"] = None
                else:
                    st.session_state["ml_data"] = df_with_indicators
                    st.success("Technical indicators added to data.")
                    st.dataframe(df_with_indicators.head())
                    
                    # Plotting indicators
                    fig_indicators = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                                vertical_spacing=0.05,
                                                row_heights=[0.5, 0.15, 0.15, 0.15],
                                                subplot_titles=("Price & Moving Averages/Bollinger Bands", "RSI", "MACD", "Volume"))

                    # Candlestick and SMAs, Bollinger Bands
                    fig_indicators.add_trace(go.Candlestick(x=df_with_indicators.index,
                                                            open=df_with_indicators['open'],
                                                            high=df_with_indicators['high'],
                                                            low=df_with_indicators['low'],
                                                            close=df_with_indicators['close'],
                                                            name='Candlestick'), row=1, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['SMA_5'], mode='lines', name='SMA 5', line=dict(color='orange')), row=1, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['SMA_20'], mode='lines', name='SMA 20', line=dict(color='purple')), row=1, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['Bollinger_High'], mode='lines', name='Bollinger High', line=dict(color='green', dash='dot')), row=1, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['Bollinger_Low'], mode='lines', name='Bollinger Low', line=dict(color='red', dash='dot')), row=1, col=1)

                    # RSI
                    fig_indicators.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['RSI'], mode='lines', name='RSI', line=dict(color='blue')), row=2, col=1)
                    fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top right", row=2, col=1)
                    fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right", row=2, col=1)

                    # MACD
                    fig_indicators.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD'], mode='lines', name='MACD', line=dict(color='green')), row=3, col=1)
                    fig_indicators.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD_signal'], mode='lines', name='MACD Signal', line=dict(color='red')), row=3, col=1)
                    
                    # Volume
                    fig_indicators.add_trace(go.Bar(x=df_with_indicators.index, y=df_with_indicators['volume'], name='Volume', marker_color='blue'), row=4, col=1)

                    fig_indicators.update_layout(title_text=f"{historical_data.columns.name if historical_data.columns.name else 'Instrument'} Price with Technical Indicators",
                                                xaxis_rangeslider_visible=False,
                                                height=900,
                                                template="plotly_white")
                    st.plotly_chart(fig_indicators, use_container_width=True)

            ml_data = st.session_state.get("ml_data")
            if ml_data is not None and not ml_data.empty:
                st.subheader("2. Machine Learning Model Training")

                model_type = st.selectbox("Select ML Model", ["Linear Regression", "Random Forest Regressor", "LightGBM Regressor"])
                target_column = st.selectbox("Select Target Variable (e.g., 'close' for next day close price)", ["close"])
                
                # Shift target for prediction: predicting next period's close price
                ml_data['target'] = ml_data[target_column].shift(-1)
                ml_data.dropna(subset=['target'], inplace=True) # Drop the last row as it won't have a target
                
                features = [col for col in ml_data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
                if not features:
                    st.warning("No features available after dropping target and basic OHLCV columns. Please ensure indicators are added and valid.")
                    st.stop()
                
                # Check for infinite values in features or target and replace with NaN, then drop
                ml_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                ml_data.dropna(subset=features + ['target'], inplace=True)


                X = ml_data[features]
                y = ml_data['target']

                if X.empty or y.empty:
                    st.warning("Not enough clean data after preprocessing to train the model. Try a longer historical period.")
                    st.stop()

                test_size = st.slider("Test Size Percentage", min_value=10, max_value=50, value=20) / 100.0
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

                st.write(f"Training data size: {len(X_train)} samples")
                st.write(f"Testing data size: {len(X_test)} samples")


                if st.button(f"Train {model_type} Model"):
                    if len(X_train) == 0 or len(X_test) == 0:
                        st.error("Insufficient data for training or testing after split. Adjust test size or fetch more data.")
                    else:
                        model = None
                        if model_type == "Linear Regression":
                            model = LinearRegression()
                        elif model_type == "Random Forest Regressor":
                            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use all cores
                        elif model_type == "LightGBM Regressor":
                            model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)

                        if model:
                            with st.spinner(f"Training {model_type} model..."):
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                            st.session_state["ml_model"] = model
                            st.session_state["y_test"] = y_test
                            st.session_state["y_pred"] = y_pred
                            st.session_state["X_test_ml"] = X_test
                            st.session_state["ml_features"] = features # Store features for real-time simulation

                            st.success(f"{model_type} Model Trained!")
                            st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
                            st.write(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

                            # Plotting predictions
                            pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
                            fig_pred = go.Figure()
                            fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'], mode='lines', name='Actual Price'))
                            fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'], mode='lines', name='Predicted Price', line=dict(dash='dot')))
                            fig_pred.update_layout(title_text=f"{model_type} Actual vs. Predicted Prices on Test Set", 
                                                height=500, xaxis_title="Date", yaxis_title="Price",
                                                template="plotly_white")
                            st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.info("Please add technical indicators first to proceed with ML training.")

            st.subheader("3. Real-time Price Prediction (Simulated)")
            st.write("This section demonstrates the *workflow* for using a trained model for real-time predictions.")
            st.write("For an actual real-time prediction, a live system would typically:")
            st.markdown("- Continuously fetch the latest OHLCV data and current LTP using the WebSocket or direct API calls.")
            st.markdown("- Aggregate this data into candles (if predicting at interval, e.g., 5-min candles).")
            st.markdown("- Compute the required technical indicators based on the most recent complete data points.")
            st.markdown("- Feed these latest features into the trained model to get a prediction for the next period's close price.")

            if st.session_state.get("ml_model") and st.session_state.get("X_test_ml") is not None:
                model = st.session_state["ml_model"]
                X_test_ml = st.session_state["X_test_ml"]
                ml_features = st.session_state["ml_features"]

                st.write(f"Model trained: {type(model).__name__}")
                st.write(f"Features used for prediction: {', '.join(ml_features)}")

                # Simulate a real-time data point using the last known features from our test set
                if not X_test_ml.empty:
                    latest_features_df = X_test_ml.iloc[[-1]] # Take the last row of features from test set
                    if st.button("Simulate Real-time Prediction"):
                        with st.spinner("Generating simulated prediction..."):
                            simulated_prediction = model.predict(latest_features_df)[0]
                        st.success(f"Simulated next period close price prediction: **{simulated_prediction:.2f}**")
                        st.info("This is a simulation using the last available test data point's features. In a live trading system, these features would be derived from fresh, real-time market data.")
                else:
                    st.warning("No test data available for simulation. Please train the model first.")
            else:
                st.info("Train a machine learning model first to see a real-time prediction simulation.")

# ---------------------------
# TAB: RISK & STRESS TESTING
# ---------------------------
with tab_risk:
    st.header("Risk & Stress Testing Models and Algorithms")

    if not k:
        st.info("Login first to perform risk analysis.")
    else:
        st.write("This module helps you understand the potential risks and perform stress tests on a specific instrument using its historical data.")
        st.write("First, ensure you have fetched historical data from the 'Market & Historical' tab.")
        
        historical_data = st.session_state.get("historical_data")

        if historical_data is None or historical_data.empty:
            st.warning("No historical data available. Please fetch data from the 'Market & Historical' tab first.")
        else:
            if historical_data.index.duplicated().any():
                historical_data = historical_data.loc[~historical_data.index.duplicated(keep='first')]
            
            # Ensure 'close' column is numeric for calculations
            historical_data['close'] = pd.to_numeric(historical_data['close'], errors='coerce')
            historical_data.dropna(subset=['close'], inplace=True)

            daily_returns = historical_data['close'].pct_change().dropna()
            
            if daily_returns.empty:
                st.warning("Not enough valid data to compute daily returns. Ensure historical data is fetched correctly and contains multiple valid 'close' prices.")
                st.stop()

            st.subheader("1. Volatility Analysis (Historical)")
            st.write("Volatility measures the degree of variation of a trading price series over time.")
            
            st.dataframe(daily_returns.describe())

            # Annualized volatility (assuming 252 trading days for 'day' interval, adjust for others if needed)
            trading_days_per_year = 252 # Typical for equities
            annualized_volatility = daily_returns.std() * np.sqrt(trading_days_per_year) * 100
            st.metric(label="Annualized Volatility (based on daily returns)", value=f"{annualized_volatility:.2f}%")

            fig_volatility = go.Figure()
            fig_volatility.add_trace(go.Histogram(x=daily_returns, nbinsx=50, name='Daily Returns', marker_color='#1f77b4'))
            fig_volatility.update_layout(title_text='Distribution of Daily Returns',
                                         xaxis_title='Daily Return (%)',
                                         yaxis_title='Frequency',
                                         height=400, template="plotly_white")
            st.plotly_chart(fig_volatility, use_container_width=True)

            st.subheader("2. Value at Risk (VaR) Calculation (Historical Method)")
            st.write("Value at Risk (VaR) estimates the maximum expected loss over a specific time horizon at a given confidence level.")
            
            confidence_level = st.slider("Confidence Level (%)", min_value=90, max_value=99, value=95, step=1)
            holding_period_var = st.number_input("Holding Period for VaR (days)", min_value=1, value=1)
            
            # Calculate VaR using the historical percentile method
            # For 1-day VaR, directly use daily_returns
            var_percentile_1day = np.percentile(daily_returns, 100 - confidence_level)

            st.write(f"For a **{holding_period_var}-day holding period**:")
            # Scale 1-day VaR for multiple days (simplified assumption: returns are independent and identically distributed)
            # This is a simplification; more complex methods exist for multi-period VaR
            var_percentile_multiday = var_percentile_1day * np.sqrt(holding_period_var) # Using root-t rule

            st.metric(label=f"Value at Risk ({confidence_level}%)", value=f"{abs(var_percentile_multiday):.2f}% (over {holding_period_var} days)")

            current_price = historical_data['close'].iloc[-1]
            potential_loss = (abs(var_percentile_multiday) / 100) * current_price
            st.write(f"This means, with {confidence_level}% confidence, the maximum potential loss on a position currently worth **₹{current_price:.2f}** is approximately **₹{potential_loss:.2f}** over {holding_period_var} day(s).")
            
            # Plotting VaR on returns distribution
            fig_var = go.Figure()
            fig_var.add_trace(go.Histogram(x=daily_returns, nbinsx=50, name='Daily Returns', marker_color='#1f77b4'))
            fig_var.add_vline(x=var_percentile_1day, line_dash="dash", line_color="red", 
                              annotation_text=f"1-Day VaR {confidence_level}%: {var_percentile_1day:.2f}%", 
                              annotation_position="top right")
            fig_var.update_layout(title_text=f'Daily Returns Distribution with {confidence_level}% VaR',
                                  xaxis_title='Daily Return (%)',
                                  yaxis_title='Frequency',
                                  height=400, template="plotly_white")
            st.plotly_chart(fig_var, use_container_width=True)

            st.subheader("3. Stress Testing (Scenario Analysis)")
            st.write("Stress testing evaluates the impact of extreme but plausible market movements on an asset's price.")
            
            # Pre-defined scenarios
            scenarios = {
                "Historical Worst Day Drop": {"type": "historical", "percent": daily_returns.min()},
                "Global Financial Crisis (-20%)": {"type": "fixed", "percent": -20.0},
                "Flash Crash (-10%)": {"type": "fixed", "percent": -10.0},
                "Moderate Correction (-5%)": {"type": "fixed", "percent": -5.0},
                "Significant Rally (+10%)": {"type": "fixed", "percent": 10.0}
            }

            scenario_key = st.selectbox("Select Stress Scenario", list(scenarios.keys()))
            
            if st.button("Run Stress Test"):
                current_price = historical_data['close'].iloc[-1]
                scenario_data = scenarios[scenario_key]
                stressed_price = 0
                scenario_change_percent = 0

                if scenario_data["type"] == "historical":
                    scenario_change_percent = scenario_data["percent"]
                    stressed_price = current_price * (1 + scenario_change_percent / 100)
                    st.write(f"Scenario: **{scenario_key}** (Worst historical daily drop: **{scenario_change_percent:.2f}%** in daily returns)")
                else: # fixed scenarios
                    scenario_change_percent = scenario_data["percent"]
                    stressed_price = current_price * (1 + scenario_change_percent / 100)
                    st.write(f"Scenario: **{scenario_key}** (Fixed change: **{scenario_change_percent:.2f}%**)")
                
                st.metric(label="Current Price", value=f"₹{current_price:.2f}")
                st.metric(label="Stressed Price", value=f"₹{stressed_price:.2f}")
                st.metric(label="Potential Price Change", value=f"₹{(stressed_price - current_price):.2f}")
                st.metric(label="Percentage Change", value=f"{scenario_change_percent:.2f}%")

                st.info("This is a simplified stress test on a single instrument. A comprehensive stress test involves modeling the impact across an entire portfolio, considering correlations between assets, and the behavior of derivatives.")


# ---------------------------
# TAB: WEBSOCKET (Ticker)
# ---------------------------
with tab_ws:
    st.header("WebSocket streaming — KiteTicker (authenticated)")
    st.write("Start the KiteTicker to receive live ticks. This component uses threads — click Start, then Stop to disconnect.")

    if not k:
        st.info("Login first to start websocket.")
    else:
        # session state for ticker & ticks
        if "kt_ticker" not in st.session_state:
            st.session_state["kt_ticker"] = None
        if "kt_thread" not in st.session_state:
            st.session_state["kt_thread"] = None
        if "kt_running" not in st.session_state:
            st.session_state["kt_running"] = False
        if "kt_ticks" not in st.session_state:
            st.session_state["kt_ticks"] = []

        # Autoload instruments if not already loaded, for convenience in getting tokens
        if "instruments_df" not in st.session_state or st.session_state["instruments_df"].empty:
            st.info("Loading instruments for NSE to facilitate instrument token lookup for WebSocket.")
            # Pass k to load_instruments, which now expects _kite_instance
            nse_instruments = load_instruments(k, "NSE") 
            if not nse_instruments.empty:
                st.session_state["instruments_df"] = nse_instruments
                st.success(f"Loaded {len(nse_instruments)} instruments for NSE.")
            else:
                st.warning("Could not load NSE instruments. WebSocket token lookup might be limited.")
        
        # Helper to look up token from trading symbol
        ws_exchange = st.selectbox("Exchange for WebSocket symbol lookup", ["NSE", "BSE", "NFO"], index=0, key="ws_lookup_ex")
        ws_tradingsymbol = st.text_input("Tradingsymbol for WebSocket (e.g., INFY)", value="INFY", key="ws_lookup_sym")
        
        instrument_token_for_ws = None
        if st.button("Lookup Instrument Token"):
            if "instruments_df" in st.session_state and not st.session_state["instruments_df"].empty:
                instrument_token_for_ws = find_instrument_token(st.session_state["instruments_df"], ws_tradingsymbol, ws_exchange)
                if instrument_token_for_ws:
                    st.success(f"Found instrument_token for {ws_tradingsymbol}: {instrument_token_for_ws}")
                    st.session_state["ws_instrument_token"] = str(instrument_token_for_ws) # Store as string for input
                else:
                    st.warning(f"Could not find instrument token for {ws_tradingsymbol} on {ws_exchange}.")
            else:
                st.warning("Please load instruments first from 'Market & Historical' tab or 'Instruments Utils' tab.")

        # Use the looked-up token or allow manual input
        symbol_for_ws = st.text_input("Instrument token(s) comma separated (e.g. 738561,3409)", 
                                      value=st.session_state.get("ws_instrument_token", ""),
                                      key="ws_symbol_input")
        st.caption("Enter numeric instrument token(s) or use the lookup above. Leave blank to subscribe none (you can subscribe later).")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start ticker") and not st.session_state["kt_running"]:
                try:
                    access_token = st.session_state["kite_access_token"]
                    user_id = st.session_state["kite_login_response"].get("user_id")
                    
                    try:
                        kt = KiteTicker(user_id, access_token, API_KEY)
                    except Exception:
                        kt = KiteTicker(API_KEY, access_token)

                    st.session_state["kt_ticker"] = kt
                    st.session_state["kt_running"] = True
                    st.session_state["kt_ticks"] = []
                    st.session_state["kt_status_message"] = "Ticker connecting..." # Initial status

                    def on_connect(ws, response):
                        st.session_state["kt_ticks"].append({"event": "connected", "time": datetime.utcnow().isoformat()})
                        st.session_state["kt_status_message"] = "Ticker connected. Subscribing..."
                        if symbol_for_ws:
                            tokens = [int(x.strip()) for x in symbol_for_ws.split(",") if x.strip()]
                            if tokens:
                                try:
                                    ws.subscribe(tokens)
                                    ws.set_mode(ws.MODE_FULL, tokens)
                                    st.session_state["kt_status_message"] = f"Subscribed to {len(tokens)} tokens."
                                except Exception as e:
                                    st.session_state["kt_ticks"].append({"event": "subscribe_error", "error": str(e), "time": datetime.utcnow().isoformat()})
                                    st.session_state["kt_status_message"] = f"Subscription error: {e}"
                            else:
                                st.session_state["kt_status_message"] = "Connected, but no tokens provided for subscription."
                        else:
                            st.session_state["kt_status_message"] = "Connected, no tokens specified for initial subscription."

                    def on_ticks(ws, ticks):
                        for t in ticks:
                            t["_ts"] = datetime.utcnow().isoformat()
                            st.session_state["kt_ticks"].append(t)
                        if len(st.session_state["kt_ticks"]) > 200:
                            st.session_state["kt_ticks"] = st.session_state["kt_ticks"][-200:]
                        # Small trick to force Streamlit to re-render the tick dataframe
                        st.session_state["_last_tick_update"] = time.time()

                    def on_close(ws, code, reason):
                        st.session_state["kt_ticks"].append({"event": "closed", "code": code, "reason": reason, "time": datetime.utcnow().isoformat()})
                        st.session_state["kt_running"] = False
                        st.session_state["kt_status_message"] = f"Ticker disconnected: {reason} (Code: {code})"

                    def on_error(ws, code, reason):
                        st.session_state["kt_ticks"].append({"event": "error", "code": code, "reason": reason, "time": datetime.utcnow().isoformat()})
                        st.session_state["kt_status_message"] = f"Ticker error: {reason} (Code: {code})"


                    kt.on_connect = on_connect
                    kt.on_ticks = on_ticks
                    kt.on_close = on_close
                    kt.on_error = on_error # Add error handler

                    def run_ticker():
                        try:
                            kt.connect(threaded=True)
                            while st.session_state["kt_running"]:
                                time.sleep(0.5) # Keep the thread alive
                        except Exception as e:
                            st.session_state["kt_ticks"].append({"event": "fatal_error", "error": str(e)})
                            st.session_state["kt_running"] = False
                            st.session_state["kt_status_message"] = f"Fatal ticker error: {e}"

                    th = threading.Thread(target=run_ticker, daemon=True)
                    st.session_state["kt_thread"] = th
                    th.start()
                    st.success("Ticker start attempt initiated. Check status below.")
                except Exception as e:
                    st.error(f"Failed to start ticker: {e}")
                    st.session_state["kt_status_message"] = f"Failed to start ticker: {e}"

        with col2:
            if st.button("Stop ticker") and st.session_state.get("kt_running"):
                try:
                    kt = st.session_state.get("kt_ticker")
                    if kt:
                        kt.disconnect()
                    st.session_state["kt_running"] = False
                    st.session_state["kt_status_message"] = "Ticker explicitly stopped."
                    st.success("Ticker stopped.")
                except Exception as e:
                    st.error(f"Failed to stop ticker: {e}")
                    st.session_state["kt_status_message"] = f"Failed to stop ticker: {e}"
        
        # Live status updates
        st.info(f"Ticker Status: {st.session_state.get('kt_status_message', 'Not started')}")

        st.markdown("#### Latest ticks (most recent 100)")
        ticks = st.session_state.get("kt_ticks", [])
        
        # Use a placeholder and update it to show live ticks without rerunning the whole app
        tick_placeholder = st.empty()
        
        if ticks:
            df_ticks = pd.json_normalize(ticks[-100:][::-1])
            # Only display a subset of columns to keep it readable, or allow user to select
            display_cols = ['_ts', 'instrument_token', 'last_price', 'ohlc.open', 'ohlc.high', 'ohlc.low', 'ohlc.close', 'volume', 'change']
            
            # Filter available columns
            available_cols = [col for col in display_cols if col in df_ticks.columns]
            
            tick_placeholder.dataframe(df_ticks[available_cols], use_container_width=True)
        else:
            tick_placeholder.write("No ticks yet. Start ticker and/or subscribe tokens.")

# ---------------------------
# TAB: INSTRUMENTS UTILS
# ---------------------------
with tab_inst:
    st.header("Instrument Lookup and Utilities")
    st.write("This utility helps you find instrument tokens, which are essential for fetching historical data or subscribing to live market data via WebSocket.")
    
    inst_exchange = st.selectbox("Select Exchange to Load Instruments", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0)
    if st.button("Load Instruments for Selected Exchange (cached)"):
        try:
            # Pass k to load_instruments, which now expects _kite_instance
            df = load_instruments(k, inst_exchange)
            st.session_state["instruments_df"] = df
            if not df.empty:
                st.success(f"Loaded {len(df)} instruments for {inst_exchange}.")
            else:
                st.warning(f"Could not load instruments for {inst_exchange}. Check API permissions or if the exchange has instruments available.")
        except Exception as e:
            st.error(f"Failed to load instruments: {e}")

    df_instruments = st.session_state.get("instruments_df", pd.DataFrame())
    if not df_instruments.empty:
        st.subheader("Search Instrument Token")
        search_symbol = st.text_input(f"Enter Tradingsymbol (e.g., INFY for {inst_exchange})", value="INFY", key="inst_search_sym")
        search_exchange = st.selectbox("Specify Exchange for Search", ["NSE", "BSE", "NFO", "CDS", "MCX"], index=0, key="inst_search_ex")
        
        if st.button("Find Token"):
            token = find_instrument_token(df_instruments, search_symbol, search_exchange)
            if token:
                st.success(f"Found instrument_token for {search_symbol} on {search_exchange}: **{token}**")
            else:
                st.warning(f"Instrument token not found for '{search_symbol}' on '{search_exchange}'. Ensure correct symbol/exchange and that instruments for this exchange are loaded.")

        st.subheader("Preview Loaded Instruments (first 200 rows)")
        st.dataframe(df_instruments.head(200), use_container_width=True)
    else:
        st.info("No instruments loaded. Click 'Load Instruments for Selected Exchange' above to fetch data.")
