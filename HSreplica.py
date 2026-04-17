import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime

# --- PAGE SETUP ---
st.set_page_config(page_title="hittezoeker", layout="centered")

st.markdown("""
<div style="text-align: center;">
    <p>realtime GEX profile, ticker: SPY</p>
</div>
""", unsafe_allow_html=True)

# --- MATH SETUP ---
def calculate_gamma(S, K, T, r, sigma):
    """Calculates Option Gamma using Black-Scholes"""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# --- DATA FETCHING ---
@st.cache_data(ttl=300)
def get_expirations(ticker_symbol):
    return yf.Ticker(ticker_symbol).options

@st.cache_data(ttl=60)
def get_options_data(ticker_symbol, expiry_date):
    ticker = yf.Ticker(ticker_symbol)
    spot_price = ticker.history(period="1d")['Close'].iloc[-1]
    
    chain = ticker.option_chain(expiry_date)
    calls = chain.calls
    puts = chain.puts
    
    expiry_dt = datetime.datetime.strptime(expiry_date, "%Y-%m-%d").date()
    today = datetime.date.today()
    days_to_expiry = max((expiry_dt - today).days, 1) 
    T = days_to_expiry / 365.0
    
    risk_free_rate = 0.053 
    
    calls['impliedVolatility'] = calls['impliedVolatility'].fillna(0.15).replace(0, 0.15)
    puts['impliedVolatility'] = puts['impliedVolatility'].fillna(0.15).replace(0, 0.15)
    calls['openInterest'] = calls['openInterest'].fillna(0)
    puts['openInterest'] = puts['openInterest'].fillna(0)
    
    calls['Gamma_Calc'] = calls.apply(lambda row: calculate_gamma(spot_price, row['strike'], T, risk_free_rate, row['impliedVolatility']), axis=1)
    calls['GEX'] = calls['Gamma_Calc'] * calls['openInterest'] * 100 * spot_price
    
    puts['Gamma_Calc'] = puts.apply(lambda row: calculate_gamma(spot_price, row['strike'], T, risk_free_rate, row['impliedVolatility']), axis=1)
    puts['GEX'] = puts['Gamma_Calc'] * puts['openInterest'] * 100 * spot_price * -1

    gex_df = pd.merge(calls[['strike', 'GEX']], puts[['strike', 'GEX']], on='strike', how='outer', suffixes=('_call', '_put')).fillna(0)
    gex_df['Net_GEX'] = gex_df['GEX_call'] + gex_df['GEX_put']
    
    lower_bound = spot_price * 0.95
    upper_bound = spot_price * 1.05
    gex_df = gex_df[(gex_df['strike'] >= lower_bound) & (gex_df['strike'] <= upper_bound)]
    
    return gex_df.sort_values('strike', ascending=False), spot_price

# --- UI CONTROLS ---
ticker_sym = "SPY"
expirations = get_expirations(ticker_sym)

if not expirations:
    st.error("Failed to fetch expirations")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    selected_expiry = st.selectbox("expiry date:", expirations, index=0)

with st.spinner(f"Fetching live data for {selected_expiry}..."):
    gex_data, spot = get_options_data(ticker_sym, selected_expiry)

with col2:
    st.markdown(f"""
<div style="padding-top: 28px;">
    <h4>prijs: <span style="color: #2ecc71;">${spot:.2f}</span></h4>
</div>
""", unsafe_allow_html=True)

# ---  VISUALIZATION ---
if gex_data is not None and not gex_data.empty:
    display_df = gex_data[['strike', 'Net_GEX']].copy()
    display_df.set_index('strike', inplace=True)
    
    max_abs_gex = display_df['Net_GEX'].abs().max()
    max_gex_idx = display_df['Net_GEX'].abs().idxmax()
    total_abs_gex = display_df['Net_GEX'].abs().sum()
    
    # no indentation on raw HTML block (avoid code snippet in streamlit)
    html = """
<style>
.gex-wrapper {
    display: flex;
    justify-content: center;
    background-color: #0e1117;
    padding: 10px;
    border-radius: 8px;
}
.gex-table {
    border-collapse: collapse;
    width: 100%;
    max-width: 500px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 14px;
    color: white;
}
.gex-table th, .gex-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #1e2530;
}
.gex-table td {
    text-align: right;
}
.gex-table td:first-child {
    text-align: left;
    font-weight: bold;
    width: 60px;
    border-right: 1px solid #1e2530;
    background-color: #1a1c24;
}
.king-row {
    background-color: #fce803 !important;
    color: black !important;
    font-weight: bold;
}
</style>
<div class="gex-wrapper">
<table class="gex-table">
"""
    
    for strike, row in display_df.iterrows():
        val = row['Net_GEX']
        
        is_neg = val < 0
        abs_val_k = abs(val) / 1000
        formatted_val = f"-${abs_val_k:,.1f}K" if is_neg else f"${abs_val_k:,.1f}K"
        
        pct = (val / total_abs_gex * 100) if total_abs_gex > 0 else 0
        sign = "+" if pct >= 0 else ""
        pct_text = f"{sign}{pct:.0f}%"
        
        row_class = ""
        row_style = ""
        pct_color = "white"
        
        if strike == max_gex_idx:
            row_class = "king-row"
            formatted_val += " ★"
            pct_color = "black"
        else:
            intensity = min(abs(val) / max_abs_gex, 1.0) if max_abs_gex > 0 else 0
            opacity = 0.2 + (intensity * 0.8) 
            
            if val > 0:
                row_style = f"background-color: rgba(46, 204, 113, {opacity});"
            else:
                row_style = f"background-color: rgba(142, 68, 173, {opacity});"
        
        pct_badge = f"<span style='float: left; font-size: 11px; color: {pct_color}; background: rgba(0,0,0,0.2); padding: 2px 4px; border-radius: 3px;'>{pct_text}</span>"
        
        # Single line HTML generation per row to prevent Markdown code block triggering
        html += f"<tr class='{row_class}' style='{row_style}'><td>{int(strike)}</td><td>{pct_badge} {formatted_val}</td></tr>"
        
    html += "</table></div>"
    
    st.markdown(html, unsafe_allow_html=True)
else:
    st.warning("No option data found. This usually happens after hours. Try another expiry date.")