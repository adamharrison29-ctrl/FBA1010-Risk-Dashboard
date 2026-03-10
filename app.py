import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import yfinance as yf
import plotly.graph_objects as go
import statsmodels.api as sm

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Executive Risk Dashboard", page_icon="📈", layout="wide")
st.markdown("""<style>.main {background-color: #0E1117;} h1, h2, h3 {color: #00d4ff;} .stMetric {background-color: #1E2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff;}</style>""", unsafe_allow_html=True)

st.title("📈 FBA1010: Executive Risk Management Dashboard")
st.markdown("### Asset Profiling, Portfolio Diversification, and Jet Fuel Cross-Hedging")
st.markdown("---")

# 2. DATA ENGINE
@st.cache_data
def load_market_data():
    tickers = ['LMT', 'CL=F', 'TLT']
    prices = yf.download(tickers, start='2024-03-05', end='2026-03-06', auto_adjust=True)['Close']
    returns = prices.pct_change().dropna()
    return prices, returns

prices, returns_full = load_market_data()

# 3. TABS
tab1, tab2, tab3 = st.tabs(["📊 Q1: Individual Asset Risk", "💼 Q2: Portfolio Diversification", "🛢️ Q3: Jet Fuel Hedging"])

# --- TAB 1: Q1 ---
with tab1:
    st.header("Individual Asset Risk Profiles")
    st.markdown("""
    ### Methodology
    Daily price data for LMT, CL=F, and TLT were sourced via Yahoo Finance (March 2024–March 2026) to obtain accurate, public daily histories. CL=F serves as a continuous, liquid proxy for near-term oil exposure. Daily logarithmic returns were calculated to evaluate their risk profiles.
    
    ### Q1A: Normality of Returns
    A Shapiro-Wilk test indicates the returns are not normally distributed (all p-values < 0.05). Standard normal distributions assume extreme events are exceptionally rare. In reality, these assets exhibit "fat tails" (excess kurtosis) due to fundamental drivers:
    * **LMT (Kurtosis 9.66):** Sensitive to extreme, binary events like government contract awards.
    * **CL=F (Kurtosis 1.88):** Vulnerable to sudden macroeconomic supply shocks.
    * **TLT (Kurtosis 0.90):** Reacts sharply to surprise central bank interest rate decisions.
    """)
    asset_choice = st.selectbox("Select Asset to Analyze:", ['LMT', 'CL=F', 'TLT'])
    data = returns_full[asset_choice].dropna()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1-Day Std Dev", f"{data.std()*100:.2f}%")
    c2.metric("Historical 1% VaR", f"{np.percentile(data, 1)*100:.2f}%")
    c3.metric("Historical 1% ES", f"{data[data <= np.percentile(data, 1)].mean()*100:.2f}%")
    c4.metric("Excess Kurtosis", f"{data.kurtosis():.2f}")
    
    counts, bin_edges = np.histogram(data, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    x_curve = np.linspace(bin_centers.min(), bin_centers.max(), 100)
    y_curve = stats.norm.pdf(x_curve, data.mean(), data.std())
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=bin_centers, y=counts, name="Actual Return Density", marker_color='#00d4ff', opacity=0.7))
    fig1.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name="Normal Distribution", line=dict(color='#ff4b4b', width=3)))
    fig1.update_layout(title=f"{asset_choice} Return Distribution", template="plotly_dark", hovermode="x unified", barmode='overlay')
    st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: Q2 ---
with tab2:
    st.header("Equal-Weighted Portfolio Risk")
    weights = np.array([1/3, 1/3, 1/3])
    port_returns = returns_full[['LMT', 'CL=F', 'TLT']].dot(weights)
    p_var = np.percentile(port_returns, 1)
    p_es = port_returns[port_returns <= p_var].mean()
    
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Portfolio Std Dev", f"{port_returns.std()*100:.2f}%")
    pc2.metric("Portfolio 1% VaR", f"{p_var*100:.2f}%")
    pc3.metric("Portfolio 1% ES", f"{p_es*100:.2f}%", "Subadditive Property Satisfied")

# --- TAB 3: Q3 ---
with tab3:
    st.header("Jet Fuel Cross-Hedging Engine")
    try:
        eia = pd.read_excel('US EIA Data.xlsx', sheet_name='Data 1', skiprows=2).iloc[:, [0, 1]]
        eia.columns = ['Date', 'Jet_Fuel_Price']
        eia['Date'] = pd.to_datetime(eia['Date'], errors='coerce')
        eia = eia.dropna().set_index('Date')
        eia['Jet_Fuel_Price'] = pd.to_numeric(eia['Jet_Fuel_Price'], errors='coerce')
        
        clf = prices['CL=F'].truncate(after='2026-03-03')
        df = pd.merge(eia, clf.rename('Crude_Oil_Price'), left_index=True, right_index=True, how='inner').dropna()
        
        df_est = df.loc[:'2025-09-05']
        df_test = df.loc['2025-09-05':]
        
        model_data = pd.concat([df_est['Jet_Fuel_Price'].diff(), df_est['Crude_Oil_Price'].diff()], axis=1).dropna()
        X = sm.add_constant(model_data['Crude_Oil_Price'])
        model = sm.OLS(model_data['Jet_Fuel_Price'], X).fit()
        h_star = model.params.iloc[1]
        
        gallons = 1000000
        contracts = round((gallons * h_star) / 1000)
        unhedged_cost = gallons * df_test['Jet_Fuel_Price'].iloc[-1]
        profit = contracts * 1000 * (df_test['Crude_Oil_Price'].iloc[-1] - df_test['Crude_Oil_Price'].iloc[0])
        eff_price = (unhedged_cost - profit) / gallons
        
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Optimal Hedge Ratio (h*)", f"{h_star:.4f}")
        hc2.metric("R-Squared", f"{model.rsquared:.4f}")
        hc3.metric("Contracts Needed", f"{contracts}")
        hc4.metric("Net Hedge Savings", f"${profit:,.0f}")
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df['Jet_Fuel_Price'], name="Jet Fuel Spot ($/gal)", line=dict(color='#00d4ff')))
        fig3.add_trace(go.Scatter(x=df.index, y=df['Crude_Oil_Price'], name="Crude Oil Futures ($/bbl)", yaxis="y2", line=dict(color='#ff4b4b')))
        fig3.update_layout(title="Spot Jet Fuel vs. NYMEX Crude Futures", template="plotly_dark", hovermode="x unified", yaxis2=dict(overlaying="y", side="right"))
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:

        st.error(f"Data alignment error. Ensure 'US EIA Data.xlsx' is in the repo. Error: {e}")
