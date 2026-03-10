import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import yfinance as yf
import plotly.graph_objects as go
import statsmodels.api as sm

# ==========================================
# 1. PAGE CONFIGURATION & UI ENGINE
# ==========================================
st.set_page_config(page_title="Executive Risk Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* 1. Terminal Grid Background */
    .stApp {
        background-color: #0E1117;
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
    }
    
    /* 2. Premium Typography */
    p, li {
        color: #cfd8dc !important; 
        font-size: 1.05rem;
        line-height: 1.8 !important;
        letter-spacing: 0.2px;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }
    
    /* 3. Elevated Metric Cards */
    [data-testid="metric-container"] {
        background-color: #161a24;
        border: 1px solid #2d3342;
        padding: 15px 20px;
        border-radius: 12px;
        border-left: 5px solid #00d4ff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
    }
    
    /* 4. Styled Pill Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #11141c;
        border-radius: 12px;
        padding: 5px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b9bb4;
        border-radius: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #002060 0%, #063970 100%) !important;
        color: #00d4ff !important;
        border: 1px solid #00d4ff !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
    }
    
    /* 5. Clean DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #2d3342;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. THE SIDEBAR (Framing the page)
# ==========================================
with st.sidebar:
    # This will load the image directly from your GitHub folder!
    try:
        st.image("dcu_logo.jpg", use_container_width=True)
    except FileNotFoundError:
        # A quick backup text header just in case the file name is slightly off
        st.markdown("<h2 style='text-align: center; color: #00d4ff;'>DCU Business School</h2>", unsafe_allow_html=True)
        
    st.header("⚙️ Dashboard Controls")
    st.markdown("This dashboard acts as an interactive companion to the FBA1010 Quantitative Risk Management Report.")
    st.markdown("---")
    st.markdown("**Module:** FBA1010")
    st.markdown("**Date:** March 2026")

# ==========================================
# 3. HERO BANNER
# ==========================================
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #002060 0%, #00d4ff 100%);
        padding: 40px; 
        border-radius: 15px; 
        text-align: center; 
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);">
        <h1 style="
            color: white !important; 
            margin: 0; 
            font-size: 3.2em; 
            font-weight: 800; 
            letter-spacing: 1px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.6);">
            📊 FBA1010: Quantitative Risk Terminal
        </h1>
        <p style="
            color: #f0f2f6 !important; 
            font-size: 1.3em; 
            margin-top: 15px; 
            margin-bottom: 0;
            font-weight: 400;
            letter-spacing: 0.5px;">
            Asset Profiling &nbsp; | &nbsp; Portfolio Diversification &nbsp; | &nbsp; Jet Fuel Cross-Hedging
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. DATA ENGINE
# ==========================================
@st.cache_data
def load_market_data():
    tickers = ['LMT', 'CL=F', 'TLT']
    prices = yf.download(tickers, start='2024-03-05', end='2026-03-06', auto_adjust=True)['Close']
    returns = prices.pct_change().dropna()
    return prices, returns

prices, returns_full = load_market_data()

# ==========================================
# 5. TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Q1: Individual Asset Risk", "💼 Q2: Portfolio Diversification", "🛢️ Q3: Jet Fuel Hedging"])

# ------------------------------------------
# --- TAB 1: Q1 ---
# ------------------------------------------
with tab1:
    st.markdown("""
    ### Methodology
    Daily price data for three assets, Lockheed Martin (LMT) representing equity, Continuous Front-Month WTI Crude Oil (CL=F) representing unexpired crude oil futures, and the iShares 20+ Year Treasury Bond ETF (TLT), were sourced via Yahoo Finance from March 2024 to March 2026. CL=F was selected as it provides a continuous, liquid proxy for near-term oil exposure, solving the issue of individual unexpired contracts lacking two full years of historical data. Daily logarithmic returns were calculated to evaluate the risk profile of each asset.
    """)
    st.markdown("---")
    
    st.markdown("""
    ### Q1A: Normality of Returns
    A Shapiro-Wilk test was conducted to determine if the full 2-year sample returns follow a normal distribution. All p-values are significantly below 0.05; thus, we reject the null hypothesis. The returns are not normally distributed. 
    
    Standard normal distributions assume extreme events are exceptionally rare. In reality, these assets exhibit "fat tails" (excess kurtosis) due to real-world fundamental drivers:
    * **LMT (Excess Kurtosis: 9.66):** Highly sensitive to extreme, binary events like government contract awards or sudden geopolitical conflicts.
    * **CL=F (Excess Kurtosis: 1.88):** Vulnerable to immediate macroeconomic supply shocks, such as unexpected OPEC+ cuts.
    * **TLT (Excess Kurtosis: 0.90):** Reacts sharply to surprise central bank interest rate decisions.
    """)
    
    asset_choice = st.selectbox("Select Asset to View Interactive Distribution:", ['LMT', 'CL=F', 'TLT'])
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
    fig1.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name="Normal Distribution", line=dict(color='#ff8c00', width=3))) 
    fig1.update_layout(title=f"{asset_choice} Return Distribution vs Normal Curve", template="plotly_dark", hovermode="x unified", barmode='overlay')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    
    st.markdown("""
    ### Q1B & Q1C: Historical Risk Metrics (Year 1 vs. Full 2 Years)
    In Year 1, TLT was the least volatile, while CL=F was the most volatile. LMT exhibited the most severe tail risk (ES: -6.35%). Comparing the full two years to Year 1, volatility noticeably increased for LMT and CL=F. Furthermore, tail risk worsened significantly; LMT’s 1% VaR dropped from -3.55% to -4.82%, indicating the second year introduced more extreme negative downside events, reinforcing the presence of fat tails.
    
    ### Q1D: Parametric Risk Metrics
    Compared to the historical metrics in Q1C, the parametric method drastically underestimates tail risk. For example, LMT's Parametric ES (-3.91%) is far weaker than its Historical ES (-6.95%). Because parametric models force a symmetrical "bell curve," they ignore the heavy left tails identified in Q1A, creating a dangerously optimistic risk assessment.
    
    **Bridging the Gap: Modified VaR**
    Because standard parametric models fail to capture excess kurtosis and skewness, institutional risk managers often apply the Cornish-Fisher expansion. This technique adjusts the standard Z-score to account for non-normal skew and heavy tails. Acknowledging this adjustment highlights why relying purely on standard normal distribution assumptions is an incomplete risk management strategy.
    """)
    st.markdown("### Historical VaR Breaches (Backtesting)")
    st.markdown("A robust risk management framework requires backtesting. This chart highlights the specific trading days where the daily return breached the historical 1% Value-at-Risk threshold, visually isolating the extreme tail events.")
    
    var_threshold = np.percentile(data, 1)
    breaches = data[data < var_threshold]
    
    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Daily Returns', line=dict(color='#8b9bb4', width=1)))
    fig_var.add_hline(y=var_threshold, line_dash="dash", line_color="#ff8c00", annotation_text=f"1% VaR ({var_threshold*100:.2f}%)", annotation_position="bottom right", annotation_font_color="#ff8c00")
    fig_var.add_trace(go.Scatter(x=breaches.index, y=breaches, mode='markers', name='VaR Breach', marker=dict(color='#ff4b4b', size=8, symbol='x')))
    
    fig_var.update_layout(title=f"{asset_choice} Return Timeline & Tail Risk Breaches", template="plotly_dark", hovermode="x unified", yaxis_title="Daily Return")
    st.plotly_chart(fig_var, use_container_width=True)
    
    st.markdown("---")

    st.markdown("### Cross-Asset Volatility & Tail Risk (Box Plot)")
    st.markdown("This box plot visually confirms the excess kurtosis calculated above. The 'whiskers' represent the standard range of volatility, while the individual dots highlight the extreme tail-risk events (fat tails) that normal distributions fail to predict.")
    
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=returns_full['LMT'], name='LMT (Equity)', marker_color='#00d4ff', boxpoints='outliers'))
    fig_box.add_trace(go.Box(y=returns_full['CL=F'], name='CL=F (Commodity)', marker_color='#ffb822', boxpoints='outliers'))
    fig_box.add_trace(go.Box(y=returns_full['TLT'], name='TLT (Bonds)', marker_color='#00fa9a', boxpoints='outliers')) 
    
    fig_box.update_layout(title="Daily Return Distribution & Outliers", template="plotly_dark", yaxis_title="Daily Return")
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Raw Data: Daily Asset Returns")
    st.dataframe(returns_full.style.format("{:.4%}"), use_container_width=True, height=300)

# ------------------------------------------
# --- TAB 2: Q2 ---
# ------------------------------------------
with tab2:
    st.markdown("""
    ### Q2: Risk Profile of Portfolio of Assets
    An equal-weighted portfolio was constructed using the three assets (LMT, CL=F, TLT), allocating 33.33% to each. Using the full 2 years of daily returns, historical simulation was applied to determine the portfolio's risk metrics.
    """)
    st.markdown("---")
    
    weights = np.array([1/3, 1/3, 1/3])
    port_returns = returns_full[['LMT', 'CL=F', 'TLT']].dot(weights)
    p_var = np.percentile(port_returns, 1)
    p_es = port_returns[port_returns <= p_var].mean()
    
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Portfolio Std Dev", f"{port_returns.std()*100:.2f}%")
    pc2.metric("Portfolio 1% VaR", f"{p_var*100:.2f}%")
    pc3.metric("Portfolio 1% ES", f"{p_es*100:.2f}%", "Subadditive Property Satisfied")

    st.markdown("""
    **Summary & Comparison to Individual Assets**
    A clear diversification benefit is observed. The portfolio's overall volatility (0.87%) and tail risk (ES: -3.05%) are drastically lower than those of its riskiest components, LMT (ES: -6.95%) and CL=F (ES: -6.58%). Because these distinct asset classes (equities, commodities, bonds) are not perfectly correlated, extreme downside movements in one asset are often offset by stability or inverse movements in another, smoothing the overall return distribution and dramatically reducing catastrophic tail risk.
    """)
    st.markdown("---")
    
    st.markdown("### Cumulative Performance: Portfolio vs. Individual Assets")
    st.markdown("This chart visually demonstrates the smoothing effect of diversification. While individual assets experience severe, volatile drawdowns, the equal-weighted portfolio mitigates these extremes, resulting in a more stable trajectory.")
    
    cum_returns = (1 + returns_full[['LMT', 'CL=F', 'TLT']]).cumprod() - 1
    cum_port = (1 + port_returns).cumprod() - 1
    
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns['LMT'], name='LMT', line=dict(color='#808080', width=1, dash='dot')))
    fig_cum.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns['CL=F'], name='CL=F', line=dict(color='#ffb822', width=1, dash='dot')))
    fig_cum.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns['TLT'], name='TLT', line=dict(color='#00fa9a', width=1, dash='dot'))) 
    fig_cum.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name='Equal-Weighted Portfolio', line=dict(color='#00d4ff', width=3)))
    fig_cum.update_layout(title="Growth of $1: Diversification in Action", template="plotly_dark", hovermode="x unified", yaxis_tickformat=".1%")
    st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown("---")
    st.markdown("### Visualizing the Subadditive Property")
    
    fig_bar = go.Figure()
    assets = ['LMT', 'CL=F', 'TLT', 'Equal-Weighted Portfolio']
    var_vals = [np.percentile(returns_full['LMT'], 1)*100, np.percentile(returns_full['CL=F'], 1)*100, np.percentile(returns_full['TLT'], 1)*100, p_var*100]
    es_vals = [returns_full['LMT'][returns_full['LMT'] <= np.percentile(returns_full['LMT'], 1)].mean()*100,
               returns_full['CL=F'][returns_full['CL=F'] <= np.percentile(returns_full['CL=F'], 1)].mean()*100,
               returns_full['TLT'][returns_full['TLT'] <= np.percentile(returns_full['TLT'], 1)].mean()*100, p_es*100]
    
    fig_bar.add_trace(go.Bar(x=assets, y=var_vals, name='1% VaR', marker_color='#ffb822'))
    fig_bar.add_trace(go.Bar(x=assets, y=es_vals, name='1% ES', marker_color='#ff8c00')) 
    
    fig_bar.update_layout(title="Tail Risk Comparison: Individual Assets vs Portfolio", template="plotly_dark", yaxis_title="Risk / Return (%)", barmode='group')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### Advanced Correlation Analysis")
    
    corr_matrix = returns_full[['LMT', 'CL=F', 'TLT']].corr()
    fig_corr = go.Figure(data=go.Heatmap(
                   z=corr_matrix.values,
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   colorscale='RdBu', zmin=-1, zmax=1,
                   text=np.round(corr_matrix.values, 2),
                   texttemplate="%{text}", textfont={"size":18, "color":"white"},
                   showscale=True))
    fig_corr.update_layout(title="Asset Correlation Matrix", template="plotly_dark", width=500, height=500, yaxis_autorange='reversed')
    
    colA, colB = st.columns([1, 1])
    with colA:
        st.plotly_chart(fig_corr, use_container_width=True)
    with colB:
        st.markdown("""
        <br><br>The diversification benefits observed are mathematically driven by the low and negative correlations between the constituent assets. As shown in the matrix, TLT (Treasuries) exhibits a negative correlation with both equities and commodities. During "flight-to-safety" market shocks, Treasury bonds typically appreciate, offsetting the portfolio's net losses.
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **The Subadditive Property**
    Expected Shortfall (ES) satisfies the subadditive property, whereas Value-at-Risk (VaR) generally does not. The subadditive property dictates that the risk of a combined portfolio must be less than or equal to the sum of the standalone risks of its individual components. ES is a mathematically coherent risk measure that always satisfies this rule. Conversely, when asset returns exhibit non-normal distributions with "fat tails"—which was definitively proven in Q1—VaR can fail subadditivity. This means VaR could theoretically and incorrectly suggest that a diversified portfolio is riskier than its individual parts.
    """)

# ------------------------------------------
# --- TAB 3: Q3 ---
# ------------------------------------------
with tab3:
    st.markdown("""
    ### Q3: Futures Markets & Hedging
    The assignment requires the use of daily price data spanning two full calendar years. Working backward from the March 2026 deadline, a two-year sample from March 2024 to March 2026 was constructed. Daily U.S. Gulf Coast spot jet fuel prices were sourced from the EIA. Because government spot reporting naturally lags behind live financial markets, the continuous crude oil futures data (CL=F) was strategically truncated to March 2, 2026, perfectly aligning the two time series to prevent data-matching errors.
    
    Following the assignment guidelines, the sample was split: the first 1.5 years of data were used to estimate the optimal hedge ratio ($h^*$), and the remaining data was used to test the hedge, simulating physical fuel consumption at the exact end of the 2-year sample. The $h^*$ was calculated via an Ordinary Least Squares (OLS) regression of the daily price changes of jet fuel against crude oil futures.
    """)
    st.markdown("---")
    
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
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=model_data['Crude_Oil_Price'], y=model_data['Jet_Fuel_Price'], mode='markers', name='Daily Changes', marker=dict(color='gray', opacity=0.6)))
        fig_scatter.add_trace(go.Scatter(x=model_data['Crude_Oil_Price'], y=model.predict(X), mode='lines', name=f'OLS Line (Slope: {h_star:.4f})', line=dict(color='#ff8c00', width=3))) 
        fig_scatter.update_layout(title="OLS Regression: Estimating Optimal Hedge Ratio", template="plotly_dark", xaxis_title="Crude Oil Daily Change ($/bbl)", yaxis_title="Jet Fuel Daily Change ($/gal)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown(f"""
        **Hedge Ratio & Contract Calculation**
        The regression yielded an $h^*$ of 0.0248 ($R^2$ = 0.5887). Because the spot data is priced per gallon and the futures data is priced per barrel, this $h^*$ implicitly captures the unit variance (closely mirroring the theoretical 1:42 gallon-to-barrel ratio).
        
        To determine the number of 1,000-barrel contracts ($N$) required to hedge the 1,000,000-gallon exposure:
        
        $$N = \\frac{{1,000,000 \\times 0.0248}}{{1,000}} = 24.79$$
        
        Rounding to the nearest whole unit, the airline must take a long position of 25 crude oil contracts.
        """)

        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Optimal Hedge Ratio (h*)", f"{h_star:.4f}")
        hc2.metric("R-Squared", f"{model.rsquared:.4f}")
        hc3.metric("Contracts Needed", f"{contracts}")
        hc4.metric("Net Hedge Savings", f"${profit:,.0f}")
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df['Jet_Fuel_Price'], name="Jet Fuel Spot ($/gal)", line=dict(color='#00d4ff')))
        fig3.add_trace(go.Scatter(x=df.index, y=df['Crude_Oil_Price'], name="Crude Oil Futures ($/bbl)", yaxis="y2", line=dict(color='#ff8c00'))) 
        fig3.update_layout(title="Spot Jet Fuel vs. NYMEX Crude Futures", template="plotly_dark", hovermode="x unified", yaxis2=dict(overlaying="y", side="right"))
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("""
        **Real-World Hedge Effectiveness**
        To test the hedge, we simulated initiating the 25-contract long position at the end of the estimation period (Sept 5, 2025) and lifting it upon physical fuel consumption at the end of the dataset (March 2, 2026).
        
        **Conclusion:**
        The cross-hedge was highly effective. Rising energy prices across the test period would have severely impacted the airline's operating costs. However, the \\$234,000 profit generated by the long futures position successfully offset the rising spot market prices, reducing the effective cost of jet fuel from \\$2.74 to \\$2.50 per gallon. This validates the $R^2$ of 58.87%, proving that while CL=F is not a perfect 1:1 proxy, it serves as an excellent mitigating instrument for jet fuel price risk.
        """)
        
        st.markdown("---")
        st.markdown("**Limitations and Basis Risk**")

        df['Basis'] = df['Jet_Fuel_Price'] - (df['Crude_Oil_Price'] / 42)
        fig_basis = go.Figure()
        fig_basis.add_trace(go.Scatter(x=df.index, y=df['Basis'], mode='lines', name='Basis Spread', line=dict(color='#e0e0e0', width=2))) 
        fig_basis.update_layout(title="Basis Risk: Price Differential Between Jet Fuel and Crude Oil", template="plotly_dark", yaxis_title="Spread ($/gal)")
        st.plotly_chart(fig_basis, use_container_width=True)

        st.markdown("""
        While the cross-hedge generated substantial savings, it is subject to basis risk—the risk that the price relationship between jet fuel and crude oil fluctuates over time. Jet fuel prices are influenced by specific refining constraints and aviation demand, which do not always perfectly track unrefined crude oil. As visualized above, this basis spread is volatile. Therefore, while $h^*$ provides an optimal static ratio, a dynamic hedging strategy would be required in practice to manage ongoing basis risk.
        """)
        
        st.markdown("---")
        st.markdown("### Historical U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB")
        
        fig_eia = go.Figure()
        fig_eia.add_trace(go.Scatter(x=eia.index, y=eia['Jet_Fuel_Price'], name="Spot Price", fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.2)', line=dict(color='#00d4ff', width=2)))
        fig_eia.update_layout(title="Jet Fuel Spot Price (Dollars per Gallon)", template="plotly_dark", hovermode="x unified", yaxis_title="$/gal")
        st.plotly_chart(fig_eia, use_container_width=True)
        
        st.markdown("### Raw Data Table")
        eia_display = eia.sort_index(ascending=False).copy()
        eia_display.index = eia_display.index.strftime('%Y-%m-%d')
        st.dataframe(eia_display, use_container_width=True, height=300)

    except Exception as e:
        st.error(f"Data alignment error. Ensure 'US EIA Data.xlsx' is in the repo. Error: {e}")










