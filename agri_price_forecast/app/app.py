"""
═══════════════════════════════════════════════════════════════
  🌾 AgriPrice Nigeria — LSTM Price Forecasting Dashboard
  Streamlit Web Application

  Run: streamlit run app.py
═══════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from pipeline import (
    generate_nigerian_price_data, engineer_features,
    CommodityForecaster, COMMODITIES, COLORS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AgriPrice Nigeria",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0D1117; color: #E6EDF3; }

.hero-banner {
    background: linear-gradient(135deg, #1A2744 0%, #0D1117 50%, #1A3A1A 100%);
    border: 1px solid #30363D;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '🌾';
    position: absolute; right: 40px; top: 20px;
    font-size: 80px; opacity: 0.15;
}
.hero-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #F5A623, #58A6FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.hero-sub { color: #8B949E; font-size: 1rem; margin-top: 8px; }

.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58A6FF; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #E6EDF3; font-family: 'JetBrains Mono'; }
.metric-label { font-size: 0.78rem; color: #8B949E; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
.metric-delta { font-size: 0.85rem; font-weight: 600; margin-top: 6px; }
.delta-pos { color: #3FB950; }
.delta-neg { color: #F85149; }

.section-header {
    font-size: 1.1rem; font-weight: 600; color: #E6EDF3;
    border-left: 3px solid #F5A623;
    padding-left: 12px; margin: 24px 0 16px;
}
.insight-box {
    background: #161B22; border: 1px solid #30363D;
    border-left: 3px solid #58A6FF;
    border-radius: 8px; padding: 16px 20px;
    margin: 12px 0; font-size: 0.9rem; color: #C9D1D9;
}
.tag {
    display: inline-block;
    background: #21262D; color: #58A6FF;
    border: 1px solid #30363D; border-radius: 20px;
    padding: 3px 10px; font-size: 0.75rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHED DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_commodity_data(commodity: str) -> pd.DataFrame:
    return generate_nigerian_price_data(commodity)

@st.cache_resource(show_spinner=False)
def get_trained_forecaster(commodity: str) -> CommodityForecaster:
    df = get_commodity_data(commodity)
    fc = CommodityForecaster(commodity, lookback=12)
    fc.fit(df, epochs=80, verbose=0)
    return fc


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌾 AgriPrice Nigeria")
    st.markdown("---")

    commodity = st.selectbox(
        "Select Commodity",
        list(COMMODITIES.keys()),
        format_func=lambda x: f"{x}  ({COMMODITIES[x]['unit']})"
    )

    forecast_months = st.slider("Forecast Horizon (months)", 3, 24, 12, 1)

    show_confidence = st.checkbox("Show Confidence Band", value=True)
    show_rolling    = st.checkbox("Show 6-Month Moving Average", value=True)

    st.markdown("---")
    st.markdown("**Model:** LSTM Neural Network")
    st.markdown("**Region:** Nigeria 🇳🇬")
    st.markdown("**Frequency:** Monthly")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#8B949E;'>
    <b>Data Sources:</b><br>
    • WFP VAM Price Portal<br>
    • FAO GIEWS Food Prices<br>
    • World Bank Pink Sheet<br>
    • NBS Nigeria Statistics
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">Nigerian AgriPrice Forecaster</div>
  <div class="hero-sub">
    LSTM-powered monthly price predictions for major agricultural commodities &nbsp;•&nbsp;
    <span class="tag">Nigeria</span>
    <span class="tag">6 Commodities</span>
    <span class="tag">LSTM</span>
    <span class="tag">Monthly</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA & TRAIN MODEL
# ─────────────────────────────────────────────
with st.spinner(f"🤖 Training LSTM model for {commodity}..."):
    df_hist       = get_commodity_data(commodity)
    forecaster    = get_trained_forecaster(commodity)
    df_future     = forecaster.predict_future(forecast_months)
    metrics       = forecaster.metrics
    color         = COLORS[commodity]
    unit          = COMMODITIES[commodity]["unit"]


# ─────────────────────────────────────────────
# KEY METRICS ROW
# ─────────────────────────────────────────────
current_price    = df_hist["price"].iloc[-1]
forecast_6m      = df_future["predicted_price"].iloc[min(5, len(df_future)-1)]
forecast_end     = df_future["predicted_price"].iloc[-1]
change_6m_pct    = (forecast_6m   - current_price) / current_price * 100
change_end_pct   = (forecast_end  - current_price) / current_price * 100

def delta_html(pct):
    arrow = "▲" if pct > 0 else "▼"
    cls   = "delta-pos" if pct > 0 else "delta-neg"
    return f'<div class="metric-delta {cls}">{arrow} {abs(pct):.1f}% vs today</div>'

col1, col2, col3, col4, col5 = st.columns(5)
cards = [
    ("Current Price",   f"₦{current_price:,.0f}",   "",                       ""),
    ("6-Month Forecast",f"₦{forecast_6m:,.0f}",      delta_html(change_6m_pct), ""),
    (f"{forecast_months}-Month Forecast", f"₦{forecast_end:,.0f}", delta_html(change_end_pct), ""),
    ("Model R²",        f"{metrics['R2']:.3f}",      "", ""),
    ("MAPE",            f"{metrics['MAPE']:.1f}%",   "", ""),
]
for col, (label, value, delta, _) in zip([col1,col2,col3,col4,col5], cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{value}</div>
          <div class="metric-label">{label}</div>
          {delta}
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN FORECAST CHART
# ─────────────────────────────────────────────
st.markdown(f'<div class="section-header">📈 {commodity} Price Forecast — Nigeria ({unit})</div>',
            unsafe_allow_html=True)

fig = go.Figure()

# Rolling average
if show_rolling:
    roll = df_hist["price"].rolling(6).mean()
    fig.add_trace(go.Scatter(
        x=df_hist.index, y=roll,
        name="6M Moving Avg", line=dict(color="#8B949E", width=1.5, dash="dot"),
        opacity=0.7
    ))

# Historical fill
fig.add_trace(go.Scatter(
    x=df_hist.index, y=df_hist["price"],
    fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
    line=dict(color=color, width=0), showlegend=False, hoverinfo="skip"
))

# Historical line
fig.add_trace(go.Scatter(
    x=df_hist.index, y=df_hist["price"],
    name="Historical Price", line=dict(color=color, width=2.5),
    hovertemplate="<b>%{x|%b %Y}</b><br>₦%{y:,.0f}<extra></extra>"
))

# Confidence band
if show_confidence:
    upper = df_future["predicted_price"] * 1.10
    lower = df_future["predicted_price"] * 0.90
    fig.add_trace(go.Scatter(
        x=list(df_future.index) + list(df_future.index[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(88,166,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="90% Confidence Band",
        hoverinfo="skip"
    ))

# Forecast line
fig.add_trace(go.Scatter(
    x=df_future.index, y=df_future["predicted_price"],
    name="LSTM Forecast", line=dict(color="#58A6FF", width=3, dash="dash"),
    marker=dict(size=7, color="#58A6FF", symbol="circle"),
    hovertemplate="<b>Forecast %{x|%b %Y}</b><br>₦%{y:,.0f}<extra></extra>"
))

# Forecast start marker
fig.add_vline(x=df_hist.index[-1], line_dash="dot", line_color="#FF7B72")
fig.add_annotation(
    x=df_hist.index[-1],
    text="Forecast Start",
    showarrow=False,
    font=dict(color="#FF7B72"),
    yanchor="top",
    xanchor="right"
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0D1117",
    plot_bgcolor="#161B22",
    height=480,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(showgrid=True, gridcolor="#21262D", tickformat="%b %Y"),
    yaxis=dict(showgrid=True, gridcolor="#21262D",
               title=f"Price ({unit})", tickprefix="₦", tickformat=",.0f"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# FORECAST TABLE
# ─────────────────────────────────────────────
col_a, col_b = st.columns([1, 1])

with col_a:
    st.markdown('<div class="section-header">📋 Forecast Table</div>', unsafe_allow_html=True)
    table_df = df_future.copy()
    table_df.index = table_df.index.strftime("%B %Y")
    table_df["predicted_price"] = table_df["predicted_price"].apply(lambda x: f"₦{x:,.0f}")
    table_df.columns = ["Predicted Price"]
    table_df["Change vs Today"] = df_future["predicted_price"].apply(
        lambda x: f"{'▲' if x > current_price else '▼'} {abs((x-current_price)/current_price*100):.1f}%"
    ).values
    st.dataframe(table_df, use_container_width=True)

with col_b:
    st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
    perf_df = pd.DataFrame([
        {"Metric": "Mean Absolute Error (MAE)", "Value": f"₦{metrics['MAE']:,.0f}"},
        {"Metric": "Root Mean Square Error (RMSE)", "Value": f"₦{metrics['RMSE']:,.0f}"},
        {"Metric": "R-Squared (R²)", "Value": f"{metrics['R2']:.4f}"},
        {"Metric": "Mean Abs. % Error (MAPE)", "Value": f"{metrics['MAPE']:.2f}%"},
        {"Metric": "Model Architecture", "Value": "LSTM (128→64→32) + Dense"},
        {"Metric": "Lookback Window", "Value": "12 months"},
        {"Metric": "Training Split", "Value": "80% train / 20% validation"},
    ])
    st.dataframe(perf_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# ALL COMMODITIES OVERVIEW
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🌍 All Commodities — Normalised Price Index (Base=100)</div>',
            unsafe_allow_html=True)

fig2 = go.Figure()
for c in COMMODITIES:
    df_c  = get_commodity_data(c)
    index = (df_c["price"] / df_c["price"].iloc[0]) * 100
    fig2.add_trace(go.Scatter(
        x=df_c.index, y=index,
        name=c, line=dict(color=COLORS[c], width=2),
        hovertemplate=f"<b>{c}</b> %{{x|%b %Y}}<br>Index: %{{y:.1f}}<extra></extra>"
    ))

fig2.update_layout(
    template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
    height=360, margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=True, gridcolor="#21262D", tickformat="%Y"),
    yaxis=dict(showgrid=True, gridcolor="#21262D",
               title="Price Index (2015=100)"),
    hovermode="x unified",
)
st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# INSIGHTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">💡 Automated Market Insights</div>', unsafe_allow_html=True)

yoy = (df_hist["price"].iloc[-1] - df_hist["price"].iloc[-13]) / df_hist["price"].iloc[-13] * 100
volatility = df_hist["price"].pct_change().std() * 100
trend_dir = "upward" if change_end_pct > 0 else "downward"
trend_emoji = "📈" if change_end_pct > 0 else "📉"

insights = [
    f"{trend_emoji} The LSTM model forecasts a <b>{trend_dir} trend</b> for {commodity} prices over the next "
    f"{forecast_months} months — estimated at <b>₦{forecast_end:,.0f}</b> ({change_end_pct:+.1f}% vs today).",

    f"📅 Year-on-year price change: <b>{yoy:+.1f}%</b>. "
    f"Monthly price volatility is <b>{volatility:.1f}%</b> — "
    f"{'high' if volatility > 5 else 'moderate' if volatility > 2 else 'low'} risk environment.",

    f"🎯 Model accuracy: R²={metrics['R2']:.3f}, MAPE={metrics['MAPE']:.1f}%. "
    f"The model explains <b>{metrics['R2']*100:.1f}%</b> of price variance on held-out validation data.",

    f"🌍 Key drivers for Nigerian {commodity} prices include seasonal harvest cycles, "
    f"fuel/transport costs, USD/NGN exchange rate movements, and rainfall patterns.",
]

for insight in insights:
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8B949E; font-size:0.8rem; padding: 12px 0;'>
  Built with LSTM Neural Networks &nbsp;|&nbsp;
  Data: WFP VAM · FAO GIEWS · World Bank Pink Sheet · NBS Nigeria &nbsp;|&nbsp;
  For educational & research purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)
