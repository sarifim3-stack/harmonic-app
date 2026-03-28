import streamlit as st
import pandas as pd
from harmonic_detector import (
    fetch_data, generate_sample_data,
    scan_patterns, build_figure, PATTERN_COLORS
)

st.set_page_config(
    page_title="Harmonic Pattern Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0D1117; color: #FFFFFF; }
[data-testid="stSidebar"] { background-color: #161B22; }
.stButton>button {
    background: linear-gradient(90deg, #1a1a2e, #16213e);
    color: #00BFFF;
    border: 1px solid #00BFFF;
    border-radius: 6px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    font-weight: bold;
    width: 100%;
    letter-spacing: 1px;
}
.stButton>button:hover { background: #00BFFF; color: #000; }
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
h1, h2, h3 { color: #FFFFFF; }
label, .stRadio label { color: #AAAAAA !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.markdown("## Harmonic Detector")
    st.markdown("---")

    st.markdown("#### Data Source")
    data_source = st.radio("", ["Yahoo Finance", "Sample Data"], index=0)

    if data_source == "Yahoo Finance":
        symbol   = st.text_input("Symbol", value="EURUSD=X").upper()
        period   = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
        interval = st.selectbox("Interval", ["1d","1wk","1h","15m"], index=0)
        st.caption("Examples: AAPL  TSLA  BTC-USD  EURUSD=X  GC=F  CL=F")
    else:
        symbol = "SAMPLE"
        period = interval = None

    st.markdown("---")
    st.markdown("#### Detection Settings")
    swing_bars = st.slider("Swing Sensitivity", 2, 10, 3)
    tolerance  = st.slider("Fibonacci Tolerance (%)", 2, 20, 12) / 100

    st.markdown("---")
    st.markdown("#### Patterns")
    active = {}
    for name, color in PATTERN_COLORS.items():
        active[name] = st.checkbox(name, value=True)

    st.markdown("---")
    run = st.button("Run Detection")

# ── Main ─────────────────────────────────────────
st.markdown("# Harmonic Pattern Detector")
st.caption("Automatic detection of harmonic patterns with price and time crosshairs")
st.markdown("---")

if not run:
    st.markdown("""
    <div style='text-align:center; padding:3rem 0;'>
        <h2 style='color:#00BFFF;'>Ready</h2>
        <p style='color:#555; font-size:1rem;'>
            Configure settings in the sidebar and click Run Detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (name, color) in zip(cols, PATTERN_COLORS.items()):
        col.markdown(f"""
        <div class='metric-card'>
            <div style='color:{color}; font-size:1.1rem; font-weight:bold;'>{name}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    with st.spinner("Loading data..."):
        try:
            if data_source == "Yahoo Finance":
                df = fetch_data(symbol, period, interval)
                if df.empty:
                    st.error(f"No data found for symbol: {symbol}")
                    st.stop()
            else:
                df = generate_sample_data()
                symbol = "SAMPLE"
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

    min_required = swing_bars * 10
    if len(df) < min_required:
        st.warning(f"Only {len(df)} candles — try a longer period or wider interval.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol",  symbol)
    c2.metric("Candles", f"{len(df):,}")
    c3.metric("From",    str(df.index[0].date()))
    c4.metric("To",      str(df.index[-1].date()))
    st.markdown("---")

    with st.spinner("Scanning for harmonic patterns..."):
        patterns = scan_patterns(df, swing_bars, tolerance, active)

    if not patterns:
        st.warning("No patterns detected. Try reducing Swing Sensitivity or increasing Tolerance.")
    else:
        st.success(f"{len(patterns)} pattern(s) detected")

        counts = pd.Series([p["pattern"] for p in patterns]).value_counts()
        cols   = st.columns(len(counts))
        for col, (name, cnt) in zip(cols, counts.items()):
            color = PATTERN_COLORS.get(name, "#FFF")
            col.markdown(f"""
            <div class='metric-card'>
                <div style='color:{color}; font-size:1.6rem; font-weight:bold;'>{cnt}</div>
                <div style='color:{color}; font-size:0.85rem;'>{name}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Chart")
        with st.spinner("Rendering chart..."):
            fig = build_figure(df, patterns, symbol)
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Report")
        rows = [{
            "Pattern":   p["pattern"],
            "Direction": p["direction"],
            "Date X":    str(p["date_X"].date()),
            "Date D":    str(p["date_D"].date()),
            "X":         p["X_price"],
            "A":         p["A_price"],
            "B":         p["B_price"],
            "C":         p["C_price"],
            "D":         p["D_price"],
            "PRZ Min":   p["PRZ_min"],
            "PRZ Max":   p["PRZ_max"],
        } for p in patterns]

        report = pd.DataFrame(rows)
        st.dataframe(report, use_container_width=True, height=280)

        csv = report.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download Report CSV", csv,
                           f"harmonic_{symbol}.csv", "text/csv")
