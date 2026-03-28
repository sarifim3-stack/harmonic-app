"""
=====================================================
  داشبورد الأنماط التوافقية — Streamlit
=====================================================
تثبيت:
    pip install streamlit pandas matplotlib numpy yfinance

تشغيل:
    streamlit run dashboard.py
=====================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False


# ══════════════════════════════════════════════════
# إعداد الصفحة
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="Harmonic Pattern Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS مخصص
st.markdown("""
<style>
    .main { background-color: #0D1117; color: #FFFFFF; }
    .stButton>button {
        background: linear-gradient(90deg, #00BFFF, #0077AA);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton>button:hover { opacity: 0.85; }
    .metric-box {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 4px;
    }
    .pattern-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# نسب فيبوناتشي وألوان الأنماط
# ══════════════════════════════════════════════════
HARMONIC_RULES = {
    "Gartley":   (0.618, 0.382, 0.886, 0.786, 0.786),
    "Bat":       (0.500, 0.382, 0.886, 0.886, 0.886),
    "Butterfly": (0.786, 0.382, 0.886, 1.272, 1.618),
    "Crab":      (0.618, 0.382, 0.886, 1.618, 1.618),
}
PATTERN_COLORS = {
    "Gartley":   "#00BFFF",
    "Bat":       "#FFD700",
    "Butterfly": "#FF69B4",
    "Crab":      "#FF4500",
}


# ══════════════════════════════════════════════════
# الشريط الجانبي — إعدادات المستخدم
# ══════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("⚙️ الإعدادات")
    st.markdown("---")

    # مصدر البيانات
    st.subheader("📥 مصدر البيانات")
    data_source = st.radio(
        "اختر المصدر",
        ["Yahoo Finance (yfinance)", "بيانات تجريبية"],
        index=0
    )

    if data_source == "Yahoo Finance (yfinance)":
        st.markdown("**🔎 الرمز**")
        symbol = st.text_input("مثال: AAPL, BTC-USD, EURUSD=X", value="AAPL").upper()

        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("المدة", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        with col2:
            interval = st.selectbox("الإطار", ["1d", "1wk", "1h"], index=0)

        st.markdown("""
        <small>💡 أمثلة على الرموز:<br>
        أسهم: AAPL TSLA NVDA<br>
        فوركس: EURUSD=X GBPUSD=X<br>
        كريبتو: BTC-USD ETH-USD<br>
        ذهب: GC=F | نفط: CL=F</small>
        """, unsafe_allow_html=True)
    else:
        symbol   = "DEMO"
        period   = "6mo"
        interval = "1d"

    st.markdown("---")

    # إعدادات الكشف
    st.subheader("🔬 إعدادات الكشف")
    swing_bars = st.slider(
        "حساسية السوينج (SWING_BARS)",
        min_value=2, max_value=15, value=5,
        help="كلما قلّ العدد كلما اكتشفت أنماطاً أكثر"
    )
    tolerance = st.slider(
        "هامش تسامح فيبو (%)",
        min_value=2, max_value=15, value=6,
        help="هامش الخطأ المسموح به في نسب فيبوناتشي"
    ) / 100

    st.markdown("---")

    # اختيار الأنماط
    st.subheader("📐 الأنماط المطلوبة")
    selected_patterns = {}
    for name, color in PATTERN_COLORS.items():
        selected_patterns[name] = st.checkbox(
            name, value=True,
            help=f"البحث عن نمط {name}"
        )

    st.markdown("---")

    # زر التشغيل
    run_btn = st.button("🚀  تشغيل الكشف")


# ══════════════════════════════════════════════════
# دوال الأداة الداخلية
# ══════════════════════════════════════════════════

def generate_sample_data(n=300, seed=42):
    np.random.seed(seed)
    dates  = pd.date_range("2023-01-01", periods=n, freq="B")
    price  = 100.0
    closes = []
    for _ in range(n):
        price *= np.exp(np.random.normal(0, 0.012))
        closes.append(round(price, 4))
    closes = pd.Series(closes)
    df = pd.DataFrame({
        "Close":  closes,
        "High":   closes * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "Low":    closes * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "Open":   closes.shift(1).fillna(closes.iloc[0]),
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)
    return df


@st.cache_data(show_spinner=False)
def fetch_yfinance(sym, per, intv):
    raw = yf.download(sym, period=per, interval=intv, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index.name = "Date"
    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()


def find_swings(df, n):
    highs    = df["High"]
    lows     = df["Low"]
    roll_max = highs.rolling(window=2 * n + 1, center=True).max()
    roll_min = lows.rolling(window=2 * n + 1, center=True).min()
    sh = sorted(set(df.index[highs == roll_max].map(df.index.get_loc)))
    sl = sorted(set(df.index[lows  == roll_min].map(df.index.get_loc)))
    return sh, sl


def near(val, target, tol):
    if target == 0:
        return abs(val) < tol
    return abs(val - target) / target <= tol


def identify_pattern(X, A, B, C, D, tol, active):
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    XD = abs(D - X)
    if XA == 0 or AB == 0:
        return None
    r_xab = AB / XA
    r_abc = BC / AB
    r_xad = XD / XA
    results = []
    for name, (xab, abc_min, abc_max, xad_min, xad_max) in HARMONIC_RULES.items():
        if not active.get(name, True):
            continue
        if not near(r_xab, xab, tol):
            continue
        if not (abc_min * (1 - tol) <= r_abc <= abc_max * (1 + tol)):
            continue
        if not (near(r_xad, xad_min, tol) or near(r_xad, xad_max, tol)):
            continue
        score = 1 - abs(r_xab - xab) / xab
        results.append((name, score))
    if not results:
        return None
    return sorted(results, key=lambda x: x[1], reverse=True)[0][0]


def scan_patterns(df, swing_bars, tolerance, active):
    sh, sl = find_swings(df, swing_bars)
    found  = []
    all_sw = (
        [(i, "H", df["High"].iloc[i]) for i in sh] +
        [(i, "L", df["Low"].iloc[i])  for i in sl]
    )
    all_sw.sort(key=lambda x: x[0])
    for start in range(len(all_sw) - 4):
        seg   = all_sw[start: start + 5]
        types = [s[1] for s in seg]
        if types not in [["H","L","H","L","H"], ["L","H","L","H","L"]]:
            continue
        idxs   = [s[0] for s in seg]
        prices = [s[2] for s in seg]
        X_p, A_p, B_p, C_p, D_p = prices
        pattern = identify_pattern(X_p, A_p, B_p, C_p, D_p, tolerance, active)
        if not pattern:
            continue
        direction = "Bullish 📈" if types[0] == "L" else "Bearish 📉"
        XA = abs(A_p - X_p)
        found.append({
            "pattern":   pattern,
            "direction": direction,
            "X_idx": idxs[0], "X_price": round(X_p, 4),
            "A_idx": idxs[1], "A_price": round(A_p, 4),
            "B_idx": idxs[2], "B_price": round(B_p, 4),
            "C_idx": idxs[3], "C_price": round(C_p, 4),
            "D_idx": idxs[4], "D_price": round(D_p, 4),
            "PRZ_min": round(D_p - XA * 0.05, 4),
            "PRZ_max": round(D_p + XA * 0.05, 4),
            "date_X":  df.index[idxs[0]],
            "date_D":  df.index[idxs[4]],
        })
    return found


def draw_crosshair(ax, x_pos, y_pos, color, label, y_min):
    ax.hlines(y=y_pos, xmin=0, xmax=x_pos,
              colors=color, linestyles="--", linewidth=0.9, alpha=0.6)
    ax.vlines(x=x_pos, ymin=y_min, ymax=y_pos,
              colors=color, linestyles="--", linewidth=0.9, alpha=0.6)
    ax.scatter(x_pos, y_pos, color=color, s=90, zorder=6,
               edgecolors="white", linewidth=0.5)
    ax.annotate(
        f" {label}\n {y_pos:.2f}",
        xy=(x_pos, y_pos), fontsize=8, color=color, fontweight="bold",
        xytext=(7, 5), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0D1117",
                  alpha=0.7, edgecolor=color)
    )


def build_figure(df, patterns):
    n    = len(patterns)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 4.5))
    fig.patch.set_facecolor("#0D1117")

    axes_flat = [axes] if n == 1 else axes.flatten()

    for i, pat in enumerate(patterns):
        ax    = axes_flat[i]
        color = PATTERN_COLORS[pat["pattern"]]
        pts   = {
            "X": (pat["X_idx"], pat["X_price"]),
            "A": (pat["A_idx"], pat["A_price"]),
            "B": (pat["B_idx"], pat["B_price"]),
            "C": (pat["C_idx"], pat["C_price"]),
            "D": (pat["D_idx"], pat["D_price"]),
        }
        x_vals = [p[0] for p in pts.values()]
        y_vals = [p[1] for p in pts.values()]
        off    = max(0, min(x_vals) - 8)
        x_end  = max(x_vals) + 8
        y_lo   = min(y_vals) * 0.975
        y_hi   = max(y_vals) * 1.025

        seg = df["Close"].iloc[off: x_end + 1].reset_index(drop=True)
        ax.plot(seg.index, seg.values, color="#444444", linewidth=0.9, alpha=0.5)
        ax.set_facecolor("#0D1117")

        xs = [p[0] - off for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.85, zorder=4)

        for label, (xi, yi) in pts.items():
            draw_crosshair(ax, xi - off, yi, color, label, y_lo)

        ax.axhspan(pat["PRZ_min"], pat["PRZ_max"], color=color, alpha=0.10)
        ax.annotate(
            f"  PRZ: {pat['PRZ_min']:.2f} – {pat['PRZ_max']:.2f}",
            xy=(xs[-1] + 1, (pat["PRZ_min"] + pat["PRZ_max"]) / 2),
            fontsize=7.5, color=color
        )
        ax.set_title(
            f"{pat['pattern']}  ·  {pat['direction']}  ·  D @ {pat['D_price']:.2f}",
            color=color, fontsize=9.5, fontweight="bold", pad=5
        )
        ax.set_xlim(0, x_end - off)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2A2A")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=c, label=nm)
        for nm, c in PATTERN_COLORS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, frameon=False, labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════
# الواجهة الرئيسية
# ══════════════════════════════════════════════════

st.title("📈 Harmonic Pattern Detector")
st.caption("كشف تلقائي عن الأنماط التوافقية مع خطوط التقاطع السعرية والزمنية")
st.markdown("---")

if not run_btn:
    # شاشة الترحيب
    st.markdown("""
    <div style='text-align:center; padding: 3rem 0;'>
        <h2 style='color:#00BFFF;'>🔍 جاهز للكشف</h2>
        <p style='color:#888; font-size:1.1rem;'>
            اضبط الإعدادات في الشريط الجانبي ثم اضغط <b>تشغيل الكشف</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, (name, color) in zip([col1, col2, col3, col4], PATTERN_COLORS.items()):
        col.markdown(f"""
        <div class='metric-box'>
            <div style='color:{color}; font-size:1.3rem; font-weight:bold;'>{name}</div>
            <div style='color:#888; font-size:0.8rem; margin-top:4px;'>
                {"XAB=61.8%" if name=="Gartley" else
                 "XAB=50.0%" if name=="Bat" else
                 "XAB=78.6%" if name=="Butterfly" else
                 "XAD=161.8%"}
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── تحميل البيانات ──────────────────────────
    with st.spinner("📥 جاري تحميل البيانات..."):
        try:
            if data_source == "Yahoo Finance (yfinance)":
                if not YFINANCE_OK:
                    st.error("⚠️ yfinance غير مثبت — قم بتشغيل: pip install yfinance")
                    st.stop()
                df = fetch_yfinance(symbol, period, interval)
                if df.empty:
                    st.error(f"❌ لم يتم العثور على بيانات للرمز: {symbol}")
                    st.stop()
            else:
                df = generate_sample_data()
                symbol = "DEMO"
        except Exception as e:
            st.error(f"❌ خطأ في تحميل البيانات: {e}")
            st.stop()

    # معلومات البيانات
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏷️ الرمز",    symbol)
    c2.metric("📊 عدد الشمعات", f"{len(df):,}")
    c3.metric("📅 من",       str(df.index[0].date()))
    c4.metric("📅 إلى",      str(df.index[-1].date()))
    st.markdown("---")

    # ── تشغيل الكشف ─────────────────────────────
    with st.spinner("🔍 جاري البحث عن الأنماط التوافقية..."):
        patterns = scan_patterns(df, swing_bars, tolerance, selected_patterns)

    # ── النتائج ──────────────────────────────────
    if not patterns:
        st.warning("⚠️ لم يُكتشف أي نمط. جرّب تقليل SWING_BARS أو زيادة هامش التسامح.")
    else:
        st.success(f"✅ تم اكتشاف **{len(patterns)}** نمط توافقي!")

        # ملخص الأنماط
        counts = pd.Series([p["pattern"] for p in patterns]).value_counts()
        cols   = st.columns(len(counts))
        for col, (name, cnt) in zip(cols, counts.items()):
            color = PATTERN_COLORS.get(name, "#FFF")
            col.markdown(f"""
            <div class='metric-box'>
                <div style='color:{color}; font-size:1.4rem; font-weight:bold;'>{cnt}</div>
                <div style='color:{color}; font-size:0.9rem;'>{name}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── الشارت ───────────────────────────────
        st.subheader("📊 الشارت")
        with st.spinner("🎨 جاري رسم الأنماط..."):
            fig = build_figure(df, patterns)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")

        # ── جدول التقرير ─────────────────────────
        st.subheader("📋 تقرير مفصّل")
        report_rows = [{
            "النمط":       p["pattern"],
            "الاتجاه":     p["direction"],
            "تاريخ X":    str(p["date_X"].date()),
            "تاريخ D":    str(p["date_D"].date()),
            "سعر X":      p["X_price"],
            "سعر A":      p["A_price"],
            "سعر B":      p["B_price"],
            "سعر C":      p["C_price"],
            "سعر D":      p["D_price"],
            "PRZ Min":    p["PRZ_min"],
            "PRZ Max":    p["PRZ_max"],
        } for p in patterns]

        report_df = pd.DataFrame(report_rows)

        # تلوين صف حسب النمط
        def color_pattern(val):
            c = PATTERN_COLORS.get(val, "#FFFFFF")
            return f"color: {c}; font-weight: bold"

        styled = report_df.style.applymap(color_pattern, subset=["النمط"])
        st.dataframe(styled, use_container_width=True, height=300)

        # ── تحميل التقرير CSV ────────────────────
        csv = report_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ تحميل التقرير CSV",
            data=csv,
            file_name=f"harmonic_{symbol}.csv",
            mime="text/csv"
        )
