"""
=====================================================
  أداة الكشف التلقائي عن الأنماط التوافقية
  Harmonic Pattern Detector — pandas + matplotlib
  الأنماط: Gartley | Bat | Butterfly | Crab
=====================================================
تثبيت:
    pip install pandas matplotlib numpy yfinance

تشغيل:
    python harmonic_detector.py

البيانات:
    DATA_SOURCE = "yfinance"   ← بيانات حقيقية من Yahoo Finance (موصى به)
    DATA_SOURCE = "csv"        ← ملف CSV: Date,Open,High,Low,Close,Volume
    DATA_SOURCE = "generate"   ← بيانات تجريبية بدون إنترنت
=====================================================
"""

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


# ─────────────────────────────────────────────────
# ⚙️  إعدادات — عدّل هنا فقط
# ─────────────────────────────────────────────────
DATA_SOURCE = "yfinance"        # "yfinance" | "csv" | "generate"
CSV_PATH    = "data.csv"        # مسار ملف CSV إن وُجد

# إعدادات yfinance
YF_SYMBOL   = "AAPL"           # الرمز: AAPL | EURUSD=X | BTC-USD | TSLA ...
YF_PERIOD   = "6mo"            # المدة: 1mo | 3mo | 6mo | 1y | 2y
YF_INTERVAL = "1d"             # الإطار: 1d | 1h | 15m | 5m

SWING_BARS  = 5                 # عدد الشمعات لتحديد القمة/القاع
TOLERANCE   = 0.06              # هامش تسامح نسب فيبو (6%)
SHOW_ALL    = True              # True = كل الأنماط | False = آخر نمط فقط


# ─────────────────────────────────────────────────
# 📐 نسب فيبوناتشي لكل نمط
#    (XAB_ratio, ABC_min, ABC_max, XAD_min, XAD_max)
# ─────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────
# 1. توليد / تحميل البيانات
# ─────────────────────────────────────────────────

def generate_sample_data(n=300, seed=42):
    """توليد بيانات شمعات تجريبية"""
    np.random.seed(seed)
    dates  = pd.date_range("2023-01-01", periods=n, freq="B")
    price  = 100.0
    closes = []
    for _ in range(n):
        price *= np.exp(np.random.normal(0, 0.012))
        closes.append(round(price, 4))
    closes = pd.Series(closes)
    df = pd.DataFrame({
        "Date":   dates,
        "Close":  closes,
        "High":   closes * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "Low":    closes * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "Open":   closes.shift(1).fillna(closes.iloc[0]),
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    })
    df.set_index("Date", inplace=True)
    return df


def load_data():
    if DATA_SOURCE == "yfinance":
        if not YFINANCE_OK:
            print("⚠️  yfinance غير مثبت — قم بتشغيل: pip install yfinance")
            print("⚠️  جاري التبديل إلى البيانات التجريبية...")
            return generate_sample_data()
        print(f"🌐 جاري تحميل {YF_SYMBOL} من Yahoo Finance...")
        raw = yf.download(YF_SYMBOL, period=YF_PERIOD, interval=YF_INTERVAL, progress=False)
        # توحيد أسماء الأعمدة (yfinance أحياناً يُعيد MultiIndex)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index.name = "Date"
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        print(f"✅ تم تحميل {len(df)} شمعة لـ {YF_SYMBOL}")
        return df
    elif DATA_SOURCE == "csv":
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"], index_col="Date")
        df.columns = [c.capitalize() for c in df.columns]
    else:
        df = generate_sample_data()
    df.sort_index(inplace=True)
    return df


# ─────────────────────────────────────────────────
# 2. الكشف عن القمم والقيعان باستخدام pandas
# ─────────────────────────────────────────────────

def find_swings(df, n=SWING_BARS):
    """
    القمة: أعلى High في نافذة [i-n : i+n]
    القاع: أدنى Low  في نافذة [i-n : i+n]
    يعتمد كلياً على pandas rolling للمقارنة
    """
    highs = df["High"]
    lows  = df["Low"]

    # rolling max/min للمقارنة
    roll_max = highs.rolling(window=2 * n + 1, center=True).max()
    roll_min = lows.rolling(window=2 * n + 1, center=True).min()

    swing_highs = list(df.index[highs == roll_max].map(df.index.get_loc))
    swing_lows  = list(df.index[lows  == roll_min].map(df.index.get_loc))

    # إزالة المتكررات المتجاورة
    swing_highs = sorted(set(swing_highs))
    swing_lows  = sorted(set(swing_lows))

    return swing_highs, swing_lows


# ─────────────────────────────────────────────────
# 3. حساب نسب فيبوناتشي
# ─────────────────────────────────────────────────

def near(val, target, tol=TOLERANCE):
    if target == 0:
        return abs(val) < tol
    return abs(val - target) / target <= tol


def identify_pattern(X, A, B, C, D):
    """يُعيد اسم النمط أو None"""
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
        if not near(r_xab, xab):
            continue
        if not (abc_min * (1 - TOLERANCE) <= r_abc <= abc_max * (1 + TOLERANCE)):
            continue
        if not (near(r_xad, xad_min) or near(r_xad, xad_max)):
            continue
        score = (1 - abs(r_xab - xab) / xab)
        results.append((name, score))

    if not results:
        return None
    return sorted(results, key=lambda x: x[1], reverse=True)[0][0]


# ─────────────────────────────────────────────────
# 4. المسح الشامل عن الأنماط
# ─────────────────────────────────────────────────

def scan_patterns(df):
    swing_h, swing_l = find_swings(df)
    found = []

    all_swings = (
        [(i, "H", df["High"].iloc[i]) for i in swing_h] +
        [(i, "L", df["Low"].iloc[i])  for i in swing_l]
    )
    all_swings.sort(key=lambda x: x[0])

    for start in range(len(all_swings) - 4):
        seg   = all_swings[start: start + 5]
        types = [s[1] for s in seg]

        if types not in [["H","L","H","L","H"], ["L","H","L","H","L"]]:
            continue

        idxs   = [s[0] for s in seg]
        prices = [s[2] for s in seg]
        X_p, A_p, B_p, C_p, D_p = prices

        pattern = identify_pattern(X_p, A_p, B_p, C_p, D_p)
        if pattern is None:
            continue

        direction = "Bullish 📈" if types[0] == "L" else "Bearish 📉"
        XA        = abs(A_p - X_p)

        found.append({
            "pattern":   pattern,
            "direction": direction,
            "X_idx": idxs[0], "X_price": X_p,
            "A_idx": idxs[1], "A_price": A_p,
            "B_idx": idxs[2], "B_price": B_p,
            "C_idx": idxs[3], "C_price": C_p,
            "D_idx": idxs[4], "D_price": D_p,
            "PRZ_min": round(D_p - XA * 0.05, 4),
            "PRZ_max": round(D_p + XA * 0.05, 4),
            "date_X":  df.index[idxs[0]],
            "date_A":  df.index[idxs[1]],
            "date_B":  df.index[idxs[2]],
            "date_C":  df.index[idxs[3]],
            "date_D":  df.index[idxs[4]],
        })

    return found


# ─────────────────────────────────────────────────
# 5. رسم خطوط التقاطع (أفقي × عمودي) عند كل نقطة
# ─────────────────────────────────────────────────

def draw_crosshair(ax, x_pos, y_pos, color, label, x_bound, y_min, y_max):
    """
    خط أفقي من x=0 حتى النقطة  ← السعر
    خط عمودي من y_min حتى النقطة ← الزمن
    يلتقيان عند نقطة XABCD
    """
    # خط أفقي (سعري)
    ax.hlines(y=y_pos, xmin=0, xmax=x_pos,
              colors=color, linestyles="--", linewidth=0.9, alpha=0.65)

    # خط عمودي (زمني)
    ax.vlines(x=x_pos, ymin=y_min, ymax=y_pos,
              colors=color, linestyles="--", linewidth=0.9, alpha=0.65)

    # نقطة الالتقاء
    ax.scatter(x_pos, y_pos, color=color, s=90, zorder=6, edgecolors="white", linewidth=0.5)

    # ملصق النقطة
    ax.annotate(
        f" {label}\n {y_pos:.2f}",
        xy=(x_pos, y_pos),
        fontsize=8.5, color=color, fontweight="bold",
        xytext=(7, 5), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0D1117", alpha=0.6, edgecolor=color)
    )


# ─────────────────────────────────────────────────
# 6. رسم نمط واحد
# ─────────────────────────────────────────────────

def plot_pattern(df, pat, ax):
    color = PATTERN_COLORS.get(pat["pattern"], "#FFFFFF")

    points = {
        "X": (pat["X_idx"], pat["X_price"]),
        "A": (pat["A_idx"], pat["A_price"]),
        "B": (pat["B_idx"], pat["B_price"]),
        "C": (pat["C_idx"], pat["C_price"]),
        "D": (pat["D_idx"], pat["D_price"]),
    }

    x_vals = [p[0] for p in points.values()]
    y_vals = [p[1] for p in points.values()]

    x_min_bound = min(x_vals) - 8
    x_max_bound = max(x_vals) + 8
    y_min_bound = min(y_vals) * 0.975
    y_max_bound = max(y_vals) * 1.025

    # شارت Close في الخلفية
    seg = df["Close"].iloc[max(0, x_min_bound): x_max_bound + 1].reset_index(drop=True)
    offset = max(0, x_min_bound)
    ax.plot(seg.index + offset - offset, seg.values,
            color="#555555", linewidth=0.9, alpha=0.5)
    ax.set_facecolor("#0D1117")

    # رسم خطوط الوصل X→A→B→C→D
    xs_rel = [p[0] - offset for p in points.values()]
    ys     = [p[1] for p in points.values()]
    ax.plot(xs_rel, ys, color=color, linewidth=2, alpha=0.85, zorder=4)

    # رسم التقاطعات
    for label, (xi, yi) in points.items():
        draw_crosshair(
            ax,
            x_pos   = xi - offset,
            y_pos   = yi,
            color   = color,
            label   = label,
            x_bound = xi - offset,
            y_min   = y_min_bound,
            y_max   = yi
        )

    # منطقة PRZ
    ax.axhspan(pat["PRZ_min"], pat["PRZ_max"],
               color=color, alpha=0.10, zorder=1)
    ax.annotate(
        f"  PRZ: {pat['PRZ_min']:.2f} – {pat['PRZ_max']:.2f}",
        xy=(xs_rel[-1] + 1, (pat["PRZ_min"] + pat["PRZ_max"]) / 2),
        fontsize=7.5, color=color, alpha=0.9
    )

    # عنوان
    ax.set_title(
        f"{pat['pattern']}  ·  {pat['direction']}  ·  D @ {pat['D_price']:.2f}",
        color=color, fontsize=9.5, fontweight="bold", pad=5
    )
    ax.set_xlim(0, x_max_bound - offset)
    ax.set_ylim(y_min_bound, y_max_bound)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2A2A")


# ─────────────────────────────────────────────────
# 7. رسم كل الأنماط في شبكة
# ─────────────────────────────────────────────────

def plot_all(df, patterns):
    if not patterns:
        print("⚠️  لم يُكتشف أي نمط. جرّب تقليل SWING_BARS أو زيادة TOLERANCE.")
        return

    n    = len(patterns)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 4.5))
    fig.patch.set_facecolor("#0D1117")
    symbol_label = YF_SYMBOL if DATA_SOURCE == "yfinance" else "Chart"
    fig.suptitle(
        f"🔍 الأنماط التوافقية — {symbol_label} — Harmonic Pattern Detector",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )

    axes_flat = [axes] if n == 1 else axes.flatten()

    for i, pat in enumerate(patterns):
        plot_pattern(df, pat, axes_flat[i])

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # legend مشتركة
    legend_patches = [
        mpatches.Patch(color=c, label=name)
        for name, c in PATTERN_COLORS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, frameon=False, labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig("harmonic_patterns.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("✅ تم حفظ الشارت: harmonic_patterns.png")
    plt.show()


# ─────────────────────────────────────────────────
# 8. تقرير pandas
# ─────────────────────────────────────────────────

def print_report(patterns):
    if not patterns:
        return
    rows = [{
        "النمط":      p["pattern"],
        "الاتجاه":    p["direction"],
        "تاريخ X":   str(p["date_X"].date()),
        "تاريخ D":   str(p["date_D"].date()),
        "سعر X":     p["X_price"],
        "سعر D":     p["D_price"],
        "PRZ Min":   p["PRZ_min"],
        "PRZ Max":   p["PRZ_max"],
    } for p in patterns]

    report = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("         📊 تقرير الأنماط التوافقية — Harmonic Report")
    print("=" * 72)
    print(report.to_string(index=False))
    print("=" * 72 + "\n")


# ─────────────────────────────────────────────────
# 9. نقطة الدخول الرئيسية
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("📥 جاري تحميل البيانات...")
    df = load_data()
    print(f"✅ {len(df)} شمعة — من {df.index[0].date()} إلى {df.index[-1].date()}")

    print("🔍 جاري البحث عن الأنماط التوافقية...")
    patterns = scan_patterns(df)
    print(f"✅ تم اكتشاف {len(patterns)} نمط")

    print_report(patterns)

    target = patterns if SHOW_ALL else ([patterns[-1]] if patterns else [])
    plot_all(df, target)
