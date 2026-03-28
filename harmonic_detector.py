import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

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

def generate_sample_data(n=500, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    price = 100.0
    closes = []
    for _ in range(n):
        price *= np.exp(np.random.normal(0, 0.015))
        closes.append(round(price, 4))
    closes = pd.Series(closes)
    df = pd.DataFrame({
        "Close":  closes,
        "High":   closes * (1 + np.abs(np.random.normal(0, 0.007, n))),
        "Low":    closes * (1 - np.abs(np.random.normal(0, 0.007, n))),
        "Open":   closes.shift(1).fillna(closes.iloc[0]),
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)
    return df

def fetch_data(symbol, period, interval):
    if not YFINANCE_OK:
        return generate_sample_data()
    raw = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index.name = "Date"
    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

def find_swings(df, n):
    highs = df["High"].values
    lows  = df["Low"].values
    sh, sl = [], []
    for i in range(n, len(df) - n):
        if highs[i] > highs[i-n:i].max() and highs[i] > highs[i+1:i+n+1].max():
            sh.append(i)
        if lows[i] < lows[i-n:i].min() and lows[i] < lows[i+1:i+n+1].min():
            sl.append(i)
    def clean(lst):
        if not lst: return lst
        out = [lst[0]]
        for x in lst[1:]:
            if x - out[-1] >= 2:
                out.append(x)
        return out
    return clean(sh), clean(sl)

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
        if not (abc_min*(1-tol) <= r_abc <= abc_max*(1+tol)):
            continue
        if not (near(r_xad, xad_min, tol) or near(r_xad, xad_max, tol)):
            continue
        results.append((name, 1 - abs(r_xab - xab) / xab))
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
        seg   = all_sw[start:start+5]
        types = [s[1] for s in seg]
        if types not in [["H","L","H","L","H"],["L","H","L","H","L"]]:
            continue
        idxs   = [s[0] for s in seg]
        prices = [s[2] for s in seg]
        X_p, A_p, B_p, C_p, D_p = prices
        pattern = identify_pattern(X_p, A_p, B_p, C_p, D_p, tolerance, active)
        if not pattern:
            continue
        direction = "Bullish" if types[0] == "L" else "Bearish"
        XA = abs(A_p - X_p)
        found.append({
            "pattern": pattern, "direction": direction,
            "X_idx": idxs[0], "X_price": round(X_p, 5),
            "A_idx": idxs[1], "A_price": round(A_p, 5),
            "B_idx": idxs[2], "B_price": round(B_p, 5),
            "C_idx": idxs[3], "C_price": round(C_p, 5),
            "D_idx": idxs[4], "D_price": round(D_p, 5),
            "PRZ_min": round(D_p - XA*0.05, 5),
            "PRZ_max": round(D_p + XA*0.05, 5),
            "date_X": df.index[idxs[0]],
            "date_D": df.index[idxs[4]],
        })
    return found

def draw_crosshair(ax, x_pos, y_pos, color, label, y_min):
    ax.hlines(y=y_pos, xmin=0, xmax=x_pos,
              colors=color, linestyles="--", linewidth=0.8, alpha=0.6)
    ax.vlines(x=x_pos, ymin=y_min, ymax=y_pos,
              colors=color, linestyles="--", linewidth=0.8, alpha=0.6)
    ax.scatter(x_pos, y_pos, color=color, s=80, zorder=6,
               edgecolors="white", linewidth=0.4)
    ax.annotate(f" {label}\n {y_pos:.4f}",
                xy=(x_pos, y_pos), fontsize=7.5, color=color,
                fontweight="bold", xytext=(6, 4),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0D1117",
                          alpha=0.7, edgecolor=color))

def build_figure(df, patterns, symbol):
    n    = len(patterns)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*7, rows*4.5))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(f"Harmonic Patterns — {symbol}",
                 color="white", fontsize=13, fontweight="bold")
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
        off    = max(0, min(x_vals) - 10)
        x_end  = max(x_vals) + 10
        y_lo   = min(y_vals) * 0.975
        y_hi   = max(y_vals) * 1.025
        seg = df["Close"].iloc[off:x_end+1].reset_index(drop=True)
        ax.plot(seg.index, seg.values, color="#444", linewidth=0.9, alpha=0.5)
        ax.set_facecolor("#0D1117")
        xs = [p[0]-off for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, zorder=4)
        for label, (xi, yi) in pts.items():
            draw_crosshair(ax, xi-off, yi, color, label, y_lo)
        ax.axhspan(pat["PRZ_min"], pat["PRZ_max"], color=color, alpha=0.10)
        ax.annotate(f"  PRZ: {pat['PRZ_min']:.4f} - {pat['PRZ_max']:.4f}",
                    xy=(xs[-1]+1, (pat["PRZ_min"]+pat["PRZ_max"])/2),
                    fontsize=7, color=color)
        ax.set_title(f"{pat['pattern']}  |  {pat['direction']}  |  D: {pat['D_price']:.4f}",
                     color=color, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlim(0, x_end-off)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(colors="#888", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2A2A2A")
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    legend_patches = [mpatches.Patch(color=c, label=nm)
                      for nm, c in PATTERN_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, frameon=False, labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig
