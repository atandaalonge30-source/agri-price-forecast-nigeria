"""
═══════════════════════════════════════════════════════════════
  RESEARCH NOTEBOOK
  Agricultural Commodity Price Forecasting — Nigeria
  Exploratory Data Analysis + Model Evaluation
═══════════════════════════════════════════════════════════════
Run as:  python notebook_analysis.py
Or copy cells into Jupyter Notebook.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from pipeline import (
    generate_nigerian_price_data, engineer_features,
    load_all_commodities, COMMODITIES, COLORS
)

os.makedirs("outputs", exist_ok=True)

# ═══════════════════════════════════════════════
# § 1. LOAD & INSPECT DATA
# ═══════════════════════════════════════════════
print("§1  Loading data for all commodities...")
all_data = {c: generate_nigerian_price_data(c) for c in COMMODITIES}

for c, df in all_data.items():
    print(f"\n  [{c}]  shape={df.shape}  "
          f"min=₦{df['price'].min():,.0f}  "
          f"max=₦{df['price'].max():,.0f}  "
          f"mean=₦{df['price'].mean():,.0f}")

# ═══════════════════════════════════════════════
# § 2. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════
print("\n§2  Descriptive Statistics")
stats_rows = []
for c, df in all_data.items():
    p = df["price"]
    stats_rows.append({
        "Commodity": c,
        "Unit": COMMODITIES[c]["unit"],
        "Mean (₦)": f"{p.mean():,.0f}",
        "Median (₦)": f"{p.median():,.0f}",
        "Std Dev": f"{p.std():,.0f}",
        "CV (%)": f"{(p.std()/p.mean()*100):.1f}",
        "Min (₦)": f"{p.min():,.0f}",
        "Max (₦)": f"{p.max():,.0f}",
        "Skewness": f"{p.skew():.2f}",
    })
stats_df = pd.DataFrame(stats_rows)
print(stats_df.to_string(index=False))

# ═══════════════════════════════════════════════
# § 3. EDA PLOTS
# ═══════════════════════════════════════════════
print("\n§3  Generating EDA plots...")

# 3a. Price distribution per commodity
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Price Distribution — All Commodities (Nigeria)",
             color="#E6EDF3", fontsize=14, fontweight="bold", y=1.01)

for ax, (commodity, df) in zip(axes.flat, all_data.items()):
    ax.set_facecolor("#161B22")
    color = COLORS[commodity]
    p = df["price"]
    ax.hist(p, bins=20, color=color, alpha=0.7, edgecolor="#0D1117")
    ax.axvline(p.mean(),   color="#FFFFFF", linewidth=1.5, linestyle="--", label=f"Mean ₦{p.mean():,.0f}")
    ax.axvline(p.median(), color="#58A6FF", linewidth=1.5, linestyle=":",  label=f"Median ₦{p.median():,.0f}")
    ax.set_title(commodity, color="#E6EDF3", fontsize=11)
    ax.tick_params(colors="#C9D1D9")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
    ax.legend(fontsize=7, facecolor="#21262D", labelcolor="#C9D1D9", edgecolor="#30363D")

plt.tight_layout()
plt.savefig("outputs/eda_distributions.png", dpi=130, bbox_inches="tight",
            facecolor="#0D1117")
plt.close()
print("  → outputs/eda_distributions.png")

# 3b. Seasonality heatmap
print("  Generating seasonality heatmaps...")
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.patch.set_facecolor("#0D1117")
fig.suptitle("Monthly Seasonality Heatmap (Average Price by Month × Year)",
             color="#E6EDF3", fontsize=13, fontweight="bold")

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

for ax, (commodity, df) in zip(axes.flat, all_data.items()):
    ax.set_facecolor("#161B22")
    pivot = df["price"].copy()
    pivot_df = pd.DataFrame({"price": pivot, "month": pivot.index.month, "year": pivot.index.year})
    heatmap_data = pivot_df.pivot_table(index="year", columns="month", values="price", aggfunc="mean")
    # Normalize per row (year) for better visual
    heatmap_norm = heatmap_data.div(heatmap_data.mean(axis=1), axis=0)

    im = ax.imshow(heatmap_norm.values, aspect="auto",
                   cmap="RdYlGn_r", vmin=0.85, vmax=1.15)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, color="#C9D1D9", fontsize=7)
    ax.set_yticks(range(len(heatmap_norm.index)))
    ax.set_yticklabels(heatmap_norm.index.tolist(), color="#C9D1D9", fontsize=7)
    ax.set_title(commodity, color="#E6EDF3", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.04)

plt.tight_layout()
plt.savefig("outputs/eda_seasonality.png", dpi=130, bbox_inches="tight",
            facecolor="#0D1117")
plt.close()
print("  → outputs/eda_seasonality.png")

# 3c. Correlation matrix
print("  Generating correlation matrix...")
price_df = pd.DataFrame({c: all_data[c]["price"] for c in COMMODITIES})
corr = price_df.corr()

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#161B22")

im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="#C9D1D9")
ax.set_yticklabels(corr.index, color="#C9D1D9")

for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.iloc[i,j]) > 0.5 else "#666", fontsize=9)

plt.colorbar(im, ax=ax)
ax.set_title("Price Correlation Matrix — Nigerian Commodities",
             color="#E6EDF3", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/eda_correlation.png", dpi=130, bbox_inches="tight",
            facecolor="#0D1117")
plt.close()
print("  → outputs/eda_correlation.png")

# ═══════════════════════════════════════════════
# § 4. STATIONARITY TEST (ADF)
# ═══════════════════════════════════════════════
print("\n§4  Augmented Dickey–Fuller Stationarity Tests")
try:
    from statsmodels.tsa.stattools import adfuller
    for c, df in all_data.items():
        result = adfuller(df["price"].dropna())
        stat   = result[0]
        pval   = result[1]
        status = "Stationary ✅" if pval < 0.05 else "Non-Stationary ⚠️"
        print(f"  {c:12s}  ADF={stat:7.3f}  p={pval:.4f}  → {status}")
except ImportError:
    print("  statsmodels not installed — skipping ADF test.")

# ═══════════════════════════════════════════════
# § 5. FEATURE IMPORTANCE (MAIZE EXAMPLE)
# ═══════════════════════════════════════════════
print("\n§5  Feature Importance — Maize (Random Forest proxy)")
try:
    from sklearn.ensemble import RandomForestRegressor

    df_maize = engineer_features(all_data["Maize"])
    feature_cols = [c for c in df_maize.columns if c not in ("price","commodity","unit")]
    X = df_maize[feature_cols].values
    y = df_maize["price"].values

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")
    colors_bar = ["#F5A623" if i < 5 else "#30363D" for i in range(len(importances[:15]))]
    importances[:15].plot(kind="barh", ax=ax, color=colors_bar)
    ax.set_title("Top 15 Feature Importances — Maize Price Prediction",
                 color="#E6EDF3", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#C9D1D9")
    ax.set_xlabel("Importance", color="#C9D1D9")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=130, bbox_inches="tight",
                facecolor="#0D1117")
    plt.close()
    print("  → outputs/feature_importance.png")
    print(f"  Top features: {', '.join(importances.index[:5].tolist())}")
except ImportError:
    print("  sklearn not installed — skipping feature importance.")

# ═══════════════════════════════════════════════
# § 6. MODEL PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════
print("\n§6  Running quick model evaluation (GBM baseline)...")
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit

    eval_rows = []
    tscv = TimeSeriesSplit(n_splits=5)

    for commodity, df in all_data.items():
        df_feat = engineer_features(df)
        feature_cols = [c for c in df_feat.columns if c not in ("price","commodity","unit")]
        X = df_feat[feature_cols].values
        y = df_feat["price"].values

        maes, rmses, r2s = [], [], []
        for train_idx, test_idx in tscv.split(X):
            model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                              max_depth=4, random_state=42)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            maes.append(mean_absolute_error(y[test_idx], pred))
            rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
            r2s.append(r2_score(y[test_idx], pred))

        eval_rows.append({
            "Commodity": commodity,
            "CV MAE (₦)": f"{np.mean(maes):,.0f} ± {np.std(maes):,.0f}",
            "CV RMSE (₦)": f"{np.mean(rmses):,.0f}",
            "CV R²": f"{np.mean(r2s):.3f}",
        })

    eval_df = pd.DataFrame(eval_rows)
    print("\n  ╔══ Cross-Validated Model Performance (5-Fold TimeSeriesCV) ══╗")
    print(eval_df.to_string(index=False))
    eval_df.to_csv("outputs/model_evaluation.csv", index=False)
    print("\n  → outputs/model_evaluation.csv")
except ImportError:
    print("  sklearn not available.")

print("\n" + "═"*55)
print("  ✅  Notebook analysis complete. Check outputs/ folder.")
print("═"*55)
