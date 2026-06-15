"""
=============================================================
  Agricultural Commodity Price Forecasting — Nigeria
  LSTM Neural Network Pipeline
  Commodities: Maize, Rice, Cassava, Tomatoes, Cocoa, Coffee
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─── Optional deep learning imports ───────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Using sklearn fallback model.")
    from sklearn.ensemble import GradientBoostingRegressor


# ═══════════════════════════════════════════════════════════
#  1. DATA GENERATION (Realistic Nigerian Market Simulation)
# ═══════════════════════════════════════════════════════════

COMMODITIES = {
    "Maize":    {"base": 45_000, "trend": 2500,  "seasonal_amp": 8000,  "noise": 3000,  "unit": "₦/100kg"},
    "Rice":     {"base": 75_000, "trend": 4000,  "seasonal_amp": 10000, "noise": 4000,  "unit": "₦/50kg"},
    "Cassava":  {"base": 18_000, "trend": 1200,  "seasonal_amp": 4000,  "noise": 2000,  "unit": "₦/100kg"},
    "Tomatoes": {"base": 12_000, "trend": 800,   "seasonal_amp": 6000,  "noise": 3500,  "unit": "₦/basket"},
    "Cocoa":    {"base": 850_000,"trend": 30000, "seasonal_amp": 60000, "noise": 20000, "unit": "₦/tonne"},
    "Coffee":   {"base": 450_000,"trend": 18000, "seasonal_amp": 40000, "noise": 15000, "unit": "₦/tonne"},
}

def generate_nigerian_price_data(commodity: str, start="2015-01", periods=None) -> pd.DataFrame:
    """
    Generate realistic monthly price data for Nigerian agricultural commodities.
    Incorporates:
      - Long-term upward trend (inflation / naira depreciation)
      - Seasonal harvest cycles
      - Rainfall shocks
      - COVID-19 shock (2020)
      - FX/fuel price shocks
    """
    # Auto-calculate periods from start date to current month if not specified
    if periods is None:
        start_date = pd.Timestamp(start)
        current_date = pd.Timestamp.now()
        # Calculate months between start and current date
        periods = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month) + 1
    
    params = COMMODITIES[commodity]
    np.random.seed(42 + list(COMMODITIES.keys()).index(commodity))

    dates = pd.date_range(start=start, periods=periods, freq="MS")
    t = np.arange(periods)

    # Base trend (exponential growth due to inflation)
    trend = params["base"] + params["trend"] * t * (1 + 0.002 * t)

    # Seasonal component — harvest months push prices down
    seasonal_phase = {"Maize": 0, "Rice": 1, "Cassava": 2,
                      "Tomatoes": 3, "Cocoa": 4, "Coffee": 5}[commodity]
    seasonal = params["seasonal_amp"] * np.sin(2 * np.pi * (t + seasonal_phase * 2) / 12)

    # COVID-19 shock: Mar–Jun 2020
    covid_shock = np.zeros(periods)
    covid_idx = [i for i, d in enumerate(dates) if d >= pd.Timestamp("2020-03-01") and d <= pd.Timestamp("2020-07-01")]
    for i in covid_idx:
        covid_shock[i] = params["base"] * 0.15 * np.random.uniform(0.8, 1.2)

    # FX devaluation shock: mid-2023
    fx_shock = np.zeros(periods)
    fx_idx = [i for i, d in enumerate(dates) if d >= pd.Timestamp("2023-06-01")]
    for i in fx_idx:
        fx_shock[i] = params["base"] * 0.20

    # Random noise
    noise = np.random.normal(0, params["noise"], periods)

    price = trend + seasonal + covid_shock + fx_shock + noise
    price = np.maximum(price, params["base"] * 0.5)  # floor

    df = pd.DataFrame({
        "date": dates,
        "price": price.round(2),
        "commodity": commodity,
        "unit": params["unit"],
    })
    df.set_index("date", inplace=True)
    return df


def load_all_commodities() -> pd.DataFrame:
    frames = [generate_nigerian_price_data(c) for c in COMMODITIES]
    return pd.concat(frames)


# ═══════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    p = df["price"]

    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = p.shift(lag)

    # Rolling statistics
    for window in [3, 6, 12]:
        df[f"rolling_mean_{window}"] = p.rolling(window).mean()
        df[f"rolling_std_{window}"]  = p.rolling(window).std()

    # Momentum
    df["mom_1"]  = p.pct_change(1)
    df["mom_3"]  = p.pct_change(3)
    df["mom_6"]  = p.pct_change(6)
    df["mom_12"] = p.pct_change(12)

    # Calendar features
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    df["year"]       = df.index.year
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    df.dropna(inplace=True)
    return df


# ═══════════════════════════════════════════════════════════
#  3. LSTM MODEL
# ═══════════════════════════════════════════════════════════

def build_lstm_model(input_shape):
    if not TF_AVAILABLE:
        return None
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(1e-3), loss="huber", metrics=["mae"])
    return model


def prepare_sequences(X: np.ndarray, y: np.ndarray, lookback: int = 12):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


class CommodityForecaster:
    def __init__(self, commodity: str, lookback: int = 12):
        self.commodity = commodity
        self.lookback  = lookback
        self.scaler_X  = MinMaxScaler()
        self.scaler_y  = MinMaxScaler()
        self.model     = None
        self.history   = None
        self.metrics   = {}

    def fit(self, df: pd.DataFrame, epochs=100, batch_size=16, verbose=0):
        df_feat = engineer_features(df)
        feature_cols = [c for c in df_feat.columns if c not in ("price", "commodity", "unit")]

        X = self.scaler_X.fit_transform(df_feat[feature_cols])
        y = self.scaler_y.fit_transform(df_feat[["price"]])

        X_seq, y_seq = prepare_sequences(X, y, self.lookback)
        split = int(len(X_seq) * 0.8)
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]

        if TF_AVAILABLE:
            self.model = build_lstm_model((self.lookback, X.shape[1]))
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
                ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6),
            ]
            self.history = self.model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=epochs, batch_size=batch_size,
                callbacks=callbacks, verbose=verbose,
            )
            y_pred_scaled = self.model.predict(X_val, verbose=0)
        else:
            # Fallback: flatten sequences for GBM
            self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05)
            self.model.fit(X_tr.reshape(len(X_tr), self.lookback * X.shape[1]), y_tr.ravel())
            y_pred_scaled = self.model.predict(X_val.reshape(len(X_val), self.lookback * X.shape[1])).reshape(-1, 1)

        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_val)

        self.metrics = {
            "MAE":  mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2":   r2_score(y_true, y_pred),
            "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        }
        self._df_feat = df_feat
        self._feature_cols = feature_cols
        self._n_features = X.shape[1]
        return self

    def predict_future(self, n_months: int = 12) -> pd.DataFrame:
        """Forecast n_months ahead using recursive prediction."""
        df_base = self._df_feat.copy()
        feature_cols = self._feature_cols
        predictions = []

        last_price = df_base["price"].iloc[-1]
        last_date  = df_base.index[-1]

        # Keep a rolling price series for re-engineering
        price_series = list(df_base["price"].values)

        for i in range(n_months):
            # Re-engineer on extended series
            tmp = pd.DataFrame({
                "price": price_series,
                "commodity": self.commodity,
                "unit": COMMODITIES[self.commodity]["unit"],
            }, index=pd.date_range(end=last_date + pd.DateOffset(months=i),
                                   periods=len(price_series), freq="MS"))
            tmp_feat = engineer_features(tmp)

            # Align to training feature columns
            for col in feature_cols:
                if col not in tmp_feat.columns:
                    tmp_feat[col] = 0
            tmp_feat = tmp_feat[feature_cols]

            if len(tmp_feat) < self.lookback:
                # Not enough rows yet — carry last price forward
                pred_price = price_series[-1]
            else:
                X_window = self.scaler_X.transform(tmp_feat.values)[-self.lookback:]
                if TF_AVAILABLE:
                    X_seq = X_window.reshape(1, self.lookback, -1)
                    pred_scaled = self.model.predict(X_seq, verbose=0)
                else:
                    X_seq = X_window.reshape(1, self.lookback * self._n_features)
                    pred_scaled = self.model.predict(X_seq).reshape(-1, 1)
                pred_price = self.scaler_y.inverse_transform(pred_scaled)[0][0]
                pred_price = max(pred_price, last_price * 0.7)

            next_date = last_date + pd.DateOffset(months=i + 1)
            predictions.append({"date": next_date, "predicted_price": pred_price})
            price_series.append(pred_price)

        return pd.DataFrame(predictions).set_index("date")


# ═══════════════════════════════════════════════════════════
#  4. VISUALISATION
# ═══════════════════════════════════════════════════════════

COLORS = {
    "Maize":    "#F5A623",
    "Rice":     "#7ED321",
    "Cassava":  "#9B59B6",
    "Tomatoes": "#E74C3C",
    "Cocoa":    "#8B4513",
    "Coffee":   "#D2691E",
}

def plot_forecast(commodity: str, df_hist: pd.DataFrame, df_future: pd.DataFrame,
                  metrics: dict, save_path: str = None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0D1117")

    for ax in axes:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#C9D1D9")
        ax.xaxis.label.set_color("#C9D1D9")
        ax.yaxis.label.set_color("#C9D1D9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    color = COLORS.get(commodity, "#58A6FF")
    ax = axes[0]

    # Historical
    ax.plot(df_hist.index, df_hist["price"], color=color, linewidth=1.8,
            label="Historical Price", alpha=0.9)
    ax.fill_between(df_hist.index, df_hist["price"], alpha=0.1, color=color)

    # Forecast
    ax.plot(df_future.index, df_future["predicted_price"], color="#58A6FF",
            linewidth=2.5, linestyle="--", label="LSTM Forecast", marker="o",
            markersize=4)

    # Confidence band (±10%)
    upper = df_future["predicted_price"] * 1.10
    lower = df_future["predicted_price"] * 0.90
    ax.fill_between(df_future.index, lower, upper, alpha=0.15, color="#58A6FF",
                    label="90% Confidence Band")

    # Divider
    ax.axvline(df_hist.index[-1], color="#FF7B72", linestyle=":", linewidth=1.5,
               label="Forecast Start")

    unit = COMMODITIES[commodity]["unit"]
    ax.set_title(f"{commodity} Price Forecast — Nigeria ({unit})",
                 color="#E6EDF3", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel(f"Price ({unit})", color="#C9D1D9")
    ax.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9",
              fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.15, color="#30363D")

    # Metrics bar
    ax2 = axes[1]
    ax2.axis("off")
    metric_text = (
        f"  MAE: ₦{metrics['MAE']:,.0f}   |   "
        f"RMSE: ₦{metrics['RMSE']:,.0f}   |   "
        f"R²: {metrics['R2']:.3f}   |   "
        f"MAPE: {metrics['MAPE']:.1f}%"
    )
    ax2.text(0.5, 0.5, metric_text, transform=ax2.transAxes,
             color="#58A6FF", fontsize=11, ha="center", va="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D",
                       edgecolor="#30363D", alpha=0.9))

    plt.tight_layout(h_pad=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_all_commodities_overview(all_data: dict, save_path: str = None):
    """Plot price index of all commodities normalised to 100."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="#C9D1D9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

    for commodity, df in all_data.items():
        base = df["price"].iloc[0]
        indexed = (df["price"] / base) * 100
        ax.plot(df.index, indexed, label=commodity, color=COLORS[commodity],
                linewidth=2, alpha=0.85)

    ax.set_title("Nigerian Agricultural Commodity Price Index (Base=100)",
                 color="#E6EDF3", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Price Index", color="#C9D1D9")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9",
              fontsize=10, ncol=3)
    ax.grid(axis="both", alpha=0.12, color="#30363D")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ═══════════════════════════════════════════════════════════
#  5. MAIN RUNNER
# ═══════════════════════════════════════════════════════════

def run_pipeline(commodities=None, forecast_months=12, output_dir="../outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    if commodities is None:
        commodities = list(COMMODITIES.keys())

    print("=" * 60)
    print("  Nigerian Agri Price Forecast — LSTM Pipeline")
    print("=" * 60)

    all_hist = {}
    all_forecasts = {}
    summary_rows = []

    for commodity in commodities:
        print(f"\n[{commodity}] Generating data & training LSTM...")
        df = generate_nigerian_price_data(commodity)
        all_hist[commodity] = df

        forecaster = CommodityForecaster(commodity, lookback=12)
        forecaster.fit(df, epochs=80, verbose=0)
        future = forecaster.predict_future(forecast_months)
        all_forecasts[commodity] = future

        m = forecaster.metrics
        print(f"  MAE={m['MAE']:,.0f}  RMSE={m['RMSE']:,.0f}  R²={m['R2']:.3f}  MAPE={m['MAPE']:.1f}%")

        # Save individual chart
        plot_forecast(commodity, df, future, m,
                      save_path=f"{output_dir}/{commodity.lower()}_forecast.png")
        print(f"  → Chart saved: {commodity.lower()}_forecast.png")

        summary_rows.append({
            "Commodity": commodity,
            "Unit": COMMODITIES[commodity]["unit"],
            "Current Price": f"₦{df['price'].iloc[-1]:,.0f}",
            "6-Month Forecast": f"₦{future['predicted_price'].iloc[5]:,.0f}",
            "12-Month Forecast": f"₦{future['predicted_price'].iloc[-1]:,.0f}",
            "MAE": f"₦{m['MAE']:,.0f}",
            "MAPE": f"{m['MAPE']:.1f}%",
            "R²": f"{m['R2']:.3f}",
        })

    # Overview chart
    plot_all_commodities_overview(all_hist,
                                  save_path=f"{output_dir}/price_index_overview.png")
    print(f"\n→ Overview chart saved: price_index_overview.png")

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{output_dir}/forecast_summary.csv", index=False)
    print(f"→ Summary CSV saved: forecast_summary.csv")

    print("\n" + "=" * 60)
    print("  FORECAST SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("\n✅  Pipeline complete.")
    return all_hist, all_forecasts, summary_df


if __name__ == "__main__":
    run_pipeline(forecast_months=12, output_dir="outputs")
