# 🌾 Agricultural Commodity Price Forecasting — Nigeria
## LSTM Neural Network Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

---

## 📌 Project Overview

This project builds an end-to-end **machine learning system** for forecasting monthly prices of six major Nigerian agricultural commodities using **LSTM (Long Short-Term Memory) neural networks**.

### Commodities Covered
| Commodity | Unit | Key Market |
|-----------|------|-----------|
| Maize | ₦/100kg | Lagos, Kano |
| Rice | ₦/50kg | Lagos, Abuja |
| Cassava | ₦/100kg | South-West |
| Tomatoes | ₦/basket | Kano, Lagos |
| Cocoa | ₦/tonne | Ondo, Cross River |
| Coffee | ₦/tonne | Mubi, Taraba |

---

## 🗂️ Project Structure

```
agri_price_forecast/
│
├── src/
│   └── pipeline.py          ← Core ML pipeline (data gen, features, LSTM, plots)
│
├── notebooks/
│   └── notebook_analysis.py ← EDA, stationarity tests, feature importance
│
├── app/
│   └── app.py               ← Streamlit interactive dashboard
│
├── data/
│   └── DATA_SOURCES.md      ← Data sources guide + API scripts
│
├── outputs/                 ← Generated charts, CSVs (auto-created)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full ML pipeline
```bash
cd src
python pipeline.py
```
Generates forecast charts + CSV summary in `outputs/`.

### 3. Run the research notebook
```bash
cd notebooks
python notebook_analysis.py
```
Generates EDA plots, correlation matrix, feature importance.

### 4. Launch the web dashboard
```bash
cd app
streamlit run app.py
```
Opens interactive dashboard at `http://localhost:8501`

---

## 🧠 LSTM Model Architecture

```
Input (lookback=12 months × N features)
    │
    ▼
LSTM(128) + BatchNorm + Dropout(0.2)
    │
    ▼
LSTM(64)  + BatchNorm + Dropout(0.2)
    │
    ▼
LSTM(32)  + BatchNorm + Dropout(0.1)
    │
    ▼
Dense(16, relu)
    │
    ▼
Dense(1)  ← Price prediction
```

**Loss:** Huber (robust to outliers)  
**Optimizer:** Adam (lr=0.001, with ReduceLROnPlateau)  
**Early stopping:** patience=15  

---

## 🔧 Feature Engineering

| Feature Group | Features |
|--------------|----------|
| Lag features | lag_1, lag_2, lag_3, lag_6, lag_12 |
| Rolling stats | rolling_mean_3/6/12, rolling_std_3/6/12 |
| Momentum | mom_1, mom_3, mom_6, mom_12 (% change) |
| Calendar | month_sin, month_cos, quarter, year |

---

## 📊 Data Sources

| Source | Commodities | Access |
|--------|-------------|--------|
| WFP VAM API | Maize, Rice, Cassava, Tomatoes | Free API |
| FAO GIEWS | All staples | Free download |
| World Bank Pink Sheet | Cocoa, Coffee | Free Excel |
| CBN | Exchange rates | Free |
| NBS Nigeria | CPI, food prices | Free PDF/Excel |

See `data/DATA_SOURCES.md` for full details and code snippets.

---

## 📈 Recommended Additional Features (for real data)

- **Rainfall (mm)** — from NASA POWER API or NIMET
- **USD/NGN exchange rate** — from CBN
- **Fuel price** — from NNPC
- **Nigeria CPI** — from NBS
- **Import volumes** — from NCS

---

## 🛠️ Replacing Synthetic Data with Real Data

1. Collect real price data from WFP/FAO (see `DATA_SOURCES.md`)
2. Load into a `pd.DataFrame` with columns: `date` (index), `price`
3. Pass directly to `CommodityForecaster.fit(your_df)`

```python
from src.pipeline import CommodityForecaster

# Load your real data
import pandas as pd
df = pd.read_csv("maize_prices_nigeria.csv", index_col="date", parse_dates=True)
df.columns = ["price"]

# Train and forecast
fc = CommodityForecaster("Maize", lookback=12)
fc.fit(df, epochs=100)
future = fc.predict_future(12)
print(future)
```

---

## ⚠️ Disclaimer
This system is for **research and educational purposes**. Price forecasts should not be used as sole basis for financial or trading decisions. Agricultural prices are affected by many unpredictable factors including weather, policy changes, and global commodity markets.

---

## 📬 Data Gaps / Next Steps

- [ ] Connect to live WFP VAM API for real data ingestion
- [ ] Add weather (rainfall) features from NASA POWER
- [ ] Add FX (USD/NGN) as exogenous feature
- [ ] Hyperparameter tuning with Optuna
- [ ] Model versioning with MLflow
- [ ] Deploy dashboard to Streamlit Cloud
