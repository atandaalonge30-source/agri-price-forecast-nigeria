# 📦 Nigerian Agricultural Commodity Price — Data Sources Guide

## 🔴 FREE & OFFICIAL DATA SOURCES

### 1. FAO GIEWS (Recommended — Free)
- **URL**: https://www.fao.org/giews/food-prices/
- **Coverage**: Nigeria market prices (Lagos, Kano, Ibadan, Port Harcourt)
- **Commodities**: Maize, Rice, Sorghum, Millet, Yam, Cassava
- **Frequency**: Weekly / Monthly
- **Format**: CSV download
- **How to access**: Tools → FPMA Tool → Select Nigeria → Export CSV

### 2. World Food Programme (VAM)
- **URL**: https://dataviz.vam.wfp.org/
- **Coverage**: Nigeria state-level market prices
- **Commodities**: All major staples
- **API**: https://api.vam.wfp.org/
```python
import requests
# WFP API Example
url = "https://api.vam.wfp.org/markets/price-weekly?iso3=NGA&commodity=maize"
response = requests.get(url)
df = pd.DataFrame(response.json()['data'])
```

### 3. World Bank Pink Sheet
- **URL**: https://www.worldbank.org/en/research/commodity-markets
- **Coverage**: Global commodity prices (USD) — Cocoa, Coffee, Rice
- **Frequency**: Monthly
- **Format**: Excel download (free)

### 4. FAOSTAT
- **URL**: https://www.fao.org/faostat/
- **Coverage**: Nigeria producer prices
- **How**: Data → Prices → Producer Prices → Select Nigeria

### 5. CBN (Central Bank of Nigeria)
- **URL**: https://www.cbn.gov.ng/rates/
- **Coverage**: Food price indices, CPI data
- **Format**: PDF / Excel

### 6. NBS (National Bureau of Statistics)
- **URL**: https://nigerianstat.gov.ng/
- **Coverage**: Nigeria Consumer Price Index, Food basket prices
- **Key Report**: "Selected Food Prices Watch" (Monthly)

---

## 💰 PAID / PREMIUM SOURCES

| Source | Coverage | Price |
|--------|----------|-------|
| Refinitiv Eikon | Global commodities + FX | Subscription |
| Bloomberg Terminal | Full commodity data | ~$24,000/yr |
| AgFlow | Agricultural trade flows | Contact |
| Mintec | Food ingredient prices | Contact |

---

## 🐍 AUTOMATED DATA COLLECTION SCRIPTS

### Script A: WFP API Collector
```python
import requests, pandas as pd

def fetch_wfp_nigeria(commodity: str, start_year: int = 2015) -> pd.DataFrame:
    """Fetch commodity prices from WFP VAM for Nigeria."""
    # commodity options: 'maize', 'rice', 'cassava', 'tomatoes'
    base_url = "https://api.vam.wfp.org/markets/price-weekly"
    params = {
        "iso3": "NGA",
        "commodity": commodity,
        "start_date": f"{start_year}-01-01"
    }
    resp = requests.get(base_url, params=params)
    data = resp.json()
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').resample('MS').mean()  # Resample to monthly
    return df

# Usage:
# maize_df = fetch_wfp_nigeria('maize', 2015)
```

### Script B: FAO GIEWS Scraper
```python
import pandas as pd

def load_fao_giews_csv(filepath: str) -> pd.DataFrame:
    """Load and clean FAO GIEWS price CSV export."""
    df = pd.read_csv(filepath, skiprows=3)
    df.columns = ['date', 'market', 'commodity', 'currency', 'unit', 'price', 'source']
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.dropna(subset=['price'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df.sort_values('date').set_index('date')
```

### Script C: World Bank Pink Sheet
```python
import pandas as pd

def fetch_worldbank_commodities() -> pd.DataFrame:
    """Download World Bank commodity price data (Cocoa, Coffee, Rice)."""
    url = (
        "https://thedocs.worldbank.org/en/doc/"
        "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/"
        "CMO-Historical-Data-Monthly.xlsx"
    )
    df = pd.read_excel(url, sheet_name="Monthly Prices", header=4, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df[['Cocoa', 'Coffee, Arabica', 'Rice, Thai 5%']]
```

---

## 📋 RECOMMENDED DATA COLLECTION WORKFLOW

```
1. WFP VAM API   ──→  Maize, Rice, Cassava, Tomatoes (Nigeria, NGN)
2. World Bank    ──→  Cocoa, Coffee (USD → convert to NGN via CBN FX rates)
3. CBN FX rates  ──→  USD/NGN exchange rate for conversion
4. NBS CPI       ──→  Inflation adjustment feature
5. NIMET         ──→  Rainfall data as weather feature
```

---

## 🌦️ ADDITIONAL FEATURES TO COLLECT

| Feature | Source | Why Important |
|---------|--------|---------------|
| Rainfall (mm) | NIMET / NASA POWER | Harvest yield prediction |
| USD/NGN FX | CBN | Import price pass-through |
| Fuel price | NNPC | Transportation cost |
| Nigeria CPI | NBS | Inflation adjustment |
| Import volumes | NCS | Supply-side pressure |
| Planting calendar | FMARD | Lead-lag seasonal features |

**NASA POWER API (Free rainfall data):**
```python
import requests

def fetch_rainfall_nigeria(lat=9.082, lon=8.675, start="2015", end="2024"):
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON"
    }
    resp = requests.get(url, params=params)
    data = resp.json()['properties']['parameter']['PRECTOTCORR']
    df = pd.DataFrame(list(data.items()), columns=['yearmonth', 'rainfall_mm'])
    df['date'] = pd.to_datetime(df['yearmonth'], format='%Y%m')
    return df.set_index('date')[['rainfall_mm']]
```
