# Data Format Guide

This document describes supported data formats and how to prepare your historical data for optimization.

## Quick Start

**Default location:** `d:\tick_data\` (configurable in `config.yaml`)

**Expected format:** `{SYMBOL}_{YEAR}_M1_formatted.csv`

**Example:** `EURUSD_2024_M1_formatted.csv`

---

## Supported Formats

### 1. CSV Format (Recommended)

**File naming:** `SYMBOL_YEAR_M1_formatted.csv`

**Required columns:**
- `timestamp` (or `time`, `datetime`, `date`) - ISO 8601 format
- `open` (or `o`, `Open`) - Opening price
- `high` (or `h`, `High`) - High price  
- `low` (or `l`, `Low`) - Low price
- `close` (or `c`, `Close`) - Closing price

**Optional columns:**
- `volume` - Trade volume
- `spread` - Bid/ask spread in pips

**Example CSV:**
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.10450,1.10465,1.10440,1.10458,1250
2024-01-01 00:01:00,1.10458,1.10472,1.10455,1.10468,980
2024-01-01 00:02:00,1.10468,1.10485,1.10465,1.10478,1120
```

**Column names are case-insensitive** - `timestamp`, `Timestamp`, `TIMESTAMP` all work.

---

## Timeframe Support

### Current: M1 (1-minute bars)

The optimizer currently processes M1 (1-minute) data.

### Planned (v3.1): Multi-Timeframe

Future versions will support:
- **Tick data** → Auto-aggregation to M1/M5/M15/H1/H4/D1
- **Higher timeframes** → Direct M5, M15, M30, H1, H4, D1 optimization

---

## Data Sources

### Recommended: Dukascopy

**Why:** Highest quality tick data, free historical data

**Download:** https://www.dukascopy.com/trading-tools/widgets/quotes/historical_data_feed

**Format:** CSV export with M1 aggregation

**Steps:**
1. Select symbol (e.g., EURUSD)
2. Select date range (e.g., 2024-01-01 to 2024-12-31)
3. Select timeframe: M1
4. Download CSV
5. Rename to: `EURUSD_2024_M1_formatted.csv`
6. Place in `d:\tick_data\`

### Alternative Sources

**MetaTrader 5:** Export M1 history to CSV
- Open Data Window → Right-click → Export to CSV
- Format columns as shown above

**TrueFX:** Tick data (requires aggregation)

**FXCM:** Historical M1 data via API

---

## Custom Data Paths

### Using config.yaml

Create `config.yaml` from `config.yaml.example`:

```yaml
data:
  data_dir: "C:/your/custom/path"  # Your data folder
  file_pattern: "{symbol}_{year}_M1_formatted.csv"
```

### Multi-Symbol Example

```
C:\trading_data\
├── EURUSD_2024_M1_formatted.csv
├── GBPUSD_2024_M1_formatted.csv
├── USDJPY_2024_M1_formatted.csv
└── XAUUSD_2024_M1_formatted.csv
```

In optimizer, specify symbol:
```bash
python blessing_optimizer_main.py
# Symbol: GBPUSD  ← Will load GBPUSD_2024_M1_formatted.csv
```

---

## Data Validation

### Before optimization, check:

1. **File exists** in data directory
2. **Correct naming** (symbol, year match)
3. **Required columns** present (timestamp, OHLC)
4. **No missing data** (gaps in timestamps)
5. **Correct date range** (covers optimization period)

### Run validation:

```bash
python check_environment.py
# [6/7] Data directory... [OK] 4 CSV files
```

---

## Common Issues

### "File not found: EURUSD_2024_M1_formatted.csv"

**Solution:** Check filename matches exactly:
- Symbol in UPPERCASE: `EURUSD` not `eurusd`
- Year: `2024` not `24`
- Extension: `.csv` not `.CSV` or `.txt`

### "KeyError: 'timestamp'"

**Solution:** Check CSV has timestamp column (any case):
```csv
timestamp,open,high,low,close  ← OK
Timestamp,Open,High,Low,Close  ← OK
time,o,h,l,c                   ← OK
date,price_open,price_high...  ← NEEDS RENAMING
```

### "No data for date range"

**Solution:** Check CSV contains data for requested dates:
```python
import pandas as pd
df = pd.read_csv('EURUSD_2024_M1_formatted.csv')
print(df['timestamp'].min())  # 2024-01-01 00:00:00
print(df['timestamp'].max())  # 2024-12-31 23:59:00
```

---

## Performance Tips

### 1. Use smaller date ranges for testing

**Instead of:** 2024-01-01 to 2024-12-31 (full year)
**Try first:** 2024-01-01 to 2024-01-31 (one month)

### 2. Parquet format (future)

Parquet is 10x faster to load than CSV:
```python
# Convert CSV to Parquet (one time)
df = pd.read_csv('EURUSD_2024_M1_formatted.csv')
df.to_parquet('EURUSD_2024_M1.parquet')
```

**Note:** Not yet supported in v3.0, planned for v3.1

---

## Example: Preparing Dukascopy Data

### Step-by-step:

1. Download from Dukascopy (gets file like `EURUSD_Ticks_01.01.2024-31.12.2024.csv`)

2. Open in Excel/Python, check columns:
```
Time (UTC),Bid,Ask,Bid Volume,Ask Volume
01.01.2024 00:00:00,1.10450,1.10455,1.5,1.2
```

3. Convert to required format:
```python
import pandas as pd

df = pd.read_csv('EURUSD_Ticks_01.01.2024-31.12.2024.csv')

# Rename columns
df = df.rename(columns={
    'Time (UTC)': 'timestamp',
    'Bid': 'close',  # Use bid as close
})

# Parse datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S')

# Resample to M1 (OHLC)
df = df.set_index('timestamp').resample('1min').agg({
    'close': ['first', 'max', 'min', 'last']
}).dropna()

df.columns = ['open', 'high', 'low', 'close']
df = df.reset_index()

# Save
df.to_csv('EURUSD_2024_M1_formatted.csv', index=False)
```

4. Verify:
```python
df = pd.read_csv('EURUSD_2024_M1_formatted.csv')
print(df.head())
#            timestamp    open    high     low   close
# 0 2024-01-01 00:00:00 1.10450 1.10465 1.10440 1.10458
```

5. Move to data folder: `d:\tick_data\EURUSD_2024_M1_formatted.csv`

6. Run optimizer!

---

## Questions?

Check [TROUBLESHOOTING](README.md#troubleshooting) in README.md

Or open an issue on GitHub: [Link to issues]

---

**Last updated:** 2026-01-03
**Version:** 3.0
