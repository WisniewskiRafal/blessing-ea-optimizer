# BLESSING OPTIMIZER - MULTI-SYMBOL GUIDE

**Jak optymalizować różne pary walutowe równolegle**

---

## 🎯 QUICK START - Jedna para (EURUSD)

### Podstawowe uruchomienie:
```bash
python run_optimizer.py
```

**Domyślne ustawienia:**
- Symbol: EURUSD
- Data: 2024-01-01 do 2024-12-31
- Checkpoint: co 10 backtestów
- Results: `data/results/continuous/`

---

## 📊 MULTI-PARA OPTIMIZATION

### Przykład: EURUSD, GBPUSD, USDJPY

**1. EURUSD (terminal #1):**
```bash
python run_optimizer.py ^
  --symbol EURUSD ^
  --start-date 2024-01-01 ^
  --end-date 2024-12-31 ^
  --state-file data/state/eurusd_state.pkl ^
  --results-dir data/results/eurusd ^
  --max-hours 8
```

**2. GBPUSD (terminal #2):**
```bash
python run_optimizer.py ^
  --symbol GBPUSD ^
  --start-date 2024-01-01 ^
  --end-date 2024-12-31 ^
  --state-file data/state/gbpusd_state.pkl ^
  --results-dir data/results/gbpusd ^
  --max-hours 8
```

**3. USDJPY (terminal #3):**
```bash
python run_optimizer.py ^
  --symbol USDJPY ^
  --start-date 2024-01-01 ^
  --end-date 2024-12-31 ^
  --state-file data/state/usdjpy_state.pkl ^
  --results-dir data/results/usdjpy ^
  --max-hours 8
```

**WAŻNE:** Każda para ma swój **osobny state file** i **results dir**!

---

## 📁 STRUKTURA WYNIKÓW

Po uruchomieniu dla 3 par:

```
data/
├── state/
│   ├── eurusd_state.pkl      ← Stan optymalizacji EURUSD
│   ├── gbpusd_state.pkl      ← Stan optymalizacji GBPUSD
│   └── usdjpy_state.pkl      ← Stan optymalizacji USDJPY
│
└── results/
    ├── eurusd/
    │   ├── phase_0_top_100.csv        ← Top 100 wyników (aktualizowane co 10 BT)
    │   └── phase_0_top_100_FINAL.csv  ← Finalne top 100
    │
    ├── gbpusd/
    │   ├── phase_0_top_100.csv
    │   └── phase_0_top_100_FINAL.csv
    │
    └── usdjpy/
        ├── phase_0_top_100.csv
        └── phase_0_top_100_FINAL.csv
```

---

## 📈 FORMAT WYNIKÓW CSV

**Przykład: `phase_0_top_100.csv`**

```csv
rank,score,net_profit,win_rate,total_trades,profit_factor,max_drawdown_pct,sharpe_ratio,final_balance,ma_entry,cci_entry,bollinger_entry,stoch_entry,macd_entry,b3_traditional,force_market_cond,use_any_entry
1,65992.99,65992.99,0.68,342,2.45,18.3,1.87,66992.99,1,2,0,1,2,False,3,True
2,45231.12,45231.12,0.62,298,1.92,22.1,1.34,46231.12,2,1,1,0,2,True,0,False
...
```

**Kolumny:**
- `rank`: Pozycja w rankingu (1 = najlepsza)
- `score` / `net_profit`: Zysk netto ($)
- `win_rate`: Wskaźnik wygranych (0-1)
- `total_trades`: Liczba transakcji
- `profit_factor`: Gross profit / Gross loss
- `max_drawdown_pct`: Maksymalny drawdown (%)
- `sharpe_ratio`: Sharpe ratio
- `final_balance`: Końcowy balance ($)
- `ma_entry` ... `macd_entry`: Ustawienia wskaźników (0=off, 1=normal, 2=reverse)
- `b3_traditional`: True=STOP/LIMIT, False=instant
- `force_market_cond`: 0=uptrend, 1=downtrend, 2=range, 3=off
- `use_any_entry`: True=ANY, False=ALL

---

## 🔄 WZNOWIENIE PO PRZERWANIU

Każda para zachowuje swój stan:

```bash
# Dzień 1 - EURUSD (8h)
python run_optimizer.py --symbol EURUSD --max-hours 8

# Dzień 2 - EURUSD (kontynuacja)
python run_optimizer.py --symbol EURUSD --max-hours 8
# ↑ Automatycznie wznawia od miejsca przerwania!
```

---

## ⚙️ DODATKOWE OPCJE

### GPU/CPU:
```bash
# Z GPU (domyślnie)
python run_optimizer.py --symbol EURUSD --gpu

# Bez GPU (tylko CPU)
python run_optimizer.py --symbol EURUSD --no-gpu
```

### Checkpoint frequency:
```bash
# Zapisuj częściej (co 5 BT)
python run_optimizer.py --symbol EURUSD --checkpoint-interval 5

# Zapisuj rzadziej (co 50 BT, szybciej)
python run_optimizer.py --symbol EURUSD --checkpoint-interval 50
```

### Limitowanie czasu:
```bash
# Max 2 godziny
python run_optimizer.py --symbol EURUSD --max-hours 2

# Max 100 backtestów
python run_optimizer.py --symbol EURUSD --max-backtests 100

# Kombinacja (co pierwsze)
python run_optimizer.py --symbol EURUSD --max-hours 4 --max-backtests 500
```

---

## 📊 MONITORING NA BIEŻĄCO

### Konsola:
System pokazuje progress co 10 backtestów:

```
[PROGRESS] Phase: entry_combinations
  Iteration: 130/3888 (3.3%)
  Best score: 65992.99  ← Najlepsza konfiguracja DO TEJ PORY
  Total backtests: 130
  Runtime: 0.2h
```

### CSV update:
Plik `phase_0_top_100.csv` jest aktualizowany **co 10 backtestów**!

Możesz otwierać go w Excel/Calc w trakcie optymalizacji.

### Stan w Python:
```python
import pickle

# Wczytaj aktualny stan
with open('data/state/eurusd_state.pkl', 'rb') as f:
    state = pickle.load(f)

print(f"Progress: {state.current_iteration}/{state.total_iterations}")
print(f"Best score: ${state.current_best_score:.2f}")
print(f"Best config: {state.current_best_config}")
```

---

## 🎯 WORKFLOW: 7 PAR × 3,888 KOMBINACJI

**Setup parallel (Windows - 7 terminali):**

```bash
# Terminal 1: EURUSD
python run_optimizer.py --symbol EURUSD --state-file data/state/eurusd.pkl --results-dir data/results/eurusd

# Terminal 2: GBPUSD
python run_optimizer.py --symbol GBPUSD --state-file data/state/gbpusd.pkl --results-dir data/results/gbpusd

# Terminal 3: USDJPY
python run_optimizer.py --symbol USDJPY --state-file data/state/usdjpy.pkl --results-dir data/results/usdjpy

# Terminal 4: AUDUSD
python run_optimizer.py --symbol AUDUSD --state-file data/state/audusd.pkl --results-dir data/results/audusd

# Terminal 5: USDCAD
python run_optimizer.py --symbol USDCAD --state-file data/state/usdcad.pkl --results-dir data/results/usdcad

# Terminal 6: NZDUSD
python run_optimizer.py --symbol NZDUSD --state-file data/state/nzdusd.pkl --results-dir data/results/nzdusd

# Terminal 7: USDCHF
python run_optimizer.py --symbol USDCHF --state-file data/state/usdchf.pkl --results-dir data/results/usdchf
```

**Każda para:** 3,888 kombinacji entry

**Total:** 7 × 3,888 = **27,216 backtestów**

**Czas (GPU):** ~35 sekund per para = **~4 minuty total** (jeśli równolegle!)

---

## 🚀 FINALNE WYNIKI

Po zakończeniu wszystkich par:

```
data/results/
├── eurusd/phase_0_top_100_FINAL.csv   ← Top 100 dla EURUSD
├── gbpusd/phase_0_top_100_FINAL.csv   ← Top 100 dla GBPUSD
├── usdjpy/phase_0_top_100_FINAL.csv   ← Top 100 dla USDJPY
...
```

**Możesz:**
1. Otworzyć każdy CSV w Excel
2. Wybrać #1 konfigurację dla każdej pary
3. Zastosować w live trading na MT5/MT4

---

## 💡 TIPS & TRICKS

### 1. Background execution (Windows):
```bash
start /B python run_optimizer.py --symbol EURUSD --max-hours 24 > eurusd.log 2>&1
```

### 2. Quick test (100 BT):
```bash
python run_optimizer.py --symbol EURUSD --max-backtests 100 --no-gpu
```

### 3. Check progress without stopping:
```python
# check_progress.py
import pickle
import sys

symbol = sys.argv[1] if len(sys.argv) > 1 else 'eurusd'
state_file = f'data/state/{symbol}_state.pkl'

with open(state_file, 'rb') as f:
    state = pickle.load(f)

pct = (state.current_iteration / state.total_iterations * 100)
print(f"{symbol.upper()}: {state.current_iteration}/{state.total_iterations} ({pct:.1f}%)")
print(f"Best: ${state.current_best_score:.2f}")
```

Uruchom: `python check_progress.py eurusd`

---

**Powodzenia w optymalizacji!** 🎯
