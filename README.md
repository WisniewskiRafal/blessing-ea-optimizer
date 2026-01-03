# BLESSING EA OPTIMIZER v3.0

**Complete optimization system for Blessing EA with 5 strategies and 134 parameters**

**Author:** Rafał Wiśniewski | Data & AI Solutions
**Version:** 3.0 (Complete System - 2026-01-03)

---

## 🚀 QUICK START

### 1. Install required libraries:
```bash
pip install scikit-optimize deap
```

### 2. Run optimizer:
```bash
python blessing_optimizer_main.py
```

### 3. Choose strategy:
```
[A] Sequential - Phase by phase (300k backtests, 2-3 weeks)
[B] Bayesian - Intelligent sampling (300k backtests, 2-3 weeks)
[C] Genetic - Evolutionary optimization of all parameters (5k backtests, 1-2 weeks) ⭐ RECOMMENDED
[D] Hybrid - Genetic + Bayesian refinement (50k backtests, 3-4 weeks)
[E] Refine - Improve previous results (5k backtests, 1 week)
```

### 4. Results:
- **CSV:** `data/results/opt_YYYYMMDD_HHMMSS/`
- **.SET files:** `data/set_files/opt_YYYYMMDD_HHMMSS/` (TOP 10)

---

## 📊 WHAT DOES IT OPTIMIZE?

### All 134 parameters of Blessing EA:

#### **PHASE 1: Entry Logic** (8 parameters) ✅ COMPLETED
- 5 indicators: MA, CCI, Bollinger, Stochastic, MACD
- Each: 0=OFF, 1=BUY, 2=SELL
- B3Traditional: True/False
- ForceMarketCond: 0-3 (any/ranging/quiet/trending)
- UseAnyEntry: True/False

**Results:** 3,888 combinations → Best: **$57,092 profit, 80% win rate**

#### **PHASE 2: Indicator Timeframes** (5 parameters)
- MA, CCI, Bollinger, Stochastic, MACD timeframes
- Each: M1, M5, M15, M30, H1, H4, D1

**Combinations:** 7^5 = 16,807

#### **PHASE 3: Indicator Parameters** (15 parameters)
- MA: period, distance
- CCI: period
- Bollinger: period, distance, deviation
- Stochastic: zone, K period, D period, slowing
- MACD: fast, slow, signal, price type
- SmartGrid: RSI period

**Combinations:** ~4.5 million (sampling/Bayesian)

#### **PHASE 4: Grid Settings** (10 parameters)
- Lot multiplier, LAF, GAF
- Grid arrays, TP arrays, Set count
- AutoCal, SmartGrid
- Entry delay

**Combinations:** 186,624

#### **PHASE 5: Risk Management** (12 parameters)
- Max trades, Break even trade
- Max drawdown, Max spread
- Close oldest settings
- Stop loss, Trailing stop

**Combinations:** ~746,496 (sampling)

---

## 🎯 OPTIMIZATION STRATEGIES

### **Option A: SEQUENTIAL**
- **Description:** Optimizes phase by phase
- **Backtests:** ~300,000
- **Time:** 2-3 weeks
- **Advantage:** Full control, visible progress
- **For:** Beginners, need for control

### **Option B: BAYESIAN**
- **Description:** Sequential + intelligent sampling (Gaussian Process)
- **Backtests:** ~300,000
- **Time:** 2-3 weeks
- **Advantage:** Finds optima faster
- **For:** Advanced users, efficiency

### **Option C: GENETIC ALGORITHM** ⭐ **RECOMMENDED**
- **Description:** Evolutionary optimization of ALL 64 parameters simultaneously
- **Backtests:** ~5,000-10,000
- **Time:** 1-2 weeks
- **Advantage:** Considers parameter interactions, fast
- **Multi-objective:** Optimizes profit + drawdown (Pareto Front)
- **For:** First optimization

### **Option D: HYBRID**
- **Description:** Genetic (5k) → TOP 5 → Bayesian refinement of each
- **Backtests:** ~50,000
- **Time:** 3-4 weeks
- **Advantage:** Best balance quality/time
- **For:** Best result

### **Option E: REFINE** ⭐ **AFTER FIRST OPTIMIZATION**
- **Description:** Use TOP 10 from previous optimization as seed population
- **Backtests:** ~5,000
- **Time:** 1 week
- **Advantage:** Improve found configurations
- **For:** Refinement after option C

---

## 📁 PROJECT STRUCTURE

```
Blessing Optymalizer/
├── blessing_optimizer_main.py      # Main launcher (RUN THIS!)
│
├── optimization/
│   ├── sequential_optimizer.py    # Option A
│   ├── bayesian_optimizer.py      # Option B
│   └── genetic_optimizer.py       # Options C, D, E
│
├── core/
│   ├── blessing_backtest_engine.py  # Backtest engine
│   └── data_loader.py               # Data loading
│
├── strategies/
│   ├── blessing_entry_generator.py  # Entry combinations generator
│   ├── blessing_grid_system.py      # Grid trading
│   └── blessing_indicators.py       # 5 indicators
│
├── utils/
│   └── set_file_generator.py       # .set file generator for MT4/MT5
│
├── data/
│   ├── results/                     # CSV results
│   │   ├── continuous/              # PHASE 1 (3,888 combinations)
│   │   └── opt_YYYYMMDD_HHMMSS/     # New optimizations
│   └── set_files/                   # .set files (TOP 10)
│
├── docs/
│   ├── PELNA_OPTYMALIZACJA_PLAN.md  # Plan for all 6 phases
│   └── MULTI_SYMBOL_GUIDE.md        # Multi-symbol optimization
│
├── QUICK_START.md                   # Quick start (3 steps)
├── INSTRUKCJA_URUCHOMIENIA.md       # Full instructions
└── README.md                        # This file
```

---

## 🔧 REQUIREMENTS

### Python 3.11+
```bash
pip install -r requirements_full.txt
```

### Main libraries:
- **pandas, numpy** - Data processing
- **torch** - GPU acceleration (optional, 15.94x speedup)
- **scikit-optimize** - Bayesian Optimization (Option B)
- **deap** - Genetic Algorithms (Options C, D, E)

### Hardware:
- **CPU:** Any (multi-core better)
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** RTX 5060 Ti or better (optional, accelerates ~15x)

---

## 📈 PHASE 1 RESULTS (3,888 combinations)

### TOP 5 configurations EURUSD 2024:

| Rank | Profit (USD) | Win Rate | Trades | Profit Factor | Max DD | Sharpe |
|------|--------------|----------|--------|---------------|--------|--------|
| #1   | 57,092       | 80%      | 55     | 2.24          | 18.3%  | 2.69   |
| #2   | 42,531       | 80%      | 35     | 2.66          | 16.4%  | 5.17   |
| #3   | 38,208       | 78%      | 32     | 2.90          | 17.1%  | 7.18   |
| #4   | 37,258       | 83%      | 42     | 2.26          | 28.2%  | 4.40   |
| #5   | 35,812       | 80%      | 30     | 3.08          | 19.9%  | 7.67   |

**Detailed analysis:** `data/results/ANALIZA_WYNIKOW.md`

---

## 🎮 HOW TO USE RESULTS ON MT4/MT5?

### 1. Find .set files:
```
d:\Blessing Optymalizer\data\set_files\opt_YYYYMMDD_HHMMSS\
```

### 2. Copy best file:
```
blessing_rank01_score57092_wr80.set → MT4/MQL4/Presets/
                                   or MT5/MQL5/Presets/
```

### 3. In MT4/MT5 platform:
1. Drag **Blessing EA** onto EURUSD chart
2. Click **"Load"**
3. Select **blessing_rank01_score57092_wr80.set**
4. Check parameters (already set!)
5. Click **OK**

### 4. DONE!
EA will start trading with optimal settings.

---

## ⚙️ ADVANCED USAGE

### Multi-Symbol Optimization:
```bash
# Terminal 1: EURUSD
python blessing_optimizer_main.py
# Choose C, symbol: EURUSD

# Terminal 2: GBPUSD
python blessing_optimizer_main.py
# Choose C, symbol: GBPUSD

# Terminal 3: USDJPY
python blessing_optimizer_main.py
# Choose C, symbol: USDJPY
```

See: `docs/MULTI_SYMBOL_GUIDE.md`

### Custom Parameters:
```bash
python blessing_optimizer_main.py
# Choose strategy
# Enter custom symbol, dates, TOP N
```

### Resume Previous Optimization:
```bash
python blessing_optimizer_main.py
# Choose E (Refine)
# System will load TOP 10 from previous results
# Genetic Algorithm will improve them further
```

---

## 📊 STRATEGY COMPARISON

| Criterion | Sequential (A) | Bayesian (B) | Genetic (C) | Hybrid (D) | Refine (E) |
|-----------|----------------|--------------|-------------|------------|------------|
| **Backtests** | 300k | 300k | 5k | 50k | 5k |
| **Time** | 2-3 weeks | 2-3 weeks | 1-2 weeks | 3-4 weeks | 1 week |
| **Parameters simultaneously** | 5-15 | 5-15 | **64** | **64** | **64** |
| **Interactions** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Pareto Front** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Seed population** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Complexity** | Medium | High | High | Very high | Medium |
| **Best for** | Control | Efficiency | Start | Best result | Refinement |

---

## ⚠️ IMPORTANT NOTES

### 1. Overfitting
The more parameters you optimize, the greater the risk of overfitting.

**Solution:**
- Walk-forward analysis
- Out-of-sample testing (2025 data)
- Test on demo 1-3 months

### 2. Backtests ≠ Live Trading
Backtest results DO NOT guarantee profits in live trading.

**Solution:**
- Always test on demo before live
- Start with small lot (0.01)
- Monitor drawdown

### 3. Spread & Slippage
In live trading, profits will be lower vs backtest.

**Solution:**
- Include spread in backtests
- Add margin of safety (50% of expected profit)

### 4. Market Regime Changes
Optimal parameters for 2024 may not work in 2025.

**Solution:**
- Periodic reoptimization (every 3-6 months)
- Portfolio of different configurations
- Adaptive parameters

---

## 🔬 TECHNOLOGY

### GPU Acceleration (RTX 5060 Ti):
- **PyTorch 2.9.1+cu128:** 15.94x speedup
- **Mixed precision (FP16):** Additional 4.4x
- **Hybrid (GPU × CPU parallel):** Up to 191x theoretically

### Optimization Algorithms:
- **Grid Search:** Full exploration of small spaces
- **Bayesian (GP):** Intelligent sampling of large spaces
- **Genetic (NSGA-II):** Multi-objective evolution

### Data Processing:
- **Dukascopy M1 data:** Highest quality tick data
- **Chronological split:** Time-based, not random
- **Walk-forward:** Expanding window validation

---

## 📖 DOCUMENTATION

### Basic:
- **[QUICK_START.md](QUICK_START.md)** - 3 steps to run
- **[INSTRUKCJA_URUCHOMIENIA.md](INSTRUKCJA_URUCHOMIENIA.md)** - Full instructions
- **[CLAUDE.md](CLAUDE.md)** - Project rules, GPU setup

### Advanced:
- **[docs/PELNA_OPTYMALIZACJA_PLAN.md](docs/PELNA_OPTYMALIZACJA_PLAN.md)** - Plan for all 6 phases
- **[docs/MULTI_SYMBOL_GUIDE.md](docs/MULTI_SYMBOL_GUIDE.md)** - Multi-symbol optimization
- **[data/results/ANALIZA_WYNIKOW.md](data/results/ANALIZA_WYNIKOW.md)** - Analysis of TOP 100 EURUSD 2024

---

## 🆘 TROUBLESHOOTING

### Problem: "scikit-optimize not installed"
```bash
pip install scikit-optimize
```

### Problem: "DEAP not installed"
```bash
pip install deap
```

### Problem: "CUDA not available"
System will automatically use CPU. You can force:
```
Use GPU? (y/n): n
```

### Problem: "File not found: EURUSD_2024_M1_formatted.csv"
Make sure you have data in:
```
d:\tick_data\EURUSD_2024_M1_formatted.csv
```

---

## 🚀 ROADMAP

### v3.1 (planned):
- [ ] Multi-pair portfolio optimization
- [ ] Walk-forward analysis automation
- [ ] Live trading integration (MT4/MT5 bridge)
- [ ] Web dashboard with results
- [ ] Auto-reoptimization scheduler

### v3.2 (future):
- [ ] Reinforcement Learning agent
- [ ] Ensemble methods (voting)
- [ ] Market regime detection
- [ ] Adaptive parameter adjustment

---

## 📧 CONTACT

**Author:** Rafał Wiśniewski
**Email:** [Your email]
**GitHub:** [Link to repo]

---

## 📄 LICENSE

Private project - for personal use only.

**WARNING:** Do not commit to public repositories:
- Proprietary strategies (strategies/)
- Real backtest results (if revealing edge)
- API keys, credentials
- Profitable configurations

---

## 🙏 ACKNOWLEDGMENTS

Blessing EA - Original by J Talon LLC/FiFtHeLeMe Nt
Dedicated to Mike McKeough (RIP)

---

**Last update:** 2026-01-03
**Version:** 3.0
**Status:** ✅ Production Ready

---

**Good luck in trading! 🚀📈**
