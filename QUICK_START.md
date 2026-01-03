# ⚡ QUICK START - Blessing EA Optimizer

## 🚀 QUICK START (3 STEPS)

### Step 1: Install required libraries

```bash
cd "d:\Blessing Optymalizer"
pip install scikit-optimize deap
```

### Step 2: Run optimizer

```bash
python blessing_optimizer_main.py
```

### Step 3: Choose strategy

```
Choose option (A/B/C/D/E/X): C [Enter]

Symbol (default: EURUSD): [Enter]
Start date (default: 2024-01-01): [Enter]
End date (default: 2024-12-31): [Enter]
Use GPU? (y/n, default: y): y [Enter]
How many TOP configurations to save? (default: 10): 10 [Enter]
Results folder name (default: auto): [Enter]

Population size (default: 100): [Enter]
Number of generations (default: 50): [Enter]
```

**DONE!** System will start optimization.

---

## 📂 WHERE ARE RESULTS?

After completion:

1. **CSV with results:**
   ```
   d:\Blessing Optymalizer\data\results\opt_YYYYMMDD_HHMMSS\genetic_top_10.csv
   ```

2. **.set files for MT4/MT5:**
   ```
   d:\Blessing Optymalizer\data\set_files\opt_YYYYMMDD_HHMMSS\
   ├── blessing_rank01_score57092_wr80.set  ← BEST
   ├── blessing_rank02_score42531_wr80.set
   ├── ...
   └── blessing_rank10_score20648_wr86.set
   ```

3. **Copy best .set to MT4/MT5:**
   ```
   Copy blessing_rank01_*.set to:
   MT4: C:\Program Files\MetaTrader 4\MQL4\Presets\
   MT5: C:\Program Files\MetaTrader 5\MQL5\Presets\
   ```

4. **Load in platform:**
   - Drag Blessing EA onto chart
   - Click "Load" → select .set file
   - Click OK → DONE!

---

## 🎯 WHICH STRATEGY TO CHOOSE?

| If you want... | Choose | Time |
|----------------|--------|------|
| Fastest | **C** (Genetic) | 1-2 weeks |
| Best result | **D** (Hybrid) | 3-4 weeks |
| Improve previous results | **E** (Refine) | 1 week |
| Full control | **A** (Sequential) | 2-3 weeks |
| Intelligent sampling | **B** (Bayesian) | 2-3 weeks |

**RECOMMENDED:** Option **C** for first optimization, then **E** to improve results.

---

## ⚠️ NOTES

- **GPU accelerates ~15x** - make sure GPU is enabled
- **Ctrl+C interrupts** - you can resume later (Sequential only)
- **Backtests ≠ live** - test on demo before live trading!
- **TOP 10 instead of 5** - system automatically generates TOP 10 .set files

---

## 📖 MORE INFORMATION

- Full instructions: `INSTRUKCJA_URUCHOMIENIA.md`
- Plan for all phases: `docs/PELNA_OPTYMALIZACJA_PLAN.md`
- PHASE 1 analysis: `data/results/ANALIZA_WYNIKOW.md`
- Technical details: `CLAUDE.md`

---

**Good luck! 🚀**
