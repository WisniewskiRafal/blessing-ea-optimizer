# ⚡ QUICK START - Blessing EA Optimizer

## 🚀 SZYBKI START (3 KROKI)

### Krok 1: Zainstaluj wymagane biblioteki

```bash
cd "d:\Blessing Optymalizer"
pip install scikit-optimize deap
```

### Krok 2: Uruchom optymalizator

```bash
python blessing_optimizer_main.py
```

### Krok 3: Wybierz strategię

```
Wybierz opcję (A/B/C/D/E/X): C [Enter]

Symbol (default: EURUSD): [Enter]
Data początkowa (default: 2024-01-01): [Enter]
Data końcowa (default: 2024-12-31): [Enter]
Użyć GPU? (y/n, default: y): y [Enter]
Ile TOP konfiguracji zapisać? (default: 10): 10 [Enter]
Nazwa folderu wyników (default: auto): [Enter]

Wielkość populacji (default: 100): [Enter]
Liczba generacji (default: 50): [Enter]
```

**GOTOWE!** System zacznie optymalizację.

---

## 📂 GDZIE SĄ WYNIKI?

Po zakończeniu:

1. **CSV z wynikami:**
   ```
   d:\Blessing Optymalizer\data\results\opt_YYYYMMDD_HHMMSS\genetic_top_10.csv
   ```

2. **Pliki .set dla MT4/MT5:**
   ```
   d:\Blessing Optymalizer\data\set_files\opt_YYYYMMDD_HHMMSS\
   ├── blessing_rank01_score57092_wr80.set  ← NAJLEPSZY
   ├── blessing_rank02_score42531_wr80.set
   ├── ...
   └── blessing_rank10_score20648_wr86.set
   ```

3. **Skopiuj najlepszy .set do MT4/MT5:**
   ```
   Skopiuj blessing_rank01_*.set do:
   MT4: C:\Program Files\MetaTrader 4\MQL4\Presets\
   MT5: C:\Program Files\MetaTrader 5\MQL5\Presets\
   ```

4. **Załaduj w platformie:**
   - Przeciągnij Blessing EA na wykres
   - Kliknij "Load" → wybierz plik .set
   - Kliknij OK → GOTOWE!

---

## 🎯 KTÓRA STRATEGIA WYBRAĆ?

| Jeśli chcesz... | Wybierz | Czas |
|------------------|---------|------|
| Najszybciej | **C** (Genetic) | 1-2 tygodnie |
| Najlepszy wynik | **D** (Hybrid) | 3-4 tygodnie |
| Poprawić poprzednie wyniki | **E** (Refine) | 1 tydzień |
| Pełną kontrolę | **A** (Sequential) | 2-3 tygodnie |
| Inteligentne próbkowanie | **B** (Bayesian) | 2-3 tygodnie |

**ZALECAM:** Opcja **C** dla pierwszej optymalizacji, potem **E** żeby poprawić wyniki.

---

## ⚠️ UWAGI

- **GPU przyśpiesza ~15x** - upewnij się że GPU jest włączone
- **Ctrl+C przerywa** - możesz wznowić później (tylko Sequential)
- **Backtesty ≠ live** - testuj na demo przed live trading!
- **TOP 10 zamiast 5** - system automatycznie generuje TOP 10 plików .set

---

## 📖 WIĘCEJ INFORMACJI

- Pełna instrukcja: `INSTRUKCJA_URUCHOMIENIA.md`
- Plan wszystkich faz: `docs/PELNA_OPTYMALIZACJA_PLAN.md`
- Analiza FAZY 1: `data/results/ANALIZA_WYNIKOW.md`
- Technicznie: `CLAUDE.md`

---

**Powodzenia! 🚀**
