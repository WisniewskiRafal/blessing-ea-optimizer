# BLESSING CONTINUOUS OPTIMIZER - INSTRUKCJA UŻYCIA

**System ciągłej optymalizacji z automatycznym zapisem stanu**

---

## 🚀 SZYBKI START

### Uruchomienie (pierwszy raz):
```bash
python run_optimizer.py
```

### Zatrzymanie:
- Naciśnij `Ctrl+C`
- Stan zostanie automatycznie zapisany

### Wznowienie (następny dzień):
```bash
python run_optimizer.py
```
- System automatycznie wczyta ostatni stan
- Kontynuuje od miejsca przerwania

---

## 📊 JAK TO DZIAŁA

### 1. Automatyczny zapis stanu
- **Co 10 backtestów** (domyślnie) system zapisuje stan do pliku
- Stan zawiera:
  - Aktualną fazę optymalizacji
  - Przetestowane kombinacje
  - Najlepsze wyniki
  - Postęp (X/Y kombinacji)
  - Czas pracy

### 2. Resume po przerwaniu
- Po włączeniu system:
  1. Sprawdza czy istnieje plik stanu
  2. Wczytuje ostatni stan
  3. Kontynuuje od przerwania
  4. Pomija już przetestowane kombinacje

### 3. Fazy optymalizacji
System przechodzi przez 5 faz:

**FAZA 1: Entry Combinations** (3,888 kombinacji)
- Testuje wszystkie kombinacje wskaźników entry
- MA: 0/1/2 × CCI: 0/1/2 × Bollinger: 0/1/2 × Stochastic: 0/1/2 × MACD: 0/1/2
- × B3Traditional × ForceMarketCond × UseAnyEntry
- Wybiera top 5 najlepszych

**FAZA 2: Indicator Parameters** (~500 kombinacji)
- Dla top 5 z Fazy 1
- Testuje różne parametry wskaźników
- MA Period, CCI Period, Bollinger settings, etc.
- Wybiera top 3

**FAZA 3: Grid Configuration** (~150 kombinacji)
- Dla top 3 z Fazy 2
- Testuje konfiguracje gridu
- AutoCal, GAF, SmartGrid, EntryDelay
- Wybiera top 3

**FAZA 4: Money Management** (~60 kombinacji)
- Dla top 3 z Fazy 3
- Testuje MM parametry
- Multiplier, LAF, MaxDD
- Wybiera top 3

**FAZA 5: Exit Strategies** (~90 kombinacji)
- Dla top 3 z Fazy 4
- Testuje strategie wyjścia
- MaximizeProfit, UseStopLoss, EarlyExit
- Finalny wynik

**TOTAL: ~4,700 backtestów** (zamiast miliardów dzięki hierarchii!)

---

## ⚙️ OPCJE URUCHOMIENIA

### Podstawowe:
```bash
# Uruchom z domyślnymi ustawieniami (nieskończony czas)
python run_optimizer.py
```

### Z limitami:
```bash
# Uruchom na 2 godziny
python run_optimizer.py --max-hours 2

# Uruchom 100 backtestów i stop
python run_optimizer.py --max-backtests 100

# Kombinacja: max 8h LUB 500 backtestów (co pierwsze)
python run_optimizer.py --max-hours 8 --max-backtests 500
```

### GPU/CPU:
```bash
# Z GPU (domyślnie)
python run_optimizer.py --gpu

# Bez GPU (tylko CPU)
python run_optimizer.py --no-gpu
```

### Checkpoint interval:
```bash
# Zapisuj stan co 5 backtestów (częściej)
python run_optimizer.py --checkpoint-interval 5

# Zapisuj co 50 backtestów (rzadziej, szybciej)
python run_optimizer.py --checkpoint-interval 50
```

### Custom pliki:
```bash
# Własna ścieżka stanu i wyników
python run_optimizer.py \
  --state-file "my_state.pkl" \
  --results-dir "my_results"
```

---

## 📁 PLIKI I STRUKTURA

### Stan optymalizacji:
```
data/state/blessing_optimizer_state.pkl
```
- Plik binarny (pickle)
- Zawiera cały stan optymalizacji
- Backup automatyczny przy każdym zapisie

### Wyniki:
```
data/results/continuous/
├── phase_1_entry_combinations.csv
├── phase_2_indicator_parameters.csv
├── phase_3_grid_configuration.csv
├── phase_4_money_management.csv
├── phase_5_exit_strategies.csv
└── final_best_configuration.json
```

### Logi:
```
logs/optimizer.log
```
- Wszystkie eventy
- Błędy
- Progress updates

---

## 🔄 TYPOWY WORKFLOW

### Dzień 1 (Poniedziałek rano):
```bash
# Uruchom na 8 godzin
python run_optimizer.py --max-hours 8
```
- System startuje od początku (Faza 1)
- Testuje ~1,000 kombinacji
- Po 8h lub Ctrl+C: zapisuje stan i kończy

### Dzień 2 (Wtorek rano):
```bash
# Wznów
python run_optimizer.py --max-hours 8
```
- System wczytuje stan z Dnia 1
- Kontynuuje od kombinacji 1,001
- Testuje kolejne ~1,000
- Zapisuje stan

### ...kontynuuj aż do końca

### Sprawdzenie postępu:
System pokazuje przy starcie:
```
[RESUME] Loaded previous state
  Phase: entry_combinations
  Level: 0
  Progress: 1542/3888 (39.7%)
  Best score: 245.67
  Total backtests: 1542
  Runtime: 12.3h
```

---

## 📊 MONITORING POSTĘPU

### Konsola (real-time):
```
[PROGRESS] Phase: entry_combinations
  Iteration: 1234/3888 (31.7%)
  Best score: 189.45
  Total backtests: 1234
  Runtime: 8.2h

[CHECKPOINT] State saved (1240 backtests)
```

### Log file:
```bash
tail -f logs/optimizer.log
```

### Plik stanu (Python):
```python
import pickle

with open('data/state/blessing_optimizer_state.pkl', 'rb') as f:
    state = pickle.load(f)

print(f"Phase: {state.current_phase}")
print(f"Progress: {state.current_iteration}/{state.total_iterations}")
print(f"Best score: {state.current_best_score}")
```

---

## ⚡ WYDAJNOŚĆ

### GPU vs CPU:
- **GPU:** ~200 backtestów/s
- **CPU:** ~30 backtestów/s

### Szacunki czasu (GPU):
- **Faza 1:** 3,888 / 200 = ~19 sekund × overhead = **~30 sekund**
- **Faza 2:** 500 / 200 = ~3 sekundy
- **Faza 3:** 150 / 200 = ~1 sekunda
- **Faza 4:** 60 / 200 = ~0.3 sekundy
- **Faza 5:** 90 / 200 = ~0.5 sekundy

**TOTAL: ~35 sekund per timeframe!** (single walk-forward period)

### Walk-forward (3 periods):
- Q1→Q2: 35s
- Q1+Q2→Q3: 35s
- Q1+Q2+Q3→Q4: 35s
**= ~2 minuty total per TF**

### Multiple timeframes (7 TF):
- M1, M5, M15, M30, H1, H4, D1
**= 2 min × 7 = ~14 minut TOTAL!**

---

## 🛡️ SAFETY & RECOVERY

### Co się dzieje przy błędzie?
1. System zapisuje stan PRZED każdym backtestem
2. Przy błędzie: stan jest już zapisany
3. Uruchom ponownie - system pominie błędną kombinację

### Co się dzieje przy crash?
1. Stan zapisany co 10 backtestów
2. Maksymalnie stracisz 10 backtestów
3. Uruchom ponownie - wznowi od ostatniego checkpoint

### Ctrl+C handling:
```python
try:
    optimizer.run_continuous()
except KeyboardInterrupt:
    # Stan zapisany automatycznie
    print("[SAVED] State saved - safe to exit")
```

---

## 🎯 PRZYKŁADY UŻYCIA

### Przykład 1: Testowanie przez weekend
```bash
# Piątek wieczorem - uruchom na 48h
python run_optimizer.py --max-hours 48
```

### Przykład 2: Codzienne 8h sesje
```bash
# Każdego ranka
python run_optimizer.py --max-hours 8
```

### Przykład 3: Szybki test (100 kombinacji)
```bash
python run_optimizer.py --max-backtests 100 --no-gpu
```

### Przykład 4: Full run do końca
```bash
# Uruchom i zostaw (zatrzyma się sam po zakończeniu)
nohup python run_optimizer.py > optimizer.out 2>&1 &
```

---

## 🐛 TROUBLESHOOTING

### Problem: "State file corrupted"
**Rozwiązanie:**
```bash
# Usuń plik stanu i zacznij od nowa
rm data/state/blessing_optimizer_state.pkl
python run_optimizer.py
```

### Problem: GPU out of memory
**Rozwiązanie:**
```bash
# Użyj CPU lub zmniejsz batch size
python run_optimizer.py --no-gpu
```

### Problem: Optimizer stuck
**Rozwiązanie:**
```bash
# Sprawdź log
tail -100 logs/optimizer.log

# Jeśli deadlock: kill i restart
Ctrl+C
python run_optimizer.py  # Wznowi od checkpoint
```

---

## 📈 NAJLEPSZE PRAKTYKI

1. **Uruchamiaj regularnie:**
   - Codziennie 8h jest lepsze niż raz w tygodniu 56h
   - Częstsze checkpointy = mniejsze ryzyko utraty

2. **Monitoruj logi:**
   - Sprawdzaj `logs/optimizer.log` regularnie
   - Szukaj błędów i ostrzeżeń

3. **Backup stanu:**
   ```bash
   # Co tydzień
   cp data/state/blessing_optimizer_state.pkl \
      data/state/backup_$(date +%Y%m%d).pkl
   ```

4. **Sprawdzaj postęp:**
   - Przed zatrzymaniem sprawdź ile zostało
   - Szacuj czas do końca

5. **Analizuj wyniki na bieżąco:**
   - Nie czekaj do końca
   - Po każdej fazie sprawdzaj top wyniki

---

## 🎉 KOŃCOWY WYNIK

Po zakończeniu wszystkich faz znajdziesz:

```
data/results/continuous/final_best_configuration.json
```

Zawiera:
- Najlepsza konfiguracja dla każdego TF
- Walk-forward results
- Wszystkie metryki
- Gotowe do użycia w live trading!

---

**Powodzenia w optymalizacji!** 🚀
