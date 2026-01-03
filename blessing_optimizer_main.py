"""
Blessing EA Optimizer - Main Launcher
Interactive menu for choosing optimization strategy
"""

import sys
import os
from pathlib import Path
from typing import Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from optimization.sequential_optimizer import SequentialOptimizer
from optimization.bayesian_optimizer import BayesianOptimizer
from optimization.genetic_optimizer import GeneticOptimizer
from utils.set_file_generator import SetFileGenerator


class BlessingOptimizerLauncher:
    """Main launcher with interactive menu"""

    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Paths
        self.project_root = Path(__file__).parent
        self.results_dir = self.project_root / "data" / "results"
        self.set_files_dir = self.project_root / "data" / "set_files"
        self.state_dir = self.project_root / "data" / "state"

        # Create directories
        for dir_path in [self.results_dir, self.set_files_dir, self.state_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('blessing_optimizer.log'),
                logging.StreamHandler()
            ]
        )

    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "="*80)
        print("  ____  _               _             _____ _              ___        _   _           _              ")
        print(" | __ )| | ___  ___ ___(_)_ __   __ _| ____/ \\            / _ \\ _ __ | |_(_)_ __ ___ (_)_______ _ __ ")
        print(" |  _ \\| |/ _ \\/ __/ __| | '_ \\ / _` |  _|/ _ \\   _____  | | | | '_ \\| __| | '_ ` _ \\| |_  / _ \\ '__|")
        print(" | |_) | |  __/\\__ \\__ \\ | | | | (_| | |_/ ___ \\ |_____| | |_| | |_) | |_| | | | | | | |/ /  __/ |   ")
        print(" |____/|_|\\___||___/___/_|_| |_|\\__, |_/_/   \\_\\          \\___/| .__/ \\__|_|_| |_| |_|_/___\\___|_|   ")
        print("                                 |___/                          |_|                                   ")
        print("="*80)
        print("  Kompletny system optymalizacji Blessing EA v3.9.6.09")
        print("  Wszystkie 134 parametry | 3 strategie optymalizacji | TOP 10 .set files")
        print("="*80 + "\n")

    def print_main_menu(self):
        """Print main menu"""
        print("\n" + "="*80)
        print("WYBIERZ STRATEGIĘ OPTYMALIZACJI:")
        print("="*80)
        print()
        print("  [A] SEKWENCYJNA OPTYMALIZACJA")
        print("      > Faza po fazie: Entry > Timeframes > Indicators > Grid > Risk")
        print("      > ~300,000 backtestow")
        print("      > Czas: 2-3 tygodnie")
        print("      > Zaleta: Pelna kontrola, widzisz postep kazdej fazy")
        print()
        print("  [B] BAYESIAN OPTIMIZATION")
        print("      > Sekwencyjna + inteligentne probkowanie")
        print("      > ~300,000 backtestow (lepiej wykorzystane)")
        print("      > Czas: 2-3 tygodnie")
        print("      > Zaleta: Szybciej znajduje optima, uczy sie z wynikow")
        print()
        print("  [C] GENETIC ALGORITHM")
        print("      > Ewolucyjna optymalizacja WSZYSTKICH 64 parametrow jednoczesnie")
        print("      > ~5,000-10,000 backtestow")
        print("      > Czas: 1-2 tygodnie")
        print("      > Zaleta: Uwzglednia interakcje miedzy parametrami")
        print()
        print("  [D] GENETIC + REFINEMENT (HYBRYDOWA)")
        print("      > Genetic (5k BT) > wybierz TOP 5 > Bayesian refinement kazdego")
        print("      > ~50,000 backtestow")
        print("      > Czas: 3-4 tygodnie")
        print("      > Zaleta: Najlepsze z obu swiatow")
        print()
        print("  [E] GENETIC - TESTUJ POPRZEDNIE WYNIKI")
        print("      > Uzyj TOP 10 z poprzedniej optymalizacji jako populacji startowej")
        print("      > ~5,000 backtestow")
        print("      > Czas: 1 tydzien")
        print("      > Zaleta: Refinement znalezionych konfiguracji")
        print()
        print("  [X] Wyjście")
        print()
        print("="*80)

    def get_optimization_params(self):
        """Get common optimization parameters"""
        print("\n" + "="*80)
        print("PARAMETRY OPTYMALIZACJI:")
        print("="*80 + "\n")

        # Symbol
        symbol = input("Symbol (default: EURUSD): ").strip() or "EURUSD"

        # Date range
        start_date = input("Data początkowa (default: 2024-01-01): ").strip() or "2024-01-01"
        end_date = input("Data końcowa (default: 2024-12-31): ").strip() or "2024-12-31"

        # GPU
        use_gpu_input = input("Użyć GPU? (y/n, default: y): ").strip().lower()
        use_gpu = use_gpu_input != 'n'

        # Top N results
        top_n = int(input("Ile TOP konfiguracji zapisać? (default: 10): ").strip() or "10")

        # Results directory
        results_subdir = input("Nazwa folderu wyników (default: auto): ").strip()
        if not results_subdir:
            from datetime import datetime
            results_subdir = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'use_gpu': use_gpu,
            'top_n': top_n,
            'results_dir': self.results_dir / results_subdir,
            'set_files_dir': self.set_files_dir / results_subdir,
        }

    def run_sequential_optimization(self, params: dict):
        """Run sequential (phase-by-phase) optimization"""
        print("\n" + "="*80)
        print("URUCHAMIANIE SEKWENCYJNEJ OPTYMALIZACJI")
        print("="*80 + "\n")

        optimizer = SequentialOptimizer(
            symbol=params['symbol'],
            start_date=params['start_date'],
            end_date=params['end_date'],
            results_dir=params['results_dir'],
            use_gpu=params['use_gpu'],
            top_n=params['top_n']
        )

        # Run all phases
        results = optimizer.run_all_phases()

        # Generate .set files
        self.generate_set_files(results, params)

        return results

    def run_bayesian_optimization(self, params: dict):
        """Run Bayesian optimization"""
        print("\n" + "="*80)
        print("URUCHAMIANIE BAYESIAN OPTIMIZATION")
        print("="*80 + "\n")

        optimizer = BayesianOptimizer(
            symbol=params['symbol'],
            start_date=params['start_date'],
            end_date=params['end_date'],
            results_dir=params['results_dir'],
            use_gpu=params['use_gpu'],
            top_n=params['top_n']
        )

        # Run optimization
        results = optimizer.run_optimization()

        # Generate .set files
        self.generate_set_files(results, params)

        return results

    def run_genetic_optimization(self, params: dict, seed_population: Optional[list] = None):
        """Run genetic algorithm optimization"""
        print("\n" + "="*80)
        if seed_population:
            print("URUCHAMIANIE GENETIC ALGORITHM (Z SEED POPULATION)")
        else:
            print("URUCHAMIANIE GENETIC ALGORITHM")
        print("="*80 + "\n")

        # Get GA parameters
        population_size = int(input("Wielkość populacji (default: 100): ").strip() or "100")
        generations = int(input("Liczba generacji (default: 50): ").strip() or "50")

        optimizer = GeneticOptimizer(
            symbol=params['symbol'],
            start_date=params['start_date'],
            end_date=params['end_date'],
            results_dir=params['results_dir'],
            use_gpu=params['use_gpu'],
            top_n=params['top_n'],
            population_size=population_size,
            generations=generations
        )

        # Run optimization
        if seed_population:
            results = optimizer.run_with_seed_population(seed_population)
        else:
            results = optimizer.run_optimization()

        # Generate .set files
        self.generate_set_files(results, params)

        return results

    def run_hybrid_optimization(self, params: dict):
        """Run hybrid (Genetic + Bayesian refinement) optimization"""
        print("\n" + "="*80)
        print("URUCHAMIANIE HYBRYDOWEJ OPTYMALIZACJI")
        print("="*80 + "\n")

        # Step 1: Genetic Algorithm
        print("\n[KROK 1/2] Genetic Algorithm - szukanie TOP 5 konfiguracji...\n")
        ga_params = {**params, 'top_n': 5}
        ga_results = self.run_genetic_optimization(ga_params)

        # Step 2: Bayesian refinement for each top config
        print("\n[KROK 2/2] Bayesian Refinement - optymalizacja każdej z TOP 5...\n")
        all_refined_results = []

        for i, config in enumerate(ga_results['top_configs'][:5], 1):
            print(f"\n>>> Refinement {i}/5: Config ze score {config.get('score', 0):.2f}\n")

            bayesian_optimizer = BayesianOptimizer(
                symbol=params['symbol'],
                start_date=params['start_date'],
                end_date=params['end_date'],
                results_dir=params['results_dir'] / f"refinement_{i}",
                use_gpu=params['use_gpu'],
                top_n=2
            )

            refined = bayesian_optimizer.run_refinement(config)
            all_refined_results.extend(refined)

        # Sort all refined results
        all_refined_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        results = {
            'top_configs': all_refined_results[:params['top_n']],
            'all_results': all_refined_results
        }

        # Generate .set files
        self.generate_set_files(results, params)

        return results

    def load_previous_results(self):
        """Load previous optimization results for seeding GA"""
        print("\n" + "="*80)
        print("WYBIERZ POPRZEDNIE WYNIKI DO ZAŁADOWANIA")
        print("="*80 + "\n")

        # List available result directories
        result_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]

        if not result_dirs:
            print("[ERROR] Brak poprzednich wyników optymalizacji!")
            return None

        print("Dostępne wyniki:\n")
        for i, dir_path in enumerate(result_dirs, 1):
            csv_files = list(dir_path.glob("*top_*.csv"))
            if csv_files:
                print(f"  [{i}] {dir_path.name} ({len(csv_files)} plików CSV)")

        print()
        choice = input("Wybierz numer (0 = anuluj): ").strip()

        if not choice or choice == '0':
            return None

        try:
            selected_dir = result_dirs[int(choice) - 1]
        except (ValueError, IndexError):
            print("[ERROR] Nieprawidłowy wybór!")
            return None

        # Load TOP 10 from CSV
        csv_files = list(selected_dir.glob("*top_*.csv"))
        if not csv_files:
            print("[ERROR] Brak plików CSV w wybranym katalogu!")
            return None

        # Use the most recent/largest CSV
        csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)

        print(f"\n[DIR] Ładowanie: {csv_file.name}")

        import pandas as pd
        df = pd.read_csv(csv_file)

        # Convert to list of config dicts
        configs = []
        for _, row in df.head(10).iterrows():
            config = row.to_dict()
            configs.append(config)

        print(f"[OK] Załadowano {len(configs)} konfiguracji")
        print(f"   Najlepsza: score={configs[0].get('score', 0):.2f}, "
              f"win_rate={configs[0].get('win_rate', 0):.1%}")

        return configs

    def generate_set_files(self, results: dict, params: dict):
        """Generate .set files for top configurations"""
        print("\n" + "="*80)
        print(f"GENEROWANIE TOP {params['top_n']} PLIKÓW .SET")
        print("="*80 + "\n")

        params['set_files_dir'].mkdir(parents=True, exist_ok=True)

        generator = SetFileGenerator()

        top_configs = results.get('top_configs', [])[:params['top_n']]

        for i, config in enumerate(top_configs, 1):
            score = config.get('score', 0)
            win_rate = config.get('win_rate', 0)

            filename = f"blessing_rank{i:02d}_score{int(score)}_wr{int(win_rate*100)}.set"
            output_path = params['set_files_dir'] / filename

            generator.generate_set_file(
                config=config,
                output_path=str(output_path),
                name=f"Rank #{i}",
                magic_number=10000 + i - 1
            )

            print(f"  [OK] [{i}/{params['top_n']}] {filename}")

        print(f"\n[DIR] Pliki .set zapisane w: {params['set_files_dir']}")
        print(f"   Skopiuj do: MT4/MQL4/Presets/ lub MT5/MQL5/Presets/")

    def run(self):
        """Main run loop"""
        self.print_banner()

        while True:
            self.print_main_menu()
            choice = input("Wybierz opcję (A/B/C/D/E/X): ").strip().upper()

            if choice == 'X':
                print("\n[BYE] Do widzenia!\n")
                break

            if choice not in ['A', 'B', 'C', 'D', 'E']:
                print("\n[ERROR] Nieprawidłowy wybór! Spróbuj ponownie.\n")
                continue

            # Get common parameters
            params = self.get_optimization_params()

            try:
                if choice == 'A':
                    results = self.run_sequential_optimization(params)

                elif choice == 'B':
                    results = self.run_bayesian_optimization(params)

                elif choice == 'C':
                    results = self.run_genetic_optimization(params)

                elif choice == 'D':
                    results = self.run_hybrid_optimization(params)

                elif choice == 'E':
                    # Load previous results
                    seed_population = self.load_previous_results()
                    if seed_population:
                        results = self.run_genetic_optimization(params, seed_population)
                    else:
                        print("\n⚠️  Anulowano - brak seed population")
                        continue

                # Print summary
                print("\n" + "="*80)
                print("[SUCCESS] OPTYMALIZACJA ZAKOŃCZONA!")
                print("="*80)
                print(f"\n[STATS] Wyniki:")
                print(f"   TOP 1: score={results['top_configs'][0].get('score', 0):.2f}, "
                      f"win_rate={results['top_configs'][0].get('win_rate', 0):.1%}")
                print(f"\n📁 Lokalizacja:")
                print(f"   CSV: {params['results_dir']}")
                print(f"   .SET: {params['set_files_dir']}")
                print()

                # Ask if continue
                continue_choice = input("Uruchomić kolejną optymalizację? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    print("\n[BYE] Do widzenia!\n")
                    break

            except KeyboardInterrupt:
                print("\n\n⚠️  Optymalizacja przerwana przez użytkownika (Ctrl+C)")
                print("   Stan został zapisany - możesz wznowić później")
                break

            except Exception as e:
                self.logger.error(f"Błąd podczas optymalizacji: {e}", exc_info=True)
                print(f"\n[ERROR] BŁĄD: {e}")
                print("   Sprawdź logi: blessing_optimizer.log")

                retry = input("\nSpróbować ponownie? (y/n): ").strip().lower()
                if retry != 'y':
                    break


def main():
    """Entry point"""
    launcher = BlessingOptimizerLauncher()
    launcher.run()


if __name__ == '__main__':
    main()
