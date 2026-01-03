# optimization/blessing_continuous_optimizer.py
# Continuous Blessing Optimizer with State Persistence
# Author: Rafał Wiśniewski | Data & AI Solutions

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from core.data_loader import DataLoader
from core.blessing_backtest_engine import BlessingBacktestEngine, BacktestConfig
from optimization.hierarchical_optimizer import HierarchicalOptimizer, OptimizationLevel, ParameterSpace
from optimization.walk_forward_analyzer import WalkForwardAnalyzer
from strategies.blessing_entry_generator import EntryConfig
from strategies.blessing_grid_system import GridConfig


@dataclass
class OptimizationState:
    """Stan optymalizacji - zapisywany do pliku po każdym kroku"""

    # Progress tracking
    current_phase: str  # 'entry_combinations', 'parameters', 'grid', 'money_mgmt', 'exit'
    current_level: int  # Poziom hierarchii (0-5)
    current_iteration: int  # Iteracja w danym poziomie
    total_iterations: int  # Total iteracji w poziomie

    # Best results so far
    best_combinations: List[Dict]  # Top kombinacje z każdego poziomu
    current_best_config: Dict  # Najlepsza konfiguracja do tej pory
    current_best_score: float

    # Completed work
    completed_combinations: List[str]  # Hash kombinacji które już przetestowano
    all_results: List[Dict]  # Wszystkie wyniki

    # Timestamps
    start_time: str
    last_update: str
    total_runtime_seconds: float

    # Walk-forward periods completed
    wf_periods_completed: List[str]
    current_wf_period: Optional[str]

    # Statistics
    total_backtests_run: int
    successful_backtests: int
    failed_backtests: int


class BlessingContinuousOptimizer:
    """
    Continuous Blessing Optimizer

    Features:
    - Auto-save state after each backtest
    - Resume from last checkpoint
    - Can run indefinitely until all combinations tested
    - Progress tracking and estimation
    """

    def __init__(self,
                 state_file: str = 'data/state/blessing_optimizer_state.pkl',
                 results_dir: str = 'data/results/continuous',
                 checkpoint_interval: int = 10,  # Save every N backtests
                 use_gpu: bool = True,
                 verbose: bool = True,
                 symbol: str = 'EURUSD',
                 start_date: str = '2024-01-01',
                 end_date: str = '2024-12-31'):
        """
        Args:
            state_file: Path to state pickle file
            results_dir: Directory for results
            checkpoint_interval: Save state every N backtests
            use_gpu: Use GPU acceleration
            verbose: Print progress
            symbol: Trading symbol (default: EURUSD)
            start_date: Start date for data
            end_date: End date for data
        """
        self.state_file = Path(state_file)
        self.results_dir = Path(results_dir)
        self.checkpoint_interval = checkpoint_interval
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

        # Create directories
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # State
        self.state: Optional[OptimizationState] = None
        self.backtest_counter = 0

        # Load market data
        self.market_data = None
        self._load_market_data()

        # Optimizer
        self.optimizer = HierarchicalOptimizer(
            objective='net_profit',
            use_gpu=use_gpu,
            n_workers=4,
            verbose=False  # Kontrolujemy output sami
        )

    def _load_market_data(self):
        """Load market data from Dukascopy"""
        try:
            self.logger.info(f"[DATA] Loading {self.symbol} data: {self.start_date} to {self.end_date}")

            loader = DataLoader()
            df = loader.load_dukascopy_m1(self.symbol, self.start_date, self.end_date)

            self.market_data = {
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'bars': len(df)
            }

            self.logger.info(f"[DATA] Loaded {self.market_data['bars']:,} bars")

        except Exception as e:
            self.logger.warning(f"[DATA] Failed to load market data: {e}")
            self.logger.warning("[DATA] Will use random data for testing")
            self.market_data = None

    def load_state(self) -> bool:
        """
        Load state from file

        Returns:
            True if state loaded, False if starting fresh
        """
        if not self.state_file.exists():
            self.logger.info("[NEW] No previous state found - starting fresh")
            return False

        try:
            with open(self.state_file, 'rb') as f:
                self.state = pickle.load(f)

            self.logger.info("[RESUME] Loaded previous state")
            self.logger.info(f"  Phase: {self.state.current_phase}")
            self.logger.info(f"  Level: {self.state.current_level}")
            self.logger.info(f"  Progress: {self.state.current_iteration}/{self.state.total_iterations}")
            self.logger.info(f"  Best score: {self.state.current_best_score:.2f}")
            self.logger.info(f"  Total backtests: {self.state.total_backtests_run}")
            self.logger.info(f"  Runtime: {self.state.total_runtime_seconds/3600:.1f}h")

            return True

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load state: {e}")
            return False

    def save_state(self):
        """Save current state to file"""
        if self.state is None:
            return

        # Update timestamps
        self.state.last_update = datetime.now().isoformat()

        # Save to temp file first (atomic write)
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'wb') as f:
            pickle.dump(self.state, f)

        # Rename (atomic on most systems)
        temp_file.replace(self.state_file)

        if self.verbose and self.backtest_counter % self.checkpoint_interval == 0:
            self.logger.info(f"[CHECKPOINT] State saved ({self.state.total_backtests_run} backtests)")

    def initialize_state(self):
        """Initialize fresh state"""
        self.state = OptimizationState(
            current_phase='entry_combinations',
            current_level=0,
            current_iteration=0,
            total_iterations=0,
            best_combinations=[],
            current_best_config={},
            current_best_score=-np.inf,
            completed_combinations=[],
            all_results=[],
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
            total_runtime_seconds=0.0,
            wf_periods_completed=[],
            current_wf_period=None,
            total_backtests_run=0,
            successful_backtests=0,
            failed_backtests=0
        )
        self.save_state()

    def generate_entry_combinations(self) -> List[Dict]:
        """
        Generate all 3,888 entry combinations

        Returns:
            List of entry configuration dicts
        """
        combinations = []

        # Entry indicators: 0=off, 1=normal, 2=reverse
        ma_options = [0, 1, 2]
        cci_options = [0, 1, 2]
        boll_options = [0, 1, 2]
        stoch_options = [0, 1, 2]
        macd_options = [0, 1, 2]

        # Entry logic
        traditional_options = [True, False]
        market_cond_options = [0, 1, 2, 3]  # uptrend, downtrend, range, off
        any_entry_options = [True, False]

        # Generate all combinations
        for ma in ma_options:
            for cci in cci_options:
                for boll in boll_options:
                    for stoch in stoch_options:
                        for macd in macd_options:
                            for trad in traditional_options:
                                for cond in market_cond_options:
                                    for any_e in any_entry_options:
                                        config = {
                                            'ma_entry': ma,
                                            'cci_entry': cci,
                                            'bollinger_entry': boll,
                                            'stoch_entry': stoch,
                                            'macd_entry': macd,
                                            'b3_traditional': trad,
                                            'force_market_cond': cond,
                                            'use_any_entry': any_e
                                        }
                                        combinations.append(config)

        self.logger.info(f"[GENERATED] {len(combinations)} entry combinations")
        return combinations

    def config_hash(self, config: Dict) -> str:
        """Generate unique hash for configuration"""
        # Sort keys for consistent hash
        sorted_items = sorted(config.items())
        return str(hash(tuple(sorted_items)))

    def run_continuous(self,
                      max_runtime_hours: Optional[float] = None,
                      max_backtests: Optional[int] = None):
        """
        Run continuous optimization

        Args:
            max_runtime_hours: Stop after this many hours (None = infinite)
            max_backtests: Stop after this many backtests (None = infinite)
        """
        # Load or initialize state
        if not self.load_state():
            self.initialize_state()

        start_session_time = time.time()
        session_backtests = 0

        self.logger.info("="*60)
        self.logger.info("BLESSING CONTINUOUS OPTIMIZER - START")
        self.logger.info("="*60)
        self.logger.info(f"Max runtime: {max_runtime_hours}h" if max_runtime_hours else "Max runtime: INFINITE")
        self.logger.info(f"Max backtests: {max_backtests}" if max_backtests else "Max backtests: INFINITE")
        self.logger.info(f"GPU: {'ENABLED' if self.use_gpu else 'DISABLED'}")
        self.logger.info("")

        try:
            # PHASE 1: Entry Combinations (3,888 combinations)
            if self.state.current_phase == 'entry_combinations':
                self._run_entry_combinations_phase(max_backtests=max_backtests)

            # PHASE 2: Indicator Parameters
            if self.state.current_phase == 'indicator_parameters':
                self._run_indicator_parameters_phase()

            # PHASE 3: Grid Configuration
            if self.state.current_phase == 'grid_configuration':
                self._run_grid_configuration_phase()

            # PHASE 4: Money Management
            if self.state.current_phase == 'money_management':
                self._run_money_management_phase()

            # PHASE 5: Exit Strategies
            if self.state.current_phase == 'exit_strategies':
                self._run_exit_strategies_phase()

            # Check stop conditions
            session_runtime = (time.time() - start_session_time) / 3600
            if max_runtime_hours and session_runtime >= max_runtime_hours:
                self.logger.info(f"[STOP] Max runtime reached: {session_runtime:.1f}h")
                return

            if max_backtests and session_backtests >= max_backtests:
                self.logger.info(f"[STOP] Max backtests reached: {session_backtests}")
                return

        except KeyboardInterrupt:
            self.logger.info("\n[INTERRUPTED] Saving state before exit...")
            self.save_state()
            self.logger.info("[SAVED] State saved successfully")
            self.logger.info("[INFO] You can resume by running again")
            raise

        finally:
            # Final save
            self.save_state()

            # Summary
            self._print_summary()

    def _run_entry_combinations_phase(self, max_backtests: Optional[int] = None):
        """Run Phase 1: Test all entry combinations"""
        self.logger.info("[PHASE 1] Entry Combinations (3,888 tests)")

        # Generate all combinations if not done
        if self.state.total_iterations == 0:
            all_combinations = self.generate_entry_combinations()
            self.state.total_iterations = len(all_combinations)
            self.save_state()
        else:
            all_combinations = self.generate_entry_combinations()

        # Filter out completed
        remaining = [
            c for c in all_combinations
            if self.config_hash(c) not in self.state.completed_combinations
        ]

        self.logger.info(f"[PROGRESS] {len(remaining)} remaining out of {len(all_combinations)}")

        # Test each combination
        for idx, combo in enumerate(remaining):
            combo_hash = self.config_hash(combo)

            # Run backtest
            result = self._run_single_backtest(combo)

            # Update state
            self.state.completed_combinations.append(combo_hash)
            self.state.current_iteration = len(self.state.completed_combinations)
            self.state.total_backtests_run += 1
            self.state.successful_backtests += 1 if result else 0

            if result and result.get('score', -np.inf) > self.state.current_best_score:
                self.state.current_best_score = result['score']
                self.state.current_best_config = combo.copy()

            # Store result in all_results
            if result:
                self.state.all_results.append(result)

            # Checkpoint
            self.backtest_counter += 1
            if self.backtest_counter % self.checkpoint_interval == 0:
                self.save_state()
                self._print_progress()
                self._save_top_results()  # Save top results to CSV

            # Check max backtests limit
            if max_backtests and self.state.total_backtests_run >= max_backtests:
                self.logger.info(f"[STOP] Max backtests limit reached: {max_backtests}")
                self._save_top_results(final=True)  # Final save
                return

        # Phase complete
        self.logger.info("[PHASE 1] Complete!")
        self.state.current_phase = 'indicator_parameters'
        self.state.current_level = 1
        self.state.current_iteration = 0
        self.save_state()

    def _run_indicator_parameters_phase(self):
        """Run Phase 2: Optimize indicator parameters"""
        self.logger.info("[PHASE 2] Indicator Parameters")
        # TODO: Implement
        pass

    def _run_grid_configuration_phase(self):
        """Run Phase 3: Grid configuration"""
        self.logger.info("[PHASE 3] Grid Configuration")
        # TODO: Implement
        pass

    def _run_money_management_phase(self):
        """Run Phase 4: Money management"""
        self.logger.info("[PHASE 4] Money Management")
        # TODO: Implement
        pass

    def _run_exit_strategies_phase(self):
        """Run Phase 5: Exit strategies"""
        self.logger.info("[PHASE 5] Exit Strategies")
        # TODO: Implement
        pass

    def _run_single_backtest(self, config: Dict, data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Run single backtest with given config

        Args:
            config: Configuration dict with entry/grid/indicator params
            data: Optional data dict with high/low/close arrays

        Returns:
            Result dict or None if failed
        """
        try:
            # Use market data if available, otherwise use provided data or generate random
            if self.market_data is not None:
                high = self.market_data['high']
                low = self.market_data['low']
                close = self.market_data['close']
            elif data is not None:
                high = data['high']
                low = data['low']
                close = data['close']
            else:
                # Fallback: generate test data
                n = 1000
                close = np.cumsum(np.random.randn(n) * 0.01) + 1.1000
                high = close + np.abs(np.random.randn(n) * 0.001)
                low = close - np.abs(np.random.randn(n) * 0.001)

            # Create EntryConfig from dict
            entry_config = EntryConfig(
                ma_entry=config.get('ma_entry', 0),
                cci_entry=config.get('cci_entry', 0),
                bollinger_entry=config.get('bollinger_entry', 0),
                stoch_entry=config.get('stoch_entry', 0),
                macd_entry=config.get('macd_entry', 0),
                b3_traditional=config.get('b3_traditional', False),
                force_market_cond=config.get('force_market_cond', 3),
                use_any_entry=config.get('use_any_entry', False)
            )

            # Indicator parameters
            indicator_params = {
                'ma_period': config.get('ma_period', 20),
                'ma_distance': config.get('ma_distance', 5.0),
                'cci_period': config.get('cci_period', 14),
                'boll_period': config.get('boll_period', 15),
                'boll_deviation': config.get('boll_deviation', 2.0),
                'boll_distance': config.get('boll_distance', 13.0),
                'stoch_k': config.get('stoch_k', 10),
                'stoch_d': config.get('stoch_d', 2),
                'stoch_slowing': config.get('stoch_slowing', 2),
                'stoch_zone': config.get('stoch_zone', 20.0),
                'macd_fast': config.get('macd_fast', 12),
                'macd_slow': config.get('macd_slow', 26),
                'macd_signal': config.get('macd_signal', 9)
            }

            # Grid configuration
            grid_config = GridConfig(
                grid_set_01=config.get('grid_set_01', 25.0),
                pip_value=0.0001,
                auto_cal=config.get('auto_cal', False),
                gaf=config.get('gaf', 1.0),
                smart_grid=config.get('smart_grid', False),
                entry_delay=config.get('entry_delay', 0),
                lot_multiplier=config.get('lot_multiplier', 2.0),
                base_lot=config.get('base_lot', 0.01),
                max_trades=config.get('max_trades', 10),
                take_profit=config.get('take_profit', 50.0),
                tp_in_money=config.get('tp_in_money', False)
            )

            # Backtest configuration
            backtest_config = BacktestConfig(
                entry_config=entry_config,
                indicator_params=indicator_params,
                grid_config=grid_config,
                initial_deposit=1000.0,
                max_drawdown_percent=30.0,
                max_total_trades=20,
                max_lot_total=1.0
            )

            # Run backtest
            engine = BlessingBacktestEngine(
                config=backtest_config,
                point_value=10.0,
                commission_per_lot=0.0,
                verbose=False
            )

            result = engine.run_backtest(high, low, close)

            # Return standardized result
            return {
                'config': config,
                'score': result.net_profit,  # Use net profit as score
                'net_profit': result.net_profit,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'profit_factor': result.profit_factor,
                'max_drawdown_percent': result.max_drawdown_percent,
                'sharpe_ratio': result.sharpe_ratio,
                'final_balance': result.final_balance
            }

        except Exception as e:
            self.logger.error(f"[ERROR] Backtest failed: {e}")
            self.state.failed_backtests += 1
            return None

    def _save_top_results(self, final: bool = False, top_n: int = 100):
        """
        Save top N results to CSV

        Args:
            final: If True, this is final save
            top_n: Number of top results to save
        """
        if not self.state.all_results:
            return

        # Sort by score (net profit)
        sorted_results = sorted(
            self.state.all_results,
            key=lambda x: x.get('score', -np.inf),
            reverse=True
        )

        # Take top N
        top_results = sorted_results[:top_n]

        # Create DataFrame
        rows = []
        for rank, result in enumerate(top_results, 1):
            config = result['config']
            row = {
                'rank': rank,
                'score': result['score'],
                'net_profit': result['net_profit'],
                'win_rate': result['win_rate'],
                'total_trades': result['total_trades'],
                'profit_factor': result['profit_factor'],
                'max_drawdown_pct': result['max_drawdown_percent'],
                'sharpe_ratio': result['sharpe_ratio'],
                'final_balance': result['final_balance'],
                # Entry config
                'ma_entry': config.get('ma_entry', 0),
                'cci_entry': config.get('cci_entry', 0),
                'bollinger_entry': config.get('bollinger_entry', 0),
                'stoch_entry': config.get('stoch_entry', 0),
                'macd_entry': config.get('macd_entry', 0),
                'b3_traditional': config.get('b3_traditional', False),
                'force_market_cond': config.get('force_market_cond', 3),
                'use_any_entry': config.get('use_any_entry', False),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Save to CSV
        csv_filename = f"phase_{self.state.current_level}_top_{top_n}"
        if final:
            csv_filename += "_FINAL"
        csv_filename += ".csv"

        csv_path = self.results_dir / csv_filename
        df.to_csv(csv_path, index=False)

        if final or self.verbose:
            self.logger.info(f"[RESULTS] Top {len(top_results)} saved to {csv_path}")

    def _print_progress(self):
        """Print current progress"""
        if not self.verbose:
            return

        pct = (self.state.current_iteration / self.state.total_iterations * 100) if self.state.total_iterations > 0 else 0

        print(f"\n[PROGRESS] Phase: {self.state.current_phase}")
        print(f"  Iteration: {self.state.current_iteration}/{self.state.total_iterations} ({pct:.1f}%)")
        print(f"  Best score: {self.state.current_best_score:.2f}")
        print(f"  Total backtests: {self.state.total_backtests_run}")
        print(f"  Runtime: {self.state.total_runtime_seconds/3600:.1f}h")

    def _print_summary(self):
        """Print final summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info("OPTIMIZATION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Phase: {self.state.current_phase}")
        self.logger.info(f"Progress: {self.state.current_iteration}/{self.state.total_iterations}")
        self.logger.info(f"Total backtests: {self.state.total_backtests_run}")
        self.logger.info(f"Successful: {self.state.successful_backtests}")
        self.logger.info(f"Failed: {self.state.failed_backtests}")
        self.logger.info(f"Runtime: {self.state.total_runtime_seconds/3600:.1f}h")
        self.logger.info(f"\nBest score: {self.state.current_best_score:.2f}")
        self.logger.info(f"Best config: {self.state.current_best_config}")
        self.logger.info("="*60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

    print("="*60)
    print("BLESSING CONTINUOUS OPTIMIZER - TEST")
    print("="*60)

    # Create optimizer
    optimizer = BlessingContinuousOptimizer(
        state_file='data/state/test_optimizer_state.pkl',
        results_dir='data/results/test_continuous',
        checkpoint_interval=5,
        use_gpu=False,  # CPU for test
        verbose=True
    )

    # Run for 30 seconds or 20 backtests (whichever first)
    print("\n[TEST] Running continuous optimizer...")
    print("[TEST] Press Ctrl+C to interrupt and test resume\n")

    try:
        optimizer.run_continuous(
            max_runtime_hours=30/3600,  # 30 seconds
            max_backtests=20
        )
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted - state should be saved")

    print("\n" + "="*60)
    print("[OK] Continuous Optimizer Test Completed")
    print("="*60)
