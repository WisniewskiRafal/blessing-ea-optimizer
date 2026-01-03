# optimization/walk_forward_analyzer.py
# Walk-Forward Analysis System
# Author: Rafał Wiśniewski | Data & AI Solutions

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
import json

from core.data_loader import DataLoader, WalkForwardSplit
from optimization.hierarchical_optimizer import HierarchicalOptimizer, OptimizationLevel, OptimizationResult
from backtesting.gpu_backtest_engine import GPUBacktestEngine
from backtesting.simple_backtest import simple_backtest_cpu
from optimization.metrics import BacktestResults


@dataclass
class WalkForwardResult:
    """Result from single walk-forward period"""
    period_name: str
    train_result: OptimizationResult
    test_result: BacktestResults
    train_score: float
    test_score: float
    train_bars: int
    test_bars: int
    config: Dict
    timestamp: datetime = field(default_factory=datetime.now)


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis System

    Implements:
    1. Train optimizer on Period N
    2. Test best config on Period N+1
    3. Aggregate results across all periods
    4. Calculate robustness metrics
    """

    def __init__(self,
                 optimizer: HierarchicalOptimizer,
                 objective: str = 'sharpe',
                 use_gpu: bool = True,
                 verbose: bool = True):
        """
        Args:
            optimizer: HierarchicalOptimizer instance
            objective: Scoring objective
            use_gpu: Use GPU for testing
            verbose: Print progress
        """
        self.optimizer = optimizer
        self.objective = objective
        self.use_gpu = use_gpu
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)

        # GPU engine for testing
        self.gpu_engine = None
        if use_gpu:
            try:
                self.gpu_engine = GPUBacktestEngine(use_mixed_precision=True)
            except Exception as e:
                self.logger.warning(f"[!] GPU init failed: {e}")

        # Results storage
        self.wf_results: List[WalkForwardResult] = []

    def run_walk_forward(self,
                        wf_splits: List[WalkForwardSplit],
                        optimization_levels: List[OptimizationLevel],
                        base_config: Optional[Dict] = None) -> List[WalkForwardResult]:
        """
        Run walk-forward analysis

        Args:
            wf_splits: List of walk-forward splits (from DataLoader)
            optimization_levels: Hierarchical optimization levels
            base_config: Base configuration

        Returns:
            List of WalkForwardResult
        """
        if base_config is None:
            base_config = {}

        self.logger.info("="*60)
        self.logger.info("WALK-FORWARD ANALYSIS")
        self.logger.info("="*60)
        self.logger.info(f"Periods: {len(wf_splits)}")
        self.logger.info(f"Objective: {self.objective}")

        results = []

        for period_idx, wf_split in enumerate(wf_splits):
            self.logger.info(f"\n{'#'*60}")
            self.logger.info(f"PERIOD {period_idx+1}/{len(wf_splits)}: {wf_split.period_name}")
            self.logger.info(f"{'#'*60}")

            # TRAIN phase
            self.logger.info(f"\n[TRAIN] {wf_split.train.name}")
            self.logger.info(f"  Bars: {wf_split.train.n_bars}")
            self.logger.info(f"  Date: {wf_split.train.start_date.date()} to {wf_split.train.end_date.date()}")

            train_results = self.optimizer.optimize_hierarchical(
                data_split=wf_split.train,
                optimization_levels=optimization_levels,
                base_config=base_config
            )

            if not train_results:
                self.logger.warning(f"[!] No results from training period {wf_split.period_name}")
                continue

            # Best config from training
            best_train_result = train_results[-1]  # Last level has complete config
            best_config = best_train_result.config
            train_score = best_train_result.score

            self.logger.info(f"\n[TRAIN BEST] Score: {train_score:.3f}")
            self.logger.info(f"Config: {best_config}")

            # TEST phase
            self.logger.info(f"\n[TEST] {wf_split.test.name}")
            self.logger.info(f"  Bars: {wf_split.test.n_bars}")
            self.logger.info(f"  Date: {wf_split.test.start_date.date()} to {wf_split.test.end_date.date()}")

            # Run backtest on test data
            test_result = self._run_test(wf_split.test.data, best_config)

            # Calculate test score
            test_score = self._calculate_score(test_result)

            self.logger.info(f"[TEST RESULT] Score: {test_score:.3f}")
            self.logger.info(f"  Net Profit: ${test_result.net_profit:.2f}")
            self.logger.info(f"  Win Rate: {test_result.win_rate*100:.1f}%")
            self.logger.info(f"  Trades: {test_result.total_trades}")
            self.logger.info(f"  Max DD: {test_result.max_drawdown_percent:.2f}%")

            # Store result
            wf_result = WalkForwardResult(
                period_name=wf_split.period_name,
                train_result=best_train_result,
                test_result=test_result,
                train_score=train_score,
                test_score=test_score,
                train_bars=wf_split.train.n_bars,
                test_bars=wf_split.test.n_bars,
                config=best_config
            )
            results.append(wf_result)
            self.wf_results.append(wf_result)

        # Final summary
        self._print_summary(results)

        return results

    def _run_test(self, ohlc: np.ndarray, config: Dict) -> BacktestResults:
        """
        Run backtest on test data

        Args:
            ohlc: OHLC data
            config: Configuration

        Returns:
            BacktestResults
        """
        if self.use_gpu and self.gpu_engine:
            # GPU single test
            result = self.gpu_engine.backtest_single(ohlc, config)
        else:
            # CPU test
            config['_ohlc_data'] = ohlc
            result = simple_backtest_cpu(config)

        return result

    def _calculate_score(self, result: BacktestResults) -> float:
        """Calculate score from backtest result"""
        if self.objective == 'sharpe':
            return result.sharpe_ratio if hasattr(result, 'sharpe_ratio') else 0.0
        elif self.objective == 'profit_factor':
            return result.profit_factor if hasattr(result, 'profit_factor') else 0.0
        elif self.objective == 'net_profit':
            return result.net_profit
        elif self.objective == 'win_rate':
            return result.win_rate
        else:
            return result.sharpe_ratio if hasattr(result, 'sharpe_ratio') else 0.0

    def _print_summary(self, results: List[WalkForwardResult]):
        """Print walk-forward summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info("WALK-FORWARD SUMMARY")
        self.logger.info("="*60)

        if not results:
            self.logger.warning("[!] No results to summarize")
            return

        # Aggregate metrics
        total_train_score = sum(r.train_score for r in results)
        total_test_score = sum(r.test_score for r in results)
        avg_train_score = total_train_score / len(results)
        avg_test_score = total_test_score / len(results)

        total_test_profit = sum(r.test_result.net_profit for r in results)
        total_test_trades = sum(r.test_result.total_trades for r in results)
        avg_win_rate = np.mean([r.test_result.win_rate for r in results])

        # Robustness: Train/Test score ratio
        score_ratios = [r.test_score / r.train_score if r.train_score != 0 else 0 for r in results]
        avg_ratio = np.mean(score_ratios)

        self.logger.info(f"\nPeriods analyzed: {len(results)}")
        self.logger.info(f"\nTRAIN Performance:")
        self.logger.info(f"  Average score: {avg_train_score:.3f}")
        self.logger.info(f"\nTEST Performance:")
        self.logger.info(f"  Average score: {avg_test_score:.3f}")
        self.logger.info(f"  Total profit: ${total_test_profit:.2f}")
        self.logger.info(f"  Total trades: {total_test_trades}")
        self.logger.info(f"  Average win rate: {avg_win_rate*100:.1f}%")
        self.logger.info(f"\nRobustness:")
        self.logger.info(f"  Train/Test ratio: {avg_ratio:.2%}")
        self.logger.info(f"  (100% = perfect match, >100% = better on test)")

        # Period breakdown
        self.logger.info(f"\nPeriod Breakdown:")
        for i, r in enumerate(results):
            self.logger.info(f"  {i+1}. {r.period_name}")
            self.logger.info(f"     Train: {r.train_score:.3f} | Test: {r.test_score:.3f} | Profit: ${r.test_result.net_profit:.2f}")

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Get summary as DataFrame

        Returns:
            DataFrame with walk-forward results
        """
        if not self.wf_results:
            return pd.DataFrame()

        data = []
        for r in self.wf_results:
            row = {
                'period': r.period_name,
                'train_score': r.train_score,
                'test_score': r.test_score,
                'test_profit': r.test_result.net_profit,
                'test_win_rate': r.test_result.win_rate,
                'test_trades': r.test_result.total_trades,
                'test_max_dd': r.test_result.max_drawdown_percent,
                'train_bars': r.train_bars,
                'test_bars': r.test_bars,
            }
            row.update({f'cfg_{k}': v for k, v in r.config.items()})
            data.append(row)

        return pd.DataFrame(data)

    def save_results(self, output_dir: str):
        """
        Save walk-forward results

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        df = self.get_summary_dataframe()
        csv_path = output_path / 'walk_forward_summary.csv'
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Summary saved to {csv_path}")

        # Detailed JSON
        json_data = []
        for r in self.wf_results:
            json_data.append({
                'period': r.period_name,
                'train': {
                    'score': float(r.train_score),
                    'bars': int(r.train_bars),
                    'config': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                              for k, v in r.config.items()}
                },
                'test': {
                    'score': float(r.test_score),
                    'net_profit': float(r.test_result.net_profit),
                    'win_rate': float(r.test_result.win_rate),
                    'total_trades': int(r.test_result.total_trades),
                    'max_drawdown_percent': float(r.test_result.max_drawdown_percent),
                    'bars': int(r.test_bars)
                },
                'timestamp': r.timestamp.isoformat()
            })

        json_path = output_path / 'walk_forward_detailed.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        self.logger.info(f"Detailed results saved to {json_path}")


if __name__ == "__main__":
    # Test WalkForwardAnalyzer
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

    print("="*60)
    print("WALK-FORWARD ANALYZER TEST")
    print("="*60)

    # Load data
    from core.data_loader import DataLoader

    loader = DataLoader()
    df, ohlc = loader.load_eurusd_m1(max_bars=50000)  # 50k bars for faster test

    # Create walk-forward splits (simulate Q1->Q2, Q1+Q2->Q3)
    # For test, just split in half twice
    n = len(ohlc)
    q1_end = n // 4
    q2_end = n // 2
    q3_end = 3 * n // 4

    from core.data_loader import DataSplit, WalkForwardSplit

    # Period 1: Q1 -> Q2
    wf1 = WalkForwardSplit(
        period_name='Period_1',
        train=DataSplit('Q1_train', ohlc[:q1_end], df['Timestamp'].min(),
                       df.iloc[q1_end-1]['Timestamp'], q1_end, 'M1'),
        test=DataSplit('Q2_test', ohlc[q1_end:q2_end], df.iloc[q1_end]['Timestamp'],
                      df.iloc[q2_end-1]['Timestamp'], q2_end-q1_end, 'M1')
    )

    # Period 2: Q1+Q2 -> Q3
    wf2 = WalkForwardSplit(
        period_name='Period_2',
        train=DataSplit('Q1Q2_train', ohlc[:q2_end], df['Timestamp'].min(),
                       df.iloc[q2_end-1]['Timestamp'], q2_end, 'M1'),
        test=DataSplit('Q3_test', ohlc[q2_end:q3_end], df.iloc[q2_end]['Timestamp'],
                      df.iloc[q3_end-1]['Timestamp'], q3_end-q2_end, 'M1')
    )

    wf_splits = [wf1, wf2]

    # Define optimization levels (simple for test)
    from optimization.hierarchical_optimizer import OptimizationLevel, ParameterSpace

    level1 = OptimizationLevel(
        name='MA Type',
        parameters=[ParameterSpace(name='ma_type', type='categorical', values=['SMA', 'EMA'], priority=0)],
        priority=0
    )

    level2 = OptimizationLevel(
        name='MA Period',
        parameters=[ParameterSpace(name='ma_period', type='integer', min_val=10, max_val=30, step=10, priority=1)],
        priority=1
    )

    # Create optimizer
    optimizer = HierarchicalOptimizer(
        objective='net_profit',
        use_gpu=True,
        n_workers=4,
        verbose=False  # Less verbose for WF test
    )

    # Create analyzer
    analyzer = WalkForwardAnalyzer(
        optimizer=optimizer,
        objective='net_profit',
        use_gpu=True,
        verbose=True
    )

    # Run walk-forward
    print("\n[TEST] Running walk-forward analysis...")

    base_config = {'initial_balance': 10000.0}

    results = analyzer.run_walk_forward(
        wf_splits=wf_splits,
        optimization_levels=[level1, level2],
        base_config=base_config
    )

    # Show detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)

    for r in results:
        print(f"\n{r.period_name}:")
        print(f"  Config: {r.config}")
        print(f"  Train score: {r.train_score:.3f}")
        print(f"  Test score: {r.test_score:.3f}")
        print(f"  Test profit: ${r.test_result.net_profit:.2f}")

    # Summary DataFrame
    summary = analyzer.get_summary_dataframe()
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(summary)

    # Save results
    analyzer.save_results('data/results/test_walk_forward')

    print("\n" + "="*60)
    print("[OK] Walk-Forward Analyzer Test Completed")
    print("="*60)
