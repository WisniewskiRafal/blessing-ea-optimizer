# optimization/hierarchical_optimizer.py
# Hierarchical Optimization System
# Author: Rafał Wiśniewski | Data & AI Solutions

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import itertools

from core.data_loader import DataSplit, WalkForwardSplit
from backtesting.gpu_backtest_engine import GPUBacktestEngine
from backtesting.parallel_runner import ParallelBacktestRunner
from optimization.metrics import BacktestResults


@dataclass
class ParameterSpace:
    """Definition of parameter search space"""
    name: str
    type: str  # 'categorical', 'integer', 'float'
    values: Optional[List[Any]] = None  # For categorical/discrete
    min_val: Optional[float] = None     # For continuous
    max_val: Optional[float] = None
    step: Optional[float] = None        # For grid search
    priority: int = 0                   # 0=highest (test first)


@dataclass
class OptimizationLevel:
    """Single level in hierarchical optimization"""
    name: str                           # e.g., 'MA Type', 'Timeframe', 'Parameters'
    parameters: List[ParameterSpace]
    priority: int = 0                   # Order of execution


@dataclass
class OptimizationResult:
    """Result from optimization run"""
    config: Dict[str, Any]
    backtest_result: BacktestResults
    score: float                        # Objective function value
    level_name: str
    timestamp: datetime = field(default_factory=datetime.now)


class HierarchicalOptimizer:
    """
    Hierarchical Optimization System

    Strategy:
    1. Test categorical features first (MA Type: SMA/EMA/WMA)
    2. For each good categorical, test all timeframes
    3. For each good TF, optimize numeric parameters
    4. Early stopping if results below threshold
    """

    def __init__(self,
                 objective: str = 'sharpe',
                 use_gpu: bool = True,
                 n_workers: int = 4,
                 early_stop_threshold: Optional[float] = None,
                 verbose: bool = True):
        """
        Args:
            objective: Optimization objective ('sharpe', 'profit_factor', 'net_profit')
            use_gpu: Use GPU acceleration
            n_workers: Number of CPU workers
            early_stop_threshold: Stop if score below this
            verbose: Print progress
        """
        self.objective = objective
        self.use_gpu = use_gpu
        self.n_workers = n_workers
        self.early_stop_threshold = early_stop_threshold
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)

        # GPU engine
        self.gpu_engine = None
        if use_gpu:
            try:
                self.gpu_engine = GPUBacktestEngine(use_mixed_precision=True)
                if self.gpu_engine.device.type == 'cuda':
                    self.logger.info("[OK] GPU acceleration enabled")
                else:
                    self.logger.warning("[!] GPU not available, using CPU")
            except Exception as e:
                self.logger.warning(f"[!] GPU init failed: {e}")

        # Results storage
        self.all_results: List[OptimizationResult] = []
        self.best_results: Dict[str, OptimizationResult] = {}

    def optimize_hierarchical(self,
                             data_split: DataSplit,
                             optimization_levels: List[OptimizationLevel],
                             base_config: Optional[Dict] = None) -> List[OptimizationResult]:
        """
        Run hierarchical optimization

        Args:
            data_split: Training data split
            optimization_levels: List of optimization levels (ordered by priority)
            base_config: Base configuration (merged with optimized params)

        Returns:
            List of best results per level
        """
        if base_config is None:
            base_config = {}

        self.logger.info(f"Starting hierarchical optimization on {data_split.name}")
        self.logger.info(f"Data: {data_split.n_bars} bars ({data_split.start_date.date()} to {data_split.end_date.date()})")
        self.logger.info(f"Levels: {len(optimization_levels)}")

        # Sort levels by priority
        levels = sorted(optimization_levels, key=lambda x: x.priority)

        # Track best config through levels
        current_best_config = base_config.copy()
        level_results = []

        for level_idx, level in enumerate(levels):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"LEVEL {level_idx+1}/{len(levels)}: {level.name}")
            self.logger.info(f"{'='*60}")

            # Optimize this level
            best_result = self._optimize_level(
                data_split=data_split,
                level=level,
                base_config=current_best_config
            )

            if best_result is None:
                self.logger.warning(f"[!] No valid results for level {level.name}")
                break

            # Update best config with this level's result
            current_best_config.update(best_result.config)
            level_results.append(best_result)

            # Early stopping check
            if self.early_stop_threshold and best_result.score < self.early_stop_threshold:
                self.logger.info(f"[STOP] Score {best_result.score:.3f} < threshold {self.early_stop_threshold}")
                break

            self.logger.info(f"[LEVEL {level_idx+1} BEST] Score: {best_result.score:.3f}")
            self.logger.info(f"Config: {best_result.config}")

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"HIERARCHICAL OPTIMIZATION COMPLETED")
        self.logger.info(f"{'='*60}")

        return level_results

    def _optimize_level(self,
                       data_split: DataSplit,
                       level: OptimizationLevel,
                       base_config: Dict) -> Optional[OptimizationResult]:
        """
        Optimize single level

        Args:
            data_split: Training data
            level: Optimization level definition
            base_config: Base configuration

        Returns:
            Best OptimizationResult for this level
        """
        # Generate configurations for this level
        configs = self._generate_configs(level.parameters, base_config)

        self.logger.info(f"Testing {len(configs)} configurations...")

        # Run backtests
        start_time = time.perf_counter()

        if self.use_gpu and self.gpu_engine:
            # GPU batch processing
            results = self.gpu_engine.backtest_batch(data_split.data, configs)
        else:
            # CPU parallel processing
            from backtesting.simple_backtest import simple_backtest_cpu
            runner = ParallelBacktestRunner(
                backtest_func=simple_backtest_cpu,
                n_workers=self.n_workers,
                use_gpu=False
            )
            # Add OHLC to configs
            for cfg in configs:
                cfg['_ohlc_data'] = data_split.data
            results = runner.run_parallel(configs, show_progress=self.verbose)

        elapsed = time.perf_counter() - start_time

        # Convert to OptimizationResults
        opt_results = []
        for cfg, res in zip(configs, results):
            score = self._calculate_score(res)
            opt_result = OptimizationResult(
                config=cfg,
                backtest_result=res,
                score=score,
                level_name=level.name
            )
            opt_results.append(opt_result)
            self.all_results.append(opt_result)

        # Find best
        if not opt_results:
            return None

        best = max(opt_results, key=lambda x: x.score)

        # Store in best_results
        self.best_results[level.name] = best

        throughput = len(configs) / elapsed
        self.logger.info(f"Completed in {elapsed:.2f}s ({throughput:.1f} configs/s)")

        # Show top 5
        top_5 = sorted(opt_results, key=lambda x: x.score, reverse=True)[:5]
        self.logger.info(f"\nTop 5 results:")
        for i, result in enumerate(top_5):
            self.logger.info(f"  {i+1}. Score: {result.score:.3f} | Config: {result.config}")

        return best

    def _generate_configs(self,
                         parameters: List[ParameterSpace],
                         base_config: Dict) -> List[Dict]:
        """
        Generate all configurations for parameter space

        Args:
            parameters: List of parameters to vary
            base_config: Base configuration

        Returns:
            List of configuration dicts
        """
        # Extract parameter ranges
        param_values = {}

        for param in parameters:
            if param.type == 'categorical':
                param_values[param.name] = param.values
            elif param.type == 'integer':
                if param.step:
                    param_values[param.name] = list(range(
                        int(param.min_val),
                        int(param.max_val) + 1,
                        int(param.step)
                    ))
                else:
                    param_values[param.name] = list(range(
                        int(param.min_val),
                        int(param.max_val) + 1
                    ))
            elif param.type == 'float':
                if param.step:
                    n_steps = int((param.max_val - param.min_val) / param.step) + 1
                    param_values[param.name] = [
                        param.min_val + i * param.step
                        for i in range(n_steps)
                    ]
                else:
                    # Default: 10 steps
                    param_values[param.name] = list(np.linspace(
                        param.min_val,
                        param.max_val,
                        10
                    ))

        # Generate all combinations
        param_names = list(param_values.keys())
        param_lists = [param_values[name] for name in param_names]

        configs = []
        for combination in itertools.product(*param_lists):
            config = base_config.copy()
            for name, value in zip(param_names, combination):
                config[name] = value
            configs.append(config)

        return configs

    def _calculate_score(self, result: BacktestResults) -> float:
        """
        Calculate objective score from backtest results

        Args:
            result: BacktestResults

        Returns:
            Score (higher is better)
        """
        if self.objective == 'sharpe':
            return result.sharpe_ratio if hasattr(result, 'sharpe_ratio') else 0.0
        elif self.objective == 'profit_factor':
            return result.profit_factor if hasattr(result, 'profit_factor') else 0.0
        elif self.objective == 'net_profit':
            return result.net_profit
        elif self.objective == 'win_rate':
            return result.win_rate
        else:
            # Default: Sharpe ratio
            return result.sharpe_ratio if hasattr(result, 'sharpe_ratio') else 0.0

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all optimization results

        Returns:
            DataFrame with results
        """
        if not self.all_results:
            return pd.DataFrame()

        data = []
        for result in self.all_results:
            row = {
                'level': result.level_name,
                'score': result.score,
                'net_profit': result.backtest_result.net_profit,
                'win_rate': result.backtest_result.win_rate,
                'total_trades': result.backtest_result.total_trades,
                'timestamp': result.timestamp
            }
            row.update(result.config)
            data.append(row)

        return pd.DataFrame(data)

    def save_results(self, output_path: str):
        """
        Save optimization results to CSV

        Args:
            output_path: Path to save results
        """
        df = self.get_summary()
        df.to_csv(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Test HierarchicalOptimizer
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

    print("="*60)
    print("HIERARCHICAL OPTIMIZER TEST")
    print("="*60)

    # Load test data
    from core.data_loader import DataLoader

    loader = DataLoader()
    df, ohlc = loader.load_eurusd_m1(max_bars=10000)  # Small test

    # Create test split
    test_split = DataSplit(
        name='test',
        data=ohlc,
        start_date=df['Timestamp'].min(),
        end_date=df['Timestamp'].max(),
        n_bars=len(ohlc),
        timeframe='M1'
    )

    # Define optimization levels
    level1 = OptimizationLevel(
        name='MA Type',
        parameters=[
            ParameterSpace(name='ma_type', type='categorical', values=['SMA', 'EMA'], priority=0)
        ],
        priority=0
    )

    level2 = OptimizationLevel(
        name='MA Period',
        parameters=[
            ParameterSpace(name='ma_period', type='integer', min_val=10, max_val=50, step=10, priority=1)
        ],
        priority=1
    )

    # Create optimizer
    optimizer = HierarchicalOptimizer(
        objective='net_profit',
        use_gpu=True,
        n_workers=4,
        verbose=True
    )

    # Run optimization
    print("\n[TEST] Running hierarchical optimization...")

    base_config = {'initial_balance': 10000.0}

    results = optimizer.optimize_hierarchical(
        data_split=test_split,
        optimization_levels=[level1, level2],
        base_config=base_config
    )

    # Show results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)

    for i, result in enumerate(results):
        print(f"\nLevel {i+1}: {result.level_name}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Config: {result.config}")
        print(f"  Net Profit: ${result.backtest_result.net_profit:.2f}")
        print(f"  Win Rate: {result.backtest_result.win_rate*100:.1f}%")
        print(f"  Trades: {result.backtest_result.total_trades}")

    # Summary
    summary = optimizer.get_summary()
    print(f"\n[SUMMARY] Total configs tested: {len(summary)}")
    print(summary.head(10))

    print("\n" + "="*60)
    print("[OK] Hierarchical Optimizer Test Completed")
    print("="*60)
