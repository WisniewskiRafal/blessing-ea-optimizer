# backtesting/parallel_runner.py
# Author: Rafał Wiśniewski | Data & AI Solutions

import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple, Any
from datetime import datetime
import logging
import time
from functools import partial
import queue

from optimization.metrics import BacktestResults, PerformanceMetrics

# Import GPU engine if available
try:
    from backtesting.gpu_backtest_engine import GPUBacktestEngine
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def _backtest_wrapper(backtest_func: Callable, progress_queue: Optional[Queue], config: Dict) -> BacktestResults:
    """
    Module-level wrapper function for pickling compatibility

    Args:
        backtest_func: The backtest function to execute
        progress_queue: Queue for progress updates (can be None)
        config: Configuration dict for backtest

    Returns:
        BacktestResults
    """
    result = backtest_func(config)
    if progress_queue:
        try:
            progress_queue.put(1, block=False)
        except:
            pass
    return result


class ParallelBacktestRunner:
    """
    Równoległe uruchamianie backtestów
    Obsługa CPU + GPU acceleration
    """
    
    def __init__(self,
                 backtest_func: Callable,
                 n_workers: Optional[int] = None,
                 use_gpu: bool = False,
                 gpu_batch_size: int = 4,
                 verbose: bool = True):
        """
        Args:
            backtest_func: Function(config) -> BacktestResults
            n_workers: Number of parallel workers (None = auto)
            use_gpu: Use GPU acceleration if available
            gpu_batch_size: Batch size for GPU processing
            verbose: Print progress
        """
        self.backtest_func = backtest_func
        self.n_workers = n_workers or max(1, mp.cpu_count() - 2)
        self.use_gpu = use_gpu
        self.gpu_batch_size = gpu_batch_size
        self.verbose = verbose
        
        self.logger = logging.getLogger(__name__)

        # GPU setup
        self.gpu_engine = None
        self.gpu_available = False

        if use_gpu and GPU_AVAILABLE:
            try:
                self.gpu_engine = GPUBacktestEngine(use_mixed_precision=True)
                self.gpu_available = self.gpu_engine.device.type == 'cuda'

                if self.gpu_available:
                    self.logger.info("[OK] GPU acceleration enabled (PyTorch)")
                else:
                    self.logger.warning("[!] GPU not available, using CPU fallback")
                    self.use_gpu = False
            except Exception as e:
                self.logger.warning(f"[!] Failed to initialize GPU engine: {e}")
                self.use_gpu = False
        elif use_gpu and not GPU_AVAILABLE:
            self.logger.warning("[!] GPU engine not available, using CPU only")
            self.use_gpu = False
    
    def run_parallel(self, 
                    configs: List[Dict],
                    show_progress: bool = True) -> List[BacktestResults]:
        """
        Uruchom backtesty równolegle
        
        Args:
            configs: Lista konfiguracji do przetestowania
            show_progress: Pokazuj progress bar
        
        Returns:
            Lista BacktestResults
        """
        n_configs = len(configs)
        
        self.logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║     PARALLEL BACKTEST RUNNER                             ║
║     Rafał Wiśniewski | Data & AI Solutions              ║
╚══════════════════════════════════════════════════════════╝

Configurations:   {n_configs}
Workers:          {self.n_workers}
GPU:              {'Enabled' if self.gpu_available else 'Disabled'}
Batch size:       {self.gpu_batch_size if self.gpu_available else 'N/A'}

Starting backtests...
        """)
        
        start_time = time.time()
        
        # Progress tracking
        if show_progress:
            manager = Manager()
            progress_queue = manager.Queue()
            
            # Start progress monitor
            progress_process = mp.Process(
                target=self._progress_monitor,
                args=(progress_queue, n_configs)
            )
            progress_process.start()
        else:
            progress_queue = None
        
        # Run backtests
        if self.gpu_available and n_configs >= self.gpu_batch_size:
            results = self._run_gpu_batched(configs, progress_queue)
        else:
            results = self._run_cpu_parallel(configs, progress_queue)
        
        # Stop progress monitor
        if show_progress:
            progress_queue.put('DONE')
            progress_process.join()
        
        elapsed = time.time() - start_time
        
        self.logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║     BACKTESTS COMPLETED                                  ║
╚══════════════════════════════════════════════════════════╝

Total time:       {elapsed:.1f}s
Time per test:    {elapsed/n_configs:.2f}s
Throughput:       {n_configs/elapsed:.1f} tests/s
        """)
        
        return results
    
    def _run_cpu_parallel(self,
                         configs: List[Dict],
                         progress_queue: Optional[Queue] = None) -> List[BacktestResults]:
        """Uruchom na CPU z multiprocessing"""

        # Use functools.partial to make it picklable
        from functools import partial

        # Wrapper for progress tracking (now at module level via partial)
        wrapper_func = partial(_backtest_wrapper, self.backtest_func, progress_queue)

        # Parallel execution
        with Pool(self.n_workers) as pool:
            results = pool.map(wrapper_func, configs)

        return results
    
    def _run_gpu_batched(self,
                        configs: List[Dict],
                        progress_queue: Optional[Queue] = None) -> List[BacktestResults]:
        """
        Uruchom w batch'ach na GPU
        
        UWAGA: Wymaga vectorized backtest function
        """
        self.logger.info(f"🎮 Running GPU-accelerated backtests")
        
        results = []
        
        # Process in batches
        for i in range(0, len(configs), self.gpu_batch_size):
            batch = configs[i:i + self.gpu_batch_size]
            
            # Run batch on GPU
            batch_results = self._process_gpu_batch(batch)
            results.extend(batch_results)
            
            # Update progress
            if progress_queue:
                for _ in range(len(batch)):
                    try:
                        progress_queue.put(1, block=False)
                    except:
                        pass
        
        return results
    
    def _process_gpu_batch(self, configs: List[Dict], prices: np.ndarray = None) -> List[BacktestResults]:
        """
        Process batch na GPU using PyTorch

        Args:
            configs: List of configuration dictionaries
            prices: OHLC price data [n_bars, 4] (open, high, low, close)
                   If None, will try to load from config

        Returns:
            List of BacktestResults
        """
        if not self.gpu_available or self.gpu_engine is None:
            self.logger.warning("[!] GPU not available, using CPU fallback")
            results = []
            for config in configs:
                result = self.backtest_func(config)
                results.append(result)
            return results

        # Use GPU engine for batch processing
        try:
            results = self.gpu_engine.backtest_batch(prices, configs)
            return results
        except Exception as e:
            self.logger.error(f"[ERROR] GPU batch processing failed: {e}")
            self.logger.info("[!] Falling back to CPU processing")

            # Fallback to CPU
            results = []
            for config in configs:
                result = self.backtest_func(config)
                results.append(result)

            return results
    
    def _progress_monitor(self, progress_queue: Queue, total: int):
        """Monitor postępu w osobnym procesie"""
        completed = 0
        start_time = time.time()
        
        while completed < total:
            try:
                msg = progress_queue.get(timeout=1)
                
                if msg == 'DONE':
                    break
                
                completed += msg
                elapsed = time.time() - start_time
                
                if elapsed > 0:
                    rate = completed / elapsed
                    eta = (total - completed) / rate if rate > 0 else 0
                    
                    # Progress bar
                    pct = (completed / total) * 100
                    bar_length = 40
                    filled = int(bar_length * completed / total)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    
                    print(f"\r🔄 Progress: [{bar}] {pct:5.1f}% | "
                          f"{completed}/{total} | "
                          f"{rate:.1f} tests/s | "
                          f"ETA: {eta:.0f}s",
                          end='', flush=True)
            
            except queue.Empty:
                continue
        
        print()  # New line after progress bar
    
    def run_walk_forward(self,
                        configs: List[Dict],
                        data: pd.DataFrame,
                        train_size: float = 0.7,
                        n_splits: int = 5,
                        step_size: Optional[int] = None) -> Dict[str, List[BacktestResults]]:
        """
        Walk-forward validation
        
        Args:
            configs: Lista konfiguracji
            data: Dane OHLC
            train_size: Rozmiar okna treningowego (0-1)
            n_splits: Liczba podziałów
            step_size: Krok przesuwania okna (None = non-overlapping)
        
        Returns:
            {
                'train_results': [...],
                'test_results': [...]
            }
        """
        self.logger.info(f"🔄 Walk-forward validation: {n_splits} splits")
        
        # Calculate split points
        total_bars = len(data)
        train_bars = int(total_bars * train_size)
        
        if step_size is None:
            test_bars = total_bars - train_bars
            step_size = test_bars
        
        train_results = []
        test_results = []
        
        for split_idx in range(n_splits):
            start_idx = split_idx * step_size
            train_end_idx = start_idx + train_bars
            test_end_idx = min(train_end_idx + step_size, total_bars)
            
            if train_end_idx >= total_bars:
                break
            
            self.logger.info(f"  Split {split_idx + 1}/{n_splits}: "
                           f"Train[{start_idx}:{train_end_idx}] "
                           f"Test[{train_end_idx}:{test_end_idx}]")
            
            # TODO: Run backtests on split data
            # Requires backtest_func to accept data parameter
            
        return {
            'train_results': train_results,
            'test_results': test_results
        }
    
    def run_monte_carlo(self,
                       best_config: Dict,
                       n_simulations: int = 1000,
                       vary_params: Optional[List[str]] = None,
                       variance_pct: float = 0.1) -> List[BacktestResults]:
        """
        Monte Carlo simulation
        
        Args:
            best_config: Najlepsza konfiguracja
            n_simulations: Liczba symulacji
            vary_params: Parametry do wariacji (None = wszystkie numeryczne)
            variance_pct: Procent wariancji (±)
        
        Returns:
            Lista wyników symulacji
        """
        self.logger.info(f"🎲 Monte Carlo: {n_simulations} simulations")
        
        # Generate variations
        configs = []
        
        for _ in range(n_simulations):
            config = best_config.copy()
            
            # Vary parameters
            for key, value in config.items():
                if vary_params and key not in vary_params:
                    continue
                
                if isinstance(value, (int, float)):
                    # Add random variation
                    delta = value * variance_pct * np.random.uniform(-1, 1)
                    
                    if isinstance(value, int):
                        config[key] = int(value + delta)
                    else:
                        config[key] = value + delta
            
            configs.append(config)
        
        # Run simulations
        results = self.run_parallel(configs, show_progress=True)
        
        # Statistics
        metrics = PerformanceMetrics()
        scores = [metrics.calculate_custom_score(r) for r in results]
        
        self.logger.info(f"""
Monte Carlo Results:
  Mean Score:  {np.mean(scores):.3f}
  Std Dev:     {np.std(scores):.3f}
  Min Score:   {np.min(scores):.3f}
  Max Score:   {np.max(scores):.3f}
  95% CI:      [{np.percentile(scores, 2.5):.3f}, {np.percentile(scores, 97.5):.3f}]
        """)
        
        return results
    
    def compare_configs(self,
                       configs: List[Dict],
                       config_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Porównaj wiele konfiguracji
        
        Returns:
            DataFrame z metrykami dla każdej konfiguracji
        """
        if config_names is None:
            config_names = [f"Config_{i+1}" for i in range(len(configs))]
        
        # Run backtests
        results = self.run_parallel(configs, show_progress=True)
        
        # Calculate metrics
        metrics_calc = PerformanceMetrics()
        
        comparison = []
        for name, result in zip(config_names, results):
            metrics = metrics_calc.calculate_all_metrics(result)
            metrics['config_name'] = name
            comparison.append(metrics)
        
        df = pd.DataFrame(comparison)
        
        # Reorder columns
        cols = ['config_name'] + [c for c in df.columns if c != 'config_name']
        df = df[cols]
        
        return df
    
    def benchmark_performance(self, 
                            test_config: Dict,
                            n_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark wydajności systemu
        
        Returns:
            {
                'avg_time': seconds,
                'std_time': seconds,
                'throughput': tests/second
            }
        """
        self.logger.info(f"⏱️ Benchmarking: {n_runs} runs")
        
        configs = [test_config] * n_runs
        
        start = time.time()
        results = self.run_parallel(configs, show_progress=False)
        elapsed = time.time() - start
        
        avg_time = elapsed / n_runs
        throughput = n_runs / elapsed
        
        benchmark = {
            'total_time': elapsed,
            'avg_time': avg_time,
            'throughput': throughput,
            'n_workers': self.n_workers,
            'gpu_enabled': self.gpu_available
        }
        
        self.logger.info(f"""
Benchmark Results:
  Total Time:   {elapsed:.2f}s
  Avg Time:     {avg_time:.3f}s
  Throughput:   {throughput:.1f} tests/s
  Workers:      {self.n_workers}
  GPU:          {'Yes' if self.gpu_available else 'No'}
        """)
        
        return benchmark


class ProgressTracker:
    """Helper do trackowania postępu"""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress"""
        self.completed += n
        self.print_progress()
    
    def print_progress(self):
        """Print progress bar"""
        elapsed = time.time() - self.start_time
        pct = (self.completed / self.total) * 100
        
        if elapsed > 0:
            rate = self.completed / elapsed
            eta = (self.total - self.completed) / rate if rate > 0 else 0
        else:
            rate = 0
            eta = 0
        
        bar_length = 40
        filled = int(bar_length * self.completed / self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r🔄 [{bar}] {pct:5.1f}% | "
              f"{self.completed}/{self.total} | "
              f"{rate:.1f}/s | "
              f"ETA: {eta:.0f}s",
              end='', flush=True)
        
        if self.completed >= self.total:
            print()


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    def dummy_backtest(config: Dict) -> BacktestResults:
        """Fake backtest"""
        import random
        time.sleep(0.1)  # Simulate work
        
        score = random.random()
        equity = pd.Series(10000 + np.cumsum(np.random.randn(100) * 100))
        
        return BacktestResults(
            total_trades=50,
            winning_trades=int(50 * score),
            losing_trades=int(50 * (1 - score)),
            gross_profit=10000 * score,
            gross_loss=-5000 * (1 - score),
            net_profit=5000 * score,
            max_drawdown=-2000,
            max_drawdown_percent=20,
            initial_deposit=10000,
            final_balance=10000 + 5000 * score,
            equity_curve=equity
        )
    
    # Create test configs
    test_configs = [
        {'MaxTrades': i, 'Multiplier': 1.4}
        for i in range(10, 20)
    ]
    
    # Run parallel
    runner = ParallelBacktestRunner(
        backtest_func=dummy_backtest,
        n_workers=4,
        use_gpu=False
    )
    
    results = runner.run_parallel(test_configs, show_progress=True)
    
    print(f"\n✅ Completed {len(results)} backtests")
    
    # Benchmark
    benchmark = runner.benchmark_performance(test_configs[0], n_runs=5)
    print(f"Benchmark: {benchmark['throughput']:.1f} tests/s")