"""
Bayesian Optimizer - Sequential optimization with smart sampling
Uses Gaussian Process to intelligently explore parameter space
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not installed. Install with: pip install scikit-optimize")

from optimization.sequential_optimizer import SequentialOptimizer


class BayesianOptimizer(SequentialOptimizer):
    """Bayesian Optimization using Gaussian Process"""

    def __init__(self, *args, n_calls: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_calls = n_calls

        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize required. Install with: pip install scikit-optimize")

    def run_phase_3_indicators_bayesian(self, best_phase_2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """FAZA 3: Bayesian optimization of indicator parameters"""
        self.logger.info("="*80)
        self.logger.info("FAZA 3: INDICATOR PARAMETERS (BAYESIAN OPTIMIZATION)")
        self.logger.info("="*80)

        # Define search space
        space = [
            Integer(50, 400, name='ma_period'),
            Real(5.0, 20.0, name='ma_distance'),
            Integer(10, 30, name='cci_period'),
            Integer(10, 30, name='boll_period'),
            Real(5.0, 20.0, name='boll_distance'),
            Real(1.5, 3.0, name='boll_deviation'),
            Integer(20, 30, name='buysell_stoch_zone'),
            Integer(5, 14, name='k_period'),
            Integer(2, 5, name='d_period'),
            Integer(1, 3, name='slowing'),
            Integer(8, 16, name='fast_period'),
            Integer(21, 34, name='slow_period'),
            Integer(7, 12, name='signal_period'),
            Categorical([0, 4, 5], name='macd_price'),
            Integer(10, 21, name='rsi_period'),
        ]

        param_names = [s.name for s in space]
        results_cache = []

        def objective(params):
            """Objective function to minimize (negative profit)"""
            config = {**best_phase_2}

            # Update config with current params
            for name, value in zip(param_names, params):
                config[name] = value

            # Run backtest
            result = self.run_backtest(config)
            results_cache.append(result)

            # Log progress
            if len(results_cache) % 10 == 0:
                self.logger.info(f"Bayesian iteration {len(results_cache)}/{self.n_calls}: "
                               f"score={result['score']:.2f}")

            # Return negative profit (minimize)
            return -result['score']

        # Run Bayesian Optimization
        self.logger.info(f"Running Bayesian Optimization with {self.n_calls} calls...")

        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            random_state=42,
            verbose=False,
            n_jobs=1  # Sequential for GPU consistency
        )

        self.logger.info(f"Bayesian Optimization completed!")
        self.logger.info(f"Best score found: {-result.fun:.2f}")

        # Sort all tested configurations
        results_cache.sort(key=lambda x: x['score'], reverse=True)
        self.save_phase_results("phase_3_indicators_bayesian", results_cache[:self.top_n])

        return results_cache

    def run_phase_5_risk_bayesian(self, best_phase_4: Dict[str, Any]) -> List[Dict[str, Any]]:
        """FAZA 5: Bayesian optimization of risk management"""
        self.logger.info("="*80)
        self.logger.info("FAZA 5: RISK MANAGEMENT (BAYESIAN OPTIMIZATION)")
        self.logger.info("="*80)

        # Define search space
        space = [
            Integer(10, 25, name='max_trades'),
            Integer(10, 15, name='break_even_trade'),
            Real(30.0, 60.0, name='max_dd_percent'),
            Real(3.0, 10.0, name='max_spread'),
            Categorical([True, False], name='use_close_oldest'),
            Integer(5, 10, name='close_trades_level'),
            Integer(3, 5, name='max_close_trades'),
            Categorical([True, False], name='force_close_oldest'),
            Categorical([True, False], name='use_stoploss'),
            Real(20.0, 50.0, name='sl_pips'),
            Real(5.0, 15.0, name='tsl_pips'),
            Real(0.0, 20.0, name='min_tp_pips'),
        ]

        param_names = [s.name for s in space]
        results_cache = []

        def objective(params):
            """Objective function"""
            config = {**best_phase_4}

            for name, value in zip(param_names, params):
                config[name] = value

            result = self.run_backtest(config)
            results_cache.append(result)

            if len(results_cache) % 10 == 0:
                self.logger.info(f"Bayesian iteration {len(results_cache)}/{self.n_calls}: "
                               f"score={result['score']:.2f}")

            return -result['score']

        # Run Bayesian Optimization
        self.logger.info(f"Running Bayesian Optimization with {self.n_calls} calls...")

        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            random_state=42,
            verbose=False,
            n_jobs=1
        )

        self.logger.info(f"Best score found: {-result.fun:.2f}")

        results_cache.sort(key=lambda x: x['score'], reverse=True)
        self.save_phase_results("phase_5_risk_bayesian", results_cache[:self.top_n])

        return results_cache

    def run_optimization(self) -> Dict[str, Any]:
        """Run full Bayesian optimization (all phases)"""
        self.load_market_data()

        # FAZA 1: Use existing results
        self.logger.info("FAZA 1: Loading existing entry logic results...")
        phase_1_csv = Path("data/results/continuous/phase_0_top_100.csv")
        if not phase_1_csv.exists():
            raise FileNotFoundError("FAZA 1 results not found!")

        df_phase_1 = pd.read_csv(phase_1_csv)
        best_phase_1 = df_phase_1.iloc[0].to_dict()

        # FAZA 2: Timeframes (grid search - small enough)
        results_phase_2 = self.run_phase_2_timeframes(best_phase_1)
        best_phase_2 = results_phase_2[0]

        # FAZA 3: Indicators (BAYESIAN)
        results_phase_3 = self.run_phase_3_indicators_bayesian(best_phase_2)
        best_phase_3 = results_phase_3[0]

        # FAZA 4: Grid (grid search - manageable size)
        results_phase_4 = self.run_phase_4_grid(best_phase_3)
        best_phase_4 = results_phase_4[0]

        # FAZA 5: Risk (BAYESIAN)
        results_phase_5 = self.run_phase_5_risk_bayesian(best_phase_4)

        return {
            'top_configs': results_phase_5[:self.top_n],
            'all_results': results_phase_5,
        }

    def run_refinement(self, config: Dict[str, Any], n_calls: int = 100) -> List[Dict[str, Any]]:
        """Refine a single configuration using Bayesian optimization"""
        self.logger.info(f"Refining configuration with score {config.get('score', 0):.2f}")

        # Create search space around current values (±20%)
        space = []
        param_ranges = {
            'ma_period': (0.8, 1.2),
            'cci_period': (0.8, 1.2),
            'boll_period': (0.8, 1.2),
            'lot_multiplier': (0.9, 1.1),
            'max_trades': (0.8, 1.2),
        }

        for param, (low_mult, high_mult) in param_ranges.items():
            if param in config:
                value = config[param]
                low = int(value * low_mult)
                high = int(value * high_mult)
                space.append(Integer(low, high, name=param))

        if not space:
            self.logger.warning("No parameters to refine!")
            return [config]

        param_names = [s.name for s in space]
        results_cache = []

        def objective(params):
            refined_config = {**config}
            for name, value in zip(param_names, params):
                refined_config[name] = value

            result = self.run_backtest(refined_config)
            results_cache.append(result)
            return -result['score']

        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=False
        )

        results_cache.sort(key=lambda x: x['score'], reverse=True)
        return results_cache[:self.top_n]
