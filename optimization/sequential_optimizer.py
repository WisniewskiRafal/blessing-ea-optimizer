"""
Sequential Optimizer - Phase-by-phase optimization
FAZA 1: Entry Logic (DONE)
FAZA 2: Timeframes
FAZA 3: Indicator Parameters
FAZA 4: Grid Settings
FAZA 5: Risk Management
FAZA 6: Advanced Features
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json
import pickle
from datetime import datetime

from strategies.blessing_entry_generator import BlessingEntryGenerator
from core.blessing_backtest_engine import BlessingBacktestEngine
from core.data_loader import DataLoader


class SequentialOptimizer:
    """Sequential phase-by-phase optimizer"""

    # FAZA 2: Timeframes
    TIMEFRAMES = [1, 5, 15, 30, 60, 240, 1440]  # M1, M5, M15, M30, H1, H4, D1

    # FAZA 3: Indicator Parameters
    MA_PERIODS = [50, 100, 150, 200, 400]
    MA_DISTANCES = [5, 10, 15, 20]
    CCI_PERIODS = [10, 14, 20, 30]
    BOLL_PERIODS = [10, 20, 30]
    BOLL_DISTANCES = [5, 10, 15, 20]
    BOLL_DEVIATIONS = [1.5, 2.0, 2.5, 3.0]
    STOCH_ZONES = [20, 30]
    K_PERIODS = [5, 10, 14]
    D_PERIODS = [2, 3, 5]
    SLOWINGS = [1, 2, 3]
    FAST_PERIODS = [8, 12, 16]
    SLOW_PERIODS = [21, 26, 34]
    SIGNAL_PERIODS = [7, 9, 12]
    MACD_PRICES = [0, 4, 5]  # close, HL/2, HLC/3
    RSI_PERIODS = [10, 14, 21]

    # FAZA 4: Grid Settings
    MULTIPLIERS = [1.2, 1.4, 1.6, 1.8, 2.0, 2.5]
    LAF_VALUES = [0.3, 0.5, 0.7, 1.0]
    GAF_VALUES = [0.8, 1.0, 1.2, 1.5]
    GRID_ARRAYS = ["25,50,100", "20,40,80", "30,60,120"]
    TP_ARRAYS = ["50,100,200", "40,80,160", "60,120,240"]
    SET_ARRAYS = ["4,4", "3,5", "5,3"]
    ENTRY_DELAYS = [1200, 2400, 3600]

    # FAZA 5: Risk Management
    MAX_TRADES_OPTIONS = [10, 15, 20, 25]
    BREAK_EVEN_TRADES = [10, 12, 15]
    MAX_DD_OPTIONS = [30, 40, 50, 60]
    MAX_SPREADS = [3, 5, 7, 10]
    CLOSE_TRADES_LEVELS = [5, 7, 10]
    MAX_CLOSE_TRADES = [3, 4, 5]
    SL_PIPS = [20, 30, 50]
    TSL_PIPS = [5, 10, 15]
    MIN_TP_PIPS = [0, 10, 20]

    def __init__(
        self,
        symbol: str = "EURUSD",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        results_dir: Path = None,
        use_gpu: bool = True,
        top_n: int = 10
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.results_dir = Path(results_dir) if results_dir else Path("data/results/sequential")
        self.use_gpu = use_gpu
        self.top_n = top_n

        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.market_data = None
        self.phase_results = {}

    def load_market_data(self):
        """Load market data"""
        self.logger.info(f"Loading {self.symbol} data: {self.start_date} to {self.end_date}")
        loader = DataLoader()
        df = loader.load_dukascopy_m1(self.symbol, self.start_date, self.end_date)
        self.market_data = {
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'bars': len(df)
        }
        self.logger.info(f"Loaded {self.market_data['bars']:,} bars")

    def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run single backtest with given configuration"""
        from core.blessing_backtest_engine import BlessingBacktestEngine, BacktestConfig
        from strategies.blessing_entry_generator import EntryConfig
        from strategies.blessing_grid_system import GridConfig

        # Convert dict config to BacktestConfig
        entry_config = EntryConfig(
            ma_entry=config.get('ma_entry', 0),
            cci_entry=config.get('cci_entry', 0),
            bollinger_entry=config.get('bollinger_entry', 0),
            stoch_entry=config.get('stoch_entry', 0),
            macd_entry=config.get('macd_entry', 0),
            b3_traditional=config.get('b3_traditional', True),
            force_market_cond=config.get('force_market_cond', 3),
            use_any_entry=config.get('use_any_entry', False)
        )

        grid_config = GridConfig(
            base_lot=config.get('base_lot', 0.01),
            lot_multiplier=config.get('lot_multiplier', 2.0),
            grid_set_01=config.get('grid_step_pips', 20.0),  # grid_step → grid_set_01
            max_trades=config.get('max_grid_levels', 15),  # max_grid_levels → max_trades
            take_profit=config.get('take_profit_pips', 50.0),  # take_profit_pips → take_profit
            auto_cal=config.get('autocal', False),  # autocal → auto_cal
            smart_grid=config.get('use_smartgrid', True),  # use_smartgrid → smart_grid
            gaf=config.get('gaf', 1.0)
        )

        backtest_config = BacktestConfig(
            entry_config=entry_config,
            grid_config=grid_config,
            initial_deposit=1000.0,
            max_drawdown_percent=config.get('max_dd_percent', 30.0),
            max_total_trades=config.get('max_trades', 20)
        )

        engine = BlessingBacktestEngine(backtest_config, verbose=False)
        result = engine.run_backtest(
            high=self.market_data['high'],
            low=self.market_data['low'],
            close=self.market_data['close']
        )

        return {
            'config': config,
            'score': result.net_profit,
            'net_profit': result.net_profit,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'max_drawdown_percent': result.max_drawdown_percent,
            'sharpe_ratio': result.sharpe_ratio,
            'final_balance': result.final_balance,
            **config  # Include all config params in result
        }

    def run_phase_2_timeframes(self, best_phase_1: Dict[str, Any]) -> List[Dict[str, Any]]:
        """FAZA 2: Optimize indicator timeframes"""
        self.logger.info("="*80)
        self.logger.info("FAZA 2: TIMEFRAMES OPTIMIZATION")
        self.logger.info("="*80)

        combinations = []
        for ma_tf in self.TIMEFRAMES:
            for cci_tf in self.TIMEFRAMES:
                for boll_tf in self.TIMEFRAMES:
                    for stoch_tf in self.TIMEFRAMES:
                        for macd_tf in self.TIMEFRAMES:
                            config = {
                                **best_phase_1,
                                'ma_timeframe': ma_tf,
                                'cci_timeframe': cci_tf,
                                'bollinger_timeframe': boll_tf,
                                'stoch_timeframe': stoch_tf,
                                'macd_timeframe': macd_tf,
                            }
                            combinations.append(config)

        total = len(combinations)
        self.logger.info(f"Testing {total:,} timeframe combinations...")

        results = []
        for i, config in enumerate(combinations, 1):
            if i % 100 == 0:
                self.logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%)")

            result = self.run_backtest(config)
            results.append(result)

        # Sort and save top N
        results.sort(key=lambda x: x['score'], reverse=True)
        self.save_phase_results("phase_2_timeframes", results[:self.top_n])

        return results

    def run_phase_3_indicators(self, best_phase_2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """FAZA 3: Optimize indicator parameters (sampled)"""
        self.logger.info("="*80)
        self.logger.info("FAZA 3: INDICATOR PARAMETERS OPTIMIZATION (SAMPLED)")
        self.logger.info("="*80)

        # Too many combinations - use systematic sampling
        import itertools
        import random

        # Generate all combinations
        all_combos = list(itertools.product(
            self.MA_PERIODS,
            self.MA_DISTANCES,
            self.CCI_PERIODS,
            self.BOLL_PERIODS,
            self.BOLL_DISTANCES,
            self.BOLL_DEVIATIONS,
            self.STOCH_ZONES,
            self.K_PERIODS,
            self.D_PERIODS,
            self.SLOWINGS,
            self.FAST_PERIODS,
            self.SLOW_PERIODS,
            self.SIGNAL_PERIODS,
            self.MACD_PRICES,
            self.RSI_PERIODS
        ))

        total_possible = len(all_combos)
        self.logger.info(f"Total possible combinations: {total_possible:,}")

        # Sample 10,000 combinations
        sample_size = min(10000, total_possible)
        sampled_combos = random.sample(all_combos, sample_size)

        self.logger.info(f"Testing {sample_size:,} sampled combinations...")

        results = []
        for i, combo in enumerate(sampled_combos, 1):
            if i % 100 == 0:
                self.logger.info(f"Progress: {i}/{sample_size} ({i/sample_size*100:.1f}%)")

            config = {
                **best_phase_2,
                'ma_period': combo[0],
                'ma_distance': combo[1],
                'cci_period': combo[2],
                'boll_period': combo[3],
                'boll_distance': combo[4],
                'boll_deviation': combo[5],
                'buysell_stoch_zone': combo[6],
                'k_period': combo[7],
                'd_period': combo[8],
                'slowing': combo[9],
                'fast_period': combo[10],
                'slow_period': combo[11],
                'signal_period': combo[12],
                'macd_price': combo[13],
                'rsi_period': combo[14],
            }

            result = self.run_backtest(config)
            results.append(result)

        results.sort(key=lambda x: x['score'], reverse=True)
        self.save_phase_results("phase_3_indicators", results[:self.top_n])

        return results

    def run_phase_4_grid(self, best_phase_3: Dict[str, Any]) -> List[Dict[str, Any]]:
        """FAZA 4: Optimize grid settings"""
        self.logger.info("="*80)
        self.logger.info("FAZA 4: GRID SETTINGS OPTIMIZATION")
        self.logger.info("="*80)

        import itertools

        combinations = []
        for mult in self.MULTIPLIERS:
            for laf in self.LAF_VALUES:
                for gaf in self.GAF_VALUES:
                    for grid_arr in self.GRID_ARRAYS:
                        for tp_arr in self.TP_ARRAYS:
                            for set_arr in self.SET_ARRAYS:
                                for autocal in [True, False]:
                                    for smartgrid in [True, False]:
                                        for entry_delay in self.ENTRY_DELAYS:
                                            config = {
                                                **best_phase_3,
                                                'lot_multiplier': mult,
                                                'laf': laf,
                                                'gaf': gaf,
                                                'grid_set_array': grid_arr,
                                                'tp_set_array': tp_arr,
                                                'set_count_array': set_arr,
                                                'autocal': autocal,
                                                'use_smartgrid': smartgrid,
                                                'entry_delay': entry_delay,
                                            }
                                            combinations.append(config)

        total = len(combinations)
        self.logger.info(f"Testing {total:,} grid combinations...")

        results = []
        for i, config in enumerate(combinations, 1):
            if i % 1000 == 0:
                self.logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%)")

            result = self.run_backtest(config)
            results.append(result)

        results.sort(key=lambda x: x['score'], reverse=True)
        self.save_phase_results("phase_4_grid", results[:self.top_n])

        return results

    def run_phase_5_risk(self, best_phase_4: Dict[str, Any]) -> List[Dict[str, Any]]:
        """FAZA 5: Optimize risk management (sampled)"""
        self.logger.info("="*80)
        self.logger.info("FAZA 5: RISK MANAGEMENT OPTIMIZATION (SAMPLED)")
        self.logger.info("="*80)

        import itertools
        import random

        all_combos = list(itertools.product(
            self.MAX_TRADES_OPTIONS,
            self.BREAK_EVEN_TRADES,
            self.MAX_DD_OPTIONS,
            self.MAX_SPREADS,
            [True, False],  # use_close_oldest
            self.CLOSE_TRADES_LEVELS,
            self.MAX_CLOSE_TRADES,
            [True, False],  # force_close_oldest
            [True, False],  # use_stoploss
            self.SL_PIPS,
            self.TSL_PIPS,
            self.MIN_TP_PIPS
        ))

        total_possible = len(all_combos)
        sample_size = min(50000, total_possible)
        sampled_combos = random.sample(all_combos, sample_size)

        self.logger.info(f"Testing {sample_size:,} sampled risk management combinations...")

        results = []
        for i, combo in enumerate(sampled_combos, 1):
            if i % 1000 == 0:
                self.logger.info(f"Progress: {i}/{sample_size} ({i/sample_size*100:.1f}%)")

            config = {
                **best_phase_4,
                'max_trades': combo[0],
                'break_even_trade': combo[1],
                'max_dd_percent': combo[2],
                'max_spread': combo[3],
                'use_close_oldest': combo[4],
                'close_trades_level': combo[5],
                'max_close_trades': combo[6],
                'force_close_oldest': combo[7],
                'use_stoploss': combo[8],
                'sl_pips': combo[9],
                'tsl_pips': combo[10],
                'min_tp_pips': combo[11],
            }

            result = self.run_backtest(config)
            results.append(result)

        results.sort(key=lambda x: x['score'], reverse=True)
        self.save_phase_results("phase_5_risk", results[:self.top_n])

        return results

    def save_phase_results(self, phase_name: str, results: List[Dict[str, Any]]):
        """Save phase results to CSV"""
        df = pd.DataFrame(results)
        csv_path = self.results_dir / f"{phase_name}_top_{self.top_n}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved {len(results)} results to {csv_path}")

    def run_all_phases(self) -> Dict[str, Any]:
        """Run all optimization phases sequentially"""
        self.load_market_data()

        # FAZA 1: Use existing results
        self.logger.info("FAZA 1: Loading existing entry logic results...")
        phase_1_csv = Path("data/results/continuous/phase_0_top_100.csv")
        if not phase_1_csv.exists():
            raise FileNotFoundError("FAZA 1 results not found! Run phase 1 first.")

        df_phase_1 = pd.read_csv(phase_1_csv)
        best_phase_1 = df_phase_1.iloc[0].to_dict()
        self.logger.info(f"Best Phase 1 score: {best_phase_1['score']:.2f}")

        # FAZA 2: Timeframes
        results_phase_2 = self.run_phase_2_timeframes(best_phase_1)
        best_phase_2 = results_phase_2[0]
        self.logger.info(f"Best Phase 2 score: {best_phase_2['score']:.2f}")

        # FAZA 3: Indicators
        results_phase_3 = self.run_phase_3_indicators(best_phase_2)
        best_phase_3 = results_phase_3[0]
        self.logger.info(f"Best Phase 3 score: {best_phase_3['score']:.2f}")

        # FAZA 4: Grid
        results_phase_4 = self.run_phase_4_grid(best_phase_3)
        best_phase_4 = results_phase_4[0]
        self.logger.info(f"Best Phase 4 score: {best_phase_4['score']:.2f}")

        # FAZA 5: Risk
        results_phase_5 = self.run_phase_5_risk(best_phase_4)
        best_phase_5 = results_phase_5[0]
        self.logger.info(f"Best Phase 5 score: {best_phase_5['score']:.2f}")

        # Final results
        final_results = {
            'top_configs': results_phase_5[:self.top_n],
            'all_results': results_phase_5,
            'phase_progression': {
                'phase_1': best_phase_1['score'],
                'phase_2': best_phase_2['score'],
                'phase_3': best_phase_3['score'],
                'phase_4': best_phase_4['score'],
                'phase_5': best_phase_5['score'],
            }
        }

        # Save final summary
        summary_path = self.results_dir / "optimization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(final_results['phase_progression'], f, indent=2)

        return final_results
