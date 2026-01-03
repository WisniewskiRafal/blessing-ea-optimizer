"""
Genetic Algorithm Optimizer - Evolutionary optimization of ALL 64 parameters simultaneously
Uses DEAP (Distributed Evolutionary Algorithms in Python)
"""

import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logging.warning("DEAP not installed. Install with: pip install deap")

from core.blessing_backtest_engine import BlessingBacktestEngine
from core.data_loader import DataLoader


class GeneticOptimizer:
    """Genetic Algorithm optimizer for Blessing EA"""

    # Parameter ranges for genetic encoding
    PARAM_RANGES = {
        # Entry signals (FAZA 1)
        'ma_entry': (0, 2),
        'cci_entry': (0, 2),
        'bollinger_entry': (0, 2),
        'stoch_entry': (0, 2),
        'macd_entry': (0, 2),
        'b3_traditional': (0, 1),  # bool as 0/1
        'force_market_cond': (0, 3),
        'use_any_entry': (0, 1),

        # Timeframes (FAZA 2) - encoded as index into [1,5,15,30,60,240,1440]
        'ma_timeframe_idx': (0, 6),
        'cci_timeframe_idx': (0, 6),
        'bollinger_timeframe_idx': (0, 6),
        'stoch_timeframe_idx': (0, 6),
        'macd_timeframe_idx': (0, 6),

        # Indicator parameters (FAZA 3)
        'ma_period': (50, 400),
        'ma_distance': (5, 20),
        'cci_period': (10, 30),
        'boll_period': (10, 30),
        'boll_distance': (5, 20),
        'boll_deviation': (15, 30),  # *10 to keep as int
        'buysell_stoch_zone': (20, 30),
        'k_period': (5, 14),
        'd_period': (2, 5),
        'slowing': (1, 3),
        'fast_period': (8, 16),
        'slow_period': (21, 34),
        'signal_period': (7, 12),
        'macd_price': (0, 2),  # encoded: 0=close, 1=HL/2, 2=HLC/3
        'rsi_period': (10, 21),

        # Grid settings (FAZA 4)
        'lot_multiplier': (12, 25),  # *10
        'laf': (3, 10),  # *10
        'gaf': (8, 15),  # *10
        'grid_set_idx': (0, 2),  # index into preset arrays
        'tp_set_idx': (0, 2),
        'set_count_idx': (0, 2),
        'autocal': (0, 1),
        'use_smartgrid': (0, 1),
        'entry_delay': (12, 36),  # *100

        # Risk management (FAZA 5)
        'max_trades': (10, 25),
        'break_even_trade': (10, 15),
        'max_dd_percent': (30, 60),
        'max_spread': (3, 10),
        'use_close_oldest': (0, 1),
        'close_trades_level': (5, 10),
        'max_close_trades': (3, 5),
        'force_close_oldest': (0, 1),
        'use_stoploss': (0, 1),
        'sl_pips': (20, 50),
        'tsl_pips': (5, 15),
        'min_tp_pips': (0, 20),
    }

    TIMEFRAME_MAP = [1, 5, 15, 30, 60, 240, 1440]
    GRID_ARRAYS = ["25,50,100", "20,40,80", "30,60,120"]
    TP_ARRAYS = ["50,100,200", "40,80,160", "60,120,240"]
    SET_ARRAYS = ["4,4", "3,5", "5,3"]
    MACD_PRICE_MAP = [0, 4, 5]

    def __init__(
        self,
        symbol: str = "EURUSD",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        results_dir: Path = None,
        use_gpu: bool = True,
        top_n: int = 10,
        population_size: int = 100,
        generations: int = 50,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3
    ):
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP required. Install with: pip install deap")

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.results_dir = Path(results_dir) if results_dir else Path("data/results/genetic")
        self.use_gpu = use_gpu
        self.top_n = top_n

        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.market_data = None

        # DEAP setup
        self.setup_deap()

    def setup_deap(self):
        """Setup DEAP framework"""
        # Create fitness and individual classes
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Multi-objective: maximize profit, minimize drawdown
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Register attribute generators for each parameter
        for param, (min_val, max_val) in self.PARAM_RANGES.items():
            self.toolbox.register(
                f"attr_{param}",
                random.randint,
                min_val,
                max_val
            )

        # Register individual generator
        param_names = list(self.PARAM_RANGES.keys())
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            tuple(getattr(self.toolbox, f"attr_{p}") for p in param_names),
            n=1
        )

        # Register population generator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selNSGA2)

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

    def decode_individual(self, individual: list) -> Dict[str, Any]:
        """Decode genetic individual to config dict"""
        param_names = list(self.PARAM_RANGES.keys())
        raw_config = dict(zip(param_names, individual))

        # Decode config
        config = {}

        # Entry signals
        config['ma_entry'] = raw_config['ma_entry']
        config['cci_entry'] = raw_config['cci_entry']
        config['bollinger_entry'] = raw_config['bollinger_entry']
        config['stoch_entry'] = raw_config['stoch_entry']
        config['macd_entry'] = raw_config['macd_entry']
        config['b3_traditional'] = bool(raw_config['b3_traditional'])
        config['force_market_cond'] = raw_config['force_market_cond']
        config['use_any_entry'] = bool(raw_config['use_any_entry'])

        # Timeframes
        config['ma_timeframe'] = self.TIMEFRAME_MAP[raw_config['ma_timeframe_idx']]
        config['cci_timeframe'] = self.TIMEFRAME_MAP[raw_config['cci_timeframe_idx']]
        config['bollinger_timeframe'] = self.TIMEFRAME_MAP[raw_config['bollinger_timeframe_idx']]
        config['stoch_timeframe'] = self.TIMEFRAME_MAP[raw_config['stoch_timeframe_idx']]
        config['macd_timeframe'] = self.TIMEFRAME_MAP[raw_config['macd_timeframe_idx']]

        # Indicators
        config['ma_period'] = raw_config['ma_period']
        config['ma_distance'] = raw_config['ma_distance']
        config['cci_period'] = raw_config['cci_period']
        config['boll_period'] = raw_config['boll_period']
        config['boll_distance'] = raw_config['boll_distance']
        config['boll_deviation'] = raw_config['boll_deviation'] / 10.0
        config['buysell_stoch_zone'] = raw_config['buysell_stoch_zone']
        config['k_period'] = raw_config['k_period']
        config['d_period'] = raw_config['d_period']
        config['slowing'] = raw_config['slowing']
        config['fast_period'] = raw_config['fast_period']
        config['slow_period'] = raw_config['slow_period']
        config['signal_period'] = raw_config['signal_period']
        config['macd_price'] = self.MACD_PRICE_MAP[raw_config['macd_price']]
        config['rsi_period'] = raw_config['rsi_period']

        # Grid
        config['lot_multiplier'] = raw_config['lot_multiplier'] / 10.0
        config['laf'] = raw_config['laf'] / 10.0
        config['gaf'] = raw_config['gaf'] / 10.0
        config['grid_set_array'] = self.GRID_ARRAYS[raw_config['grid_set_idx']]
        config['tp_set_array'] = self.TP_ARRAYS[raw_config['tp_set_idx']]
        config['set_count_array'] = self.SET_ARRAYS[raw_config['set_count_idx']]
        config['autocal'] = bool(raw_config['autocal'])
        config['use_smartgrid'] = bool(raw_config['use_smartgrid'])
        config['entry_delay'] = raw_config['entry_delay'] * 100

        # Risk
        config['max_trades'] = raw_config['max_trades']
        config['break_even_trade'] = raw_config['break_even_trade']
        config['max_dd_percent'] = raw_config['max_dd_percent']
        config['max_spread'] = raw_config['max_spread']
        config['use_close_oldest'] = bool(raw_config['use_close_oldest'])
        config['close_trades_level'] = raw_config['close_trades_level']
        config['max_close_trades'] = raw_config['max_close_trades']
        config['force_close_oldest'] = bool(raw_config['force_close_oldest'])
        config['use_stoploss'] = bool(raw_config['use_stoploss'])
        config['sl_pips'] = raw_config['sl_pips']
        config['tsl_pips'] = raw_config['tsl_pips']
        config['min_tp_pips'] = raw_config['min_tp_pips']

        # Defaults for other params
        config['base_lot'] = 0.01
        config['take_profit_pips'] = 50
        config['max_grid_levels'] = 15

        return config

    def encode_config(self, config: Dict[str, Any]) -> list:
        """Encode config dict to genetic individual"""
        individual = []
        param_names = list(self.PARAM_RANGES.keys())

        for param in param_names:
            if param.endswith('_idx'):
                # Special handling for indexed parameters
                base_param = param.replace('_idx', '')
                if base_param + '_timeframe' in config:
                    tf_value = config[base_param + '_timeframe']
                    idx = self.TIMEFRAME_MAP.index(tf_value) if tf_value in self.TIMEFRAME_MAP else 4
                    individual.append(idx)
                elif 'grid_set' in param:
                    arr_value = config.get('grid_set_array', "25,50,100")
                    idx = self.GRID_ARRAYS.index(arr_value) if arr_value in self.GRID_ARRAYS else 0
                    individual.append(idx)
                elif 'tp_set' in param:
                    arr_value = config.get('tp_set_array', "50,100,200")
                    idx = self.TP_ARRAYS.index(arr_value) if arr_value in self.TP_ARRAYS else 0
                    individual.append(idx)
                elif 'set_count' in param:
                    arr_value = config.get('set_count_array', "4,4")
                    idx = self.SET_ARRAYS.index(arr_value) if arr_value in self.SET_ARRAYS else 0
                    individual.append(idx)
                else:
                    individual.append(0)
            elif param in ['lot_multiplier', 'laf', 'gaf', 'boll_deviation']:
                # Scaled parameters
                value = config.get(param, 1.0)
                individual.append(int(value * 10))
            elif param == 'entry_delay':
                value = config.get(param, 2400)
                individual.append(int(value / 100))
            elif param in ['b3_traditional', 'use_any_entry', 'autocal', 'use_smartgrid',
                          'use_close_oldest', 'force_close_oldest', 'use_stoploss']:
                # Booleans
                individual.append(1 if config.get(param, False) else 0)
            elif param == 'macd_price':
                value = config.get(param, 0)
                idx = self.MACD_PRICE_MAP.index(value) if value in self.MACD_PRICE_MAP else 0
                individual.append(idx)
            else:
                # Regular int parameters
                individual.append(config.get(param, self.PARAM_RANGES[param][0]))

        return creator.Individual(individual)

    def evaluate_individual(self, individual: list) -> tuple:
        """Evaluate fitness of individual"""
        from core.blessing_backtest_engine import BacktestConfig
        from strategies.blessing_entry_generator import EntryConfig
        from strategies.blessing_grid_system import GridConfig

        config = self.decode_individual(individual)

        try:
            # Convert dict to BacktestConfig
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
                grid_set_01=config.get('grid_step_pips', 20.0),
                max_trades=config.get('max_grid_levels', 15),
                take_profit=config.get('take_profit_pips', 50.0),
                auto_cal=config.get('autocal', False),
                smart_grid=config.get('use_smartgrid', True),
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

            # Multi-objective: (profit, drawdown)
            fitness = (result.net_profit, result.max_drawdown_percent)

        except Exception as e:
            self.logger.warning(f"Backtest failed: {e}")
            fitness = (-10000, 100)  # Penalize failed backtests

        return fitness

    def mutate_individual(self, individual: list) -> tuple:
        """Mutate individual with bounds checking"""
        param_names = list(self.PARAM_RANGES.keys())

        for i, param in enumerate(param_names):
            if random.random() < 0.1:  # 10% mutation rate per gene
                min_val, max_val = self.PARAM_RANGES[param]
                # Gaussian mutation
                individual[i] = int(individual[i] + random.gauss(0, (max_val - min_val) * 0.1))
                # Clip to bounds
                individual[i] = max(min_val, min(max_val, individual[i]))

        return (individual,)

    def create_seed_population(self, seed_configs: List[Dict[str, Any]]) -> list:
        """Create initial population from seed configurations"""
        population = []

        # Add seed configurations
        for config in seed_configs:
            individual = self.encode_config(config)
            population.append(individual)

        # Fill rest with random individuals
        remaining = self.population_size - len(population)
        if remaining > 0:
            population.extend(self.toolbox.population(n=remaining))

        return population

    def run_optimization(self) -> Dict[str, Any]:
        """Run genetic algorithm optimization"""
        self.load_market_data()

        self.logger.info("="*80)
        self.logger.info("GENETIC ALGORITHM OPTIMIZATION")
        self.logger.info(f"Population: {self.population_size} | Generations: {self.generations}")
        self.logger.info("="*80)

        # Create initial population
        pop = self.toolbox.population(n=self.population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("min", np.min, axis=0)

        # Hall of Fame (Pareto Front)
        hof = tools.ParetoFront()

        # Run evolution
        pop, logbook = algorithms.eaMuPlusLambda(
            pop,
            self.toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # Extract results
        results = []
        for individual in hof:
            config = self.decode_individual(individual)
            fitness = individual.fitness.values

            result = {
                'config': config,
                'score': fitness[0],  # Profit
                'net_profit': fitness[0],
                'max_drawdown_percent': fitness[1],
                **config
            }
            results.append(result)

        # Save results
        df = pd.DataFrame(results)
        csv_path = self.results_dir / f"genetic_top_{self.top_n}.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved {len(results)} Pareto optimal solutions to {csv_path}")

        return {
            'top_configs': results[:self.top_n],
            'all_results': results,
            'pareto_front': hof
        }

    def run_with_seed_population(self, seed_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run GA with seed population from previous results"""
        self.load_market_data()

        self.logger.info("="*80)
        self.logger.info("GENETIC ALGORITHM WITH SEED POPULATION")
        self.logger.info(f"Seeds: {len(seed_configs)} | Population: {self.population_size}")
        self.logger.info("="*80)

        # Create population with seeds
        pop = self.create_seed_population(seed_configs)

        # Run evolution
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("max", np.max, axis=0)

        hof = tools.ParetoFront()

        pop, logbook = algorithms.eaMuPlusLambda(
            pop,
            self.toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # Extract results
        results = []
        for individual in hof:
            config = self.decode_individual(individual)
            fitness = individual.fitness.values

            result = {
                'config': config,
                'score': fitness[0],
                'net_profit': fitness[0],
                'max_drawdown_percent': fitness[1],
                **config
            }
            results.append(result)

        df = pd.DataFrame(results)
        csv_path = self.results_dir / f"genetic_refined_top_{self.top_n}.csv"
        df.to_csv(csv_path, index=False)

        return {
            'top_configs': results[:self.top_n],
            'all_results': results
        }
