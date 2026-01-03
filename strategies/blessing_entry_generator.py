# strategies/blessing_entry_generator.py
# Blessing Entry Combination Generator
# Author: Rafał Wiśniewski | Data & AI Solutions

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

from indicators.blessing_indicators import BlessingIndicators


@dataclass
class EntryConfig:
    """Entry configuration for Blessing EA"""
    # Indicator states (0=off, 1=normal, 2=reverse)
    ma_entry: int  # 0/1/2
    cci_entry: int  # 0/1/2
    bollinger_entry: int  # 0/1/2
    stoch_entry: int  # 0/1/2
    macd_entry: int  # 0/1/2

    # Entry logic
    b3_traditional: bool  # True=STOP/LIMIT, False=instant BUY/SELL
    force_market_cond: int  # 0=uptrend, 1=downtrend, 2=range, 3=off
    use_any_entry: bool  # True=ANY indicator, False=ALL indicators

    def to_dict(self) -> Dict:
        """Convert to dict"""
        return {
            'ma_entry': self.ma_entry,
            'cci_entry': self.cci_entry,
            'bollinger_entry': self.bollinger_entry,
            'stoch_entry': self.stoch_entry,
            'macd_entry': self.macd_entry,
            'b3_traditional': self.b3_traditional,
            'force_market_cond': self.force_market_cond,
            'use_any_entry': self.use_any_entry
        }

    def hash(self) -> str:
        """Generate unique hash"""
        return f"{self.ma_entry}_{self.cci_entry}_{self.bollinger_entry}_{self.stoch_entry}_{self.macd_entry}_{int(self.b3_traditional)}_{self.force_market_cond}_{int(self.use_any_entry)}"


class BlessingEntryGenerator:
    """
    Blessing Entry Combination Generator

    Generates all 3,888 entry combinations:
    - 5 indicators × 3 states each = 3^5 = 243 combinations
    - × 2 (traditional/instant) = 486
    - × 4 (market conditions) = 1,944
    - × 2 (any/all entry) = 3,888 combinations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators = BlessingIndicators()

    def generate_all_combinations(self) -> List[EntryConfig]:
        """
        Generate all 3,888 entry combinations

        Returns:
            List of EntryConfig objects
        """
        combinations = []

        # Indicator options (0=off, 1=normal, 2=reverse)
        indicator_options = [0, 1, 2]

        # Logic options
        traditional_options = [True, False]
        market_cond_options = [0, 1, 2, 3]  # up/down/range/off
        any_entry_options = [True, False]

        # Generate all combinations
        for ma in indicator_options:
            for cci in indicator_options:
                for boll in indicator_options:
                    for stoch in indicator_options:
                        for macd in indicator_options:
                            for trad in traditional_options:
                                for cond in market_cond_options:
                                    for any_e in any_entry_options:
                                        config = EntryConfig(
                                            ma_entry=ma,
                                            cci_entry=cci,
                                            bollinger_entry=boll,
                                            stoch_entry=stoch,
                                            macd_entry=macd,
                                            b3_traditional=trad,
                                            force_market_cond=cond,
                                            use_any_entry=any_e
                                        )
                                        combinations.append(config)

        self.logger.info(f"[GENERATED] {len(combinations)} entry combinations")
        return combinations

    def calculate_entry_signal(self,
                              config: EntryConfig,
                              high: np.ndarray,
                              low: np.ndarray,
                              close: np.ndarray,
                              indicator_params: Dict) -> int:
        """
        Calculate entry signal for given configuration

        Args:
            config: Entry configuration
            high: High prices
            low: Low prices
            close: Close prices
            indicator_params: Dict with indicator parameters
                - ma_period, ma_distance
                - cci_period
                - boll_period, boll_deviation, boll_distance
                - stoch_k, stoch_d, stoch_slowing, stoch_zone
                - macd_fast, macd_slow, macd_signal

        Returns:
            signal: 1=BUY, -1=SELL, 0=NO SIGNAL
        """
        signals = []

        # 1. MA Signal
        if config.ma_entry != 0:
            ma_signal, _ = self.indicators.ma_signal(
                close,
                period=indicator_params.get('ma_period', 20),
                ma_distance=indicator_params.get('ma_distance', 5.0)
            )

            # Apply reverse if needed
            if config.ma_entry == 2:
                ma_signal = -ma_signal

            signals.append(('ma', ma_signal))

        # 2. CCI Signal
        if config.cci_entry != 0:
            cci_signal, _ = self.indicators.cci_signal(
                high, low, close,
                period=indicator_params.get('cci_period', 14)
            )

            # Apply reverse
            if config.cci_entry == 2:
                cci_signal = -cci_signal

            signals.append(('cci', cci_signal))

        # 3. Bollinger Signal
        if config.bollinger_entry != 0:
            bb_signal, _ = self.indicators.bollinger_signal(
                close,
                period=indicator_params.get('boll_period', 15),
                deviation=indicator_params.get('boll_deviation', 2.0),
                boll_distance=indicator_params.get('boll_distance', 13.0)
            )

            # Apply reverse
            if config.bollinger_entry == 2:
                bb_signal = -bb_signal

            signals.append(('bollinger', bb_signal))

        # 4. Stochastic Signal
        if config.stoch_entry != 0:
            stoch_signal, _ = self.indicators.stochastic_signal(
                high, low, close,
                k_period=indicator_params.get('stoch_k', 10),
                d_period=indicator_params.get('stoch_d', 2),
                slowing=indicator_params.get('stoch_slowing', 2),
                zone=indicator_params.get('stoch_zone', 20.0)
            )

            # Apply reverse
            if config.stoch_entry == 2:
                stoch_signal = -stoch_signal

            signals.append(('stochastic', stoch_signal))

        # 5. MACD Signal
        if config.macd_entry != 0:
            macd_signal, _ = self.indicators.macd_signal(
                close,
                fast_period=indicator_params.get('macd_fast', 12),
                slow_period=indicator_params.get('macd_slow', 26),
                signal_period=indicator_params.get('macd_signal', 9)
            )

            # Apply reverse
            if config.macd_entry == 2:
                macd_signal = -macd_signal

            signals.append(('macd', macd_signal))

        # Combine signals
        if not signals:
            return 0  # No indicators active

        # Extract signal values
        signal_values = [s[1] for s in signals]

        # Apply UseAnyEntry logic
        if config.use_any_entry:
            # ANY indicator can trigger
            # If ANY is bullish (1), return 1
            # If ANY is bearish (-1), return -1
            # Priority: first non-zero signal
            for sig in signal_values:
                if sig != 0:
                    final_signal = sig
                    break
            else:
                final_signal = 0
        else:
            # ALL indicators must agree
            if all(s == 1 for s in signal_values):
                final_signal = 1  # All bullish
            elif all(s == -1 for s in signal_values):
                final_signal = -1  # All bearish
            elif all(s == 0 for s in signal_values):
                final_signal = 0  # All neutral
            else:
                final_signal = 0  # Mixed signals

        # Apply ForceMarketCond override
        if config.force_market_cond == 0:  # Force uptrend
            if final_signal == -1:
                final_signal = 0  # No sell in forced uptrend
        elif config.force_market_cond == 1:  # Force downtrend
            if final_signal == 1:
                final_signal = 0  # No buy in forced downtrend
        elif config.force_market_cond == 2:  # Force range
            final_signal = 0  # No signals in range
        # elif config.force_market_cond == 3: off - no override

        return final_signal


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("BLESSING ENTRY GENERATOR TEST")
    print("="*60)

    # Create generator
    generator = BlessingEntryGenerator()

    # Generate all combinations
    print("\n[TEST 1] Generate all combinations")
    all_combinations = generator.generate_all_combinations()
    print(f"  Total: {len(all_combinations)} combinations")
    print(f"  Expected: 3888")
    print(f"  Match: {'YES' if len(all_combinations) == 3888 else 'NO'}")

    # Test signal calculation
    print("\n[TEST 2] Calculate entry signals")

    # Generate test data
    np.random.seed(42)
    n = 100
    close = np.cumsum(np.random.randn(n) * 0.01) + 1.1000
    high = close + np.abs(np.random.randn(n) * 0.001)
    low = close - np.abs(np.random.randn(n) * 0.001)

    # Test configurations
    test_configs = [
        EntryConfig(1, 0, 0, 0, 0, True, 3, False),  # MA only
        EntryConfig(1, 1, 0, 0, 0, True, 3, False),  # MA + CCI (all must agree)
        EntryConfig(1, 1, 0, 0, 0, True, 3, True),   # MA + CCI (any can trigger)
        EntryConfig(1, 1, 1, 1, 1, True, 3, False),  # All indicators (all agree)
        EntryConfig(2, 0, 0, 0, 0, True, 3, False),  # MA reverse
    ]

    indicator_params = {
        'ma_period': 20,
        'ma_distance': 5.0,
        'cci_period': 14,
        'boll_period': 15,
        'boll_deviation': 2.0,
        'boll_distance': 13.0,
        'stoch_k': 10,
        'stoch_d': 2,
        'stoch_slowing': 2,
        'stoch_zone': 20.0,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }

    for i, config in enumerate(test_configs):
        signal = generator.calculate_entry_signal(
            config, high, low, close, indicator_params
        )
        signal_str = 'BUY' if signal == 1 else 'SELL' if signal == -1 else 'NO SIGNAL'
        print(f"  Config {i+1}: {signal_str} ({signal})")
        print(f"    MA:{config.ma_entry} CCI:{config.cci_entry} BB:{config.bollinger_entry} " +
              f"Stoch:{config.stoch_entry} MACD:{config.macd_entry}")

    # Test hash uniqueness
    print("\n[TEST 3] Hash uniqueness")
    hashes = set(c.hash() for c in all_combinations)
    print(f"  Unique hashes: {len(hashes)}")
    print(f"  Total configs: {len(all_combinations)}")
    print(f"  All unique: {'YES' if len(hashes) == len(all_combinations) else 'NO'}")

    # Distribution analysis
    print("\n[TEST 4] Distribution analysis")
    print(f"  MA-only configs: {sum(1 for c in all_combinations if c.ma_entry != 0 and c.cci_entry == 0 and c.bollinger_entry == 0 and c.stoch_entry == 0 and c.macd_entry == 0)}")
    print(f"  All-indicators configs: {sum(1 for c in all_combinations if all([c.ma_entry != 0, c.cci_entry != 0, c.bollinger_entry != 0, c.stoch_entry != 0, c.macd_entry != 0]))}")
    print(f"  Traditional entry: {sum(1 for c in all_combinations if c.b3_traditional)}")
    print(f"  Instant entry: {sum(1 for c in all_combinations if not c.b3_traditional)}")

    print("\n" + "="*60)
    print("[OK] Blessing Entry Generator Test Completed")
    print("="*60)
