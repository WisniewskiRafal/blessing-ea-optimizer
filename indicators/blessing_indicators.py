# indicators/blessing_indicators.py
# Blessing EA Indicators - Python Implementation
# Author: Rafał Wiśniewski | Data & AI Solutions

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging


class BlessingIndicators:
    """
    Blessing EA Technical Indicators

    Implements all 5 indicators used in Blessing EA:
    1. Moving Average (MA) with channel
    2. Commodities Channel Index (CCI) - multi-timeframe
    3. Bollinger Bands (BB)
    4. Stochastic Oscillator
    5. MACD

    Each indicator returns signals compatible with Blessing entry logic:
    - 0 = ranging/neutral
    - 1 = bullish/long
    - -1 = bearish/short
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # ===================================================================
    # 1. MOVING AVERAGE (MA) WITH CHANNEL
    # ===================================================================

    def ma_signal(self,
                  close: np.ndarray,
                  period: int = 20,
                  ma_distance: float = 5.0,
                  pip_value: float = 0.0001) -> Tuple[int, float]:
        """
        Calculate MA-based signal with ranging channel

        Args:
            close: Close prices
            period: MA period
            ma_distance: Channel distance in pips
            pip_value: Pip value (0.0001 for most pairs)

        Returns:
            (signal, ma_value)
            signal: 1=bullish, -1=bearish, 0=ranging
        """
        # Calculate MA
        ma = self._calculate_sma(close, period)

        if np.isnan(ma[-1]):
            return 0, np.nan

        # Current price and MA
        current_price = close[-1]
        current_ma = ma[-1]

        # Channel boundaries
        channel_pips = ma_distance * pip_value
        upper_channel = current_ma + channel_pips
        lower_channel = current_ma - channel_pips

        # Determine signal
        if current_price > upper_channel:
            signal = 1  # Bullish
        elif current_price < lower_channel:
            signal = -1  # Bearish
        else:
            signal = 0  # Ranging (price inside channel)

        return signal, current_ma

    # ===================================================================
    # 2. CCI (COMMODITIES CHANNEL INDEX) - MULTI-TIMEFRAME
    # ===================================================================

    def cci_signal(self,
                   high: np.ndarray,
                   low: np.ndarray,
                   close: np.ndarray,
                   period: int = 14) -> Tuple[int, float]:
        """
        Calculate CCI signal

        Note: Blessing uses multi-TF CCI (M5, M15, M30, H1).
        This is single-TF version. Multi-TF requires separate data.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: CCI period

        Returns:
            (signal, cci_value)
            signal: 1=bullish (CCI>0), -1=bearish (CCI<0), 0=neutral
        """
        cci = self._calculate_cci(high, low, close, period)

        if np.isnan(cci[-1]):
            return 0, np.nan

        current_cci = cci[-1]

        # CCI signal
        if current_cci > 0:
            signal = 1  # Bullish
        elif current_cci < 0:
            signal = -1  # Bearish
        else:
            signal = 0  # Neutral

        return signal, current_cci

    def cci_multi_timeframe_signal(self,
                                   cci_m5: float,
                                   cci_m15: float,
                                   cci_m30: float,
                                   cci_h1: float) -> int:
        """
        Multi-timeframe CCI signal (Blessing logic)

        Args:
            cci_m5: CCI on M5
            cci_m15: CCI on M15
            cci_m30: CCI on M30
            cci_h1: CCI on H1

        Returns:
            signal: 1=all bullish, -1=all bearish, 0=mixed
        """
        # Check if all timeframes agree
        ccis = [cci_m5, cci_m15, cci_m30, cci_h1]

        if all(cci > 0 for cci in ccis):
            return 1  # All bullish
        elif all(cci < 0 for cci in ccis):
            return -1  # All bearish
        else:
            return 0  # Mixed/ranging

    # ===================================================================
    # 3. BOLLINGER BANDS
    # ===================================================================

    def bollinger_signal(self,
                        close: np.ndarray,
                        period: int = 15,
                        deviation: float = 2.0,
                        boll_distance: float = 13.0,
                        pip_value: float = 0.0001) -> Tuple[int, Tuple[float, float, float]]:
        """
        Calculate Bollinger Bands signal

        Blessing logic: Trade when price OUTSIDE bands (mean reversion)

        Args:
            close: Close prices
            period: BB period
            deviation: Standard deviation multiplier
            boll_distance: Distance parameter (affects band width)
            pip_value: Pip value

        Returns:
            (signal, (upper, middle, lower))
            signal: 1=price below lower (buy), -1=price above upper (sell), 0=inside
        """
        upper, middle, lower = self._calculate_bollinger_bands(
            close, period, deviation
        )

        if np.isnan(upper[-1]):
            return 0, (np.nan, np.nan, np.nan)

        current_price = close[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]

        # Blessing: reversal strategy (price outside bands)
        if current_price > current_upper:
            signal = -1  # Price above upper band -> SELL
        elif current_price < current_lower:
            signal = 1  # Price below lower band -> BUY
        else:
            signal = 0  # Inside bands

        return signal, (current_upper, middle[-1], current_lower)

    # ===================================================================
    # 4. STOCHASTIC OSCILLATOR
    # ===================================================================

    def stochastic_signal(self,
                         high: np.ndarray,
                         low: np.ndarray,
                         close: np.ndarray,
                         k_period: int = 10,
                         d_period: int = 2,
                         slowing: int = 2,
                         zone: float = 20.0) -> Tuple[int, Tuple[float, float]]:
        """
        Calculate Stochastic signal

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period (signal line)
            slowing: Slowing parameter
            zone: Overbought/oversold zone (default 20)

        Returns:
            (signal, (k_value, d_value))
            signal: 1=oversold, -1=overbought, 0=neutral
        """
        k, d = self._calculate_stochastic(
            high, low, close, k_period, d_period, slowing
        )

        if np.isnan(k[-1]) or np.isnan(d[-1]):
            return 0, (np.nan, np.nan)

        current_k = k[-1]
        current_d = d[-1]

        # Stochastic zones
        oversold = zone
        overbought = 100 - zone

        # Signal based on zone
        if current_k < oversold and current_d < oversold:
            signal = 1  # Oversold -> BUY
        elif current_k > overbought and current_d > overbought:
            signal = -1  # Overbought -> SELL
        else:
            signal = 0  # Neutral

        return signal, (current_k, current_d)

    # ===================================================================
    # 5. MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE)
    # ===================================================================

    def macd_signal(self,
                   close: np.ndarray,
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> Tuple[int, Tuple[float, float, float]]:
        """
        Calculate MACD signal

        Args:
            close: Close prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            (signal, (macd_line, signal_line, histogram))
            signal: 1=bullish (MACD>signal), -1=bearish (MACD<signal), 0=neutral
        """
        macd_line, signal_line, histogram = self._calculate_macd(
            close, fast_period, slow_period, signal_period
        )

        if np.isnan(macd_line[-1]):
            return 0, (np.nan, np.nan, np.nan)

        current_macd = macd_line[-1]
        current_signal = signal_line[-1]

        # MACD signal
        if current_macd > current_signal:
            signal = 1  # Bullish (MACD above signal)
        elif current_macd < current_signal:
            signal = -1  # Bearish (MACD below signal)
        else:
            signal = 0  # Neutral

        return signal, (current_macd, current_signal, histogram[-1])

    # ===================================================================
    # HELPER FUNCTIONS - INTERNAL CALCULATIONS
    # ===================================================================

    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

    def _calculate_cci(self,
                      high: np.ndarray,
                      low: np.ndarray,
                      close: np.ndarray,
                      period: int) -> np.ndarray:
        """Commodities Channel Index"""
        # Typical Price
        tp = (high + low + close) / 3.0

        # SMA of Typical Price
        sma_tp = pd.Series(tp).rolling(window=period).mean().values

        # Mean Deviation
        mad = pd.Series(tp).rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        ).values

        # CCI
        cci = (tp - sma_tp) / (0.015 * mad)

        return cci

    def _calculate_bollinger_bands(self,
                                  close: np.ndarray,
                                  period: int,
                                  deviation: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        # Middle band (SMA)
        middle = self._calculate_sma(close, period)

        # Standard deviation
        std = pd.Series(close).rolling(window=period).std().values

        # Upper and lower bands
        upper = middle + (deviation * std)
        lower = middle - (deviation * std)

        return upper, middle, lower

    def _calculate_stochastic(self,
                             high: np.ndarray,
                             low: np.ndarray,
                             close: np.ndarray,
                             k_period: int,
                             d_period: int,
                             slowing: int) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        # Lowest low and highest high over k_period
        lowest_low = pd.Series(low).rolling(window=k_period).min().values
        highest_high = pd.Series(high).rolling(window=k_period).max().values

        # %K (raw stochastic)
        k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

        # Smooth %K with slowing
        k = pd.Series(k_raw).rolling(window=slowing).mean().values

        # %D (signal line - SMA of %K)
        d = pd.Series(k).rolling(window=d_period).mean().values

        return k, d

    def _calculate_macd(self,
                       close: np.ndarray,
                       fast_period: int,
                       slow_period: int,
                       signal_period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD Indicator"""
        # Fast and slow EMAs
        ema_fast = self._calculate_ema(close, fast_period)
        ema_slow = self._calculate_ema(close, slow_period)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        signal_line = self._calculate_ema(macd_line, signal_period)

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram


if __name__ == "__main__":
    # Test indicators
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("BLESSING INDICATORS TEST")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 100

    # Simulate price data
    close = np.cumsum(np.random.randn(n) * 0.01) + 1.1000
    high = close + np.abs(np.random.randn(n) * 0.001)
    low = close - np.abs(np.random.randn(n) * 0.001)

    # Create indicator instance
    indicators = BlessingIndicators()

    print("\n[TEST 1] Moving Average Signal")
    ma_signal, ma_value = indicators.ma_signal(close, period=20, ma_distance=5.0)
    print(f"  Signal: {ma_signal} ({'BULLISH' if ma_signal==1 else 'BEARISH' if ma_signal==-1 else 'RANGING'})")
    print(f"  MA Value: {ma_value:.5f}")
    print(f"  Current Price: {close[-1]:.5f}")

    print("\n[TEST 2] CCI Signal")
    cci_signal, cci_value = indicators.cci_signal(high, low, close, period=14)
    print(f"  Signal: {cci_signal} ({'BULLISH' if cci_signal==1 else 'BEARISH' if cci_signal==-1 else 'NEUTRAL'})")
    print(f"  CCI Value: {cci_value:.2f}")

    print("\n[TEST 3] Bollinger Bands Signal")
    bb_signal, (bb_upper, bb_middle, bb_lower) = indicators.bollinger_signal(
        close, period=15, deviation=2.0
    )
    print(f"  Signal: {bb_signal} ({'SELL' if bb_signal==-1 else 'BUY' if bb_signal==1 else 'INSIDE'})")
    print(f"  Upper: {bb_upper:.5f}")
    print(f"  Middle: {bb_middle:.5f}")
    print(f"  Lower: {bb_lower:.5f}")
    print(f"  Price: {close[-1]:.5f}")

    print("\n[TEST 4] Stochastic Signal")
    stoch_signal, (k_value, d_value) = indicators.stochastic_signal(
        high, low, close, k_period=10, d_period=2, slowing=2, zone=20
    )
    print(f"  Signal: {stoch_signal} ({'OVERSOLD' if stoch_signal==1 else 'OVERBOUGHT' if stoch_signal==-1 else 'NEUTRAL'})")
    print(f"  %K: {k_value:.2f}")
    print(f"  %D: {d_value:.2f}")

    print("\n[TEST 5] MACD Signal")
    macd_signal, (macd_line, signal_line, histogram) = indicators.macd_signal(
        close, fast_period=12, slow_period=26, signal_period=9
    )
    print(f"  Signal: {macd_signal} ({'BULLISH' if macd_signal==1 else 'BEARISH' if macd_signal==-1 else 'NEUTRAL'})")
    print(f"  MACD Line: {macd_line:.5f}")
    print(f"  Signal Line: {signal_line:.5f}")
    print(f"  Histogram: {histogram:.5f}")

    print("\n[TEST 6] Multi-Timeframe CCI")
    # Simulate different TF CCIs
    cci_m5 = 50.0
    cci_m15 = 30.0
    cci_m30 = 45.0
    cci_h1 = 55.0

    mtf_signal = indicators.cci_multi_timeframe_signal(cci_m5, cci_m15, cci_m30, cci_h1)
    print(f"  M5 CCI: {cci_m5}")
    print(f"  M15 CCI: {cci_m15}")
    print(f"  M30 CCI: {cci_m30}")
    print(f"  H1 CCI: {cci_h1}")
    print(f"  MTF Signal: {mtf_signal} ({'ALL BULLISH' if mtf_signal==1 else 'ALL BEARISH' if mtf_signal==-1 else 'MIXED'})")

    print("\n" + "="*60)
    print("[OK] Blessing Indicators Test Completed")
    print("="*60)
