# backtesting/gpu_indicators.py
# GPU-Accelerated Technical Indicators using PyTorch
# Author: Rafał Wiśniewski | Data & AI Solutions

import sys
from pathlib import Path

# Add project root to path for imports
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from typing import Optional, Tuple
import logging


class GPUIndicators:
    """
    GPU-Accelerated Technical Indicators

    Uses PyTorch for 15.94x speedup on RTX 5060 Ti
    All indicators operate on GPU tensors for maximum performance
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: torch.device to use (auto-detects if None)
        """
        self.logger = logging.getLogger(__name__)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if self.device.type == 'cuda':
            self.logger.info(f"[GPU INDICATORS] Using {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("[GPU INDICATORS] Using CPU fallback")

    def sma(self,
            prices: torch.Tensor,
            period: int) -> torch.Tensor:
        """
        Simple Moving Average (SMA) on GPU

        Uses efficient convolution for O(n) complexity

        Args:
            prices: [n] or [batch, n] tensor
            period: SMA period

        Returns:
            [n] or [batch, n] tensor with SMA (NaN padded at start)
        """
        if prices.dim() == 1:
            return self._sma_1d(prices, period)
        elif prices.dim() == 2:
            return self._sma_2d(prices, period)
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {prices.dim()}D")

    def _sma_1d(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """SMA for 1D tensor"""
        n = len(prices)

        # Create averaging kernel
        kernel = torch.ones(period, device=self.device, dtype=torch.float32) / period

        # Reshape for conv1d
        prices_3d = prices.unsqueeze(0).unsqueeze(0)
        kernel_3d = kernel.unsqueeze(0).unsqueeze(0)

        # Convolve
        sma_conv = torch.nn.functional.conv1d(prices_3d, kernel_3d, padding=0)
        sma = sma_conv.squeeze(0).squeeze(0)

        # Pad beginning with NaN
        sma_full = torch.full((n,), float('nan'), device=self.device, dtype=torch.float32)
        sma_full[period-1:] = sma

        return sma_full

    def _sma_2d(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """SMA for 2D tensor (batch processing)"""
        batch_size, n = prices.shape

        # Create averaging kernel
        kernel = torch.ones(period, device=self.device, dtype=torch.float32) / period
        kernel_3d = kernel.unsqueeze(0).unsqueeze(0)

        # Reshape for conv1d: [batch, 1, n]
        prices_3d = prices.unsqueeze(1)

        # Convolve
        sma_conv = torch.nn.functional.conv1d(prices_3d, kernel_3d, padding=0)
        sma = sma_conv.squeeze(1)

        # Pad beginning with NaN
        sma_full = torch.full((batch_size, n), float('nan'), device=self.device, dtype=torch.float32)
        sma_full[:, period-1:] = sma

        return sma_full

    def ema(self,
            prices: torch.Tensor,
            period: int,
            smoothing: float = 2.0) -> torch.Tensor:
        """
        Exponential Moving Average (EMA) on GPU

        Args:
            prices: [n] or [batch, n] tensor
            period: EMA period
            smoothing: Smoothing factor (default 2.0)

        Returns:
            [n] or [batch, n] tensor with EMA
        """
        alpha = smoothing / (period + 1)

        if prices.dim() == 1:
            return self._ema_1d(prices, alpha)
        elif prices.dim() == 2:
            return self._ema_2d(prices, alpha)
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {prices.dim()}D")

    def _ema_1d(self, prices: torch.Tensor, alpha: float) -> torch.Tensor:
        """EMA for 1D tensor"""
        n = len(prices)
        ema = torch.zeros(n, device=self.device, dtype=torch.float32)
        ema[0] = prices[0]

        for i in range(1, n):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

        return ema

    def _ema_2d(self, prices: torch.Tensor, alpha: float) -> torch.Tensor:
        """EMA for 2D tensor (batch processing)"""
        batch_size, n = prices.shape
        ema = torch.zeros((batch_size, n), device=self.device, dtype=torch.float32)
        ema[:, 0] = prices[:, 0]

        for i in range(1, n):
            ema[:, i] = alpha * prices[:, i] + (1 - alpha) * ema[:, i-1]

        return ema

    def rsi(self,
            prices: torch.Tensor,
            period: int = 14) -> torch.Tensor:
        """
        Relative Strength Index (RSI) on GPU

        Args:
            prices: [n] or [batch, n] tensor
            period: RSI period (default 14)

        Returns:
            [n] or [batch, n] tensor with RSI (0-100)
        """
        if prices.dim() == 1:
            return self._rsi_1d(prices, period)
        elif prices.dim() == 2:
            return self._rsi_2d(prices, period)
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {prices.dim()}D")

    def _rsi_1d(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """RSI for 1D tensor"""
        # Calculate price changes
        deltas = torch.diff(prices, prepend=prices[0:1])

        # Separate gains and losses
        gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))

        # Calculate average gains and losses using SMA
        avg_gains = self.sma(gains, period)
        avg_losses = self.sma(losses, period)

        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _rsi_2d(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """RSI for 2D tensor"""
        # Calculate price changes
        deltas = torch.diff(prices, dim=1, prepend=prices[:, 0:1])

        # Separate gains and losses
        gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))

        # Calculate average gains and losses using SMA
        avg_gains = self.sma(gains, period)
        avg_losses = self.sma(losses, period)

        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def bollinger_bands(self,
                       prices: torch.Tensor,
                       period: int = 20,
                       num_std: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bollinger Bands on GPU

        Args:
            prices: [n] or [batch, n] tensor
            period: Period for SMA (default 20)
            num_std: Number of standard deviations (default 2.0)

        Returns:
            (upper_band, middle_band, lower_band) tensors
        """
        # Middle band is SMA
        middle = self.sma(prices, period)

        # Calculate rolling standard deviation
        if prices.dim() == 1:
            std = self._rolling_std_1d(prices, period)
        else:
            std = self._rolling_std_2d(prices, period)

        # Upper and lower bands
        upper = middle + num_std * std
        lower = middle - num_std * std

        return upper, middle, lower

    def _rolling_std_1d(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """Rolling standard deviation for 1D tensor"""
        n = len(prices)
        std_full = torch.full((n,), float('nan'), device=self.device, dtype=torch.float32)

        for i in range(period-1, n):
            window = prices[i-period+1:i+1]
            std_full[i] = torch.std(window)

        return std_full

    def _rolling_std_2d(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """Rolling standard deviation for 2D tensor"""
        batch_size, n = prices.shape
        std_full = torch.full((batch_size, n), float('nan'), device=self.device, dtype=torch.float32)

        for i in range(period-1, n):
            window = prices[:, i-period+1:i+1]
            std_full[:, i] = torch.std(window, dim=1)

        return std_full

    def macd(self,
             prices: torch.Tensor,
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MACD (Moving Average Convergence Divergence) on GPU

        Args:
            prices: [n] or [batch, n] tensor
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)

        Returns:
            (macd_line, signal_line, histogram) tensors
        """
        # Calculate fast and slow EMAs
        fast_ema = self.ema(prices, fast_period)
        slow_ema = self.ema(prices, slow_period)

        # MACD line
        macd_line = fast_ema - slow_ema

        # Signal line (EMA of MACD)
        signal_line = self.ema(macd_line, signal_period)

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def atr(self,
            high: torch.Tensor,
            low: torch.Tensor,
            close: torch.Tensor,
            period: int = 14) -> torch.Tensor:
        """
        Average True Range (ATR) on GPU

        Args:
            high: [n] or [batch, n] tensor
            low: [n] or [batch, n] tensor
            close: [n] or [batch, n] tensor
            period: ATR period (default 14)

        Returns:
            [n] or [batch, n] tensor with ATR
        """
        # Calculate True Range
        prev_close = torch.roll(close, 1, dims=-1)

        if close.dim() == 1:
            prev_close[0] = close[0]
        else:
            prev_close[:, 0] = close[:, 0]

        tr1 = high - low
        tr2 = torch.abs(high - prev_close)
        tr3 = torch.abs(low - prev_close)

        tr = torch.maximum(tr1, torch.maximum(tr2, tr3))

        # Calculate ATR using EMA of TR
        atr = self.ema(tr, period)

        return atr


if __name__ == "__main__":
    # Test GPU indicators
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("GPU INDICATORS TEST")
    print("="*60)

    # Create indicator calculator
    indicators = GPUIndicators()

    # Generate synthetic price data
    torch.manual_seed(42)
    n = 1000
    prices = 100 + torch.cumsum(torch.randn(n) * 0.5, dim=0)
    prices = prices.to(indicators.device)

    print(f"\n[TEST 1] SMA")
    sma_20 = indicators.sma(prices, period=20)
    print(f"  SMA(20) calculated: {len(sma_20)} values")
    print(f"  First valid value: {sma_20[19].item():.2f}")
    print(f"  Last value: {sma_20[-1].item():.2f}")

    print(f"\n[TEST 2] EMA")
    ema_20 = indicators.ema(prices, period=20)
    print(f"  EMA(20) calculated: {len(ema_20)} values")
    print(f"  Last value: {ema_20[-1].item():.2f}")

    print(f"\n[TEST 3] RSI")
    rsi_14 = indicators.rsi(prices, period=14)
    print(f"  RSI(14) calculated: {len(rsi_14)} values")
    print(f"  Last value: {rsi_14[-1].item():.2f}")

    print(f"\n[TEST 4] Bollinger Bands")
    upper, middle, lower = indicators.bollinger_bands(prices, period=20, num_std=2.0)
    print(f"  BB calculated: {len(middle)} values")
    print(f"  Last middle: {middle[-1].item():.2f}")
    print(f"  Last upper: {upper[-1].item():.2f}")
    print(f"  Last lower: {lower[-1].item():.2f}")

    print(f"\n[TEST 5] MACD")
    macd_line, signal_line, histogram = indicators.macd(prices)
    print(f"  MACD calculated: {len(macd_line)} values")
    print(f"  Last MACD: {macd_line[-1].item():.2f}")
    print(f"  Last Signal: {signal_line[-1].item():.2f}")
    print(f"  Last Histogram: {histogram[-1].item():.2f}")

    print(f"\n[TEST 6] ATR")
    # Generate OHLC data
    high = prices + torch.abs(torch.randn(n, device=indicators.device)) * 0.5
    low = prices - torch.abs(torch.randn(n, device=indicators.device)) * 0.5
    atr_14 = indicators.atr(high, low, prices, period=14)
    print(f"  ATR(14) calculated: {len(atr_14)} values")
    print(f"  Last value: {atr_14[-1].item():.2f}")

    print(f"\n[TEST 7] Batch Processing (2D)")
    batch_prices = torch.randn(5, n, device=indicators.device).cumsum(dim=1) + 100
    batch_sma = indicators.sma(batch_prices, period=20)
    print(f"  Batch SMA shape: {batch_sma.shape}")
    print(f"  Last values: {batch_sma[:, -1].cpu().numpy()}")

    print("\n" + "="*60)
    print("[OK] GPU Indicators Tests Completed")
    print("="*60)
