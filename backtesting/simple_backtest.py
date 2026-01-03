# backtesting/simple_backtest.py
# Simple CPU-based Backtest for Comparison
# Author: Rafał Wiśniewski | Data & AI Solutions

import numpy as np
from typing import Dict
from optimization.metrics import BacktestResults


def simple_backtest_cpu(config: Dict, ohlc: np.ndarray = None) -> BacktestResults:
    """
    Simple CPU-based backtest for comparison

    Strategy: Moving Average Crossover
    - Long when price > MA
    - Short when price < MA

    Args:
        config: Configuration dict with:
            - ma_period: MA period (default 20)
            - ma_type: 'SMA' or 'EMA' (default 'SMA')
            - initial_balance: Starting balance (default 10000.0)
            - _ohlc_data: OHLC data if not passed separately
        ohlc: OHLC data [n_bars, 4] (optional)

    Returns:
        BacktestResults
    """
    # Get OHLC data
    if ohlc is None:
        ohlc = config.get('_ohlc_data')

    if ohlc is None:
        raise ValueError("No OHLC data provided")

    # Extract parameters
    close = ohlc[:, 3]
    ma_period = config.get('ma_period', 20)
    ma_type = config.get('ma_type', 'SMA')
    initial_balance = config.get('initial_balance', 10000.0)

    # Calculate MA
    if ma_type == 'EMA':
        ma = _calculate_ema(close, ma_period)
    else:  # SMA
        ma = _calculate_sma(close, ma_period)

    # Generate signals
    # 1 = long, -1 = short
    signals = np.where(close > ma, 1, -1)

    # Calculate returns
    returns = np.diff(close) / close[:-1]
    strategy_returns = returns * signals[:-1]

    # Calculate equity curve
    equity_curve = initial_balance * np.cumprod(1 + strategy_returns)

    # Final balance
    final_balance = equity_curve[-1] if len(equity_curve) > 0 else initial_balance
    net_profit = final_balance - initial_balance

    # Count trades (signal changes)
    signal_changes = np.diff(signals)
    total_trades = np.count_nonzero(signal_changes) // 2

    # Calculate winning/losing trades
    trade_pnls = []
    in_trade = False
    entry_price = 0.0
    trade_direction = 0

    for i in range(1, len(signals)):
        if signals[i] != signals[i-1]:
            if in_trade:
                # Close trade
                exit_price = close[i]
                if trade_direction == 1:  # Long
                    pnl = (exit_price - entry_price) / entry_price
                else:  # Short
                    pnl = (entry_price - exit_price) / entry_price
                trade_pnls.append(pnl * initial_balance)
                in_trade = False
            else:
                # Open trade
                entry_price = close[i]
                trade_direction = signals[i]
                in_trade = True

    # Trade statistics
    if trade_pnls:
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        losing_trades = sum(1 for pnl in trade_pnls if pnl < 0)
        gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0.0
        largest_win = max(trade_pnls) if trade_pnls else 0.0
        largest_loss = min(trade_pnls) if trade_pnls else 0.0
    else:
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        largest_win = 0.0
        largest_loss = 0.0

    # Drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - cummax
    max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    max_drawdown_percent = (max_drawdown / initial_balance) * 100 if initial_balance > 0 else 0.0

    # Create result
    results = BacktestResults(
        total_trades=int(total_trades),
        winning_trades=int(winning_trades),
        losing_trades=int(losing_trades),
        gross_profit=float(gross_profit),
        gross_loss=float(gross_loss),
        net_profit=float(net_profit),
        max_drawdown=float(max_drawdown),
        max_drawdown_percent=float(max_drawdown_percent),
        initial_deposit=float(initial_balance),
        final_balance=float(final_balance),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        largest_win=float(largest_win),
        largest_loss=float(largest_loss)
    )

    return results


def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average"""
    n = len(prices)
    ma = np.full(n, np.nan)

    if n < period:
        return ma

    # Use convolution for efficiency
    kernel = np.ones(period) / period
    ma_valid = np.convolve(prices, kernel, mode='valid')

    # Fill from period-1 onwards
    ma[period-1:period-1+len(ma_valid)] = ma_valid

    return ma


def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average"""
    n = len(prices)
    ema = np.full(n, np.nan)

    if n < period:
        return ema

    # EMA multiplier
    multiplier = 2.0 / (period + 1)

    # First EMA is SMA
    ema[period-1] = np.mean(prices[:period])

    # Calculate remaining EMAs
    for i in range(period, n):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]

    return ema
