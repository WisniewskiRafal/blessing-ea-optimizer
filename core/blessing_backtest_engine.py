# core/blessing_backtest_engine.py
# Blessing EA Backtest Engine with Grid Management
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

from indicators.blessing_indicators import BlessingIndicators
from strategies.blessing_entry_generator import BlessingEntryGenerator, EntryConfig
from strategies.blessing_grid_system import BlessingGridSystem, GridConfig


@dataclass
class BacktestConfig:
    """Complete Blessing EA backtest configuration"""
    # Entry configuration
    entry_config: EntryConfig

    # Indicator parameters
    indicator_params: Dict = field(default_factory=dict)

    # Grid configuration
    grid_config: GridConfig = field(default_factory=GridConfig)

    # Money management
    initial_deposit: float = 1000.0
    max_drawdown_percent: float = 30.0  # Stop trading if DD > 30%
    equity_protection: bool = False  # Close all if equity < threshold

    # Risk settings
    max_total_trades: int = 20  # Maximum open trades (buy + sell)
    max_lot_total: float = 1.0  # Maximum total lot size

    # Exit settings
    maximize_profit: bool = False  # Don't close on small profit
    use_stop_loss: bool = False
    stop_loss_pips: float = 0.0
    early_exit_hours: int = 0  # Close all after X hours

    # Break Even
    use_break_even: bool = False
    break_even_pips: float = 50.0


@dataclass
class Trade:
    """Single trade"""
    ticket: int  # Unique ID
    open_time: int  # Bar index
    open_price: float
    lot_size: float
    is_buy: bool
    grid_level: int
    is_open: bool = True
    close_time: Optional[int] = None
    close_price: Optional[float] = None
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0


@dataclass
class BacktestResult:
    """Backtest results"""
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float

    # Drawdown
    max_drawdown: float
    max_drawdown_percent: float

    # Risk metrics
    sharpe_ratio: float
    recovery_factor: float  # Net profit / Max DD

    # Equity curve
    equity_curve: np.ndarray
    balance_curve: np.ndarray

    # Trade list
    trades: List[Trade]

    # Extra info
    final_balance: float
    bars_tested: int


class BlessingBacktestEngine:
    """
    Blessing EA Backtest Engine

    Features:
    - Grid trading with dynamic levels
    - Lot multiplication per level
    - TP synchronization across grid
    - Break Even handling
    - Equity protection
    - MaximizeProfit mode
    - Comprehensive metrics
    """

    def __init__(self,
                 config: BacktestConfig,
                 point_value: float = 10.0,  # $10 per pip for 0.01 lot
                 commission_per_lot: float = 0.0,  # Commission per lot
                 verbose: bool = False):
        """
        Args:
            config: Complete backtest configuration
            point_value: Dollar value per pip for 0.01 lot
            commission_per_lot: Commission per 1.0 lot (both sides)
            verbose: Print progress
        """
        self.config = config
        self.point_value = point_value
        self.commission_per_lot = commission_per_lot
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)

        # Components
        self.indicators = BlessingIndicators()
        self.entry_generator = BlessingEntryGenerator()
        self.grid_system = BlessingGridSystem(config.grid_config)

        # State
        self.trades: List[Trade] = []
        self.next_ticket = 1
        self.balance = config.initial_deposit
        self.equity = config.initial_deposit
        self.max_equity = config.initial_deposit

        # Equity curve tracking
        self.equity_history: List[float] = []
        self.balance_history: List[float] = []

    def run_backtest(self,
                    high: np.ndarray,
                    low: np.ndarray,
                    close: np.ndarray) -> BacktestResult:
        """
        Run complete backtest on OHLC data

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            BacktestResult with all metrics
        """
        n_bars = len(close)

        # Reset state
        self._reset()

        # Bar-by-bar simulation
        for bar in range(50, n_bars):  # Start after indicators warm up
            current_price = close[bar]

            # 1. Calculate entry signal
            signal = self.entry_generator.calculate_entry_signal(
                self.config.entry_config,
                high[:bar+1],
                low[:bar+1],
                close[:bar+1],
                self.config.indicator_params
            )

            # 2. Check for TP hit (close grid if hit)
            self._check_take_profit(current_price, bar)

            # 3. Open new grid levels if signal present
            if signal != 0:
                self._process_entry_signal(signal, current_price, bar, high[:bar+1], low[:bar+1], close[:bar+1])

            # 4. Update equity (floating P&L)
            self._update_equity(current_price)

            # 5. Check equity protection / max DD stop
            if self._check_stop_conditions():
                self.logger.warning(f"[STOP] Max DD or equity protection triggered at bar {bar}")
                self._close_all_trades(current_price, bar, reason="stop_condition")
                break

            # 6. Record equity curve
            self.equity_history.append(self.equity)
            self.balance_history.append(self.balance)

        # Close any remaining open trades at end
        if self._count_open_trades() > 0:
            self._close_all_trades(close[-1], n_bars-1, reason="backtest_end")

        # Calculate results
        result = self._calculate_results(n_bars)

        return result

    def _process_entry_signal(self,
                              signal: int,
                              price: float,
                              bar: int,
                              high: np.ndarray,
                              low: np.ndarray,
                              close: np.ndarray):
        """Process entry signal and open grid levels"""
        is_buy = (signal == 1)

        # Check if we can open new trades (risk limits)
        if not self._can_open_trade(is_buy):
            return

        # Calculate grid distance (dynamic if AutoCal/SmartGrid enabled)
        grid_distance = self.grid_system.calculate_grid_distance(close, high, low, bar)

        # Apply SmartGrid adjustment
        smart_adjustment = self.grid_system.smart_grid_adjustment(close, high, low, bar)
        grid_distance *= smart_adjustment

        # Check if next grid level should be opened
        if self.grid_system.should_open_next_level(price, is_buy, bar, grid_distance):
            # Open new grid level
            grid_level = self.grid_system.open_grid_level(price, is_buy, bar)

            # Create trade
            trade = Trade(
                ticket=self.next_ticket,
                open_time=bar,
                open_price=price,
                lot_size=grid_level.lot_size,
                is_buy=is_buy,
                grid_level=grid_level.level,
                is_open=True
            )

            # Commission
            trade.commission = -self.commission_per_lot * (trade.lot_size / 1.0) * 2  # Both open/close

            self.trades.append(trade)
            self.next_ticket += 1

    def _check_take_profit(self, price: float, bar: int):
        """Check if TP is hit and close grid"""
        tp_hit, grid_type = self.grid_system.check_take_profit(price, self.point_value)

        if tp_hit:
            # Close all trades of this grid type
            self._close_grid_trades(price, bar, grid_type)

            # Close grid in grid system
            self.grid_system.close_all_levels(grid_type)

    def _close_grid_trades(self, price: float, bar: int, grid_type: str):
        """Close all trades belonging to grid"""
        for trade in self.trades:
            if not trade.is_open:
                continue

            # Check if trade belongs to this grid
            if grid_type == 'buy' and trade.is_buy:
                self._close_trade(trade, price, bar)
            elif grid_type == 'sell' and not trade.is_buy:
                self._close_trade(trade, price, bar)

    def _close_all_trades(self, price: float, bar: int, reason: str = "manual"):
        """Close all open trades"""
        for trade in self.trades:
            if trade.is_open:
                self._close_trade(trade, price, bar)

        # Close all grids
        self.grid_system.close_all_levels('both')

    def _close_trade(self, trade: Trade, price: float, bar: int):
        """Close single trade and update balance"""
        trade.close_price = price
        trade.close_time = bar
        trade.is_open = False

        # Calculate P&L
        if trade.is_buy:
            pips = (price - trade.open_price) / self.config.grid_config.pip_value
        else:
            pips = (trade.open_price - price) / self.config.grid_config.pip_value

        # Profit in dollars
        trade.profit = pips * (trade.lot_size / 0.01) * self.point_value

        # Update balance
        self.balance += trade.profit + trade.commission + trade.swap

    def _update_equity(self, price: float):
        """Update equity (balance + floating P&L)"""
        floating_pnl = self.grid_system.calculate_floating_pnl(price, self.point_value)

        self.equity = self.balance + floating_pnl['total_pnl']

        # Track max equity
        if self.equity > self.max_equity:
            self.max_equity = self.equity

    def _can_open_trade(self, is_buy: bool) -> bool:
        """Check if we can open new trade (risk limits)"""
        # Check if balance is positive (no trading if broke)
        if self.balance <= 0:
            return False

        # Check max trades
        if self._count_open_trades() >= self.config.max_total_trades:
            return False

        # Check max lot
        total_lot = sum(t.lot_size for t in self.trades if t.is_open)
        next_level = len(self.grid_system.buy_grid if is_buy else self.grid_system.sell_grid)
        next_lot = self.config.grid_config.base_lot * (self.config.grid_config.lot_multiplier ** next_level)

        if total_lot + next_lot > self.config.max_lot_total:
            return False

        return True

    def _check_stop_conditions(self) -> bool:
        """Check if stop conditions met (max DD, equity protection)"""
        # Max drawdown check
        drawdown_percent = (self.max_equity - self.equity) / self.max_equity * 100

        if drawdown_percent > self.config.max_drawdown_percent:
            return True

        # Equity protection
        if self.config.equity_protection:
            if self.equity < self.config.initial_deposit * 0.5:  # Lost 50% of capital
                return True

        return False

    def _count_open_trades(self) -> int:
        """Count open trades"""
        return sum(1 for t in self.trades if t.is_open)

    def _reset(self):
        """Reset backtest state"""
        self.trades = []
        self.next_ticket = 1
        self.balance = self.config.initial_deposit
        self.equity = self.config.initial_deposit
        self.max_equity = self.config.initial_deposit
        self.equity_history = []
        self.balance_history = []
        self.grid_system.reset()

    def _calculate_results(self, n_bars: int) -> BacktestResult:
        """Calculate backtest metrics"""
        closed_trades = [t for t in self.trades if not t.is_open]

        if not closed_trades:
            # No trades
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
                gross_profit=0.0, gross_loss=0.0, net_profit=0.0, profit_factor=0.0,
                max_drawdown=0.0, max_drawdown_percent=0.0,
                sharpe_ratio=0.0, recovery_factor=0.0,
                equity_curve=np.array(self.equity_history),
                balance_curve=np.array(self.balance_history),
                trades=self.trades,
                final_balance=self.balance,
                bars_tested=n_bars
            )

        # Trade statistics
        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t.profit > 0)
        losing_trades = sum(1 for t in closed_trades if t.profit < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # P&L
        gross_profit = sum(t.profit for t in closed_trades if t.profit > 0)
        gross_loss = abs(sum(t.profit for t in closed_trades if t.profit < 0))
        net_profit = self.balance - self.config.initial_deposit
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Drawdown
        equity_curve = np.array(self.equity_history)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = running_max - equity_curve
        max_drawdown = np.max(drawdown)
        max_drawdown_percent = (max_drawdown / self.max_equity * 100) if self.max_equity > 0 else 0.0

        # Sharpe ratio (returns / std)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Recovery factor
        recovery_factor = net_profit / max_drawdown if max_drawdown > 0 else 0.0

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            recovery_factor=recovery_factor,
            equity_curve=equity_curve,
            balance_curve=np.array(self.balance_history),
            trades=self.trades,
            final_balance=self.balance,
            bars_tested=n_bars
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("BLESSING BACKTEST ENGINE TEST")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 1000
    close = np.cumsum(np.random.randn(n) * 0.01) + 1.1000
    high = close + np.abs(np.random.randn(n) * 0.001)
    low = close - np.abs(np.random.randn(n) * 0.001)

    # Create configuration
    entry_config = EntryConfig(
        ma_entry=1,  # MA normal
        cci_entry=0,  # CCI off
        bollinger_entry=0,
        stoch_entry=0,
        macd_entry=0,
        b3_traditional=False,  # Instant entry
        force_market_cond=3,  # Off
        use_any_entry=False  # All must agree (only MA active)
    )

    indicator_params = {
        'ma_period': 20,
        'ma_distance': 5.0
    }

    grid_config = GridConfig(
        grid_set_01=25.0,
        pip_value=0.0001,
        base_lot=0.01,
        lot_multiplier=2.0,
        max_trades=5,
        take_profit=50.0,
        tp_in_money=False
    )

    backtest_config = BacktestConfig(
        entry_config=entry_config,
        indicator_params=indicator_params,
        grid_config=grid_config,
        initial_deposit=1000.0,
        max_drawdown_percent=30.0,
        max_total_trades=10,
        max_lot_total=1.0
    )

    # Create engine
    engine = BlessingBacktestEngine(
        config=backtest_config,
        point_value=10.0,
        commission_per_lot=0.0,
        verbose=True
    )

    # Run backtest
    print("\n[TEST] Running backtest on 1000 bars...")
    result = engine.run_backtest(high, low, close)

    # Print trade details
    print("\n[TRADES]")
    for i, trade in enumerate(result.trades[:10]):  # First 10 trades
        status = "OPEN" if trade.is_open else "CLOSED"
        direction = "BUY" if trade.is_buy else "SELL"
        print(f"  #{i+1} {direction} Level{trade.grid_level} Lot:{trade.lot_size:.2f} " +
              f"Open:{trade.open_price:.5f} Close:{trade.close_price if trade.close_price else 'N/A'} " +
              f"P/L: ${trade.profit:.2f} [{status}]")

    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total trades: {result.total_trades}")
    print(f"Winning trades: {result.winning_trades}")
    print(f"Losing trades: {result.losing_trades}")
    print(f"Win rate: {result.win_rate*100:.1f}%")
    print(f"\nGross profit: ${result.gross_profit:.2f}")
    print(f"Gross loss: ${result.gross_loss:.2f}")
    print(f"Net profit: ${result.net_profit:.2f}")
    print(f"Profit factor: {result.profit_factor:.2f}")
    print(f"\nMax drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_percent:.1f}%)")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"Recovery factor: {result.recovery_factor:.2f}")
    print(f"\nFinal balance: ${result.final_balance:.2f}")
    print(f"Initial deposit: $1000.00")
    print(f"Return: {((result.final_balance/1000.0 - 1) * 100):.1f}%")
    print("="*60)

    print("\n[OK] Blessing Backtest Engine Test Completed")
