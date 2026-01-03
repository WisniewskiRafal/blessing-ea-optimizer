# strategies/blessing_grid_system.py
# Blessing EA Grid Management System
# Author: Rafał Wiśniewski | Data & AI Solutions

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class GridLevel:
    """Single grid level"""
    level: int  # 0=entry, 1, 2, 3...
    price: float  # Entry price
    lot_size: float  # Lot size at this level
    is_buy: bool  # True=BUY, False=SELL
    is_open: bool = True  # Is position open?
    entry_time: int = 0  # Bar index when opened


@dataclass
class GridConfig:
    """Grid configuration parameters"""
    # Grid spacing
    grid_set_01: float = 25.0  # Grid distance in pips (default 25)
    pip_value: float = 0.0001  # Pip value (0.01 for JPY pairs)

    # AutoCal (ATR-based automatic grid sizing)
    auto_cal: bool = False
    atr_period: int = 14
    atr_multiplier: float = 1.0

    # GAF (Grid Adjustment Factor) - widens/squishes grid
    gaf: float = 1.0  # 1.0=normal, >1=wider, <1=tighter

    # SmartGrid (RSI/MA based intelligent placement)
    smart_grid: bool = False
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Entry delay
    entry_delay: int = 0  # Bars to wait before opening next level

    # Lot management
    lot_multiplier: float = 2.0  # Each level multiplies lot (default 2.0)
    base_lot: float = 0.01  # Starting lot size
    max_trades: int = 10  # Maximum grid levels

    # TP management
    tp_in_money: bool = False  # False=pips, True=dollars
    take_profit: float = 50.0  # TP in pips or dollars


class BlessingGridSystem:
    """
    Blessing EA Grid Management System

    Features:
    - Dynamic grid array management
    - Lot multiplication per level (default 2.0x)
    - TP synchronization across all trades
    - AutoCal (ATR-based grid sizing)
    - GAF (Grid Adjustment Factor)
    - SmartGrid (RSI/MA based placement)
    - Entry delay
    """

    def __init__(self, config: GridConfig = None):
        self.config = config or GridConfig()
        self.logger = logging.getLogger(__name__)

        # Grid state
        self.buy_grid: List[GridLevel] = []
        self.sell_grid: List[GridLevel] = []

        # Statistics
        self.total_buy_lots = 0.0
        self.total_sell_lots = 0.0
        self.avg_buy_price = 0.0
        self.avg_sell_price = 0.0

    def reset(self):
        """Reset grid state"""
        self.buy_grid = []
        self.sell_grid = []
        self.total_buy_lots = 0.0
        self.total_sell_lots = 0.0
        self.avg_buy_price = 0.0
        self.avg_sell_price = 0.0

    def calculate_grid_distance(self,
                                close: np.ndarray,
                                high: np.ndarray = None,
                                low: np.ndarray = None,
                                current_bar: int = -1) -> float:
        """
        Calculate grid distance in price units

        Args:
            close: Close prices
            high: High prices (for ATR)
            low: Low prices (for ATR)
            current_bar: Current bar index

        Returns:
            Grid distance in price units
        """
        base_distance = self.config.grid_set_01 * self.config.pip_value

        # AutoCal: Use ATR for dynamic grid sizing
        if self.config.auto_cal and high is not None and low is not None:
            atr = self._calculate_atr(high, low, close, self.config.atr_period)

            if not np.isnan(atr[current_bar]):
                base_distance = atr[current_bar] * self.config.atr_multiplier

        # GAF: Grid Adjustment Factor
        adjusted_distance = base_distance * self.config.gaf

        return adjusted_distance

    def should_open_next_level(self,
                               current_price: float,
                               is_buy: bool,
                               current_bar: int,
                               grid_distance: float) -> bool:
        """
        Check if next grid level should be opened

        Args:
            current_price: Current market price
            is_buy: True=checking BUY grid, False=SELL grid
            current_bar: Current bar index
            grid_distance: Grid distance in price

        Returns:
            True if next level should open
        """
        grid = self.buy_grid if is_buy else self.sell_grid

        # No open positions - can open first level
        if not grid:
            return True

        # Max trades reached
        if len(grid) >= self.config.max_trades:
            return False

        # Get last opened level
        last_level = grid[-1]

        # Entry delay check
        if self.config.entry_delay > 0:
            bars_since_entry = current_bar - last_level.entry_time
            if bars_since_entry < self.config.entry_delay:
                return False

        # Price movement check
        if is_buy:
            # BUY grid: open next level when price drops
            price_distance = last_level.price - current_price
        else:
            # SELL grid: open next level when price rises
            price_distance = current_price - last_level.price

        # Open if price moved >= grid_distance
        return price_distance >= grid_distance

    def open_grid_level(self,
                       price: float,
                       is_buy: bool,
                       current_bar: int) -> GridLevel:
        """
        Open new grid level

        Args:
            price: Entry price
            is_buy: True=BUY, False=SELL
            current_bar: Current bar index

        Returns:
            Created GridLevel
        """
        grid = self.buy_grid if is_buy else self.sell_grid

        # Calculate lot size (multiply by level)
        level_num = len(grid)
        lot_size = self.config.base_lot * (self.config.lot_multiplier ** level_num)

        # Create level
        level = GridLevel(
            level=level_num,
            price=price,
            lot_size=lot_size,
            is_buy=is_buy,
            is_open=True,
            entry_time=current_bar
        )

        # Add to grid
        grid.append(level)

        # Update statistics
        self._update_grid_stats()

        return level

    def close_all_levels(self, grid_type: str = 'both') -> List[GridLevel]:
        """
        Close all grid levels (on TP hit)

        Args:
            grid_type: 'buy', 'sell', or 'both'

        Returns:
            List of closed levels
        """
        closed = []

        if grid_type in ['buy', 'both']:
            for level in self.buy_grid:
                level.is_open = False
                closed.append(level)
            self.buy_grid = []

        if grid_type in ['sell', 'both']:
            for level in self.sell_grid:
                level.is_open = False
                closed.append(level)
            self.sell_grid = []

        # Update stats
        self._update_grid_stats()

        return closed

    def calculate_floating_pnl(self,
                              current_price: float,
                              point_value: float = 10.0) -> Dict[str, float]:
        """
        Calculate floating P&L for all open positions

        Args:
            current_price: Current market price
            point_value: Dollar value of 1 pip for 0.01 lot (default $10)

        Returns:
            Dict with buy_pnl, sell_pnl, total_pnl
        """
        buy_pnl = 0.0
        sell_pnl = 0.0

        # BUY positions
        for level in self.buy_grid:
            if level.is_open:
                pips = (current_price - level.price) / self.config.pip_value
                pnl = pips * (level.lot_size / 0.01) * point_value
                buy_pnl += pnl

        # SELL positions
        for level in self.sell_grid:
            if level.is_open:
                pips = (level.price - current_price) / self.config.pip_value
                pnl = pips * (level.lot_size / 0.01) * point_value
                sell_pnl += pnl

        total_pnl = buy_pnl + sell_pnl

        return {
            'buy_pnl': buy_pnl,
            'sell_pnl': sell_pnl,
            'total_pnl': total_pnl
        }

    def check_take_profit(self,
                         current_price: float,
                         point_value: float = 10.0) -> Tuple[bool, str]:
        """
        Check if TP is hit for any grid

        Args:
            current_price: Current market price
            point_value: Dollar value per pip

        Returns:
            (tp_hit, grid_type) where grid_type = 'buy', 'sell', or None
        """
        pnl = self.calculate_floating_pnl(current_price, point_value)

        if self.config.tp_in_money:
            # TP in dollars
            target = self.config.take_profit

            if pnl['buy_pnl'] >= target and self.buy_grid:
                return True, 'buy'
            if pnl['sell_pnl'] >= target and self.sell_grid:
                return True, 'sell'
        else:
            # TP in pips
            # Calculate average TP price for each grid
            if self.buy_grid and self.avg_buy_price > 0:
                tp_price = self.avg_buy_price + (self.config.take_profit * self.config.pip_value)
                if current_price >= tp_price:
                    return True, 'buy'

            if self.sell_grid and self.avg_sell_price > 0:
                tp_price = self.avg_sell_price - (self.config.take_profit * self.config.pip_value)
                if current_price <= tp_price:
                    return True, 'sell'

        return False, None

    def smart_grid_adjustment(self,
                             close: np.ndarray,
                             high: np.ndarray,
                             low: np.ndarray,
                             current_bar: int) -> float:
        """
        SmartGrid: Adjust grid distance based on RSI/MA

        Args:
            close: Close prices
            high: High prices
            low: Low prices
            current_bar: Current bar index

        Returns:
            Adjustment multiplier (0.5-2.0)
        """
        if not self.config.smart_grid:
            return 1.0

        # Calculate RSI
        rsi = self._calculate_rsi(close, self.config.rsi_period)
        current_rsi = rsi[current_bar]

        if np.isnan(current_rsi):
            return 1.0

        # SmartGrid logic:
        # - Overbought (RSI > 70): Widen grid for SELL (expect reversal)
        # - Oversold (RSI < 30): Widen grid for BUY (expect reversal)
        # - Neutral: Normal grid

        if current_rsi > self.config.rsi_overbought:
            # Overbought - widen grid (safer)
            return 1.5
        elif current_rsi < self.config.rsi_oversold:
            # Oversold - widen grid (safer)
            return 1.5
        else:
            # Neutral - normal grid
            return 1.0

    def get_grid_info(self) -> Dict:
        """Get current grid state info"""
        return {
            'buy_levels': len(self.buy_grid),
            'sell_levels': len(self.sell_grid),
            'total_buy_lots': self.total_buy_lots,
            'total_sell_lots': self.total_sell_lots,
            'avg_buy_price': self.avg_buy_price,
            'avg_sell_price': self.avg_sell_price,
            'buy_grid': [
                {
                    'level': l.level,
                    'price': l.price,
                    'lot': l.lot_size,
                    'open': l.is_open
                }
                for l in self.buy_grid
            ],
            'sell_grid': [
                {
                    'level': l.level,
                    'price': l.price,
                    'lot': l.lot_size,
                    'open': l.is_open
                }
                for l in self.sell_grid
            ]
        }

    # ===================================================================
    # INTERNAL HELPER FUNCTIONS
    # ===================================================================

    def _update_grid_stats(self):
        """Update grid statistics (avg price, total lots)"""
        # BUY grid
        if self.buy_grid:
            total_volume = sum(l.lot_size for l in self.buy_grid if l.is_open)
            if total_volume > 0:
                weighted_price = sum(l.price * l.lot_size for l in self.buy_grid if l.is_open)
                self.avg_buy_price = weighted_price / total_volume
                self.total_buy_lots = total_volume
            else:
                self.avg_buy_price = 0.0
                self.total_buy_lots = 0.0
        else:
            self.avg_buy_price = 0.0
            self.total_buy_lots = 0.0

        # SELL grid
        if self.sell_grid:
            total_volume = sum(l.lot_size for l in self.sell_grid if l.is_open)
            if total_volume > 0:
                weighted_price = sum(l.price * l.lot_size for l in self.sell_grid if l.is_open)
                self.avg_sell_price = weighted_price / total_volume
                self.total_sell_lots = total_volume
            else:
                self.avg_sell_price = 0.0
                self.total_sell_lots = 0.0
        else:
            self.avg_sell_price = 0.0
            self.total_sell_lots = 0.0

    def _calculate_atr(self,
                      high: np.ndarray,
                      low: np.ndarray,
                      close: np.ndarray,
                      period: int) -> np.ndarray:
        """Calculate Average True Range"""
        import pandas as pd

        # True Range
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))

        tr = np.maximum(high_low, np.maximum(high_close, low_close))

        # ATR (EMA of TR)
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values

        return atr

    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI"""
        import pandas as pd

        # Price changes
        delta = np.diff(close, prepend=close[0])

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Average gains and losses (EMA)
        avg_gains = pd.Series(gains).ewm(span=period, adjust=False).mean().values
        avg_losses = pd.Series(losses).ewm(span=period, adjust=False).mean().values

        # RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("BLESSING GRID SYSTEM TEST")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 100
    close = np.cumsum(np.random.randn(n) * 0.01) + 1.1000
    high = close + np.abs(np.random.randn(n) * 0.001)
    low = close - np.abs(np.random.randn(n) * 0.001)

    # Create grid system
    config = GridConfig(
        grid_set_01=25.0,
        pip_value=0.0001,
        base_lot=0.01,
        lot_multiplier=2.0,
        max_trades=5,
        take_profit=50.0,
        tp_in_money=False
    )

    grid = BlessingGridSystem(config)

    print("\n[TEST 1] Grid distance calculation")
    grid_distance = grid.calculate_grid_distance(close, high, low)
    print(f"  Base grid: {config.grid_set_01} pips")
    print(f"  Grid distance: {grid_distance:.5f}")
    print(f"  In pips: {grid_distance / config.pip_value:.1f}")

    print("\n[TEST 2] Open BUY grid levels")
    current_price = close[50]
    current_bar = 50

    # Open first level
    if grid.should_open_next_level(current_price, True, current_bar, grid_distance):
        level = grid.open_grid_level(current_price, True, current_bar)
        print(f"  Level {level.level}: Price={level.price:.5f}, Lot={level.lot_size:.2f}")

    # Simulate grid opening as price drops
    for i in range(4):
        # Simulate price drop
        current_price -= grid_distance
        current_bar += 1

        if grid.should_open_next_level(current_price, True, current_bar, grid_distance):
            level = grid.open_grid_level(current_price, True, current_bar)
            print(f"  Level {level.level}: Price={level.price:.5f}, Lot={level.lot_size:.2f}")

    print("\n[TEST 3] Grid statistics")
    info = grid.get_grid_info()
    print(f"  Buy levels: {info['buy_levels']}")
    print(f"  Total buy lots: {info['total_buy_lots']:.2f}")
    print(f"  Avg buy price: {info['avg_buy_price']:.5f}")

    print("\n[TEST 4] Floating P&L calculation")
    test_price = info['avg_buy_price'] + 30 * config.pip_value  # +30 pips
    pnl = grid.calculate_floating_pnl(test_price, point_value=10.0)
    print(f"  Test price: {test_price:.5f} (+30 pips from avg)")
    print(f"  Buy P&L: ${pnl['buy_pnl']:.2f}")
    print(f"  Total P&L: ${pnl['total_pnl']:.2f}")

    print("\n[TEST 5] Take Profit check")
    tp_price = info['avg_buy_price'] + config.take_profit * config.pip_value
    tp_hit, grid_type = grid.check_take_profit(tp_price, point_value=10.0)
    print(f"  TP target: {config.take_profit} pips")
    print(f"  TP price: {tp_price:.5f}")
    print(f"  TP hit: {tp_hit}")
    print(f"  Grid type: {grid_type}")

    if tp_hit:
        print("\n[TEST 6] Close grid on TP")
        closed = grid.close_all_levels(grid_type)
        print(f"  Closed {len(closed)} levels")
        info = grid.get_grid_info()
        print(f"  Remaining buy levels: {info['buy_levels']}")

    print("\n[TEST 7] AutoCal (ATR-based grid)")
    config_autocal = GridConfig(
        auto_cal=True,
        atr_period=14,
        atr_multiplier=1.0,
        pip_value=0.0001
    )
    grid_autocal = BlessingGridSystem(config_autocal)
    atr_distance = grid_autocal.calculate_grid_distance(close, high, low, current_bar=50)
    print(f"  AutoCal grid distance: {atr_distance:.5f}")
    print(f"  In pips: {atr_distance / config.pip_value:.1f}")

    print("\n[TEST 8] SmartGrid (RSI adjustment)")
    config_smart = GridConfig(
        smart_grid=True,
        rsi_period=14,
        rsi_overbought=70.0,
        rsi_oversold=30.0
    )
    grid_smart = BlessingGridSystem(config_smart)
    adjustment = grid_smart.smart_grid_adjustment(close, high, low, current_bar=50)
    print(f"  SmartGrid adjustment: {adjustment:.2f}x")

    print("\n" + "="*60)
    print("[OK] Blessing Grid System Test Completed")
    print("="*60)
