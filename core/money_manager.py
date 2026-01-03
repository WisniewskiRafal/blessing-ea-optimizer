# core/money_manager.py
# Money Management System for Forex Trading
# Author: Rafał Wiśniewski | Data & AI Solutions

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import logging
import numpy as np


@dataclass
class MoneyManagementConfig:
    """Money management configuration"""
    # Account
    initial_deposit: float = 1000.0
    leverage: int = 100
    currency: str = 'USD'

    # Position sizing
    lot_size: float = 0.01              # Fixed 0.01 (micro lot)
    lot_size_mode: str = 'fixed'        # 'fixed' or 'risk_based'
    min_lot_size: float = 0.01
    max_lot_size: float = 0.10

    # Risk limits
    max_drawdown_percent: float = 20.0          # Max 20% DD
    max_daily_loss_percent: float = 5.0         # Max 5% daily loss
    max_consecutive_losses: int = 10            # Max 10 losses in row

    # Stop loss/Take profit
    require_stop_loss: bool = True
    min_stop_loss_pips: int = 10
    max_stop_loss_pips: int = 100
    min_take_profit_pips: int = 10
    max_take_profit_pips: int = 200

    # Risk per trade (for risk_based mode)
    risk_per_trade_percent: float = 2.0         # 2% per trade


@dataclass
class TradeRisk:
    """Trade risk calculation result"""
    lot_size: float
    risk_amount: float              # Dollar amount at risk
    risk_percent: float             # Percent of equity at risk
    margin_required: float          # Margin needed
    max_loss: float                 # Max loss if SL hit
    max_profit: float               # Max profit if TP hit
    risk_reward_ratio: float        # TP/SL ratio


class MoneyManager:
    """
    Money Management System

    Handles:
    - Position sizing (fixed or risk-based)
    - Risk calculations
    - Circuit breakers (DD limits, daily loss)
    - Trade validation
    """

    def __init__(self, config: Optional[MoneyManagementConfig] = None):
        """
        Args:
            config: Money management configuration
        """
        self.config = config or MoneyManagementConfig()
        self.logger = logging.getLogger(__name__)

        # Instrument specifications (EURUSD)
        self.pip_value_per_micro_lot = 0.10  # $0.10 per pip for 0.01 lot
        self.contract_size = 100000          # Standard lot size

    def calculate_lot_size(self,
                          equity: float,
                          stop_loss_pips: int,
                          instrument: str = 'EURUSD') -> float:
        """
        Calculate lot size based on risk

        Args:
            equity: Current account equity
            stop_loss_pips: Stop loss in pips
            instrument: Trading instrument (default EURUSD)

        Returns:
            Lot size
        """
        if self.config.lot_size_mode == 'fixed':
            return self.config.lot_size

        # Risk-based sizing
        risk_amount = equity * (self.config.risk_per_trade_percent / 100.0)

        # Calculate required lot size
        lot_size = risk_amount / (stop_loss_pips * self.pip_value_per_micro_lot)

        # Clamp to limits
        lot_size = max(self.config.min_lot_size, lot_size)
        lot_size = min(self.config.max_lot_size, lot_size)

        return round(lot_size, 2)

    def calculate_trade_risk(self,
                            equity: float,
                            entry_price: float,
                            stop_loss_pips: int,
                            take_profit_pips: int,
                            lot_size: Optional[float] = None,
                            instrument: str = 'EURUSD') -> TradeRisk:
        """
        Calculate comprehensive trade risk

        Args:
            equity: Current equity
            entry_price: Entry price
            stop_loss_pips: SL in pips
            take_profit_pips: TP in pips
            lot_size: Lot size (if None, will calculate)
            instrument: Trading instrument

        Returns:
            TradeRisk object
        """
        # Calculate lot size if not provided
        if lot_size is None:
            lot_size = self.calculate_lot_size(equity, stop_loss_pips, instrument)

        # Calculate max loss/profit
        max_loss = stop_loss_pips * self.pip_value_per_micro_lot * (lot_size / 0.01)
        max_profit = take_profit_pips * self.pip_value_per_micro_lot * (lot_size / 0.01)

        # Calculate margin required
        notional_value = lot_size * self.contract_size * entry_price
        margin_required = notional_value / self.config.leverage

        # Risk percent
        risk_percent = (max_loss / equity) * 100.0

        # Risk/Reward
        risk_reward = max_profit / max_loss if max_loss > 0 else 0.0

        return TradeRisk(
            lot_size=lot_size,
            risk_amount=max_loss,
            risk_percent=risk_percent,
            margin_required=margin_required,
            max_loss=max_loss,
            max_profit=max_profit,
            risk_reward_ratio=risk_reward
        )

    def calculate_pnl(self,
                     entry_price: float,
                     exit_price: float,
                     direction: str,
                     lot_size: Optional[float] = None) -> float:
        """
        Calculate P&L for a trade

        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: 'long' or 'short'
            lot_size: Lot size (default: config.lot_size)

        Returns:
            P&L in USD
        """
        if lot_size is None:
            lot_size = self.config.lot_size

        # Calculate pips
        if direction == 'long':
            pips = (exit_price - entry_price) * 10000
        elif direction == 'short':
            pips = (entry_price - exit_price) * 10000
        else:
            raise ValueError(f"Invalid direction: {direction}")

        # Calculate P&L
        pnl = pips * self.pip_value_per_micro_lot * (lot_size / 0.01)

        return pnl

    def validate_trade(self,
                      equity: float,
                      stop_loss_pips: int,
                      take_profit_pips: int,
                      lot_size: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate if trade meets risk requirements

        Args:
            equity: Current equity
            stop_loss_pips: SL in pips
            take_profit_pips: TP in pips
            lot_size: Lot size

        Returns:
            (is_valid, reason)
        """
        # Check SL requirements
        if self.config.require_stop_loss and stop_loss_pips == 0:
            return False, "Stop loss is required"

        if stop_loss_pips < self.config.min_stop_loss_pips:
            return False, f"SL {stop_loss_pips} < min {self.config.min_stop_loss_pips}"

        if stop_loss_pips > self.config.max_stop_loss_pips:
            return False, f"SL {stop_loss_pips} > max {self.config.max_stop_loss_pips}"

        # Check TP requirements
        if take_profit_pips < self.config.min_take_profit_pips:
            return False, f"TP {take_profit_pips} < min {self.config.min_take_profit_pips}"

        if take_profit_pips > self.config.max_take_profit_pips:
            return False, f"TP {take_profit_pips} > max {self.config.max_take_profit_pips}"

        # Check lot size
        if lot_size is None:
            lot_size = self.config.lot_size

        if lot_size < self.config.min_lot_size:
            return False, f"Lot {lot_size} < min {self.config.min_lot_size}"

        if lot_size > self.config.max_lot_size:
            return False, f"Lot {lot_size} > max {self.config.max_lot_size}"

        return True, "OK"

    def check_circuit_breaker(self,
                             equity_curve: List[float],
                             initial_deposit: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if circuit breaker should trigger (stop trading)

        Args:
            equity_curve: List of equity values
            initial_deposit: Initial deposit (default: config.initial_deposit)

        Returns:
            (should_stop, reason)
        """
        if len(equity_curve) == 0:
            return False, ""

        if initial_deposit is None:
            initial_deposit = self.config.initial_deposit

        current_equity = equity_curve[-1]

        # Check daily loss (if we have >= 1 day of data)
        # Assuming M1 data: 1440 bars = 1 day
        if len(equity_curve) > 1440:
            yesterday_equity = equity_curve[-1440]
            daily_loss_pct = ((current_equity - yesterday_equity) / yesterday_equity) * 100

            if daily_loss_pct < -self.config.max_daily_loss_percent:
                return True, f"Daily loss {abs(daily_loss_pct):.1f}% > limit {self.config.max_daily_loss_percent}%"

        # Check total drawdown
        peak_equity = max(equity_curve)
        if peak_equity > 0:
            current_dd_pct = ((peak_equity - current_equity) / peak_equity) * 100

            if current_dd_pct > self.config.max_drawdown_percent:
                return True, f"Drawdown {current_dd_pct:.1f}% > limit {self.config.max_drawdown_percent}%"

        return False, ""

    def calculate_drawdown(self, equity_curve: List[float]) -> Dict[str, float]:
        """
        Calculate drawdown statistics

        Args:
            equity_curve: List of equity values

        Returns:
            Dictionary with DD stats
        """
        if len(equity_curve) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_percent': 0.0,
                'current_drawdown': 0.0,
                'current_drawdown_percent': 0.0
            }

        equity_array = np.array(equity_curve)

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)

        # Calculate drawdown
        drawdown = equity_array - running_max
        drawdown_pct = (drawdown / running_max) * 100

        # Max drawdown
        max_dd = np.min(drawdown)
        max_dd_pct = np.min(drawdown_pct)

        # Current drawdown
        current_dd = drawdown[-1]
        current_dd_pct = drawdown_pct[-1]

        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_percent': abs(max_dd_pct),
            'current_drawdown': abs(current_dd),
            'current_drawdown_percent': abs(current_dd_pct)
        }

    def check_consecutive_losses(self, trades: List[Dict]) -> Tuple[bool, int]:
        """
        Check for consecutive losses

        Args:
            trades: List of trade dicts with 'pnl' key

        Returns:
            (exceeded_limit, consecutive_count)
        """
        if len(trades) == 0:
            return False, 0

        consecutive = 0
        max_consecutive = 0

        for trade in trades:
            if trade.get('pnl', 0) < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        exceeded = max_consecutive > self.config.max_consecutive_losses

        return exceeded, max_consecutive


if __name__ == "__main__":
    # Test MoneyManager
    import logging
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("MONEY MANAGER TEST")
    print("="*60)

    # Create config
    config = MoneyManagementConfig(
        initial_deposit=1000.0,
        leverage=100,
        lot_size=0.01,
        lot_size_mode='fixed'
    )

    mm = MoneyManager(config)

    # Test 1: Calculate trade risk
    print("\n[TEST 1] Calculate Trade Risk")
    equity = 1000.0
    entry_price = 1.1000
    sl_pips = 30
    tp_pips = 60

    risk = mm.calculate_trade_risk(equity, entry_price, sl_pips, tp_pips)

    print(f"  Equity: ${equity:.2f}")
    print(f"  Entry: {entry_price:.5f}")
    print(f"  SL: {sl_pips} pips, TP: {tp_pips} pips")
    print(f"  Lot Size: {risk.lot_size}")
    print(f"  Max Loss: ${risk.max_loss:.2f} ({risk.risk_percent:.2f}%)")
    print(f"  Max Profit: ${risk.max_profit:.2f}")
    print(f"  Risk/Reward: {risk.risk_reward_ratio:.2f}:1")
    print(f"  Margin: ${risk.margin_required:.2f}")

    # Test 2: Calculate P&L
    print("\n[TEST 2] Calculate P&L")
    pnl_long = mm.calculate_pnl(1.1000, 1.1030, 'long', 0.01)
    pnl_short = mm.calculate_pnl(1.1000, 1.0970, 'short', 0.01)

    print(f"  Long: 1.1000 to 1.1030 = ${pnl_long:.2f}")
    print(f"  Short: 1.1000 to 1.0970 = ${pnl_short:.2f}")

    # Test 3: Validate trade
    print("\n[TEST 3] Validate Trade")
    valid, reason = mm.validate_trade(equity, 30, 60, 0.01)
    print(f"  SL=30, TP=60: {valid} - {reason}")

    valid, reason = mm.validate_trade(equity, 5, 60, 0.01)
    print(f"  SL=5, TP=60: {valid} - {reason}")

    # Test 4: Circuit breaker
    print("\n[TEST 4] Circuit Breaker")
    equity_curve = [1000, 990, 980, 970, 960, 950, 800]  # 20% DD
    should_stop, reason = mm.check_circuit_breaker(equity_curve)
    print(f"  Equity curve: {equity_curve}")
    print(f"  Stop trading: {should_stop} - {reason}")

    # Test 5: Drawdown calculation
    print("\n[TEST 5] Drawdown Calculation")
    dd_stats = mm.calculate_drawdown(equity_curve)
    print(f"  Max DD: ${dd_stats['max_drawdown']:.2f} ({dd_stats['max_drawdown_percent']:.1f}%)")
    print(f"  Current DD: ${dd_stats['current_drawdown']:.2f} ({dd_stats['current_drawdown_percent']:.1f}%)")

    print("\n" + "="*60)
    print("[OK] Money Manager Tests Completed")
    print("="*60)
