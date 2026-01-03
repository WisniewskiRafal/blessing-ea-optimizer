# optimization/metrics.py
# Author: Rafał Wiśniewski | Data & AI Solutions

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class BacktestResults:
    """Wyniki pojedynczego backtestu"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    
    initial_deposit: float = 10000.0
    final_balance: float = 10000.0
    
    # Trade statistics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Time statistics
    total_days: int = 0
    avg_trade_duration_hours: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Equity curve
    equity_curve: Optional[pd.Series] = None
    
    # Trade log
    trades: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        else:
            self.win_rate = 0.0


class PerformanceMetrics:
    """
    Kalkulacja metryk wydajności tradingowej
    Sharpe, Sortino, Profit Factor, Recovery Factor, etc.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_metrics(self, results: BacktestResults) -> Dict[str, float]:
        """Oblicz wszystkie metryki"""
        
        metrics = {
            # Basic metrics
            'net_profit': results.net_profit,
            'total_trades': results.total_trades,
            'win_rate': results.win_rate,
            'profit_factor': self.profit_factor(results),
            
            # Risk metrics
            'max_drawdown': results.max_drawdown,
            'max_drawdown_percent': results.max_drawdown_percent,
            'recovery_factor': self.recovery_factor(results),
            
            # Risk-adjusted returns
            'sharpe_ratio': self.sharpe_ratio(results),
            'sortino_ratio': self.sortino_ratio(results),
            'calmar_ratio': self.calmar_ratio(results),
            
            # Trade quality
            'avg_win_loss_ratio': self.avg_win_loss_ratio(results),
            'expectancy': self.expectancy(results),
            'kelly_criterion': self.kelly_criterion(results),
            
            # Consistency
            'profit_stability': self.profit_stability(results),
            'trade_efficiency': self.trade_efficiency(results),
            
            # Time-based
            'annual_return': self.annual_return(results),
            'monthly_return': self.monthly_return(results),
        }
        
        return metrics
    
    def profit_factor(self, results: BacktestResults) -> float:
        """
        Profit Factor = Gross Profit / Gross Loss
        >1.0 = profitable, >2.0 = excellent
        """
        if results.gross_loss == 0:
            return float('inf') if results.gross_profit > 0 else 0.0
        
        return results.gross_profit / abs(results.gross_loss)
    
    def recovery_factor(self, results: BacktestResults) -> float:
        """
        Recovery Factor = Net Profit / Max Drawdown
        Measures ability to recover from losses
        """
        if results.max_drawdown == 0:
            return float('inf') if results.net_profit > 0 else 0.0
        
        return results.net_profit / abs(results.max_drawdown)
    
    def sharpe_ratio(self, results: BacktestResults, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe Ratio = (Return - RiskFreeRate) / StdDev(Returns)
        Annualized, risk-free rate default 2%
        
        >1.0 = good, >2.0 = very good, >3.0 = excellent
        """
        if results.equity_curve is None or len(results.equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        returns = results.equity_curve.pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Annualize
        periods_per_year = self._get_periods_per_year(results)
        
        mean_return = returns.mean() * periods_per_year
        std_return = returns.std() * np.sqrt(periods_per_year)
        
        return (mean_return - risk_free_rate) / std_return
    
    def sortino_ratio(self, results: BacktestResults, risk_free_rate: float = 0.02) -> float:
        """
        Sortino Ratio = (Return - RiskFreeRate) / Downside Deviation
        Like Sharpe but only penalizes downside volatility
        """
        if results.equity_curve is None or len(results.equity_curve) < 2:
            return 0.0
        
        returns = results.equity_curve.pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        periods_per_year = self._get_periods_per_year(results)
        
        mean_return = returns.mean() * periods_per_year
        downside_dev = downside_std * np.sqrt(periods_per_year)
        
        return (mean_return - risk_free_rate) / downside_dev
    
    def calmar_ratio(self, results: BacktestResults) -> float:
        """
        Calmar Ratio = Annual Return / Max Drawdown
        >3.0 = excellent
        """
        if results.max_drawdown_percent == 0:
            return float('inf') if results.net_profit > 0 else 0.0
        
        annual_ret = self.annual_return(results)
        
        return annual_ret / (results.max_drawdown_percent / 100)
    
    def avg_win_loss_ratio(self, results: BacktestResults) -> float:
        """Average Win / Average Loss"""
        if results.avg_loss == 0:
            return float('inf') if results.avg_win > 0 else 0.0
        
        return results.avg_win / abs(results.avg_loss)
    
    def expectancy(self, results: BacktestResults) -> float:
        """
        Expectancy = (WinRate * AvgWin) - (LossRate * AvgLoss)
        Average profit per trade
        """
        if results.total_trades == 0:
            return 0.0
        
        win_rate = results.win_rate
        loss_rate = 1 - win_rate
        
        return (win_rate * results.avg_win) - (loss_rate * abs(results.avg_loss))
    
    def kelly_criterion(self, results: BacktestResults) -> float:
        """
        Kelly % = W - [(1-W) / R]
        where W = win rate, R = avg_win/avg_loss
        
        Optimal position size (fraction of capital)
        """
        if results.avg_loss == 0 or results.total_trades == 0:
            return 0.0
        
        W = results.win_rate
        R = results.avg_win / abs(results.avg_loss)
        
        kelly = W - ((1 - W) / R)
        
        # Kelly can be negative (don't trade) or >1 (over-leverage)
        return max(0.0, min(kelly, 1.0))  # Clamp to [0, 1]
    
    def profit_stability(self, results: BacktestResults) -> float:
        """
        Coefficient of variation of equity curve
        Lower = more stable
        """
        if results.equity_curve is None or len(results.equity_curve) < 2:
            return float('inf')
        
        mean = results.equity_curve.mean()
        std = results.equity_curve.std()
        
        if mean == 0:
            return float('inf')
        
        return std / mean
    
    def trade_efficiency(self, results: BacktestResults) -> float:
        """
        Net Profit / Gross Profit
        Measures how much profit is retained vs given back
        """
        if results.gross_profit == 0:
            return 0.0
        
        return results.net_profit / results.gross_profit
    
    def annual_return(self, results: BacktestResults) -> float:
        """Annualized return %"""
        if results.total_days == 0:
            return 0.0
        
        years = results.total_days / 365.25
        
        if years == 0:
            return 0.0
        
        total_return = (results.final_balance - results.initial_deposit) / results.initial_deposit
        
        return (total_return / years) * 100
    
    def monthly_return(self, results: BacktestResults) -> float:
        """Average monthly return %"""
        return self.annual_return(results) / 12
    
    def max_adverse_excursion(self, results: BacktestResults) -> float:
        """
        MAE - maksymalna strata w trakcie trwania trade'u
        Wymaga szczegółowego trade log
        """
        if results.trades is None or 'mae' not in results.trades.columns:
            return 0.0
        
        return results.trades['mae'].min()
    
    def max_favorable_excursion(self, results: BacktestResults) -> float:
        """
        MFE - maksymalny zysk w trakcie trwania trade'u
        """
        if results.trades is None or 'mfe' not in results.trades.columns:
            return 0.0
        
        return results.trades['mfe'].max()
    
    def _get_periods_per_year(self, results: BacktestResults) -> int:
        """Estimate trading periods per year from data"""
        if results.equity_curve is None:
            return 252  # Default daily
        
        # Estimate from equity curve frequency
        if len(results.equity_curve) < 2:
            return 252
        
        # Simple heuristic based on total days and curve length
        if results.total_days > 0:
            bars_per_day = len(results.equity_curve) / results.total_days
            
            if bars_per_day >= 20:  # Hourly or faster
                return 252 * 24
            elif bars_per_day >= 4:  # 4H
                return 252 * 6
            else:  # Daily
                return 252
        
        return 252
    
    def calculate_custom_score(self, results: BacktestResults,
                               weights: Optional[Dict[str, float]] = None) -> float:
        """
        Composite score z wagami
        
        Default weights:
        - Sharpe: 40%
        - Profit Factor: 30%
        - Recovery Factor: 20%
        - Win Rate: 10%
        """
        if weights is None:
            weights = {
                'sharpe_ratio': 0.40,
                'profit_factor': 0.30,
                'recovery_factor': 0.20,
                'win_rate': 0.10
            }
        
        metrics = self.calculate_all_metrics(results)
        
        # Normalize metrics to [0, 1] scale
        normalized = {
            'sharpe_ratio': self._normalize_sharpe(metrics['sharpe_ratio']),
            'profit_factor': self._normalize_pf(metrics['profit_factor']),
            'recovery_factor': self._normalize_rf(metrics['recovery_factor']),
            'win_rate': metrics['win_rate']
        }
        
        # Weighted sum
        score = sum(normalized[k] * weights[k] for k in weights.keys())
        
        return score
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe to [0, 1], where 3.0 = 1.0"""
        return min(max(sharpe / 3.0, 0.0), 1.0)
    
    def _normalize_pf(self, pf: float) -> float:
        """Normalize PF to [0, 1], where 3.0 = 1.0"""
        if pf == float('inf'):
            return 1.0
        return min(max((pf - 1.0) / 2.0, 0.0), 1.0)
    
    def _normalize_rf(self, rf: float) -> float:
        """Normalize RF to [0, 1], where 5.0 = 1.0"""
        if rf == float('inf'):
            return 1.0
        return min(max(rf / 5.0, 0.0), 1.0)
    
    def compare_results(self, results_list: List[BacktestResults],
                       metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """Porównaj wiele wyników"""
        
        comparison = []
        
        for i, result in enumerate(results_list):
            metrics = self.calculate_all_metrics(result)
            metrics['config_id'] = i
            comparison.append(metrics)
        
        df = pd.DataFrame(comparison)
        df = df.sort_values(metric, ascending=False)
        
        return df
    
    def generate_report(self, results: BacktestResults) -> str:
        """Wygeneruj tekstowy raport"""
        
        metrics = self.calculate_all_metrics(results)
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║        BACKTEST PERFORMANCE REPORT                       ║
║        Rafał Wiśniewski | Data & AI Solutions           ║
╚══════════════════════════════════════════════════════════╝

SUMMARY
-------
Net Profit:              ${metrics['net_profit']:,.2f}
Total Trades:            {results.total_trades}
Win Rate:                {metrics['win_rate']*100:.1f}%

PROFITABILITY
-------------
Profit Factor:           {metrics['profit_factor']:.2f}
Recovery Factor:         {metrics['recovery_factor']:.2f}
Expectancy:              ${metrics['expectancy']:.2f}
Annual Return:           {metrics['annual_return']:.1f}%

RISK METRICS
------------
Max Drawdown:            ${results.max_drawdown:,.2f} ({results.max_drawdown_percent:.1f}%)
Sharpe Ratio:            {metrics['sharpe_ratio']:.2f}
Sortino Ratio:           {metrics['sortino_ratio']:.2f}
Calmar Ratio:            {metrics['calmar_ratio']:.2f}

TRADE STATISTICS
----------------
Avg Win:                 ${results.avg_win:.2f}
Avg Loss:                ${results.avg_loss:.2f}
Win/Loss Ratio:          {metrics['avg_win_loss_ratio']:.2f}
Largest Win:             ${results.largest_win:.2f}
Largest Loss:            ${results.largest_loss:.2f}

CONSISTENCY
-----------
Trade Efficiency:        {metrics['trade_efficiency']*100:.1f}%
Kelly Criterion:         {metrics['kelly_criterion']*100:.1f}%
Max Consecutive Wins:    {results.max_consecutive_wins}
Max Consecutive Losses:  {results.max_consecutive_losses}

═══════════════════════════════════════════════════════════
"""
        
        return report


if __name__ == "__main__":
    # Test with dummy data
    
    # Create fake equity curve
    np.random.seed(42)
    equity = pd.Series(10000 + np.cumsum(np.random.randn(1000) * 100))
    
    results = BacktestResults(
        total_trades=100,
        winning_trades=60,
        losing_trades=40,
        gross_profit=15000,
        gross_loss=-5000,
        net_profit=10000,
        max_drawdown=-2000,
        max_drawdown_percent=20,
        initial_deposit=10000,
        final_balance=20000,
        avg_win=250,
        avg_loss=-125,
        largest_win=1000,
        largest_loss=-500,
        total_days=365,
        max_consecutive_wins=8,
        max_consecutive_losses=5,
        equity_curve=equity
    )
    
    metrics = PerformanceMetrics()
    
    # Calculate all metrics
    all_metrics = metrics.calculate_all_metrics(results)
    
    print("\n📊 All Metrics:")
    for k, v in all_metrics.items():
        print(f"  {k:25} = {v:.3f}")
    
    # Generate report
    print(metrics.generate_report(results))
    
    # Custom score
    score = metrics.calculate_custom_score(results)
    print(f"\n🎯 Custom Score: {score:.3f}")