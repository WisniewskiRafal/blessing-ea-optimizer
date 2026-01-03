# backtesting/__init__.py
# Author: Rafał Wiśniewski | Data & AI Solutions

"""
Blessing EA Optimizer - Backtesting Module
Automatyzacja testów MT4/MT5 Strategy Tester
"""

__version__ = "1.0.0"
__author__ = "Rafał Wiśniewski | Data & AI Solutions"

from .mt4_tester import MT4Tester
from .mt5_tester import MT5Tester
from .parallel_runner import ParallelBacktestRunner

__all__ = [
    'MT4Tester',
    'MT5Tester',
    'ParallelBacktestRunner'
]