# core/__init__.py
# Author: Rafał Wiśniewski | Data & AI Solutions

"""
Blessing EA Optimizer - Core Module
Główne komponenty systemu optymalizacji
"""

__version__ = "1.0.0"
__author__ = "Rafał Wiśniewski | Data & AI Solutions"

from .hardware_detector import HardwareDetector
from .mt5_bridge import MT5Bridge
from .mt4_bridge import MT4Bridge
from .data_processor import DataProcessor
from .symbol_mapper import SymbolMapper

__all__ = [
    'HardwareDetector',
    'MT5Bridge', 
    'MT4Bridge',
    'DataProcessor',
    'SymbolMapper'
]