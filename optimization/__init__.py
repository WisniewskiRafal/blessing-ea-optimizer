# optimization/__init__.py
# Author: Rafał Wiśniewski | Data & AI Solutions

"""
Blessing EA Optimizer - Optimization Module
Algorytmy optymalizacji parametrów EA
"""

__version__ = "1.0.0"
__author__ = "Rafał Wiśniewski | Data & AI Solutions"

from .parameter_space import BlessingParameters
from .genetic_optimizer import GeneticOptimizer
from .bayesian_optimizer import BayesianOptimizer
from .metrics import PerformanceMetrics

__all__ = [
    'BlessingParameters',
    'GeneticOptimizer',
    'BayesianOptimizer',
    'PerformanceMetrics'
]