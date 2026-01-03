# optimization/parameter_space.py
# Author: Rafał Wiśniewski | Data & AI Solutions

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from enum import Enum
from datetime import datetime
from pathlib import Path

class ParameterType(Enum):
    """Typy parametrów"""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    CHOICE = "choice"
    ARRAY = "array"

@dataclass
class Parameter:
    """Definicja pojedynczego parametru"""
    name: str
    type: ParameterType
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    optimize: bool = True  # Czy optymalizować ten parametr
    description: str = ""
    
    def validate(self, value: Any) -> bool:
        """Walidacja wartości"""
        if self.type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
        
        elif self.type == ParameterType.INTEGER:
            if not isinstance(value, int):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            return True
        
        elif self.type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            return True
        
        elif self.type == ParameterType.CHOICE:
            return value in self.choices
        
        elif self.type == ParameterType.ARRAY:
            return isinstance(value, (list, tuple, np.ndarray))
        
        return False
    
    def random_value(self) -> Any:
        """Generuj losową wartość w zakresie"""
        if self.type == ParameterType.BOOLEAN:
            return np.random.choice([True, False])
        
        elif self.type == ParameterType.INTEGER:
            return np.random.randint(self.min_value, self.max_value + 1)
        
        elif self.type == ParameterType.FLOAT:
            return np.random.uniform(self.min_value, self.max_value)
        
        elif self.type == ParameterType.CHOICE:
            return np.random.choice(self.choices)
        
        return self.default


class BlessingParameters:
    """
    Definicja wszystkich parametrów Blessing EA
    Zgrupowane według kategorii z manuala
    """
    
    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}
        self._define_parameters()
    
    def _define_parameters(self):
        """Definiuj wszystkie parametry z Blessing EA"""
        
        # ============================================
        # GRID STRUCTURE
        # ============================================
        self.add_parameter(Parameter(
            name="GridSetArray",
            type=ParameterType.ARRAY,
            default=[25, 50, 100],
            description="Grid spacing in pips [level1, level2, level3+]"
        ))
        
        self.add_parameter(Parameter(
            name="TP_SetArray",
            type=ParameterType.ARRAY,
            default=[50, 100, 200],
            description="Take profit in pips for each grid level"
        ))
        
        self.add_parameter(Parameter(
            name="SetCountArray",
            type=ParameterType.ARRAY,
            default=[5, 4],
            description="Number of trades per grid level"
        ))
        
        self.add_parameter(Parameter(
            name="MaxTrades",
            type=ParameterType.INTEGER,
            default=12,
            min_value=5,
            max_value=30,
            step=1,
            description="Maximum number of trades in basket"
        ))
        
        self.add_parameter(Parameter(
            name="AutoCal",
            type=ParameterType.BOOLEAN,
            default=False,
            description="Auto-calculate grid based on ATR"
        ))
        
        self.add_parameter(Parameter(
            name="GAF",
            type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.3,
            max_value=3.0,
            step=0.1,
            description="Grid Adjustment Factor"
        ))
        
        # ============================================
        # ENTRY INDICATORS
        # ============================================
        self.add_parameter(Parameter(
            name="MAEntry",
            type=ParameterType.CHOICE,
            default=1,
            choices=[0, 1, 2],
            description="MA Entry: 0=off, 1=normal, 2=reverse"
        ))
        
        self.add_parameter(Parameter(
            name="MAPeriod",
            type=ParameterType.INTEGER,
            default=100,
            min_value=10,
            max_value=200,
            step=10,
            description="Moving Average period"
        ))
        
        self.add_parameter(Parameter(
            name="MADistance",
            type=ParameterType.INTEGER,
            default=5,
            min_value=0,
            max_value=50,
            step=5,
            description="MA channel distance in pips"
        ))
        
        self.add_parameter(Parameter(
            name="CCIEntry",
            type=ParameterType.CHOICE,
            default=0,
            choices=[0, 1, 2],
            description="CCI Entry: 0=off, 1=normal, 2=reverse"
        ))
        
        self.add_parameter(Parameter(
            name="CCIPeriod",
            type=ParameterType.INTEGER,
            default=14,
            min_value=5,
            max_value=50,
            step=1,
            description="CCI period"
        ))
        
        self.add_parameter(Parameter(
            name="BollingerEntry",
            type=ParameterType.CHOICE,
            default=0,
            choices=[0, 1, 2],
            description="Bollinger Bands Entry"
        ))
        
        self.add_parameter(Parameter(
            name="BollPeriod",
            type=ParameterType.INTEGER,
            default=15,
            min_value=10,
            max_value=50,
            step=5,
            description="Bollinger Bands period"
        ))
        
        self.add_parameter(Parameter(
            name="BollDistance",
            type=ParameterType.INTEGER,
            default=13,
            min_value=5,
            max_value=30,
            step=1,
            description="Bollinger distance"
        ))
        
        self.add_parameter(Parameter(
            name="BollDeviation",
            type=ParameterType.FLOAT,
            default=2.0,
            min_value=1.0,
            max_value=3.0,
            step=0.1,
            description="Bollinger standard deviation"
        ))
        
        self.add_parameter(Parameter(
            name="StochEntry",
            type=ParameterType.CHOICE,
            default=0,
            choices=[0, 1, 2],
            description="Stochastic Entry"
        ))
        
        self.add_parameter(Parameter(
            name="BuySellStochZone",
            type=ParameterType.INTEGER,
            default=20,
            min_value=10,
            max_value=50,
            step=5,
            description="Stochastic overbought/oversold zone"
        ))
        
        self.add_parameter(Parameter(
            name="MACDEntry",
            type=ParameterType.CHOICE,
            default=0,
            choices=[0, 1, 2],
            description="MACD Entry"
        ))
        
        self.add_parameter(Parameter(
            name="UseAnyEntry",
            type=ParameterType.BOOLEAN,
            default=False,
            description="Use any entry signal vs all signals"
        ))
        
        self.add_parameter(Parameter(
            name="EntryDelay",
            type=ParameterType.INTEGER,
            default=240,
            min_value=0,
            max_value=3600,
            step=60,
            description="Entry delay in seconds"
        ))
        
        self.add_parameter(Parameter(
            name="ForceMarketCond",
            type=ParameterType.CHOICE,
            default=0,
            choices=[0, 1, 2],
            description="Force market: 0=auto, 1=long, 2=short"
        ))
        
        # ============================================
        # MONEY MANAGEMENT
        # ============================================
        self.add_parameter(Parameter(
            name="Lot",
            type=ParameterType.FLOAT,
            default=0.01,
            min_value=0.01,
            max_value=10.0,
            step=0.01,
            description="Base lot size"
        ))
        
        self.add_parameter(Parameter(
            name="Multiplier",
            type=ParameterType.FLOAT,
            default=1.4,
            min_value=1.0,
            max_value=3.0,
            step=0.1,
            description="Lot multiplier for each level"
        ))
        
        self.add_parameter(Parameter(
            name="LAF",
            type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.1,
            max_value=5.0,
            step=0.1,
            description="Lot Adjustment Factor"
        ))
        
        self.add_parameter(Parameter(
            name="PortionPC",
            type=ParameterType.INTEGER,
            default=100,
            min_value=10,
            max_value=100,
            step=5,
            description="Portion of account to use (%)"
        ))
        
        # ============================================
        # RISK MANAGEMENT
        # ============================================
        self.add_parameter(Parameter(
            name="MaxDDPercent",
            type=ParameterType.INTEGER,
            default=50,
            min_value=20,
            max_value=80,
            step=5,
            description="Max drawdown % before basket close"
        ))
        
        self.add_parameter(Parameter(
            name="StopTradePercent",
            type=ParameterType.INTEGER,
            default=10,
            min_value=5,
            max_value=30,
            step=5,
            description="Stop trading below this % of balance"
        ))
        
        self.add_parameter(Parameter(
            name="BreakEvenTrade",
            type=ParameterType.INTEGER,
            default=12,
            min_value=5,
            max_value=30,
            step=1,
            description="Close at breakeven at this trade count"
        ))
        
        self.add_parameter(Parameter(
            name="BEPlusPips",
            type=ParameterType.INTEGER,
            default=0,
            min_value=-50,
            max_value=50,
            step=5,
            description="Breakeven + pips"
        ))
        
        # ============================================
        # PROFIT MANAGEMENT
        # ============================================
        self.add_parameter(Parameter(
            name="MaximizeProfit",
            type=ParameterType.BOOLEAN,
            default=True,
            description="Enable profit trailing stop"
        ))
        
        self.add_parameter(Parameter(
            name="ProfitSet",
            type=ParameterType.INTEGER,
            default=70,
            min_value=50,
            max_value=95,
            step=5,
            description="Profit trailing stop % of potential"
        ))
        
        self.add_parameter(Parameter(
            name="MoveTP",
            type=ParameterType.INTEGER,
            default=0,
            min_value=0,
            max_value=100,
            step=10,
            description="Move TP by pips when trailing"
        ))
        
        self.add_parameter(Parameter(
            name="TotalMoves",
            type=ParameterType.INTEGER,
            default=1,
            min_value=0,
            max_value=5,
            step=1,
            description="Number of TP moves allowed"
        ))
        
        self.add_parameter(Parameter(
            name="ForceTPPips",
            type=ParameterType.INTEGER,
            default=0,
            min_value=0,
            max_value=500,
            step=10,
            description="Force TP to fixed pips (0=disabled)"
        ))
        
        self.add_parameter(Parameter(
            name="MinTPPips",
            type=ParameterType.INTEGER,
            default=10,
            min_value=5,
            max_value=100,
            step=5,
            description="Minimum TP in pips"
        ))
        
        # ============================================
        # ADVANCED FEATURES
        # ============================================
        self.add_parameter(Parameter(
            name="UseSmartGrid",
            type=ParameterType.BOOLEAN,
            default=False,
            description="Use SmartGrid (RSI-based entry)"
        ))
        
        self.add_parameter(Parameter(
            name="UseCloseOldest",
            type=ParameterType.BOOLEAN,
            default=False,
            description="Close oldest trades to reduce DD"
        ))
        
        self.add_parameter(Parameter(
            name="RecoupClosedLoss",
            type=ParameterType.BOOLEAN,
            default=True,
            description="Recoup losses from closed trades"
        ))
        
        self.add_parameter(Parameter(
            name="UseEarlyExit",
            type=ParameterType.BOOLEAN,
            default=False,
            description="Enable early exit strategy"
        ))
        
        self.add_parameter(Parameter(
            name="EEStartHours",
            type=ParameterType.INTEGER,
            default=24,
            min_value=1,
            max_value=168,
            step=1,
            description="Early exit start after hours"
        ))
        
        self.add_parameter(Parameter(
            name="EEHoursPC",
            type=ParameterType.INTEGER,
            default=1,
            min_value=0,
            max_value=10,
            step=1,
            description="Early exit % reduction per hour"
        ))
        
        # ============================================
        # HEDGING
        # ============================================
        self.add_parameter(Parameter(
            name="UseHedge",
            type=ParameterType.BOOLEAN,
            default=False,
            description="Enable hedging"
        ))
        
        self.add_parameter(Parameter(
            name="HedgeSymbol",
            type=ParameterType.CHOICE,
            default="",
            choices=["", "SAME", "CORRELATED"],
            description="Hedge symbol (empty=same pair)"
        ))
        
        self.add_parameter(Parameter(
            name="DDorLevel",
            type=ParameterType.CHOICE,
            default="D",
            choices=["D", "L"],
            description="Hedge trigger: D=drawdown, L=level"
        ))
        
        self.add_parameter(Parameter(
            name="HedgeStart",
            type=ParameterType.INTEGER,
            default=20,
            min_value=10,
            max_value=50,
            step=5,
            description="Hedge start % (DD) or level"
        ))
        
        self.add_parameter(Parameter(
            name="HedgeLotMult",
            type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.5,
            max_value=2.0,
            step=0.1,
            description="Hedge lot multiplier"
        ))
        
        # ============================================
        # STOP LOSS
        # ============================================
        self.add_parameter(Parameter(
            name="UsePowerOutSL",
            type=ParameterType.BOOLEAN,
            default=True,
            description="Send SL to broker (power-out protection)"
        ))
        
        self.add_parameter(Parameter(
            name="POSLPips",
            type=ParameterType.INTEGER,
            default=600,
            min_value=100,
            max_value=2000,
            step=50,
            description="Power-out SL max pips"
        ))
        
        self.add_parameter(Parameter(
            name="UseStopLoss",
            type=ParameterType.BOOLEAN,
            default=False,
            description="Use trailing stop loss"
        ))
        
        self.add_parameter(Parameter(
            name="TSLPips",
            type=ParameterType.INTEGER,
            default=50,
            min_value=10,
            max_value=500,
            step=10,
            description="Trailing stop loss pips"
        ))
        
        # ============================================
        # OTHER
        # ============================================
        self.add_parameter(Parameter(
            name="B3Traditional",
            type=ParameterType.BOOLEAN,
            default=True,
            description="Use traditional pending orders"
        ))
        
        self.add_parameter(Parameter(
            name="MaxSpread",
            type=ParameterType.INTEGER,
            default=50,
            min_value=10,
            max_value=200,
            step=10,
            description="Max spread to allow trading (points)"
        ))
        
        self.add_parameter(Parameter(
            name="Slip",
            type=ParameterType.INTEGER,
            default=3,
            min_value=0,
            max_value=20,
            step=1,
            description="Slippage tolerance (points)"
        ))
    
    def add_parameter(self, param: Parameter):
        """Dodaj parametr do space"""
        self.parameters[param.name] = param
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """Pobierz parametr"""
        return self.parameters.get(name)
    
    def get_optimizable_parameters(self) -> Dict[str, Parameter]:
        """Pobierz tylko parametry do optymalizacji"""
        return {k: v for k, v in self.parameters.items() if v.optimize}
    
    def set_optimize_flags(self, categories: List[str]):
        """
        Ustaw które kategorie parametrów optymalizować
        
        Categories:
        - grid_structure
        - entry_indicators
        - money_management
        - risk_management
        - profit_management
        - advanced
        - hedging
        - stop_loss
        """
        category_params = {
            'grid_structure': ['GridSetArray', 'TP_SetArray', 'SetCountArray', 'MaxTrades', 'AutoCal', 'GAF'],
            'entry_indicators': ['MAEntry', 'MAPeriod', 'MADistance', 'CCIEntry', 'CCIPeriod', 
                                'BollingerEntry', 'BollPeriod', 'BollDistance', 'StochEntry', 
                                'MACDEntry', 'UseAnyEntry', 'EntryDelay', 'ForceMarketCond'],
            'money_management': ['Lot', 'Multiplier', 'LAF', 'PortionPC'],
            'risk_management': ['MaxDDPercent', 'StopTradePercent', 'BreakEvenTrade', 'BEPlusPips'],
            'profit_management': ['MaximizeProfit', 'ProfitSet', 'MoveTP', 'TotalMoves', 'ForceTPPips', 'MinTPPips'],
            'advanced': ['UseSmartGrid', 'UseCloseOldest', 'RecoupClosedLoss', 'UseEarlyExit', 'EEStartHours', 'EEHoursPC'],
            'hedging': ['UseHedge', 'HedgeSymbol', 'DDorLevel', 'HedgeStart', 'HedgeLotMult'],
            'stop_loss': ['UsePowerOutSL', 'POSLPips', 'UseStopLoss', 'TSLPips']
        }
        
        # Disable all first
        for param in self.parameters.values():
            param.optimize = False
        
        # Enable selected categories
        for category in categories:
            if category in category_params:
                for param_name in category_params[category]:
                    if param_name in self.parameters:
                        self.parameters[param_name].optimize = True
    
    def generate_random_config(self) -> Dict[str, Any]:
        """Generuj losową konfigurację parametrów"""
        config = {}
        for name, param in self.parameters.items():
            if param.optimize:
                config[name] = param.random_value()
            else:
                config[name] = param.default
        return config
    
    def to_set_file(self, config: Dict[str, Any], output_path: str):
        """
        Eksportuj konfigurację do pliku .set MT4/MT5
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-16') as f:
            f.write("; Blessing EA Optimized Settings\n")
            f.write("; Author: Rafał Wiśniewski | Data & AI Solutions\n")
            f.write("; Generated: " + str(datetime.now()) + "\n\n")
            
            for name, value in config.items():
                param = self.get_parameter(name)
                if not param:
                    continue
                
                # Format value based on type
                if param.type == ParameterType.BOOLEAN:
                    value_str = "true" if value else "false"
                elif param.type == ParameterType.ARRAY:
                    value_str = ",".join(map(str, value))
                else:
                    value_str = str(value)
                
                f.write(f"{name}={value_str}\n")
        
        print(f"💾 Saved .set file: {output_path}")


if __name__ == "__main__":
    # Test
    params = BlessingParameters()
    
    print(f"Total parameters: {len(params.parameters)}")
    print(f"Optimizable: {len(params.get_optimizable_parameters())}")
    
    # Enable specific categories
    params.set_optimize_flags(['grid_structure', 'risk_management'])
    print(f"After filtering: {len(params.get_optimizable_parameters())}")
    
    # Generate random config
    config = params.generate_random_config()
    print(f"\nRandom config sample:")
    for k, v in list(config.items())[:5]:
        print(f"  {k}: {v}")