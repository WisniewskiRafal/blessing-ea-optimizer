# backtesting/mt5_tester.py
# Author: Rafał Wiśniewski | Data & AI Solutions

import MetaTrader5 as mt5
import time
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from optimization.metrics import BacktestResults


class MT5Tester:
    """
    Automatyzacja MT5 Strategy Tester
    
    UWAGA: MT5 API ma ograniczone wsparcie dla Strategy Tester.
    Wykorzystuje kombinację:
    1. MQL5 script do uruchomienia testu
    2. Monitorowanie plików wynikowych
    """
    
    def __init__(self, 
                 mt5_path: Optional[Path] = None,
                 ea_name: str = "Blessing_3_v3_9_6_09"):
        """
        Args:
            mt5_path: Ścieżka do katalogu MT5 (None = auto-detect)
            ea_name: Nazwa EA (bez .ex5)
        """
        self.mt5_path = Path(mt5_path) if mt5_path else None
        self.ea_name = ea_name
        self.logger = logging.getLogger(__name__)
        
        # MT5 directories
        if self.mt5_path:
            self.terminal_exe = self.mt5_path / "terminal64.exe"
            self.tester_dir = self.mt5_path / "tester"
        else:
            # Will be set after MT5 initialization
            self.terminal_exe = None
            self.tester_dir = None
        
        self.connected = False
    
    def connect(self, login: Optional[int] = None, 
                password: Optional[str] = None,
                server: Optional[str] = None) -> bool:
        """Połącz z MT5"""
        try:
            # Initialize MT5
            if self.mt5_path:
                if not mt5.initialize(path=str(self.mt5_path)):
                    self.logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize():
                    self.logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
            
            # Get terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info:
                self.mt5_path = Path(terminal_info.data_path).parent
                self.terminal_exe = self.mt5_path / "terminal64.exe"
                self.tester_dir = Path(terminal_info.data_path) / "MQL5" / "tester"
            
            # Login if credentials provided
            if login and password and server:
                if not mt5.login(login=login, password=password, server=server):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
            
            self.connected = True
            self.logger.info(f"✅ Connected to MT5")
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Rozłącz MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 disconnected")
    
    def create_set_file(self, config: Dict, set_file_path: Path):
        """
        Stwórz plik .set dla MT5
        
        Format MT5 .set (podobny do MT4 ale UTF-8):
        GridSetArray=25,50,100||25||50||100
        MaxTrades=12||12||1||30||N
        """
        set_file_path = Path(set_file_path)
        set_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(set_file_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"; Blessing EA Parameters\n")
            f.write(f"; Author: Rafał Wiśniewski | Data & AI Solutions\n")
            f.write(f"; Created: {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Parameters in MT5 format
            for key, value in config.items():
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                elif isinstance(value, (list, tuple)):
                    # MT5 array format: value||default||min||max||step
                    value_str = ",".join(map(str, value))
                elif isinstance(value, float):
                    value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                
                f.write(f"{key}={value_str}\n")
        
        self.logger.info(f"📝 Created MT5 .set file: {set_file_path}")
    
    def create_test_script(self, 
                          symbol: str,
                          timeframe: str,
                          date_from: datetime,
                          date_to: datetime,
                          deposit: float = 10000.0,
                          leverage: int = 100,
                          set_file: Optional[Path] = None) -> Path:
        """
        Stwórz MQL5 script do uruchomienia backtestu
        
        MT5 nie ma pełnej API do testera, więc używamy scriptu MQL5
        """
        # Timeframe mapping
        tf_map = {
            'M1': 'PERIOD_M1', 'M5': 'PERIOD_M5', 'M15': 'PERIOD_M15',
            'M30': 'PERIOD_M30', 'H1': 'PERIOD_H1', 'H4': 'PERIOD_H4',
            'D1': 'PERIOD_D1', 'W1': 'PERIOD_W1', 'MN1': 'PERIOD_MN1'
        }
        
        period = tf_map.get(timeframe.upper(), 'PERIOD_H1')
        
        # Create script
        script_code = f'''//+------------------------------------------------------------------+
//|                                          BlessingBacktest.mq5     |
//|                           Rafał Wiśniewski | Data & AI Solutions |
//+------------------------------------------------------------------+
#property copyright "Rafał Wiśniewski"
#property link      "github.com/RafalWisniewski"
#property version   "1.0"
#property script_show_inputs

input string Symbol = "{symbol}";
input ENUM_TIMEFRAMES Timeframe = {period};
input datetime DateFrom = D'{date_from.strftime("%Y.%m.%d")}';
input datetime DateTo = D'{date_to.strftime("%Y.%m.%d")}';
input double Deposit = {deposit};
input int Leverage = {leverage};

//+------------------------------------------------------------------+
void OnStart()
{{
    Print("Starting backtest...");
    Print("Symbol: ", Symbol);
    Print("Timeframe: ", EnumToString(Timeframe));
    Print("Period: ", DateFrom, " to ", DateTo);
    
    // Note: MT5 Strategy Tester must be run manually
    // This script prepares the configuration
    
    // Save config to file
    int file_handle = FileOpen("backtest_config.txt", FILE_WRITE|FILE_TXT);
    if(file_handle != INVALID_HANDLE)
    {{
        FileWriteString(file_handle, "Symbol=" + Symbol + "\\n");
        FileWriteString(file_handle, "Timeframe=" + EnumToString(Timeframe) + "\\n");
        FileWriteString(file_handle, "DateFrom=" + TimeToString(DateFrom) + "\\n");
        FileWriteString(file_handle, "DateTo=" + TimeToString(DateTo) + "\\n");
        FileWriteString(file_handle, "Deposit=" + DoubleToString(Deposit) + "\\n");
        FileWriteString(file_handle, "Leverage=" + IntegerToString(Leverage) + "\\n");
        FileClose(file_handle);
        
        Print("✅ Configuration saved");
    }}
    else
    {{
        Print("❌ Failed to save configuration");
    }}
}}
//+------------------------------------------------------------------+
'''
        
        # Save script
        if not self.tester_dir:
            self.logger.error("MT5 tester directory not set")
            return None
        
        scripts_dir = self.tester_dir.parent.parent / "Scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = scripts_dir / "BlessingBacktest.mq5"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_code)
        
        self.logger.info(f"📝 Created test script: {script_path}")
        return script_path
    
    def run_backtest(self,
                    symbol: str,
                    timeframe: str,
                    date_from: datetime,
                    date_to: datetime,
                    config: Dict,
                    deposit: float = 10000.0,
                    leverage: int = 100,
                    model: int = 0,
                    optimization: bool = False) -> Optional[BacktestResults]:
        """
        Uruchom backtest w MT5
        
        Args:
            symbol: Para walutowa
            timeframe: M1/M5/M15/M30/H1/H4/D1
            date_from: Data początkowa
            date_to: Data końcowa
            config: Parametry EA
            deposit: Depozyt startowy
            leverage: Dźwignia
            model: 0=Every tick, 1=1 minute OHLC, 2=Open prices, 3=Math calculations
            optimization: Czy tryb optymalizacji
        
        Returns:
            BacktestResults lub None
        """
        self.logger.info(f"🚀 Starting MT5 backtest: {symbol} {timeframe}")
        
        if not self.connected:
            self.logger.error("❌ MT5 not connected")
            return None
        
        # Create set file
        if self.tester_dir:
            set_file = self.tester_dir / "sets" / f"{self.ea_name}_temp.set"
            set_file.parent.mkdir(parents=True, exist_ok=True)
            self.create_set_file(config, set_file)
        
        # UWAGA: MT5 Strategy Tester wymaga manualnego uruchomienia
        # lub użycia zewnętrznych narzędzi
        
        self.logger.warning("⚠️ MT5 Strategy Tester API is limited")
        self.logger.warning("⚠️ Please run backtest manually in MT5 MetaEditor")
        self.logger.warning("⚠️ Or use workaround with MQL5 automation")
        
        # Workaround: Monitor results directory
        results = self._wait_for_results(timeout=600)
        
        return results
    
    def _wait_for_results(self, timeout: int = 600) -> Optional[BacktestResults]:
        """
        Czekaj na wyniki testu (monitoruj katalog)
        """
        if not self.tester_dir:
            return None
        
        results_dir = self.tester_dir
        
        self.logger.info(f"⏳ Waiting for results (timeout={timeout}s)...")
        
        start_time = time.time()
        last_file_count = 0
        
        while time.time() - start_time < timeout:
            # Check for XML/HTM reports
            xml_files = list(results_dir.glob("*.xml"))
            htm_files = list(results_dir.glob("*.htm"))
            
            current_count = len(xml_files) + len(htm_files)
            
            if current_count > last_file_count:
                # New file appeared
                self.logger.info("📊 New report detected")
                time.sleep(2)  # Wait for file to be fully written
                
                # Parse latest
                if xml_files:
                    latest = max(xml_files, key=lambda p: p.stat().st_mtime)
                    return self.parse_xml_report(latest)
                elif htm_files:
                    latest = max(htm_files, key=lambda p: p.stat().st_mtime)
                    return self.parse_html_report(latest)
            
            last_file_count = current_count
            time.sleep(5)
        
        self.logger.error(f"❌ Timeout waiting for results")
        return None
    
    def parse_xml_report(self, report_path: Path) -> Optional[BacktestResults]:
        """
        Parse raportu XML z MT5
        """
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            results = BacktestResults()
            
            # Parse XML structure
            # MT5 XML format varies, adjust as needed
            
            for elem in root.iter():
                tag = elem.tag.lower()
                
                if tag == 'initial_deposit':
                    results.initial_deposit = float(elem.text or 0)
                
                elif tag == 'total_net_profit':
                    results.net_profit = float(elem.text or 0)
                
                elif tag == 'gross_profit':
                    results.gross_profit = float(elem.text or 0)
                
                elif tag == 'gross_loss':
                    results.gross_loss = float(elem.text or 0)
                
                elif tag == 'max_drawdown':
                    results.max_drawdown = -abs(float(elem.text or 0))
                
                elif tag == 'max_drawdown_percent':
                    results.max_drawdown_percent = float(elem.text or 0)
                
                elif tag == 'total_trades':
                    results.total_trades = int(elem.text or 0)
            
            self.logger.info(f"✅ Parsed XML: Net Profit=${results.net_profit:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ XML parse error: {str(e)}")
            return None
    
    def parse_html_report(self, report_path: Path) -> Optional[BacktestResults]:
        """
        Parse raportu HTML z MT5
        Podobny format jak MT4
        """
        from bs4 import BeautifulSoup
        import re
        
        try:
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                html = f.read()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            results = BacktestResults()
            
            # Extract from table
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    
                    if 'Initial deposit' in key:
                        results.initial_deposit = self._parse_float(value)
                    
                    elif 'Total net profit' in key:
                        results.net_profit = self._parse_float(value)
                    
                    elif 'Gross profit' in key:
                        results.gross_profit = self._parse_float(value)
                    
                    elif 'Gross loss' in key:
                        results.gross_loss = self._parse_float(value)
                    
                    elif 'Total trades' in key:
                        results.total_trades = self._parse_int(value)
                    
                    elif 'Maximal drawdown' in key:
                        # Format: "2000.00 (20.00%)"
                        match = re.search(r'([\d.]+)\s*\(([\d.]+)%\)', value)
                        if match:
                            results.max_drawdown = -abs(float(match.group(1)))
                            results.max_drawdown_percent = float(match.group(2))
            
            self.logger.info(f"✅ Parsed HTML: Net Profit=${results.net_profit:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ HTML parse error: {str(e)}")
            return None
    
    def _parse_float(self, value: str) -> float:
        """Parse float from string"""
        try:
            clean = re.sub(r'[^\d.-]', '', value)
            return float(clean) if clean else 0.0
        except:
            return 0.0
    
    def _parse_int(self, value: str) -> int:
        """Parse int from string"""
        try:
            clean = re.sub(r'[^\d-]', '', value)
            return int(clean) if clean else 0
        except:
            return 0
    
    def get_optimization_results(self) -> pd.DataFrame:
        """
        Pobierz wyniki optymalizacji z MT5
        
        MT5 zapisuje wyniki w katalogu tester/cache
        """
        if not self.tester_dir:
            return pd.DataFrame()
        
        cache_dir = self.tester_dir / "cache"
        
        if not cache_dir.exists():
            self.logger.warning("Cache directory not found")
            return pd.DataFrame()
        
        # MT5 optimization results are in binary format
        # Would require reverse engineering or MQL5 script to export
        
        self.logger.warning("MT5 optimization results parsing not implemented")
        return pd.DataFrame()
    
    def __enter__(self):
        """Context manager"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.disconnect()


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    with MT5Tester() as tester:
        if tester.connected:
            print("✅ MT5 Tester initialized")
            
            # Test config
            test_config = {
                'MaxTrades': 12,
                'Multiplier': 1.4,
                'GridSetArray': [25, 50, 100]
            }
            
            # Create set file
            test_set = Path("test_blessing_mt5.set")
            tester.create_set_file(test_config, test_set)
            
            print(f"✅ Created test .set file: {test_set}")
            
            # Cleanup
            if test_set.exists():
                test_set.unlink()
        else:
            print("❌ MT5 not available")