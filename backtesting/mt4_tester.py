# backtesting/mt4_tester.py
# Author: Rafał Wiśniewski | Data & AI Solutions

import subprocess
import time
import re
import configparser
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd
import logging
from bs4 import BeautifulSoup

from optimization.metrics import BacktestResults


class MT4Tester:
    """
    Automatyzacja MT4 Strategy Tester
    Uruchamianie backtestów i parsowanie wyników
    """
    
    def __init__(self, mt4_path: Path, ea_name: str = "Blessing_3_v3_9_6_09"):
        """
        Args:
            mt4_path: Ścieżka do katalogu MT4
            ea_name: Nazwa EA (bez .ex4)
        """
        self.mt4_path = Path(mt4_path)
        self.ea_name = ea_name
        self.logger = logging.getLogger(__name__)
        
        # Validate paths
        self.terminal_exe = self.mt4_path / "terminal.exe"
        if not self.terminal_exe.exists():
            raise FileNotFoundError(f"MT4 terminal not found: {self.terminal_exe}")
        
        self.config_dir = self.mt4_path / "config"
        self.tester_dir = self.mt4_path / "tester"
        self.reports_dir = self.tester_dir / "reports"
        
        # Create dirs if needed
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.tester_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def create_set_file(self, config: Dict, set_file_path: Path):
        """
        Stwórz plik .set z parametrami EA
        
        Format MT4 .set:
        ; saved automatically on 2024.12.28 10:30
        GridSetArray=25,50,100
        MaxTrades=12
        Multiplier=1.40
        """
        set_file_path = Path(set_file_path)
        set_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(set_file_path, 'w', encoding='utf-16-le') as f:
            # Header
            f.write(f"; Blessing EA Parameters\n")
            f.write(f"; Author: Rafał Wiśniewski | Data & AI Solutions\n")
            f.write(f"; Created: {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Parameters
            for key, value in config.items():
                # Format value based on type
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                elif isinstance(value, (list, tuple)):
                    value_str = ",".join(map(str, value))
                elif isinstance(value, float):
                    value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                
                f.write(f"{key}={value_str}\n")
        
        self.logger.info(f"📝 Created .set file: {set_file_path}")
    
    def configure_tester(self,
                        symbol: str,
                        timeframe: str,
                        date_from: datetime,
                        date_to: datetime,
                        deposit: float = 10000.0,
                        leverage: int = 100,
                        optimization_mode: int = 0,
                        model: int = 0,
                        set_file: Optional[Path] = None):
        """
        Konfiguruj MT4 Strategy Tester przez plik tester.ini
        
        Args:
            symbol: Para walutowa (np. EURUSD)
            timeframe: M1/M5/M15/M30/H1/H4/D1
            date_from: Data początkowa
            date_to: Data końcowa
            deposit: Depozyt startowy
            leverage: Dźwignia
            optimization_mode: 0=disabled, 1=slow, 2=fast, 3=genetic
            model: 0=Every tick, 1=Control points, 2=Open prices
            set_file: Ścieżka do pliku .set (opcjonalne)
        """
        # Timeframe mapping
        tf_map = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440,
            'W1': 10080, 'MN1': 43200
        }
        
        period = tf_map.get(timeframe.upper(), 60)
        
        # Create ini config
        config = configparser.ConfigParser()
        config.optionxform = str  # Preserve case
        
        config['Tester'] = {
            'Expert': self.ea_name,
            'Symbol': symbol,
            'Period': str(period),
            'Model': str(model),
            'FromDate': date_from.strftime('%Y.%m.%d'),
            'ToDate': date_to.strftime('%Y.%m.%d'),
            'Deposit': str(int(deposit)),
            'Leverage': f'1:{leverage}',
            'Optimization': str(optimization_mode),
            'ShutdownTerminal': '1',
            'Visual': '0',
            'UseLocal': '0',
            'ForwardMode': '0'
        }
        
        if set_file:
            config['Tester']['ExpertParameters'] = set_file.name
        
        # Save ini
        ini_path = self.config_dir / "tester.ini"
        with open(ini_path, 'w') as f:
            config.write(f, space_around_delimiters=False)
        
        self.logger.info(f"⚙️ Configured tester: {symbol} {timeframe} ({date_from.date()} to {date_to.date()})")
    
    def run_backtest(self,
                    symbol: str,
                    timeframe: str,
                    date_from: datetime,
                    date_to: datetime,
                    config: Dict,
                    deposit: float = 10000.0,
                    leverage: int = 100,
                    timeout: int = 600) -> Optional[BacktestResults]:
        """
        Uruchom backtest
        
        Returns:
            BacktestResults lub None jeśli błąd
        """
        self.logger.info(f"🚀 Starting backtest: {symbol} {timeframe}")
        
        # Create set file
        set_file = self.tester_dir / "sets" / f"{self.ea_name}_temp.set"
        set_file.parent.mkdir(parents=True, exist_ok=True)
        self.create_set_file(config, set_file)
        
        # Configure tester
        self.configure_tester(
            symbol=symbol,
            timeframe=timeframe,
            date_from=date_from,
            date_to=date_to,
            deposit=deposit,
            leverage=leverage,
            set_file=set_file
        )
        
        # Run MT4 in test mode
        process = subprocess.Popen(
            [str(self.terminal_exe), "/testmode"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for completion
        start_time = time.time()
        while True:
            if process.poll() is not None:
                break
            
            if time.time() - start_time > timeout:
                process.kill()
                self.logger.error(f"❌ Backtest timeout ({timeout}s)")
                return None
            
            time.sleep(1)
        
        self.logger.info(f"✅ Backtest completed in {time.time() - start_time:.1f}s")
        
        # Parse results
        results = self.parse_latest_report()
        
        return results
    
    def parse_latest_report(self) -> Optional[BacktestResults]:
        """
        Parse najnowszego raportu HTML z MT4
        """
        # Find latest report
        reports = list(self.reports_dir.glob("*.htm"))
        
        if not reports:
            self.logger.error("❌ No reports found")
            return None
        
        latest_report = max(reports, key=lambda p: p.stat().st_mtime)
        
        self.logger.info(f"📊 Parsing report: {latest_report.name}")
        
        return self.parse_report(latest_report)
    
    def parse_report(self, report_path: Path) -> Optional[BacktestResults]:
        """
        Parse raportu HTML MT4
        
        Struktura raportu MT4:
        - Strategy Tester Report
        - Tabela z metrykami
        """
        try:
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metrics from table
            metrics = {}
            
            # Find all table rows
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    metrics[key] = value
            
            # Parse key metrics
            results = BacktestResults()
            
            # Basic info
            results.initial_deposit = self._parse_float(metrics.get('Initial deposit', '10000'))
            
            # Trade statistics
            results.total_trades = self._parse_int(metrics.get('Total trades', '0'))
            
            # Profit/Loss
            results.net_profit = self._parse_float(metrics.get('Total net profit', '0'))
            results.gross_profit = self._parse_float(metrics.get('Gross profit', '0'))
            results.gross_loss = self._parse_float(metrics.get('Gross loss', '0'))
            
            # Profit factor
            pf = self._parse_float(metrics.get('Profit factor', '0'))
            
            # Drawdown
            dd_str = metrics.get('Maximal drawdown', '0')
            # Format: "2000.00 (20.00%)"
            dd_match = re.search(r'([\d.]+)\s*\(([\d.]+)%\)', dd_str)
            if dd_match:
                results.max_drawdown = -abs(float(dd_match.group(1)))
                results.max_drawdown_percent = float(dd_match.group(2))
            else:
                results.max_drawdown = -abs(self._parse_float(dd_str))
            
            # Balance
            results.final_balance = self._parse_float(metrics.get('Balance', str(results.initial_deposit)))
            
            # Winning/Losing trades
            # Format: "60 (60.00%)"
            win_str = metrics.get('Short positions (won %)', '0')
            win_match = re.search(r'(\d+)', win_str)
            if win_match:
                short_wins = int(win_match.group(1))
            else:
                short_wins = 0
            
            # Long wins
            win_str = metrics.get('Long positions (won %)', '0')
            win_match = re.search(r'(\d+)', win_str)
            if win_match:
                long_wins = int(win_match.group(1))
            else:
                long_wins = 0
            
            results.winning_trades = short_wins + long_wins
            results.losing_trades = results.total_trades - results.winning_trades
            
            # Average trade
            if results.winning_trades > 0:
                results.avg_win = results.gross_profit / results.winning_trades
            
            if results.losing_trades > 0:
                results.avg_loss = results.gross_loss / results.losing_trades
            
            # Largest trade
            results.largest_win = self._parse_float(metrics.get('Largest profit trade', '0'))
            results.largest_loss = self._parse_float(metrics.get('Largest loss trade', '0'))
            
            # Consecutive
            results.max_consecutive_wins = self._parse_int(metrics.get('Maximal consecutive wins ($)', '0').split()[0])
            results.max_consecutive_losses = self._parse_int(metrics.get('Maximal consecutive losses ($)', '0').split()[0])
            
            self.logger.info(f"✅ Parsed: Net Profit=${results.net_profit:.2f}, DD={results.max_drawdown_percent:.1f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Parse error: {str(e)}")
            return None
    
    def _parse_float(self, value: str) -> float:
        """Parse float from string, handling spaces and formatting"""
        try:
            # Remove spaces, currency symbols
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
    
    def get_report_list(self) -> List[Path]:
        """Lista wszystkich raportów"""
        return sorted(self.reports_dir.glob("*.htm"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    def clean_old_reports(self, keep_last: int = 10):
        """Usuń stare raporty (zostaw ostatnie N)"""
        reports = self.get_report_list()
        
        if len(reports) > keep_last:
            for report in reports[keep_last:]:
                report.unlink()
                self.logger.info(f"🗑️ Deleted old report: {report.name}")


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    mt4_path = Path("C:/Program Files/MetaTrader 4")
    
    if not mt4_path.exists():
        print("⚠️ MT4 not found at default path")
        print("Update mt4_path variable for testing")
    else:
        tester = MT4Tester(mt4_path)
        
        # Test config creation
        test_config = {
            'MaxTrades': 12,
            'Multiplier': 1.4,
            'GridSetArray': [25, 50, 100]
        }
        
        test_set = Path("test_blessing.set")
        tester.create_set_file(test_config, test_set)
        
        print(f"✅ Created test .set file: {test_set}")
        
        # Cleanup
        test_set.unlink()