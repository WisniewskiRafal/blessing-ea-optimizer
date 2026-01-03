# core/mt5_bridge.py
# Author: Rafał Wiśniewski | Data & AI Solutions

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import time
import logging

class MT5Bridge:
    """Most do MT5 - logowanie, pobieranie danych, uruchamianie testów"""
    
    def __init__(self, login: int = None, password: str = None, 
                 server: str = None, path: str = None):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Połącz z MT5"""
        try:
            # Initialize MT5
            if self.path:
                if not mt5.initialize(path=self.path):
                    self.logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize():
                    self.logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
            
            # Login if credentials provided
            if self.login and self.password and self.server:
                if not mt5.login(login=self.login, password=self.password, server=self.server):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
                
                self.logger.info(f"✅ Connected to MT5: {self.server}")
            else:
                self.logger.info("✅ MT5 initialized (no login)")
            
            self.connected = True
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
    
    def get_account_info(self) -> Optional[Dict]:
        """Pobierz info o koncie"""
        if not self.connected:
            return None
        
        account = mt5.account_info()
        if account is None:
            return None
        
        return {
            'login': account.login,
            'server': account.server,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'margin_free': account.margin_free,
            'margin_level': account.margin_level,
            'leverage': account.leverage,
            'currency': account.currency
        }
    
    def get_symbols(self) -> List[str]:
        """Lista dostępnych symboli"""
        if not self.connected:
            return []
        
        symbols = mt5.symbols_get()
        return [s.name for s in symbols if s.visible]
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Info o symbolu"""
        if not self.connected:
            return None
        
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return {
            'name': info.name,
            'digits': info.digits,
            'point': info.point,
            'spread': info.spread,
            'trade_contract_size': info.trade_contract_size,
            'trade_tick_value': info.trade_tick_value,
            'trade_tick_size': info.trade_tick_size,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'swap_long': info.swap_long,
            'swap_short': info.swap_short
        }
    
    def download_ticks(self, symbol: str, date_from: datetime, 
                       date_to: datetime, save_path: Optional[Path] = None) -> pd.DataFrame:
        """Pobierz dane tick"""
        if not self.connected:
            raise RuntimeError("MT5 not connected")
        
        self.logger.info(f"Downloading ticks: {symbol} from {date_from} to {date_to}")
        
        # Copy ticks range
        ticks = mt5.copy_ticks_range(symbol, date_from, date_to, mt5.COPY_TICKS_ALL)
        
        if ticks is None or len(ticks) == 0:
            self.logger.warning(f"No ticks downloaded for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        self.logger.info(f"✅ Downloaded {len(df):,} ticks")
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet (compressed)
            df.to_parquet(save_path, compression='gzip', index=False)
            self.logger.info(f"💾 Saved to {save_path}")
        
        return df
    
    def download_ohlc(self, symbol: str, timeframe: str, 
                      date_from: datetime, date_to: datetime,
                      save_path: Optional[Path] = None) -> pd.DataFrame:
        """Pobierz dane OHLC"""
        if not self.connected:
            raise RuntimeError("MT5 not connected")
        
        # Map timeframe string to MT5 constant
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        tf = tf_map.get(timeframe)
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        self.logger.info(f"Downloading OHLC: {symbol} {timeframe} from {date_from} to {date_to}")
        
        # Copy rates
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        
        if rates is None or len(rates) == 0:
            self.logger.warning(f"No rates downloaded for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        self.logger.info(f"✅ Downloaded {len(df):,} bars")
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path, compression='gzip', index=False)
            self.logger.info(f"💾 Saved to {save_path}")
        
        return df
    
    def run_backtest(self, symbol: str, timeframe: str, ea_path: Path,
                     set_file: Path, date_from: datetime, date_to: datetime,
                     deposit: float = 10000, leverage: int = 100) -> Optional[Dict]:
        """
        Uruchom backtest EA
        
        UWAGA: MT5 nie ma pełnej API do Strategy Tester.
        Ta metoda używa workaround przez pliki .ini
        """
        if not self.connected:
            raise RuntimeError("MT5 not connected")
        
        self.logger.warning("MT5 Strategy Tester API is limited - using workaround")
        
        # MT5 Strategy Tester configuration
        # Wymaga manualnego uruchomienia przez terminal_path/tester/tester.ini
        
        tester_config = {
            'Expert': ea_path.name,
            'ExpertParameters': set_file.name,
            'Symbol': symbol,
            'Period': timeframe,
            'FromDate': date_from.strftime('%Y.%m.%d'),
            'ToDate': date_to.strftime('%Y.%m.%d'),
            'Deposit': deposit,
            'Leverage': leverage,
            'OptimizationMode': 0  # 0=disabled, 1=slow, 2=fast, 3=genetic
        }
        
        # TODO: Implement full MT5 tester automation
        # Obecnie wymaga MetaEditor i manualnego uruchomienia
        
        self.logger.error("MT5 backtest automation not fully implemented")
        self.logger.info("Alternative: Use MQL5 script + file monitoring")
        
        return None
    
    def get_optimization_results(self, results_path: Path) -> pd.DataFrame:
        """Wczytaj wyniki optymalizacji z MT5"""
        # MT5 zapisuje wyniki w XML lub HTML
        # Parser pending implementation
        
        self.logger.error("MT5 results parser not implemented")
        return pd.DataFrame()
    
    def __enter__(self):
        """Context manager"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.disconnect()


if __name__ == "__main__":
    # Test connection
    logging.basicConfig(level=logging.INFO)
    
    with MT5Bridge() as mt5:
        if mt5.connected:
            print("✅ MT5 Bridge Test OK")
            
            # Test data download
            symbols = mt5.get_symbols()
            print(f"Available symbols: {len(symbols)}")
            
            if 'EURUSD' in symbols:
                info = mt5.get_symbol_info('EURUSD')
                print(f"EURUSD info: {info}")