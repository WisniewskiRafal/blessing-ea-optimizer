# core/mt4_bridge.py
# Author: Rafał Wiśniewski | Data & AI Solutions

import socket
import json
import struct
import time
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
import configparser
import zipfile
import shutil

class MT4Bridge:
    """
    Most do MT4 przez ZeroMQ socket communication
    Wymaga MT4 EA z ZMQ server (MQL4 side)
    """
    
    def __init__(self, host: str = "localhost", port: int = 5555, 
                 mt4_path: Optional[Path] = None):
        self.host = host
        self.port = port
        self.mt4_path = Path(mt4_path) if mt4_path else None
        self.socket = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
    def connect(self, timeout: int = 10) -> bool:
        """Połącz przez ZeroMQ"""
        try:
            import zmq
            
            context = zmq.Context()
            self.socket = context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            
            # Set timeout
            self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
            self.socket.setsockopt(zmq.SNDTIMEO, timeout * 1000)
            
            # Test connection
            response = self._send_command({"action": "ping"})
            if response and response.get("status") == "pong":
                self.connected = True
                self.logger.info(f"✅ Connected to MT4 via ZMQ: {self.host}:{self.port}")
                return True
            
            self.logger.error("MT4 ZMQ ping failed")
            return False
            
        except ImportError:
            self.logger.error("ZMQ not installed. Run: pip install pyzmq")
            return False
        except Exception as e:
            self.logger.error(f"MT4 connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Rozłącz"""
        if self.socket:
            self.socket.close()
        self.connected = False
        self.logger.info("MT4 disconnected")
    
    def _send_command(self, command: Dict) -> Optional[Dict]:
        """Wyślij komendę do MT4 i odbierz odpowiedź"""
        if not self.socket:
            self.logger.error("Socket not initialized")
            return None
        
        try:
            # Send JSON command
            self.socket.send_json(command)
            
            # Receive response
            response = self.socket.recv_json()
            return response
            
        except Exception as e:
            self.logger.error(f"Command failed: {str(e)}")
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """Pobierz info o koncie"""
        response = self._send_command({"action": "account_info"})
        
        if response and response.get("status") == "success":
            return response.get("data")
        return None
    
    def get_symbols(self) -> List[str]:
        """Lista symboli"""
        response = self._send_command({"action": "symbols_list"})
        
        if response and response.get("status") == "success":
            return response.get("data", [])
        return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Info o symbolu"""
        response = self._send_command({
            "action": "symbol_info",
            "symbol": symbol
        })
        
        if response and response.get("status") == "success":
            return response.get("data")
        return None
    
    def download_history(self, symbol: str, timeframe: int, 
                         bars: int = 10000) -> pd.DataFrame:
        """
        Pobierz dane historyczne
        
        Timeframes MT4:
        1=M1, 5=M5, 15=M15, 30=M30, 60=H1, 240=H4, 1440=D1, 10080=W1, 43200=MN1
        """
        response = self._send_command({
            "action": "download_history",
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": bars
        })
        
        if response and response.get("status") == "success":
            data = response.get("data", [])
            
            if data:
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                self.logger.info(f"✅ Downloaded {len(df):,} bars for {symbol}")
                return df
        
        self.logger.warning(f"No data for {symbol}")
        return pd.DataFrame()
    
    def install_zmq_ea(self) -> bool:
        """
        Instaluj EA z ZMQ server do MT4
        Kopiuje plik do MT4/MQL4/Experts/
        """
        if not self.mt4_path:
            self.logger.error("MT4 path not set")
            return False
        
        experts_dir = self.mt4_path / "MQL4" / "Experts"
        if not experts_dir.exists():
            self.logger.error(f"MT4 Experts dir not found: {experts_dir}")
            return False
        
        # Create ZMQ EA template
        zmq_ea_code = '''//+------------------------------------------------------------------+
//|                                             BlessingZMQ.mq4      |
//|                           Rafał Wiśniewski | Data & AI Solutions |
//+------------------------------------------------------------------+
#property copyright "Rafał Wiśniewski"
#property link      "github.com/RafalWisniewski"
#property version   "1.0"
#property strict

#include <Zmq/Zmq.mqh>

input string ZMQ_HOST = "tcp://*:5555";

Context context("BlessingZMQ");
Socket socket(context, ZMQ_REP);

//+------------------------------------------------------------------+
int OnInit() {
    if(!socket.bind(ZMQ_HOST)) {
        Print("❌ ZMQ bind failed");
        return INIT_FAILED;
    }
    
    Print("✅ ZMQ Server started: ", ZMQ_HOST);
    EventSetTimer(1);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    socket.unbind(ZMQ_HOST);
    EventKillTimer();
}

//+------------------------------------------------------------------+
void OnTimer() {
    ZmqMsg request;
    
    if(socket.recv(request, true)) {
        string cmd = request.getData();
        string response = ProcessCommand(cmd);
        
        ZmqMsg reply(response);
        socket.send(reply);
    }
}

//+------------------------------------------------------------------+
string ProcessCommand(string cmd) {
    // Parse JSON command
    // {"action": "ping"}
    
    if(StringFind(cmd, "ping") >= 0) {
        return "{\\"status\\":\\"pong\\"}";
    }
    
    if(StringFind(cmd, "account_info") >= 0) {
        return GetAccountInfo();
    }
    
    if(StringFind(cmd, "symbols_list") >= 0) {
        return GetSymbolsList();
    }
    
    if(StringFind(cmd, "download_history") >= 0) {
        return DownloadHistory(cmd);
    }
    
    return "{\\"status\\":\\"error\\",\\"message\\":\\"Unknown command\\"}";
}

//+------------------------------------------------------------------+
string GetAccountInfo() {
    string json = "{\\"status\\":\\"success\\",\\"data\\":{";
    json += "\\"balance\\":" + DoubleToString(AccountBalance(), 2) + ",";
    json += "\\"equity\\":" + DoubleToString(AccountEquity(), 2) + ",";
    json += "\\"margin\\":" + DoubleToString(AccountMargin(), 2) + ",";
    json += "\\"free_margin\\":" + DoubleToString(AccountFreeMargin(), 2) + ",";
    json += "\\"leverage\\":" + IntegerToString(AccountLeverage()) + ",";
    json += "\\"currency\\":\\"" + AccountCurrency() + "\\"";
    json += "}}";
    return json;
}

//+------------------------------------------------------------------+
string GetSymbolsList() {
    string json = "{\\"status\\":\\"success\\",\\"data\\":[";
    
    for(int i = 0; i < SymbolsTotal(true); i++) {
        string symbol = SymbolName(i, true);
        if(i > 0) json += ",";
        json += "\\"" + symbol + "\\"";
    }
    
    json += "]}";
    return json;
}

//+------------------------------------------------------------------+
string DownloadHistory(string cmd) {
    // Parse: {"action":"download_history","symbol":"EURUSD","timeframe":60,"bars":1000}
    
    string symbol = "EURUSD";  // TODO: Parse from cmd
    int timeframe = 60;
    int bars = 1000;
    
    string json = "{\\"status\\":\\"success\\",\\"data\\":[";
    
    for(int i = bars - 1; i >= 0; i--) {
        if(i < bars - 1) json += ",";
        
        json += "{";
        json += "\\"time\\":" + IntegerToString(iTime(symbol, timeframe, i)) + ",";
        json += "\\"open\\":" + DoubleToString(iOpen(symbol, timeframe, i), 5) + ",";
        json += "\\"high\\":" + DoubleToString(iHigh(symbol, timeframe, i), 5) + ",";
        json += "\\"low\\":" + DoubleToString(iLow(symbol, timeframe, i), 5) + ",";
        json += "\\"close\\":" + DoubleToString(iClose(symbol, timeframe, i), 5) + ",";
        json += "\\"volume\\":" + IntegerToString(iVolume(symbol, timeframe, i));
        json += "}";
    }
    
    json += "]}";
    return json;
}
//+------------------------------------------------------------------+
'''
        
        # Save EA file
        ea_file = experts_dir / "BlessingZMQ.mq4"
        with open(ea_file, "w", encoding="utf-8") as f:
            f.write(zmq_ea_code)
        
        self.logger.info(f"✅ ZMQ EA installed: {ea_file}")
        self.logger.warning("⚠️ Compile BlessingZMQ.mq4 in MetaEditor")
        self.logger.warning("⚠️ Attach BlessingZMQ to chart in MT4")
        
        return True
    
    def run_backtest(self, symbol: str, timeframe: str, ea_path: Path,
                     set_file: Path, date_from: datetime, date_to: datetime,
                     deposit: float = 10000, leverage: int = 100) -> Optional[Dict]:
        """
        Uruchom backtest w MT4 Strategy Tester
        
        Metoda:
        1. Modyfikuj terminal.ini
        2. Uruchom MT4 w trybie /testmode
        3. Monitoruj wyniki w reports/
        """
        if not self.mt4_path:
            self.logger.error("MT4 path not set")
            return None
        
        # Timeframe mapping
        tf_map = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200
        }
        
        tf_value = tf_map.get(timeframe, 60)
        
        # Prepare tester config
        tester_ini = self.mt4_path / "config" / "tester.ini"
        
        config = configparser.ConfigParser()
        config['Tester'] = {
            'Expert': ea_path.stem,
            'ExpertParameters': set_file.name,
            'Symbol': symbol,
            'Period': str(tf_value),
            'Model': '0',  # 0=Every tick, 1=Control points, 2=Open prices
            'FromDate': date_from.strftime('%Y.%m.%d'),
            'ToDate': date_to.strftime('%Y.%m.%d'),
            'Deposit': str(deposit),
            'Leverage': f'1:{leverage}',
            'Optimization': '0',  # 0=disabled
            'ShutdownTerminal': '1'
        }
        
        # Write config
        tester_ini.parent.mkdir(parents=True, exist_ok=True)
        with open(tester_ini, 'w') as f:
            config.write(f)
        
        self.logger.info(f"📝 Tester config: {tester_ini}")
        
        # Run MT4 in test mode
        terminal_exe = self.mt4_path / "terminal.exe"
        
        if not terminal_exe.exists():
            self.logger.error(f"MT4 terminal not found: {terminal_exe}")
            return None
        
        self.logger.info("🚀 Starting MT4 Strategy Tester...")
        
        process = subprocess.Popen([
            str(terminal_exe),
            "/testmode"
        ])
        
        # Wait for results
        report_dir = self.mt4_path / "tester" / "reports"
        timeout = 300  # 5 min
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not process.poll() is None:
                break
            time.sleep(5)
        
        # Parse results
        results = self._parse_mt4_report(report_dir)
        
        return results
    
    def _parse_mt4_report(self, report_dir: Path) -> Optional[Dict]:
        """Parse MT4 HTML report"""
        reports = list(report_dir.glob("*.htm"))
        
        if not reports:
            self.logger.error("No reports found")
            return None
        
        # Get latest report
        latest_report = max(reports, key=lambda p: p.stat().st_mtime)
        
        self.logger.info(f"📊 Parsing report: {latest_report}")
        
        # Simple HTML parsing (regex or BeautifulSoup)
        # TODO: Implement full parser
        
        with open(latest_report, 'r', encoding='utf-8') as f:
            html = f.read()
        
        # Extract key metrics (simplified)
        import re
        
        results = {}
        
        patterns = {
            'total_net_profit': r'Total net profit</td><td[^>]*>([-\d.]+)',
            'profit_factor': r'Profit factor</td><td[^>]*>([\d.]+)',
            'max_drawdown': r'Maximal drawdown</td><td[^>]*>([-\d.]+)',
            'total_trades': r'Total trades</td><td[^>]*>(\d+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, html)
            if match:
                results[key] = float(match.group(1))
        
        self.logger.info(f"✅ Results: {results}")
        return results
    
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
    
    bridge = MT4Bridge()
    
    # Install ZMQ EA
    # bridge.install_zmq_ea()
    
    # Test connection (requires ZMQ EA running)
    if bridge.connect():
        print("✅ MT4 Bridge Test OK")
        
        account = bridge.get_account_info()
        print(f"Account: {account}")