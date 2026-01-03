# core/data_processor.py
# Author: Rafał Wiśniewski | Data & AI Solutions

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Union
import logging
from numba import jit, prange
import multiprocessing as mp
from functools import partial

class DataProcessor:
    """
    Przetwarzanie danych tick → OHLC
    Obsługa wielowątkowa + CUDA (jeśli dostępne)
    """
    
    def __init__(self, use_gpu: bool = False, n_workers: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        self.use_gpu = use_gpu
        self.n_workers = n_workers or max(1, mp.cpu_count() - 2)
        
        # Check GPU availability
        if use_gpu:
            try:
                import cupy as cp
                self.cp = cp
                self.gpu_available = True
                self.logger.info(f"✅ GPU acceleration enabled: {cp.cuda.Device().name}")
            except:
                self.logger.warning("⚠️ CuPy not available, falling back to CPU")
                self.gpu_available = False
                self.use_gpu = False
        else:
            self.gpu_available = False
    
    def load_tick_data(self, file_path: Union[str, Path], 
                       chunk_size: Optional[int] = None) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Wczytaj dane tick z pliku
        Wspiera: CSV, Parquet, HDF5
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"📂 Loading tick data: {file_path}")
        
        # Detect format
        suffix = file_path.suffix.lower()
        
        if suffix == '.parquet':
            if chunk_size:
                # Chunked reading with polars (faster)
                df = pl.scan_parquet(file_path).collect()
                self.logger.info(f"✅ Loaded {len(df):,} ticks (Polars)")
                return df
            else:
                df = pd.read_parquet(file_path)
                self.logger.info(f"✅ Loaded {len(df):,} ticks (Pandas)")
                return df
        
        elif suffix == '.csv':
            if chunk_size:
                # Read in chunks
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path)
            
            self.logger.info(f"✅ Loaded {len(df):,} ticks (CSV)")
            return df
        
        elif suffix in ['.h5', '.hdf5']:
            df = pd.read_hdf(file_path)
            self.logger.info(f"✅ Loaded {len(df):,} ticks (HDF5)")
            return df
        
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def ticks_to_ohlc(self, df: Union[pd.DataFrame, pl.DataFrame], 
                      timeframe: str, use_polars: bool = True) -> pd.DataFrame:
        """
        Konwersja tick → OHLC
        
        Timeframes: '1min', '5min', '15min', '30min', '1H', '4H', '1D'
        """
        self.logger.info(f"🔄 Converting ticks to OHLC ({timeframe})...")
        
        # Convert polars to pandas if needed
        if isinstance(df, pl.DataFrame):
            if use_polars:
                return self._ticks_to_ohlc_polars(df, timeframe)
            else:
                df = df.to_pandas()
        
        # Pandas implementation
        return self._ticks_to_ohlc_pandas(df, timeframe)
    
    def _ticks_to_ohlc_pandas(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Pandas implementation (slower but compatible)"""
        
        # Ensure datetime index
        if 'time' in df.columns:
            df = df.set_index('time')
        
        df.index = pd.to_datetime(df.index)
        
        # Map timeframe
        tf_map = {
            '1min': '1T', 'M1': '1T',
            '5min': '5T', 'M5': '5T',
            '15min': '15T', 'M15': '15T',
            '30min': '30T', 'M30': '30T',
            '1H': '1H', 'H1': '1H',
            '4H': '4H', 'H4': '4H',
            '1D': '1D', 'D1': '1D'
        }
        
        freq = tf_map.get(timeframe, '1H')
        
        # Resample (use bid/ask or last price)
        price_col = 'bid' if 'bid' in df.columns else 'last'
        
        ohlc = df[price_col].resample(freq).ohlc()
        ohlc['volume'] = df['volume'].resample(freq).sum() if 'volume' in df.columns else 0
        
        # Add tick_volume (count of ticks per bar)
        ohlc['tick_volume'] = df[price_col].resample(freq).count()
        
        ohlc = ohlc.dropna()
        ohlc = ohlc.reset_index()
        
        self.logger.info(f"✅ Created {len(ohlc):,} bars")
        
        return ohlc
    
    def _ticks_to_ohlc_polars(self, df: pl.DataFrame, timeframe: str) -> pd.DataFrame:
        """Polars implementation (faster)"""
        
        # Map timeframe to duration
        tf_map = {
            '1min': '1m', 'M1': '1m',
            '5min': '5m', 'M5': '5m',
            '15min': '15m', 'M15': '15m',
            '30min': '30m', 'M30': '30m',
            '1H': '1h', 'H1': '1h',
            '4H': '4h', 'H4': '4h',
            '1D': '1d', 'D1': '1d'
        }
        
        interval = tf_map.get(timeframe, '1h')
        
        # Ensure time column
        if 'time' not in df.columns:
            raise ValueError("DataFrame must have 'time' column")
        
        # Convert to datetime
        df = df.with_columns([
            pl.col('time').cast(pl.Datetime)
        ])
        
        # Price column
        price_col = 'bid' if 'bid' in df.columns else 'last'
        
        # Group by time intervals
        ohlc = df.group_by_dynamic(
            'time',
            every=interval
        ).agg([
            pl.col(price_col).first().alias('open'),
            pl.col(price_col).max().alias('high'),
            pl.col(price_col).min().alias('low'),
            pl.col(price_col).last().alias('close'),
            pl.col('volume').sum().alias('volume') if 'volume' in df.columns else pl.lit(0).alias('volume'),
            pl.count().alias('tick_volume')
        ])
        
        # Convert to pandas
        result = ohlc.to_pandas()
        
        self.logger.info(f"✅ Created {len(result):,} bars (Polars)")
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _resample_ticks_numba(times: np.ndarray, prices: np.ndarray, 
                              volumes: np.ndarray, interval_seconds: int) -> Tuple:
        """
        Ultra-fast resampling using Numba JIT
        """
        n = len(times)
        
        # Pre-allocate arrays
        max_bars = n // 100 + 1000
        bar_times = np.zeros(max_bars, dtype=np.int64)
        opens = np.zeros(max_bars, dtype=np.float64)
        highs = np.zeros(max_bars, dtype=np.float64)
        lows = np.zeros(max_bars, dtype=np.float64)
        closes = np.zeros(max_bars, dtype=np.float64)
        bar_volumes = np.zeros(max_bars, dtype=np.float64)
        
        bar_idx = 0
        current_bar_start = times[0] - (times[0] % interval_seconds)
        bar_open = prices[0]
        bar_high = prices[0]
        bar_low = prices[0]
        bar_volume = 0.0
        
        for i in prange(n):
            bar_time = times[i] - (times[i] % interval_seconds)
            
            if bar_time > current_bar_start:
                # Save current bar
                bar_times[bar_idx] = current_bar_start
                opens[bar_idx] = bar_open
                highs[bar_idx] = bar_high
                lows[bar_idx] = bar_low
                closes[bar_idx] = prices[i-1]
                bar_volumes[bar_idx] = bar_volume
                
                # Start new bar
                bar_idx += 1
                current_bar_start = bar_time
                bar_open = prices[i]
                bar_high = prices[i]
                bar_low = prices[i]
                bar_volume = volumes[i]
            else:
                # Update current bar
                if prices[i] > bar_high:
                    bar_high = prices[i]
                if prices[i] < bar_low:
                    bar_low = prices[i]
                bar_volume += volumes[i]
        
        # Save last bar
        bar_times[bar_idx] = current_bar_start
        opens[bar_idx] = bar_open
        highs[bar_idx] = bar_high
        lows[bar_idx] = bar_low
        closes[bar_idx] = prices[-1]
        bar_volumes[bar_idx] = bar_volume
        
        # Trim arrays
        return (bar_times[:bar_idx+1], opens[:bar_idx+1], highs[:bar_idx+1],
                lows[:bar_idx+1], closes[:bar_idx+1], bar_volumes[:bar_idx+1])
    
    def ticks_to_ohlc_fast(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Ultra-fast conversion using Numba
        """
        self.logger.info(f"⚡ Fast conversion (Numba): {timeframe}")
        
        # Map timeframe to seconds
        tf_seconds = {
            '1min': 60, 'M1': 60,
            '5min': 300, 'M5': 300,
            '15min': 900, 'M15': 900,
            '30min': 1800, 'M30': 1800,
            '1H': 3600, 'H1': 3600,
            '4H': 14400, 'H4': 14400,
            '1D': 86400, 'D1': 86400
        }
        
        interval = tf_seconds.get(timeframe, 3600)
        
        # Prepare arrays
        times = df['time'].astype(np.int64) // 10**9  # to unix timestamp
        price_col = 'bid' if 'bid' in df.columns else 'last'
        prices = df[price_col].values
        volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        
        # Run Numba
        bar_times, opens, highs, lows, closes, bar_volumes = self._resample_ticks_numba(
            times.values, prices, volumes, interval
        )
        
        # Create DataFrame
        result = pd.DataFrame({
            'time': pd.to_datetime(bar_times, unit='s'),
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': bar_volumes
        })
        
        self.logger.info(f"✅ Created {len(result):,} bars (Numba)")
        
        return result
    
    def parallel_process_files(self, file_paths: List[Path], 
                               timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Równoległe przetwarzanie wielu plików tick
        """
        self.logger.info(f"🚀 Processing {len(file_paths)} files in parallel ({self.n_workers} workers)")
        
        with mp.Pool(self.n_workers) as pool:
            func = partial(self._process_single_file, timeframe=timeframe)
            results = pool.map(func, file_paths)
        
        # Create dict {symbol: dataframe}
        output = {}
        for file_path, df in zip(file_paths, results):
            symbol = file_path.stem.split('_')[0]  # Extract symbol from filename
            output[symbol] = df
        
        self.logger.info(f"✅ Processed {len(output)} symbols")
        
        return output
    
    def _process_single_file(self, file_path: Path, timeframe: str) -> pd.DataFrame:
        """Helper dla parallel processing"""
        try:
            df = self.load_tick_data(file_path)
            ohlc = self.ticks_to_ohlc_fast(df, timeframe)
            return ohlc
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def save_ohlc(self, df: pd.DataFrame, output_path: Path, 
                  format: str = 'parquet'):
        """Zapisz OHLC do pliku"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            df.to_parquet(output_path, compression='gzip', index=False)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'hdf5':
            df.to_hdf(output_path, key='ohlc', mode='w')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"💾 Saved: {output_path}")
    
    def split_train_test(self, df: pd.DataFrame, 
                         train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Podział na train/test"""
        split_idx = int(len(df) * train_ratio)
        
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        self.logger.info(f"📊 Train: {len(train):,} bars | Test: {len(test):,} bars")
        
        return train, test
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaj wskaźniki techniczne (opcjonalne dla ML)"""
        df = df.copy()
        
        # Simple features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        
        # Volatility
        df['volatility_10'] = df['returns'].rolling(10).std()
        
        df = df.dropna()
        
        return df


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    processor = DataProcessor(use_gpu=False, n_workers=4)
    
    # Test data
    test_data = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=100000, freq='1s'),
        'bid': np.random.randn(100000).cumsum() + 1.1000,
        'volume': np.random.randint(1, 100, 100000)
    })
    
    # Save test file
    test_file = Path("test_ticks.parquet")
    test_data.to_parquet(test_file, index=False)
    
    # Load and convert
    df = processor.load_tick_data(test_file)
    ohlc = processor.ticks_to_ohlc_fast(df, 'M5')
    
    print(ohlc.head())
    print(f"\n✅ Test completed: {len(ohlc)} bars created")
    
    # Cleanup
    test_file.unlink()