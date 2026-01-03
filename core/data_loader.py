# core/data_loader.py
# Data Loading and Splitting for Optimization
# Author: Rafał Wiśniewski | Data & AI Solutions

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class DataSplit:
    """Data split with metadata"""
    name: str                    # e.g., 'Q1_train', 'Q2_test'
    data: np.ndarray            # OHLC data [n_bars, 4]
    start_date: datetime
    end_date: datetime
    n_bars: int
    timeframe: str              # 'M1', 'M5', etc.


@dataclass
class WalkForwardSplit:
    """Walk-forward train/test split"""
    period_name: str            # e.g., 'Period_1'
    train: DataSplit
    test: DataSplit


class DataLoader:
    """
    Data Loader for Forex OHLC Data

    Handles:
    - Loading CSV files
    - Quarterly splits (Q1, Q2, Q3, Q4)
    - Walk-forward splits
    - Data validation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_dukascopy_m1(self,
                          symbol: str,
                          start_date: str,
                          end_date: str,
                          data_dir: str = r"d:\tick_data") -> pd.DataFrame:
        """
        Load M1 data from Dukascopy format

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            data_dir: Directory with data files

        Returns:
            DataFrame with OHLC data
        """
        # Construct file path
        # Expected format: d:\tick_data\EURUSD_2024_M1_formatted.csv
        year = start_date[:4]
        file_path = Path(data_dir) / f"{symbol}_{year}_M1_formatted.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        self.logger.info(f"Loading {symbol} data from {file_path.name}")

        # Load CSV
        df = pd.read_csv(file_path)

        self.logger.info(f"Loaded {len(df):,} bars")

        # Rename columns to lowercase first
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Timestamp': 'timestamp'
        })

        # Parse timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("No timestamp column found")

        # Sort by time
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()

        self.logger.info(f"Filtered to {len(df):,} bars ({start_date} to {end_date})")
        if len(df) > 0:
            self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            raise ValueError(f"No data found for date range {start_date} to {end_date}")

        return df

    def load_eurusd_m1(self,
                       file_path: str = r"d:\tick_data\EURUSD_2024_M1_formatted.csv",
                       max_bars: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load EURUSD M1 data from CSV

        Args:
            file_path: Path to CSV file
            max_bars: Maximum bars to load (None = all)

        Returns:
            (dataframe, ohlc_array)
        """
        self.logger.info(f"Loading data from {Path(file_path).name}")

        # Load CSV
        df = pd.read_csv(file_path, nrows=max_bars)

        self.logger.info(f"Loaded {len(df):,} bars")
        self.logger.info(f"Columns: {list(df.columns)}")

        # Parse timestamps
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Sort by time
        df = df.sort_values('Timestamp').reset_index(drop=True)

        # Extract OHLC
        ohlc = df[['Open', 'High', 'Low', 'Close']].values

        self.logger.info(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        self.logger.info(f"OHLC shape: {ohlc.shape}")
        self.logger.info(f"Price range: {ohlc[:, 3].min():.5f} - {ohlc[:, 3].max():.5f}")

        return df, ohlc

    def split_by_quarters(self,
                         df: pd.DataFrame,
                         year: int = 2024) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Split data into quarters (Q1, Q2, Q3, Q4)

        Args:
            df: DataFrame with 'Timestamp' column
            year: Year to split (default 2024)

        Returns:
            Dictionary with Q1, Q2, Q3, Q4 splits
        """
        self.logger.info(f"Splitting data into quarters for {year}")

        quarters = {}

        # Define quarter boundaries
        quarter_ranges = {
            'Q1': (f'{year}-01-01', f'{year}-03-31'),
            'Q2': (f'{year}-04-01', f'{year}-06-30'),
            'Q3': (f'{year}-07-01', f'{year}-09-30'),
            'Q4': (f'{year}-10-01', f'{year}-12-31'),
        }

        for q_name, (start, end) in quarter_ranges.items():
            # Filter by date range
            mask = (df['Timestamp'] >= start) & (df['Timestamp'] <= end)
            q_df = df[mask].copy()

            if len(q_df) == 0:
                self.logger.warning(f"{q_name}: No data found")
                continue

            # Extract OHLC
            ohlc = q_df[['Open', 'High', 'Low', 'Close']].values

            quarters[q_name] = (q_df, ohlc)

            self.logger.info(
                f"{q_name}: {len(q_df):,} bars "
                f"({q_df['Timestamp'].min().date()} to {q_df['Timestamp'].max().date()})"
            )

        return quarters

    def create_data_split(self,
                         df: pd.DataFrame,
                         name: str,
                         timeframe: str = 'M1') -> DataSplit:
        """
        Create DataSplit object from DataFrame

        Args:
            df: DataFrame with OHLC data
            name: Name of split
            timeframe: Timeframe (default M1)

        Returns:
            DataSplit object
        """
        ohlc = df[['Open', 'High', 'Low', 'Close']].values

        return DataSplit(
            name=name,
            data=ohlc,
            start_date=df['Timestamp'].min(),
            end_date=df['Timestamp'].max(),
            n_bars=len(df),
            timeframe=timeframe
        )

    def create_walk_forward_splits(self,
                                   df: pd.DataFrame,
                                   year: int = 2024) -> List[WalkForwardSplit]:
        """
        Create walk-forward train/test splits

        Period 1: Train Q1 → Test Q2
        Period 2: Train Q1+Q2 → Test Q3
        Period 3: Train Q1+Q2+Q3 → Test Q4

        Args:
            df: Full DataFrame
            year: Year to split

        Returns:
            List of WalkForwardSplit objects
        """
        self.logger.info("Creating walk-forward splits")

        # Get quarterly splits
        quarters = self.split_by_quarters(df, year)

        if len(quarters) < 4:
            raise ValueError(f"Need 4 quarters, got {len(quarters)}")

        # Extract DataFrames
        q1_df, _ = quarters['Q1']
        q2_df, _ = quarters['Q2']
        q3_df, _ = quarters['Q3']
        q4_df, _ = quarters['Q4']

        splits = []

        # Period 1: Q1 train, Q2 test
        period1_train = pd.concat([q1_df], ignore_index=True)
        period1_test = q2_df

        splits.append(WalkForwardSplit(
            period_name='Period_1',
            train=self.create_data_split(period1_train, 'Q1_train'),
            test=self.create_data_split(period1_test, 'Q2_test')
        ))

        # Period 2: Q1+Q2 train, Q3 test
        period2_train = pd.concat([q1_df, q2_df], ignore_index=True)
        period2_test = q3_df

        splits.append(WalkForwardSplit(
            period_name='Period_2',
            train=self.create_data_split(period2_train, 'Q1Q2_train'),
            test=self.create_data_split(period2_test, 'Q3_test')
        ))

        # Period 3: Q1+Q2+Q3 train, Q4 test
        period3_train = pd.concat([q1_df, q2_df, q3_df], ignore_index=True)
        period3_test = q4_df

        splits.append(WalkForwardSplit(
            period_name='Period_3',
            train=self.create_data_split(period3_train, 'Q1Q2Q3_train'),
            test=self.create_data_split(period3_test, 'Q4_test')
        ))

        # Log summary
        for split in splits:
            self.logger.info(
                f"{split.period_name}: "
                f"Train {split.train.n_bars:,} bars, "
                f"Test {split.test.n_bars:,} bars"
            )

        return splits

    def validate_data(self, ohlc: np.ndarray) -> Tuple[bool, str]:
        """
        Validate OHLC data

        Args:
            ohlc: OHLC array [n_bars, 4]

        Returns:
            (is_valid, message)
        """
        # Check shape
        if ohlc.ndim != 2:
            return False, f"Expected 2D array, got {ohlc.ndim}D"

        if ohlc.shape[1] != 4:
            return False, f"Expected 4 columns (OHLC), got {ohlc.shape[1]}"

        # Check for NaN
        if np.isnan(ohlc).any():
            n_nans = np.isnan(ohlc).sum()
            return False, f"Found {n_nans} NaN values"

        # Check for negative prices
        if (ohlc < 0).any():
            return False, "Found negative prices"

        # Check OHLC relationships (High >= Low)
        high = ohlc[:, 1]
        low = ohlc[:, 2]

        if (high < low).any():
            n_invalid = (high < low).sum()
            return False, f"Found {n_invalid} bars where High < Low"

        return True, "OK"


if __name__ == "__main__":
    # Test DataLoader
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

    print("="*60)
    print("DATA LOADER TEST")
    print("="*60)

    loader = DataLoader()

    # Test 1: Load data
    print("\n[TEST 1] Load EURUSD M1 Data")
    df, ohlc = loader.load_eurusd_m1()

    # Test 2: Validate data
    print("\n[TEST 2] Validate Data")
    valid, msg = loader.validate_data(ohlc)
    print(f"  Valid: {valid} - {msg}")

    # Test 3: Split by quarters
    print("\n[TEST 3] Split by Quarters")
    quarters = loader.split_by_quarters(df, year=2024)

    for q_name, (q_df, q_ohlc) in quarters.items():
        print(f"  {q_name}: {len(q_df):,} bars, "
              f"{q_df['Timestamp'].min().date()} to {q_df['Timestamp'].max().date()}")

    # Test 4: Walk-forward splits
    print("\n[TEST 4] Walk-Forward Splits")
    wf_splits = loader.create_walk_forward_splits(df, year=2024)

    for split in wf_splits:
        print(f"\n  {split.period_name}:")
        print(f"    TRAIN: {split.train.name} - {split.train.n_bars:,} bars "
              f"({split.train.start_date.date()} to {split.train.end_date.date()})")
        print(f"    TEST:  {split.test.name} - {split.test.n_bars:,} bars "
              f"({split.test.start_date.date()} to {split.test.end_date.date()})")

        # Validate
        train_valid, train_msg = loader.validate_data(split.train.data)
        test_valid, test_msg = loader.validate_data(split.test.data)
        print(f"    Train valid: {train_valid}, Test valid: {test_valid}")

    print("\n" + "="*60)
    print("[OK] Data Loader Tests Completed")
    print("="*60)
