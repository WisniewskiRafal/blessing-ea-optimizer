# Contributing to Blessing EA Optimizer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### 1. Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

**When reporting bugs, include:**
- Python version (`python --version`)
- Operating system (Windows/Linux/Mac)
- GPU/CPU specs (run `python check_environment.py`)
- Steps to reproduce the bug
- Expected vs actual behavior
- Error messages and stack traces

**Example:**
```markdown
## Bug: Genetic optimizer crashes on large datasets

**Environment:**
- Python 3.11.5
- Windows 11
- RTX 3060 Ti, 16GB RAM

**Steps:**
1. Run `python blessing_optimizer_main.py`
2. Choose option C (Genetic)
3. Symbol: EURUSD, dates: 2024-01-01 to 2024-12-31
4. Crash after 50 generations

**Error:**
```
RuntimeError: CUDA out of memory
```

**Expected:** Should complete all 100 generations
```

### 2. Suggesting Enhancements

Open an issue with the label "enhancement".

**Include:**
- Clear description of the enhancement
- Why this would be useful
- Possible implementation approach
- Examples (if applicable)

### 3. Pull Requests

**Before submitting:**
1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

**PR Guidelines:**
- Follow the existing code style (Black formatter, type hints)
- Add tests for new features
- Update documentation (README, docstrings)
- Keep PRs focused (one feature/fix per PR)
- Reference related issues

**Example PR title:**
```
[FEAT] Add Parquet data format support
[FIX] Correct parameter mapping in grid config
[DOCS] Update DATA_FORMAT.md with examples
[TEST] Add unit tests for DataLoader
```

---

## Development Setup

### 1. Clone and setup:
```bash
git clone https://github.com/your-username/blessing-optimizer.git
cd blessing-optimizer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements_full.txt
pip install -r requirements_dev.txt  # Development dependencies
```

### 2. Install dev tools:
```bash
pip install black ruff mypy pytest
```

### 3. Run tests:
```bash
pytest tests/
```

---

## Code Style

### Python Style Guide

We follow **PEP 8** with these specifics:

**Formatting:**
- Use **Black** formatter (line length: 100)
- Use **Ruff** for linting
- Full type hints (checked with **mypy --strict**)

**Example:**
```python
from typing import Dict, List, Optional
import pandas as pd


def optimize_parameters(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    top_n: int = 10,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Optimize EA parameters for given symbol and date range.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        top_n: Number of best configurations to return
        use_gpu: Enable GPU acceleration if available
        
    Returns:
        Dictionary containing top_n optimized configurations
        
    Raises:
        FileNotFoundError: If historical data file not found
        ValueError: If date range is invalid
    """
    # Implementation...
    pass
```

### Commit Messages

Format: `[TYPE] Short description (max 72 chars)`

**Types:**
- `FEAT` - New feature
- `FIX` - Bug fix
- `DOCS` - Documentation only
- `TEST` - Adding/updating tests
- `REFACTOR` - Code restructuring
- `PERF` - Performance improvement

**Examples:**
```
[FEAT] Add multi-timeframe data aggregation
[FIX] Correct Sharpe ratio calculation
[DOCS] Update installation instructions for macOS
[TEST] Add integration tests for genetic optimizer
[REFACTOR] Extract data loading to separate module
[PERF] Optimize backtest engine with vectorization
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_data_loader.py

# With coverage
pytest --cov=core --cov=optimization
```

### Writing Tests

Use **pytest** with fixtures:

```python
import pytest
import pandas as pd
from core.data_loader import DataLoader


@pytest.fixture
def sample_data():
    """Sample OHLC data for testing"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'open': [1.0] * 1000,
        'high': [1.1] * 1000,
        'low': [0.9] * 1000,
        'close': [1.05] * 1000,
    })


def test_data_loader_loads_csv(sample_data, tmp_path):
    """Test that DataLoader correctly loads CSV files"""
    # Arrange
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    loader = DataLoader()
    
    # Act
    df = loader.load_csv(csv_path)
    
    # Assert
    assert len(df) == 1000
    assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close']
    assert df['close'].mean() == pytest.approx(1.05)
```

---

## Documentation

### Docstrings

Use **Google style**:

```python
def backtest(config: BacktestConfig, data: pd.DataFrame) -> BacktestResult:
    """
    Run backtest with given configuration.
    
    This function simulates trading based on the provided configuration
    and historical data, returning performance metrics.
    
    Args:
        config: Backtest configuration including entry rules, grid settings
        data: Historical OHLC data with columns [timestamp, open, high, low, close]
        
    Returns:
        BacktestResult object containing:
            - net_profit: Total profit/loss in account currency
            - win_rate: Percentage of profitable trades (0.0 to 1.0)
            - max_drawdown_percent: Maximum equity drawdown percentage
            - total_trades: Number of executed trades
            
    Raises:
        ValueError: If data is empty or missing required columns
        RuntimeError: If backtest encounters unrecoverable error
        
    Example:
        >>> config = BacktestConfig(...)
        >>> data = pd.read_csv('EURUSD_2024_M1.csv')
        >>> result = backtest(config, data)
        >>> print(f"Profit: ${result.net_profit:.2f}")
        Profit: $5420.50
        
    Note:
        Backtests do not account for slippage or spread. Results may differ
        from live trading performance.
    """
    pass
```

### README Updates

When adding features, update:
- `README.md` - Main documentation
- `QUICK_START.md` - If affecting quick start steps
- `DATA_FORMAT.md` - If adding data format support

---

## What NOT to Contribute

Please **DO NOT** submit:
- Proprietary trading strategies with profitable edge
- Real backtest results revealing profitable configurations
- API keys, passwords, or credentials
- Large binary files (data, models) - use Git LFS or external hosting
- Unrelated features (keep PRs focused)

---

## Questions?

- Open an issue with label "question"
- Check existing discussions
- Email: [your-email@example.com]

Thank you for contributing! 🚀

---

**Last updated:** 2026-01-03
