# CLAUDE.md - Universal Instructions for Trading/Data Processing Projects

> **Language Note:** This file is in English for technical precision and GitHub compatibility.  
> **Chat Language:** We communicate in Polish during conversations for efficiency.

---

## GENERAL PRINCIPLES (ALWAYS APPLY)

### Communication & Workflow
- **Always ask before coding** - Propose approach, wait for confirmation
- **Be maximally concise** - Skip detailed explanations unless explicitly requested
- **Token efficiency** - Focus on substance, minimize verbose explanations
- **Iterative approach**: Plan → Code → Execute/Test (always ask permission for commands)
- **Chat in Polish** - Technical docs in English, conversations in Polish

### Code Standards
- **Language:** Python 3.11+
- **Style:** Black + Ruff formatting, full type hints (mypy strict)
- **Docstrings:** Google style
- **Testing:** Always add unit tests (pytest) for new/modified functions, especially data processing and backtests
- **Security:** NEVER commit sensitive data (API keys, passwords, proprietary trading logic). Use placeholders or environment variables.

---

## DATA MANAGEMENT

### Tick Data from Dukascopy
- **Central storage:** `D:\tick_data\` (external to projects)
- **NEVER copy data to project folders**
- **Always use absolute path:** `D:\tick_data\` (or via config.yaml / environment variables)
- **Data splitting:** Always chronological (time-based), default 50/50
  - First half = train set (pattern discovery)
  - Second half = test set (backtests)

### Project Data Structure
```
project_root/
├── data/
│   ├── README.md           # Only structure description (committed)
│   ├── aggregated/         # Processed data (NOT committed)
│   ├── enriched/           # With indicators (NOT committed)
│   └── results/            # Backtest outputs (NOT committed)
```

**Never commit to Git:** `/data/*` (except README.md), `/logs/`, `*.env`, `/proprietary/`

---

## PERFORMANCE & ACCELERATION (CRITICAL - ALWAYS OPTIMIZE!)

### Environment Diagnostics
**At the start of each major change or new session, run diagnostics:**

```bash
nvidia-smi
nvcc --version
python -m torch.utils.collect_env
pip list | grep -E "torch|numba|cupy|dask|pandas|numpy"
```

**Always check GPU availability in code:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Hardware Context
- **CPU:** Ryzen 5 5600 (12 threads)
- **RAM:** 32GB
- **GPU:** RTX 5060 Ti 16GB (CUDA Compute Capability sm_120 - Blackwell Architecture)
  - **CUDA Toolkit:** 12.4 + 13.1 installed
  - **Driver:** 591.44 (supports CUDA 13.1)

### GPU Acceleration - RTX 5060 Ti (sm_120) Verified Solutions

**⚠️ PARTIALLY WORKING (Tested 2026-01-01):**

1. **CuPy 13.6.0** - PARTIAL SUPPORT ⭐⭐
   ```bash
   pip install cupy-cuda12x  # For CUDA 12.x
   ```
   - ✅ Array creation and transfers work
   - ❌ **Compiled kernels FAIL** on sm_120 (mean, sum, matmul, etc.)
   - ⚠️ Built with CUDA 12.0.90, needs CUDA 12.4+ rebuild
   - ⚠️ **Status:** Wait for CuPy update or rebuild from source
   - 📌 **Alternative:** Use NumPy + CPU parallelization instead

2. **nvmath-python 0.7.0** - ADVANCED MATH ⭐⭐⭐⭐⭐
   ```bash
   pip install nvmath-python
   ```
   - ✅ **CUDA 13.1 support** (sm_120 compatible)
   - ✅ Official NVIDIA library for advanced math operations
   - ✅ BLAS, FFT, sparse matrix operations
   - ✅ Best for complex numerical algorithms

3. **cuda-python 13.1.1** - LOW-LEVEL ACCESS ⭐⭐⭐⭐
   ```bash
   pip install cuda-python
   ```
   - ✅ **CUDA 13.1 bindings** (full Blackwell support)
   - ✅ Direct access to CUDA Driver/Runtime APIs
   - ✅ For advanced GPU programming

**✅ FULLY WORKING:**

2. **PyTorch 2.9.1+cu128** - GPU WORKS! ⭐⭐⭐⭐⭐
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   - ✅ **FULL GPU support on sm_120** (verified 2026-01-01)
   - ✅ **15.94x average speedup** vs CPU
   - ✅ Mixed precision (FP16) works - additional 4.4x speedup
   - ✅ All tensor operations functional on RTX 5060 Ti
   - 📊 Benchmark: Small (8.9x), Medium (21x), Large (17.9x)
   - 🎯 **PRIMARY CHOICE** for Blessing Optimizer

**⚠️ PARTIALLY WORKING:**

**❌ NOT RECOMMENDED:**

5. **TensorFlow** - SKIP
   - ❌ No official sm_120 support in stable builds
   - ❌ Requires custom compilation or Docker workarounds
   - ❌ Overkill for numerical computations

### RTX 5060 Ti GPU Strategy

**For Blessing Optimizer - Recommended Stack (Updated 2026-01-01):**
```python
# PRIMARY: PyTorch GPU Acceleration (15.94x speedup!)
import torch

device = torch.device('cuda')  # RTX 5060 Ti sm_120 works!

def backtest_gpu(config, data):
    # Convert to GPU tensors
    prices = torch.tensor(data, device=device, dtype=torch.float32)

    # All operations on GPU (15.94x faster)
    ma = torch.mean(prices, dim=1, keepdim=True)
    signals = (prices > ma).float()
    returns = torch.diff(prices, dim=1)

    # Mixed precision for extra speed (4.4x)
    with torch.cuda.amp.autocast():
        cum_returns = torch.cumsum(returns * signals[:, :-1], dim=1)

    return cum_returns.cpu().numpy()  # Transfer back only result

# SECONDARY: CPU parallelization for multi-config optimization
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(backtest_gpu)(config, data) for config in configs
)

# TERTIARY: Numba JIT for CPU-bound hotspots (if needed)
from numba import njit
@njit
def calculate_indicators_fast(prices):
    return indicators
```

**Key Points (VERIFIED 2026-01-01):**
- ✅ **PyTorch 2.9.1+cu128 WORKS on RTX 5060 Ti!** 🎉
- ✅ **15.94x average GPU speedup** (Small: 8.9x, Medium: 21x, Large: 17.9x)
- ✅ **Mixed precision (FP16): 4.4x additional speedup**
- ✅ **Hybrid strategy:** GPU (15.94x) × CPU parallel (12x) = **191x potential!**
- ⚠️ CuPy 13.6.0: Partial support (wait for update)
- ✅ nvmath-python, cuda-python: Working

**Performance Strategy:**
1. **GPU (PyTorch):** Heavy numerical computations (15.94x)
2. **CPU parallelization:** Multi-config backtests (12x)
3. **Numba JIT:** CPU hotspots if needed (2-100x)
4. **Combined potential:** Up to **191x speedup** with hybrid approach!

### Optimization Strategy (in order of preference)

1. **Vectorization first:** NumPy/pandas (avoid for loops)
   ```python
   # Good
   df['signal'] = (df['close'] > df['ma']).astype(int)
   
   # Bad
   for i in range(len(df)):
       df.loc[i, 'signal'] = 1 if df.loc[i, 'close'] > df.loc[i, 'ma'] else 0
   ```

2. **Large tick files:** Read in chunks
   ```python
   for chunk in pd.read_csv('ticks.csv', chunksize=1_000_000):
       process(chunk)
   ```

3. **Parallelization:**
   - `multiprocessing` or `joblib` for CPU-bound tasks
   - `concurrent.futures` for I/O-bound tasks
   - Example:
     ```python
     from joblib import Parallel, delayed
     results = Parallel(n_jobs=-1)(delayed(process)(chunk) for chunk in chunks)
     ```

4. **GPU acceleration** (RTX 5060 Ti sm_120 - WORKING!):
   - **PyTorch (PRIMARY):** Use for GPU tensor operations (15.94x speedup)
     ```python
     import torch
     device = torch.device('cuda')  # RTX 5060 Ti sm_120 works!
     x = torch.tensor(data, device=device)
     y = torch.matmul(x, x.T)  # Computed on GPU

     # Mixed precision for extra speed
     with torch.cuda.amp.autocast():
         result = model(x)  # 4.4x faster with FP16
     ```
   - **nvmath-python:** For advanced math (BLAS, FFT, sparse matrices)
   - **CuPy:** Partial support (array creation OK, kernels fail - wait for update)
   - **Important:** PyTorch 2.9.1+cu128 fully supports RTX 5060 Ti!

5. **Additional optimizations:**
   - **Numba:** `@njit`, `parallel=True` for numeric loops
   - **Polars:** For very large dataframes (3-10x faster than pandas)
   - **Dask:** For distributed processing
   - **PyArrow:** For Parquet I/O

### Performance Measurement
**Always before final code:**
- Propose performance measurement (`%timeit`, `time.perf_counter`, `torch.utils.benchmark`)
- Explain acceleration choice or why not using it
- If task can benefit from GPU - use it; on problems, fallback to CPU with info

---

## DEPENDENCY MANAGEMENT

### Virtual Environments
- **Always use dedicated venv/conda** per project
- **Never mix environments** between projects

### Missing Packages
If a library is needed (numba, cupy, vectorbt, ta-lib, polars, dask, torch nightly, etc.):
1. Propose exact install command tailored to current environment
2. Ask for permission to execute `pip install` / `conda install`

### GPU Libraries Installation (RTX 5060 Ti)

**Recommended Installation Order:**

1. **PyTorch** (Primary GPU library - WORKING!):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   # ✅ CUDA 12.8 - Full sm_120 support on RTX 5060 Ti!
   # ✅ 15.94x average speedup verified
   ```

2. **nvmath-python** (Advanced math):
   ```bash
   pip install nvmath-python  # CUDA 13.1 support (verified working)
   ```

3. **cuda-python** (Low-level access):
   ```bash
   pip install cuda-python  # CUDA 13.1 bindings (verified working)
   ```

4. **CuPy** (Optional - partial support):
   ```bash
   pip install cupy-cuda12x  # For CUDA 12.x
   # ⚠️ Array creation works, compiled kernels fail (wait for update)
   ```

**Check Installation:**
```bash
nvidia-smi  # Check CUDA version
nvcc --version  # Check compiler
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; x = torch.randn(100,100, device='cuda'); print('GPU works!')"
```

### Requirements File
```txt
# requirements.txt - Always use pinned versions
pandas==2.1.0          # NOT >=2.0 or ~=2.0
numpy==1.24.3
torch==2.1.0+cu118     # Specify CUDA version
# ... etc
```

---

## PROJECT STRUCTURE (STANDARD)

```
project_root/
├── .env.example              # Template for environment variables
├── .gitignore                # ALWAYS exclude: data, logs, .env, proprietary
├── requirements.txt          # Pinned versions
├── README.md                 # Public: capabilities. Private: full docs
├── CLAUDE.md                 # This file
│
├── core/                     # Framework (can be public)
│   ├── __init__.py
│   ├── data_loader.py
│   ├── indicators.py
│   └── backtest_engine.py
│
├── strategies/               # PROPRIETARY (.gitignore)
│   ├── fractal_logic.py
│   └── ichimoku_rules.py
│
├── agents/                   # Multi-agent systems
│   ├── agent_1_analyzer.py
│   ├── agent_2_philosopher.py
│   └── agent_N_executor.py
│
├── utils/                    # Helpers
│   ├── __init__.py
│   ├── logging_system.py
│   └── gpu_utils.py
│
├── tests/                    # ALWAYS add tests
│   ├── __init__.py
│   └── test_*.py
│
├── data/                     # NEVER commit to Git
│   └── README.md             # Only structure description
│
└── logs/                     # NEVER commit to Git
    └── .gitkeep
```

---

## NAMING CONVENTIONS

- **Files:** `snake_case.py` (e.g., `ichimoku_analyzer.py`)
- **Classes:** `PascalCase` (e.g., `class Agent1Analyzer`)
- **Functions/variables:** `snake_case` (e.g., `def calculate_indicators`)
- **Constants:** `UPPER_CASE` (e.g., `MAX_DRAWDOWN = 0.20`)
- **Private methods:** `_leading_underscore` (e.g., `def _internal_calc`)
- **Booleans:** `is_/has_` prefix (e.g., `is_bullish`, `has_signal`)

---

## LOGGING & DEBUGGING

### Always Use Logging (Not print!)
```python
import logging

logger = logging.getLogger(__name__)
logger.info(f"Processing {symbol}...")
logger.debug(f"Intermediate result: {value}")
logger.warning(f"Low quality data for {symbol}")
logger.error(f"Failed to process: {error}")
```

### Log Levels
- **DEBUG:** Development/detailed diagnostics
- **INFO:** Production events
- **WARNING:** Unexpected but handled situations
- **ERROR:** Errors with stack traces

### Log Structure
```
logs/
├── system.log          # All events
├── trading.log         # Signals, trades
├── error.log          # Errors with tracebacks
└── performance.log    # Timing, resource usage
```

### File Rotation
- Max size: 10MB per file
- Keep: 5 backups
- Daily rotation for production

### Performance Tracking
```python
import time

start = time.perf_counter()
# ... code ...
elapsed = time.perf_counter() - start
logger.info(f"Execution time: {elapsed:.3f}s")
```

Or use decorator:
```python
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

@timeit
def expensive_function():
    # ...
```

---

## BACKTESTS & PATTERN RECOGNITION

### Backtest Requirements
- **Always run on unseen (test) data set**
- **Minimum metrics:** Sharpe ratio, max drawdown, win rate, profit factor
- **Display in table format**
- **Visualizations:** Equity curve, drawdown chart (matplotlib or plotly)

### Walk-Forward Analysis
When appropriate, use walk-forward validation:
1. Train on period 1
2. Test on period 2
3. Retrain on periods 1+2
4. Test on period 3
5. Continue...

---

## GIT WORKFLOW (for Claude Code)

### Branch Strategy
- `main` - stable, production-ready
- `develop` - active development
- `feature/xxx` - new features
- `fix/xxx` - bug fixes

### Commit Messages
Format: `[TYPE] Short description`

Types:
- `FEAT` - New feature
- `FIX` - Bug fix
- `REFACTOR` - Code restructuring (no functionality change)
- `DOCS` - Documentation only
- `TEST` - Adding/updating tests
- `PERF` - Performance improvements

Examples:
```
[FEAT] Add GPU acceleration for indicator calculation
[FIX] Correct symbol mapping for OANDA broker
[REFACTOR] Extract data loading to separate module
[DOCS] Add examples to README
[TEST] Add unit tests for Ichimoku calculation
[PERF] Optimize tick aggregation with Polars
```

### .gitignore (ALWAYS include)
```gitignore
# Environment
.env
*.env
venv/
.venv/

# Data (NEVER commit)
/data/tick_history/
/data/aggregated/
/data/enriched/
/logs/
*.log

# Proprietary code
/strategies/
/proprietary/
config_private.yaml

# IDE
.vscode/
.idea/
*.pyc
__pycache__/

# OS
.DS_Store
Thumbs.db
```

### Pre-commit Hooks
```bash
# Auto-format before commit
pip install pre-commit
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
```

---

## GITHUB & INTELLECTUAL PROPERTY

### Public vs Private Repositories

**Public Repositories** (showcase/demo only):
- Blessing Optimizer (general framework)
- YouTube RAG (culinary recipes)
- Pattern Recognition (generic)
- Framework code without strategies

**Private Repositories** (proprietary):
- Fractal Archaeologist
- Ichimoku Hosoda AI
- Projekt Michał
- Any profitable trading strategies

### Protection Strategy

**NEVER commit to public repos:**
- ❌ Proprietary trading logic
- ❌ Profitable strategies
- ❌ Optimized parameters
- ❌ Real backtest results (if revealing edge)
- ❌ API keys, passwords, credentials

**Safe to commit to public repos:**
- ✅ General framework architecture
- ✅ Data processing utilities
- ✅ Indicator calculations (standard)
- ✅ Visualization tools
- ✅ Documentation (capabilities, not logic)

### Public Repository Structure
```
public_project/
├── core/                    # Framework - PUBLIC
│   ├── data_loader.py
│   ├── indicators.py       # Standard indicators only
│   └── backtest_engine.py  # Generic engine
│
├── strategies/              # NOT COMMITTED (.gitignore)
│   └── strategy_template.py.example  # Template only
│
├── configs/
│   ├── config.yaml.example  # Template with placeholders
│   └── config.yaml          # Real config - .gitignore
│
└── README.md                # Describes capabilities, not logic
```

### Example: config.yaml.example
```yaml
# config.yaml.example - Public template
# Copy to config.yaml and fill in your values

data:
  tick_path: "/path/to/tick/data"  # PLACEHOLDER
  
broker:
  name: "YOUR_BROKER"
  api_key: "YOUR_API_KEY_HERE"     # PLACEHOLDER
  
strategy:
  # Parameters are proprietary - see documentation
  param1: 0.0  # PLACEHOLDER
  param2: 0.0  # PLACEHOLDER
```

---

## BROKER-SPECIFIC CONFIGURATIONS

### Symbol Mapping
Each project should have `config.yaml` with broker symbol mapping:

```yaml
symbol_mapping:
  EURUSD:
    oanda: "EUR_USD"
    dukascopy: "EURUSD"
    mt5_suffix: ".pro"        # Results in: EURUSD.pro
    
  GBPUSD:
    oanda: "GBP_USD"
    dukascopy: "GBPUSD"
    mt5_suffix: ".pro"
```

### MT5 Symbol Verification
```python
import MetaTrader5 as mt5

def verify_symbol(symbol: str) -> bool:
    """Verify symbol availability on broker"""
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"Symbol {symbol} not found")
        return False
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to enable {symbol}")
            return False
    return True
```

---

## TESTING (MANDATORY)

### Unit Tests
- **Requirement:** Unit tests for EVERY new function processing data
- **Framework:** pytest
- **Coverage:** Minimum 70% for core logic

### Test Structure
```
tests/
├── __init__.py
├── conftest.py              # Fixtures
├── test_indicators.py
├── test_data_processing.py
└── test_backtest.py
```

### Example Test with Fixtures
```python
import pytest
import pandas as pd

@pytest.fixture
def sample_ohlc():
    """Sample OHLC data for testing"""
    return pd.DataFrame({
        'open': [1.0, 1.1, 1.2],
        'high': [1.1, 1.2, 1.3],
        'low': [0.9, 1.0, 1.1],
        'close': [1.05, 1.15, 1.25],
    })

def test_calculate_ma(sample_ohlc):
    result = calculate_ma(sample_ohlc, period=2)
    assert len(result) == len(sample_ohlc)
    assert result.iloc[-1] == pytest.approx(1.20)
```

### Backtest Validation
- **Always:** Out-of-sample testing (second half of data)
- **Time-based split:** Never random (preserves temporal order)
- **No look-ahead bias:** Ensure all calculations use only past data

```python
# Good - time-based split
split_date = df.index[len(df)//2]
train = df[df.index < split_date]
test = df[df.index >= split_date]

# Bad - random split (breaks temporal order)
train = df.sample(frac=0.5)  # WRONG
```

---

## SKILLS / TECHNIQUES (Available in workspace)

### Public Skills (from /mnt/skills/public/)
- **docx:** Word document creation, editing, tracked changes
- **pdf:** PDF manipulation, form filling, text extraction
- **pptx:** Presentation creation and editing
- **xlsx:** Spreadsheet creation, formulas, data analysis
- **product-self-knowledge:** Anthropic product capabilities reference
- **frontend-design:** Production-grade UI/web components

### Example Skills (from /mnt/skills/examples/)
- **skill-creator:** Guide for creating custom skills

**Usage:** Before starting complex document/data work, view relevant SKILL.md:
```python
# Example: Before creating Excel report
view_tool('/mnt/skills/public/xlsx/SKILL.md')
```

---

## EXAMPLE DECISION FLOW

### Request: "Summarize this attached file"
→ File is in context → Use provided content, do NOT use view tool

### Request: "Fix the bug in my Python file" + attachment
→ File mentioned → Check `/mnt/user-data/uploads` → Copy to `/home/claude` to iterate/lint/test → Provide back in `/mnt/user-data/outputs`

### Request: "What are the top video game companies by net worth?"
→ Knowledge question → Answer directly, NO tools needed

### Request: "Write a blog post about AI trends"
→ Content creation → CREATE actual `.md` file in `/mnt/user-data/outputs`, don't just output text

### Request: "Create a React component for user login"
→ Code component → CREATE actual `.jsx` file(s) in `/home/claude` then move to `/mnt/user-data/outputs`

### Request: "Search for and compare how NYT vs WSJ covered the Fed rate decision"
→ Web search task → Respond CONVERSATIONALLY in chat (no file creation, no report-style headers, concise prose)

---

## CRITICAL REMINDERS

1. **NO CODING WITHOUT CONFIRMATION** - Always propose approach first
2. **CONCISE RESPONSES** - Token efficiency is priority
3. **OPTIMIZATION ALWAYS** - Propose GPU/multiprocessing/vectorization when applicable
4. **SECURITY FIRST** - Never expose credentials or proprietary logic
5. **TEST EVERYTHING** - Unit tests for data processing, out-of-sample for backtests
6. **COMMIT SAFELY** - Use .gitignore, never commit data/logs/secrets
7. **DIAGNOSTICS FIRST** - Check environment before major changes
8. **FALLBACK READY** - If GPU issues (RTX 5060 Ti sm_120), use CPU without blocking
9. **ASK FOR PERMISSION** - Before running commands, installing packages, or major changes
10. **CHAT IN POLISH** - Technical docs in English, conversations in Polish for efficiency

---

## SUMMARY

This file ensures:
- ✅ **Fast, efficient work** - No wasted tokens, straight to the point
- ✅ **Optimized performance** - GPU when possible, CPU fallback when needed
- ✅ **Secure development** - Protected IP, no credential leaks
- ✅ **Professional quality** - Tests, logging, documentation
- ✅ **Reproducible results** - Structured projects, pinned dependencies
- ✅ **GitHub-ready** - Public showcase, private proprietary code

**Let's build great trading systems together!** 🚀

---

*Last updated: 2026-01-01*  
*Author: Rafał Wiśniewski | Data & AI Solutions*
