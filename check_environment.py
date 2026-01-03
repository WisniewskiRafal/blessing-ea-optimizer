"""
Blessing EA Optimizer - Environment Check
Detects hardware, verifies dependencies, provides recommendations
"""

import sys
import subprocess
import platform
from pathlib import Path


def check_python():
    v = sys.version_info
    print(f'[1/7] Python version... {v.major}.{v.minor}.{v.micro}', end=' ')
    if v.major < 3 or (v.major == 3 and v.minor < 11):
        print('[ERROR]')
        return False
    print('[OK]')
    return True


def check_packages():
    print('[2/7] Core packages...', end=' ')
    try:
        import pandas, numpy, matplotlib
        print('[OK]')
        return True
    except ImportError as e:
        print(f'[ERROR] {e}')
        return False


def check_optimization():
    print('[3/7] Optimization libs...', end=' ')
    missing = []
    try:
        import skopt
    except:
        missing.append('scikit-optimize')
    try:
        import deap
    except:
        missing.append('deap')
    
    if missing:
        print(f'[WARNING] Missing: {missing}')
        return False
    print('[OK]')
    return True


def check_gpu():
    print('[4/7] GPU support...', end=' ')
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            print(f'[OK] {gpu_name}')
            
            try:
                import torch
                if torch.cuda.is_available():
                    print(f'    PyTorch CUDA: [OK]')
                else:
                    print(f'    PyTorch CUDA: [WARNING] Not available')
            except:
                print(f'    PyTorch: [WARNING] Not installed')
            return True
    except:
        pass
    print('[INFO] CPU only')
    return False


def check_cpu():
    import multiprocessing
    cores = multiprocessing.cpu_count()
    print(f'[5/7] CPU cores... {cores} [OK]')
    return True


def check_data():
    print('[6/7] Data directory...', end=' ')
    data_dir = Path('d:/tick_data')
    if data_dir.exists():
        csv_count = len(list(data_dir.glob('*.csv')))
        print(f'[OK] {csv_count} CSV files')
    else:
        print('[WARNING] Not found')
    return True


def check_structure():
    print('[7/7] Project files...', end=' ')
    required = ['blessing_optimizer_main.py', 'core/blessing_backtest_engine.py']
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print(f'[ERROR] Missing: {missing}')
        return False
    print('[OK]')
    return True


if __name__ == '__main__':
    print('='*60)
    print('BLESSING EA OPTIMIZER - ENVIRONMENT CHECK')
    print('='*60)
    
    all_ok = all([
        check_python(),
        check_packages(),
        check_optimization(),
        check_gpu(),
        check_cpu(),
        check_data(),
        check_structure()
    ])
    
    print('='*60)
    if all_ok:
        print('[SUCCESS] Ready to run!')
        print('Next: python blessing_optimizer_main.py')
    else:
        print('[WARNING] Some issues found')
        print('Install missing: pip install -r requirements.txt')
    print('='*60)
