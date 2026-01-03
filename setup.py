"""
Blessing EA Optimizer - Automated Setup
Detects environment and installs all required dependencies

Author: Rafał Wiśniewski | Data & AI Solutions
Version: 3.0
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

class BlessingInstaller:
    def __init__(self):
        self.python_version = sys.version_info
        self.os_type = platform.system()
        self.errors = []
        self.project_root = Path(__file__).parent
        self.venv_dir = self.project_root / "venv"
        self.is_in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
    def check_python_version(self):
        """Sprawdź wersję Python (min 3.10)"""
        print("🔍 Checking Python version...")
        if self.python_version < (3, 10):
            print(f"❌ Python 3.10+ required. Current: {self.python_version.major}.{self.python_version.minor}")
            print("\n📥 Please install Python 3.10 or higher from https://www.python.org/downloads/")
            input("\nPress Enter to exit...")
            sys.exit(1)
        else:
            print(f"✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
    
    def create_venv(self):
        """Utwórz virtual environment"""
        print("\n🔨 Creating virtual environment...")
        
        if self.venv_dir.exists():
            print(f" ℹ️ Virtual environment already exists at {self.venv_dir}")
            response = input("  Delete and recreate? (y/N): ").lower()
            if response == 'y':
                import shutil
                print("  🗑️ Removing old venv...")
                shutil.rmtree(self.venv_dir)
            else:
                print("  ✅ Using existing venv")
                return
        
        try:
            print(f"  Creating venv at: {self.venv_dir}")
            venv.create(self.venv_dir, with_pip=True, clear=True)
            print("  ✅ Virtual environment created")
        except Exception as e:
            print(f"  ❌ Failed to create venv: {str(e)}")
            input("\nPress Enter to exit...")
            sys.exit(1)
    
    def get_venv_python(self):
        """Pobierz ścieżkę do Python w venv"""
        if self.os_type == "Windows":
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"
    
    def get_venv_pip(self):
        """Pobierz ścieżkę do pip w venv"""
        if self.os_type == "Windows":
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"
    
    def upgrade_pip(self):
        """Upgrade pip w venv"""
        print("\n📦 Upgrading pip...")
        
        pip_path = self.get_venv_pip()
        python_path = self.get_venv_python()
        
        try:
            subprocess.run(
                [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True
            )
            print("  ✅ pip upgraded")
        except Exception as e:
            print(f"  ⚠️ pip upgrade warning: {str(e)}")
    
    def detect_hardware(self):
        """Wykryj CPU/GPU"""
        print("\n🖥️ Hardware Detection:")
        
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"  CPU Cores: {cpu_count}")
        
        # GPU check
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    name, memory = line.split(',')
                    print(f"  GPU [{i}]: {name.strip()} ({memory.strip()})")
                return {'cpu': cpu_count, 'gpu': True}
            else:
                print("  GPU: Not detected (CPU only)")
                return {'cpu': cpu_count, 'gpu': False}
        except:
            print("  GPU: Not detected (CPU only)")
            return {'cpu': cpu_count, 'gpu': False}
    
    def install_requirements(self):
        """Instaluj dependencies w venv"""
        print("\n📦 Installing packages in virtual environment...")
        print("  This may take several minutes...")
        
        pip_path = self.get_venv_pip()
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print(f"  ❌ requirements.txt not found at {requirements_file}")
            input("\nPress Enter to exit...")
            sys.exit(1)
        
        try:
            # Install from requirements.txt
            print(f"\n  Installing from {requirements_file}...")
            result = subprocess.run(
                [str(pip_path), "install", "-r", str(requirements_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("  ✅ All packages installed")
            else:
                print("  ⚠️ Some packages had warnings")
                if result.stderr:
                    # Don't print full errors, just summary
                    print("  Check installation_log.txt for details")
                    with open("installation_log.txt", "w", encoding="utf-8") as f:
                        f.write(result.stderr)
            
        except Exception as e:
            print(f"  ❌ Installation failed: {str(e)}")
            self.errors.append("requirements installation")
    
    def install_gpu_packages(self, hw):
        """Instaluj pakiety GPU (opcjonalnie)"""
        if not hw.get('gpu'):
            return
        
        print("\n🎮 GPU packages detected. Install GPU support?")
        print("  This will install PyTorch and CuPy (~2GB download)")
        response = input("  Install GPU packages? (y/N): ").lower()
        
        if response != 'y':
            print("  ⏭️ Skipping GPU packages")
            return
        
        pip_path = self.get_venv_pip()
        
        gpu_packages = [
            'torch>=2.1.0',
            'cupy-cuda12x>=12.3.0',
        ]
        
        print("\n  Installing GPU packages...")
        for pkg in gpu_packages:
            try:
                pkg_name = pkg.split('>=')[0]
                print(f"  Installing {pkg_name}...")
                subprocess.run(
                    [str(pip_path), "install", pkg],
                    capture_output=True,
                    check=True
                )
                print(f"  ✅ {pkg_name}")
            except:
                print(f"  ⚠️ {pkg_name} - skipped")
    
    def create_config_template(self):
        """Stwórz .env template"""
        print("\n⚙️ Creating config template...")
        
        env_template = """# Blessing EA Optimizer Configuration
# Author: Rafal Wisniewski | Data & AI Solutions

MT4_LOGIN=
MT4_PASSWORD=
MT4_SERVER=
MT4_PATH=

MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
MT5_PATH=

MAX_PARALLEL_JOBS=auto
USE_GPU=auto
CACHE_DIR=./cache
OUTPUT_DIR=./output/optimized_sets

LOG_LEVEL=INFO
"""
        
        env_file = self.project_root / ".env.template"
        if not env_file.exists():
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(env_template)
            print("  ✅ .env.template created")
    
    def create_directories(self):
        """Utwórz niezbędne katalogi"""
        print("\n📁 Creating directories...")
        
        dirs = [
            'cache/tick_data',
            'cache/ohlc_data',
            'output/optimized_sets',
            'output/results',
            'output/reports',
            'output/charts',
            'logs',
        ]
        
        for dir_path in dirs:
            (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        print("  ✅ Directories created")
    
    def create_activation_scripts(self):
        """Utwórz skrypty aktywacji venv"""
        print("\n📝 Creating activation scripts...")
        
        # Windows batch - PROSTE ZNAKI ASCII
        if self.os_type == "Windows":
            activate_bat = self.project_root / "activate.bat"
            with open(activate_bat, "w", encoding="ascii", errors="ignore") as f:
                f.write("""@echo off
echo.
echo ============================================================
echo   Blessing EA Optimizer - Virtual Environment
echo   Rafal Wisniewski - Data and AI Solutions
echo ============================================================
echo.
call venv\\Scripts\\activate.bat
echo [OK] Virtual environment activated
echo.
echo Available commands:
echo   streamlit run dashboard.py  - Start dashboard
echo   python setup.py             - Reinstall packages
echo   deactivate                  - Exit venv
echo.
""")
            
            # Run script
            run_bat = self.project_root / "run.bat"
            with open(run_bat, "w", encoding="ascii", errors="ignore") as f:
                f.write("""@echo off
call venv\\Scripts\\activate.bat
streamlit run dashboard.py
""")
            
            print("  ✅ activate.bat created")
            print("  ✅ run.bat created")
        
        # Linux/Mac bash
        else:
            activate_sh = self.project_root / "activate.sh"
            with open(activate_sh, "w", encoding="utf-8") as f:
                f.write("""#!/bin/bash
echo ""
echo "============================================================"
echo "  Blessing EA Optimizer - Virtual Environment"
echo "  Rafal Wisniewski - Data and AI Solutions"
echo "============================================================"
echo ""
source venv/bin/activate
echo "[OK] Virtual environment activated"
echo ""
echo "Available commands:"
echo "  streamlit run dashboard.py  - Start dashboard"
echo "  python setup.py             - Reinstall packages"
echo "  deactivate                  - Exit venv"
echo ""
""")
            
            # Run script
            run_sh = self.project_root / "run.sh"
            with open(run_sh, "w", encoding="utf-8") as f:
                f.write("""#!/bin/bash
source venv/bin/activate
streamlit run dashboard.py
""")
            
            # Make executable
            activate_sh.chmod(0o755)
            run_sh.chmod(0o755)
            
            print("  ✅ activate.sh created")
            print("  ✅ run.sh created")
    
    def run(self):
        """Uruchom pełną instalację"""
        print("""
============================================================
                                                          
          BLESSING EA OPTIMIZER - SETUP                  
                                                          
        Rafal Wisniewski | Data & AI Solutions          
                                                          
        Installing in isolated virtual environment       
                                                          
============================================================
        """)
        
        try:
            # Check if already in venv
            if self.is_in_venv:
                print("⚠️ Already running in a virtual environment!")
                print("   This may cause conflicts.")
                response = input("\nContinue anyway? (y/N): ").lower()
                if response != 'y':
                    print("\n✋ Installation cancelled")
                    input("\nPress Enter to exit...")
                    sys.exit(0)
            
            self.check_python_version()
            self.create_venv()
            self.upgrade_pip()
            
            hw = self.detect_hardware()
            
            self.install_requirements()
            self.install_gpu_packages(hw)
            self.create_config_template()
            self.create_directories()
            self.create_activation_scripts()
            
            print("\n" + "="*60)
            if self.errors:
                print(f"⚠️ Installation completed with {len(self.errors)} warnings:")
                for err in self.errors:
                    print(f"  - {err}")
            else:
                print("✅ Installation completed successfully!")
            
            print("\n📌 Next steps:")
            print("="*60)
            
            if self.os_type == "Windows":
                print("\n  1. Run: activate.bat")
                print("     (or double-click activate.bat)")
                print("\n  2. Copy .env.template to .env and fill credentials")
                print("\n  3. Run: run.bat")
                print("     (or: streamlit run dashboard.py)")
            else:
                print("\n  1. Run: source activate.sh")
                print("\n  2. Copy .env.template to .env and fill credentials")
                print("\n  3. Run: ./run.sh")
                print("     (or: streamlit run dashboard.py)")
            
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Installation cancelled by user")
        except Exception as e:
            print(f"\n❌ Installation failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            input("\nPress Enter to exit...")

if __name__ == "__main__":
    installer = BlessingInstaller()
    installer.run()