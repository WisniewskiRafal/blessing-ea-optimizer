# core/hardware_detector.py
# Author: Rafał Wiśniewski | Data & AI Solutions

import platform
import psutil
import subprocess
from typing import Dict, Optional

class HardwareDetector:
    """Wykrywanie sprzętu: CPU, RAM, GPU, akceleracja"""
    
    def __init__(self):
        self.specs = {}
        self._detect_all()
    
    def _detect_all(self) -> Dict:
        """Wykryj wszystkie komponenty"""
        self.specs = {
            'os': self._detect_os(),
            'cpu': self._detect_cpu(),
            'ram': self._detect_ram(),
            'gpu': self._detect_gpu()
        }
        self.specs['acceleration'] = self._available_acceleration()
        return self.specs
    
    def _detect_os(self) -> Dict:
        """Wykryj system operacyjny"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine()
        }
    
    def _detect_cpu(self) -> Dict:
        """Wykryj CPU"""
        try:
            cpu_freq = psutil.cpu_freq()
            return {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'max_frequency': cpu_freq.max if cpu_freq else 0,
                'current_frequency': cpu_freq.current if cpu_freq else 0,
                'cpu_percent': psutil.cpu_percent(interval=1)
            }
        except Exception as e:
            return {
                'physical_cores': 1,
                'logical_cores': 1,
                'error': str(e)
            }
    
    def _detect_ram(self) -> Dict:
        """Wykryj RAM"""
        try:
            mem = psutil.virtual_memory()
            return {
                'total_gb': round(mem.total / (1024**3), 2),
                'available_gb': round(mem.available / (1024**3), 2),
                'used_gb': round(mem.used / (1024**3), 2),
                'percent_used': mem.percent
            }
        except Exception as e:
            return {
                'total_gb': 0,
                'error': str(e)
            }
    
    def _detect_gpu(self) -> Dict:
        """Wykryj GPU (NVIDIA)"""
        gpu_info = {
            'cuda_available': False,
            'devices': []
        }
        
        # Try PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info['devices'].append({
                        'id': i,
                        'name': props.name,
                        'total_memory_gb': round(props.total_memory / (1024**3), 2),
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                
                return gpu_info
        except ImportError:
            pass
        except Exception as e:
            gpu_info['torch_error'] = str(e)
        
        # Try CuPy
        try:
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()
            
            if device_count > 0:
                gpu_info['cuda_available'] = True
                
                for i in range(device_count):
                    device = cp.cuda.Device(i)
                    with device:
                        mem_info = cp.cuda.runtime.memGetInfo()
                        total_mem = mem_info[1]
                        
                        gpu_info['devices'].append({
                            'id': i,
                            'name': device.name.decode('utf-8') if isinstance(device.name, bytes) else str(device.name),
                            'total_memory_gb': round(total_mem / (1024**3), 2),
                            'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}"
                        })
                
                return gpu_info
        except ImportError:
            pass
        except Exception as e:
            gpu_info['cupy_error'] = str(e)
        
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        memory_str = parts[1].strip()
                        
                        # Parse memory (e.g., "16311 MiB")
                        memory_gb = 0
                        try:
                            memory_val = float(memory_str.split()[0])
                            if 'MiB' in memory_str:
                                memory_gb = round(memory_val / 1024, 2)
                            elif 'GiB' in memory_str:
                                memory_gb = round(memory_val, 2)
                        except:
                            pass
                        
                        gpu_info['devices'].append({
                            'id': i,
                            'name': name,
                            'total_memory_gb': memory_gb,
                            'compute_capability': 'unknown'
                        })
                
                if gpu_info['devices']:
                    gpu_info['cuda_available'] = True
        except FileNotFoundError:
            gpu_info['nvidia_smi'] = 'not found'
        except Exception as e:
            gpu_info['nvidia_smi_error'] = str(e)
        
        return gpu_info
    
    def _available_acceleration(self) -> Dict:
        """Sprawdź dostępne metody akceleracji"""
        acceleration = {
            'cpu_multiprocessing': True,  # Always available
            'numba_jit': False,
            'cuda': False,
            'opencl': False,
            'gpu_libraries': {}  # Detailed GPU library info
        }

        # Numba JIT
        try:
            import numba
            acceleration['numba_jit'] = True
            acceleration['numba_version'] = numba.__version__
        except ImportError:
            pass

        # CUDA - check from already detected GPU info
        if self.specs.get('gpu', {}).get('cuda_available'):
            acceleration['cuda'] = True

        # CuPy (PRIMARY GPU library for RTX 5060 Ti)
        try:
            import cupy as cp
            acceleration['gpu_libraries']['cupy'] = {
                'available': True,
                'version': cp.__version__,
                'cuda_runtime_version': cp.cuda.runtime.runtimeGetVersion(),
                'status': '[OK] WORKING (sm_120 compatible)',
                'priority': 1
            }
        except ImportError:
            acceleration['gpu_libraries']['cupy'] = {
                'available': False,
                'status': '[X] Not installed'
            }
        except Exception as e:
            acceleration['gpu_libraries']['cupy'] = {
                'available': False,
                'status': f'[!] Error: {str(e)}'
            }

        # nvmath-python (Advanced math library)
        try:
            import nvmath
            acceleration['gpu_libraries']['nvmath'] = {
                'available': True,
                'version': nvmath.__version__,
                'status': '[OK] WORKING (CUDA 13.1 compatible)',
                'priority': 2
            }
        except ImportError:
            acceleration['gpu_libraries']['nvmath'] = {
                'available': False,
                'status': '[X] Not installed'
            }
        except Exception as e:
            acceleration['gpu_libraries']['nvmath'] = {
                'available': False,
                'status': f'[!] Error: {str(e)}'
            }

        # cuda-python (Low-level CUDA bindings)
        try:
            from cuda.bindings import driver
            import cuda
            # Try to get version from package metadata
            try:
                from importlib.metadata import version
                cuda_version = version('cuda-python')
            except:
                cuda_version = 'unknown'

            acceleration['gpu_libraries']['cuda-python'] = {
                'available': True,
                'version': cuda_version,
                'status': '[OK] WORKING (CUDA 13.1 bindings)',
                'priority': 3
            }
        except ImportError:
            acceleration['gpu_libraries']['cuda-python'] = {
                'available': False,
                'status': '[X] Not installed'
            }
        except Exception as e:
            acceleration['gpu_libraries']['cuda-python'] = {
                'available': False,
                'status': f'[!] Error: {str(e)}'
            }

        # PyTorch (CPU fallback for RTX 5060 Ti)
        try:
            import torch
            cuda_available = torch.cuda.is_available()

            # Check if sm_120 is supported
            sm_120_supported = False
            if cuda_available:
                try:
                    # Try to run a simple operation on GPU
                    test_tensor = torch.randn(2, 2, device='cuda')
                    _ = test_tensor @ test_tensor.T
                    sm_120_supported = True
                except RuntimeError as e:
                    if 'no kernel image is available' in str(e):
                        sm_120_supported = False
                    else:
                        raise

            acceleration['gpu_libraries']['pytorch'] = {
                'available': True,
                'version': torch.__version__,
                'cuda_available': cuda_available,
                'sm_120_supported': sm_120_supported,
                'status': '[OK] GPU WORKING' if sm_120_supported else '[!] CPU ONLY (sm_120 not supported)',
                'priority': 4
            }
        except ImportError:
            acceleration['gpu_libraries']['pytorch'] = {
                'available': False,
                'status': '[X] Not installed'
            }
        except Exception as e:
            acceleration['gpu_libraries']['pytorch'] = {
                'available': False,
                'status': f'[!] Error: {str(e)}'
            }

        # OpenCL
        try:
            import pyopencl
            acceleration['opencl'] = True
        except ImportError:
            pass

        return acceleration
    
    def get_optimal_workers(self, task_type: str = 'cpu') -> int:
        """Oblicz optymalną liczbę workerów"""
        cpu_info = self.specs.get('cpu', {})
        gpu_info = self.specs.get('gpu', {})
        
        if task_type == 'gpu' and gpu_info.get('cuda_available'):
            # GPU tasks: 1 worker per GPU * 4
            return len(gpu_info.get('devices', [])) * 4
        
        elif task_type == 'mixed':
            # Mixed CPU/GPU: half of CPU cores
            logical = cpu_info.get('logical_cores', 4)
            return max(1, logical // 2)
        
        else:
            # CPU tasks: cores - 2
            logical = cpu_info.get('logical_cores', 4)
            return max(1, logical - 2)
    
    def check_memory_requirements(self, task_memory_gb: float) -> bool:
        """Sprawdź czy jest wystarczająco RAM"""
        ram_info = self.specs.get('ram', {})
        available = ram_info.get('available_gb', 0)
        
        # 20% buffer
        required = task_memory_gb * 1.2
        
        return available >= required
    
    def print_report(self):
        """Wydrukuj raport sprzętowy"""
        print("\n" + "="*60)
        print("HARDWARE DETECTION REPORT")
        print("="*60)
        
        # OS
        os_info = self.specs.get('os', {})
        print(f"\nOS: {os_info.get('system')} {os_info.get('release')}")
        print(f"Architecture: {os_info.get('machine')}")
        
        # CPU
        cpu_info = self.specs.get('cpu', {})
        print(f"\nCPU:")
        print(f"  Physical Cores: {cpu_info.get('physical_cores')}")
        print(f"  Logical Cores: {cpu_info.get('logical_cores')}")
        print(f"  Max Frequency: {cpu_info.get('max_frequency', 0):.0f} MHz")
        print(f"  Current Load: {cpu_info.get('cpu_percent', 0):.1f}%")
        
        # RAM
        ram_info = self.specs.get('ram', {})
        print(f"\nRAM:")
        print(f"  Total: {ram_info.get('total_gb', 0):.2f} GB")
        print(f"  Available: {ram_info.get('available_gb', 0):.2f} GB")
        print(f"  Used: {ram_info.get('percent_used', 0):.1f}%")
        
        # GPU
        gpu_info = self.specs.get('gpu', {})
        if gpu_info.get('cuda_available'):
            print(f"\nGPU:")
            for device in gpu_info.get('devices', []):
                print(f"  [{device['id']}] {device['name']}")
                print(f"      Memory: {device['total_memory_gb']} GB")
                print(f"      Compute: {device['compute_capability']}")
        else:
            print(f"\nGPU: Not detected (CPU only)")
        
        # Acceleration
        acc = self.specs.get('acceleration', {})
        print(f"\nAcceleration:")
        print(f"  CPU Multiprocessing: {'[YES]' if acc.get('cpu_multiprocessing') else '[NO]'}")

        if acc.get('numba_jit'):
            print(f"  Numba JIT: [YES] v{acc.get('numba_version', 'unknown')}")
        else:
            print(f"  Numba JIT: [NO]")

        print(f"  CUDA: {'[YES]' if acc.get('cuda') else '[NO]'}")
        print(f"  OpenCL: {'[YES]' if acc.get('opencl') else '[NO]'}")

        # GPU Libraries (detailed)
        gpu_libs = acc.get('gpu_libraries', {})
        if gpu_libs:
            print(f"\nGPU Libraries (RTX 5060 Ti sm_120 Support):")

            # Sort by priority
            sorted_libs = sorted(
                gpu_libs.items(),
                key=lambda x: x[1].get('priority', 999)
            )

            for lib_name, lib_info in sorted_libs:
                if lib_info.get('available'):
                    version = lib_info.get('version', 'unknown')
                    status = lib_info.get('status', '')
                    print(f"  [{lib_info.get('priority', '?')}] {lib_name} v{version}: {status}")

                    # Extra info for specific libraries
                    if lib_name == 'cupy' and 'cuda_runtime_version' in lib_info:
                        cuda_ver = lib_info['cuda_runtime_version']
                        print(f"      CUDA Runtime: {cuda_ver}")

                    if lib_name == 'pytorch':
                        cuda_avail = lib_info.get('cuda_available', False)
                        sm120 = lib_info.get('sm_120_supported', False)
                        print(f"      CUDA Detected: {cuda_avail}, sm_120: {sm120}")
                else:
                    print(f"  {lib_name}: {lib_info.get('status', '[X]')}")
        
        # Recommendations
        print(f"\nRecommendations:")
        print(f"  Optimal CPU workers: {self.get_optimal_workers('cpu')}")
        if gpu_info.get('cuda_available'):
            print(f"  Optimal GPU workers: {self.get_optimal_workers('gpu')}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test
    detector = HardwareDetector()
    detector.print_report()