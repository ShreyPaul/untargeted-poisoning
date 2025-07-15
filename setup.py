"""
Environment Setup Module
=======================

This module is responsible for:
- Checking system dependencies
- Initializing experiment environment
- Validating configuration parameters
- Preparing datasets

Author: Security Research Team
Date: 2024
"""

import os
import sys
import importlib
import warnings
from typing import Dict, List, Tuple, Any
from pathlib import Path

# 抑制警告信息
warnings.filterwarnings('ignore')

class EnvironmentSetup:
    """環境設置類"""
    
    def __init__(self):
        self.required_modules = [
            'torch', 'torchvision', 'numpy', 'json', 'datetime'
        ]
        self.optional_modules = [
            'matplotlib', 'seaborn', 'pandas'
        ]
        self.setup_status = {
            'dependencies': False,
            'data_directory': False,
            'config_loaded': False
        }
        
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check required dependencies"""
        missing_modules = []
        print("Checking system dependencies...")
        for module in self.required_modules:
            try:
                if module == 'torch':
                    import torch
                elif module == 'torchvision':
                    import torchvision
                elif module == 'numpy':
                    import numpy as np
                else:
                    importlib.import_module(module)
            except ImportError:
                missing_modules.append(module)
        if missing_modules:
            print(f"Missing required modules: {missing_modules}")
        self.setup_status['dependencies'] = len(missing_modules) == 0
        return len(missing_modules) == 0, missing_modules
    
    def setup_data_directory(self) -> bool:
        """Set up data directory"""
        data_dir = Path('./data')
        try:
            data_dir.mkdir(exist_ok=True)
            self.setup_status['data_directory'] = True
            return True
        except Exception as e:
            print(f"Failed to create data directory: {e}")
            return False
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            from config import AttackConfig
            config = AttackConfig()
            self.setup_status['config_loaded'] = True
            return config
        except ImportError:
            print("config.py not found, using default config.")
            return self._get_default_config()
    
    def _get_default_config(self) -> object:
        """獲取默認配置"""
        class DefaultConfig:
            NUM_HONEST_CLIENTS = 5
            NUM_SYBIL_CLIENTS = 3
            DATASET_NAME = 'MNIST'
            LEARNING_RATE_HONEST = 0.01
            LEARNING_RATE_SYBIL = 0.05
            POISON_RATIO = 0.3
            TOTAL_ROUNDS = 12
            ATTACK_START_ROUND = 3
            
        return DefaultConfig()
    
    def test_torch_functionality(self) -> bool:
        """Test basic PyTorch functionality"""
        try:
            import torch
            import torch.nn as nn
            x = torch.randn(2, 3)
            y = torch.randn(2, 3)
            z = x + y
            model = nn.Linear(3, 1)
            output = model(x)
            if torch.cuda.is_available():
                pass  # CUDA available, no need to print
            return True
        except Exception as e:
            print(f"PyTorch test failed: {e}")
            return False
    
    def display_system_info(self):
        """Display system information"""
        print(f"Python version: {sys.version.split()[0]}")
        print(f"OS: {os.name}")
        print(f"Working directory: {os.getcwd()}")
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
        except ImportError:
            pass
    
    def run_complete_setup(self) -> Tuple[bool, object]:
        """Run complete environment setup"""
        print("Starting environment setup...")
        self.display_system_info()
        deps_ok, missing = self.check_dependencies()
        if not deps_ok:
            print("Please install the missing modules and retry.")
            return False, None
        if not self.setup_data_directory():
            print("Data directory setup failed.")
            return False, None
        if not self.test_torch_functionality():
            print("PyTorch functionality test failed.")
            return False, None
        config = self.load_config()
        print("Environment setup complete.")
        return True, config
    
    def get_setup_status(self) -> Dict[str, bool]:
        """獲取設置狀態"""
        return self.setup_status.copy()

def quick_setup() -> Tuple[bool, object]:
    """快速環境設置"""
    setup = EnvironmentSetup()
    return setup.run_complete_setup()

def validate_environment() -> bool:
    """Validate if environment is set up correctly"""
    setup = EnvironmentSetup()
    required_files = ['environment.py', 'attack.py']
    for file in required_files:
        if not Path(file).exists():
            print(f"Missing required file: {file}")
            return False
    deps_ok, _ = setup.check_dependencies()
    if not deps_ok:
        return False
    try:
        from environment import FederatedLearningEnvironment
        from attack import SybilAttackOrchestrator
        return True
    except ImportError as e:
        print(f"Module import failed: {e}")
        return False

def install_requirements():
    """Dependency installation guide"""
    print("To install dependencies, run:")
    print("  pip install -r requirements.txt")
    print("For CUDA support, visit https://pytorch.org/ for the correct command.")

if __name__ == "__main__":
    success, config = quick_setup()
    if success:
        print("Setup successful. You can now run the attack script with: python main.py")
    else:
        print("Setup failed. Please check the error messages above.")
        install_requirements() 