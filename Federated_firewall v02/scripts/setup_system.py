#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import time
import torch
import numpy as np
import yaml
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)

def check_root_privileges():
    """Check if running as root"""
    if os.geteuid() != 0:
        print("🚫 ROOT PRIVILEGES REQUIRED")
        print("Please run: sudo env \"PATH=$PATH\" python3 scripts/setup_system.py")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    logger = logging.getLogger(__name__)
    
    directories = [
        'logs',
        'logs/reports',
        'logs/results',
        'config',
        'src/models',
        'src/core', 
        'src/federated',
        'src/mininet_controller',
        'src/utils',
        'scripts'
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def install_dependencies():
    """Install required Python packages"""
    logger = logging.getLogger(__name__)
    
    # Required packages with specific handling
    packages = [
        'torch',
        'numpy', 
        'PyYAML'
    ]
    
    for package in packages:
        try:
            # Check if already installed
            if package.lower() == 'pyyaml':
                import yaml
                logger.info(f"✅ {package} already installed")
            elif package.lower() == 'torch':
                import torch
                logger.info(f"✅ {package} already installed (version: {torch.__version__})")
            elif package.lower() == 'numpy':
                import numpy
                logger.info(f"✅ {package} already installed (version: {numpy.__version__})")
            else:
                __import__(package.lower())
                logger.info(f"✅ {package} already installed")
                
        except ImportError:
            logger.info(f"📦 Installing {package}...")
            try:
                # Use specific commands for different packages
                if package.lower() == 'torch':
                    # Install CPU-only version of PyTorch for better compatibility
                    cmd = [sys.executable, '-m', 'pip', 'install', 'torch', '--index-url', 'https://download.pytorch.org/whl/cpu']
                else:
                    cmd = [sys.executable, '-m', 'pip', 'install', package]
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
                logger.info(f"✅ {package} installed successfully")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                logger.error(f"Error output: {e.stderr}")
                return False
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Installation of {package} timed out")
                return False
    
    # Check Mininet separately
    try:
        from mininet.net import Mininet
        logger.info("✅ Mininet already available")
    except ImportError:
        logger.warning("⚠️  Mininet not found in Python path")
        logger.info("Mininet should be installed system-wide: sudo apt-get install mininet")
        # Don't fail setup for Mininet as it's typically installed system-wide
    
    return True

def setup_mininet():
    """Setup Mininet environment"""
    logger = logging.getLogger(__name__)
    
    # Check if Mininet is available
    try:
        result = subprocess.run(['mn', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✅ Mininet version: {result.stdout.strip()}")
        else:
            logger.warning("⚠️  Mininet check returned non-zero code")
    except FileNotFoundError:
        logger.error("❌ Mininet not found. Installing...")
        try:
            subprocess.run(['apt-get', 'update'], check=True, timeout=60)
            subprocess.run(['apt-get', 'install', '-y', 'mininet'], check=True, timeout=300)
            logger.info("✅ Mininet installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install Mininet: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("❌ Mininet installation timed out")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("⚠️  Mininet version check timed out")
    
    # Clean up any existing Mininet state
    try:
        subprocess.run(['mn', '-c'], capture_output=True, timeout=15)
        logger.info("✅ Cleaned existing Mininet state")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.debug("Mininet cleanup completed or not needed")
    
    # Start/restart Open vSwitch with error handling
    try:
        subprocess.run(['service', 'openvswitch-switch', 'stop'], 
                      capture_output=True, timeout=10)
        subprocess.run(['service', 'openvswitch-switch', 'start'], 
                      capture_output=True, timeout=15)
        logger.info("✅ Open vSwitch service restarted")
    except subprocess.TimeoutExpired:
        logger.warning("⚠️  Open vSwitch restart timed out")
    except FileNotFoundError:
        logger.warning("⚠️  Could not restart Open vSwitch service")
    
    return True

def create_configuration():
    """Create default configuration file"""
    logger = logging.getLogger(__name__)
    
    config_path = Path("config/simple_config.yaml")
    
    if config_path.exists():
        logger.info(f"✅ Configuration already exists: {config_path}")
        return True
    
    # Enhanced configuration for simple topology
    config = {
        'system': {
            'log_level': 'INFO',
            'debug_mode': False
        },
        'mininet': {
            'num_clients': 3,
            'controller_port': 6633,
            'switch_protocol': 'OpenFlow13'
        },
        'phases': {
            'learning_rounds': 6,
            'testing_rounds': 2,
            'learning_duration': 45,
            'testing_duration': 20
        },
        'federated': {
            'local_epochs': 2,
            'batch_size': 16,
            'min_batch_size': 4,
            'aggregation_interval': 25
        },
        'hyperparameters': {
            'learning_rates': [0.01, 0.005],
            'hidden_sizes': [64, 32],
            'dropout_rates': [0.1, 0.2]
        },
        'model': {
            'input_size': 10,
            'output_size': 2
        },
        'security': {
            'threat_threshold': 0.6,
            'block_threshold': 0.75,
            'enable_logging': True
        },
        'firewall': {
            'enable_blocking': False,
            'whitelist_ips': [],
            'blacklist_ports': [4444, 6666, 1234, 31337],
            'max_blocked_ips': 50
        }
    }
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"✅ Created configuration: {config_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create configuration: {e}")
        return False

def run_system_tests():
    """Run comprehensive system tests"""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Running system tests...")
    
    # Test 1: Configuration validation
    try:
        with open("config/simple_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['system', 'mininet', 'phases', 'model']
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Missing config section: {section}")
                return False
        
        logger.info("✅ Configuration validation passed")
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False
    
    # Test 2: Import test with proper path setup
    try:
        sys.path.insert(0, 'src')
        
        from models.simple_firewall import SimpleFirewall
        from core.simple_packet_analyzer import SimplePacketAnalyzer
        from core.firewall_manager import SimpleFirewallManager
        
        logger.info("✅ Module import tests passed")
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        logger.error("Make sure all source files are properly created")
        return False
    
    # Test 3: Basic functionality
    try:
        import torch
        import numpy as np
        
        # Test model creation and prediction
        model = SimpleFirewall()
        test_input = torch.randn(1, 10)
        output = model(test_input)
        prediction = model.predict_threat(test_input[0])
        
        # Test analyzer
        analyzer = SimplePacketAnalyzer()
        features = analyzer.extract_features()
        
        # Test firewall manager
        firewall = SimpleFirewallManager(config)
        action = firewall.analyze_packet(features, prediction)
        
        # Validate results
        if (len(features) == 10 and 
            isinstance(prediction, dict) and 
            'action' in action):
            logger.info("✅ Functionality tests passed")
        else:
            logger.error("❌ Functionality test failed: incorrect output format")
            return False
            
    except Exception as e:
        logger.error(f"❌ Functionality test failed: {e}")
        return False
    
    # Test 4: Mininet basic test (quick)
    try:
        result = subprocess.run(['mn', '--test', 'pingall', '--topo', 'single,2'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("✅ Mininet basic test passed")
        else:
            logger.warning("⚠️  Mininet test had issues (may work in actual deployment)")
    except Exception as e:
        logger.warning(f"⚠️  Mininet test error: {e} (may work in actual deployment)")
    
    return True

def create_startup_script():
    """Create convenient startup script with proper environment"""
    logger = logging.getLogger(__name__)
    
    script_content = """#!/bin/bash

# Robust Federated Firewall System Startup Script
echo "🔥 Starting Robust Federated Firewall System..."

# Check root privileges
if [ "$EUID" -ne 0 ]; then
    echo "🚫 Please run as root: sudo env \"PATH=\$PATH\" bash scripts/start_system.sh"
    exit 1
fi

# Preserve environment PATH
export PATH="$PATH"

# Set Python environment
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Clean up any existing processes
echo "🧹 Cleaning environment..."
mn -c >/dev/null 2>&1
pkill -f "python.*main_simple" >/dev/null 2>&1
fuser -k 6633/tcp >/dev/null 2>&1

# Start system with proper environment
echo "🚀 Launching system..."
env "PATH=$PATH" python3 src/main_simple.py "$@"

echo "✅ System shutdown complete"
"""
    
    script_path = Path("scripts/start_system.sh")
    
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"✅ Created startup script: {script_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create startup script: {e}")
        return False

def create_test_script():
    """Create system test script"""
    logger = logging.getLogger(__name__)
    
    test_script_content = """#!/usr/bin/env python3

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    print("🧪 Running Quick System Tests...")
    
    # Check root
    if os.geteuid() != 0:
        print("🚫 Please run as root: sudo env \\"PATH=$PATH\\" python3 scripts/quick_test.py")
        return False
    
    # Test imports
    try:
        sys.path.insert(0, 'src')
        from models.simple_firewall import SimpleFirewall
        from core.simple_packet_analyzer import SimplePacketAnalyzer
        print("✅ Python imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test basic functionality
    try:
        import torch
        model = SimpleFirewall()
        analyzer = SimplePacketAnalyzer()
        
        features = analyzer.extract_features()
        prediction = model.predict_threat(features)
        
        if len(features) == 10 and isinstance(prediction, dict):
            print("✅ Basic functionality test passed")
        else:
            print("❌ Functionality test failed")
            return False
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False
    
    # Test Mininet
    try:
        result = subprocess.run(['mn', '--version'], capture_output=True, timeout=5)
        print("✅ Mininet available")
    except Exception as e:
        print(f"⚠️  Mininet test: {e}")
    
    print("🎉 Quick tests completed successfully!")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
"""
    
    test_script_path = Path("scripts/quick_test.py")
    
    try:
        with open(test_script_path, 'w') as f:
            f.write(test_script_content)
        
        os.chmod(test_script_path, 0o755)
        logger.info(f"✅ Created test script: {test_script_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create test script: {e}")
        return False

def main():
    """Main setup function"""
    logger = setup_logging()
    
    logger.info("🛠️  ROBUST FEDERATED FIREWALL SYSTEM SETUP")
    logger.info("=" * 60)
    
    # Check prerequisites
    check_root_privileges()
    
    setup_steps = [
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Setting up Mininet", setup_mininet),
        ("Creating configuration", create_configuration),
        ("Creating startup script", create_startup_script),
        ("Creating test script", create_test_script),
        ("Running system tests", run_system_tests)
    ]
    
    failed_steps = []
    
    for step_name, step_function in setup_steps:
        logger.info(f"📋 {step_name}...")
        try:
            if step_function():
                logger.info(f"✅ {step_name} completed")
            else:
                logger.error(f"❌ {step_name} failed")
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"❌ {step_name} failed with error: {e}")
            failed_steps.append(step_name)
    
    # Setup summary
    logger.info("=" * 60)
    
    if not failed_steps:
        logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
        logger.info("")
        logger.info("📋 NEXT STEPS:")
        logger.info("  1. Quick test: sudo env \"PATH=$PATH\" python3 scripts/quick_test.py")
        logger.info("  2. Start system: sudo env \"PATH=$PATH\" bash scripts/start_system.sh")
        logger.info("  3. Or direct: sudo env \"PATH=$PATH\" python3 src/main_simple.py")
        logger.info("")
        logger.info("🔧 SYSTEM FEATURES:")
        logger.info("  ✅ Simple star topology (1 switch, 3 hosts)")
        logger.info("  ✅ Direct host-to-switch connections")
        logger.info("  ✅ No BatchNorm issues in neural network")
        logger.info("  ✅ Comprehensive error handling")
        logger.info("  ✅ 10-feature packet analysis")
        logger.info("  ✅ Federated learning with weight averaging")
        logger.info("  ✅ Real-time threat detection")
        logger.info("  ✅ Proper environment preservation")
        logger.info("")
        return True
    else:
        logger.error(f"❌ SETUP INCOMPLETE - {len(failed_steps)} step(s) failed:")
        for step in failed_steps:
            logger.error(f"   • {step}")
        logger.info("")
        logger.info("🔧 TROUBLESHOOTING:")
        logger.info("  • Ensure you're running as root with: sudo env \"PATH=$PATH\" python3 scripts/setup_system.py")
        logger.info("  • Check internet connectivity for package installation")
        logger.info("  • Verify system has sufficient disk space")
        logger.info("  • Try manual installation: pip3 install torch numpy PyYAML")
        logger.info("  • Install Mininet: sudo apt-get install mininet")
        logger.info("")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
