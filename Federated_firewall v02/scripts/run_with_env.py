#!/usr/bin/env python3
"""
Script to run the federated firewall system with proper environment preservation
Usage: sudo env "PATH=$PATH" python3 scripts/run_with_env.py [options]
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)

def check_environment():
    """Check if environment is properly preserved"""
    logger = logging.getLogger(__name__)
    
    # Check root privileges
    if os.geteuid() != 0:
        logger.error("🚫 ROOT PRIVILEGES REQUIRED")
        logger.error("Please run: sudo env \"PATH=$PATH\" python3 scripts/run_with_env.py")
        return False
    
    # Check PATH preservation
    current_path = os.environ.get('PATH', '')
    if '/usr/local/bin' not in current_path:
        logger.warning("⚠️  PATH may not be properly preserved")
        logger.info("Current PATH:", current_path[:100] + "...")
    
    # Check Python modules
    try:
        import torch
        import numpy
        import yaml
        logger.info("✅ All required Python modules available")
    except ImportError as e:
        logger.error(f"❌ Missing Python module: {e}")
        return False
    
    return True

def main():
    """Main function to run the system with proper environment"""
    logger = setup_logging()
    
    logger.info("🔥 FEDERATED FIREWALL - ENVIRONMENT RUNNER")
    logger.info("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Set up Python path
    project_root = Path(__file__).parent.parent
    src_path = str(project_root / "src")
    
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if src_path not in current_pythonpath:
        os.environ['PYTHONPATH'] = f"{src_path}:{current_pythonpath}"
        logger.info(f"✅ Added to PYTHONPATH: {src_path}")
    
    # Clean up environment first
    logger.info("🧹 Cleaning environment...")
    cleanup_commands = [
        ['mn', '-c'],
        ['pkill', '-f', 'python.*main_simple'],
        ['fuser', '-k', '6633/tcp']
    ]
    
    for cmd in cleanup_commands:
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
        except Exception:
            pass  # Ignore cleanup errors
    
    # Run the main system
    main_script = project_root / "src" / "main_simple.py"
    
    if not main_script.exists():
        logger.error(f"❌ Main script not found: {main_script}")
        logger.info("Please run setup first: sudo env \"PATH=$PATH\" python3 scripts/setup_system.py")
        sys.exit(1)
    
    try:
        logger.info("🚀 Starting federated firewall system...")
        logger.info(f"Script: {main_script}")
        logger.info(f"Args: {sys.argv[1:]}")
        
        # Run with preserved environment
        cmd = [sys.executable, str(main_script)] + sys.argv[1:]
        
        # Execute the main system
        result = subprocess.run(cmd, env=os.environ.copy())
        
        logger.info(f"✅ System exited with code: {result.returncode}")
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        logger.info("🛑 System interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Execution error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()