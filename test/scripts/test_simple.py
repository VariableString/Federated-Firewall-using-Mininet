#!/usr/bin/env python3

import subprocess
import logging
import time
import os

def setup_logging():
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')

def test_mininet():
    """Test basic Mininet functionality"""
    logging.info("Testing Mininet...")
    
    try:
        result = subprocess.run(['mn', '--test', 'pingall', '--topo', 'single,2'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logging.info("✓ Mininet test passed")
            return True
        else:
            logging.error("✗ Mininet test failed")
            return False
            
    except Exception as e:
        logging.error(f"✗ Mininet test error: {e}")
        return False

def test_python_deps():
    """Test Python dependencies"""
    logging.info("Testing Python dependencies...")
    
    deps = ['torch', 'numpy', 'scapy', 'yaml', 'mininet']
    
    for dep in deps:
        try:
            __import__(dep)
            logging.info(f"✓ {dep} found")
        except ImportError:
            logging.error(f"✗ {dep} not found")
            return False
    
    return True

def main():
    setup_logging()
    
    if os.geteuid() != 0:
        logging.error("Please run as root: sudo python3 scripts/test_simple.py")
        return 1
    
    logging.info("Running simple system tests...")
    
    tests = [
        ("Python Dependencies", test_python_deps),
        ("Mininet Basic", test_mininet)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        logging.info(f"Running {test_name} test...")
        if test_func():
            passed += 1
    
    logging.info(f"Tests completed: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        logging.info("🎉 All tests passed! System is ready.")
        return 0
    else:
        logging.error("❌ Some tests failed. Please check your setup.")
        return 1

if __name__ == "__main__":
    exit(main())