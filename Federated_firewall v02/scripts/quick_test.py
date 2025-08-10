#!/usr/bin/env python3

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    print("🧪 Running Quick System Tests...")
    
    # Check root
    if os.geteuid() != 0:
        print("🚫 Please run as root: sudo env \"PATH=$PATH\" python3 scripts/quick_test.py")
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
