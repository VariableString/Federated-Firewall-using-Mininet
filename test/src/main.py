import os
import sys
import asyncio
import yaml
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from mininet_controller.simple_network import SimpleNetworkManager
from utils.logging_utils import setup_logging

async def main():
    """Main entry point"""
    # Check root privileges
    if os.geteuid() != 0:
        print("This script must be run as root for Mininet operations!")
        print("Please run: sudo python3 src/main.py")
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting Simplified Federated Firewall System")
    
    # Initialize network manager
    network_manager = SimpleNetworkManager(config)
    
    try:
        # Start the system
        await network_manager.start_system()
        
        # Keep running
        logger.info("System running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(5)
            logger.info("System status: Running normally")
            
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await network_manager.stop_system()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())