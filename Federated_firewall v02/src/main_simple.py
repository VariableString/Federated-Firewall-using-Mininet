import os
import sys
import asyncio
import yaml
import logging
import time
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import with comprehensive error handling
try:
    from mininet_controller.simple_network import RobustNetworkManager
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are installed and the project structure is correct.")
    sys.exit(1)

def check_environment():
    """Check system environment and requirements"""
    
    # Check root privileges
    if os.geteuid() != 0:
        print("🚫 ROOT PRIVILEGES REQUIRED")
        print("This script requires root access for Mininet operations.")
        print(f"Please run: sudo env \"PATH=$PATH\" python3 {' '.join(sys.argv)}")
        return False
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("🚫 Python 3.7+ required")
        print(f"Current version: {sys.version}")
        return False
    
    # Check required modules
    required_modules = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'), 
        ('yaml', 'PyYAML'),
        ('mininet', 'Mininet')
    ]
    
    missing_modules = []
    for module, display_name in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(display_name)
    
    if missing_modules:
        print(f"🚫 Missing required modules: {', '.join(missing_modules)}")
        print("Install with: pip3 install torch numpy PyYAML")
        print("Install Mininet: sudo apt-get install mininet")
        return False
    
    return True

async def main():
    """Enhanced main entry point with comprehensive error handling"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robust Federated Firewall System')
    parser.add_argument('--config', '-c', default='config/simple_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--log-level', '-l', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Skip initial cleanup (for debugging)')
    
    args = parser.parse_args()
    
    # Check environment before proceeding
    if not check_environment():
        sys.exit(1)
    
    # Setup logging with error handling
    try:
        logger = setup_logging(args.log_level, args.debug)
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please ensure the configuration file exists or run the setup script.")
        logger.info("Run: sudo env \"PATH=$PATH\" python3 scripts/setup_system.py")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from: {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Validate configuration
    try:
        required_sections = ['mininet', 'phases', 'federated', 'model']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required configuration section: {section}")
                sys.exit(1)
        logger.info("Configuration validation passed")
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        sys.exit(1)
    
    # Display system information
    logger.info("🔥 ROBUST FEDERATED FIREWALL SYSTEM")
    logger.info("=" * 50)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"Debug Mode: {args.debug}")
    logger.info(f"Clients: {config['mininet']['num_clients']}")
    logger.info(f"Learning Rounds: {config['phases']['learning_rounds']}")
    logger.info(f"Testing Rounds: {config['phases']['testing_rounds']}")
    logger.info(f"Simple Topology: 1 switch, {config['mininet']['num_clients']} hosts")
    logger.info("=" * 50)
    
    # Initialize network manager
    try:
        network_manager = RobustNetworkManager(config)
    except Exception as e:
        logger.error(f"Failed to initialize network manager: {e}")
        sys.exit(1)
    
    # Record start time
    start_time = time.time()
    network_manager.start_time = start_time
    
    try:
        # Start the system
        logger.info("🚀 LAUNCHING SYSTEM...")
        await network_manager.start_system()
        
        # System running loop
        logger.info("✅ SYSTEM OPERATIONAL")
        logger.info("Press Ctrl+C to stop the system gracefully")
        
        # Keep the system running with periodic status updates
        status_interval = 30
        next_status_time = time.time() + status_interval
        
        while network_manager.is_running:
            current_time = time.time()
            
            # Status update
            if current_time >= next_status_time:
                try:
                    elapsed_time = current_time - start_time
                    system_status = network_manager.get_system_status()
                    
                    logger.info(f"⏱️  SYSTEM STATUS - Runtime: {elapsed_time:.0f}s, "
                              f"Phase: {system_status['current_phase']}, "
                              f"Clients: {system_status['num_clients']} active")
                    
                    next_status_time = current_time + status_interval
                except Exception as e:
                    logger.error(f"Status update error: {e}")
                    next_status_time = current_time + status_interval
            
            await asyncio.sleep(5)
        
    except KeyboardInterrupt:
        logger.info("🛑 SHUTDOWN REQUESTED BY USER")
    except Exception as e:
        logger.error(f"💥 SYSTEM ERROR: {e}")
        if args.debug:
            import traceback
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
    finally:
        # Graceful shutdown
        try:
            total_runtime = time.time() - start_time
            logger.info(f"⏹️  INITIATING SHUTDOWN (Total Runtime: {total_runtime:.1f}s)")
            
            await network_manager.stop_system()
            
            logger.info("✅ SHUTDOWN COMPLETE")
            logger.info("Thank you for using the Robust Federated Firewall System!")
            
        except Exception as shutdown_error:
            logger.error(f"Shutdown error: {shutdown_error}")
            sys.exit(1)

if __name__ == "__main__":
    # Environment check
    if not check_environment():
        sys.exit(1)
    
    # Run the main application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)