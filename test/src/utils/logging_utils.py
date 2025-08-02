import logging
import logging.config
from pathlib import Path

def setup_logging():
    """Setup simple logging configuration"""
    
    # Create log directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/system.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set Mininet logging level
    mininet_logger = logging.getLogger('mininet')
    mininet_logger.setLevel(logging.WARNING)