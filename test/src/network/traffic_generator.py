import asyncio
import random
import logging
from mininet.net import Mininet

logger = logging.getLogger(__name__)

class SimpleTrafficGenerator:
    """Simple traffic generator for Mininet"""
    
    def __init__(self, net: Mininet):
        self.net = net
        self.is_generating = False
    
    async def start(self):
        """Start simple traffic generation"""
        self.is_generating = True
        logger.info("Starting simple traffic generation")
        
        while self.is_generating:
            try:
                hosts = self.net.hosts
                if len(hosts) >= 2:
                    # Pick random source and destination
                    src = random.choice(hosts)
                    dst = random.choice([h for h in hosts if h != src])
                    
                    # Generate simple ping traffic
                    src.cmd(f"ping -c 1 {dst.IP()} &")
                
                await asyncio.sleep(random.uniform(1, 5))
                
            except Exception as e:
                logger.error(f"Traffic generation error: {e}")
                await asyncio.sleep(2)
    
    async def stop(self):
        """Stop traffic generation"""
        self.is_generating = False
        logger.info("Stopped traffic generation")
