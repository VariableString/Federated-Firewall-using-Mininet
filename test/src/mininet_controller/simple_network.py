import asyncio
import logging
import time
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import Controller
from mininet.cli import CLI
from mininet.log import setLogLevel

from federated.simple_coordinator import SimpleCoordinator
from federated.simple_client import SimpleClient

logger = logging.getLogger(__name__)

class LinearTopology(Topo):
    """Simple linear topology"""
    
    def __init__(self, num_clients=3):
        super(LinearTopology, self).__init__()
        
        # Add coordinator host
        coordinator = self.addHost('coord', ip='10.0.0.1/24')
        
        # Add switch
        switch = self.addSwitch('s1')
        
        # Connect coordinator to switch
        self.addLink(coordinator, switch)
        
        # Add client hosts
        for i in range(num_clients):
            client = self.addHost(f'h{i+1}', ip=f'10.0.0.{i+2}/24')
            self.addLink(client, switch)

class SimpleNetworkManager:
    """Simple network manager for Mininet"""
    
    def __init__(self, config):
        self.config = config
        self.net = None
        self.coordinator = None
        self.clients = []
        self.is_running = False
        
        setLogLevel('info')
        
    async def start_system(self):
        """Start Mininet network and federated system"""
        logger.info("Starting simple Mininet network...")
        
        try:
            # Create topology
            topo = LinearTopology(self.config['mininet']['num_clients'])
            
            # Create network
            self.net = Mininet(
                topo=topo,
                controller=Controller,
                autoSetMacs=True,
                autoStaticArp=True
            )
            
            # Start network
            self.net.start()
            logger.info("Mininet network started")
            
            # Wait for network to stabilize
            await asyncio.sleep(2)
            
            # Start federated learning
            await self._start_federated_learning()
            
            self.is_running = True
            logger.info("Federated system is now running")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop_system()
            raise
    
    async def stop_system(self):
        """Stop the system"""
        logger.info("Stopping system...")
        
        self.is_running = False
        
        try:
            # Stop federated components
            if self.coordinator:
                await self.coordinator.stop()
            
            for client in self.clients:
                await client.stop()
            
            # Stop Mininet
            if self.net:
                self.net.stop()
                
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
        
        # Clean up
        import subprocess
        subprocess.run(['sudo', 'mn', '-c'], capture_output=True)
        logger.info("System stopped")
    
    async def _start_federated_learning(self):
        """Start federated learning components"""
        logger.info("Starting federated learning...")
        
        hosts = self.net.hosts
        
        # First host is coordinator
        coordinator_host = hosts[0]
        client_hosts = hosts[1:]
        
        # Start coordinator
        self.coordinator = SimpleCoordinator(self.config, coordinator_host)
        asyncio.create_task(self.coordinator.start())
        
        # Start clients
        for i, host in enumerate(client_hosts):
            client = SimpleClient(i, self.config, host)
            self.clients.append(client)
            asyncio.create_task(client.start())
        
        logger.info(f"Started 1 coordinator and {len(self.clients)} clients")
