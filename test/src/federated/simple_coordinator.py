import asyncio
import torch
import logging
import time
from models.simple_firewall import SimpleFirewall
from models.fed_utils import federated_average

logger = logging.getLogger(__name__)

class SimpleCoordinator:
    """Simple federated learning coordinator"""
    
    def __init__(self, config, mininet_host):
        self.config = config
        self.mininet_host = mininet_host
        self.host_name = mininet_host.name
        
        self.global_model = SimpleFirewall(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            output_size=config['model']['output_size']
        )
        
        self.current_round = 0
        self.max_rounds = config['federated']['rounds']
        self.is_running = False
        
    async def start(self):
        """Start coordination"""
        self.is_running = True
        logger.info(f"Starting coordinator on {self.host_name}")
        
        # Wait for clients to initialize
        await asyncio.sleep(3)
        
        try:
            while self.is_running and self.current_round < self.max_rounds:
                await self._run_round()
                self.current_round += 1
                
                # Wait between rounds
                await asyncio.sleep(10)
                
            logger.info("Federated learning completed")
            
        except Exception as e:
            logger.error(f"Coordinator error: {e}")
    
    async def stop(self):
        """Stop coordination"""
        self.is_running = False
        logger.info(f"Stopped coordinator on {self.host_name}")
    
    async def _run_round(self):
        """Execute one federated round"""
        logger.info(f"Starting round {self.current_round + 1}/{self.max_rounds}")
        
        try:
            # Simulate collecting client updates
            client_weights = []
            client_sizes = []
            
            num_clients = self.config['mininet']['num_clients']
            
            for client_id in range(num_clients):
                # Simulate client update
                weights = self._simulate_client_update(client_id)
                size = 100 + client_id * 20  # Simulated data size
                
                client_weights.append(weights)
                client_sizes.append(size)
            
            if client_weights:
                # Perform federated averaging
                averaged_weights = federated_average(client_weights, client_sizes)
                
                # Update global model
                self.global_model.set_weights(averaged_weights)
                
                logger.info(f"Round {self.current_round + 1} completed. "
                           f"Averaged {len(client_weights)} client updates.")
            
        except Exception as e:
            logger.error(f"Round error: {e}")
    
    def _simulate_client_update(self, client_id):
        """Simulate getting client model update"""
        # Get current weights and add some noise to simulate training
        weights = self.global_model.get_weights()
        
        # Add client-specific noise
        noise_scale = 0.01 * (1 + client_id * 0.1)
        for key in weights:
            noise = torch.randn_like(weights[key]) * noise_scale
            weights[key] += noise
        
        return weights