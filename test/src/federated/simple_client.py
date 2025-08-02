import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
from scapy.all import sniff
from models.simple_firewall import SimpleFirewall
from core.threat_detector import SimpleThreatDetector

logger = logging.getLogger(__name__)

class SimpleClient:
    """Simple federated learning client"""
    
    def __init__(self, client_id, config, mininet_host):
        self.client_id = client_id
        self.config = config
        self.mininet_host = mininet_host
        self.host_name = mininet_host.name
        self.host_ip = mininet_host.IP()
        
        # Initialize model
        self.model = SimpleFirewall(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            output_size=config['model']['output_size']
        )
        
        self.threat_detector = SimpleThreatDetector(config, self.host_name)
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=config['federated']['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        
        self.training_data = []
        self.is_running = False
        self.packets_processed = 0
        
    async def start(self):
        """Start client operations"""
        self.is_running = True
        logger.info(f"Starting client {self.host_name} on {self.host_ip}")
        
        try:
            # Generate initial training data
            await self._generate_training_data()
            
            # Start packet monitoring
            monitor_task = asyncio.create_task(self._monitor_packets())
            
            # Start training loop
            training_task = asyncio.create_task(self._training_loop())
            
            # Wait for tasks (they run continuously)
            await asyncio.gather(monitor_task, training_task)
            
        except Exception as e:
            logger.error(f"Client error: {e}")
    
    async def stop(self):
        """Stop client operations"""
        self.is_running = False
        logger.info(f"Stopped client {self.host_name}")
    
    async def _monitor_packets(self):
        """Monitor network packets"""
        logger.info(f"Starting packet monitoring on {self.host_name}")
        
        def packet_handler(packet):
            try:
                if self.is_running:
                    self.packets_processed += 1
                    analysis = self.threat_detector.analyze_packet(packet)
                    
                    # Add to training data
                    if 'features' in analysis:
                        label = 1 if analysis['is_threat'] else 0
                        self.training_data.append({
                            'features': analysis['features'],
                            'label': label
                        })
                        
                        # Keep data size manageable
                        if len(self.training_data) > 1000:
                            self.training_data = self.training_data[-800:]
                            
            except Exception as e:
                logger.error(f"Packet handler error: {e}")
        
        try:
            # Start packet sniffing (simplified - capture on any interface)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sniff(
                    prn=packet_handler,
                    stop_filter=lambda x: not self.is_running,
                    timeout=1,
                    store=False
                )
            )
        except Exception as e:
            logger.error(f"Packet monitoring error: {e}")
    
    async def _training_loop(self):
        """Main training loop"""
        while self.is_running:
            try:
                # Train model if we have enough data
                if len(self.training_data) > 50:
                    await self._train_local_model()
                
                # Log status every 50 packets
                if self.packets_processed % 50 == 0 and self.packets_processed > 0:
                    stats = self.threat_detector.get_statistics()
                    logger.info(f"{self.host_name}: Processed {self.packets_processed} packets, "
                              f"Detected {stats['total_threats']} threats")
                
                await asyncio.sleep(8)  # Training interval
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
    
    async def _generate_training_data(self):
        """Generate simple synthetic training data"""
        logger.info(f"Generating training data for {self.host_name}")
        
        # Generate 200 normal samples
        for _ in range(200):
            features = np.random.normal(0, 1, 10)  # Normal traffic features
            features[0] = np.random.normal(800, 200)  # Packet size
            features[6] = np.random.choice([80, 443, 22, 53])  # Common ports
            
            self.training_data.append({
                'features': features,
                'label': 0  # Normal
            })
        
        # Generate 50 threat samples
        for _ in range(50):
            features = np.random.normal(0, 2, 10)  # Threat traffic (more variance)
            features[0] = np.random.choice([64, 1500])  # Unusual packet sizes
            features[6] = np.random.choice([4444, 6666, 1433])  # Suspicious ports
            features[8] = np.random.choice([1, 4, 16])  # Suspicious TCP flags
            
            self.training_data.append({
                'features': features,
                'label': 1  # Threat
            })
        
        logger.info(f"Generated {len(self.training_data)} training samples")
    
    async def _train_local_model(self):
        """Train local model on collected data"""
        logger.info(f"Training model on {self.host_name}")
        
        # Prepare data
        features = torch.FloatTensor([d['features'] for d in self.training_data])
        labels = torch.LongTensor([d['label'] for d in self.training_data])
        
        # Simple normalization
        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        
        # Training
        self.model.train()
        total_loss = 0.0
        batch_size = self.config['federated']['batch_size']
        
        for epoch in range(self.config['federated']['local_epochs']):
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(features) // batch_size)
        logger.info(f"Training completed on {self.host_name}. Loss: {avg_loss:.4f}")
        
        # Update threat detector model
        self.threat_detector.model = self.model