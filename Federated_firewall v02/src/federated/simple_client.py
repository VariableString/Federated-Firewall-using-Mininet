import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import json
from pathlib import Path

# Import with error handling
try:
    from models.simple_firewall import SimpleFirewall
    from core.simple_packet_analyzer import SimplePacketAnalyzer
    from core.firewall_manager import SimpleFirewallManager
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Import error in federated client: {e}")
    raise

logger = logging.getLogger(__name__)

class SimpleFederatedClient:
    """Enhanced federated learning client with comprehensive error handling"""
    
    def __init__(self, client_id, config, mininet_host=None):
        try:
            self.client_id = client_id
            self.config = config
            self.mininet_host = mininet_host
            
            # Host information with error handling
            if mininet_host:
                try:
                    self.host_name = mininet_host.name
                    self.host_ip = mininet_host.IP()
                except Exception as e:
                    logger.error(f"Error getting host info: {e}")
                    self.host_name = f"client_{client_id}"
                    self.host_ip = f"10.0.0.{client_id + 1}"
            else:
                self.host_name = f"client_{client_id}"
                self.host_ip = f"10.0.0.{client_id + 1}"
            
            # Initialize components with error handling
            try:
                self.model = SimpleFirewall(
                    input_size=config['model']['input_size'],
                    hidden_size=config['hyperparameters']['hidden_sizes'][0],
                    output_size=config['model']['output_size'],
                    dropout_rate=config['hyperparameters']['dropout_rates'][0]
                )
            except Exception as e:
                logger.error(f"Model initialization error: {e}")
                # Use safe defaults
                self.model = SimpleFirewall(input_size=10, hidden_size=64, output_size=2, dropout_rate=0.1)
            
            self.packet_analyzer = SimplePacketAnalyzer()
            self.firewall_manager = SimpleFirewallManager(config)
            
            # Learning configuration with error handling
            try:
                learning_rate = config['hyperparameters']['learning_rates'][0]
            except (KeyError, IndexError):
                learning_rate = 0.01
                logger.warning("Using default learning rate")
                
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=1e-5
            )
            self.criterion = nn.CrossEntropyLoss()
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=False
            )
            
            # Data management
            self.training_data = []
            self.validation_data = []
            self.max_training_samples = 2000
            self.validation_split = 0.2
            
            # State management
            self.is_running = False
            self.current_phase = "learning"
            self.training_round = 0
            
            # Performance tracking
            self.performance_metrics = {
                'packets_processed': 0,
                'threats_detected': 0,
                'model_accuracy': 0.0,
                'training_loss': 0.0,
                'last_update_time': time.time()
            }
            
            # Task management
            self.running_tasks = []
            
            logger.info(f"Client {self.host_name} initialized at {self.host_ip}")
            
        except Exception as e:
            logger.error(f"Client initialization error: {e}")
            raise
    
    async def start(self):
        """Start all client operations with comprehensive error handling"""
        if self.is_running:
            logger.warning(f"Client {self.host_name} already running")
            return
        
        try:
            self.is_running = True
            logger.info(f"Starting federated client {self.host_name}")
            
            # Generate initial training data
            await self._generate_initial_training_data()
            
            # Start all background tasks
            tasks = [
                self._packet_processing_loop(),
                self._training_loop(),
                self._status_reporting_loop(),
                self._data_cleanup_loop()
            ]
            
            self.running_tasks = [asyncio.create_task(task) for task in tasks]
            
            # Wait for tasks to complete
            await asyncio.gather(*self.running_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Client {self.host_name} startup error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop all client operations gracefully"""
        if not self.is_running:
            return
        
        try:
            logger.info(f"Stopping client {self.host_name}")
            self.is_running = False
            
            # Cancel all running tasks
            for task in self.running_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to finish with timeout
            if self.running_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.running_tasks, return_exceptions=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Client {self.host_name} tasks didn't stop within timeout")
            
            self.running_tasks.clear()
            logger.info(f"Client {self.host_name} stopped")
            
        except Exception as e:
            logger.error(f"Client stop error: {e}")
    
    async def _packet_processing_loop(self):
        """Continuously process packets and generate training data"""
        packet_interval = 0.5  # Process packet every 500ms
        
        while self.is_running:
            try:
                # Extract packet features
                features = self.packet_analyzer.extract_features()
                
                # Get model prediction
                prediction_info = self.model.predict_threat(features)
                
                # Analyze with firewall
                firewall_action = self.firewall_manager.analyze_packet(features, prediction_info)
                
                # Generate training label (synthetic)
                true_label = self._generate_synthetic_label(features, prediction_info)
                
                # Add to training data
                training_sample = {
                    'features': features.copy(),
                    'label': true_label,
                    'prediction': prediction_info['prediction'],
                    'confidence': prediction_info['confidence'],
                    'timestamp': time.time(),
                    'firewall_action': firewall_action['action']
                }
                
                self.training_data.append(training_sample)
                
                # Update performance metrics
                self.performance_metrics['packets_processed'] += 1
                if true_label == 1:
                    self.performance_metrics['threats_detected'] += 1
                
                # Log significant threats
                if prediction_info.get('is_threat', False) and prediction_info.get('confidence', 0) > 0.7:
                    logger.debug(f"{self.host_name}: HIGH THREAT - "
                               f"Confidence: {prediction_info['confidence']:.3f}, "
                               f"Action: {firewall_action['action']}")
                
                await asyncio.sleep(packet_interval)
                
            except Exception as e:
                logger.error(f"Packet processing error in {self.host_name}: {e}")
                await asyncio.sleep(1)
    
    async def _training_loop(self):
        """Training loop with improved error handling"""
        training_interval = 15  # Train every 15 seconds
        
        while self.is_running:
            try:
                min_batch_size = self.config.get('federated', {}).get('min_batch_size', 4)
                if len(self.training_data) >= min_batch_size:
                    await self._train_model()
                    self.training_round += 1
                    
                    if self.training_round % 4 == 0:
                        logger.info(f"{self.host_name}: Completed {self.training_round} training rounds")
                
                await asyncio.sleep(training_interval)
                
            except Exception as e:
                logger.error(f"Training loop error in {self.host_name}: {e}")
                await asyncio.sleep(5)
    
    async def _train_model(self):
        """Enhanced model training with validation and error handling"""
        try:
            min_batch_size = self.config.get('federated', {}).get('min_batch_size', 4)
            if len(self.training_data) < min_batch_size:
                return
            
            # Prepare balanced dataset
            threat_samples = [d for d in self.training_data if d['label'] == 1]
            normal_samples = [d for d in self.training_data if d['label'] == 0]
            
            # Balance the classes
            min_samples = min(len(threat_samples), len(normal_samples))
            max_samples_per_class = min(min_samples, 300)  # Limit dataset size
            
            if max_samples_per_class < min_batch_size // 2:
                # Use all available data if we don't have enough
                selected_samples = self.training_data[-200:] if len(self.training_data) > 200 else self.training_data
            else:
                # Randomly sample balanced data
                if len(threat_samples) >= max_samples_per_class:
                    selected_threat = np.random.choice(threat_samples, max_samples_per_class, replace=False)
                else:
                    selected_threat = threat_samples
                    
                if len(normal_samples) >= max_samples_per_class:
                    selected_normal = np.random.choice(normal_samples, max_samples_per_class, replace=False)
                else:
                    selected_normal = normal_samples
                    
                selected_samples = list(selected_threat) + list(selected_normal)
            
            if len(selected_samples) < min_batch_size:
                return
            
            # Prepare tensors with error handling
            try:
                features_list = [s['features'] for s in selected_samples]
                labels_list = [s['label'] for s in selected_samples]
                
                features = torch.FloatTensor(features_list)
                labels = torch.LongTensor(labels_list)
                
                # Validate tensor shapes
                if features.size(1) != self.model.input_size:
                    logger.error(f"Feature size mismatch: {features.size(1)} != {self.model.input_size}")
                    return
                
            except Exception as e:
                logger.error(f"Tensor preparation error: {e}")
                return
            
            # Feature normalization with error handling
            try:
                if features.std() > 1e-6:  # Avoid division by zero
                    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
            except Exception as e:
                logger.warning(f"Feature normalization error: {e}")
            
            # Split into training and validation
            dataset_size = len(features)
            val_size = max(1, int(dataset_size * self.validation_split))
            train_size = dataset_size - val_size
            
            # Shuffle data
            indices = torch.randperm(dataset_size)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            train_features = features[train_indices]
            train_labels = labels[train_indices]
            val_features = features[val_indices]
            val_labels = labels[val_indices]
            
            # Training with error handling
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            batch_size = min(self.config.get('federated', {}).get('batch_size', 32), train_size // 2)
            batch_size = max(batch_size, min_batch_size)
            
            local_epochs = self.config.get('federated', {}).get('local_epochs', 2)
            
            for epoch in range(local_epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                
                # Create batches
                for i in range(0, len(train_features), batch_size):
                    try:
                        batch_features = train_features[i:i+batch_size]
                        batch_labels = train_labels[i:i+batch_size]
                        
                        if len(batch_features) < 2:  # Skip very small batches
                            continue
                        
                        # Forward pass
                        self.optimizer.zero_grad()
                        outputs = self.model(batch_features)
                        
                        # Check for valid outputs
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            logger.warning("Invalid model outputs detected, skipping batch")
                            continue
                        
                        loss = self.criterion(outputs, batch_labels)
                        
                        # Check for valid loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning("Invalid loss detected, skipping batch")
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        self.optimizer.step()
                        
                        epoch_loss += loss.item()
                        epoch_batches += 1
                        
                    except Exception as e:
                        logger.error(f"Batch training error: {e}")
                        continue
                
                if epoch_batches > 0:
                    total_loss += epoch_loss / epoch_batches
                    num_batches += 1
            
            # Validation with error handling
            if len(val_features) > 0:
                try:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(val_features)
                        
                        if not (torch.isnan(val_outputs).any() or torch.isinf(val_outputs).any()):
                            val_predictions = torch.argmax(val_outputs, dim=1)
                            val_accuracy = (val_predictions == val_labels).float().mean().item()
                            val_loss = self.criterion(val_outputs, val_labels).item()
                            
                            # Update metrics
                            self.performance_metrics['model_accuracy'] = val_accuracy
                            self.performance_metrics['training_loss'] = total_loss / max(num_batches, 1)
                            self.performance_metrics['last_update_time'] = time.time()
                            
                            # Learning rate scheduling
                            self.scheduler.step(val_loss)
                            
                            logger.debug(f"{self.host_name}: Training - Loss: {total_loss/max(num_batches, 1):.4f}, "
                                       f"Val Accuracy: {val_accuracy:.3f}, Samples: {len(selected_samples)}")
                        else:
                            logger.warning("Invalid validation outputs, skipping metrics update")
                            
                except Exception as e:
                    logger.error(f"Validation error: {e}")
            
        except Exception as e:
            logger.error(f"Model training error in {self.host_name}: {e}")
    
    async def _status_reporting_loop(self):
        """Regular status reporting"""
        report_interval = 45  # Report every 45 seconds
        
        while self.is_running:
            try:
                await asyncio.sleep(report_interval)
                
                # Get comprehensive status
                analyzer_stats = self.packet_analyzer.get_analyzer_stats()
                firewall_stats = self.firewall_manager.get_firewall_stats()
                model_info = self.model.get_model_info()
                
                status_report = {
                    'host': self.host_name,
                    'phase': self.current_phase,
                    'training_round': self.training_round,
                    'performance': self.performance_metrics.copy(),
                    'analyzer': analyzer_stats,
                    'firewall': firewall_stats,
                    'model': {k: v for k, v in model_info.items() if k != 'total_parameters'},
                    'data': {
                        'training_samples': len(self.training_data),
                        'validation_samples': len(self.validation_data)
                    }
                }
                
                logger.info(f"{self.host_name} Status Report:")
                logger.info(f"  Phase: {status_report['phase']}, Round: {status_report['training_round']}")
                logger.info(f"  Packets: {status_report['performance']['packets_processed']}, "
                          f"Threats: {status_report['performance']['threats_detected']}")
                logger.info(f"  Accuracy: {status_report['performance']['model_accuracy']:.3f}, "
                          f"Training Samples: {status_report['data']['training_samples']}")
                logger.info(f"  Firewall: {status_report['firewall']['threats_detected']} threats, "
                          f"{status_report['firewall']['packets_blocked']} blocked")
                
                # Save status to file for monitoring
                await self._save_status_report(status_report)
                
            except Exception as e:
                logger.error(f"Status reporting error in {self.host_name}: {e}")
    
    async def _data_cleanup_loop(self):
        """Periodic data cleanup to manage memory"""
        cleanup_interval = 120  # Cleanup every 2 minutes
        
        while self.is_running:
            try:
                await asyncio.sleep(cleanup_interval)
                
                # Limit training data size
                if len(self.training_data) > self.max_training_samples:
                    # Keep recent samples and some older samples for diversity
                    recent_samples = self.training_data[-int(self.max_training_samples * 0.8):]
                    older_samples = self.training_data[:int(self.max_training_samples * 0.2)]
                    self.training_data = older_samples + recent_samples
                    
                    logger.debug(f"{self.host_name}: Cleaned training data to {len(self.training_data)} samples")
                
                # Reset analyzer stats periodically to prevent overflow
                if self.performance_metrics['packets_processed'] > 100000:
                    self.packet_analyzer.reset_stats()
                    self.firewall_manager.reset_stats()
                    
                    # Reset performance metrics but keep accuracy
                    current_accuracy = self.performance_metrics['model_accuracy']
                    self.performance_metrics = {
                        'packets_processed': 0,
                        'threats_detected': 0,
                        'model_accuracy': current_accuracy,
                        'training_loss': 0.0,
                        'last_update_time': time.time()
                    }
                    
                    logger.info(f"{self.host_name}: Reset statistics due to high packet count")
                
            except Exception as e:
                logger.error(f"Data cleanup error in {self.host_name}: {e}")
    
    def _generate_synthetic_label(self, features, prediction_info):
        """Generate synthetic training labels with some noise"""
        try:
            # Base threat score from features
            threat_score = 0.0
            
            # Feature-based indicators with error handling
            try:
                if len(features) >= 10:
                    if features[0] > 0.8:  # Large packets
                        threat_score += 0.25
                    if features[8] > 0.7:  # High suspicious score
                        threat_score += 0.4
                    if features[9] > 0.8:  # High entropy
                        threat_score += 0.25
                    if features[5] < 0.2:  # Unusual protocol
                        threat_score += 0.15
                    if features[6] < 0.3 or features[6] > 0.9:  # Unusual TTL
                        threat_score += 0.1
            except (IndexError, TypeError) as e:
                logger.debug(f"Feature indexing error in label generation: {e}")
            
            # Add some randomness for model diversity
            threat_score += np.random.uniform(-0.15, 0.15)
            
            # Convert to binary label with some noise
            base_threshold = 0.5
            noise_factor = np.random.uniform(-0.1, 0.1)
            final_threshold = base_threshold + noise_factor
            
            return 1 if threat_score > final_threshold else 0
            
        except Exception as e:
            logger.error(f"Label generation error: {e}")
            return np.random.choice([0, 1], p=[0.7, 0.3])  # 30% threats by default
    
    async def _generate_initial_training_data(self):
        """Generate diverse initial training data"""
        logger.info(f"Generating initial training data for {self.host_name}")
        
        initial_samples = 150
        successful_samples = 0
        
        for i in range(initial_samples):
            try:
                # Generate diverse packet features
                features = self.packet_analyzer.extract_features()
                prediction = self.model.predict_threat(features)
                label = self._generate_synthetic_label(features, prediction)
                
                sample = {
                    'features': features.copy(),
                    'label': label,
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'timestamp': time.time(),
                    'firewall_action': 'allow'
                }
                
                self.training_data.append(sample)
                successful_samples += 1
                
                # Small delay to simulate realistic packet timing
                if i % 20 == 0:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Initial data generation error: {e}")
                continue
        
        threat_count = sum(1 for s in self.training_data if s['label'] == 1)
        logger.info(f"{self.host_name}: Generated {successful_samples} initial samples "
                   f"({threat_count} threats, {successful_samples-threat_count} normal)")
    
    async def _save_status_report(self, report):
        """Save status report to file"""
        try:
            reports_dir = Path("logs/reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"{self.host_name}_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        except Exception as e:
            logger.debug(f"Could not save status report: {e}")
    
    def update_phase(self, new_phase):
        """Update the current operational phase"""
        try:
            old_phase = self.current_phase
            self.current_phase = new_phase
            logger.info(f"{self.host_name}: Phase transition {old_phase} -> {new_phase}")
        except Exception as e:
            logger.error(f"Phase update error: {e}")
    
    def get_model_weights(self):
        """Get model weights for federated averaging"""
        try:
            return self.model.get_weights()
        except Exception as e:
            logger.error(f"Error getting model weights from {self.host_name}: {e}")
            return {}
    
    def set_model_weights(self, weights):
        """Set model weights from federated averaging"""
        try:
            self.model.set_weights(weights)
            logger.debug(f"{self.host_name}: Updated model weights from federated averaging")
        except Exception as e:
            logger.error(f"Error setting model weights for {self.host_name}: {e}")
    
    def get_training_stats(self):
        """Get comprehensive training statistics"""
        try:
            return {
                'host': self.host_name,
                'host_ip': self.host_ip,
                'client_id': self.client_id,
                'current_phase': self.current_phase,
                'training_round': self.training_round,
                'performance_metrics': self.performance_metrics.copy(),
                'data_stats': {
                    'training_samples': len(self.training_data),
                    'validation_samples': len(self.validation_data),
                    'threat_ratio': self.performance_metrics['threats_detected'] / max(self.performance_metrics['packets_processed'], 1)
                },
                'firewall_stats': self.firewall_manager.get_firewall_stats(),
                'analyzer_stats': self.packet_analyzer.get_analyzer_stats()
            }
        except Exception as e:
            logger.error(f"Error getting training stats: {e}")
            return {
                'host': self.host_name,
                'error': str(e)
            }