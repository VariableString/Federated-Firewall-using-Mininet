import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SimpleFirewall(nn.Module):
    """Robust neural network for threat detection without BatchNorm issues"""
    
    def __init__(self, input_size=10, hidden_size=64, output_size=2, dropout_rate=0.1):
        super(SimpleFirewall, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Simple feedforward architecture - NO BATCH NORMALIZATION
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        logger.debug("Model weights initialized with Xavier uniform")
    
    def forward(self, x):
        """Forward pass through the network"""
        try:
            # Ensure proper input shape and type
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
                
            # Handle single sample
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            # Validate input size
            if x.size(-1) != self.input_size:
                logger.error(f"Input size mismatch: expected {self.input_size}, got {x.size(-1)}")
                # Return safe default
                batch_size = x.size(0) if len(x.shape) > 1 else 1
                return torch.zeros((batch_size, self.output_size))
            
            # Forward pass
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            
            return x
            
        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            batch_size = x.size(0) if isinstance(x, torch.Tensor) and len(x.shape) > 1 else 1
            return torch.zeros((batch_size, self.output_size))
    
    def predict_threat(self, features):
        """Predict threat with confidence scores and error handling"""
        self.eval()
        
        try:
            with torch.no_grad():
                # Input validation and conversion
                if isinstance(features, np.ndarray):
                    features = torch.FloatTensor(features)
                elif not isinstance(features, torch.Tensor):
                    features = torch.FloatTensor(list(features))
                
                # Ensure proper shape
                if len(features.shape) == 1:
                    features = features.unsqueeze(0)
                
                # Check for invalid values
                if torch.isnan(features).any() or torch.isinf(features).any():
                    logger.warning("Invalid values in input features, replacing with defaults")
                    features = torch.nan_to_num(features, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Forward pass
                logits = self.forward(features)
                probabilities = F.softmax(logits, dim=-1)
                
                # Extract results safely
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
                threat_probability = probabilities[0][1].item() if probabilities.size(0) > 0 else 0.5
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'threat_probability': threat_probability,
                    'is_threat': prediction == 1
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return safe default prediction
            return {
                'prediction': 0,
                'confidence': 0.5,
                'threat_probability': 0.5,
                'is_threat': False,
                'error': str(e)
            }
    
    def get_weights(self):
        """Get model parameters for federated learning"""
        try:
            return {name: param.clone().detach() for name, param in self.named_parameters()}
        except Exception as e:
            logger.error(f"Error getting weights: {e}")
            return {}
    
    def set_weights(self, weights):
        """Set model parameters from federated averaging"""
        try:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in weights and weights[name].shape == param.shape:
                        param.copy_(weights[name])
                    else:
                        logger.warning(f"Skipping weight {name} due to shape mismatch or missing data")
        except Exception as e:
            logger.error(f"Error setting weights: {e}")
    
    def get_model_info(self):
        """Get model architecture information"""
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'dropout_rate': self.dropout_rate
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'error': str(e)}