import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleFirewall(nn.Module):
    """Simple neural network for threat detection"""
    
    def __init__(self, input_size=10, hidden_size=32, output_size=2):
        super(SimpleFirewall, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def predict_threat(self, features):
        """Predict if traffic is malicious"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features)
            probabilities = F.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities).item()
        return prediction, confidence
    
    def get_weights(self):
        """Get model weights for federated learning"""
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_weights(self, weights):
        """Set model weights from coordinator"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    param.data.copy_(weights[name])