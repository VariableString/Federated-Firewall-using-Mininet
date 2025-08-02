import unittest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, patch
from src.models.simple_firewall import SimpleFirewall
from src.core.packet_analyzer import SimplePacketAnalyzer
from src.federated.simple_coordinator import SimpleCoordinator

class TestSimpleFirewall(unittest.TestCase):
    
    def setUp(self):
        self.model = SimpleFirewall(input_size=10, hidden_size=32, output_size=2)
    
    def test_model_forward(self):
        """Test model forward pass"""
        batch_size = 5
        input_tensor = torch.randn(batch_size, 10)
        
        output = self.model(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_threat_prediction(self):
        """Test threat prediction"""
        features = torch.randn(1, 10)
        
        prediction, confidence = self.model.predict_threat(features)
        
        self.assertIn(prediction, [0, 1])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_weight_operations(self):
        """Test weight get/set operations"""
        original_weights = self.model.get_weights()
        
        # Modify weights
        new_weights = {}
        for key, value in original_weights.items():
            new_weights[key] = value + 0.1
        
        self.model.set_weights(new_weights)
        updated_weights = self.model.get_weights()
        
        # Check weights were updated
        for key in original_weights.keys():
            self.assertFalse(torch.allclose(original_weights[key], updated_weights[key]))

class TestSimplePacketAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = SimplePacketAnalyzer()
    
    def test_feature_extraction(self):
        """Test basic feature extraction"""
        # Mock packet
        class MockPacket:
            def __len__(self):
                return 1000
        
        packet = MockPacket()
        features = self.analyzer.extract_features(packet)
        
        self.assertEqual(len(features), 10)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features[0], 1000)  # Packet size

class TestSimpleCoordinator(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'model': {'input_size': 10, 'hidden_size': 32, 'output_size': 2},
            'federated': {'rounds': 3},
            'mininet': {'num_clients': 2}
        }
        self.mock_host = Mock()
        self.mock_host.name = 'coordinator'
        
    def test_coordinator_initialization(self):
        """Test coordinator initialization"""
        coordinator = SimpleCoordinator(self.config, self.mock_host)
        
        self.assertEqual(coordinator.host_name, 'coordinator')
        self.assertEqual(coordinator.max_rounds, 3)
        self.assertFalse(coordinator.is_running)

if __name__ == '__main__':
    unittest.main()