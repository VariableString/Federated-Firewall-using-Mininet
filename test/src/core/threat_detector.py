import torch
import numpy as np
import logging
from models.simple_firewall import SimpleFirewall
from core.packet_analyzer import SimplePacketAnalyzer

logger = logging.getLogger(__name__)

class SimpleThreatDetector:
    """Simple threat detector using neural network"""
    
    def __init__(self, config, host_id):
        self.config = config
        self.host_id = host_id
        self.model = SimpleFirewall(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            output_size=config['model']['output_size']
        )
        
        self.analyzer = SimplePacketAnalyzer()
        self.threat_threshold = config['security']['threat_threshold']
        self.detected_threats = []
        
    def analyze_packet(self, packet):
        """Analyze packet for threats"""
        try:
            # Extract features
            features = self.analyzer.extract_features(packet)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Predict threat
            prediction, confidence = self.model.predict_threat(features_tensor)
            
            is_threat = prediction == 1 and confidence > self.threat_threshold
            
            if is_threat:
                self.detected_threats.append({
                    'timestamp': time.time(),
                    'host_id': self.host_id,
                    'confidence': confidence,
                    'features': features.tolist()
                })
                logger.warning(f"THREAT detected on {self.host_id} (confidence: {confidence:.3f})")
            
            return {
                'is_threat': is_threat,
                'confidence': confidence,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Packet analysis error: {e}")
            return {'is_threat': False, 'confidence': 0.0}
    
    def get_statistics(self):
        """Get threat detection statistics"""
        return {
            'host_id': self.host_id,
            'total_threats': len(self.detected_threats),
            'recent_threats': self.detected_threats[-5:] if self.detected_threats else []
        }
