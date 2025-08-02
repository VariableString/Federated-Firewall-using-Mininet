import numpy as np
import time
import logging
from scapy.all import *
from collections import defaultdict

logger = logging.getLogger(__name__)

class SimplePacketAnalyzer:
    """Simple packet analyzer for basic feature extraction"""
    
    def __init__(self):
        self.packet_count = 0
        self.threat_count = 0
        
    def extract_features(self, packet):
        """Extract 10 simple features from packet"""
        features = np.zeros(10)
        
        try:
            # Basic packet features
            features[0] = len(packet)  # Packet size
            features[1] = time.time() % 3600  # Time of day (hour)
            
            if IP in packet:
                ip_layer = packet[IP]
                features[2] = int(ip_layer.src.split('.')[-1])  # Last octet of src IP
                features[3] = int(ip_layer.dst.split('.')[-1])  # Last octet of dst IP
                features[4] = ip_layer.proto  # Protocol number
                features[5] = ip_layer.ttl    # TTL
            
            if TCP in packet:
                tcp_layer = packet[TCP]
                features[6] = tcp_layer.sport % 1000  # Source port (mod 1000)
                features[7] = tcp_layer.dport % 1000  # Dest port (mod 1000)
                features[8] = tcp_layer.flags  # TCP flags
                features[9] = len(tcp_layer.payload) if tcp_layer.payload else 0
            elif UDP in packet:
                udp_layer = packet[UDP]
                features[6] = udp_layer.sport % 1000
                features[7] = udp_layer.dport % 1000
                features[8] = 0  # No flags for UDP
                features[9] = len(udp_layer.payload) if udp_layer.payload else 0
                
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
        
        return features
