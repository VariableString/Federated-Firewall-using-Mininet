# src/core/simple_packet_analyzer.py
import numpy as np
import time
import logging
import hashlib
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class SimplePacketAnalyzer:
    """Advanced packet feature extraction with 10 key features"""
    
    def __init__(self):
        self.packet_count = 0
        self.start_time = time.time()
        self.connection_stats = defaultdict(lambda: {
            'packets': 0, 
            'bytes': 0, 
            'start_time': time.time(),
            'last_seen': time.time()
        })
        self.recent_ips = deque(maxlen=100)
        self.recent_ports = deque(maxlen=100)
        
        # Feature normalization ranges (more conservative)
        self.feature_ranges = {
            0: (64, 1500),      # packet_size
            1: (0, 86400),      # time_of_day (seconds in day)
            2: (0, 1000),       # packet_sequence
            3: (1, 254),        # src_ip_last_octet
            4: (1, 254),        # dst_ip_last_octet  
            5: (1, 255),        # protocol_type
            6: (1, 255),        # ttl_value
            7: (1, 100),        # connection_count (reduced range)
            8: (0, 1),          # suspicious_score
            9: (0, 1)           # traffic_entropy
        }
        
        logger.debug("Packet analyzer initialized")
        
    def extract_features(self, packet_data=None):
        """Extract comprehensive 10-feature vector from packet"""
        
        try:
            current_time = time.time()
            features = np.zeros(10, dtype=np.float32)
            
            # Generate realistic packet characteristics
            if packet_data is None:
                packet_data = self._generate_realistic_packet()
            
            # Feature 0: Packet Size (normalized)
            packet_size = packet_data.get('size', np.random.randint(64, 1501))
            features[0] = self._normalize_feature(packet_size, 0)
            
            # Feature 1: Time of Day (normalized)  
            time_of_day = current_time % 86400  # seconds since midnight
            features[1] = self._normalize_feature(time_of_day, 1)
            
            # Feature 2: Packet Sequence (normalized)
            features[2] = self._normalize_feature(self.packet_count % 1000, 2)
            
            # Feature 3: Source IP Last Octet
            src_ip = packet_data.get('src_ip', f"10.0.0.{np.random.randint(1, 4)}")
            try:
                src_last = int(src_ip.split('.')[-1])
            except (ValueError, IndexError):
                src_last = np.random.randint(1, 254)
            features[3] = self._normalize_feature(src_last, 3)
            
            # Feature 4: Destination IP Last Octet
            dst_ip = packet_data.get('dst_ip', f"10.0.0.{np.random.randint(1, 4)}")
            try:
                dst_last = int(dst_ip.split('.')[-1])
            except (ValueError, IndexError):
                dst_last = np.random.randint(1, 254)
            features[4] = self._normalize_feature(dst_last, 4)
            
            # Feature 5: Protocol Type
            protocol = packet_data.get('protocol', np.random.choice([6, 17, 1]))  # TCP, UDP, ICMP
            features[5] = self._normalize_feature(protocol, 5)
            
            # Feature 6: TTL Value
            ttl = packet_data.get('ttl', np.random.randint(32, 129))
            features[6] = self._normalize_feature(ttl, 6)
            
            # Feature 7: Connection Count (estimated)
            connection_key = f"{src_ip}-{dst_ip}"
            self.connection_stats[connection_key]['packets'] += 1
            self.connection_stats[connection_key]['last_seen'] = current_time
            active_connections = len([c for c in self.connection_stats.values() 
                                    if current_time - c['last_seen'] < 300])  # 5 min window
            features[7] = self._normalize_feature(min(active_connections, 100), 7)
            
            # Feature 8: Suspicious Score
            suspicious_score = self._calculate_suspicious_score(packet_data, features)
            features[8] = max(0.0, min(1.0, suspicious_score))  # Explicit clipping
            
            # Feature 9: Traffic Entropy
            entropy = self._calculate_traffic_entropy(packet_data)
            features[9] = max(0.0, min(1.0, entropy))  # Explicit clipping
            
            # Update tracking
            self.packet_count += 1
            self.recent_ips.append(src_ip)
            if 'src_port' in packet_data:
                self.recent_ports.append(packet_data['src_port'])
            
            # Final validation and cleaning
            features = self._validate_features(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return safe default features
            return self._get_safe_default_features()
    
    def _generate_realistic_packet(self):
        """Generate realistic packet data for simulation"""
        try:
            protocols = [6, 17, 1]  # TCP, UDP, ICMP
            protocol = np.random.choice(protocols)
            
            # Generate realistic packet based on protocol
            if protocol == 6:  # TCP
                # Fixed probability arrays that sum to exactly 1.0
                size_choices = [64, 128, 512, 1024, 1460]
                size_probs = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
                size_probs = size_probs / size_probs.sum()  # Ensure sum = 1.0
                
                # Create port choices with proper probabilities
                common_ports = [80, 443, 22, 21, 25, 53]
                high_ports = list(range(1024, 65536))
                
                # Use simpler approach for port selection
                if np.random.random() < 0.6:  # 60% chance for common ports
                    src_port = np.random.choice(common_ports)
                else:  # 40% chance for high ports
                    src_port = np.random.choice(high_ports)
                
                packet = {
                    'size': np.random.choice(size_choices, p=size_probs),
                    'protocol': 6,
                    'src_port': src_port,
                    'ttl': np.random.randint(32, 129)
                }
                
            elif protocol == 17:  # UDP  
                # Simpler approach for UDP ports
                dns_ports = [53, 67, 68, 123]
                if np.random.random() < 0.7:  # 70% chance for DNS/DHCP/NTP
                    src_port = np.random.choice(dns_ports)
                else:  # 30% chance for high ports
                    src_port = np.random.randint(1024, 65536)
                
                packet = {
                    'size': np.random.randint(64, 1025),
                    'protocol': 17,
                    'src_port': src_port,
                    'ttl': np.random.randint(32, 129)
                }
                
            else:  # ICMP
                packet = {
                    'size': np.random.randint(64, 128),
                    'protocol': 1,
                    'ttl': np.random.randint(32, 129)
                }
            
            # Add IP addresses - simple topology (10.0.0.x)
            packet['src_ip'] = f"10.0.0.{np.random.randint(1, 4)}"
            packet['dst_ip'] = f"10.0.0.{np.random.randint(1, 4)}"
            
            return packet
            
        except Exception as e:
            logger.error(f"Packet generation error: {e}")
            return {
                'size': 64,
                'protocol': 6,
                'src_ip': '10.0.0.1',
                'dst_ip': '10.0.0.2',
                'ttl': 64,
                'src_port': 80
            }
    
    def _normalize_feature(self, value, feature_index):
        """Normalize feature to [0, 1] range with error handling"""
        try:
            min_val, max_val = self.feature_ranges[feature_index]
            normalized = (value - min_val) / (max_val - min_val)
            return np.clip(normalized, 0.0, 1.0)
        except Exception as e:
            logger.error(f"Normalization error for feature {feature_index}: {e}")
            return 0.5  # Safe default
    
    def _calculate_suspicious_score(self, packet_data, features):
        """Calculate suspiciousness score based on various indicators"""
        try:
            score = 0.0
            
            # Large packet size indicator
            size = packet_data.get('size', 0)
            if size > 1200:
                score += 0.2
            elif size < 64:
                score += 0.1
            
            # Unusual port usage
            src_port = packet_data.get('src_port', 0)
            if src_port in [4444, 6666, 1234, 31337]:  # Known malicious ports
                score += 0.3
            elif src_port > 60000:  # High port numbers
                score += 0.1
            
            # Protocol anomalies
            protocol = packet_data.get('protocol', 6)
            if protocol == 1:  # ICMP - check for floods
                recent_icmp = sum(1 for ip in list(self.recent_ips)[-10:] if '10.0.0' in str(ip))
                if recent_icmp > 5:
                    score += 0.2
            
            # TTL anomalies
            ttl = packet_data.get('ttl', 64)
            if ttl < 32 or ttl > 128:
                score += 0.15
            
            # Time-based patterns
            current_time = time.time()
            time_of_day = current_time % 86400
            if time_of_day < 21600 or time_of_day > 79200:  # Night hours
                score += 0.05
            
            # Add controlled randomness
            score += np.random.uniform(-0.05, 0.1)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.debug(f"Suspicious score calculation error: {e}")
            return np.random.uniform(0.0, 0.5)
    
    def _calculate_traffic_entropy(self, packet_data):
        """Calculate entropy of recent traffic patterns"""
        try:
            if len(self.recent_ips) < 3:
                return np.random.uniform(0.3, 0.7)
            
            # Calculate IP entropy from recent traffic
            ip_counts = {}
            recent_ips_list = list(self.recent_ips)[-20:]  # Last 20 IPs
            
            for ip in recent_ips_list:
                ip_counts[ip] = ip_counts.get(ip, 0) + 1
            
            total = sum(ip_counts.values())
            if total == 0:
                return 0.5
            
            entropy = 0.0
            for count in ip_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # Normalize entropy (max for 20 different IPs ≈ 4.32)
            max_entropy = min(4.32, np.log2(len(ip_counts)))
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0.5
            
            return max(0.0, min(1.0, normalized_entropy))
            
        except Exception as e:
            logger.debug(f"Entropy calculation error: {e}")
            return np.random.uniform(0.3, 0.7)
    
    def _validate_features(self, features):
        """Validate and clean feature vector"""
        try:
            # Replace any NaN or inf values
            features = np.nan_to_num(features, nan=0.5, posinf=1.0, neginf=0.0)
            
            # Ensure all features are in [0, 1] range
            features = np.clip(features, 0.0, 1.0)
            
            # Ensure correct data type
            features = features.astype(np.float32)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature validation error: {e}")
            return self._get_safe_default_features()
    
    def _get_safe_default_features(self):
        """Return safe default features in case of errors"""
        return np.random.uniform(0.2, 0.8, 10).astype(np.float32)
    
    def get_analyzer_stats(self):
        """Get analyzer statistics"""
        try:
            current_time = time.time()
            active_connections = len([c for c in self.connection_stats.values() 
                                    if current_time - c['last_seen'] < 300])
            
            return {
                'packets_processed': self.packet_count,
                'active_connections': active_connections,
                'unique_ips_seen': len(set(self.recent_ips)),
                'uptime_seconds': current_time - self.start_time
            }
        except Exception as e:
            logger.error(f"Error getting analyzer stats: {e}")
            return {
                'packets_processed': self.packet_count,
                'active_connections': 0,
                'unique_ips_seen': 0,
                'uptime_seconds': time.time() - self.start_time
            }
    
    def reset_stats(self):
        """Reset analyzer statistics"""
        try:
            self.packet_count = 0
            self.start_time = time.time()
            self.connection_stats.clear()
            self.recent_ips.clear()
            self.recent_ports.clear()
            logger.debug("Analyzer statistics reset")
        except Exception as e:
            logger.error(f"Error resetting analyzer stats: {e}")