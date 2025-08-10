import logging
import threading
import time
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

class SimpleFirewallManager:
    """Simple firewall manager with comprehensive error handling"""
    
    def __init__(self, config):
        try:
            self.config = config
            self.blocked_ips: Set[str] = set()
            self.blocked_ports: Set[int] = set(config.get('firewall', {}).get('blacklist_ports', []))
            self.whitelist_ips: Set[str] = set(config.get('firewall', {}).get('whitelist_ips', []))
            self.enable_blocking = config.get('firewall', {}).get('enable_blocking', False)
            self.max_blocked_ips = config.get('firewall', {}).get('max_blocked_ips', 100)
            
            # Statistics with thread safety
            self.stats = {
                'packets_analyzed': 0,
                'threats_detected': 0,
                'packets_blocked': 0,
                'false_positives': 0
            }
            
            # Thread safety
            self.lock = threading.RLock()
            
            logger.info(f"Firewall manager initialized - Blocking: {self.enable_blocking}")
            
        except Exception as e:
            logger.error(f"Firewall manager initialization error: {e}")
            # Set safe defaults
            self.config = config or {}
            self.blocked_ips = set()
            self.blocked_ports = set()
            self.whitelist_ips = set()
            self.enable_blocking = False
            self.max_blocked_ips = 100
            self.stats = {'packets_analyzed': 0, 'threats_detected': 0, 'packets_blocked': 0, 'false_positives': 0}
            self.lock = threading.RLock()
        
    def analyze_packet(self, features, model_prediction):
        """Analyze packet and decide on blocking action"""
        
        try:
            with self.lock:
                self.stats['packets_analyzed'] += 1
                
                # Extract threat information with error handling
                is_threat = model_prediction.get('is_threat', False)
                confidence = model_prediction.get('confidence', 0.0)
                threat_probability = model_prediction.get('threat_probability', 0.0)
                
                # Validate confidence values
                confidence = max(0.0, min(1.0, confidence))
                threat_probability = max(0.0, min(1.0, threat_probability))
                
                # Determine action based on thresholds
                threat_threshold = self.config.get('security', {}).get('threat_threshold', 0.6)
                block_threshold = self.config.get('security', {}).get('block_threshold', 0.7)
                
                action_info = {
                    'is_threat': is_threat,
                    'confidence': confidence,
                    'threat_probability': threat_probability,
                    'action': 'allow'
                }
                
                if is_threat and confidence > threat_threshold:
                    self.stats['threats_detected'] += 1
                    action_info['action'] = 'detect'
                    
                    if self.enable_blocking and confidence > block_threshold:
                        action_info['action'] = 'block'
                        self.stats['packets_blocked'] += 1
                        
                        # Simulate blocking action
                        self._simulate_block_action(features, action_info)
                
                return action_info
                
        except Exception as e:
            logger.error(f"Packet analysis error: {e}")
            return {
                'action': 'allow', 
                'error': str(e),
                'is_threat': False,
                'confidence': 0.5,
                'threat_probability': 0.5
            }
    
    def _simulate_block_action(self, features, action_info):
        """Simulate blocking action without actual iptables rules"""
        
        try:
            # Generate simulated source IP from features
            if len(features) >= 4:
                src_ip_octet = int(features[3] * 253) + 1  # Convert normalized feature back
                simulated_src_ip = f"10.0.0.{src_ip_octet}"
            else:
                simulated_src_ip = "10.0.0.100"  # Fallback
            
            # Add to blocked IPs (simulation)
            if simulated_src_ip not in self.whitelist_ips:
                self.blocked_ips.add(simulated_src_ip)
                
                # Maintain size limit
                if len(self.blocked_ips) > self.max_blocked_ips:
                    # Remove oldest (convert to list, remove first, convert back)
                    blocked_list = list(self.blocked_ips)
                    oldest_ip = blocked_list[0]
                    self.blocked_ips.remove(oldest_ip)
                    logger.debug(f"Removed oldest blocked IP: {oldest_ip}")
                
                logger.info(f"BLOCKED (simulated): {simulated_src_ip} - "
                          f"Confidence: {action_info['confidence']:.3f}")
                
                action_info['blocked_ip'] = simulated_src_ip
            else:
                logger.warning(f"IP {simulated_src_ip} in whitelist, not blocking")
                action_info['whitelisted'] = True
                
        except Exception as e:
            logger.error(f"Block simulation error: {e}")
            action_info['block_error'] = str(e)
    
    def add_whitelist_ip(self, ip_address):
        """Add IP to whitelist"""
        try:
            with self.lock:
                self.whitelist_ips.add(ip_address)
                # Remove from blocked if present
                self.blocked_ips.discard(ip_address)
                logger.info(f"Added {ip_address} to whitelist")
                return True
        except Exception as e:
            logger.error(f"Error adding IP to whitelist: {e}")
            return False
    
    def remove_blocked_ip(self, ip_address):
        """Remove IP from blocked list"""
        try:
            with self.lock:
                if ip_address in self.blocked_ips:
                    self.blocked_ips.remove(ip_address)
                    logger.info(f"Removed {ip_address} from blocked list")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error removing blocked IP: {e}")
            return False
    
    def get_firewall_stats(self):
        """Get firewall statistics"""
        try:
            with self.lock:
                return {
                    **self.stats.copy(),
                    'blocked_ips_count': len(self.blocked_ips),
                    'whitelist_count': len(self.whitelist_ips),
                    'blocked_ports_count': len(self.blocked_ports),
                    'blocking_enabled': self.enable_blocking
                }
        except Exception as e:
            logger.error(f"Error getting firewall stats: {e}")
            return {
                'packets_analyzed': 0,
                'threats_detected': 0,
                'packets_blocked': 0,
                'false_positives': 0,
                'blocked_ips_count': 0,
                'whitelist_count': 0,
                'blocked_ports_count': 0,
                'blocking_enabled': False
            }
    
    def get_blocked_ips(self):
        """Get list of blocked IPs"""
        try:
            with self.lock:
                return list(self.blocked_ips)
        except Exception as e:
            logger.error(f"Error getting blocked IPs: {e}")
            return []
    
    def reset_stats(self):
        """Reset firewall statistics"""
        try:
            with self.lock:
                self.stats = {
                    'packets_analyzed': 0,
                    'threats_detected': 0,
                    'packets_blocked': 0,
                    'false_positives': 0
                }
                logger.info("Firewall statistics reset")
        except Exception as e:
            logger.error(f"Error resetting firewall stats: {e}")