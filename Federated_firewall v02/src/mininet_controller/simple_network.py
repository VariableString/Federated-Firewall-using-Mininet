import asyncio
import logging
import time
import subprocess
import signal
import sys
from pathlib import Path

# Import with error handling
try:
    from mininet.net import Mininet
    from mininet.topo import Topo
    from mininet.node import Controller, OVSKernelSwitch
    from mininet.link import TCLink, Intf
    from mininet.log import setLogLevel
    from mininet.cli import CLI
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Mininet import error: {e}")
    logger.error("Please install Mininet: sudo apt-get install mininet")
    raise

try:
    from federated.simple_client import SimpleFederatedClient
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Federated client import error: {e}")
    raise

logger = logging.getLogger(__name__)

class SimpleTopology(Topo):
    """Simple star topology - 1 switch, 3 hosts with direct connections"""
    
    def __init__(self, num_hosts=3):
        super(SimpleTopology, self).__init__()
        
        logger.info(f"Creating simple star topology with {num_hosts} hosts")
        
        # Add single switch
        switch = self.addSwitch('s1', 
                               cls=OVSKernelSwitch,
                               protocols='OpenFlow13',
                               failMode='standalone')
        
        # Add hosts with direct links to switch
        for i in range(num_hosts):
            host_name = f'h{i+1}'
            host_ip = f'10.0.0.{i+1}'
            
            # Add host
            host = self.addHost(host_name, 
                               ip=f'{host_ip}/24',
                               mac=f'00:00:00:00:00:0{i+1}')
            
            # Add direct link to switch
            self.addLink(host, switch,
                        cls=TCLink,
                        bw=100,        # 100 Mbps
                        delay='1ms',   # 1ms delay
                        loss=0)        # No packet loss
            
            logger.debug(f"Added host {host_name} ({host_ip}) with direct link to switch")
        
        logger.info(f"Simple topology created: {num_hosts} hosts -> 1 switch")

class RobustNetworkManager:
    """Network manager optimized for simple topology and connectivity"""
    
    def __init__(self, config):
        self.config = config
        self.net = None
        self.clients = []
        self.is_running = False
        self.current_phase = "learning"
        
        # Configuration
        self.max_connectivity_attempts = 3
        self.connectivity_timeout = 5
        
        # Set Mininet logging to reduce noise
        setLogLevel('error')
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Network manager initialized for simple topology")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if self.is_running:
            asyncio.create_task(self.stop_system())
    
    async def start_system(self):
        """Start the complete federated firewall system"""
        logger.info("=" * 60)
        logger.info("STARTING ROBUST FEDERATED FIREWALL SYSTEM")
        logger.info("=" * 60)
        
        try:
            # Pre-startup cleanup
            await self._cleanup_environment()
            
            # Create simple network topology
            await self._create_simple_network()
            
            # Configure network for optimal connectivity
            await self._configure_simple_network()
            
            # Verify basic connectivity
            await self._verify_simple_connectivity()
            
            # Start federated learning system
            await self._start_federated_learning()
            
            # Start system management
            await self._start_system_management()
            
            self.is_running = True
            logger.info("🚀 SYSTEM FULLY OPERATIONAL")
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            await self.stop_system()
            raise
    
    async def stop_system(self):
        """Stop the system gracefully"""
        if not self.is_running:
            return
        
        logger.info("🛑 INITIATING SYSTEM SHUTDOWN")
        self.is_running = False
        
        try:
            # Stop federated clients
            logger.info("Stopping federated clients...")
            stop_tasks = [client.stop() for client in self.clients]
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
            self.clients.clear()
            
            # Stop network
            if self.net:
                logger.info("Stopping Mininet network...")
                try:
                    self.net.stop()
                except Exception as e:
                    logger.error(f"Error stopping network: {e}")
                self.net = None
            
            # Final cleanup
            await self._cleanup_environment()
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
        
        logger.info("✅ SYSTEM SHUTDOWN COMPLETE")
    
    async def _cleanup_environment(self):
        """Comprehensive environment cleanup"""
        logger.info("Cleaning up environment...")
        
        cleanup_commands = [
            ['sudo', 'mn', '-c'],
            ['sudo', 'pkill', '-f', 'controller'],
            ['sudo', 'fuser', '-k', '6633/tcp']
        ]
        
        for cmd in cleanup_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                logger.debug(f"Cleanup command {' '.join(cmd)}: {result.returncode}")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.debug(f"Cleanup command {' '.join(cmd)} failed: {e}")
        
        await asyncio.sleep(2)
        logger.info("Environment cleanup completed")
    
    async def _create_simple_network(self):
        """Create simple Mininet network optimized for connectivity"""
        logger.info("Creating simple star topology network...")
        
        try:
            # Create simple topology
            num_clients = self.config.get('mininet', {}).get('num_clients', 3)
            topo = SimpleTopology(num_clients)
            
            # Create network with minimal configuration for maximum reliability
            self.net = Mininet(
                topo=topo,
                switch=OVSKernelSwitch,
                controller=Controller,
                link=TCLink,
                autoSetMacs=True,
                autoStaticArp=True,
                waitConnected=True
            )
            
            # Start the network
            logger.info("Starting Mininet network...")
            self.net.start()
            
            # Wait for network to stabilize
            await asyncio.sleep(3)
            
            logger.info("✅ Simple network created and started")
            
        except Exception as e:
            logger.error(f"Network creation failed: {e}")
            raise
    
    async def _configure_simple_network(self):
        """Configure network for simple topology"""
        logger.info("Configuring simple network...")
        
        try:
            # Configure switch for normal forwarding
            switch = self.net.get('s1')
            if switch:
                # Add basic flow rule for normal L2 switching
                switch.cmd('ovs-ofctl add-flow s1 actions=normal')
                logger.info("Switch configured for normal L2 forwarding")
            
            # Configure each host
            for i, host in enumerate(self.net.hosts):
                host_name = host.name
                host_ip = host.IP()
                
                logger.info(f"Configuring host {host_name} ({host_ip})")
                
                # Basic network configuration
                host.cmd('ip link set lo up')
                host.cmd(f'ip link set {host_name}-eth0 up')
                
                # Disable IPv6 to simplify networking
                host.cmd('sysctl -w net.ipv6.conf.all.disable_ipv6=1 2>/dev/null')
                host.cmd('sysctl -w net.ipv6.conf.default.disable_ipv6=1 2>/dev/null')
                
                # Set up static ARP entries for all other hosts
                for j, other_host in enumerate(self.net.hosts):
                    if host != other_host:
                        other_ip = other_host.IP()
                        other_mac = other_host.MAC()
                        host.cmd(f'arp -s {other_ip} {other_mac}')
                        logger.debug(f"Added ARP entry: {other_ip} -> {other_mac}")
                
                # Configure default route
                host.cmd(f'ip route add default dev {host_name}-eth0')
            
            logger.info("✅ Simple network configuration completed")
            
        except Exception as e:
            logger.error(f"Network configuration failed: {e}")
            raise
    
    async def _verify_simple_connectivity(self):
        """Verify connectivity in simple topology"""
        logger.info("Verifying network connectivity...")
        
        for attempt in range(1, self.max_connectivity_attempts + 1):
            logger.info(f"Connectivity test {attempt}/{self.max_connectivity_attempts}")
            
            try:
                # Use simple ping test with short timeout
                logger.info("Running pingall test...")
                result = self.net.pingAll(timeout='2')
                
                if result == 0.0:
                    logger.info("🎉 PERFECT CONNECTIVITY: 0% packet loss")
                    return True
                elif result <= 10.0:
                    logger.info(f"✅ GOOD CONNECTIVITY: {result}% packet loss (acceptable)")
                    return True
                else:
                    logger.warning(f"⚠️ POOR CONNECTIVITY: {result}% packet loss")
                
                # If poor connectivity, try to fix
                if attempt < self.max_connectivity_attempts:
                    await self._fix_simple_connectivity()
                    await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Connectivity test {attempt} failed: {e}")
                if attempt < self.max_connectivity_attempts:
                    await asyncio.sleep(2)
        
        # Accept even imperfect connectivity for simple topology
        logger.warning("Proceeding with current connectivity level")
        return True
    
    async def _fix_simple_connectivity(self):
        """Fix connectivity issues in simple topology"""
        logger.info("Attempting to fix connectivity...")
        
        try:
            # Restart switch flows
            switch = self.net.get('s1')
            if switch:
                switch.cmd('ovs-ofctl del-flows s1')
                await asyncio.sleep(1)
                switch.cmd('ovs-ofctl add-flow s1 actions=normal')
                logger.debug("Reset switch flows")
            
            # Refresh ARP tables
            for host in self.net.hosts:
                try:
                    host.cmd('ip neigh flush all')
                    # Re-add static ARP entries
                    for other_host in self.net.hosts:
                        if host != other_host:
                            host.cmd(f'arp -s {other_host.IP()} {other_host.MAC()}')
                except Exception as e:
                    logger.debug(f"ARP refresh error for {host.name}: {e}")
            
            logger.info("Connectivity fix completed")
            
        except Exception as e:
            logger.error(f"Connectivity fix error: {e}")
    
    async def _start_federated_learning(self):
        """Start federated learning clients"""
        logger.info("Starting federated learning system...")
        
        hosts = self.net.hosts
        successful_clients = 0
        
        # Create and start clients
        for i, host in enumerate(hosts):
            try:
                client = SimpleFederatedClient(i, self.config, host)
                self.clients.append(client)
                
                # Start client as background task
                asyncio.create_task(client.start())
                successful_clients += 1
                
                logger.info(f"✅ Started federated client on {host.name} ({host.IP()})")
                
                # Small delay between client starts
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to start client on {host.name}: {e}")
        
        if successful_clients == 0:
            raise Exception("No federated clients started successfully")
        
        logger.info(f"🤖 Started {successful_clients} federated learning clients")
    
    async def _start_system_management(self):
        """Start system management tasks"""
        logger.info("Starting system management...")
        
        # Start management tasks
        management_tasks = [
            self._phase_management_loop(),
            self._federated_averaging_loop(),
            self._system_monitoring_loop()
        ]
        
        for task in management_tasks:
            asyncio.create_task(task)
        
        logger.info("🔧 System management tasks started")
    
    async def _phase_management_loop(self):
        """Manage learning and testing phases"""
        logger.info("🔄 Starting phase management")
        
        try:
            # Learning phase
            learning_rounds = self.config.get('phases', {}).get('learning_rounds', 5)
            learning_duration = self.config.get('phases', {}).get('learning_duration', 60)
            total_learning_time = learning_rounds * learning_duration
            
            logger.info(f"📚 LEARNING PHASE: {total_learning_time} seconds ({learning_rounds} rounds)")
            
            await asyncio.sleep(total_learning_time)
            
            # Switch to testing phase
            logger.info("🔄 SWITCHING TO TESTING PHASE")
            self.current_phase = "testing"
            
            for client in self.clients:
                try:
                    client.update_phase("testing")
                except Exception as e:
                    logger.error(f"Error updating phase for client: {e}")
            
            # Testing phase
            testing_rounds = self.config.get('phases', {}).get('testing_rounds', 2)
            testing_duration = self.config.get('phases', {}).get('testing_duration', 30)
            total_testing_time = testing_rounds * testing_duration
            
            logger.info(f"🧪 TESTING PHASE: {total_testing_time} seconds ({testing_rounds} rounds)")
            
            await asyncio.sleep(total_testing_time)
            
            # Final evaluation
            await self._perform_final_evaluation()
            
            logger.info("✅ ALL PHASES COMPLETED")
            
        except Exception as e:
            logger.error(f"Phase management error: {e}")
    
    async def _federated_averaging_loop(self):
        """Perform federated averaging between clients"""
        try:
            averaging_interval = self.config.get('federated', {}).get('aggregation_interval', 30)
            
            while self.is_running:
                await asyncio.sleep(averaging_interval)
                
                if len(self.clients) > 1:
                    await self._perform_federated_averaging()
                    
        except Exception as e:
            logger.error(f"Federated averaging loop error: {e}")
    
    async def _perform_federated_averaging(self):
        """Execute federated averaging algorithm with error handling"""
        try:
            logger.info("🔄 Performing federated averaging...")
            
            # Collect weights from all clients
            client_weights = []
            client_sizes = []
            successful_clients = []
            
            for client in self.clients:
                try:
                    weights = client.get_model_weights()
                    stats = client.get_training_stats()
                    
                    if weights and 'data_stats' in stats and stats['data_stats']['training_samples'] > 0:
                        client_weights.append(weights)
                        client_sizes.append(stats['data_stats']['training_samples'])
                        successful_clients.append(client.host_name)
                        
                except Exception as e:
                    logger.error(f"Error collecting weights from {getattr(client, 'host_name', 'unknown')}: {e}")
            
            if len(client_weights) < 2:
                logger.warning("Insufficient clients for federated averaging")
                return
            
            # Perform weighted averaging
            averaged_weights = self._federated_average_weights(client_weights, client_sizes)
            
            if not averaged_weights:
                logger.error("Federated averaging failed")
                return
            
            # Distribute averaged weights back to clients
            distribution_count = 0
            for client in self.clients:
                try:
                    client.set_model_weights(averaged_weights)
                    distribution_count += 1
                except Exception as e:
                    logger.error(f"Error distributing weights to {getattr(client, 'host_name', 'unknown')}: {e}")
            
            logger.info(f"✅ Federated averaging completed: "
                       f"{len(successful_clients)} contributors, "
                       f"{distribution_count} recipients")
            
        except Exception as e:
            logger.error(f"Federated averaging error: {e}")
    
    def _federated_average_weights(self, weights_list, client_sizes):
        """Perform weighted federated averaging with error handling"""
        try:
            import torch
            
            if not weights_list or not client_sizes:
                logger.error("Empty weights list or client sizes")
                return {}
            
            # Calculate total samples for weighting
            total_samples = sum(client_sizes)
            if total_samples == 0:
                logger.error("Total samples is zero")
                return weights_list[0] if weights_list else {}
            
            # Initialize averaged weights
            averaged_weights = {}
            
            # Get parameter names from first client
            param_names = list(weights_list[0].keys())
            
            # Average each parameter
            for param_name in param_names:
                try:
                    weighted_params = []
                    
                    for i, client_weights in enumerate(weights_list):
                        if param_name in client_weights:
                            weight = client_sizes[i] / total_samples
                            weighted_param = weight * client_weights[param_name]
                            weighted_params.append(weighted_param)
                    
                    if weighted_params:
                        averaged_weights[param_name] = torch.stack(weighted_params).sum(dim=0)
                    else:
                        logger.warning(f"No weights found for parameter {param_name}")
                        
                except Exception as e:
                    logger.error(f"Error averaging parameter {param_name}: {e}")
                    continue
            
            return averaged_weights
            
        except Exception as e:
            logger.error(f"Weight averaging error: {e}")
            return weights_list[0] if weights_list else {}
    
    async def _system_monitoring_loop(self):
        """Monitor overall system health and performance"""
        monitoring_interval = 60  # Monitor every minute
        
        while self.is_running:
            try:
                await asyncio.sleep(monitoring_interval)
                await self._generate_system_report()
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
    
    async def _generate_system_report(self):
        """Generate comprehensive system performance report"""
        try:
            logger.info("=" * 50)
            logger.info("SYSTEM PERFORMANCE REPORT")
            logger.info("=" * 50)
            
            # Collect statistics from all clients
            system_stats = {
                'total_packets': 0,
                'total_threats': 0,
                'total_training_samples': 0,
                'total_blocked_packets': 0,
                'average_accuracy': 0.0,
                'active_clients': 0,
                'client_details': []
            }
            
            for client in self.clients:
                try:
                    stats = client.get_training_stats()
                    
                    if 'performance_metrics' in stats:
                        system_stats['total_packets'] += stats['performance_metrics']['packets_processed']
                        system_stats['total_threats'] += stats['performance_metrics']['threats_detected']
                        system_stats['average_accuracy'] += stats['performance_metrics']['model_accuracy']
                        
                    if 'data_stats' in stats:
                        system_stats['total_training_samples'] += stats['data_stats']['training_samples']
                        
                    if 'firewall_stats' in stats:
                        system_stats['total_blocked_packets'] += stats['firewall_stats']['packets_blocked']
                    
                    system_stats['active_clients'] += 1
                    
                    system_stats['client_details'].append({
                        'host': stats.get('host', 'unknown'),
                        'packets': stats.get('performance_metrics', {}).get('packets_processed', 0),
                        'threats': stats.get('performance_metrics', {}).get('threats_detected', 0),
                        'accuracy': stats.get('performance_metrics', {}).get('model_accuracy', 0.0),
                        'samples': stats.get('data_stats', {}).get('training_samples', 0)
                    })
                    
                except Exception as e:
                    logger.error(f"Error collecting stats from client: {e}")
            
            # Calculate averages
            if system_stats['active_clients'] > 0:
                system_stats['average_accuracy'] /= system_stats['active_clients']
                system_stats['average_threat_ratio'] = (
                    system_stats['total_threats'] / max(system_stats['total_packets'], 1)
                )
            else:
                system_stats['average_threat_ratio'] = 0.0
            
            # Log system-wide statistics
            logger.info(f"Phase: {self.current_phase.upper()}")
            logger.info(f"Active Clients: {system_stats['active_clients']}")
            logger.info(f"Total Packets Processed: {system_stats['total_packets']:,}")
            logger.info(f"Total Threats Detected: {system_stats['total_threats']:,}")
            logger.info(f"Total Training Samples: {system_stats['total_training_samples']:,}")
            logger.info(f"Average Model Accuracy: {system_stats['average_accuracy']:.3f}")
            logger.info(f"System Threat Detection Rate: {system_stats['average_threat_ratio']:.3f}")
            
            # Client breakdown
            logger.info("\nCLIENT BREAKDOWN:")
            for client_info in system_stats['client_details']:
                threat_ratio = client_info['threats'] / max(client_info['packets'], 1)
                logger.info(f"  {client_info['host']}: "
                          f"{client_info['packets']:,} packets, "
                          f"{client_info['threats']:,} threats ({threat_ratio:.3f}), "
                          f"{client_info['accuracy']:.3f} accuracy, "
                          f"{client_info['samples']:,} samples")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"System report generation error: {e}")
    
    async def _perform_final_evaluation(self):
        """Perform comprehensive final system evaluation"""
        logger.info("🎯 PERFORMING FINAL SYSTEM EVALUATION")
        logger.info("=" * 60)
        
        try:
            # Collect final statistics
            final_stats = {
                'evaluation_time': time.time(),
                'total_runtime': time.time() - getattr(self, 'start_time', time.time()),
                'system_phase': self.current_phase,
                'clients': []
            }
            
            system_totals = {
                'packets': 0, 'threats': 0, 'samples': 0, 'blocked': 0,
                'accuracy_sum': 0.0, 'active_count': 0
            }
            
            # Collect from each client
            for client in self.clients:
                try:
                    stats = client.get_training_stats()
                    
                    client_summary = {
                        'host_name': stats.get('host', 'unknown'),
                        'host_ip': stats.get('host_ip', 'unknown'),
                        'packets_processed': stats.get('performance_metrics', {}).get('packets_processed', 0),
                        'threats_detected': stats.get('performance_metrics', {}).get('threats_detected', 0),
                        'model_accuracy': stats.get('performance_metrics', {}).get('model_accuracy', 0.0),
                        'training_samples': stats.get('data_stats', {}).get('training_samples', 0),
                        'threat_detection_rate': stats.get('data_stats', {}).get('threat_ratio', 0.0),
                        'firewall_blocks': stats.get('firewall_stats', {}).get('packets_blocked', 0),
                        'training_rounds': stats.get('training_round', 0)
                    }
                    
                    final_stats['clients'].append(client_summary)
                    
                    # Add to system totals
                    system_totals['packets'] += client_summary['packets_processed']
                    system_totals['threats'] += client_summary['threats_detected']
                    system_totals['samples'] += client_summary['training_samples']
                    system_totals['blocked'] += client_summary['firewall_blocks']
                    system_totals['accuracy_sum'] += client_summary['model_accuracy']
                    system_totals['active_count'] += 1
                    
                except Exception as e:
                    logger.error(f"Error collecting final stats from client: {e}")
            
            # Calculate system-wide metrics
            avg_accuracy = system_totals['accuracy_sum'] / max(system_totals['active_count'], 1)
            overall_threat_rate = system_totals['threats'] / max(system_totals['packets'], 1)
            blocking_rate = system_totals['blocked'] / max(system_totals['packets'], 1)
            
            # Display final results
            logger.info("🏆 FINAL SYSTEM RESULTS")
            logger.info(f"Runtime: {final_stats['total_runtime']:.1f} seconds")
            logger.info(f"Active Clients: {system_totals['active_count']}")
            logger.info(f"Total Packets Analyzed: {system_totals['packets']:,}")
            logger.info(f"Total Threats Detected: {system_totals['threats']:,}")
            logger.info(f"Total Training Samples Generated: {system_totals['samples']:,}")
            logger.info(f"Average Model Accuracy: {avg_accuracy:.3f}")
            logger.info(f"System Threat Detection Rate: {overall_threat_rate:.3f}")
            logger.info(f"Packet Blocking Rate: {blocking_rate:.3f}")
            
            logger.info("\n📊 CLIENT PERFORMANCE SUMMARY:")
            for client_info in final_stats['clients']:
                logger.info(f"  🖥️  {client_info['host_name']} ({client_info['host_ip']}):")
                logger.info(f"      Packets: {client_info['packets_processed']:,}")
                logger.info(f"      Threats: {client_info['threats_detected']:,} "
                          f"({client_info['threat_detection_rate']:.3f} rate)")
                logger.info(f"      Accuracy: {client_info['model_accuracy']:.3f}")
                logger.info(f"      Training: {client_info['training_rounds']} rounds, "
                          f"{client_info['training_samples']:,} samples")
                logger.info(f"      Blocked: {client_info['firewall_blocks']:,} packets")
            
            # Performance assessment
            logger.info("\n🎯 SYSTEM ASSESSMENT:")
            
            if avg_accuracy >= 0.8:
                logger.info("✅ EXCELLENT: High model accuracy achieved")
            elif avg_accuracy >= 0.7:
                logger.info("✅ GOOD: Acceptable model accuracy")
            else:
                logger.info("⚠️  NEEDS IMPROVEMENT: Low model accuracy")
            
            if overall_threat_rate >= 0.1:
                logger.info("✅ GOOD: Healthy threat detection rate")
            else:
                logger.info("ℹ️  INFO: Low threat detection (expected in simulation)")
            
            if system_totals['packets'] >= 1000:
                logger.info("✅ EXCELLENT: High packet processing volume")
            elif system_totals['packets'] >= 500:
                logger.info("✅ GOOD: Adequate packet processing")
            else:
                logger.info("⚠️  LOW: Limited packet processing")
            
            logger.info("\n🎉 FEDERATED FIREWALL EVALUATION COMPLETE!")
            logger.info("=" * 60)
            
            # Save results to file
            await self._save_final_results(final_stats, system_totals, avg_accuracy, overall_threat_rate)
            
        except Exception as e:
            logger.error(f"Final evaluation error: {e}")
    
    async def _save_final_results(self, final_stats, system_totals, avg_accuracy, overall_threat_rate):
        """Save final evaluation results to file"""
        try:
            import json
            from datetime import datetime
            
            results_dir = Path("logs/results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"final_results_{timestamp}.json"
            
            results_data = {
                'evaluation_timestamp': timestamp,
                'system_configuration': {
                    'num_clients': len(self.clients),
                    'learning_rounds': self.config.get('phases', {}).get('learning_rounds', 0),
                    'testing_rounds': self.config.get('phases', {}).get('testing_rounds', 0)
                },
                'system_totals': system_totals,
                'performance_metrics': {
                    'average_accuracy': avg_accuracy,
                    'overall_threat_rate': overall_threat_rate,
                    'blocking_rate': system_totals['blocked'] / max(system_totals['packets'], 1)
                },
                'client_details': final_stats['clients'],
                'runtime_seconds': final_stats.get('total_runtime', 0)
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"📄 Final results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
    
    def get_current_phase(self):
        """Get current system phase"""
        return self.current_phase
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            return {
                'is_running': self.is_running,
                'current_phase': self.current_phase,
                'num_clients': len(self.clients),
                'network_active': self.net is not None,
                'clients_status': [
                    {
                        'host_name': getattr(client, 'host_name', 'unknown'),
                        'host_ip': getattr(client, 'host_ip', 'unknown'),
                        'is_running': getattr(client, 'is_running', False),
                        'current_phase': getattr(client, 'current_phase', 'unknown')
                    }
                    for client in self.clients
                ]
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'is_running': self.is_running,
                'current_phase': self.current_phase,
                'num_clients': 0,
                'network_active': False,
                'clients_status': [],
                'error': str(e)
            }