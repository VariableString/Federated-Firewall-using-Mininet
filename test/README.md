# Federated Firewall with Mininet Integration

This document provides specific information about using the Federated Firewall system with Mininet.

## Mininet Integration Features

### Network Simulation
- **Realistic Network Topology**: Creates virtual networks with configurable topologies (tree, star, mesh)
- **Host Isolation**: Each federated client runs on a separate Mininet host
- **Traffic Control**: Configurable bandwidth, delay, and packet loss
- **SDN Integration**: Optional SDN controller for advanced network management

### Enhanced Threat Detection
- **25-Dimensional Feature Vector**: Expanded from 20 to include Mininet-specific features
- **Signature-Based Detection**: Combined with ML-based detection for better accuracy
- **Network Context Awareness**: Understands Mininet network topology and routing
- **Real-time Blocking**: Can block malicious flows through SDN controller

### Federated Learning Improvements
- **Secure Aggregation**: Differential privacy support for enhanced security
- **Byzantine Fault Tolerance**: Detects and excludes malicious clients
- **Quality-Weighted Aggregation**: Considers client data quality in aggregation
- **Network-Aware Distribution**: Optimized for Mininet's virtual network constraints

## System Requirements

### Software Dependencies
- Python 3.8+
- Mininet 2.3.0+
- Open vSwitch
- Linux environment (Ubuntu/Debian recommended)
- Root/sudo access

### Hardware Requirements
- Minimum 4GB RAM (8GB recommended)
- Multi-core CPU (for running multiple virtual hosts)
- 10GB free disk space

## Installation Guide

### 1. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install mininet openvswitch-testcontroller
sudo apt-get install tcpdump wireshark-common netcat-openbsd iperf3
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Mininet Environment
```bash
sudo bash scripts/setup_mininet.sh
```

### 4. Test Installation
```bash
sudo python3 scripts/test_mininet.py
```

## Configuration

### Mininet-Specific Settings
```yaml
mininet:
  topology: tree          # Options: tree, star, mesh
  depth: 2               # Tree depth (for tree topology)
  fanout: 2              # Fanout factor (for tree topology)
  controller_port: 6633  # SDN controller port
  switch_type: ovsk      # Switch type
  link_loss: 0.0         # Link packet loss percentage
  link_delay: 10ms       # Link delay
```

### Enhanced Security Settings
```yaml
security:
  threat_threshold: 0.7
  max_connection_rate: 100
  suspicious_ports: [22, 23, 80, 443, 3389, 1433, 5432]
  block_threats: true    # Enable real-time threat blocking
```

## Usage

### Basic Usage
```bash
# Start the federated firewall system
sudo python3 src/main.py

# Or use the startup script
sudo bash scripts/start_system.sh
```

### Advanced Deployment
```bash
# Deploy with custom configuration
sudo python3 scripts/deploy_mininet.py
```

### Monitoring
- **Network Topology**: View in `logs/mininet/topology.log`
- **Traffic Patterns**: Monitor in `logs/mininet/traffic.log`
- **Threat Detection**: Check `logs/threats.log`
- **Federation Progress**: View `logs/federation.log`

## Architecture Overview

### Network Layer
```
┌─────────────────────────────────────────────────────────────┐
│                    Mininet Network                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Coordinator │    │   Client    │    │   Client    │      │
│  │    Host     │    │   Host 1    │    │   Host 2    │      │
│  │             │    │             │    │             │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┤           │
│                            │                   │           │
│                     ┌─────────────┐    ┌─────────────┐      │
│                     │   Switch    │────│   Switch    │      │
│                     │     s1      │    │     s2      │      │
│                     └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Application Layer
```
┌─────────────────────────────────────────────────────────────┐
│                Federated Learning Layer                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Global    │    │   Local     │    │   Local     │      │
│  │    Model    │    │   Model     │    │   Model     │      │
│  │Aggregation  │    │ Training    │    │ Training    │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Threat Detection Layer                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Neural    │    │ Signature   │    │   Flow      │      │
│  │  Firewall   │    │ Detection   │    │  Analysis   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### Enhanced Feature Extraction
The system extracts 25 features from network packets:
1. **Basic Features (0-8)**: Packet size, timestamp, IP addresses, protocol, TTL, IP flags, subnet detection
2. **Transport Features (9-16)**: Port numbers, TCP flags, window size, sequence numbers, suspicious port detection
3. **Flow Features (17-24)**: Packet count, byte count, duration, statistical measures, port scan detection

### Threat Detection Methods
1. **Machine Learning**: Neural network-based classification
2. **Signature-Based**: Known attack pattern detection
3. **Statistical Analysis**: Anomaly detection based on flow statistics
4. **Hybrid Approach**: Combines all methods for robust detection

### Federated Learning Enhancements
- **Differential Privacy**: Adds noise to protect client privacy
- **Byzantine Tolerance**: Detects and excludes malicious participants
- **Adaptive Aggregation**: Weights clients based on data quality
- **Secure Communication**: Optional encryption for model updates

## Topology Options

### Tree Topology
- **Use Case**: Hierarchical networks, enterprise environments
- **Configuration**: Set `depth` and `fanout` parameters
- **Benefits**: Scalable, realistic network structure

### Star Topology
- **Use Case**: Simple centralized networks
- **Configuration**: All clients connect to central switch
- **Benefits**: Easy management, low latency

### Mesh Topology
- **Use Case**: Distributed networks, high availability
- **Configuration**: Partial mesh connectivity
- **Benefits**: Fault tolerance, multiple paths

## Traffic Generation

### Normal Traffic Patterns
- **Web Browsing**: HTTP/HTTPS traffic on ports 80, 443
- **SSH Connections**: Secure shell traffic on port 22
- **DNS Queries**: Domain name resolution on port 53
- **Email**: SMTP, IMAP traffic on various ports

### Attack Patterns
- **Port Scanning**: Sequential port probing
- **DDoS Attacks**: High-volume traffic floods
- **Malware Communication**: C&C server connections
- **Brute Force**: Repeated login attempts

## Performance Considerations

### Resource Usage
- **Memory**: ~500MB per Mininet host
- **CPU**: ~10% per active client during training
- **Network**: Minimal overhead for virtual links
- **Storage**: ~100MB for logs and models

### Scaling Guidelines
- **Small Scale**: 2-5 clients, single machine
- **Medium Scale**: 5-10 clients, multi-core system
- **Large Scale**: 10+ clients, distributed setup

### Optimization Tips
1. **Reduce Feature Dimensionality**: Use feature selection
2. **Batch Processing**: Group packets for processing
3. **Model Compression**: Use lighter neural networks
4. **Sampling**: Process subset of traffic during high load

## Troubleshooting

### Common Issues

#### Mininet Won't Start
```bash
# Clean up existing networks
sudo mn -c

# Check Open vSwitch status
sudo systemctl status openvswitch-switch

# Restart OVS if needed
sudo systemctl restart openvswitch-switch
```

#### Permission Errors
```bash
# Ensure running as root
sudo python3 src/main.py

# Check file permissions
sudo chown -R $(whoami) logs/ data/
```

#### Network Connectivity Issues
```bash
# Test basic connectivity
sudo mn --test pingall

# Check network interfaces
ip link show

# Verify routing
ip route show
```

#### High Resource Usage
```bash
# Monitor system resources
htop

# Check Mininet processes
ps aux | grep mininet

# Reduce number of clients in config
# Set federated.num_clients to lower value
```

### Debug Mode
Enable debug logging:
```yaml
system:
  log_level: DEBUG
```

View detailed logs:
```bash
tail -f logs/system.log
tail -f logs/mininet/network.log
```

## Advanced Features

### SDN Controller Integration
Optional Ryu-based controller for advanced network management:
```bash
# Start Ryu controller
ryu-manager src/mininet_controller/sdn_controller.py

# Configure controller in config.yaml
mininet:
  controller_port: 6633
```

### Quality of Service (QoS)
Configure bandwidth and latency per host:
```python
# In topology configuration
qos_rules = [
    {'host': 'h1', 'bandwidth': '10Mbit', 'delay': '10ms'},
    {'host': 'h2', 'bandwidth': '5Mbit', 'delay': '20ms'}
]
```

### Real-time Monitoring
Web-based dashboard (optional):
```bash
# Install dashboard dependencies
pip install flask socketio

# Start monitoring dashboard
python3 scripts/monitoring_dashboard.py
```

## Testing and Validation

### Unit Tests
```bash
# Run basic tests
python3 -m pytest tests/

# Run Mininet-specific tests (requires root)
sudo python3 -m pytest tests/test_mininet_integration.py
```

### Integration Tests
```bash
# Full system test
sudo python3 scripts/test_mininet.py

# Performance benchmarking
sudo python3 scripts/benchmark_system.py
```

### Validation Metrics
- **Detection Accuracy**: True positive/negative rates
- **Federation Convergence**: Model performance over rounds
- **Network Performance**: Latency, throughput, packet loss
- **Resource Utilization**: CPU, memory, network usage

## Best Practices

### Security
1. **Run in Isolated Environment**: Use VMs or containers
2. **Network Segmentation**: Separate management and data networks
3. **Access Control**: Limit root access, use sudo where needed
4. **Regular Updates**: Keep dependencies updated

### Performance
1. **Resource Monitoring**: Track CPU, memory usage
2. **Load Balancing**: Distribute clients across hosts
3. **Batch Processing**: Group operations for efficiency
4. **Caching**: Store frequently accessed data

### Reliability
1. **Error Handling**: Robust exception management
2. **Graceful Shutdown**: Clean up resources properly
3. **State Persistence**: Save model checkpoints
4. **Health Checks**: Monitor system components

## Contributing

### Development Environment
```bash
# Clone repository
git clone <repository-url>
cd federated_firewall

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### Code Structure
- `src/mininet_controller/`: Mininet network management
- `src/network/`: Topology and traffic generation
- `src/federated/`: Federated learning components
- `src/core/`: Threat detection and analysis
- `src/utils/`: Utility functions and helpers

### Testing Guidelines
1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test system interactions
3. **Performance Tests**: Benchmark critical paths
4. **Security Tests**: Validate threat detection


@software{federated_firewall_mininet,
  title={Federated Learning Distributed Firewall with Mininet},
  author={VariableString},
  year={2025},
  url={https://github.com/example/federated-firewall-mininet}
}
```

---

For more information, see the main README.md file and source code documentation.