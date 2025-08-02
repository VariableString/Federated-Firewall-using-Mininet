#!/bin/bash

# Simple startup script

echo "Starting Simplified Federated Firewall..."

# Check root privileges
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash scripts/start_simple.sh"
    exit 1
fi

# Create directories
mkdir -p logs

# Clean up existing Mininet
mn -c > /dev/null 2>&1

# Set Python path
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Start system
echo "Launching system..."
python3 src/main.py

echo "System started!"