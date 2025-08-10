#!/bin/bash

# Robust Federated Firewall System Startup Script
echo "🔥 Starting Robust Federated Firewall System..."

# Check root privileges
if [ "$EUID" -ne 0 ]; then
    echo "🚫 Please run as root: sudo env "PATH=\$PATH" bash scripts/start_system.sh"
    exit 1
fi

# Preserve environment PATH
export PATH="$PATH"

# Set Python environment
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Clean up any existing processes
echo "🧹 Cleaning environment..."
mn -c >/dev/null 2>&1
pkill -f "python.*main_simple" >/dev/null 2>&1
fuser -k 6633/tcp >/dev/null 2>&1

# Start system with proper environment
echo "🚀 Launching system..."
env "PATH=$PATH" python3 src/main_simple.py "$@"

echo "✅ System shutdown complete"
