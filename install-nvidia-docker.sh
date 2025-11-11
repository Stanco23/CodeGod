#!/bin/bash
# Install NVIDIA Container Toolkit for Docker GPU access

set -e

echo "🔧 Installing NVIDIA Container Toolkit"
echo ""

# Check if nvidia-smi works
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ NVIDIA drivers not found!"
    echo "   Install NVIDIA drivers first:"
    echo "   sudo apt-get install nvidia-driver-535"
    exit 1
fi

echo "✓ NVIDIA drivers detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo ""
echo "📦 Installing NVIDIA Container Toolkit..."

# Remove old broken list if exists
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Add NVIDIA package repository GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Use generic deb repository (works for Ubuntu, Debian, Pop!_OS)
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo ""
echo "🔧 Configuring Docker to use NVIDIA runtime..."

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 Testing GPU access in Docker..."

# Test GPU access
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✓ Docker can access GPU!"
    docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
    echo ""
    echo "🚀 Ready to start multi-agent system:"
    echo "   ./start-agents.sh up"
else
    echo "❌ Docker still can't access GPU"
    echo "   Try: sudo systemctl restart docker"
    echo "   Or reboot your system"
fi
