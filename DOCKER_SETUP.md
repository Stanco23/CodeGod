# Docker Setup for Multi-Agent System

## Fix Docker Permissions

### Option 1: Add User to Docker Group (Recommended)

```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Apply group changes (logout/login or use newgrp)
newgrp docker

# Verify it works
docker ps
```

**Note:** You may need to log out and log back in for group changes to take effect.

### Option 2: Use sudo (Quick test only)

```bash
sudo ./start-agents.sh up
```

**Warning:** This is not recommended for production use.

## Install Python Dependencies

### Option 1: Use Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Now run Code-God
python codegod.py --use-agents
```

### Option 2: System-wide Install

```bash
pip install -r requirements.txt
# Or
pip3 install -r requirements.txt
```

## Quick Start After Setup

```bash
# 1. Start Docker services
./start-agents.sh up

# 2. In another terminal, run Code-God
source venv/bin/activate  # If using venv
python codegod.py --use-agents --build "Create a Flask REST API"
```

## Verify Setup

```bash
# Check Docker access
docker ps

# Check Python dependencies
python -c "import rich; print('✓ Rich installed')"
python -c "import redis; print('✓ Redis installed')"
python -c "import openai; print('✓ OpenAI installed')"
```

## Install NVIDIA Container Toolkit (Required for GPU)

The multi-agent system needs GPU access through Docker.

### Automated Install

```bash
./install-nvidia-docker.sh
```

### Manual Install

```bash
# 1. Remove old broken list if exists
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Add NVIDIA package repository GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 3. Add generic deb repository (works for Ubuntu, Debian, Pop!_OS)
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 4. Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 5. Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 6. Test it works
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## Troubleshooting

### "could not select device driver nvidia with capabilities: [[gpu]]"
This means NVIDIA Container Toolkit is not installed. Run:
```bash
./install-nvidia-docker.sh
```

### "Cannot connect to Docker daemon"
- Docker service not running: `sudo systemctl start docker`
- Docker not installed: `sudo apt-get install docker.io docker-compose`

### "docker-compose: command not found"
```bash
# Install docker-compose
sudo apt-get install docker-compose
# Or use docker compose (v2)
docker compose version
```

### GPU not available for vLLM
vLLM requires NVIDIA GPU. If you don't have one:
- Use single-agent mode: `python codegod.py --build "..."`
- Or modify docker-compose.agents.yml to use CPU-only models (much slower)
