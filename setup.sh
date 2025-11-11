#!/bin/bash
# Quick setup script for Code-God

set -e

echo "🔧 Code-God Setup Script"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.11"

echo "📋 Checking requirements..."
echo "   Python version: $(python3 --version)"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment exists"
fi

# Activate venv
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

echo "   ✓ Dependencies installed"

# Check Docker access
echo ""
echo "🐳 Checking Docker access..."
if docker ps > /dev/null 2>&1; then
    echo "   ✓ Docker access OK"
    DOCKER_OK=true
else
    echo "   ✗ Docker access denied"
    echo ""
    echo "   To fix Docker permissions:"
    echo "   sudo usermod -aG docker $USER"
    echo "   newgrp docker"
    echo ""
    echo "   Or see DOCKER_SETUP.md for details"
    DOCKER_OK=false
fi

# Check GPU
echo ""
echo "🎮 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        echo "   ✓ NVIDIA GPU detected"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs echo "   GPU:"
    else
        echo "   ⚠️  nvidia-smi found but failed to run"
    fi
else
    echo "   ⚠️  No NVIDIA GPU detected"
    echo "   Multi-agent mode requires GPU. Use single-agent mode instead."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo ""

if [ "$DOCKER_OK" = true ]; then
    echo "   For MULTI-AGENT mode (with Docker):"
    echo "   1. ./start-agents.sh up"
    echo "   2. source venv/bin/activate"
    echo "   3. python codegod.py --use-agents --build \"Your project\""
    echo ""
fi

echo "   For SINGLE-AGENT mode (no Docker needed):"
echo "   1. source venv/bin/activate"
echo "   2. python codegod.py --build \"Your project\""
echo ""
echo "   Interactive mode:"
echo "   1. source venv/bin/activate"
echo "   2. ./codegod"
echo ""

# Create activation helper
cat > activate.sh << 'EOF'
#!/bin/bash
# Helper script to activate venv
source venv/bin/activate
echo "✓ Virtual environment activated"
echo "Run: python codegod.py or ./codegod"
EOF
chmod +x activate.sh

echo "💡 Tip: Run 'source activate.sh' to activate venv quickly"
echo ""
