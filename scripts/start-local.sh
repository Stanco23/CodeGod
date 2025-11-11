#!/bin/bash

# Code-God Local Deployment Startup Script
# Starts all services with LOCAL MODELS ONLY - No API keys required

set -e

echo "======================================"
echo "  Code-God LOCAL Deployment"
echo "  100% Local - No API Keys Needed"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠ .env file not found. Creating from local template...${NC}"
    if [ -f .env.local.example ]; then
        cp .env.local.example .env
        echo -e "${GREEN}✓ Created .env from .env.local.example${NC}"
        echo -e "${BLUE}ℹ No API keys required for local deployment!${NC}"
    else
        echo -e "${RED}✗ .env.local.example not found${NC}"
        exit 1
    fi
fi

# Load environment variables
source .env

echo -e "${BLUE}ℹ Configuration loaded${NC}"
echo -e "${BLUE}  Master Model: ${MASTER_MODEL:-llama3.1:70b}${NC}"
echo -e "${BLUE}  Worker Model: ${WORKER_MODEL:-phi-3:3b}${NC}"
echo -e "${BLUE}  Mode: ${GREEN}100% LOCAL${NC}"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    echo "  Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ Docker Compose is not installed${NC}"
    echo "  Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker and Docker Compose found${NC}"

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠ Ollama not found locally${NC}"
    echo "  Ollama will be installed in containers"
    echo "  To use local models, install Ollama: https://ollama.ai/"
else
    echo -e "${GREEN}✓ Ollama found${NC}"

    # Check if models are pulled
    echo ""
    echo "Checking for required models..."

    MASTER_MODEL_NAME=${MASTER_MODEL:-llama3.1:70b}
    WORKER_MODEL_NAME=${WORKER_MODEL:-phi-3:3b}

    if [[ "$MASTER_MODEL_NAME" != "claude-"* ]] && [[ "$MASTER_MODEL_NAME" != "gpt-"* ]]; then
        if ollama list | grep -q "$MASTER_MODEL_NAME"; then
            echo -e "${GREEN}✓ Master model $MASTER_MODEL_NAME available${NC}"
        else
            echo -e "${YELLOW}⚠ Master model $MASTER_MODEL_NAME not found${NC}"
            echo "  Pulling $MASTER_MODEL_NAME (this will take some time)..."
            ollama pull "$MASTER_MODEL_NAME"
        fi
    fi

    if ollama list | grep -q "$WORKER_MODEL_NAME"; then
        echo -e "${GREEN}✓ Worker model $WORKER_MODEL_NAME available${NC}"
    else
        echo -e "${YELLOW}⚠ Worker model $WORKER_MODEL_NAME not found${NC}"
        echo "  Pulling $WORKER_MODEL_NAME..."
        ollama pull "$WORKER_MODEL_NAME"
    fi
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p projects logs config sandbox/{backend,frontend,testing,devops} mcp_servers

echo -e "${GREEN}✓ Directories created${NC}"

# Check system resources
echo ""
echo "Checking system resources..."

TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
echo "  Total RAM: ${TOTAL_RAM}GB"

if [ "$TOTAL_RAM" -lt 32 ]; then
    echo -e "${YELLOW}⚠ Warning: Less than 32GB RAM detected${NC}"
    echo "  Recommended: 48GB+ for Master AI with 70B model"
    echo "  Consider using a smaller model like mixtral:8x7b"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    GPU_MEM_GB=$((GPU_MEM / 1024))
    echo -e "${GREEN}✓ GPU detected: ${GPU_MEM_GB}GB VRAM${NC}"

    if [ "$GPU_MEM_GB" -lt 24 ]; then
        echo -e "${YELLOW}⚠ Warning: Less than 24GB VRAM${NC}"
        echo "  70B models need ~40GB VRAM"
        echo "  Consider using CPU or a smaller model"
    fi
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected - using CPU${NC}"
    echo "  This will be slower. Consider using a cloud GPU provider."
fi

# Build images
echo ""
echo "Building Docker images (this may take a few minutes)..."
docker-compose -f docker-compose.local.yml build

echo -e "${GREEN}✓ Images built${NC}"

# Start infrastructure services first
echo ""
echo "Starting infrastructure services..."
docker-compose -f docker-compose.local.yml up -d chromadb redis postgres

echo "Waiting for services to be ready..."
sleep 10

# Check ChromaDB health
echo -n "Checking ChromaDB... "
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ ChromaDB failed to start${NC}"
        docker-compose -f docker-compose.local.yml logs chromadb
        exit 1
    fi
done

# Check Redis health
echo -n "Checking Redis... "
for i in {1..30}; do
    if docker-compose -f docker-compose.local.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Redis failed to start${NC}"
        exit 1
    fi
done

# Start Master AI
echo ""
echo "Starting Master AI Orchestrator (with local model support)..."
echo "This may take a moment while models are loaded..."
docker-compose -f docker-compose.local.yml up -d master-ai-local

echo "Waiting for Master AI to be ready..."
sleep 30

# Check Master AI health
echo -n "Checking Master AI... "
for i in {1..120}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 120 ]; then
        echo -e "${RED}✗ Master AI failed to start${NC}"
        echo "Check logs with: docker-compose -f docker-compose.local.yml logs master-ai-local"
        exit 1
    fi
done

# Get model info
echo ""
echo "Checking Master AI model..."
MODEL_INFO=$(curl -s http://localhost:8001/status | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data.get('model', 'unknown')} ({data.get('backend', 'unknown')})\")" 2>/dev/null || echo "unknown")
echo -e "  ${GREEN}Model: $MODEL_INFO${NC}"

# Clone MCP servers repository
echo ""
echo "Setting up MCP servers..."
if [ ! -d "./mcp_servers/servers" ]; then
    echo "Cloning MCP servers repository..."
    git clone https://github.com/modelcontextprotocol/servers.git ./mcp_servers/servers
    echo -e "${GREEN}✓ MCP servers cloned${NC}"
else
    echo -e "${GREEN}✓ MCP servers already cloned${NC}"
fi

# Install essential MCP servers
echo "Installing essential MCP servers..."
cd mcp_servers/servers

# Install filesystem server
if [ -d "src/filesystem" ]; then
    echo "  Installing filesystem MCP server..."
    cd src/filesystem && npm install && npm run build && cd ../..
fi

# Install git server
if [ -d "src/git" ]; then
    echo "  Installing git MCP server..."
    cd src/git && npm install && npm run build && cd ../..
fi

# Install fetch server
if [ -d "src/fetch" ]; then
    echo "  Installing fetch MCP server..."
    cd src/fetch && npm install && npm run build && cd ../..
fi

cd ../..
echo -e "${GREEN}✓ Essential MCP servers installed${NC}"

# Start Worker Agents
echo ""
echo "Starting Worker Agents..."
docker-compose -f docker-compose.local.yml up -d \
    worker-backend-mcp \
    worker-frontend-mcp \
    worker-testing-mcp \
    worker-devops-mcp

echo "Waiting for workers to initialize..."
sleep 20

echo ""
echo -e "${GREEN}✓ All services started successfully!${NC}"
echo ""
echo "======================================"
echo "  System Status"
echo "======================================"
docker-compose -f docker-compose.local.yml ps

echo ""
echo "======================================"
echo "  Access Points"
echo "======================================"
echo "  Master AI API:  http://localhost:8001"
echo "  API Docs:       http://localhost:8001/docs"
echo "  ChromaDB:       http://localhost:8000"
echo "  MCP Servers:    http://localhost:8001/mcp/servers"
echo ""
echo "======================================"
echo "  Quick Start"
echo "======================================"
echo ""
echo "Create a project:"
echo '  curl -X POST http://localhost:8001/build \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"prompt": "Create a REST API for task management", "tech_stack": {"backend": "FastAPI"}}'"'"
echo ""
echo "Check status:"
echo "  curl http://localhost:8001/status"
echo ""
echo "View MCP servers:"
echo "  curl http://localhost:8001/mcp/servers"
echo ""
echo "View logs:"
echo "  docker-compose -f docker-compose.local.yml logs -f master-ai-local"
echo "  docker-compose -f docker-compose.local.yml logs -f worker-backend-mcp"
echo ""
echo -e "${GREEN}System ready - 100% local, no API keys needed! 🚀${NC}"
echo ""
