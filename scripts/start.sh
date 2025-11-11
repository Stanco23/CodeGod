#!/bin/bash

# Code-God Startup Script
# Starts all services for the autonomous multi-agent AI system

set -e

echo "======================================"
echo "  Code-God Autonomous AI System"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠ .env file not found. Creating from template...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}⚠ Please edit .env and add your API keys before continuing${NC}"
        exit 1
    else
        echo -e "${RED}✗ .env.example not found. Creating minimal .env...${NC}"
        cat > .env << EOF
# API Keys (REQUIRED)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Master AI Configuration
MASTER_MODEL=claude-sonnet-4.5
MAX_PARALLEL_TASKS=5

# Worker Configuration
WORKER_MODEL=phi-3:3b

# Database
POSTGRES_PASSWORD=changeme

# Monitoring (optional)
GRAFANA_PASSWORD=admin
EOF
        echo -e "${YELLOW}⚠ Please edit .env and add your API keys before continuing${NC}"
        exit 1
    fi
fi

# Load environment variables
source .env

# Check for required API keys
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}✗ Error: No API keys found in .env${NC}"
    echo "  Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY"
    exit 1
fi

echo -e "${GREEN}✓ Configuration loaded${NC}"

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

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p projects logs config sandbox/{backend,frontend,testing,devops}

echo -e "${GREEN}✓ Directories created${NC}"

# Pull/Build images
echo ""
echo "Building Docker images (this may take a few minutes)..."
docker-compose build

echo -e "${GREEN}✓ Images built${NC}"

# Start infrastructure services first
echo ""
echo "Starting infrastructure services..."
docker-compose up -d chromadb redis postgres

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
        docker-compose logs chromadb
        exit 1
    fi
done

# Check Redis health
echo -n "Checking Redis... "
for i in {1..30}; do
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
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
echo "Starting Master AI Orchestrator..."
docker-compose up -d master-ai

echo "Waiting for Master AI to be ready..."
sleep 15

# Check Master AI health
echo -n "Checking Master AI... "
for i in {1..60}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 60 ]; then
        echo -e "${RED}✗ Master AI failed to start${NC}"
        docker-compose logs master-ai
        exit 1
    fi
done

# Start Worker Agents
echo ""
echo "Starting Worker Agents..."
docker-compose up -d worker-backend worker-frontend worker-testing worker-devops

echo "Waiting for workers to initialize (pulling models if needed)..."
sleep 20

echo ""
echo -e "${GREEN}✓ All services started successfully!${NC}"
echo ""
echo "======================================"
echo "  System Status"
echo "======================================"
docker-compose ps

echo ""
echo "======================================"
echo "  Access Points"
echo "======================================"
echo "  Master AI API:  http://localhost:8001"
echo "  API Docs:       http://localhost:8001/docs"
echo "  ChromaDB:       http://localhost:8000"
echo ""
echo "Optional services (start with --profile):"
echo "  Grafana:        http://localhost:3000  (docker-compose --profile monitoring up -d)"
echo "  Web UI:         http://localhost:3001  (docker-compose --profile ui up -d)"
echo ""
echo "======================================"
echo "  Quick Start"
echo "======================================"
echo ""
echo "Create a project:"
echo '  curl -X POST http://localhost:8001/build \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"prompt": "Create a REST API for a todo app", "tech_stack": {"backend": "FastAPI"}}'"'"
echo ""
echo "Check status:"
echo "  curl http://localhost:8001/status"
echo ""
echo "View logs:"
echo "  docker-compose logs -f master-ai"
echo "  docker-compose logs -f worker-backend"
echo ""
echo -e "${GREEN}System ready! 🚀${NC}"
echo ""
