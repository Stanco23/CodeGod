#!/bin/bash
# Start multi-agent system with vLLM and Redis
# Usage: ./start-agents.sh [up|down|logs|status|wipe]

set -e

COMPOSE_FILE="docker-compose.agents.yml"
PROJECT_NAME="codegod-agents"

case "${1:-up}" in
    up)
        echo "🚀 Starting Code-God Multi-Agent System..."

        # Check Docker access
        if ! docker ps &> /dev/null; then
            echo ""
            echo "❌ Cannot connect to Docker daemon"
            echo ""
            echo "Possible causes:"
            echo "  1. Docker is not running"
            echo "     Fix: sudo systemctl start docker"
            echo ""
            echo "  2. Permission denied (most common)"
            echo "     Fix: sudo usermod -aG docker $USER"
            echo "          newgrp docker"
            echo ""
            echo "  3. Docker not installed"
            echo "     Fix: sudo apt-get install docker.io docker-compose"
            echo ""
            echo "See DOCKER_SETUP.md for detailed instructions"
            echo ""
            echo "Quick workaround: sudo ./start-agents.sh up"
            exit 1
        fi

        # Check for GPU
        if ! command -v nvidia-smi &> /dev/null; then
            echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
            echo "   vLLM requires NVIDIA GPU with CUDA support."
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi

        # Check for NVIDIA Container Toolkit
        echo ""
        echo "🎮 Checking NVIDIA Docker support..."
        if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo "   ✓ Docker can access GPU"
        else
            echo ""
            echo "❌ Docker cannot access GPU!"
            echo ""
            echo "NVIDIA Container Toolkit is not installed or configured."
            echo ""
            echo "Fix this by running:"
            echo "   ./install-nvidia-docker.sh"
            echo ""
            echo "Or install manually:"
            echo "   1. Install NVIDIA Container Toolkit"
            echo "   2. Configure Docker: sudo nvidia-ctk runtime configure --runtime=docker"
            echo "   3. Restart Docker: sudo systemctl restart docker"
            echo ""
            exit 1
        fi

        # Create required directories
        mkdir -p ./models ./projects ./logs

        echo "📦 Pulling Docker images..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME pull

        echo "🔨 Building agent containers..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME build

        echo "▶️  Starting services..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d

        echo ""
        echo "✅ Multi-Agent System started!"
        echo ""
        echo "📊 Monitoring URLs:"
        echo "   • Redis Commander: http://localhost:8081"
        echo "   • vLLM API: http://localhost:8000"
        echo ""
        echo "📝 Useful commands:"
        echo "   • View logs: ./start-agents.sh logs"
        echo "   • Check status: ./start-agents.sh status"
        echo "   • Wipe memories: ./start-agents.sh wipe"
        echo "   • Stop system: ./start-agents.sh down"
        echo ""
        ;;

    down)
        echo "🛑 Stopping Multi-Agent System..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down
        echo "✅ System stopped"
        ;;

    restart)
        echo "🔄 Restarting Multi-Agent System..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME restart
        echo "✅ System restarted"
        ;;

    logs)
        echo "📋 Showing logs (Ctrl+C to exit)..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f "${@:2}"
        ;;

    status)
        echo "📊 System Status:"
        echo ""
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
        echo ""

        echo "💾 Redis Connection:"
        if docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T redis redis-cli ping &> /dev/null; then
            echo "   ✅ Redis: Connected"

            # Show task queue stats
            TASK_COUNT=$(docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T redis redis-cli LLEN codegod:tasks:backend 2>/dev/null || echo "0")
            echo "   📋 Backend queue: $TASK_COUNT tasks"
        else
            echo "   ❌ Redis: Not responding"
        fi

        echo ""
        echo "🤖 Active Workers:"
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T redis redis-cli KEYS "codegod:workers:*" 2>/dev/null | wc -l | xargs echo "   Workers registered:"
        echo ""
        ;;

    wipe)
        echo "🧹 Wiping all agent memories..."

        # Clear task queues
        for role in backend frontend devops testing debugging; do
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T redis redis-cli DEL "codegod:tasks:$role" &> /dev/null || true
        done

        # Clear worker registrations
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T redis redis-cli --scan --pattern "codegod:workers:*" | xargs docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T redis redis-cli DEL &> /dev/null || true

        # Clear task data
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T redis redis-cli --scan --pattern "codegod:task:*" | xargs docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T redis redis-cli DEL &> /dev/null || true

        echo "✅ All agent memories wiped"
        echo "   Note: Restart agents to reset their internal state:"
        echo "   ./start-agents.sh restart"
        ;;

    scale)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: ./start-agents.sh scale <agent-type> <count>"
            echo ""
            echo "Available agent types:"
            echo "  • backend-agent"
            echo "  • frontend-agent"
            echo "  • testing-agent"
            echo ""
            echo "Example: ./start-agents.sh scale backend-agent 4"
            exit 1
        fi

        echo "⚖️  Scaling $2 to $3 replicas..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d --scale "$2=$3" --no-recreate
        echo "✅ Scaled $2 to $3 replicas"
        ;;

    exec)
        if [ -z "$2" ]; then
            echo "Usage: ./start-agents.sh exec <service> [command]"
            echo ""
            echo "Available services:"
            echo "  • master-orchestrator"
            echo "  • backend-agent"
            echo "  • frontend-agent"
            echo "  • vllm-server"
            echo "  • redis"
            echo ""
            echo "Example: ./start-agents.sh exec master-orchestrator bash"
            exit 1
        fi

        SERVICE="$2"
        COMMAND="${@:3}"
        if [ -z "$COMMAND" ]; then
            COMMAND="bash"
        fi

        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec "$SERVICE" $COMMAND
        ;;

    pull-model)
        MODEL="${2:-meta-llama/Llama-2-70b-chat-hf}"
        echo "📥 Pulling model: $MODEL"
        echo "   This may take a while (models are 40-140GB)..."

        docker run --rm -v ./models:/root/.cache/huggingface \
            huggingface/transformers-pytorch-gpu \
            python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('$MODEL'); AutoTokenizer.from_pretrained('$MODEL')"

        echo "✅ Model downloaded to ./models/"
        ;;

    *)
        echo "Code-God Multi-Agent System Manager"
        echo ""
        echo "Usage: ./start-agents.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  up              Start all services (default)"
        echo "  down            Stop all services"
        echo "  restart         Restart all services"
        echo "  logs [service]  Show logs (optionally for specific service)"
        echo "  status          Show system status"
        echo "  wipe            Wipe all agent memories and task queues"
        echo "  scale <agent> <count>  Scale agent replicas"
        echo "  exec <service> [cmd]   Execute command in container"
        echo "  pull-model [model]     Download LLM model"
        echo ""
        echo "Examples:"
        echo "  ./start-agents.sh up              # Start system"
        echo "  ./start-agents.sh logs backend    # View backend agent logs"
        echo "  ./start-agents.sh scale backend-agent 4  # Run 4 backend agents"
        echo "  ./start-agents.sh wipe            # Clear all memories"
        echo ""
        ;;
esac
