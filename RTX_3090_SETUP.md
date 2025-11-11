# RTX 3090 (24GB) Setup

The multi-agent system is optimized for a single RTX 3090 with 24GB VRAM.

## Configuration

**Model:** Llama 3.1 8B Instruct
**VRAM Usage:** ~16GB (leaves room for agent workers)
**Agents:** 1 of each type (Backend, Frontend, DevOps, Testing)
**Parallelism:** Sequential agent execution, parallel task reasoning

## Quick Start

```bash
# 1. Fix Docker permissions (if needed)
sudo usermod -aG docker $USER
newgrp docker

# 2. Setup Python environment
./setup.sh

# 3. Start multi-agent system
./start-agents.sh up

# Wait ~2 minutes for model download and vLLM startup

# 4. Run Code-God with agents
source venv/bin/activate
python codegod.py --use-agents --build "Create a Flask REST API"
```

## Memory Breakdown (24GB)

- **vLLM Model (Llama 3.1 8B):** ~16GB
- **Agent workers (5 containers):** ~2GB total
- **Redis + overhead:** ~1GB
- **Free buffer:** ~5GB

## Performance

- **Inference speed:** ~50-80 tokens/sec (vLLM optimized)
- **vs 70B models:** 3-4x faster inference, slightly lower quality
- **vs single-agent:** True reasoning at every step, parallelizable tasks

## Alternative Models for 3090

Edit `docker-compose.agents.yml` line 15:

```yaml
# Faster, good for code (recommended)
--model Qwen/Qwen2.5-7B-Instruct

# Balanced speed/quality
--model mistralai/Mistral-7B-Instruct-v0.3

# Google's model
--model google/gemma-2-9b-it
```

## Scaling

Since all agents share one vLLM server, you can scale workers without GPU memory issues:

```bash
# More backend workers for heavy Python projects
./start-agents.sh scale backend-agent 2

# More testing workers for test-heavy projects
./start-agents.sh scale testing-agent 2
```

Workers queue requests to vLLM, so scaling doesn't increase VRAM usage.

## Troubleshooting

### vLLM won't start / CUDA out of memory

```bash
# Reduce max context length in docker-compose.agents.yml
--max-model-len 4096  # Instead of 8192
```

### Agents timeout waiting for vLLM

```bash
# Check vLLM logs
./start-agents.sh logs vllm-server

# Model still downloading?
docker exec codegod-vllm ls -lh /root/.cache/huggingface/hub/
```

### Want even smaller model

```yaml
# Use Phi-3 Mini (3.8B) - fits easily but lower quality
--model microsoft/Phi-3-mini-128k-instruct
```

## Monitor GPU Usage

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Should see ~16GB used by vllm-server
```
