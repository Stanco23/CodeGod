# Code-God: 100% Local Deployment

<div align="center">

**Build complete applications with NO API KEYS - Fully local with real MCP servers**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Local First](https://img.shields.io/badge/Local-First-green.svg)]()
[![MCP Integrated](https://img.shields.io/badge/MCP-Integrated-purple.svg)](https://github.com/modelcontextprotocol/servers)

</div>

---

## What's New: Local-First Architecture

### Key Features

✅ **100% Local** - No API keys required, runs entirely on your hardware
✅ **Real MCP Integration** - Uses official MCP servers from modelcontextprotocol/servers
✅ **Explicit Tool Calls** - Workers receive exact MCP tool instructions
✅ **Local Master AI** - Support for Llama 3.1 70B/405B, Qwen 2.5 72B, Mixtral 8x7B
✅ **Optional APIs** - Use Claude/GPT-4 only if you want to
✅ **Full MCP Server Database** - Filesystem, Git, GitHub, Postgres, Fetch, and more

### Architecture Changes

**Master AI**:
- Now supports local models via Ollama (Llama, Qwen, Mixtral, DeepSeek)
- API models (Claude/GPT-4) completely optional
- Automatic model selection based on availability

**Worker Agents**:
- Receive explicit MCP tool call instructions
- Direct integration with real MCP protocol
- Execute tools via stdio communication with MCP servers

**MCP Integration**:
- Official MCP servers from https://github.com/modelcontextprotocol/servers
- Automatic installation and management
- 13+ servers available: filesystem, git, github, postgres, fetch, etc.

---

## Quick Start (No API Keys)

### Prerequisites

- Docker & Docker Compose
- 48GB+ RAM (64GB recommended)
- 40GB+ VRAM for GPU OR 64GB+ RAM for CPU-only
- 100GB storage

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/code-god.git
cd code-god

# 2. Start the system (no configuration needed!)
./scripts/start-local.sh

# That's it! No API keys, no configuration required.
```

The startup script will:
- Create .env from local template
- Pull required models (llama3.1:70b, phi-3:3b)
- Clone and install MCP servers
- Start all services
- Verify everything is working

### First Project

```bash
curl -X POST http://localhost:8001/build \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a REST API for managing books with CRUD operations and SQLite database",
    "tech_stack": {
      "backend": "FastAPI",
      "database": "SQLite"
    }
  }'
```

---

## Model Selection

### Master AI Models (for orchestration and planning)

| Model | Size | VRAM | RAM (CPU) | Quality | Speed |
|-------|------|------|-----------|---------|-------|
| llama3.1:70b | 70B | 40GB | 64GB | ⭐⭐⭐⭐⭐ | Medium |
| llama3.1:405b | 405B | 200GB+ | 512GB | ⭐⭐⭐⭐⭐ | Slow |
| qwen2.5:72b | 72B | 40GB | 64GB | ⭐⭐⭐⭐⭐ | Medium |
| mixtral:8x7b | 56B | 26GB | 48GB | ⭐⭐⭐⭐ | Fast |
| deepseek-coder:33b | 33B | 20GB | 32GB | ⭐⭐⭐⭐ | Fast |

**Recommended**: `llama3.1:70b` or `qwen2.5:72b`

### Worker Models (for code generation)

| Model | Size | VRAM | RAM (CPU) | Quality | Speed |
|-------|------|------|-----------|---------|-------|
| phi-3:3b | 3B | 2GB | 4GB | ⭐⭐⭐ | Very Fast |
| llama-3.2:3b | 3B | 2GB | 4GB | ⭐⭐⭐ | Very Fast |
| qwen2.5:3b | 3B | 2GB | 4GB | ⭐⭐⭐⭐ | Fast |
| mistral:7b | 7B | 4GB | 8GB | ⭐⭐⭐⭐ | Medium |
| codellama:7b | 7B | 4GB | 8GB | ⭐⭐⭐⭐ | Medium |

**Recommended**: `phi-3:3b` (fast) or `mistral:7b` (better quality)

### Configuration

Edit `.env`:

```bash
# For high-end GPU (40GB+ VRAM)
MASTER_MODEL=llama3.1:70b
WORKER_MODEL=mistral:7b

# For mid-range GPU (24GB VRAM)
MASTER_MODEL=mixtral:8x7b
WORKER_MODEL=phi-3:3b

# For CPU-only (64GB+ RAM)
MASTER_MODEL=mixtral:8x7b
WORKER_MODEL=phi-3:3b
```

---

## MCP Server Integration

### Available MCP Servers

The system integrates with **13 official MCP servers**:

| Server | Tools | Description |
|--------|-------|-------------|
| **filesystem** | 8 tools | File operations: read, write, list, search |
| **git** | 11 tools | Git: status, commit, diff, log, merge |
| **github** | 9 tools | GitHub API: repos, issues, PRs |
| **postgres** | 7 tools | PostgreSQL operations |
| **sqlite** | 6 tools | SQLite operations |
| **fetch** | 4 tools | Web scraping and fetching |
| **brave-search** | 2 tools | Web search via Brave |
| **google-maps** | 5 tools | Location and mapping |
| **slack** | 5 tools | Slack messaging |
| **memory** | 6 tools | Knowledge graph memory |
| **puppeteer** | 6 tools | Browser automation |
| **sequential-thinking** | 4 tools | Multi-step reasoning |
| **time** | 3 tools | Time and timezone |

### How It Works

1. **Master AI** decomposes tasks and specifies which MCP tools to use
2. **Workers** receive explicit tool call instructions in their prompts
3. **MCP Servers** execute tools via stdio communication
4. **Results** are integrated back into the workflow

### Example MCP Tool Call

Worker receives:

```json
{
  "task": "Create user.py file",
  "mcp_tools": [
    {
      "server": "filesystem",
      "tool": "write_file",
      "arguments": {
        "path": "backend/models/user.py",
        "content": "... generated code ..."
      },
      "when": "after generating the code"
    },
    {
      "server": "git",
      "tool": "git_add",
      "arguments": {
        "paths": ["backend/models/user.py"]
      },
      "when": "after writing file"
    }
  ]
}
```

Worker generates code and includes tool calls in response:

```json
{
  "success": true,
  "code": "class User: ...",
  "tool_calls": [
    {
      "server": "filesystem",
      "tool": "write_file",
      "arguments": {"path": "backend/models/user.py", "content": "class User: ..."}
    },
    {
      "server": "git",
      "tool": "git_add",
      "arguments": {"paths": ["backend/models/user.py"]}
    }
  ]
}
```

System executes tools automatically and returns results.

---

## Hardware Requirements

### Minimum Configuration

**GPU**:
- NVIDIA GPU with 40GB VRAM
- 48GB System RAM
- 8 CPU cores
- 100GB SSD storage

**CPU-Only**:
- 64GB System RAM
- 16 CPU cores
- 200GB SSD storage
- Note: 5-10x slower than GPU

### Recommended Configuration

**GPU**:
- NVIDIA GPU with 48GB+ VRAM (e.g., A6000, A100)
- 64GB System RAM
- 16 CPU cores
- 200GB NVMe storage

**Multi-GPU**:
- 2x NVIDIA GPUs with 24GB each
- 128GB System RAM
- 32 CPU cores
- 500GB NVMe storage

### Budget Options

**Cloud GPU** (recommended for trying out):
- RunPod: ~$0.50-1.00/hour for A6000
- Vast.ai: ~$0.30-0.80/hour for various GPUs
- Lambda Labs: ~$1.10/hour for A100

**Local Budget Build**:
- Used RTX 3090 (24GB): $800-1200
- 64GB RAM: $150-250
- Ryzen 9 or i9: $300-500
- Total: ~$1500-2000

---

## API Key Optional Mode

You can optionally use API models for better quality:

```bash
# .env
PREFER_LOCAL=false  # Prefer API over local
MASTER_MODEL=claude-sonnet-4.5
ANTHROPIC_API_KEY=sk-ant-...
```

This gives you the best of both worlds:
- Local workers (cheap, private)
- API master (higher quality planning)

---

## Performance Benchmarks

### Build Times (Local vs API)

| Project Complexity | Local (70B) | API (Claude) | Local Cost | API Cost |
|-------------------|-------------|--------------|------------|----------|
| Simple API | 8-12 min | 5-10 min | $0 | $0.50 |
| Medium Web App | 25-40 min | 15-30 min | $0 | $2.00 |
| Full-Stack Platform | 50-80 min | 30-60 min | $0 | $5.00 |

### Resource Usage (Local)

| Model | VRAM/RAM | Tokens/sec | Power |
|-------|----------|------------|-------|
| llama3.1:70b | 40GB | 15-25 | 300W |
| mixtral:8x7b | 26GB | 25-40 | 200W |
| phi-3:3b | 2GB | 100-150 | 50W |

---

## MCP Server Management

### View Available Servers

```bash
curl http://localhost:8001/mcp/servers
```

### Install Additional Server

Servers are auto-installed when needed, but you can manually install:

```bash
# SSH into master container
docker-compose -f docker-compose.local.yml exec master-ai-local bash

# Install server
cd /app/mcp_servers/servers/src/SERVER_NAME
npm install && npm run build
```

### Configure Server with API Keys

Some servers need API keys:

```bash
# .env
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
BRAVE_API_KEY=BSA...
GOOGLE_MAPS_API_KEY=AIza...
```

Restart services:

```bash
docker-compose -f docker-compose.local.yml restart master-ai-local
```

---

## Troubleshooting

### Out of Memory

```bash
# Use smaller models
MASTER_MODEL=mixtral:8x7b  # Instead of llama3.1:70b
WORKER_MODEL=phi-3:3b      # Instead of mistral:7b

# Reduce parallelism
MAX_PARALLEL_TASKS=2  # Instead of 5
```

### Slow Performance (CPU)

```bash
# Increase RAM if available
# Use swap space
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Use smaller, faster models
MASTER_MODEL=mixtral:8x7b
```

### MCP Server Issues

```bash
# Check server status
curl http://localhost:8001/mcp/servers

# View server logs
docker-compose -f docker-compose.local.yml logs master-ai-local | grep MCP

# Reinstall servers
rm -rf ./mcp_servers/servers
./scripts/start-local.sh
```

### Model Download Issues

```bash
# Pull models manually
ollama pull llama3.1:70b
ollama pull phi-3:3b

# Check available space
df -h

# Clean up old models
ollama rm OLD_MODEL_NAME
```

---

## Cost Comparison

### Local vs API (Monthly)

**100% Local**:
- Hardware: $1500-2000 one-time OR $50-100/month cloud
- Electricity: ~$30-50/month (24/7 operation)
- **Total monthly**: $50-150 (cloud) OR $30-50 (owned hardware)

**API Only**:
- API calls: $200-1000/month (depending on usage)
- No hardware needed
- **Total monthly**: $200-1000

**Break-even**: 2-4 months if buying hardware, immediate if using cloud

---

## What's Different from Standard Deployment

| Feature | Standard | Local-First |
|---------|----------|-------------|
| Master AI | API-only (Claude/GPT-4) | Local models + optional API |
| Worker Models | 3B local | Same |
| MCP Servers | Custom implementations | Official MCP servers |
| Tool Integration | Simulated | Real MCP protocol |
| API Keys | Required | Optional |
| Cost | API fees | Hardware/electricity |
| Privacy | Data sent to APIs | 100% local |
| Internet | Required | Optional (after setup) |

---

## Next Steps

1. **Try it**: Run `./scripts/start-local.sh`
2. **Build something**: Start with a simple API
3. **Monitor**: Watch resource usage
4. **Optimize**: Adjust models based on your hardware
5. **Scale**: Add more workers or better models

---

## Contributing

The MCP server integration is extensible. To add a new MCP server:

1. Add server spec to `mcp_server_registry.py`
2. Update Docker volumes to mount MCP servers directory
3. Add installation commands to startup script
4. Update documentation

---

## Support

- **Local Deployment Issues**: [GitHub Issues](https://github.com/yourusername/code-god/issues)
- **MCP Server Issues**: [MCP Servers Repo](https://github.com/modelcontextprotocol/servers/issues)
- **Model Issues**: [Ollama Repo](https://github.com/ollama/ollama/issues)

---

**Made with ⚡ and 🤖 - 100% local, 0% API lock-in**
