# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Code-God is an autonomous AI development assistant that builds complete applications from natural language descriptions. It features:
- Terminal-based interactive CLI (`codegod.py`)
- Multi-agent system with Master AI orchestrator and specialized Worker Agents
- Local model support (Llama, Qwen, Mistral) via Ollama
- Optional API model support (Claude, GPT-4)
- MCP (Model Context Protocol) server discovery and integration
- Autonomous project building with sandboxed execution

## Running Code-God

### Quick Start (Recommended)
```bash
./codegod              # Linux/macOS - auto-installs dependencies
codegod.bat            # Windows - auto-installs dependencies
```

### Direct Python Execution
```bash
# From virtual environment
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
python codegod.py

# With arguments
python codegod.py --model llama3.1:70b --prefer-local
python codegod.py --build "Create a REST API for user management"
```

### Development Mode
```bash
# Install dependencies manually
pip install -r requirements.txt

# Run with debug logging
LOG_LEVEL=DEBUG python codegod.py
```

## Docker Deployment

### API-Based Models (Claude/GPT-4)
```bash
# Configure API keys in .env
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY or OPENAI_API_KEY

# Start services
./scripts/start.sh         # Interactive
docker-compose up -d       # Daemon mode
docker-compose logs -f     # View logs
docker-compose down        # Stop services
```

### Local Models (Ollama)
```bash
# Configure for local models
cp .env.local.example .env

# Start services with local model support
./scripts/start-local.sh
docker-compose -f docker-compose.local.yml up -d
```

## Architecture

### Core Components

**Entry Point:**
- `codegod.py` - Main CLI application with Rich terminal UI
- `codegod` / `codegod.bat` - Platform-specific launcher scripts

**Master Orchestrators:**
- `master_orchestrator.py` - API-based master AI (Claude/GPT-4)
- `master_orchestrator_local.py` - Local model-based master AI (Llama/Qwen/Mixtral)
- Responsibilities: Project analysis, task decomposition, worker coordination, output integration

**Worker Agents:**
- `worker_agent.py` - Small model workers for API deployment (3B-7B models)
- `worker_agent_mcp.py` - Workers with enhanced MCP tool integration
- Specializations: Backend, Frontend, Testing, DevOps, Documentation

**Core Systems:**
- `memory_system.py` - Vector database (ChromaDB) for RAG and context management
- `task_orchestration.py` - Task queue, dependency resolution, scheduling
- `validation_system.py` - Syntax checking, testing, code quality analysis

**MCP Integration:**
- `mcp_discovery.py` - Auto-discovers MCP servers from 1mcpserver.com API
- `mcp_integration.py` - Tool allocation and execution for workers
- `mcp_server_registry.py` - Server installation and lifecycle management

**User-Facing:**
- `conversation_manager.py` - Chat history and context management
- `project_builder.py` - Autonomous project generation from descriptions
- `local_model_executor.py` - Unified interface for Ollama local models

**Prompts:**
- `prompt_templates.py` - System prompts for Master AI and specialized Workers

### Multi-Agent Flow

1. **User Input** → Terminal CLI or non-interactive mode
2. **Master AI** analyzes prompt, decomposes into atomic tasks, builds dependency graph
3. **Task Queue** schedules tasks based on priorities and dependencies
4. **Worker Agents** execute tasks in parallel with MCP tools (filesystem, git, database)
5. **Validation** checks syntax, runs tests, analyzes code quality
6. **Integration** merges outputs, resolves conflicts
7. **DevOps** handles git operations, builds, deployment prep
8. **Output** complete project in `projects/` directory

### Memory System (RAG)

- **Vector DB:** ChromaDB stores API docs, code modules, task metadata
- **Embedding:** all-MiniLM-L6-v2 for semantic search
- **Retrieval:** Workers get relevant context before task execution
- **Storage:** `prepare_worker_context()` fetches from memory for each task

### MCP Tools Available

Workers can use:
- **filesystem:** Read/write files, execute code
- **git:** Version control operations
- **memory:** Store/retrieve from vector DB
- **fetch:** Web scraping, documentation retrieval
- **sequential_thinking:** Multi-step reasoning
- **database:** PostgreSQL/SQLite operations
- **github:** Repository operations (if installed)
- **postgres, sqlite, puppeteer, etc.** (60+ tools via `/install`)

## Testing

No formal test suite is currently present. Manual testing approach:
```bash
# Test terminal CLI with new features
./codegod
> /help                    # Show all commands
> /search database         # Search for database-related MCP servers
> /categories              # Show all MCP categories
> /mcp                     # Should show 50+ servers

# Test arrow key history
./codegod
> /search git              # Type a command
> [Press ↑]                # Should recall previous command
> [Press ↓]                # Navigate through history

# Test shell mode toggle
./codegod
> /shell                   # Enter shell mode
$ ls -la                   # Execute shell commands
$ git status              # Any shell command works
$ exit                    # Return to AI mode
> /help                    # Back in AI mode

# Test smart directory handling
./codegod
> /build Create a simple API --dir ~/test-exact
# Should create project in ~/test-exact (no subdirectory)

> /build Create another API
# Should create in ./projects/create_another_api_20250111_123456/

# Test project building
./codegod --build "Create a simple FastAPI hello world"
# Or interactive:
./codegod
> /build Create a todo app --dir ~/my-projects/todo-api

# Test MCP search and installation
./codegod
> /search stripe           # Find payment-related servers
> /search web              # Find web scraping servers
> /install github          # Install GitHub MCP server

# Test with verbose logging
LOG_LEVEL=DEBUG ./codegod
> /install filesystem      # Shows detailed installation steps
> /search postgresql       # Shows search algorithm in action
```

## Development Patterns

### Adding New Commands
Edit `codegod.py`:
```python
async def _cmd_yourcommand(self, args: str):
    """Your command handler"""
    pass

# Register in _handle_command():
elif command == "yourcommand":
    await self._cmd_yourcommand(args)
```

### Adding Worker Specializations
1. Add TaskType to `task_orchestration.py`:
   ```python
   class TaskType(Enum):
       YOUR_TYPE = "your_type"
   ```

2. Add system prompt to `prompt_templates.py`:
   ```python
   YOUR_WORKER_SYSTEM_PROMPT = """..."""
   ```

3. Update worker agent to handle new type

### Adding MCP Tools
MCP servers are discovered automatically from 1mcpserver.com. To manually add:
1. Update `mcp_server_registry.py` with server definition
2. Add tool allocation logic in `mcp_integration.py`
3. Workers automatically receive tools based on task type

### Model Configuration

**Local Models (via Ollama):**
- Master: `llama3.1:70b`, `qwen2.5:72b`, `llama3.1:405b`
- Workers: `phi-3:3b`, `llama3.2:3b`, `mistral:7b`
- Config: `~/.codegod/config.json` or env vars

**API Models:**
- Master: `claude-sonnet-4.5`, `gpt-4-turbo-preview`
- Config: `~/.codegod/.env` with API keys

## Key Files to Modify

- **CLI Interface:** `codegod.py`
- **Master Logic:** `master_orchestrator.py` or `master_orchestrator_local.py`
- **Task System:** `task_orchestration.py`
- **Worker Behavior:** `worker_agent.py`, `worker_agent_mcp.py`
- **MCP Integration:** `mcp_discovery.py`, `mcp_integration.py`
- **Prompts:** `prompt_templates.py`

## Project Structure Locations

- **Generated Projects:** `projects/`
- **Logs:** `logs/` (when using Docker)
- **Config:** `~/.codegod/config.json` or `./config/`
- **MCP Servers:** `~/.codegod/mcp_servers/`

## Dependencies

All Python dependencies in `requirements.txt`:
- `rich>=13.7.0` - Terminal UI
- `aiohttp>=3.9.0` - Async HTTP
- `ollama>=0.1.6` - Local model interface
- `anthropic>=0.25.0` - Claude API (optional)
- `openai>=1.12.0` - OpenAI API (optional)
- `structlog>=24.1.0` - Logging
- `python-dotenv>=1.0.0` - Environment config

External requirements:
- **Ollama:** For local models (`curl -fsSL https://ollama.com/install.sh | sh`)
- **Node.js 16+:** For MCP server installation
- **Docker & Docker Compose:** For multi-agent system deployment

## Common Issues

**Model Not Found:**
```bash
ollama pull llama3.1:70b  # Pull missing model
```

**MCP Server Installation Fails:**
```bash
node --version  # Ensure Node.js 16+ installed
cd ~/.codegod/mcp_servers/servers/src/filesystem
npm install && npm run build
```

**Permission Denied (Unix):**
```bash
chmod +x codegod
```

**Out of Memory:**
- Use smaller model: `./codegod --model mixtral:8x7b`
- Reduce context: Edit `~/.codegod/config.json`, set `max_history: 5`

## Recent Features & Fixes

### Comprehensive MCP Server Catalog (2025-01-11)
- **File:** `mcp_servers_catalog.json`, `mcp_discovery.py:71`
- **Change:** Added 50+ MCP servers covering 10+ categories
- **Categories:** core, development, database, web, ai, cloud, devops, productivity, communication, etc.
- **Servers:** GitHub, Stripe, Slack, Notion, Docker, Kubernetes, AWS, Cloudflare, MongoDB, PostgreSQL, etc.
- **Benefit:** Comprehensive ecosystem of tools available for AI agents

### MCP Server Search & Discovery (2025-01-11)
- **Files:** `mcp_discovery.py:581`, `codegod.py:393`
- **New Commands:**
  - `/search <query>` - Search servers by name, description, tools, or category
  - `/categories` - Show all available server categories
- **Usage:** `/search database`, `/search git`, `/categories`
- **Benefit:** Easy discovery of relevant MCP tools from 50+ available servers

### Smart Directory Handling (2025-01-11)
- **Files:** `codegod.py:256`, `project_builder.py:44`
- **Behavior:**
  - `--dir /path/to/dir` → Uses exact directory (no timestamp subdirectory)
  - No `--dir` flag → Creates `./projects/project_name_timestamp/`
- **Usage:** `/build Create API --dir ~/myproject`
- **Benefit:** Full control over project location - exact path or auto-generated

### Shell Mode Toggle (2025-01-11)
- **Files:** `codegod.py:145`, `codegod.py:590`
- **New Command:** `/shell` - Enter shell mode for direct terminal commands
- **Features:**
  - Execute any shell command directly
  - Type `exit` or `quit` to return to AI mode
  - Separate prompt indicator (`$` vs `You>`)
- **Usage:** `/shell` → `ls -la` → `git status` → `exit`
- **Benefit:** Seamlessly switch between AI and terminal without closing app

### Arrow Key Command History (2025-01-11)
- **Files:** `codegod.py:26`, `codegod.py:59`, `requirements.txt`
- **Change:** Integrated `prompt_toolkit` for input handling
- **Features:**
  - ↑/↓ arrows navigate command history
  - Auto-suggestions from history
  - Persistent history across sessions (`~/.codegod/history`)
  - Ctrl+C cancels input, Ctrl+D exits
- **Benefit:** Standard terminal UX with full history navigation

### Earlier Fixes (2025-01-11)

**File Creation Robustness**
- **File:** `project_builder.py:222`
- Removed unreliable MCP filesystem server dependency
- Direct file writing with proper error handling

**MCP Discovery Fallback**
- **File:** `mcp_discovery.py:39`
- Always falls back to default servers if API unavailable
- Handles 404, timeout, and network errors gracefully

**MCP Installation Robustness**
- **File:** `mcp_discovery.py:238`
- Prerequisite checks (Node.js, git)
- Multiple fallback paths for server source
- Comprehensive error messages and logging

## Key Differences from Typical Projects

1. **Dual Architecture:** Two separate orchestrators (API-based vs local-model)
2. **No Traditional Tests:** Validation happens at runtime via `validation_system.py`
3. **Dynamic Tool Discovery:** MCP servers discovered from external API, not hardcoded
4. **Sandboxed Execution:** Docker containers isolate worker operations
5. **Auto-installer Launchers:** `codegod`/`codegod.bat` handle dependency setup
6. **Multi-modal Operations:** CLI can be terminal-interactive or single-command
7. **Direct File Operations:** Recent updates bypass MCP for file I/O reliability
