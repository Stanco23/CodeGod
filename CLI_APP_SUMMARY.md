# Code-God CLI Application - Complete Transformation

## Overview

Code-God has been completely redesigned as a **terminal-based interactive application** similar to Claude Code, with automatic MCP server discovery and cross-platform support.

---

## What Changed

### ❌ Removed: REST API/Docker Architecture
- No more FastAPI REST endpoints
- No Docker containers
- No Redis/PostgreSQL infrastructure
- No complex multi-service setup

### ✅ Added: Terminal CLI Application
- Interactive terminal interface (like Claude Code)
- Beautiful Rich UI with tables, panels, markdown
- Conversation-based interaction
- Command system with `/` prefix
- Cross-platform launchers

---

## New Architecture

```
User
  ↓
Terminal Interface (Rich TUI)
  ↓
Code-God CLI App (codegod.py)
  ├─→ Local Model Executor (Ollama / API)
  ├─→ MCP Discovery (1mcpserver.com)
  ├─→ Conversation Manager
  └─→ Project Builder
  ↓
Generated Projects
```

---

## Core Files

### 1. **codegod.py** (Main Application)
- Terminal UI with Rich library
- Command system (`/build`, `/mcp`, `/help`, etc.)
- Interactive conversation loop
- Beautiful formatted output
- Progress indicators

**Key Features**:
```python
class CodeGod:
    - initialize() - Setup models, MCP, etc.
    - run() - Main interactive loop
    - _handle_command() - Process /commands
    - _handle_message() - Natural conversation
```

### 2. **mcp_discovery.py** (MCP Integration)
- Integrates with **1mcpserver.com API**
- Automatically discovers available MCP servers
- One-command installation: `/install <server>`
- Manages server lifecycle
- Real MCP protocol communication

**Key Features**:
```python
class MCPDiscovery:
    - discover_servers() - Fetch from 1mcpserver.com
    - install_server() - Clone, build, install
    - start_server() - Launch MCP server
    - call_tool() - Execute MCP tools
```

### 3. **conversation_manager.py** (Chat Handling)
- Multi-turn conversations
- Context management
- History tracking
- Smart prompting

**Key Features**:
```python
class ConversationManager:
    - send_message() - Send to AI, get response
    - _build_context() - Prepare conversation context
    - _get_system_prompt() - Dynamic system prompts with MCP info
```

### 4. **project_builder.py** (Autonomous Building)
- Analyzes requirements
- Generates complete projects
- Uses MCP tools for file operations
- Git initialization
- Progress tracking

**Key Features**:
```python
class ProjectBuilder:
    - build_project() - Main build workflow
    - _analyze_requirements() - AI planning
    - _generate_file() - Code generation
    - _init_git() - Git setup via MCP
```

### 5. **local_model_executor.py** (Model Interface)
- Supports local models (Llama, Qwen, Mixtral)
- API fallback (Claude, GPT-4)
- Automatic model selection
- Backend abstraction

**Key Features**:
```python
class LocalModelExecutor:
    - execute() - Run prompt with model
    - _execute_ollama() - Local execution
    - _execute_api() - API execution

def get_master_model() -> LocalModelExecutor:
    # Auto-selects best available model
```

### 6. **Launchers**
- `codegod` (Unix) - Bash script
- `codegod.bat` (Windows) - Batch script

**Features**:
- Auto-create venv
- Install dependencies
- Launch application
- Cross-platform

---

## Usage Flow

### 1. Launch

```bash
./codegod                    # Linux/macOS
codegod.bat                  # Windows
```

### 2. Interactive Session

```
╔═══════════════════════════════════════════╗
║         CODE-GOD                          ║
║   Autonomous AI Development Assistant    ║
╚═══════════════════════════════════════════╝

✓ Model loaded: llama3.1:70b
✓ Found 13 MCP servers
✓ Project builder ready

You>
```

### 3. Commands

**Build Projects**:
```
You> /build Create a REST API for task management

[Analyzing requirements...]
[Creating structure...]
[Generating code...]

✓ Complete! → projects/task_api_20250115/
```

**Chat**:
```
You> How do I implement JWT auth in FastAPI?

Code-God> Here's how to implement JWT authentication...
```

**Manage MCP Servers**:
```
You> /mcp

Available MCP Servers:
┌────────────┬─────────────────┬──────────┬───────────┐
│ Name       │ Description     │ Tools    │ Installed │
├────────────┼─────────────────┼──────────┼───────────┤
│ filesystem │ File operations │ 8        │ ✓         │
│ git        │ Git operations  │ 11       │ ✓         │
│ github     │ GitHub API      │ 9        │ ✗         │
└────────────┴─────────────────┴──────────┴───────────┘

You> /install github
[Installing...]
✓ github installed successfully
```

---

## MCP Server Discovery

### Integration with 1mcpserver.com

**How It Works**:
1. On startup, fetches server list from `https://1mcpserver.com/api/servers`
2. Displays available servers with `/mcp` command
3. User installs with `/install <server>`
4. Automatically clones, builds, and integrates
5. Available to AI for tool usage

**Fallback**:
- If 1mcpserver.com unreachable, uses built-in list
- Includes all official MCP servers from modelcontextprotocol/servers

**Supported Servers**:
- filesystem, git, github, postgres, sqlite
- fetch, memory, puppeteer, sequential-thinking
- brave-search, google-maps, slack, time

---

## Key Differences from REST API Version

| Aspect | REST API Version | CLI Version |
|--------|------------------|-------------|
| **Interface** | HTTP endpoints | Terminal TUI |
| **Usage** | curl/HTTP clients | Interactive CLI |
| **Deployment** | Docker containers | Single script |
| **Complexity** | High (multi-service) | Low (single app) |
| **Startup** | Docker compose up | ./codegod |
| **Installation** | Docker required | Python only |
| **MCP Discovery** | Manual config | Automatic from 1mcpserver |
| **Workers** | Separate containers | Integrated |
| **Conversation** | Stateless API | Stateful session |

---

## Benefits of CLI Approach

### 1. **Simplicity**
- One command to start: `./codegod`
- No Docker, no containers
- No infrastructure to manage
- Just Python and dependencies

### 2. **Portability**
- Works on Linux, macOS, Windows
- No platform-specific quirks
- Single script launcher
- Virtual environment isolation

### 3. **User Experience**
- Interactive conversation
- Immediate feedback
- Rich terminal UI
- Command history
- Tab completion (future)

### 4. **Development**
- Easier to develop
- Simpler to debug
- Faster iteration
- Less overhead

### 5. **MCP Integration**
- Automatic discovery from 1mcpserver.com
- One-command installation
- Seamless integration
- Always up-to-date

---

## Installation Steps

### Prerequisites
- Python 3.8+ (required)
- Node.js 16+ (for MCP servers)
- Git (for installation)
- Ollama (for local models) OR API key

### Quick Install

```bash
# 1. Clone
git clone https://github.com/yourusername/code-god.git
cd code-god

# 2. Run
./codegod              # Auto-installs everything
```

### Manual Install

```bash
# 1. Clone
git clone https://github.com/yourusername/code-god.git
cd code-god

# 2. Create venv
python3 -m venv venv
source venv/bin/activate

# 3. Install deps
pip install -r requirements.txt

# 4. Run
python codegod.py
```

---

## Configuration

### Config File: `~/.codegod/config.json`

```json
{
  "model": "llama3.1:70b",
  "prefer_local": true,
  "max_history": 10,
  "mcp_servers_dir": "~/.codegod/mcp_servers"
}
```

### Environment Variables

```bash
export CODEGOD_MODEL=qwen2.5:72b
export PREFER_LOCAL=true
export ANTHROPIC_API_KEY=sk-ant-...  # Optional
export OPENAI_API_KEY=sk-...         # Optional
```

### Command-Line Arguments

```bash
./codegod --model llama3.1:70b
./codegod --prefer-api
./codegod --build "Create a REST API..."  # Non-interactive
```

---

## Example Workflows

### Workflow 1: Build a Project

```
$ ./codegod

You> /build Create a FastAPI REST API for todo management with
     SQLite, user authentication, and CRUD operations

Code-God> [Analyzing requirements...]
          Planning FastAPI application with:
          - SQLite database
          - JWT authentication
          - CRUD endpoints for todos
          - User management

          [Creating project structure...]
          projects/todo_api_20250115/
          ├── main.py
          ├── models.py
          ├── auth.py
          ├── database.py
          ├── requirements.txt
          └── README.md

          [Generating code...]
          ✓ main.py created
          ✓ models.py created
          ✓ auth.py created
          [Using filesystem MCP server]

          [Initializing git...]
          [Using git MCP server]
          ✓ Repository initialized
          ✓ Initial commit created

          ✓ Project complete!

You> Great! Now add documentation

Code-God> [Generating API documentation...]
          [Using filesystem to create docs/...]
```

### Workflow 2: Chat and Learn

```
You> What's the best way to handle authentication in FastAPI?

Code-God> For FastAPI authentication, here are the best approaches:

          1. JWT Tokens (recommended):
             - Use `python-jose` for JWT
             - Implement OAuth2 with password flow
             ...

          2. Session-based:
             - Use `fastapi-sessions`
             ...

          Would you like me to generate an example?

You> Yes, show JWT

Code-God> Here's a complete JWT authentication example:
          [Detailed code with explanations...]
```

### Workflow 3: MCP Server Management

```
You> /mcp

[Shows available servers...]

You> /install github

Code-God> Installing github MCP server...
          ✓ Cloned repository
          ✓ Installed dependencies
          ✓ Built server
          ✓ github is now available

You> Now build a tool that creates GitHub repos

Code-God> [Uses github MCP server to enable GitHub operations...]
```

---

## Technical Details

### Dependencies

**Required**:
- `rich>=13.7.0` - Terminal UI
- `aiohttp>=3.9.0` - HTTP client
- `ollama>=0.1.6` - Local models

**Optional**:
- `anthropic>=0.25.0` - Claude API
- `openai>=1.12.0` - GPT API

### File Structure

```
code-god/
├── codegod.py              # Main CLI app (600+ lines)
├── local_model_executor.py # Model interface (380 lines)
├── mcp_discovery.py        # MCP integration (500+ lines)
├── conversation_manager.py # Chat handling (200 lines)
├── project_builder.py      # Project builder (400+ lines)
├── codegod                 # Unix launcher
├── codegod.bat             # Windows launcher
├── requirements.txt        # Dependencies
├── README.md               # Full documentation
├── QUICKSTART.md           # Quick start guide
└── CLI_APP_SUMMARY.md      # This file
```

### State Management

**User Data**: `~/.codegod/`
```
~/.codegod/
├── config.json           # User configuration
├── .env                  # API keys (optional)
├── mcp_servers/          # Installed MCP servers
│   ├── repos/            # Cloned repositories
│   ├── filesystem/       # Built server
│   ├── git/              # Built server
│   └── ...
└── logs/                 # Application logs
```

**Project Output**: `./projects/`
```
./projects/
├── project_name_20250115_120000/
│   ├── src/
│   ├── tests/
│   ├── README.md
│   └── .git/
└── ...
```

---

## Comparison to Claude Code

| Feature | Code-God | Claude Code |
|---------|----------|-------------|
| **Platform** | Linux/macOS/Windows | macOS only |
| **Cost** | Free (local) or API | API only |
| **Models** | Any local + API | Claude API |
| **MCP Discovery** | Automatic (1mcpserver) | Manual config |
| **Project Building** | Autonomous | Assisted |
| **Installation** | One script | Installer |
| **Requirements** | Python + Ollama | macOS + API key |

---

## Future Enhancements

### Planned Features
- [ ] Enhanced TUI with panels/splits
- [ ] Tab completion for commands
- [ ] Project templates
- [ ] Multi-file editing
- [ ] Git integration UI
- [ ] Debugging support
- [ ] Testing integration
- [ ] CI/CD templates

### MCP Improvements
- [ ] Custom server creation wizard
- [ ] Server marketplace
- [ ] Server ratings/reviews
- [ ] Auto-update servers
- [ ] Server dependencies

### UX Improvements
- [ ] Syntax highlighting
- [ ] Code preview
- [ ] File browser
- [ ] Diff viewer
- [ ] Search history
- [ ] Keyboard shortcuts

---

## Migration from Docker Version

If you were using the Docker/REST API version:

### What to Keep
- All the core logic (models, MCP, validation) is still there
- Just repackaged as CLI instead of API

### What to Change
- Instead of `curl http://localhost:8001/build`, use `/build`
- Instead of Docker containers, use Python script
- Instead of API calls, use interactive commands

### Migration Steps

1. Clone new version
2. Run `./codegod`
3. Use `/build` instead of API calls
4. That's it!

---

## Status

✅ **Complete and Ready**

All functionality implemented:
- Interactive terminal application
- Local model support
- MCP discovery from 1mcpserver.com
- Project building
- Conversation handling
- Cross-platform launchers
- Comprehensive documentation

---

## Quick Reference

### Launch
```bash
./codegod              # Unix
codegod.bat            # Windows
```

### Essential Commands
```
/build <prompt>        # Build project
/mcp                   # Show servers
/install <server>      # Install server
/help                  # All commands
/exit                  # Quit
```

### Configuration
```bash
~/.codegod/config.json  # Config file
~/.codegod/.env         # API keys
```

### Documentation
```
README.md              # Full guide
QUICKSTART.md          # 2-minute start
CLI_APP_SUMMARY.md     # This file
```

---

**The CLI version is production-ready and superior to the REST API approach for this use case!** 🎉
