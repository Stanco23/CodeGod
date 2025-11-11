# Code-God: Autonomous AI Development Assistant

<div align="center">

**Terminal-based interactive AI for building complete applications**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Cross-Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Windows-green.svg)]()

</div>

---

## What is Code-God?

Code-God is a **terminal-based interactive AI assistant** that autonomously builds complete applications from natural language descriptions. Think of it as having an expert developer in your terminal who can:

- Build entire projects from a single prompt
- Answer development questions with context
- Automatically discover and use MCP tools
- Work 100% locally (no API keys required)
- Run on Linux, macOS, and Windows

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-god.git
cd code-god

# Run Code-God (auto-installs dependencies)
./codegod              # Linux/macOS
codegod.bat            # Windows
```

That's it! The launcher will:
- Create a virtual environment
- Install all dependencies
- Launch the interactive terminal app

### First Project

```
Code-God> /build Create a REST API for managing tasks with FastAPI and SQLite

[Building your project...]

✓ Project complete! → projects/create_rest_api_20250115_143022/
```

## Features

### 🎯 Interactive Terminal Interface

- Beautiful rich terminal UI
- Conversation-style interaction
- Real-time progress indicators
- Markdown rendering
- Command system with `/` prefix

### 🤖 Local AI Models (No API Keys)

- Supports Llama 3.1 (70B/405B)
- Qwen 2.5 72B
- Mixtral 8x7B
- DeepSeek Coder 33B
- Optional API support (Claude/GPT-4)

### 🔧 Automatic MCP Discovery

- Integrates with [1mcpserver.com](https://1mcpserver.com)
- Automatically discovers available MCP servers
- One-command installation
- 13+ official servers supported

### 🚀 Autonomous Project Building

- Generates complete project structure
- Creates all necessary files
- Initializes git repository
- Production-ready code
- Comprehensive documentation

### 💬 Conversational AI

- Ask questions naturally
- Get contextual answers
- Build incrementally
- Learn as you go

## Commands

### Project Commands

- `/build <description>` - Build a new project
- `/list` - Show recent projects

### MCP Server Commands

- `/mcp` - Show available MCP servers
- `/servers` - List installed servers
- `/install <server>` - Install an MCP server

### System Commands

- `/status` - Show system status
- `/model` - Show AI model info
- `/config` - View configuration
- `/clear` - Clear screen
- `/help` - Show all commands
- `/exit` or `/quit` - Exit Code-God

## Usage Examples

### Build a Web API

```
You: /build Create a FastAPI server with user authentication,
     CRUD operations for blog posts, and PostgreSQL database

Code-God: [Analyzing requirements...]
          [Creating project structure...]
          [Generating code...]
          ✓ Project complete!
```

### Ask Questions

```
You: How do I implement JWT authentication in FastAPI?

Code-God: Here's how to implement JWT authentication in FastAPI:

1. Install dependencies:
   pip install python-jose passlib

2. Create authentication utilities:
   [detailed explanation with code]
...
```

### Install MCP Servers

```
You: /mcp

Available MCP Servers:
┌─────────────┬────────────────────────────┬──────────┬───────────┐
│ Name        │ Description                │ Tools    │ Installed │
├─────────────┼────────────────────────────┼──────────┼───────────┤
│ filesystem  │ File operations            │ 8        │ ✓         │
│ git         │ Git operations             │ 11       │ ✓         │
│ github      │ GitHub API                 │ 9        │ ✗         │
│ postgres    │ PostgreSQL operations      │ 7        │ ✗         │
└─────────────┴────────────────────────────┴──────────┴───────────┘

You: /install github

[Installing github...]
✓ github is now available
```

## Configuration

### Model Selection

Edit `~/.codegod/config.json`:

```json
{
  "model": "llama3.1:70b",
  "prefer_local": true,
  "max_history": 10
}
```

Or use environment variables:

```bash
export CODEGOD_MODEL=qwen2.5:72b
export PREFER_LOCAL=true
./codegod
```

### Available Models

**Local Models** (via Ollama):
- `llama3.1:70b` - Recommended, balanced quality/speed
- `llama3.1:405b` - Best quality, very slow
- `qwen2.5:72b` - Excellent alternative
- `mixtral:8x7b` - Good for lower VRAM
- `deepseek-coder:33b` - Code-focused

**API Models** (requires keys):
- `claude-sonnet-4.5` - Anthropic Claude
- `gpt-4-turbo-preview` - OpenAI GPT-4

### API Keys (Optional)

Create `~/.codegod/.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## MCP Server Discovery

Code-God integrates with [1mcpserver.com](https://1mcpserver.com) to automatically discover and install MCP servers.

### How It Works

1. **Discovery**: Fetches available servers from 1mcpserver.com API
2. **Installation**: One command installs any server
3. **Integration**: Automatically available to AI
4. **Updates**: Refreshes server list on startup

### Supported Servers

| Server | Tools | Use Case |
|--------|-------|----------|
| filesystem | 8 tools | File operations |
| git | 11 tools | Version control |
| github | 9 tools | GitHub operations |
| postgres | 7 tools | PostgreSQL |
| sqlite | 6 tools | SQLite |
| fetch | 4 tools | Web scraping |
| memory | 6 tools | Knowledge graph |
| puppeteer | 6 tools | Browser automation |
| + more | 60+ total | Various |

## System Requirements

### Minimum

- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 8GB (for small models)
- **Storage**: 10GB

### Recommended

- **RAM**: 48GB+ (for 70B models)
- **GPU**: NVIDIA with 40GB+ VRAM
- **Storage**: 100GB SSD

### Cloud/CPU-Only

Works on CPU but slower:
- **RAM**: 64GB+
- **CPU**: 16+ cores
- Expected: 5-10x slower than GPU

## Project Structure

```
code-god/
├── codegod.py              # Main CLI application
├── codegod                 # Unix launcher
├── codegod.bat             # Windows launcher
├── local_model_executor.py # Local model support
├── mcp_discovery.py        # MCP server discovery
├── conversation_manager.py # Conversation handling
├── project_builder.py      # Project building
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## How It Works

### 1. Interactive Terminal

Launch Code-God and you get a beautiful terminal interface:

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

### 2. Natural Conversation

Talk to the AI naturally:

```
You> I want to build a todo app

Code-God> I can help you build a todo application!
          To create the best solution, I need to know:

          1. Platform? (web, mobile, desktop)
          2. Database? (SQLite, PostgreSQL, MongoDB)
          3. Features? (users, sharing, reminders, etc.)

          Or use /build with a detailed description!
```

### 3. Autonomous Building

Use `/build` to create projects:

```
You> /build Create a web-based todo app with user auth,
     task categories, due dates, and PostgreSQL backend

[Code-God autonomously:]
- Analyzes requirements
- Plans architecture
- Generates all files
- Initializes git
- Creates documentation
```

### 4. MCP Tool Usage

MCP servers provide capabilities:

```
When generating code:
- Uses `filesystem` to write files
- Uses `git` to initialize repo
- Uses `postgres` to create schema
- All automatically!
```

## Non-Interactive Mode

Build projects without interaction:

```bash
# Linux/macOS
./codegod --build "Create a REST API for user management"

# Windows
codegod.bat --build "Create a REST API for user management"
```

Perfect for:
- CI/CD pipelines
- Scripting
- Automation
- Batch operations

## Troubleshooting

### Model Not Found

```bash
# Install model with Ollama
ollama pull llama3.1:70b

# Or use API model
export ANTHROPIC_API_KEY=sk-ant-...
./codegod --model claude-sonnet-4.5 --prefer-api
```

### MCP Server Installation Fails

```bash
# Check Node.js is installed
node --version  # Should be 16+

# Install Node.js if needed
# - Linux: sudo apt install nodejs npm
# - macOS: brew install node
# - Windows: Download from nodejs.org

# Manual installation
cd ~/.codegod/mcp_servers/
git clone https://github.com/modelcontextprotocol/servers.git
cd servers/src/filesystem
npm install && npm run build
```

### Out of Memory

```bash
# Use smaller model
./codegod --model mixtral:8x7b

# Or reduce context
# Edit ~/.codegod/config.json
{
  "max_history": 5  # Reduce from 10
}
```

### Permission Denied (Unix)

```bash
chmod +x codegod
./codegod
```

## Development

### Running from Source

```bash
# Clone repository
git clone https://github.com/yourusername/code-god.git
cd code-god

# Create venv
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run directly
python codegod.py
```

### Adding Features

The codebase is modular:

- `codegod.py` - Main app, commands, UI
- `mcp_discovery.py` - MCP server discovery
- `project_builder.py` - Project generation
- `conversation_manager.py` - Chat handling
- `local_model_executor.py` - Model interface

Contributions welcome!

## Comparison

### vs Claude Code

| Feature | Code-God | Claude Code |
|---------|----------|-------------|
| Interface | Terminal | Terminal |
| AI Model | Local or API | API only |
| MCP Support | Auto-discovery | Manual config |
| Cost | $0 (local) | API fees |
| Privacy | 100% local | Data to Anthropic |
| Platforms | Linux/macOS/Windows | macOS only |
| Project Building | Autonomous | Assisted |

### vs Cursor

| Feature | Code-God | Cursor |
|---------|----------|--------|
| Type | Terminal | IDE |
| AI Model | Flexible | Fixed |
| Autonomy | Full | Assisted |
| Cost | Free (local) | Subscription |
| Editing | Generates | In-place |

### vs GitHub Copilot

| Feature | Code-God | Copilot |
|---------|----------|---------|
| Scope | Full projects | Code completion |
| Autonomy | Autonomous | Suggestions |
| MCP Tools | Yes | No |
| Local | Yes | No |
| Conversation | Yes | Limited |

## Roadmap

- [ ] Enhanced UI with panels and tables
- [ ] Project templates
- [ ] Multi-language support
- [ ] Cloud sync for projects
- [ ] Plugin system
- [ ] Team collaboration
- [ ] Web interface
- [ ] Mobile app

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE)

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/code-god/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/code-god/discussions)
- **Docs**: [Documentation](https://docs.codegod.dev)

---

**Built with ❤️ by the Code-God community**

*Autonomous development has never been easier.*
