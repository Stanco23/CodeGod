# Code-God Quick Start

Get up and running in 2 minutes.

## Install

```bash
# Clone
git clone https://github.com/yourusername/code-god.git
cd code-god

# Run (auto-installs everything)
./codegod           # Linux/macOS
codegod.bat         # Windows
```

## First Project

```
You> /build Create a REST API for managing tasks with FastAPI

[Building...]

✓ Complete! → projects/create_rest_api_20250115/
```

## Essential Commands

- `/build <description>` - Build a project
- `/mcp` - Show available MCP servers
- `/install <server>` - Install MCP server
- `/help` - All commands
- `/exit` - Quit

## Install Ollama (for local models)

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:70b
```

### macOS

```bash
brew install ollama
ollama pull llama3.1:70b
```

### Windows

Download from [ollama.com](https://ollama.com/download)

```powershell
ollama pull llama3.1:70b
```

## Use API Models (if no GPU)

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run with API
./codegod --model claude-sonnet-4.5 --prefer-api
```

## Troubleshooting

**Python not found**:
```bash
# Install Python 3.8+
# - Linux: sudo apt install python3
# - macOS: brew install python3
# - Windows: python.org
```

**Permission denied**:
```bash
chmod +x codegod
```

**Model not found**:
```bash
ollama pull llama3.1:70b
```

## Next Steps

- Read [README.md](README.md) for full documentation
- Try `/build` with different project types
- Install MCP servers with `/install`
- Chat naturally with the AI

## Examples

**Web API**:
```
/build Create a FastAPI REST API for user management with JWT auth and PostgreSQL
```

**Frontend**:
```
/build Create a React todo app with TypeScript, state management, and local storage
```

**Full Stack**:
```
/build Create a blog platform with React frontend, FastAPI backend, and PostgreSQL
```

**Mobile**:
```
/build Create a React Native weather app using OpenWeather API
```

That's it! Start building! 🚀
