# MCP Server Installation - Fixed!

## What Was Wrong

The git MCP server installation was failing because:
- Git server is **Python-based** (uses `pyproject.toml`)
- Installer was trying to run **npm commands** on it
- Error: `Could not read package.json` (because it doesn't have one!)

## What's Fixed

The installer now **auto-detects** the project type:

### TypeScript/Node.js Servers
Detects: `package.json`
Runs: `npm install && npm run build`

Examples: filesystem, github, puppeteer, memory

### Python Servers
Detects: `pyproject.toml` or `requirements.txt`
Runs: `pip install -e .` or `uv pip install -e .` (if uv available)

Examples: git, and future Python MCP servers

## Try It Now

```bash
./codegod --model qwen2.5:1.5b

# Install TypeScript server
You> /install filesystem
✓ Detected Node.js/TypeScript project
✓ filesystem installed successfully

# Install Python server (THIS NOW WORKS!)
You> /install git
✓ Detected Python project (pyproject.toml)
✓ git installed successfully

You> /servers
# Shows both installed
```

## Verify Git Server

```bash
You> Check my git status

Code-God> <TOOL_CALL>
server: git
tool: git_status
arguments:
  repo_path: .
</TOOL_CALL>

[Tool Result]
On branch master
nothing to commit, working tree clean
```

## Available MCP Servers

| Server | Language | Status |
|--------|----------|--------|
| filesystem | TypeScript | ✓ Works |
| git | Python | ✓ FIXED! |
| github | TypeScript | ✓ Works |
| memory | TypeScript | ✓ Works |
| puppeteer | TypeScript | ✓ Works |

All should install correctly now!

## What Happens Under the Hood

### Before (Broken)
```
/install git
→ Copies git server files
→ Runs: npm install && npm run build
→ ERROR: No package.json found
```

### After (Fixed)
```
/install git
→ Copies git server files
→ Detects: pyproject.toml exists
→ Runs: pip install -e .
→ SUCCESS!
```

## Troubleshooting

### If pip install fails
Make sure you have Python 3.10+ installed:
```bash
python3 --version  # Should be 3.10+
```

### If you want faster Python installs
Install `uv` (much faster than pip):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The installer will automatically use `uv` if available!

### Old broken installation
If you have a broken git installation from before:
```bash
You> /install git
# It will reinstall correctly now
```

Or manually clean up:
```bash
rm -rf ~/.codegod/mcp_servers/git
./codegod
You> /install git
```

## What This Means

You can now install **ANY** MCP server from the official repository:
- TypeScript servers work ✓
- Python servers work ✓
- Auto-detection handles it all ✓

No more installation errors!
