# MCP Tools with Small Models

Now your small models (1B-3B) can actually USE MCP tools! Here's how it works.

## What Changed

**Before**: Small models said "I don't have access to MCP tools"
**Now**: Small models can execute MCP tools and get results

## How It Works

When you chat with Code-God, the model can now call MCP tools using a simple format:

```
<TOOL_CALL>
server: filesystem
tool: list_directory
arguments:
  path: /home/user/projects
</TOOL_CALL>
```

The system automatically:
1. Parses the tool call
2. Executes it via MCP protocol
3. Returns results to you
4. Continues the conversation

## Available Tools

Check what tools you have installed:
```bash
./codegod

You> /servers
```

Common tools include:
- **filesystem**: Read, write, list files
- **git**: Git operations (status, commit, diff)
- **github**: GitHub API operations
- Plus any MCP servers you install

## Examples

### List Files
```
You> Show me the files in my current directory

Code-God> Let me check that for you.
<TOOL_CALL>
server: filesystem
tool: list_directory
arguments:
  path: .
</TOOL_CALL>

[Tool Result - list_directory]
src/
  main.py
  utils.py
tests/
README.md
requirements.txt
```

### Create a File
```
You> Create a file called hello.py with a simple print statement

Code-God> I'll create that file for you.
<TOOL_CALL>
server: filesystem
tool: write_file
arguments:
  path: hello.py
  content: print("Hello from Code-God!")
</TOOL_CALL>

[Tool Result - write_file]
File created successfully
```

### Check Git Status
```
You> What's my git status?

Code-God> Let me check git status.
<TOOL_CALL>
server: git
tool: git_status
arguments:
  repo_path: .
</TOOL_CALL>

[Tool Result - git_status]
On branch master
Changes not staged for commit:
  modified: README.md
```

## Installing MCP Servers

To get more tools, install MCP servers:

```bash
./codegod

You> /mcp
# Shows available servers

You> /install filesystem
# Installs filesystem server

You> /install git
# Installs git server
```

## Tips for Small Models

### 1. Be Specific
Small models work better with clear requests:

**Good**: "List all .py files in the src directory"
**Bad**: "Show me stuff"

### 2. One Tool at a Time
Small models handle single tool calls better:

**Good**: "Create test.txt" → "Now add hello world to it"
**Bad**: "Create test.txt and add hello world and commit it"

### 3. Simple Operations First
Start with basic operations:
- List files
- Read a file
- Create a file
- Check git status

Then progress to complex ones:
- Multi-file operations
- Git workflows
- API calls

### 4. If Tool Fails
The model will show an error. Just rephrase:

```
You> Read the config

[Tool Error - read_file]
File not found: config

You> Read the config.json file

[Tool Result - read_file]
{
  "model": "qwen2.5:1.5b"
}
```

## Troubleshooting

### Model doesn't use tools
The model should now automatically use tools when appropriate. If it doesn't:

1. **Make the request more action-oriented**:
   - Instead of: "Can you list files?"
   - Try: "List the files in this directory"

2. **Be explicit about the action**:
   - "Show me the contents of README.md"
   - "Create a new file called test.py"
   - "Check the git status"

### Tool call format is wrong
The system will ignore malformed tool calls. The model should learn from examples in the system prompt, but if issues persist:

1. Use `/build` command for complex tasks
2. Try a slightly larger model (3B instead of 1B)
3. Be more specific in your requests

### Server not installed
```
[Tool Error - list_directory]
Could not start server filesystem

Solution:
You> /install filesystem
```

## What Tools Can Do

### Filesystem Server
- ✅ List directory contents
- ✅ Read files
- ✅ Write files
- ✅ Create directories
- ✅ Move/copy files
- ✅ Search for files
- ✅ Get file info (size, modified date, etc.)

### Git Server
- ✅ Check status
- ✅ View diffs
- ✅ Stage files (git add)
- ✅ Commit changes
- ✅ View log
- ✅ Create branches
- ✅ Checkout branches
- ✅ Merge branches

### GitHub Server
- ✅ Search repositories
- ✅ Create repositories
- ✅ Read file contents
- ✅ Create/update files
- ✅ Create issues
- ✅ Create pull requests
- ✅ Fork repositories

## Practical Workflows

### Workflow 1: Explore a Project
```
You> List all files in the current directory
You> Show me what's in main.py
You> Check the git status
You> Read the README file
```

### Workflow 2: Create New Code
```
You> Create a new file utils.py
You> Add a function to validate email addresses
You> Show me the file you created
You> Add it to git
```

### Workflow 3: Project Setup
```
You> Create a new directory called my_project
You> Create a README.md file with project info
You> Create a main.py file with a basic structure
You> Initialize git in this directory
```

## Combining with /build

For complex projects, use `/build`:

```
You> /build Create a REST API for todo management with FastAPI
```

This uses MCP tools automatically to:
1. Create project structure
2. Generate all files
3. Initialize git repository
4. Create documentation

Then you can chat about it:
```
You> Show me the main.py file
You> Modify the todo model to add a priority field
You> Check what changed in git
```

## Model-Specific Tips

### Qwen 2.5 1.5B
- Best for: Simple file operations, reading files
- Works well: List, read, write single files
- May struggle: Complex multi-step operations

### Llama 3.2 3B
- Best for: File operations + basic git
- Works well: Most filesystem and git tools
- Good at: Following tool call format

### Phi-3 Mini
- Best for: Structured operations
- Works well: Git workflows, file management
- Good at: Chaining operations logically

## Advanced: Tool Chaining

With 3B models, you can chain operations:

```
You> Create a new feature branch, create a file with a test function,
     and commit it

Code-God> I'll do that step by step.

<TOOL_CALL>
server: git
tool: git_create_branch
arguments:
  branch_name: feature/new-test
</TOOL_CALL>

<TOOL_CALL>
server: filesystem
tool: write_file
arguments:
  path: test_feature.py
  content: def test_feature():\n    assert True
</TOOL_CALL>

<TOOL_CALL>
server: git
tool: git_add
arguments:
  paths: ["test_feature.py"]
</TOOL_CALL>

<TOOL_CALL>
server: git
tool: git_commit
arguments:
  message: "Add test feature"
</TOOL_CALL>

[Multiple tool results shown...]
```

## Summary

**Small models (1B-3B) can now**:
- ✅ Use MCP tools in conversation
- ✅ Execute file operations
- ✅ Run git commands
- ✅ Access any installed MCP server
- ✅ See tool results in real-time
- ✅ Chain multiple operations

**This makes them much more useful** for:
- Code exploration
- File management
- Git workflows
- Project setup
- Quick modifications

Just chat naturally and the model will use tools when needed!
