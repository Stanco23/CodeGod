# Using Small Models (1B-3B)

Quick guide for running Code-God with small models on low-resource systems.

## TL;DR - Fastest Setup

```bash
# 1. Install a small model
ollama pull qwen2.5:1.5b

# 2. Run Code-God with it
./codegod --model qwen2.5:1.5b
```

## Recommended Small Models

### Qwen 2.5 1.5B (Best for low resource)
```bash
ollama pull qwen2.5:1.5b
./codegod --model qwen2.5:1.5b
```
- **Download**: ~900MB
- **RAM needed**: 1-2GB
- **Speed**: ⚡⚡⚡⚡⚡ (Very fast)
- **Quality**: ⭐⭐⭐ (Good for simple tasks)

### Llama 3.2 3B (Best balance)
```bash
ollama pull llama3.2:3b
./codegod --model llama3.2:3b
```
- **Download**: ~2GB
- **RAM needed**: 2-4GB
- **Speed**: ⚡⚡⚡⚡ (Fast)
- **Quality**: ⭐⭐⭐⭐ (Good reasoning)

### Qwen 2.5 3B (Alternative)
```bash
ollama pull qwen2.5:3b
./codegod --model qwen2.5:3b
```
- **Download**: ~1.9GB
- **RAM needed**: 2-3GB
- **Speed**: ⚡⚡⚡⚡ (Fast)
- **Quality**: ⭐⭐⭐⭐ (Strong for size)

## Configuration Options

### Option 1: Command Line (Recommended for testing)
```bash
./codegod --model qwen2.5:1.5b
```

### Option 2: Environment Variable
```bash
# Create .env file
echo "MASTER_MODEL=qwen2.5:1.5b" > .env

# Run normally
./codegod
```

### Option 3: Config File
```bash
# Create config
mkdir -p ~/.codegod
cat > ~/.codegod/config.json << EOF
{
  "model": "qwen2.5:1.5b",
  "prefer_local": true,
  "max_history": 5
}
EOF

# Run normally
./codegod
```

## Performance Comparison

| Model | Download | RAM | Startup | Response | Quality |
|-------|----------|-----|---------|----------|---------|
| qwen2.5:1.5b | 900MB | 1-2GB | 2-3s | 3-5s | Good |
| llama3.2:1b | 1.3GB | 1-2GB | 2-3s | 3-5s | Basic |
| gemma2:2b | 1.6GB | 2-3GB | 3-4s | 4-6s | Good |
| qwen2.5:3b | 1.9GB | 2-3GB | 3-4s | 4-7s | Better |
| llama3.2:3b | 2GB | 2-4GB | 3-4s | 5-8s | Better |
| phi3:mini | 2.3GB | 3-4GB | 4-5s | 6-9s | Good |

## What to Expect

### With 1B-1.5B Models:
- ✅ Simple code generation (functions, classes)
- ✅ Basic project scaffolding
- ✅ Documentation generation
- ✅ Code explanations
- ⚠️ May struggle with complex architectures
- ⚠️ Limited multi-step reasoning

### With 3B Models:
- ✅ All of the above
- ✅ Better code quality
- ✅ More complex logic
- ✅ Better project structure
- ⚠️ Still limited on very complex tasks

## Tips for Small Models

1. **Be specific**: Give clear, detailed prompts
   ```
   Good: "Create a Python function that validates email addresses using regex"
   Bad: "Make an email thing"
   ```

2. **Break down complex tasks**: Use multiple steps
   ```
   Step 1: /build Create basic FastAPI server
   Step 2: Add authentication endpoint
   Step 3: Add database integration
   ```

3. **Use simpler architectures**: Small models work better with straightforward designs

4. **Reduce context**: Set `max_history: 5` in config to save memory

5. **Be patient**: Small models are fast but may need guidance

## When to Upgrade

Consider upgrading to larger models if:
- You need more complex reasoning
- Working on production-quality code
- Need better architecture decisions
- Have GPU/more RAM available

**Next step up**: `llama3.1:8b` (needs 8GB RAM)
```bash
ollama pull llama3.1:8b
./codegod --model llama3.1:8b
```

## Troubleshooting

### "Model not found"
```bash
# Install it first
ollama pull qwen2.5:1.5b
```

### "Out of memory"
```bash
# Use smaller model
./codegod --model qwen2.5:1.5b

# Or reduce history
echo '{"max_history": 3}' > ~/.codegod/config.json
```

### "Slow responses"
- This is normal for CPU-only systems
- Small models are already the fastest option
- Consider API models if speed is critical:
  ```bash
  ./codegod --prefer-api --model claude-sonnet-4.5
  ```

### "Poor code quality"
- Try a 3B model instead of 1B
- Be more specific in your prompts
- Break complex tasks into smaller steps
- Or upgrade to 8B+ model if you have RAM

## Examples

### Simple REST API
```bash
./codegod --model qwen2.5:3b

You> /build Create a simple REST API with FastAPI that has
     GET and POST endpoints for managing a todo list
```

### Basic Web App
```bash
./codegod --model llama3.2:3b

You> /build Create a static HTML/CSS/JS todo app with
     local storage persistence
```

### Code Explanation
```bash
./codegod --model qwen2.5:1.5b

You> Explain how async/await works in Python
```

## Summary

**For 4GB RAM or less**: Use `qwen2.5:1.5b`
```bash
ollama pull qwen2.5:1.5b
./codegod --model qwen2.5:1.5b
```

**For 4-8GB RAM**: Use `llama3.2:3b` or `qwen2.5:3b`
```bash
ollama pull llama3.2:3b
./codegod --model llama3.2:3b
```

**For 8GB+ RAM**: Consider upgrading to `llama3.1:8b` for better quality
```bash
ollama pull llama3.1:8b
./codegod --model llama3.1:8b
```

Small models are perfect for:
- Learning and experimentation
- Simple projects
- Low-resource systems
- Fast prototyping
- CPU-only machines

Happy coding! 🚀
