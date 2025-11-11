# Model Configuration Guide

Code-God supports a wide range of AI models, from tiny 1B parameter models to massive 405B models.

## Configuration Methods

### 1. Environment Variables (.env file)

Create a `.env` file in the project root:

```bash
# Use a small model
PREFER_LOCAL=true
MASTER_MODEL=qwen2.5:1.5b
```

### 2. Config File (~/.codegod/config.json)

```json
{
  "model": "qwen2.5:1.5b",
  "prefer_local": true,
  "max_history": 10
}
```

### 3. Command Line

```bash
./codegod --model qwen2.5:1.5b
```

## Small Models (1B-3B) - Perfect for Testing

### Qwen 2.5 1.5B (Recommended for small)
```bash
# Install
ollama pull qwen2.5:1.5b

# Use
MASTER_MODEL=qwen2.5:1.5b ./codegod
```
- **Size**: ~900MB
- **RAM**: 1-2GB
- **Speed**: Very fast
- **Quality**: Good for simple tasks

### Qwen 2.5 3B
```bash
ollama pull qwen2.5:3b
MASTER_MODEL=qwen2.5:3b ./codegod
```
- **Size**: ~1.9GB
- **RAM**: 2-3GB
- **Speed**: Fast
- **Quality**: Better reasoning

### Llama 3.2 1B
```bash
ollama pull llama3.2:1b
MASTER_MODEL=llama3.2:1b ./codegod
```
- **Size**: ~1.3GB
- **RAM**: 1-2GB
- **Speed**: Very fast
- **Quality**: Good for basic tasks

### Llama 3.2 3B
```bash
ollama pull llama3.2:3b
MASTER_MODEL=llama3.2:3b ./codegod
```
- **Size**: ~2GB
- **RAM**: 2-4GB
- **Speed**: Fast
- **Quality**: Balanced

### Phi-3 Mini (3.8B)
```bash
ollama pull phi3:mini
MASTER_MODEL=phi3:mini ./codegod
```
- **Size**: ~2.3GB
- **RAM**: 3-4GB
- **Speed**: Fast
- **Quality**: Microsoft's compact model

### Gemma 2 2B
```bash
ollama pull gemma2:2b
MASTER_MODEL=gemma2:2b ./codegod
```
- **Size**: ~1.6GB
- **RAM**: 2-3GB
- **Speed**: Fast
- **Quality**: Google's efficient model

## Medium Models (7B-13B)

### Llama 3.1 8B
```bash
ollama pull llama3.1:8b
MASTER_MODEL=llama3.1:8b ./codegod
```
- **Size**: ~4.7GB
- **RAM**: 8GB
- **Quality**: Excellent balance

### Mistral 7B
```bash
ollama pull mistral:7b
MASTER_MODEL=mistral:7b ./codegod
```
- **Size**: ~4.1GB
- **RAM**: 8GB
- **Quality**: Strong reasoning

### DeepSeek Coder 6.7B
```bash
ollama pull deepseek-coder:6.7b
MASTER_MODEL=deepseek-coder:6.7b ./codegod
```
- **Size**: ~3.8GB
- **RAM**: 8GB
- **Quality**: Code-specialized

## Large Models (70B+)

### Llama 3.1 70B (Recommended)
```bash
ollama pull llama3.1:70b
MASTER_MODEL=llama3.1:70b ./codegod
```
- **Size**: ~40GB
- **RAM**: 48GB+
- **Quality**: Production-ready

### Qwen 2.5 72B
```bash
ollama pull qwen2.5:72b
MASTER_MODEL=qwen2.5:72b ./codegod
```
- **Size**: ~43GB
- **RAM**: 48GB+
- **Quality**: Excellent

## API Models (No local installation needed)

### Claude Sonnet 4.5
```bash
# Set in .env
ANTHROPIC_API_KEY=sk-ant-...
PREFER_LOCAL=false
MASTER_MODEL=claude-sonnet-4.5

./codegod --prefer-api --model claude-sonnet-4.5
```

### GPT-4 Turbo
```bash
# Set in .env
OPENAI_API_KEY=sk-...
PREFER_LOCAL=false
MASTER_MODEL=gpt-4-turbo-preview

./codegod --prefer-api --model gpt-4-turbo-preview
```

## Quick Start Examples

### For Low-End Systems (4GB RAM)
```bash
# Use Qwen 1.5B
echo "MASTER_MODEL=qwen2.5:1.5b" > .env
ollama pull qwen2.5:1.5b
./codegod
```

### For Mid-Range Systems (16GB RAM)
```bash
# Use Llama 3.1 8B
echo "MASTER_MODEL=llama3.1:8b" > .env
ollama pull llama3.1:8b
./codegod
```

### For High-End Systems (64GB+ RAM)
```bash
# Use Llama 3.1 70B
echo "MASTER_MODEL=llama3.1:70b" > .env
ollama pull llama3.1:70b
./codegod
```

### For API Usage (No GPU needed)
```bash
# Use Claude API
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-your-key-here
PREFER_LOCAL=false
MASTER_MODEL=claude-sonnet-4.5
EOF

./codegod --prefer-api
```

## Model Selection Priority

Code-God selects models in this order:

1. **Command line** `--model` flag (highest priority)
2. **Environment variable** `MASTER_MODEL`
3. **Config file** `~/.codegod/config.json`
4. **Auto-detection** - Scans for available Ollama models

## Performance Comparison

| Model | Size | RAM | Speed | Quality | Use Case |
|-------|------|-----|-------|---------|----------|
| qwen2.5:1.5b | 900MB | 1-2GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Testing, low-end |
| qwen2.5:3b | 1.9GB | 2-3GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Dev, learning |
| llama3.2:3b | 2GB | 2-4GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| llama3.1:8b | 4.7GB | 8GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Development |
| mistral:7b | 4.1GB | 8GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | General purpose |
| llama3.1:70b | 40GB | 48GB+ | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ | Production |
| qwen2.5:72b | 43GB | 48GB+ | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ | Production |
| claude-sonnet-4.5 | API | 0GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐⭐⭐ | Cloud, best quality |

## Checking Available Models

```bash
# List installed Ollama models
ollama list

# See what Code-God will use
./codegod --model auto
# (Check startup message)
```

## Troubleshooting

### Model Not Found
```bash
# Pull the model first
ollama pull qwen2.5:1.5b
```

### Out of Memory
```bash
# Use a smaller model
MASTER_MODEL=qwen2.5:1.5b ./codegod
```

### Slow Performance
```bash
# GPU required for good performance
# Or use smaller model
# Or use API
./codegod --prefer-api --model claude-sonnet-4.5
```

## Recommendations

- **Just testing?** → `qwen2.5:1.5b`
- **Learning/Development?** → `llama3.1:8b`
- **Production projects?** → `llama3.1:70b` or `claude-sonnet-4.5`
- **No GPU?** → `qwen2.5:3b` or API models
- **Best quality?** → `claude-sonnet-4.5` (API)
