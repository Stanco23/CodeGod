```
# Multi-Agent System Architecture

## Overview

True multi-agent system with:
- **Master Orchestrator** - Breaks down problems, coordinates agents
- **Specialized Worker Agents** - Parallel execution with reasoning
- **Containerized Deployment** - vLLM for fast inference
- **Memory Management** - Wipeable, checkpointable state
- **Reasoning at Every Step** - Observe → Reason → Plan → Act → Validate

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MASTER ORCHESTRATOR                   │
│  • Problem decomposition                                 │
│  • Task assignment                                       │
│  • Coordination                                          │
│  • Memory management                                     │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────┐       ┌──────────────┐
│   REDIS      │       │    vLLM      │
│ Task Queue   │       │   Server     │
│ & Memory     │       │ (LLama 70B)  │
└──────┬───────┘       └──────┬───────┘
       │                      │
       │                      │
   ┌───┴───────┬──────────┬───┴──────┬──────────┐
   │           │          │          │          │
   ▼           ▼          ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│Backend│  │Backend│  │Front │  │DevOps│  │Test  │
│Agent  │  │Agent  │  │end   │  │Agent │  │Agent │
│  #1   │  │  #2   │  │Agent │  │      │  │  #1  │
└───────┘  └───────┘  └───────┘  └───────┘  └───────┘
   │           │          │          │          │
   └───────────┴──────────┴──────────┴──────────┘
                    │
                    ▼
             [Project Output]
```

## Agent Roles

### Master Orchestrator
**Responsibilities:**
- Analyze high-level problems
- Decompose into atomic tasks
- Assign to specialized agents
- Coordinate execution
- Resolve dependencies
- Wipe memory when stuck

**Reasoning:**
```python
Problem: "Flask not found"
→ Decompose:
   - Task 1: Check venv exists (Backend Agent)
   - Task 2: Install Flask (Backend Agent) [depends on Task 1]
   - Task 3: Verify import (Testing Agent) [depends on Task 2]
→ Execute in order, wait for completion
```

### Backend Agent
**Specialization:** Python, Flask, Django, FastAPI, dependencies, venv

**Reasoning Loop:**
1. **Observe:** Check venv, installed packages, requirements.txt
2. **Reason:** "Flask not in pip list, but in requirements.txt"
3. **Plan:** ["install flask via pip", "verify import"]
4. **Act:** Execute pip install
5. **Validate:** Try importing flask - SUCCESS

**Actions:**
- install_dependency
- check_file_exists
- run_command
- update_requirements
- create_venv

### Frontend Agent
**Specialization:** React, Vue, npm, package.json, webpack

**Reasoning Loop:**
1. **Observe:** Check node_modules, package.json, src files
2. **Reason:** "Entry point index.js not in root, but in src/"
3. **Plan:** ["update package.json main field", "run npm install"]
4. **Act:** Edit package.json, run npm install
5. **Validate:** Check package.json valid, node_modules exists

**Actions:**
- npm_install
- npm_run
- update_package_json
- check_entry_point
- build_frontend

### DevOps Agent
**Specialization:** Docker, deployment, environment, CI/CD

**Actions:**
- create_dockerfile
- build_image
- run_container
- setup_env
- deploy

### Testing Agent
**Specialization:** Running tests, validation, debugging

**Reasoning Loop:**
1. **Observe:** Run tests, collect output
2. **Reason:** "3 tests failed, all import errors"
3. **Plan:** ["identify missing deps", "trigger Backend agent"]
4. **Act:** Report to Master
5. **Validate:** Re-run tests

**Actions:**
- run_tests
- validate_output
- check_syntax
- debug_error

## Reasoning at Every Step

Every agent follows: **Observe → Reason → Plan → Act → Validate**

### Example: Backend Agent Installing Flask

**Observe:**
```json
{
  "venv_exists": true,
  "dependencies_installed": ["pip", "setuptools", "wheel"],
  "requirements_file": ["flask", "sqlalchemy"],
  "findings": [
    "Virtual environment: exists",
    "Installed packages: 3",
    "requirements.txt: 2 dependencies"
  ]
}
```

**Reason:**
```json
{
  "problem_analysis": "Flask is listed in requirements.txt but not installed in venv",
  "root_causes": ["pip install not run", "requirements.txt not processed"],
  "approach": "Install Flask using pip in the virtual environment",
  "risks": ["Network failure", "Version conflicts"],
  "success_criteria": "Flask importable via python -c 'import flask'",
  "conclusion": "Need to run pip install flask in venv"
}
```

**Plan:**
```json
{
  "steps": [
    {
      "action": "install_dependency",
      "target": "flask",
      "method": "pip",
      "expected": "flask importable"
    }
  ],
  "estimated_duration": "30 seconds",
  "confidence": 0.95
}
```

**Act:**
```bash
/path/to/venv/bin/pip install flask
✓ Successfully installed Flask-3.0.0
```

**Validate:**
```bash
/path/to/venv/bin/python -c "import flask"
✓ Module imported successfully
```

**Result:**
```json
{
  "success": true,
  "output": "Flask successfully installed and verified",
  "reasoning": [
    "Observed: Flask missing from venv",
    "Reasoning: Need to install via pip",
    "Plan: 1 step - install_dependency"
  ],
  "actions_taken": [
    {
      "step": 1,
      "action": "install_dependency",
      "success": true,
      "output": "Installed flask"
    }
  ],
  "validation": {
    "valid": true,
    "output": "Module flask successfully installed and importable"
  }
}
```

## Memory Management

### Per-Agent Memory
```python
AgentMemory:
  - agent_role: BACKEND
  - current_task: Task(id='task_002', description='Install Flask')
  - completed_tasks: ['task_001', 'task_002']
  - knowledge: {"venv_path": "/path/to/venv", "python_version": "3.11"}
  - observations: ["Flask missing", "requirements.txt exists"]
  - reasoning_history: [...]
  - blocked_on: None
```

### Operations

**Wipe Memory:**
```python
orchestrator.wipe_all_memory()
```
- Clears current task
- Clears completed tasks
- Clears observations
- Clears reasoning history
- Keeps: Role, core knowledge

**Checkpoint:**
```python
checkpoint = agent.memory.checkpoint()
# Save to disk or Redis
```

**Restore:**
```python
agent.memory.restore(checkpoint)
```

## Deployment

### Local Development
```bash
# Use local Ollama
python codegod.py --build "Create a Flask app"
```

### Multi-Agent with vLLM (Production)
```bash
# Quick start - all services
./start-agents.sh up

# View logs
./start-agents.sh logs
./start-agents.sh logs backend-agent  # Specific agent

# Monitor task queue and status
./start-agents.sh status
# Or open browser: http://localhost:8081 (Redis Commander)

# Wipe all agent memory
./start-agents.sh wipe

# Scale agents
./start-agents.sh scale backend-agent 4

# Stop system
./start-agents.sh down

# Or use docker-compose directly:
docker-compose -f docker-compose.agents.yml up -d
docker-compose -f docker-compose.agents.yml logs -f
docker-compose -f docker-compose.agents.yml down
```

### Architecture Benefits

1. **True Parallelism:** Multiple agents work simultaneously
2. **Specialization:** Each agent is expert in its domain
3. **Scalable:** Add more agent replicas as needed
4. **Fast Inference:** vLLM provides 10-20x faster inference than standard
5. **Memory Management:** Can wipe and reset when stuck
6. **Reasoning:** Every step is thought through
7. **Validation:** Every action is verified

### Task Execution Flow

```
1. User: "Build Flask + React app"

2. Master: Decompose
   → Task 1: Setup backend venv (Backend Agent)
   → Task 2: Install Flask deps (Backend Agent, depends on 1)
   → Task 3: Setup frontend (Frontend Agent, parallel)
   → Task 4: Install npm deps (Frontend Agent, depends on 3)
   → Task 5: Test backend (Testing Agent, depends on 2)
   → Task 6: Test frontend (Testing Agent, depends on 4)

3. Execute:
   [Task 1, Task 3] → Execute in parallel (no dependencies)
   ↓
   [Task 2, Task 4] → Execute in parallel (dependencies met)
   ↓
   [Task 5, Task 6] → Execute in parallel (dependencies met)

4. Each task: Observe → Reason → Plan → Act → Validate

5. If stuck: Wipe memory, retry with fresh reasoning
```

## Configuration

### Environment Variables
```bash
# Agent role
AGENT_ROLE=backend|frontend|devops|testing|master

# vLLM endpoint
VLLM_API_URL=http://vllm-server:8000/v1

# Redis for task queue
REDIS_URL=redis://redis:6379/0

# Logging
LOG_LEVEL=INFO|DEBUG
```

### vLLM Models

**Current (RTX 3090 24GB):**
```bash
--model meta-llama/Meta-Llama-3.1-8B-Instruct  # 8B, fits in 24GB
```

**Other options for 24GB VRAM:**
```bash
--model mistralai/Mistral-7B-Instruct-v0.3     # 7B, fast
--model Qwen/Qwen2.5-7B-Instruct               # 7B, good at code
--model google/gemma-2-9b-it                   # 9B, Google model
```

**For larger GPUs (40GB+ VRAM):**
```bash
--model meta-llama/Meta-Llama-3-70B-Instruct   # 70B, requires 2x A100
--model meta-llama/Meta-Llama-3.1-405B-Instruct # 405B, requires 8x A100
```

## Monitoring

### Redis Commander
- URL: http://localhost:8081
- View task queue in real-time
- Monitor agent memory
- See task dependencies

### Logs
```bash
# All agents
docker-compose -f docker-compose.agents.yml logs -f

# Specific agent
docker-compose -f docker-compose.agents.yml logs -f backend-agent

# Master only
docker-compose -f docker-compose.agents.yml logs -f master-orchestrator
```

### Metrics
```bash
# Task completion rate
redis-cli --raw HGETALL codegod:metrics:tasks

# Agent status
redis-cli --raw HGETALL codegod:agents:status
```

## Comparison: Old vs New

| Aspect | Old (Single Agent) | New (Multi-Agent) |
|--------|-------------------|-------------------|
| Execution | Sequential | Parallel |
| Reasoning | Pattern matching | Step-by-step reasoning |
| Specialization | None | Domain experts |
| Stuck handling | Infinite loop | Wipe memory, retry |
| Validation | Hope it works | Verify every action |
| Scalability | 1 instance | N replicas per agent type |
| Inference Speed | Standard | vLLM (10-20x faster) |
| Memory | Can't wipe | Checkpoint & wipe |

## Next Steps

1. **Deploy vLLM:**
   ```bash
   docker-compose -f docker-compose.agents.yml up -d vllm-server
   ```

2. **Start agents:**
   ```bash
   docker-compose -f docker-compose.agents.yml up -d
   ```

3. **Build project:**
   ```bash
   python codegod.py --build "Create Flask + React app" --use-agents
   ```

4. **Monitor:**
   - Open http://localhost:8081 for Redis Commander
   - Watch `docker-compose logs -f`

5. **If stuck:**
   ```bash
   # Wipe all memory
   python -c "from multi_agent_system import orchestrator; orchestrator.wipe_all_memory()"

   # Or restart fresh
   docker-compose -f docker-compose.agents.yml restart
   ```

## Success Criteria

✅ Each agent reasons before acting
✅ Actions are validated before moving on
✅ Parallel execution when possible
✅ Can wipe memory when stuck
✅ Each agent is a domain expert
✅ Fast inference with vLLM
✅ Scalable with Docker Compose
```
