# Usage Guide

Complete guide for using Code-God to build applications autonomously.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Creating Projects](#creating-projects)
3. [Best Practices](#best-practices)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Start the System

```bash
./scripts/start.sh
```

Wait for all services to become healthy (about 1-2 minutes).

### Verify System is Running

```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

---

## Creating Projects

### Basic Project

Create a simple API:

```bash
curl -X POST http://localhost:8001/build \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a REST API for managing books with CRUD operations",
    "tech_stack": {
      "backend": "FastAPI",
      "database": "PostgreSQL"
    },
    "target_platforms": ["backend"]
  }'
```

### With API Specifications

Include external API documentation:

```bash
curl -X POST http://localhost:8001/build \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Build a weather dashboard that displays current weather and forecasts",
    "api_specs": ["https://openweathermap.org/api"],
    "tech_stack": {
      "backend": "FastAPI",
      "frontend": "React + TypeScript"
    },
    "target_platforms": ["web", "backend"]
  }'
```

### Full-Stack Application

```bash
curl -X POST http://localhost:8001/build \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a social media platform with user authentication, posts, comments, likes, and real-time notifications. Include image upload and user profiles.",
    "tech_stack": {
      "backend": "FastAPI",
      "frontend": "React + TypeScript",
      "database": "PostgreSQL",
      "realtime": "WebSockets"
    },
    "target_platforms": ["web", "backend"]
  }'
```

### Mobile Application

```bash
curl -X POST http://localhost:8001/build \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Build a fitness tracking mobile app with workout logging, progress charts, and goal setting",
    "tech_stack": {
      "mobile": "React Native",
      "backend": "Node.js",
      "database": "MongoDB"
    },
    "target_platforms": ["ios", "android", "backend"]
  }'
```

---

## Monitoring Progress

### Check Build Status

```bash
curl http://localhost:8001/status
```

Response:
```json
{
  "statistics": {
    "total": 24,
    "pending": 3,
    "in_progress": 5,
    "completed": 16,
    "failed": 0,
    "blocked": 0
  },
  "project_id": "proj_abc123"
}
```

### Real-time Logs

Watch Master AI orchestrator:
```bash
docker-compose logs -f master-ai
```

Watch specific worker:
```bash
docker-compose logs -f worker-backend
```

Watch all services:
```bash
docker-compose logs -f
```

---

## Best Practices

### Writing Effective Prompts

**DO**:
- Be specific about functionality
- Mention key features explicitly
- Specify technology stack
- Include non-functional requirements (security, performance)
- Mention testing requirements

**DON'T**:
- Be too vague ("build a website")
- Contradict yourself
- Omit critical requirements
- Assume implicit behavior

### Good Prompt Examples

**Good**:
> "Create a REST API for an e-commerce platform with user authentication (JWT), product catalog with search and filtering, shopping cart management, order processing with Stripe integration, and admin dashboard. Include unit tests with 80%+ coverage and API documentation."

**Bad**:
> "Make an online shop"

### Recommended Tech Stacks

**Web Application**:
```json
{
  "backend": "FastAPI",
  "frontend": "React + TypeScript",
  "database": "PostgreSQL",
  "cache": "Redis"
}
```

**Mobile App**:
```json
{
  "mobile": "React Native",
  "backend": "Node.js",
  "database": "MongoDB"
}
```

**Microservice**:
```json
{
  "backend": "Go",
  "database": "PostgreSQL",
  "messaging": "RabbitMQ"
}
```

### Project Complexity Guidelines

| Complexity | Features | Estimated Time | Recommended Workers |
|------------|----------|----------------|---------------------|
| Simple | 1-3 endpoints, basic CRUD | 5-10 min | 2 backend |
| Medium | 5-10 endpoints, auth, tests | 15-30 min | 2 backend, 1 testing |
| Complex | Full-stack, multiple services | 30-60 min | 2-4 of each type |
| Enterprise | Microservices, advanced features | 1-2 hours | 4+ of each type |

---

## Advanced Usage

### Custom Configuration

Create a configuration file `project-config.json`:

```json
{
  "prompt": "Build a task management system",
  "tech_stack": {
    "backend": "FastAPI",
    "frontend": "React",
    "database": "PostgreSQL"
  },
  "target_platforms": ["web", "backend"],
  "requirements": {
    "authentication": "JWT",
    "test_coverage": 85,
    "code_style": "black + pylint",
    "documentation": "OpenAPI + README",
    "deployment": "Docker"
  },
  "features": [
    "User registration and login",
    "Create, update, delete tasks",
    "Assign tasks to users",
    "Task priorities and deadlines",
    "Email notifications",
    "Dashboard with statistics"
  ],
  "constraints": {
    "max_execution_time": 1800,
    "security": "OWASP top 10 compliant"
  }
}
```

Build:
```bash
curl -X POST http://localhost:8001/build \
  -H "Content-Type: application/json" \
  -d @project-config.json
```

### Using API Specifications

Provide OpenAPI/Swagger specs or documentation URLs:

```json
{
  "prompt": "Integrate with Stripe for payment processing",
  "api_specs": [
    "https://stripe.com/docs/api",
    "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.yaml"
  ],
  "tech_stack": {
    "backend": "Python + FastAPI"
  }
}
```

The system will:
1. Fetch and parse the API documentation
2. Store it in the vector database
3. Use it as context for code generation
4. Ensure correct API usage

### Iterative Development

Build in phases:

**Phase 1 - MVP**:
```bash
curl -X POST http://localhost:8001/build \
  -d '{"prompt": "Create basic user authentication API"}'
```

**Phase 2 - Add Features**:
```bash
curl -X POST http://localhost:8001/build \
  -d '{"prompt": "Add password reset and email verification to existing auth API"}'
```

**Phase 3 - Frontend**:
```bash
curl -X POST http://localhost:8001/build \
  -d '{"prompt": "Create React frontend for the authentication system"}'
```

### Platform-Specific Builds

**iOS Only**:
```json
{
  "prompt": "Build a note-taking app with local storage",
  "tech_stack": {
    "mobile": "SwiftUI"
  },
  "target_platforms": ["ios"]
}
```

**Android Only**:
```json
{
  "prompt": "Build a note-taking app with local storage",
  "tech_stack": {
    "mobile": "Kotlin + Jetpack Compose"
  },
  "target_platforms": ["android"]
}
```

**Cross-Platform**:
```json
{
  "prompt": "Build a note-taking app with cloud sync",
  "tech_stack": {
    "mobile": "Flutter",
    "backend": "Firebase"
  },
  "target_platforms": ["ios", "android"]
}
```

---

## Retrieving Results

### Get Project Files

After build completes, files are in `./projects/PROJECT_ID/`:

```bash
# List generated files
ls -R projects/PROJECT_ID/

# View specific file
cat projects/PROJECT_ID/backend/main.py
```

### Clone from Git

The system automatically commits to git:

```bash
cd projects/PROJECT_ID
git log --oneline
git show HEAD
```

### Export as Archive

```bash
tar czf my-project.tar.gz projects/PROJECT_ID/
```

---

## System Configuration

### Adjust Parallelism

Edit `.env`:
```bash
MAX_PARALLEL_TASKS=10  # Increase for faster builds
```

Restart:
```bash
docker-compose restart master-ai
```

### Change Worker Models

For better quality (slower):
```bash
WORKER_MODEL=mistral:7b
docker-compose up -d --force-recreate worker-backend
```

For faster execution (lower quality):
```bash
WORKER_MODEL=phi-3:3b
docker-compose up -d --force-recreate worker-backend
```

### Scale Workers

```bash
# More backend workers for API-heavy projects
docker-compose up -d --scale worker-backend=6

# More frontend workers for UI-heavy projects
docker-compose up -d --scale worker-frontend=4
```

---

## Troubleshooting

### Build Taking Too Long

1. Check task progress:
```bash
curl http://localhost:8001/status
```

2. Check for blocked tasks (dependency issues)

3. Increase parallelism:
```bash
MAX_PARALLEL_TASKS=10
```

4. Scale up workers:
```bash
docker-compose up -d --scale worker-backend=4
```

### Build Failed

1. Check logs:
```bash
docker-compose logs master-ai | grep ERROR
```

2. Review failed tasks in status response

3. Common causes:
   - Ambiguous prompt
   - Conflicting requirements
   - API spec fetch failure
   - Model timeout

4. Retry with clearer prompt

### Poor Code Quality

1. Use larger worker models:
```bash
WORKER_MODEL=mistral:7b
```

2. Be more specific in prompt:
```
"Include comprehensive error handling, input validation, and security best practices"
```

3. Request specific patterns:
```
"Use repository pattern for data access, dependency injection, and clean architecture"
```

### Memory Issues

1. Check Docker resources:
```bash
docker stats
```

2. Increase Docker memory limit (Docker Desktop settings)

3. Reduce parallel tasks:
```bash
MAX_PARALLEL_TASKS=2
```

4. Use smaller models:
```bash
WORKER_MODEL=phi-3:3b
```

---

## Examples

### Example 1: Blog Platform

```json
{
  "prompt": "Create a blog platform with user authentication, post creation with markdown support, comments, tags, search functionality, and RSS feed. Include admin panel for content moderation.",
  "tech_stack": {
    "backend": "FastAPI + SQLAlchemy",
    "frontend": "Next.js + TypeScript",
    "database": "PostgreSQL",
    "search": "Elasticsearch"
  },
  "target_platforms": ["web", "backend"],
  "api_specs": []
}
```

### Example 2: Real-time Chat

```json
{
  "prompt": "Build a real-time chat application with user presence, private and group chats, message history, file sharing, and typing indicators. Include push notifications.",
  "tech_stack": {
    "backend": "Node.js + Socket.io",
    "frontend": "React + TypeScript",
    "database": "MongoDB",
    "cache": "Redis"
  },
  "target_platforms": ["web", "backend"]
}
```

### Example 3: Analytics Dashboard

```json
{
  "prompt": "Create an analytics dashboard that displays user metrics, revenue charts, conversion funnels, and real-time visitor tracking. Include data export to CSV/PDF.",
  "tech_stack": {
    "backend": "Python + FastAPI",
    "frontend": "React + D3.js",
    "database": "PostgreSQL + TimescaleDB"
  },
  "target_platforms": ["web", "backend"]
}
```

---

## Tips & Tricks

### Speed Up Development

- Use simpler models for prototyping
- Build incrementally
- Reuse generated components
- Cache API documentation

### Improve Quality

- Be very specific in prompts
- Request tests explicitly
- Mention code standards
- Include security requirements
- Request documentation

### Save Costs

- Use local models when possible
- Batch related features
- Leverage caching
- Monitor token usage

### Best Results

- Clear, detailed prompts
- Realistic scope
- Standard tech stacks
- Include API specs when available
- Request comprehensive tests

---

## Next Steps

1. **Experiment**: Start with simple projects
2. **Learn**: Review generated code
3. **Iterate**: Refine based on results
4. **Scale**: Deploy to production
5. **Customize**: Extend for your needs

For more information, see:
- [Architecture Documentation](../ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [API Reference](../README.md#api-reference)
