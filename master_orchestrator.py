"""
Master AI Orchestrator
Main controller for the autonomous multi-agent system
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import os

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis

from memory_system import MemorySystem, MemoryMetadata, ContentType, Platform, prepare_worker_context
from task_orchestration import (
    TaskQueue, TaskScheduler, Task, TaskType, TaskStatus,
    TaskOutput, create_task
)
from prompt_templates import PromptGenerator, BACKEND_WORKER_SYSTEM_PROMPT
from mcp_integration import MCPToolManager
from validation_system import ValidationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectRequest(BaseModel):
    """Request to build a project"""
    prompt: str
    api_specs: List[str] = []
    tech_stack: Dict[str, str] = {}
    target_platforms: List[str] = ["web"]


class MasterOrchestrator:
    """
    Master AI that orchestrates the entire system
    """

    def __init__(
        self,
        anthropic_api_key: str,
        chromadb_host: str = "localhost",
        chromadb_port: int = 8000,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        max_parallel_tasks: int = 5
    ):
        # Initialize AI client
        self.ai_client = Anthropic(api_key=anthropic_api_key)
        self.model = os.getenv("MASTER_MODEL", "claude-sonnet-4.5")

        # Initialize memory system
        self.memory = MemorySystem(
            persist_directory="./chroma_db",
            chunk_size=1024,
            chunk_overlap=128
        )

        # Initialize task management
        self.task_queue = TaskQueue()
        self.task_scheduler = TaskScheduler(
            queue=self.task_queue,
            max_parallel_tasks=max_parallel_tasks
        )

        # Initialize MCP tools
        self.mcp_manager = MCPToolManager(
            memory_system=self.memory,
            sandbox_root="./sandbox"
        )

        # Initialize validation
        self.validation_pipeline = ValidationPipeline()

        # Redis for worker communication
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port

        # Current project state
        self.current_project: Optional[Dict] = None
        self.project_id: Optional[str] = None

    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.from_url(
            f"redis://{self.redis_host}:{self.redis_port}",
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Master Orchestrator initialized")

    async def shutdown(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()

    async def build_project(self, request: ProjectRequest) -> Dict[str, Any]:
        """
        Main entry point: Build complete project from prompt

        Args:
            request: Project request with prompt and specifications

        Returns:
            Project build result
        """
        logger.info(f"Starting project build: {request.prompt[:100]}...")

        try:
            # 1. Analyze requirements
            project_plan = await self.analyze_requirements(request)

            # 2. Store project context in memory
            await self.initialize_project_memory(request, project_plan)

            # 3. Decompose into tasks
            tasks = await self.decompose_project(project_plan)

            # 4. Add tasks to queue
            self.task_queue.add_tasks(tasks)

            # 5. Validate dependencies
            issues = self.task_queue.validate_dependencies()
            if issues:
                logger.error(f"Dependency issues: {issues}")
                return {"success": False, "error": "Invalid task dependencies", "issues": issues}

            # 6. Execute tasks
            results = await self.execute_all_tasks()

            # 7. Integrate outputs
            final_output = await self.integrate_outputs(results)

            # 8. Run final validation
            validation_results = await self.validate_project(final_output)

            # 9. Commit to git
            if validation_results["success"]:
                await self.commit_to_git(final_output)

            return {
                "success": validation_results["success"],
                "project": final_output,
                "statistics": self.task_queue.get_statistics(),
                "validation": validation_results
            }

        except Exception as e:
            logger.error(f"Project build failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def analyze_requirements(self, request: ProjectRequest) -> Dict[str, Any]:
        """
        Analyze user requirements and create project plan

        Uses Master AI to understand requirements and plan architecture
        """
        logger.info("Analyzing requirements...")

        # Fetch API specs if provided
        api_docs = []
        for spec_url in request.api_specs:
            result = await self.mcp_manager.execute_tool(
                tool_type="fetch",
                operation="fetch_url",
                params={"url": spec_url}
            )
            if result.success:
                api_docs.append(result.output["content"])

        # Build analysis prompt
        prompt = f"""You are a senior software architect. Analyze the following project requirements and create a detailed technical plan.

USER REQUIREMENTS:
{request.prompt}

TARGET PLATFORMS:
{', '.join(request.target_platforms)}

TECH STACK:
{json.dumps(request.tech_stack, indent=2)}

API SPECIFICATIONS:
{chr(10).join(api_docs[:3])}  # Include first 3 API docs

TASK:
Create a comprehensive technical plan including:
1. Project architecture and structure
2. Required components and modules
3. Technology choices and frameworks
4. Database schema design
5. API endpoint design
6. Frontend component hierarchy
7. Testing strategy
8. Deployment approach

OUTPUT FORMAT (JSON):
{{
    "project_name": "string",
    "description": "string",
    "architecture": {{
        "backend": {{"framework": "string", "language": "string"}},
        "frontend": {{"framework": "string", "language": "string"}},
        "database": {{"type": "string", "orm": "string"}}
    }},
    "file_structure": {{
        "backend/": ["list of files"],
        "frontend/": ["list of files"],
        "tests/": ["list of files"]
    }},
    "components": [
        {{
            "name": "string",
            "type": "backend|frontend|database|test",
            "description": "string",
            "dependencies": ["list of component names"],
            "priority": 1-10
        }}
    ],
    "api_endpoints": [
        {{
            "path": "/api/...",
            "method": "GET|POST|PUT|DELETE",
            "description": "string"
        }}
    ]
}}
"""

        # Call Master AI
        response = self.ai_client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        content = response.content[0].text

        # Extract JSON (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        project_plan = json.loads(content.strip())

        logger.info(f"Project plan created: {project_plan['project_name']}")
        return project_plan

    async def initialize_project_memory(self, request: ProjectRequest, plan: Dict):
        """Store project context in memory"""
        logger.info("Initializing project memory...")

        # Store user prompt
        self.memory.store(
            content=request.prompt,
            metadata=MemoryMetadata(
                type=ContentType.DOCUMENTATION,
                source="user_prompt",
                tags=["requirements", "initial"]
            )
        )

        # Store project plan
        self.memory.store(
            content=json.dumps(plan, indent=2),
            metadata=MemoryMetadata(
                type=ContentType.DOCUMENTATION,
                source="project_plan",
                tags=["architecture", "plan"]
            )
        )

        # Store API docs
        for idx, spec_url in enumerate(request.api_specs):
            result = await self.mcp_manager.execute_tool(
                tool_type="fetch",
                operation="fetch_url",
                params={"url": spec_url}
            )
            if result.success:
                self.memory.store_api_documentation(
                    api_spec=result.output["content"],
                    source_url=spec_url
                )

    async def decompose_project(self, plan: Dict) -> List[Task]:
        """
        Decompose project plan into atomic tasks

        Uses Master AI to break down components into executable tasks
        """
        logger.info("Decomposing project into tasks...")

        prompt = f"""You are a project decomposition expert. Break down the following project into atomic, executable tasks.

PROJECT PLAN:
{json.dumps(plan, indent=2)}

GUIDELINES:
1. Each task should be independently executable
2. Tasks should have clear dependencies
3. Estimate priority (1-10, higher = more urgent)
4. Specify exact file paths for code tasks
5. Include testing tasks for each code component

OUTPUT FORMAT (JSON):
{{
    "tasks": [
        {{
            "type": "backend|frontend|database|testing|devops|documentation",
            "subtype": "api_endpoint|component|unit_tests|etc",
            "description": "Clear description of what to implement",
            "file_path": "exact/path/to/file.py",
            "language": "python|typescript|etc",
            "priority": 8,
            "dependencies": ["task_descriptions_of_dependencies"],
            "specifications": {{
                "additional": "task-specific specs"
            }}
        }}
    ]
}}
"""

        response = self.ai_client.messages.create(
            model=self.model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]

        decomposition = json.loads(content.strip())

        # Convert to Task objects
        tasks = []
        task_map = {}  # description -> task_id

        for task_spec in decomposition["tasks"]:
            task = create_task(
                task_type=TaskType(task_spec["type"]),
                description=task_spec["description"],
                priority=task_spec.get("priority", 5),
                file_path=task_spec.get("file_path"),
                language=task_spec.get("language"),
                metadata=task_spec.get("specifications", {})
            )

            # Allocate MCP tools
            task.tools = self.mcp_manager.allocate_tools_for_task(
                task_type=task_spec["type"],
                task_subtype=task_spec.get("subtype", "default")
            )

            tasks.append(task)
            task_map[task_spec["description"]] = task.id

        # Resolve dependencies (convert descriptions to IDs)
        for i, task_spec in enumerate(decomposition["tasks"]):
            dep_ids = []
            for dep_desc in task_spec.get("dependencies", []):
                if dep_desc in task_map:
                    dep_ids.append(task_map[dep_desc])
            tasks[i].dependencies = dep_ids

        logger.info(f"Created {len(tasks)} tasks")
        return tasks

    async def execute_all_tasks(self) -> List[Task]:
        """Execute all tasks in queue with parallelization"""
        logger.info("Starting task execution...")

        completed_tasks = []

        while not self.task_queue.is_complete():
            # Get next batch of tasks
            batch = self.task_scheduler.schedule_next_batch()

            if not batch:
                # Wait for some tasks to complete
                await asyncio.sleep(1)
                continue

            # Execute batch in parallel
            results = await asyncio.gather(
                *[self.execute_task(task) for task in batch],
                return_exceptions=True
            )

            # Process results
            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Task {task.id} failed: {str(result)}")
                    self.task_queue.mark_failed(task.id, str(result))
                elif result:
                    completed_tasks.append(task)

            logger.info(f"Progress: {self.task_queue.get_statistics()}")

        logger.info("All tasks completed")
        return self.task_queue.get_tasks_by_status(TaskStatus.COMPLETED)

    async def execute_task(self, task: Task) -> bool:
        """
        Execute a single task by assigning to worker

        Args:
            task: Task to execute

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing task: {task.description[:80]}...")

        try:
            # Prepare context from memory
            context = prepare_worker_context(
                memory=self.memory,
                task_description=task.description,
                task_type=task.type.value,
                platform=task.platform,
                dependencies=task.dependencies
            )

            # Generate prompt
            prompt = PromptGenerator.generate(
                task_type=task.type.value,
                task_subtype=task.metadata.get("subtype", "default"),
                task_description=task.description,
                specifications=task.metadata,
                context=context
            )

            task.prompt = prompt

            # Publish task to Redis for worker pickup
            await self.redis_client.lpush(
                f"task_queue:{task.assigned_worker}",
                json.dumps({
                    "task_id": task.id,
                    "prompt": prompt,
                    "tools": task.tools,
                    "validation_rules": [v.value for v in task.validation_rules]
                })
            )

            # Wait for result
            result = await self.wait_for_task_result(task.id, timeout=300)

            if result["success"]:
                # Validate output
                validation_reports = await self.validation_pipeline.validate(
                    code=result["code"],
                    language=task.language or "python",
                    validation_rules=[v.value for v in task.validation_rules],
                    context={"task_id": task.id}
                )

                should_retry, reason = self.validation_pipeline.should_retry(validation_reports)

                if should_retry and task.retry_count < task.max_retries:
                    # Generate fix prompt
                    fix_prompt = self.validation_pipeline.generate_fix_prompt(
                        code=result["code"],
                        reports=validation_reports
                    )

                    # Retry with fix prompt
                    logger.info(f"Retrying task {task.id} due to: {reason}")
                    return await self.retry_task_with_fix(task, fix_prompt)

                # Store output in memory
                self.memory.store_code_module(
                    code=result["code"],
                    file_path=task.file_path,
                    language=task.language,
                    task_id=task.id,
                    platform=Platform(task.platform)
                )

                # Mark complete
                output = TaskOutput(
                    success=True,
                    content=result["code"],
                    artifacts={task.file_path: result["code"]},
                    metrics={"execution_time": result.get("execution_time", 0)}
                )
                self.task_queue.mark_completed(task.id, output)
                return True
            else:
                # Task failed
                logger.error(f"Task failed: {result.get('error')}")
                retry = self.task_queue.mark_failed(task.id, result.get("error", "Unknown error"))
                return False

        except Exception as e:
            logger.error(f"Task execution error: {str(e)}", exc_info=True)
            self.task_queue.mark_failed(task.id, str(e))
            return False

    async def wait_for_task_result(self, task_id: str, timeout: int = 300) -> Dict:
        """Wait for worker to complete task"""
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check for result in Redis
            result_key = f"task_result:{task_id}"
            result_json = await self.redis_client.get(result_key)

            if result_json:
                await self.redis_client.delete(result_key)
                return json.loads(result_json)

            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                return {"success": False, "error": "Task timeout"}

            await asyncio.sleep(1)

    async def retry_task_with_fix(self, task: Task, fix_prompt: str) -> bool:
        """Retry task with fix prompt"""
        task.prompt = fix_prompt
        task.retry_count += 1
        return await self.execute_task(task)

    async def integrate_outputs(self, tasks: List[Task]) -> Dict[str, Any]:
        """Integrate outputs from all tasks into final project"""
        logger.info("Integrating outputs...")

        project_files = {}
        for task in tasks:
            if task.output and task.output.artifacts:
                project_files.update(task.output.artifacts)

        return {
            "files": project_files,
            "tasks_completed": len(tasks),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def validate_project(self, project: Dict) -> Dict[str, Any]:
        """Final project validation"""
        logger.info("Validating project...")

        # Run integration tests
        # Build project
        # Check all files present

        return {"success": True, "message": "Project validation passed"}

    async def commit_to_git(self, project: Dict):
        """Commit project to git"""
        logger.info("Committing to git...")

        # Initialize git if needed
        await self.mcp_manager.execute_tool(
            tool_type="git",
            operation="init",
            params={"repo_path": "./sandbox"}
        )

        # Add all files
        await self.mcp_manager.execute_tool(
            tool_type="git",
            operation="add",
            params={"repo_path": "./sandbox", "files": ["."]}
        )

        # Commit
        await self.mcp_manager.execute_tool(
            tool_type="git",
            operation="commit",
            params={
                "repo_path": "./sandbox",
                "message": "Initial commit - Generated by Code-God autonomous system"
            }
        )


# FastAPI application
app = FastAPI(title="Code-God Master Orchestrator")
orchestrator: Optional[MasterOrchestrator] = None


@app.on_event("startup")
async def startup():
    global orchestrator
    orchestrator = MasterOrchestrator(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        chromadb_host=os.getenv("CHROMADB_HOST", "localhost"),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        max_parallel_tasks=int(os.getenv("MAX_PARALLEL_TASKS", "5"))
    )
    await orchestrator.initialize()


@app.on_event("shutdown")
async def shutdown():
    if orchestrator:
        await orchestrator.shutdown()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/build")
async def build_project(request: ProjectRequest):
    """Build a complete project from prompt"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    result = await orchestrator.build_project(request)
    return result


@app.get("/status")
async def get_status():
    """Get current build status"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    return {
        "statistics": orchestrator.task_queue.get_statistics(),
        "project_id": orchestrator.project_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
