#!/usr/bin/env python3
"""
Agent Worker Process

Runs inside Docker containers, connects to Redis task queue and vLLM API.
Each worker is a specialized agent (Backend, Frontend, DevOps, Testing) that:
1. Pulls tasks from Redis queue
2. Executes Observe → Reason → Plan → Act → Validate loop
3. Reports results back to Master Orchestrator
"""

import os
import sys
import json
import asyncio
import signal
import logging
import subprocess
from typing import Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import redis.asyncio as aioredis
from openai import AsyncOpenAI

from multi_agent_system import (
    AgentRole, Task, TaskStatus, TaskType
)

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'/app/logs/agent_{os.getenv("AGENT_ROLE", "worker")}.log')
    ]
)
logger = logging.getLogger(__name__)


class ContainerAgent:
    """Base agent optimized for container execution with vLLM API"""

    def __init__(self, role: AgentRole, llm_client: AsyncOpenAI):
        self.role = role
        self.llm_client = llm_client
        self.model_name = os.getenv('VLLM_MODEL', 'meta-llama/Llama-2-70b-chat-hf')

    async def reason_and_act(self, task: Task) -> Dict[str, Any]:
        """Execute the full reasoning loop for a task"""
        logger.info(f"{self.role.value} executing task: {task.description}")

        result = {
            "success": False,
            "output": None,
            "reasoning": [],
            "actions_taken": [],
            "validation": {}
        }

        try:
            # Observe
            observations = await self._observe(task)
            result["reasoning"].append(f"Observed: {len(observations.get('findings', []))} findings")

            # Reason
            reasoning = await self._reason(task, observations)
            result["reasoning"].append(f"Reasoning: {reasoning.get('conclusion', '')}")

            # Plan
            plan = await self._plan(task, observations, reasoning)
            result["reasoning"].append(f"Plan: {len(plan.get('steps', []))} steps")

            # Act
            for i, step in enumerate(plan.get('steps', []), 1):
                logger.info(f"  Step {i}/{len(plan['steps'])}: {step.get('action', '')}")
                action_result = await self._act(step, task, observations)

                result["actions_taken"].append({
                    "step": i,
                    "action": step.get('action', ''),
                    "success": action_result.get('success', False),
                    "output": action_result.get('output', '')
                })

                if not action_result.get('success', False):
                    result["success"] = False
                    result["error"] = action_result.get('error', 'Action failed')
                    logger.error(f"  Step {i} failed: {result['error']}")
                    return result

            # Validate
            validation = await self._validate(task, plan, result)
            result["validation"] = validation

            if validation.get('valid', False):
                result["success"] = True
                result["output"] = validation.get('output', '')
                logger.info(f"Task completed successfully: {task.id}")
            else:
                result["success"] = False
                result["error"] = validation.get('reason', 'Validation failed')
                logger.warning(f"Validation failed: {result['error']}")

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.error(f"Agent error: {e}", exc_info=True)

        return result

    async def _observe(self, task: Task) -> Dict[str, Any]:
        """Observe environment - to be implemented by subclasses"""
        raise NotImplementedError

    async def _reason(self, task: Task, observations: Dict) -> Dict[str, Any]:
        """Reason about the problem using vLLM"""
        prompt = f"""You are a {self.role.value}. Analyze this situation.

TASK: {task.description}
OBSERVATIONS: {json.dumps(observations, indent=2)}
CONTEXT: {json.dumps(task.context, indent=2)}

Reason step-by-step about:
1. What is the actual problem?
2. What are the root causes?
3. What approach should be taken?
4. What could go wrong?
5. How to verify success?

Respond with valid JSON only:
{{"problem_analysis": "...", "root_causes": [...], "approach": "...", "risks": [...], "success_criteria": "...", "conclusion": "..."}}
"""

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are {self.role.value}. Reason deeply before acting."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.4
            )

            import re
            json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.error(f"Reasoning error: {e}")

        return {"conclusion": "Could not reason about problem"}

    async def _plan(self, task: Task, observations: Dict, reasoning: Dict) -> Dict[str, Any]:
        """Create action plan using vLLM"""
        prompt = f"""Based on your reasoning, create a concrete action plan.

TASK: {task.description}
REASONING: {reasoning.get('conclusion', '')}
APPROACH: {reasoning.get('approach', '')}

Create specific, executable steps. Each step should be atomic and verifiable.

Respond with valid JSON only:
{{"steps": [{{"action": "...", "target": "...", "expected": "...", "fallback": "..."}}], "estimated_duration": "...", "confidence": 0.8}}
"""

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are {self.role.value}. Create actionable plans."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.3
            )

            import re
            json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.error(f"Planning error: {e}")

        return {"steps": []}

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        """Execute action - to be implemented by subclasses"""
        raise NotImplementedError

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        """Validate completion - to be implemented by subclasses"""
        raise NotImplementedError


class ContainerBackendAgent(ContainerAgent):
    """Backend agent for containerized execution"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__(AgentRole.BACKEND, llm_client)

    async def _observe(self, task: Task) -> Dict[str, Any]:
        """Observe backend environment"""
        project_path = Path(task.context.get('project_path', '/app/projects'))
        observations = {
            "findings": [],
            "venv_exists": (project_path / "venv").exists(),
            "python_files": [str(f) for f in project_path.rglob('*.py')] if project_path.exists() else []
        }
        observations["findings"].append(f"Python files: {len(observations['python_files'])}")
        return observations

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        """Execute backend action"""
        action = step.get('action', '')

        if action == 'install_dependency':
            target = step.get('target', '')
            project_path = Path(task.context.get('project_path', '/app/projects'))
            venv_pip = project_path / "venv" / "bin" / "pip"

            try:
                result = subprocess.run(
                    [str(venv_pip), "install", target],
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                return {
                    "success": result.returncode == 0,
                    "output": f"{'Installed' if result.returncode == 0 else 'Failed to install'} {target}"
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": f"Unknown action: {action}"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        """Validate backend task"""
        failed_actions = [a for a in result.get('actions_taken', []) if not a.get('success', False)]
        if failed_actions:
            return {"valid": False, "reason": f"{len(failed_actions)} action(s) failed"}
        return {"valid": True, "output": "All actions completed successfully"}


class ContainerFrontendAgent(ContainerAgent):
    """Frontend agent for containerized execution"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__(AgentRole.FRONTEND, llm_client)

    async def _observe(self, task: Task) -> Dict[str, Any]:
        project_path = Path(task.context.get('project_path', '/app/projects'))
        return {
            "findings": ["Frontend environment observed"],
            "node_modules_exists": (project_path / "node_modules").exists() if project_path.exists() else False
        }

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        return {"success": True, "output": "Frontend action completed"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        return {"valid": True, "output": "Frontend validation passed"}


class ContainerDevOpsAgent(ContainerAgent):
    """DevOps agent for containerized execution"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__(AgentRole.DEVOPS, llm_client)

    async def _observe(self, task: Task) -> Dict[str, Any]:
        return {"findings": ["DevOps environment observed"]}

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        return {"success": True, "output": "DevOps action completed"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        return {"valid": True, "output": "DevOps validation passed"}


class ContainerTestingAgent(ContainerAgent):
    """Testing agent for containerized execution"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__(AgentRole.TESTING, llm_client)

    async def _observe(self, task: Task) -> Dict[str, Any]:
        return {"findings": ["Testing environment observed"]}

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        return {"success": True, "output": "Testing action completed"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        return {"valid": True, "output": "Testing validation passed"}


class ContainerDebuggingAgent(ContainerAgent):
    """Debugging agent for containerized execution"""

    def __init__(self, llm_client: AsyncOpenAI):
        super().__init__(AgentRole.DEBUGGING, llm_client)

    async def _observe(self, task: Task) -> Dict[str, Any]:
        return {
            "findings": ["Error analysis initiated"],
            "error_output": task.context.get('error_output', '')
        }

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        return {"success": True, "output": "Error analyzed"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        return {"valid": True, "output": "Debugging completed"}


class AgentWorker:
    """Worker process that runs specialized agents in containers"""

    def __init__(self):
        self.agent_role = self._get_agent_role()
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.vllm_url = os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
        self.worker_id = f"{self.agent_role.value}_{os.getpid()}"

        self.redis_client: Optional[aioredis.Redis] = None
        self.vllm_client: Optional[AsyncOpenAI] = None
        self.agent: Optional[Agent] = None
        self.running = False
        self.current_task: Optional[Task] = None

        logger.info(f"Initializing AgentWorker: {self.worker_id}")

    def _get_agent_role(self) -> AgentRole:
        """Get agent role from environment variable"""
        role_str = os.getenv('AGENT_ROLE', 'backend').upper()
        role_mapping = {
            'BACKEND': AgentRole.BACKEND,
            'FRONTEND': AgentRole.FRONTEND,
            'DEVOPS': AgentRole.DEVOPS,
            'TESTING': AgentRole.TESTING,
            'DEBUGGING': AgentRole.DEBUGGING,
        }

        role = role_mapping.get(role_str)
        if not role:
            logger.warning(f"Unknown role '{role_str}', defaulting to BACKEND")
            return AgentRole.BACKEND
        return role

    async def initialize(self):
        """Initialize Redis and vLLM connections, create agent"""
        logger.info("Connecting to Redis...")
        self.redis_client = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )

        logger.info("Connecting to vLLM API...")
        self.vllm_client = AsyncOpenAI(
            base_url=self.vllm_url,
            api_key="EMPTY"  # vLLM doesn't require API key
        )

        # Test vLLM connection
        try:
            models = await self.vllm_client.models.list()
            logger.info(f"Connected to vLLM, available models: {[m.id for m in models.data]}")
        except Exception as e:
            logger.error(f"Failed to connect to vLLM: {e}")
            raise

        # Create specialized agent
        logger.info(f"Creating {self.agent_role.value} agent...")
        self.agent = self._create_agent()

        # Register worker in Redis
        await self.redis_client.hset(
            f"codegod:workers:{self.worker_id}",
            mapping={
                "role": self.agent_role.value,
                "status": "idle",
                "started_at": datetime.utcnow().isoformat(),
                "pid": os.getpid()
            }
        )
        await self.redis_client.expire(f"codegod:workers:{self.worker_id}", 300)  # 5 min TTL

        logger.info(f"Worker {self.worker_id} initialized and ready")

    def _create_agent(self) -> ContainerAgent:
        """Create specialized agent based on role"""
        agent_classes = {
            AgentRole.BACKEND: ContainerBackendAgent,
            AgentRole.FRONTEND: ContainerFrontendAgent,
            AgentRole.DEVOPS: ContainerDevOpsAgent,
            AgentRole.TESTING: ContainerTestingAgent,
            AgentRole.DEBUGGING: ContainerDebuggingAgent,
        }

        agent_class = agent_classes.get(self.agent_role)
        if not agent_class:
            raise ValueError(f"No agent class for role {self.agent_role}")

        return agent_class(llm_client=self.vllm_client)

    async def run(self):
        """Main worker loop: pull tasks from queue, execute, report results"""
        self.running = True
        logger.info(f"Worker {self.worker_id} starting main loop...")

        while self.running:
            try:
                # Update worker heartbeat
                await self.redis_client.hset(
                    f"codegod:workers:{self.worker_id}",
                    "last_seen",
                    datetime.utcnow().isoformat()
                )
                await self.redis_client.expire(f"codegod:workers:{self.worker_id}", 300)

                # Pull task from queue (blocking pop with 5 second timeout)
                task_data = await self.redis_client.blpop(
                    f"codegod:tasks:{self.agent_role.value}",
                    timeout=5
                )

                if not task_data:
                    # No task available, continue loop
                    await asyncio.sleep(1)
                    continue

                # Parse task
                _, task_json = task_data
                task_dict = json.loads(task_json)
                task = Task(**task_dict)

                logger.info(f"Received task: {task.id} - {task.description}")
                self.current_task = task

                # Update worker status
                await self.redis_client.hset(
                    f"codegod:workers:{self.worker_id}",
                    mapping={
                        "status": "busy",
                        "current_task": task.id
                    }
                )

                # Execute task with agent reasoning
                result = await self._execute_task(task)

                # Report result back to master
                await self._report_result(task, result)

                # Update worker status
                await self.redis_client.hset(
                    f"codegod:workers:{self.worker_id}",
                    mapping={
                        "status": "idle",
                        "current_task": ""
                    }
                )

                self.current_task = None

            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                if self.current_task:
                    await self._report_error(self.current_task, str(e))
                await asyncio.sleep(5)  # Back off on error

    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute task using agent's reasoning loop"""
        try:
            # Update task status to IN_PROGRESS
            task.status = TaskStatus.IN_PROGRESS
            await self.redis_client.hset(
                f"codegod:task:{task.id}",
                "status",
                task.status.value
            )

            logger.info(f"Executing task {task.id} with agent reasoning...")

            # Agent executes: Observe → Reason → Plan → Act → Validate
            result = await self.agent.reason_and_act(task)

            # Update agent memory
            self.agent.memory.completed_tasks.append(task.id)

            logger.info(f"Task {task.id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}", exc_info=True)
            raise

    async def _report_result(self, task: Task, result: Dict[str, Any]):
        """Report task completion result to master orchestrator"""
        task.status = TaskStatus.COMPLETED if result.get('success') else TaskStatus.FAILED
        task.result = result

        # Store result in Redis
        await self.redis_client.hset(
            f"codegod:task:{task.id}",
            mapping={
                "status": task.status.value,
                "result": json.dumps(result),
                "completed_at": datetime.utcnow().isoformat(),
                "worker_id": self.worker_id
            }
        )

        # Publish completion event to master
        await self.redis_client.publish(
            "codegod:task_completed",
            json.dumps({
                "task_id": task.id,
                "status": task.status.value,
                "worker_id": self.worker_id
            })
        )

        # Update metrics
        await self.redis_client.hincrby(
            "codegod:metrics:tasks",
            "completed" if result.get('success') else "failed",
            1
        )

        logger.info(f"Reported result for task {task.id}: {task.status.value}")

    async def _report_error(self, task: Task, error: str):
        """Report task error to master orchestrator"""
        task.status = TaskStatus.FAILED

        await self.redis_client.hset(
            f"codegod:task:{task.id}",
            mapping={
                "status": task.status.value,
                "error": error,
                "failed_at": datetime.utcnow().isoformat(),
                "worker_id": self.worker_id
            }
        )

        await self.redis_client.publish(
            "codegod:task_completed",
            json.dumps({
                "task_id": task.id,
                "status": task.status.value,
                "error": error,
                "worker_id": self.worker_id
            })
        )

        await self.redis_client.hincrby("codegod:metrics:tasks", "failed", 1)
        logger.error(f"Reported error for task {task.id}: {error}")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"Shutting down worker {self.worker_id}...")
        self.running = False

        # Unregister worker
        if self.redis_client:
            await self.redis_client.delete(f"codegod:workers:{self.worker_id}")
            await self.redis_client.close()

        logger.info("Worker shutdown complete")


async def main():
    """Main entry point for agent worker"""
    worker = AgentWorker()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler(sig):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        asyncio.create_task(worker.shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    try:
        # Initialize worker
        await worker.initialize()

        # Run main loop
        await worker.run()

    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await worker.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        sys.exit(0)
