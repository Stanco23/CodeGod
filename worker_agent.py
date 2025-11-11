"""
Worker Agent Framework
Executes individual tasks using small local models (3B-7B)
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional
import subprocess

import redis.asyncio as redis
from prompt_templates import (
    BACKEND_WORKER_SYSTEM_PROMPT,
    FRONTEND_WORKER_SYSTEM_PROMPT,
    TESTING_WORKER_SYSTEM_PROMPT,
    DEVOPS_WORKER_SYSTEM_PROMPT,
    DOCUMENTATION_WORKER_SYSTEM_PROMPT
)
from mcp_integration import MCPToolManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalModelExecutor:
    """
    Executes prompts using local small models via Ollama
    """

    def __init__(self, model_name: str = "phi-3:3b"):
        self.model_name = model_name
        self._ensure_model_available()

    def _ensure_model_available(self):
        """Ensure model is pulled and available"""
        try:
            # Check if model exists
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if self.model_name not in result.stdout:
                logger.info(f"Pulling model {self.model_name}...")
                subprocess.run(
                    ["ollama", "pull", self.model_name],
                    check=True,
                    timeout=600
                )
                logger.info(f"Model {self.model_name} ready")

        except Exception as e:
            logger.error(f"Failed to ensure model availability: {e}")

    async def execute(self, system_prompt: str, user_prompt: str) -> str:
        """
        Execute prompt with local model

        Args:
            system_prompt: System/role prompt
            user_prompt: User instruction

        Returns:
            Model response
        """
        try:
            # Use Ollama API
            process = await asyncio.create_subprocess_exec(
                "ollama", "run", self.model_name,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Construct full prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            stdout, stderr = await process.communicate(full_prompt.encode())

            if process.returncode != 0:
                logger.error(f"Model execution failed: {stderr.decode()}")
                return json.dumps({
                    "success": False,
                    "error": "Model execution failed"
                })

            response = stdout.decode().strip()
            return response

        except Exception as e:
            logger.error(f"Model execution error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class WorkerAgent:
    """
    Worker agent that executes individual tasks
    """

    def __init__(
        self,
        worker_type: str,
        model_name: str = "phi-3:3b",
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        self.worker_type = worker_type
        self.worker_id = f"{worker_type}_{os.getpid()}"

        # Initialize model executor
        self.model_executor = LocalModelExecutor(model_name)

        # Redis for task queue
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port

        # MCP tools
        self.mcp_manager = MCPToolManager(
            memory_system=None,  # Workers don't directly access memory
            sandbox_root=os.getenv("SANDBOX_ROOT", "./sandbox")
        )

        # System prompt based on worker type
        self.system_prompt = self._get_system_prompt()

        # Statistics
        self.tasks_completed = 0
        self.tasks_failed = 0

    def _get_system_prompt(self) -> str:
        """Get appropriate system prompt for worker type"""
        prompts = {
            "backend": BACKEND_WORKER_SYSTEM_PROMPT,
            "frontend": FRONTEND_WORKER_SYSTEM_PROMPT,
            "testing": TESTING_WORKER_SYSTEM_PROMPT,
            "devops": DEVOPS_WORKER_SYSTEM_PROMPT,
            "documentation": DOCUMENTATION_WORKER_SYSTEM_PROMPT
        }
        return prompts.get(self.worker_type, BACKEND_WORKER_SYSTEM_PROMPT)

    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.from_url(
            f"redis://{self.redis_host}:{self.redis_port}",
            encoding="utf-8",
            decode_responses=True
        )
        logger.info(f"Worker {self.worker_id} initialized")

    async def shutdown(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()

    async def run(self):
        """Main worker loop - listen for tasks and execute"""
        logger.info(f"Worker {self.worker_id} started")

        while True:
            try:
                # Block and wait for task from queue
                queue_name = f"task_queue:{self.worker_type}_specialist"
                task_json = await self.redis_client.brpop(queue_name, timeout=5)

                if not task_json:
                    continue

                _, task_data = task_json
                task = json.loads(task_data)

                # Execute task
                result = await self.execute_task(task)

                # Publish result back to master
                result_key = f"task_result:{task['task_id']}"
                await self.redis_client.set(
                    result_key,
                    json.dumps(result),
                    ex=3600  # Expire after 1 hour
                )

                if result["success"]:
                    self.tasks_completed += 1
                else:
                    self.tasks_failed += 1

                logger.info(f"Worker stats - Completed: {self.tasks_completed}, Failed: {self.tasks_failed}")

            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def execute_task(self, task: Dict) -> Dict:
        """
        Execute a single task

        Args:
            task: Task specification with prompt and tools

        Returns:
            Task result
        """
        task_id = task["task_id"]
        prompt = task["prompt"]
        tools = task.get("tools", [])

        logger.info(f"Executing task {task_id}")
        start_time = time.time()

        try:
            # Execute prompt with model
            response = await self.model_executor.execute(
                system_prompt=self.system_prompt,
                user_prompt=prompt
            )

            # Parse response (expect JSON)
            result = self._parse_response(response)

            # If task requires tool usage, handle it
            if tools and result.get("requires_tools"):
                result = await self._execute_with_tools(result, tools)

            # Validate output format
            if not self._validate_output(result):
                return {
                    "success": False,
                    "error": "Invalid output format",
                    "raw_response": response
                }

            # Add execution time
            result["execution_time"] = time.time() - start_time

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    def _parse_response(self, response: str) -> Dict:
        """Parse model response into structured format"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            elif "{" in response and "}" in response:
                # Find JSON object
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                # Assume entire response is JSON
                json_str = response

            return json.loads(json_str.strip())

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {
                "success": False,
                "error": f"Invalid JSON response: {str(e)}",
                "raw_response": response
            }

    async def _execute_with_tools(self, partial_result: Dict, tools: List[str]) -> Dict:
        """
        Execute task that requires MCP tools

        This allows worker to use filesystem, git, etc.
        """
        # Extract tool calls from result
        tool_calls = partial_result.get("tool_calls", [])

        for tool_call in tool_calls:
            tool_name = tool_call["tool"]
            operation = tool_call["operation"]
            params = tool_call["params"]

            # Execute tool
            result = await self.mcp_manager.execute_tool(
                tool_type=tool_name,
                operation=operation,
                params=params
            )

            # Store result for next iteration
            partial_result["tool_results"] = partial_result.get("tool_results", [])
            partial_result["tool_results"].append({
                "tool": tool_name,
                "success": result.success,
                "output": result.output
            })

        return partial_result

    def _validate_output(self, result: Dict) -> bool:
        """Validate that output has required fields"""
        if not isinstance(result, dict):
            return False

        # Must have success field
        if "success" not in result:
            return False

        # If success, must have code or content
        if result["success"]:
            if "code" not in result and "content" not in result:
                return False

        return True


async def main():
    """Main entry point for worker agent"""
    worker_type = os.getenv("WORKER_TYPE", "backend")
    model_name = os.getenv("WORKER_MODEL", "phi-3:3b")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    logger.info(f"Starting {worker_type} worker with model {model_name}")

    worker = WorkerAgent(
        worker_type=worker_type,
        model_name=model_name,
        redis_host=redis_host,
        redis_port=redis_port
    )

    try:
        await worker.initialize()
        await worker.run()
    except KeyboardInterrupt:
        logger.info("Worker shutting down...")
    finally:
        await worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
