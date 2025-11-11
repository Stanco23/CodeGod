"""
Worker Agent Framework with Real MCP Integration
Executes individual tasks using small local models with explicit MCP tool calls
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
from mcp_server_registry import MCPServerRegistry, setup_mcp_servers

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
        """Execute prompt with local model"""
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama", "run", self.model_name,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            stdout, stderr = await asyncio.wait_for(
                process.communicate(full_prompt.encode()),
                timeout=120  # 2 minutes for worker tasks
            )

            if process.returncode != 0:
                logger.error(f"Model execution failed: {stderr.decode()}")
                return json.dumps({
                    "success": False,
                    "error": "Model execution failed"
                })

            response = stdout.decode().strip()
            return response

        except asyncio.TimeoutError:
            logger.error("Model execution timed out")
            return json.dumps({
                "success": False,
                "error": "Model execution timed out"
            })
        except Exception as e:
            logger.error(f"Model execution error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class WorkerAgentMCP:
    """
    Worker agent that executes individual tasks with MCP tool integration
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

        # MCP server registry
        self.mcp_registry: Optional[MCPServerRegistry] = None

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
        base_prompt = prompts.get(self.worker_type, BACKEND_WORKER_SYSTEM_PROMPT)

        # Add MCP tool usage instructions
        mcp_instructions = """

MCP TOOL USAGE:
When you need to use external tools (file operations, git, etc.), you must include explicit tool calls in your response.

Format for tool calls in your JSON response:
{
    "success": true,
    "code": "... your generated code ...",
    "tool_calls": [
        {
            "server": "filesystem",
            "tool": "write_file",
            "arguments": {
                "path": "path/to/file.py",
                "content": "... code content ..."
            },
            "description": "Save generated code to file"
        },
        {
            "server": "git",
            "tool": "git_add",
            "arguments": {
                "paths": ["path/to/file.py"]
            },
            "description": "Stage file for commit"
        }
    ]
}

Available MCP servers:
- filesystem: read_file, write_file, list_directory, create_directory, search_files
- git: git_status, git_add, git_commit, git_diff_unstaged, git_log
- fetch: fetch (download from URL)
- memory: create_entities, search_nodes (knowledge graph)
- sequential-thinking: create_thinking_sequence, add_thought

IMPORTANT: Include tool_calls in your response whenever the task requires file operations, git operations, or external actions.
"""

        return base_prompt + mcp_instructions

    async def initialize(self):
        """Initialize async components"""
        self.redis_client = await redis.from_url(
            f"redis://{self.redis_host}:{self.redis_port}",
            encoding="utf-8",
            decode_responses=True
        )

        # Setup MCP servers
        logger.info(f"Setting up MCP servers for worker {self.worker_id}...")
        self.mcp_registry = await setup_mcp_servers({})

        logger.info(f"Worker {self.worker_id} initialized with MCP integration")

    async def shutdown(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()

        if self.mcp_registry:
            await self.mcp_registry.shutdown_all()

    async def run(self):
        """Main worker loop - listen for tasks and execute"""
        logger.info(f"Worker {self.worker_id} started")

        while True:
            try:
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
                    ex=3600
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
        Execute a single task with MCP tool integration

        Args:
            task: Task specification with prompt, MCP servers, and tool calls

        Returns:
            Task result
        """
        task_id = task["task_id"]
        prompt = task["prompt"]
        mcp_servers = task.get("mcp_servers", [])
        expected_tool_calls = task.get("mcp_tool_calls", [])

        logger.info(f"Executing task {task_id} with MCP servers: {mcp_servers}")
        start_time = time.time()

        try:
            # Ensure required MCP servers are running
            for server_name in mcp_servers:
                if server_name not in self.mcp_registry.running_servers:
                    logger.info(f"Starting MCP server: {server_name}")
                    await self.mcp_registry.start_server(server_name)

            # Add MCP tool call instructions to prompt
            if expected_tool_calls:
                tool_instructions = "\n\nEXPECTED MCP TOOL CALLS:\n"
                for tc in expected_tool_calls:
                    tool_instructions += f"- Use {tc['server']}.{tc['tool']} {tc.get('when', '')}\n"
                prompt = prompt + tool_instructions

            # Execute prompt with model
            response = await self.model_executor.execute(
                system_prompt=self.system_prompt,
                user_prompt=prompt
            )

            # Parse response
            result = self._parse_response(response)

            # Validate output format
            if not self._validate_output(result):
                return {
                    "success": False,
                    "error": "Invalid output format",
                    "raw_response": response
                }

            # Execute MCP tool calls if present
            if "tool_calls" in result and result["tool_calls"]:
                tool_results = await self._execute_tool_calls(result["tool_calls"])
                result["tool_results"] = tool_results

                # Check if any tool call failed
                if any(not tr.get("success", True) for tr in tool_results):
                    logger.warning("Some tool calls failed")
                    result["warnings"] = ["Some MCP tool calls failed"]

            # Add execution time
            result["execution_time"] = time.time() - start_time

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
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
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                json_str = response

            return json.loads(json_str.strip())

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {
                "success": False,
                "error": f"Invalid JSON response: {str(e)}",
                "raw_response": response
            }

    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute MCP tool calls

        Args:
            tool_calls: List of tool call specifications

        Returns:
            List of tool results
        """
        results = []

        for tool_call in tool_calls:
            server_name = tool_call.get("server")
            tool_name = tool_call.get("tool")
            arguments = tool_call.get("arguments", {})
            description = tool_call.get("description", "")

            logger.info(f"Calling MCP tool: {server_name}.{tool_name} - {description}")

            try:
                result = await self.mcp_registry.call_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=arguments
                )

                results.append({
                    "server": server_name,
                    "tool": tool_name,
                    "success": "error" not in result,
                    "result": result,
                    "description": description
                })

            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                results.append({
                    "server": server_name,
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                    "description": description
                })

        return results

    def _validate_output(self, result: Dict) -> bool:
        """Validate that output has required fields"""
        if not isinstance(result, dict):
            return False

        if "success" not in result:
            return False

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

    logger.info(f"Starting {worker_type} worker with model {model_name} and MCP integration")

    worker = WorkerAgentMCP(
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
