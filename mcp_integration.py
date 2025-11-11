"""
MCP (Model Context Protocol) Tool Integration System
Manages allocation and execution of MCP tools for worker agents
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from abc import ABC, abstractmethod


class MCPToolType(Enum):
    """Available MCP tool types"""
    FETCH = "fetch"
    MEMORY = "memory"
    GIT = "git"
    FILESYSTEM = "filesystem"
    SEQUENTIAL_THINKING = "sequential_thinking"
    DATABASE = "database"
    TIME = "time"
    BROWSER = "browser"


@dataclass
class MCPTool:
    """Represents a single MCP tool"""
    name: str
    type: MCPToolType
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    optional_params: List[str] = field(default_factory=list)


@dataclass
class MCPToolResult:
    """Result from MCP tool execution"""
    success: bool
    tool_name: str
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


class MCPToolExecutor(ABC):
    """Abstract base class for MCP tool executors"""

    @abstractmethod
    async def execute(self, tool: MCPTool, params: Dict) -> MCPToolResult:
        """Execute the tool with given parameters"""
        pass


# ========== MCP Tool Implementations ==========

class FetchMCPExecutor(MCPToolExecutor):
    """
    Fetch MCP - Web scraping and documentation retrieval
    """

    async def execute(self, tool: MCPTool, params: Dict) -> MCPToolResult:
        """Execute fetch operation"""
        import time
        import aiohttp

        start_time = time.time()

        try:
            url = params.get("url")
            method = params.get("method", "GET")

            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url) as response:
                        content = await response.text()
                        return MCPToolResult(
                            success=True,
                            tool_name="fetch",
                            output={
                                "content": content,
                                "status_code": response.status,
                                "headers": dict(response.headers)
                            },
                            execution_time=time.time() - start_time
                        )
                elif method == "POST":
                    data = params.get("data", {})
                    async with session.post(url, json=data) as response:
                        content = await response.text()
                        return MCPToolResult(
                            success=True,
                            tool_name="fetch",
                            output={
                                "content": content,
                                "status_code": response.status
                            },
                            execution_time=time.time() - start_time
                        )

        except Exception as e:
            return MCPToolResult(
                success=False,
                tool_name="fetch",
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )


class MemoryMCPExecutor(MCPToolExecutor):
    """
    Memory MCP - Vector database operations
    """

    def __init__(self, memory_system):
        self.memory = memory_system

    async def execute(self, tool: MCPTool, params: Dict) -> MCPToolResult:
        """Execute memory operation"""
        import time

        start_time = time.time()
        operation = params.get("operation")

        try:
            if operation == "store":
                content = params["content"]
                metadata = params["metadata"]
                result = self.memory.store(content, metadata)
                output = {"stored_ids": result}

            elif operation == "retrieve":
                query = params["query"]
                filters = params.get("filters")
                top_k = params.get("top_k", 5)
                result = self.memory.retrieve(query, filters, top_k)
                output = {"results": result}

            elif operation == "search":
                filters = params["filters"]
                result = self.memory.search_by_metadata(filters)
                output = {"results": result}

            else:
                raise ValueError(f"Unknown operation: {operation}")

            return MCPToolResult(
                success=True,
                tool_name="memory",
                output=output,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return MCPToolResult(
                success=False,
                tool_name="memory",
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )


class GitMCPExecutor(MCPToolExecutor):
    """
    Git MCP - Version control operations
    """

    async def execute(self, tool: MCPTool, params: Dict) -> MCPToolResult:
        """Execute git operation"""
        import time
        import subprocess

        start_time = time.time()
        operation = params.get("operation")
        repo_path = params.get("repo_path", ".")

        try:
            if operation == "init":
                result = subprocess.run(
                    ["git", "init"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True
                )

            elif operation == "add":
                files = params.get("files", ["."])
                result = subprocess.run(
                    ["git", "add"] + files,
                    cwd=repo_path,
                    capture_output=True,
                    text=True
                )

            elif operation == "commit":
                message = params["message"]
                result = subprocess.run(
                    ["git", "commit", "-m", message],
                    cwd=repo_path,
                    capture_output=True,
                    text=True
                )

            elif operation == "branch":
                branch_name = params["branch_name"]
                result = subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    cwd=repo_path,
                    capture_output=True,
                    text=True
                )

            elif operation == "status":
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True
                )

            else:
                raise ValueError(f"Unknown operation: {operation}")

            return MCPToolResult(
                success=result.returncode == 0,
                tool_name="git",
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                },
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return MCPToolResult(
                success=False,
                tool_name="git",
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )


class FilesystemMCPExecutor(MCPToolExecutor):
    """
    Filesystem MCP - File operations in sandboxed environment
    """

    def __init__(self, sandbox_root: str = "./sandbox"):
        self.sandbox_root = sandbox_root

    async def execute(self, tool: MCPTool, params: Dict) -> MCPToolResult:
        """Execute filesystem operation"""
        import time
        import os
        from pathlib import Path

        start_time = time.time()
        operation = params.get("operation")

        try:
            if operation == "read":
                file_path = params["file_path"]
                full_path = os.path.join(self.sandbox_root, file_path)
                with open(full_path, 'r') as f:
                    content = f.read()
                output = {"content": content}

            elif operation == "write":
                file_path = params["file_path"]
                content = params["content"]
                full_path = os.path.join(self.sandbox_root, file_path)

                # Create directories if needed
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                with open(full_path, 'w') as f:
                    f.write(content)
                output = {"file_path": full_path, "bytes_written": len(content)}

            elif operation == "list":
                dir_path = params.get("dir_path", ".")
                full_path = os.path.join(self.sandbox_root, dir_path)
                files = os.listdir(full_path)
                output = {"files": files}

            elif operation == "delete":
                file_path = params["file_path"]
                full_path = os.path.join(self.sandbox_root, file_path)
                os.remove(full_path)
                output = {"deleted": file_path}

            elif operation == "execute":
                # Execute code in sandboxed environment
                code = params["code"]
                language = params.get("language", "python")

                # This is a simplified example - production would use proper sandboxing
                if language == "python":
                    import subprocess
                    result = subprocess.run(
                        ["python", "-c", code],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=self.sandbox_root
                    )
                    output = {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode
                    }
                else:
                    raise ValueError(f"Unsupported language: {language}")

            else:
                raise ValueError(f"Unknown operation: {operation}")

            return MCPToolResult(
                success=True,
                tool_name="filesystem",
                output=output,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return MCPToolResult(
                success=False,
                tool_name="filesystem",
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )


class SequentialThinkingMCPExecutor(MCPToolExecutor):
    """
    Sequential Thinking MCP - Multi-step reasoning for complex tasks
    """

    async def execute(self, tool: MCPTool, params: Dict) -> MCPToolResult:
        """Execute sequential thinking operation"""
        import time

        start_time = time.time()
        operation = params.get("operation")

        try:
            if operation == "plan_steps":
                task = params["task"]
                # Generate step-by-step plan
                steps = self._decompose_task(task)
                output = {"steps": steps}

            elif operation == "execute_step":
                step = params["step"]
                context = params.get("context", {})
                # Execute a single step
                result = await self._execute_single_step(step, context)
                output = {"result": result}

            elif operation == "validate_step":
                step_output = params["step_output"]
                expected = params.get("expected")
                # Validate step output
                is_valid = self._validate_output(step_output, expected)
                output = {"valid": is_valid}

            else:
                raise ValueError(f"Unknown operation: {operation}")

            return MCPToolResult(
                success=True,
                tool_name="sequential_thinking",
                output=output,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return MCPToolResult(
                success=False,
                tool_name="sequential_thinking",
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _decompose_task(self, task: str) -> List[str]:
        """Decompose task into steps"""
        # Simplified - in production would use LLM
        return [
            f"Analyze requirements for: {task}",
            f"Design solution for: {task}",
            f"Implement solution for: {task}",
            f"Test solution for: {task}",
            f"Validate output for: {task}"
        ]

    async def _execute_single_step(self, step: str, context: Dict) -> str:
        """Execute a single reasoning step"""
        # Simplified - in production would use LLM
        return f"Completed: {step}"

    def _validate_output(self, output: Any, expected: Any) -> bool:
        """Validate step output"""
        # Simplified validation
        return output is not None


class DatabaseMCPExecutor(MCPToolExecutor):
    """
    Database MCP - Database operations
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def execute(self, tool: MCPTool, params: Dict) -> MCPToolResult:
        """Execute database operation"""
        import time

        start_time = time.time()
        operation = params.get("operation")

        try:
            # This is a simplified example
            # Production would use actual database connections
            if operation == "query":
                query = params["query"]
                # Execute query
                output = {"rows": [], "count": 0}

            elif operation == "execute":
                statement = params["statement"]
                # Execute statement
                output = {"affected_rows": 0}

            elif operation == "create_schema":
                schema = params["schema"]
                # Create schema
                output = {"created": True}

            else:
                raise ValueError(f"Unknown operation: {operation}")

            return MCPToolResult(
                success=True,
                tool_name="database",
                output=output,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return MCPToolResult(
                success=False,
                tool_name="database",
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )


# ========== MCP Tool Manager ==========

class MCPToolManager:
    """
    Manages MCP tools and their allocation to tasks
    """

    def __init__(self, memory_system=None, sandbox_root: str = "./sandbox"):
        self.executors: Dict[MCPToolType, MCPToolExecutor] = {
            MCPToolType.FETCH: FetchMCPExecutor(),
            MCPToolType.MEMORY: MemoryMCPExecutor(memory_system) if memory_system else None,
            MCPToolType.GIT: GitMCPExecutor(),
            MCPToolType.FILESYSTEM: FilesystemMCPExecutor(sandbox_root),
            MCPToolType.SEQUENTIAL_THINKING: SequentialThinkingMCPExecutor(),
            MCPToolType.DATABASE: DatabaseMCPExecutor("sqlite:///default.db")
        }

        self.tool_definitions = self._define_tools()

    def _define_tools(self) -> Dict[str, List[MCPTool]]:
        """Define all available MCP tools"""
        return {
            "fetch": [
                MCPTool(
                    name="fetch_url",
                    type=MCPToolType.FETCH,
                    description="Fetch content from a URL",
                    parameters={"url": "string", "method": "GET|POST"},
                    required_params=["url"]
                ),
                MCPTool(
                    name="fetch_documentation",
                    type=MCPToolType.FETCH,
                    description="Fetch API documentation",
                    parameters={"url": "string"},
                    required_params=["url"]
                )
            ],
            "memory": [
                MCPTool(
                    name="store_knowledge",
                    type=MCPToolType.MEMORY,
                    description="Store content in vector database",
                    parameters={"content": "string", "metadata": "object"},
                    required_params=["content", "metadata"]
                ),
                MCPTool(
                    name="retrieve_context",
                    type=MCPToolType.MEMORY,
                    description="Retrieve relevant context using RAG",
                    parameters={"query": "string", "filters": "object", "top_k": "number"},
                    required_params=["query"]
                ),
                MCPTool(
                    name="search_knowledge",
                    type=MCPToolType.MEMORY,
                    description="Search by metadata",
                    parameters={"filters": "object"},
                    required_params=["filters"]
                )
            ],
            "git": [
                MCPTool(
                    name="git_init",
                    type=MCPToolType.GIT,
                    description="Initialize git repository",
                    parameters={"repo_path": "string"},
                    required_params=[]
                ),
                MCPTool(
                    name="git_commit",
                    type=MCPToolType.GIT,
                    description="Create git commit",
                    parameters={"message": "string", "files": "array"},
                    required_params=["message"]
                ),
                MCPTool(
                    name="git_branch",
                    type=MCPToolType.GIT,
                    description="Create git branch",
                    parameters={"branch_name": "string"},
                    required_params=["branch_name"]
                )
            ],
            "filesystem": [
                MCPTool(
                    name="read_file",
                    type=MCPToolType.FILESYSTEM,
                    description="Read file contents",
                    parameters={"file_path": "string"},
                    required_params=["file_path"]
                ),
                MCPTool(
                    name="write_file",
                    type=MCPToolType.FILESYSTEM,
                    description="Write content to file",
                    parameters={"file_path": "string", "content": "string"},
                    required_params=["file_path", "content"]
                ),
                MCPTool(
                    name="execute_code",
                    type=MCPToolType.FILESYSTEM,
                    description="Execute code in sandbox",
                    parameters={"code": "string", "language": "string"},
                    required_params=["code"]
                )
            ],
            "sequential_thinking": [
                MCPTool(
                    name="plan_steps",
                    type=MCPToolType.SEQUENTIAL_THINKING,
                    description="Break down task into steps",
                    parameters={"task": "string"},
                    required_params=["task"]
                ),
                MCPTool(
                    name="execute_step",
                    type=MCPToolType.SEQUENTIAL_THINKING,
                    description="Execute a reasoning step",
                    parameters={"step": "string", "context": "object"},
                    required_params=["step"]
                )
            ],
            "database": [
                MCPTool(
                    name="execute_query",
                    type=MCPToolType.DATABASE,
                    description="Execute database query",
                    parameters={"query": "string"},
                    required_params=["query"]
                )
            ]
        }

    def allocate_tools_for_task(self, task_type: str, task_subtype: str) -> List[str]:
        """
        Allocate appropriate MCP tools for a given task

        Args:
            task_type: Type of task (backend, frontend, testing, etc.)
            task_subtype: Specific task subtype

        Returns:
            List of tool names allocated for this task
        """

        allocation_rules = {
            "backend": {
                "api_endpoint": ["filesystem", "memory", "sequential_thinking"],
                "database_model": ["filesystem", "database", "memory"],
                "business_logic": ["filesystem", "memory", "sequential_thinking"]
            },
            "frontend": {
                "component": ["filesystem", "memory", "sequential_thinking"],
                "state_management": ["filesystem", "memory"]
            },
            "testing": {
                "unit_tests": ["filesystem", "memory", "sequential_thinking"],
                "integration_tests": ["filesystem", "database", "memory"]
            },
            "devops": {
                "dockerfile": ["filesystem", "memory"],
                "ci_cd": ["filesystem", "git", "memory"]
            },
            "documentation": {
                "api_docs": ["filesystem", "fetch", "memory"]
            }
        }

        return allocation_rules.get(task_type, {}).get(
            task_subtype,
            ["filesystem", "memory"]  # Default tools
        )

    async def execute_tool(
        self,
        tool_type: MCPToolType,
        operation: str,
        params: Dict
    ) -> MCPToolResult:
        """Execute a tool operation"""
        executor = self.executors.get(tool_type)

        if not executor:
            return MCPToolResult(
                success=False,
                tool_name=tool_type.value,
                output=None,
                error=f"No executor found for tool type: {tool_type}"
            )

        # Create tool object
        tool = MCPTool(
            name=operation,
            type=tool_type,
            description="",
            parameters=params,
            required_params=[]
        )

        # Add operation to params
        params["operation"] = operation

        # Execute
        return await executor.execute(tool, params)

    def get_tool_descriptions(self, tool_names: List[str]) -> str:
        """
        Get formatted descriptions of tools for prompt

        Args:
            tool_names: List of tool names to include

        Returns:
            Formatted string describing available tools
        """
        descriptions = ["=== AVAILABLE TOOLS ===\n"]

        for tool_name in tool_names:
            if tool_name in self.tool_definitions:
                tools = self.tool_definitions[tool_name]
                descriptions.append(f"\n{tool_name.upper()} Tools:")
                for tool in tools:
                    descriptions.append(f"- {tool.name}: {tool.description}")
                    descriptions.append(f"  Parameters: {tool.parameters}")
                    descriptions.append(f"  Required: {tool.required_params}")

        return "\n".join(descriptions)
