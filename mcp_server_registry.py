"""
MCP Server Registry and Manager
Integrates with official MCP servers from https://github.com/modelcontextprotocol/servers
"""

import json
import logging
import subprocess
import asyncio
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MCPServerSpec:
    """Specification for an MCP server"""
    name: str
    repo: str
    description: str
    tools: List[str]
    install_command: str
    run_command: str
    port: Optional[int] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    installed: bool = False


class MCPServerRegistry:
    """
    Registry of official MCP servers
    Based on https://github.com/modelcontextprotocol/servers
    """

    def __init__(self, registry_path: str = "./mcp_servers"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.servers: Dict[str, MCPServerSpec] = {}
        self.running_servers: Dict[str, subprocess.Popen] = {}

        self._load_official_servers()

    def _load_official_servers(self):
        """Load official MCP servers from the ecosystem"""

        # Official MCP servers from modelcontextprotocol/servers
        official_servers = [
            MCPServerSpec(
                name="filesystem",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="File system operations - read, write, list, search files",
                tools=[
                    "read_file",
                    "read_multiple_files",
                    "write_file",
                    "create_directory",
                    "list_directory",
                    "move_file",
                    "search_files",
                    "get_file_info"
                ],
                install_command="cd src/filesystem && npm install && npm run build",
                run_command="node dist/index.js",
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="github",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="GitHub API operations - repos, issues, PRs, search",
                tools=[
                    "create_or_update_file",
                    "search_repositories",
                    "create_repository",
                    "get_file_contents",
                    "push_files",
                    "create_issue",
                    "create_pull_request",
                    "fork_repository",
                    "create_branch"
                ],
                install_command="cd src/github && npm install && npm run build",
                run_command="node dist/index.js",
                env_vars={"GITHUB_PERSONAL_ACCESS_TOKEN": ""},
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="git",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="Git operations - clone, commit, push, status, diff",
                tools=[
                    "git_status",
                    "git_diff_unstaged",
                    "git_diff_staged",
                    "git_commit",
                    "git_add",
                    "git_reset",
                    "git_log",
                    "git_clone",
                    "git_checkout",
                    "git_create_branch",
                    "git_merge"
                ],
                install_command="cd src/git && npm install && npm run build",
                run_command="node dist/index.js",
                dependencies=["node", "npm", "git"]
            ),
            MCPServerSpec(
                name="postgres",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="PostgreSQL database operations",
                tools=[
                    "query",
                    "execute",
                    "list_tables",
                    "describe_table",
                    "create_table",
                    "insert_data",
                    "update_data"
                ],
                install_command="cd src/postgres && npm install && npm run build",
                run_command="node dist/index.js",
                env_vars={"POSTGRES_CONNECTION_STRING": ""},
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="sqlite",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="SQLite database operations",
                tools=[
                    "read_query",
                    "write_query",
                    "create_table",
                    "list_tables",
                    "describe_table",
                    "append_insight"
                ],
                install_command="cd src/sqlite && npm install && npm run build",
                run_command="node dist/index.js",
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="fetch",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="Web fetching and scraping",
                tools=[
                    "fetch",
                    "fetch_html",
                    "fetch_json",
                    "fetch_text"
                ],
                install_command="cd src/fetch && npm install && npm run build",
                run_command="node dist/index.js",
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="brave-search",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="Web search using Brave Search API",
                tools=[
                    "brave_web_search",
                    "brave_local_search"
                ],
                install_command="cd src/brave-search && npm install && npm run build",
                run_command="node dist/index.js",
                env_vars={"BRAVE_API_KEY": ""},
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="google-maps",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="Google Maps API operations",
                tools=[
                    "search_places",
                    "get_place_details",
                    "get_directions",
                    "geocode",
                    "reverse_geocode"
                ],
                install_command="cd src/google-maps && npm install && npm run build",
                run_command="node dist/index.js",
                env_vars={"GOOGLE_MAPS_API_KEY": ""},
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="slack",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="Slack API operations",
                tools=[
                    "post_message",
                    "list_channels",
                    "get_channel_history",
                    "add_reaction",
                    "upload_file"
                ],
                install_command="cd src/slack && npm install && npm run build",
                run_command="node dist/index.js",
                env_vars={"SLACK_BOT_TOKEN": "", "SLACK_TEAM_ID": ""},
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="memory",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="Knowledge graph memory with entities and relations",
                tools=[
                    "create_entities",
                    "create_relations",
                    "search_nodes",
                    "open_nodes",
                    "delete_entities",
                    "delete_relations"
                ],
                install_command="cd src/memory && npm install && npm run build",
                run_command="node dist/index.js",
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="puppeteer",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="Browser automation with Puppeteer",
                tools=[
                    "puppeteer_navigate",
                    "puppeteer_screenshot",
                    "puppeteer_click",
                    "puppeteer_fill",
                    "puppeteer_select",
                    "puppeteer_evaluate"
                ],
                install_command="cd src/puppeteer && npm install && npm run build",
                run_command="node dist/index.js",
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="sequential-thinking",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="Dynamic problem-solving through thinking sequences",
                tools=[
                    "create_thinking_sequence",
                    "add_thought",
                    "revise_thought",
                    "mark_complete"
                ],
                install_command="cd src/sequentialthinking && npm install && npm run build",
                run_command="node dist/index.js",
                dependencies=["node", "npm"]
            ),
            MCPServerSpec(
                name="time",
                repo="https://github.com/modelcontextprotocol/servers.git",
                description="Time and timezone operations",
                tools=[
                    "get_current_time",
                    "convert_time",
                    "get_timezone"
                ],
                install_command="cd src/time && npm install && npm run build",
                run_command="node dist/index.js",
                dependencies=["node", "npm"]
            ),
        ]

        for server in official_servers:
            self.servers[server.name] = server

        logger.info(f"Loaded {len(self.servers)} official MCP servers")

    async def install_server(self, server_name: str) -> bool:
        """
        Install an MCP server

        Args:
            server_name: Name of server to install

        Returns:
            True if successful, False otherwise
        """
        if server_name not in self.servers:
            logger.error(f"Unknown MCP server: {server_name}")
            return False

        server = self.servers[server_name]

        if server.installed:
            logger.info(f"Server {server_name} already installed")
            return True

        try:
            logger.info(f"Installing MCP server: {server_name}")

            # Check dependencies
            for dep in server.dependencies:
                if not self._check_dependency(dep):
                    logger.error(f"Missing dependency: {dep}")
                    return False

            # Clone repo if needed
            repo_path = self.registry_path / "servers"
            if not repo_path.exists():
                logger.info("Cloning MCP servers repository...")
                subprocess.run(
                    ["git", "clone", server.repo, str(repo_path)],
                    check=True,
                    capture_output=True
                )

            # Run install command
            logger.info(f"Running install command: {server.install_command}")
            result = subprocess.run(
                server.install_command,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                logger.error(f"Installation failed: {result.stderr}")
                return False

            server.installed = True
            logger.info(f"Successfully installed {server_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to install {server_name}: {e}")
            return False

    async def start_server(self, server_name: str, config: Dict[str, Any] = None) -> bool:
        """
        Start an MCP server

        Args:
            server_name: Name of server to start
            config: Configuration including environment variables

        Returns:
            True if started successfully
        """
        if server_name not in self.servers:
            logger.error(f"Unknown MCP server: {server_name}")
            return False

        server = self.servers[server_name]

        if not server.installed:
            logger.info(f"Server not installed, installing {server_name}...")
            success = await self.install_server(server_name)
            if not success:
                return False

        if server_name in self.running_servers:
            logger.warning(f"Server {server_name} already running")
            return True

        try:
            # Prepare environment
            env = os.environ.copy()
            if config:
                env.update(config)
            env.update(server.env_vars)

            # Start server
            repo_path = self.registry_path / "servers"
            server_path = repo_path / "src" / server_name

            logger.info(f"Starting MCP server: {server_name}")

            process = subprocess.Popen(
                server.run_command,
                shell=True,
                cwd=server_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.running_servers[server_name] = process

            # Wait a bit to ensure it started
            await asyncio.sleep(2)

            if process.poll() is not None:
                # Process died
                stdout, stderr = process.communicate()
                logger.error(f"Server {server_name} failed to start: {stderr.decode()}")
                return False

            logger.info(f"Successfully started {server_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start {server_name}: {e}")
            return False

    async def stop_server(self, server_name: str):
        """Stop an MCP server"""
        if server_name in self.running_servers:
            process = self.running_servers[server_name]
            process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.create_task(asyncio.to_thread(process.wait)),
                    timeout=5
                )
            except asyncio.TimeoutError:
                process.kill()

            del self.running_servers[server_name]
            logger.info(f"Stopped MCP server: {server_name}")

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server via stdio

        Args:
            server_name: Name of MCP server
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if server_name not in self.running_servers:
            # Try to start it
            success = await self.start_server(server_name)
            if not success:
                return {"error": f"Could not start server {server_name}"}

        # MCP protocol: send JSON-RPC request via stdin
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        try:
            process = self.running_servers[server_name]

            # Send request
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json.encode())
            process.stdin.flush()

            # Read response (blocking, but should be quick)
            response_line = process.stdout.readline()
            response = json.loads(response_line.decode())

            if "error" in response:
                logger.error(f"Tool call error: {response['error']}")
                return {"error": response["error"]}

            return response.get("result", {})

        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on {server_name}: {e}")
            return {"error": str(e)}

    def _check_dependency(self, dependency: str) -> bool:
        """Check if a dependency is installed"""
        try:
            subprocess.run(
                [dependency, "--version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_available_tools(self, server_names: List[str] = None) -> Dict[str, List[str]]:
        """
        Get all available tools from specified servers

        Args:
            server_names: List of server names, or None for all servers

        Returns:
            Dict mapping server name to list of tools
        """
        if server_names is None:
            server_names = list(self.servers.keys())

        tools = {}
        for name in server_names:
            if name in self.servers:
                tools[name] = self.servers[name].tools

        return tools

    def get_tool_description(self, server_name: str, tool_name: str) -> str:
        """Get description of a specific tool"""
        if server_name not in self.servers:
            return "Unknown server"

        server = self.servers[server_name]
        if tool_name not in server.tools:
            return "Unknown tool"

        # Tool descriptions (would ideally come from MCP server schema)
        descriptions = {
            "read_file": "Read contents of a file",
            "write_file": "Write content to a file",
            "list_directory": "List files in a directory",
            "search_files": "Search for files by pattern",
            "git_status": "Get git repository status",
            "git_commit": "Create a git commit",
            "git_diff_unstaged": "Show unstaged changes",
            "query": "Execute SQL query",
            "execute": "Execute SQL statement",
            "fetch": "Fetch content from URL",
            "brave_web_search": "Search the web with Brave",
            # ... add more as needed
        }

        return descriptions.get(tool_name, f"Execute {tool_name}")

    async def shutdown_all(self):
        """Shutdown all running MCP servers"""
        logger.info("Shutting down all MCP servers...")
        for server_name in list(self.running_servers.keys()):
            await self.stop_server(server_name)


async def setup_mcp_servers(config: Dict[str, Any] = None) -> MCPServerRegistry:
    """
    Setup and start essential MCP servers

    Args:
        config: Configuration dict with API keys and settings

    Returns:
        Configured MCPServerRegistry
    """
    registry = MCPServerRegistry()

    # Essential servers to start by default
    essential_servers = [
        "filesystem",
        "git",
        "fetch",
        "memory",
        "sequential-thinking",
        "time"
    ]

    # Optional servers (require API keys)
    optional_servers = {
        "github": "GITHUB_PERSONAL_ACCESS_TOKEN",
        "postgres": "POSTGRES_CONNECTION_STRING",
        "brave-search": "BRAVE_API_KEY",
        "google-maps": "GOOGLE_MAPS_API_KEY",
        "slack": "SLACK_BOT_TOKEN"
    }

    # Start essential servers
    for server_name in essential_servers:
        success = await registry.start_server(server_name, config)
        if not success:
            logger.warning(f"Could not start {server_name}, continuing anyway")

    # Start optional servers if configured
    if config:
        for server_name, env_var in optional_servers.items():
            if env_var in config and config[env_var]:
                await registry.start_server(server_name, config)

    return registry
