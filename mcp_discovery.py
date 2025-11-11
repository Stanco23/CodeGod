"""
MCP Discovery Module
Integrates with 1mcpserver.com to discover and install MCP servers automatically
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiohttp
import subprocess

logger = logging.getLogger(__name__)


class MCPDiscovery:
    """
    Discovers and manages MCP servers using 1mcpserver.com
    """

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.servers_dir = config_dir / "mcp_servers"
        self.servers_dir.mkdir(parents=True, exist_ok=True)

        self.available_servers: List[Dict[str, Any]] = []
        self.installed_servers: Dict[str, Dict] = {}
        self.running_servers: Dict[str, subprocess.Popen] = {}

        # 1mcpserver API endpoint
        self.api_url = "https://1mcpserver.com/api/servers"

    async def initialize(self):
        """Initialize and discover MCP servers"""
        await self.discover_servers()
        self.load_installed_servers()

    async def discover_servers(self):
        """Discover available MCP servers from 1mcpserver.com"""
        # Try API first, but always fall back to defaults
        api_success = False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        servers = data.get("servers", [])
                        if servers:
                            self.available_servers = servers
                            logger.info(f"Discovered {len(self.available_servers)} MCP servers from API")
                            api_success = True
                        else:
                            logger.warning("API returned empty server list")
                    else:
                        logger.warning(f"API returned HTTP {response.status}")

        except asyncio.TimeoutError:
            logger.warning("Timeout fetching servers from 1mcpserver.com (5s limit)")
        except aiohttp.ClientError as e:
            logger.warning(f"Network error fetching servers: {e}")
        except Exception as e:
            logger.warning(f"Error discovering servers from API: {e}")

        # Always load defaults if API failed or returned nothing
        if not api_success:
            logger.info("Using default MCP server list")
            self._load_default_servers()

    def _load_default_servers(self):
        """Load default MCP servers (fallback)"""
        # Try to load from catalog JSON file first
        catalog_path = Path(__file__).parent / "mcp_servers_catalog.json"

        if catalog_path.exists():
            try:
                with open(catalog_path, 'r') as f:
                    data = json.load(f)
                    self.available_servers = data.get("servers", [])
                    logger.info(f"Loaded {len(self.available_servers)} servers from catalog")
                    return
            except Exception as e:
                logger.warning(f"Failed to load catalog: {e}, using hardcoded defaults")

        # Fallback to hardcoded list if catalog doesn't exist
        self.available_servers = [
            {
                "name": "filesystem",
                "repo": "https://github.com/modelcontextprotocol/servers.git",
                "path": "src/filesystem",
                "description": "File system operations - read, write, list, search files",
                "tools": [
                    "read_file", "read_multiple_files", "write_file",
                    "create_directory", "list_directory", "move_file",
                    "search_files", "get_file_info"
                ],
                "install_cmd": "npm install && npm run build",
                "run_cmd": "node dist/index.js",
                "language": "typescript"
            },
            {
                "name": "git",
                "repo": "https://github.com/modelcontextprotocol/servers.git",
                "path": "src/git",
                "description": "Git operations - status, commit, diff, log",
                "tools": [
                    "git_status", "git_diff_unstaged", "git_diff_staged",
                    "git_commit", "git_add", "git_reset", "git_log",
                    "git_clone", "git_checkout", "git_create_branch", "git_merge"
                ],
                "install_cmd": None,  # Auto-detect (Python project)
                "run_cmd": "python -m mcp_server_git",
                "language": "python"
            },
            {
                "name": "github",
                "repo": "https://github.com/modelcontextprotocol/servers.git",
                "path": "src/github",
                "description": "GitHub API operations",
                "tools": [
                    "create_or_update_file", "search_repositories",
                    "create_repository", "get_file_contents", "push_files",
                    "create_issue", "create_pull_request", "fork_repository"
                ],
                "install_cmd": "npm install && npm run build",
                "run_cmd": "node dist/index.js",
                "language": "typescript",
                "env_vars": {"GITHUB_PERSONAL_ACCESS_TOKEN": ""}
            },
            {
                "name": "postgres",
                "repo": "https://github.com/modelcontextprotocol/servers.git",
                "path": "src/postgres",
                "description": "PostgreSQL database operations",
                "tools": [
                    "query", "execute", "list_tables",
                    "describe_table", "create_table", "insert_data"
                ],
                "install_cmd": "npm install && npm run build",
                "run_cmd": "node dist/index.js",
                "language": "typescript",
                "env_vars": {"POSTGRES_CONNECTION_STRING": ""}
            },
            {
                "name": "fetch",
                "repo": "https://github.com/modelcontextprotocol/servers.git",
                "path": "src/fetch",
                "description": "Web fetching and scraping",
                "tools": ["fetch", "fetch_html", "fetch_json", "fetch_text"],
                "install_cmd": "npm install && npm run build",
                "run_cmd": "node dist/index.js",
                "language": "typescript"
            },
            {
                "name": "sqlite",
                "repo": "https://github.com/modelcontextprotocol/servers.git",
                "path": "src/sqlite",
                "description": "SQLite database operations",
                "tools": [
                    "read_query", "write_query", "create_table",
                    "list_tables", "describe_table"
                ],
                "install_cmd": "npm install && npm run build",
                "run_cmd": "node dist/index.js",
                "language": "typescript"
            },
            {
                "name": "memory",
                "repo": "https://github.com/modelcontextprotocol/servers.git",
                "path": "src/memory",
                "description": "Knowledge graph memory",
                "tools": [
                    "create_entities", "create_relations", "search_nodes",
                    "open_nodes", "delete_entities"
                ],
                "install_cmd": "npm install && npm run build",
                "run_cmd": "node dist/index.js",
                "language": "typescript"
            },
            {
                "name": "puppeteer",
                "repo": "https://github.com/modelcontextprotocol/servers.git",
                "path": "src/puppeteer",
                "description": "Browser automation",
                "tools": [
                    "puppeteer_navigate", "puppeteer_screenshot",
                    "puppeteer_click", "puppeteer_fill", "puppeteer_evaluate"
                ],
                "install_cmd": "npm install && npm run build",
                "run_cmd": "node dist/index.js",
                "language": "typescript"
            }
        ]

        logger.info(f"Loaded {len(self.available_servers)} default servers")

    def load_installed_servers(self):
        """Load information about installed servers"""
        for server_spec in self.available_servers:
            server_name = server_spec["name"]
            server_path = self.servers_dir / server_name

            if server_path.exists():
                server_spec["installed"] = True
                server_spec["path_local"] = str(server_path)
                self.installed_servers[server_name] = server_spec
            else:
                server_spec["installed"] = False

    def _detect_and_get_install_cmd(self, server_path: Path, server_spec: Dict) -> str:
        """
        Auto-detect project type and return appropriate install command

        Args:
            server_path: Path to server directory
            server_spec: Server specification

        Returns:
            Install command string
        """
        # Check for Python project
        if (server_path / "pyproject.toml").exists():
            logger.info(f"Detected Python project (pyproject.toml)")
            # Try uv first (faster), fall back to pip
            if subprocess.run(["which", "uv"], capture_output=True).returncode == 0:
                return "uv pip install -e ."
            else:
                return "pip install -e ."

        # Check for Node.js/TypeScript project
        elif (server_path / "package.json").exists():
            logger.info(f"Detected Node.js/TypeScript project (package.json)")
            return "npm install && npm run build"

        # Check for requirements.txt (Python without pyproject.toml)
        elif (server_path / "requirements.txt").exists():
            logger.info(f"Detected Python project (requirements.txt)")
            return "pip install -r requirements.txt"

        # Fall back to spec's install_cmd if provided
        elif server_spec.get("install_cmd"):
            logger.info(f"Using spec install command")
            return server_spec.get("install_cmd")

        # No installation needed
        else:
            logger.info(f"No installation command detected")
            return None

    async def install_server(self, server_name: str) -> bool:
        """
        Install an MCP server

        Args:
            server_name: Name of server to install

        Returns:
            True if successful
        """
        server_spec = next((s for s in self.available_servers if s["name"] == server_name), None)

        if not server_spec:
            logger.error(f"Server '{server_name}' not found in available servers")
            return False

        if server_spec.get("installed"):
            logger.info(f"Server '{server_name}' is already installed")
            return True

        try:
            logger.info(f"Installing MCP server: {server_name}")

            # Check prerequisites
            if server_spec.get("language") == "typescript":
                try:
                    result = subprocess.run(
                        ["node", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode != 0:
                        raise FileNotFoundError
                    logger.info(f"Node.js version: {result.stdout.strip()}")
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    logger.error("Node.js is required but not installed. Install from https://nodejs.org/")
                    return False

            # Check git is available
            try:
                subprocess.run(
                    ["git", "--version"],
                    capture_output=True,
                    timeout=5,
                    check=True
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.error("Git is required but not installed")
                return False

            # Clone repo if needed
            repo_url = server_spec["repo"]
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_dir = self.servers_dir / "repos" / repo_name

            if not repo_dir.exists():
                logger.info(f"Cloning repository: {repo_url}")
                try:
                    result = subprocess.run(
                        ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode != 0:
                        logger.error(f"Git clone failed: {result.stderr}")
                        return False
                    logger.info("Repository cloned successfully")
                except subprocess.TimeoutExpired:
                    logger.error("Git clone timed out after 60 seconds")
                    return False
            else:
                logger.info(f"Repository already exists at {repo_dir}")

            # Find server source with multiple fallback paths
            server_src = repo_dir / server_spec.get("path", server_name)

            if not server_src.exists():
                # Try alternative paths
                alt_paths = [
                    repo_dir / server_name,
                    repo_dir / f"src/{server_name}",
                    repo_dir / f"servers/{server_name}",
                    repo_dir,  # Might be at root
                ]

                logger.warning(f"Primary path not found: {server_src}")
                logger.info(f"Trying alternative paths...")

                found = False
                for alt_path in alt_paths:
                    logger.info(f"  Checking: {alt_path}")
                    if alt_path.exists():
                        server_src = alt_path
                        logger.info(f"  Found server at: {alt_path}")
                        found = True
                        break

                if not found:
                    logger.error(f"Server source not found in repository")
                    logger.error(f"Tried paths: {server_src}, {alt_paths}")
                    return False

            # Copy server files
            import shutil
            server_dst = self.servers_dir / server_name

            if server_dst.exists():
                logger.info(f"Removing existing installation at {server_dst}")
                shutil.rmtree(server_dst)

            logger.info(f"Copying server files from {server_src} to {server_dst}")
            shutil.copytree(server_src, server_dst, symlinks=True)

            # Auto-detect project type and install dependencies
            install_cmd = self._detect_and_get_install_cmd(server_dst, server_spec)

            if install_cmd:
                logger.info(f"Installing dependencies with: {install_cmd}")
                try:
                    result = subprocess.run(
                        install_cmd,
                        shell=True,
                        cwd=server_dst,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result.returncode != 0:
                        logger.error(f"Installation failed with exit code {result.returncode}")
                        logger.error(f"STDOUT: {result.stdout}")
                        logger.error(f"STDERR: {result.stderr}")
                        return False

                    logger.info(f"Dependencies installed successfully")
                    if result.stdout:
                        logger.debug(f"Build output: {result.stdout}")

                except subprocess.TimeoutExpired:
                    logger.error("Installation timed out after 300 seconds")
                    return False
            else:
                logger.info("No installation command needed")

            # Mark as installed
            server_spec["installed"] = True
            server_spec["path_local"] = str(server_dst)
            self.installed_servers[server_name] = server_spec

            logger.info(f"Successfully installed '{server_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to install '{server_name}': {e}", exc_info=True)
            return False

    async def start_server(self, server_name: str, config: Dict = None) -> bool:
        """
        Start an MCP server

        Args:
            server_name: Name of server
            config: Configuration including env vars

        Returns:
            True if started successfully
        """
        if server_name in self.running_servers:
            logger.warning(f"Server {server_name} already running")
            return True

        server_spec = self.installed_servers.get(server_name)
        if not server_spec:
            logger.error(f"Server {server_name} not installed")
            # Try to install it
            success = await self.install_server(server_name)
            if not success:
                return False
            server_spec = self.installed_servers[server_name]

        try:
            # Prepare environment
            import os
            env = os.environ.copy()
            if config:
                env.update(config)
            if server_spec.get("env_vars"):
                env.update(server_spec["env_vars"])

            # Start server
            server_path = Path(server_spec["path_local"])
            run_cmd = server_spec.get("run_cmd", "node dist/index.js")

            logger.info(f"Starting MCP server: {server_name}")

            process = subprocess.Popen(
                run_cmd,
                shell=True,
                cwd=server_path,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.running_servers[server_name] = process

            # Wait a bit to ensure it started
            await asyncio.sleep(1)

            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Server {server_name} failed to start: {stderr.decode()}")
                return False

            logger.info(f"Successfully started {server_name}")
            server_spec["running"] = True
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

            # Update spec
            for spec in self.available_servers:
                if spec["name"] == server_name:
                    spec["running"] = False

            logger.info(f"Stopped MCP server: {server_name}")

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server

        Args:
            server_name: Name of MCP server
            tool_name: Name of tool
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if server_name not in self.running_servers:
            # Try to start it
            success = await self.start_server(server_name)
            if not success:
                return {"error": f"Could not start server {server_name}"}

        # MCP protocol: JSON-RPC request
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

            # Read response
            response_line = process.stdout.readline()
            if not response_line:
                return {"error": "No response from server"}

            response = json.loads(response_line.decode())

            if "error" in response:
                logger.error(f"Tool call error: {response['error']}")
                return {"error": response["error"]}

            return response.get("result", {})

        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on {server_name}: {e}")
            return {"error": str(e)}

    async def refresh_servers(self):
        """Refresh server list from 1mcpserver.com"""
        await self.discover_servers()
        self.load_installed_servers()

    def get_server_tools(self, server_name: str) -> List[str]:
        """Get list of tools for a server"""
        server_spec = next((s for s in self.available_servers if s["name"] == server_name), None)
        return server_spec.get("tools", []) if server_spec else []

    def get_all_tools(self) -> Dict[str, List[str]]:
        """Get all available tools from all installed servers"""
        tools = {}
        for server_name, server_spec in self.installed_servers.items():
            tools[server_name] = server_spec.get("tools", [])
        return tools

    async def shutdown_all(self):
        """Shutdown all running servers"""
        logger.info("Shutting down all MCP servers...")
        for server_name in list(self.running_servers.keys()):
            await self.stop_server(server_name)

    def search_servers(self, query: str) -> List[Dict[str, Any]]:
        """
        Search MCP servers by name, description, tools, or category

        Args:
            query: Search query string

        Returns:
            List of matching server specifications
        """
        query = query.lower()
        results = []

        for server in self.available_servers:
            # Search in name
            if query in server.get("name", "").lower():
                results.append(server)
                continue

            # Search in description
            if query in server.get("description", "").lower():
                results.append(server)
                continue

            # Search in category
            if query in server.get("category", "").lower():
                results.append(server)
                continue

            # Search in tools
            tools = server.get("tools", [])
            if any(query in tool.lower() for tool in tools):
                results.append(server)
                continue

        return results

    def get_categories(self) -> List[str]:
        """Get list of all available categories"""
        categories = set()
        for server in self.available_servers:
            if "category" in server:
                categories.add(server["category"])
        return sorted(list(categories))

    def get_servers_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all servers in a specific category"""
        return [s for s in self.available_servers if s.get("category", "").lower() == category.lower()]
