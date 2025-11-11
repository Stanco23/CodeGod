#!/usr/bin/env python3
"""
Code-God: Autonomous AI Development Assistant
Terminal-based interactive app for building applications with AI
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse

# Rich for beautiful terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box

# Prompt toolkit for input with history
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style as PTStyle

# Core components
from local_model_executor import get_master_model, LocalModelExecutor
from mcp_discovery import MCPDiscovery
from project_builder import ProjectBuilder
from conversation_manager import ConversationManager
import subprocess

console = Console()


class CodeGod:
    """
    Main Code-God CLI application
    Interactive terminal interface for autonomous development
    """

    def __init__(self, model_name: Optional[str] = None, prefer_local: bool = True):
        self.console = console
        self.model_name = model_name
        self.prefer_local = prefer_local

        # Initialize components
        self.model: Optional[LocalModelExecutor] = None
        self.mcp_discovery: Optional[MCPDiscovery] = None
        self.project_builder: Optional[ProjectBuilder] = None
        self.conversation: Optional[ConversationManager] = None

        # Initialize prompt session with history
        history_file = Path.home() / ".codegod" / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self.session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
        )

        self.shell_mode = False  # Track if in shell mode

        # State
        self.current_project: Optional[str] = None
        self.running = True

        # Config
        self.config_dir = Path.home() / ".codegod"
        self.config_dir.mkdir(exist_ok=True)

    async def initialize(self):
        """Initialize the application"""
        self.console.clear()

        # Show banner
        self._show_banner()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            # Initialize model
            task = progress.add_task("Loading AI model...", total=None)
            try:
                self.model = get_master_model(
                    model_name=self.model_name,
                    prefer_local=self.prefer_local
                )
                progress.update(task, description=f"✓ Model loaded: {self.model.model_name}")
            except Exception as e:
                self.console.print(f"[red]✗ Failed to load model: {e}[/red]")
                sys.exit(1)

            # Initialize MCP discovery
            task = progress.add_task("Discovering MCP servers...", total=None)
            self.mcp_discovery = MCPDiscovery(config_dir=self.config_dir)
            await self.mcp_discovery.initialize()
            server_count = len(self.mcp_discovery.available_servers)
            progress.update(task, description=f"✓ Found {server_count} MCP servers")

            # Initialize project builder
            task = progress.add_task("Initializing project builder...", total=None)
            self.project_builder = ProjectBuilder(
                model=self.model,
                mcp_discovery=self.mcp_discovery
            )
            progress.update(task, description="✓ Project builder ready")

            # Initialize conversation manager
            self.conversation = ConversationManager(
                model=self.model,
                mcp_discovery=self.mcp_discovery
            )

        self.console.print()
        self.console.print("[green]✓ Code-God initialized successfully![/green]")
        self.console.print()

    def _show_banner(self):
        """Show welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██████╗ ██████╗ ███████╗     ██████╗  ██████╗ ██████╗  ║
║  ██╔════╝██╔═══██╗██╔══██╗██╔════╝    ██╔════╝ ██╔═══██╗██╔══██╗ ║
║  ██║     ██║   ██║██║  ██║█████╗      ██║  ███╗██║   ██║██║  ██║ ║
║  ██║     ██║   ██║██║  ██║██╔══╝      ██║   ██║██║   ██║██║  ██║ ║
║  ╚██████╗╚██████╔╝██████╔╝███████╗    ╚██████╔╝╚██████╔╝██████╔╝ ║
║   ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝     ╚═════╝  ╚═════╝ ╚═════╝  ║
║                                                               ║
║          Autonomous AI Development Assistant                 ║
║              Build Apps from Natural Language                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold cyan")

    async def run(self):
        """Main application loop"""
        # Show help on first run
        self._show_quick_help()

        while self.running:
            try:
                # Determine prompt based on mode
                if self.shell_mode:
                    prompt_text = "\n$ "
                else:
                    prompt_text = "\nYou> "

                # Get user input with history support
                user_input = await asyncio.to_thread(
                    self.session.prompt,
                    prompt_text
                )
                user_input = user_input.strip()

                if not user_input:
                    continue

                # Handle shell mode
                if self.shell_mode:
                    if user_input.lower() in ['exit', 'quit']:
                        self.shell_mode = False
                        self.console.print("[cyan]Exited shell mode. Back to AI mode.[/cyan]")
                    else:
                        self._execute_shell_command(user_input)
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                else:
                    # Normal conversation
                    await self._handle_message(user_input)

            except KeyboardInterrupt:
                if self.shell_mode:
                    # In shell mode, Ctrl+C just cancels current line
                    self.console.print()
                    continue
                if Confirm.ask("\n[yellow]Exit Code-God?[/yellow]"):
                    self.running = False
            except EOFError:
                # Ctrl+D exits
                self.running = False
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

        self.console.print("\n[cyan]Goodbye! 👋[/cyan]\n")

    def _show_quick_help(self):
        """Show quick help message"""
        help_text = """
[bold]Quick Start:[/bold]
• Type naturally to chat with AI
• Use [cyan]/build[/cyan] to start a new project
• Use [cyan]/search <query>[/cyan] to find MCP tools
• Use [cyan]/shell[/cyan] to run shell commands
• Use [cyan]↑/↓ arrows[/cyan] to navigate command history
• Use [cyan]/help[/cyan] for all commands

[bold]Example:[/bold]
  /build Create a REST API --dir ~/my-projects
        """
        self.console.print(Panel(help_text, title="Welcome to Code-God", border_style="cyan"))

    async def _handle_command(self, command: str):
        """Handle slash commands"""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        commands = {
            '/help': self._cmd_help,
            '/build': self._cmd_build,
            '/mcp': self._cmd_mcp,
            '/servers': self._cmd_servers,
            '/install': self._cmd_install,
            '/search': self._cmd_search,
            '/categories': self._cmd_categories,
            '/list': self._cmd_list,
            '/status': self._cmd_status,
            '/shell': self._cmd_shell,
            '/clear': self._cmd_clear,
            '/exit': self._cmd_exit,
            '/quit': self._cmd_exit,
            '/model': self._cmd_model,
            '/config': self._cmd_config,
        }

        if cmd in commands:
            await commands[cmd](args)
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("Type [cyan]/help[/cyan] for available commands")

    async def _handle_message(self, message: str):
        """Handle normal conversation message"""
        self.console.print()

        with Live(
            Panel("Thinking...", border_style="cyan"),
            console=self.console,
            refresh_per_second=4
        ) as live:
            try:
                response = await self.conversation.send_message(message)

                live.update(Panel(
                    Markdown(response),
                    title="[bold cyan]Code-God[/bold cyan]",
                    border_style="cyan"
                ))

            except Exception as e:
                live.update(Panel(
                    f"[red]Error: {e}[/red]",
                    border_style="red"
                ))

    async def _cmd_help(self, args: str):
        """Show help"""
        help_table = Table(title="Code-God Commands", box=box.ROUNDED)
        help_table.add_column("Command", style="cyan", no_wrap=True)
        help_table.add_column("Description", style="white")

        commands = [
            ("/build <prompt>", "Build a new project (use --dir to specify location)"),
            ("/mcp", "Show all available MCP servers"),
            ("/search <query>", "Search MCP servers by name, tool, or category"),
            ("/categories", "Show MCP server categories"),
            ("/servers", "List installed MCP servers"),
            ("/install <server>", "Install an MCP server"),
            ("/list", "List recent projects"),
            ("/status", "Show current status and configuration"),
            ("/model", "Show current AI model information"),
            ("/config", "Show/edit configuration"),
            ("/shell", "Enter shell mode (type 'exit' to return)"),
            ("/clear", "Clear the screen"),
            ("/exit or /quit", "Exit Code-God"),
        ]

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        self.console.print(help_table)

    async def _cmd_build(self, args: str):
        """Build a new project"""
        if not args:
            self.console.print("[yellow]Usage: /build <project description> [--dir <directory>][/yellow]")
            self.console.print("[dim]Example: /build Create a REST API for managing tasks[/dim]")
            self.console.print("[dim]Example: /build Create a REST API --dir ~/my-projects[/dim]")
            return

        # Parse directory flag
        import shlex
        try:
            parts = shlex.split(args)
        except ValueError:
            # If shlex fails, just use simple split
            parts = args.split()

        output_dir = None
        description_parts = []

        i = 0
        while i < len(parts):
            if parts[i] == "--dir" and i + 1 < len(parts):
                output_dir = parts[i + 1]
                i += 2
            else:
                description_parts.append(parts[i])
                i += 1

        description = " ".join(description_parts)

        if not description:
            self.console.print("[yellow]Please provide a project description[/yellow]")
            return

        self.console.print()
        self.console.print(f"[bold]Building project:[/bold] {description}")
        if output_dir:
            self.console.print(f"[bold]Output directory:[/bold] {output_dir}")
        self.console.print()

        try:
            project_path = await self.project_builder.build_project(
                description,
                console=self.console,
                output_dir=output_dir
            )

            self.console.print()
            self.console.print(Panel(
                f"[green]✓ Project built successfully![/green]\n\n"
                f"Location: [cyan]{project_path}[/cyan]\n\n"
                f"[bold]Next steps:[/bold]\n"
                f"  cd {project_path}\n"
                f"  # Review the generated code\n"
                f"  # Run tests\n"
                f"  # Deploy!",
                title="Success",
                border_style="green"
            ))

            self.current_project = str(project_path)

        except Exception as e:
            self.console.print(Panel(
                f"[red]Build failed:[/red] {str(e)}",
                border_style="red"
            ))

    async def _cmd_mcp(self, args: str):
        """Show MCP servers"""
        await self.mcp_discovery.refresh_servers()

        table = Table(title="Available MCP Servers", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Tools", style="green")
        table.add_column("Installed", style="yellow")

        for server in self.mcp_discovery.available_servers:
            installed = "✓" if server.get("installed", False) else "✗"
            tools = str(len(server.get("tools", [])))
            table.add_row(
                server["name"],
                server.get("description", "")[:50],
                tools,
                installed
            )

        self.console.print(table)
        self.console.print()
        self.console.print(f"[dim]Total: {len(self.mcp_discovery.available_servers)} servers[/dim]")
        self.console.print(f"[dim]Use [cyan]/install <server>[/cyan] to install a server[/dim]")

    async def _cmd_servers(self, args: str):
        """List installed servers"""
        installed = [s for s in self.mcp_discovery.available_servers if s.get("installed")]

        if not installed:
            self.console.print("[yellow]No MCP servers installed yet[/yellow]")
            self.console.print("[dim]Use [cyan]/mcp[/cyan] to see available servers[/dim]")
            return

        table = Table(title="Installed MCP Servers", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Tools", style="green")
        table.add_column("Status", style="yellow")

        for server in installed:
            tools_list = ", ".join(server.get("tools", [])[:3])
            if len(server.get("tools", [])) > 3:
                tools_list += "..."
            table.add_row(
                server["name"],
                tools_list,
                "Running" if server.get("running") else "Stopped"
            )

        self.console.print(table)

    async def _cmd_install(self, args: str):
        """Install an MCP server"""
        if not args:
            self.console.print("[yellow]Usage: /install <server-name>[/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Installing {args}...", total=None)

            try:
                success = await self.mcp_discovery.install_server(args)

                if success:
                    progress.update(task, description=f"✓ {args} installed successfully")
                    self.console.print(f"[green]✓ {args} is now available[/green]")
                else:
                    progress.update(task, description=f"✗ Failed to install {args}")
                    self.console.print(f"[red]✗ Installation failed[/red]")

            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    async def _cmd_search(self, args: str):
        """Search MCP servers and tools"""
        if not args:
            self.console.print("[yellow]Usage: /search <query>[/yellow]")
            self.console.print("[dim]Example: /search database[/dim]")
            self.console.print("[dim]Example: /search git[/dim]")
            return

        results = self.mcp_discovery.search_servers(args)

        if not results:
            self.console.print(f"[yellow]No servers found matching '{args}'[/yellow]")
            return

        table = Table(title=f"Search Results for '{args}'", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("Tools", style="green")
        table.add_column("Installed", style="yellow")

        for server in results:
            installed = "✓" if server.get("installed", False) else "✗"
            tools_count = str(len(server.get("tools", [])))
            category = server.get("category", "other")

            table.add_row(
                server["name"],
                category,
                server.get("description", "")[:60],
                tools_count,
                installed
            )

        self.console.print(table)
        self.console.print(f"\n[dim]Found {len(results)} server(s)[/dim]")
        self.console.print(f"[dim]Use [cyan]/install <name>[/cyan] to install a server[/dim]")

    async def _cmd_categories(self, args: str):
        """Show MCP server categories"""
        categories = self.mcp_discovery.get_categories()

        table = Table(title="MCP Server Categories", box=box.ROUNDED)
        table.add_column("Category", style="cyan")
        table.add_column("Servers", style="green")
        table.add_column("Example Servers", style="white")

        for category in categories:
            servers = self.mcp_discovery.get_servers_by_category(category)
            server_names = [s["name"] for s in servers[:3]]
            example_text = ", ".join(server_names)
            if len(servers) > 3:
                example_text += f", +{len(servers) - 3} more"

            table.add_row(
                category,
                str(len(servers)),
                example_text
            )

        self.console.print(table)
        self.console.print()
        self.console.print(f"[dim]Total: {len(categories)} categories[/dim]")
        self.console.print(f"[dim]Use [cyan]/search <category>[/cyan] to filter by category[/dim]")

    async def _cmd_list(self, args: str):
        """List recent projects"""
        projects_dir = Path.cwd() / "projects"

        if not projects_dir.exists():
            self.console.print("[yellow]No projects found[/yellow]")
            return

        projects = sorted(projects_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

        table = Table(title="Recent Projects", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Created", style="white")
        table.add_column("Path", style="dim")

        for project in projects[:10]:
            if project.is_dir():
                from datetime import datetime
                mtime = datetime.fromtimestamp(project.stat().st_mtime)
                table.add_row(
                    project.name,
                    mtime.strftime("%Y-%m-%d %H:%M"),
                    str(project)
                )

        self.console.print(table)

    async def _cmd_status(self, args: str):
        """Show current status"""
        status_table = Table(box=box.ROUNDED)
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="white")

        status_table.add_row("AI Model", self.model.model_name)
        status_table.add_row("Backend", self.model.backend.value)
        status_table.add_row("MCP Servers", str(len(self.mcp_discovery.available_servers)))
        status_table.add_row("Installed Servers", str(len([s for s in self.mcp_discovery.available_servers if s.get("installed")])))

        if self.current_project:
            status_table.add_row("Current Project", self.current_project)

        self.console.print(Panel(status_table, title="Status", border_style="cyan"))

    async def _cmd_clear(self, args: str):
        """Clear screen"""
        self.console.clear()
        self._show_banner()

    async def _cmd_exit(self, args: str):
        """Exit application"""
        self.running = False

    async def _cmd_model(self, args: str):
        """Show model information"""
        info_table = Table(title="AI Model Information", box=box.ROUNDED)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("Model Name", self.model.model_name)
        info_table.add_row("Backend", self.model.backend.value)

        if self.model.backend.value == "ollama":
            info_table.add_row("Type", "Local Model")
            info_table.add_row("Cost", "$0 per request")
        else:
            info_table.add_row("Type", "API Model")

        self.console.print(info_table)

    async def _cmd_config(self, args: str):
        """Show/edit configuration"""
        config_file = self.config_dir / "config.json"

        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)

            config_table = Table(title="Configuration", box=box.ROUNDED)
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="white")

            for key, value in config.items():
                config_table.add_row(key, str(value))

            self.console.print(config_table)
        else:
            self.console.print("[yellow]No configuration file found[/yellow]")

    async def _cmd_shell(self, args: str):
        """Enter shell mode"""
        self.shell_mode = True
        self.console.print("[cyan]Entered shell mode. Type shell commands directly.[/cyan]")
        self.console.print("[dim]Type 'exit' or 'quit' to return to AI mode[/dim]")

    def _execute_shell_command(self, command: str):
        """Execute a shell command"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.stdout:
                self.console.print(result.stdout, end='')
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]", end='')

            if result.returncode != 0:
                self.console.print(f"[yellow]Exit code: {result.returncode}[/yellow]")

        except subprocess.TimeoutExpired:
            self.console.print("[red]Command timed out (5 minute limit)[/red]")
        except Exception as e:
            self.console.print(f"[red]Error executing command: {e}[/red]")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Code-God: Autonomous AI Development Assistant"
    )
    parser.add_argument(
        "--model",
        help="AI model to use (default: auto-detect)",
        default=None
    )
    parser.add_argument(
        "--prefer-api",
        action="store_true",
        help="Prefer API models over local models"
    )
    parser.add_argument(
        "--build",
        help="Build a project and exit (non-interactive mode)",
        default=None
    )

    args = parser.parse_args()

    # Create and initialize app
    app = CodeGod(
        model_name=args.model,
        prefer_local=not args.prefer_api
    )

    try:
        await app.initialize()

        # Non-interactive mode
        if args.build:
            await app._cmd_build(args.build)
            return

        # Interactive mode
        await app.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
