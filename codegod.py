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
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box

# Core components
from local_model_executor import get_master_model, LocalModelExecutor
from mcp_discovery import MCPDiscovery
from project_builder import ProjectBuilder
from conversation_manager import ConversationManager

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
                # Get user input
                user_input = Prompt.ask(
                    "\n[bold cyan]You[/bold cyan]",
                    default=""
                ).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                else:
                    # Normal conversation
                    await self._handle_message(user_input)

            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Exit Code-God?[/yellow]"):
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
• Use [cyan]/mcp[/cyan] to manage MCP servers
• Use [cyan]/help[/cyan] for all commands

[bold]Example:[/bold]
  /build Create a REST API for todo management with FastAPI
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
            '/list': self._cmd_list,
            '/status': self._cmd_status,
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
            ("/build <prompt>", "Build a new project from natural language"),
            ("/mcp", "Show available MCP servers"),
            ("/servers", "List installed MCP servers"),
            ("/install <server>", "Install an MCP server"),
            ("/list", "List recent projects"),
            ("/status", "Show current status and configuration"),
            ("/model", "Show current AI model information"),
            ("/config", "Show/edit configuration"),
            ("/clear", "Clear the screen"),
            ("/exit or /quit", "Exit Code-God"),
        ]

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        self.console.print(help_table)

    async def _cmd_build(self, args: str):
        """Build a new project"""
        if not args:
            self.console.print("[yellow]Usage: /build <project description>[/yellow]")
            self.console.print("[dim]Example: /build Create a REST API for managing tasks[/dim]")
            return

        self.console.print()
        self.console.print(f"[bold]Building project:[/bold] {args}")
        self.console.print()

        try:
            project_path = await self.project_builder.build_project(args, console=self.console)

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
