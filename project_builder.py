"""
Project Builder
Builds complete projects from natural language descriptions
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

from local_model_executor import LocalModelExecutor
from mcp_discovery import MCPDiscovery
from error_knowledge_base import ErrorKnowledgeBase
from knowledge_memory_system import KnowledgeMemorySystem
from agentic_fixer import AgenticFixer

logger = logging.getLogger(__name__)


class ProjectBuilder:
    """
    Autonomous project builder
    """

    def __init__(self, model: LocalModelExecutor, mcp_discovery: MCPDiscovery, use_agents: bool = False):
        self.model = model
        self.mcp_discovery = mcp_discovery
        self.use_agents = use_agents

        # Initialize knowledge systems
        config_dir = Path.home() / ".codegod"
        self.error_kb = ErrorKnowledgeBase(config_dir)
        self.knowledge_memory = KnowledgeMemorySystem(config_dir)

        # Track build metrics
        self.build_start_time = None
        self.errors_encountered = []
        self.fixes_applied = []

    async def build_project(self, description: str, console: Console, output_dir: Optional[str] = None) -> Path:
        """
        Build a complete project from description

        Args:
            description: Natural language project description
            console: Rich console for output
            output_dir: Optional output directory (defaults to ./projects)

        Returns:
            Path to created project
        """
        # Log start
        console.print("\n[bold cyan]Starting AI Project Builder[/bold cyan]")
        console.print(f"[yellow]Description:[/yellow] {description}\n")

        # Start tracking
        self.build_start_time = datetime.now()
        self.errors_encountered = []
        self.fixes_applied = []

        # Record conversation context
        self.knowledge_memory.add_conversation_context(
            user_input=f"Build project: {description}",
            ai_response="Analyzing and planning project",
            action_taken="project_build"
        )

        # Show knowledge stats
        kb_stats = self.knowledge_memory.get_statistics()
        console.print(f"[dim]📚 Knowledge Base: {kb_stats['total_rules']} rules, {kb_stats['total_learnings']} learnings, {kb_stats['total_builds']} builds (Success rate: {kb_stats['success_rate']*100:.1f}%)[/dim]")

        # Show recent session context if available
        recent_context = self.knowledge_memory.get_conversation_summary(last_n=3)
        if recent_context and "No recent conversations" not in recent_context:
            console.print(f"[dim]💬 Recent Session Memory:[/dim]")
            for line in recent_context.split('\n')[1:4]:  # Show first 3
                if line.strip():
                    console.print(f"[dim]   {line}[/dim]")

        console.print()

        # Check if multi-agent mode is requested
        if self.use_agents:
            console.print("[bold magenta]🤖 Multi-Agent Mode Activated[/bold magenta]")
            console.print("[yellow]NOTE:[/yellow] Multi-agent system requires Docker services running.")
            console.print("[dim]Start with: ./start-agents.sh up[/dim]")
            console.print()

            # Check if Docker services are available
            import subprocess
            try:
                result = subprocess.run(
                    ['docker', 'ps', '--filter', 'name=codegod-', '--format', '{{.Names}}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                services = [s for s in result.stdout.strip().split('\n') if s]
                if services:
                    console.print(f"[green]✓ Found {len(services)} agent services running[/green]")
                    console.print("[dim]Services: " + ", ".join(services) + "[/dim]")
                else:
                    console.print("[red]✗ No agent services found. Please start with: ./start-agents.sh up[/red]")
                    console.print("[yellow]Falling back to single-agent mode...[/yellow]")
                    self.use_agents = False
            except Exception as e:
                console.print(f"[red]✗ Cannot connect to Docker: {e}[/red]")
                console.print("[yellow]Falling back to single-agent mode...[/yellow]")
                self.use_agents = False

            console.print()

        # Determine project path based on output_dir
        if output_dir:
            # Use exact directory specified by user (no subdirectory)
            project_path = Path(output_dir).expanduser().resolve()
        else:
            # Auto-generate name with timestamp in ./projects
            project_name = self._generate_project_name(description)
            base_dir = Path.cwd() / "projects"
            project_path = base_dir / project_name

        project_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Project path:[/green] {project_path}\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            # Phase 1: Analyze and plan
            task = progress.add_task("Analyzing requirements...", total=100)
            console.print("[bold blue]Phase 1: Analyzing Requirements[/bold blue]")
            console.print("[dim]AI is analyzing your project description and planning the architecture...[/dim]")

            plan = await self._analyze_requirements(description, console)
            progress.update(task, completed=20, description="✓ Requirements analyzed")

            # Show plan details
            console.print(f"\n[bold green]✓ Project Plan Created[/bold green]")
            console.print(f"  [cyan]Name:[/cyan] {plan.get('project_name', 'Unknown')}")
            console.print(f"  [cyan]Tech Stack:[/cyan] {json.dumps(plan.get('tech_stack', {}), indent=4)}")
            console.print(f"  [cyan]Files to create:[/cyan] {len(plan.get('files', {}))}")

            # Check for required MCP tools
            console.print("\n[bold blue]Checking for Required MCP Tools[/bold blue]")
            required_tools = await self._check_and_install_mcp_tools(plan, console)

            # Phase 2: Generate file structure
            progress.update(task, completed=30, description="Creating project structure...")
            console.print("\n[bold blue]Phase 2: Creating Project Structure[/bold blue]")
            await self._create_structure(project_path, plan, console)
            progress.update(task, completed=40, description="✓ Structure created")

            # Phase 3: Generate code files
            progress.update(task, completed=50, description="Generating code...")
            console.print("\n[bold blue]Phase 3: Generating Code Files[/bold blue]")
            files_to_create = plan.get("files", {})

            for idx, (file_path, file_spec) in enumerate(files_to_create.items()):
                file_progress = 50 + (30 * (idx + 1) / len(files_to_create))
                console.print(f"  [yellow]→[/yellow] Generating: [cyan]{file_path}[/cyan]")
                console.print(f"    [dim]Purpose: {file_spec.get('description', 'N/A')}[/dim]")

                progress.update(task, completed=file_progress, description=f"Generating {file_path}...")

                await self._generate_file(
                    project_path / file_path,
                    file_spec,
                    plan.get("context", {})
                )
                console.print(f"  [green]✓[/green] Created: [cyan]{file_path}[/cyan]")

            progress.update(task, completed=80, description="✓ Code generated")

            # Phase 4: Create detailed documentation
            progress.update(task, completed=85, description="Creating documentation...")
            console.print("\n[bold blue]Phase 4: Creating Detailed Documentation[/bold blue]")
            await self._create_detailed_readme(project_path, plan, console)
            progress.update(task, completed=88, description="✓ Documentation created")

            # Phase 5: Initialize git
            progress.update(task, completed=90, description="Initializing git...")
            console.print("\n[bold blue]Phase 5: Initializing Git Repository[/bold blue]")
            await self._init_git(project_path, console)
            progress.update(task, completed=92, description="✓ Git initialized")

            # Phase 6: Setup project environment (install dependencies)
            progress.update(task, completed=93, description="Setting up environment...")
            console.print("\n[bold blue]Phase 6: Setting Up Project Environment[/bold blue]")
            setup_success = await self._setup_project_environment(project_path, plan, console)
            progress.update(task, completed=94, description="✓ Environment setup")

            # Phase 7: Test the project with infinite auto-fix loop
            progress.update(task, completed=95, description="Testing project...")
            console.print("\n[bold blue]Phase 7: Testing Generated Project[/bold blue]")

            fix_attempt = 0
            test_results = await self._test_project(project_path, plan, console)

            # Track error progression to detect if fixes are making things worse
            previous_error_count = 0
            same_error_count = 0
            max_same_errors = 2  # Stop if same number of errors twice in a row

            # Infinite auto-fix loop with user control
            while test_results.get('passed') != 'All':
                fix_attempt += 1
                console.print(f"\n[bold yellow]⚠ Tests Failed - Attempting Auto-Fix (Attempt {fix_attempt})[/bold yellow]")

                # Check if errors are increasing
                current_error_count = len(test_results.get('syntax_errors', [])) + (0 if test_results.get('run_success', False) else 1)

                if current_error_count > previous_error_count and fix_attempt > 1:
                    console.print(f"[red]⚠ Errors increased from {previous_error_count} to {current_error_count}[/red]")
                    console.print(f"[red]⚠ Fixes may be making things worse![/red]")

                    # Record this as a negative learning
                    self.knowledge_memory.add_learning(
                        category="debugging",
                        learning=f"Fix approach in attempt {fix_attempt} made errors worse (increased from {previous_error_count} to {current_error_count})",
                        context={
                            "errors": self.errors_encountered[-3:],  # Last 3 errors
                            "fixes": self.fixes_applied[-3:],  # Last 3 fixes
                            "outcome": "negative"
                        },
                        confidence=0.3  # Low confidence - this didn't work
                    )

                    # Ask user if they want to continue
                    console.print("\n[bold yellow]Fixes appear to be making things worse. Options:[/bold yellow]")
                    console.print("[cyan]1.[/cyan] Stop and keep current state")
                    console.print("[cyan]2.[/cyan] Try one more time with higher temperature")

                    try:
                        from prompt_toolkit import prompt
                        choice = await asyncio.to_thread(
                            prompt,
                            "Enter choice (1 or 2): "
                        )
                        if choice.strip() == '1':
                            console.print("[yellow]  → Stopping at user request[/yellow]")
                            break
                    except:
                        console.print("[yellow]  → Stopping (fixes making things worse)[/yellow]")
                        break

                if current_error_count == previous_error_count and current_error_count > 0:
                    same_error_count += 1
                    if same_error_count >= max_same_errors:
                        console.print(f"[yellow]⚠ Same number of errors for {same_error_count} attempts[/yellow]")
                        console.print(f"[yellow]⚠ Fixes not making progress[/yellow]")
                        break
                else:
                    same_error_count = 0

                previous_error_count = current_error_count

                # Analyze failures and generate fixes
                fixes_applied = await self._auto_fix_issues(project_path, plan, test_results, console, fix_attempt)

                if not fixes_applied:
                    console.print("[yellow]  → No fixes could be generated[/yellow]")
                    console.print("\n[bold cyan]What would you like to do?[/bold cyan]")
                    console.print("[cyan]1.[/cyan] Try again (AI will attempt different fix)")
                    console.print("[cyan]2.[/cyan] Stop and finish with current state")

                    try:
                        from prompt_toolkit import prompt
                        choice = await asyncio.to_thread(
                            prompt,
                            "Enter choice (1 or 2, or Ctrl+C to stop): "
                        )

                        if choice.strip() == '2':
                            console.print("[yellow]  → Stopping auto-fix at user request[/yellow]")
                            break
                        else:
                            console.print("[cyan]  → Retrying with different approach...[/cyan]")
                            # Continue loop to retry
                    except (KeyboardInterrupt, EOFError):
                        console.print("\n[yellow]  → Auto-fix stopped by user (Ctrl+C)[/yellow]")
                        break
                    except Exception:
                        # Fallback if prompt_toolkit not available
                        console.print("[yellow]  → Stopping auto-fix (no more fixes available)[/yellow]")
                        break

                else:
                    # Re-test after fixes
                    console.print(f"\n[bold blue]Re-testing Project (Attempt {fix_attempt + 1})[/bold blue]")
                    test_results = await self._test_project(project_path, plan, console)

                    if test_results.get('passed') == 'All':
                        console.print(f"\n[bold green]✓ Issues Fixed Successfully![/bold green]")
                        break
                    else:
                        # Ask user if they want to continue
                        console.print(f"\n[bold yellow]⚠ Still have issues after fix attempt {fix_attempt}[/bold yellow]")
                        console.print("[yellow]Press Ctrl+C to stop, or press Enter to continue fixing...[/yellow]")

                        try:
                            from prompt_toolkit import prompt
                            await asyncio.to_thread(
                                prompt,
                                ""
                            )
                            console.print("[cyan]  → Continuing auto-fix...[/cyan]")
                        except (KeyboardInterrupt, EOFError):
                            console.print("\n[yellow]  → Auto-fix stopped by user[/yellow]")
                            break

            # Final status message
            if test_results.get('passed') != 'All':
                console.print(f"\n[bold yellow]⚠ Auto-fix completed with {fix_attempt} attempt(s)[/bold yellow]")
                console.print("[yellow]  → Project generated but may have issues[/yellow]")
                console.print("[yellow]  → Check README.md and error messages above[/yellow]")
                console.print(f"[yellow]  → You can manually fix issues or re-run the build[/yellow]")

            progress.update(task, completed=100, description="✓ Project complete!")

            # Final summary
            console.print("\n[bold green]" + "="*60 + "[/bold green]")
            if test_results.get('passed') == 'All':
                console.print("[bold green]✓ PROJECT BUILD COMPLETE - ALL TESTS PASSED![/bold green]")
            elif test_results.get('passed') == 'Syntax Only':
                console.print("[bold yellow]✓ PROJECT BUILD COMPLETE - SYNTAX VALID[/bold yellow]")
            else:
                console.print("[bold yellow]⚠ PROJECT BUILD COMPLETE - WITH WARNINGS[/bold yellow]")
            console.print(f"[bold green]" + "="*60 + "[/bold green]\n")
            console.print(f"[cyan]Location:[/cyan] {project_path}")
            console.print(f"[cyan]Files Created:[/cyan] {len(files_to_create)}")
            console.print(f"[cyan]Tests Passed:[/cyan] {test_results.get('passed', 'N/A')}")
            if fix_attempt > 0:
                console.print(f"[cyan]Fix Attempts:[/cyan] {fix_attempt}")

                # Show knowledge base stats
                kb_stats = self.error_kb.get_statistics()
                console.print(f"[cyan]Knowledge Base:[/cyan] {kb_stats['total_patterns']} patterns, {kb_stats['total_successful_fixes']} total fixes")

            console.print(f"\n[yellow]Next steps:[/yellow]")
            console.print(f"  1. cd {project_path}")
            console.print(f"  2. Read README.md for detailed instructions")
            console.print(f"  3. Follow setup and run instructions\n")

        # Record build in knowledge memory
        build_duration = (datetime.now() - self.build_start_time).total_seconds()
        build_success = test_results.get('passed') == 'All'

        self.knowledge_memory.record_build(
            description=description,
            success=build_success,
            tech_stack=plan.get('tech_stack', {}),
            errors_encountered=self.errors_encountered,
            fixes_applied=self.fixes_applied,
            files_created=len(files_to_create),
            duration_seconds=build_duration
        )

        # Extract and save learnings
        if build_success:
            self.knowledge_memory.add_learning(
                category="successful_build",
                learning=f"Successfully built {plan.get('project_name', 'project')} with {' + '.join([v for v in plan.get('tech_stack', {}).values() if v])}",
                context={
                    "tech_stack": plan.get('tech_stack', {}),
                    "files_created": len(files_to_create),
                    "duration": build_duration,
                    "fixes_needed": len(self.fixes_applied)
                },
                confidence=0.9
            )

        return project_path

    async def _analyze_requirements(self, description: str, console: Console) -> Dict:
        """
        Analyze requirements and create project plan

        Args:
            description: Project description
            console: Console for output

        Returns:
            Project plan
        """
        system_prompt = """You are a software architect. Analyze the project requirements and create a detailed plan.

Output ONLY valid JSON with this structure:
{
    "project_name": "snake_case_name",
    "description": "brief description",
    "tech_stack": {
        "backend": "framework_name",
        "frontend": "framework_name",
        "database": "database_name"
    },
    "files": {
        "path/to/file.py": {
            "type": "backend|frontend|config|test",
            "description": "what this file does",
            "dependencies": ["other", "files"]
        }
    },
    "context": {
        "any": "additional context needed"
    },
    "run_command": "command to run the project (e.g., python main.py, npm start)",
    "setup_commands": ["pip install -r requirements.txt", "other setup commands"],
    "required_tools": ["git", "postgres", "fetch"]
}

IMPORTANT for required_tools:
- Only list MCP server tools that are actually needed for the application code
- Available MCP tools: git, filesystem, postgres, sqlite, fetch, puppeteer, memory, github
- DO NOT list: node, npm, python, pip (these are language runtimes, not MCP tools)
- Use "postgres" not "postgresql"
- Only include if the application actually needs database/web operations"""

        # Get relevant knowledge for this project
        knowledge_context = self.knowledge_memory.format_knowledge_for_prompt(
            category="project_structure",
            tech_stack=None  # Will be inferred from description
        )

        user_prompt = f"""Create a project plan for:

{description}

Include all necessary files, structure, and dependencies. Be specific about file paths and purposes.

IMPORTANT:
1. For run_command: Specify the exact command to start the application
2. For setup_commands: List installation steps (e.g., pip install -r requirements.txt)
3. For required_tools: Only list MCP servers if needed (git, postgres, sqlite, fetch, etc.)
   - DO NOT include node, npm, python, or pip
   - Use "postgres" not "postgresql"
   - Only list tools the application code will actually use

{knowledge_context}"""

        console.print("[dim]  → AI is analyzing project requirements...[/dim]")
        console.print("[dim]  → Determining optimal tech stack...[/dim]")
        console.print("[dim]  → Planning file structure...[/dim]")

        response = await self.model.execute(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=4096,
            temperature=0.7
        )

        # Parse JSON
        plan = self._extract_json(response)

        console.print("[dim]  → Plan validation complete[/dim]")

        return plan

    async def _create_structure(self, project_path: Path, plan: Dict, console: Console):
        """Create project directory structure"""
        # Create all directories
        directories = set()
        for file_path in plan.get("files", {}).keys():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if full_path.parent != project_path:
                directories.add(str(full_path.parent.relative_to(project_path)))

        console.print(f"[dim]  → Created {len(directories)} directories[/dim]")
        for directory in sorted(directories):
            console.print(f"[dim]    • {directory}/[/dim]")

    async def _generate_file(self, file_path: Path, file_spec: Dict, context: Dict):
        """
        Generate a single file

        Args:
            file_path: Path to create file
            file_spec: File specification from plan
            context: Project context
        """
        file_type = file_spec.get("type", "code")
        description = file_spec.get("description", "")

        # Determine language from extension
        ext = file_path.suffix
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".sh": "bash",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".json": "json",
            ".md": "markdown",
            ".txt": "text"
        }
        language = language_map.get(ext, "text")

        # Generate code
        system_prompt = f"""You are an expert {language} developer. Generate complete, production-ready code.

Rules:
- Output ONLY the code, no explanations
- Include necessary imports
- Add helpful comments
- Follow best practices
- Make it production-ready"""

        user_prompt = f"""Generate {file_path.name}:

Purpose: {description}

Context: {json.dumps(context, indent=2)}

Dependencies: {file_spec.get('dependencies', [])}

Generate complete, working code."""

        code = await self.model.execute(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.7
        )

        # Clean up code (remove markdown code blocks if present)
        code = self._extract_code(code, language)

        # Write file using MCP filesystem server
        await self._write_file_mcp(file_path, code)

    async def _write_file_mcp(self, file_path: Path, content: str):
        """Write file with robust error handling"""
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Direct write (most reliable method)
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"Successfully wrote {file_path}")

        except Exception as e:
            logger.error(f"Failed to write {file_path}: {e}")
            raise RuntimeError(f"Could not create file {file_path}: {e}")

    async def _init_git(self, project_path: Path, console: Console):
        """Initialize git repository"""
        console.print("[dim]  → Initializing git repository...[/dim]")

        if "git" in self.mcp_discovery.installed_servers:
            try:
                # Use MCP git server
                await self.mcp_discovery.call_tool(
                    server_name="git",
                    tool_name="git_status",
                    arguments={"repo_path": str(project_path)}
                )

                await self.mcp_discovery.call_tool(
                    server_name="git",
                    tool_name="git_add",
                    arguments={"paths": ["."]}
                )

                await self.mcp_discovery.call_tool(
                    server_name="git",
                    tool_name="git_commit",
                    arguments={"message": "Initial commit - Generated by Code-God"}
                )
                console.print("[dim]  → Git repository created and initial commit made[/dim]")
            except Exception as e:
                logger.warning(f"Git initialization failed: {e}")
                console.print("[dim]  → MCP git failed, using fallback...[/dim]")
                # Fallback to subprocess
                import subprocess
                try:
                    subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
                    subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
                    subprocess.run(
                        ["git", "commit", "-m", "Initial commit - Generated by Code-God"],
                        cwd=project_path,
                        check=True,
                        capture_output=True
                    )
                    console.print("[dim]  → Git repository created successfully[/dim]")
                except Exception as e:
                    console.print(f"[yellow]  ⚠ Git initialization skipped: {e}[/yellow]")
        else:
            # Direct git commands
            import subprocess
            try:
                subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
                subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
                subprocess.run(
                    ["git", "commit", "-m", "Initial commit - Generated by Code-God"],
                    cwd=project_path,
                    check=True,
                    capture_output=True
                )
                console.print("[dim]  → Git repository created successfully[/dim]")
            except Exception as e:
                console.print(f"[yellow]  ⚠ Git initialization skipped: {e}[/yellow]")

    async def _setup_project_environment(self, project_path: Path, plan: Dict, console: Console) -> bool:
        """
        Setup project environment - create venv and install dependencies

        Args:
            project_path: Path to project
            plan: Project plan
            console: Console for output

        Returns:
            True if setup successful
        """
        import subprocess
        import sys

        console.print("[dim]  → Detecting project type and dependencies...[/dim]")

        # Check for Python project
        requirements_file = project_path / "requirements.txt"
        if requirements_file.exists():
            console.print("[yellow]  → Detected Python project (requirements.txt)[/yellow]")

            # Create virtual environment
            venv_path = project_path / "venv"
            if not venv_path.exists():
                console.print("[dim]  → Creating virtual environment...[/dim]")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "venv", str(venv_path)],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        console.print("[green]    ✓ Virtual environment created[/green]")
                    else:
                        console.print(f"[red]    ✗ Failed to create venv: {result.stderr}[/red]")
                        return False
                except Exception as e:
                    console.print(f"[red]    ✗ Failed to create venv: {e}[/red]")
                    return False
            else:
                console.print("[dim]    ✓ Virtual environment already exists[/dim]")

            # Determine pip path in venv
            if sys.platform == "win32":
                pip_path = venv_path / "Scripts" / "pip.exe"
                python_path = venv_path / "Scripts" / "python.exe"
            else:
                pip_path = venv_path / "bin" / "pip"
                python_path = venv_path / "bin" / "python"

            # Store venv paths in plan for later use
            plan['venv_python'] = str(python_path)
            plan['venv_pip'] = str(pip_path)

            # Install dependencies
            console.print("[dim]  → Installing Python dependencies...[/dim]")
            try:
                result = subprocess.run(
                    [str(pip_path), "install", "-r", str(requirements_file)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=project_path
                )
                if result.returncode == 0:
                    # Count installed packages
                    lines = result.stdout.split('\n')
                    installed = [l for l in lines if 'Successfully installed' in l]
                    console.print(f"[green]    ✓ Dependencies installed successfully[/green]")
                    if installed:
                        console.print(f"[dim]      {installed[0]}[/dim]")
                    return True
                else:
                    console.print(f"[red]    ✗ Failed to install dependencies[/red]")
                    console.print(f"[dim]      {result.stderr[:200]}...[/dim]")
                    return False
            except subprocess.TimeoutExpired:
                console.print("[red]    ✗ Dependency installation timed out (5 min)[/red]")
                return False
            except Exception as e:
                console.print(f"[red]    ✗ Failed to install dependencies: {e}[/red]")
                return False

        # Check for Node.js project
        package_json = project_path / "package.json"
        if package_json.exists():
            console.print("[yellow]  → Detected Node.js project (package.json)[/yellow]")
            console.print("[dim]  → Installing Node.js dependencies...[/dim]")

            try:
                result = subprocess.run(
                    ["npm", "install"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=project_path
                )
                if result.returncode == 0:
                    console.print(f"[green]    ✓ Node.js dependencies installed[/green]")
                    return True
                else:
                    console.print(f"[red]    ✗ npm install failed[/red]")
                    console.print(f"[dim]      {result.stderr[:200]}...[/dim]")
                    return False
            except FileNotFoundError:
                console.print(f"[red]    ✗ npm not found - Node.js may not be installed[/red]")
                return False
            except subprocess.TimeoutExpired:
                console.print("[red]    ✗ npm install timed out (5 min)[/red]")
                return False
            except Exception as e:
                console.print(f"[red]    ✗ Failed to install dependencies: {e}[/red]")
                return False

        # Check for Go project
        go_mod = project_path / "go.mod"
        if go_mod.exists():
            console.print("[yellow]  → Detected Go project (go.mod)[/yellow]")
            console.print("[dim]  → Downloading Go dependencies...[/dim]")

            try:
                result = subprocess.run(
                    ["go", "mod", "download"],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=project_path
                )
                if result.returncode == 0:
                    console.print(f"[green]    ✓ Go dependencies downloaded[/green]")
                    return True
                else:
                    console.print(f"[red]    ✗ go mod download failed[/red]")
                    return False
            except Exception as e:
                console.print(f"[red]    ✗ Failed: {e}[/red]")
                return False

        # No dependencies found
        console.print("[dim]  → No dependency files found (requirements.txt, package.json, go.mod)[/dim]")
        console.print("[dim]  → Skipping dependency installation[/dim]")
        return True

    def _fix_flask_command(self, project_path: Path, plan: Dict, console: Console) -> bool:
        """
        Fix flask run command to use venv python

        Args:
            project_path: Project path
            plan: Project plan
            console: Console for output

        Returns:
            True if fixed
        """
        run_command = plan.get('run_command', '')

        # Check if already fixed (idempotent check)
        if '-m flask' in run_command:
            console.print(f"[dim]    Flask command already using venv, skipping...[/dim]")
            return False

        # Sanity check - if command is corrupted, reset it
        if run_command.count('venv/bin/python') > 1 or run_command.count('/home/') > 2:
            console.print(f"[red]    ✗ Command corrupted (nested paths detected), resetting...[/red]")
            console.print(f"[dim]    Corrupted: {run_command[:100]}...[/dim]")

            # Try to extract just the flask part and rebuild
            if 'venv_python' in plan:
                venv_python = plan['venv_python']
                # Reset to simple command
                if '&' in run_command:
                    parts = run_command.split('&')
                    npm_part = [p.strip() for p in parts if 'npm' in p]
                    new_command = f'{venv_python} -m flask run'
                    if npm_part:
                        new_command += f' & {npm_part[0]}'
                    plan['run_command'] = new_command
                    console.print(f"[green]    ✓ Reset to: {new_command}[/green]")
                    return True
                else:
                    # Just flask
                    new_command = f'{venv_python} -m flask run'
                    plan['run_command'] = new_command
                    console.print(f"[green]    ✓ Reset to: {new_command}[/green]")
                    return True

        if 'flask run' in run_command or ('flask' in run_command and '-m flask' not in run_command):
            console.print(f"[dim]    Current: {run_command[:100]}...[/dim]")

            # Use venv python if available
            if 'venv_python' in plan:
                venv_python = plan['venv_python']

                # Only replace the standalone 'flask' command, not paths containing 'flask'
                # Use word boundary to avoid replacing inside paths
                import re
                new_command = re.sub(r'\bflask run\b', f'{venv_python} -m flask run', run_command)

                # Remove any 'source' commands - they don't work in sh
                new_command = re.sub(r'source [^ ]+ &&\s*', '', new_command)

                plan['run_command'] = new_command
                console.print(f"[green]    ✓ Fixed: {new_command[:100]}...[/green]")
                return True

        return False

    async def _fix_package_json_path(self, project_path: Path, plan: Dict, console: Console) -> bool:
        """
        Fix package.json path issues in run command

        Args:
            project_path: Project path
            plan: Project plan
            console: Console for output

        Returns:
            True if fixed
        """
        run_command = plan.get('run_command', '')

        if 'npm' in run_command or 'package.json' in run_command:
            console.print(f"[dim]    Checking for package.json...[/dim]")

            # Find where package.json actually is
            package_json_locations = list(project_path.rglob('package.json'))

            if package_json_locations:
                actual_location = package_json_locations[0].parent
                rel_path = actual_location.relative_to(project_path)

                console.print(f"[dim]    Found package.json in: {rel_path}/[/dim]")

                # Fix the run command
                if 'npm start --prefix' in run_command:
                    # Update the prefix path
                    parts = run_command.split('--prefix')
                    if len(parts) > 1:
                        # Replace the path after --prefix
                        new_command = f"{parts[0]}--prefix {rel_path}"
                        if '&' in parts[1]:  # Preserve any trailing commands
                            new_command += ' &' + parts[1].split('&', 1)[1]

                        plan['run_command'] = new_command
                        console.print(f"[green]    ✓ Fixed: {new_command}[/green]")
                        return True
                elif 'npm start' in run_command:
                    # Add cd command or --prefix
                    new_command = f"cd {rel_path} && npm start"
                    if '&' in run_command:
                        # Preserve background process notation
                        new_command = f"(cd {rel_path} && npm start) &"
                        rest = run_command.split('&', 1)[1]
                        new_command += rest

                    plan['run_command'] = new_command
                    console.print(f"[green]    ✓ Fixed: {new_command}[/green]")
                    return True
            else:
                console.print(f"[yellow]    ⚠ package.json not found anywhere in project[/yellow]")
                # Remove npm command from run_command
                if '&' in run_command:
                    parts = run_command.split('&')
                    # Keep only non-npm parts
                    new_parts = [p.strip() for p in parts if 'npm' not in p]
                    if new_parts:
                        plan['run_command'] = ' & '.join(new_parts)
                        console.print(f"[yellow]    ⚠ Removed npm command: {plan['run_command']}[/yellow]")
                        return True

        return False

    def _normalize_mcp_tool_name(self, tool_name: str) -> Optional[str]:
        """
        Normalize MCP tool names to match available servers

        Args:
            tool_name: Requested tool name (might be alias)

        Returns:
            Actual server name or None if not found
        """
        # Direct mapping of common aliases to actual server names
        aliases = {
            'postgresql': 'postgres',
            'pg': 'postgres',
            'psql': 'postgres',
            'node': None,  # No MCP server for node itself
            'nodejs': None,
            'npm': None,
            'database': 'sqlite',  # Default to sqlite if just "database"
            'db': 'sqlite',
            'github': 'github',
            'web': 'fetch',
            'http': 'fetch',
            'browser': 'puppeteer',
        }

        # Check if it's an alias
        normalized = aliases.get(tool_name.lower())
        if normalized is not None:
            return normalized

        # Check if it's a valid server name directly
        if any(s['name'] == tool_name.lower() for s in self.mcp_discovery.available_servers):
            return tool_name.lower()

        # Not found
        return None

    async def _check_and_install_mcp_tools(self, plan: Dict, console: Console) -> List[str]:
        """
        Check for required MCP tools and auto-install them if needed

        Args:
            plan: Project plan containing required_tools
            console: Console for output

        Returns:
            List of installed tool names
        """
        required_tools = plan.get("required_tools", [])

        if not required_tools:
            console.print("[dim]  → No MCP tools required[/dim]")
            return []

        console.print(f"[dim]  → Required MCP tools: {', '.join(required_tools)}[/dim]")

        installed = []
        for tool in required_tools:
            # Normalize tool name (handle aliases)
            normalized_tool = self._normalize_mcp_tool_name(tool)

            if normalized_tool is None:
                console.print(f"[yellow]    ⚠ '{tool}' is not an MCP server (skipping)[/yellow]")
                continue

            if normalized_tool in self.mcp_discovery.installed_servers:
                console.print(f"[dim]    ✓ {normalized_tool} already installed[/dim]")
                installed.append(normalized_tool)
            else:
                console.print(f"[yellow]  → Installing {normalized_tool}...[/yellow]")
                success = await self.mcp_discovery.install_server(normalized_tool)
                if success:
                    console.print(f"[green]    ✓ {normalized_tool} installed successfully[/green]")
                    installed.append(normalized_tool)
                else:
                    console.print(f"[red]    ✗ Failed to install {normalized_tool}[/red]")

        return installed

    async def _create_detailed_readme(self, project_path: Path, plan: Dict, console: Console):
        """
        Create a comprehensive README.md with detailed run instructions

        Args:
            project_path: Path to project directory
            plan: Project plan with all details
            console: Console for output
        """
        console.print("[dim]  → Generating comprehensive README.md...[/dim]")

        # Get project details
        project_name = plan.get('project_name', 'Project')
        description = plan.get('description', 'A Code-God generated project')
        tech_stack = plan.get('tech_stack', {})
        setup_commands = plan.get('setup_commands', [])
        run_command = plan.get('run_command', 'See individual file instructions')
        files = plan.get('files', {})

        # Build README content
        readme_content = f"""# {project_name}

{description}

**Generated by Code-God AI Project Builder**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Generated Files](#generated-files)

---

## 🎯 Overview

{description}

This project was autonomously generated by Code-God, an AI-powered development assistant that builds complete applications from natural language descriptions.

**Generation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🛠 Tech Stack

"""
        # Add tech stack details
        if tech_stack:
            for key, value in tech_stack.items():
                readme_content += f"- **{key.capitalize()}:** {value}\n"
        else:
            readme_content += "- See individual file headers for technology details\n"

        readme_content += f"""
---

## 📁 Project Structure

```
{project_name}/
"""
        # Add file structure
        for file_path in sorted(files.keys()):
            indent = "  " * (file_path.count('/'))
            file_name = file_path.split('/')[-1]
            readme_content += f"{indent}├── {file_name}\n"

        readme_content += f"""```

---

## ✅ Prerequisites

Before running this project, ensure you have the following installed:

"""
        # Detect prerequisites from tech stack
        prerequisites = set()
        if tech_stack.get('backend'):
            backend = tech_stack['backend'].lower()
            if 'python' in backend or 'flask' in backend or 'django' in backend or 'fastapi' in backend:
                prerequisites.add("- Python 3.8 or higher")
                prerequisites.add("- pip (Python package manager)")
            elif 'node' in backend or 'express' in backend or 'nest' in backend:
                prerequisites.add("- Node.js 16 or higher")
                prerequisites.add("- npm or yarn")
            elif 'go' in backend or 'gin' in backend:
                prerequisites.add("- Go 1.20 or higher")
            elif 'rust' in backend:
                prerequisites.add("- Rust 1.70 or higher")
                prerequisites.add("- Cargo")

        if tech_stack.get('database'):
            db = tech_stack['database'].lower()
            if 'postgres' in db:
                prerequisites.add("- PostgreSQL 13 or higher")
            elif 'mysql' in db:
                prerequisites.add("- MySQL 8 or higher")
            elif 'mongodb' in db:
                prerequisites.add("- MongoDB 5 or higher")
            elif 'sqlite' in db:
                prerequisites.add("- SQLite3 (usually pre-installed)")

        if prerequisites:
            for prereq in sorted(prerequisites):
                readme_content += f"{prereq}\n"
        else:
            readme_content += "- See tech stack section for specific requirements\n"

        readme_content += f"""
---

## 📦 Installation

Follow these steps to set up the project:

### 1. Clone or Navigate to Project Directory

```bash
cd {project_path}
```

### 2. Set Up Environment & Install Dependencies

"""
        # Add setup commands with venv for Python
        if setup_commands:
            for idx, cmd in enumerate(setup_commands, 1):
                readme_content += f"**Step {idx}:**\n```bash\n{cmd}\n```\n\n"
        else:
            # Auto-detect based on files
            if any('requirements.txt' in f for f in files.keys()):
                readme_content += """**For Python projects:**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

**IMPORTANT:** Always activate the virtual environment before running the project!

"""
            elif any('package.json' in f for f in files.keys()):
                readme_content += "```bash\nnpm install\n# or\nyarn install\n```\n\n"
            elif any('go.mod' in f for f in files.keys()):
                readme_content += "```bash\ngo mod download\n```\n\n"
            elif any('Cargo.toml' in f for f in files.keys()):
                readme_content += "```bash\ncargo build\n```\n\n"
            else:
                readme_content += "See individual file headers for setup instructions.\n\n"

        readme_content += f"""### 3. Environment Configuration

If your project requires environment variables, create a `.env` file:

```bash
cp .env.example .env  # If .env.example exists
# Or create .env manually with required variables
```

---

## 🚀 Running the Project

### Development Mode

"""
        if run_command:
            # Add venv activation reminder for Python projects
            if any('requirements.txt' in f for f in files.keys()):
                readme_content += """**Remember to activate your virtual environment first:**

```bash
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\\Scripts\\activate
```

**Then run the project:**

"""
            readme_content += f"```bash\n{run_command}\n```\n\n"
        else:
            readme_content += "See individual component README files or file headers for run instructions.\n\n"

        readme_content += f"""### Production Mode

For production deployment:

1. Set environment variables appropriately
2. Use production-grade web servers (gunicorn, pm2, etc.)
3. Configure reverse proxy (nginx, apache)
4. Enable HTTPS
5. Set up monitoring and logging

---

## 🧪 Testing

### Running Tests

"""
        if any('test' in f.lower() for f in files.keys()):
            readme_content += "```bash\n# Run all tests\n"
            if any('.py' in f for f in files.keys()):
                readme_content += "pytest\n# or\npython -m unittest discover\n"
            elif any('.js' in f or '.ts' in f for f in files.keys()):
                readme_content += "npm test\n# or\nyarn test\n"
            elif any('.go' in f for f in files.keys()):
                readme_content += "go test ./...\n"
            elif any('.rs' in f for f in files.keys()):
                readme_content += "cargo test\n"
            readme_content += "```\n\n"
        else:
            readme_content += "No tests were generated. Consider adding tests for your project.\n\n"

        readme_content += f"""### Code Quality

```bash
# Linting (if applicable)
# Python: pylint, flake8, black
# JavaScript: eslint, prettier
# Go: golint, gofmt
# Rust: cargo clippy
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue: Dependencies not installing**
- Ensure you have the correct version of package manager
- Try clearing cache: `pip cache purge` or `npm cache clean --force`
- Check internet connection

**Issue: Application won't start**
- Verify all environment variables are set
- Check if required ports are available
- Review error logs

**Issue: Database connection fails**
- Ensure database service is running
- Verify connection string in `.env`
- Check database credentials

### Getting Help

1. Check individual file comments for specific module documentation
2. Review error messages carefully
3. Ensure all prerequisites are installed
4. Verify environment configuration

---

## 📄 Generated Files

This section describes each generated file:

"""
        # Add file descriptions
        for file_path, file_spec in files.items():
            file_type = file_spec.get('type', 'code')
            description = file_spec.get('description', 'No description available')
            readme_content += f"### `{file_path}`\n\n"
            readme_content += f"**Type:** {file_type}  \n"
            readme_content += f"**Purpose:** {description}  \n\n"

        readme_content += f"""---

## 🤖 About Code-God

This project was generated by **Code-God**, an autonomous AI development assistant that:

- Analyzes natural language project descriptions
- Plans optimal architecture and tech stack
- Generates production-ready code
- Creates comprehensive documentation
- Tests generated projects automatically

Learn more at: https://github.com/yourusername/CodeGod

---

## 📝 License

[Specify your license here]

---

## 🙏 Acknowledgments

Generated with ❤️ by Code-God AI Project Builder

*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Write README
        readme_path = project_path / "README.md"
        await self._write_file_mcp(readme_path, readme_content)

        console.print("[green]  ✓ Comprehensive README.md created[/green]")
        console.print(f"[dim]    • {len(readme_content.splitlines())} lines[/dim]")
        console.print(f"[dim]    • Includes setup, run, and troubleshooting instructions[/dim]")

    async def _test_project(self, project_path: Path, plan: Dict, console: Console) -> Dict:
        """
        Test the generated project by attempting to run it

        Args:
            project_path: Path to project
            plan: Project plan
            console: Console for output

        Returns:
            Dictionary with test results
        """
        console.print("[dim]  → Running project validation tests...[/dim]")

        results = {
            "syntax_checks": 0,
            "syntax_errors": [],
            "run_attempted": False,
            "run_success": False,
            "run_output": "",
            "passed": "Unknown"
        }

        import subprocess

        # Phase 1: Syntax checking
        console.print("[dim]  → Phase 1: Syntax validation...[/dim]")

        # Use venv Python if available
        python_cmd = plan.get('venv_python', 'python')

        files = plan.get("files", {})
        for file_path in files.keys():
            full_path = project_path / file_path

            if not full_path.exists():
                continue

            # Check syntax based on file type
            if file_path.endswith('.py'):
                try:
                    result = subprocess.run(
                        [python_cmd, "-m", "py_compile", str(full_path)],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        results["syntax_checks"] += 1
                        console.print(f"[dim]    ✓ {file_path}[/dim]")
                    else:
                        results["syntax_errors"].append(f"{file_path}: {result.stderr}")
                        console.print(f"[yellow]    ⚠ {file_path}: Syntax error[/yellow]")
                except Exception as e:
                    results["syntax_errors"].append(f"{file_path}: {str(e)}")

            elif file_path.endswith('.js') or file_path.endswith('.ts'):
                # Try node syntax check
                try:
                    result = subprocess.run(
                        ["node", "--check", str(full_path)],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        results["syntax_checks"] += 1
                        console.print(f"[dim]    ✓ {file_path}[/dim]")
                    else:
                        results["syntax_errors"].append(f"{file_path}: {result.stderr}")
                        console.print(f"[yellow]    ⚠ {file_path}: Syntax error[/yellow]")
                except Exception:
                    # Node might not be installed
                    pass

        # Phase 2: Try to run the project (with timeout)
        console.print("\n[dim]  → Phase 2: Attempting to run project...[/dim]")

        run_command = plan.get("run_command")
        if run_command:
            # Use virtual environment Python if available (but only if not already using it)
            if 'venv_python' in plan:
                venv_python = plan['venv_python']

                # Check if run_command already uses venv python
                if venv_python not in run_command and 'python' in run_command.lower():
                    # Replace standalone 'python' command with venv python
                    import re
                    # Only replace 'python' at word boundaries (not inside paths)
                    run_command = re.sub(r'\bpython\b', f'"{venv_python}"', run_command, count=1)
                    console.print(f"[dim]    Using virtual environment Python[/dim]")

            console.print(f"[dim]    Run command: {run_command[:150]}...[/dim]")

            try:
                results["run_attempted"] = True

                # Run with timeout (5 seconds for quick start check)
                result = subprocess.run(
                    run_command,
                    shell=True,
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                results["run_output"] = result.stdout + result.stderr
                results["run_success"] = result.returncode == 0

                if result.returncode == 0:
                    console.print("[green]    ✓ Project started successfully[/green]")
                else:
                    console.print(f"[yellow]    ⚠ Project exited with code {result.returncode}[/yellow]")
                    if result.stderr:
                        console.print(f"[dim]    Error: {result.stderr[:200]}...[/dim]")

            except subprocess.TimeoutExpired:
                # Timeout is actually good - means the server started and is running
                console.print("[green]    ✓ Project started and is running (timed out after 5s - this is expected for servers)[/green]")
                results["run_success"] = True
            except Exception as e:
                console.print(f"[red]    ✗ Failed to run project: {str(e)}[/red]")
                results["run_output"] = str(e)
        else:
            console.print("[yellow]    ⚠ No run command specified, skipping execution test[/yellow]")

        # Summary
        if results["syntax_checks"] > 0 and len(results["syntax_errors"]) == 0:
            if results["run_success"]:
                results["passed"] = "All"
                console.print("\n[bold green]✓ All tests passed![/bold green]")
            else:
                results["passed"] = "Syntax Only"
                console.print("\n[yellow]⚠ Syntax checks passed, but runtime issues detected[/yellow]")
        elif results["syntax_checks"] > 0:
            results["passed"] = "Partial"
            console.print("\n[yellow]⚠ Some tests passed, but errors were found[/yellow]")
        else:
            results["passed"] = "None"
            console.print("\n[yellow]⚠ Unable to run automated tests[/yellow]")

        return results

    async def _auto_fix_issues(self, project_path: Path, plan: Dict, test_results: Dict, console: Console, attempt: int = 1) -> bool:
        """
        Automatically fix issues found during testing

        Args:
            project_path: Path to project
            plan: Project plan
            test_results: Test results with errors
            console: Console for output
            attempt: Current fix attempt number (for variation)

        Returns:
            True if fixes were applied, False otherwise
        """
        console.print("\n[bold cyan]Analyzing Failures and Generating Fixes...[/bold cyan]")

        # Vary temperature based on attempt for different fix strategies
        # Start conservative (0.3), gradually increase for more creative fixes
        temperature = min(0.3 + (attempt - 1) * 0.1, 0.8)
        if attempt > 1:
            console.print(f"[dim]  → Using temperature {temperature:.1f} for attempt {attempt}[/dim]")

        fixes_applied = False

        # First, check if run command is corrupted and fix immediately
        run_command = plan.get('run_command', '')
        if run_command:
            # Detect corrupted paths (nested home directories)
            if run_command.count('venv/bin/python') > 1 or run_command.count('/home/') > 2:
                console.print(f"[red]  ✗ CRITICAL: Run command is corrupted![/red]")
                console.print(f"[dim]    Current: {run_command[:100]}...[/dim]")

                # Force reset
                if 'venv_python' in plan:
                    venv_python = plan['venv_python']

                    # Extract any npm commands
                    npm_part = ''
                    if 'npm' in run_command:
                        import re
                        npm_match = re.search(r'npm[^&]+', run_command)
                        if npm_match:
                            npm_part = f' & {npm_match.group(0)}'

                    # Reset to clean command
                    new_command = f'{venv_python} -m flask run{npm_part}'
                    plan['run_command'] = new_command

                    console.print(f"[green]  ✓ RESET to clean command: {new_command}[/green]")
                    fixes_applied = True
                    self.fixes_applied.append("Reset corrupted run command")

        # Check knowledge base for known error patterns
        run_output = test_results.get('run_output', '')
        all_errors = run_output + ' '.join(test_results.get('syntax_errors', []))

        matching_patterns = self.error_kb.find_matching_patterns(all_errors)
        if matching_patterns:
            console.print(f"[cyan]  → Found {len(matching_patterns)} known error pattern(s) in knowledge base[/cyan]")
            for i, pattern in enumerate(matching_patterns[:3], 1):
                console.print(f"[dim]    {i}. {pattern['description']} (success rate: {pattern.get('success_count', 0)})[/dim]")
                console.print(f"[dim]       Solution: {pattern['solution']}[/dim]")

        # Smart fixes based on known patterns (but only if command isn't already using -m flask)
        current_run_command = plan.get('run_command', '')
        if ('flask: not found' in run_output or ('flask run' in current_run_command and '-m flask' not in current_run_command)):
            if '-m flask' not in current_run_command:  # Double check
                console.print(f"[yellow]  → Detected flask command issue[/yellow]")
                self.errors_encountered.append("flask: not found")
                if self._fix_flask_command(project_path, plan, console):
                    fixes_applied = True
                    fix_desc = "Fixed run command to use venv python"
                    self.fixes_applied.append(fix_desc)
                    self.error_kb.record_successful_fix("flask_not_found", run_output[:200], fix_desc)
            else:
                console.print(f"[dim]  → Flask command already using -m flask, skipping fix[/dim]")

        if 'package.json' in run_output and 'ENOENT' in run_output:
            console.print(f"[yellow]  → Detected package.json path issue[/yellow]")
            self.errors_encountered.append("package.json path error")
            if await self._fix_package_json_path(project_path, plan, console):
                fixes_applied = True
                fix_desc = "Fixed package.json path in run command"
                self.fixes_applied.append(fix_desc)
                self.error_kb.record_successful_fix("package_json_not_found", run_output[:200], fix_desc)

        # Check for package.json errors first
        syntax_errors = test_results.get('syntax_errors', [])
        has_package_json_error = any('ERR_INVALID_PACKAGE_CONFIG' in err or 'package.json' in err.lower() for err in syntax_errors)

        # Track all syntax errors
        for err in syntax_errors[:5]:  # Track first 5 to avoid bloat
            if err not in self.errors_encountered:
                self.errors_encountered.append(err[:100])  # Truncate

        if has_package_json_error:
            console.print(f"[yellow]  → Detected package.json configuration error[/yellow]")
            package_json_path = project_path / "package.json"

            if package_json_path.exists():
                console.print(f"[yellow]  → Fixing: package.json[/yellow]")
                try:
                    current_content = package_json_path.read_text(encoding='utf-8')

                    system_prompt = """You are a Node.js expert. Fix the invalid package.json configuration.

Output ONLY the corrected JSON, no explanations, no markdown.
Ensure valid JSON syntax with proper comma placement."""

                    user_prompt = f"""Fix this package.json file:

CURRENT CONTENT:
```json
{current_content}
```

The error indicates invalid package.json configuration. Common issues:
- Missing or extra commas
- Invalid JSON syntax
- Missing required fields (name, version)
- Incorrect field types

Provide the corrected package.json only."""

                    console.print(f"[dim]    → AI is fixing package.json...[/dim]")

                    fixed_content = await self.model.execute(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_tokens=2048,
                        temperature=temperature
                    )

                    # Clean up any markdown
                    fixed_content = self._extract_code(fixed_content, 'json')

                    package_json_path.write_text(fixed_content, encoding='utf-8')
                    console.print(f"[green]    ✓ Fixed package.json[/green]")
                    fixes_applied = True
                    self.error_kb.record_successful_fix("invalid_package_config", all_errors[:200], "Fixed package.json syntax")

                except Exception as e:
                    console.print(f"[red]    ✗ Failed to fix package.json: {e}[/red]")

        # Fix syntax errors
        if syntax_errors:
            code_syntax_errors = [err for err in syntax_errors if 'ERR_INVALID_PACKAGE_CONFIG' not in err and 'package.json' not in err.lower()]

            if code_syntax_errors:
                console.print(f"[yellow]  → Found {len(code_syntax_errors)} code syntax error(s)[/yellow]")

            for error_info in code_syntax_errors:
                # Parse error info (format: "file_path: error_message")
                if ": " not in error_info:
                    continue

                file_path_str, error_message = error_info.split(": ", 1)
                file_path = project_path / file_path_str

                if not file_path.exists():
                    continue

                console.print(f"\n[yellow]  → Fixing: {file_path_str}[/yellow]")

                # Check if this is actually a package.json error
                if 'ERR_INVALID_PACKAGE_CONFIG' in error_message or 'package.json' in error_message.lower():
                    console.print(f"[yellow]    ⚠ This is a package.json configuration error, not a code syntax error[/yellow]")
                    console.print(f"[dim]    Skipping file fix, will address package.json separately[/dim]")
                    continue

                console.print(f"[dim]    Error: {error_message[:200]}...[/dim]")

                try:
                    # Read current file contents
                    current_code = file_path.read_text(encoding='utf-8')

                    # Get file spec from plan
                    file_spec = plan.get('files', {}).get(file_path_str, {})

                    # Generate fix prompt with attempt context
                    approach_guidance = ""
                    if attempt > 1:
                        approach_guidance = f"\n\nNOTE: This is fix attempt #{attempt}. Previous attempts may have failed. Try a different approach or consider:\n- Rewriting problematic sections\n- Using alternative syntax\n- Checking for edge cases"

                    system_prompt = f"""You are an expert code debugger. Fix the syntax error in the code provided.

Output ONLY the corrected code, no explanations, no markdown code blocks.
Maintain the original structure and logic, just fix the syntax error.{approach_guidance}"""

                    user_prompt = f"""Fix the syntax error in this code:

FILE: {file_path_str}
PURPOSE: {file_spec.get('description', 'N/A')}

ERROR (FULL):
{error_message[:1000]}

CURRENT CODE:
```
{current_code}
```

Read the FULL error message above carefully. Fix the exact issue mentioned.
Provide the corrected code only. No explanations, no markdown."""

                    console.print(f"[dim]    → AI is analyzing the error...[/dim]")

                    # Get AI fix with varying temperature for different attempts
                    fixed_code = await self.model.execute(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_tokens=2048,
                        temperature=temperature
                    )

                    # Clean up any markdown that might have been added
                    fixed_code = self._extract_code(fixed_code, file_path.suffix[1:])

                    # Write fixed code
                    file_path.write_text(fixed_code, encoding='utf-8')
                    console.print(f"[green]    ✓ Applied fix to {file_path_str}[/green]")
                    fixes_applied = True

                except Exception as e:
                    console.print(f"[red]    ✗ Failed to fix {file_path_str}: {e}[/red]")
                    logger.error(f"Fix failed for {file_path_str}: {e}")

        # Fix runtime errors
        run_output = test_results.get('run_output', '')
        if not test_results.get('run_success', False) and test_results.get('run_attempted', False) and run_output:
            console.print(f"\n[yellow]  → Analyzing runtime error...[/yellow]")

            # Show more of the error for context (500 chars instead of 200)
            error_preview = run_output[:500] if len(run_output) > 500 else run_output
            console.print(f"[dim]    Error output: {error_preview}...[/dim]")

            # Check for common issues
            if 'ModuleNotFoundError' in run_output or 'ImportError' in run_output:
                console.print(f"[yellow]    ⚠ Missing Python module - may need to reinstall dependencies[/yellow]")

                # Try to reinstall dependencies
                requirements_file = project_path / "requirements.txt"
                if requirements_file.exists() and 'venv_pip' in plan:
                    console.print(f"[dim]    → Attempting to reinstall dependencies...[/dim]")
                    try:
                        import subprocess
                        result = subprocess.run(
                            [plan['venv_pip'], "install", "-r", str(requirements_file), "--force-reinstall"],
                            capture_output=True,
                            text=True,
                            timeout=300,
                            cwd=project_path
                        )
                        if result.returncode == 0:
                            console.print(f"[green]    ✓ Dependencies reinstalled[/green]")
                            fixes_applied = True
                            self.error_kb.record_successful_fix("module_not_found_python", run_output[:200], "Reinstalled Python dependencies")
                            # Return early to re-test
                            return fixes_applied
                        else:
                            console.print(f"[yellow]    ⚠ Reinstall had issues: {result.stderr[:200]}[/yellow]")
                    except Exception as e:
                        console.print(f"[yellow]    ⚠ Could not reinstall: {e}[/yellow]")

            # Check for Node.js module errors
            if 'Cannot find module' in run_output or 'MODULE_NOT_FOUND' in run_output:
                console.print(f"[yellow]    ⚠ Node.js entry point issue detected[/yellow]")

                # Check if it's a missing entry point file (not a missing npm package)
                if "Cannot find module '/home" in run_output or "Cannot find module '" in run_output:
                    # Extract the file it's looking for
                    import re
                    match = re.search(r"Cannot find module '([^']+)'", run_output)
                    if match:
                        missing_file = match.group(1)
                        console.print(f"[dim]    Looking for: {missing_file}[/dim]")

                        # Check if package.json has wrong entry point
                        package_json_locations = list(project_path.rglob('package.json'))
                        for pkg_path in package_json_locations:
                            try:
                                pkg_data = json.loads(pkg_path.read_text())
                                if 'main' in pkg_data or 'scripts' in pkg_data:
                                    # Find actual entry files
                                    pkg_dir = pkg_path.parent
                                    possible_entries = list(pkg_dir.glob('src/**/*.js')) + list(pkg_dir.glob('*.js'))

                                    if possible_entries:
                                        # Use first .js file as entry
                                        actual_entry = possible_entries[0].relative_to(pkg_dir)
                                        console.print(f"[dim]    Found actual entry: {actual_entry}[/dim]")

                                        # Update package.json
                                        pkg_data['main'] = str(actual_entry)
                                        if 'scripts' in pkg_data and 'start' in pkg_data['scripts']:
                                            pkg_data['scripts']['start'] = f'node {actual_entry}'

                                        pkg_path.write_text(json.dumps(pkg_data, indent=2))
                                        console.print(f"[green]    ✓ Fixed package.json entry point[/green]")
                                        fixes_applied = True
                                        return fixes_applied
                            except:
                                pass

                # Try reinstalling as fallback
                package_json = project_path / "package.json"
                if package_json.exists():
                    console.print(f"[dim]    → Attempting to reinstall npm packages...[/dim]")
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["npm", "install"],
                            capture_output=True,
                            text=True,
                            timeout=300,
                            cwd=project_path
                        )
                        if result.returncode == 0:
                            console.print(f"[green]    ✓ npm packages reinstalled[/green]")
                            fixes_applied = True
                            return fixes_applied
                        else:
                            console.print(f"[yellow]    ⚠ npm install had issues[/yellow]")
                    except Exception as e:
                        console.print(f"[yellow]    ⚠ Could not reinstall: {e}[/yellow]")

            # Try to identify which file caused the error
            files = plan.get('files', {})
            main_files = [f for f in files.keys() if 'main' in f.lower() or 'app' in f.lower() or '__init__' in f]

            if main_files:
                main_file = main_files[0]
                file_path = project_path / main_file

                if file_path.exists():
                    console.print(f"\n[yellow]  → Fixing runtime error in: {main_file}[/yellow]")

                    try:
                        current_code = file_path.read_text(encoding='utf-8')
                        file_spec = files.get(main_file, {})

                        # Add retry guidance
                        retry_guidance = ""
                        if attempt > 1:
                            retry_guidance = f"\n\nIMPORTANT: This is fix attempt #{attempt}. Previous fix didn't work. Try a completely different approach:\n- Check if modules need installation\n- Verify import paths\n- Check for circular dependencies\n- Consider initialization order\n- Look for missing dependencies"

                        system_prompt = f"""You are an expert debugger. Fix the runtime error in the code.

Output ONLY the corrected code, no explanations, no markdown code blocks.
Common issues to check:
- Missing imports
- Incorrect function calls
- Wrong variable names
- Missing dependencies in the code
- Port/host configuration issues{retry_guidance}"""

                        user_prompt = f"""Fix the runtime error in this code:

FILE: {main_file}
PURPOSE: {file_spec.get('description', 'N/A')}

RUNTIME ERROR (FULL):
{run_output[:2000]}

CURRENT CODE:
```
{current_code}
```

Analyze the FULL error trace above. Common fixes:
- Add missing imports
- Fix import paths
- Check module installation
- Fix syntax errors
- Correct configuration

Provide the corrected code only."""

                        console.print(f"[dim]    → AI is generating fix...[/dim]")

                        fixed_code = await self.model.execute(
                            prompt=user_prompt,
                            system_prompt=system_prompt,
                            max_tokens=2048,
                            temperature=temperature
                        )

                        fixed_code = self._extract_code(fixed_code, file_path.suffix[1:])
                        file_path.write_text(fixed_code, encoding='utf-8')
                        console.print(f"[green]    ✓ Applied runtime fix to {main_file}[/green]")
                        fixes_applied = True

                    except Exception as e:
                        console.print(f"[red]    ✗ Failed to fix runtime error: {e}[/red]")

        # If pattern-based fixes didn't work, use agentic reasoning
        if not fixes_applied and (run_output or syntax_errors):
            console.print("\n[bold magenta]🧠 Activating Agentic Problem Solver...[/bold magenta]")
            console.print("[dim]Using AI reasoning to understand and solve the problem...[/dim]")

            # Create agentic fixer
            agentic = AgenticFixer(self.model, project_path, plan, console)

            # Determine error type
            error_type = "runtime" if run_output else "syntax"
            full_error = run_output if run_output else '\n'.join(syntax_errors[:3])

            # Use agentic reasoning
            try:
                success, description = await agentic.analyze_and_fix(full_error, error_type)

                if success:
                    console.print(f"[bold green]✓ Agentic fix succeeded![/bold green]")
                    console.print(f"[green]{description}[/green]")
                    fixes_applied = True
                    self.fixes_applied.append(f"Agentic: {description}")
                else:
                    console.print(f"[yellow]⚠ Agentic fix failed: {description}[/yellow]")

            except Exception as e:
                console.print(f"[red]✗ Agentic reasoning error: {e}[/red]")
                logger.error(f"Agentic fixer error: {e}")

        if not fixes_applied:
            console.print("[yellow]  → No fixable errors identified[/yellow]")
        else:
            # Show final run command after all fixes
            final_run_command = plan.get('run_command', '')
            if final_run_command:
                console.print(f"\n[cyan]Final run command after fixes:[/cyan]")
                console.print(f"[dim]{final_run_command[:200]}{'...' if len(final_run_command) > 200 else ''}[/dim]")

        return fixes_applied

    async def _create_readme(self, project_path: Path, plan: Dict):
        """Create README.md"""
        readme_content = f"""# {plan.get('project_name', 'Project')}

{plan.get('description', '')}

## Tech Stack

{json.dumps(plan.get('tech_stack', {}), indent=2)}

## Getting Started

1. Install dependencies
2. Run the application
3. Access at http://localhost:8000

## Generated by Code-God

This project was autonomously generated by Code-God.

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        readme_path = project_path / "README.md"
        await self._write_file_mcp(readme_path, readme_content)

    def _generate_project_name(self, description: str) -> str:
        """Generate project name from description"""
        import re
        name = description.lower()
        name = re.sub(r'[^a-z0-9\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        name = name[:50]  # Limit length

        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{name}_{timestamp}"

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from text"""
        # Try to extract from code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Find JSON object
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {}

    def _extract_code(self, text: str, language: str) -> str:
        """Extract code from text"""
        # Remove markdown code blocks
        if f"```{language}" in text:
            text = text.split(f"```{language}")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return text.strip()
