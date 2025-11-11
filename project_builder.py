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

logger = logging.getLogger(__name__)


class ProjectBuilder:
    """
    Autonomous project builder
    """

    def __init__(self, model: LocalModelExecutor, mcp_discovery: MCPDiscovery):
        self.model = model
        self.mcp_discovery = mcp_discovery

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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            # Phase 1: Analyze and plan
            task = progress.add_task("Analyzing requirements...", total=100)
            plan = await self._analyze_requirements(description)
            progress.update(task, completed=20, description="✓ Requirements analyzed")

            # Phase 2: Generate file structure
            progress.update(task, completed=30, description="Creating project structure...")
            await self._create_structure(project_path, plan)
            progress.update(task, completed=40, description="✓ Structure created")

            # Phase 3: Generate code files
            progress.update(task, completed=50, description="Generating code...")
            files_to_create = plan.get("files", {})

            for idx, (file_path, file_spec) in enumerate(files_to_create.items()):
                file_progress = 50 + (30 * (idx + 1) / len(files_to_create))
                progress.update(task, completed=file_progress, description=f"Generating {file_path}...")

                await self._generate_file(
                    project_path / file_path,
                    file_spec,
                    plan.get("context", {})
                )

            progress.update(task, completed=80, description="✓ Code generated")

            # Phase 4: Initialize git
            progress.update(task, completed=85, description="Initializing git...")
            await self._init_git(project_path)
            progress.update(task, completed=90, description="✓ Git initialized")

            # Phase 5: Create documentation
            progress.update(task, completed=95, description="Creating documentation...")
            await self._create_readme(project_path, plan)
            progress.update(task, completed=100, description="✓ Project complete!")

        return project_path

    async def _analyze_requirements(self, description: str) -> Dict:
        """
        Analyze requirements and create project plan

        Args:
            description: Project description

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
    }
}"""

        user_prompt = f"""Create a project plan for:

{description}

Include all necessary files, structure, and dependencies. Be specific about file paths and purposes."""

        response = await self.model.execute(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=4096,
            temperature=0.7
        )

        # Parse JSON
        plan = self._extract_json(response)

        return plan

    async def _create_structure(self, project_path: Path, plan: Dict):
        """Create project directory structure"""
        # Create all directories
        for file_path in plan.get("files", {}).keys():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

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

    async def _init_git(self, project_path: Path):
        """Initialize git repository"""
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
            except Exception as e:
                logger.warning(f"Git initialization failed: {e}")
                # Fallback to subprocess
                import subprocess
                try:
                    subprocess.run(["git", "init"], cwd=project_path, check=True)
                    subprocess.run(["git", "add", "."], cwd=project_path, check=True)
                    subprocess.run(
                        ["git", "commit", "-m", "Initial commit - Generated by Code-God"],
                        cwd=project_path,
                        check=True
                    )
                except:
                    pass
        else:
            # Direct git commands
            import subprocess
            try:
                subprocess.run(["git", "init"], cwd=project_path, check=True)
                subprocess.run(["git", "add", "."], cwd=project_path, check=True)
                subprocess.run(
                    ["git", "commit", "-m", "Initial commit - Generated by Code-God"],
                    cwd=project_path,
                    check=True
                )
            except:
                pass

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
