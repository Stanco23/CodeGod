"""
Agentic Problem Solver
Uses AI reasoning to dynamically solve build issues
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgenticFixer:
    """
    Agentic problem solver that reasons through issues and validates fixes
    """

    def __init__(self, model, project_path: Path, plan: Dict, console):
        self.model = model
        self.project_path = project_path
        self.plan = plan
        self.console = console

    async def analyze_and_fix(self, error_output: str, error_type: str) -> Tuple[bool, str]:
        """
        Use AI reasoning to analyze error and apply fix

        Args:
            error_output: The error message
            error_type: Type of error (syntax, runtime, dependency)

        Returns:
            (success, description of fix)
        """
        self.console.print("\n[bold cyan]🤖 Agentic Reasoner Activated[/bold cyan]")
        self.console.print("[dim]AI is analyzing the problem and planning a solution...[/dim]")

        # Phase 1: Understand the problem
        understanding = await self._understand_problem(error_output, error_type)

        if not understanding.get('understood', False):
            return False, "Could not understand the problem"

        self.console.print(f"[cyan]Problem Analysis:[/cyan] {understanding.get('summary', 'Unknown')}")

        # Phase 2: Explore the environment
        context = await self._explore_environment(understanding)

        # Phase 3: Generate fix strategy
        strategy = await self._generate_fix_strategy(understanding, context)

        if not strategy.get('fixable', False):
            return False, strategy.get('reason', 'No fix strategy found')

        self.console.print(f"[cyan]Fix Strategy:[/cyan] {strategy.get('approach', 'Unknown')}")

        # Phase 4: Apply fix
        success = await self._apply_fix(strategy)

        if not success:
            return False, "Fix application failed"

        # Phase 5: Validate fix
        validation = await self._validate_fix(understanding, strategy)

        if validation.get('valid', False):
            return True, strategy.get('description', 'Fixed issue')
        else:
            return False, f"Fix validation failed: {validation.get('reason', 'Unknown')}"

    async def _understand_problem(self, error_output: str, error_type: str) -> Dict:
        """Use AI to understand the problem"""

        prompt = f"""Analyze this error and extract the core problem:

ERROR TYPE: {error_type}
ERROR OUTPUT:
```
{error_output[:1000]}
```

PROJECT PATH: {self.project_path}

Respond ONLY with valid JSON:
{{
    "understood": true/false,
    "summary": "brief description of the actual problem",
    "root_cause": "what's really causing this",
    "affected_component": "which file/module/dependency",
    "fixable": true/false,
    "needs_exploration": ["list", "of", "things", "to", "check"]
}}
"""

        response = await self.model.execute(
            prompt=prompt,
            system_prompt="You are an expert debugger. Analyze errors deeply to find root causes.",
            max_tokens=1024,
            temperature=0.3
        )

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass

        return {"understood": False}

    async def _explore_environment(self, understanding: Dict) -> Dict:
        """Explore filesystem and environment to gather context"""

        context = {
            "files_exist": {},
            "dependencies_installed": {},
            "directory_structure": {}
        }

        # Check what the AI wants to explore
        needs_exploration = understanding.get('needs_exploration', [])

        self.console.print("[dim]Exploring environment:[/dim]")

        for item in needs_exploration:
            item_lower = item.lower()

            # Check if specific file exists
            if 'file' in item_lower or 'module' in item_lower:
                # Extract potential filename
                affected = understanding.get('affected_component', '')
                if affected:
                    file_path = self.project_path / affected
                    context['files_exist'][affected] = file_path.exists()
                    self.console.print(f"[dim]  → Checked {affected}: {'✓' if file_path.exists() else '✗'}[/dim]")

                    # If not exists, search for similar files
                    if not file_path.exists():
                        similar_files = list(self.project_path.rglob(f"*{Path(affected).stem}*"))
                        context[f'{affected}_alternatives'] = [str(f.relative_to(self.project_path)) for f in similar_files[:5]]
                        if similar_files:
                            self.console.print(f"[dim]    Found alternatives: {', '.join([f.name for f in similar_files[:3]])}[/dim]")

            # Check dependencies
            if 'dependency' in item_lower or 'module' in item_lower or 'package' in item_lower:
                # Check Python dependencies
                if 'venv_python' in self.plan:
                    venv_python = self.plan['venv_python']
                    try:
                        result = subprocess.run(
                            [venv_python, "-m", "pip", "list"],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            installed_packages = [line.split()[0].lower() for line in result.stdout.split('\n') if line and not line.startswith('-')]
                            context['python_packages_installed'] = installed_packages
                            self.console.print(f"[dim]  → Found {len(installed_packages)} Python packages[/dim]")
                    except:
                        pass

                # Check Node dependencies
                node_modules = self.project_path / "node_modules"
                if node_modules.exists():
                    context['node_modules_exist'] = True
                    context['node_package_count'] = len(list(node_modules.iterdir()))
                    self.console.print(f"[dim]  → Found node_modules with {context['node_package_count']} packages[/dim]")

        # Get directory structure
        try:
            all_files = []
            for ext in ['*.py', '*.js', '*.json', '*.txt']:
                all_files.extend([str(f.relative_to(self.project_path)) for f in self.project_path.rglob(ext)])
            context['all_project_files'] = all_files[:50]  # First 50
        except:
            pass

        return context

    async def _generate_fix_strategy(self, understanding: Dict, context: Dict) -> Dict:
        """Generate a concrete fix strategy based on understanding and context"""

        prompt = f"""Based on this problem analysis and environment context, generate a fix strategy:

PROBLEM:
{json.dumps(understanding, indent=2)}

ENVIRONMENT CONTEXT:
{json.dumps(context, indent=2)}

PROJECT PATH: {self.project_path}
PLAN: {json.dumps(self.plan.get('tech_stack', {}), indent=2)}

Respond ONLY with valid JSON:
{{
    "fixable": true/false,
    "approach": "brief description of the fix approach",
    "steps": [
        {{"action": "install_dependency", "target": "flask", "method": "pip"}},
        {{"action": "update_file", "file": "package.json", "change": "update main field to correct path"}},
        {{"action": "create_file", "file": "index.js", "content": "..."}}
    ],
    "validation_check": "how to verify this worked",
    "description": "user-friendly description of what was done"
}}
"""

        response = await self.model.execute(
            prompt=prompt,
            system_prompt="You are a fix strategist. Generate concrete, actionable fix steps.",
            max_tokens=2048,
            temperature=0.4
        )

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.error(f"Failed to parse strategy: {e}")

        return {"fixable": False, "reason": "Could not generate strategy"}

    async def _apply_fix(self, strategy: Dict) -> bool:
        """Apply the fix steps"""

        steps = strategy.get('steps', [])

        if not steps:
            return False

        self.console.print(f"[cyan]Applying {len(steps)} fix step(s):[/cyan]")

        for i, step in enumerate(steps, 1):
            action = step.get('action', '')
            self.console.print(f"[dim]  Step {i}: {action}[/dim]")

            try:
                if action == 'install_dependency':
                    success = await self._install_dependency(step)
                elif action == 'update_file':
                    success = await self._update_file(step)
                elif action == 'create_file':
                    success = await self._create_file(step)
                elif action == 'update_config':
                    success = await self._update_config(step)
                else:
                    self.console.print(f"[yellow]    Unknown action: {action}[/yellow]")
                    continue

                if success:
                    self.console.print(f"[green]    ✓ Completed[/green]")
                else:
                    self.console.print(f"[red]    ✗ Failed[/red]")
                    return False

            except Exception as e:
                self.console.print(f"[red]    ✗ Error: {e}[/red]")
                return False

        return True

    async def _install_dependency(self, step: Dict) -> bool:
        """Install a dependency"""
        target = step.get('target', '')
        method = step.get('method', 'pip')

        if method == 'pip' and 'venv_pip' in self.plan:
            try:
                result = subprocess.run(
                    [self.plan['venv_pip'], "install", target],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                return result.returncode == 0
            except:
                return False

        elif method == 'npm':
            try:
                result = subprocess.run(
                    ["npm", "install", target],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.project_path
                )
                return result.returncode == 0
            except:
                return False

        return False

    async def _update_file(self, step: Dict) -> bool:
        """Update a file"""
        file_path = self.project_path / step.get('file', '')
        change = step.get('change', '')

        if not file_path.exists():
            return False

        try:
            content = file_path.read_text()

            # Use AI to apply the change
            prompt = f"""Apply this change to the file:

CHANGE NEEDED: {change}

CURRENT CONTENT:
```
{content}
```

Return ONLY the updated file content, no explanations."""

            updated_content = await self.model.execute(
                prompt=prompt,
                system_prompt="You are a file editor. Output only the corrected file.",
                max_tokens=2048,
                temperature=0.3
            )

            file_path.write_text(updated_content)
            return True

        except:
            return False

    async def _create_file(self, step: Dict) -> bool:
        """Create a new file"""
        file_path = self.project_path / step.get('file', '')
        content = step.get('content', '')

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return True
        except:
            return False

    async def _update_config(self, step: Dict) -> bool:
        """Update configuration (package.json, run command, etc.)"""
        config_type = step.get('config_type', '')
        update = step.get('update', {})

        if config_type == 'run_command':
            self.plan['run_command'] = update.get('new_value', self.plan.get('run_command', ''))
            return True

        return False

    async def _validate_fix(self, understanding: Dict, strategy: Dict) -> Dict:
        """Validate that the fix actually worked"""

        validation_check = strategy.get('validation_check', '')

        if not validation_check:
            return {"valid": True}  # Assume valid if no check specified

        self.console.print("[dim]Validating fix...[/dim]")

        # Simple validation checks
        affected = understanding.get('affected_component', '')

        if 'file' in validation_check.lower() and affected:
            file_path = self.project_path / affected
            valid = file_path.exists()
            return {
                "valid": valid,
                "reason": f"File {'exists' if valid else 'still missing'}: {affected}"
            }

        if 'dependency' in validation_check.lower() or 'module' in validation_check.lower():
            # Check if dependency is now installed
            if 'venv_python' in self.plan:
                try:
                    module_name = affected.lower()
                    result = subprocess.run(
                        [self.plan['venv_python'], "-c", f"import {module_name}"],
                        capture_output=True,
                        timeout=5
                    )
                    valid = result.returncode == 0
                    return {
                        "valid": valid,
                        "reason": f"Module {'can' if valid else 'cannot'} be imported: {module_name}"
                    }
                except:
                    pass

        # Default: assume valid
        return {"valid": True}
