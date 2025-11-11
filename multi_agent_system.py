"""
True Multi-Agent System
Master orchestrator coordinating specialized worker agents
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent specialization roles"""
    MASTER = "master_orchestrator"
    BACKEND = "backend_specialist"
    FRONTEND = "frontend_specialist"
    DEVOPS = "devops_specialist"
    TESTING = "testing_specialist"
    DEBUGGING = "debugging_specialist"


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(Enum):
    """Task types for agent specialization"""
    BACKEND_SETUP = "backend_setup"
    BACKEND_CODE = "backend_code"
    FRONTEND_SETUP = "frontend_setup"
    FRONTEND_CODE = "frontend_code"
    DEPENDENCY_INSTALL = "dependency_install"
    TESTING = "testing"
    DEBUGGING = "debugging"
    DEVOPS_SETUP = "devops_setup"
    DEVOPS_DEPLOY = "devops_deploy"
    CODE_GENERATION = "code_generation"
    FILE_OPERATION = "file_operation"


@dataclass
class Task:
    """Task for an agent"""
    id: str
    description: str
    assigned_to: Optional[AgentRole]
    status: TaskStatus
    dependencies: List[str]
    context: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    reasoning: List[str] = None

    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = []


@dataclass
class AgentMemory:
    """Agent's working memory"""
    agent_role: AgentRole
    current_task: Optional[Task]
    completed_tasks: List[str]
    knowledge: Dict[str, Any]
    observations: List[str]
    reasoning_history: List[Dict[str, str]]
    blocked_on: Optional[str] = None

    def wipe(self):
        """Wipe memory (keep role and knowledge)"""
        self.current_task = None
        self.completed_tasks = []
        self.observations = []
        self.reasoning_history = []
        self.blocked_on = None
        logger.info(f"Wiped memory for {self.agent_role.value}")

    def checkpoint(self) -> Dict:
        """Save memory state"""
        return {
            "agent_role": self.agent_role.value,
            "completed_tasks": self.completed_tasks.copy(),
            "knowledge": self.knowledge.copy(),
            "observations": self.observations.copy(),
            "reasoning_history": self.reasoning_history.copy()
        }

    def restore(self, checkpoint: Dict):
        """Restore from checkpoint"""
        self.completed_tasks = checkpoint.get("completed_tasks", [])
        self.knowledge.update(checkpoint.get("knowledge", {}))
        self.observations = checkpoint.get("observations", [])
        self.reasoning_history = checkpoint.get("reasoning_history", [])


class Agent:
    """Base agent with reasoning capabilities"""

    def __init__(self, role: AgentRole, model, project_path: Path, console):
        self.role = role
        self.model = model
        self.project_path = project_path
        self.console = console
        self.memory = AgentMemory(
            agent_role=role,
            current_task=None,
            completed_tasks=[],
            knowledge={},
            observations=[],
            reasoning_history=[]
        )

    async def reason_and_act(self, task: Task) -> Dict[str, Any]:
        """
        Reasoning loop: Observe → Reason → Plan → Act → Validate
        """
        self.memory.current_task = task
        self.console.print(f"\n[bold cyan]{self.role.value.upper()}[/bold cyan] starting task: {task.description}")

        result = {
            "success": False,
            "output": None,
            "reasoning": [],
            "actions_taken": [],
            "validation": {}
        }

        try:
            # Step 1: Observe
            observations = await self._observe(task)
            result["reasoning"].append(f"Observed: {observations}")
            self.console.print(f"[dim]  📋 Observations: {len(observations.get('findings', []))} findings[/dim]")

            # Step 2: Reason
            reasoning = await self._reason(task, observations)
            result["reasoning"].append(f"Reasoning: {reasoning.get('conclusion', '')}")
            self.console.print(f"[cyan]  🧠 Reasoning: {reasoning.get('conclusion', '')[:80]}...[/dim]")

            # Step 3: Plan
            plan = await self._plan(task, observations, reasoning)
            result["reasoning"].append(f"Plan: {len(plan.get('steps', []))} steps")
            self.console.print(f"[dim]  📝 Plan: {len(plan.get('steps', []))} action steps[/dim]")

            # Step 4: Act
            for i, step in enumerate(plan.get('steps', []), 1):
                self.console.print(f"[dim]    Step {i}/{len(plan['steps'])}: {step.get('action', '')}[/dim]")
                action_result = await self._act(step, task, observations)

                result["actions_taken"].append({
                    "step": i,
                    "action": step.get('action', ''),
                    "success": action_result.get('success', False),
                    "output": action_result.get('output', '')
                })

                if not action_result.get('success', False):
                    result["success"] = False
                    result["error"] = action_result.get('error', 'Action failed')
                    self.console.print(f"[red]    ✗ Step {i} failed: {result['error']}[/red]")
                    return result
                else:
                    self.console.print(f"[green]    ✓ Step {i} completed[/green]")

            # Step 5: Validate
            validation = await self._validate(task, plan, result)
            result["validation"] = validation

            if validation.get('valid', False):
                result["success"] = True
                result["output"] = validation.get('output', '')
                self.console.print(f"[bold green]  ✓ Task completed successfully[/bold green]")
            else:
                result["success"] = False
                result["error"] = validation.get('reason', 'Validation failed')
                self.console.print(f"[yellow]  ⚠ Validation failed: {result['error']}[/yellow]")

            # Store in memory
            self.memory.completed_tasks.append(task.id)
            self.memory.reasoning_history.append({
                "task_id": task.id,
                "timestamp": datetime.now().isoformat(),
                "reasoning": result["reasoning"],
                "success": result["success"]
            })

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            self.console.print(f"[red]  ✗ Agent error: {e}[/red]")
            logger.error(f"Agent {self.role.value} error: {e}")

        return result

    async def _observe(self, task: Task) -> Dict[str, Any]:
        """Observe environment and gather information"""
        raise NotImplementedError("Subclasses must implement _observe")

    async def _reason(self, task: Task, observations: Dict) -> Dict[str, Any]:
        """Reason about the problem"""
        prompt = f"""You are a {self.role.value}. Analyze this situation and reason about what to do.

TASK: {task.description}

OBSERVATIONS:
{json.dumps(observations, indent=2)}

CONTEXT:
{json.dumps(task.context, indent=2)}

Reason step-by-step about:
1. What is the actual problem?
2. What are the root causes?
3. What approach should be taken?
4. What could go wrong?
5. How to verify success?

Respond ONLY with valid JSON:
{{
    "problem_analysis": "what's actually wrong",
    "root_causes": ["cause 1", "cause 2"],
    "approach": "strategy to solve this",
    "risks": ["risk 1", "risk 2"],
    "success_criteria": "how to verify it worked",
    "conclusion": "brief summary of reasoning"
}}
"""

        response = await self.model.execute(
            prompt=prompt,
            system_prompt=f"You are {self.role.value}. Reason deeply before acting.",
            max_tokens=2048,
            temperature=0.4
        )

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                reasoning = json.loads(json_match.group(0))
                self.memory.observations.append(reasoning.get('conclusion', ''))
                return reasoning
        except:
            pass

        return {"conclusion": "Could not reason about problem"}

    async def _plan(self, task: Task, observations: Dict, reasoning: Dict) -> Dict[str, Any]:
        """Create action plan"""
        prompt = f"""Based on your reasoning, create a concrete action plan.

TASK: {task.description}
REASONING: {reasoning.get('conclusion', '')}
APPROACH: {reasoning.get('approach', '')}

Create specific, executable steps. Each step should be atomic and verifiable.

Respond ONLY with valid JSON:
{{
    "steps": [
        {{
            "action": "check_file_exists",
            "target": "/path/to/file",
            "expected": "file should exist",
            "fallback": "create file if missing"
        }},
        {{
            "action": "install_dependency",
            "target": "flask",
            "method": "pip",
            "expected": "flask importable"
        }}
    ],
    "estimated_duration": "2 minutes",
    "confidence": 0.8
}}
"""

        response = await self.model.execute(
            prompt=prompt,
            system_prompt=f"You are {self.role.value}. Create actionable plans.",
            max_tokens=2048,
            temperature=0.3
        )

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass

        return {"steps": []}

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        """Execute a single action step"""
        raise NotImplementedError("Subclasses must implement _act")

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        """Validate that the task was completed successfully"""
        raise NotImplementedError("Subclasses must implement _validate")


class BackendAgent(Agent):
    """Specialized agent for backend tasks"""

    def __init__(self, model, project_path: Path, console, plan: Dict):
        super().__init__(AgentRole.BACKEND, model, project_path, console)
        self.plan = plan

    async def _observe(self, task: Task) -> Dict[str, Any]:
        """Observe backend environment"""
        import subprocess

        observations = {
            "findings": [],
            "venv_exists": False,
            "dependencies_installed": [],
            "python_files": [],
            "requirements_file": None
        }

        # Check venv
        venv_path = self.project_path / "venv"
        observations["venv_exists"] = venv_path.exists()
        observations["findings"].append(f"Virtual environment: {'exists' if venv_path.exists() else 'missing'}")

        # Check installed packages
        if 'venv_python' in self.plan:
            try:
                result = subprocess.run(
                    [self.plan['venv_python'], "-m", "pip", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    observations["dependencies_installed"] = [
                        line.split()[0].lower()
                        for line in result.stdout.split('\n')
                        if line and not line.startswith('-')
                    ]
                    observations["findings"].append(f"Installed packages: {len(observations['dependencies_installed'])}")
            except:
                pass

        # Check Python files
        observations["python_files"] = [
            str(f.relative_to(self.project_path))
            for f in self.project_path.rglob('*.py')
        ]
        observations["findings"].append(f"Python files: {len(observations['python_files'])}")

        # Check requirements.txt
        req_file = self.project_path / "requirements.txt"
        if req_file.exists():
            observations["requirements_file"] = req_file.read_text().split('\n')
            observations["findings"].append(f"requirements.txt: {len(observations['requirements_file'])} dependencies")

        return observations

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        """Execute backend action"""
        import subprocess

        action = step.get('action', '')
        target = step.get('target', '')

        if action == 'install_dependency':
            if 'venv_pip' not in self.plan:
                return {"success": False, "error": "No venv pip available"}

            try:
                result = subprocess.run(
                    [self.plan['venv_pip'], "install", target],
                    capture_output=True,
                    text=True,
                    timeout=180
                )

                if result.returncode == 0:
                    return {
                        "success": True,
                        "output": f"Installed {target}",
                        "details": result.stdout
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Installation failed: {result.stderr}",
                        "output": result.stderr
                    }
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == 'check_file_exists':
            file_path = self.project_path / target
            exists = file_path.exists()

            return {
                "success": exists,
                "output": f"File {'exists' if exists else 'missing'}: {target}",
                "exists": exists
            }

        elif action == 'run_command':
            try:
                result = subprocess.run(
                    step.get('command', ''),
                    shell=True,
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=step.get('timeout', 30)
                )

                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr if result.returncode != 0 else None
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": f"Unknown action: {action}"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        """Validate backend task completion"""
        import subprocess

        # Check if we can import required modules
        if 'install' in task.description.lower():
            module_name = task.context.get('module_name', '')
            if module_name and 'venv_python' in self.plan:
                try:
                    result_check = subprocess.run(
                        [self.plan['venv_python'], "-c", f"import {module_name}"],
                        capture_output=True,
                        timeout=5
                    )

                    if result_check.returncode == 0:
                        return {
                            "valid": True,
                            "output": f"Module {module_name} successfully installed and importable"
                        }
                    else:
                        return {
                            "valid": False,
                            "reason": f"Module {module_name} cannot be imported"
                        }
                except:
                    pass

        # Default: check if any actions failed
        failed_actions = [a for a in result.get('actions_taken', []) if not a.get('success', False)]

        if failed_actions:
            return {
                "valid": False,
                "reason": f"{len(failed_actions)} action(s) failed"
            }

        return {"valid": True, "output": "All actions completed successfully"}


class FrontendAgent(Agent):
    """Specialized agent for frontend tasks"""

    def __init__(self, model, project_path: Path, console, plan: Dict):
        super().__init__(AgentRole.FRONTEND, model, project_path, console)
        self.plan = plan

    async def _observe(self, task: Task) -> Dict[str, Any]:
        """Observe frontend environment"""
        import subprocess

        observations = {
            "findings": [],
            "node_modules_exists": False,
            "package_json_exists": False,
            "frontend_files": [],
            "entry_points": []
        }

        # Check node_modules
        node_modules = self.project_path / "node_modules"
        observations["node_modules_exists"] = node_modules.exists()
        observations["findings"].append(f"node_modules: {'exists' if node_modules.exists() else 'missing'}")

        # Check package.json
        package_json = self.project_path / "package.json"
        observations["package_json_exists"] = package_json.exists()
        if package_json.exists():
            try:
                import json
                pkg_data = json.loads(package_json.read_text())
                observations["findings"].append(f"package.json: {len(pkg_data.get('dependencies', {}))} dependencies")
                observations["entry_points"] = [pkg_data.get('main', ''), pkg_data.get('module', '')]
            except:
                observations["findings"].append("package.json: parse error")

        # Check frontend files
        for ext in ['*.js', '*.jsx', '*.ts', '*.tsx', '*.vue']:
            observations["frontend_files"].extend([
                str(f.relative_to(self.project_path))
                for f in self.project_path.rglob(ext)
            ])
        observations["findings"].append(f"Frontend files: {len(observations['frontend_files'])}")

        return observations

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        """Execute frontend action"""
        import subprocess

        action = step.get('action', '')
        target = step.get('target', '')

        if action == 'npm_install':
            try:
                result = subprocess.run(
                    ['npm', 'install'],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                return {
                    "success": result.returncode == 0,
                    "output": f"npm install {'succeeded' if result.returncode == 0 else 'failed'}",
                    "details": result.stdout if result.returncode == 0 else result.stderr
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == 'update_package_json':
            try:
                import json
                package_json = self.project_path / "package.json"
                data = json.loads(package_json.read_text())

                # Apply updates from step
                updates = step.get('updates', {})
                for key, value in updates.items():
                    data[key] = value

                package_json.write_text(json.dumps(data, indent=2))

                return {
                    "success": True,
                    "output": f"Updated package.json: {list(updates.keys())}"
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == 'npm_run':
            script = step.get('script', 'build')
            try:
                result = subprocess.run(
                    ['npm', 'run', script],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=step.get('timeout', 120)
                )

                return {
                    "success": result.returncode == 0,
                    "output": f"npm run {script} {'succeeded' if result.returncode == 0 else 'failed'}",
                    "details": result.stdout if result.returncode == 0 else result.stderr
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": f"Unknown action: {action}"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        """Validate frontend task completion"""
        # Check if node_modules exists after install
        if 'install' in task.description.lower():
            node_modules = self.project_path / "node_modules"
            if node_modules.exists():
                return {"valid": True, "output": "node_modules directory created successfully"}
            else:
                return {"valid": False, "reason": "node_modules directory not found"}

        # Default validation
        failed_actions = [a for a in result.get('actions_taken', []) if not a.get('success', False)]
        if failed_actions:
            return {"valid": False, "reason": f"{len(failed_actions)} action(s) failed"}

        return {"valid": True, "output": "All actions completed successfully"}


class DevOpsAgent(Agent):
    """Specialized agent for DevOps tasks"""

    def __init__(self, model, project_path: Path, console, plan: Dict):
        super().__init__(AgentRole.DEVOPS, model, project_path, console)
        self.plan = plan

    async def _observe(self, task: Task) -> Dict[str, Any]:
        """Observe DevOps environment"""
        import subprocess

        observations = {
            "findings": [],
            "dockerfile_exists": False,
            "docker_available": False,
            "docker_compose_exists": False
        }

        # Check Dockerfile
        dockerfile = self.project_path / "Dockerfile"
        observations["dockerfile_exists"] = dockerfile.exists()
        observations["findings"].append(f"Dockerfile: {'exists' if dockerfile.exists() else 'missing'}")

        # Check Docker availability
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, timeout=5)
            observations["docker_available"] = result.returncode == 0
            observations["findings"].append("Docker: available")
        except:
            observations["findings"].append("Docker: not available")

        # Check docker-compose
        compose_file = self.project_path / "docker-compose.yml"
        observations["docker_compose_exists"] = compose_file.exists()
        observations["findings"].append(f"docker-compose.yml: {'exists' if compose_file.exists() else 'missing'}")

        return observations

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        """Execute DevOps action"""
        import subprocess

        action = step.get('action', '')

        if action == 'create_dockerfile':
            content = step.get('content', '')
            try:
                dockerfile = self.project_path / "Dockerfile"
                dockerfile.write_text(content)
                return {"success": True, "output": "Dockerfile created"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == 'build_image':
            tag = step.get('tag', 'app:latest')
            try:
                result = subprocess.run(
                    ['docker', 'build', '-t', tag, '.'],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                return {
                    "success": result.returncode == 0,
                    "output": f"Docker image {tag} {'built' if result.returncode == 0 else 'build failed'}",
                    "details": result.stdout if result.returncode == 0 else result.stderr
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == 'setup_env':
            env_vars = step.get('env_vars', {})
            try:
                env_file = self.project_path / ".env"
                content = '\n'.join([f"{k}={v}" for k, v in env_vars.items()])
                env_file.write_text(content)
                return {"success": True, "output": f"Created .env with {len(env_vars)} variables"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": f"Unknown action: {action}"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        """Validate DevOps task completion"""
        failed_actions = [a for a in result.get('actions_taken', []) if not a.get('success', False)]
        if failed_actions:
            return {"valid": False, "reason": f"{len(failed_actions)} action(s) failed"}
        return {"valid": True, "output": "All DevOps actions completed"}


class TestingAgent(Agent):
    """Specialized agent for testing and validation"""

    def __init__(self, model, project_path: Path, console, plan: Dict):
        super().__init__(AgentRole.TESTING, model, project_path, console)
        self.plan = plan

    async def _observe(self, task: Task) -> Dict[str, Any]:
        """Observe testing environment"""
        import subprocess

        observations = {
            "findings": [],
            "test_files": [],
            "test_frameworks": []
        }

        # Find test files
        for pattern in ['test_*.py', '*_test.py', '*.test.js', '*.spec.js']:
            observations["test_files"].extend([
                str(f.relative_to(self.project_path))
                for f in self.project_path.rglob(pattern)
            ])
        observations["findings"].append(f"Test files: {len(observations['test_files'])}")

        # Check for test frameworks
        if (self.project_path / "pytest.ini").exists() or any('pytest' in f for f in observations["test_files"]):
            observations["test_frameworks"].append("pytest")
        if (self.project_path / "package.json").exists():
            observations["test_frameworks"].append("jest/mocha")

        observations["findings"].append(f"Test frameworks: {', '.join(observations['test_frameworks']) or 'none detected'}")

        return observations

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        """Execute testing action"""
        import subprocess

        action = step.get('action', '')

        if action == 'run_tests':
            framework = step.get('framework', 'pytest')
            try:
                if framework == 'pytest' and 'venv_python' in self.plan:
                    cmd = [self.plan['venv_python'], '-m', 'pytest', '-v']
                elif framework == 'npm':
                    cmd = ['npm', 'test']
                else:
                    return {"success": False, "error": f"Unknown test framework: {framework}"}

                result = subprocess.run(
                    cmd,
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=step.get('timeout', 300)
                )

                return {
                    "success": result.returncode == 0,
                    "output": f"Tests {'passed' if result.returncode == 0 else 'failed'}",
                    "details": result.stdout,
                    "errors": result.stderr if result.returncode != 0 else None
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == 'validate_syntax':
            target = step.get('target', '')
            file_path = self.project_path / target

            if not file_path.exists():
                return {"success": False, "error": f"File not found: {target}"}

            try:
                if file_path.suffix == '.py':
                    import ast
                    ast.parse(file_path.read_text())
                    return {"success": True, "output": f"Python syntax valid: {target}"}
                elif file_path.suffix in ['.js', '.jsx']:
                    # Just check if node can parse it
                    result = subprocess.run(
                        ['node', '--check', str(file_path)],
                        capture_output=True,
                        timeout=10
                    )
                    return {
                        "success": result.returncode == 0,
                        "output": f"JavaScript syntax {'valid' if result.returncode == 0 else 'invalid'}: {target}"
                    }
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == 'verify_import':
            module = step.get('module', '')
            if 'venv_python' in self.plan:
                try:
                    result = subprocess.run(
                        [self.plan['venv_python'], '-c', f'import {module}'],
                        capture_output=True,
                        timeout=10
                    )
                    return {
                        "success": result.returncode == 0,
                        "output": f"Module {module} {'importable' if result.returncode == 0 else 'not found'}"
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}

        return {"success": False, "error": f"Unknown action: {action}"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        """Validate testing task completion"""
        # For test runs, success means tests passed
        if 'run_tests' in task.description.lower():
            actions = result.get('actions_taken', [])
            test_actions = [a for a in actions if a.get('action') == 'run_tests']

            if test_actions:
                all_passed = all(a.get('success', False) for a in test_actions)
                if all_passed:
                    return {"valid": True, "output": "All tests passed"}
                else:
                    return {"valid": False, "reason": "Some tests failed"}

        # Default validation
        failed_actions = [a for a in result.get('actions_taken', []) if not a.get('success', False)]
        if failed_actions:
            return {"valid": False, "reason": f"{len(failed_actions)} action(s) failed"}

        return {"valid": True, "output": "All testing actions completed"}


class DebuggingAgent(Agent):
    """Specialized agent for debugging and error analysis"""

    def __init__(self, model, project_path: Path, console, plan: Dict):
        super().__init__(AgentRole.DEBUGGING, model, project_path, console)
        self.plan = plan

    async def _observe(self, task: Task) -> Dict[str, Any]:
        """Observe error conditions"""
        observations = {
            "findings": [],
            "error_output": task.context.get('error_output', ''),
            "error_type": "unknown",
            "affected_files": []
        }

        error_output = observations["error_output"]

        # Classify error
        if "ModuleNotFoundError" in error_output or "ImportError" in error_output:
            observations["error_type"] = "missing_dependency"
        elif "SyntaxError" in error_output:
            observations["error_type"] = "syntax_error"
        elif "not found" in error_output.lower() and "command" in error_output.lower():
            observations["error_type"] = "command_not_found"
        elif "ENOENT" in error_output or "No such file" in error_output:
            observations["error_type"] = "file_not_found"

        observations["findings"].append(f"Error type: {observations['error_type']}")

        # Extract affected files from error output
        import re
        file_matches = re.findall(r'File "([^"]+)"', error_output)
        observations["affected_files"] = file_matches
        if file_matches:
            observations["findings"].append(f"Affected files: {len(file_matches)}")

        return observations

    async def _act(self, step: Dict, task: Task, observations: Dict) -> Dict[str, Any]:
        """Execute debugging action"""
        action = step.get('action', '')

        if action == 'analyze_error':
            error_type = observations.get('error_type', 'unknown')
            error_output = observations.get('error_output', '')

            analysis = {
                "error_type": error_type,
                "likely_causes": [],
                "suggested_fixes": []
            }

            if error_type == "missing_dependency":
                analysis["likely_causes"] = ["Package not installed", "Wrong venv active"]
                analysis["suggested_fixes"] = ["Install missing package via pip/npm", "Verify venv activation"]
            elif error_type == "command_not_found":
                analysis["likely_causes"] = ["Command not in PATH", "Tool not installed", "Need to use venv"]
                analysis["suggested_fixes"] = ["Use full path to venv binary", "Install missing tool", "Activate venv"]

            return {
                "success": True,
                "output": f"Error analyzed: {error_type}",
                "analysis": analysis
            }

        return {"success": False, "error": f"Unknown action: {action}"}

    async def _validate(self, task: Task, plan: Dict, result: Dict) -> Dict[str, Any]:
        """Validate debugging task completion"""
        # Debugging tasks are informational, always succeed if analysis was performed
        actions = result.get('actions_taken', [])
        if any(a.get('action') == 'analyze_error' for a in actions):
            return {"valid": True, "output": "Error analysis completed"}

        return {"valid": True, "output": "Debugging actions completed"}


class MasterOrchestrator:
    """Master orchestrator coordinating all agents"""

    def __init__(self, model, project_path: Path, plan: Dict, console):
        self.model = model
        self.project_path = project_path
        self.plan = plan
        self.console = console

        # Initialize agents
        self.agents = {
            AgentRole.BACKEND: BackendAgent(model, project_path, console, plan),
            AgentRole.FRONTEND: FrontendAgent(model, project_path, console, plan),
            AgentRole.DEVOPS: DevOpsAgent(model, project_path, console, plan),
            AgentRole.TESTING: TestingAgent(model, project_path, console, plan),
            AgentRole.DEBUGGING: DebuggingAgent(model, project_path, console, plan),
        }

        # Task queue
        self.pending_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []

    def wipe_all_memory(self):
        """Wipe memory of all agents"""
        self.console.print("[bold red]🧹 Wiping all agent memories...[/bold red]")
        for agent in self.agents.values():
            agent.memory.wipe()
        self.pending_tasks = []
        self.console.print("[green]✓ All agent memories wiped[/green]")

    async def decompose_problem(self, problem_description: str, error_output: str) -> List[Task]:
        """Decompose problem into tasks for agents"""
        self.console.print("\n[bold magenta]🎯 MASTER: Decomposing problem into tasks...[/bold magenta]")

        prompt = f"""You are a master orchestrator. Decompose this problem into specific tasks for specialized agents.

PROBLEM: {problem_description}
ERROR OUTPUT: {error_output[:500]}

PROJECT PATH: {self.project_path}
TECH STACK: {json.dumps(self.plan.get('tech_stack', {}), indent=2)}

Available agents:
- BACKEND: Python/Flask/Django dependencies, venv, backend code, pip operations
- FRONTEND: npm, React/Vue/Angular, frontend dependencies, package.json management
- DEVOPS: Docker, deployment, environment setup, CI/CD, container operations
- TESTING: Run tests, validate syntax, verify imports, debugging test failures
- DEBUGGING: Error analysis, problem diagnosis, suggesting fixes

Decompose into atomic, parallelizable tasks.

Respond ONLY with valid JSON:
{{
    "tasks": [
        {{
            "id": "task_001",
            "description": "Install Flask in virtual environment",
            "assigned_to": "BACKEND",
            "dependencies": [],
            "context": {{"module_name": "flask"}}
        }},
        {{
            "id": "task_002",
            "description": "Verify backend can start",
            "assigned_to": "TESTING",
            "dependencies": ["task_001"],
            "context": {{}}
        }}
    ]
}}
"""

        response = await self.model.execute(
            prompt=prompt,
            system_prompt="You are a master orchestrator. Decompose problems into parallel tasks.",
            max_tokens=3072,
            temperature=0.4
        )

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                tasks = []

                for task_data in data.get('tasks', []):
                    role_str = task_data.get('assigned_to', '').upper()
                    try:
                        role = AgentRole[role_str]
                    except:
                        role = AgentRole.BACKEND  # Default

                    task = Task(
                        id=task_data['id'],
                        description=task_data['description'],
                        assigned_to=role,
                        status=TaskStatus.PENDING,
                        dependencies=task_data.get('dependencies', []),
                        context=task_data.get('context', {})
                    )
                    tasks.append(task)

                self.console.print(f"[cyan]  → Decomposed into {len(tasks)} tasks[/cyan]")
                for task in tasks:
                    self.console.print(f"[dim]    • {task.id}: {task.description} (Agent: {task.assigned_to.value})[/dim]")

                return tasks
        except Exception as e:
            logger.error(f"Failed to decompose problem: {e}")

        return []

    async def execute_tasks(self, tasks: List[Task]) -> bool:
        """Execute tasks with dependency resolution"""
        self.pending_tasks = tasks.copy()
        all_success = True

        while self.pending_tasks:
            # Find tasks that can run (no pending dependencies)
            ready_tasks = [
                t for t in self.pending_tasks
                if all(dep_id in [ct.id for ct in self.completed_tasks] for dep_id in t.dependencies)
            ]

            if not ready_tasks:
                self.console.print("[red]✗ Deadlock: No tasks ready to execute[/red]")
                break

            # Execute ready tasks (can be parallelized)
            for task in ready_tasks:
                agent = self.agents.get(task.assigned_to)
                if not agent:
                    self.console.print(f"[red]✗ No agent for {task.assigned_to}[/red]")
                    task.status = TaskStatus.FAILED
                    self.failed_tasks.append(task)
                    self.pending_tasks.remove(task)
                    all_success = False
                    continue

                # Execute task
                result = await agent.reason_and_act(task)

                if result.get('success', False):
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    self.completed_tasks.append(task)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result.get('error', 'Unknown error')
                    self.failed_tasks.append(task)
                    all_success = False

                self.pending_tasks.remove(task)

        return all_success
