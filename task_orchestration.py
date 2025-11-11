"""
Task Orchestration System
Manages task queue, dependencies, scheduling, and execution
"""

from typing import List, Dict, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import heapq
import json
import uuid
import time


class TaskType(Enum):
    """Types of development tasks"""
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    API = "api"
    TESTING = "testing"
    DEVOPS = "devops"
    DOCUMENTATION = "documentation"
    INTEGRATION = "integration"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    READY = "ready"  # Dependencies resolved
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    RETRYING = "retrying"


class ValidationRule(Enum):
    """Types of validation for task outputs"""
    SYNTAX_CHECK = "syntax_check"
    TYPE_CHECK = "type_check"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    LINT = "lint"
    SECURITY_SCAN = "security_scan"
    CUSTOM = "custom"


@dataclass
class TaskOutput:
    """Structured output from task execution"""
    success: bool
    content: str
    artifacts: Dict[str, str] = field(default_factory=dict)  # filename -> content
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)  # execution_time, tokens_used, etc.


@dataclass
class Task:
    """
    Represents a single atomic task in the system
    """
    id: str
    type: TaskType
    description: str
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, higher = more urgent
    estimated_time: int = 300  # seconds
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    prompt: str = ""
    tools: List[str] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    output: Optional[TaskOutput] = None
    validation_rules: List[ValidationRule] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    platform: str = "shared"
    language: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority > other.priority  # Higher priority first

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        data['status'] = self.status.value
        data['validation_rules'] = [v.value for v in self.validation_rules]
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create from dictionary"""
        data['type'] = TaskType(data['type'])
        data['status'] = TaskStatus(data['status'])
        data['validation_rules'] = [ValidationRule(v) for v in data.get('validation_rules', [])]
        if data.get('output'):
            data['output'] = TaskOutput(**data['output'])
        return cls(**data)


class DependencyGraph:
    """
    Manages task dependencies and determines execution order
    """

    def __init__(self):
        self.graph: Dict[str, Set[str]] = {}  # task_id -> set of dependency task_ids
        self.reverse_graph: Dict[str, Set[str]] = {}  # task_id -> set of dependent task_ids

    def add_task(self, task_id: str, dependencies: List[str]):
        """Add a task with its dependencies"""
        self.graph[task_id] = set(dependencies)

        # Update reverse graph
        if task_id not in self.reverse_graph:
            self.reverse_graph[task_id] = set()

        for dep in dependencies:
            if dep not in self.reverse_graph:
                self.reverse_graph[dep] = set()
            self.reverse_graph[dep].add(task_id)

    def get_dependencies(self, task_id: str) -> Set[str]:
        """Get all dependencies for a task"""
        return self.graph.get(task_id, set())

    def get_dependents(self, task_id: str) -> Set[str]:
        """Get all tasks that depend on this task"""
        return self.reverse_graph.get(task_id, set())

    def is_ready(self, task_id: str, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are completed"""
        dependencies = self.get_dependencies(task_id)
        return dependencies.issubset(completed_tasks)

    def get_ready_tasks(self, pending_tasks: Set[str], completed_tasks: Set[str]) -> Set[str]:
        """Get all tasks that are ready to execute"""
        ready = set()
        for task_id in pending_tasks:
            if self.is_ready(task_id, completed_tasks):
                ready.add(task_id)
        return ready

    def detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies"""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.reverse_graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            rec_stack.remove(node)

        for node in self.graph.keys():
            if node not in visited:
                dfs(node, [])

        return cycles

    def topological_sort(self) -> List[str]:
        """Return tasks in topological order"""
        in_degree = {task: len(deps) for task, deps in self.graph.items()}
        queue = [task for task, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            task = queue.pop(0)
            result.append(task)

            for dependent in self.reverse_graph.get(task, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result


class TaskQueue:
    """
    Priority queue with dependency resolution and retry logic
    """

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.heap: List[Task] = []
        self.dependency_graph = DependencyGraph()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.in_progress_tasks: Set[str] = set()

    def add_task(self, task: Task) -> None:
        """Add a task to the queue"""
        self.tasks[task.id] = task
        self.dependency_graph.add_task(task.id, task.dependencies)

        # Add to heap if ready
        if self.dependency_graph.is_ready(task.id, self.completed_tasks):
            task.status = TaskStatus.READY
            heapq.heappush(self.heap, task)

    def add_tasks(self, tasks: List[Task]) -> None:
        """Add multiple tasks"""
        for task in tasks:
            self.add_task(task)

    def get_next_task(self) -> Optional[Task]:
        """Get highest priority ready task"""
        while self.heap:
            task = heapq.heappop(self.heap)

            # Double-check task is still ready and not in progress
            if (task.id in self.tasks and
                task.status == TaskStatus.READY and
                task.id not in self.in_progress_tasks):

                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.utcnow().isoformat()
                self.in_progress_tasks.add(task.id)
                return task

        return None

    def get_ready_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """Get multiple ready tasks for parallel execution"""
        ready_task_ids = self.dependency_graph.get_ready_tasks(
            set(self.tasks.keys()) - self.completed_tasks - self.in_progress_tasks,
            self.completed_tasks
        )

        ready_tasks = []
        for task_id in ready_task_ids:
            task = self.tasks[task_id]
            if task.status != TaskStatus.IN_PROGRESS:
                ready_tasks.append(task)

        # Sort by priority
        ready_tasks.sort(reverse=True)

        if limit:
            return ready_tasks[:limit]
        return ready_tasks

    def mark_completed(self, task_id: str, output: TaskOutput) -> None:
        """Mark task as completed and update dependents"""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.output = output
        task.completed_at = datetime.utcnow().isoformat()

        self.completed_tasks.add(task_id)
        self.in_progress_tasks.discard(task_id)

        # Check and update dependent tasks
        dependents = self.dependency_graph.get_dependents(task_id)
        for dep_id in dependents:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if self.dependency_graph.is_ready(dep_id, self.completed_tasks):
                    dep_task.status = TaskStatus.READY
                    heapq.heappush(self.heap, dep_task)

    def mark_failed(self, task_id: str, error: str) -> bool:
        """
        Mark task as failed and determine if retry should happen

        Returns:
            True if task will be retried, False otherwise
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.retry_count += 1

        if task.retry_count < task.max_retries:
            # Retry with exponential backoff
            task.status = TaskStatus.RETRYING
            self.in_progress_tasks.discard(task_id)

            # Add back to queue with slight priority reduction
            task.priority = max(1, task.priority - 1)
            heapq.heappush(self.heap, task)

            return True
        else:
            # Max retries exceeded
            task.status = TaskStatus.FAILED
            task.output = TaskOutput(
                success=False,
                content="",
                errors=[error]
            )
            self.failed_tasks.add(task_id)
            self.in_progress_tasks.discard(task_id)

            # Block dependent tasks
            dependents = self.dependency_graph.get_dependents(task_id)
            for dep_id in dependents:
                if dep_id in self.tasks:
                    self.tasks[dep_id].status = TaskStatus.BLOCKED

            return False

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with given status"""
        return [t for t in self.tasks.values() if t.status == status]

    def get_statistics(self) -> Dict[str, int]:
        """Get queue statistics"""
        return {
            "total": len(self.tasks),
            "pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            "ready": len([t for t in self.tasks.values() if t.status == TaskStatus.READY]),
            "in_progress": len(self.in_progress_tasks),
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
            "blocked": len([t for t in self.tasks.values() if t.status == TaskStatus.BLOCKED])
        }

    def is_complete(self) -> bool:
        """Check if all tasks are completed or failed"""
        return (len(self.completed_tasks) + len(self.failed_tasks)) == len(self.tasks)

    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate task dependencies and return issues"""
        issues = {}

        # Check for circular dependencies
        cycles = self.dependency_graph.detect_cycles()
        if cycles:
            issues["circular_dependencies"] = [" -> ".join(cycle) for cycle in cycles]

        # Check for missing dependencies
        missing = []
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    missing.append(f"{task_id} depends on missing task {dep_id}")

        if missing:
            issues["missing_dependencies"] = missing

        return issues


class TaskScheduler:
    """
    Manages task execution scheduling and worker assignment
    """

    def __init__(
        self,
        queue: TaskQueue,
        max_parallel_tasks: int = 5,
        worker_assignment_fn: Optional[Callable] = None
    ):
        self.queue = queue
        self.max_parallel_tasks = max_parallel_tasks
        self.worker_assignment_fn = worker_assignment_fn or self._default_worker_assignment

    def _default_worker_assignment(self, task: Task) -> str:
        """Default worker assignment based on task type"""
        worker_map = {
            TaskType.BACKEND: "backend_specialist",
            TaskType.FRONTEND: "frontend_specialist",
            TaskType.DATABASE: "backend_specialist",
            TaskType.API: "backend_specialist",
            TaskType.TESTING: "testing_specialist",
            TaskType.DEVOPS: "devops_specialist",
            TaskType.DOCUMENTATION: "documentation_specialist",
            TaskType.INTEGRATION: "integration_specialist",
            TaskType.BUGFIX: "general_specialist",
            TaskType.REFACTOR: "general_specialist"
        }
        return worker_map.get(task.type, "general_specialist")

    def schedule_next_batch(self) -> List[Task]:
        """
        Schedule next batch of tasks for execution

        Returns:
            List of tasks to execute in parallel
        """
        available_slots = self.max_parallel_tasks - len(self.queue.in_progress_tasks)

        if available_slots <= 0:
            return []

        ready_tasks = self.queue.get_ready_tasks(limit=available_slots)

        # Assign workers
        for task in ready_tasks:
            if not task.assigned_worker:
                task.assigned_worker = self.worker_assignment_fn(task)

        return ready_tasks

    def can_schedule_more(self) -> bool:
        """Check if more tasks can be scheduled"""
        return len(self.queue.in_progress_tasks) < self.max_parallel_tasks


# Task Creation Helpers

def create_task(
    task_type: TaskType,
    description: str,
    dependencies: List[str] = None,
    priority: int = 5,
    platform: str = "shared",
    validation_rules: List[ValidationRule] = None,
    **kwargs
) -> Task:
    """Helper to create a task"""
    return Task(
        id=str(uuid.uuid4()),
        type=task_type,
        description=description,
        dependencies=dependencies or [],
        priority=priority,
        platform=platform,
        validation_rules=validation_rules or [],
        **kwargs
    )


def create_backend_task(
    description: str,
    file_path: str,
    language: str = "python",
    **kwargs
) -> Task:
    """Create a backend development task"""
    return create_task(
        task_type=TaskType.BACKEND,
        description=description,
        file_path=file_path,
        language=language,
        validation_rules=[
            ValidationRule.SYNTAX_CHECK,
            ValidationRule.UNIT_TEST
        ],
        tools=["filesystem", "sequential_thinking"],
        **kwargs
    )


def create_frontend_task(
    description: str,
    file_path: str,
    language: str = "typescript",
    **kwargs
) -> Task:
    """Create a frontend development task"""
    return create_task(
        task_type=TaskType.FRONTEND,
        description=description,
        file_path=file_path,
        language=language,
        platform="web",
        validation_rules=[
            ValidationRule.SYNTAX_CHECK,
            ValidationRule.LINT
        ],
        tools=["filesystem", "sequential_thinking"],
        **kwargs
    )


def create_test_task(
    description: str,
    target_task_id: str,
    file_path: str,
    **kwargs
) -> Task:
    """Create a testing task"""
    return create_task(
        task_type=TaskType.TESTING,
        description=description,
        dependencies=[target_task_id],
        file_path=file_path,
        validation_rules=[ValidationRule.CUSTOM],
        tools=["filesystem", "sequential_thinking"],
        **kwargs
    )


def create_devops_task(
    description: str,
    **kwargs
) -> Task:
    """Create a DevOps task"""
    return create_task(
        task_type=TaskType.DEVOPS,
        description=description,
        validation_rules=[ValidationRule.CUSTOM],
        tools=["git", "filesystem", "sequential_thinking"],
        **kwargs
    )


# Example: Decompose a project into tasks

def decompose_simple_api_project() -> List[Task]:
    """
    Example decomposition of a simple API project
    """
    tasks = []

    # Database schema
    db_task = create_backend_task(
        description="Create database schema for user management",
        file_path="backend/models/user.py",
        priority=10
    )
    tasks.append(db_task)

    # API endpoints
    api_task = create_backend_task(
        description="Implement user CRUD API endpoints",
        file_path="backend/api/users.py",
        dependencies=[db_task.id],
        priority=9
    )
    tasks.append(api_task)

    # Authentication
    auth_task = create_backend_task(
        description="Implement JWT authentication middleware",
        file_path="backend/auth/jwt.py",
        dependencies=[db_task.id],
        priority=9
    )
    tasks.append(auth_task)

    # Tests
    test_task = create_test_task(
        description="Create unit tests for user API",
        target_task_id=api_task.id,
        file_path="tests/test_user_api.py",
        priority=7
    )
    tasks.append(test_task)

    # DevOps
    docker_task = create_devops_task(
        description="Create Dockerfile and docker-compose.yml",
        dependencies=[api_task.id, auth_task.id],
        priority=6
    )
    tasks.append(docker_task)

    return tasks
