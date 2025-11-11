"""
Knowledge Memory System
Persistent general knowledge that improves over time
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class KnowledgeMemorySystem:
    """
    Maintains general knowledge across sessions, learns from every build
    """

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.memory_path = config_dir / "knowledge_memory.json"
        self.memory: Dict = self._load()

    def _load(self) -> Dict:
        """Load persistent memory from disk"""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, 'r') as f:
                    memory = json.load(f)
                    logger.info(f"Loaded knowledge memory with {len(memory.get('learnings', []))} learnings")
                    return memory
            except Exception as e:
                logger.error(f"Failed to load knowledge memory: {e}")

        # Initialize default knowledge base
        return {
            "version": "1.0",
            "learnings": [],
            "rules": [
                {
                    "id": "rule_001",
                    "category": "project_structure",
                    "rule": "Python projects should always use virtual environments",
                    "rationale": "Prevents dependency conflicts and ensures isolation",
                    "confidence": 1.0,
                    "applications": 0
                },
                {
                    "id": "rule_002",
                    "category": "run_commands",
                    "rule": "Flask commands should use 'python -m flask' not 'flask' directly",
                    "rationale": "Direct flask command may not be in PATH, module invocation is more reliable",
                    "confidence": 1.0,
                    "applications": 0
                },
                {
                    "id": "rule_003",
                    "category": "dependencies",
                    "rule": "Always install dependencies before running tests",
                    "rationale": "Tests will fail if dependencies are missing",
                    "confidence": 1.0,
                    "applications": 0
                },
                {
                    "id": "rule_004",
                    "category": "file_structure",
                    "rule": "Frontend and backend should be in separate directories",
                    "rationale": "Improves organization and allows independent deployment",
                    "confidence": 0.8,
                    "applications": 0
                },
                {
                    "id": "rule_005",
                    "category": "run_commands",
                    "rule": "For multi-service projects, verify all service paths exist before running",
                    "rationale": "Prevents path-not-found errors at runtime",
                    "confidence": 0.9,
                    "applications": 0
                }
            ],
            "patterns": [
                {
                    "pattern_type": "tech_stack",
                    "condition": "Python + Flask + PostgreSQL",
                    "best_practices": [
                        "Use Flask-SQLAlchemy for ORM",
                        "Use Flask-Migrate for database migrations",
                        "Store database URL in environment variables",
                        "Create separate config files for dev/prod"
                    ],
                    "common_files": [
                        "requirements.txt",
                        "config.py",
                        "models.py",
                        "routes.py",
                        "__init__.py"
                    ]
                },
                {
                    "pattern_type": "tech_stack",
                    "condition": "React + Node.js backend",
                    "best_practices": [
                        "Use create-react-app or Vite for frontend",
                        "Separate frontend and backend into different directories",
                        "Use environment variables for API URLs",
                        "Add CORS configuration to backend"
                    ],
                    "common_files": [
                        "package.json (frontend)",
                        "package.json (backend)",
                        "src/App.js",
                        "server.js or index.js"
                    ]
                }
            ],
            "build_history": [],
            "conversation_context": [],
            "last_updated": datetime.now().isoformat()
        }

    def _save(self):
        """Save memory to disk"""
        try:
            self.memory["last_updated"] = datetime.now().isoformat()
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.memory_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
            logger.info("Knowledge memory saved")
        except Exception as e:
            logger.error(f"Failed to save knowledge memory: {e}")

    def add_learning(self, category: str, learning: str, context: Dict, confidence: float = 0.7):
        """
        Add a new learning to memory

        Args:
            category: Category of learning (project_structure, debugging, etc.)
            learning: The actual learning/insight
            context: Context in which this was learned
            confidence: Confidence level (0.0 - 1.0)
        """
        learning_entry = {
            "id": f"learning_{len(self.memory.get('learnings', []))}",
            "category": category,
            "learning": learning,
            "context": context,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "applications": 0
        }

        self.memory.setdefault("learnings", []).append(learning_entry)

        # Auto-generate rule if confidence is high enough
        if confidence >= 0.8:
            self._try_generate_rule(learning_entry)

        self._save()
        logger.info(f"Added learning: {learning[:50]}...")

    def _try_generate_rule(self, learning_entry: Dict):
        """Try to generate a rule from a learning"""
        # If learning has been applied successfully multiple times, elevate to rule
        if learning_entry.get("applications", 0) >= 3:
            rule = {
                "id": f"rule_{len(self.memory.get('rules', []))}",
                "category": learning_entry["category"],
                "rule": learning_entry["learning"],
                "rationale": learning_entry.get("context", {}).get("reason", "Derived from successful applications"),
                "confidence": learning_entry["confidence"],
                "applications": learning_entry["applications"]
            }

            self.memory.setdefault("rules", []).append(rule)
            logger.info(f"Generated new rule: {rule['rule'][:50]}...")

    def record_build(self, description: str, success: bool, tech_stack: Dict,
                    errors_encountered: List[str], fixes_applied: List[str],
                    files_created: int, duration_seconds: float):
        """
        Record a build in history for learning

        Args:
            description: Project description
            success: Whether build succeeded
            tech_stack: Tech stack used
            errors_encountered: List of errors
            fixes_applied: List of fixes
            files_created: Number of files
            duration_seconds: Build duration
        """
        build_entry = {
            "id": f"build_{len(self.memory.get('build_history', []))}",
            "description": description,
            "success": success,
            "tech_stack": tech_stack,
            "errors_encountered": errors_encountered,
            "fixes_applied": fixes_applied,
            "files_created": files_created,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat()
        }

        self.memory.setdefault("build_history", []).append(build_entry)

        # Extract learnings from this build
        self._extract_learnings_from_build(build_entry)

        # Keep only last 50 builds
        if len(self.memory["build_history"]) > 50:
            self.memory["build_history"] = self.memory["build_history"][-50:]

        self._save()

    def _extract_learnings_from_build(self, build_entry: Dict):
        """Automatically extract learnings from a build"""
        # Learn from successful builds
        if build_entry["success"]:
            tech_combo = " + ".join([v for v in build_entry["tech_stack"].values() if v])

            # Check if this tech combination is new
            existing_patterns = [p for p in self.memory.get("patterns", [])
                               if p.get("condition") == tech_combo]

            if not existing_patterns and tech_combo:
                self.add_learning(
                    category="tech_stack",
                    learning=f"Successfully built project with {tech_combo}",
                    context={
                        "tech_stack": build_entry["tech_stack"],
                        "files_created": build_entry["files_created"],
                        "reason": "Successful build completion"
                    },
                    confidence=0.7
                )

        # Learn from errors and fixes
        for i, error in enumerate(build_entry.get("errors_encountered", [])):
            if i < len(build_entry.get("fixes_applied", [])):
                fix = build_entry["fixes_applied"][i]
                self.add_learning(
                    category="debugging",
                    learning=f"When encountering '{error[:50]}...', apply: {fix}",
                    context={
                        "error": error,
                        "fix": fix,
                        "tech_stack": build_entry["tech_stack"]
                    },
                    confidence=0.8
                )

    def add_conversation_context(self, user_input: str, ai_response: str,
                                action_taken: Optional[str] = None):
        """
        Add conversation to context memory

        Args:
            user_input: What user said/requested
            ai_response: AI's response/plan
            action_taken: What action was actually taken
        """
        context_entry = {
            "user_input": user_input,
            "ai_response": ai_response[:500],  # Truncate long responses
            "action_taken": action_taken,
            "timestamp": datetime.now().isoformat()
        }

        self.memory.setdefault("conversation_context", []).append(context_entry)

        # Keep only last 100 conversations
        if len(self.memory["conversation_context"]) > 100:
            self.memory["conversation_context"] = self.memory["conversation_context"][-100:]

        self._save()

    def get_relevant_knowledge(self, category: Optional[str] = None,
                              tech_stack: Optional[Dict] = None) -> Dict:
        """
        Get relevant knowledge for current context

        Args:
            category: Filter by category
            tech_stack: Tech stack to find relevant patterns

        Returns:
            Relevant rules, learnings, and patterns
        """
        relevant = {
            "rules": [],
            "learnings": [],
            "patterns": [],
            "recent_successes": []
        }

        # Get applicable rules
        rules = self.memory.get("rules", [])
        if category:
            rules = [r for r in rules if r.get("category") == category]
        relevant["rules"] = sorted(rules, key=lambda x: x.get("confidence", 0), reverse=True)[:5]

        # Get relevant learnings
        learnings = self.memory.get("learnings", [])
        if category:
            learnings = [l for l in learnings if l.get("category") == category]
        relevant["learnings"] = sorted(learnings, key=lambda x: x.get("confidence", 0), reverse=True)[:5]

        # Get tech stack patterns
        if tech_stack:
            tech_combo = " + ".join([v for v in tech_stack.values() if v])
            patterns = [p for p in self.memory.get("patterns", [])
                       if tech_combo in p.get("condition", "")]
            relevant["patterns"] = patterns

        # Get recent successful builds
        recent_builds = self.memory.get("build_history", [])[-10:]
        successful = [b for b in recent_builds if b.get("success")]
        relevant["recent_successes"] = successful[-3:]

        return relevant

    def format_knowledge_for_prompt(self, category: Optional[str] = None,
                                   tech_stack: Optional[Dict] = None) -> str:
        """
        Format relevant knowledge as a prompt addition

        Args:
            category: Filter by category
            tech_stack: Tech stack context

        Returns:
            Formatted knowledge string for AI prompt
        """
        knowledge = self.get_relevant_knowledge(category, tech_stack)

        prompt_addition = "\n\n=== ACCUMULATED KNOWLEDGE ===\n\n"

        # Add rules
        if knowledge["rules"]:
            prompt_addition += "IMPORTANT RULES (from past experience):\n"
            for rule in knowledge["rules"]:
                prompt_addition += f"- {rule['rule']}\n"
                prompt_addition += f"  Rationale: {rule['rationale']}\n"

        # Add learnings
        if knowledge["learnings"]:
            prompt_addition += "\nKEY LEARNINGS:\n"
            for learning in knowledge["learnings"]:
                prompt_addition += f"- {learning['learning']}\n"

        # Add tech stack patterns
        if knowledge["patterns"]:
            prompt_addition += "\nBEST PRACTICES FOR THIS TECH STACK:\n"
            for pattern in knowledge["patterns"]:
                for practice in pattern.get("best_practices", []):
                    prompt_addition += f"- {practice}\n"

        # Add recent successes
        if knowledge["recent_successes"]:
            prompt_addition += "\nRECENT SUCCESSFUL BUILDS:\n"
            for build in knowledge["recent_successes"]:
                tech = " + ".join([v for v in build.get("tech_stack", {}).values() if v])
                prompt_addition += f"- {tech}: {build.get('files_created', 0)} files, "
                prompt_addition += f"{len(build.get('fixes_applied', []))} fixes applied\n"

        prompt_addition += "\n=== END KNOWLEDGE ===\n"

        return prompt_addition

    def apply_rule(self, rule_id: str):
        """Record that a rule was applied successfully"""
        for rule in self.memory.get("rules", []):
            if rule["id"] == rule_id:
                rule["applications"] = rule.get("applications", 0) + 1
                rule["confidence"] = min(1.0, rule.get("confidence", 0.5) + 0.05)
                self._save()
                break

    def apply_learning(self, learning_id: str):
        """Record that a learning was applied successfully"""
        for learning in self.memory.get("learnings", []):
            if learning["id"] == learning_id:
                learning["applications"] = learning.get("applications", 0) + 1
                learning["confidence"] = min(1.0, learning.get("confidence", 0.5) + 0.1)

                # Try to elevate to rule
                if learning["applications"] >= 3:
                    self._try_generate_rule(learning)

                self._save()
                break

    def get_statistics(self) -> Dict:
        """Get memory system statistics"""
        rules = self.memory.get("rules", [])
        learnings = self.memory.get("learnings", [])
        builds = self.memory.get("build_history", [])

        successful_builds = [b for b in builds if b.get("success")]
        recent_builds = builds[-10:]

        return {
            "total_rules": len(rules),
            "total_learnings": len(learnings),
            "total_builds": len(builds),
            "successful_builds": len(successful_builds),
            "success_rate": len(successful_builds) / len(builds) if builds else 0,
            "rules_with_high_confidence": len([r for r in rules if r.get("confidence", 0) >= 0.9]),
            "most_applied_rule": max(rules, key=lambda x: x.get("applications", 0))["rule"] if rules else None,
            "recent_improvements": len([b for b in recent_builds if b.get("success")])
        }

    def get_conversation_summary(self, last_n: int = 5) -> str:
        """Get summary of recent conversations"""
        conversations = self.memory.get("conversation_context", [])[-last_n:]

        if not conversations:
            return "No recent conversations."

        summary = "RECENT SESSION CONTEXT:\n"
        for conv in conversations:
            timestamp = datetime.fromisoformat(conv["timestamp"]).strftime("%Y-%m-%d %H:%M")
            summary += f"[{timestamp}] User: {conv['user_input'][:80]}...\n"
            if conv.get("action_taken"):
                summary += f"  Action: {conv['action_taken']}\n"

        return summary
