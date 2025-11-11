"""
Error Knowledge Base
Persistent storage of common errors and their fixes
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorKnowledgeBase:
    """
    Stores and retrieves known error patterns and their fixes
    """

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.kb_path = config_dir / "error_knowledge_base.json"
        self.knowledge_base: Dict = self._load()

    def _load(self) -> Dict:
        """Load knowledge base from disk"""
        if self.kb_path.exists():
            try:
                with open(self.kb_path, 'r') as f:
                    kb = json.load(f)
                    logger.info(f"Loaded {len(kb.get('patterns', []))} error patterns from knowledge base")
                    return kb
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")

        # Default knowledge base with common patterns
        return {
            "version": "1.0",
            "patterns": [
                {
                    "id": "flask_not_found",
                    "error_pattern": "flask: not found",
                    "error_type": "runtime",
                    "description": "Flask command not found - needs to use venv",
                    "fix_type": "run_command",
                    "solution": "Use venv Python with -m flask instead of flask command",
                    "success_count": 0,
                    "last_seen": None
                },
                {
                    "id": "package_json_not_found",
                    "error_pattern": "Could not read package.json",
                    "error_type": "runtime",
                    "description": "package.json not found at expected path",
                    "fix_type": "run_command",
                    "solution": "Check if package.json exists before running npm commands, adjust path",
                    "success_count": 0,
                    "last_seen": None
                },
                {
                    "id": "module_not_found_python",
                    "error_pattern": "ModuleNotFoundError:",
                    "error_type": "runtime",
                    "description": "Python module not installed",
                    "fix_type": "dependency",
                    "solution": "Reinstall dependencies with pip install -r requirements.txt",
                    "success_count": 0,
                    "last_seen": None
                },
                {
                    "id": "cannot_find_module_node",
                    "error_pattern": "Cannot find module",
                    "error_type": "runtime",
                    "description": "Node.js module not installed",
                    "fix_type": "dependency",
                    "solution": "Run npm install to install missing packages",
                    "success_count": 0,
                    "last_seen": None
                },
                {
                    "id": "invalid_package_config",
                    "error_pattern": "ERR_INVALID_PACKAGE_CONFIG",
                    "error_type": "syntax",
                    "description": "Invalid package.json configuration",
                    "fix_type": "config",
                    "solution": "Fix JSON syntax in package.json - check commas, quotes, structure",
                    "success_count": 0,
                    "last_seen": None
                },
                {
                    "id": "python_syntax_error",
                    "error_pattern": "SyntaxError:",
                    "error_type": "syntax",
                    "description": "Python syntax error",
                    "fix_type": "code",
                    "solution": "Fix Python syntax - check indentation, colons, parentheses",
                    "success_count": 0,
                    "last_seen": None
                },
                {
                    "id": "import_error",
                    "error_pattern": "ImportError:",
                    "error_type": "runtime",
                    "description": "Python import error",
                    "fix_type": "code",
                    "solution": "Fix import statement - check module path and installation",
                    "success_count": 0,
                    "last_seen": None
                },
                {
                    "id": "enoent_error",
                    "error_pattern": "ENOENT: no such file or directory",
                    "error_type": "runtime",
                    "description": "File or directory not found",
                    "fix_type": "path",
                    "solution": "Check if file/directory exists, correct the path in run command",
                    "success_count": 0,
                    "last_seen": None
                }
            ],
            "successful_fixes": [],
            "last_updated": datetime.now().isoformat()
        }

    def _save(self):
        """Save knowledge base to disk"""
        try:
            self.knowledge_base["last_updated"] = datetime.now().isoformat()
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.kb_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            logger.info("Knowledge base saved")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def find_matching_patterns(self, error_message: str) -> List[Dict]:
        """
        Find error patterns that match the given error message

        Args:
            error_message: The error message to match

        Returns:
            List of matching patterns, sorted by success count
        """
        matches = []
        for pattern in self.knowledge_base.get("patterns", []):
            if pattern["error_pattern"].lower() in error_message.lower():
                matches.append(pattern)

        # Sort by success count (most successful first)
        matches.sort(key=lambda x: x.get("success_count", 0), reverse=True)

        if matches:
            logger.info(f"Found {len(matches)} matching error patterns")

        return matches

    def record_successful_fix(self, pattern_id: str, error_message: str, fix_applied: str):
        """
        Record a successful fix for a pattern

        Args:
            pattern_id: ID of the pattern that was fixed
            error_message: The original error message
            fix_applied: Description of the fix that worked
        """
        # Update pattern success count
        for pattern in self.knowledge_base.get("patterns", []):
            if pattern["id"] == pattern_id:
                pattern["success_count"] = pattern.get("success_count", 0) + 1
                pattern["last_seen"] = datetime.now().isoformat()
                logger.info(f"Recorded success for pattern '{pattern_id}' (count: {pattern['success_count']})")
                break

        # Record in successful fixes log
        self.knowledge_base.setdefault("successful_fixes", []).append({
            "pattern_id": pattern_id,
            "error_message": error_message[:200],
            "fix_applied": fix_applied,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 100 successful fixes
        if len(self.knowledge_base["successful_fixes"]) > 100:
            self.knowledge_base["successful_fixes"] = self.knowledge_base["successful_fixes"][-100:]

        self._save()

    def add_new_pattern(self, error_pattern: str, error_type: str, description: str,
                       fix_type: str, solution: str) -> str:
        """
        Add a new error pattern to the knowledge base

        Args:
            error_pattern: Pattern to match in error messages
            error_type: Type of error (syntax, runtime, etc.)
            description: Human-readable description
            fix_type: Type of fix needed
            solution: Solution description

        Returns:
            Pattern ID
        """
        pattern_id = f"custom_{len(self.knowledge_base['patterns'])}"

        new_pattern = {
            "id": pattern_id,
            "error_pattern": error_pattern,
            "error_type": error_type,
            "description": description,
            "fix_type": fix_type,
            "solution": solution,
            "success_count": 0,
            "last_seen": datetime.now().isoformat()
        }

        self.knowledge_base.setdefault("patterns", []).append(new_pattern)
        self._save()

        logger.info(f"Added new error pattern: {pattern_id}")
        return pattern_id

    def get_fix_suggestions(self, error_message: str) -> List[str]:
        """
        Get fix suggestions based on matching patterns

        Args:
            error_message: The error message

        Returns:
            List of fix suggestions
        """
        matches = self.find_matching_patterns(error_message)
        return [match["solution"] for match in matches[:3]]  # Top 3 suggestions

    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        patterns = self.knowledge_base.get("patterns", [])
        successful_fixes = self.knowledge_base.get("successful_fixes", [])

        return {
            "total_patterns": len(patterns),
            "patterns_with_successes": len([p for p in patterns if p.get("success_count", 0) > 0]),
            "total_successful_fixes": sum(p.get("success_count", 0) for p in patterns),
            "recent_fixes": len(successful_fixes),
            "most_common_error": max(patterns, key=lambda x: x.get("success_count", 0))["id"] if patterns else None
        }
