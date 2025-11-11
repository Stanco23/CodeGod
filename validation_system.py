"""
Validation and Error Handling System
Ensures quality, correctness, and handles failures autonomously
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
import subprocess
import ast


class ValidationResult(Enum):
    """Result of validation"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: str  # error, warning, info
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report"""
    result: ValidationResult
    validator_name: str
    issues: List[ValidationIssue]
    execution_time: float
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def has_errors(self) -> bool:
        """Check if report has any errors"""
        return any(issue.severity == "error" for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if report has any warnings"""
        return any(issue.severity == "warning" for issue in self.issues)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        error_count = sum(1 for i in self.issues if i.severity == "error")
        warning_count = sum(1 for i in self.issues if i.severity == "warning")

        return (
            f"{self.validator_name}: "
            f"{error_count} errors, {warning_count} warnings - "
            f"{self.result.value}"
        )


class Validator:
    """Base validator class"""

    def __init__(self, name: str):
        self.name = name

    async def validate(self, code: str, language: str, context: Dict = None) -> ValidationReport:
        """Validate code and return report"""
        raise NotImplementedError


class SyntaxValidator(Validator):
    """Validates code syntax"""

    def __init__(self):
        super().__init__("SyntaxValidator")

    async def validate(self, code: str, language: str, context: Dict = None) -> ValidationReport:
        """Check syntax validity"""
        import time

        start_time = time.time()
        issues = []

        try:
            if language == "python":
                issues = self._validate_python_syntax(code)
            elif language in ["javascript", "typescript"]:
                issues = self._validate_javascript_syntax(code, language)
            elif language == "go":
                issues = self._validate_go_syntax(code)
            else:
                issues.append(ValidationIssue(
                    severity="warning",
                    message=f"Syntax validation not supported for {language}"
                ))

            result = ValidationResult.FAIL if any(i.severity == "error" for i in issues) else ValidationResult.PASS

            return ValidationReport(
                result=result,
                validator_name=self.name,
                issues=issues,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationReport(
                result=ValidationResult.FAIL,
                validator_name=self.name,
                issues=[ValidationIssue(severity="error", message=f"Validation error: {str(e)}")],
                execution_time=time.time() - start_time
            )

    def _validate_python_syntax(self, code: str) -> List[ValidationIssue]:
        """Validate Python syntax"""
        issues = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity="error",
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                column=e.offset,
                code_snippet=e.text
            ))

        return issues

    def _validate_javascript_syntax(self, code: str, language: str) -> List[ValidationIssue]:
        """Validate JavaScript/TypeScript syntax"""
        issues = []

        try:
            # Use Node.js to check syntax
            result = subprocess.run(
                ["node", "--check"],
                input=code,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"Syntax error: {result.stderr}"
                ))

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                severity="error",
                message="Syntax check timed out"
            ))
        except FileNotFoundError:
            issues.append(ValidationIssue(
                severity="warning",
                message="Node.js not available for syntax checking"
            ))

        return issues

    def _validate_go_syntax(self, code: str) -> List[ValidationIssue]:
        """Validate Go syntax"""
        issues = []

        try:
            # Use go fmt to check syntax
            result = subprocess.run(
                ["go", "fmt"],
                input=code,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"Syntax error: {result.stderr}"
                ))

        except FileNotFoundError:
            issues.append(ValidationIssue(
                severity="warning",
                message="Go compiler not available for syntax checking"
            ))

        return issues


class TypeValidator(Validator):
    """Validates type correctness"""

    def __init__(self):
        super().__init__("TypeValidator")

    async def validate(self, code: str, language: str, context: Dict = None) -> ValidationReport:
        """Check type correctness"""
        import time

        start_time = time.time()
        issues = []

        try:
            if language == "python":
                issues = await self._validate_python_types(code, context)
            elif language == "typescript":
                issues = await self._validate_typescript_types(code, context)
            else:
                return ValidationReport(
                    result=ValidationResult.SKIP,
                    validator_name=self.name,
                    issues=[],
                    execution_time=time.time() - start_time
                )

            result = ValidationResult.FAIL if any(i.severity == "error" for i in issues) else ValidationResult.PASS

            return ValidationReport(
                result=result,
                validator_name=self.name,
                issues=issues,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationReport(
                result=ValidationResult.FAIL,
                validator_name=self.name,
                issues=[ValidationIssue(severity="error", message=str(e))],
                execution_time=time.time() - start_time
            )

    async def _validate_python_types(self, code: str, context: Dict) -> List[ValidationIssue]:
        """Validate Python types with mypy"""
        issues = []

        try:
            # Write code to temp file and run mypy
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    ["mypy", temp_file, "--strict"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    # Parse mypy output
                    for line in result.stdout.split('\n'):
                        if ':' in line:
                            issues.append(ValidationIssue(
                                severity="error",
                                message=line
                            ))
            finally:
                os.unlink(temp_file)

        except FileNotFoundError:
            issues.append(ValidationIssue(
                severity="warning",
                message="mypy not available for type checking"
            ))

        return issues

    async def _validate_typescript_types(self, code: str, context: Dict) -> List[ValidationIssue]:
        """Validate TypeScript types"""
        issues = []

        try:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    ["tsc", "--noEmit", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    for line in result.stdout.split('\n'):
                        if 'error TS' in line:
                            issues.append(ValidationIssue(
                                severity="error",
                                message=line
                            ))
            finally:
                os.unlink(temp_file)

        except FileNotFoundError:
            issues.append(ValidationIssue(
                severity="warning",
                message="TypeScript compiler not available"
            ))

        return issues


class LintValidator(Validator):
    """Validates code style and quality"""

    def __init__(self):
        super().__init__("LintValidator")

    async def validate(self, code: str, language: str, context: Dict = None) -> ValidationReport:
        """Run linting checks"""
        import time

        start_time = time.time()
        issues = []

        try:
            if language == "python":
                issues = await self._lint_python(code)
            elif language in ["javascript", "typescript"]:
                issues = await self._lint_javascript(code, language)
            else:
                return ValidationReport(
                    result=ValidationResult.SKIP,
                    validator_name=self.name,
                    issues=[],
                    execution_time=time.time() - start_time
                )

            result = ValidationResult.FAIL if any(i.severity == "error" for i in issues) else ValidationResult.PASS

            return ValidationReport(
                result=result,
                validator_name=self.name,
                issues=issues,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationReport(
                result=ValidationResult.FAIL,
                validator_name=self.name,
                issues=[ValidationIssue(severity="error", message=str(e))],
                execution_time=time.time() - start_time
            )

    async def _lint_python(self, code: str) -> List[ValidationIssue]:
        """Lint Python code with pylint"""
        issues = []

        try:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    ["pylint", temp_file, "--output-format=json"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )

                if result.stdout:
                    lint_results = json.loads(result.stdout)
                    for item in lint_results:
                        issues.append(ValidationIssue(
                            severity="warning" if item['type'] == 'warning' else "error",
                            message=item['message'],
                            line_number=item['line'],
                            column=item['column']
                        ))
            finally:
                os.unlink(temp_file)

        except FileNotFoundError:
            # pylint not available, skip
            pass

        return issues

    async def _lint_javascript(self, code: str, language: str) -> List[ValidationIssue]:
        """Lint JavaScript/TypeScript with eslint"""
        issues = []

        try:
            import tempfile
            import os

            ext = '.ts' if language == 'typescript' else '.js'
            with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    ["eslint", temp_file, "--format=json"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )

                if result.stdout:
                    lint_results = json.loads(result.stdout)
                    for file_result in lint_results:
                        for message in file_result.get('messages', []):
                            issues.append(ValidationIssue(
                                severity=message['severity'] == 2 and "error" or "warning",
                                message=message['message'],
                                line_number=message['line'],
                                column=message['column']
                            ))
            finally:
                os.unlink(temp_file)

        except FileNotFoundError:
            # eslint not available, skip
            pass

        return issues


class SecurityValidator(Validator):
    """Validates security best practices"""

    def __init__(self):
        super().__init__("SecurityValidator")

    async def validate(self, code: str, language: str, context: Dict = None) -> ValidationReport:
        """Check for common security issues"""
        import time

        start_time = time.time()
        issues = []

        # Check for common security patterns
        security_patterns = {
            r"eval\(": "Avoid using eval() - security risk",
            r"exec\(": "Avoid using exec() - security risk",
            r"__import__": "Avoid dynamic imports with __import__",
            r"shell=True": "Avoid shell=True in subprocess calls",
            r"pickle\.loads": "Avoid pickle.loads() - can execute arbitrary code",
            r"innerHTML\s*=": "Avoid innerHTML assignment - XSS risk",
            r"dangerouslySetInnerHTML": "Use dangerouslySetInnerHTML carefully - XSS risk",
            r"SELECT\s+.*\s+FROM.*\+": "Possible SQL injection - use parameterized queries",
            r"password\s*=\s*['\"]": "Avoid hardcoded passwords",
            r"api_key\s*=\s*['\"]": "Avoid hardcoded API keys",
        }

        for pattern, message in security_patterns.items():
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                # Find line number
                line_num = code[:match.start()].count('\n') + 1
                issues.append(ValidationIssue(
                    severity="warning",
                    message=message,
                    line_number=line_num
                ))

        result = ValidationResult.PASS if not issues else ValidationResult.WARNING

        return ValidationReport(
            result=result,
            validator_name=self.name,
            issues=issues,
            execution_time=time.time() - start_time
        )


class TestValidator(Validator):
    """Runs unit tests"""

    def __init__(self):
        super().__init__("TestValidator")

    async def validate(self, code: str, language: str, context: Dict = None) -> ValidationReport:
        """Run unit tests"""
        import time

        start_time = time.time()
        issues = []

        try:
            test_file = context.get("test_file") if context else None

            if not test_file:
                return ValidationReport(
                    result=ValidationResult.SKIP,
                    validator_name=self.name,
                    issues=[ValidationIssue(severity="info", message="No test file provided")],
                    execution_time=time.time() - start_time
                )

            if language == "python":
                issues = await self._run_pytest(test_file)
            elif language in ["javascript", "typescript"]:
                issues = await self._run_jest(test_file)
            else:
                return ValidationReport(
                    result=ValidationResult.SKIP,
                    validator_name=self.name,
                    issues=[],
                    execution_time=time.time() - start_time
                )

            result = ValidationResult.FAIL if any(i.severity == "error" for i in issues) else ValidationResult.PASS

            return ValidationReport(
                result=result,
                validator_name=self.name,
                issues=issues,
                execution_time=time.time() - start_time,
                metadata=context or {}
            )

        except Exception as e:
            return ValidationReport(
                result=ValidationResult.FAIL,
                validator_name=self.name,
                issues=[ValidationIssue(severity="error", message=str(e))],
                execution_time=time.time() - start_time
            )

    async def _run_pytest(self, test_file: str) -> List[ValidationIssue]:
        """Run pytest"""
        issues = []

        try:
            result = subprocess.run(
                ["pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                # Parse pytest output for failures
                for line in result.stdout.split('\n'):
                    if 'FAILED' in line or 'ERROR' in line:
                        issues.append(ValidationIssue(
                            severity="error",
                            message=line.strip()
                        ))

        except FileNotFoundError:
            issues.append(ValidationIssue(
                severity="error",
                message="pytest not available"
            ))

        return issues

    async def _run_jest(self, test_file: str) -> List[ValidationIssue]:
        """Run jest"""
        issues = []

        try:
            result = subprocess.run(
                ["jest", test_file, "--verbose"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                for line in result.stdout.split('\n'):
                    if 'FAIL' in line or '✕' in line:
                        issues.append(ValidationIssue(
                            severity="error",
                            message=line.strip()
                        ))

        except FileNotFoundError:
            issues.append(ValidationIssue(
                severity="error",
                message="jest not available"
            ))

        return issues


class ValidationPipeline:
    """
    Orchestrates multiple validators
    """

    def __init__(self):
        self.validators: List[Validator] = [
            SyntaxValidator(),
            TypeValidator(),
            LintValidator(),
            SecurityValidator(),
            TestValidator()
        ]

    async def validate(
        self,
        code: str,
        language: str,
        validation_rules: List[str],
        context: Dict = None
    ) -> List[ValidationReport]:
        """
        Run validation pipeline

        Args:
            code: Code to validate
            language: Programming language
            validation_rules: List of validation types to run
            context: Additional context (e.g., test files)

        Returns:
            List of validation reports
        """
        reports = []

        validator_map = {
            "syntax_check": SyntaxValidator(),
            "type_check": TypeValidator(),
            "lint": LintValidator(),
            "security_scan": SecurityValidator(),
            "unit_test": TestValidator()
        }

        for rule in validation_rules:
            validator = validator_map.get(rule)
            if validator:
                report = await validator.validate(code, language, context)
                reports.append(report)

        return reports

    def should_retry(self, reports: List[ValidationReport]) -> Tuple[bool, str]:
        """
        Determine if task should be retried based on validation

        Returns:
            (should_retry, reason)
        """
        for report in reports:
            if report.result == ValidationResult.FAIL:
                error_messages = [i.message for i in report.issues if i.severity == "error"]
                return True, f"{report.validator_name} failed: {'; '.join(error_messages[:3])}"

        return False, ""

    def generate_fix_prompt(self, code: str, reports: List[ValidationReport]) -> str:
        """
        Generate prompt for fixing validation issues

        Args:
            code: Original code
            reports: Validation reports

        Returns:
            Prompt for worker agent to fix issues
        """
        issues_summary = []

        for report in reports:
            if report.has_errors():
                issues_summary.append(f"\n{report.validator_name} Errors:")
                for issue in report.issues:
                    if issue.severity == "error":
                        location = f" (Line {issue.line_number})" if issue.line_number else ""
                        issues_summary.append(f"- {issue.message}{location}")
                        if issue.suggestion:
                            issues_summary.append(f"  Suggestion: {issue.suggestion}")

        prompt = f"""The following code has validation errors that need to be fixed:

```
{code}
```

VALIDATION ISSUES:
{''.join(issues_summary)}

TASK:
Fix all validation errors while preserving the original functionality.
Ensure the fixed code passes all validation checks.

OUTPUT FORMAT:
{{
    "success": true,
    "fixed_code": "... complete corrected code ...",
    "changes_made": ["list", "of", "changes"],
    "explanation": "brief explanation of fixes"
}}

CRITICAL:
- Fix ALL errors
- Maintain original functionality
- Do not introduce new issues
"""

        return prompt
