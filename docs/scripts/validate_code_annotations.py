#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "docs.scripts.validate_code_annotations",
#   "purpose": "Documentation tooling for validate code annotations workflows",
#   "sections": [
#     {
#       "id": "codeannotationvalidator",
#       "name": "CodeAnnotationValidator",
#       "anchor": "class-codeannotationvalidator",
#       "kind": "class"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Code Annotation Validation Script

This script validates that Python code follows the established annotation
standards for automated documentation generation. It checks for:

- Required docstrings on public interfaces
- Proper parameter and return value documentation
- Consistent terminology and formatting
- Exception documentation
- Module-level documentation

Usage:
    python docs/scripts/validate_code_annotations.py [source_directory]
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Public Classes ---


class CodeAnnotationValidator:
    """Validates code annotations against established standards."""

    def __init__(self, source_dir: str = "src", *, style_checks: bool = False):
        self.source_dir = Path(source_dir)
        self.issues: List[Dict] = []
        self.style_checks = style_checks

        # Standards definitions
        self.required_terms = ["document", "process", "search", "index"]
        self.forbidden_patterns = [r"TODO.*", r"FIXME.*", r"XXX.*", r"HACK.*"]

    def validate_file(self, file_path: Path) -> List[Dict]:
        """Validate annotations in a single Python file."""
        issues = []
        file_name = file_path.name

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return [{"type": "error", "message": f"Could not read file: {e}", "file": file_name}]

        # Parse the AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return [{"type": "error", "message": f"Syntax error: {e}", "file": file_name}]

        # Validate NAVMAP (if present) before other checks so ordering issues are surfaced early.
        lines = content.splitlines()
        issues.extend(self._validate_navmap(file_path, tree, lines))

        # Extract module docstring
        module_docstring = ast.get_docstring(tree)
        if not module_docstring:
            issues.append(
                {
                    "type": "warning",
                    "message": "Module missing docstring",
                    "file": file_name,
                    "line": 1,
                }
            )
        else:
            # Validate module docstring quality
            issues.extend(self._validate_module_docstring(module_docstring, file_name))

        # Check classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                issues.extend(self._validate_class(node, file_name))
            elif isinstance(node, ast.FunctionDef):
                issues.extend(self._validate_function(node, file_name))

        # Check for forbidden patterns
        issues.extend(self._check_forbidden_patterns(content, file_name))

        return issues

    def _validate_navmap(self, file_path: Path, tree: ast.Module, lines: List[str]) -> List[Dict]:
        """Ensure the NAVMAP sections enumerate classes and functions in order."""

        navmap = self._extract_navmap(lines)
        if navmap is None:
            return []

        sections = navmap.get("sections", [])
        expected: List[Tuple[str, str]] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                expected.append((node.name, "class"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                expected.append((node.name, "function"))

        actual: List[Tuple[str, str]] = [
            (section.get("name", ""), section.get("kind", "")) for section in sections
        ]

        if expected != actual:
            return [
                {
                    "type": "error",
                    "message": (
                        "NAVMAP mismatch: sections must list top-level classes and functions "
                        "in source order."
                    ),
                    "file": str(file_path),
                    "details": {
                        "expected": expected,
                        "actual": actual,
                    },
                }
            ]
        return []

    def _extract_navmap(self, lines: List[str]) -> Optional[Dict]:
        """Parse NAVMAP JSON payload from ``lines`` when present."""

        start = None
        for idx, line in enumerate(lines):
            if line.strip() == "# === NAVMAP v1 ===":
                start = idx
                break
        if start is None:
            return None

        end = None
        for idx in range(start + 1, len(lines)):
            if lines[idx].strip() == "# === /NAVMAP ===":
                end = idx
                break
        if end is None:
            return None

        payload_lines = []
        for line in lines[start + 1 : end]:
            stripped = line.lstrip("# ").rstrip("\n")
            payload_lines.append(stripped)
        payload = "\n".join(payload_lines).strip()
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _validate_module_docstring(self, docstring: str, file_name: str) -> List[Dict]:
        """Validate module-level docstring quality."""
        issues = []

        if not self.style_checks:
            return issues

        # Check minimum length
        if len(docstring.strip()) < 50:
            issues.append(
                {
                    "type": "warning",
                    "message": "Module docstring too short (minimum 50 characters)",
                    "file": file_name,
                    "line": 1,
                }
            )

        # Check for key sections
        required_sections = ["Args:", "Returns:", "Raises:"]
        for section in required_sections:
            if section in docstring:
                # Check that the section has content
                section_pattern = f"{section}\\s*\\n(.*?)(?=\\n\\n|\\n[A-Z]|$)"
                match = re.search(section_pattern, docstring, re.DOTALL)
                if match and not match.group(1).strip():
                    issues.append(
                        {
                            "type": "warning",
                            "message": f"Empty {section} section in module docstring",
                            "file": file_name,
                        }
                    )

        return issues

    def _validate_class(self, class_node: ast.ClassDef, file_name: str) -> List[Dict]:
        """Validate class definition and documentation."""
        issues = []

        # Check if class has docstring
        if not ast.get_docstring(class_node):
            if not class_node.name.startswith("_"):  # Only check public classes
                issues.append(
                    {
                        "type": "error",
                        "message": f"Public class '{class_node.name}' missing docstring",
                        "file": file_name,
                        "line": class_node.lineno,
                    }
                )
        else:
            docstring = ast.get_docstring(class_node)
            issues.extend(self._validate_class_docstring(docstring, class_node.name, file_name))

        # Check methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                issues.extend(self._validate_method(node, class_node.name, file_name))

        return issues

    def _validate_class_docstring(
        self, docstring: str, class_name: str, file_name: str
    ) -> List[Dict]:
        """Validate class docstring quality."""
        issues = []

        if not self.style_checks:
            return issues

        # Check for key sections
        if "Attributes:" not in docstring and "Args:" not in docstring:
            issues.append(
                {
                    "type": "info",
                    "message": f"Class '{class_name}' docstring missing Attributes or Args section",
                    "file": file_name,
                }
            )

        # Check for examples
        if "Examples:" not in docstring and ">>>" not in docstring:
            issues.append(
                {
                    "type": "info",
                    "message": f"Class '{class_name}' docstring missing usage examples",
                    "file": file_name,
                }
            )

        return issues

    def _validate_function(self, func_node: ast.FunctionDef, file_name: str) -> List[Dict]:
        """Validate function definition and documentation."""
        issues = []

        # Skip private functions for basic checks
        if func_node.name.startswith("_"):
            return issues

        # Check if function has docstring
        if not ast.get_docstring(func_node):
            issues.append(
                {
                    "type": "error",
                    "message": f"Public function '{func_node.name}' missing docstring",
                    "file": file_name,
                    "line": func_node.lineno,
                }
            )
        else:
            docstring = ast.get_docstring(func_node)
            issues.extend(self._validate_function_docstring(docstring, func_node.name, file_name))

        return issues

    def _validate_method(
        self, method_node: ast.FunctionDef, class_name: str, file_name: str
    ) -> List[Dict]:
        """Validate method definition and documentation."""
        issues = []

        # Check if method has docstring (skip private methods)
        if not method_node.name.startswith("_") and not ast.get_docstring(method_node):
            issues.append(
                {
                    "type": "error",
                    "message": f"Public method '{class_name}.{method_node.name}' missing docstring",
                    "file": file_name,
                    "line": method_node.lineno,
                }
            )
        else:
            docstring = ast.get_docstring(method_node)
            if docstring:
                issues.extend(
                    self._validate_method_docstring(
                        docstring, method_node.name, class_name, file_name
                    )
                )

        return issues

    def _validate_function_docstring(
        self, docstring: str, func_name: str, file_name: str
    ) -> List[Dict]:
        """Validate function docstring structure."""
        issues = []

        if not self.style_checks:
            return issues

        # Check for Args section
        if "Args:" not in docstring and not self._has_parameters(docstring):
            issues.append(
                {
                    "type": "warning",
                    "message": f"Function '{func_name}' docstring missing Args section",
                    "file": file_name,
                }
            )

        # Check for Returns section
        if "Returns:" not in docstring and not self._has_returns(docstring):
            issues.append(
                {
                    "type": "warning",
                    "message": f"Function '{func_name}' docstring missing Returns section",
                    "file": file_name,
                }
            )

        # Check for Raises section (for functions that might raise exceptions)
        if not self._has_raises(docstring) and self._function_might_raise_exceptions(func_name):
            issues.append(
                {
                    "type": "info",
                    "message": f"Function '{func_name}' should document potential exceptions",
                    "file": file_name,
                }
            )

        return issues

    def _validate_method_docstring(
        self, docstring: str, method_name: str, class_name: str, file_name: str
    ) -> List[Dict]:
        """Validate method docstring structure."""
        issues = []

        if not self.style_checks:
            return issues

        # Methods should have Args section if they have parameters
        if "Args:" not in docstring and not self._method_has_parameters(docstring):
            issues.append(
                {
                    "type": "warning",
                    "message": f"Method '{class_name}.{method_name}' docstring missing Args section",
                    "file": file_name,
                }
            )

        # Check for Returns section for non-void methods
        if "Returns:" not in docstring and not self._method_has_returns(docstring):
            issues.append(
                {
                    "type": "warning",
                    "message": f"Method '{class_name}.{method_name}' docstring missing Returns section",
                    "file": file_name,
                }
            )

        return issues

    def _has_parameters(self, docstring: str) -> bool:
        """Check if docstring documents parameters."""
        return "Args:" in docstring or "Parameters:" in docstring

    def _has_returns(self, docstring: str) -> bool:
        """Check if docstring documents return values."""
        return "Returns:" in docstring or "Return:" in docstring

    def _has_raises(self, docstring: str) -> bool:
        """Check if docstring documents exceptions."""
        return "Raises:" in docstring or "Raise:" in docstring

    def _method_has_parameters(self, docstring: str) -> bool:
        """Check if method docstring has parameter documentation."""
        return "Args:" in docstring or "Parameters:" in docstring

    def _method_has_returns(self, docstring: str) -> bool:
        """Check if method docstring has return documentation."""
        return "Returns:" in docstring or "Return:" in docstring

    def _function_might_raise_exceptions(self, func_name: str) -> bool:
        """Determine if a function might raise exceptions based on its name."""
        exception_indicators = [
            "load",
            "parse",
            "validate",
            "process",
            "extract",
            "generate",
            "create",
            "update",
            "delete",
        ]

        return any(indicator in func_name.lower() for indicator in exception_indicators)

    def _check_forbidden_patterns(self, content: str, file_name: str) -> List[Dict]:
        """Check for forbidden patterns in code."""
        issues = []

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern in self.forbidden_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(
                        {
                            "type": "warning",
                            "message": f"Forbidden pattern '{pattern}' found",
                            "file": file_name,
                            "line": i,
                            "context": line.strip(),
                        }
                    )

        return issues

    def validate_all_files(self) -> List[Dict]:
        """Validate annotations in all Python files."""
        print("🔍 Validating code annotations...")

        all_issues = []

        if not self.source_dir.exists():
            print(f"❌ Source directory not found: {self.source_dir}")
            return [{"type": "error", "message": "Source directory not found"}]

        # Find all Python files
        py_files = list(self.source_dir.rglob("*.py"))

        if not py_files:
            print("❌ No Python files found to validate")
            return [{"type": "error", "message": "No Python files found"}]

        print(f"📁 Found {len(py_files)} Python files to validate")

        for file_path in py_files:
            # Skip test files and generated files
            if any(
                skip in str(file_path)
                for skip in ["test", "__pycache__", "migrations", "generated"]
            ):
                continue

            issues = self.validate_file(file_path)
            all_issues.extend(issues)

            if issues:
                print(f"⚠️  Found {len(issues)} annotation issues in {file_path.name}")

        return all_issues

    def print_report(self, issues: List[Dict]):
        """Print a formatted validation report."""
        if not issues:
            print("\n✅ No code annotation issues found!")
            return

        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)

        print("\n📋 Code Annotation Validation Report")
        print("=" * 60)

        total_issues = len(issues)
        print(f"Total issues found: {total_issues}")

        for issue_type in ["error", "warning", "info"]:
            type_issues = issues_by_type.get(issue_type, [])
            if type_issues:
                print(f"\n{issue_type.upper()} ({len(type_issues)}):")
                for issue in type_issues[:10]:  # Show first 10 issues of each type
                    file_name = issue.get("file", "unknown")
                    message = issue.get("message", "No message")
                    line = issue.get("line", "")
                    line_str = f":{line}" if line else ""
                    print(f"  • {file_name}{line_str} - {message}")

                if len(type_issues) > 10:
                    print(f"  • ... and {len(type_issues) - 10} more {issue_type} issues")

        print(f"\n📊 Summary: {total_issues} total issues")
        print("Run this script regularly to maintain code annotation quality")


# --- Public Functions ---
# --- Module Entry Points ---


def main():
    """Main entry point for code annotation validation."""
    parser = argparse.ArgumentParser(description="Validate code annotations against standards")
    parser.add_argument(
        "source_dir", nargs="?", default="src", help="Source directory to validate (default: src)"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Attempt to automatically fix some annotation issues"
    )
    parser.add_argument(
        "--style",
        dest="style_checks",
        action="store_true",
        help="Enable docstring style checks (Args/Returns/Raises sections).",
    )

    args = parser.parse_args()

    print("🔍 Starting Code Annotation Validation")
    print("=" * 60)

    validator = CodeAnnotationValidator(args.source_dir, style_checks=args.style_checks)
    issues = validator.validate_all_files()

    validator.print_report(issues)

    # Exit with error code if there are errors
    error_count = len([i for i in issues if i.get("type") == "error"])

    if error_count > 0:
        print(f"\n❌ Validation failed: {error_count} errors found")
        sys.exit(1)
    else:
        print("\n✅ Code annotation validation passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
