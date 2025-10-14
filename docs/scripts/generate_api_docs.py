#!/usr/bin/env python3
"""
Automated API Documentation Generator

This script generates API documentation from Python docstrings and code structure.
It scans the source code, extracts function and class definitions, and generates
Markdown documentation that can be included in the docs.
"""

import ast
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import re


class APIDocGenerator:
    """Generates API documentation from Python source code."""

    def __init__(self, source_dir: str = "src", output_dir: str = "docs/04-api"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scan_source_files(self) -> List[Path]:
        """Scan for Python source files to document."""
        python_files = []

        if not self.source_dir.exists():
            print(f"Warning: Source directory {self.source_dir} does not exist")
            return python_files

        for py_file in self.source_dir.rglob("*.py"):
            # Skip test files, __pycache__, and migration files
            if any(skip in str(py_file) for skip in ["test", "__pycache__", "migrations"]):
                continue
            python_files.append(py_file)

        return python_files

    def extract_module_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract module-level information from a Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Extract module docstring
            module_docstring = ast.get_docstring(tree)

            # Find all classes and functions
            classes = []
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(
                        {
                            "name": node.name,
                            "lineno": node.lineno,
                            "docstring": ast.get_docstring(node),
                        }
                    )
                elif isinstance(node, ast.FunctionDef) and node.name != "__init__":
                    functions.append(
                        {
                            "name": node.name,
                            "lineno": node.lineno,
                            "docstring": ast.get_docstring(node),
                            "args": [arg.arg for arg in node.args.args],
                        }
                    )

            return {
                "file_path": file_path,
                "module_name": file_path.stem,
                "module_docstring": module_docstring,
                "classes": classes,
                "functions": functions,
            }

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {}

    def generate_module_docs(self, module_info: Dict[str, Any]) -> str:
        """Generate Markdown documentation for a module."""
        lines = []

        # Module header
        lines.append(f"# Module: {module_info['module_name']}")
        lines.append("")

        # Module description
        if module_info.get("module_docstring"):
            lines.append(module_info["module_docstring"])
            lines.append("")

        # Functions section
        if module_info.get("functions"):
            lines.append("## Functions")
            lines.append("")

            for func in module_info["functions"]:
                lines.append(f"### `{func['name']}({', '.join(func['args'])})`")
                lines.append("")

                if func.get("docstring"):
                    # Clean up the docstring
                    docstring = self._clean_docstring(func["docstring"])
                    lines.append(docstring)
                else:
                    lines.append("*No documentation available.*")

                lines.append("")

        # Classes section
        if module_info.get("classes"):
            lines.append("## Classes")
            lines.append("")

            for cls in module_info["classes"]:
                lines.append(f"### `{cls['name']}`")
                lines.append("")

                if cls.get("docstring"):
                    # Clean up the docstring
                    docstring = self._clean_docstring(cls["docstring"])
                    lines.append(docstring)
                else:
                    lines.append("*No documentation available.*")

                lines.append("")

        return "\n".join(lines)

    def _clean_docstring(self, docstring: str) -> str:
        """Clean and format docstring for Markdown."""
        if not docstring:
            return ""

        # Remove leading/trailing whitespace
        lines = docstring.strip().split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove common leading whitespace
            cleaned_line = re.sub(r"^\s{4,}", "", line)
            cleaned_lines.append(cleaned_line)

        return "\n".join(cleaned_lines)

    def generate_all_docs(self):
        """Generate documentation for all modules."""
        print("🔍 Scanning source files...")
        source_files = self.scan_source_files()

        if not source_files:
            print("❌ No source files found to document")
            return

        print(f"📝 Found {len(source_files)} Python files to document")

        generated_files = []

        for file_path in source_files:
            print(f"📖 Processing {file_path}...")

            module_info = self.extract_module_info(file_path)

            if not module_info:
                continue

            # Generate documentation
            docs_content = self.generate_module_docs(module_info)

            # Write to output file
            output_file = self.output_dir / f"{module_info['module_name']}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(docs_content)

            generated_files.append(output_file)
            print(f"✅ Generated {output_file}")

        print(f"\n🎉 Generated documentation for {len(generated_files)} modules")
        return generated_files


def main():
    """Main entry point for API documentation generation."""
    print("🚀 Starting API Documentation Generation")
    print("=" * 50)

    # Default paths - can be overridden via command line
    source_dir = "src"
    output_dir = "docs/04-api"

    # Check for command line arguments
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    generator = APIDocGenerator(source_dir, output_dir)
    generated_files = generator.generate_all_docs()

    if generated_files:
        print("\n📋 Generated Files:")
        for file_path in generated_files:
            print(f"  - {file_path}")
    else:
        print("\n❌ No documentation files were generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
