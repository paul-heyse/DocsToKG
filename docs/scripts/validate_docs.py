#!/usr/bin/env python3
"""
Documentation Validation Script

This script validates documentation quality by checking:
- Style guide compliance
- Broken links
- Missing required sections
- Proper formatting
- Consistent terminology
"""

import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml


class DocumentationValidator:
    """Validates documentation quality and consistency."""

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.issues: List[Dict] = []
        self.style_guide = self._load_style_guide()

    def _load_style_guide(self) -> Dict:
        """Load style guide rules from the STYLE_GUIDE.md file."""
        style_file = self.docs_dir / "STYLE_GUIDE.md"

        if not style_file.exists():
            print(f"⚠️  Style guide not found: {style_file}")
            return {}

        # For now, return basic rules - in a real implementation,
        # this would parse the style guide file
        return {
            "required_sections": ["Overview", "Prerequisites", "Installation"],
            "forbidden_terms": ["TODO", "FIXME", "XXX"],
            "required_terms": ["DocsToKG"],
            "heading_levels": {"max_depth": 4, "require_numbers": True},
        }

    def validate_file(self, file_path: Path) -> List[Dict]:
        """Validate a single documentation file."""
        issues = []
        file_name = file_path.name

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception as e:
            return [{"type": "error", "message": f"Could not read file: {e}", "file": file_name}]

        # Check for required sections
        issues.extend(self._check_required_sections(content, file_name))

        # Check for forbidden terms
        issues.extend(self._check_forbidden_terms(content, file_name))

        # Check for required terms
        issues.extend(self._check_required_terms(content, file_name))

        # Check heading structure
        issues.extend(self._check_heading_structure(lines, file_name))

        # Check for broken internal links
        issues.extend(self._check_internal_links(content, file_path, file_name))

        # Check for consistent formatting
        issues.extend(self._check_formatting(lines, file_name))

        return issues

    def _check_required_sections(self, content: str, file_name: str) -> List[Dict]:
        """Check if required sections are present."""
        issues = []

        # Only check certain file types for required sections
        if file_name in ["index.md"] or "setup" in file_name or "installation" in file_name:
            missing_sections = []

            for section in self.style_guide.get("required_sections", []):
                if f"## {section}" not in content and f"# {section}" not in content:
                    missing_sections.append(section)

            if missing_sections:
                issues.append(
                    {
                        "type": "warning",
                        "message": f"Missing required sections: {', '.join(missing_sections)}",
                        "file": file_name,
                    }
                )

        return issues

    def _check_forbidden_terms(self, content: str, file_name: str) -> List[Dict]:
        """Check for forbidden terms that should not be in documentation."""
        issues = []

        for term in self.style_guide.get("forbidden_terms", []):
            if term.lower() in content.lower():
                issues.append(
                    {
                        "type": "warning",
                        "message": f"Found forbidden term '{term}' - consider removing or replacing",
                        "file": file_name,
                    }
                )

        return issues

    def _check_required_terms(self, content: str, file_name: str) -> List[Dict]:
        """Check for required terms that should be mentioned."""
        issues = []

        for term in self.style_guide.get("required_terms", []):
            if term not in content:
                issues.append(
                    {
                        "type": "info",
                        "message": f"Consider mentioning '{term}' in this documentation",
                        "file": file_name,
                    }
                )

        return issues

    def _check_heading_structure(self, lines: List[str], file_name: str) -> List[Dict]:
        """Check heading structure and numbering."""
        issues = []
        prev_level = 0

        for i, line in enumerate(lines, 1):
            if line.startswith("#"):
                # Count the number of # symbols
                level = len(line) - len(line.lstrip("#"))

                if level > self.style_guide.get("heading_levels", {}).get("max_depth", 4):
                    issues.append(
                        {
                            "type": "warning",
                            "message": f"Heading too deep (level {level}) at line {i}",
                            "file": file_name,
                            "line": i,
                        }
                    )

                # Check for proper numbering in numbered sections
                if self.style_guide.get("heading_levels", {}).get("require_numbers", True):
                    if level <= 2 and not re.match(r"^#{1,2}\s+\d+\.", line):
                        issues.append(
                            {
                                "type": "info",
                                "message": f"Consider numbering section at line {i}",
                                "file": file_name,
                                "line": i,
                            }
                        )

        return issues

    def _check_internal_links(self, content: str, file_path: Path, file_name: str) -> List[Dict]:
        """Check for broken internal links."""
        issues = []

        # Find all markdown links
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        matches = re.findall(link_pattern, content)

        for link_text, link_url in matches:
            if link_url.startswith("#"):  # Anchor link
                # Check if anchor exists in the same file
                if not self._check_anchor_exists(content, link_url[1:]):
                    issues.append(
                        {
                            "type": "warning",
                            "message": f"Broken anchor link: {link_url}",
                            "file": file_name,
                        }
                    )

            elif not link_url.startswith("http"):  # Internal relative link
                # Check if the linked file exists
                linked_file = file_path.parent / link_url
                if not linked_file.exists():
                    issues.append(
                        {
                            "type": "warning",
                            "message": f"Broken internal link: {link_url}",
                            "file": file_name,
                        }
                    )

        return issues

    def _check_anchor_exists(self, content: str, anchor: str) -> bool:
        """Check if an anchor exists in the content."""
        # Look for heading with id attribute or direct anchor
        anchor_pattern = f'id="{anchor}"'
        if anchor_pattern in content:
            return True

        # Also check for implicit anchors from headings
        # This is a simplified check - in practice you'd want more sophisticated parsing
        return False

    def _check_formatting(self, lines: List[str], file_name: str) -> List[Dict]:
        """Check for consistent formatting."""
        issues = []

        for i, line in enumerate(lines, 1):
            # Check for trailing whitespace
            if line.rstrip() != line:
                issues.append(
                    {
                        "type": "info",
                        "message": f"Trailing whitespace at line {i}",
                        "file": file_name,
                        "line": i,
                    }
                )

            # Check for very long lines
            if len(line) > 100:
                issues.append(
                    {
                        "type": "info",
                        "message": f"Very long line ({len(line)} chars) at line {i}",
                        "file": file_name,
                        "line": i,
                    }
                )

        return issues

    def validate_all_docs(self) -> List[Dict]:
        """Validate all documentation files."""
        print("🔍 Validating documentation...")

        all_issues = []

        if not self.docs_dir.exists():
            print(f"❌ Documentation directory not found: {self.docs_dir}")
            return [{"type": "error", "message": "Documentation directory not found"}]

        # Find all markdown files
        md_files = list(self.docs_dir.rglob("*.md"))

        if not md_files:
            print("❌ No markdown files found to validate")
            return [{"type": "error", "message": "No markdown files found"}]

        print(f"📁 Found {len(md_files)} markdown files to validate")

        for file_path in md_files:
            issues = self.validate_file(file_path)
            all_issues.extend(issues)

            if issues:
                print(f"⚠️  Found {len(issues)} issues in {file_path.name}")

        return all_issues

    def print_report(self, issues: List[Dict]):
        """Print a formatted validation report."""
        if not issues:
            print("\n✅ No documentation issues found!")
            return

        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)

        print(f"\n📋 Documentation Validation Report")
        print("=" * 50)

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
        print("Run 'python docs/scripts/generate_all_docs.py --validate-only' to see all details")


def main():
    """Main entry point for documentation validation."""
    print("🔍 Starting Documentation Validation")
    print("=" * 50)

    validator = DocumentationValidator()
    issues = validator.validate_all_docs()

    validator.print_report(issues)

    # Exit with error code if there are errors or warnings
    error_count = len([i for i in issues if i.get("type") == "error"])
    warning_count = len([i for i in issues if i.get("type") == "warning"])

    if error_count > 0:
        print(f"\n❌ Validation failed: {error_count} errors, {warning_count} warnings")
        sys.exit(1)
    elif warning_count > 0:
        print(f"\n⚠️  Validation completed with warnings: {warning_count} warnings")
        sys.exit(0)
    else:
        print("\n✅ Validation passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
