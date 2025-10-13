#!/usr/bin/env python3
"""
Master Documentation Generation Script

This script coordinates all documentation generation tasks:
1. Generate API documentation from source code
2. Build Sphinx documentation
3. Validate documentation quality
4. Check for broken links

Usage:
    python docs/scripts/generate_all_docs.py [--validate-only] [--quick]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class DocumentationGenerator:
    """Coordinates all documentation generation tasks."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "docs"
        self.scripts_dir = self.docs_dir / "scripts"

    def run_script(self, script_name: str, args: List[str] = None) -> bool:
        """Run a documentation script with optional arguments."""
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False

        cmd = [str(script_path)]
        if args:
            cmd.extend(args)

        print(f"🏃 Running {script_name}...")
        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=True)
            print(f"✅ {script_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {script_name} failed with exit code {e.returncode}")
            return False

    def generate_api_docs(self) -> bool:
        """Generate API documentation from source code."""
        return self.run_script("generate_api_docs.py")

    def build_html_docs(self) -> bool:
        """Build HTML documentation using Sphinx."""
        return self.run_script("build_docs.py", ["--format", "html"])

    def validate_docs(self) -> bool:
        """Validate documentation quality and check for issues."""
        print("🔍 Running documentation validation...")

        success = True

        # Run quality checks
        if not self.run_script("build_docs.py", ["--format", "quality"]):
            success = False

        return success

    def check_broken_links(self) -> bool:
        """Check for broken links in documentation."""
        return self.run_script("build_docs.py", ["--format", "linkcheck"])

    def full_generation(self) -> bool:
        """Run complete documentation generation process."""
        print("🚀 Starting full documentation generation...")
        print("=" * 60)

        success = True

        # Step 1: Generate API documentation
        if not self.generate_api_docs():
            success = False

        # Step 2: Build HTML documentation
        if not self.build_html_docs():
            success = False

        # Step 3: Validate documentation
        if not self.validate_docs():
            success = False

        # Step 4: Check for broken links (optional)
        print("\n🔗 Checking for broken links...")
        if not self.check_broken_links():
            print("⚠️  Found broken links - review and fix them")
            # Don't fail the build for broken links, just warn

        return success

    def quick_generation(self) -> bool:
        """Run quick documentation generation (skip validation)."""
        print("⚡ Running quick documentation generation...")
        print("=" * 60)

        success = True

        # Just generate API docs and build HTML
        if not self.generate_api_docs():
            success = False

        if not self.build_html_docs():
            success = False

        return success


def main():
    """Main entry point for documentation generation."""
    parser = argparse.ArgumentParser(description="Generate DocsToKG documentation")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation checks, don't regenerate docs",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Skip validation and link checks for faster builds"
    )
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    # Create generator instance
    generator = DocumentationGenerator(args.project_root)

    # Run the requested operation
    success = False

    if args.validate_only:
        success = generator.validate_docs()
    elif args.quick:
        success = generator.quick_generation()
    else:
        success = generator.full_generation()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 Documentation generation completed successfully!")
        print("\n📋 Generated files:")
        print("  - API documentation: docs/04-api/")
        print("  - HTML documentation: docs/html/")
        print("  - Sphinx build: docs/build/_build/")
    else:
        print("❌ Documentation generation failed!")
        print("\n🔧 Check the error messages above and fix any issues.")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
