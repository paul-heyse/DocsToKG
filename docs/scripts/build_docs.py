#!/usr/bin/env python3
# === NAVMAP v1 ===
# {
#   "module": "docs.scripts.build_docs",
#   "purpose": "Documentation tooling for build docs workflows",
#   "sections": [
#     {
#       "id": "docs_builder",
#       "name": "DocsBuilder",
#       "anchor": "DOCS",
#       "kind": "class"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "MAIN",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Documentation Build Script

This script builds the Sphinx documentation and handles various build configurations.
It can build HTML documentation, check for broken links, and validate documentation quality.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


class DocsBuilder:
    """Builds Sphinx documentation with various configurations."""

    def __init__(self, source_dir: str = "docs/build/sphinx", build_dir: str = "docs/build/_build"):
        self.source_dir = Path(source_dir)
        self.build_dir = Path(build_dir)
        self.docs_root = Path("docs")
        self.sphinx_cmd = "sphinx-build"

    def check_sphinx_installation(self) -> bool:
        """Check if Sphinx is installed and available."""
        try:
            result = subprocess.run(
                [self.sphinx_cmd, "--version"], capture_output=True, text=True, check=True
            )
            print(f"✅ Sphinx found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Sphinx not found. Please install with: pip install sphinx")
            return False

    def clean_build_directory(self):
        """Clean the build directory before building."""
        if self.build_dir.exists():
            print(f"🧹 Cleaning build directory: {self.build_dir}")
            shutil.rmtree(self.build_dir)

    def prepare_source_tree(self) -> None:
        """Mirror the Markdown documentation tree into the Sphinx source directory."""

        docs_root = self.docs_root
        if not docs_root.exists():
            return

        skip_dirs = {"build", "scripts", "__pycache__"}

        for src_path in docs_root.iterdir():
            if not src_path.is_dir():
                continue
            if src_path.name in skip_dirs:
                continue

            dest_path = self.source_dir / src_path.name

            if dest_path.exists() or dest_path.is_symlink():
                if dest_path.is_symlink() or dest_path.is_file():
                    dest_path.unlink()
                elif dest_path.is_dir():
                    shutil.rmtree(dest_path)

            shutil.copytree(src_path, dest_path)

        for src_file in docs_root.glob("*.md"):
            dest_file = self.source_dir / src_file.name
            if dest_file.exists():
                dest_file.unlink()
            shutil.copy2(src_file, dest_file)

    def build_html(self, clean: bool = True) -> bool:
        """Build HTML documentation."""
        if clean:
            self.clean_build_directory()

        self.prepare_source_tree()

        print("🔨 Building HTML documentation...")

        cmd = [
            self.sphinx_cmd,
            "-b",
            "html",
            "-W",  # Treat warnings as errors
            "-E",  # Don't use a saved environment
            str(self.source_dir),
            str(self.build_dir / "html"),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ HTML documentation built successfully")

            # Copy to main docs directory if it exists
            html_output = self.build_dir / "html"
            if html_output.exists():
                docs_html = Path("docs") / "html"
                if docs_html.exists():
                    shutil.rmtree(docs_html)
                shutil.copytree(html_output, docs_html)
                print(f"📋 HTML docs copied to: {docs_html}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ HTML build failed: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False

    def build_latex(self) -> bool:
        """Build LaTeX documentation (for PDF generation)."""
        print("📄 Building LaTeX documentation...")

        self.prepare_source_tree()

        cmd = [self.sphinx_cmd, "-b", "latex", str(self.source_dir), str(self.build_dir / "latex")]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ LaTeX documentation built successfully")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ LaTeX build failed: {e}")
            return False

    def check_links(self) -> bool:
        """Check for broken links in the documentation."""
        print("🔗 Checking for broken links...")

        self.prepare_source_tree()

        cmd = [
            self.sphinx_cmd,
            "-b",
            "linkcheck",
            str(self.source_dir),
            str(self.build_dir / "linkcheck"),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Link check completed")

            # Check if there are any broken links
            linkcheck_output = self.build_dir / "linkcheck" / "output.txt"
            if linkcheck_output.exists():
                with open(linkcheck_output, "r") as f:
                    content = f.read()

                if "broken" in content.lower():
                    print("⚠️  Found broken links in documentation")
                    print("Check the linkcheck output for details")
                    return False
                else:
                    print("✅ No broken links found")
                    return True

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Link check failed: {e}")
            return False

    def check_coverage(self) -> bool:
        """Check documentation coverage."""
        print("📊 Checking documentation coverage...")

        self.prepare_source_tree()

        cmd = [
            self.sphinx_cmd,
            "-b",
            "coverage",
            str(self.source_dir),
            str(self.build_dir / "coverage"),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Coverage check completed")

            # Show coverage report
            coverage_file = self.build_dir / "coverage" / "python.txt"
            if coverage_file.exists():
                print("\n📋 Documentation Coverage Report:")
                with open(coverage_file, "r") as f:
                    content = f.read()
                    print(content)

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Coverage check failed: {e}")
            return False

    def build_all(self) -> bool:
        """Build all documentation formats."""
        print("🚀 Building all documentation formats...")

        success = True

        # Build HTML (required)
        if not self.build_html():
            success = False

        # Build LaTeX (optional, don't fail if it fails)
        if not self.build_latex():
            print("⚠️  LaTeX build failed, but continuing...")

        return success

    def run_quality_checks(self) -> bool:
        """Run all quality checks on the documentation."""
        print("🔍 Running documentation quality checks...")

        success = True

        # Check links
        if not self.check_links():
            success = False

        # Check coverage
        if not self.check_coverage():
            success = False

        return success


def main():
    """Main entry point for documentation building."""
    parser = argparse.ArgumentParser(description="Build DocsToKG documentation")
    parser.add_argument(
        "--format",
        "-f",
        choices=["html", "latex", "linkcheck", "coverage", "all", "quality"],
        default="html",
        help="Documentation format to build",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean build directory before building"
    )
    parser.add_argument("--source-dir", default="docs/build/sphinx", help="Sphinx source directory")
    parser.add_argument("--build-dir", default="docs/build/_build", help="Build output directory")

    args = parser.parse_args()

    # Create builder instance
    builder = DocsBuilder(args.source_dir, args.build_dir)

    # Check Sphinx installation
    if not builder.check_sphinx_installation():
        sys.exit(1)

    # Run the requested build
    success = False

    if args.format == "html":
        success = builder.build_html(clean=args.clean)
    elif args.format == "latex":
        success = builder.build_latex()
    elif args.format == "linkcheck":
        success = builder.check_links()
    elif args.format == "coverage":
        success = builder.check_coverage()
    elif args.format == "all":
        success = builder.build_all()
    elif args.format == "quality":
        success = builder.run_quality_checks()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
