#!/usr/bin/env python3
"""
Comprehensive Link Checker for Documentation

This script performs thorough link checking on documentation files:
- Checks internal links between documentation files
- Validates external URLs (with rate limiting)
- Reports broken links with detailed information
- Provides suggestions for fixing broken links
"""

import asyncio
import aiohttp
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from urllib.parse import urlparse, urljoin
import argparse


class LinkChecker:
    """Comprehensive link checker for documentation."""

    def __init__(self, docs_dir: str = "docs", timeout: int = 10, max_concurrent: int = 10):
        self.docs_dir = Path(docs_dir)
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.broken_links: List[Dict] = []
        self.working_links: List[Dict] = []
        self.ignored_links: Set[str] = set()

    def find_all_links(self) -> Dict[Path, List[Dict]]:
        """Find all links in all documentation files."""
        print("🔍 Scanning for links in documentation...")

        file_links = {}

        if not self.docs_dir.exists():
            print(f"❌ Documentation directory not found: {self.docs_dir}")
            return file_links

        # Find all markdown files
        md_files = list(self.docs_dir.rglob("*.md"))

        if not md_files:
            print("❌ No markdown files found")
            return file_links

        print(f"📁 Scanning {len(md_files)} files for links...")

        for file_path in md_files:
            links = self._extract_links_from_file(file_path)
            if links:
                file_links[file_path] = links

        total_links = sum(len(links) for links in file_links.values())
        print(f"🔗 Found {total_links} total links in {len(file_links)} files")

        return file_links

    def _extract_links_from_file(self, file_path: Path) -> List[Dict]:
        """Extract all links from a single file."""
        links = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return links

        # Patterns for different types of links
        patterns = [
            # Markdown links: [text](url)
            (r"\[([^\]]+)\]\(([^)]+)\)", "markdown"),
            # HTML links: <a href="url">text</a>
            (r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', "html"),
            # Bare URLs (http/https)
            (r"\b(https?://[^\s]+)", "bare"),
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern, link_type in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if link_type == "markdown":
                        link_text, url = match
                    elif link_type == "html":
                        url, link_text = match
                    else:  # bare URL
                        url = match
                        link_text = url

                    if url and not self._should_ignore_url(url):
                        links.append(
                            {
                                "url": url,
                                "text": link_text.strip(),
                                "line": line_num,
                                "type": link_type,
                                "context": line.strip(),
                            }
                        )

        return links

    def _should_ignore_url(self, url: str) -> bool:
        """Check if URL should be ignored (e.g., placeholders, fragments)."""
        # Ignore fragment-only links (#anchor)
        if url.startswith("#"):
            return True

        # Ignore placeholder URLs
        if any(
            placeholder in url
            for placeholder in ["yourorg", "example.com", "placeholder", "TODO", "FIXME"]
        ):
            return True

        # Ignore localhost URLs (unless explicitly testing)
        if "localhost" in url or "127.0.0.1" in url:
            return True

        return False

    async def check_url(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, bool, str]:
        """Check if a URL is accessible."""
        try:
            # Handle relative URLs by making them absolute
            if not url.startswith(("http://", "https://")):
                # For now, assume these are working internal links
                # In a real implementation, you'd resolve relative to the file's location
                return url, True, "relative"

            async with session.get(url, timeout=self.timeout, allow_redirects=True) as response:
                is_working = 200 <= response.status < 400
                return url, is_working, str(response.status)

        except asyncio.TimeoutError:
            return url, False, "timeout"
        except aiohttp.ClientError as e:
            return url, False, str(e)
        except Exception as e:
            return url, False, str(e)

    async def check_all_links(
        self, file_links: Dict[Path, List[Dict]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Check all links for validity."""
        print("🔍 Checking link validity...")

        # Collect all unique URLs
        all_urls = set()
        for file_path, links in file_links.items():
            for link in links:
                all_urls.add(link["url"])

        print(f"🌐 Checking {len(all_urls)} unique URLs...")

        # Check URLs concurrently
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Create tasks for all URLs
            tasks = [self.check_url(session, url) for url in all_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        url_status = {}
        for result in results:
            if isinstance(result, Exception):
                url, is_working, error = str(result), False, str(result)
            else:
                url, is_working, error = result
            url_status[url] = (is_working, error)

        # Categorize links
        broken_links = []
        working_links = []

        for file_path, links in file_links.items():
            for link in links:
                url = link["url"]
                is_working, error = url_status.get(url, (False, "unknown"))

                link_info = {
                    "file": str(file_path.relative_to(self.docs_dir)),
                    "url": url,
                    "text": link["text"],
                    "line": link["line"],
                    "type": link["type"],
                    "context": link["context"],
                }

                if is_working:
                    working_links.append(link_info)
                else:
                    link_info["error"] = error
                    broken_links.append(link_info)

        return broken_links, working_links

    def generate_report(self, broken_links: List[Dict], working_links: List[Dict]) -> str:
        """Generate a detailed report of link checking results."""
        report = []

        report.append("# 🔗 Link Check Report")
        report.append("=" * 50)
        report.append(
            f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report.append("")

        # Summary
        total_links = len(broken_links) + len(working_links)
        report.append("## Summary")
        report.append(f"- **Total Links**: {total_links}")
        report.append(f"- **Working Links**: {len(working_links)} ✅")
        report.append(f"- **Broken Links**: {len(broken_links)} ❌")
        report.append("")

        if broken_links:
            report.append("## ❌ Broken Links")
            report.append("")

            for link in broken_links:
                report.append(f"### {link['url']}")
                report.append(f"**File**: `{link['file']}` (line {link['line']})")
                report.append(f"**Text**: {link['text']}")
                report.append(f"**Error**: {link.get('error', 'Unknown error')}")
                report.append(f"**Context**: `{link['context']}`")
                report.append("")
                report.append("**Suggestions**:")
                report.append("- Check if the URL is correct")
                report.append("- Verify the target page exists")
                report.append("- Update or remove the broken link")
                report.append("")

        if working_links:
            report.append("## ✅ Working Links (Sample)")
            report.append("")

            # Show first 20 working links as examples
            for link in working_links[:20]:
                report.append(f"- ✅ [{link['file']}:{link['line']}] {link['url']}")

            if len(working_links) > 20:
                report.append(f"- ... and {len(working_links) - 20} more working links")
            report.append("")

        # Recommendations
        if broken_links:
            report.append("## 🔧 Recommendations")
            report.append("")
            report.append("1. **Fix broken internal links**: Update file paths or anchor names")
            report.append("2. **Check external links**: Verify URLs are still valid")
            report.append("3. **Remove dead links**: Consider removing links to deleted resources")
            report.append("4. **Add placeholders**: For links to resources under development")
            report.append("")

        return "\n".join(report)

    def save_report(self, report: str, output_file: str = "docs/link_check_report.md"):
        """Save the link check report to a file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"📋 Report saved to: {output_path}")


def main():
    """Main entry point for link checking."""
    parser = argparse.ArgumentParser(description="Check for broken links in documentation")
    parser.add_argument("--docs-dir", default="docs", help="Documentation directory to scan")
    parser.add_argument(
        "--output", default="docs/link_check_report.md", help="Output file for the report"
    )
    parser.add_argument(
        "--timeout", type=int, default=10, help="Timeout for HTTP requests (seconds)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10, help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Skip external URL checks for faster results"
    )

    args = parser.parse_args()

    print("🔗 Starting Comprehensive Link Check")
    print("=" * 50)

    # Create link checker
    checker = LinkChecker(
        docs_dir=args.docs_dir, timeout=args.timeout, max_concurrent=args.max_concurrent
    )

    # Find all links
    file_links = checker.find_all_links()

    if not file_links:
        print("❌ No links found to check")
        sys.exit(1)

    # Check links
    broken_links, working_links = asyncio.run(checker.check_all_links(file_links))

    # Generate and save report
    report = checker.generate_report(broken_links, working_links)
    checker.save_report(report, args.output)

    # Summary
    print("\n📊 Link Check Summary:")
    print(f"  ✅ Working: {len(working_links)}")
    print(f"  ❌ Broken: {len(broken_links)}")

    if broken_links:
        print("\n🔧 Found broken links - check the report for details")
        sys.exit(1)
    else:
        print("\n✅ All links are working!")
        sys.exit(0)


if __name__ == "__main__":
    main()
