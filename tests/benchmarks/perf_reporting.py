# === NAVMAP v1 ===
# {
#   "module": "tests.benchmarks.perf_reporting",
#   "purpose": "Performance reporting and documentation generation",
#   "sections": [
#     {"id": "perf-report", "name": "PerformanceReport", "anchor": "class-perf-report", "kind": "class"},
#     {"id": "report-formatter", "name": "ReportFormatter", "anchor": "class-report-formatter", "kind": "class"},
#     {"id": "doc-generator", "name": "DocumentationGenerator", "anchor": "class-doc-generator", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""
Performance reporting and documentation generation for Optimization 10 Phase 4.

Generates CLI reports, markdown documentation, and performance summaries
for dashboards and communication.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime


@dataclass
class BudgetStatus:
    """Status of a performance budget."""

    name: str
    threshold_ms: float
    current_ms: float
    status: str  # "pass", "warn", "fail"
    delta_pct: float


class PerformanceReport:
    """Generate comprehensive performance reports."""

    def __init__(self, baseline_data: dict[str, Any], current_data: dict[str, Any]):
        """
        Initialize performance report.

        Args:
            baseline_data: Baseline measurements
            current_data: Current measurements
        """
        self.baseline_data = baseline_data
        self.current_data = current_data
        self.timestamp = datetime.now().isoformat()
        self.budget_statuses: list[BudgetStatus] = []

    def add_budget_status(self, status: BudgetStatus) -> None:
        """Add budget status to report."""
        self.budget_statuses.append(status)

    def get_summary(self) -> dict[str, Any]:
        """
        Get report summary.

        Returns:
            Summary dict with pass/warn/fail counts
        """
        summary = {
            "timestamp": self.timestamp,
            "total_budgets": len(self.budget_statuses),
            "passed": sum(1 for s in self.budget_statuses if s.status == "pass"),
            "warnings": sum(1 for s in self.budget_statuses if s.status == "warn"),
            "failures": sum(1 for s in self.budget_statuses if s.status == "fail"),
            "budgets": [
                {
                    "name": s.name,
                    "threshold_ms": s.threshold_ms,
                    "current_ms": s.current_ms,
                    "delta_pct": s.delta_pct,
                    "status": s.status,
                }
                for s in self.budget_statuses
            ],
        }
        return summary

    def to_json(self) -> str:
        """Serialize report to JSON."""
        return json.dumps(self.get_summary(), indent=2)


class ReportFormatter:
    """Format performance reports for different outputs."""

    @staticmethod
    def format_table(report: PerformanceReport) -> str:
        """
        Format report as ASCII table.

        Args:
            report: PerformanceReport instance

        Returns:
            ASCII table string
        """
        lines = [
            "╔════════════════════════════════════════════════════════════╗",
            "║              PERFORMANCE BUDGET REPORT                    ║",
            "╠════════════════════════════════════════════════════════════╣",
        ]

        for status in report.budget_statuses:
            status_icon = (
                "✅" if status.status == "pass"
                else "⚠️ " if status.status == "warn"
                else "❌"
            )
            line = (
                f"║ {status_icon} {status.name:30s} "
                f"{status.current_ms:8.2f}ms "
                f"({status.delta_pct:+6.1f}%) ║"
            )
            lines.append(line)

        summary = report.get_summary()
        lines.extend([
            "╠════════════════════════════════════════════════════════════╣",
            f"║ PASSED: {summary['passed']:2d}  WARNED: {summary['warnings']:2d}  FAILED: {summary['failures']:2d}                       ║",
            "╚════════════════════════════════════════════════════════════╝",
        ])

        return "\n".join(lines)

    @staticmethod
    def format_markdown(report: PerformanceReport) -> str:
        """
        Format report as Markdown.

        Args:
            report: PerformanceReport instance

        Returns:
            Markdown string
        """
        lines = [
            "# Performance Budget Report\n",
            f"**Generated:** {report.timestamp}\n",
        ]

        summary = report.get_summary()
        lines.extend([
            "## Summary\n",
            f"- **Total Budgets:** {summary['total_budgets']}",
            f"- **Passed:** {summary['passed']} ✅",
            f"- **Warnings:** {summary['warnings']} ⚠️",
            f"- **Failures:** {summary['failures']} ❌\n",
        ])

        lines.append("## Budget Details\n")
        lines.append("| Benchmark | Threshold | Current | Delta | Status |")
        lines.append("|-----------|-----------|---------|-------|--------|")

        for status in report.budget_statuses:
            status_emoji = (
                "✅" if status.status == "pass"
                else "⚠️" if status.status == "warn"
                else "❌"
            )
            line = (
                f"| {status.name} | {status.threshold_ms:.1f}ms | "
                f"{status.current_ms:.2f}ms | {status.delta_pct:+.1f}% | {status_emoji} |"
            )
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def format_json(report: PerformanceReport) -> str:
        """Format report as JSON."""
        return report.to_json()


class DocumentationGenerator:
    """Generate performance documentation."""

    def __init__(self, output_dir: Path):
        """
        Initialize documentation generator.

        Args:
            output_dir: Directory for documentation output
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_budgets_doc(self, budgets: dict[str, dict[str, Any]]) -> Path:
        """
        Generate budgets documentation.

        Args:
            budgets: Dict of budget name -> details

        Returns:
            Path to generated documentation
        """
        lines = [
            "# Performance Budgets\n",
            "Performance budgets ensure critical paths stay within acceptable time limits.\n",
        ]

        for name, details in sorted(budgets.items()):
            lines.extend([
                f"## {name}\n",
                f"**Threshold:** {details.get('threshold_ms', 'N/A')} ms (p95)",
                f"**Description:** {details.get('description', 'N/A')}\n",
            ])

        doc_path = self.output_dir / "BUDGETS.md"
        doc_path.write_text("\n".join(lines))
        return doc_path

    def generate_profiling_guide(self) -> Path:
        """
        Generate profiling guide documentation.

        Returns:
            Path to generated documentation
        """
        lines = [
            "# Performance Profiling Guide\n",
            "## How to Profile Locally\n",
            "### CPU Profiling with pyinstrument\n",
            "```bash\n",
            "PYINSTRUMENT=1 pytest tests/benchmarks/bench_micro.py -k httpx --profile\n",
            "```\n",
            "### Memory Profiling with psutil\n",
            "```bash\n",
            "pytest tests/benchmarks/bench_micro.py --benchmark-only\n",
            "```\n",
            "### DuckDB Query Profiling\n",
            "```bash\n",
            "DUCKDB_EXPLAIN=1 pytest tests/benchmarks/bench_micro.py::test_bench_duckdb_query\n",
            "```\n",
            "### Regression Detection\n",
            "```bash\n",
            "pytest tests/benchmarks/ --benchmark-autosave\n",
            "```\n",
        ]

        doc_path = self.output_dir / "PROFILING_GUIDE.md"
        doc_path.write_text("\n".join(lines))
        return doc_path

    def generate_regression_guide(self) -> Path:
        """
        Generate regression detection guide.

        Returns:
            Path to generated documentation
        """
        lines = [
            "# Regression Detection Guide\n",
            "## CI Regression Behavior\n",
            "### PR Lane (Pull Requests)\n",
            "- **Threshold:** >15% slower than baseline → WARNING\n",
            "- **Action:** Notify in PR, continue\n",
            "- **Baseline:** Stored in `.ci/perf/baselines/`\n",
            "### Nightly Suite (Scheduled Runs)\n",
            "- **Threshold:** >20% slower than baseline → FAILURE\n",
            "- **Action:** Fail CI, halt merge\n",
            "- **Baseline:** Updated after approval\n",
            "## Updating Baselines\n",
            "```bash\n",
            "# After performance improvement or intentional change:\n",
            "./.venv/bin/pytest tests/benchmarks/ --benchmark-autosave\n",
            "git add .ci/perf/baselines/\n",
            "git commit -m 'Update performance baselines (reason)'\n",
            "```\n",
        ]

        doc_path = self.output_dir / "REGRESSION_GUIDE.md"
        doc_path.write_text("\n".join(lines))
        return doc_path

    def generate_summary(self, report: PerformanceReport) -> Path:
        """
        Generate performance summary report.

        Args:
            report: PerformanceReport instance

        Returns:
            Path to generated summary
        """
        summary_text = ReportFormatter.format_markdown(report)
        summary_path = self.output_dir / f"REPORT_{report.timestamp[:10]}.md"
        summary_path.write_text(summary_text)
        return summary_path
