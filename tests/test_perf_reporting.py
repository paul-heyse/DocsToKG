# === NAVMAP v1 ===
# {
#   "module": "tests.test_perf_reporting",
#   "purpose": "Tests for performance reporting and documentation generation",
#   "sections": [
#     {"id": "report-tests", "name": "Report Tests", "anchor": "report-tests", "kind": "section"},
#     {"id": "formatter-tests", "name": "Formatter Tests", "anchor": "formatter-tests", "kind": "section"},
#     {"id": "doc-generator-tests", "name": "Documentation Tests", "anchor": "doc-generator-tests", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Tests for performance reporting and documentation generation.

Tests report generation, formatting, and documentation output.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from tests.benchmarks.perf_reporting import (
    BudgetStatus,
    PerformanceReport,
    ReportFormatter,
    DocumentationGenerator,
)


# --- PerformanceReport Tests ---


@pytest.mark.unit
def test_performance_report_init():
    """Test performance report initialization."""
    baseline = {"httpx": 3.5}
    current = {"httpx": 3.6}

    report = PerformanceReport(baseline, current)

    assert report.baseline_data == baseline
    assert report.current_data == current
    assert report.timestamp is not None


@pytest.mark.unit
def test_performance_report_add_status():
    """Test adding budget status to report."""
    report = PerformanceReport({}, {})

    status = BudgetStatus(
        name="HTTPX GET",
        threshold_ms=5.0,
        current_ms=3.5,
        status="pass",
        delta_pct=-30.0,
    )

    report.add_budget_status(status)

    assert len(report.budget_statuses) == 1
    assert report.budget_statuses[0].name == "HTTPX GET"


@pytest.mark.unit
def test_performance_report_summary():
    """Test report summary generation."""
    report = PerformanceReport({}, {})

    # Add mix of statuses
    report.add_budget_status(
        BudgetStatus("bench1", 100.0, 90.0, "pass", -10.0)
    )
    report.add_budget_status(
        BudgetStatus("bench2", 100.0, 115.0, "warn", 15.0)
    )
    report.add_budget_status(
        BudgetStatus("bench3", 100.0, 125.0, "fail", 25.0)
    )

    summary = report.get_summary()

    assert summary["total_budgets"] == 3
    assert summary["passed"] == 1
    assert summary["warnings"] == 1
    assert summary["failures"] == 1


@pytest.mark.unit
def test_performance_report_to_json():
    """Test report serialization to JSON."""
    report = PerformanceReport({}, {})
    report.add_budget_status(
        BudgetStatus("test", 100.0, 95.0, "pass", -5.0)
    )

    json_str = report.to_json()

    assert "test" in json_str
    assert "pass" in json_str
    assert "timestamp" in json_str


# --- ReportFormatter Tests ---


@pytest.mark.unit
def test_format_table():
    """Test ASCII table formatting."""
    report = PerformanceReport({}, {})
    report.add_budget_status(
        BudgetStatus("HTTPX", 5.0, 3.5, "pass", -30.0)
    )
    report.add_budget_status(
        BudgetStatus("DuckDB", 200.0, 250.0, "warn", 25.0)
    )

    table = ReportFormatter.format_table(report)

    assert "PERFORMANCE BUDGET REPORT" in table
    assert "HTTPX" in table
    assert "DuckDB" in table
    assert "✅" in table or "⚠️" in table


@pytest.mark.unit
def test_format_markdown():
    """Test Markdown formatting."""
    report = PerformanceReport({}, {})
    report.add_budget_status(
        BudgetStatus("HTTPX", 5.0, 3.5, "pass", -30.0)
    )
    report.add_budget_status(
        BudgetStatus("DuckDB", 200.0, 250.0, "warn", 25.0)
    )

    markdown = ReportFormatter.format_markdown(report)

    assert "# Performance Budget Report" in markdown
    assert "**Passed:** 1" in markdown
    assert "**Warnings:** 1" in markdown
    assert "| Benchmark |" in markdown
    assert "HTTPX" in markdown


@pytest.mark.unit
def test_format_json():
    """Test JSON formatting."""
    report = PerformanceReport({}, {})
    report.add_budget_status(
        BudgetStatus("test", 100.0, 95.0, "pass", -5.0)
    )

    json_str = ReportFormatter.format_json(report)

    assert "test" in json_str
    assert "pass" in json_str


# --- DocumentationGenerator Tests ---


@pytest.mark.unit
def test_doc_generator_init(tmp_path):
    """Test documentation generator initialization."""
    gen = DocumentationGenerator(tmp_path)

    assert gen.output_dir.exists()


@pytest.mark.unit
def test_generate_budgets_doc(tmp_path):
    """Test budgets documentation generation."""
    gen = DocumentationGenerator(tmp_path)

    budgets = {
        "HTTPX": {
            "threshold_ms": 5.0,
            "description": "GET 200 with 128 KiB body",
        },
        "DuckDB": {
            "threshold_ms": 200.0,
            "description": "Query on 200k rows",
        },
    }

    doc_path = gen.generate_budgets_doc(budgets)

    assert doc_path.exists()
    content = doc_path.read_text()
    assert "# Performance Budgets" in content
    assert "HTTPX" in content
    assert "DuckDB" in content


@pytest.mark.unit
def test_generate_profiling_guide(tmp_path):
    """Test profiling guide generation."""
    gen = DocumentationGenerator(tmp_path)

    doc_path = gen.generate_profiling_guide()

    assert doc_path.exists()
    content = doc_path.read_text()
    assert "# Performance Profiling Guide" in content
    assert "pyinstrument" in content or "CPU Profiling" in content


@pytest.mark.unit
def test_generate_regression_guide(tmp_path):
    """Test regression guide generation."""
    gen = DocumentationGenerator(tmp_path)

    doc_path = gen.generate_regression_guide()

    assert doc_path.exists()
    content = doc_path.read_text()
    assert "# Regression Detection Guide" in content
    assert "PR Lane" in content
    assert "Nightly Suite" in content


@pytest.mark.unit
def test_generate_summary(tmp_path):
    """Test summary report generation."""
    gen = DocumentationGenerator(tmp_path)

    report = PerformanceReport({}, {})
    report.add_budget_status(
        BudgetStatus("test", 100.0, 95.0, "pass", -5.0)
    )

    summary_path = gen.generate_summary(report)

    assert summary_path.exists()
    content = summary_path.read_text()
    assert "# Performance Budget Report" in content


# --- Integration Tests ---


@pytest.mark.component
def test_reporting_full_workflow(tmp_path):
    """Test full reporting workflow."""
    # Create report with sample data
    report = PerformanceReport({}, {})

    report.add_budget_status(
        BudgetStatus("HTTPX GET", 5.0, 3.5, "pass", -30.0)
    )
    report.add_budget_status(
        BudgetStatus("DuckDB Query", 200.0, 195.0, "pass", -2.5)
    )
    report.add_budget_status(
        BudgetStatus("Polars Pipeline", 2000.0, 2100.0, "warn", 5.0)
    )

    # Format as multiple outputs
    table = ReportFormatter.format_table(report)
    markdown = ReportFormatter.format_markdown(report)
    json_str = ReportFormatter.format_json(report)

    # Verify all formats produced
    assert "HTTPX GET" in table
    assert "HTTPX GET" in markdown
    assert "HTTPX GET" in json_str

    # Generate documentation
    gen = DocumentationGenerator(tmp_path)
    summary_path = gen.generate_summary(report)

    assert summary_path.exists()

    # Verify docs generated
    budgets_doc = gen.generate_budgets_doc(
        {
            "HTTPX": {"threshold_ms": 5.0, "description": "GET 200"},
            "DuckDB": {"threshold_ms": 200.0, "description": "Query"},
        }
    )
    profiling_doc = gen.generate_profiling_guide()
    regression_doc = gen.generate_regression_guide()

    assert budgets_doc.exists()
    assert profiling_doc.exists()
    assert regression_doc.exists()
