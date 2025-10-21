"""Tests for CLI commands and formatters."""

from __future__ import annotations

import json

import pytest

try:  # pragma: no cover
    import polars as pl
except ImportError:  # pragma: no cover
    pytest.skip("polars not installed", allow_module_level=True)

from DocsToKG.OntologyDownload.analytics.cli_commands import (
    cmd_report_growth,
    cmd_report_latest,
    cmd_report_validation,
    format_growth_report,
    format_latest_report,
    format_validation_report,
)
from DocsToKG.OntologyDownload.analytics.reports import (
    generate_growth_report,
    generate_latest_report,
    generate_validation_report,
)


@pytest.fixture
def sample_files() -> pl.DataFrame:
    """Create sample files."""
    return pl.DataFrame(
        {
            "file_id": ["f1", "f2", "f3", "f4", "f5"],
            "relpath": ["a.ttl", "b.rdf", "c.txt", "d.owl", "e.jsonld"],
            "size": [1024, 2048, 512, 4096, 256],
            "format": ["ttl", "rdf", "txt", "owl", "jsonld"],
        }
    )


@pytest.fixture
def sample_validations() -> pl.DataFrame:
    """Create sample validations."""
    return pl.DataFrame(
        {
            "validation_id": ["v1", "v2", "v3", "v4", "v5"],
            "file_id": ["f1", "f2", "f3", "f4", "f5"],
            "validator": ["rdflib", "owlready2", "rdflib", "owlready2", "rdflib"],
            "status": ["pass", "pass", "fail", "pass", "fail"],
        }
    )


class TestLatestReportCommand:
    """Test latest report CLI command."""

    def test_cmd_report_latest_table(self, sample_files: pl.DataFrame) -> None:
        """Test latest report with table format."""
        result = cmd_report_latest(sample_files, output_format="table")

        assert isinstance(result, str)
        assert "LatestVersionReport" in result
        assert "Total Files" in result  # Formatted key

    def test_cmd_report_latest_json(self, sample_files: pl.DataFrame) -> None:
        """Test latest report with JSON format."""
        result = cmd_report_latest(sample_files, output_format="json")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "total_files" in data
        assert data["total_files"] == 5

    def test_cmd_report_latest_csv(self, sample_files: pl.DataFrame) -> None:
        """Test latest report with CSV format."""
        result = cmd_report_latest(sample_files, output_format="csv")

        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 2  # Header + data row


class TestGrowthReportCommand:
    """Test growth report CLI command."""

    def test_cmd_report_growth_table(self, sample_files: pl.DataFrame) -> None:
        """Test growth report with table format."""
        v2_data = {
            "file_id": ["f1", "f2", "f3", "f6"],
            "relpath": ["a.ttl", "b.rdf", "c.txt", "f.xml"],
            "size": [1024, 2048, 512, 2000],
            "format": ["ttl", "rdf", "txt", "xml"],
        }
        v2_files = pl.DataFrame(v2_data)

        result = cmd_report_growth(sample_files, v2_files, "v1", "v2", output_format="table")

        assert isinstance(result, str)
        assert "GrowthReport" in result
        assert "From Version" in result  # Formatted key

    def test_cmd_report_growth_json(self, sample_files: pl.DataFrame) -> None:
        """Test growth report with JSON format."""
        v2_data = {
            "file_id": ["f1", "f2"],
            "relpath": ["a.ttl", "b.rdf"],
            "size": [1024, 2048],
            "format": ["ttl", "rdf"],
        }
        v2_files = pl.DataFrame(v2_data)

        result = cmd_report_growth(sample_files, v2_files, "v1", "v2", output_format="json")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "from_version" in data
        assert data["from_version"] == "v1"

    def test_cmd_report_growth_csv(self, sample_files: pl.DataFrame) -> None:
        """Test growth report with CSV format."""
        v2_data = {
            "file_id": ["f1", "f2"],
            "relpath": ["a.ttl", "b.rdf"],
            "size": [1024, 2048],
            "format": ["ttl", "rdf"],
        }
        v2_files = pl.DataFrame(v2_data)

        result = cmd_report_growth(sample_files, v2_files, "v1", "v2", output_format="csv")

        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 2


class TestValidationReportCommand:
    """Test validation report CLI command."""

    def test_cmd_report_validation_table(self, sample_validations: pl.DataFrame) -> None:
        """Test validation report with table format."""
        result = cmd_report_validation("v1", sample_validations, output_format="table")

        assert isinstance(result, str)
        assert "ValidationReport" in result
        assert "Pass Rate" in result  # Formatted key

    def test_cmd_report_validation_json(self, sample_validations: pl.DataFrame) -> None:
        """Test validation report with JSON format."""
        result = cmd_report_validation("v1", sample_validations, output_format="json")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "pass_count" in data
        assert data["pass_count"] == 3

    def test_cmd_report_validation_csv(self, sample_validations: pl.DataFrame) -> None:
        """Test validation report with CSV format."""
        result = cmd_report_validation("v1", sample_validations, output_format="csv")

        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 2


class TestFormatters:
    """Test report formatters."""

    def test_format_latest_report_table(self, sample_files: pl.DataFrame) -> None:
        """Test formatting latest report as table."""
        report = generate_latest_report(sample_files)
        result = format_latest_report(report, "table")

        assert isinstance(result, str)
        assert "LatestVersionReport" in result

    def test_format_latest_report_json(self, sample_files: pl.DataFrame) -> None:
        """Test formatting latest report as JSON."""
        report = generate_latest_report(sample_files)
        result = format_latest_report(report, "json")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "total_files" in data

    def test_format_growth_report_table(self, sample_files: pl.DataFrame) -> None:
        """Test formatting growth report as table."""
        v2_files = pl.DataFrame(
            {
                "file_id": ["f1", "f2"],
                "relpath": ["a.ttl", "b.rdf"],
                "size": [1024, 2048],
                "format": ["ttl", "rdf"],
            }
        )
        report = generate_growth_report(sample_files, v2_files, "v1", "v2")
        result = format_growth_report(report, "table")

        assert isinstance(result, str)
        assert "GrowthReport" in result

    def test_format_growth_report_json(self, sample_files: pl.DataFrame) -> None:
        """Test formatting growth report as JSON."""
        v2_files = pl.DataFrame(
            {
                "file_id": ["f1", "f2"],
                "relpath": ["a.ttl", "b.rdf"],
                "size": [1024, 2048],
                "format": ["ttl", "rdf"],
            }
        )
        report = generate_growth_report(sample_files, v2_files, "v1", "v2")
        result = format_growth_report(report, "json")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "from_version" in data

    def test_format_validation_report_table(self, sample_validations: pl.DataFrame) -> None:
        """Test formatting validation report as table."""
        report = generate_validation_report("v1", sample_validations)
        result = format_validation_report(report, "table")

        assert isinstance(result, str)
        assert "ValidationReport" in result

    def test_format_validation_report_json(self, sample_validations: pl.DataFrame) -> None:
        """Test formatting validation report as JSON."""
        report = generate_validation_report("v1", sample_validations)
        result = format_validation_report(report, "json")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "pass_count" in data

    def test_format_validation_report_csv(self, sample_validations: pl.DataFrame) -> None:
        """Test formatting validation report as CSV."""
        report = generate_validation_report("v1", sample_validations)
        result = format_validation_report(report, "csv")

        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 2


class TestFormatterEdgeCases:
    """Test formatter edge cases."""

    def test_json_format_with_large_numbers(self, sample_files: pl.DataFrame) -> None:
        """Test JSON format with large numbers."""
        report = generate_latest_report(sample_files)
        result = format_latest_report(report, "json")

        data = json.loads(result)
        assert isinstance(data["total_bytes"], int)

    def test_table_format_alignment(self, sample_files: pl.DataFrame) -> None:
        """Test table format alignment."""
        report = generate_latest_report(sample_files)
        result = format_latest_report(report, "table")

        # Check for consistent formatting
        lines = result.split("\n")
        assert len(lines) >= 3  # Header + separator + data
