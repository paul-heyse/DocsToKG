"""Tests for analytics reports."""

from __future__ import annotations

import pytest

try:  # pragma: no cover
    import polars as pl
except ImportError:  # pragma: no cover
    pytest.skip("polars not installed", allow_module_level=True)

from DocsToKG.OntologyDownload.analytics.reports import (
    LatestVersionReport,
    ValidationReport,
    generate_growth_report,
    generate_latest_report,
    generate_validation_report,
    report_to_dict,
    report_to_table,
)


@pytest.fixture
def sample_files() -> pl.DataFrame:
    """Create sample files data."""
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
    """Create sample validations data."""
    return pl.DataFrame(
        {
            "validation_id": ["v1", "v2", "v3", "v4", "v5"],
            "file_id": ["f1", "f2", "f3", "f4", "f5"],
            "validator": ["rdflib", "owlready2", "rdflib", "owlready2", "rdflib"],
            "status": ["pass", "pass", "fail", "pass", "fail"],
        }
    )


class TestLatestVersionReport:
    """Test latest version report generation."""

    def test_generate_latest_report_basic(self, sample_files: pl.DataFrame) -> None:
        """Test basic latest report."""
        report = generate_latest_report(sample_files)

        assert isinstance(report, LatestVersionReport)
        assert report.total_files == 5
        assert report.total_bytes == 1024 + 2048 + 512 + 4096 + 256

    def test_generate_latest_report_formats(self, sample_files: pl.DataFrame) -> None:
        """Test format distribution in report."""
        report = generate_latest_report(sample_files)

        assert "ttl" in report.formats
        assert "rdf" in report.formats
        assert report.formats["ttl"] == 1

    def test_generate_latest_report_top_files(self, sample_files: pl.DataFrame) -> None:
        """Test top files in report."""
        report = generate_latest_report(sample_files)

        assert len(report.largest_files) >= 1
        # Largest file is 4096
        assert report.largest_files[0][1] == 4096

    def test_generate_latest_report_with_validations(
        self, sample_files: pl.DataFrame, sample_validations: pl.DataFrame
    ) -> None:
        """Test report with validation data."""
        report = generate_latest_report(sample_files, sample_validations)

        assert report.validation_pass_rate == 0.6  # 3 pass out of 5


class TestGrowthReport:
    """Test growth report generation."""

    def test_generate_growth_report_additions(self, sample_files: pl.DataFrame) -> None:
        """Test growth report with additions."""
        # v2 has one extra file
        v2_data = {
            "file_id": ["f1", "f2", "f3", "f6"],
            "relpath": ["a.ttl", "b.rdf", "c.txt", "f.xml"],
            "size": [1024, 2048, 512, 2000],
            "format": ["ttl", "rdf", "txt", "xml"],
        }
        v2_files = pl.DataFrame(v2_data)

        report = generate_growth_report(sample_files, v2_files, "v1", "v2")

        assert report.files_added == 1
        assert report.bytes_added == 2000

    def test_generate_growth_report_removals(self, sample_files: pl.DataFrame) -> None:
        """Test growth report with removals."""
        # v2 is subset of v1
        v2_data = {
            "file_id": ["f1", "f2"],
            "relpath": ["a.ttl", "b.rdf"],
            "size": [1024, 2048],
            "format": ["ttl", "rdf"],
        }
        v2_files = pl.DataFrame(v2_data)

        report = generate_growth_report(sample_files, v2_files, "v1", "v2")

        assert report.files_removed == 3

    def test_generate_growth_report_net_growth(self, sample_files: pl.DataFrame) -> None:
        """Test net growth calculation."""
        v2_data = {
            "file_id": ["f1", "f6"],
            "relpath": ["a.ttl", "f.xml"],
            "size": [1024, 5000],
            "format": ["ttl", "xml"],
        }
        v2_files = pl.DataFrame(v2_data)

        report = generate_growth_report(sample_files, v2_files, "v1", "v2")

        # Added: f6 (5000), Removed: f2, f3, f4, f5 (2048+512+4096+256=6912)
        expected_net = 5000 - 6912
        assert report.net_growth == expected_net


class TestValidationReport:
    """Test validation report generation."""

    def test_generate_validation_report_basic(self, sample_validations: pl.DataFrame) -> None:
        """Test basic validation report."""
        report = generate_validation_report("v1", sample_validations)

        assert isinstance(report, ValidationReport)
        assert report.total_validations == 5
        assert report.pass_count == 3
        assert report.fail_count == 2

    def test_generate_validation_report_pass_rate(self, sample_validations: pl.DataFrame) -> None:
        """Test pass rate calculation."""
        report = generate_validation_report("v1", sample_validations)

        assert report.pass_rate == 0.6


class TestReportExport:
    """Test report export functions."""

    def test_report_to_dict(self, sample_files: pl.DataFrame) -> None:
        """Test report to dictionary conversion."""
        report = generate_latest_report(sample_files)
        d = report_to_dict(report)

        assert isinstance(d, dict)
        assert "total_files" in d
        assert d["total_files"] == 5

    def test_report_to_table(self, sample_files: pl.DataFrame) -> None:
        """Test report to table string conversion."""
        report = generate_latest_report(sample_files)
        table_str = report_to_table(report)

        assert isinstance(table_str, str)
        assert "LatestVersionReport" in table_str
        assert "total_files" in table_str
