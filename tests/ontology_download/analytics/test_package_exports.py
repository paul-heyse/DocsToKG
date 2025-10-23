"""Smoke tests for OntologyDownload analytics package exports."""

from DocsToKG.OntologyDownload.analytics import (
    LatestSummary,
    cmd_report_latest,
    generate_latest_report,
)


def test_symbols_are_importable() -> None:
    """Ensure key analytics exports remain accessible."""

    assert LatestSummary.__name__ == "LatestSummary"
    assert callable(generate_latest_report)
    assert callable(cmd_report_latest)
