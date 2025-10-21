"""Tests for Polars analytics pipelines."""

from __future__ import annotations

import pytest

try:  # pragma: no cover
    import polars as pl
except ImportError:  # pragma: no cover
    pytest.skip("polars not installed", allow_module_level=True)

from DocsToKG.OntologyDownload.analytics.pipelines import (
    LatestSummary,
    VersionDelta,
    arrow_to_lazy_frame,
    build_latest_summary_pipeline,
    build_version_delta_pipeline,
    compute_latest_summary,
    compute_version_delta,
    lazy_frame_to_arrow,
)


@pytest.fixture
def sample_files_df() -> pl.LazyFrame:
    """Create sample files dataframe."""
    data = {
        "file_id": ["f1", "f2", "f3", "f4", "f5"],
        "relpath": ["data/a.ttl", "data/b.rdf", "docs/c.txt", "data/d.owl", "other/e.jsonld"],
        "size": [1024, 2048, 512, 4096, 256],
        "format": ["ttl", "rdf", "txt", "owl", "jsonld"],
        "extracted_at": ["2025-01-01"] * 5,
    }
    return pl.DataFrame(data).lazy()


@pytest.fixture
def sample_validations_df() -> pl.LazyFrame:
    """Create sample validations dataframe."""
    data = {
        "validation_id": ["v1", "v2", "v3", "v4", "v5"],
        "file_id": ["f1", "f2", "f3", "f4", "f5"],
        "validator": ["rdflib", "owlready2", "rdflib", "owlready2", "rdflib"],
        "status": ["pass", "pass", "fail", "pass", "fail"],
    }
    return pl.DataFrame(data).lazy()


class TestLatestSummaryPipeline:
    """Test latest version summary pipeline."""

    def test_build_latest_summary_pipeline(self, sample_files_df: pl.LazyFrame) -> None:
        """Test pipeline construction."""
        pipeline = build_latest_summary_pipeline(sample_files_df)
        
        assert isinstance(pipeline, pl.LazyFrame)
        # Pipeline should include key columns
        collected = pipeline.collect()
        assert "relpath" in collected.columns
        assert "size" in collected.columns
        assert "format" in collected.columns

    def test_compute_latest_summary_basic(self, sample_files_df: pl.LazyFrame) -> None:
        """Test latest summary computation."""
        summary = compute_latest_summary(sample_files_df)
        
        assert isinstance(summary, LatestSummary)
        assert summary.total_files == 5
        assert summary.total_bytes == 1024 + 2048 + 512 + 4096 + 256

    def test_compute_latest_summary_format_breakdown(self, sample_files_df: pl.LazyFrame) -> None:
        """Test format-based aggregation."""
        summary = compute_latest_summary(sample_files_df)
        
        assert "ttl" in summary.files_by_format
        assert summary.files_by_format["ttl"] == 1
        assert "rdf" in summary.files_by_format
        assert summary.files_by_format["rdf"] == 1

    def test_compute_latest_summary_top_files(self, sample_files_df: pl.LazyFrame) -> None:
        """Test top N files identification."""
        summary = compute_latest_summary(sample_files_df, top_n=3)
        
        assert len(summary.top_files) == 3
        # First should be largest (4096)
        assert summary.top_files[0][1] == 4096

    def test_compute_latest_summary_with_validations(
        self, sample_files_df: pl.LazyFrame, sample_validations_df: pl.LazyFrame
    ) -> None:
        """Test validation summary integration."""
        summary = compute_latest_summary(sample_files_df, sample_validations_df)
        
        assert len(summary.validation_summary) > 0
        assert "pass" in summary.validation_summary
        assert "fail" in summary.validation_summary


class TestVersionDeltaPipeline:
    """Test version delta computation."""

    def test_build_version_delta_pipeline(self, sample_files_df: pl.LazyFrame) -> None:
        """Test delta pipeline construction."""
        # Create second version (subset of first)
        data_v2 = {
            "file_id": ["f1", "f2", "f6"],  # f6 is new
            "relpath": ["data/a.ttl", "data/b.rdf", "data/new.ttl"],
            "size": [1024, 2048, 1000],
            "format": ["ttl", "rdf", "ttl"],
            "extracted_at": ["2025-01-02"] * 3,
        }
        v2_df = pl.DataFrame(data_v2).lazy()

        added, removed, common = build_version_delta_pipeline(sample_files_df, v2_df)

        # Should return DataFrame or LazyFrame (after collection)
        assert isinstance(added, (pl.LazyFrame, pl.DataFrame))
        assert isinstance(removed, (pl.LazyFrame, pl.DataFrame))
        assert isinstance(common, (pl.LazyFrame, pl.DataFrame))

    def test_compute_version_delta_additions(self, sample_files_df: pl.LazyFrame) -> None:
        """Test detection of added files."""
        # v2 has one extra file
        data_v2 = {
            "file_id": ["f1", "f2", "f3", "f6"],  # f6 is new
            "relpath": ["data/a.ttl", "data/b.rdf", "docs/c.txt", "data/new.ttl"],
            "size": [1024, 2048, 512, 1000],
            "format": ["ttl", "rdf", "txt", "ttl"],
            "extracted_at": ["2025-01-02"] * 4,
        }
        v2_df = pl.DataFrame(data_v2).lazy()

        delta = compute_version_delta(sample_files_df, v2_df)

        assert delta.added_files == 1
        assert delta.added_bytes == 1000

    def test_compute_version_delta_removals(self, sample_files_df: pl.LazyFrame) -> None:
        """Test detection of removed files."""
        # v2 is subset of v1
        data_v2 = {
            "file_id": ["f1", "f2"],
            "relpath": ["data/a.ttl", "data/b.rdf"],
            "size": [1024, 2048],
            "format": ["ttl", "rdf"],
            "extracted_at": ["2025-01-02"] * 2,
        }
        v2_df = pl.DataFrame(data_v2).lazy()

        delta = compute_version_delta(sample_files_df, v2_df)

        assert delta.removed_files == 3
        assert delta.removed_bytes == 512 + 4096 + 256

    def test_compute_version_delta_net_bytes(self, sample_files_df: pl.LazyFrame) -> None:
        """Test net bytes delta calculation."""
        # v2: add 1000, remove 2560 = net -1560
        data_v2 = {
            "file_id": ["f1", "f6"],  # Removed f2 (2048) and others, added f6
            "relpath": ["data/a.ttl", "data/new.ttl"],
            "size": [1024, 1000],
            "format": ["ttl", "ttl"],
            "extracted_at": ["2025-01-02"] * 2,
        }
        v2_df = pl.DataFrame(data_v2).lazy()

        delta = compute_version_delta(sample_files_df, v2_df)

        # Added: f6 (1000), Removed: f2, f3, f4, f5 (2048+512+4096+256)
        assert delta.added_bytes == 1000
        assert delta.removed_bytes == 2048 + 512 + 4096 + 256


class TestPipelineInterop:
    """Test Arrow/Polars interop."""

    def test_arrow_to_lazy_frame(self) -> None:
        """Test Arrow to LazyFrame conversion."""
        # Create a Polars DF, convert to Arrow, then back to LazyFrame
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        arrow = df.to_arrow()
        
        lazy = arrow_to_lazy_frame(arrow)
        
        assert isinstance(lazy, pl.LazyFrame)
        collected = lazy.collect()
        assert collected.height == 3
        assert "a" in collected.columns

    def test_lazy_frame_to_arrow(self) -> None:
        """Test LazyFrame to Arrow conversion."""
        lf = pl.DataFrame({"x": [1, 2, 3]}).lazy()
        
        arrow = lazy_frame_to_arrow(lf)
        
        assert arrow.num_rows == 3
        assert "x" in arrow.column_names
