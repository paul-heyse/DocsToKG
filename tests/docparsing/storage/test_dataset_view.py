"""Tests for DatasetView (lazy dataset inspection and summarization)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pytest

from DocsToKG.DocParsing.storage.dataset_view import (
    DatasetSummary,
    _extract_doc_id_from_filename,
    _extract_partition_from_path,
    open_chunks,
    open_vectors,
    summarize,
)
from DocsToKG.DocParsing.storage.chunks_writer import ParquetChunksWriter


class TestExtractPartitionFromPath:
    """Tests for partition extraction from file paths."""

    def test_extract_partition_chunks_path(self):
        """Test extraction from Chunks path."""
        path = "/data/Chunks/fmt=parquet/2025/10/doc.parquet"
        partition = _extract_partition_from_path(path)
        assert partition == "2025-10"

    def test_extract_partition_vectors_path(self):
        """Test extraction from Vectors path."""
        path = "/data/Vectors/family=dense/fmt=parquet/2025/10/doc.parquet"
        partition = _extract_partition_from_path(path)
        assert partition == "2025-10"

    def test_extract_partition_invalid_path(self):
        """Test extraction from invalid path returns None."""
        path = "/data/invalid/structure/doc.parquet"
        partition = _extract_partition_from_path(path)
        assert partition is None


class TestExtractDocIdFromFilename:
    """Tests for doc_id extraction from filenames."""

    def test_extract_doc_id_simple(self):
        """Test extraction from simple filename."""
        filename = "doc123.parquet"
        doc_id = _extract_doc_id_from_filename(filename)
        assert doc_id == "doc123"

    def test_extract_doc_id_nested(self):
        """Test extraction from nested doc_id."""
        filename = "papers_ai_ml_1.parquet"
        doc_id = _extract_doc_id_from_filename(filename)
        assert doc_id == "papers_ai_ml_1"

    def test_extract_doc_id_invalid_extension(self):
        """Test extraction from non-parquet file returns None."""
        filename = "doc.txt"
        doc_id = _extract_doc_id_from_filename(filename)
        assert doc_id is None


class TestDatasetSummary:
    """Tests for DatasetSummary dataclass."""

    def test_summary_creation(self):
        """Test DatasetSummary creation."""
        schema = pa.schema([("doc_id", pa.string()), ("text", pa.string())])
        summary = DatasetSummary(
            dataset_type="chunks",
            schema=schema,
            file_count=5,
            total_bytes=1024000,
            approx_rows=1000,
            partitions={"2025-10": 3, "2025-11": 2},
            sample_doc_ids=["doc1", "doc2"],
        )

        assert summary.dataset_type == "chunks"
        assert summary.file_count == 5
        assert summary.approx_rows == 1000

    def test_summary_to_dict(self):
        """Test DatasetSummary.to_dict() conversion."""
        schema = pa.schema([("doc_id", pa.string())])
        summary = DatasetSummary(
            dataset_type="dense",
            schema=schema,
            file_count=2,
            total_bytes=500000,
            approx_rows=500,
            partitions={"2025-10": 2},
            sample_doc_ids=["doc1"],
        )

        summary_dict = summary.to_dict()
        assert isinstance(summary_dict, dict)
        assert summary_dict["dataset_type"] == "dense"
        assert summary_dict["file_count"] == 2
        assert summary_dict["approx_rows"] == 500


class TestOpenChunks:
    """Tests for open_chunks function."""

    def test_open_chunks_success(self):
        """Test opening chunks dataset successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            writer = ParquetChunksWriter()

            rows = [
                {
                    "doc_id": "doc1",
                    "chunk_id": 0,
                    "text": "Content",
                    "tokens": 1,
                    "span": {"start": 0, "end": 7},
                    "created_at": datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
                    "schema_version": "docparse/chunks/1.0.0",
                },
            ]

            result = writer.write(
                rows,
                data_root=temp_root,
                rel_id="doc1",
                cfg_hash="cfg123",
                dt_utc=datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
            )

            dataset = open_chunks(temp_root)
            assert dataset is not None
            # Dataset should be lazy and ready for operations

    def test_open_chunks_with_columns(self):
        """Test opening chunks with specific columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            writer = ParquetChunksWriter()

            rows = [
                {
                    "doc_id": "doc1",
                    "chunk_id": 0,
                    "text": "Content",
                    "tokens": 1,
                    "span": {"start": 0, "end": 7},
                    "created_at": datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
                    "schema_version": "docparse/chunks/1.0.0",
                },
            ]

            writer.write(
                rows,
                data_root=temp_root,
                rel_id="doc1",
                cfg_hash="cfg123",
            )

            dataset = open_chunks(temp_root, columns=["doc_id", "text"])
            assert dataset is not None

    def test_open_chunks_nonexistent(self):
        """Test opening non-existent chunks raises error."""
        nonexistent = Path("/tmp/nonexistent_chunks_12345")
        with pytest.raises(FileNotFoundError):
            open_chunks(nonexistent)


class TestOpenVectors:
    """Tests for open_vectors function."""

    def test_open_vectors_invalid_family(self):
        """Test opening vectors with invalid family raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            with pytest.raises(ValueError, match="Invalid family"):
                open_vectors(temp_root, "invalid_family")

    def test_open_vectors_nonexistent(self):
        """Test opening non-existent vectors raises error."""
        nonexistent = Path("/tmp/nonexistent_vectors_12345")
        with pytest.raises(FileNotFoundError):
            open_vectors(nonexistent, "dense")


class TestSummarize:
    """Tests for summarize function."""

    def test_summarize_chunks_dataset(self):
        """Test summarizing a chunks dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            writer = ParquetChunksWriter()

            rows = [
                {
                    "doc_id": f"doc{i}",
                    "chunk_id": 0,
                    "text": f"Content {i}",
                    "tokens": i,
                    "span": {"start": 0, "end": len(f"Content {i}")},
                    "created_at": datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
                    "schema_version": "docparse/chunks/1.0.0",
                }
                for i in range(3)
            ]

            writer.write(
                rows,
                data_root=temp_root,
                rel_id="test",
                cfg_hash="cfg123",
                dt_utc=datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
            )

            dataset = open_chunks(temp_root)
            summary = summarize(dataset, dataset_type="chunks")

            assert summary.dataset_type == "chunks"
            assert summary.file_count > 0
            assert summary.total_bytes > 0
            assert len(summary.sample_doc_ids) > 0

    def test_summarize_contains_schema(self):
        """Test that summary includes schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            writer = ParquetChunksWriter()

            rows = [
                {
                    "doc_id": f"doc{i}",
                    "chunk_id": 0,
                    "text": f"Content {i}",
                    "tokens": i,
                    "span": {"start": 0, "end": len(f"Content {i}")},
                    "created_at": datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
                    "schema_version": "docparse/chunks/1.0.0",
                }
                for i in range(3)
            ]

            writer.write(
                rows,
                data_root=temp_root,
                rel_id="test",
                cfg_hash="cfg123",
                dt_utc=datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
            )

            dataset = open_chunks(temp_root)
            summary = summarize(dataset)

            assert summary.schema is not None
            assert len(summary.schema.names) > 0

    def test_summarize_partitions(self):
        """Test that summary captures partitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            writer = ParquetChunksWriter()

            rows = [
                {
                    "doc_id": f"doc{i}",
                    "chunk_id": 0,
                    "text": f"Content {i}",
                    "tokens": i,
                    "span": {"start": 0, "end": len(f"Content {i}")},
                    "created_at": datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
                    "schema_version": "docparse/chunks/1.0.0",
                }
                for i in range(3)
            ]

            writer.write(
                rows,
                data_root=temp_root,
                rel_id="test",
                cfg_hash="cfg123",
                dt_utc=datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
            )

            dataset = open_chunks(temp_root)
            summary = summarize(dataset)

            # Should have partition info for 2025-10
            assert len(summary.partitions) > 0
            assert any("2025-10" in key for key in summary.partitions.keys())
