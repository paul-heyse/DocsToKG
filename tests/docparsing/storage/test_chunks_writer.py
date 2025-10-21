"""Tests for ParquetChunksWriter (Parquet Chunks output)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from DocsToKG.DocParsing.storage.chunks_writer import ParquetChunksWriter, WriteResult
from DocsToKG.DocParsing.storage.parquet_schemas import (
    SCHEMA_VERSION_CHUNKS,
    chunks_schema,
    validate_parquet_file,
)
from DocsToKG.DocParsing.storage import paths


class TestParquetChunksWriter:
    """Tests for ParquetChunksWriter class."""

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def writer(self):
        """Create a ParquetChunksWriter instance."""
        return ParquetChunksWriter()

    def test_writer_initialization(self, writer):
        """Test writer initialization with defaults."""
        assert writer.schema is not None
        assert writer.compression == "zstd"
        assert writer.compression_level == 5
        assert writer.target_row_group_mb == 32

    def test_writer_schema_matches_chunks(self, writer):
        """Test that writer schema matches expected chunks schema."""
        expected_schema = chunks_schema()
        actual_schema = writer.schema
        assert actual_schema.names == expected_schema.names

    def test_validate_row_success(self, writer):
        """Test row validation with valid row."""
        row = {
            "doc_id": "test_doc",
            "chunk_id": 0,
            "text": "Sample text",
            "tokens": 10,
            "span": {"start": 0, "end": 11},
        }
        # Should not raise
        writer._validate_row(row)

    def test_validate_row_missing_doc_id(self, writer):
        """Test row validation rejects missing doc_id."""
        row = {
            "chunk_id": 0,
            "text": "Sample text",
            "tokens": 10,
            "span": {"start": 0, "end": 11},
        }
        with pytest.raises(ValueError, match="doc_id is required"):
            writer._validate_row(row)

    def test_validate_row_invalid_chunk_id(self, writer):
        """Test row validation rejects negative chunk_id."""
        row = {
            "doc_id": "test_doc",
            "chunk_id": -1,
            "text": "Sample text",
            "tokens": 10,
            "span": {"start": 0, "end": 11},
        }
        with pytest.raises(ValueError, match="chunk_id must be non-negative"):
            writer._validate_row(row)

    def test_validate_row_empty_text(self, writer):
        """Test row validation rejects empty text."""
        row = {
            "doc_id": "test_doc",
            "chunk_id": 0,
            "text": "",
            "tokens": 0,
            "span": {"start": 0, "end": 0},
        }
        with pytest.raises(ValueError, match="text is required"):
            writer._validate_row(row)

    def test_validate_row_invalid_span(self, writer):
        """Test row validation rejects invalid span."""
        row = {
            "doc_id": "test_doc",
            "chunk_id": 0,
            "text": "Sample text",
            "tokens": 10,
            "span": {"start": 10, "end": 5},  # start > end
        }
        with pytest.raises(ValueError, match="span invariant"):
            writer._validate_row(row)

    def test_compute_span_hash(self):
        """Test deterministic span hash computation."""
        text = "The quick brown fox jumps over the lazy dog"
        hash1 = ParquetChunksWriter._compute_span_hash(text, 0, 9)  # "The quick"
        hash2 = ParquetChunksWriter._compute_span_hash(text, 0, 9)  # Same span
        hash3 = ParquetChunksWriter._compute_span_hash(text, 10, 15)  # Different span

        assert hash1 == hash2  # Deterministic
        assert hash1 != hash3  # Different spans produce different hashes
        assert len(hash1) == 64  # SHA256 hex digest is 64 chars

    def test_write_success(self, writer, temp_dir):
        """Test successful write of chunks to Parquet."""
        rows = [
            {
                "doc_id": "doc1",
                "chunk_id": 0,
                "text": "First chunk text",
                "tokens": 5,
                "span": {"start": 0, "end": 16},
                "created_at": datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
                "schema_version": "docparse/chunks/1.0.0",
            },
            {
                "doc_id": "doc1",
                "chunk_id": 1,
                "text": "Second chunk text",
                "tokens": 6,
                "span": {"start": 17, "end": 34},
                "created_at": datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
                "schema_version": "docparse/chunks/1.0.0",
            },
        ]

        result = writer.write(
            rows,
            data_root=temp_dir,
            rel_id="doc1",
            cfg_hash="test_hash_123",
            dt_utc=datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
        )

        assert result.rows_written == 2
        assert result.parquet_bytes > 0
        assert len(result.paths) == 1
        assert result.paths[0].exists()

    def test_write_result_to_dict(self, writer, temp_dir):
        """Test WriteResult.to_dict() conversion."""
        rows = [
            {
                "doc_id": "doc1",
                "chunk_id": 0,
                "text": "Chunk",
                "tokens": 1,
                "span": {"start": 0, "end": 5},
                "created_at": datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
                "schema_version": "docparse/chunks/1.0.0",
            },
        ]

        result = writer.write(
            rows,
            data_root=temp_dir,
            rel_id="doc1",
            cfg_hash="test_hash_123",
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "rows_written" in result_dict
        assert "parquet_bytes" in result_dict
        assert result_dict["rows_written"] == 1

    def test_write_creates_partitioned_output(self, writer, temp_dir):
        """Test that output is written to partitioned directory structure."""
        rows = [
            {
                "doc_id": "papers/ai/ml",
                "chunk_id": 0,
                "text": "Content",
                "tokens": 2,
                "span": {"start": 0, "end": 7},
                "created_at": datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
                "schema_version": "docparse/chunks/1.0.0",
            },
        ]

        result = writer.write(
            rows,
            data_root=temp_dir,
            rel_id="papers/ai/ml",
            cfg_hash="cfg123",
            dt_utc=datetime(2025, 10, 21, 12, 0, 0, tzinfo=timezone.utc),
        )

        output_path = result.paths[0]
        # Should be: Chunks/fmt=parquet/2025/10/papers/ai/ml.parquet
        assert "Chunks" in str(output_path)
        assert "fmt=parquet" in str(output_path)
        assert "2025/10" in str(output_path)

    def test_write_parquet_footer_metadata(self, writer, temp_dir):
        """Test that Parquet footer contains required metadata."""
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
            data_root=temp_dir,
            rel_id="doc1",
            cfg_hash="cfg_hash_value",
        )

        output_path = result.paths[0]
        pf = pq.ParquetFile(str(output_path))
        footer = pf.metadata.metadata

        assert footer is not None
        assert b"docparse.schema_version" in footer
        assert b"docparse.cfg_hash" in footer
        assert b"docparse.created_by" in footer
        assert b"docparse.created_at" in footer

    def test_write_atomic_recovery(self, writer, temp_dir):
        """Test that failed write leaves no partial files."""
        rows_invalid = [{"bad": "row"}]  # Invalid row
        output_path = temp_dir / "test.parquet"

        with pytest.raises(Exception):
            writer.write(
                rows_invalid,
                data_root=temp_dir,
                rel_id="test",
                cfg_hash="cfg123",
            )

        # Check that no final file was created
        assert not output_path.exists()
        # Check that no temp files were left behind
        temp_files = list(temp_dir.glob("*.tmp.*"))
        assert len(temp_files) == 0

    def test_write_empty_rows_raises(self, writer, temp_dir):
        """Test that writing empty rows raises ValueError."""
        with pytest.raises(ValueError, match="No rows to write"):
            writer.write(
                [],
                data_root=temp_dir,
                rel_id="doc1",
                cfg_hash="cfg123",
            )

    def test_estimate_row_group_count(self):
        """Test row group count estimation."""
        # 64MB file with 32MB avg row groups = 2 groups
        count = ParquetChunksWriter._estimate_row_group_count(
            file_size=64 * 1024 * 1024, avg_row_group_mb=32
        )
        assert count == 2

        # Minimum is 1
        count = ParquetChunksWriter._estimate_row_group_count(
            file_size=1024 * 1024,
            avg_row_group_mb=32,  # 1MB < 32MB
        )
        assert count == 1


class TestWriteResultDataclass:
    """Tests for WriteResult dataclass."""

    def test_write_result_creation(self):
        """Test WriteResult creation with sample data."""
        paths = [Path("/tmp/test.parquet")]
        result = WriteResult(
            paths=paths, rows_written=100, row_group_count=2, parquet_bytes=5000000
        )

        assert result.rows_written == 100
        assert result.row_group_count == 2
        assert result.parquet_bytes == 5000000
        assert len(result.paths) == 1
