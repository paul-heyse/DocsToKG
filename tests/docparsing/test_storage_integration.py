"""
Integration tests for storage layer with embedding stage.

Tests round-trip data flow through the new storage layer and validates
Parquet footer metadata and schema enforcement.
"""

import json
import tempfile
from pathlib import Path

import pytest

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not available")
class TestStorageIntegration:
    """Test storage layer integration."""

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not available")
    def test_unified_vector_writer_parquet_format(self, tmp_path: Path) -> None:
        """Test writing vectors in Parquet format using unified writer."""
        from DocsToKG.DocParsing.storage.embedding_integration import (
            create_unified_vector_writer,
        )

        output_file = tmp_path / "vectors.parquet"
        test_rows = [
            {
                "UUID": "uuid-1",
                "BM25": {"terms": ["hello"], "weights": [1.0], "avgdl": 10.0, "N": 100},
                "SPLADEv3": {"tokens": ["hello"], "weights": [0.5]},
                "Qwen3-4B": {"model_id": "qwen", "vector": [0.1, 0.2], "dimension": 2},
                "model_metadata": "{}",
                "schema_version": "1.0.0",
            }
        ]

        with create_unified_vector_writer(output_file, fmt="parquet") as writer:
            writer.write_rows(test_rows)

        # Verify Parquet was written
        assert output_file.exists()
        table = pq.read_table(output_file)
        assert len(table) == 1
        assert table.column("UUID")[0].as_py() == "uuid-1"

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not available")
    def test_roundtrip_parquet_to_parquet(self, tmp_path: Path) -> None:
        """Test round-trip: write Parquet, read as Parquet."""
        from DocsToKG.DocParsing.storage.embedding_integration import (
            create_unified_vector_writer,
            iter_vector_rows,
        )

        output_file = tmp_path / "vectors.parquet"
        test_rows = [
            {
                "UUID": f"uuid-{i}",
                "BM25": {"terms": ["test"], "weights": [float(i)], "avgdl": 10.0, "N": 100},
                "SPLADEv3": {"tokens": ["test"], "weights": [0.5]},
                "Qwen3-4B": {"model_id": "qwen", "vector": [0.1, 0.2], "dimension": 2},
                "model_metadata": "{}",
                "schema_version": "1.0.0",
            }
            for i in range(3)
        ]

        # Write as Parquet
        with create_unified_vector_writer(output_file, fmt="parquet") as writer:
            writer.write_rows(test_rows)

        # Read back with iter_vector_rows
        batches = list(iter_vector_rows(output_file, "parquet", batch_size=2))
        assert len(batches) == 2  # 3 rows with batch_size=2 -> 2 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

        # Verify round-trip preserves data
        all_rows = [row for batch in batches for row in batch]
        assert len(all_rows) == 3
        assert all_rows[0]["UUID"] == "uuid-0"
        assert all_rows[1]["UUID"] == "uuid-1"
        assert all_rows[2]["UUID"] == "uuid-2"

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not available")
    def test_parquet_schema_validation(self, tmp_path: Path) -> None:
        """Test that Parquet writer enforces correct schema."""
        from DocsToKG.DocParsing.storage import parquet_schemas

        # Create a table with the legacy vector schema
        test_data = {
            "UUID": ["uuid-1"],
            "BM25": [{"terms": ["hello"], "weights": [1.0], "avgdl": 10.0, "N": 100}],
            "SPLADEv3": [{"tokens": ["hello"], "weights": [0.5]}],
            "Qwen3-4B": [{"model_id": "qwen", "vector": [0.1, 0.2], "dimension": 2}],
            "model_metadata": ["{}"],
            "schema_version": ["1.0.0"],
        }

        schema = parquet_schemas._legacy_vector_schema()
        table = pa.Table.from_pydict(test_data, schema=schema)
        output_file = tmp_path / "test.parquet"
        pq.write_table(table, str(output_file))

        # Verify file was written
        assert output_file.exists()
        read_table = pq.read_table(output_file)
        assert read_table.schema == schema

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not available")
    def test_dense_schema_creation(self) -> None:
        """Test dense vector schema creation."""
        from DocsToKG.DocParsing.storage import parquet_schemas

        schema = parquet_schemas.dense_schema(dim=768, fixed_size=True)
        assert "vec" in schema.names
        assert "dim" in schema.names
        assert "normalize_l2" in schema.names

    def test_format_validation(self, tmp_path: Path) -> None:
        """Test that invalid format raises error."""
        from DocsToKG.DocParsing.storage.embedding_integration import (
            create_unified_vector_writer,
        )

        output_file = tmp_path / "vectors.invalid"
        with pytest.raises(ValueError, match="Parquet is the only supported format"):
            create_unified_vector_writer(output_file, fmt="invalid")


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not available")
class TestLegacySchemaSupport:
    """Test backward compatibility with legacy vector format."""

    def test_legacy_vector_schema_roundtrip(self, tmp_path: Path) -> None:
        """Test that legacy vector schema can be written and read."""
        from DocsToKG.DocParsing.storage import parquet_schemas

        test_data = {
            "UUID": ["uuid-1", "uuid-2"],
            "BM25": [
                {
                    "terms": ["hello", "world"],
                    "weights": [1.0, 0.5],
                    "avgdl": 10.0,
                    "N": 100,
                },
                {
                    "terms": ["foo", "bar"],
                    "weights": [0.8, 0.2],
                    "avgdl": 10.0,
                    "N": 100,
                },
            ],
            "SPLADEv3": [
                {"tokens": ["hello"], "weights": [0.5]},
                {"tokens": ["world"], "weights": [0.3]},
            ],
            "Qwen3-4B": [
                {"model_id": "qwen", "vector": [0.1, 0.2], "dimension": 2},
                {"model_id": "qwen", "vector": [0.3, 0.4], "dimension": 2},
            ],
            "model_metadata": ["{}", "{}"],
            "schema_version": ["1.0.0", "1.0.0"],
        }

        schema = parquet_schemas._legacy_vector_schema()
        table = pa.Table.from_pydict(test_data, schema=schema)
        output_file = tmp_path / "legacy_vectors.parquet"
        pq.write_table(table, str(output_file))

        # Verify round-trip
        read_table = pq.read_table(output_file)
        assert len(read_table) == 2
        assert read_table.column("UUID")[0].as_py() == "uuid-1"
        assert read_table.column("UUID")[1].as_py() == "uuid-2"


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not available")
class TestFooterMetadata:
    """Test Parquet footer metadata handling."""

    def test_chunks_footer_metadata(self) -> None:
        """Test that chunks footer metadata can be built."""
        from DocsToKG.DocParsing.storage import parquet_schemas

        footer_meta = parquet_schemas.build_footer_common(
            schema_version=parquet_schemas.SCHEMA_VERSION_CHUNKS,
            cfg_hash="abc123",
            created_by="test-builder",
        )

        assert "docparse.schema_version" in footer_meta
        assert "docparse.cfg_hash" in footer_meta
        assert "docparse.created_by" in footer_meta
        assert footer_meta["docparse.schema_version"] == parquet_schemas.SCHEMA_VERSION_CHUNKS

    def test_dense_footer_metadata(self) -> None:
        """Test that dense vector footer metadata can be built."""
        from DocsToKG.DocParsing.storage import parquet_schemas

        footer_meta = parquet_schemas.build_footer_dense(
            provider="dense.qwen_vllm",
            model_id="Qwen2-7B-Embedding",
            dim=768,
            cfg_hash="abc123",
        )

        assert footer_meta["docparse.family"] == "dense"
        assert footer_meta["docparse.provider"] == "dense.qwen_vllm"
        assert footer_meta["docparse.model_id"] == "Qwen2-7B-Embedding"
        assert int(footer_meta["docparse.dim"]) == 768

    def test_footer_validation_chunks(self) -> None:
        """Test validation of chunks footer metadata."""
        from DocsToKG.DocParsing.storage import parquet_schemas

        # Valid footer
        valid_footer = parquet_schemas.build_footer_common(
            schema_version=parquet_schemas.SCHEMA_VERSION_CHUNKS,
            cfg_hash="abc",
            created_by="test",
        )
        result = parquet_schemas.validate_footer_common(valid_footer)
        assert result.ok

        # Missing required key
        invalid_footer = {"docparse.cfg_hash": "abc"}
        result = parquet_schemas.validate_footer_common(invalid_footer)
        assert not result.ok
        assert len(result.errors) > 0
