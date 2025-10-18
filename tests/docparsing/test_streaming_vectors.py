# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_streaming_vectors",
#   "purpose": "Pytest coverage for docparsing streaming vectors scenarios",
#   "sections": [
#     {
#       "id": "test-iter-rows-in-batches",
#       "name": "test_iter_rows_in_batches",
#       "anchor": "function-test-iter-rows-in-batches",
#       "kind": "function"
#     },
#     {
#       "id": "test-iter-rows-in-batches-empty-file",
#       "name": "test_iter_rows_in_batches_empty_file",
#       "anchor": "function-test-iter-rows-in-batches-empty-file",
#       "kind": "function"
#     },
#     {
#       "id": "test-iter-rows-in-batches-single-row",
#       "name": "test_iter_rows_in_batches_single_row",
#       "anchor": "function-test-iter-rows-in-batches-single-row",
#       "kind": "function"
#     },
#     {
#       "id": "test-iter-rows-in-batches-large-batch-size",
#       "name": "test_iter_rows_in_batches_large_batch_size",
#       "anchor": "function-test-iter-rows-in-batches-large-batch-size",
#       "kind": "function"
#     },
#     {
#       "id": "test-iter-rows-in-batches-skips-empty-lines",
#       "name": "test_iter_rows_in_batches_skips_empty_lines",
#       "anchor": "function-test-iter-rows-in-batches-skips-empty-lines",
#       "kind": "function"
#     },
#     {
#       "id": "test-iter-rows-in-batches-memory-efficiency",
#       "name": "test_iter_rows_in_batches_memory_efficiency",
#       "anchor": "function-test-iter-rows-in-batches-memory-efficiency",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Test streaming vector writing to reduce peak RAM usage."""

import json
import tempfile
from pathlib import Path

import pytest

from DocsToKG.DocParsing._embedding.runtime import iter_rows_in_batches

# --- Test Cases ---


def test_iter_rows_in_batches():
    """Test that iter_rows_in_batches processes JSONL files in batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        test_data = [
            {"uuid": "uuid1", "text": "text1", "doc_id": "doc1"},
            {"uuid": "uuid2", "text": "text2", "doc_id": "doc2"},
            {"uuid": "uuid3", "text": "text3", "doc_id": "doc3"},
            {"uuid": "uuid4", "text": "text4", "doc_id": "doc4"},
            {"uuid": "uuid5", "text": "text5", "doc_id": "doc5"},
        ]

        with jsonl_file.open("w") as f:
            for row in test_data:
                f.write(json.dumps(row) + "\n")

        # Test batch size 2
        batches = list(iter_rows_in_batches(jsonl_file, batch_size=2))

        assert len(batches) == 3  # 5 rows / 2 = 3 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

        # Verify data integrity
        all_rows = []
        for batch in batches:
            all_rows.extend(batch)

        assert len(all_rows) == 5
        assert all_rows[0]["uuid"] == "uuid1"
        assert all_rows[4]["uuid"] == "uuid5"


def test_iter_rows_in_batches_empty_file():
    """Test iter_rows_in_batches with empty file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create empty JSONL file
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        batches = list(iter_rows_in_batches(jsonl_file, batch_size=10))
        assert len(batches) == 0


def test_iter_rows_in_batches_single_row():
    """Test iter_rows_in_batches with single row."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create JSONL file with single row
        jsonl_file = tmp_path / "single.jsonl"
        test_data = {"uuid": "uuid1", "text": "text1", "doc_id": "doc1"}

        with jsonl_file.open("w") as f:
            f.write(json.dumps(test_data) + "\n")

        batches = list(iter_rows_in_batches(jsonl_file, batch_size=10))

        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0]["uuid"] == "uuid1"


def test_iter_rows_in_batches_large_batch_size():
    """Test iter_rows_in_batches with batch size larger than file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create JSONL file with 3 rows
        jsonl_file = tmp_path / "small.jsonl"
        test_data = [
            {"uuid": "uuid1", "text": "text1", "doc_id": "doc1"},
            {"uuid": "uuid2", "text": "text2", "doc_id": "doc2"},
            {"uuid": "uuid3", "text": "text3", "doc_id": "doc3"},
        ]

        with jsonl_file.open("w") as f:
            for row in test_data:
                f.write(json.dumps(row) + "\n")

        # Test with batch size larger than file
        batches = list(iter_rows_in_batches(jsonl_file, batch_size=10))

        assert len(batches) == 1
        assert len(batches[0]) == 3


def test_iter_rows_in_batches_skips_empty_lines():
    """Test that iter_rows_in_batches skips empty lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create JSONL file with empty lines
        jsonl_file = tmp_path / "with_empty.jsonl"
        test_data = [
            {"uuid": "uuid1", "text": "text1", "doc_id": "doc1"},
            {"uuid": "uuid2", "text": "text2", "doc_id": "doc2"},
        ]

        with jsonl_file.open("w") as f:
            f.write(json.dumps(test_data[0]) + "\n")
            f.write("\n")  # Empty line
            f.write(json.dumps(test_data[1]) + "\n")
            f.write("\n")  # Empty line

        batches = list(iter_rows_in_batches(jsonl_file, batch_size=10))

        assert len(batches) == 1
        assert len(batches[0]) == 2
        assert batches[0][0]["uuid"] == "uuid1"
        assert batches[0][1]["uuid"] == "uuid2"


def test_iter_rows_in_batches_memory_efficiency():
    """Test that iter_rows_in_batches processes files without loading entire content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create large JSONL file
        jsonl_file = tmp_path / "large.jsonl"
        num_rows = 1000

        with jsonl_file.open("w") as f:
            for i in range(num_rows):
                row = {"uuid": f"uuid{i}", "text": f"text{i}", "doc_id": f"doc{i}"}
                f.write(json.dumps(row) + "\n")

        # Process in small batches
        batch_size = 10
        total_rows = 0

        for batch in iter_rows_in_batches(jsonl_file, batch_size=batch_size):
            assert len(batch) <= batch_size
            total_rows += len(batch)

        assert total_rows == num_rows


# --- Module Entry Points ---


if __name__ == "__main__":
    pytest.main([__file__])
