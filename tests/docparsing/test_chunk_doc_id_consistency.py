# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_chunk_doc_id_consistency",
#   "purpose": "Pytest coverage for docparsing chunk doc id consistency scenarios",
#   "sections": [
#     {
#       "id": "test-compute-relative-doc-id",
#       "name": "test_compute_relative_doc_id",
#       "anchor": "function-test-compute-relative-doc-id",
#       "kind": "function"
#     },
#     {
#       "id": "test-relative-doc-id-uniqueness",
#       "name": "test_relative_doc_id_uniqueness",
#       "anchor": "function-test-relative-doc-id-uniqueness",
#       "kind": "function"
#     },
#     {
#       "id": "test-relative-doc-id-edge-cases",
#       "name": "test_relative_doc_id_edge_cases",
#       "anchor": "function-test-relative-doc-id-edge-cases",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Test chunk doc_id consistency with relative paths."""

import tempfile
from pathlib import Path

import pytest

from DocsToKG.DocParsing._common import compute_relative_doc_id
# --- Test Cases ---


def test_compute_relative_doc_id():
    """Test that relative doc IDs are computed correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create nested directory structure
        root_dir = tmp_path / "root"
        subdir1 = root_dir / "subdir1"
        subdir2 = root_dir / "subdir2" / "nested"

        subdir1.mkdir(parents=True)
        subdir2.mkdir(parents=True)

        # Test files at different levels
        file1 = subdir1 / "document.pdf"
        file2 = subdir2 / "document.pdf"
        file3 = root_dir / "document.pdf"

        # Create dummy files
        file1.write_text("dummy")
        file2.write_text("dummy")
        file3.write_text("dummy")

        # Test relative ID computation
        rel_id1 = compute_relative_doc_id(file1, root_dir)
        rel_id2 = compute_relative_doc_id(file2, root_dir)
        rel_id3 = compute_relative_doc_id(file3, root_dir)

        # Verify relative IDs
        assert rel_id1 == "subdir1/document.pdf"
        assert rel_id2 == "subdir2/nested/document.pdf"
        assert rel_id3 == "document.pdf"

        # Verify POSIX-style paths
        assert "/" in rel_id1
        assert "/" in rel_id2
        assert "\\" not in rel_id1  # No Windows-style separators
        assert "\\" not in rel_id2


def test_relative_doc_id_uniqueness():
    """Test that relative doc IDs are unique even with same basenames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        root_dir = tmp_path / "root"

        # Create multiple files with same basename in different directories
        files = [
            root_dir / "document.pdf",
            root_dir / "subdir1" / "document.pdf",
            root_dir / "subdir2" / "document.pdf",
            root_dir / "subdir1" / "nested" / "document.pdf",
        ]

        # Create directories and files
        for file_path in files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("dummy")

        # Compute relative IDs
        rel_ids = [compute_relative_doc_id(f, root_dir) for f in files]

        # Verify all IDs are unique
        assert len(rel_ids) == len(set(rel_ids))

        # Verify expected IDs
        expected_ids = [
            "document.pdf",
            "subdir1/document.pdf",
            "subdir2/document.pdf",
            "subdir1/nested/document.pdf",
        ]

        for expected_id in expected_ids:
            assert expected_id in rel_ids


def test_relative_doc_id_edge_cases():
    """Test edge cases for relative doc ID computation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        root_dir = tmp_path / "root"
        root_dir.mkdir()

        # Test file in root directory
        root_file = root_dir / "root_file.pdf"
        root_file.write_text("dummy")

        rel_id = compute_relative_doc_id(root_file, root_dir)
        assert rel_id == "root_file.pdf"

        # Test file with spaces and special characters
        special_file = root_dir / "file with spaces & symbols.pdf"
        special_file.write_text("dummy")

        rel_id = compute_relative_doc_id(special_file, root_dir)
        assert rel_id == "file with spaces & symbols.pdf"
# --- Module Entry Points ---


if __name__ == "__main__":
    pytest.main([__file__])
