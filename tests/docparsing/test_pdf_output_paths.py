# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_pdf_output_paths",
#   "purpose": "Pytest coverage for docparsing pdf output paths scenarios",
#   "sections": [
#     {
#       "id": "test-pdf-output-path-mirroring",
#       "name": "test_pdf_output_path_mirroring",
#       "anchor": "function-test-pdf-output-path-mirroring",
#       "kind": "function"
#     },
#     {
#       "id": "test-pdf-task-vlm-parameters",
#       "name": "test_pdf_task_vlm_parameters",
#       "anchor": "function-test-pdf-task-vlm-parameters",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Test PDF output path mirroring to prevent basename collisions."""

import tempfile
from pathlib import Path

import pytest

from DocsToKG.DocParsing.pipelines import PdfTask
# --- Test Cases ---


def test_pdf_output_path_mirroring():
    """Test that PDF outputs mirror input directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create nested directory structure
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        # Create nested PDFs with same basename
        pdf1 = input_dir / "subdir1" / "document.pdf"
        pdf2 = input_dir / "subdir2" / "document.pdf"

        pdf1.parent.mkdir(parents=True)
        pdf2.parent.mkdir(parents=True)

        # Create dummy PDF files
        pdf1.write_text("dummy pdf 1")
        pdf2.write_text("dummy pdf 2")

        # Test PdfTask creation
        task1 = PdfTask(
            pdf_path=pdf1,
            output_dir=output_dir,
            port=8000,
            input_hash="hash1",
            doc_id="subdir1/document",
            output_path=output_dir / "subdir1" / "document.doctags",
            served_model_names=("model",),
            inference_model="model",
            vlm_prompt="Convert this page to docling.",
            vlm_stop=("</doctag>", "<|end_of_text|>"),
        )

        task2 = PdfTask(
            pdf_path=pdf2,
            output_dir=output_dir,
            port=8000,
            input_hash="hash2",
            doc_id="subdir2/document",
            output_path=output_dir / "subdir2" / "document.doctags",
            served_model_names=("model",),
            inference_model="model",
            vlm_prompt="Convert this page to docling.",
            vlm_stop=("</doctag>", "<|end_of_text|>"),
        )

        # Verify output paths don't collide
        assert task1.output_path != task2.output_path
        assert task1.output_path.name == "document.doctags"
        assert task2.output_path.name == "document.doctags"

        # Verify directory structure is preserved
        assert "subdir1" in str(task1.output_path)
        assert "subdir2" in str(task2.output_path)

        # Verify doc_id reflects relative path
        assert task1.doc_id == "subdir1/document"
        assert task2.doc_id == "subdir2/document"


def test_pdf_task_vlm_parameters():
    """Test that VLM parameters are correctly stored and accessible."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        task = PdfTask(
            pdf_path=tmp_path / "test.pdf",
            output_dir=tmp_path / "output",
            port=8000,
            input_hash="hash",
            doc_id="test",
            output_path=tmp_path / "output" / "test.doctags",
            served_model_names=("model",),
            inference_model="model",
            vlm_prompt="Custom prompt for testing",
            vlm_stop=("custom_stop1", "custom_stop2"),
        )

        # Verify VLM parameters
        assert task.vlm_prompt == "Custom prompt for testing"
        assert task.vlm_stop == ("custom_stop1", "custom_stop2")
        assert len(task.vlm_stop) == 2
        assert "custom_stop1" in task.vlm_stop
        assert "custom_stop2" in task.vlm_stop
# --- Module Entry Points ---


if __name__ == "__main__":
    pytest.main([__file__])
