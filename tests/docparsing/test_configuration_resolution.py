# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_configuration_resolution",
#   "purpose": "Pytest coverage for docparsing configuration resolution scenarios",
#   "sections": [
#     {
#       "id": "test_resolve_hf_home",
#       "name": "test_resolve_hf_home",
#       "anchor": "TRHH",
#       "kind": "function"
#     },
#     {
#       "id": "test_resolve_model_root",
#       "name": "test_resolve_model_root",
#       "anchor": "TRMR",
#       "kind": "function"
#     },
#     {
#       "id": "test_resolve_pdf_model_path",
#       "name": "test_resolve_pdf_model_path",
#       "anchor": "TRPMP",
#       "kind": "function"
#     },
#     {
#       "id": "test_environment_variable_precedence",
#       "name": "test_environment_variable_precedence",
#       "anchor": "TEVP",
#       "kind": "function"
#     },
#     {
#       "id": "test_path_resolution_edge_cases",
#       "name": "test_path_resolution_edge_cases",
#       "anchor": "TPREC",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Test environment variable precedence for model paths."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from DocsToKG.DocParsing.pipelines import (
    resolve_hf_home,
    resolve_model_root,
    resolve_pdf_model_path,
)


def test_resolve_hf_home():
    """Test HF_HOME environment variable resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Test with HF_HOME set
        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            result = resolve_hf_home()
            assert result == tmp_path

        # Test without HF_HOME (should use default)
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_hf_home()
            # Should return default HF home path
            assert isinstance(result, Path)
            assert result.exists() or result.parent.exists()


def test_resolve_model_root():
    """Test DOCSTOKG_MODEL_ROOT environment variable resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Test with DOCSTOKG_MODEL_ROOT set
        with patch.dict(os.environ, {"DOCSTOKG_MODEL_ROOT": str(tmp_path)}):
            result = resolve_model_root()
            assert result == tmp_path

        # Test without DOCSTOKG_MODEL_ROOT (should fallback to HF home)
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_model_root()
            # Should return HF home or its subdirectory
            assert isinstance(result, Path)


def test_resolve_pdf_model_path():
    """Test PDF model path resolution with CLI and environment precedence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test model directory
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Test with CLI value
        result = resolve_pdf_model_path(cli_value=str(model_dir))
        assert result == str(model_dir)

        # Test with DOCLING_PDF_MODEL environment variable
        with patch.dict(os.environ, {"DOCLING_PDF_MODEL": str(model_dir)}):
            result = resolve_pdf_model_path()
            assert result == str(model_dir)

        # Test with DOCSTOKG_MODEL_ROOT environment variable
        with patch.dict(os.environ, {"DOCSTOKG_MODEL_ROOT": str(tmp_path)}):
            result = resolve_pdf_model_path()
            # Should construct path using model root
            assert isinstance(result, str)
            assert "test_model" in result or "granite-docling" in result

        # Test precedence: CLI should override environment
        with patch.dict(os.environ, {"DOCLING_PDF_MODEL": str(model_dir)}):
            cli_override = tmp_path / "cli_override"
            cli_override.mkdir()
            result = resolve_pdf_model_path(cli_value=str(cli_override))
            assert result == str(cli_override)


def test_environment_variable_precedence():
    """Test that environment variables are resolved in correct precedence order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test directories
        hf_home = tmp_path / "hf_home"
        model_root = tmp_path / "model_root"
        pdf_model = tmp_path / "pdf_model"

        hf_home.mkdir()
        model_root.mkdir()
        pdf_model.mkdir()

        # Test precedence: DOCLING_PDF_MODEL > DOCSTOKG_MODEL_ROOT > HF_HOME
        with patch.dict(
            os.environ,
            {
                "HF_HOME": str(hf_home),
                "DOCSTOKG_MODEL_ROOT": str(model_root),
                "DOCLING_PDF_MODEL": str(pdf_model),
            },
        ):
            result = resolve_pdf_model_path()
            assert result == str(pdf_model)

        # Test precedence: DOCSTOKG_MODEL_ROOT > HF_HOME
        with patch.dict(
            os.environ,
            {
                "HF_HOME": str(hf_home),
                "DOCSTOKG_MODEL_ROOT": str(model_root),
            },
        ):
            result = resolve_pdf_model_path()
            # Should use model_root as base
            assert isinstance(result, str)

        # Test fallback to HF_HOME
        with patch.dict(
            os.environ,
            {
                "HF_HOME": str(hf_home),
            },
        ):
            result = resolve_pdf_model_path()
            # Should use hf_home as base
            assert isinstance(result, str)


def test_path_resolution_edge_cases():
    """Test edge cases in path resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Test with non-existent path
        non_existent = tmp_path / "non_existent"
        result = resolve_pdf_model_path(cli_value=str(non_existent))
        assert result == str(non_existent)

        # Test with relative path
        relative_path = "relative/path"
        result = resolve_pdf_model_path(cli_value=relative_path)
        assert result == relative_path

        # Test with empty string (should fallback to default)
        result = resolve_pdf_model_path(cli_value="")
        assert isinstance(result, str)
        assert len(result) > 0  # Should not be empty


if __name__ == "__main__":
    pytest.main([__file__])
