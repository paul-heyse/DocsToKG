# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_configuration_resolution",
#   "purpose": "Pytest coverage for docparsing configuration resolution scenarios",
#   "sections": [
#     {
#       "id": "test-resolve-hf-home",
#       "name": "test_resolve_hf_home",
#       "anchor": "function-test-resolve-hf-home",
#       "kind": "function"
#     },
#     {
#       "id": "test-resolve-model-root",
#       "name": "test_resolve_model_root",
#       "anchor": "function-test-resolve-model-root",
#       "kind": "function"
#     },
#     {
#       "id": "test-resolve-pdf-model-path",
#       "name": "test_resolve_pdf_model_path",
#       "anchor": "function-test-resolve-pdf-model-path",
#       "kind": "function"
#     },
#     {
#       "id": "test-environment-variable-precedence",
#       "name": "test_environment_variable_precedence",
#       "anchor": "function-test-environment-variable-precedence",
#       "kind": "function"
#     },
#     {
#       "id": "test-path-resolution-edge-cases",
#       "name": "test_path_resolution_edge_cases",
#       "anchor": "function-test-path-resolution-edge-cases",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Exercise environment-driven configuration resolution for DocParsing.

These tests cover the helper functions in `env.py` that discover HuggingFace
caches, model roots, and PDF model paths. They verify precedence rules between
environment variables and defaults, handle path edge cases (relative segments,
user expansion), and ensure downstream stages inherit consistent directories.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from DocsToKG.DocParsing.chunking.config import ChunkerCfg
from DocsToKG.DocParsing.doctags import (
    DoctagsCfg,
    resolve_hf_home,
    resolve_model_root,
    resolve_pdf_model_path,
)
from DocsToKG.DocParsing.env import init_hf_env
from DocsToKG.DocParsing.embedding.config import EmbedCfg

# --- Test Cases ---


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

        # Test without DOCSTOKG_MODEL_ROOT using default HF home fallback
        with patch.dict(os.environ, {}, clear=True):
            expected = resolve_hf_home().parent / "docs-to-kg" / "models"
            result = resolve_model_root()
            assert result == expected

        # Test fallback derived from a custom HF_HOME
        custom_hf = tmp_path / "alt-cache" / "huggingface"
        custom_hf.mkdir(parents=True, exist_ok=True)

        with patch.dict(os.environ, {"HF_HOME": str(custom_hf)}):
            expected = custom_hf.parent / "docs-to-kg" / "models"
            result = resolve_model_root()
            assert result == expected.resolve()

        direct = resolve_model_root(hf_home=custom_hf)
        assert direct == (custom_hf.parent / "docs-to-kg" / "models").resolve()


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

        # Test with DOCLING_PDF_MODEL pointing at a Hugging Face repo ID
        hf_repo = "ibm-granite/test-docling"
        with patch.dict(os.environ, {"DOCLING_PDF_MODEL": hf_repo}):
            result = resolve_pdf_model_path()
            assert result == hf_repo

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


def test_init_hf_env_default_model_root():
    """Ensure init_hf_env derives the default model root from the HF cache."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_home = tmp_path / "hf-cache"
        hf_home.mkdir()

        with patch.dict(os.environ, {}, clear=True):
            resolved_hf, resolved_model_root = init_hf_env(hf_home=hf_home)

        expected_model_root = hf_home.parent / "docs-to-kg" / "models"
        assert resolved_hf == hf_home.resolve()
        assert resolved_model_root == expected_model_root.resolve()
        assert os.environ["DOCSTOKG_MODEL_ROOT"] == str(expected_model_root.resolve())


def test_init_hf_env_accepts_string_overrides():
    """String inputs for overrides are expanded prior to environment setup."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hf_home = tmp_path / "hf-cache"
        model_root = tmp_path / "models"
        hf_home.mkdir()
        model_root.mkdir()

        with patch.dict(os.environ, {}, clear=True):
            resolved_hf, resolved_model_root = init_hf_env(
                hf_home=str(hf_home),
                model_root=str(model_root),
            )

        assert resolved_hf == hf_home.resolve()
        assert resolved_model_root == model_root.resolve()
        assert os.environ["HF_HOME"] == str(hf_home.resolve())
        assert os.environ["DOCSTOKG_MODEL_ROOT"] == str(model_root.resolve())


def test_path_resolution_edge_cases():
    """Test edge cases in path resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Test with non-existent path
        non_existent = tmp_path / "non_existent"
        result = resolve_pdf_model_path(cli_value=str(non_existent))
        assert result == str(non_existent)

        # Test DOCLING_PDF_MODEL with filesystem-like value expands to absolute path
        env_path = tmp_path / "env_model"
        env_path.mkdir()
        with patch.dict(os.environ, {"DOCLING_PDF_MODEL": str(env_path)}):
            result = resolve_pdf_model_path()
            assert result == str(env_path.resolve())

        # Test with relative path
        relative_path = "relative/path"
        result = resolve_pdf_model_path(cli_value=relative_path)
        assert result == relative_path

        # Test with empty string (should fallback to default)
        result = resolve_pdf_model_path(cli_value="")
        assert isinstance(result, str)
        assert len(result) > 0  # Should not be empty


def test_chunker_finalize_respects_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    os.environ["DOCSTOKG_DATA_ROOT"] = str(data_root)
    cfg = ChunkerCfg()
    cfg.finalize()
    assert cfg.data_root == data_root.resolve()
    assert cfg.in_dir == (data_root / "DocTagsFiles").resolve()
    assert cfg.out_dir == (data_root / "ChunkedDocTagFiles").resolve()


def test_chunker_finalize_token_window_invariant() -> None:
    os.environ.pop("DOCSTOKG_DATA_ROOT", None)
    cfg = ChunkerCfg(min_tokens=10, max_tokens=5)
    with pytest.raises(ValueError):
        cfg.finalize()


def test_embed_finalize_parallelism_invariant(tmp_path: Path) -> None:
    os.environ["DOCSTOKG_DATA_ROOT"] = str(tmp_path / "Data")
    cfg = EmbedCfg(files_parallel=0)
    with pytest.raises(ValueError):
        cfg.finalize()


def test_doctags_finalize_html_defaults(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    os.environ["DOCSTOKG_DATA_ROOT"] = str(data_root)
    cfg = DoctagsCfg(mode="html")
    cfg.finalize()
    assert cfg.data_root == data_root.resolve()
    assert cfg.input == (data_root / "HTML").resolve()
    assert cfg.output == (data_root / "DocTagsFiles").resolve()


def test_doctags_finalize_workers_invariant() -> None:
    cfg = DoctagsCfg(workers=0)
    with pytest.raises(ValueError):
        cfg.finalize()


# --- Module Entry Points ---


if __name__ == "__main__":
    pytest.main([__file__])
