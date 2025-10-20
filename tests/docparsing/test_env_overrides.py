"""Tests for environment override behaviour in DocsToKG.DocParsing.env."""

from __future__ import annotations

import os
from pathlib import Path

from DocsToKG.DocParsing.env import ensure_qwen_environment, ensure_splade_environment


def test_ensure_splade_environment_cli_overrides_existing_env(monkeypatch, tmp_path: Path) -> None:
    """CLI-provided values should overwrite existing SPLADE environment settings."""

    monkeypatch.setenv("DOCSTOKG_SPLADE_DEVICE", "cuda:legacy")
    monkeypatch.setenv("SPLADE_DEVICE", "cuda:legacy")
    monkeypatch.setenv("DOCSTOKG_SPLADE_MODEL_DIR", str(tmp_path / "legacy"))

    override_dir = tmp_path / "override"
    env_info = ensure_splade_environment(device="cpu", cache_dir=override_dir)

    resolved_override = str(override_dir.expanduser().resolve())
    assert os.environ["DOCSTOKG_SPLADE_DEVICE"] == "cpu"
    assert os.environ["SPLADE_DEVICE"] == "cpu"
    assert os.environ["DOCSTOKG_SPLADE_MODEL_DIR"] == resolved_override
    assert env_info == {"device": "cpu", "model_dir": resolved_override}


def test_ensure_qwen_environment_cli_overrides_existing_env(monkeypatch, tmp_path: Path) -> None:
    """CLI-provided values should overwrite existing Qwen environment settings."""

    monkeypatch.setenv("DOCSTOKG_QWEN_DEVICE", "cuda:legacy")
    monkeypatch.setenv("VLLM_DEVICE", "cuda:legacy")
    monkeypatch.setenv("DOCSTOKG_QWEN_DTYPE", "float16")
    monkeypatch.setenv("DOCSTOKG_QWEN_MODEL_DIR", str(tmp_path / "legacy"))

    override_dir = tmp_path / "override"
    env_info = ensure_qwen_environment(device="cpu", dtype="float32", model_dir=override_dir)

    resolved_override = str(override_dir.expanduser().resolve())
    assert os.environ["DOCSTOKG_QWEN_DEVICE"] == "cpu"
    assert os.environ["VLLM_DEVICE"] == "cpu"
    assert os.environ["DOCSTOKG_QWEN_DTYPE"] == "float32"
    assert os.environ["DOCSTOKG_QWEN_MODEL_DIR"] == resolved_override
    assert env_info == {"device": "cpu", "dtype": "float32", "model_dir": resolved_override}
