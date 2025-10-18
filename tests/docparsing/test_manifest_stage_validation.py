"""Validation coverage for manifest stage argument handling."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest

from DocsToKG.DocParsing.cli_errors import CLIValidationError
from tests.docparsing.stubs import dependency_stubs


def _iter_stub(stages: list[str]) -> Iterable[Dict[str, Any]]:
    """Yield minimal manifest entries for the provided stages."""

    for stage in stages:
        yield {
            "timestamp": "2025-01-01T00:00:00",
            "stage": stage,
            "doc_id": f"{stage}-doc",
            "status": "success",
            "duration_s": 0.1,
        }


def _install_logging_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the minimal logging utilities required by the CLI import."""

    logging_utils_module = types.ModuleType("DocsToKG.OntologyDownload.logging_utils")

    class _JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - passthrough
            return super().format(record)

    logging_utils_module.JSONFormatter = _JSONFormatter
    monkeypatch.setitem(sys.modules, "DocsToKG.OntologyDownload.logging_utils", logging_utils_module)


def _install_http_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install lightweight HTTP client stubs used by the CLI import path."""

    requests_module = types.ModuleType("requests")

    class _RequestException(Exception):
        pass

    class _HTTPError(_RequestException):
        def __init__(self, response=None) -> None:
            super().__init__("HTTP error")
            self.response = response

    class _Session:
        def __init__(self) -> None:
            self.headers: Dict[str, str] = {}

        def mount(self, *_args, **_kwargs) -> None:  # pragma: no cover - stub
            return None

    adapters_module = types.ModuleType("requests.adapters")

    class _HTTPAdapter:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            self.args = args
            self.kwargs = kwargs

    adapters_module.HTTPAdapter = _HTTPAdapter
    requests_module.Session = _Session
    requests_module.RequestException = _RequestException
    requests_module.HTTPError = _HTTPError
    requests_module.ConnectionError = _RequestException
    requests_module.Timeout = _RequestException
    requests_module.adapters = adapters_module
    requests_module.exceptions = types.SimpleNamespace(SSLError=_RequestException)

    urllib3_module = types.ModuleType("urllib3")
    urllib3_util_module = types.ModuleType("urllib3.util")
    urllib3_retry_module = types.ModuleType("urllib3.util.retry")

    class _Retry:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            self.args = args
            self.kwargs = kwargs

    urllib3_retry_module.Retry = _Retry
    urllib3_module.util = urllib3_util_module

    monkeypatch.setitem(sys.modules, "requests", requests_module)
    monkeypatch.setitem(sys.modules, "requests.adapters", adapters_module)
    monkeypatch.setitem(sys.modules, "urllib3", urllib3_module)
    monkeypatch.setitem(sys.modules, "urllib3.util", urllib3_util_module)
    monkeypatch.setitem(sys.modules, "urllib3.util.retry", urllib3_retry_module)


def _install_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure optional dependency stubs are available before importing the CLI."""

    fake_vllm = types.ModuleType("tests.docparsing.fake_deps.vllm")

    class _StubEmbeddingOutputs:
        def __init__(self, embedding) -> None:  # pragma: no cover - stub
            self.embedding = embedding

    class _StubEmbedding:
        def __init__(self, embedding) -> None:  # pragma: no cover - stub
            self.outputs = _StubEmbeddingOutputs(embedding)

    class _StubLLM:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            self.dense_dim = kwargs.get("dense_dim", 2560)

        def embed(self, prompts) -> list[_StubEmbedding]:  # pragma: no cover - stub
            embedding = [float(index) for index in range(self.dense_dim)]
            return [_StubEmbedding(embedding) for _ in prompts]

    fake_vllm.LLM = _StubLLM
    fake_vllm.DEFAULT_DENSE_DIM = 2560
    fake_vllm.PoolingParams = type("PoolingParams", (), {})

    monkeypatch.setitem(sys.modules, "tests.docparsing.fake_deps.vllm", fake_vllm)
    dependency_stubs()


def _prepare_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install all test doubles needed to import the CLI module."""

    _install_dependency_stubs(monkeypatch)
    _install_logging_stub(monkeypatch)
    _install_http_stubs(monkeypatch)


def test_manifest_stage_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Mixed-case stages are normalised, deduplicated, and forwarded in order."""

    _prepare_runtime(monkeypatch)
    from DocsToKG.DocParsing.core import cli

    observed: dict[str, Any] = {}

    def fake_iter_manifest_entries(stages: list[str], data_root: Path):
        observed["stages"] = stages
        observed["data_root"] = data_root
        return _iter_stub(stages)

    monkeypatch.setattr(cli, "iter_manifest_entries", fake_iter_manifest_entries)
    monkeypatch.setattr(
        cli, "data_manifests", lambda _root, *, ensure=True: tmp_path
    )
    monkeypatch.setattr(
        cli,
        "known_stages",
        ["doctags", "chunk", "embeddings"],
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "known_stage_set",
        {"doctags", "chunk", "embeddings"},
        raising=False,
    )

    exit_code = cli.manifest(
        [
            "--stage",
            "Doctags",
            "--stage",
            "doctags",
            "--stage",
            "EMBEDdings",
            "--stage",
            "chunk",
            "--stage",
            "CHUNK",
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert observed["stages"] == ["doctags", "embeddings", "chunk"]
    assert observed["data_root"] == tmp_path


def test_manifest_rejects_unknown_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unknown ``--stage`` value raises a structured validation error."""

    _prepare_runtime(monkeypatch)
    from DocsToKG.DocParsing.core import cli

    monkeypatch.setattr(
        cli, "data_manifests", lambda _root, *, ensure=True: tmp_path
    )
    monkeypatch.setattr(
        cli,
        "known_stages",
        ["doctags", "chunk", "embeddings"],
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "known_stage_set",
        {"doctags", "chunk", "embeddings"},
        raising=False,
    )

    with pytest.raises(CLIValidationError) as excinfo:
        cli.manifest([
            "--stage",
            "invalid",
            "--data-root",
            str(tmp_path),
        ])

    error = excinfo.value
    assert error.option == "--stage"
    assert "invalid" in error.message


def test_manifest_help_describes_stage_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The manifest help text documents the discovery default and fallback."""

    _prepare_runtime(monkeypatch)
    from DocsToKG.DocParsing.core import cli

    with pytest.raises(SystemExit):
        cli.manifest(["--help"])

    output = " ".join(capsys.readouterr().out.split())
    assert "Defaults to stages discovered from manifest files" in output
    assert "falls back to embeddings when no manifests are present." in output
