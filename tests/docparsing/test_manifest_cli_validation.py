"""Validate error handling and request plumbing for `docparse manifest`.

The manifest CLI exposes streaming and tailing capabilities that hinge on robust
input validation and optional HTTP calls. This module stubs network clients and
verifies that invalid flag combinations raise precise `CLIValidationError`
messages while valid inputs wire up telemetry and streaming helpers correctly.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from DocsToKG.DocParsing.cli_errors import CLIValidationError
from tests.conftest import PatchManager
from tests.docparsing.test_manifest_streaming_cli import _prepare_manifest_cli_stubs


def _install_requests_stub(patcher: PatchManager) -> None:
    """Register lightweight ``requests`` modules for CLI imports."""

    if "requests" in sys.modules:
        return

    requests_module = types.ModuleType("requests")

    class _RequestException(Exception):
        pass

    class _HTTPError(_RequestException):
        def __init__(self, response=None) -> None:
            super().__init__("HTTP error")
            self.response = response

    class _Session:
        def __init__(self) -> None:
            self.headers = {}

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

    patcher.setitem(sys.modules, "requests", requests_module)
    patcher.setitem(sys.modules, "requests.adapters", adapters_module)

    urllib3_module = types.ModuleType("urllib3")
    urllib3_util_module = types.ModuleType("urllib3.util")
    urllib3_retry_module = types.ModuleType("urllib3.util.retry")

    class _Retry:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            self.args = args
            self.kwargs = kwargs

    urllib3_retry_module.Retry = _Retry
    urllib3_module.util = urllib3_util_module

    patcher.setitem(sys.modules, "urllib3", urllib3_module)
    patcher.setitem(sys.modules, "urllib3.util", urllib3_util_module)
    patcher.setitem(sys.modules, "urllib3.util.retry", urllib3_retry_module)


def _load_cli(patcher: PatchManager):
    """Prepare dependency stubs and return the CLI module."""

    _prepare_manifest_cli_stubs(patcher)
    _install_requests_stub(patcher)

    from DocsToKG.DocParsing.core import cli

    return cli


def test_manifest_accepts_known_stage(patcher, tmp_path) -> None:
    """CLI should accept known manifest stages and reject unsupported ones."""

    cli = _load_cli(patcher)

    manifests_dir = tmp_path / "Manifests"
    manifests_dir.mkdir()

    captured: dict[str, object] = {}
    call_count = {"value": 0}

    def fake_iter_manifest_entries(
        stages,
        data_root: Path,
        *,
        limit=None,
    ):
        call_count["value"] += 1
        captured["stages"] = list(stages)
        captured["data_root"] = data_root
        return iter(())

    patcher.setattr(cli, "iter_manifest_entries", fake_iter_manifest_entries)
    patcher.setattr(
        cli,
        "data_manifests",
        lambda _root, *, ensure=False: manifests_dir,
    )

    exit_code = cli.manifest(["--stage", "doctags", "--data-root", str(tmp_path)])

    assert exit_code == 0
    assert captured["stages"] == ["doctags-html", "doctags-pdf"]
    assert captured["data_root"] == tmp_path

    with pytest.raises(CLIValidationError) as excinfo:
        cli._manifest_main(["--stage", "unknown-stage", "--data-root", str(tmp_path)])

    assert "Unsupported stage" in str(excinfo.value)
    assert call_count["value"] == 1


def test_manifest_aliases_chunk_and_embed(patcher, tmp_path) -> None:
    """Singular stage aliases resolve to the canonical manifest identifiers."""

    cli = _load_cli(patcher)

    manifests_dir = tmp_path / "Manifests"
    manifests_dir.mkdir()

    observed: list[tuple[list[str], Path]] = []

    def fake_iter_manifest_entries(
        stages,
        data_root: Path,
        *,
        limit=None,
    ):
        observed.append((list(stages), data_root))
        return iter(())

    patcher.setattr(cli, "iter_manifest_entries", fake_iter_manifest_entries)
    patcher.setattr(
        cli,
        "data_manifests",
        lambda _root, *, ensure=False: manifests_dir,
    )

    exit_code_chunk = cli.manifest(["--stage", "chunk", "--data-root", str(tmp_path)])
    exit_code_embed = cli.manifest(["--stage", "embed", "--data-root", str(tmp_path)])

    assert exit_code_chunk == 0
    assert exit_code_embed == 0
    assert observed == [(["chunks"], tmp_path), (["embeddings"], tmp_path)]


def test_manifest_accepts_discovered_stage(patcher, tmp_path) -> None:
    """A manifest present on disk extends allowable ``--stage`` selections."""

    cli = _load_cli(patcher)

    manifests_dir = tmp_path / "Manifests"
    manifests_dir.mkdir()

    novel_stage = "synthetic"
    manifest_path = manifests_dir / f"docparse.{novel_stage}.manifest.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_iter_manifest_entries(
        stages,
        data_root: Path,
        *,
        limit=None,
    ):
        captured["stages"] = list(stages)
        captured["data_root"] = data_root
        return iter(())

    patcher.setattr(cli, "iter_manifest_entries", fake_iter_manifest_entries)
    patcher.setattr(
        cli,
        "data_manifests",
        lambda _root, *, ensure=False: manifests_dir,
    )

    exit_code = cli.manifest(["--stage", novel_stage, "--data-root", str(tmp_path)])

    assert exit_code == 0
    assert captured["stages"] == [novel_stage]
    assert captured["data_root"] == tmp_path
