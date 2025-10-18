"""CLI validation tests for the ``docparse manifest`` entry point."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from DocsToKG.DocParsing.cli_errors import CLIValidationError

from tests.docparsing.test_manifest_streaming_cli import _prepare_manifest_cli_stubs


def _install_requests_stub(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setitem(sys.modules, "requests", requests_module)
    monkeypatch.setitem(sys.modules, "requests.adapters", adapters_module)

    urllib3_module = types.ModuleType("urllib3")
    urllib3_util_module = types.ModuleType("urllib3.util")
    urllib3_retry_module = types.ModuleType("urllib3.util.retry")

    class _Retry:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            self.args = args
            self.kwargs = kwargs

    urllib3_retry_module.Retry = _Retry
    urllib3_module.util = urllib3_util_module

    monkeypatch.setitem(sys.modules, "urllib3", urllib3_module)
    monkeypatch.setitem(sys.modules, "urllib3.util", urllib3_util_module)
    monkeypatch.setitem(sys.modules, "urllib3.util.retry", urllib3_retry_module)


def _load_cli(monkeypatch: pytest.MonkeyPatch):
    """Prepare dependency stubs and return the CLI module."""

    _prepare_manifest_cli_stubs(monkeypatch)
    _install_requests_stub(monkeypatch)

    from DocsToKG.DocParsing.core import cli

    return cli


def test_manifest_accepts_known_stage(monkeypatch, tmp_path) -> None:
    """CLI should accept known manifest stages and reject unsupported ones."""

    cli = _load_cli(monkeypatch)

    manifests_dir = tmp_path / "Manifests"
    manifests_dir.mkdir()

    captured: dict[str, object] = {}
    call_count = {"value": 0}

    def fake_iter_manifest_entries(stages, data_root: Path):
        call_count["value"] += 1
        captured["stages"] = list(stages)
        captured["data_root"] = data_root
        return iter(())

    monkeypatch.setattr(cli, "iter_manifest_entries", fake_iter_manifest_entries)
    monkeypatch.setattr(
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


def test_manifest_accepts_discovered_stage(monkeypatch, tmp_path) -> None:
    """A manifest present on disk extends allowable ``--stage`` selections."""

    cli = _load_cli(monkeypatch)

    manifests_dir = tmp_path / "Manifests"
    manifests_dir.mkdir()

    novel_stage = "synthetic"
    manifest_path = manifests_dir / f"docparse.{novel_stage}.manifest.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_iter_manifest_entries(stages, data_root: Path):
        captured["stages"] = list(stages)
        captured["data_root"] = data_root
        return iter(())

    monkeypatch.setattr(cli, "iter_manifest_entries", fake_iter_manifest_entries)
    monkeypatch.setattr(
        cli,
        "data_manifests",
        lambda _root, *, ensure=False: manifests_dir,
    )

    exit_code = cli.manifest(["--stage", novel_stage, "--data-root", str(tmp_path)])

    assert exit_code == 0
    assert captured["stages"] == [novel_stage]
    assert captured["data_root"] == tmp_path
