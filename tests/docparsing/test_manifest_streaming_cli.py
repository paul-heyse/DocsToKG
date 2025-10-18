"""Regression tests for manifest CLI streaming behaviour."""

from __future__ import annotations

import json
import logging
import tracemalloc
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typing import Dict

import pytest

from DocsToKG.DocParsing.env import data_manifests
from DocsToKG.DocParsing.io import iter_manifest_entries

from tests.docparsing.stubs import dependency_stubs


def _prepare_manifest_cli_stubs(monkeypatch) -> None:
    """Install shared dependency stubs required to import the DocParsing CLI."""

    if "tests.docparsing.fake_deps.vllm" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "tests.docparsing.fake_deps.vllm",
            types.ModuleType("tests.docparsing.fake_deps.vllm"),
        )

    dependency_stubs()

    logging_utils_module = types.ModuleType("DocsToKG.OntologyDownload.logging_utils")

    class _JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - stub
            return super().format(record)

    logging_utils_module.JSONFormatter = _JSONFormatter
    monkeypatch.setitem(
        sys.modules, "DocsToKG.OntologyDownload.logging_utils", logging_utils_module
    )

    yaml_stub = types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {}, YAMLError=Exception)
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    if "requests" not in sys.modules:
        requests_module = types.ModuleType("requests")

        class _RequestException(Exception):
            """Base request exception stub."""

        class _HTTPError(_RequestException):
            def __init__(self, response=None) -> None:  # pragma: no cover - stub
                super().__init__("HTTP error")
                self.response = response

        class _Session:
            def __init__(self) -> None:  # pragma: no cover - stub
                self.headers: dict[str, str] = {}
                self.auth = None
                self.cookies: dict[str, str] = {}
                self.params: dict[str, str] = {}
                self.proxies: dict[str, str] = {}
                self.verify = True
                self.cert = None
                self.trust_env = True
                self.max_redirects = 30
                self.stream = False
                self.hooks: dict[str, list[object]] = {"response": []}
                self.adapters: dict[str, object] = {}

            def mount(self, prefix: str, adapter: object) -> None:  # pragma: no cover - stub
                self.adapters[prefix] = adapter

        requests_module.Session = _Session
        requests_module.RequestException = _RequestException
        requests_module.HTTPError = _HTTPError
        requests_module.ConnectionError = _RequestException
        requests_module.Timeout = _RequestException
        requests_module.exceptions = types.SimpleNamespace(SSLError=_RequestException)
        monkeypatch.setitem(sys.modules, "requests", requests_module)

    if "requests.adapters" not in sys.modules:
        adapters_module = types.ModuleType("requests.adapters")

        class _HTTPAdapter:
            def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
                self.args = args
                self.kwargs = kwargs

        adapters_module.HTTPAdapter = _HTTPAdapter
        monkeypatch.setitem(sys.modules, "requests.adapters", adapters_module)
        sys.modules["requests"].adapters = adapters_module  # type: ignore[index]

    if "urllib3" not in sys.modules:
        urllib3_module = types.ModuleType("urllib3")
        monkeypatch.setitem(sys.modules, "urllib3", urllib3_module)

    if "urllib3.util" not in sys.modules:
        urllib3_util_module = types.ModuleType("urllib3.util")
        monkeypatch.setitem(sys.modules, "urllib3.util", urllib3_util_module)
        sys.modules["urllib3"].util = urllib3_util_module  # type: ignore[index]

    if "urllib3.util.retry" not in sys.modules:
        urllib3_retry_module = types.ModuleType("urllib3.util.retry")

        class _Retry:
            def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
                self.args = args
                self.kwargs = kwargs

        urllib3_retry_module.Retry = _Retry
        monkeypatch.setitem(sys.modules, "urllib3.util.retry", urllib3_retry_module)


@pytest.mark.parametrize("tail", [7])
def test_manifest_streams_large_tail(monkeypatch, tmp_path, capsys, tail: int) -> None:
    """Ensure ``manifest`` streams large iterators while keeping tail accuracy."""

    total_entries = 10_050
    stage_name = "chunk"

    _prepare_manifest_cli_stubs(monkeypatch)

    vllm_module = types.ModuleType("vllm")

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

    vllm_module.LLM = _StubLLM
    vllm_module.DEFAULT_DENSE_DIM = 2560
    vllm_module.PoolingParams = type("PoolingParams", (), {})
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)

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

    requests_module.Session = _Session
    requests_module.RequestException = _RequestException
    requests_module.HTTPError = _HTTPError
    requests_module.ConnectionError = _RequestException
    requests_module.Timeout = _RequestException
    requests_module.exceptions = types.SimpleNamespace(SSLError=_RequestException)
    monkeypatch.setitem(sys.modules, "requests", requests_module)

    adapters_module = types.ModuleType("requests.adapters")

    class _HTTPAdapter:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            self.args = args
            self.kwargs = kwargs

    adapters_module.HTTPAdapter = _HTTPAdapter
    monkeypatch.setitem(sys.modules, "requests.adapters", adapters_module)
    requests_module.adapters = adapters_module

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

    from DocsToKG.DocParsing.core import cli

    def fake_iter_manifest_entries(stages, data_root):
        assert stages == [stage_name]
        assert data_root == tmp_path

        for index in range(total_entries):
            yield {
                "timestamp": f"2025-01-01T00:{index // 60:02d}:{index % 60:02d}",
                "stage": stage_name,
                "doc_id": f"doc-{index}",
                "status": "failure" if index % 2000 == 0 else "success",
                "duration_s": 0.5,
            }

    monkeypatch.setattr(
        cli,
        "iter_manifest_entries",
        fake_iter_manifest_entries,
    )
    monkeypatch.setattr(
        cli, "data_manifests", lambda data_root, *, ensure=True: tmp_path
    )
    monkeypatch.setattr(cli, "known_stages", [stage_name], raising=False)
    monkeypatch.setattr(cli, "known_stage_set", {stage_name}, raising=False)

    exit_code = cli.manifest(
        [
            "--stage",
            stage_name,
            "--tail",
            str(tail),
            "--summarize",
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0

    stdout = capsys.readouterr().out.splitlines()
    assert stdout[0] == f"docparse manifest tail (last {tail} entries)"

    expected_ids = [f"doc-{index}" for index in range(total_entries - tail, total_entries)]
    tail_lines = stdout[1 : 1 + tail]
    for expected_id, line in zip(expected_ids, tail_lines):
        assert expected_id in line
        assert "status=" in line

    manifest_header_index = stdout.index("Manifest summary")
    summary_line = stdout[manifest_header_index + 1]
    assert summary_line == f"- {stage_name}: total={total_entries} duration_s=5025.0"

    status_line = stdout[manifest_header_index + 2]
    assert status_line == "  statuses: failure=6, success=10044"


def test_manifest_missing_directory_read_only(tmp_path, monkeypatch, capsys) -> None:
    """CLI should warn when the manifest directory is absent without creating it."""

    _prepare_manifest_cli_stubs(monkeypatch)

    from DocsToKG.DocParsing.core import cli

    read_only_root = tmp_path / "readonly"
    read_only_root.mkdir()
    manifests_dir = read_only_root / "Manifests"

    def _fail_iter(*_args, **_kwargs):
        raise AssertionError("iter_manifest_entries should not run when manifests are missing")

    monkeypatch.setattr(cli, "iter_manifest_entries", _fail_iter)
    monkeypatch.setattr(cli, "known_stages", ["chunk"], raising=False)
    monkeypatch.setattr(cli, "known_stage_set", {"chunk"}, raising=False)

    read_only_root.chmod(0o555)
    try:
        exit_code = cli.manifest(["--data-root", str(read_only_root)])
    finally:
        read_only_root.chmod(0o755)

    assert exit_code == 0
    assert not manifests_dir.exists()

    output = capsys.readouterr()
    assert "No manifest directory found" in output.out


def test_manifest_read_only_root_existing_manifests(tmp_path, monkeypatch, capsys) -> None:
    """CLI should read manifests without creating directories when root is read-only."""

    _prepare_manifest_cli_stubs(monkeypatch)

    from DocsToKG.DocParsing.core import cli

    read_only_root = tmp_path / "readonly"
    manifests_dir = read_only_root / "Manifests"
    manifests_dir.mkdir(parents=True)

    stage_name = "chunk"
    manifest_path = manifests_dir / f"docparse.{stage_name}.manifest.jsonl"
    entries = [
        {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "stage": stage_name,
            "doc_id": "doc-1",
            "status": "success",
            "duration_s": 1.23,
        },
        {
            "timestamp": "2025-01-01T00:01:00+00:00",
            "stage": stage_name,
            "doc_id": "doc-2",
            "status": "failure",
            "duration_s": 2.34,
            "error": "boom",
        },
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "known_stages", [stage_name], raising=False)
    monkeypatch.setattr(cli, "known_stage_set", {stage_name}, raising=False)

    expected_contents = sorted(path.name for path in read_only_root.iterdir())

    read_only_root.chmod(0o555)
    manifests_dir.chmod(0o555)
    try:
        exit_code = cli.manifest(["--data-root", str(read_only_root), "--tail", "1"])
    finally:
        manifests_dir.chmod(0o755)
        read_only_root.chmod(0o755)

    assert exit_code == 0
    assert sorted(path.name for path in read_only_root.iterdir()) == expected_contents

    stdout = capsys.readouterr().out.splitlines()
    assert stdout[0] == "docparse manifest tail (last 1 entries)"
    assert "doc-2" in stdout[1]
    assert "status=failure" in stdout[1]
