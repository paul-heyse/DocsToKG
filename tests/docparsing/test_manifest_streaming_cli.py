"""Regression tests for manifest CLI streaming behaviour."""

from __future__ import annotations

import json
import logging
import tracemalloc
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from DocsToKG.DocParsing.env import data_manifests
from DocsToKG.DocParsing.io import iter_manifest_entries

from tests.docparsing.stubs import dependency_stubs


@pytest.mark.parametrize("tail", [7])
def test_manifest_streams_large_tail(monkeypatch, tmp_path, capsys, tail: int) -> None:
    """Ensure ``manifest`` streams large iterators while keeping tail accuracy."""

    total_entries = 10_050
    stage_name = "chunk"

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
    monkeypatch.setattr(cli, "data_manifests", lambda data_root: tmp_path)

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


def test_iter_manifest_entries_streams_large_manifest(tmp_path: Path) -> None:
    """Iterating over large manifests should remain memory efficient."""

    stages = ["doctags", "chunk", "embeddings"]
    entries_per_stage = 60_000
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)

    manifest_dir = data_manifests(tmp_path)
    for offset, stage in enumerate(stages):
        stage_path = manifest_dir / f"docparse.{stage}.manifest.jsonl"
        with stage_path.open("w", encoding="utf-8") as handle:
            for index in range(entries_per_stage):
                timestamp = (base + timedelta(seconds=index + offset * entries_per_stage)).isoformat()
                payload = {
                    "timestamp": timestamp,
                    "stage": stage,
                    "doc_id": f"{stage}-doc-{index}",
                    "status": "success",
                    "duration_s": 0.1,
                }
                handle.write(json.dumps(payload) + "\n")

    tracemalloc.start()
    total_entries = 0
    previous_timestamp = ""
    for entry in iter_manifest_entries(stages, tmp_path):
        assert entry["stage"] in stages
        current_timestamp = entry.get("timestamp", "")
        assert current_timestamp >= previous_timestamp
        previous_timestamp = current_timestamp
        total_entries += 1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert total_entries == entries_per_stage * len(stages)
    assert peak < 32_000_000
