"""Validate telemetry enrichment emitted during chunk validation flows.

Chunk validation generates structured telemetry that downstream monitoring expects.
These tests construct lightweight tokenizer stubs and inspect emitted telemetry
records to ensure metadata such as chunk counts, token spans, and checksum
details are captured. By mocking third-party dependencies we verify enrichment
logic without requiring GPU or Docling assets.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - test stub
            return cls()

    transformers_stub.AutoTokenizer = _AutoTokenizer
    transformers_stub.PreTrainedTokenizerBase = type(
        "PreTrainedTokenizerBase",
        (),
        {},
    )
    sys.modules["transformers"] = transformers_stub

if "docling_core.transforms.chunker.tokenizer.huggingface" not in sys.modules:
    huggingface_stub = types.ModuleType("docling_core.transforms.chunker.tokenizer.huggingface")

    class _HuggingFaceTokenizer:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs) -> None:
            self.tokenizer = kwargs.get("tokenizer")

    huggingface_stub.HuggingFaceTokenizer = _HuggingFaceTokenizer
    sys.modules["docling_core.transforms.chunker.tokenizer.huggingface"] = huggingface_stub

from DocsToKG.DocParsing.chunking import runtime
from DocsToKG.DocParsing.io import resolve_hash_algorithm
from DocsToKG.DocParsing.logging import get_logger
from DocsToKG.DocParsing.telemetry import StageTelemetry, TelemetrySink


def test_validate_chunk_files_failure_logs_status_and_hash_alg(tmp_path) -> None:
    """Invalid chunk files should record status and hash algorithm metadata."""

    chunk_file = tmp_path / "invalid.chunks.jsonl"
    chunk_file.write_text('{"bad": }\n', encoding="utf-8")

    attempts_path = tmp_path / "attempts.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    telemetry = StageTelemetry(
        TelemetrySink(attempts_path, manifest_path),
        run_id="test-run",
        stage="chunking",
    )
    logger = get_logger("test.validate_chunk_files", level="CRITICAL")

    stats = runtime._validate_chunk_files(
        [chunk_file],
        logger,
        data_root=tmp_path,
        telemetry=telemetry,
    )

    assert stats["quarantined"] == 1

    attempt_entries = attempts_path.read_text(encoding="utf-8").strip().splitlines()
    assert attempt_entries, "expected an attempt entry to be written"
    attempt_record = json.loads(attempt_entries[0])
    expected_hash_alg = resolve_hash_algorithm()
    assert attempt_record["status"] == "failure"
    assert attempt_record["hash_alg"] == expected_hash_alg

    manifest_entries = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert manifest_entries, "expected a manifest entry to be written"
    manifest_record = json.loads(manifest_entries[0])
    assert manifest_record["status"] == "failure"
    assert manifest_record["hash_alg"] == expected_hash_alg


def test_log_skip_preserves_null_schema_version(tmp_path) -> None:
    """Skip manifests should preserve ``None`` schema versions and metadata."""

    attempts_path = tmp_path / "attempts.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    telemetry = StageTelemetry(
        TelemetrySink(attempts_path, manifest_path),
        run_id="test-run",
        stage="chunking",
    )

    metadata = {
        "schema_version": None,
        "output_path": str(tmp_path / "skipped.jsonl"),
        "hash_alg": "sha256",
        "skip_reason": "unchanged input",
    }

    telemetry.log_skip(
        doc_id="doc-123",
        input_path=tmp_path / "input.jsonl",
        reason="unchanged input",
        metadata=metadata,
    )

    manifest_entries = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert manifest_entries, "expected a manifest entry to be written"
    manifest_record = json.loads(manifest_entries[0])

    assert "schema_version" in manifest_record, "schema_version field missing"
    assert manifest_record["schema_version"] is None
    assert manifest_record["hash_alg"] == metadata["hash_alg"]
    assert manifest_record["skip_reason"] == metadata["skip_reason"]


def test_stage_telemetry_supports_custom_writer(tmp_path) -> None:
    """Telemetry should allow injection of a custom writer callable."""

    attempts_path = tmp_path / "attempts.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    calls: list[tuple[Path, list[dict[str, Any]]]] = []

    def fake_writer(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
        payload = [dict(row) for row in rows]
        calls.append((path, payload))
        return len(payload)

    sink = TelemetrySink(attempts_path, manifest_path)
    telemetry = StageTelemetry(sink, run_id="run-1", stage="chunk", writer=fake_writer)

    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")

    telemetry.record_attempt(
        doc_id="doc-1",
        input_path=input_path,
        status="success",
        metadata={"extra": "attempt"},
    )
    telemetry.write_manifest(
        doc_id="doc-1",
        output_path=tmp_path / "output.jsonl",
        schema_version="v1",
        duration_s=0.1,
        metadata={"extra": "manifest"},
    )

    assert len(calls) == 2
    attempt_call, manifest_call = calls
    assert attempt_call[0] == attempts_path
    assert manifest_call[0] == manifest_path
    assert attempt_call[1][0]["extra"] == "attempt"
    assert manifest_call[1][0]["extra"] == "manifest"
