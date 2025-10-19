"""Tests for chunk validation telemetry metadata enrichment."""

from __future__ import annotations

import json
import sys
import types

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
    huggingface_stub = types.ModuleType(
        "docling_core.transforms.chunker.tokenizer.huggingface"
    )

    class _HuggingFaceTokenizer:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs) -> None:
            self.tokenizer = kwargs.get("tokenizer")

    huggingface_stub.HuggingFaceTokenizer = _HuggingFaceTokenizer
    sys.modules[
        "docling_core.transforms.chunker.tokenizer.huggingface"
    ] = huggingface_stub

from DocsToKG.DocParsing.chunking import runtime
from DocsToKG.DocParsing.io import resolve_hash_algorithm
from DocsToKG.DocParsing.logging import get_logger
from DocsToKG.DocParsing.telemetry import StageTelemetry, TelemetrySink


def test_validate_chunk_files_failure_logs_status_and_hash_alg(tmp_path) -> None:
    """Invalid chunk files should record status and hash algorithm metadata."""

    chunk_file = tmp_path / "invalid.chunks.jsonl"
    chunk_file.write_text("{\"bad\": }\n", encoding="utf-8")

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
