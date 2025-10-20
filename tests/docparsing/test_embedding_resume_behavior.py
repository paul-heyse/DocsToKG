"""Exercise embedding resume logic, hashing, and manifest safeguards.

The embedding stage should skip work when manifests already contain matching
hashes while still recomputing BM25 statistics and vectors when inputs change.
These integration-style tests construct chunk fixtures, invoke the runtime, and
validate that resume guards, manifest paths, and hash computations behave as
expected across successive runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pytest

from DocsToKG.DocParsing.core import BM25Stats
from DocsToKG.DocParsing.io import compute_content_hash, resolve_manifest_path
from tests.conftest import PatchManager

_DOC_ID = "doc-1.doctags"


def _write_chunk_file(path: Path) -> None:
    rows = [
        {
            "schema_version": "docparse/1.0.0",
            "uuid": "chunk-1",
            "doc_id": "doc-1",
            "text": "hello world",
            "num_tokens": 2,
            "source_path": "doc-1.doctags",
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _configure_runtime(
    patcher: PatchManager, tmp_path: Path, vector_format: str
) -> Tuple[object, Path, Path, Path, Path, Dict[str, int]]:
    import DocsToKG.DocParsing.embedding.runtime as runtime

    data_root = tmp_path / "Data"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    vectors_dir = data_root / "Embeddings"
    model_root = tmp_path / "models"
    qwen_dir = model_root / "qwen"
    splade_dir = model_root / "splade"
    qwen_dir.mkdir(parents=True)
    splade_dir.mkdir(parents=True)
    chunks_dir.mkdir(parents=True)
    vectors_dir.mkdir(parents=True)

    chunk_file = chunks_dir / "doc-1.chunks.jsonl"
    _write_chunk_file(chunk_file)

    patcher.setattr(runtime, "ensure_model_environment", lambda: (model_root, model_root))

    def _expand_path(candidate: object) -> Path:
        if candidate is None:
            return model_root
        return Path(candidate).expanduser().resolve()

    patcher.setattr(runtime, "expand_path", _expand_path)
    patcher.setattr(runtime, "_resolve_qwen_dir", lambda _root: qwen_dir)
    patcher.setattr(runtime, "_resolve_splade_dir", lambda _root: splade_dir)
    patcher.setattr(runtime, "ensure_splade_environment", lambda **_: {"device": "cpu"})
    patcher.setattr(
        runtime, "ensure_qwen_environment", lambda **_: {"device": "cpu", "dtype": "fp16"}
    )
    patcher.setattr(runtime, "_ensure_splade_dependencies", lambda: None)
    patcher.setattr(runtime, "_ensure_qwen_dependencies", lambda: None)
    patcher.setattr(runtime, "prepare_data_root", lambda override, detected: data_root)
    patcher.setattr(runtime, "detect_data_root", lambda: data_root)
    patcher.setattr(runtime, "data_chunks", lambda _root, ensure=False: chunks_dir)
    patcher.setattr(runtime, "data_vectors", lambda _root, ensure=False: vectors_dir)
    patcher.setattr(
        runtime,
        "resolve_pipeline_path",
        lambda cli_value, default_path, **_: cli_value or default_path,
    )
    patcher.setattr(
        runtime,
        "process_pass_a",
        lambda _files, _logger: BM25Stats(N=1, avgdl=1.0, df={}),
    )

    counters = {"hash": 0, "process": 0}
    original_hash = runtime.compute_content_hash

    def _counting_hash(path: Path) -> str:
        counters["hash"] += 1
        return original_hash(path)

    patcher.setattr(runtime, "compute_content_hash", _counting_hash)

    def _stub_process_chunk_file_vectors(
        chunk_file: Path,
        out_path: Path,
        stats,
        args,
        validator,
        logger,
        *,
        content_hasher=None,
    ):
        counters["process"] += 1
        rows = []
        with chunk_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if content_hasher is not None:
                    content_hasher.update(raw_line)
                line = raw_line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vector_rows = [
            {
                "UUID": row.get("uuid", ""),
                "BM25": {
                    "terms": ["hello", "world"],
                    "weights": [1.0, 1.0],
                    "avgdl": 1.0,
                    "N": 1,
                },
                "SPLADEv3": {
                    "tokens": ["hello", "world"],
                    "weights": [0.5, 0.4],
                },
                "Qwen3-4B": {
                    "model_id": "stub",
                    "vector": [0.0, 0.0],
                    "dimension": 2,
                },
                "model_metadata": {},
                "schema_version": runtime.VECTOR_SCHEMA_VERSION,
            }
            for row in rows
        ]
        writer = runtime.create_vector_writer(
            out_path, str(getattr(args, "vector_format", vector_format))
        )
        with writer:
            writer.write_rows(vector_rows)
        count = len(rows)
        return count, [0] * count, [1.0] * count

    patcher.setattr(runtime, "process_chunk_file_vectors", _stub_process_chunk_file_vectors)

    return runtime, data_root, chunks_dir, vectors_dir, chunk_file, counters


def _read_manifest_entries(stage: str, root: Path) -> list[dict]:
    manifest_path = resolve_manifest_path(stage, root)
    if not manifest_path.exists():
        return []
    entries = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_embed_without_resume_streams_hash(
    patcher: PatchManager, tmp_path: Path, vector_format: str
) -> None:
    runtime, data_root, chunks_dir, vectors_dir, chunk_file, counters = _configure_runtime(
        patcher, tmp_path, vector_format
    )

    base_args = [
        "--data-root",
        str(data_root),
        "--chunks-dir",
        str(chunks_dir),
        "--out-dir",
        str(vectors_dir),
        "--qwen-dim",
        "2",
        "--batch-size-qwen",
        "1",
        "--batch-size-splade",
        "1",
        "--format",
        vector_format,
    ]

    exit_code = runtime.main(base_args)

    assert exit_code == 0
    assert counters["process"] == 1
    assert counters["hash"] == 0

    entries = _read_manifest_entries(runtime.MANIFEST_STAGE, data_root)
    success_entries = [
        entry
        for entry in entries
        if entry.get("doc_id") == _DOC_ID and entry.get("status") == "success"
    ]
    assert success_entries, "Expected a success manifest entry for doc-1"
    recorded_hash = success_entries[-1]["input_hash"]
    expected_hash = compute_content_hash(chunk_file)
    assert recorded_hash == expected_hash
    assert success_entries[-1]["vector_format"] == vector_format
    suffix = ".vectors.parquet" if vector_format == "parquet" else ".vectors.jsonl"
    vector_path = vectors_dir / f"doc-1{suffix}"
    assert vector_path.exists()


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_embed_resume_skips_unchanged(
    patcher: PatchManager, tmp_path: Path, vector_format: str
) -> None:
    runtime, data_root, chunks_dir, vectors_dir, chunk_file, counters = _configure_runtime(
        patcher, tmp_path, vector_format
    )

    base_args = [
        "--data-root",
        str(data_root),
        "--chunks-dir",
        str(chunks_dir),
        "--out-dir",
        str(vectors_dir),
        "--qwen-dim",
        "2",
        "--batch-size-qwen",
        "1",
        "--batch-size-splade",
        "1",
        "--format",
        vector_format,
    ]
    assert runtime.main(base_args) == 0

    counters["hash"] = 0
    counters["process"] = 0

    assert runtime.main(base_args + ["--resume"]) == 0
    assert counters["hash"] == 1
    assert counters["process"] == 0

    entries = _read_manifest_entries(runtime.MANIFEST_STAGE, data_root)
    skip_entries = [
        entry
        for entry in entries
        if entry.get("doc_id") == _DOC_ID and entry.get("status") == "skip"
    ]
    assert skip_entries, "Expected a skip manifest entry after resume"
    expected_hash = compute_content_hash(chunk_file)
    assert skip_entries[-1]["input_hash"] == expected_hash
    assert skip_entries[-1]["vector_format"] == vector_format


@pytest.mark.parametrize("vector_format", ["jsonl", "parquet"])
def test_embed_validate_only_respects_format(
    patcher: PatchManager, tmp_path: Path, vector_format: str
) -> None:
    runtime, data_root, chunks_dir, vectors_dir, _chunk_file, _ = _configure_runtime(
        patcher, tmp_path, vector_format
    )

    base_args = [
        "--data-root",
        str(data_root),
        "--chunks-dir",
        str(chunks_dir),
        "--out-dir",
        str(vectors_dir),
        "--qwen-dim",
        "2",
        "--batch-size-qwen",
        "1",
        "--batch-size-splade",
        "1",
        "--format",
        vector_format,
    ]
    assert runtime.main(base_args) == 0

    validate_args = [
        "--data-root",
        str(data_root),
        "--chunks-dir",
        str(chunks_dir),
        "--out-dir",
        str(vectors_dir),
        "--validate-only",
        "--format",
        vector_format,
    ]

    exit_code = runtime.main(validate_args)
    assert exit_code == 0

    entries = _read_manifest_entries(runtime.MANIFEST_STAGE, data_root)
    validate_entries = [
        entry
        for entry in entries
        if entry.get("status") == "validate-only" and entry.get("vector_format") == vector_format
    ]
    assert validate_entries, "Expected validate-only manifest entry recording vector format"
