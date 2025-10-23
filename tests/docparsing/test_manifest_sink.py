"""Regression tests for ``JsonlManifestSink`` rotation and compaction."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if "filelock" not in sys.modules:
    class _TestFileLock:
        def __init__(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial stub
            pass

        def __enter__(self) -> "_TestFileLock":  # pragma: no cover - trivial stub
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: types.TracebackType | None,
        ) -> bool:  # pragma: no cover - trivial stub
            return False

    sys.modules["filelock"] = types.SimpleNamespace(FileLock=_TestFileLock)

MANIFEST_SINK_PATH = SRC_ROOT / "DocsToKG" / "DocParsing" / "core" / "manifest_sink.py"
SPEC = importlib.util.spec_from_file_location("manifest_sink", MANIFEST_SINK_PATH)
assert SPEC and SPEC.loader  # pragma: no cover - sanity check for loader availability
manifest_sink = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest_sink
SPEC.loader.exec_module(manifest_sink)

JsonlManifestSink = manifest_sink.JsonlManifestSink


def _write_success_entries(sink: JsonlManifestSink, count: int) -> None:
    for idx in range(count):
        sink.log_success(
            stage="chunk",
            item_id=f"doc-{idx}",
            input_path=f"input-{idx}.json",
            output_paths={"chunks": Path(f"chunks-{idx}.json")},
            duration_s=0.1,
            schema_version="v1",
            extras={"worker": "test"},
        )


def test_rotate_if_needed_creates_snapshot(tmp_path: Path) -> None:
    manifest_path = tmp_path / "docparse.manifest.jsonl"
    sink = JsonlManifestSink(manifest_path)

    _write_success_entries(sink, count=10)

    size_before = manifest_path.stat().st_size

    result = sink.rotate_if_needed(max_bytes=size_before - 1)

    assert result is not None
    assert result.bytes_before == size_before
    assert result.entry_count == 10
    assert result.rotated_path.exists()

    with open(result.rotated_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    assert len(lines) == 10
    parsed = [json.loads(line) for line in lines]
    assert {entry["doc_id"] for entry in parsed} == {f"doc-{i}" for i in range(10)}

    # Original manifest should have been recreated and ready for appends.
    assert manifest_path.exists()
    assert manifest_path.stat().st_size == 0

    sink.log_skip(
        stage="chunk",
        item_id="doc-10",
        input_path="input-10.json",
        output_path="chunks-10.json",
        duration_s=0.01,
        schema_version="v1",
    )

    with open(manifest_path, "r", encoding="utf-8") as handle:
        skip_entry = json.loads(handle.readline())

    assert skip_entry["status"] == "skip"
    assert skip_entry["doc_id"] == "doc-10"


def test_rotate_with_compaction_deduplicates_entries(tmp_path: Path) -> None:
    manifest_path = tmp_path / "docparse.manifest.jsonl"
    sink = JsonlManifestSink(manifest_path)

    for idx in range(5):
        sink.log_success(
            stage="chunk",
            item_id="duplicate-doc",
            input_path=f"input-{idx}.json",
            output_paths={"chunks": Path(f"chunks-{idx}.json")},
            duration_s=idx,
            schema_version="v1",
            extras={"revision": idx},
        )

    size_before = manifest_path.stat().st_size
    result = sink.rotate_if_needed(max_bytes=size_before - 1, compact=True)

    assert result is not None
    assert result.compacted_path is not None

    with open(result.compacted_path, "r", encoding="utf-8") as handle:
        compacted_lines = [json.loads(line) for line in handle]

    assert len(compacted_lines) == 1
    entry = compacted_lines[0]
    assert entry["doc_id"] == "duplicate-doc"
    assert entry["extras"]["revision"] == 4

    with open(result.rotated_path, "r", encoding="utf-8") as handle:
        rotated_lines = [json.loads(line) for line in handle]

    assert len(rotated_lines) == 5
    assert rotated_lines[-1]["extras"]["revision"] == 4
