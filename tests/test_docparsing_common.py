"""Unit tests for ``DocsToKG.DocParsing._common`` utilities."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from DocsToKG.DocParsing import _common


@pytest.fixture(autouse=True)
def _reset_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Force utilities to operate inside a temporary data root."""

    data_root = tmp_path / "Data"
    data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(data_root))
    yield
    monkeypatch.delenv("DOCSTOKG_DATA_ROOT", raising=False)


def test_detect_data_root_env(tmp_path: Path) -> None:
    data = tmp_path / "Data"
    data.mkdir(exist_ok=True)
    root = _common.detect_data_root()
    assert root == data.resolve()


def test_detect_data_root_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCSTOKG_DATA_ROOT", raising=False)
    project = tmp_path / "workspace" / "DocsToKG" / "src"
    target = tmp_path / "workspace" / "DocsToKG" / "Data"
    project.mkdir(parents=True)
    (target / "DocTagsFiles").mkdir(parents=True)
    monkeypatch.chdir(project)
    resolved = _common.detect_data_root()
    assert resolved == target.resolve()


def test_detect_data_root_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCSTOKG_DATA_ROOT", raising=False)
    start = tmp_path / "no_data_here"
    start.mkdir()
    resolved = _common.detect_data_root(start)
    assert resolved == (start / "Data").resolve()


def test_data_directories_created(tmp_path: Path) -> None:
    expected = {
        _common.data_doctags(): "DocTagsFiles",
        _common.data_chunks(): "ChunkedDocTagFiles",
        _common.data_vectors(): "Vectors",
        _common.data_manifests(): "Manifests",
        _common.data_pdfs(): "PDFs",
        _common.data_html(): "HTML",
    }
    for path, folder in expected.items():
        assert folder in str(path)
        assert path.exists()


def test_get_logger_idempotent() -> None:
    logger1 = _common.get_logger("docparse-test")
    logger2 = _common.get_logger("docparse-test")
    assert logger1 is logger2


def test_find_free_port_basic() -> None:
    port = _common.find_free_port(start=9000, span=4)
    assert isinstance(port, int)


def test_find_free_port_fallback() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    busy_port = sock.getsockname()[1]
    try:
        result = _common.find_free_port(start=busy_port, span=1)
        assert result != busy_port
    finally:
        sock.close()


def test_atomic_write_success(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    with _common.atomic_write(target) as handle:
        handle.write("data")
    assert target.read_text(encoding="utf-8") == "data"


def test_atomic_write_failure(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    with pytest.raises(RuntimeError):
        with _common.atomic_write(target) as handle:
            handle.write("data")
            raise RuntimeError("boom")
    assert not target.exists()
    assert not target.with_suffix(".json.tmp").exists()


def test_iter_doctags(tmp_path: Path) -> None:
    doctags_dir = _common.data_doctags()
    target = Path(doctags_dir)
    files = [target / "a.doctags", target / "b.doctag"]
    for file in files:
        file.write_text("content", encoding="utf-8")
    results = list(_common.iter_doctags(target))
    assert results == sorted(file.resolve() for file in files)


def test_iter_chunks(tmp_path: Path) -> None:
    chunks_dir = _common.data_chunks()
    good = chunks_dir / "doc.chunks.jsonl"
    other = chunks_dir / "doc.jsonl"
    good.write_text("{}\n", encoding="utf-8")
    other.write_text("{}\n", encoding="utf-8")
    results = list(_common.iter_chunks(chunks_dir))
    assert results == [good.resolve()]


def test_jsonl_load_and_save(tmp_path: Path) -> None:
    target = tmp_path / "example.jsonl"
    _common.jsonl_save(target, [{"a": 1}, {"b": 2}])
    rows = _common.jsonl_load(target)
    assert rows == [{"a": 1}, {"b": 2}]


def test_jsonl_load_skip_invalid(tmp_path: Path) -> None:
    target = tmp_path / "bad.jsonl"
    target.write_text("{\ninvalid\n", encoding="utf-8")
    rows = _common.jsonl_load(target, skip_invalid=True, max_errors=1)
    assert rows == []


def test_jsonl_save_validation_error(tmp_path: Path) -> None:
    target = tmp_path / "example.jsonl"

    def validate(row: dict) -> None:
        raise ValueError("bad row")

    with pytest.raises(ValueError, match="Validation failed"):
        _common.jsonl_save(target, [{"a": 1}], validate=validate)
    assert not target.exists()


def test_batcher() -> None:
    assert list(_common.Batcher([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(_common.Batcher([], 3)) == []


def test_manifest_append(tmp_path: Path) -> None:
    manifest = _common.data_manifests() / "docparse.manifest.jsonl"
    _common.manifest_append("chunk", "doc-1", "success", duration_s=1.23)
    content = manifest.read_text(encoding="utf-8").strip()
    record = json.loads(content)
    assert record["doc_id"] == "doc-1"
    assert record["status"] == "success"


def test_manifest_append_invalid_status() -> None:
    with pytest.raises(ValueError):
        _common.manifest_append("chunk", "doc-1", "bad")


def test_compute_content_hash(tmp_path: Path) -> None:
    target = tmp_path / "hash.txt"
    target.write_text("hello", encoding="utf-8")
    digest = _common.compute_content_hash(target)
    assert digest == hashlib.sha1(b"hello").hexdigest()


def test_acquire_lock_success(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    with _common.acquire_lock(target):
        assert target.with_suffix(".txt.lock").exists()
    assert not target.with_suffix(".txt.lock").exists()


def test_acquire_lock_timeout(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    lock = target.with_suffix(".txt.lock")
    lock.write_text("123", encoding="utf-8")
    monkeypatch.setattr(_common, "_pid_is_running", lambda pid: True)
    with pytest.raises(TimeoutError):
        with _common.acquire_lock(target, timeout=0.1):
            pass
    lock.unlink()


def test_acquire_lock_stale_lock_cleanup(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    lock = target.with_suffix(".txt.lock")
    lock.write_text("123", encoding="utf-8")
    monkeypatch.setattr(_common, "_pid_is_running", lambda pid: False)
    with _common.acquire_lock(target, timeout=0.1):
        assert lock.exists()
