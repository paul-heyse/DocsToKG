from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

filelock_module = pytest.importorskip("filelock")
FileLock = filelock_module.FileLock

manifest_sink_path = SRC_DIR / "DocsToKG" / "DocParsing" / "core" / "manifest_sink.py"
spec = importlib.util.spec_from_file_location("manifest_sink", manifest_sink_path)
assert spec and spec.loader, "Failed to load manifest_sink module"
manifest_sink = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = manifest_sink
spec.loader.exec_module(manifest_sink)

JsonlManifestSink = manifest_sink.JsonlManifestSink
ManifestLockTimeoutError = manifest_sink.ManifestLockTimeoutError


def test_manifest_sink_uses_environment_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCSTOKG_MANIFEST_LOCK_TIMEOUT", "45.5")
    sink = JsonlManifestSink(tmp_path / "manifest.jsonl")

    assert sink.lock_timeout_s == pytest.approx(45.5)


def test_manifest_sink_parameter_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCSTOKG_MANIFEST_LOCK_TIMEOUT", "12.0")
    sink = JsonlManifestSink(tmp_path / "manifest.jsonl", lock_timeout_s=0.25)

    assert sink.lock_timeout_s == pytest.approx(0.25)


def test_manifest_sink_timeout_error_contains_context(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    sink = JsonlManifestSink(manifest_path, lock_timeout_s=0.1)

    # Acquire the lock to force a timeout when the sink attempts to append.
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    held_lock = FileLock(str(lock_path))
    held_lock.acquire(timeout=1)

    try:
        with pytest.raises(ManifestLockTimeoutError) as excinfo:
            sink.log_success(
                stage="test",
                item_id="doc-1",
                input_path="/tmp/in",
                output_paths={"primary": manifest_path},
                duration_s=0.1,
                schema_version="v1",
            )

        error = excinfo.value
        assert error.lock_path == sink.lock_path
        assert error.timeout_s == pytest.approx(sink.lock_timeout_s)
        assert "increase the timeout" in error.hint
        assert "DOCSTOKG_MANIFEST_LOCK_TIMEOUT" in str(error)
    finally:
        held_lock.release()
