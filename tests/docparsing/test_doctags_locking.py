"""Regression tests for DocTags locking helpers."""

from __future__ import annotations

from pathlib import Path

from DocsToKG.DocParsing.core.concurrency import _acquire_lock, safe_write


def test_doctags_import_and_lock_round_trip(tmp_path) -> None:
    """Import DocTags and ensure legacy locking helper interoperates."""

    # Import inside the test to mirror real usage and detect import regressions.
    from DocsToKG.DocParsing import doctags  # noqa: F401  (import smoke test)

    out_path = Path(tmp_path) / "sample.json"
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")

    with _acquire_lock(out_path, timeout=1.0):
        assert not out_path.exists()
        out_path.write_text("payload")

    assert out_path.read_text() == "payload"
    assert not lock_path.exists(), "lock file should be cleaned up after context exit"

    # ``safe_write`` should detect the existing file and respect ``skip_if_exists``
    wrote = safe_write(out_path, lambda: out_path.write_text("new"), skip_if_exists=True)
    assert wrote is False
    assert out_path.read_text() == "payload"

    out_path.unlink()
    wrote = safe_write(out_path, lambda: out_path.write_text("replacement"), skip_if_exists=True)
    assert wrote is True
    assert out_path.read_text() == "replacement"
