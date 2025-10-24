import pytest

from DocsToKG.DocParsing.app_context import build_app_context
from DocsToKG.DocParsing.core.discovery import (
    configure_discovery_ignore,
    get_discovery_ignore_patterns,
    iter_chunks,
)


@pytest.fixture(autouse=True)
def reset_discovery_ignore():
    configure_discovery_ignore(None)
    yield
    configure_discovery_ignore(None)


def _touch_chunk(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")


def test_iter_chunks_skips_default_patterns(tmp_path):
    _touch_chunk(tmp_path / "root.chunks.jsonl")

    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    _touch_chunk(hidden_dir / "ignored.chunks.jsonl")

    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    _touch_chunk(tmp_dir / "skipped.chunks.jsonl")

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    _touch_chunk(temp_dir / "ignored.chunks.jsonl")

    nested_ok = tmp_path / "nested" / "ok"
    nested_ok.mkdir(parents=True)
    _touch_chunk(nested_ok / "kept.chunks.jsonl")

    pycache_dir = tmp_path / "nested" / "__pycache__"
    pycache_dir.mkdir(parents=True)
    _touch_chunk(pycache_dir / "ignored.chunks.jsonl")

    seen = {entry.logical_path.as_posix() for entry in iter_chunks(tmp_path)}

    assert seen == {"root.chunks.jsonl", "nested/ok/kept.chunks.jsonl"}


def test_iter_chunks_honours_env_override(monkeypatch, tmp_path):
    _touch_chunk(tmp_path / "keep" / "value.chunks.jsonl")
    _touch_chunk(tmp_path / "special" / "skip.chunks.jsonl")

    monkeypatch.setenv("DOCSTOKG_DISCOVERY_IGNORE", "special")

    seen = {entry.logical_path.as_posix() for entry in iter_chunks(tmp_path)}

    assert seen == {"keep/value.chunks.jsonl"}


def test_build_app_context_configures_discovery_ignore(tmp_path):
    _touch_chunk(tmp_path / "keep.chunks.jsonl")
    _touch_chunk(tmp_path / "special" / "skip.chunks.jsonl")

    build_app_context(data_root=tmp_path, discovery_ignore=("special",))

    patterns = get_discovery_ignore_patterns()
    assert "special" in patterns
    assert "tmp" in patterns  # defaults are preserved

    seen = {entry.logical_path.as_posix() for entry in iter_chunks(tmp_path)}
    assert seen == {"keep.chunks.jsonl"}
