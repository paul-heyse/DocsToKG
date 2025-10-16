"""Normalization determinism tests for streaming and in-memory pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")
pytest.importorskip("rdflib")

from DocsToKG.OntologyDownload import DefaultsConfig, ResolvedConfig, ValidationRequest
from DocsToKG.OntologyDownload.ontology_download import normalize_streaming, validate_rdflib

_COMPLEX_FIXTURE = Path("tests/data/ontology_normalization/complex.ttl")
_EXPECTED_STREAMING_HASH = "a4455411fb31c754effffaf74218f21304c64a8e6c9a0c72634c2af45fa29bb4"


def _make_config(threshold_mb: int = 2048) -> ResolvedConfig:
    defaults = DefaultsConfig()
    defaults.validation = defaults.validation.model_copy(
        update={"streaming_normalization_threshold_mb": threshold_mb}
    )
    return ResolvedConfig(defaults=defaults, specs=[])


def _noop_logger() -> logging.Logger:
    logger = logging.getLogger("ontology-normalization-test")
    logger.setLevel(logging.INFO)
    return logger


def _run_rdflib(path: Path, tmp_path: Path, threshold_mb: int) -> dict[str, object]:
    config = _make_config(threshold_mb)
    request = ValidationRequest(
        "rdflib",
        path,
        tmp_path / "normalized",
        tmp_path / "validation",
        config,
    )
    result = validate_rdflib(request, _noop_logger())
    assert result.ok
    return result.details


def test_streaming_hash_is_deterministic(tmp_path: Path) -> None:
    hashes = []
    outputs = []
    for iteration in range(5):
        destination = tmp_path / f"stream-{iteration}.ttl"
        digest = normalize_streaming(_COMPLEX_FIXTURE, output_path=destination)
        hashes.append(digest)
        outputs.append(destination.read_bytes())
    assert len(set(hashes)) == 1
    assert len({content for content in outputs}) == 1
    if _EXPECTED_STREAMING_HASH:
        assert hashes[0] == _EXPECTED_STREAMING_HASH
    else:
        pytest.skip("Expected hash value not recorded yet")


def test_streaming_matches_in_memory(tmp_path: Path) -> None:
    in_memory = _run_rdflib(_COMPLEX_FIXTURE, tmp_path / "memory", threshold_mb=4096)
    streaming = _run_rdflib(_COMPLEX_FIXTURE, tmp_path / "stream", threshold_mb=1)
    assert streaming["normalization_mode"] == "streaming"
    assert in_memory["normalization_mode"] == "in-memory"
    normalized_stream = tmp_path / "stream" / "normalized"
    stream_files = list(normalized_stream.glob("*.ttl"))
    assert stream_files, "streaming normalization did not emit TTL output"
    direct_stream_hash = normalize_streaming(_COMPLEX_FIXTURE)
    assert streaming.get("streaming_nt_sha256") == direct_stream_hash
    direct_stream_hash = normalize_streaming(_COMPLEX_FIXTURE)
    assert streaming.get("streaming_nt_sha256") == direct_stream_hash


@pytest.mark.parametrize(
    "content",
    [
        "@prefix ex: <http://example.org/> .\n",  # empty graph with prefix
        "@prefix ex: <http://example.org/> .\n\nex:a ex:b ex:c .\n",  # single triple
        "@prefix ex: <http://example.org/> .\n\n[] ex:b ex:c .\n",  # blank node subject
    ],
)
def test_streaming_edge_cases(tmp_path: Path, content: str) -> None:
    source = tmp_path / "source.ttl"
    source.write_text(content, encoding="utf-8")
    streaming = _run_rdflib(source, tmp_path / "stream", threshold_mb=0)
    stream_hash = normalize_streaming(source)
    assert streaming.get("streaming_nt_sha256") == stream_hash


def test_streaming_flushes_chunks(monkeypatch, tmp_path: Path) -> None:
    class _Tracker:
        def __init__(self) -> None:
            self.updates: list[int] = []

        def update(self, data: bytes) -> None:
            self.updates.append(len(data))

        def hexdigest(self) -> str:
            return "stub-digest"

    tracker = _Tracker()

    monkeypatch.setattr(
        "DocsToKG.OntologyDownload.ontology_download.hashlib.sha256", lambda: tracker
    )

    destination = tmp_path / "chunked.ttl"
    digest = normalize_streaming(
        _COMPLEX_FIXTURE, output_path=destination, chunk_bytes=64
    )

    assert digest == "stub-digest"
    assert len(tracker.updates) > 1

    emitted = destination.read_bytes()
    assert sum(tracker.updates) == len(emitted)
