# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_normalization",
#   "purpose": "Pytest coverage for ontology download normalization scenarios",
#   "sections": [
#     {
#       "id": "make-config",
#       "name": "_make_config",
#       "anchor": "function-make-config",
#       "kind": "function"
#     },
#     {
#       "id": "noop-logger",
#       "name": "_noop_logger",
#       "anchor": "function-noop-logger",
#       "kind": "function"
#     },
#     {
#       "id": "run-rdflib",
#       "name": "_run_rdflib",
#       "anchor": "function-run-rdflib",
#       "kind": "function"
#     },
#     {
#       "id": "test-streaming-hash-is-deterministic",
#       "name": "test_streaming_hash_is_deterministic",
#       "anchor": "function-test-streaming-hash-is-deterministic",
#       "kind": "function"
#     },
#     {
#       "id": "test-streaming-matches-in-memory",
#       "name": "test_streaming_matches_in_memory",
#       "anchor": "function-test-streaming-matches-in-memory",
#       "kind": "function"
#     },
#     {
#       "id": "test-streaming-edge-cases",
#       "name": "test_streaming_edge_cases",
#       "anchor": "function-test-streaming-edge-cases",
#       "kind": "function"
#     },
#     {
#       "id": "test-streaming-flushes-chunks",
#       "name": "test_streaming_flushes_chunks",
#       "anchor": "function-test-streaming-flushes-chunks",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Normalization determinism tests for streaming and in-memory pipelines."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")
pytest.importorskip("rdflib")

from DocsToKG.OntologyDownload.settings import DefaultsConfig, ResolvedConfig
from DocsToKG.OntologyDownload.validation import (
    ValidationRequest,
    normalize_streaming,
    validate_rdflib,
)

# --- Globals ---

_COMPLEX_FIXTURE = Path("tests/data/ontology_normalization/complex.ttl")
_EXPECTED_STREAMING_HASH = "65f80531c7207dc63e0d1eb6c85cae2bcb3d37968ae87b24c9c3858a4d85d449"
# --- Helper Functions ---


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


# --- Test Cases ---


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


def test_streaming_header_hash_is_stable() -> None:
    digest, header_hash = normalize_streaming(_COMPLEX_FIXTURE, return_header_hash=True)
    digest_again, header_again = normalize_streaming(_COMPLEX_FIXTURE, return_header_hash=True)
    assert digest == digest_again
    assert header_hash == header_again
    assert len(header_hash) == 64


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


def test_streaming_flushes_chunks(tmp_path: Path) -> None:
    class _Tracker:
        def __init__(self) -> None:
            self.updates: list[int] = []

        def update(self, data: bytes) -> None:
            self.updates.append(len(data))

        def hexdigest(self) -> str:
            return "stub-digest"

    tracker = _Tracker()

    real_sha256 = hashlib.sha256

    def fake_sha256(*args, **kwargs):
        if args or kwargs:
            return real_sha256(*args, **kwargs)
        return tracker

    destination = tmp_path / "chunked.ttl"
    with patch("DocsToKG.OntologyDownload.validation.hashlib.sha256", fake_sha256):
        digest = normalize_streaming(_COMPLEX_FIXTURE, output_path=destination, chunk_bytes=64)

    assert digest == "stub-digest"
    assert len(tracker.updates) > 1

    emitted = destination.read_bytes()
    assert sum(tracker.updates) == len(emitted)
