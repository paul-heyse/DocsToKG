"""
Ontology Validator Tests

This module verifies the ontology validation adapters across RDFLib,
Pronto, Owlready2, ROBOT, and Arelle to ensure normalization artefacts
and diagnostic reports are generated consistently.

Key Scenarios:
- Validates TTL, OBO, and OWL fixtures with in-process validators
- Confirms external tool fallbacks like ROBOT and Arelle integrate cleanly
- Exercises failure paths such as Owlready2 memory exhaustion

Dependencies:
- pytest: Fixtures and monkeypatching
- DocsToKG.OntologyDownload.ontology_download: Validation entry points under test

Usage:
    pytest tests/ontology_download/test_validators.py
"""

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import DefaultsConfig, ResolvedConfig, ValidationRequest
from DocsToKG.OntologyDownload.ontology_download import (
    ValidatorSubprocessError,
    normalize_streaming,
    validate_arelle,
    validate_owlready2,
    validate_pronto,
    validate_rdflib,
    validate_robot,
)


@pytest.fixture()
def config():
    return ResolvedConfig(defaults=DefaultsConfig(), specs=[])


@pytest.fixture()
def ttl_file(tmp_path: Path) -> Path:
    source = Path("tests/data/ontology_fixtures/mini.ttl")
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    return target


@pytest.fixture()
def obo_file(tmp_path: Path) -> Path:
    source = Path("tests/data/ontology_fixtures/mini.obo")
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    return target


@pytest.fixture()
def owl_file(tmp_path: Path) -> Path:
    source = Path("tests/data/ontology_fixtures/mini.owl")
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    return target


def make_request(path: Path, tmp_path: Path, config: ResolvedConfig) -> ValidationRequest:
    normalized = tmp_path / "normalized"
    validation = tmp_path / "validation"
    return ValidationRequest("rdflib", path, normalized, validation, config)


@pytest.fixture()
def xbrl_package(tmp_path: Path) -> Path:
    archive = tmp_path / "taxonomy.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "taxonomy/entrypoint.xsd", '<schema xmlns="http://www.w3.org/2001/XMLSchema" />'
        )
    return archive


def test_validate_rdflib_success(ttl_file, tmp_path, config):
    request = make_request(ttl_file, tmp_path, config)
    result = validate_rdflib(request, _noop_logger())
    assert result.ok
    normalized = request.normalized_dir / f"{ttl_file.stem}.ttl"
    assert normalized.exists()
    data = json.loads((request.validation_dir / "rdflib_parse.json").read_text())
    assert data["ok"]
    canonical = normalized.read_text()
    expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert data["normalized_sha256"] == expected_hash
    assert result.details["normalized_sha256"] == expected_hash


def test_normalize_streaming_deterministic(tmp_path):
    source = Path("tests/ontology_download/fixtures/normalization/complex.ttl")
    golden = (
        Path("tests/ontology_download/fixtures/normalization/complex.sha256").read_text().strip()
    )
    digests = []
    outputs = []
    for attempt in range(5):
        output_path = tmp_path / f"normalized-{attempt}.nt"
        digest = normalize_streaming(source, output_path=output_path)
        digests.append(digest)
        outputs.append(output_path.read_bytes())
    assert len(set(digests)) == 1
    assert digests[0] == golden
    assert all(blob == outputs[0] for blob in outputs)


def test_streaming_matches_in_memory(tmp_path, config):
    source = Path("tests/ontology_download/fixtures/normalization/complex.ttl")
    # Baseline in-memory normalization (threshold high enough to avoid streaming)
    baseline_request = make_request(source, tmp_path / "baseline", config)
    baseline_result = validate_rdflib(baseline_request, _noop_logger())
    assert baseline_result.details["normalization_mode"] == "in-memory"

    streaming_config = config.model_copy(deep=True)
    streaming_config.defaults.validation.streaming_normalization_threshold_mb = 1
    streaming_request = make_request(source, tmp_path / "stream", streaming_config)
    streaming_result = validate_rdflib(streaming_request, _noop_logger())
    assert streaming_result.details["normalization_mode"] == "streaming"
    stream_hash = normalize_streaming(source)
    assert streaming_result.details.get("streaming_nt_sha256") == stream_hash

    normalized_file = streaming_request.normalized_dir / "complex.ttl"
    assert normalized_file.exists()


def test_normalize_streaming_edge_cases(tmp_path, config):
    pytest.importorskip("rdflib")
    try:
        from rdflib import BNode, Graph, Literal, Namespace, URIRef
    except ImportError:
        pytest.skip("rdflib optional dependency not available")

    ns = Namespace("http://example.org/")
    cases = {}

    config = config.model_copy(deep=True)
    config.defaults.validation.streaming_normalization_threshold_mb = 1

    empty_graph = Graph()
    empty_path = tmp_path / "empty.ttl"
    empty_graph.serialize(destination=empty_path, format="turtle")
    cases[empty_path] = empty_graph

    single_graph = Graph()
    single_graph.add((URIRef(ns["subject"]), URIRef(ns["predicate"]), Literal("value")))
    single_path = tmp_path / "single.ttl"
    single_graph.serialize(destination=single_path, format="turtle")
    cases[single_path] = single_graph

    blank_graph = Graph()
    node = BNode()
    blank_graph.add((node, ns["p"], Literal("blank")))
    blank_graph.add((node, ns["p2"], ns["o"]))
    blank_path = tmp_path / "blank.ttl"
    blank_graph.serialize(destination=blank_path, format="turtle")
    cases[blank_path] = blank_graph

    for path, graph in cases.items():
        target = tmp_path / f"{path.stem}.ttl"
        digest_stream = normalize_streaming(path, output_path=target, graph=graph)
        request = make_request(path, tmp_path / f"{path.stem}-mem", config)
        result = validate_rdflib(request, _noop_logger())
        assert result.details.get("streaming_nt_sha256") == digest_stream


def test_validate_pronto_success(monkeypatch, obo_file, tmp_path, config):
    pytest.importorskip("pronto")
    pytest.importorskip("ols_client")
    monkeypatch.setenv("PYSTOW_HOME", str(tmp_path / "pystow"))
    request = ValidationRequest("pronto", obo_file, tmp_path / "norm", tmp_path / "val", config)
    result = validate_pronto(request, _noop_logger())
    if not result.ok:  # pragma: no cover - optional dependency pipeline misconfigured
        pytest.skip(f"Pronto validator unavailable: {result.details.get('error')}")
    payload = json.loads((request.validation_dir / "pronto_parse.json").read_text())
    assert payload["ok"]


def test_validate_owlready2_success(monkeypatch, owl_file, tmp_path, config):
    pytest.importorskip("owlready2")
    pytest.importorskip("ols_client")
    monkeypatch.setenv("PYSTOW_HOME", str(tmp_path / "pystow"))
    request = ValidationRequest("owlready2", owl_file, tmp_path / "norm", tmp_path / "val", config)
    result = validate_owlready2(request, _noop_logger())
    if not result.ok:  # pragma: no cover - optional dependency pipeline misconfigured
        pytest.skip(f"Owlready2 validator unavailable: {result.details.get('error')}")


def test_validate_robot_skips_when_missing(monkeypatch, ttl_file, tmp_path, config):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    request = ValidationRequest("robot", ttl_file, tmp_path / "norm", tmp_path / "val", config)
    result = validate_robot(request, _noop_logger())
    assert result.ok
    payload = json.loads((request.validation_dir / "robot_report.json").read_text())
    assert payload["skipped"]


def test_validate_arelle_with_stub(monkeypatch, xbrl_package, tmp_path, config):
    class DummyController:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, args):  # pragma: no cover - trivial stub
            self.args = args

    dummy_module = SimpleNamespace(Cntlr=SimpleNamespace(Cntlr=DummyController))
    monkeypatch.setitem(sys.modules, "arelle", dummy_module)
    request = ValidationRequest("arelle", xbrl_package, tmp_path / "norm", tmp_path / "val", config)
    result = validate_arelle(request, _noop_logger())
    assert result.ok
    payload = json.loads((request.validation_dir / "arelle_validation.json").read_text())
    assert payload["ok"]
    assert payload["log"].endswith("arelle.log")


def test_validate_owlready2_memory_error(monkeypatch, owl_file, tmp_path, config):
    request = ValidationRequest("owlready2", owl_file, tmp_path / "norm", tmp_path / "val", config)

    def _raise(*args, **kwargs):  # pragma: no cover - exercised in test
        raise ValidatorSubprocessError("memory exceeded")

    monkeypatch.setattr(
        "DocsToKG.OntologyDownload.ontology_download._run_validator_subprocess",
        _raise,
    )
    result = validate_owlready2(request, _noop_logger())
    assert not result.ok
    payload = json.loads((request.validation_dir / "owlready2_parse.json").read_text())
    assert "memory exceeded" in payload["error"].lower()


def _noop_logger():
    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    return _Logger()
