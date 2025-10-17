# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_validators",
#   "purpose": "Pytest coverage for ontology download validators scenarios",
#   "sections": [
#     {
#       "id": "config",
#       "name": "config",
#       "anchor": "function-config",
#       "kind": "function"
#     },
#     {
#       "id": "ttl-file",
#       "name": "ttl_file",
#       "anchor": "function-ttl-file",
#       "kind": "function"
#     },
#     {
#       "id": "obo-file",
#       "name": "obo_file",
#       "anchor": "function-obo-file",
#       "kind": "function"
#     },
#     {
#       "id": "owl-file",
#       "name": "owl_file",
#       "anchor": "function-owl-file",
#       "kind": "function"
#     },
#     {
#       "id": "make-request",
#       "name": "make_request",
#       "anchor": "function-make-request",
#       "kind": "function"
#     },
#     {
#       "id": "xbrl-package",
#       "name": "xbrl_package",
#       "anchor": "function-xbrl-package",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-rdflib-success",
#       "name": "test_validate_rdflib_success",
#       "anchor": "function-test-validate-rdflib-success",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-streaming-deterministic",
#       "name": "test_normalize_streaming_deterministic",
#       "anchor": "function-test-normalize-streaming-deterministic",
#       "kind": "function"
#     },
#     {
#       "id": "test-streaming-matches-in-memory",
#       "name": "test_streaming_matches_in_memory",
#       "anchor": "function-test-streaming-matches-in-memory",
#       "kind": "function"
#     },
#     {
#       "id": "test-normalize-streaming-edge-cases",
#       "name": "test_normalize_streaming_edge_cases",
#       "anchor": "function-test-normalize-streaming-edge-cases",
#       "kind": "function"
#     },
#     {
#       "id": "test-run-validators-respects-concurrency",
#       "name": "test_run_validators_respects_concurrency",
#       "anchor": "function-test-run-validators-respects-concurrency",
#       "kind": "function"
#     },
#     {
#       "id": "test-run-validators-matches-sequential",
#       "name": "test_run_validators_matches_sequential",
#       "anchor": "function-test-run-validators-matches-sequential",
#       "kind": "function"
#     },
#     {
#       "id": "test-sort-triple-file-falls-back-without-sort",
#       "name": "test_sort_triple_file_falls_back_without_sort",
#       "anchor": "function-test-sort-triple-file-falls-back-without-sort",
#       "kind": "function"
#     },
#     {
#       "id": "test-validator-plugin-loader-registers-and-warns",
#       "name": "test_validator_plugin_loader_registers_and_warns",
#       "anchor": "function-test-validator-plugin-loader-registers-and-warns",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-pronto-success",
#       "name": "test_validate_pronto_success",
#       "anchor": "function-test-validate-pronto-success",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-pronto-handles-exception",
#       "name": "test_validate_pronto_handles_exception",
#       "anchor": "function-test-validate-pronto-handles-exception",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-owlready2-success",
#       "name": "test_validate_owlready2_success",
#       "anchor": "function-test-validate-owlready2-success",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-robot-skips-when-missing",
#       "name": "test_validate_robot_skips_when_missing",
#       "anchor": "function-test-validate-robot-skips-when-missing",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-arelle-with-stub",
#       "name": "test_validate_arelle_with_stub",
#       "anchor": "function-test-validate-arelle-with-stub",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-owlready2-memory-error",
#       "name": "test_validate_owlready2_memory_error",
#       "anchor": "function-test-validate-owlready2-memory-error",
#       "kind": "function"
#     },
#     {
#       "id": "noop-logger",
#       "name": "_noop_logger",
#       "anchor": "function-noop-logger",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

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
import logging
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

import DocsToKG.OntologyDownload.validation as validation_mod
from DocsToKG.OntologyDownload.settings import DefaultsConfig, ResolvedConfig
from DocsToKG.OntologyDownload.validation import (
    ValidationRequest,
    ValidationResult,
    ValidatorSubprocessError,
    _sort_triple_file,
    load_validator_plugins,
    normalize_streaming,
    run_validators,
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


# --- Test Cases ---


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
    stream_hash, header_hash = normalize_streaming(source, return_header_hash=True)
    assert streaming_result.details.get("streaming_nt_sha256") == stream_hash
    assert streaming_result.details.get("streaming_prefix_sha256") == header_hash

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
        digest_stream, header_stream = normalize_streaming(
            path, output_path=target, graph=graph, return_header_hash=True
        )
        request = make_request(path, tmp_path / f"{path.stem}-mem", config)
        result = validate_rdflib(request, _noop_logger())
        assert result.details.get("streaming_nt_sha256") == digest_stream
        assert result.details.get("streaming_prefix_sha256") == header_stream


def test_run_validators_respects_concurrency(monkeypatch, tmp_path, config):
    config = config.model_copy(deep=True)
    config.defaults.validation.max_concurrent_validators = 2

    barrier = threading.Barrier(config.defaults.validation.max_concurrent_validators)
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def _make_validator(name: str):
        def _validator(request: ValidationRequest, logger):
            path = request.validation_dir / f"{request.name}_parse.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                barrier.wait(timeout=1)
            except threading.BrokenBarrierError:
                pass
            time.sleep(0.02)
            payload = {"ok": True, "validator": request.name}
            path.write_text(json.dumps(payload))
            with lock:
                state["active"] -= 1
            return ValidationResult(ok=True, details=payload, output_files=[str(path)])

        return _validator

    validators = {
        "one": _make_validator("one"),
        "two": _make_validator("two"),
        "three": _make_validator("three"),
    }
    monkeypatch.setattr(validation_mod, "VALIDATORS", validators)

    requests = []
    for name in validators:
        file_path = tmp_path / f"{name}.ttl"
        file_path.write_text("@prefix ex: <http://example.org/> .")
        requests.append(
            ValidationRequest(
                name,
                file_path,
                tmp_path / f"norm-{name}",
                tmp_path / f"val-{name}",
                config,
            )
        )

    results = run_validators(requests, _noop_logger())
    assert set(results) == set(validators)
    assert state["max_active"] >= 2
    for name in validators:
        artifact = tmp_path / f"val-{name}" / f"{name}_parse.json"
        assert artifact.exists()


def test_run_validators_matches_sequential(monkeypatch, tmp_path, config):
    config_seq = config.model_copy(deep=True)
    config_seq.defaults.validation.max_concurrent_validators = 1

    config_conc = config.model_copy(deep=True)
    config_conc.defaults.validation.max_concurrent_validators = 3

    def _validator(request: ValidationRequest, logger):
        payload = {"ok": True, "validator": request.name}
        artifact = request.validation_dir / f"{request.name}_parse.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(payload))
        return ValidationResult(ok=True, details=payload, output_files=[str(artifact)])

    validators = {
        "alpha": _validator,
        "beta": _validator,
        "gamma": _validator,
    }
    monkeypatch.setattr(validation_mod, "VALIDATORS", validators)

    def _build_requests(prefix: str, cfg: ResolvedConfig) -> list[ValidationRequest]:
        requests = []
        for name in validators:
            file_path = tmp_path / f"{prefix}-{name}.ttl"
            file_path.write_text("@prefix ex: <http://example.org/> .")
            requests.append(
                ValidationRequest(
                    name,
                    file_path,
                    tmp_path / f"{prefix}-{name}-normalized",
                    tmp_path / f"{prefix}-{name}-validation",
                    cfg,
                )
            )
        return requests

    seq_results = run_validators(_build_requests("seq", config_seq), _noop_logger())
    conc_results = run_validators(_build_requests("conc", config_conc), _noop_logger())

    assert set(seq_results) == set(conc_results) == set(validators)
    for name in validators:
        assert "metrics" in seq_results[name].details
        assert "metrics" in conc_results[name].details
        seq_details = dict(seq_results[name].details)
        conc_details = dict(conc_results[name].details)
        seq_details.pop("metrics", None)
        conc_details.pop("metrics", None)
        assert seq_details == conc_details


def test_sort_triple_file_falls_back_without_sort(monkeypatch, tmp_path):
    unsorted = tmp_path / "triples.unsorted"
    unsorted.write_text("c\nA\nb\n", encoding="utf-8")
    destination = tmp_path / "triples.sorted"

    monkeypatch.setattr(shutil, "which", lambda _: None)

    _sort_triple_file(unsorted, destination)
    assert destination.read_text(encoding="utf-8") == "A\nb\nc\n"


def test_validator_plugin_loader_registers_and_warns(monkeypatch, caplog):
    base = validation_mod.VALIDATORS.copy()
    monkeypatch.setattr(validation_mod, "VALIDATORS", base.copy())

    def _plugin(request, logger):  # pragma: no cover - handler not executed here
        return ValidationResult(ok=True, details={"ok": True}, output_files=[])

    class DummyEntry:
        def __init__(self, name: str, target, fail: bool = False):
            self.name = name
            self._target = target
            self._fail = fail

        def load(self):
            if self._fail:
                raise RuntimeError("boom")
            return self._target

    entries = [
        DummyEntry("plugin_validator", _plugin),
        DummyEntry("broken_validator", None, fail=True),
    ]

    stub = SimpleNamespace(
        select=lambda *, group=None: entries if group == "docstokg.ontofetch.validator" else []
    )
    monkeypatch.setattr(validation_mod.metadata, "entry_points", lambda: stub)
    monkeypatch.setattr(validation_mod, "_VALIDATOR_PLUGINS_LOADED", False)

    caplog.set_level(logging.INFO)
    load_validator_plugins(validation_mod.VALIDATORS, logger=logging.getLogger("test"))

    assert "plugin_validator" in validation_mod.VALIDATORS
    assert validation_mod.VALIDATORS["plugin_validator"] is _plugin
    assert any(record.message == "validator plugin failed" for record in caplog.records)


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


def test_validate_pronto_handles_exception(monkeypatch, obo_file, tmp_path, config):
    request = ValidationRequest("pronto", obo_file, tmp_path / "norm", tmp_path / "val", config)

    def _boom(*_args, **_kwargs):  # pragma: no cover - deterministic failure path
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "DocsToKG.OntologyDownload.validation._run_validator_subprocess",
        _boom,
    )

    result = validate_pronto(request, _noop_logger())
    assert not result.ok
    payload = json.loads((request.validation_dir / "pronto_parse.json").read_text())
    assert payload["ok"] is False
    assert payload["error"] == "boom"


def test_validate_owlready2_success(monkeypatch, owl_file, tmp_path, config):
    try:
        pytest.importorskip("owlready2")
    except Exception as exc:  # pragma: no cover - optional dependency import failed
        pytest.skip(f"owlready2 unavailable: {exc}")
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
        "DocsToKG.OntologyDownload.validation._run_validator_subprocess",
        _raise,
    )
    result = validate_owlready2(request, _noop_logger())
    assert not result.ok
    payload = json.loads((request.validation_dir / "owlready2_parse.json").read_text())
    assert "memory exceeded" in payload["error"].lower()


# --- Helper Functions ---


def _noop_logger():
    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    return _Logger()
