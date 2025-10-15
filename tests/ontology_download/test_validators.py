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
- DocsToKG.OntologyDownload.validators: Validation entry points under test

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

from DocsToKG.OntologyDownload.config import DefaultsConfig, ResolvedConfig
from DocsToKG.OntologyDownload.validators import (
    ValidationRequest,
    ValidatorSubprocessError,
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


def test_validate_pronto_success(obo_file, tmp_path, config):
    request = ValidationRequest("pronto", obo_file, tmp_path / "norm", tmp_path / "val", config)
    result = validate_pronto(request, _noop_logger())
    assert result.ok
    payload = json.loads((request.validation_dir / "pronto_parse.json").read_text())
    assert payload["ok"]


def test_validate_owlready2_success(owl_file, tmp_path, config):
    request = ValidationRequest("owlready2", owl_file, tmp_path / "norm", tmp_path / "val", config)
    result = validate_owlready2(request, _noop_logger())
    assert result.ok


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
        "DocsToKG.OntologyDownload.validators._run_validator_subprocess",
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
