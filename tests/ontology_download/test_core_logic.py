"""Core module behavior tests."""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import core
from DocsToKG.OntologyDownload.config import ConfigError, DefaultsConfig, ResolvedConfig
from DocsToKG.OntologyDownload.download import DownloadFailure, DownloadResult
from DocsToKG.OntologyDownload.resolvers import FetchPlan
from DocsToKG.OntologyDownload.validators import ValidationResult


def _make_plan() -> FetchPlan:
    return FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint=None,
        version="2024-01-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )


def test_plan_one_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the primary resolver fails, plan_one should fall back to the next option."""

    class FailingResolver:
        def plan(self, spec, config, logger):
            raise ConfigError("boom")

    class SuccessfulResolver:
        def plan(self, spec, config, logger):
            return _make_plan()

    monkeypatch.setitem(core.RESOLVERS, "obo", FailingResolver())
    monkeypatch.setitem(core.RESOLVERS, "lov", SuccessfulResolver())

    defaults = DefaultsConfig(prefer_source=["obo", "lov"])
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    planned = core.plan_one(spec, config=config)

    assert planned.resolver == "lov"
    assert planned.plan.url.endswith("hp.owl")
    assert [candidate.resolver for candidate in planned.candidates] == ["lov"]


def test_plan_one_respects_disabled_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback should be skipped when resolver_fallback_enabled is False."""

    class FailingResolver:
        def plan(self, spec, config, logger):
            raise ConfigError("boom")

    monkeypatch.setitem(core.RESOLVERS, "obo", FailingResolver())

    defaults = DefaultsConfig(
        prefer_source=["obo", "lov"],
        resolver_fallback_enabled=False,
    )
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    with pytest.raises(core.ResolverError):
        core.plan_one(spec, config=config)


def test_fetch_one_download_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Download should fall back to the next resolver on retryable failure."""

    primary_plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={"Accept": "application/rdf+xml"},
        filename_hint="hp.owl",
        version="2024-01-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="obo",
    )
    fallback_plan = FetchPlan(
        url="https://mirror.example.org/hp.owl",
        headers={"Accept": "application/rdf+xml"},
        filename_hint="hp.owl",
        version="2024-01-01",
        license="CC-BY",
        media_type="application/rdf+xml",
        service="lov",
    )

    class PrimaryResolver:
        def plan(self, spec, config, logger):
            return primary_plan

    class SecondaryResolver:
        def plan(self, spec, config, logger):
            return fallback_plan

    monkeypatch.setitem(core.RESOLVERS, "obo", PrimaryResolver())
    monkeypatch.setitem(core.RESOLVERS, "lov", SecondaryResolver())

    defaults = DefaultsConfig(prefer_source=["lov"])
    config = ResolvedConfig(defaults=defaults, specs=[])
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    cache_dir = tmp_path / "cache"
    ontology_dir = tmp_path / "ontologies"
    monkeypatch.setattr(core, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(core, "ONTOLOGY_DIR", ontology_dir)

    class _StubStorage:
        def finalize_version(self, ontology_id: str, version: str, base_dir: Path) -> None:
            pass

        def ensure_local_version(self, ontology_id: str, version: str) -> Path:
            path = ontology_dir / ontology_id / version
            path.mkdir(parents=True, exist_ok=True)
            return path

    monkeypatch.setattr(core, "STORAGE", _StubStorage())

    def _fake_run_validators(requests, logger):
        results = {}
        for request in requests:
            details = {"normalized_sha256": "norm"} if request.name == "rdflib" else {}
            results[request.name] = ValidationResult(ok=True, details=details, output_files=[])
        return results

    monkeypatch.setattr(core, "run_validators", _fake_run_validators)
    monkeypatch.setattr(core, "validate_url_security", lambda url, config=None: url)

    attempts = {"count": 0}

    def _fake_download_stream(**kwargs):
        attempts["count"] += 1
        destination: Path = kwargs["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if attempts["count"] == 1:
            raise DownloadFailure("temporary outage", status_code=503, retryable=True)
        payload = b"ontology"
        destination.write_bytes(payload)
        sha256 = hashlib.sha256(payload).hexdigest()
        return DownloadResult(
            path=destination,
            status="fresh",
            sha256=sha256,
            etag=None,
            last_modified=None,
        )

    monkeypatch.setattr(core, "download_stream", _fake_download_stream)

    logger = logging.getLogger("ontology-download-test")
    logger.setLevel(logging.INFO)

    result = core.fetch_one(spec, config=config, force=True, logger=logger)

    assert attempts["count"] == 2
    assert result.spec.resolver == "lov"
    assert result.local_path.exists()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["resolver"] == "lov"
    assert manifest["url"] == "https://mirror.example.org/hp.owl"
    chain = manifest["resolver_attempts"]
    assert len(chain) == 2
    assert chain[0]["resolver"] == "obo"
    assert chain[0]["status"] == "failed"
    assert chain[1]["resolver"] == "lov"
    assert chain[1]["status"] == "success"
