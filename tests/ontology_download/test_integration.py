# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_integration",
#   "purpose": "Pytest coverage for ontology download integration scenarios",
#   "sections": [
#     {
#       "id": "patched-dirs",
#       "name": "patched_dirs",
#       "anchor": "function-patched-dirs",
#       "kind": "function"
#     },
#     {
#       "id": "stubresolver",
#       "name": "_StubResolver",
#       "anchor": "class-stubresolver",
#       "kind": "class"
#     },
#     {
#       "id": "stubbed-validators",
#       "name": "stubbed_validators",
#       "anchor": "function-stubbed-validators",
#       "kind": "function"
#     },
#     {
#       "id": "test-fetch-all-writes-manifests",
#       "name": "test_fetch_all_writes_manifests",
#       "anchor": "function-test-fetch-all-writes-manifests",
#       "kind": "function"
#     },
#     {
#       "id": "test-force-download-bypasses-manifest",
#       "name": "test_force_download_bypasses_manifest",
#       "anchor": "function-test-force-download-bypasses-manifest",
#       "kind": "function"
#     },
#     {
#       "id": "test-multi-version-storage",
#       "name": "test_multi_version_storage",
#       "anchor": "function-test-multi-version-storage",
#       "kind": "function"
#     },
#     {
#       "id": "test-fetch-all-logs-progress",
#       "name": "test_fetch_all_logs_progress",
#       "anchor": "function-test-fetch-all-logs-progress",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Ontology Download Integration Tests

This module exercises the ontology download workflow end-to-end using
fixture resolvers, validating manifest generation, validator integration,
multi-version storage, and logging side effects.

Key Scenarios:
- Fetches multiple ontologies and persists manifests plus normalized artefacts
- Ensures force downloads bypass cached manifests
- Confirms versioned storage structure and progress logging

Dependencies:
- pytest: Fixtures and monkeypatching
- DocsToKG.OntologyDownload: Core orchestration under test

Usage:
    pytest tests/ontology_download/test_integration.py
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

import DocsToKG.OntologyDownload.pipeline as pipeline_mod
from DocsToKG.OntologyDownload import DefaultsConfig, ResolvedConfig, resolvers
from DocsToKG.OntologyDownload import ontology_download as core
from DocsToKG.OntologyDownload import storage as storage_mod


@pytest.fixture()
def patched_dirs(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    configs = tmp_path / "configs"
    logs = tmp_path / "logs"
    ontologies = tmp_path / "ontologies"
    for directory in (cache, configs, logs, ontologies):
        directory.mkdir(parents=True, exist_ok=True)
    overrides = {
        "CACHE_DIR": cache,
        "CONFIG_DIR": configs,
        "LOG_DIR": logs,
        "LOCAL_ONTOLOGY_DIR": ontologies,
    }
    for attr, value in overrides.items():
        monkeypatch.setattr(storage_mod, attr, value, raising=False)
        monkeypatch.setattr(pipeline_mod, attr, value, raising=False)
        monkeypatch.setattr(core, attr, value, raising=False)
    monkeypatch.setattr(core, "ONTOLOGY_DIR", ontologies, raising=False)

    class _StubStorage:
        def ensure_local_version(self, ontology_id: str, version: str) -> Path:
            path = ontologies / ontology_id / version
            path.mkdir(parents=True, exist_ok=True)
            return path

        def finalize_version(self, ontology_id: str, version: str, base_dir: Path) -> None:
            pass

        def set_latest_version(
            self, ontology_id: str, path: Path
        ) -> None:  # pragma: no cover - not used
            pass

    stub_storage = _StubStorage()
    monkeypatch.setattr(storage_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(pipeline_mod, "STORAGE", stub_storage, raising=False)
    monkeypatch.setattr(core, "STORAGE", stub_storage, raising=False)
    return ontologies


class _StubResolver:
    def __init__(self, fixture: Path, version: str):
        self.fixture = fixture
        self.version = version

    def plan(self, spec, config, logger):
        return resolvers.FetchPlan(
            url=f"https://example.org/{spec.id}.owl",
            headers={},
            filename_hint=self.fixture.name,
            version=self.version,
            license="CC0-1.0",
            media_type="application/rdf+xml",
            service=spec.resolver,
        )


@pytest.fixture()
def stubbed_validators(monkeypatch):
    def _runner(requests, logger):
        results = {}
        for req in requests:
            req.normalized_dir.mkdir(parents=True, exist_ok=True)
            ttl_path = req.normalized_dir / f"{req.file_path.stem}.ttl"
            json_path = req.normalized_dir / f"{req.file_path.stem}.json"
            ttl_path.write_text("normalized")
            json_path.write_text("{}")
            details = {"ok": True}
            if req.name == "rdflib":
                details["normalized_sha256"] = hashlib.sha256(b"normalized").hexdigest()
            results[req.name] = core.ValidationResult(
                ok=True,
                details=details,
                output_files=[str(ttl_path), str(json_path)],
            )
        return results

    monkeypatch.setattr(pipeline_mod, "run_validators", _runner, raising=False)
    monkeypatch.setattr(core, "run_validators", _runner, raising=False)


# --- Test Cases ---


def test_fetch_all_writes_manifests(monkeypatch, patched_dirs, stubbed_validators):
    fixture_dir = Path("tests/data/ontology_fixtures")
    pato_fixture = fixture_dir / "mini.ttl"
    bfo_fixture = fixture_dir / "mini.obo"

    monkeypatch.setitem(resolvers.RESOLVERS, "obo", _StubResolver(pato_fixture, "2024-01-01"))
    monkeypatch.setitem(resolvers.RESOLVERS, "ols", _StubResolver(bfo_fixture, "2024-02-01"))

    def _download_with_fixture(**kwargs):
        headers = dict(kwargs["headers"])
        if kwargs["url"].endswith("pato.owl"):
            headers["__fixture__"] = pato_fixture
        else:
            headers["__fixture__"] = bfo_fixture
        kwargs["headers"] = headers
        destination = kwargs["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(headers["__fixture__"], destination)
        sha256 = core.sha256_file(destination)
        content_length = destination.stat().st_size
        return core.DownloadResult(
            path=destination,
            status="fresh",
            sha256=sha256,
            etag="etag",
            last_modified="yesterday",
            content_type="application/rdf+xml",
            content_length=content_length,
        )

    monkeypatch.setattr(
        pipeline_mod, "download_stream", _download_with_fixture, raising=False
    )
    monkeypatch.setattr(core, "download_stream", _download_with_fixture, raising=False)

    results = core.fetch_all(
        [
            core.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=["owl"]),
            core.FetchSpec(id="bfo", resolver="ols", extras={}, target_formats=["owl"]),
        ],
        config=ResolvedConfig(defaults=DefaultsConfig(prefer_source=["obo", "ols"]), specs=[]),
        force=True,
    )
    assert len(results) == 2
    for result in results:
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["id"] == result.spec.id
        assert manifest["validation"]
        local_file = result.local_path
        assert manifest["sha256"] == core.sha256_file(local_file)
        assert manifest["normalized_sha256"]
        assert len(manifest["fingerprint"]) == 64
        assert manifest["content_type"] == "application/rdf+xml"
        assert manifest["content_length"] == local_file.stat().st_size
        assert manifest["source_media_type_label"] == "RDF/XML"
        normalized_dir = result.manifest_path.parent / "normalized"
        assert any(normalized_dir.glob("*.ttl"))
        assert any(normalized_dir.glob("*.json"))


def test_force_download_bypasses_manifest(monkeypatch, patched_dirs, stubbed_validators):
    fixture = Path("tests/data/ontology_fixtures/mini.ttl")
    captured = {"previous": None}

    def _download(**kwargs):
        captured["previous"] = kwargs.get("previous_manifest")
        kwargs["destination"].write_bytes(fixture.read_bytes())
        content_length = kwargs["destination"].stat().st_size
        return core.DownloadResult(
            path=kwargs["destination"],
            status="fresh",
            sha256="sha",
            etag=None,
            last_modified=None,
            content_type="application/rdf+xml",
            content_length=content_length,
        )

    monkeypatch.setattr(pipeline_mod, "download_stream", _download, raising=False)
    monkeypatch.setattr(core, "download_stream", _download, raising=False)
    spec = core.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=["owl"])
    monkeypatch.setitem(resolvers.RESOLVERS, "obo", _StubResolver(fixture, "2024-01-01"))
    manifest_path = patched_dirs / "pato" / "2024-01-01" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"sha256": "old"}))
    core.fetch_one(spec, config=ResolvedConfig.from_defaults(), force=True)
    assert captured["previous"] is None


def test_multi_version_storage(monkeypatch, patched_dirs, stubbed_validators):
    fixture = Path("tests/data/ontology_fixtures/mini.ttl")
    monkeypatch.setitem(resolvers.RESOLVERS, "obo", _StubResolver(fixture, "2024-01-01"))

    def _download(**kwargs):
        kwargs["destination"].write_bytes(fixture.read_bytes())
        content_length = kwargs["destination"].stat().st_size
        return core.DownloadResult(
            path=kwargs["destination"],
            status="fresh",
            sha256="sha",
            etag=None,
            last_modified=None,
            content_type="application/rdf+xml",
            content_length=content_length,
        )

    monkeypatch.setattr(pipeline_mod, "download_stream", _download, raising=False)
    monkeypatch.setattr(core, "download_stream", _download, raising=False)
    spec = core.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=["owl"])
    config = ResolvedConfig(defaults=DefaultsConfig(), specs=[])
    core.fetch_one(spec, config=config, force=True)
    monkeypatch.setitem(resolvers.RESOLVERS, "obo", _StubResolver(fixture, "2024-02-01"))
    core.fetch_one(spec, config=config, force=True)
    versions = sorted((patched_dirs / "pato").iterdir())
    assert {v.name for v in versions} == {"2024-01-01", "2024-02-01"}


def test_fetch_all_logs_progress(monkeypatch, patched_dirs, stubbed_validators, caplog):
    fixture = Path("tests/data/ontology_fixtures/mini.ttl")

    def _download(**kwargs):
        kwargs["destination"].write_bytes(fixture.read_bytes())
        content_length = kwargs["destination"].stat().st_size
        return core.DownloadResult(
            path=kwargs["destination"],
            status="fresh",
            sha256="sha",
            etag=None,
            last_modified=None,
            content_type="application/rdf+xml",
            content_length=content_length,
        )

    monkeypatch.setattr(pipeline_mod, "download_stream", _download, raising=False)
    monkeypatch.setattr(core, "download_stream", _download, raising=False)
    monkeypatch.setitem(resolvers.RESOLVERS, "obo", _StubResolver(fixture, "2024-01-01"))
    config = ResolvedConfig(defaults=DefaultsConfig(), specs=[])
    caplog.set_level(logging.INFO)
    core.fetch_all(
        [core.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=["owl"])],
        config=config,
    )
    assert any("progress" in record.getMessage() for record in caplog.records)
