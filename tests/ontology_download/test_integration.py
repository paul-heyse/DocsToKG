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
- pytest: Fixtures and patch helpers
- DocsToKG.OntologyDownload: Core orchestration under test

Usage:
    pytest tests/ontology_download/test_integration.py
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from unittest.mock import patch

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

import DocsToKG.OntologyDownload.planning as pipeline_mod
from DocsToKG.OntologyDownload import api as core
from DocsToKG.OntologyDownload import io as io_mod
from DocsToKG.OntologyDownload import settings as settings_mod
from DocsToKG.OntologyDownload.io import network as network_mod
from DocsToKG.OntologyDownload.planning import RESOLVERS, FetchPlan
from DocsToKG.OntologyDownload.settings import DefaultsConfig, ResolvedConfig
from DocsToKG.OntologyDownload.testing import temporary_resolver
from DocsToKG.OntologyDownload.validation import ValidationResult


@pytest.fixture()
def patched_dirs(ontology_env):
    return ontology_env.ontology_dir


class _StubResolver:
    def __init__(self, fixture: Path, version: str):
        self.fixture = fixture
        self.version = version

    def plan(self, spec, config, logger):
        return FetchPlan(
            url=f"https://example.org/{spec.id}.owl",
            headers={},
            filename_hint=self.fixture.name,
            version=self.version,
            license="CC0-1.0",
            media_type="application/rdf+xml",
            service=spec.resolver,
        )


@pytest.fixture()
def stubbed_validators():
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
            results[req.name] = ValidationResult(
                ok=True,
                details=details,
                output_files=[str(ttl_path), str(json_path)],
            )
        return results

    with patch.object(pipeline_mod, "run_validators", _runner), patch.object(
        core, "run_validators", _runner
    ):
        yield



# --- Test Cases ---


def test_fetch_all_writes_manifests(ontology_env, patched_dirs, stubbed_validators):
    fixture_dir = Path("tests/data/ontology_fixtures")
    pato_fixture = fixture_dir / "mini.ttl"
    bfo_fixture = fixture_dir / "mini.obo"

    config = ontology_env.build_resolved_config()
    config.defaults.http.allowed_hosts = ["example.org"]

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
        sha256 = io_mod.sha256_file(destination)
        content_length = destination.stat().st_size
        return io_mod.DownloadResult(
            path=destination,
            status="fresh",
            sha256=sha256,
            etag="etag",
            last_modified="yesterday",
            content_type="application/rdf+xml",
            content_length=content_length,
        )

    pato_resolver = _StubResolver(pato_fixture, "2024-01-01")
    bfo_resolver = _StubResolver(bfo_fixture, "2024-02-01")

    specs = [
        pipeline_mod.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=("owl",)),
        pipeline_mod.FetchSpec(id="bfo", resolver="ols", extras={}, target_formats=("owl",)),
    ]

    with temporary_resolver("obo", pato_resolver), temporary_resolver("ols", bfo_resolver):
        with patch.object(pipeline_mod, "download_stream", _download_with_fixture), patch.object(
            core, "download_stream", _download_with_fixture
        ):
            results = core.fetch_all(
                specs,
                config=config,
                force=True,
            )
    assert len(results) == 2
    for result in results:
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["id"] == result.spec.id
        assert manifest["validation"]
        local_file = result.local_path
        assert manifest["sha256"] == io_mod.sha256_file(local_file)
        assert manifest["normalized_sha256"]
        assert len(manifest["fingerprint"]) == 64
        assert manifest["content_type"] == "application/rdf+xml"
        assert manifest["content_length"] == local_file.stat().st_size
        assert manifest["source_media_type_label"] == "RDF/XML"
        normalized_dir = result.manifest_path.parent / "normalized"
        assert any(normalized_dir.glob("*.ttl"))
        assert any(normalized_dir.glob("*.json"))


def test_force_download_bypasses_manifest(ontology_env, patched_dirs, stubbed_validators):
    fixture = Path("tests/data/ontology_fixtures/mini.ttl")
    captured = {"previous": None}

    def _download(**kwargs):
        captured["previous"] = kwargs.get("previous_manifest")
        kwargs["destination"].write_bytes(fixture.read_bytes())
        content_length = kwargs["destination"].stat().st_size
        return io_mod.DownloadResult(
            path=kwargs["destination"],
            status="fresh",
            sha256="sha",
            etag=None,
            last_modified=None,
            content_type="application/rdf+xml",
            content_length=content_length,
        )

    config = ontology_env.build_resolved_config()
    config.defaults.http.allowed_hosts = ["example.org"]
    spec = pipeline_mod.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=("owl",))

    with temporary_resolver("obo", _StubResolver(fixture, "2024-01-01")):
        with patch.object(pipeline_mod, "download_stream", _download), patch.object(
            core, "download_stream", _download
        ):
            manifest_path = patched_dirs / "pato" / "2024-01-01" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"sha256": "old"}))
            core.fetch_one(spec, config=config, force=True)

    assert captured["previous"] is None


def test_multi_version_storage(ontology_env, patched_dirs, stubbed_validators):
    fixture = Path("tests/data/ontology_fixtures/mini.ttl")
    config = ontology_env.build_resolved_config()
    config.defaults.http.allowed_hosts = ["example.org"]

    def _download(**kwargs):
        kwargs["destination"].write_bytes(fixture.read_bytes())
        content_length = kwargs["destination"].stat().st_size
        return io_mod.DownloadResult(
            path=kwargs["destination"],
            status="fresh",
            sha256="sha",
            etag=None,
            last_modified=None,
            content_type="application/rdf+xml",
            content_length=content_length,
        )

    spec = pipeline_mod.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=("owl",))

    with temporary_resolver("obo", _StubResolver(fixture, "2024-01-01")):
        with patch.object(pipeline_mod, "download_stream", _download), patch.object(
            core, "download_stream", _download
        ):
            core.fetch_one(spec, config=config, force=True)

    with temporary_resolver("obo", _StubResolver(fixture, "2024-02-01")):
        with patch.object(pipeline_mod, "download_stream", _download), patch.object(
            core, "download_stream", _download
        ):
            core.fetch_one(spec, config=config, force=True)
    versions = sorted((patched_dirs / "pato").iterdir())
    dir_names = {v.name for v in versions if v.is_dir()}
    assert dir_names == {"2024-01-01", "2024-02-01"}


def test_fetch_all_logs_progress(ontology_env, patched_dirs, stubbed_validators, caplog):
    fixture = Path("tests/data/ontology_fixtures/mini.ttl")

    def _download(**kwargs):
        kwargs["destination"].write_bytes(fixture.read_bytes())
        content_length = kwargs["destination"].stat().st_size
        return io_mod.DownloadResult(
            path=kwargs["destination"],
            status="fresh",
            sha256="sha",
            etag=None,
            last_modified=None,
            content_type="application/rdf+xml",
            content_length=content_length,
        )

    config = ontology_env.build_resolved_config()
    config.defaults.http.allowed_hosts = ["example.org"]
    caplog.set_level(logging.INFO, logger="DocsToKG.OntologyDownload")
    logger = logging.getLogger("DocsToKG.OntologyDownload")
    recorded = []

    class _RecordingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - test helper
            recorded.append(record)

    handler = _RecordingHandler()
    logger.addHandler(handler)
    try:
        spec = pipeline_mod.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=("owl",))
        with temporary_resolver("obo", _StubResolver(fixture, "2024-01-01")):
            with patch.object(pipeline_mod, "download_stream", _download), patch.object(
                core, "download_stream", _download
            ):
                core.fetch_all([spec], config=config)
    finally:
        logger.removeHandler(handler)

    assert any("progress update" in record.getMessage() for record in recorded)
