import json
import logging
import shutil
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload import core, download, resolvers, validators
from DocsToKG.OntologyDownload.config import DefaultsConfiguration, ResolvedConfig


@pytest.fixture()
def patched_dirs(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    configs = tmp_path / "configs"
    logs = tmp_path / "logs"
    ontologies = tmp_path / "ontologies"
    for directory in (cache, configs, logs, ontologies):
        directory.mkdir()
    monkeypatch.setattr(core, "CACHE_DIR", cache)
    monkeypatch.setattr(core, "CONFIG_DIR", configs)
    monkeypatch.setattr(core, "LOG_DIR", logs)
    monkeypatch.setattr(core, "ONTOLOGY_DIR", ontologies)
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
            results[req.name] = validators.ValidationResult(
                ok=True,
                details={"ok": True},
                output_files=[str(ttl_path), str(json_path)],
            )
        return results

    monkeypatch.setattr(core, "run_validators", _runner)


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
        sha256 = download.sha256_file(destination)
        return download.DownloadResult(
            path=destination,
            status="fresh",
            sha256=sha256,
            etag="etag",
            last_modified="yesterday",
        )

    monkeypatch.setattr(core, "download_stream", _download_with_fixture)

    results = core.fetch_all(
        [
            core.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=["owl"]),
            core.FetchSpec(id="bfo", resolver="ols", extras={}, target_formats=["owl"]),
        ],
        config=ResolvedConfig(defaults=DefaultsConfiguration(), specs=()),
        force=True,
    )
    assert len(results) == 2
    for result in results:
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["id"] == result.spec.id
        assert manifest["validation"]
        local_file = result.local_path
        assert manifest["sha256"] == download.sha256_file(local_file)
        normalized_dir = result.manifest_path.parent / "normalized"
        assert any(normalized_dir.glob("*.ttl"))
        assert any(normalized_dir.glob("*.json"))


def test_force_download_bypasses_manifest(monkeypatch, patched_dirs, stubbed_validators):
    fixture = Path("tests/data/ontology_fixtures/mini.ttl")
    captured = {"previous": None}

    def _download(**kwargs):
        captured["previous"] = kwargs.get("previous_manifest")
        kwargs["destination"].write_bytes(fixture.read_bytes())
        return download.DownloadResult(
            path=kwargs["destination"],
            status="fresh",
            sha256="sha",
            etag=None,
            last_modified=None,
        )

    monkeypatch.setattr(core, "download_stream", _download)
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
        return download.DownloadResult(
            path=kwargs["destination"],
            status="fresh",
            sha256="sha",
            etag=None,
            last_modified=None,
        )

    monkeypatch.setattr(core, "download_stream", _download)
    spec = core.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=["owl"])
    config = ResolvedConfig(defaults=DefaultsConfiguration(), specs=())
    core.fetch_one(spec, config=config, force=True)
    monkeypatch.setitem(resolvers.RESOLVERS, "obo", _StubResolver(fixture, "2024-02-01"))
    core.fetch_one(spec, config=config, force=True)
    versions = sorted((patched_dirs / "pato").iterdir())
    assert {v.name for v in versions} == {"2024-01-01", "2024-02-01"}


def test_fetch_all_logs_progress(monkeypatch, patched_dirs, stubbed_validators, caplog):
    fixture = Path("tests/data/ontology_fixtures/mini.ttl")

    def _download(**kwargs):
        kwargs["destination"].write_bytes(fixture.read_bytes())
        return download.DownloadResult(
            path=kwargs["destination"],
            status="fresh",
            sha256="sha",
            etag=None,
            last_modified=None,
        )

    monkeypatch.setattr(core, "download_stream", _download)
    monkeypatch.setitem(resolvers.RESOLVERS, "obo", _StubResolver(fixture, "2024-01-01"))
    config = ResolvedConfig(defaults=DefaultsConfiguration(), specs=())
    caplog.set_level(logging.INFO)
    core.fetch_all([core.FetchSpec(id="pato", resolver="obo", extras={}, target_formats=["owl"])], config=config)
    assert any("progress" in record.getMessage() for record in caplog.records)
