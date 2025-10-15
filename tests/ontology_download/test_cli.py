"""
Ontology Download CLI Tests

This module exercises the ontology download command line interface to
ensure pull, validate, and configuration commands emit structured output
for both human-readable and JSON modes.

Key Scenarios:
- Fetches ontologies via `pull` with table and JSON output formatting
- Validates manifests using optional backends such as RDFLib
- Checks configuration validation including missing file error handling

Dependencies:
- pytest: Monkeypatching and fixtures
- DocsToKG.OntologyDownload.cli: CLI entry point under test

Usage:
    pytest tests/ontology_download/test_cli.py
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import cli
from DocsToKG.OntologyDownload.config import DefaultsConfig, ResolvedConfig
from DocsToKG.OntologyDownload.core import FetchResult, FetchSpec, PlannedFetch
from DocsToKG.OntologyDownload.resolvers import FetchPlan


@pytest.fixture()
def stub_logger():
    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    return _Logger()


class _StubStorage:
    def __init__(self, root: Path):
        self.root = root

    def available_versions(self, ontology_id: str):
        target = self.root / ontology_id
        if not target.exists():
            return []
        return sorted([entry.name for entry in target.iterdir() if entry.is_dir()])

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        path = self.root / ontology_id / version
        path.mkdir(parents=True, exist_ok=True)
        return path


def test_cli_pull_json_output(monkeypatch, stub_logger, tmp_path, capsys):
    result = FetchResult(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        local_path=tmp_path / "hp.owl",
        status="fresh",
        sha256="abc123",
        manifest_path=tmp_path / "manifest.json",
        artifacts=[str(tmp_path / "hp.owl")],
    )
    monkeypatch.setattr(cli, "fetch_all", lambda specs, config, force: [result])
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    assert cli.main(["pull", "hp", "--json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload[0]["id"] == "hp"


def test_cli_pull_table_output(monkeypatch, stub_logger, tmp_path, capsys):
    result = FetchResult(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        local_path=tmp_path / "hp.owl",
        status="fresh",
        sha256="abc123",
        manifest_path=tmp_path / "manifest.json",
        artifacts=[str(tmp_path / "hp.owl")],
    )
    monkeypatch.setattr(cli, "fetch_all", lambda specs, config, force: [result])
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    assert cli.main(["pull", "hp"]) == 0
    output = capsys.readouterr().out
    lines = output.splitlines()
    assert "resolver" in lines[0]
    assert any("hp | obo" in line for line in lines)


def test_cli_pull_dry_run(monkeypatch, stub_logger, capsys):
    planned = PlannedFetch(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        resolver="obo",
        plan=FetchPlan(
            url="https://example.org/hp.owl",
            headers={},
            filename_hint=None,
            version="2024",
            license="CC-BY",
            media_type="application/rdf+xml",
            service="obo",
        ),
    )

    monkeypatch.setattr(cli, "plan_all", lambda specs, config: [planned])
    monkeypatch.setattr(
        cli, "fetch_all", lambda *_, **__: pytest.fail("dry-run should not download")
    )
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)

    assert cli.main(["pull", "hp", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "hp" in output
    assert "application/rdf+xml" in output


def test_cli_plan_json_output(monkeypatch, stub_logger, capsys):
    planned = PlannedFetch(
        spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"]),
        resolver="obo",
        plan=FetchPlan(
            url="https://example.org/hp.owl",
            headers={},
            filename_hint=None,
            version="2024",
            license="CC-BY",
            media_type="application/rdf+xml",
            service="obo",
        ),
    )

    monkeypatch.setattr(cli, "plan_all", lambda specs, config: [planned])
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)

    assert cli.main(["plan", "hp", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["id"] == "hp"
    assert payload[0]["url"] == "https://example.org/hp.owl"


def test_cli_validate_json_output(monkeypatch, stub_logger, tmp_path, capsys):
    manifest_dir = tmp_path / "hp" / "2024"
    original_dir = manifest_dir / "original"
    validation_dir = manifest_dir / "validation"
    normalized_dir = manifest_dir / "normalized"
    original_dir.mkdir(parents=True)
    validation_dir.mkdir()
    normalized_dir.mkdir()
    original_file = original_dir / "hp.owl"
    original_file.write_text("content")
    manifest = {
        "id": "hp",
        "resolver": "obo",
        "url": "https://example.org/hp.owl",
        "filename": "hp.owl",
        "version": "2024",
        "license": "CC-BY",
        "status": "fresh",
        "sha256": "abc",
        "normalized_sha256": "def",
        "fingerprint": "f" * 64,
        "etag": "tag",
        "last_modified": "today",
        "downloaded_at": "now",
        "target_formats": ["owl"],
        "validation": {},
        "artifacts": [str(original_file)],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(
        cli,
        "run_validators",
        lambda requests, logger: {
            req.name: type("R", (), {"to_dict": lambda self: {"ok": True}})() for req in requests
        },
    )
    monkeypatch.setattr(cli, "STORAGE", _StubStorage(tmp_path))
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    assert cli.main(["validate", "hp", "2024", "--json", "--rdflib"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert "rdflib" in payload


def test_cli_validate_table_output(monkeypatch, stub_logger, tmp_path, capsys):
    manifest_dir = tmp_path / "hp" / "2024"
    original_dir = manifest_dir / "original"
    validation_dir = manifest_dir / "validation"
    normalized_dir = manifest_dir / "normalized"
    original_dir.mkdir(parents=True)
    validation_dir.mkdir()
    normalized_dir.mkdir()
    original_file = original_dir / "hp.owl"
    original_file.write_text("content")
    manifest = {
        "id": "hp",
        "resolver": "obo",
        "url": "https://example.org/hp.owl",
        "filename": "hp.owl",
        "version": "2024",
        "license": "CC-BY",
        "status": "fresh",
        "sha256": "abc",
        "normalized_sha256": "def",
        "fingerprint": "f" * 64,
        "etag": "tag",
        "last_modified": "today",
        "downloaded_at": "now",
        "target_formats": ["owl"],
        "validation": {},
        "artifacts": [str(original_file)],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)

    class _Result:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return self._payload

    monkeypatch.setattr(
        cli,
        "run_validators",
        lambda requests, logger: {
            "rdflib": _Result({"ok": True, "details": {"triples": 1234}}),
        },
    )
    monkeypatch.setattr(cli, "STORAGE", _StubStorage(tmp_path))
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )

    assert cli.main(["validate", "hp", "2024", "--rdflib"]) == 0
    output = capsys.readouterr().out
    lines = output.splitlines()
    assert "validator" in lines[0]
    assert any("rdflib" in line and "triples=1234" in line for line in lines)


def test_cli_config_validate(monkeypatch, stub_logger, tmp_path, capsys):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text("ontologies:\n  - id: hp\n    resolver: obo\n")
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    assert cli.main(["config", "validate", "--spec", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Configuration passed" in output


def test_cli_config_validate_json(monkeypatch, stub_logger, tmp_path, capsys):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text("ontologies:\n  - id: hp\n    resolver: obo\n")
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    assert cli.main(["config", "validate", "--spec", str(config_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"]


def test_cli_config_validate_missing_file(monkeypatch, stub_logger, tmp_path):
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(
        cli.ResolvedConfig,
        "from_defaults",
        classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfig(), specs=())),
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["config", "validate", "--spec", str(missing)])
    assert exc_info.value.code == 2


def test_cli_init_writes_example(monkeypatch, stub_logger, tmp_path, capsys):
    target = tmp_path / "sources.yaml"
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)

    exit_code = cli.main(["init", str(target)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote example configuration" in output
    contents = target.read_text()
    assert "defaults:" in contents
    assert "ontologies:" in contents


def test_cli_init_refuses_overwrite(monkeypatch, stub_logger, tmp_path, capsys):
    target = tmp_path / "sources.yaml"
    target.write_text("existing")
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)

    exit_code = cli.main(["init", str(target)])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Refusing to overwrite" in stderr


def test_cli_doctor_json(monkeypatch, stub_logger, capsys):
    monkeypatch.setattr(
        cli.requests, "get", lambda url, timeout: SimpleNamespace(ok=True, status_code=200)
    )
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)

    assert cli.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "directories" in payload
    assert "dependencies" in payload
