
import json

import pytest

from DocsToKG.OntologyDownload import cli
from DocsToKG.OntologyDownload.config import DefaultsConfiguration, ResolvedConfig
from DocsToKG.OntologyDownload.core import FetchResult, FetchSpec


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
    monkeypatch.setattr(cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfiguration(), specs=())))
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
    monkeypatch.setattr(cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfiguration(), specs=())))
    assert cli.main(["pull", "hp"]) == 0
    output = capsys.readouterr().out
    assert "hp" in output
    assert "resolver" in output.splitlines()[0]


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
        "etag": "tag",
        "last_modified": "today",
        "downloaded_at": "now",
        "target_formats": ["owl"],
        "validation": {},
        "artifacts": [str(original_file)],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(cli, "run_validators", lambda requests, logger: {req.name: type("R", (), {"to_dict": lambda self: {"ok": True}})() for req in requests})
    monkeypatch.setattr(cli, "ONTOLOGY_DIR", tmp_path)
    monkeypatch.setattr(cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfiguration(), specs=())))
    assert cli.main(["validate", "hp", "2024", "--json", "--rdflib"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert "rdflib" in payload


def test_cli_config_validate(monkeypatch, stub_logger, tmp_path, capsys):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text("ontologies:\n  - id: hp\n    resolver: obo\n")
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfiguration(), specs=())))
    assert cli.main(["config", "validate", "--spec", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Configuration passed" in output


def test_cli_config_validate_json(monkeypatch, stub_logger, tmp_path, capsys):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text("ontologies:\n  - id: hp\n    resolver: obo\n")
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfiguration(), specs=())))
    assert cli.main(["config", "validate", "--spec", str(config_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"]


def test_cli_config_validate_missing_file(monkeypatch, stub_logger, tmp_path):
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(cli, "setup_logging", lambda *_: stub_logger)
    monkeypatch.setattr(cli.ResolvedConfig, "from_defaults", classmethod(lambda cls: ResolvedConfig(defaults=DefaultsConfiguration(), specs=())))
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["config", "validate", "--spec", str(missing)])
    assert exc_info.value.code == 2
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
