"""
Ontology Configuration Tests

This module validates configuration parsing, environment overrides,
resolver selection, and error handling for the ontology download system.

Key Scenarios:
- Parses defaults/ontologies from YAML with environment variable merges
- Rejects invalid schema keys or malformed extras blocks
- Propagates resolver and download failures as domain-specific errors

Dependencies:
- pytest: Assertions and monkeypatch utilities
- DocsToKG.OntologyDownload.config/core: Configuration helpers under test

Usage:
    pytest tests/ontology_download/test_config.py
"""

import logging
import textwrap

import io
import sys
import textwrap

import pytest

from DocsToKG.OntologyDownload import core, resolvers
from DocsToKG.OntologyDownload.config import (
    ConfigError,
    DefaultsConfiguration,
    ResolvedConfig,
    load_raw_yaml,
    load_config,
    merge_defaults,
    validate_config,
)


def test_load_config_parses_defaults(tmp_path):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            defaults:
              http:
                max_retries: 7
              validation:
                parser_timeout_sec: 45
            ontologies:
              - id: hp
                resolver: obo
            """
        )
    )
    config = load_config(config_path)
    assert config.defaults.http.max_retries == 7
    assert config.specs[0].id == "hp"


def test_validate_config_rejects_missing_id(tmp_path):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text("ontologies:\n  - resolver: obo\n")
    with pytest.raises(ConfigError):
        validate_config(config_path)


def test_env_override_max_retries(monkeypatch):
    monkeypatch.setenv("ONTOFETCH_MAX_RETRIES", "3")
    config = merge_defaults({"ontologies": []})
    assert config.defaults.http.max_retries == 3


def test_fetch_one_rejects_disallowed_license(monkeypatch):
    class StubResolver:
        def plan(self, spec, config, logger):
            return resolvers.FetchPlan(
                url="https://example.org/onto.owl",
                headers={},
                filename_hint="onto.owl",
                version="2024",
                license="Proprietary",
                media_type="application/rdf+xml",
            )

    monkeypatch.setitem(resolvers.RESOLVERS, "stub", StubResolver())
    config = ResolvedConfig(defaults=DefaultsConfiguration(accept_licenses=["CC0-1.0"]), specs=())
    spec = core.FetchSpec(id="example", resolver="stub", extras={}, target_formats=["owl"])
    with pytest.raises(core.ConfigurationError):
        core.fetch_one(spec, config=config, force=True, logger=_noop_logger())


def test_fetch_one_unknown_resolver():
    spec = core.FetchSpec(id="example", resolver="missing", extras={}, target_formats=["owl"])
    with pytest.raises(core.ResolverError):
        core.fetch_one(spec, config=ResolvedConfig.from_defaults(), force=True, logger=_noop_logger())


def test_fetch_one_download_error(monkeypatch):
    spec = core.FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])

    class StubResolver:
        def plan(self, spec, config, logger):
            return resolvers.FetchPlan(
                url="https://example.org/hp.owl",
                headers={},
                filename_hint="hp.owl",
                version="2024",
                license="CC0-1.0",
                media_type="application/rdf+xml",
            )

    monkeypatch.setitem(resolvers.RESOLVERS, "obo", StubResolver())
    monkeypatch.setattr(core, "download_stream", lambda **_: (_ for _ in ()).throw(ConfigError("boom")))
    with pytest.raises(core.OntologyDownloadError):
        core.fetch_one(spec, config=ResolvedConfig.from_defaults(), force=True, logger=_noop_logger())


def test_validate_config_unknown_key(tmp_path):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        "defaults:\n  unknown: value\nontologies:\n  - id: hp\n"
    )
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config_path)
    assert "Unknown key" in str(exc_info.value)


def test_validate_config_invalid_yaml(tmp_path):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text("defaults: [invalid")
    with pytest.raises(ConfigError):
        validate_config(config_path)


def test_load_raw_yaml_missing_file_exits(tmp_path, monkeypatch):
    path = tmp_path / "missing.yaml"
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    with pytest.raises(SystemExit) as exc_info:
        load_raw_yaml(path)
    assert exc_info.value.code == 2
    assert "Configuration file not found" in stderr.getvalue()


def test_validate_config_rejects_non_mapping_extras(tmp_path):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            ontologies:
              - id: hp
                resolver: obo
                extras: invalid
            """
        )
    )
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config_path)
    assert "extras" in str(exc_info.value)


def _noop_logger():
    logger = logging.getLogger("ontology-download-test")
    logger.setLevel(logging.INFO)
    return logger
