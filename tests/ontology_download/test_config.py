"""Tests for ontology downloader configuration models and helpers."""

from __future__ import annotations

import io
import logging
import sys
import textwrap
from pathlib import Path
from typing import Dict

import pytest

pytest.importorskip("pydantic")

from pydantic import BaseModel, ValidationError

from DocsToKG.OntologyDownload import core, resolvers
from DocsToKG.OntologyDownload.config import (
    ConfigError,
    DefaultsConfig,
    DownloadConfiguration,
    LoggingConfiguration,
    ResolvedConfig,
    ValidationConfig,
    build_resolved_config,
    get_env_overrides,
    load_config,
    load_raw_yaml,
    merge_defaults,
    validate_config,
)


def test_merge_defaults_creates_fetch_spec() -> None:
    """merge_defaults should return a FetchSpec when defaults are supplied."""

    defaults = DefaultsConfig()
    spec = merge_defaults({"id": "hp"}, defaults)

    assert isinstance(spec, core.FetchSpec)
    assert spec.id == "hp"


def test_logging_config_defaults() -> None:
    """LoggingConfiguration should expose documented defaults."""

    config = LoggingConfiguration()
    assert config.level == "INFO"
    assert config.max_log_size_mb == 100
    assert config.retention_days == 30


def test_logging_config_validates_level() -> None:
    """Invalid logging level should raise a ValidationError."""

    with pytest.raises(ValidationError) as exc_info:
        LoggingConfiguration(level="invalid")
    assert "level must be one of" in str(exc_info.value)


def test_logging_config_case_insensitive_level() -> None:
    """Lowercase logging level should be normalized to uppercase."""

    config = LoggingConfiguration(level="debug")
    assert config.level == "DEBUG"


def test_logging_config_validates_positive_values() -> None:
    """Max log size must be strictly positive."""

    with pytest.raises(ValidationError):
        LoggingConfiguration(max_log_size_mb=0)
    with pytest.raises(ValidationError):
        LoggingConfiguration(max_log_size_mb=-5)


def test_validation_config_timeout_bounds() -> None:
    """ValidationConfig should enforce sensible timeout bounds."""

    with pytest.raises(ValidationError):
        ValidationConfig(parser_timeout_sec=0)
    with pytest.raises(ValidationError):
        ValidationConfig(parser_timeout_sec=4000)

    config = ValidationConfig(parser_timeout_sec=120)
    assert config.parser_timeout_sec == 120


def test_download_config_rate_limit_formats() -> None:
    """DownloadConfiguration should normalize rate limit units."""

    config = DownloadConfiguration(per_host_rate_limit="5/second")
    assert config.rate_limit_per_second() == 5.0

    config = DownloadConfiguration(per_host_rate_limit="60/minute")
    assert config.rate_limit_per_second() == 1.0

    config = DownloadConfiguration(per_host_rate_limit="0.5/second")
    assert config.rate_limit_per_second() == 0.5

    config = DownloadConfiguration(per_host_rate_limit="3600/hour")
    assert config.rate_limit_per_second() == pytest.approx(1.0)


def test_download_config_service_rate_limits() -> None:
    """Per-service rate limits should parse to per-second floats."""

    config = DownloadConfiguration(
        rate_limits={"obo": "2/second", "ols": "5/minute", "bioportal": "100/hour"}
    )

    assert config.parse_service_rate_limit("obo") == 2.0
    assert config.parse_service_rate_limit("ols") == pytest.approx(5.0 / 60.0)
    assert config.parse_service_rate_limit("bioportal") == pytest.approx(100.0 / 3600.0)
    assert config.parse_service_rate_limit("unknown") is None


def test_download_config_invalid_rate_limit() -> None:
    """Invalid rate limit strings should raise a ValidationError."""

    with pytest.raises(ValidationError):
        DownloadConfiguration(per_host_rate_limit="invalid")


def test_download_config_allowed_hosts() -> None:
    """Allowed host configuration should persist values."""

    config = DownloadConfiguration(allowed_hosts=["example.org", "purl.obolibrary.org"])
    assert "example.org" in config.allowed_hosts
    assert "purl.obolibrary.org" in config.allowed_hosts


def test_download_config_normalizes_allowed_hosts() -> None:
    """normalized_allowed_hosts should return punycoded hostnames."""

    config = DownloadConfiguration(
        allowed_hosts=["Example.org", "*.Example.com", " münchen.example.org "]
    )

    normalized = config.normalized_allowed_hosts()
    assert normalized is not None
    exact, suffixes = normalized
    assert "example.org" in exact
    assert "example.com" in suffixes
    assert "xn--mnchen-3ya.example.org" in exact


def test_download_config_polite_headers_defaults() -> None:
    """Polite headers should provide defaults when not configured."""

    config = DownloadConfiguration()
    headers = config.build_polite_headers(correlation_id="corr-123")
    assert headers["User-Agent"].startswith("DocsToKG-OntologyDownloader/")
    assert headers["X-Request-ID"].startswith("corr-123-")


def test_download_config_polite_headers_overrides() -> None:
    """Configured polite headers should be preserved and augmented."""

    config = DownloadConfiguration(
        polite_headers={"User-Agent": "Custom/1.0", "From": "ops@example.org"}
    )
    headers = config.build_polite_headers()
    assert headers["User-Agent"] == "Custom/1.0"
    assert headers["From"] == "ops@example.org"
    assert headers["X-Request-ID"].startswith("ontofetch-")


def test_defaults_config_pydantic() -> None:
    """DefaultsConfig should inherit from BaseModel for Pydantic features."""

    assert issubclass(DefaultsConfig, BaseModel)


def test_defaults_config_json_schema() -> None:
    """DefaultsConfig should provide a JSON schema."""

    schema = DefaultsConfig.model_json_schema()
    assert "properties" in schema
    assert "accept_licenses" in schema["properties"]
    assert schema["properties"]["accept_licenses"]["type"] == "array"


def test_defaults_config_prefer_source_validation() -> None:
    """Unknown resolver names should trigger validation errors."""

    with pytest.raises(ValidationError):
        DefaultsConfig(prefer_source=["obo", "invalid-resolver"])


def test_env_override_applies_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment overrides should be merged into defaults via Pydantic settings."""

    monkeypatch.setenv("ONTOFETCH_MAX_RETRIES", "3")
    monkeypatch.setenv("ONTOFETCH_TIMEOUT_SEC", "45")
    raw_config: Dict[str, object] = {"ontologies": []}

    resolved = build_resolved_config(raw_config)
    assert resolved.defaults.http.max_retries == 3
    assert resolved.defaults.http.timeout_sec == 45


def test_get_env_overrides_backwards_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_env_overrides should expose overrides for legacy code paths."""

    monkeypatch.setenv("ONTOFETCH_BACKOFF_FACTOR", "1.5")
    overrides = get_env_overrides()
    assert overrides["backoff_factor"] == "1.5"


def test_load_config_parses_defaults(tmp_path: Path) -> None:
    """load_config should merge defaults and ontologies from YAML."""

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


def test_validate_config_rejects_missing_id(tmp_path: Path) -> None:
    """Validation should reject ontology entries missing an identifier."""

    config_path = tmp_path / "sources.yaml"
    config_path.write_text("ontologies:\n  - resolver: obo\n")
    with pytest.raises(ConfigError):
        validate_config(config_path)


def test_validate_config_unknown_key(tmp_path: Path) -> None:
    """Unknown keys in defaults should raise ConfigError."""

    config_path = tmp_path / "sources.yaml"
    config_path.write_text("defaults:\n  unknown: value\nontologies:\n  - id: hp\n")
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config_path)
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_validate_config_rejects_non_mapping_extras(tmp_path: Path) -> None:
    """Extras must be mappings when provided."""

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


def test_load_raw_yaml_missing_file_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing YAML files should exit with status code 2 and helpful message."""

    path = tmp_path / "missing.yaml"
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    with pytest.raises(SystemExit) as exc_info:
        load_raw_yaml(path)
    assert exc_info.value.code == 2
    assert "Configuration file not found" in stderr.getvalue()


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    """Invalid YAML should raise ConfigError with context."""

    config_path = tmp_path / "sources.yaml"
    config_path.write_text("defaults: [invalid")
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_fetch_one_rejects_disallowed_license(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_one should reject ontologies with disallowed licenses."""

    class StubResolver:
        def plan(self, spec, config, logger):
            return resolvers.FetchPlan(
                url="https://example.org/onto.owl",
                headers={},
                filename_hint="onto.owl",
                version="2024",
                license="Proprietary",
                media_type="application/rdf+xml",
                service=spec.resolver,
            )

    monkeypatch.setitem(resolvers.RESOLVERS, "stub", StubResolver())
    config = ResolvedConfig(defaults=DefaultsConfig(accept_licenses=["CC0-1.0"]), specs=[])
    spec = core.FetchSpec(id="example", resolver="stub", extras={}, target_formats=["owl"])

    with pytest.raises(core.ConfigurationError):
        core.fetch_one(spec, config=config, force=True, logger=_noop_logger())


def test_fetch_one_unknown_resolver() -> None:
    """Unknown resolver should raise ResolverError."""

    spec = core.FetchSpec(id="example", resolver="missing", extras={}, target_formats=["owl"])
    with pytest.raises(core.ResolverError):
        core.fetch_one(
            spec, config=ResolvedConfig.from_defaults(), force=True, logger=_noop_logger()
        )


def test_fetch_one_download_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Download failures should be wrapped in OntologyDownloadError."""

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
                service=spec.resolver,
            )

    monkeypatch.setitem(resolvers.RESOLVERS, "obo", StubResolver())
    monkeypatch.setattr(
        core, "download_stream", lambda **_: (_ for _ in ()).throw(ConfigError("boom"))
    )

    with pytest.raises(core.OntologyDownloadError):
        core.fetch_one(
            spec, config=ResolvedConfig.from_defaults(), force=True, logger=_noop_logger()
        )


def _noop_logger() -> logging.Logger:
    logger = logging.getLogger("ontology-download-test")
    logger.setLevel(logging.INFO)
    return logger
