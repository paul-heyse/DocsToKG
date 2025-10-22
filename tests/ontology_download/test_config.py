# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_config",
#   "purpose": "Configuration model regression tests.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Configuration model regression tests.

Exercises the layered configuration system: YAML parsing, defaults merging,
environment overrides, validation of allowed hosts/rate limits, and backwards
compatibility helpers. Ensures the CLI and API receive well-formed
``ResolvedConfig`` objects and that misconfigurations raise descriptive errors."""

from __future__ import annotations

import logging
import os
import textwrap
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from pydantic import BaseModel, ValidationError

from DocsToKG.OntologyDownload import api as core
from DocsToKG.OntologyDownload import planning as pipeline_mod
from DocsToKG.OntologyDownload.errors import OntologyDownloadError
from DocsToKG.OntologyDownload.planning import (
    ConfigurationError,
    FetchPlan,
    ResolverError,
    merge_defaults,
)
from DocsToKG.OntologyDownload.settings import (
    ConfigError,
    DefaultsConfig,
    DownloadConfiguration,
    LoggingConfiguration,
    LoggingSettings,
    ResolvedConfig,
    ValidationConfig,
    build_resolved_config,
    get_env_overrides,
    load_config,
    load_raw_yaml,
    validate_config,
)
from DocsToKG.OntologyDownload.testing import ResponseSpec, temporary_resolver


@contextmanager
def temporary_env(**overrides: str):
    previous = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


# --- Test Cases ---


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


def test_resolved_config_serializes_logging_settings(recwarn) -> None:
    """ResolvedConfig should respect emit_json_logs and serialize without warnings."""

    resolved = ResolvedConfig(defaults=DefaultsConfig(), specs=[])
    runtime_logging = LoggingSettings(json=False)

    assert runtime_logging.emit_json_logs is False

    payload = resolved.model_dump_json()
    assert isinstance(payload, str)
    assert payload
    assert len(recwarn) == 0


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


def test_download_config_ignores_legacy_max_bytes() -> None:
    """Legacy max-bytes values should be ignored instead of raising errors."""

    raw_config = {
        "defaults": {
            "http": {
                "max_download_size_gb": 5,
                "timeout_sec": 45,
            }
        },
        "ontologies": [],
    }

    resolved = build_resolved_config(raw_config)
    http_settings = resolved.defaults.http.model_dump()
    assert "max_download_size_gb" not in http_settings


def test_download_config_service_rate_limits() -> None:
    """Per-service rate limits should parse to per-second floats."""

    config = DownloadConfiguration(
        rate_limits={"obo": "2/second", "ols": "5/minute", "bioportal": "100/hour"}
    )

    assert config.parse_service_rate_limit("obo") == 2.0
    assert config.parse_service_rate_limit("ols") == pytest.approx(5.0 / 60.0)
    assert config.parse_service_rate_limit("bioportal") == pytest.approx(100.0 / 3600.0)
    assert config.parse_service_rate_limit("unknown") is None


def test_download_config_service_rate_limits_invalid() -> None:
    """Bad per-service rate limits should surface immediately."""

    config = DownloadConfiguration()
    config.rate_limits["obo"] = "garbage"
    with pytest.raises(ValueError, match="Invalid rate limit 'garbage'"):
        config.parse_service_rate_limit("obo")


def test_download_config_invalid_rate_limit() -> None:
    """Invalid rate limit strings should raise a ValidationError."""

    with pytest.raises(ValidationError):
        DownloadConfiguration(per_host_rate_limit="invalid")


def test_download_config_allowed_hosts() -> None:
    """Allowed host configuration should persist values."""

    config = DownloadConfiguration(allowed_hosts=["example.org", "purl.obolibrary.org"])
    assert "example.org" in config.allowed_hosts
    assert "purl.obolibrary.org" in config.allowed_hosts


def test_download_config_defaults_include_service_limits() -> None:
    """Default configuration should expose service-specific limits and planners."""

    config = DownloadConfiguration()

    assert config.concurrent_plans == 8
    assert config.rate_limits["ols"] == "4/second"
    assert config.rate_limits["bioportal"] == "2/second"
    assert config.rate_limits["lov"] == "1/second"


def test_download_config_normalizes_allowed_hosts() -> None:
    """normalized_allowed_hosts should return punycoded hostnames."""

    config = DownloadConfiguration(
        allowed_hosts=["Example.org", "*.Example.com", " münchen.example.org "]
    )

    normalized = config.normalized_allowed_hosts()
    assert normalized is not None
    exact, suffixes, ports, ip_literals = normalized
    assert "example.org" in exact
    assert "example.com" in suffixes
    assert "xn--mnchen-3ya.example.org" in exact
    assert ports == {}
    assert ip_literals == set()


def test_download_config_polite_headers_defaults() -> None:
    """Polite HTTP headers should include defaults and tracing identifiers."""

    config = DownloadConfiguration(polite_headers={})
    headers = config.polite_http_headers(
        correlation_id="corr123",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    assert headers["User-Agent"].startswith("DocsToKG-OntologyDownloader/1.0")
    assert headers["X-Request-ID"].startswith("corr123-20240101T000000Z")
    assert "From" not in headers


def test_download_config_parses_allowed_host_ports() -> None:
    config = DownloadConfiguration(allowed_hosts=["example.org:8443"])

    normalized = config.normalized_allowed_hosts()
    assert normalized is not None
    exact, suffixes, ports, ip_literals = normalized
    assert "example.org" in exact
    assert not suffixes
    assert ports["example.org"] == {8443}
    assert ip_literals == set()


def test_download_config_identifies_ip_literal_allowlist_entries() -> None:
    config = DownloadConfiguration(allowed_hosts=["10.0.0.5", "[2001:db8::1]"])

    normalized = config.normalized_allowed_hosts()
    assert normalized is not None
    exact, suffixes, ports, ip_literals = normalized
    assert {"10.0.0.5", "2001:db8::1"}.issubset(exact)
    assert ip_literals == {"10.0.0.5", "2001:db8::1"}
    assert ports == {}
    assert not suffixes


def test_download_config_rejects_wildcard_port_usage() -> None:
    config = DownloadConfiguration(allowed_hosts=["*.example.org:8443"])

    with pytest.raises(ValueError):
        config.normalized_allowed_hosts()


def test_download_config_polite_headers_custom_values() -> None:
    """Configured polite headers should override defaults and promote mailto."""

    config = DownloadConfiguration(
        polite_headers={
            "User-Agent": "CustomAgent/1.0",
            "mailto": "team@example.org",
        }
    )

    headers = config.polite_http_headers(
        request_id="custom",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    assert headers["User-Agent"] == "CustomAgent/1.0"
    assert headers["From"] == "team@example.org"
    assert headers["X-Request-ID"] == "custom"


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


def test_env_override_applies_settings(tmp_path: Path) -> None:
    """Environment overrides should be merged into defaults via Pydantic settings."""

    raw_config: Dict[str, object] = {"ontologies": []}

    with temporary_env(
        ONTOFETCH_MAX_RETRIES="3",
        ONTOFETCH_TIMEOUT_SEC="45",
        ONTOFETCH_SHARED_RATE_LIMIT_DIR=str(tmp_path),
        ONTOFETCH_MAX_UNCOMPRESSED_SIZE_GB="12.5",
    ):
        resolved = build_resolved_config(raw_config)
    assert resolved.defaults.http.max_retries == 3
    assert resolved.defaults.http.timeout_sec == 45
    assert resolved.defaults.http.shared_rate_limit_dir == tmp_path
    assert resolved.defaults.http.max_uncompressed_size_gb == pytest.approx(12.5)


def test_get_env_overrides_backwards_compatible(tmp_path: Path) -> None:
    """get_env_overrides should expose overrides for legacy code paths."""

    with temporary_env(
        ONTOFETCH_BACKOFF_FACTOR="1.5",
        ONTOFETCH_SHARED_RATE_LIMIT_DIR=str(tmp_path),
        ONTOFETCH_MAX_UNCOMPRESSED_SIZE_GB="15",
    ):
        overrides = get_env_overrides()
    assert overrides["backoff_factor"] == "1.5"
    assert overrides["shared_rate_limit_dir"] == str(tmp_path)
    assert float(overrides["max_uncompressed_size_gb"]) == pytest.approx(15.0)


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


def test_load_raw_yaml_missing_file_raises_config_error(tmp_path: Path) -> None:
    """Missing YAML files should raise ConfigError instead of exiting."""

    path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError) as exc_info:
        load_raw_yaml(path)
    assert "Configuration file not found" in str(exc_info.value)


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    """Invalid YAML should raise ConfigError with context."""

    config_path = tmp_path / "sources.yaml"
    config_path.write_text("defaults: [invalid")
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_fetch_one_rejects_disallowed_license() -> None:
    """fetch_one should reject ontologies with disallowed licenses."""

    class StubResolver:
        def plan(self, spec, config, logger, *, cancellation_token=None):
            return FetchPlan(
                url="https://example.org/onto.owl",
                headers={},
                filename_hint="onto.owl",
                version="2024",
                license="Proprietary",
                media_type="application/rdf+xml",
                service=spec.resolver,
            )

    config = ResolvedConfig(defaults=DefaultsConfig(accept_licenses=["CC0-1.0"]), specs=[])
    spec = pipeline_mod.FetchSpec(id="example", resolver="stub", extras={}, target_formats=["owl"])

    with temporary_resolver("stub", StubResolver()):
        with pytest.raises(ConfigurationError):
            pipeline_mod.fetch_one(spec, config=config, force=True, logger=_noop_logger())


def test_fetch_one_unknown_resolver() -> None:
    """Unknown resolver should raise ResolverError."""

    spec = core.FetchSpec(id="example", resolver="missing", extras={}, target_formats=["owl"])
    with pytest.raises(ResolverError):
        core.fetch_one(
            spec, config=ResolvedConfig.from_defaults(), force=True, logger=_noop_logger()
        )


def test_fetch_one_download_error(ontology_env) -> None:
    """Download failures should be wrapped in OntologyDownloadError."""

    resolver_name = "obo-test-error"
    failing_path = "failures/error.owl"
    failing_url = ontology_env.http_url(failing_path)
    headers = {"Content-Type": "application/rdf+xml"}
    ontology_env.queue_response(
        failing_path, ResponseSpec(method="HEAD", status=200, headers=headers)
    )
    ontology_env.queue_response(
        failing_path, ResponseSpec(method="GET", status=503, headers=headers)
    )

    class StubResolver:
        NAME = resolver_name

        def plan(self, spec, config, logger):
            return FetchPlan(
                url=failing_url,
                headers={},
                filename_hint="error.owl",
                version="2024",
                license="CC0-1.0",
                media_type="application/rdf+xml",
                service=spec.resolver,
            )

    spec = pipeline_mod.FetchSpec(
        id="hp", resolver=resolver_name, extras={}, target_formats=["owl"]
    )

    config = ontology_env.build_resolved_config()
    config.defaults.http = ontology_env.build_download_config()

    with temporary_resolver(resolver_name, StubResolver()):
        with pytest.raises(OntologyDownloadError):
            pipeline_mod.fetch_one(spec, config=config, force=True, logger=_noop_logger())


# --- Helper Functions ---


def _noop_logger() -> logging.Logger:
    logger = logging.getLogger("ontology-download-test")
    logger.setLevel(logging.INFO)
    return logger
