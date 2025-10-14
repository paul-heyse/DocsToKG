"""
Ontology Download Configuration

This module centralizes configuration models, parsing helpers, and validation
logic for the ontology downloader service. It reads YAML configuration files,
merges default values, applies environment overrides, and ensures that the
resulting settings are ready for downstream document processing and download
pipelines.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import yaml

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .core import FetchSpec


class ConfigError(RuntimeError):
    """Raised when ontology configuration files are invalid or inconsistent."""


PYTHON_MIN_VERSION = (3, 9)


def ensure_python_version() -> None:
    """Ensure the interpreter meets the minimum supported version.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: If Python is older than the minimum version supported by
            the ontology downloader tooling.
    """

    if sys.version_info < PYTHON_MIN_VERSION:
        print("Error: Python 3.9+ required", file=sys.stderr)
        raise SystemExit(1)


@dataclass(slots=True)
class LoggingConfiguration:
    """Structured logging options for ontology download operations.

    Attributes:
        level: Logging level name (e.g., `INFO`, `DEBUG`).
        max_log_size_mb: Maximum individual log file size before rotation.
        retention_days: Number of days to keep logs prior to compression.
    """
    level: str = "INFO"
    max_log_size_mb: int = 100
    retention_days: int = 30


@dataclass(slots=True)
class ValidationConfiguration:
    """Settings that control ontology validation behaviours.

    Attributes:
        skip_reasoning_if_size_mb: Threshold that disables reasoning for large
            ontology downloads, preventing excessive processing time.
        parser_timeout_sec: Maximum parsing duration before timing out.
    """
    skip_reasoning_if_size_mb: int = 500
    parser_timeout_sec: int = 60


@dataclass(slots=True)
class DownloadConfiguration:
    """HTTP download and retry settings for ontology sources.

    Attributes:
        max_retries: Number of retries for transient download failures.
        timeout_sec: Timeout for request connection and read operations.
        download_timeout_sec: Upper bound for whole file downloads.
        backoff_factor: Backoff multiplier used between retry attempts.
        per_host_rate_limit: Token bucket rate limit definition per host.
        max_download_size_gb: Maximum allowed ontology archive size.
    """
    max_retries: int = 5
    timeout_sec: int = 30
    download_timeout_sec: int = 300
    backoff_factor: float = 0.5
    per_host_rate_limit: str = "4/second"
    max_download_size_gb: int = 5

    def rate_limit_per_second(self) -> float:
        """Convert the textual rate limit to a per-second float value.

        Args:
            None

        Returns:
            Number of allowed download tokens per second for each host.

        Raises:
            ConfigError: If the configured rate limit does not use a per-second
            unit or the numeric portion cannot be parsed.
        """
        value, _, unit = self.per_host_rate_limit.partition("/")
        try:
            tokens = float(value)
        except ValueError as exc:  # pragma: no cover - validated earlier
            raise ConfigError(f"Invalid per_host_rate_limit value: {self.per_host_rate_limit}") from exc
        if unit.strip() not in {"second", "sec", "s"}:
            raise ConfigError("per_host_rate_limit must be expressed per second")
        return tokens


@dataclass(slots=True)
class DefaultsConfiguration:
    """Collection of default settings applied to ontology fetch specifications.

    Attributes:
        accept_licenses: License identifiers allowed for automated downloads.
        normalize_to: Output formats to produce after ontology normalization.
        prefer_source: Resolver preference order when one is not specified.
        http: Download configuration shared across ontology fetch operations.
        validation: Validation configuration for ontologies post-download.
        logging: Logging configuration applied to the downloader runtime.
        continue_on_error: Whether to proceed after non-fatal download errors.
        concurrent_downloads: Maximum concurrent download workers to run.
    """
    accept_licenses: Sequence[str] = field(default_factory=lambda: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"])
    normalize_to: Sequence[str] = field(default_factory=lambda: ["ttl"])
    prefer_source: Sequence[str] = field(default_factory=lambda: ["obo", "ols", "bioportal", "direct"])
    http: DownloadConfiguration = field(default_factory=DownloadConfiguration)
    validation: ValidationConfiguration = field(default_factory=ValidationConfiguration)
    logging: LoggingConfiguration = field(default_factory=LoggingConfiguration)
    continue_on_error: bool = True
    concurrent_downloads: int = 1


DEFAULT_MAX_CONCURRENT_DOWNLOADS = 3


@dataclass(slots=True)
class ResolvedConfig:
    """Container for merged configuration defaults and fetch specifications.

    Attributes:
        defaults: Finalized default settings applied to fetch specs.
        specs: Sequence of ontology fetch specifications to execute.
    """
    defaults: DefaultsConfiguration
    specs: Sequence["FetchSpec"]

    @classmethod
    def from_defaults(cls) -> "ResolvedConfig":
        """Build an empty resolved configuration using library defaults.

        Args:
            None

        Returns:
            ResolvedConfig with default settings and no fetch specifications.
        """
        return cls(defaults=DefaultsConfiguration(), specs=())


def _coerce_sequence(value: Optional[Iterable[str]]) -> List[str]:
    """Normalize configuration entries into a list of strings.

    Args:
        value: Raw configuration value which may be None, a string, or an
            iterable of string-like items.

    Returns:
        List of string values that are safe for downstream processing.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def get_env_overrides() -> Dict[str, str]:
    """Extract ontology downloader overrides from environment variables.

    Args:
        None

    Returns:
        Mapping of normalized environment keys (lowercase without prefix) to
        their string values for supported overrides.
    """
    prefix = "ONTOFETCH_"
    return {key[len(prefix) :].lower(): value for key, value in os.environ.items() if key.startswith(prefix)}


def merge_defaults(raw_config: Mapping[str, object]) -> ResolvedConfig:
    """Merge YAML configuration with defaults and environment overrides.

    Args:
        raw_config: Configuration tree parsed from `sources.yaml`.

    Returns:
        Resolved configuration that includes defaults and concrete fetch specs.

    Raises:
        ConfigError: If the configuration structure is invalid.
    """
    defaults_section = raw_config.get("defaults", {})
    if not isinstance(defaults_section, Mapping):
        raise ConfigError("'defaults' section must be a mapping")

    http_defaults = defaults_section.get("http", {})
    validation_defaults = defaults_section.get("validation", {})
    logging_defaults = defaults_section.get("logging", {})

    defaults = DefaultsConfiguration(
        accept_licenses=_coerce_sequence(defaults_section.get("accept_licenses")) or DefaultsConfiguration().accept_licenses,
        normalize_to=_coerce_sequence(defaults_section.get("normalize_to")) or DefaultsConfiguration().normalize_to,
        prefer_source=_coerce_sequence(defaults_section.get("prefer_source")) or DefaultsConfiguration().prefer_source,
        http=DownloadConfiguration(
            max_retries=int(http_defaults.get("max_retries", 5)),
            timeout_sec=int(http_defaults.get("timeout_sec", 30)),
            download_timeout_sec=int(http_defaults.get("download_timeout_sec", 300)),
            backoff_factor=float(http_defaults.get("backoff_factor", 0.5)),
            per_host_rate_limit=str(http_defaults.get("per_host_rate_limit", "4/second")),
            max_download_size_gb=int(http_defaults.get("max_download_size_gb", 5)),
        ),
        validation=ValidationConfiguration(
            skip_reasoning_if_size_mb=int(validation_defaults.get("skip_reasoning_if_size_mb", 500)),
            parser_timeout_sec=int(validation_defaults.get("parser_timeout_sec", 60)),
        ),
        logging=LoggingConfiguration(
            level=str(logging_defaults.get("level", "INFO")),
            max_log_size_mb=int(logging_defaults.get("max_log_size_mb", 100)),
            retention_days=int(logging_defaults.get("retention_days", 30)),
        ),
        continue_on_error=bool(defaults_section.get("continue_on_error", True)),
        concurrent_downloads=int(defaults_section.get("concurrent_downloads", 1)),
    )

    env_overrides = get_env_overrides()
    if "max_retries" in env_overrides:
        defaults.http.max_retries = int(env_overrides["max_retries"])
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_MAX_RETRIES", "value": env_overrides["max_retries"]},
        )
    if "timeout_sec" in env_overrides:
        defaults.http.timeout_sec = int(env_overrides["timeout_sec"])
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_TIMEOUT_SEC", "value": env_overrides["timeout_sec"]},
        )
    if "per_host_rate_limit" in env_overrides:
        defaults.http.per_host_rate_limit = env_overrides["per_host_rate_limit"]
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_PER_HOST_RATE_LIMIT", "value": env_overrides["per_host_rate_limit"]},
        )
    if "backoff_factor" in env_overrides:
        defaults.http.backoff_factor = float(env_overrides["backoff_factor"])
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_BACKOFF_FACTOR", "value": env_overrides["backoff_factor"]},
        )
    if "log_level" in env_overrides:
        defaults.logging.level = str(env_overrides["log_level"])
        LOGGER.info(
            "config overridden by env var",
            extra={"stage": "config", "key": "ONTOFETCH_LOG_LEVEL", "value": env_overrides["log_level"]},
        )

    ontologies = raw_config.get("ontologies", [])
    if not isinstance(ontologies, list):
        raise ConfigError("'ontologies' must be a list")

    fetch_specs: List["FetchSpec"] = []
    for index, entry in enumerate(ontologies, start=1):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"Ontology entry #{index} must be a mapping")
        ontology_id = entry.get("id")
        if not ontology_id:
            raise ConfigError(f"Ontology entry #{index} is missing required field 'id'")
        resolver_value = entry.get("resolver")
        resolver = str(resolver_value or defaults.prefer_source[0])
        extras_value = entry.get("extras", {})
        if extras_value is None:
            extras_value = {}
        if not isinstance(extras_value, Mapping):
            raise ConfigError(f"Ontology entry '{ontology_id}' extras must be a mapping if provided")
        target_formats = _coerce_sequence(entry.get("target_formats") or defaults.normalize_to)
        fetch_specs.append(
            _make_fetch_spec(str(ontology_id), resolver, extras_value, target_formats)
        )

    return ResolvedConfig(defaults=defaults, specs=fetch_specs)


def _make_fetch_spec(
    ontology_id: str,
    resolver: str,
    extras: Mapping[str, object],
    target_formats: Sequence[str],
) -> "FetchSpec":
    """Instantiate a FetchSpec from raw YAML fields.

    Args:
        ontology_id: Identifier of the ontology to retrieve.
        resolver: Resolver backend to use when locating documents.
        extras: Additional resolver-specific settings.
        target_formats: Desired normalization output formats.

    Returns:
        Fetch specification ready for ontology download orchestration.
    """
    from .core import FetchSpec as _FetchSpec  # Local import avoids circular dependency

    return _FetchSpec(id=ontology_id, resolver=resolver, extras=dict(extras), target_formats=list(target_formats))


def validate_config(config_path: Path) -> ResolvedConfig:
    """Validate a configuration file and return the resolved settings.

    Args:
        config_path: File path to a YAML config describing ontology downloads.

    Returns:
        Resolved configuration containing defaults and fetch specifications.

    Raises:
        ConfigError: If the configuration fails schema validation.
        SystemExit: If the file does not exist.
    """
    raw = load_raw_yaml(config_path)
    config = merge_defaults(raw)
    _validate_schema(raw)
    return config


def _validate_schema(raw: Mapping[str, object]) -> None:
    """Perform structural validation checks on the raw configuration mapping.

    Args:
        raw: Raw configuration mapping as loaded from YAML.

    Raises:
        ConfigError: If validation errors are discovered.
    """
    errors: List[str] = []

    defaults = raw.get("defaults")
    if defaults is not None and not isinstance(defaults, Mapping):
        errors.append("'defaults' section must be a mapping when provided")
    if isinstance(defaults, Mapping):
        for key in defaults:
            if key not in {"accept_licenses", "normalize_to", "prefer_source", "http", "validation", "logging", "continue_on_error", "concurrent_downloads"}:
                errors.append(f"Unknown key in defaults: {key}")

        http_section = defaults.get("http") if isinstance(defaults, Mapping) else None
        if http_section is not None and not isinstance(http_section, Mapping):
            errors.append("'defaults.http' must be a mapping")

        validation_section = defaults.get("validation") if isinstance(defaults, Mapping) else None
        if validation_section is not None and not isinstance(validation_section, Mapping):
            errors.append("'defaults.validation' must be a mapping")

        logging_section = defaults.get("logging") if isinstance(defaults, Mapping) else None
        if logging_section is not None and not isinstance(logging_section, Mapping):
            errors.append("'defaults.logging' must be a mapping")

    ontologies = raw.get("ontologies")
    if ontologies is None:
        errors.append("'ontologies' section is required")
    elif not isinstance(ontologies, list):
        errors.append("'ontologies' must be a list of ontology entries")
    else:
        for index, entry in enumerate(ontologies, start=1):
            if not isinstance(entry, Mapping):
                errors.append(f"Ontology entry #{index} must be a mapping")
                continue
            if "id" not in entry:
                errors.append(f"Ontology entry #{index} missing required 'id'")
            if "resolver" in entry and not isinstance(entry["resolver"], str):
                errors.append(
                    f"Ontology entry '{entry.get('id', index)}' field 'resolver' must be a string if provided"
                )
            if "extras" in entry and not isinstance(entry["extras"], Mapping):
                errors.append(
                    f"Ontology entry '{entry.get('id', index)}' field 'extras' must be a mapping if provided"
                )
            if "target_formats" in entry and not isinstance(entry["target_formats"], Iterable):
                errors.append(
                    f"Ontology entry '{entry.get('id', index)}' field 'target_formats' must be a list or iterable"
                )

    if errors:
        raise ConfigError("Configuration validation failed:\n- " + "\n- ".join(errors))


def load_raw_yaml(config_path: Path) -> MutableMapping[str, object]:
    """Load and parse a YAML configuration file into a mutable mapping.

    Args:
        config_path: Path to the YAML file to parse.

    Returns:
        Parsed mapping-compatible object suitable for validation.

    Raises:
        SystemExit: If the file cannot be located.
        ConfigError: If the YAML structure is invalid.
    """
    try:
        text = config_path.read_text()
    except FileNotFoundError:  # pragma: no cover - depends on filesystem state
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        raise SystemExit(2)

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            mark = exc.problem_mark
            raise ConfigError(f"Error in sources.yaml line {mark.line + 1}: {exc}") from exc
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(data, MutableMapping):
        raise ConfigError("Configuration root must be a mapping")
    return data


def load_config(config_path: Path) -> ResolvedConfig:
    """Load configuration from disk without performing schema validation.

    Args:
        config_path: Path to the YAML file describing ontology downloads.

    Returns:
        Resolved configuration with defaults merged and fetch specs created.
    """
    raw = load_raw_yaml(config_path)
    return merge_defaults(raw)


__all__ = [
    "ConfigError",
    "DefaultsConfiguration",
    "DownloadConfiguration",
    "LoggingConfiguration",
    "ResolvedConfig",
    "ValidationConfiguration",
    "DEFAULT_MAX_CONCURRENT_DOWNLOADS",
    "ensure_python_version",
    "get_env_overrides",
    "load_config",
    "validate_config",
]
LOGGER = logging.getLogger("DocsToKG.OntologyDownload")
