"""Configuration models, parsing, and validation for the ontology downloader."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - dependency check
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - explicit guidance for users
    raise ImportError(
        "PyYAML is required for configuration parsing. Install it with: pip install pyyaml"
    ) from exc

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .core import FetchSpec
else:  # pragma: no cover - runtime fallback to break circular import
    FetchSpec = Any  # type: ignore[misc,assignment]


PYTHON_MIN_VERSION = (3, 9)


class ConfigError(RuntimeError):
    """Raised when ontology configuration files are invalid or inconsistent."""


def ensure_python_version() -> None:
    """Ensure the interpreter meets the minimum supported Python version."""

    if sys.version_info < PYTHON_MIN_VERSION:
        print("Error: Python 3.9+ required", file=sys.stderr)
        raise SystemExit(1)


def _coerce_sequence(value: Optional[Iterable[str]]) -> List[str]:
    """Normalize configuration entries into a list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


class LoggingConfiguration(BaseModel):
    """Structured logging options for ontology download operations."""

    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    max_log_size_mb: int = Field(
        default=100,
        gt=0,
        description="Maximum log file size in megabytes before rotation",
    )
    retention_days: int = Field(
        default=30,
        ge=1,
        description="Number of days to retain log files",
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        """Validate logging level values."""

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = value.upper()
        if upper not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return upper

    model_config = {
        "frozen": False,
        "validate_assignment": True,
    }


class ValidationConfig(BaseModel):
    """Validation limits governing parser execution."""

    parser_timeout_sec: int = Field(
        default=60,
        gt=0,
        le=3600,
        description="Parser timeout in seconds (maximum one hour)",
    )
    max_memory_mb: int = Field(
        default=2048,
        gt=0,
        description="Maximum memory in megabytes allocated to validators",
    )
    skip_reasoning_if_size_mb: int = Field(
        default=500,
        gt=0,
        description="Disable reasoning when ontology exceeds this size in MB",
    )

    model_config = {
        "frozen": False,
        "validate_assignment": True,
    }


_RATE_LIMIT_PATTERN = re.compile(r"^([\d.]+)/(second|sec|s|minute|min|m|hour|h)$")


class DownloadConfiguration(BaseModel):
    """HTTP download and retry settings."""

    max_retries: int = Field(default=5, ge=0, le=20)
    timeout_sec: int = Field(default=30, gt=0, le=300)
    download_timeout_sec: int = Field(default=300, gt=0, le=3600)
    backoff_factor: float = Field(default=0.5, ge=0.1, le=10.0)
    per_host_rate_limit: str = Field(
        default="4/second",
        pattern=_RATE_LIMIT_PATTERN.pattern,
        description="Token bucket style rate limit expressed as <number>/<unit>",
    )
    max_download_size_gb: float = Field(default=5.0, gt=0, le=100.0)
    concurrent_downloads: int = Field(default=1, ge=1, le=10)
    validate_media_type: bool = Field(default=True)
    rate_limits: Dict[str, str] = Field(default_factory=dict)
    allowed_hosts: Optional[List[str]] = Field(default=None)

    @field_validator("rate_limits")
    @classmethod
    def validate_rate_limits(cls, value: Dict[str, str]) -> Dict[str, str]:
        """Ensure rate limit strings follow the expected pattern."""

        for service, limit in value.items():
            if not _RATE_LIMIT_PATTERN.match(limit):
                raise ValueError(
                    f"Invalid rate limit '{limit}' for service '{service}'. "
                    "Expected format: <number>/<unit> (e.g., '5/second', '60/min')"
                )
        return value

    def rate_limit_per_second(self) -> float:
        """Convert ``per_host_rate_limit`` to a requests-per-second value."""

        match = _RATE_LIMIT_PATTERN.match(self.per_host_rate_limit)
        if not match:
            raise ValueError(f"Invalid rate limit format: {self.per_host_rate_limit}")
        value = float(match.group(1))
        unit = match.group(2)
        if unit in {"second", "sec", "s"}:
            return value
        if unit in {"minute", "min", "m"}:
            return value / 60.0
        if unit in {"hour", "h"}:
            return value / 3600.0
        raise ValueError(f"Unknown rate limit unit: {unit}")

    def parse_service_rate_limit(self, service: str) -> Optional[float]:
        """Parse a per-service rate limit to requests per second."""

        limit_str = self.rate_limits.get(service)
        if not limit_str:
            return None
        match = _RATE_LIMIT_PATTERN.match(limit_str)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2)
        if unit in {"second", "sec", "s"}:
            return value
        if unit in {"minute", "min", "m"}:
            return value / 60.0
        if unit in {"hour", "h"}:
            return value / 3600.0
        return None

    model_config = {
        "frozen": False,
        "validate_assignment": True,
    }


_VALID_RESOLVERS = {
    "obo",
    "bioregistry",
    "ols",
    "bioportal",
    "skos",
    "xbrl",
    "lov",
    "ontobee",
    "direct",
}


class DefaultsConfig(BaseModel):
    """Collection of default settings for ontology fetch specifications."""

    accept_licenses: List[str] = Field(
        default_factory=lambda: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"]
    )
    normalize_to: List[str] = Field(default_factory=lambda: ["ttl"])
    prefer_source: List[str] = Field(
        default_factory=lambda: ["obo", "ols", "bioportal", "direct"]
    )
    http: DownloadConfiguration = Field(default_factory=DownloadConfiguration)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfiguration = Field(default_factory=LoggingConfiguration)
    continue_on_error: bool = Field(default=True)

    @field_validator("prefer_source")
    @classmethod
    def validate_prefer_source(cls, value: List[str]) -> List[str]:
        """Ensure resolver names are recognized."""

        for resolver in value:
            if resolver not in _VALID_RESOLVERS:
                raise ValueError(f"Unknown resolver '{resolver}'. Valid: {_VALID_RESOLVERS}")
        return value

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "extra": "forbid",
    }


class ResolvedConfig(BaseModel):
    """Container for merged configuration defaults and fetch specifications."""

    defaults: DefaultsConfig
    specs: List["FetchSpec"] = Field(default_factory=list)

    @classmethod
    def from_defaults(cls) -> "ResolvedConfig":
        """Create an empty resolved configuration with library defaults."""

        return cls(defaults=DefaultsConfig(), specs=[])

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


ResolvedConfig.model_rebuild()


class EnvironmentOverrides(BaseSettings):
    """Environment variable overrides for ontology downloader defaults."""

    max_retries: Optional[int] = Field(default=None, alias="ONTOFETCH_MAX_RETRIES")
    timeout_sec: Optional[int] = Field(default=None, alias="ONTOFETCH_TIMEOUT_SEC")
    download_timeout_sec: Optional[int] = Field(
        default=None, alias="ONTOFETCH_DOWNLOAD_TIMEOUT_SEC"
    )
    per_host_rate_limit: Optional[str] = Field(
        default=None, alias="ONTOFETCH_PER_HOST_RATE_LIMIT"
    )
    backoff_factor: Optional[float] = Field(
        default=None, alias="ONTOFETCH_BACKOFF_FACTOR"
    )
    log_level: Optional[str] = Field(default=None, alias="ONTOFETCH_LOG_LEVEL")

    model_config = SettingsConfigDict(env_prefix="ONTOFETCH_", case_sensitive=False, extra="ignore")


def get_env_overrides() -> Dict[str, str]:
    """Return raw environment override values for backwards compatibility."""

    env = EnvironmentOverrides()
    return {
        key: str(value)
        for key, value in env.model_dump(by_alias=False, exclude_none=True).items()
    }


def _apply_env_overrides(defaults: DefaultsConfig) -> None:
    """Apply environment variable overrides to default configuration."""

    env = EnvironmentOverrides()
    logger = logging.getLogger("DocsToKG.OntologyDownload")

    if env.max_retries is not None:
        defaults.http.max_retries = env.max_retries
        logger.info(
            "Config overridden: max_retries=%s",
            env.max_retries,
            extra={"stage": "config"},
        )
    if env.timeout_sec is not None:
        defaults.http.timeout_sec = env.timeout_sec
        logger.info(
            "Config overridden: timeout_sec=%s",
            env.timeout_sec,
            extra={"stage": "config"},
        )
    if env.download_timeout_sec is not None:
        defaults.http.download_timeout_sec = env.download_timeout_sec
        logger.info(
            "Config overridden: download_timeout_sec=%s",
            env.download_timeout_sec,
            extra={"stage": "config"},
        )
    if env.per_host_rate_limit is not None:
        defaults.http.per_host_rate_limit = env.per_host_rate_limit
        logger.info(
            "Config overridden: per_host_rate_limit=%s",
            env.per_host_rate_limit,
            extra={"stage": "config"},
        )
    if env.backoff_factor is not None:
        defaults.http.backoff_factor = env.backoff_factor
        logger.info(
            "Config overridden: backoff_factor=%s",
            env.backoff_factor,
            extra={"stage": "config"},
        )
    if env.log_level is not None:
        defaults.logging.level = env.log_level
        logger.info(
            "Config overridden: log_level=%s",
            env.log_level,
            extra={"stage": "config"},
        )


def _make_fetch_spec(
    ontology_id: str,
    resolver: str,
    extras: Mapping[str, object],
    target_formats: Sequence[str],
) -> "FetchSpec":
    """Instantiate a FetchSpec from raw YAML fields."""

    from .core import FetchSpec as _FetchSpec  # Local import avoids circular dependency

    return _FetchSpec(
        id=ontology_id,
        resolver=resolver,
        extras=dict(extras),
        target_formats=list(target_formats),
    )


def merge_defaults(
    ontology_spec: Mapping[str, object],
    defaults: Optional[DefaultsConfig] = None,
):
    """Merge an ontology specification with resolved default settings."""

    if defaults is None:
        return build_resolved_config(ontology_spec)

    ontology_id = str(ontology_spec.get("id", "")).strip()
    if not ontology_id:
        raise ConfigError("Ontology specification missing required field 'id'")

    resolver = str(ontology_spec.get("resolver", defaults.prefer_source[0])).strip()
    extras_value = ontology_spec.get("extras") or {}
    if extras_value is None:
        extras_value = {}
    if not isinstance(extras_value, Mapping):
        raise ConfigError(f"Ontology '{ontology_id}' extras must be a mapping if provided")

    target_formats_raw = ontology_spec.get("target_formats")
    target_formats = _coerce_sequence(target_formats_raw) or list(defaults.normalize_to)

    return _make_fetch_spec(ontology_id, resolver, extras_value, target_formats)


def build_resolved_config(raw_config: Mapping[str, object]) -> ResolvedConfig:
    """Construct fully-resolved configuration from raw YAML contents."""

    try:
        defaults_section = raw_config.get("defaults", {})
        if defaults_section and not isinstance(defaults_section, Mapping):
            raise ConfigError("'defaults' section must be a mapping")
        defaults = DefaultsConfig.model_validate(defaults_section)
    except ValidationError as exc:
        messages = []
        for error in exc.errors():
            location = " -> ".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}")
        raise ConfigError(
            "Configuration validation failed:\n  " + "\n  ".join(messages)
        ) from exc

    _apply_env_overrides(defaults)

    ontologies = raw_config.get("ontologies")
    if ontologies is None:
        raise ConfigError("'ontologies' section is required")
    if not isinstance(ontologies, list):
        raise ConfigError("'ontologies' must be a list")

    fetch_specs: List["FetchSpec"] = []
    for index, entry in enumerate(ontologies, start=1):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"Ontology entry #{index} must be a mapping")
        fetch_specs.append(merge_defaults(entry, defaults))

    return ResolvedConfig(defaults=defaults, specs=fetch_specs)


def _validate_schema(raw: Mapping[str, object], config: Optional[ResolvedConfig] = None) -> None:
    """Perform additional structural validation beyond Pydantic models."""

    errors: List[str] = []
    defaults = raw.get("defaults")
    if defaults is not None and not isinstance(defaults, Mapping):
        errors.append("'defaults' section must be a mapping when provided")

    if isinstance(defaults, Mapping):
        allowed_keys = {
            "accept_licenses",
            "normalize_to",
            "prefer_source",
            "http",
            "validation",
            "logging",
            "continue_on_error",
        }
        for key in defaults:
            if key not in allowed_keys:
                errors.append(f"Unknown key in defaults: {key}")
        http_section = defaults.get("http")
        if http_section is not None and not isinstance(http_section, Mapping):
            errors.append("'defaults.http' must be a mapping")
        validation_section = defaults.get("validation")
        if validation_section is not None and not isinstance(validation_section, Mapping):
            errors.append("'defaults.validation' must be a mapping")
        logging_section = defaults.get("logging")
        if logging_section is not None and not isinstance(logging_section, Mapping):
            errors.append("'defaults.logging' must be a mapping")

    if errors:
        raise ConfigError("Configuration validation failed:\n- " + "\n- ".join(errors))

    if config is not None:
        if config.defaults.http.max_retries < 0:
            raise ConfigError("defaults.http.max_retries must be >= 0")
        if config.defaults.http.timeout_sec <= 0:
            raise ConfigError("defaults.http.timeout_sec must be > 0")


def load_raw_yaml(config_path: Path) -> MutableMapping[str, object]:
    """Load and parse a YAML configuration file into a mutable mapping."""

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
    """Load configuration from disk without performing additional schema validation."""

    raw = load_raw_yaml(config_path)
    return build_resolved_config(raw)


def validate_config(config_path: Path) -> ResolvedConfig:
    """Validate a configuration file and return the resolved settings."""

    raw = load_raw_yaml(config_path)
    config = build_resolved_config(raw)
    _validate_schema(raw, config)
    return config


__all__ = [
    "ConfigError",
    "DefaultsConfig",
    "DownloadConfiguration",
    "LoggingConfiguration",
    "ResolvedConfig",
    "ValidationConfig",
    "ensure_python_version",
    "get_env_overrides",
    "build_resolved_config",
    "load_config",
    "load_raw_yaml",
    "merge_defaults",
    "validate_config",
]

DefaultsConfiguration = DefaultsConfig
LoggingConfig = LoggingConfiguration
ValidationConfiguration = ValidationConfig
