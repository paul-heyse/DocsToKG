# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.config",
#   "purpose": "Configuration models and helpers for the ontology downloader",
#   "sections": []
# }
# === /NAVMAP ===

"""Configuration models, parsing, and validation helpers."""

from __future__ import annotations

import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:  # pragma: no cover - dependency check
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - explicit guidance for users
    raise ImportError(
        "PyYAML is required for configuration parsing. Install it with: pip install pyyaml"
    ) from exc

try:  # pragma: no cover - optional dependency
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
except ModuleNotFoundError as exc:  # pragma: no cover - provide actionable guidance
    raise ImportError(
        "jsonschema is required for ontology validation. Install it with: pip install jsonschema"
    ) from exc

from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as PydanticValidationError
from pydantic_core import ValidationError as CoreValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from .pipeline import ConfigurationError as _ConfigurationError

PYTHON_MIN_VERSION = (3, 9)


class ConfigError(RuntimeError):
    """Raised when ontology configuration files are invalid or inconsistent."""


def ensure_python_version() -> None:
    """Ensure the interpreter meets the minimum supported Python version."""

    if sys.version_info < PYTHON_MIN_VERSION:
        print("Error: Python 3.9+ required", file=sys.stderr)
        raise SystemExit(1)


def _coerce_sequence(value: Optional[Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


__all__ = [
    "ConfigError",
    "DefaultsConfig",
    "DownloadConfiguration",
    "LoggingConfiguration",
    "ResolvedConfig",
    "ValidationConfig",
    "build_resolved_config",
    "ensure_python_version",
    "get_env_overrides",
    "load_config",
    "load_raw_yaml",
    "parse_rate_limit_to_rps",
    "validate_config",
    "ConfigurationError",
]


def __getattr__(name: str):
    """Lazily expose pipeline exceptions without introducing import cycles."""

    if name == "ConfigurationError":
        from .pipeline import ConfigurationError as _ConfigurationError

        return _ConfigurationError
    raise AttributeError(name)


_RATE_LIMIT_PATTERN = re.compile(r"^([\d.]+)/(second|sec|s|minute|min|m|hour|h)$")


def parse_rate_limit_to_rps(limit_str: str) -> Optional[float]:
    """Convert a rate limit expression into requests-per-second."""

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


class LoggingConfiguration(BaseModel):
    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    max_log_size_mb: int = Field(default=100, gt=0, description="Maximum size of rotated log files")
    retention_days: int = Field(default=30, ge=1, description="Retention period for log files")

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = value.upper()
        if upper not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return upper

    model_config = {"validate_assignment": True}


class ValidationConfig(BaseModel):
    parser_timeout_sec: int = Field(default=60, gt=0, le=3600)
    max_memory_mb: int = Field(default=2048, gt=0)
    skip_reasoning_if_size_mb: int = Field(default=500, gt=0)
    streaming_normalization_threshold_mb: int = Field(default=200, ge=1)
    max_concurrent_validators: int = Field(default=2, ge=1, le=8)

    model_config = {"validate_assignment": True}


class DownloadConfiguration(BaseModel):
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
    concurrent_plans: int = Field(default=8, ge=1, le=32)
    validate_media_type: bool = Field(default=True)
    rate_limits: Dict[str, str] = Field(
        default_factory=lambda: {
            "ols": "4/second",
            "bioportal": "2/second",
            "lov": "1/second",
        }
    )
    allowed_hosts: Optional[List[str]] = Field(default=None)
    polite_headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "DocsToKG-OntologyDownloader/1.0 (+https://github.com/allenai/DocsToKG)",
        }
    )

    @field_validator("rate_limits")
    @classmethod
    def validate_rate_limits(cls, value: Dict[str, str]) -> Dict[str, str]:
        for service, limit in value.items():
            if not _RATE_LIMIT_PATTERN.match(limit):
                raise ValueError(
                    f"Invalid rate limit '{limit}' for service '{service}'. "
                    "Expected format: <number>/<unit> (e.g., '5/second', '60/min')"
                )
        return value

    def rate_limit_per_second(self) -> float:
        parsed = parse_rate_limit_to_rps(self.per_host_rate_limit)
        if parsed is None:
            raise ValueError(f"Invalid rate limit format: {self.per_host_rate_limit}")
        return parsed

    def parse_service_rate_limit(self, service: str) -> Optional[float]:
        limit_str = self.rate_limits.get(service)
        if limit_str is None:
            return None
        return parse_rate_limit_to_rps(limit_str)

    def normalized_allowed_hosts(self) -> Optional[Tuple[Set[str], Set[str]]]:
        if not self.allowed_hosts:
            return None

        exact: Set[str] = set()
        suffixes: Set[str] = set()

        for entry in self.allowed_hosts:
            candidate = entry.strip()
            if not candidate:
                continue

            wildcard = False
            if candidate.startswith("*."):
                wildcard = True
                candidate = candidate[2:]
            elif candidate.startswith("."):
                wildcard = True
                candidate = candidate[1:]

            try:
                normalized = candidate.encode("idna").decode("ascii").lower()
            except UnicodeError as exc:
                raise ValueError(f"Invalid hostname in allowlist: {entry}") from exc

            if wildcard:
                suffixes.add(normalized)
            else:
                exact.add(normalized)

        if not exact and not suffixes:
            return None

        return exact, suffixes

    def polite_http_headers(
        self,
        *,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            str(key): str(value)
            for key, value in (self.polite_headers or {}).items()
            if str(value).strip()
        }

        if not any(key.lower() == "user-agent" for key in headers):
            headers["User-Agent"] = (
                "DocsToKG-OntologyDownloader/1.0 (+https://github.com/allenai/DocsToKG)"
            )

        moment = timestamp or datetime.now(timezone.utc)
        if request_id is None:
            seed = correlation_id or uuid.uuid4().hex[:12]
            request_id = f"{seed}-{moment.strftime('%Y%m%dT%H%M%SZ')}"
        headers.setdefault("X-Request-ID", request_id)

        if "From" not in headers and "mailto" in headers:
            headers.setdefault("From", headers["mailto"])

        return headers

    model_config = {"validate_assignment": True}


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
    accept_licenses: List[str] = Field(
        default_factory=lambda: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"]
    )
    normalize_to: List[str] = Field(default_factory=lambda: ["ttl"])
    prefer_source: List[str] = Field(default_factory=lambda: ["obo", "ols", "bioportal", "direct"])
    http: DownloadConfiguration = Field(default_factory=DownloadConfiguration)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfiguration = Field(default_factory=LoggingConfiguration)
    continue_on_error: bool = Field(default=True)
    resolver_fallback_enabled: bool = Field(default=True)

    @field_validator("prefer_source")
    @classmethod
    def validate_prefer_source(cls, value: List[str]) -> List[str]:
        unknown = [resolver for resolver in value if resolver not in _VALID_RESOLVERS]
        if unknown:
            raise ValueError(
                "Unknown resolver(s) in prefer_source: " + ", ".join(sorted(set(unknown)))
            )
        return value

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }


class ResolvedConfig(BaseModel):
    defaults: DefaultsConfig
    specs: List["FetchSpec"] = Field(default_factory=list)

    @classmethod
    def from_defaults(cls) -> "ResolvedConfig":
        return cls(defaults=DefaultsConfig(), specs=[])

    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


class EnvironmentOverrides(BaseSettings):
    max_retries: Optional[int] = Field(default=None, alias="ONTOFETCH_MAX_RETRIES")
    timeout_sec: Optional[int] = Field(default=None, alias="ONTOFETCH_TIMEOUT_SEC")
    download_timeout_sec: Optional[int] = Field(
        default=None, alias="ONTOFETCH_DOWNLOAD_TIMEOUT_SEC"
    )
    per_host_rate_limit: Optional[str] = Field(default=None, alias="ONTOFETCH_PER_HOST_RATE_LIMIT")
    backoff_factor: Optional[float] = Field(default=None, alias="ONTOFETCH_BACKOFF_FACTOR")
    log_level: Optional[str] = Field(default=None, alias="ONTOFETCH_LOG_LEVEL")

    model_config = SettingsConfigDict(env_prefix="ONTOFETCH_", case_sensitive=False, extra="ignore")


def get_env_overrides() -> Dict[str, str]:
    env = EnvironmentOverrides()
    return {
        key: str(value) for key, value in env.model_dump(by_alias=False, exclude_none=True).items()
    }


def _apply_env_overrides(defaults: DefaultsConfig) -> None:
    env = EnvironmentOverrides()
    logger = logging.getLogger("DocsToKG.OntologyDownload")

    if env.max_retries is not None:
        defaults.http.max_retries = env.max_retries
        logger.info("Config overridden: max_retries=%s", env.max_retries, extra={"stage": "config"})
    if env.timeout_sec is not None:
        defaults.http.timeout_sec = env.timeout_sec
        logger.info("Config overridden: timeout_sec=%s", env.timeout_sec, extra={"stage": "config"})
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
        logger.info("Config overridden: log_level=%s", env.log_level, extra={"stage": "config"})


def build_resolved_config(raw_config: Mapping[str, object]) -> ResolvedConfig:
    try:
        defaults_section = raw_config.get("defaults", {})
        if defaults_section and not isinstance(defaults_section, Mapping):
            raise ConfigError("'defaults' section must be a mapping")
        defaults = DefaultsConfig.model_validate(defaults_section)
    except (PydanticValidationError, CoreValidationError) as exc:
        messages = []
        for error in exc.errors():
            location = " -> ".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}")
        raise ConfigError("Configuration validation failed:\n  " + "\n  ".join(messages)) from exc

    _apply_env_overrides(defaults)

    ontologies = raw_config.get("ontologies")
    if ontologies is None:
        raise ConfigError("'ontologies' section is required")
    if not isinstance(ontologies, list):
        raise ConfigError("'ontologies' must be a list")

    from .pipeline import merge_defaults  # imported lazily to avoid circular dependency

    fetch_specs: List["FetchSpec"] = []
    for index, entry in enumerate(ontologies, start=1):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"Ontology entry #{index} must be a mapping")
        fetch_specs.append(merge_defaults(entry, defaults))

    return ResolvedConfig(defaults=defaults, specs=fetch_specs)


def _validate_schema(raw: Mapping[str, object], config: Optional[ResolvedConfig] = None) -> None:
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
            "resolver_fallback_enabled",
        }
        for key in defaults:
            if key not in allowed_keys:
                errors.append(f"Unknown key in defaults: {key}")
        for section_name in ("http", "validation", "logging"):
            section = defaults.get(section_name)
            if section is not None and not isinstance(section, Mapping):
                errors.append(f"'defaults.{section_name}' must be a mapping")

    if errors:
        raise ConfigError("Configuration validation failed:\n- " + "\n- ".join(errors))


def load_raw_yaml(config_path: Path) -> Mapping[str, object]:
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        raise SystemExit(2)

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via tests
        raise ConfigError(
            f"Configuration file '{config_path}' contains invalid YAML"
        ) from exc

    if not isinstance(data, Mapping):
        raise ConfigError("Configuration file must contain a mapping at the root")
    return data


def load_config(config_path: Path) -> ResolvedConfig:
    raw = load_raw_yaml(config_path)
    config = build_resolved_config(raw)
    _validate_schema(raw, config)
    return config


def validate_config(config_path: Path) -> ResolvedConfig:
    raw = load_raw_yaml(config_path)
    config = build_resolved_config(raw)
    _validate_schema(raw, config)
    return config
