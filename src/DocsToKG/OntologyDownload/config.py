"""
Ontology Downloader Configuration

This module centralizes configuration schema definitions, environment
overrides, and YAML parsing for DocsToKG's ontology downloader. It builds on
Pydantic v2 models to provide strong validation, type-safe defaults, automatic
JSON Schema generation, and runtime mutability where operational overrides are
required.

Key Features:
- Declarative Pydantic models for HTTP, validation, and logging settings
- YAML loading with structural validation and friendly error messages
- Environment variable overrides merged via :class:`pydantic_settings.BaseSettings`
- JSON Schema export for documentation and validation tooling
- Utilities to merge defaults with ad-hoc fetch specifications

Dependencies:
- PyYAML for configuration parsing
- pydantic and pydantic-settings for model validation

Usage:
    from DocsToKG.OntologyDownload.config import load_config

    resolved = load_config(Path(\"sources.yaml\"))
    for spec in resolved.specs:
        print(spec.id)
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

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
    """Raised when ontology configuration files are invalid or inconsistent.

    Attributes:
        message: Human-readable explanation of the configuration flaw.

    Examples:
        >>> try:
        ...     raise ConfigError("missing id")
        ... except ConfigError as exc:
        ...     assert "missing id" in str(exc)
    """


def ensure_python_version() -> None:
    """Ensure the interpreter meets the minimum supported Python version.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: If the current interpreter version is below the minimum.
    """

    if sys.version_info < PYTHON_MIN_VERSION:
        print("Error: Python 3.9+ required", file=sys.stderr)
        raise SystemExit(1)


def _coerce_sequence(value: Optional[Iterable[str]]) -> List[str]:
    """Normalize configuration entries into a list of strings.

    Args:
        value: Input value that may be ``None``, a string, or an iterable of strings.

    Returns:
        List of strings suitable for downstream configuration processing.
    """

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


class LoggingConfiguration(BaseModel):
    """Structured logging options for ontology download operations.

    Attributes:
        level: Logging level for downloader telemetry (DEBUG, INFO, etc.).
        max_log_size_mb: Maximum size of log files before rotation occurs.
        retention_days: Number of days log files are retained on disk.

    Examples:
        >>> config = LoggingConfiguration(level="debug")
        >>> config.level
        'DEBUG'
    """

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
        """Validate logging level values.

        Args:
            value: Logging level provided in configuration.

        Returns:
            Uppercase logging level string accepted by :mod:`logging`.

        Raises:
            ValueError: If the supplied level is not among the supported options.
        """

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
    """Validation limits governing parser execution.

    Attributes:
        parser_timeout_sec: Maximum runtime allowed for ontology parsing.
        max_memory_mb: Memory ceiling allocated to validation routines.
        skip_reasoning_if_size_mb: Threshold above which reasoning is skipped.

    Examples:
        >>> ValidationConfig(parser_timeout_sec=120).parser_timeout_sec
        120
    """

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
    """HTTP download, throttling, and polite header settings for resolvers.

    Attributes:
        max_retries: Maximum retry attempts for failed download requests.
        timeout_sec: Base timeout applied to metadata requests.
        download_timeout_sec: Timeout applied to streaming download operations.
        backoff_factor: Exponential backoff multiplier between retry attempts.
        per_host_rate_limit: Token bucket rate limit string for host throttling.
        max_download_size_gb: Maximum permitted download size in gigabytes.
        concurrent_downloads: Allowed number of simultaneous downloads.
        validate_media_type: Whether to enforce Content-Type validation.
        rate_limits: Optional per-service rate limit overrides.
        allowed_hosts: Optional allowlist restricting download hostnames.
        polite_headers: Default polite HTTP headers applied to resolver API calls.

    Examples:
        >>> cfg = DownloadConfiguration(per_host_rate_limit="10/second")
        >>> round(cfg.rate_limit_per_second(), 2)
        10.0
    """

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
    polite_headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "DocsToKG-OntologyDownloader/1.0 (+https://github.com/allenai/DocsToKG)",
        }
    )

    @field_validator("rate_limits")
    @classmethod
    def validate_rate_limits(cls, value: Dict[str, str]) -> Dict[str, str]:
        """Ensure rate limit strings follow the expected pattern.

        Args:
            value: Mapping of service name to rate limit expression.

        Returns:
            Validated mapping preserving the original values.

        Raises:
            ValueError: If any rate limit fails to match the accepted pattern.
        """

        for service, limit in value.items():
            if not _RATE_LIMIT_PATTERN.match(limit):
                raise ValueError(
                    f"Invalid rate limit '{limit}' for service '{service}'. "
                    "Expected format: <number>/<unit> (e.g., '5/second', '60/min')"
                )
        return value

    def rate_limit_per_second(self) -> float:
        """Convert ``per_host_rate_limit`` to a requests-per-second value.

        Args:
            None

        Returns:
            Floating-point requests-per-second value derived from configuration.

        Raises:
            ValueError: If the configured rate limit string is invalid.
        """

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
        """Parse a per-service rate limit to requests per second.

        Args:
            service: Logical service name to look up.

        Returns:
            Requests-per-second value when configured, otherwise ``None``.

        Raises:
            None.
        """

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

    def normalized_allowed_hosts(self) -> Optional[Tuple[Set[str], Set[str]]]:
        """Return allowlist entries normalized to lowercase punycode labels.

        Args:
            None

        Returns:
            Tuple of (exact hostnames, wildcard suffixes) when entries exist,
            otherwise ``None`` if no valid allowlist entries are configured.

        Raises:
            ValueError: If any configured hostname cannot be converted to punycode.
        """

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
        """Return polite HTTP headers suitable for resolver API calls.

        The headers include a deterministic ``User-Agent`` string, propagate a
        ``From`` contact address when configured, and synthesize an ``X-Request-ID``
        correlated with the current fetch so API providers can trace requests.

        Args:
            correlation_id: Identifier attached to the current batch for log correlation.
            request_id: Optional override for ``X-Request-ID`` header; auto-generated when ``None``.
            timestamp: Optional timestamp used when constructing the request identifier.

        Returns:
            Mapping of header names to polite values complying with provider guidelines.
        """

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
    """Collection of default settings for ontology fetch specifications.

    Attributes:
        accept_licenses: Licenses accepted by default during downloads.
        normalize_to: Preferred formats for normalized ontology output.
        prefer_source: Ordered list of resolver priorities.
        http: Download configuration defaults.
        validation: Validation configuration defaults.
        logging: Logging configuration defaults.
        continue_on_error: Whether processing continues after failures.
        resolver_fallback_enabled: Whether automatic resolver fallback is enabled.

    Examples:
        >>> defaults = DefaultsConfig()
        >>> "obo" in defaults.prefer_source
        True
    """

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
        """Ensure resolver names are recognized.

        Args:
            value: Ordered list of resolver names provided in configuration.

        Returns:
            Validated list containing only supported resolver identifiers.

        Raises:
            ValueError: If any resolver name is not part of the supported set.
        """

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
    """Container for merged configuration defaults and fetch specifications.

    Attributes:
        defaults: Default configuration applied to all ontology fetch specs.
        specs: List of individual fetch specifications after merging defaults.

    Examples:
        >>> config = ResolvedConfig.from_defaults()
        >>> isinstance(config.defaults, DefaultsConfig)
        True
    """

    defaults: DefaultsConfig
    specs: List["FetchSpec"] = Field(default_factory=list)

    @classmethod
    def from_defaults(cls) -> "ResolvedConfig":
        """Create an empty resolved configuration with library defaults.

        Args:
            None

        Returns:
            ResolvedConfig populated with default settings and no fetch specs.
        """

        return cls(defaults=DefaultsConfig(), specs=[])

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


ResolvedConfig.model_rebuild()


class EnvironmentOverrides(BaseSettings):
    """Environment variable overrides for ontology downloader defaults.

    Attributes:
        max_retries: Override for retry count via ``ONTOFETCH_MAX_RETRIES``.
        timeout_sec: Override for metadata timeout via ``ONTOFETCH_TIMEOUT_SEC``.
        download_timeout_sec: Override for streaming timeout.
        per_host_rate_limit: Override for per-host rate limit string.
        backoff_factor: Override for exponential backoff multiplier.
        log_level: Override for logging level.

    Examples:
        >>> overrides = EnvironmentOverrides()
        >>> overrides.model_config['env_prefix']
        'ONTOFETCH_'
    """

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
    """Return raw environment override values for backwards compatibility.

    Args:
        None

    Returns:
        Mapping of configuration field name to override value sourced from the environment.
    """

    env = EnvironmentOverrides()
    return {
        key: str(value) for key, value in env.model_dump(by_alias=False, exclude_none=True).items()
    }


def _apply_env_overrides(defaults: DefaultsConfig) -> None:
    """Apply environment variable overrides to default configuration.

    Args:
        defaults: Defaults configuration object that will be mutated in place.

    Returns:
        None
    """

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
    """Instantiate a FetchSpec from raw YAML fields.

    Args:
        ontology_id: Identifier of the ontology being configured.
        resolver: Name of the resolver responsible for fetching content.
        extras: Additional resolver-specific configuration parameters.
        target_formats: Desired output formats for the ontology artefact.

    Returns:
        Fully initialised ``FetchSpec`` object ready for execution.
    """

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
    """Merge an ontology specification with resolved default settings.

    Args:
        ontology_spec: Raw ontology specification mapping loaded from YAML.
        defaults: Optional resolved defaults to merge with the specification.

    Returns:
        FetchSpec instance describing the fully-merged ontology configuration.

    Raises:
        ConfigError: If required fields are missing or invalid in the specification.
    """

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
    """Construct fully-resolved configuration from raw YAML contents.

    Args:
        raw_config: Parsed YAML mapping defining defaults and ontologies.

    Returns:
        ResolvedConfig combining defaults and individual fetch specifications.

    Raises:
        ConfigError: If validation fails or required sections are missing.
    """

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
        raise ConfigError("Configuration validation failed:\n  " + "\n  ".join(messages)) from exc

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
    """Perform additional structural validation beyond Pydantic models.

    Args:
        raw: Raw configuration mapping used for structural checks.
        config: Optional resolved configuration for cross-field validation.

    Returns:
        None

    Raises:
        ConfigError: When structural constraints are violated.
    """

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
    """Load and parse a YAML configuration file into a mutable mapping.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Mutable mapping representation of the YAML content.

    Raises:
        SystemExit: If the file is not found on disk.
        ConfigError: If the YAML cannot be parsed or is structurally invalid.
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
    """Load configuration from disk without performing additional schema validation.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ResolvedConfig produced from the raw file contents.

    Raises:
        ConfigError: If the configuration cannot be parsed or validated.
    """

    raw = load_raw_yaml(config_path)
    return build_resolved_config(raw)


def validate_config(config_path: Path) -> ResolvedConfig:
    """Validate a configuration file and return the resolved settings.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ResolvedConfig object after validation.

    Raises:
        ConfigError: If validation fails.
    """

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
