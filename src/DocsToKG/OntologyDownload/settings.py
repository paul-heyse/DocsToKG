# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.settings",
#   "purpose": "Define configuration models, environment overrides, optional dependency shims, and storage backends",
#   "sections": [
#     {"id": "config-models", "name": "Configuration Models", "anchor": "CFG", "kind": "api"},
#     {"id": "env-overrides", "name": "Environment Overrides", "anchor": "ENV", "kind": "helpers"},
#     {"id": "config-loading", "name": "Configuration Loading & Validation", "anchor": "LOD", "kind": "api"},
#     {"id": "opt-deps", "name": "Optional Dependency Shims", "anchor": "OPT", "kind": "helpers"},
#     {"id": "storage", "name": "Storage Backends", "anchor": "STO", "kind": "api"}
#   ]
# }
# === /NAVMAP ===

"""Configuration, optional dependency, and storage utilities for ontology downloads.

This module defines the typed settings models used by the CLI, environment
overrides that surface as ``ONTOFETCH_*`` variables, rate-limit parsing (including
the `rate_limiter` backend switch), local cache layout, and optional dependency
wiring.  It also exposes helpers for validating interpreter support, selecting
local or fsspec storage (including CAS mirrors), and coercing user-provided
configuration into strongly typed structures that downstream planners and
resolvers can consume safely.
"""

from __future__ import annotations

import importlib
import ipaddress
import logging
import os
import re
import shutil
import stat
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
)

import httpx

try:  # pragma: no cover - dependency check
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - explicit guidance for users
    # Guardrail: align messaging with OntologyDownload/AGENTS.md environment policy.
    raise ImportError(
        "PyYAML is required for configuration parsing. "
        "Ensure the project-managed .venv is set up (rerun the approved bootstrap script) "
        "instead of installing packages directly."
    ) from exc

from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pydantic import ValidationError as PydanticValidationError
from pydantic_core import ValidationError as CoreValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import (
    DownloadFailure,
    OntologyDownloadError,
    PolicyError,
    ResolverError,
    UnsupportedPythonError,
    UserConfigError,
    ValidationError,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from .planning import FetchSpec


PYTHON_MIN_VERSION = (3, 9)


def ensure_python_version(
    min_major: int = PYTHON_MIN_VERSION[0],
    min_minor: int = PYTHON_MIN_VERSION[1],
) -> None:
    """Ensure the interpreter meets the minimum supported Python version."""

    if (sys.version_info.major, sys.version_info.minor) < (min_major, min_minor):
        required = f"{min_major}.{min_minor}"
        found = sys.version.split()[0]
        raise UnsupportedPythonError(f"Python >= {required} required; found {found}")


def _coerce_sequence(value: Optional[Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


ConfigError = UserConfigError  # Backwards compatibility alias


_RATE_LIMIT_PATTERN = re.compile(r"^([\d.]+)/(second|sec|s|minute|min|m|hour|h)$")


def parse_rate_limit_to_rps(limit_str: Optional[str]) -> Optional[float]:
    """Convert a rate limit expression into requests-per-second."""

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


class LoggingConfiguration(BaseModel):
    """Logging-related configuration for ontology downloads."""

    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    max_log_size_mb: int = Field(default=100, gt=0, description="Maximum size of rotated log files")
    retention_days: int = Field(default=30, ge=1, description="Retention period for log files")

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        """Normalise logging levels and ensure they match the supported set."""

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = value.upper()
        if upper not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return upper

    model_config = {"validate_assignment": True}


class ValidationConfig(BaseModel):
    """Settings controlling ontology validation throughput and limits."""

    parser_timeout_sec: int = Field(default=60, gt=0, le=3600)
    max_memory_mb: int = Field(default=2048, gt=0)
    skip_reasoning_if_size_mb: int = Field(default=500, gt=0)
    streaming_normalization_threshold_mb: int = Field(default=200, ge=1)
    max_concurrent_validators: int = Field(default=2, ge=1, le=8)
    use_process_pool: bool = Field(
        default=False,
        description="When true, run heavy validators (rdflib, owlready2) in a process pool for isolation.",
    )
    process_pool_validators: List[str] = Field(
        default_factory=lambda: ["rdflib", "owlready2"],
        description="Validator names that should run in the process pool when enabled.",
    )

    model_config = {"validate_assignment": True, "extra": "ignore"}


class DownloadConfiguration(BaseModel):
    """HTTP download, retry, and politeness settings for resolvers.

    The configuration now drives the HTTPX-based streaming pipeline (`io.network.download_stream`):
    * `perform_head_precheck` toggles the optional HEAD probe before issuing the GET.
    * `progress_log_percent_step` / `progress_log_bytes_threshold` control progress telemetry cadence.
    * HTTPX + Hishel is the only supported engine (legacy `SessionPool`/requests fallbacks and the
      `network.engine` feature flag were removed; custom clients should be installed via
      :func:`DocsToKG.OntologyDownload.net.configure_http_client`).
    """

    _session_factory: Optional[Callable[[], Any]] = PrivateAttr(default=None)
    _bucket_provider: Optional[
        Callable[[Optional[str], "DownloadConfiguration", Optional[str]], Any]
    ] = PrivateAttr(default=None)

    max_retries: int = Field(default=5, ge=0, le=20)
    timeout_sec: int = Field(default=30, gt=0, le=300)
    download_timeout_sec: int = Field(default=300, gt=0, le=3600)
    connect_timeout_sec: float = Field(default=5.0, gt=0.0, le=60.0)
    pool_timeout_sec: float = Field(default=5.0, gt=0.0, le=60.0)
    backoff_factor: float = Field(default=0.5, ge=0.1, le=10.0)
    max_httpx_connections: int = Field(default=128, ge=1, le=1024)
    max_keepalive_connections: int = Field(default=32, ge=0, le=1024)
    keepalive_expiry_sec: float = Field(default=30.0, gt=0.0, le=600.0)
    http2_enabled: bool = Field(default=True)
    per_host_rate_limit: str = Field(
        default="4/second",
        pattern=_RATE_LIMIT_PATTERN.pattern,
        description="Token bucket style rate limit expressed as <number>/<unit>",
    )
    max_uncompressed_size_gb: float = Field(
        default=10.0,
        gt=0,
        le=200.0,
        description="Maximum allowed uncompressed archive size in gigabytes",
    )
    max_checksum_response_bytes: int = Field(
        default=262_144,
        ge=1_024,
        le=8_388_608,
        description="Maximum number of bytes permitted when downloading checksum manifests",
    )
    concurrent_downloads: int = Field(default=1, ge=1, le=10)
    concurrent_plans: int = Field(default=8, ge=1, le=32)
    validate_media_type: bool = Field(default=True)
    perform_head_precheck: bool = Field(
        default=True,
        description=(
            "When true, downloads issue a HEAD request before GET to collect metadata and "
            "enforce early policy checks."
        ),
    )
    progress_log_percent_step: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Fractional progress interval used for download telemetry. "
            "Set to 0 to disable percentage-based progress logs."
        ),
    )
    progress_log_bytes_threshold: int = Field(
        default=5_242_880,
        ge=0,
        description=(
            "Byte interval for progress logs when the total content length is unknown. "
            "Set to 0 to disable byte-based progress logging."
        ),
    )
    strict_dns: bool = Field(
        default=False,
        description="When true, abort on DNS resolution failures instead of logging a warning.",
    )
    rate_limits: Dict[str, str] = Field(
        default_factory=lambda: {
            "ols": "4/second",
            "bioportal": "2/second",
            "lov": "1/second",
        }
    )
    allowed_hosts: Optional[List[str]] = Field(default=None)
    allow_private_networks_for_host_allowlist: bool = Field(
        default=False,
        description=(
            "When true, allowlisted hosts may resolve to private or loopback addresses."
            " Defaults to False to prevent accidental SSRF via misconfigured DNS."
        ),
    )
    allow_plain_http_for_host_allowlist: bool = Field(
        default=False,
        description=(
            "When true, allowlisted hosts may be fetched over HTTP without automatic upgrades to HTTPS."
            " Defaults to False so host allowlists continue to enforce TLS unless explicitly opted in."
        ),
    )
    allowed_ports: Optional[List[int]] = Field(default=None)
    polite_headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "DocsToKG-OntologyDownloader/1.0 (+https://github.com/allenai/DocsToKG)",
        }
    )
    rate_limiter: str = Field(
        default="pyrate",
        description="Selects the rate limiter backend. Only the pyrate-limiter manager is supported.",
    )
    shared_rate_limit_dir: Optional[Path] = Field(
        default=None,
        description="Directory used to persist shared token bucket state across processes",
    )

    @field_validator("rate_limits")
    @classmethod
    def validate_rate_limits(cls, value: Dict[str, str]) -> Dict[str, str]:
        """Ensure per-resolver rate limits follow the supported syntax."""

        for service, limit in value.items():
            if not _RATE_LIMIT_PATTERN.match(limit):
                raise ValueError(
                    f"Invalid rate limit '{limit}' for service '{service}'. "
                    "Expected format: <number>/<unit> (e.g., '5/second', '60/min')"
                )
        return value

    @field_validator("allowed_ports")
    @classmethod
    def validate_allowed_ports(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        """Normalise and validate the optional port allowlist from configuration.

        Args:
            value: Collection of port numbers supplied in the settings payload.

        Returns:
            Sanitised list of ports preserving input order when provided, otherwise ``None``.

        Raises:
            ValueError: If any port is not an integer or falls outside the TCP range 1-65535.
        """
        if value is None:
            return None
        normalized: List[int] = []
        for port in value:
            if not isinstance(port, int):
                raise ValueError("allowed_ports entries must be integers")
            if port < 1 or port > 65535:
                raise ValueError("allowed_ports values must be between 1 and 65535")
            normalized.append(port)
        return normalized

    def rate_limit_per_second(self) -> float:
        """Return the configured per-host rate limit in requests per second."""

        parsed = parse_rate_limit_to_rps(self.per_host_rate_limit)
        if parsed is None:
            raise ValueError(f"Invalid rate limit format: {self.per_host_rate_limit}")
        return parsed

    def max_uncompressed_bytes(self) -> int:
        """Return the maximum allowed uncompressed archive size in bytes."""

        return int(self.max_uncompressed_size_gb * (1024**3))

    def max_checksum_bytes(self) -> int:
        """Return the configured ceiling for checksum downloads in bytes."""

        return int(self.max_checksum_response_bytes)

    def parse_service_rate_limit(self, service: str) -> Optional[float]:
        """Return service-specific rate limits expressed as requests per second."""

        limit_str = self.rate_limits.get(service)
        if limit_str is None:
            return None
        parsed = parse_rate_limit_to_rps(limit_str)
        if parsed is None:
            raise ValueError(
                f"Invalid rate limit '{limit_str}' for service '{service}'. "
                "Expected format: <number>/<unit> (e.g., '5/second', '60/min')"
            )
        return parsed

    def allowed_port_set(self) -> Set[int]:
        """Return the union of default ports and user-configured allowances."""

        ports: Set[int] = {80, 443}
        if self.allowed_ports:
            ports.update(self.allowed_ports)
        return ports

    def normalized_allowed_hosts(
        self,
    ) -> Optional[Tuple[Set[str], Set[str], Dict[str, Set[int]], Set[str]]]:
        """Split allowed host list into exact domains, wildcard suffixes, per-host ports, and IP literals."""

        if not self.allowed_hosts:
            return None

        exact: Set[str] = set()
        suffixes: Set[str] = set()
        host_ports: Dict[str, Set[int]] = {}
        ip_literals: Set[str] = set()

        for entry in self.allowed_hosts:
            candidate = entry.strip()
            if not candidate:
                continue

            port: Optional[int] = None
            working = candidate
            if working.startswith("["):
                end_bracket = working.find("]")
                if end_bracket == -1:
                    raise ValueError(f"Invalid IPv6 host '{entry}' in allowlist")
                literal = working[1:end_bracket]
                remainder = working[end_bracket + 1 :]
                if remainder:
                    if not remainder.startswith(":"):
                        raise ValueError(f"Invalid IPv6 host '{entry}' in allowlist")
                    port_str = remainder[1:]
                    if not port_str.isdigit():
                        raise ValueError(f"Invalid port in allowlist entry: {entry}")
                    port = int(port_str)
                working = literal
            elif ":" in working and working.count(":") == 1:
                host_candidate, maybe_port = working.rsplit(":", 1)
                if maybe_port.isdigit():
                    port_value = int(maybe_port)
                    if port_value < 1 or port_value > 65535:
                        raise ValueError(f"Invalid port in allowlist entry: {entry}")
                    port = port_value
                    working = host_candidate
                else:
                    raise ValueError(f"Invalid port in allowlist entry: {entry}")

            wildcard = False
            if working.startswith("*."):
                wildcard = True
                working = working[2:]
            elif working.startswith("."):
                wildcard = True
                working = working[1:]

            try:
                ipaddress.ip_address(working)
            except ValueError:
                try:
                    normalized = working.encode("idna").decode("ascii").lower()
                except UnicodeError as exc:
                    raise ValueError(f"Invalid hostname in allowlist: {entry}") from exc
            else:
                normalized = working.lower()
                ip_literals.add(normalized)

            if wildcard and port is not None:
                raise ValueError("Wildcard allowlist entries cannot specify ports")

            if wildcard:
                suffixes.add(normalized)
            else:
                exact.add(normalized)
                if port is not None:
                    host_ports.setdefault(normalized, set()).add(port)

        if not exact and not suffixes and not host_ports:
            return None

        return exact, suffixes, host_ports, ip_literals

    def polite_http_headers(
        self,
        *,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, str]:
        """Compute polite HTTP headers for outbound resolver requests."""

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

    def set_session_factory(self, factory: Optional[Callable[[], Any]]) -> None:
        """Set a custom factory used to construct HTTPX clients."""

        if factory is None:
            self._session_factory = None
            return
        if not callable(factory):
            raise TypeError("session_factory must be callable and return an httpx.Client or None")

        def _wrapped() -> Optional[httpx.Client]:
            candidate = factory()
            if candidate is None or isinstance(candidate, httpx.Client):
                return candidate
            raise TypeError("session_factory must return an httpx.Client or None")

        self._session_factory = _wrapped

    def get_session_factory(self) -> Optional[Callable[[], Any]]:
        """Return the custom HTTPX client factory, if one has been configured."""

        return self._session_factory

    def set_bucket_provider(
        self,
        provider: Optional[Callable[[Optional[str], "DownloadConfiguration", Optional[str]], Any]],
    ) -> None:
        """Set a custom provider responsible for returning token buckets."""

        self._bucket_provider = provider

    def get_bucket_provider(
        self,
    ) -> Optional[Callable[[Optional[str], "DownloadConfiguration", Optional[str]], Any]]:
        """Return the configured token bucket provider, if present."""

        return self._bucket_provider

    def model_copy(
        self,
        *,
        deep: bool = False,
        update: Optional[Mapping[str, Any]] = None,
    ) -> "DownloadConfiguration":
        """Copy the model ensuring private attributes propagate."""

        copied = super().model_copy(deep=deep, update=update)
        copied._session_factory = self._session_factory
        copied._bucket_provider = self._bucket_provider
        return copied

    model_config = {"validate_assignment": True, "extra": "ignore"}


class PlannerConfig(BaseModel):
    """Planner-specific HTTP probing behaviour."""

    probing_enabled: bool = Field(
        default=True,
        description="When false, planner metadata HTTP probes are skipped after URL validation.",
    )
    head_precheck_hosts: List[str] = Field(
        default_factory=list,
        description=(
            "Collection of hostnames (exact or suffixes prefixed with a dot) that require a "
            "HEAD request before issuing the metadata GET probe."
        ),
    )

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
    """Composite configuration applied when no per-spec overrides exist."""

    accept_licenses: List[str] = Field(
        default_factory=lambda: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"]
    )
    normalize_to: List[str] = Field(default_factory=lambda: ["ttl"])
    prefer_source: List[str] = Field(default_factory=lambda: ["obo", "ols", "bioportal", "direct"])
    http: DownloadConfiguration = Field(default_factory=DownloadConfiguration)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfiguration = Field(default_factory=LoggingConfiguration)
    continue_on_error: bool = Field(default=True)
    resolver_fallback_enabled: bool = Field(default=True)
    enable_cas_mirror: bool = Field(
        default=False,
        description="Mirror original artefacts under by-<hash>/<digest> for de-duplication",
    )

    @field_validator("prefer_source")
    @classmethod
    def validate_prefer_source(cls, value: List[str]) -> List[str]:
        """Ensure preferred resolvers belong to the supported resolver set."""

        unknown = [resolver for resolver in value if resolver not in _VALID_RESOLVERS]
        if unknown:
            raise ValueError(
                "Unknown resolver(s) in prefer_source: " + ", ".join(sorted(set(unknown)))
            )
        return value

    @field_validator("accept_licenses")
    @classmethod
    def normalize_accept_licenses(cls, value: List[str]) -> List[str]:
        """Normalize accept_licenses entries to canonical SPDX identifiers."""

        from .resolvers import normalize_license_to_spdx

        normalized: List[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                continue
            normalized_value = normalize_license_to_spdx(entry) or entry.strip()
            normalized.append(normalized_value)
        return normalized

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }


class ResolvedConfig(BaseModel):
    """Materialised configuration combining defaults and fetch specifications."""

    defaults: DefaultsConfig
    specs: List["FetchSpec"] = Field(default_factory=list)

    @classmethod
    def from_defaults(cls) -> "ResolvedConfig":
        """Construct a resolved configuration populated with default values only."""

        defaults = DefaultsConfig()
        _apply_env_overrides(defaults)
        return cls(defaults=defaults, specs=[])

    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


_DEFAULT_CONFIG_LOCK = threading.RLock()
_DEFAULT_CONFIG_CACHE: Optional[ResolvedConfig] = None

_HAS_PYDANTIC_SETTINGS = hasattr(BaseSettings, "model_dump")


def _read_env_value(name: str) -> Optional[str]:
    """Fetch and normalise an environment variable, treating empty values as absent."""

    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _read_env_int(name: str) -> Optional[int]:
    value = _read_env_value(name)
    if value is None:
        return None
    return int(value)


def _read_env_float(name: str) -> Optional[float]:
    value = _read_env_value(name)
    if value is None:
        return None
    return float(value)


def _read_env_path(name: str) -> Optional[Path]:
    value = _read_env_value(name)
    if value is None:
        return None
    return Path(value)


def get_default_config(*, copy: bool = False) -> ResolvedConfig:
    """Return a memoised :class:`ResolvedConfig` constructed from defaults."""

    global _DEFAULT_CONFIG_CACHE  # noqa: PLW0603

    with _DEFAULT_CONFIG_LOCK:
        if _DEFAULT_CONFIG_CACHE is None:
            _DEFAULT_CONFIG_CACHE = ResolvedConfig.from_defaults()
        cached = _DEFAULT_CONFIG_CACHE
    if copy:
        return cached.model_copy(deep=True)
    return cached


def invalidate_default_config_cache() -> None:
    """Invalidate the cached default configuration."""

    global _DEFAULT_CONFIG_CACHE  # noqa: PLW0603

    with _DEFAULT_CONFIG_LOCK:
        _DEFAULT_CONFIG_CACHE = None


if _HAS_PYDANTIC_SETTINGS:

    class EnvironmentOverrides(BaseSettings):
        """Pydantic settings model exposing environment-derived overrides."""

        max_retries: Optional[int] = Field(default=None, alias="ONTOFETCH_MAX_RETRIES")
        timeout_sec: Optional[int] = Field(default=None, alias="ONTOFETCH_TIMEOUT_SEC")
        download_timeout_sec: Optional[int] = Field(
            default=None, alias="ONTOFETCH_DOWNLOAD_TIMEOUT_SEC"
        )
        per_host_rate_limit: Optional[str] = Field(
            default=None, alias="ONTOFETCH_PER_HOST_RATE_LIMIT"
        )
        backoff_factor: Optional[float] = Field(default=None, alias="ONTOFETCH_BACKOFF_FACTOR")
        log_level: Optional[str] = Field(default=None, alias="ONTOFETCH_LOG_LEVEL")
        shared_rate_limit_dir: Optional[Path] = Field(
            default=None, alias="ONTOFETCH_SHARED_RATE_LIMIT_DIR"
        )
        max_uncompressed_size_gb: Optional[float] = Field(
            default=None, alias="ONTOFETCH_MAX_UNCOMPRESSED_SIZE_GB"
        )

        model_config = SettingsConfigDict(
            env_prefix="ONTOFETCH_", case_sensitive=False, extra="ignore"
        )

else:

    class EnvironmentOverrides:
        """Fallback environment reader when ``pydantic-settings`` is unavailable."""

        model_config: Dict[str, object] = {}

        def __init__(self) -> None:
            self.max_retries = _read_env_int("ONTOFETCH_MAX_RETRIES")
            self.timeout_sec = _read_env_int("ONTOFETCH_TIMEOUT_SEC")
            self.download_timeout_sec = _read_env_int("ONTOFETCH_DOWNLOAD_TIMEOUT_SEC")
            self.per_host_rate_limit = _read_env_value("ONTOFETCH_PER_HOST_RATE_LIMIT")
            self.backoff_factor = _read_env_float("ONTOFETCH_BACKOFF_FACTOR")
            self.log_level = _read_env_value("ONTOFETCH_LOG_LEVEL")
            self.shared_rate_limit_dir = _read_env_path("ONTOFETCH_SHARED_RATE_LIMIT_DIR")
            self.max_uncompressed_size_gb = _read_env_float("ONTOFETCH_MAX_UNCOMPRESSED_SIZE_GB")

        max_retries: Optional[int]
        timeout_sec: Optional[int]
        download_timeout_sec: Optional[int]
        per_host_rate_limit: Optional[str]
        backoff_factor: Optional[float]
        log_level: Optional[str]
        shared_rate_limit_dir: Optional[Path]
        max_uncompressed_size_gb: Optional[float]

        def model_dump(
            self, *, by_alias: bool = False, exclude_none: bool = False
        ) -> Dict[str, object]:
            """Return environment-derived overrides mirroring ``BaseSettings.model_dump``.

            Args:
                by_alias: Included for API parity; ignored by the fallback implementation.
                exclude_none: When ``True``, omit keys whose values evaluate to ``None``.

            Returns:
                Dict[str, object]: Mapping of configuration keys to environment-provided values.
            """
            data: Dict[str, object] = {
                "max_retries": self.max_retries,
                "timeout_sec": self.timeout_sec,
                "download_timeout_sec": self.download_timeout_sec,
                "per_host_rate_limit": self.per_host_rate_limit,
                "backoff_factor": self.backoff_factor,
                "log_level": self.log_level,
                "shared_rate_limit_dir": self.shared_rate_limit_dir,
                "max_uncompressed_size_gb": self.max_uncompressed_size_gb,
            }
            if exclude_none:
                data = {key: value for key, value in data.items() if value is not None}
            return data


def get_env_overrides() -> Dict[str, str]:
    """Return environment-derived overrides as stringified key/value pairs."""

    env = EnvironmentOverrides()
    return {
        key: str(value) for key, value in env.model_dump(by_alias=False, exclude_none=True).items()
    }


def _apply_env_overrides(defaults: DefaultsConfig) -> None:
    """Mutate ``defaults`` in-place using values from :class:`EnvironmentOverrides`."""

    env = EnvironmentOverrides()
    logger = logging.getLogger("DocsToKG.OntologyDownload")

    if env.max_retries is not None:
        defaults.http.max_retries = int(env.max_retries)
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
    if env.shared_rate_limit_dir is not None:
        defaults.http.shared_rate_limit_dir = env.shared_rate_limit_dir
        logger.info(
            "Config overridden: shared_rate_limit_dir=%s",
            env.shared_rate_limit_dir,
            extra={"stage": "config"},
        )
    if env.max_uncompressed_size_gb is not None:
        defaults.http.max_uncompressed_size_gb = env.max_uncompressed_size_gb
        logger.info(
            "Config overridden: max_uncompressed_size_gb=%s",
            env.max_uncompressed_size_gb,
            extra={"stage": "config"},
        )


def build_resolved_config(raw_config: Mapping[str, object]) -> ResolvedConfig:
    """Materialise a :class:`ResolvedConfig` from a raw mapping loaded from disk."""

    try:
        defaults_section = raw_config.get("defaults", {})
        if defaults_section and not isinstance(defaults_section, Mapping):
            raise UserConfigError("'defaults' section must be a mapping")
        defaults = DefaultsConfig.model_validate(defaults_section)
    except (PydanticValidationError, CoreValidationError) as exc:
        messages = []
        for error in exc.errors():
            location = " -> ".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}")
        raise UserConfigError(
            "Configuration validation failed:\n  " + "\n  ".join(messages)
        ) from exc

    _apply_env_overrides(defaults)

    ontologies = raw_config.get("ontologies")
    if ontologies is None:
        raise UserConfigError("'ontologies' section is required")
    if not isinstance(ontologies, list):
        raise UserConfigError("'ontologies' must be a list")

    from .planning import merge_defaults  # imported lazily to avoid circular dependency

    fetch_specs: List["FetchSpec"] = []
    for index, entry in enumerate(ontologies, start=1):
        if not isinstance(entry, Mapping):
            raise UserConfigError(f"Ontology entry #{index} must be a mapping")
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
            "planner",
            "validation",
            "logging",
            "continue_on_error",
            "resolver_fallback_enabled",
        }
        for key in defaults:
            if key not in allowed_keys:
                errors.append(f"Unknown key in defaults: {key}")
        for section_name in ("http", "planner", "validation", "logging"):
            section = defaults.get(section_name)
            if section is not None and not isinstance(section, Mapping):
                errors.append(f"'defaults.{section_name}' must be a mapping")

    if errors:
        raise UserConfigError("Configuration validation failed:\n- " + "\n- ".join(errors))


def normalize_config_path(config_path: Path) -> Path:
    """Return a user-supplied configuration path with ``~`` and symlinks resolved."""

    expanded = Path(config_path).expanduser()
    try:
        return expanded.resolve(strict=False)
    except (OSError, RuntimeError):  # pragma: no cover - only triggered on rare filesystems
        return expanded


def load_raw_yaml(config_path: Path) -> Mapping[str, object]:
    """Read a YAML configuration file and return its top-level mapping."""

    normalized_path = normalize_config_path(config_path)

    if not normalized_path.exists():
        raise ConfigError(f"Configuration file not found: {normalized_path}")

    try:
        with normalized_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via tests
        raise UserConfigError(
            f"Configuration file '{normalized_path}' contains invalid YAML"
        ) from exc

    if not isinstance(data, Mapping):
        raise UserConfigError("Configuration file must contain a mapping at the root")
    return data


def load_config(config_path: Path) -> ResolvedConfig:
    """Load, validate, and resolve configuration suitable for execution."""

    normalized_path = normalize_config_path(config_path)
    raw = load_raw_yaml(normalized_path)
    config = build_resolved_config(raw)
    _validate_schema(raw, config)
    return config


def validate_config(config_path: Path) -> ResolvedConfig:
    """Load a configuration solely for validation feedback."""

    normalized_path = normalize_config_path(config_path)
    raw = load_raw_yaml(normalized_path)
    config = build_resolved_config(raw)
    _validate_schema(raw, config)
    return config


# --- Optional dependency helpers merged from optdeps.py ---

_STUB_ATTR = "_ontofetch_stub"
_BNODE_COUNTER = 0

_pystow: Optional[Any] = None
_rdflib: Optional[Any] = None
_pronto: Optional[Any] = None
_owlready2: Optional[Any] = None


def _create_stub_module(name: str, attrs: Dict[str, Any]) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    setattr(module, _STUB_ATTR, True)
    return module


def _create_stub_bnode(value: Optional[str] = None) -> str:
    global _BNODE_COUNTER
    if value is not None:
        return value
    _BNODE_COUNTER += 1
    return f"_:b{_BNODE_COUNTER}"


def _create_stub_literal(value: Any = None) -> str:
    if value is None:
        return '""'
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _create_stub_uri(value: Optional[str] = None) -> str:
    if value is None:
        return "<>"
    if value.startswith("<") and value.endswith(">"):
        return value
    return f"<{value}>"


class _StubNamespace:
    def __init__(self, base: str) -> None:
        self._base = base

    def __getitem__(self, key: str) -> str:
        return f"{self._base}{key}"


class _StubNamespaceManager:
    def __init__(self) -> None:
        self._bindings: Dict[str, str] = {}

    def bind(self, prefix: str, namespace: str) -> None:
        """Register a namespace binding in the lightweight stub manager."""

        self._bindings[prefix] = namespace

    def namespaces(self) -> Iterable[tuple[str, str]]:
        """Yield currently registered ``(prefix, namespace)`` pairs."""

        return self._bindings.items()


class _StubGraph:
    _ontofetch_stub = True

    def __init__(self) -> None:
        self._triples: List[tuple[str, str, str]] = []
        self._last_text = "# Stub TTL output\n"
        self.namespace_manager = _StubNamespaceManager()

    def parse(self, source: str, format: Optional[str] = None, **_kwargs: object) -> "_StubGraph":
        """Parse a Turtle-like text file into an in-memory triple list."""

        text = Path(source).read_text()
        self._last_text = text
        triples: List[tuple[str, str, str]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@prefix"):
                try:
                    _, remainder = line.split(None, 1)
                except ValueError:
                    continue
                parts = [segment.strip() for segment in remainder.split(None, 2)]
                if len(parts) >= 2:
                    prefix = parts[0].rstrip(":")
                    namespace = parts[1].strip("<>").rstrip(".")
                    self.namespace_manager.bind(prefix, namespace)
                line = parts[2].strip() if len(parts) == 3 else ""
                if not line:
                    continue
            if line.endswith("."):
                line = line[:-1].strip()
            if not line:
                continue
            pieces = line.split(None, 2)
            if len(pieces) < 3:
                continue
            triples.append(tuple(pieces))
        self._triples = triples
        return self

    def serialize(
        self, destination: Optional[Any] = None, format: Optional[str] = None, **_kwargs: object
    ):
        """Serialise parsed triples to the supplied destination."""

        if destination is None:
            return self._last_text
        if isinstance(destination, (str, Path)):
            Path(destination).write_text(self._last_text)
            return destination
        destination.write(b"# Stub TTL output\n")
        return destination

    def add(self, triple: Iterable[Any]) -> None:
        """Append a triple to the stub graph, mirroring rdflib behaviour."""

        items = list(triple)
        if len(items) != 3:
            raise ValueError("rdflib.Graph.add expects a triple")
        subject, predicate, obj = (str(part) for part in items)
        self._triples.append((subject, predicate, obj))

    def bind(self, prefix: str, namespace: str) -> None:
        """Register a namespace binding within the stub graph."""

        self.namespace_manager.bind(prefix, namespace)

    def namespaces(self) -> Iterable[tuple[str, str]]:
        """Yield namespace bindings previously registered via :meth:`bind`."""

        return self.namespace_manager.namespaces()

    def __len__(self) -> int:
        return len(self._triples)

    def __iter__(self):
        return iter(self._triples)


def _import_module(name: str) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _STUB_ATTR, False):
        sys.modules.pop(name, None)
    return importlib.import_module(name)


def _create_pystow_stub(root: Path) -> ModuleType:
    root.mkdir(parents=True, exist_ok=True)

    def join(*segments: str) -> Path:
        """Mimic :func:`pystow.join` by joining segments onto the stub root."""

        return root.joinpath(*segments)

    def module(*segments: str, ensure_exists: bool = True) -> Path:
        """Return a child directory while optionally creating it."""

        target = root.joinpath(*segments)
        if ensure_exists:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def joinpath(*segments: str) -> Path:
        """Compatibility alias for :func:`join`."""

        return join(*segments)

    def ensure(*segments: str, directory: bool = True) -> Path:
        """Ensure a file or directory exists beneath the stub root."""

        target = root.joinpath(*segments)
        if directory:
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)
        return target

    def get_config(_module: str, _key: str, *, default=None, **_kwargs):
        """Return default configuration values."""

        return default

    module = _create_stub_module(
        "pystow",
        {
            "join": join,
            "joinpath": joinpath,
            "module": module,
            "ensure": ensure,
            "get_config": get_config,
        },
    )
    module._root = root  # type: ignore[attr-defined]
    return module


def _create_rdflib_stub() -> ModuleType:
    namespace_module = _create_stub_module(
        "rdflib.namespace",
        {
            "Namespace": _StubNamespace,
            "NamespaceManager": _StubNamespaceManager,
        },
    )
    graph_module = _create_stub_module(
        "rdflib.graph",
        {
            "Graph": _StubGraph,
        },
    )
    stub = _create_stub_module(
        "rdflib",
        {
            "Graph": _StubGraph,
            "Namespace": _StubNamespace,
            "NamespaceManager": _StubNamespaceManager,
            "BNode": _create_stub_bnode,
            "Literal": _create_stub_literal,
            "URIRef": _create_stub_uri,
        },
    )
    sys.modules.setdefault("rdflib.namespace", namespace_module)
    sys.modules.setdefault("rdflib.graph", graph_module)
    return stub


def _create_pronto_stub() -> ModuleType:
    class _StubOntology:
        _ontofetch_stub = True

        def __init__(self, _path: Optional[str] = None) -> None:
            self.path = _path

        def terms(self) -> Iterable[str]:
            """Return a deterministic collection of ontology term identifiers."""

            return ["TERM:0000001", "TERM:0000002"]

        def dump(self, destination: str, format: str = "obojson") -> None:
            """Write minimal ontology contents to ``destination`` for tests."""

            Path(destination).write_text('{"graphs": []}')

    return _create_stub_module("pronto", {"Ontology": _StubOntology})


def _create_owlready_stub() -> ModuleType:
    class _StubOntology:
        _ontofetch_stub = True

        def __init__(self, iri: str) -> None:
            self.iri = iri

        def load(self) -> "_StubOntology":
            """Provide fluent API parity with owlready2 ontologies."""

            return self

        def classes(self) -> List[str]:
            """Return example ontology classes for tests and fallbacks."""

            return ["Class1", "Class2", "Class3"]

    def get_ontology(iri: str) -> _StubOntology:
        """Return a stub ontology instance for the provided IRI."""

        return _StubOntology(iri)

    return _create_stub_module("owlready2", {"get_ontology": get_ontology})


def get_pystow() -> Any:
    """Return the ``pystow`` module, supplying a stub when unavailable."""

    global _pystow
    if _pystow is not None:
        return _pystow
    try:
        _pystow = _import_module("pystow")
    except ModuleNotFoundError:
        root = Path(os.environ.get("PYSTOW_HOME") or (Path.home() / ".data"))
        _pystow = _create_pystow_stub(root)
        sys.modules.setdefault("pystow", _pystow)
    return _pystow


def get_rdflib() -> Any:
    """Return the ``rdflib`` module, supplying a stub when unavailable."""

    global _rdflib
    if _rdflib is not None:
        return _rdflib
    try:
        _rdflib = _import_module("rdflib")
    except ModuleNotFoundError:
        _rdflib = _create_rdflib_stub()
        sys.modules.setdefault("rdflib", _rdflib)
    return _rdflib


def get_pronto() -> Any:
    """Return the ``pronto`` module, supplying a stub when unavailable."""

    global _pronto
    if _pronto is not None:
        return _pronto
    try:
        _pronto = _import_module("pronto")
    except ModuleNotFoundError:
        _pronto = _create_pronto_stub()
        sys.modules.setdefault("pronto", _pronto)
    return _pronto


def get_owlready2() -> Any:
    """Return the ``owlready2`` module, supplying a stub when unavailable."""

    global _owlready2
    if _owlready2 is not None:
        return _owlready2
    try:
        _owlready2 = _import_module("owlready2")
    except ModuleNotFoundError:
        _owlready2 = _create_owlready_stub()
        sys.modules.setdefault("owlready2", _owlready2)
    return _owlready2


# --- Storage backends merged from storage.py ---

try:  # pragma: no cover - optional dependency
    import fsspec  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when not installed
    fsspec = None  # type: ignore[assignment]

pystow = get_pystow()

DATA_ROOT = pystow.join("ontology-fetcher")
CONFIG_DIR = DATA_ROOT / "configs"
CACHE_DIR = DATA_ROOT / "cache"
LOG_DIR = DATA_ROOT / "logs"
LOCAL_ONTOLOGY_DIR = DATA_ROOT / "ontologies"

for directory in (CONFIG_DIR, CACHE_DIR, LOG_DIR, LOCAL_ONTOLOGY_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class StorageBackend(Protocol):
    """Protocol describing storage operations required by the pipeline."""

    root: Path

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Return a workspace for the given ontology/version combination."""

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Ensure a local directory exists for the requested ontology version."""

    def available_versions(self, ontology_id: str) -> List[str]:
        """Return sorted version identifiers available for *ontology_id*."""

    def available_ontologies(self) -> List[str]:
        """Return sorted ontology identifiers known to the backend."""

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Persist processed artefacts for *ontology_id*/*version*."""

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the canonical path for *ontology_id*/*version*."""

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Delete stored data for *ontology_id*/*version*, returning bytes reclaimed."""

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Record the latest processed version for *ontology_id*."""

    def mirror_cas_artifact(self, algorithm: str, digest: str, source: Path) -> Path:
        """Mirror ``source`` into a content-addressable cache and return its path."""

    def directory_size(self, path: Path) -> int:
        """Return the total size in bytes for files rooted at *path*."""


def _safe_identifiers(ontology_id: str, version: str) -> Tuple[str, str]:
    """Return identifiers sanitised for filesystem usage."""

    from .io import sanitize_filename  # Local import to avoid circular dependency

    safe_id = sanitize_filename(ontology_id)
    safe_version = sanitize_filename(version)
    return safe_id, safe_version


def _directory_size(path: Path) -> int:
    """Return the total size in bytes for files rooted at *path*."""

    total = 0
    for entry in path.rglob("*"):
        try:
            info = entry.stat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


class LocalStorageBackend:
    """Storage backend that keeps artefacts on the local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root: Path = root

    def _version_dir(self, ontology_id: str, version: str) -> Path:
        safe_id, safe_version = _safe_identifiers(ontology_id, version)
        return self.root / safe_id / safe_version

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Create a workspace for ``ontology_id``/``version`` with required subdirs."""

        base = self.ensure_local_version(ontology_id, version)
        for subdir in ("original", "normalized", "validation"):
            (base / subdir).mkdir(parents=True, exist_ok=True)
        return base

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Ensure the version directory exists and return its path."""

        base = self._version_dir(ontology_id, version)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def available_versions(self, ontology_id: str) -> List[str]:
        """List version identifiers currently stored for ``ontology_id``."""

        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        base = self.root / safe_id
        if not base.exists():
            return []
        versions = [entry.name for entry in base.iterdir() if entry.is_dir()]
        return sorted(versions)

    def available_ontologies(self) -> List[str]:
        """List ontology identifiers known to the local backend."""

        if not self.root.exists():
            return []
        return sorted([entry.name for entry in self.root.iterdir() if entry.is_dir()])

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Hook for subclasses; local backend writes occur in-place already."""

        _ = (ontology_id, version, local_dir)  # pragma: no cover - intentional no-op

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the filesystem path holding ``ontology_id``/``version`` data."""

        return self._version_dir(ontology_id, version)

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Remove stored data for ``ontology_id``/``version`` and return bytes reclaimed."""

        path = self._version_dir(ontology_id, version)
        if not path.exists():
            return 0
        reclaimed = _directory_size(path)
        shutil.rmtree(path)
        return reclaimed

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Update symbolic links/markers to highlight the latest processed version."""

        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        base = self.root / safe_id
        base.mkdir(parents=True, exist_ok=True)
        link = base / "latest"
        marker = base / "latest.txt"
        target = Path(version)

        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if marker.exists():
                marker.unlink()
            marker.write_text(version)
        else:
            if marker.exists():
                marker.unlink()

    def mirror_cas_artifact(self, algorithm: str, digest: str, source: Path) -> Path:
        """Copy ``source`` into the content-addressable cache."""

        algo = algorithm.lower() or "sha256"
        cas_root = self.root / f"by-{algo}"
        prefix = digest[:2] if len(digest) >= 2 else digest or "00"
        cas_dir = cas_root / prefix
        cas_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix if source.suffix else ""
        cas_path = cas_dir / f"{digest}{suffix}"
        if not cas_path.exists():
            shutil.copy2(source, cas_path)
        return cas_path

    def directory_size(self, path: Path) -> int:
        """Return the byte size for files rooted at ``path``."""

        return _directory_size(path)


class FsspecStorageBackend(LocalStorageBackend):
    """Hybrid backend that mirrors artefacts to an fsspec location."""

    def __init__(self, url: str) -> None:
        if fsspec is None:  # pragma: no cover - dependency missing
            raise ConfigError(
                "fsspec required for remote storage. Install it via 'pip install fsspec'."
            )
        fs, path = fsspec.core.url_to_fs(url)  # type: ignore[attr-defined]
        self.fs = fs
        self.base_path = PurePosixPath(path)
        super().__init__(LOCAL_ONTOLOGY_DIR)

    def _remote_version_path(self, ontology_id: str, version: str) -> PurePosixPath:
        """Return the remote storage path for ``ontology_id``/``version``."""

        safe_id, safe_version = _safe_identifiers(ontology_id, version)
        return (self.base_path / safe_id / safe_version).with_suffix("")

    def available_versions(self, ontology_id: str) -> List[str]:
        """Combine local and remote version identifiers for ``ontology_id``."""

        local_versions = super().available_versions(ontology_id)
        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        remote_dir = self.base_path / safe_id
        try:
            entries = self.fs.ls(str(remote_dir), detail=False)
        except FileNotFoundError:
            entries = []
        remote_versions = [
            PurePosixPath(entry).name for entry in entries if entry and not entry.endswith(".tmp")
        ]
        return sorted({*local_versions, *remote_versions})

    def available_ontologies(self) -> List[str]:
        """Return the union of ontology ids present locally and in remote storage."""

        local = set(super().available_ontologies())
        try:
            entries = self.fs.ls(str(self.base_path), detail=False)
        except FileNotFoundError:
            entries = []
        remote = {PurePosixPath(entry).name for entry in entries if entry}
        return sorted(local | remote)

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Mirror processed artefacts to remote storage after local completion."""

        remote_dir = self._remote_version_path(ontology_id, version)
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(local_dir)
            remote_path = remote_dir / PurePosixPath(str(relative).replace("\\", "/"))
            self.fs.makedirs(str(remote_path.parent), exist_ok=True)
            self.fs.put_file(str(path), str(remote_path))

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Remove local and remote artefacts for ``ontology_id``/``version``."""

        reclaimed = super().delete_version(ontology_id, version)
        remote_dir = self._remote_version_path(ontology_id, version)
        if not self.fs.exists(str(remote_dir)):
            return reclaimed

        try:
            remote_files = self.fs.find(str(remote_dir))
        except FileNotFoundError:
            remote_files = []
        for remote_file in remote_files:
            try:
                info = self.fs.info(remote_file)
            except FileNotFoundError:
                continue
            size = info.get("size") if isinstance(info, dict) else None
            if isinstance(size, (int, float)):
                reclaimed += int(size)
        self.fs.rm(str(remote_dir), recursive=True)
        return reclaimed

    def mirror_cas_artifact(self, algorithm: str, digest: str, source: Path) -> Path:
        """Mirror CAS artefact locally and to remote storage."""

        local_path = super().mirror_cas_artifact(algorithm, digest, source)
        algo = algorithm.lower() or "sha256"
        prefix = digest[:2] if len(digest) >= 2 else digest or "00"
        remote_dir = self.base_path / f"by-{algo}" / prefix
        remote_filename = PurePosixPath(f"{digest}{source.suffix if source.suffix else ''}")
        remote_path = remote_dir / remote_filename
        self.fs.makedirs(str(remote_path.parent), exist_ok=True)
        self.fs.put_file(str(local_path), str(remote_path))
        return local_path


def get_storage_backend() -> StorageBackend:
    """Instantiate the storage backend based on environment configuration."""

    storage_url = os.getenv("ONTOFETCH_STORAGE_URL")
    if storage_url:
        return FsspecStorageBackend(storage_url)
    return LocalStorageBackend(LOCAL_ONTOLOGY_DIR)


STORAGE: StorageBackend = get_storage_backend()


__all__ = [
    "UserConfigError",
    "OntologyDownloadError",
    "ResolverError",
    "ValidationError",
    "PolicyError",
    "DownloadFailure",
    "ConfigError",
    "DefaultsConfig",
    "DownloadConfiguration",
    "LoggingConfiguration",
    "ValidationConfig",
    "PlannerConfig",
    "ResolvedConfig",
    "EnvironmentOverrides",
    "build_resolved_config",
    "ensure_python_version",
    "get_env_overrides",
    "load_config",
    "load_raw_yaml",
    "normalize_config_path",
    "parse_rate_limit_to_rps",
    "validate_config",
    "get_pystow",
    "get_rdflib",
    "get_pronto",
    "get_owlready2",
    "DATA_ROOT",
    "CONFIG_DIR",
    "CACHE_DIR",
    "LOG_DIR",
    "LOCAL_ONTOLOGY_DIR",
    "StorageBackend",
    "LocalStorageBackend",
    "FsspecStorageBackend",
    "get_storage_backend",
    "STORAGE",
]
