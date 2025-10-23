# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.settings",
#   "purpose": "Define configuration models, environment overrides, optional dependency shims, and storage backends",
#   "sections": [
#     {
#       "id": "ensure-python-version",
#       "name": "ensure_python_version",
#       "anchor": "function-ensure-python-version",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-sequence",
#       "name": "_coerce_sequence",
#       "anchor": "function-coerce-sequence",
#       "kind": "function"
#     },
#     {
#       "id": "parse-rate-limit-to-rps",
#       "name": "parse_rate_limit_to_rps",
#       "anchor": "function-parse-rate-limit-to-rps",
#       "kind": "function"
#     },
#     {
#       "id": "loggingconfiguration",
#       "name": "LoggingConfiguration",
#       "anchor": "class-loggingconfiguration",
#       "kind": "class"
#     },
#     {
#       "id": "databaseconfiguration",
#       "name": "DatabaseConfiguration",
#       "anchor": "class-databaseconfiguration",
#       "kind": "class"
#     },
#     {
#       "id": "validationconfig",
#       "name": "ValidationConfig",
#       "anchor": "class-validationconfig",
#       "kind": "class"
#     },
#     {
#       "id": "downloadconfiguration",
#       "name": "DownloadConfiguration",
#       "anchor": "class-downloadconfiguration",
#       "kind": "class"
#     },
#     {
#       "id": "plannerconfig",
#       "name": "PlannerConfig",
#       "anchor": "class-plannerconfig",
#       "kind": "class"
#     },
#     {
#       "id": "duckdbsettings",
#       "name": "DuckDBSettings",
#       "anchor": "class-duckdbsettings",
#       "kind": "class"
#     },
#     {
#       "id": "storagesettings",
#       "name": "StorageSettings",
#       "anchor": "class-storagesettings",
#       "kind": "class"
#     },
#     {
#       "id": "defaultsconfig",
#       "name": "DefaultsConfig",
#       "anchor": "class-defaultsconfig",
#       "kind": "class"
#     },
#     {
#       "id": "resolvedconfig",
#       "name": "ResolvedConfig",
#       "anchor": "class-resolvedconfig",
#       "kind": "class"
#     },
#     {
#       "id": "rebuild-pydantic-models",
#       "name": "_rebuild_pydantic_models",
#       "anchor": "function-rebuild-pydantic-models",
#       "kind": "function"
#     },
#     {
#       "id": "read-env-value",
#       "name": "_read_env_value",
#       "anchor": "function-read-env-value",
#       "kind": "function"
#     },
#     {
#       "id": "read-env-int",
#       "name": "_read_env_int",
#       "anchor": "function-read-env-int",
#       "kind": "function"
#     },
#     {
#       "id": "read-env-float",
#       "name": "_read_env_float",
#       "anchor": "function-read-env-float",
#       "kind": "function"
#     },
#     {
#       "id": "read-env-path",
#       "name": "_read_env_path",
#       "anchor": "function-read-env-path",
#       "kind": "function"
#     },
#     {
#       "id": "get-default-config",
#       "name": "get_default_config",
#       "anchor": "function-get-default-config",
#       "kind": "function"
#     },
#     {
#       "id": "invalidate-default-config-cache",
#       "name": "invalidate_default_config_cache",
#       "anchor": "function-invalidate-default-config-cache",
#       "kind": "function"
#     },
#     {
#       "id": "get-env-overrides",
#       "name": "get_env_overrides",
#       "anchor": "function-get-env-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "apply-env-overrides",
#       "name": "_apply_env_overrides",
#       "anchor": "function-apply-env-overrides",
#       "kind": "function"
#     },
#     {
#       "id": "build-resolved-config",
#       "name": "build_resolved_config",
#       "anchor": "function-build-resolved-config",
#       "kind": "function"
#     },
#     {
#       "id": "validate-schema",
#       "name": "_validate_schema",
#       "anchor": "function-validate-schema",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-config-path",
#       "name": "normalize_config_path",
#       "anchor": "function-normalize-config-path",
#       "kind": "function"
#     },
#     {
#       "id": "load-raw-yaml",
#       "name": "load_raw_yaml",
#       "anchor": "function-load-raw-yaml",
#       "kind": "function"
#     },
#     {
#       "id": "load-config",
#       "name": "load_config",
#       "anchor": "function-load-config",
#       "kind": "function"
#     },
#     {
#       "id": "validate-config",
#       "name": "validate_config",
#       "anchor": "function-validate-config",
#       "kind": "function"
#     },
#     {
#       "id": "create-stub-module",
#       "name": "_create_stub_module",
#       "anchor": "function-create-stub-module",
#       "kind": "function"
#     },
#     {
#       "id": "create-stub-bnode",
#       "name": "_create_stub_bnode",
#       "anchor": "function-create-stub-bnode",
#       "kind": "function"
#     },
#     {
#       "id": "create-stub-literal",
#       "name": "_create_stub_literal",
#       "anchor": "function-create-stub-literal",
#       "kind": "function"
#     },
#     {
#       "id": "create-stub-uri",
#       "name": "_create_stub_uri",
#       "anchor": "function-create-stub-uri",
#       "kind": "function"
#     },
#     {
#       "id": "stubnamespace",
#       "name": "_StubNamespace",
#       "anchor": "class-stubnamespace",
#       "kind": "class"
#     },
#     {
#       "id": "stubnamespacemanager",
#       "name": "_StubNamespaceManager",
#       "anchor": "class-stubnamespacemanager",
#       "kind": "class"
#     },
#     {
#       "id": "stubgraph",
#       "name": "_StubGraph",
#       "anchor": "class-stubgraph",
#       "kind": "class"
#     },
#     {
#       "id": "import-module",
#       "name": "_import_module",
#       "anchor": "function-import-module",
#       "kind": "function"
#     },
#     {
#       "id": "create-pystow-stub",
#       "name": "_create_pystow_stub",
#       "anchor": "function-create-pystow-stub",
#       "kind": "function"
#     },
#     {
#       "id": "create-rdflib-stub",
#       "name": "_create_rdflib_stub",
#       "anchor": "function-create-rdflib-stub",
#       "kind": "function"
#     },
#     {
#       "id": "create-pronto-stub",
#       "name": "_create_pronto_stub",
#       "anchor": "function-create-pronto-stub",
#       "kind": "function"
#     },
#     {
#       "id": "create-owlready-stub",
#       "name": "_create_owlready_stub",
#       "anchor": "function-create-owlready-stub",
#       "kind": "function"
#     },
#     {
#       "id": "get-pystow",
#       "name": "get_pystow",
#       "anchor": "function-get-pystow",
#       "kind": "function"
#     },
#     {
#       "id": "get-rdflib",
#       "name": "get_rdflib",
#       "anchor": "function-get-rdflib",
#       "kind": "function"
#     },
#     {
#       "id": "get-pronto",
#       "name": "get_pronto",
#       "anchor": "function-get-pronto",
#       "kind": "function"
#     },
#     {
#       "id": "get-owlready2",
#       "name": "get_owlready2",
#       "anchor": "function-get-owlready2",
#       "kind": "function"
#     },
#     {
#       "id": "storagebackend",
#       "name": "StorageBackend",
#       "anchor": "class-storagebackend",
#       "kind": "class"
#     },
#     {
#       "id": "safe-identifiers",
#       "name": "_safe_identifiers",
#       "anchor": "function-safe-identifiers",
#       "kind": "function"
#     },
#     {
#       "id": "directory-size",
#       "name": "_directory_size",
#       "anchor": "function-directory-size",
#       "kind": "function"
#     },
#     {
#       "id": "localstoragebackend",
#       "name": "LocalStorageBackend",
#       "anchor": "class-localstoragebackend",
#       "kind": "class"
#     },
#     {
#       "id": "fsspecstoragebackend",
#       "name": "FsspecStorageBackend",
#       "anchor": "class-fsspecstoragebackend",
#       "kind": "class"
#     },
#     {
#       "id": "get-storage-backend",
#       "name": "get_storage_backend",
#       "anchor": "function-get-storage-backend",
#       "kind": "function"
#     },
#     {
#       "id": "httpsettings",
#       "name": "HttpSettings",
#       "anchor": "class-httpsettings",
#       "kind": "class"
#     },
#     {
#       "id": "cachesettings",
#       "name": "CacheSettings",
#       "anchor": "class-cachesettings",
#       "kind": "class"
#     },
#     {
#       "id": "retrysettings",
#       "name": "RetrySettings",
#       "anchor": "class-retrysettings",
#       "kind": "class"
#     },
#     {
#       "id": "loggingsettings",
#       "name": "LoggingSettings",
#       "anchor": "class-loggingsettings",
#       "kind": "class"
#     },
#     {
#       "id": "telemetrysettings",
#       "name": "TelemetrySettings",
#       "anchor": "class-telemetrysettings",
#       "kind": "class"
#     },
#     {
#       "id": "securitysettings",
#       "name": "SecuritySettings",
#       "anchor": "class-securitysettings",
#       "kind": "class"
#     },
#     {
#       "id": "ratelimitsettings",
#       "name": "RateLimitSettings",
#       "anchor": "class-ratelimitsettings",
#       "kind": "class"
#     },
#     {
#       "id": "extractionsettings",
#       "name": "ExtractionSettings",
#       "anchor": "class-extractionsettings",
#       "kind": "class"
#     }
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
import json
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

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from pydantic import ValidationError as PydanticValidationError
from pydantic_core import ValidationError as CoreValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import (
    UnsupportedPythonError,
    UserConfigError,
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


class DatabaseConfiguration(BaseModel):
    """DuckDB catalog configuration for ontology metadata tracking."""

    db_path: Optional[Path] = Field(
        default=None,
        description="Path to DuckDB file; defaults to ~/.data/ontology-fetcher/.catalog/ontofetch.duckdb",
    )
    readonly: bool = Field(default=False, description="Open database in read-only mode")
    enable_locks: bool = Field(
        default=True,
        description="Enable file-based locks to serialize writers; readers bypass locks",
    )
    threads: Optional[int] = Field(
        default=None,
        description="Number of threads for query execution; None uses CPU count",
    )
    memory_limit: Optional[str] = Field(
        default=None,
        description="Memory limit as string (e.g., '8GB'); None uses auto",
    )
    enable_object_cache: bool = Field(
        default=True,
        description="Enable object cache for repeated remote file metadata scans",
    )
    parquet_events: bool = Field(
        default=False,
        description="Store events as Parquet files instead of table; attach on demand",
    )

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


class DuckDBSettings(BaseModel):
    """DuckDB catalog database configuration.

    Controls the DuckDB catalog (brain) that tracks ontology versions,
    artifacts, extracted files, and validations.
    """

    path: Path = Field(
        default_factory=lambda: Path.home() / ".data" / ".catalog" / "ontofetch.duckdb",
        description="Path to DuckDB file",
    )
    threads: int = Field(
        default=8, gt=0, le=256, description="Number of threads for DuckDB query execution"
    )
    readonly: bool = Field(default=False, description="Open in read-only mode")
    writer_lock: bool = Field(default=True, description="Use writer lock")

    @field_validator("path", mode="before")
    @classmethod
    def normalize_path(cls, v: Any) -> Path:
        """Normalize to absolute path."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        raise ValueError("path must be string or Path")

    model_config = {"validate_assignment": True, "extra": "forbid"}


class StorageSettings(BaseModel):
    """Storage backend configuration for ontology files."""

    root: Path = Field(
        default_factory=lambda: Path.home() / "ontologies", description="Root directory"
    )
    latest_name: str = Field(default="LATEST.json", description="Latest marker filename")
    write_latest_mirror: bool = Field(default=True, description="Write JSON mirror")

    @field_validator("root", mode="before")
    @classmethod
    def normalize_root(cls, v: Any) -> Path:
        """Normalize root to absolute path."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        raise ValueError("root must be string or Path")

    @field_validator("latest_name")
    @classmethod
    def validate_latest_name(cls, v: str) -> str:
        """Ensure latest_name is a valid filename."""
        if "/" in v or "\\" in v:
            raise ValueError("latest_name must be a filename, not a path")
        if not v.strip():
            raise ValueError("latest_name cannot be empty")
        return v.strip()

    model_config = {"validate_assignment": True, "extra": "forbid"}


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
    db: DuckDBSettings = Field(
        default_factory=DuckDBSettings, description="DuckDB catalog (brain) configuration"
    )
    storage: StorageSettings = Field(
        default_factory=StorageSettings,
        description="Storage backend configuration for ontology files",
    )
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

    def config_hash(self) -> str:
        """Compute a deterministic hash of all configuration for provenance tracking."""
        import hashlib

        config_dict = {
            "http": self.defaults.http.model_dump(mode="json"),
            "planner": self.defaults.planner.model_dump(mode="json"),
            "validation": self.defaults.validation.model_dump(mode="json"),
            "logging": self.defaults.logging.model_dump(mode="json"),
            "db": {
                "path": str(self.defaults.db.path),
                "threads": self.defaults.db.threads,
                "readonly": self.defaults.db.readonly,
                "writer_lock": self.defaults.db.writer_lock,
            },
            "storage": {
                "root": str(self.defaults.storage.root),
                "latest_name": self.defaults.storage.latest_name,
                "write_latest_mirror": self.defaults.storage.write_latest_mirror,
            },
        }
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


_DEFAULT_CONFIG_LOCK = threading.RLock()
_DEFAULT_CONFIG_CACHE: Optional[ResolvedConfig] = None

_HAS_PYDANTIC_SETTINGS = hasattr(BaseSettings, "model_dump")


def _rebuild_pydantic_models() -> None:
    """Rebuild Pydantic models that have forward references to types defined in other modules.

    ResolvedConfig has a forward reference to FetchSpec (defined in planning.py).
    This function must be called after all referenced types are imported to ensure
    the model is fully constructed and can be instantiated without errors.

    Deferred until first config access to avoid circular imports at module load time.
    """
    try:
        from . import planning  # noqa: F401  # Ensure FetchSpec is imported

        ResolvedConfig.model_rebuild()
    except ImportError:
        # If planning fails to import, rebuild will be attempted on next call
        pass


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
            # Rebuild models on first access to resolve forward references
            _rebuild_pydantic_models()
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


# ============================================================================
# PHASE 5: PYDANTIC v2 SETTINGS DOMAIN MODELS (Foundation)
# ============================================================================
# Phase 5.1: Domain Models Foundation
# Implements: HttpSettings, CacheSettings, RetrySettings, LoggingSettings, TelemetrySettings
# Status: In Progress (Phase 5.1)
# ============================================================================


class HttpSettings(BaseModel):
    """HTTP client settings for HTTPX + Hishel integration.

    Controls timeout, pool, HTTP/2, user agent, and proxy trust behavior.
    """

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    http2: bool = Field(default=True, description="Enable HTTP/2 support")
    timeout_connect: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description="Connect timeout in seconds",
    )
    timeout_read: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        description="Read timeout in seconds",
    )
    timeout_write: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        description="Write timeout in seconds",
    )
    timeout_pool: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description="Acquire-from-pool timeout in seconds",
    )
    pool_max_connections: int = Field(
        default=64,
        ge=1,
        le=1024,
        description="Max concurrent connections",
    )
    pool_keepalive_max: int = Field(
        default=20,
        ge=0,
        le=1024,
        description="Keepalive pool size",
    )
    keepalive_expiry: float = Field(
        default=30.0,
        ge=0.0,
        le=600.0,
        description="Idle connection expiry in seconds",
    )
    trust_env: bool = Field(
        default=True,
        description="Honor HTTP(S)_PROXY and NO_PROXY environment variables",
    )
    user_agent: str = Field(
        default="DocsToKG/OntoFetch (+https://github.com/allenai/DocsToKG)",
        description="User-Agent header value",
    )


class CacheSettings(BaseModel):
    """Hishel HTTP cache settings."""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    enabled: bool = Field(default=True, description="Enable Hishel RFC-9111 cache")
    dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "ontofetch" / "http",
        description="Cache directory (auto-created if needed)",
    )
    bypass: bool = Field(
        default=False,
        description="Force bypass cache (no revalidation)",
    )

    @field_validator("dir", mode="before")
    @classmethod
    def normalize_cache_dir(cls, v: Any) -> Path:
        """Normalize cache directory to absolute path."""
        if isinstance(v, str):
            p = Path(v).expanduser()
        elif isinstance(v, Path):
            p = v.expanduser()
        else:
            p = Path(v).expanduser()
        return p.resolve()


class RetrySettings(BaseModel):
    """HTTP retry settings for transient failures."""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    connect_retries: int = Field(
        default=2,
        ge=0,
        le=20,
        description="Number of connect retries",
    )
    backoff_base: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Backoff start (seconds)",
    )
    backoff_max: float = Field(
        default=2.0,
        ge=0.0,
        le=60.0,
        description="Backoff cap (seconds)",
    )


class LoggingSettings(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    emit_json_logs: bool = Field(
        default=True,
        description="Emit JSON-formatted logs (legacy configs may use 'json')",
        validation_alias=AliasChoices("emit_json_logs", "json"),
    )

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, v: str) -> str:
        """Normalize and validate logging level."""
        upper = v.upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if upper not in valid_levels:
            raise ValueError(f"level must be one of {sorted(valid_levels)}, got '{v}'")
        return upper

    def level_int(self) -> int:
        """Convert level string to logging module integer."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        return level_map[self.level]


class TelemetrySettings(BaseModel):
    """Telemetry and observability settings."""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    run_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique run identifier for tracing and provenance",
    )
    emit_events: bool = Field(
        default=True,
        description="Emit telemetry events to logs and database",
    )

    @field_validator("run_id", mode="before")
    @classmethod
    def coerce_run_id(cls, v: Any) -> uuid.UUID:
        """Convert run_id from string or UUID."""
        if isinstance(v, uuid.UUID):
            return v
        if isinstance(v, str):
            return uuid.UUID(v)
        raise ValueError(f"run_id must be UUID or valid UUID string, got {type(v)}")


# ============================================================================
# PHASE 5.2: COMPLEX DOMAIN MODELS
# ============================================================================
# Phase 5.2: Complex Domains with Advanced Parsing
# Implements: SecuritySettings, RateLimitSettings, ExtractionSettings, StorageSettings, DuckDBSettings
# Status: In Progress (Phase 5.2)
# ============================================================================


class SecuritySettings(BaseModel):
    """URL security and DNS settings."""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    allowed_hosts: Optional[List[str]] = Field(
        default=None,
        description="Comma-separated allowed hosts (supports *.suffix, IP/CIDR, host:port)",
    )
    allowed_ports: Optional[List[int]] = Field(
        default=None,
        description="Allowed ports; defaults to 80,443 if not specified",
    )
    allow_private_networks: bool = Field(
        default=False,
        description="Allow private/loopback addresses if allowlisted",
    )
    allow_plain_http: bool = Field(
        default=False,
        description="Allow plain HTTP (non-HTTPS) for allowlisted hosts",
    )
    strict_dns: bool = Field(
        default=True,
        description="Fail on DNS resolution errors",
    )

    @field_validator("allowed_ports", mode="before")
    @classmethod
    def parse_ports(cls, v: Any) -> Optional[List[int]]:
        """Parse and validate port list."""
        if v is None:
            return None
        if isinstance(v, str):
            v = [int(p.strip()) for p in v.split(",") if p.strip()]
        if isinstance(v, list):
            for port in v:
                if not isinstance(port, int) or port < 1 or port > 65535:
                    raise ValueError(f"Port must be 1-65535, got {port}")
            return v
        raise ValueError(f"allowed_ports must be list or CSV string, got {type(v)}")

    def normalized_allowed_hosts(
        self,
    ) -> Optional[Tuple[Set[str], Set[str], Dict[str, Set[int]], Set[str]]]:
        """Parse allowed_hosts into exact domains, wildcard suffixes, per-host ports, and IP literals."""
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

            # Handle IPv6 literals [::1]:port
            if working.startswith("["):
                end_bracket = working.find("]")
                if end_bracket == -1:
                    raise ValueError(f"Invalid IPv6 literal '{entry}'")
                literal = working[1:end_bracket]
                remainder = working[end_bracket + 1 :]
                if remainder:
                    if not remainder.startswith(":"):
                        raise ValueError(f"Invalid port in entry '{entry}'")
                    port_str = remainder[1:]
                    if not port_str.isdigit():
                        raise ValueError(f"Invalid port '{port_str}' in entry '{entry}'")
                    port = int(port_str)
                    if port < 1 or port > 65535:
                        raise ValueError(f"Port out of range in entry '{entry}'")
                working = literal
            # Handle IPv4:port or hostname:port
            elif ":" in working and working.count(":") == 1:
                host_candidate, maybe_port = working.rsplit(":", 1)
                if maybe_port.isdigit():
                    port_value = int(maybe_port)
                    if port_value < 1 or port_value > 65535:
                        raise ValueError(f"Port out of range in entry '{entry}'")
                    port = port_value
                    working = host_candidate
                else:
                    raise ValueError(f"Invalid port in entry '{entry}'")

            # Handle wildcards
            wildcard = False
            if working.startswith("*."):
                wildcard = True
                working = working[2:]
            elif working.startswith("."):
                wildcard = True
                working = working[1:]

            # Parse as IP or hostname
            try:
                ipaddress.ip_address(working)
                normalized = working.lower()
                if wildcard:
                    raise ValueError(f"Wildcard not allowed for IP address '{entry}'")
                ip_literals.add(normalized)
            except ValueError:
                # Try hostname
                try:
                    normalized = working.encode("idna").decode("ascii").lower()
                except UnicodeError:
                    raise ValueError(f"Invalid hostname in entry '{entry}'")

                if wildcard and port is not None:
                    raise ValueError(f"Wildcard entries cannot specify ports: '{entry}'")

                if wildcard:
                    suffixes.add(normalized)
                else:
                    exact.add(normalized)
                    if port is not None:
                        host_ports.setdefault(normalized, set()).add(port)

        if not exact and not suffixes and not ip_literals and not host_ports:
            return None

        return exact, suffixes, host_ports, ip_literals

    def allowed_port_set(self) -> Set[int]:
        """Return the set of allowed ports (defaults to 80, 443)."""
        if self.allowed_ports:
            return set(self.allowed_ports)
        return {80, 443}


class RateLimitSettings(BaseModel):
    """Rate limiting configuration."""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    default: Optional[str] = Field(
        default=None,
        description="Default rate limit (e.g., '10/second', '60/minute')",
    )
    per_service: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-service rate limits (e.g., {'ols': '4/second'})",
    )
    shared_dir: Optional[Path] = Field(
        default=None,
        description="Directory for shared SQLite token bucket state",
    )
    engine: str = Field(
        default="pyrate",
        description="Rate limit engine (currently only 'pyrate' supported)",
    )

    @field_validator("default", mode="before")
    @classmethod
    def validate_rate_string(cls, v: Optional[str]) -> Optional[str]:
        """Validate rate limit string format."""
        if v is None:
            return None
        if not _RATE_LIMIT_PATTERN.match(v):
            raise ValueError(f"Invalid rate limit format '{v}'; expected 'N/(second|minute|hour)'")
        return v

    @field_validator("per_service", mode="before")
    @classmethod
    def validate_per_service(cls, v: Any) -> Dict[str, str]:
        """Parse and validate per-service rate limits."""
        if isinstance(v, str):
            # Parse CSV format: "ols:4/second;bioportal:2/second"
            v = {
                service: rate
                for pair in v.split(";")
                for service, rate in [pair.split(":")]
                if service and rate
            }
        if isinstance(v, dict):
            for service, rate in v.items():
                if not _RATE_LIMIT_PATTERN.match(rate):
                    raise ValueError(f"Invalid rate limit '{rate}' for service '{service}'")
            return v
        return {}

    def parse_service_rate_limit(self, service: str) -> Optional[float]:
        """Parse service-specific rate limit to requests-per-second."""
        rate_str = self.per_service.get(service)
        if rate_str is None:
            return parse_rate_limit_to_rps(self.default)
        return parse_rate_limit_to_rps(rate_str)


class ExtractionSettings(BaseModel):
    """Archive extraction policy (safety, throughput, integrity)."""

    model_config = ConfigDict(frozen=True, validate_assignment=False)

    # Safety settings
    encapsulate: bool = Field(default=True, description="Extract in deterministic root")
    encapsulation_name: str = Field(
        default="sha256",
        description="Encapsulation strategy (sha256 or basename)",
    )
    max_depth: int = Field(default=32, ge=1, le=255, description="Max path depth")
    max_components_len: int = Field(
        default=240, ge=1, le=4096, description="Max bytes per path component"
    )
    max_path_len: int = Field(default=4096, ge=1, le=32768, description="Max bytes per full path")
    max_entries: int = Field(default=50000, ge=1, le=1000000, description="Max extractable entries")
    max_file_size_bytes: int = Field(
        default=2147483648, ge=1, description="Per-file size cap (2GB default)"
    )
    max_total_ratio: float = Field(
        default=10.0, ge=1.0, le=1000.0, description="Zip-bomb ratio (uncompressed/compressed)"
    )
    max_entry_ratio: float = Field(
        default=100.0, ge=1.0, le=10000.0, description="Per-entry ratio cap"
    )
    unicode_form: str = Field(default="NFC", description="Unicode normalization (NFC or NFD)")
    casefold_collision_policy: str = Field(
        default="reject", description="Case collision policy (reject or allow)"
    )
    overwrite: str = Field(
        default="reject",
        description="Overwrite policy (reject, replace, keep_existing)",
    )
    duplicate_policy: str = Field(
        default="reject", description="Duplicate policy (reject, first_wins, last_wins)"
    )

    # Throughput settings
    space_safety_margin: float = Field(
        default=1.10, ge=1.0, le=10.0, description="Free-space headroom"
    )
    preallocate: bool = Field(default=True, description="Preallocate files when size known")
    copy_buffer_min: int = Field(
        default=65536, ge=1024, description="Min copy buffer bytes (64 KiB)"
    )
    copy_buffer_max: int = Field(
        default=1048576, ge=65536, description="Max copy buffer bytes (1 MiB)"
    )
    group_fsync: int = Field(default=32, ge=1, le=1000, description="fsync directory every N files")
    max_wall_time_seconds: int = Field(
        default=120, ge=1, le=3600, description="Soft time budget per archive"
    )

    # Integrity settings
    hash_enable: bool = Field(default=True, description="Compute file digests")
    hash_algorithms: List[str] = Field(
        default_factory=lambda: ["sha256"],
        description="Hash algorithms (e.g., sha256, sha1)",
    )
    include_globs: List[str] = Field(
        default_factory=list, description="Include patterns (empty = all)"
    )
    exclude_globs: List[str] = Field(default_factory=list, description="Exclude patterns")
    timestamps_mode: str = Field(
        default="preserve",
        description="Timestamp handling (preserve, normalize, source_date_epoch)",
    )
    timestamps_normalize_to: str = Field(
        default="archive_mtime",
        description="When normalizing: archive_mtime or now",
    )

    @field_validator("encapsulation_name", mode="before")
    @classmethod
    def validate_encapsulation_name(cls, v: str) -> str:
        """Validate encapsulation strategy."""
        valid = {"sha256", "basename"}
        if v.lower() not in valid:
            raise ValueError(f"encapsulation_name must be one of {valid}")
        return v.lower()

    @field_validator("unicode_form", mode="before")
    @classmethod
    def validate_unicode_form(cls, v: str) -> str:
        """Validate Unicode normalization form."""
        valid = {"NFC", "NFD"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"unicode_form must be one of {valid}")
        return upper

    @field_validator("overwrite", "duplicate_policy", "casefold_collision_policy", mode="before")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        """Validate extraction policies."""
        return v.lower()
