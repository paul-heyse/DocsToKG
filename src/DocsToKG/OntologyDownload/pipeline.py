"""Download planning and orchestration helpers for ontology fetching."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)
from urllib.parse import urlparse

try:  # pragma: no cover - platform specific availability
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - windows
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no cover - platform specific availability
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-windows
    msvcrt = None  # type: ignore[assignment]

from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler

import requests
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from . import net as net_module
from .config import (
    ConfigError,
    DefaultsConfig,
    ResolvedConfig,
    _coerce_sequence,
    ensure_python_version,
)
from .io_safe import (
    extract_archive_safe,
    generate_correlation_id,
    mask_sensitive_data,
    sanitize_filename,
    validate_url_security,
)
from .net import DownloadFailure, RDF_MIME_ALIASES, RDF_MIME_FORMAT_LABELS, download_stream
from .resolvers import RESOLVERS, FetchPlan, normalize_license_to_spdx
from .storage import CACHE_DIR, LOCAL_ONTOLOGY_DIR, LOG_DIR, STORAGE
from .validation_core import ValidationRequest, ValidationResult, run_validators

MANIFEST_SCHEMA_VERSION = "1.0"

_SUPPORTED_CHECKSUM_ALGORITHMS = {"md5", "sha1", "sha256", "sha512"}
_CHECKSUM_HEX_RE = re.compile(r"^[0-9a-f]{32,128}$")

MANIFEST_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DocsToKG Ontology Manifest",
    "type": "object",
    "required": [
        "schema_version",
        "id",
        "resolver",
        "url",
        "filename",
        "status",
        "sha256",
        "downloaded_at",
        "target_formats",
        "validation",
        "artifacts",
        "resolver_attempts",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "id": {"type": "string", "minLength": 1},
        "resolver": {"type": "string", "minLength": 1},
        "url": {
            "type": "string",
            "format": "uri",
            "pattern": r"^https?://",
        },
        "filename": {"type": "string", "minLength": 1},
        "version": {"type": ["string", "null"]},
        "license": {"type": ["string", "null"]},
        "status": {"type": "string", "minLength": 1},
        "sha256": {"type": "string", "minLength": 1},
        "normalized_sha256": {"type": ["string", "null"]},
        "fingerprint": {"type": ["string", "null"]},
        "etag": {"type": ["string", "null"]},
        "last_modified": {"type": ["string", "null"]},
        "content_type": {"type": ["string", "null"]},
        "content_length": {"type": ["integer", "null"], "minimum": 0},
        "source_media_type_label": {"type": ["string", "null"]},
        "downloaded_at": {"type": "string", "format": "date-time"},
        "target_formats": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "validation": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                    "details": {"type": "object"},
                    "output_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["ok", "details", "output_files"],
            },
        },
        "artifacts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "resolver_attempts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "resolver": {"type": "string"},
                    "url": {"type": "string"},
                    "attempt": {"type": "integer", "minimum": 1},
                    "status": {"type": "string"},
                    "error": {"type": "string"},
                },
                "required": ["resolver"],
            },
        },
    },
    "additionalProperties": True,
}


def _normalize_algorithm(algorithm: Optional[str], *, context: str) -> str:
    candidate = (algorithm or "sha256").strip().lower()
    if candidate not in _SUPPORTED_CHECKSUM_ALGORITHMS:
        raise ConfigError(f"{context}: unsupported checksum algorithm '{candidate}'")
    return candidate


def _normalize_checksum(algorithm: str, value: str, *, context: str) -> Tuple[str, str]:
    normalized_algorithm = _normalize_algorithm(algorithm, context=context)
    if not isinstance(value, str):
        raise ConfigError(f"{context}: checksum value must be a string")
    checksum = value.strip().lower()
    if not _CHECKSUM_HEX_RE.fullmatch(checksum):
        raise ConfigError(f"{context}: checksum must be a hexadecimal digest")
    return normalized_algorithm, checksum


def _checksum_from_extras(extras: Mapping[str, object], *, context: str) -> Tuple[Optional[str], Optional[str]]:
    payload = extras.get("checksum") if isinstance(extras, Mapping) else None
    if payload is None:
        return None, None
    if isinstance(payload, str):
        return _normalize_checksum("sha256", payload, context=context)
    if isinstance(payload, Mapping):
        algorithm = payload.get("algorithm", "sha256")
        value = payload.get("value")
        if not isinstance(algorithm, str):
            raise ConfigError(f"{context}: checksum algorithm must be a string")
        if not isinstance(value, str):
            raise ConfigError(f"{context}: checksum value must be a string")
        return _normalize_checksum(algorithm, value, context=context)
    raise ConfigError(f"{context}: checksum must be provided as a string or mapping")


def _checksum_url_from_extras(extras: Mapping[str, object], *, context: str) -> Tuple[Optional[str], Optional[str]]:
    payload = extras.get("checksum_url") if isinstance(extras, Mapping) else None
    if payload is None:
        return None, None
    if isinstance(payload, str):
        url = payload.strip()
        if not url:
            raise ConfigError(f"{context}: checksum_url must not be empty")
        return url, None
    if isinstance(payload, Mapping):
        url_value = payload.get("url")
        algorithm_value = payload.get("algorithm")
        if not isinstance(url_value, str) or not url_value.strip():
            raise ConfigError(f"{context}: checksum_url must include a non-empty 'url'")
        algorithm = None
        if algorithm_value is not None:
            if not isinstance(algorithm_value, str):
                raise ConfigError(f"{context}: checksum_url algorithm must be a string when provided")
            algorithm = _normalize_algorithm(algorithm_value, context=context)
        return url_value.strip(), algorithm
    raise ConfigError(f"{context}: checksum_url must be provided as a string or mapping")


def _extract_checksum_from_text(text: str, *, context: str) -> str:
    match = re.search(r"[0-9a-fA-F]{32,128}", text)
    if not match:
        raise OntologyDownloadError(f"Unable to parse checksum from {context}")
    return match.group(0).lower()


def _fetch_checksum_from_url(
    *,
    url: str,
    algorithm: str,
    http_config: DownloadConfiguration,
    logger: logging.Logger,
) -> str:
    secure_url = validate_url_security(url, http_config)
    try:
        response = requests.get(secure_url, timeout=http_config.timeout_sec)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OntologyDownloadError(f"Failed to fetch checksum from {secure_url}: {exc}") from exc
    digest = _extract_checksum_from_text(response.text, context=secure_url)
    logger.info(
        "fetched checksum",
        extra={
            "stage": "download",
            "checksum_url": secure_url,
            "algorithm": algorithm,
        },
    )
    return digest


def _resolve_expected_checksum(
    *,
    spec: FetchSpec,
    plan: FetchPlan,
    download_config: DownloadConfiguration,
    logger: logging.Logger,
) -> Optional[str]:
    """Determine the expected checksum string passed to the downloader."""

    context = f"ontology '{spec.id}'"
    plan_checksum: Optional[Tuple[str, str]] = None
    if plan.checksum:
        algorithm = plan.checksum_algorithm or "sha256"
        plan_checksum = _normalize_checksum(algorithm, plan.checksum, context=f"{context} resolver checksum")

    spec_checksum = _checksum_from_extras(spec.extras, context=context)
    if spec_checksum[1] is not None and plan_checksum is not None and spec_checksum[1] != plan_checksum[1]:
        raise ConfigError(f"{context}: conflicting checksum values between resolver and specification extras")

    algorithm: Optional[str] = None
    value: Optional[str] = None
    if plan_checksum is not None:
        algorithm, value = plan_checksum
    if spec_checksum[1] is not None:
        algorithm, value = spec_checksum

    checksum_url_source: Optional[Tuple[str, Optional[str]]] = None
    if plan.checksum_url:
        checksum_url_source = (plan.checksum_url, plan.checksum_algorithm)
    else:
        url_from_extras = _checksum_url_from_extras(spec.extras, context=context)
        if url_from_extras[0]:
            checksum_url_source = url_from_extras

    if value is None and checksum_url_source:
        raw_url, url_algorithm = checksum_url_source
        normalized_algorithm = _normalize_algorithm(url_algorithm or algorithm or "sha256", context=context)
        value = _fetch_checksum_from_url(
            url=raw_url,
            algorithm=normalized_algorithm,
            http_config=download_config,
            logger=logger,
        )
        algorithm = normalized_algorithm

    if value is None or algorithm is None:
        return None

    normalized_algorithm, normalized_value = _normalize_checksum(algorithm, value, context=context)
    checksum_string = f"{normalized_algorithm}:{normalized_value}"
    logger.info(
        "using expected checksum",
        extra={"stage": "download", "checksum": checksum_string, "ontology_id": spec.id},
    )
    return checksum_string

Draft202012Validator.check_schema(MANIFEST_JSON_SCHEMA)
_MANIFEST_VALIDATOR = Draft202012Validator(MANIFEST_JSON_SCHEMA)


class JSONFormatter(logging.Formatter):
    """Formatter emitting JSON structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a JSON string with DocsToKG-specific fields."""

        now = datetime.now(timezone.utc)
        payload = {
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "ontology_id": getattr(record, "ontology_id", None),
            "stage": getattr(record, "stage", None),
        }
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(mask_sensitive_data(payload))


def _compress_old_log(path: Path) -> None:
    compressed_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(compressed_path, "wb") as target:
        target.write(source.read())
    path.unlink(missing_ok=True)


def _cleanup_logs(log_dir: Path, retention_days: int) -> None:
    now = datetime.now(timezone.utc)
    retention_delta = timedelta(days=retention_days)
    for file in log_dir.glob("*.jsonl"):
        mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
        if now - mtime > retention_delta:
            _compress_old_log(file)
    for file in log_dir.glob("*.jsonl.gz"):
        mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
        if now - mtime > retention_delta:
            file.unlink(missing_ok=True)


def setup_logging(
    *,
    level: str = "INFO",
    retention_days: int = 30,
    max_log_size_mb: int = 100,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Configure ontology downloader logging with rotation and JSON sidecars.

    Args:
        level: Logging level name applied to the root ontology logger.
        retention_days: Number of days to retain historical log files.
        max_log_size_mb: Threshold triggering log rotation for JSONL outputs.
        log_dir: Optional directory override for persisted logs.

    Returns:
        Configured logger instance ready for use by the pipeline.
    """
    resolved_dir = log_dir or Path(os.environ.get("ONTOFETCH_LOG_DIR", ""))
    if not resolved_dir:
        resolved_dir = LOG_DIR
    resolved_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_logs(resolved_dir, retention_days)

    logger = logging.getLogger("DocsToKG.OntologyDownload")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(logger.handlers):
        if getattr(handler, "_ontofetch_managed", False):
            logger.removeHandler(handler)
            handler.close()

    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(console_formatter)
    stream_handler._ontofetch_managed = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    file_name = sanitize_filename(f"ontofetch-{today}.jsonl")
    file_handler = RotatingFileHandler(
        resolved_dir / file_name,
        maxBytes=int(max_log_size_mb * 1024 * 1024),
        backupCount=5,
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler._ontofetch_managed = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)

    logger.propagate = True
    return logger


def get_manifest_schema() -> Dict[str, Any]:
    """Return a deep copy of the manifest JSON Schema definition.

    Args:
        None

    Returns:
        Dictionary describing the manifest JSON Schema.
    """

    return deepcopy(MANIFEST_JSON_SCHEMA)


def validate_manifest_dict(payload: Mapping[str, Any], *, source: Optional[Path] = None) -> None:
    """Validate manifest payload against the JSON Schema definition.

    Args:
        payload: Manifest dictionary loaded from JSON.
        source: Optional filesystem path for contextual error reporting.

    Returns:
        None

    Raises:
        ConfigurationError: If validation fails.
    """

    try:
        _MANIFEST_VALIDATOR.validate(payload)
    except JSONSchemaValidationError as exc:
        location = " -> ".join(str(part) for part in exc.path)
        message = exc.message
        if location:
            message = f"{location}: {message}"
        context = f" for {source}" if source else ""
        raise ConfigurationError(f"Manifest validation failed{context}: {message}") from exc


class OntologyDownloadError(RuntimeError):
    """Base exception for ontology download failures.

    Args:
        message: Description of the failure encountered.

    Examples:
        >>> raise OntologyDownloadError("unexpected error")
        Traceback (most recent call last):
        ...
        OntologyDownloadError: unexpected error
    """


class ResolverError(OntologyDownloadError):
    """Raised when resolver planning fails.

    Args:
        message: Description of the resolver failure.

    Examples:
        >>> raise ResolverError("resolver unavailable")
        Traceback (most recent call last):
        ...
        ResolverError: resolver unavailable
    """


class ValidationError(OntologyDownloadError):
    """Raised when validation encounters unrecoverable issues.

    Args:
        message: Human-readable description of the validation failure.

    Examples:
        >>> raise ValidationError("robot validator crashed")
        Traceback (most recent call last):
        ...
        ValidationError: robot validator crashed
    """


class ConfigurationError(OntologyDownloadError):
    """Raised when configuration or manifest validation fails.

    Args:
        message: Details about the configuration inconsistency.

    Examples:
        >>> raise ConfigurationError("manifest missing sha256")
        Traceback (most recent call last):
        ...
        ConfigurationError: manifest missing sha256
    """


@dataclass(slots=True)
class FetchSpec:
    """Specification describing a single ontology download.

    Attributes:
        id: Stable identifier for the ontology to fetch.
        resolver: Name of the resolver strategy used to locate resources.
        extras: Resolver-specific configuration overrides.
        target_formats: Normalized ontology formats that should be produced.

    Examples:
        >>> spec = FetchSpec(id="CHEBI", resolver="obo", extras={}, target_formats=("owl",))
        >>> spec.resolver
        'obo'
    """

    id: str
    resolver: str
    extras: Dict[str, object]
    target_formats: Sequence[str]


def _make_fetch_spec(
    raw_spec: Mapping[str, object],
    defaults: DefaultsConfig,
    *,
    allow_missing_resolvers: bool = False,
) -> FetchSpec:
    """Create a fetch specification from raw configuration and defaults."""

    if "id" not in raw_spec:
        raise ConfigError("Ontology specification missing required field 'id'")

    ontology_id = str(raw_spec["id"]).strip()
    if not ontology_id:
        raise ConfigError("Ontology ID cannot be empty")

    prefer_source = _coerce_sequence(raw_spec.get("prefer_source")) or list(defaults.prefer_source)
    normalize_to = _coerce_sequence(raw_spec.get("normalize_to")) or list(defaults.normalize_to)
    extras = raw_spec.get("extras")
    if extras is not None and not isinstance(extras, Mapping):
        raise ConfigError(f"Ontology '{ontology_id}' extras must be a mapping if provided")

    if not allow_missing_resolvers:
        missing = [resolver for resolver in prefer_source if resolver not in RESOLVERS]
        if missing:
            raise ConfigError("Unknown resolver(s) specified: " + ", ".join(sorted(set(missing))))

    resolver_override = raw_spec.get("resolver")
    if resolver_override:
        resolver_name = str(resolver_override).strip()
    elif prefer_source:
        resolver_name = prefer_source[0]
    elif defaults.prefer_source:
        resolver_name = defaults.prefer_source[0]
    else:
        resolver_name = "obo"

    target_formats = tuple(normalize_to) if normalize_to else tuple(defaults.normalize_to)

    return FetchSpec(
        id=ontology_id,
        resolver=resolver_name,
        extras=dict(extras or {}),
        target_formats=target_formats or ("ttl",),
    )


def merge_defaults(
    raw_spec: Mapping[str, object], defaults: DefaultsConfig, *, index: Optional[int] = None
) -> FetchSpec:
    """Merge user-provided specification with defaults to create a fetch spec."""

    allow_missing_resolvers = bool(raw_spec.get("allow_missing_resolvers", False))

    try:
        return _make_fetch_spec(
            raw_spec,
            defaults,
            allow_missing_resolvers=allow_missing_resolvers,
        )
    except ConfigError as exc:
        location = f"ontologies[{index}]" if index is not None else "ontologies[]"
        raise ConfigError(f"{location}: {exc}") from exc


@dataclass(slots=True)
class FetchResult:
    """Outcome of a single ontology fetch operation.

    Attributes:
        spec: Fetch specification that initiated the download.
        local_path: Path to the downloaded ontology document.
        status: Final download status (e.g., `success`, `skipped`).
        sha256: SHA-256 digest of the downloaded file.
        manifest_path: Path to the generated manifest JSON file.
        artifacts: Ancillary files produced during extraction or validation.

    Examples:
        >>> from pathlib import Path
        >>> spec = FetchSpec(id="CHEBI", resolver="obo", extras={}, target_formats=("owl",))
        >>> result = FetchResult(
        ...     spec=spec,
        ...     local_path=Path("CHEBI.owl"),
        ...     status="success",
        ...     sha256="deadbeef",
        ...     manifest_path=Path("manifest.json"),
        ...     artifacts=(),
        ... )
        >>> result.status
        'success'
    """

    spec: FetchSpec
    local_path: Path
    status: str
    sha256: str
    manifest_path: Path
    artifacts: Sequence[str]


ResolvedConfig.model_rebuild()


@dataclass(slots=True)
class Manifest:
    """Provenance information for a downloaded ontology artifact.

    Attributes:
        schema_version: Manifest schema version identifier.
        id: Ontology identifier recorded in the manifest.
        resolver: Resolver used to retrieve the ontology.
        url: Final URL from which the ontology was fetched.
        filename: Local filename of the downloaded artifact.
        version: Resolver-reported ontology version, if available.
        license: License identifier associated with the ontology.
        status: Result status reported by the downloader.
        sha256: Hash of the downloaded artifact for integrity checking.
        normalized_sha256: Hash of the canonical normalized TTL output.
        fingerprint: Composite fingerprint combining key provenance values.
        etag: HTTP ETag returned by the upstream server, when provided.
        last_modified: Upstream last-modified timestamp, if supplied.
        content_type: MIME type reported by upstream servers when available.
        content_length: Content-Length reported by upstream servers when available.
        source_media_type_label: Friendly label describing the source media type.
        downloaded_at: UTC timestamp of the completed download.
        target_formats: Desired conversion targets for normalization.
        validation: Mapping of validator names to their results.
        artifacts: Additional file paths generated during processing.
        resolver_attempts: Ordered record of resolver attempts during download.

    Examples:
        >>> manifest = Manifest(
        ...     schema_version="1.0",
        ...     id="CHEBI",
        ...     resolver="obo",
        ...     url="https://example.org/chebi.owl",
        ...     filename="chebi.owl",
        ...     version=None,
        ...     license="CC-BY",
        ...     status="success",
        ...     sha256="deadbeef",
        ...     normalized_sha256=None,
        ...     fingerprint=None,
        ...     etag=None,
        ...     last_modified=None,
        ...     content_type=None,
        ...     content_length=None,
        ...     source_media_type_label=None,
        ...     downloaded_at="2024-01-01T00:00:00Z",
        ...     target_formats=("owl",),
        ...     validation={},
        ...     artifacts=(),
        ...     resolver_attempts=(),
        ... )
        >>> manifest.resolver
        'obo'
    """

    schema_version: str
    id: str
    resolver: str
    url: str
    filename: str
    version: Optional[str]
    license: Optional[str]
    status: str
    sha256: str
    normalized_sha256: Optional[str]
    fingerprint: Optional[str]
    etag: Optional[str]
    last_modified: Optional[str]
    downloaded_at: str
    target_formats: Sequence[str]
    validation: Dict[str, ValidationResult]
    artifacts: Sequence[str]
    resolver_attempts: Sequence[Dict[str, object]]
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    source_media_type_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary for the manifest.

        Args:
            None

        Returns:
            Dictionary representing the manifest payload.
        """

        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "resolver": self.resolver,
            "url": self.url,
            "filename": self.filename,
            "version": self.version,
            "license": self.license,
            "status": self.status,
            "sha256": self.sha256,
            "normalized_sha256": self.normalized_sha256,
            "fingerprint": self.fingerprint,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "source_media_type_label": self.source_media_type_label,
            "downloaded_at": self.downloaded_at,
            "target_formats": list(self.target_formats),
            "validation": {name: result.to_dict() for name, result in self.validation.items()},
            "artifacts": list(self.artifacts),
            "resolver_attempts": [dict(entry) for entry in self.resolver_attempts],
        }

    def to_json(self) -> str:
        """Serialize the manifest to a stable, human-readable JSON string.

        Args:
            None

        Returns:
            JSON document encoding the manifest metadata.
        """

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class Resolver(Protocol):
    """Protocol describing resolver planning behaviour.

    Attributes:
        None

    Examples:
        >>> import logging
        >>> spec = FetchSpec(id="CHEBI", resolver="dummy", extras={}, target_formats=("owl",))
        >>> class DummyResolver:
        ...     def plan(self, spec, config, logger):
        ...         return FetchPlan(
        ...             url="https://example.org/chebi.owl",
        ...             headers={},
        ...             filename_hint="chebi.owl",
        ...             version="v1",
        ...             license="CC-BY",
        ...             media_type="application/rdf+xml",
        ...         )
        ...
        >>> plan = DummyResolver().plan(spec, ResolvedConfig.from_defaults(), logging.getLogger("test"))
        >>> plan.url
        'https://example.org/chebi.owl'
    """

    def plan(self, spec: FetchSpec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Return a FetchPlan describing how to obtain the ontology.

        Args:
            spec: Ontology fetch specification under consideration.
            config: Fully resolved configuration containing defaults.
            logger: Logger adapter scoped to the current fetch request.

        Returns:
            Concrete plan containing download URL, headers, and metadata.
        """
        ...


@dataclass(slots=True)
class ResolverCandidate:
    """Resolver plan captured for download-time fallback.

    Attributes:
        resolver: Name of the resolver that produced the plan.
        plan: Concrete :class:`FetchPlan` describing how to fetch the ontology.

    Examples:
        >>> candidate = ResolverCandidate(
        ...     resolver="obo",
        ...     plan=FetchPlan(
        ...         url="https://example.org/hp.owl",
        ...         headers={},
        ...         filename_hint=None,
        ...         version="2024-01-01",
        ...         license="CC-BY",
        ...         media_type="application/rdf+xml",
        ...         service="obo",
        ...     ),
        ... )
        >>> candidate.resolver
        'obo'
    """

    resolver: str
    plan: FetchPlan


@dataclass(slots=True)
class PlannedFetch:
    """Plan describing how an ontology would be fetched without side effects.

    Attributes:
        spec: Original fetch specification provided by the caller.
        resolver: Name of the resolver selected to satisfy the plan.
        plan: Concrete :class:`FetchPlan` generated by the resolver.
        candidates: Ordered list of resolver candidates available for fallback.

    Examples:
        >>> fetch_plan = PlannedFetch(
        ...     spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=("owl",)),
        ...     resolver="obo",
        ...     plan=FetchPlan(
        ...         url="https://example.org/hp.owl",
        ...         headers={},
        ...         filename_hint="hp.owl",
        ...         version="2024-01-01",
        ...         license="CC-BY-4.0",
        ...         media_type="application/rdf+xml",
        ...     ),
        ...     candidates=(
        ...         ResolverCandidate(
        ...             resolver="obo",
        ...             plan=FetchPlan(
        ...                 url="https://example.org/hp.owl",
        ...                 headers={},
        ...                 filename_hint="hp.owl",
        ...                 version="2024-01-01",
        ...                 license="CC-BY-4.0",
        ...                 media_type="application/rdf+xml",
        ...             ),
        ...         ),
        ...     ),
        ... )
        >>> fetch_plan.resolver
        'obo'
    """

    spec: FetchSpec
    resolver: str
    plan: FetchPlan
    candidates: Sequence[ResolverCandidate]
    metadata: Dict[str, object] = field(default_factory=dict)
    last_modified: Optional[str] = None
    last_modified_at: Optional[datetime] = None
    size: Optional[int] = None


def parse_http_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse HTTP ``Last-Modified`` style timestamps into UTC datetimes.

    Args:
        value: Timestamp string from HTTP headers such as ``Last-Modified``.

    Returns:
        Optional[datetime]: Normalized UTC datetime when parsing succeeds.

    Raises:
        None: Parsing failures are converted into a ``None`` return value.
    """

    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamps into timezone-aware UTC datetimes.

    Args:
        value: ISO-8601 formatted timestamp string.

    Returns:
        Optional[datetime]: Normalized UTC datetime when parsing succeeds.

    Raises:
        None: Invalid values return ``None`` instead of raising.
    """

    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_version_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse version strings or manifest timestamps into UTC datetimes.

    Args:
        value: Version identifier or timestamp string to normalize.

    Returns:
        Optional[datetime]: Parsed UTC datetime when the input matches supported formats.

    Raises:
        None: All parsing failures result in ``None``.
    """

    if not value or not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    parsed = parse_iso_datetime(text)
    if parsed is not None:
        return parsed

    candidates: List[str] = []

    def _add_candidate(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    _add_candidate(text)
    _add_candidate(text.replace("_", "-"))
    _add_candidate(text.replace("/", "-"))
    _add_candidate(text.replace("_", ""))
    _add_candidate(text.replace("-", ""))

    patterns = (
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y%m%dT%H%M%S",
        "%Y-%m-%d-%H-%M-%S",
    )

    for candidate in candidates:
        parsed = parse_iso_datetime(candidate)
        if parsed is not None:
            return parsed
        for fmt in patterns:
            try:
                naive = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            return naive.replace(tzinfo=timezone.utc)
    return None


def infer_version_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Infer a timestamp from resolver version identifiers.

    Args:
        value: Resolver version string containing date-like fragments.

    Returns:
        Optional[datetime]: Parsed UTC datetime when the value contains recoverable dates.

    Raises:
        None: Returns ``None`` instead of raising on unparseable inputs.
    """

    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    parsed = parse_version_timestamp(text)
    if parsed is not None:
        return parsed

    # Attempt to recover from composite strings like "2024-01-01-release" by
    # extracting contiguous digit blocks that resemble dates.
    matches = re.findall(r"(\d{4}[\d-]{4,})", text)
    for match in matches:
        parsed = parse_version_timestamp(match)
        if parsed is not None:
            return parsed

    digits_only = re.sub(r"\D", "", text)
    if len(digits_only) >= 8:
        parsed = parse_version_timestamp(digits_only[:14])
        if parsed is not None:
            return parsed
    parsed = parse_iso_datetime(value)
    if parsed is not None:
        return parsed
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            naive = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return naive.replace(tzinfo=timezone.utc)
    return None


def _coerce_datetime(value: Optional[str]) -> Optional[datetime]:
    """Return timezone-aware datetime parsed from HTTP or ISO timestamp."""

    parsed = parse_http_datetime(value)
    if parsed is not None:
        return parsed
    return parse_iso_datetime(value)


def _normalize_timestamp(value: Optional[str]) -> Optional[str]:
    """Return canonical ISO8601 string for HTTP timestamp headers."""

    parsed = _coerce_datetime(value)
    if parsed is None:
        return value
    return parsed.isoformat().replace("+00:00", "Z")


def _canonical_media_type(value: Optional[str]) -> Optional[str]:
    """Return a normalized MIME type without parameters."""

    if not value:
        return None
    return value.split(";", 1)[0].strip().lower() or None


DEFAULT_VALIDATOR_NAMES: Tuple[str, ...] = (
    "rdflib",
    "pronto",
    "owlready2",
    "robot",
    "arelle",
)

_MEDIA_VALIDATOR_WHITELIST = {
    "application/zip": {"arelle"},
}


def _select_validators(media_type: Optional[str]) -> List[str]:
    """Return validator names appropriate for ``media_type``."""

    canonical = _canonical_media_type(media_type)
    if canonical and canonical in _MEDIA_VALIDATOR_WHITELIST:
        allowed = set(_MEDIA_VALIDATOR_WHITELIST[canonical])
    else:
        allowed = set(DEFAULT_VALIDATOR_NAMES)
        if canonical and canonical not in RDF_MIME_ALIASES:
            allowed -= {"rdflib", "robot"}
    return [name for name in DEFAULT_VALIDATOR_NAMES if name in allowed]


def _populate_plan_metadata(
    planned: PlannedFetch,
    config: ResolvedConfig,
    adapter: logging.LoggerAdapter,
) -> PlannedFetch:
    """Augment planned fetch with HTTP metadata when available."""

    if not isinstance(planned.metadata, dict):
        planned.metadata = dict(planned.metadata)
    metadata = planned.metadata

    if planned.plan.content_length is not None and planned.size is None:
        planned.size = planned.plan.content_length
    if planned.plan.content_length is not None:
        metadata.setdefault("content_length", planned.plan.content_length)
    plan_media_type = _canonical_media_type(planned.plan.media_type)
    if plan_media_type:
        metadata.setdefault("content_type", plan_media_type)
        label = RDF_MIME_FORMAT_LABELS.get(plan_media_type)
        if label:
            metadata.setdefault("source_media_type_label", label)
    if planned.plan.last_modified and not planned.last_modified:
        normalized = _normalize_timestamp(planned.plan.last_modified)
        planned.last_modified = normalized
        planned.last_modified_at = _coerce_datetime(normalized)
        planned.plan.last_modified = normalized
    elif planned.last_modified:
        normalized = _normalize_timestamp(planned.last_modified)
        planned.last_modified = normalized
        planned.last_modified_at = _coerce_datetime(normalized)
        if normalized:
            planned.plan.last_modified = normalized
            metadata.setdefault("last_modified", normalized)

    needs_size = planned.size is None
    needs_last_modified = planned.last_modified is None
    if not (needs_size or needs_last_modified):
        return planned

    try:
        validate_url_security(planned.plan.url, config.defaults.http)
    except ConfigError as exc:
        adapter.error(
            "metadata probe blocked by URL policy",
            extra={
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "url": planned.plan.url,
                "error": str(exc),
            },
        )
        raise

    timeout = getattr(config.defaults.http, "timeout_sec", 30)
    headers = dict(planned.plan.headers or {})

    try:
        head_response = requests.head(
            planned.plan.url,
            headers=headers,
            allow_redirects=True,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        adapter.warning(
            "metadata probe failed",
            extra={
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "url": planned.plan.url,
                "error": str(exc),
            },
        )
        return planned

    headers_map = head_response.headers
    status = head_response.status_code
    ok = head_response.ok
    head_response.close()

    if status == 405:
        try:
            get_response = requests.get(
                planned.plan.url,
                headers=headers,
                allow_redirects=True,
                timeout=timeout,
                stream=True,
            )
        except requests.RequestException as exc:
            adapter.warning(
                "metadata probe failed",
                extra={
                    "stage": "plan",
                    "ontology_id": planned.spec.id,
                    "url": planned.plan.url,
                    "error": str(exc),
                },
            )
            return planned
        headers_map = get_response.headers
        ok = get_response.ok
        status = get_response.status_code
        get_response.close()

    if not ok:
        adapter.warning(
            "metadata probe rejected",
            extra={
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "url": planned.plan.url,
                "status": status,
            },
        )
        return planned

    last_modified_value = headers_map.get("Last-Modified") or headers_map.get("last-modified")
    if last_modified_value:
        normalized = _normalize_timestamp(last_modified_value)
        planned.last_modified = normalized or last_modified_value
        planned.last_modified_at = _coerce_datetime(normalized or last_modified_value)
        planned.plan.last_modified = normalized or last_modified_value
        metadata["last_modified"] = planned.plan.last_modified

    content_type_value = headers_map.get("Content-Type") or headers_map.get("content-type")
    canonical_type = _canonical_media_type(content_type_value)
    if canonical_type:
        metadata["content_type"] = canonical_type
        planned.plan.media_type = canonical_type
        label = RDF_MIME_FORMAT_LABELS.get(canonical_type)
        if label:
            metadata["source_media_type_label"] = label

    if planned.size is None:
        content_length_value = headers_map.get("Content-Length") or headers_map.get(
            "content-length"
        )
        if content_length_value:
            try:
                parsed_length = int(content_length_value)
            except ValueError:
                parsed_length = None
            if parsed_length is not None:
                planned.size = parsed_length
                planned.plan.content_length = parsed_length
                metadata["content_length"] = parsed_length

    etag = headers_map.get("ETag") or headers_map.get("etag")
    if etag:
        metadata["etag"] = etag

    return planned


def _migrate_manifest_inplace(payload: dict) -> None:
    """Upgrade manifests created with older schema versions in place."""

    version = str(payload.get("schema_version", "") or "")
    if version in {"", "1.0"}:
        payload.setdefault("schema_version", "1.0")
        return
    if version == "0.9":
        payload["schema_version"] = "1.0"
        payload.setdefault("resolver_attempts", [])
        return
    logging.getLogger(__name__).warning(
        "unknown manifest schema version",
        extra={"stage": "manifest", "schema_version": version},
    )


def _read_manifest(manifest_path: Path) -> Optional[dict]:
    """Return previously recorded manifest data if a valid JSON file exists.

    Args:
        manifest_path: Filesystem path where the manifest is stored.

    Returns:
        Parsed manifest dictionary when available and valid, otherwise ``None``.
    """
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    _migrate_manifest_inplace(payload)
    net_module.validate_manifest_dict(payload, source=manifest_path)
    return payload


def _validate_manifest(manifest: Manifest) -> None:
    """Check that a manifest instance satisfies structural and type requirements.

    Args:
        manifest: Manifest produced after a download completes.

    Raises:
        ConfigurationError: If required fields are missing or contain invalid types.
    """
    net_module.validate_manifest_dict(manifest.to_dict())

    required_fields = [
        "id",
        "resolver",
        "url",
        "filename",
        "status",
        "sha256",
        "downloaded_at",
    ]
    for field_name in required_fields:
        value = getattr(manifest, field_name)
        if value in {None, ""}:
            raise ConfigurationError(f"Manifest field '{field_name}' must be populated")
    if not manifest.url.startswith(("https://", "http://")):
        raise ConfigurationError("Manifest URL must use http or https scheme")
    if not isinstance(manifest.schema_version, str):
        raise ConfigurationError("Manifest schema_version must be a string")
    if not isinstance(manifest.validation, dict):
        raise ConfigurationError("Manifest validation payload must be a dictionary")
    if not isinstance(manifest.artifacts, Sequence):
        raise ConfigurationError("Manifest artifacts must be a sequence of paths")
    for item in manifest.artifacts:
        if not isinstance(item, str):
            raise ConfigurationError("Manifest artifacts must contain only string paths")
    if not isinstance(manifest.resolver_attempts, Sequence):
        raise ConfigurationError("Manifest resolver_attempts must be a sequence")
    for entry in manifest.resolver_attempts:
        if not isinstance(entry, dict):
            raise ConfigurationError("Manifest resolver_attempts must contain dictionaries")
    if manifest.normalized_sha256 is not None and not isinstance(manifest.normalized_sha256, str):
        raise ConfigurationError("Manifest normalized_sha256 must be a string when provided")
    if manifest.fingerprint is not None and not isinstance(manifest.fingerprint, str):
        raise ConfigurationError("Manifest fingerprint must be a string when provided")


def _parse_last_modified(value: Optional[str]) -> Optional[datetime]:
    """Return a timezone-aware datetime parsed from HTTP date headers."""

    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fetch_last_modified(
    plan: FetchPlan, config: ResolvedConfig, logger: logging.Logger
) -> Optional[str]:
    """Probe the upstream plan URL for a Last-Modified header."""

    timeout = max(1, getattr(config.defaults.http, "timeout_sec", 30) or 30)
    headers = dict(plan.headers or {})
    try:
        response = requests.head(
            plan.url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        if response.status_code == 405:
            response.close()
            response = requests.get(
                plan.url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )
        header = response.headers.get("Last-Modified")
        response.close()
        return header
    except requests.RequestException as exc:  # pragma: no cover - depends on network
        logger.warning(
            "last-modified probe failed",
            extra={"stage": "plan", "resolver": plan.service or plan.url, "error": str(exc)},
        )
        return None


def _write_manifest(manifest_path: Path, manifest: Manifest) -> None:
    """Persist a validated manifest to disk as JSON.

    Args:
        manifest_path: Destination path for the manifest file.
        manifest: Manifest describing the downloaded ontology artifact.
    """
    _validate_manifest(manifest)
    manifest_path.write_text(manifest.to_json())


def _append_index_entry(ontology_dir: Path, entry: Dict[str, Any]) -> None:
    """Append or update the ontology-level ``index.json`` with ``entry``."""

    index_path = ontology_dir / "index.json"
    try:
        existing = json.loads(index_path.read_text())
        if not isinstance(existing, list):
            existing = []
    except FileNotFoundError:
        existing = []
    except json.JSONDecodeError:
        existing = []

    filtered: List[Dict[str, Any]] = []
    for item in existing:
        if not isinstance(item, dict):
            continue
        same_version = item.get("version") == entry.get("version")
        same_hash = item.get("sha256") == entry.get("sha256")
        if same_version and same_hash:
            continue
        filtered.append(item)

    filtered.insert(0, entry)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(filtered, indent=2, sort_keys=True))


def _build_destination(
    spec: FetchSpec, plan: FetchPlan, config: ResolvedConfig
) -> Tuple[Path, str, Path]:
    """Determine the output directory and filename for a download.

    Args:
        spec: Fetch specification identifying the ontology.
        plan: Resolver plan containing URL metadata and optional hints.
        config: Resolved configuration with storage layout parameters.

    Returns:
        Tuple containing the target file path, resolved version, and base directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = plan.version or timestamp
    base_dir = LOCAL_ONTOLOGY_DIR / sanitize_filename(spec.id) / sanitize_filename(version)
    for subdir in ("original", "normalized", "validation"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    parsed = urlparse(plan.url)
    candidate = Path(parsed.path).name if parsed.path else f"{spec.id}.owl"
    filename = plan.filename_hint or sanitize_filename(candidate)
    destination = base_dir / "original" / filename
    return destination, version, base_dir


def _ensure_license_allowed(plan: FetchPlan, config: ResolvedConfig, spec: FetchSpec) -> None:
    """Confirm the ontology license is present in the configured allow list.

    Args:
        plan: Resolver plan returned for the ontology.
        config: Resolved configuration containing accepted licenses.
        spec: Fetch specification for contextual error reporting.

    Raises:
        ConfigurationError: If the plan's license is not permitted.
    """
    allowed = {
        normalize_license_to_spdx(entry) or entry for entry in config.defaults.accept_licenses
    }
    plan_license = normalize_license_to_spdx(plan.license)
    if not allowed or plan_license is None:
        return
    if plan_license not in allowed:
        raise ConfigurationError(
            f"License '{plan.license}' for ontology '{spec.id}' is not in the allowlist: {sorted(allowed)}"
        )


def _resolver_candidates(spec: FetchSpec, config: ResolvedConfig) -> List[str]:
    candidates: List[str] = []
    seen = set()

    def _add(name: Optional[str]) -> None:
        if not name or name in seen:
            return
        candidates.append(name)
        seen.add(name)

    _add(spec.resolver)
    if config.defaults.resolver_fallback_enabled:
        for name in config.defaults.prefer_source:
            _add(name)
    return candidates


def _resolve_plan_with_fallback(
    spec: FetchSpec, config: ResolvedConfig, adapter: logging.LoggerAdapter
) -> Tuple[ResolverCandidate, Sequence[ResolverCandidate]]:
    attempts: List[str] = []
    candidates: List[ResolverCandidate] = []
    primary: Optional[ResolverCandidate] = None
    for attempt_number, resolver_name in enumerate(_resolver_candidates(spec, config), start=1):
        resolver = RESOLVERS.get(resolver_name)
        if resolver is None:
            message = "resolver not registered"
            attempts.append(f"{resolver_name}: {message}")
            adapter.warning(
                "resolver missing",
                extra={
                    "stage": "plan",
                    "resolver": resolver_name,
                    "attempt": attempt_number,
                    "error": message,
                },
            )
            continue
        adapter.info(
            "resolver attempt",
            extra={
                "stage": "plan",
                "resolver": resolver_name,
                "attempt": attempt_number,
            },
        )
        try:
            plan = resolver.plan(spec, config, adapter)
        except (ConfigError, DownloadFailure) as exc:
            message = str(exc)
            attempts.append(f"{resolver_name}: {message}")
            adapter.warning(
                "resolver failed",
                extra={
                    "stage": "plan",
                    "resolver": resolver_name,
                    "attempt": attempt_number,
                    "error": message,
                },
            )
            continue
        except Exception as exc:  # pylint: disable=broad-except
            message = str(exc)
            attempts.append(f"{resolver_name}: {message}")
            adapter.warning(
                "resolver failed",
                extra={
                    "stage": "plan",
                    "resolver": resolver_name,
                    "attempt": attempt_number,
                    "error": message,
                },
            )
            continue

        candidate = ResolverCandidate(resolver=resolver_name, plan=plan)
        candidates.append(candidate)
        if primary is None:
            primary = candidate
            if resolver_name != spec.resolver:
                adapter.info(
                    "resolver fallback success",
                    extra={
                        "stage": "plan",
                        "resolver": resolver_name,
                        "attempt": attempt_number,
                    },
                )
        else:
            adapter.info(
                "resolver fallback candidate",
                extra={
                    "stage": "plan",
                    "resolver": resolver_name,
                    "attempt": attempt_number,
                },
            )
    if primary is None:
        details = "; ".join(attempts) if attempts else "no resolvers attempted"
        raise ResolverError(f"All resolvers exhausted for ontology '{spec.id}': {details}")
    return primary, candidates


def fetch_one(
    spec: FetchSpec,
    *,
    config: Optional[ResolvedConfig] = None,
    correlation_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> FetchResult:
    """Fetch, validate, and persist a single ontology described by *spec*.

    Args:
        spec: Ontology fetch specification describing sources and formats.
        config: Optional resolved configuration overriding global defaults.
        correlation_id: Correlation identifier for structured logging.
        logger: Optional logger to reuse instead of configuring a new one.
        force: When ``True``, bypass local cache checks and redownload artifacts.

    Returns:
        FetchResult: Structured result containing manifest metadata and resolver attempts.

    Raises:
        ResolverError: If all resolver candidates fail to retrieve the ontology.
    """

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    logging_config = active_config.defaults.logging
    log = logger or setup_logging(
        level=logging_config.level,
        retention_days=logging_config.retention_days,
        max_log_size_mb=logging_config.max_log_size_mb,
    )
    correlation = correlation_id or generate_correlation_id()
    adapter = logging.LoggerAdapter(
        log, extra={"correlation_id": correlation, "ontology_id": spec.id}
    )
    adapter.info("planning fetch", extra={"stage": "plan"})

    primary, candidates = _resolve_plan_with_fallback(spec, active_config, adapter)
    download_config = active_config.defaults.http
    candidate_list = list(candidates) or [primary]

    resolver_attempts: List[Dict[str, object]] = []
    last_error: Optional[Exception] = None

    for attempt_number, candidate in enumerate(candidate_list, start=1):
        attempt_record: Dict[str, object] = {
            "resolver": candidate.resolver,
            "url": candidate.plan.url,
            "attempt": attempt_number,
        }

        pending_spec = FetchSpec(
            id=spec.id,
            resolver=candidate.resolver,
            extras=spec.extras,
            target_formats=spec.target_formats,
        )
        try:
            _ensure_license_allowed(candidate.plan, active_config, pending_spec)
        except ConfigurationError as exc:
            adapter.warning(
                "resolver license rejected",
                extra={
                    "stage": "plan",
                    "resolver": candidate.resolver,
                    "attempt": attempt_number,
                    "error": str(exc),
                },
            )
            attempt_record.update({"status": "rejected", "error": str(exc)})
            resolver_attempts.append(dict(attempt_record))
            last_error = exc
            continue

        if candidate.plan.service:
            adapter.extra["service"] = candidate.plan.service
        else:
            adapter.extra.pop("service", None)

        def _execute_candidate() -> FetchResult:
            pending_destination, pending_version, pending_base_dir = _build_destination(
                pending_spec, candidate.plan, active_config
            )
            pending_manifest_path = pending_base_dir / "manifest.json"

            with _version_lock(pending_spec.id, pending_version):
                previous_manifest = None
                if not force:
                    STORAGE.ensure_local_version(pending_spec.id, pending_version)
                    previous_manifest = _read_manifest(pending_manifest_path)

                adapter.info(
                    "downloading",
                    extra={
                        "stage": "download",
                        "url": candidate.plan.url,
                        "destination": str(pending_destination),
                        "version": pending_version,
                        "resolver": candidate.resolver,
                        "attempt": attempt_number,
                    },
                )

                pending_secure_url = validate_url_security(candidate.plan.url, download_config)
                expected_hash = _resolve_expected_checksum(
                    spec=pending_spec,
                    plan=candidate.plan,
                    download_config=download_config,
                    logger=adapter,
                )
                result = download_stream(
                    url=pending_secure_url,
                    destination=pending_destination,
                    headers=candidate.plan.headers,
                    previous_manifest=previous_manifest,
                    http_config=download_config,
                    cache_dir=CACHE_DIR,
                    logger=adapter,
                    expected_media_type=candidate.plan.media_type,
                    service=candidate.plan.service,
                    expected_hash=expected_hash,
                )

                if expected_hash:
                    attempt_record["expected_checksum"] = expected_hash

                effective_spec = pending_spec
                destination = pending_destination
                version = pending_version
                base_dir = pending_base_dir
                manifest_path = pending_manifest_path
                secure_url = pending_secure_url
                plan = candidate.plan

                normalized_dir = base_dir / "normalized"
                validation_dir = base_dir / "validation"
                validator_names = _select_validators(plan.media_type)
                validation_requests = [
                    ValidationRequest(
                        name=validator,
                        file_path=destination,
                        normalized_dir=normalized_dir,
                        validation_dir=validation_dir,
                        config=active_config,
                    )
                    for validator in validator_names
                ]

                canonical_media = _canonical_media_type(plan.media_type)
                if (
                    canonical_media
                    and canonical_media not in RDF_MIME_ALIASES
                    and {"rdflib", "robot"} - set(validator_names)
                ):
                    adapter.info(
                        "skipping rdf validators",
                        extra={
                            "stage": "validate",
                            "media_type": canonical_media,
                            "validator": "rdf",
                        },
                    )

                artifacts = [str(destination)]
                if plan.media_type == "application/zip" or destination.suffix.lower() == ".zip":
                    extraction_dir = destination.parent / f"{destination.stem}_extracted"
                    try:
                        extracted_paths = extract_archive_safe(
                            destination, extraction_dir, logger=adapter
                        )
                        artifacts.extend(str(path) for path in extracted_paths)
                    except ConfigError as exc:
                        adapter.error(
                            "zip extraction failed",
                            extra={"stage": "extract", "error": str(exc)},
                        )
                        if not active_config.defaults.continue_on_error:
                            raise OntologyDownloadError(
                                f"Extraction failed for '{effective_spec.id}': {exc}"
                            ) from exc

                validation_results = run_validators(validation_requests, adapter)

                normalized_hash = None
                normalization_mode = "none"
                rdflib_result = validation_results.get("rdflib")
                if rdflib_result and isinstance(rdflib_result.details, dict):
                    maybe_hash = rdflib_result.details.get("normalized_sha256")
                    if isinstance(maybe_hash, str):
                        normalized_hash = maybe_hash
                    maybe_mode = rdflib_result.details.get("normalization_mode")
                    if isinstance(maybe_mode, str):
                        normalization_mode = maybe_mode

                target_formats_sorted = ",".join(sorted(effective_spec.target_formats))

                fingerprint_components = [
                    MANIFEST_SCHEMA_VERSION,
                    effective_spec.id,
                    effective_spec.resolver,
                    version,
                    result.sha256,
                    normalized_hash or "",
                    secure_url,
                    target_formats_sorted,
                    normalization_mode,
                ]
                fingerprint = hashlib.sha256(
                    "|".join(fingerprint_components).encode("utf-8")
                ).hexdigest()

                attempt_record["status"] = "success"
                resolver_attempts.append(dict(attempt_record))

                content_type = _canonical_media_type(result.content_type) or (
                    _canonical_media_type(plan.media_type)
                )
                content_length = result.content_length or plan.content_length
                label_key = _canonical_media_type(result.content_type) or canonical_media
                source_media_label = RDF_MIME_FORMAT_LABELS.get(label_key) if label_key else None

                manifest = Manifest(
                    schema_version=MANIFEST_SCHEMA_VERSION,
                    id=effective_spec.id,
                    resolver=effective_spec.resolver,
                    url=secure_url,
                    filename=destination.name,
                    version=version,
                    license=plan.license,
                    status=result.status,
                    sha256=result.sha256,
                    normalized_sha256=normalized_hash,
                    fingerprint=fingerprint,
                    etag=result.etag,
                    last_modified=result.last_modified,
                    content_type=content_type,
                    content_length=content_length,
                    source_media_type_label=source_media_label,
                    downloaded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    target_formats=effective_spec.target_formats,
                    validation=validation_results,
                    artifacts=artifacts,
                    resolver_attempts=resolver_attempts,
                )
                _write_manifest(manifest_path, manifest)
                index_entry = {
                    "version": manifest.version,
                    "downloaded_at": manifest.downloaded_at,
                    "sha256": manifest.sha256,
                    "normalized_sha256": manifest.normalized_sha256,
                    "etag": manifest.etag,
                    "size": destination.stat().st_size if destination.exists() else None,
                    "source_url": manifest.url,
                    "content_type": manifest.content_type,
                    "content_length": manifest.content_length,
                    "status": manifest.status,
                }
                if expected_hash:
                    index_entry["expected_checksum"] = expected_hash
                _append_index_entry(base_dir.parent, index_entry)
                STORAGE.finalize_version(effective_spec.id, version, base_dir)

                adapter.info(
                    "fetch complete",
                    extra={
                        "stage": "complete",
                        "status": result.status,
                        "sha256": result.sha256,
                        "manifest": str(manifest_path),
                    },
                )

                return FetchResult(
                    spec=effective_spec,
                    local_path=destination,
                    status=result.status,
                    sha256=result.sha256,
                    manifest_path=manifest_path,
                    artifacts=artifacts,
                )

        try:
            return _execute_candidate()
        except (ConfigError, DownloadFailure) as exc:
            attempt_record.update({"status": "failed", "error": str(exc)})
            resolver_attempts.append(dict(attempt_record))
            adapter.warning(
                "download attempt failed",
                extra={
                    "stage": "download",
                    "resolver": candidate.resolver,
                    "attempt": attempt_number,
                    "error": str(exc),
                },
            )
            last_error = exc
            retryable = getattr(exc, "retryable", False)
            if retryable:
                adapter.info(
                    "trying fallback resolver",
                    extra={
                        "stage": "download",
                        "resolver": candidate.resolver,
                        "attempt": attempt_number,
                    },
                )
                continue
            raise OntologyDownloadError(f"Download failed for '{pending_spec.id}': {exc}") from exc
        except Exception as exc:
            last_error = exc
            attempt_record.update({"status": "error", "error": str(exc)})
            resolver_attempts.append(dict(attempt_record))
            raise

    if last_error is None:
        raise OntologyDownloadError(f"All resolver candidates failed for '{spec.id}'")
    if isinstance(last_error, ConfigurationError):
        raise last_error
    raise OntologyDownloadError(f"Download failed for '{spec.id}': {last_error}") from last_error


def plan_one(
    spec: FetchSpec,
    *,
    config: Optional[ResolvedConfig] = None,
    correlation_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> PlannedFetch:
    """Return a resolver plan for a single ontology without performing downloads.

    Args:
        spec: Fetch specification describing the ontology to plan.
        config: Optional resolved configuration providing defaults and limits.
        correlation_id: Correlation identifier reused for logging context.
        logger: Logger instance used to emit resolver telemetry.

    Returns:
        PlannedFetch containing the normalized spec, resolver name, and plan.

    Raises:
        ResolverError: If all resolvers fail to produce a plan for ``spec``.
        ConfigurationError: If licence checks reject the planned ontology.
    """

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    logging_config = active_config.defaults.logging
    log = logger or setup_logging(
        level=logging_config.level,
        retention_days=logging_config.retention_days,
        max_log_size_mb=logging_config.max_log_size_mb,
    )
    correlation = correlation_id or generate_correlation_id()
    adapter = logging.LoggerAdapter(
        log, extra={"correlation_id": correlation, "ontology_id": spec.id}
    )
    adapter.info("planning fetch", extra={"stage": "plan"})

    primary, candidates = _resolve_plan_with_fallback(spec, active_config, adapter)
    effective_spec = FetchSpec(
        id=spec.id,
        resolver=primary.resolver,
        extras=spec.extras,
        target_formats=spec.target_formats,
    )
    _ensure_license_allowed(primary.plan, active_config, effective_spec)
    planned = PlannedFetch(
        spec=effective_spec,
        resolver=primary.resolver,
        plan=primary.plan,
        candidates=tuple(candidates),
        last_modified=primary.plan.last_modified,
        size=primary.plan.content_length,
    )
    return _populate_plan_metadata(planned, active_config, adapter)


def plan_all(
    specs: Iterable[FetchSpec],
    *,
    config: Optional[ResolvedConfig] = None,
    logger: Optional[logging.Logger] = None,
    since: Optional[datetime] = None,
) -> List[PlannedFetch]:
    """Return resolver plans for a collection of ontologies.

    Args:
        specs: Iterable of fetch specifications to resolve.
        config: Optional resolved configuration reused across plans.
        logger: Logger instance used for annotation-aware logging.
        since: Optional cutoff date; plans older than this timestamp are filtered out.

    Returns:
        List of PlannedFetch entries describing each ontology plan.

    Raises:
        ResolverError: Propagated when fallback planning fails for any spec.
        ConfigurationError: When licence enforcement rejects a planned ontology.
    """

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    logging_config = active_config.defaults.logging
    log = logger or setup_logging(
        level=logging_config.level,
        retention_days=logging_config.retention_days,
        max_log_size_mb=logging_config.max_log_size_mb,
    )
    correlation = generate_correlation_id()
    adapter = logging.LoggerAdapter(log, extra={"correlation_id": correlation})

    spec_list = list(specs)
    if not spec_list:
        return []

    max_workers = max(1, active_config.defaults.http.concurrent_plans)
    adapter.info(
        "planning batch",
        extra={
            "stage": "plan",
            "progress": {"total": len(spec_list)},
            "workers": max_workers,
        },
    )

    results: Dict[int, PlannedFetch] = {}
    futures: Dict[object, tuple[int, FetchSpec]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, spec in enumerate(spec_list):
            future = executor.submit(
                plan_one,
                spec,
                config=active_config,
                correlation_id=correlation,
                logger=log,
            )
            futures[future] = (index, spec)

        for future in as_completed(futures):
            index, spec = futures[future]
            try:
                planned = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                adapter.error(
                    "planning failed",
                    extra={
                        "stage": "plan",
                        "ontology_id": spec.id,
                        "error": str(exc),
                    },
                )
                if isinstance(exc, (ConfigError, ConfigurationError)):
                    for pending in futures:
                        pending.cancel()
                    raise
                if not active_config.defaults.continue_on_error:
                    for pending in futures:
                        pending.cancel()
                    raise
            else:
                results[index] = planned

    ordered_indices = sorted(results)
    ordered_plans = [results[i] for i in ordered_indices]

    if since is None:
        return ordered_plans

    filtered: List[PlannedFetch] = []
    for plan in ordered_plans:
        last_modified = plan.last_modified_at or _coerce_datetime(plan.last_modified)
        if last_modified is None:
            header = _fetch_last_modified(plan.plan, active_config, log)
            if header:
                plan.plan.last_modified = header
                plan.last_modified = header
                plan.last_modified_at = _coerce_datetime(header)
                last_modified = plan.last_modified_at
        if last_modified and last_modified < since:
            adapter.info(
                "plan filtered by since",
                extra={
                    "stage": "plan",
                    "ontology_id": plan.spec.id,
                    "last_modified": plan.last_modified,
                    "since": since.isoformat().replace("+00:00", "Z"),
                },
            )
            continue
        filtered.append(plan)
    return filtered


def fetch_all(
    specs: Iterable[FetchSpec],
    *,
    config: Optional[ResolvedConfig] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> List[FetchResult]:
    """Fetch a sequence of ontologies sequentially.

    Args:
        specs: Iterable of fetch specifications to process.
        config: Optional resolved configuration shared across downloads.
        logger: Logger used to emit progress and error events.
        force: When True, skip manifest reuse and download everything again.

    Returns:
        List of FetchResult entries corresponding to completed downloads.

    Raises:
        OntologyDownloadError: Propagated when downloads fail and the pipeline
            is configured to stop on error.
    """

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    if logger is not None:
        log = logger
    else:
        candidate = logging.getLogger("DocsToKG.OntologyDownload")
        if candidate.handlers:
            log = candidate
        else:
            logging_config = active_config.defaults.logging
            log = setup_logging(
                level=logging_config.level,
                retention_days=logging_config.retention_days,
                max_log_size_mb=logging_config.max_log_size_mb,
            )
    correlation = generate_correlation_id()
    adapter = logging.LoggerAdapter(log, extra={"correlation_id": correlation})

    spec_list = list(specs)
    total = len(spec_list)
    if not spec_list:
        return []

    max_workers = max(1, active_config.defaults.http.concurrent_downloads)
    adapter.info(
        "starting batch",
        extra={"stage": "batch", "progress": {"total": total}, "workers": max_workers},
    )

    results_map: Dict[int, FetchResult] = {}
    futures: Dict[object, tuple[int, FetchSpec]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, spec in enumerate(spec_list, start=1):
            adapter.info(
                "starting ontology fetch",
                extra={
                    "stage": "start",
                    "ontology_id": spec.id,
                    "progress": {"current": index, "total": total},
                },
            )
            future = executor.submit(
                fetch_one,
                spec,
                config=active_config,
                correlation_id=correlation,
                logger=log,
                force=force,
            )
            futures[future] = (index, spec)

        for future in as_completed(futures):
            index, spec = futures[future]
            try:
                result = future.result()
                results_map[index] = result
                adapter.info(
                    "progress update",
                    extra={
                        "stage": "progress",
                        "ontology_id": spec.id,
                        "progress": {"current": len(results_map), "total": total},
                    },
                )
            except Exception as exc:  # pylint: disable=broad-except
                adapter.error(
                    "ontology fetch failed",
                    extra={"stage": "error", "ontology_id": spec.id, "error": str(exc)},
                )
                if not active_config.defaults.continue_on_error:
                    for pending in futures:
                        pending.cancel()
                    raise

    ordered_results = [results_map[i] for i in sorted(results_map)]
    return ordered_results


# --- Globals ---

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_JSON_SCHEMA",
    "FetchSpec",
    "FetchResult",
    "Manifest",
    "ResolverCandidate",
    "PlannedFetch",
    "OntologyDownloadError",
    "ResolverError",
    "ValidationError",
    "ConfigurationError",
    "merge_defaults",
    "fetch_one",
    "fetch_all",
    "plan_one",
    "plan_all",
    "get_manifest_schema",
    "validate_manifest_dict",
]


def _safe_lock_component(value: str) -> str:
    """Return a filesystem-safe token for lock filenames."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    sanitized = sanitized.strip("._") or "lock"
    return sanitized


@contextmanager
def _version_lock(ontology_id: str, version: str) -> Iterator[None]:
    """Acquire an inter-process lock for a specific ontology version."""

    lock_dir = CACHE_DIR / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = (
        lock_dir / f"{_safe_lock_component(ontology_id)}__{_safe_lock_component(version)}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()

        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:  # pragma: no cover - fallback when no locking backend available
            yield
            return

        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
