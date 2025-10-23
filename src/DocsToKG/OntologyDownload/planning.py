# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.planning",
#   "purpose": "Plan ontology downloads, coordinate execution, and emit manifests and lockfiles",
#   "sections": [
#     {
#       "id": "log-with-extra",
#       "name": "_log_with_extra",
#       "anchor": "function-log-with-extra",
#       "kind": "function"
#     },
#     {
#       "id": "get-manifest-schema",
#       "name": "get_manifest_schema",
#       "anchor": "function-get-manifest-schema",
#       "kind": "function"
#     },
#     {
#       "id": "validate-manifest-dict",
#       "name": "validate_manifest_dict",
#       "anchor": "function-validate-manifest-dict",
#       "kind": "function"
#     },
#     {
#       "id": "fetchspec",
#       "name": "FetchSpec",
#       "anchor": "class-fetchspec",
#       "kind": "class"
#     },
#     {
#       "id": "make-fetch-spec",
#       "name": "_make_fetch_spec",
#       "anchor": "function-make-fetch-spec",
#       "kind": "function"
#     },
#     {
#       "id": "merge-defaults",
#       "name": "merge_defaults",
#       "anchor": "function-merge-defaults",
#       "kind": "function"
#     },
#     {
#       "id": "fetchresult",
#       "name": "FetchResult",
#       "anchor": "class-fetchresult",
#       "kind": "class"
#     },
#     {
#       "id": "batchplanningerror",
#       "name": "BatchPlanningError",
#       "anchor": "class-batchplanningerror",
#       "kind": "class"
#     },
#     {
#       "id": "batchfetcherror",
#       "name": "BatchFetchError",
#       "anchor": "class-batchfetcherror",
#       "kind": "class"
#     },
#     {
#       "id": "cancel-pending-futures",
#       "name": "_cancel_pending_futures",
#       "anchor": "function-cancel-pending-futures",
#       "kind": "function"
#     },
#     {
#       "id": "executor-is-shutting-down",
#       "name": "_executor_is_shutting_down",
#       "anchor": "function-executor-is-shutting-down",
#       "kind": "function"
#     },
#     {
#       "id": "shutdown-executor-nowait",
#       "name": "_shutdown_executor_nowait",
#       "anchor": "function-shutdown-executor-nowait",
#       "kind": "function"
#     },
#     {
#       "id": "manifest",
#       "name": "Manifest",
#       "anchor": "class-manifest",
#       "kind": "class"
#     },
#     {
#       "id": "plannedfetch",
#       "name": "PlannedFetch",
#       "anchor": "class-plannedfetch",
#       "kind": "class"
#     },
#     {
#       "id": "parse-http-datetime",
#       "name": "parse_http_datetime",
#       "anchor": "function-parse-http-datetime",
#       "kind": "function"
#     },
#     {
#       "id": "parse-iso-datetime",
#       "name": "parse_iso_datetime",
#       "anchor": "function-parse-iso-datetime",
#       "kind": "function"
#     },
#     {
#       "id": "parse-version-timestamp",
#       "name": "parse_version_timestamp",
#       "anchor": "function-parse-version-timestamp",
#       "kind": "function"
#     },
#     {
#       "id": "infer-version-timestamp",
#       "name": "infer_version_timestamp",
#       "anchor": "function-infer-version-timestamp",
#       "kind": "function"
#     },
#     {
#       "id": "coerce-datetime",
#       "name": "_coerce_datetime",
#       "anchor": "function-coerce-datetime",
#       "kind": "function"
#     },
#     {
#       "id": "normalize-timestamp",
#       "name": "_normalize_timestamp",
#       "anchor": "function-normalize-timestamp",
#       "kind": "function"
#     },
#     {
#       "id": "canonical-media-type",
#       "name": "_canonical_media_type",
#       "anchor": "function-canonical-media-type",
#       "kind": "function"
#     },
#     {
#       "id": "select-validators",
#       "name": "_select_validators",
#       "anchor": "function-select-validators",
#       "kind": "function"
#     },
#     {
#       "id": "plannerproberesult",
#       "name": "PlannerProbeResult",
#       "anchor": "class-plannerproberesult",
#       "kind": "class"
#     },
#     {
#       "id": "planner-http-probe",
#       "name": "planner_http_probe",
#       "anchor": "function-planner-http-probe",
#       "kind": "function"
#     },
#     {
#       "id": "populate-plan-metadata",
#       "name": "_populate_plan_metadata",
#       "anchor": "function-populate-plan-metadata",
#       "kind": "function"
#     },
#     {
#       "id": "read-manifest",
#       "name": "_read_manifest",
#       "anchor": "function-read-manifest",
#       "kind": "function"
#     },
#     {
#       "id": "validate-manifest",
#       "name": "_validate_manifest",
#       "anchor": "function-validate-manifest",
#       "kind": "function"
#     },
#     {
#       "id": "parse-last-modified",
#       "name": "_parse_last_modified",
#       "anchor": "function-parse-last-modified",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-last-modified",
#       "name": "_fetch_last_modified",
#       "anchor": "function-fetch-last-modified",
#       "kind": "function"
#     },
#     {
#       "id": "atomic-write-text",
#       "name": "_atomic_write_text",
#       "anchor": "function-atomic-write-text",
#       "kind": "function"
#     },
#     {
#       "id": "atomic-write-json",
#       "name": "_atomic_write_json",
#       "anchor": "function-atomic-write-json",
#       "kind": "function"
#     },
#     {
#       "id": "write-manifest",
#       "name": "_write_manifest",
#       "anchor": "function-write-manifest",
#       "kind": "function"
#     },
#     {
#       "id": "mirror-to-cas-if-enabled",
#       "name": "_mirror_to_cas_if_enabled",
#       "anchor": "function-mirror-to-cas-if-enabled",
#       "kind": "function"
#     },
#     {
#       "id": "cleanup-failed-validation-artifacts",
#       "name": "_cleanup_failed_validation_artifacts",
#       "anchor": "function-cleanup-failed-validation-artifacts",
#       "kind": "function"
#     },
#     {
#       "id": "build-destination",
#       "name": "_build_destination",
#       "anchor": "function-build-destination",
#       "kind": "function"
#     },
#     {
#       "id": "ensure-license-allowed",
#       "name": "_ensure_license_allowed",
#       "anchor": "function-ensure-license-allowed",
#       "kind": "function"
#     },
#     {
#       "id": "resolver-candidates",
#       "name": "_resolver_candidates",
#       "anchor": "function-resolver-candidates",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-plan-with-fallback",
#       "name": "_resolve_plan_with_fallback",
#       "anchor": "function-resolve-plan-with-fallback",
#       "kind": "function"
#     },
#     {
#       "id": "get-duckdb-conn",
#       "name": "_get_duckdb_conn",
#       "anchor": "function-get-duckdb-conn",
#       "kind": "function"
#     },
#     {
#       "id": "safe-record-boundary",
#       "name": "_safe_record_boundary",
#       "anchor": "function-safe-record-boundary",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-one",
#       "name": "fetch_one",
#       "anchor": "function-fetch-one",
#       "kind": "function"
#     },
#     {
#       "id": "planned-fetch-to-dict",
#       "name": "_planned_fetch_to_dict",
#       "anchor": "function-planned-fetch-to-dict",
#       "kind": "function"
#     },
#     {
#       "id": "dict-to-planned-fetch",
#       "name": "_dict_to_planned_fetch",
#       "anchor": "function-dict-to-planned-fetch",
#       "kind": "function"
#     },
#     {
#       "id": "get-cached-plan",
#       "name": "_get_cached_plan",
#       "anchor": "function-get-cached-plan",
#       "kind": "function"
#     },
#     {
#       "id": "save-plan-to-db",
#       "name": "_save_plan_to_db",
#       "anchor": "function-save-plan-to-db",
#       "kind": "function"
#     },
#     {
#       "id": "compare-plans",
#       "name": "_compare_plans",
#       "anchor": "function-compare-plans",
#       "kind": "function"
#     },
#     {
#       "id": "save-plan-diff-to-db",
#       "name": "_save_plan_diff_to_db",
#       "anchor": "function-save-plan-diff-to-db",
#       "kind": "function"
#     },
#     {
#       "id": "plan-one",
#       "name": "plan_one",
#       "anchor": "function-plan-one",
#       "kind": "function"
#     },
#     {
#       "id": "plan-all",
#       "name": "plan_all",
#       "anchor": "function-plan-all",
#       "kind": "function"
#     },
#     {
#       "id": "fetch-all",
#       "name": "fetch_all",
#       "anchor": "function-fetch-all",
#       "kind": "function"
#     },
#     {
#       "id": "safe-lock-component",
#       "name": "_safe_lock_component",
#       "anchor": "function-safe-lock-component",
#       "kind": "function"
#     },
#     {
#       "id": "version-lock",
#       "name": "_version_lock",
#       "anchor": "function-version-lock",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Download planning, execution orchestration, and manifest production.

The planner converts configuration and resolver metadata into concrete fetch
plans, executes parallel downloads with retry/backoff controls, and records the
outcomes into manifests.  It enforces security policies (checksum verification,
allowed hosts, archive sanity checks), cooperates with the storage backend for
CAS mirroring, honours the pyrate-limiter façade (including Retry-After delays),
and records streaming hash fingerprints alongside traditional provenance fields.
Once artefacts land on disk the planner coordinates validator execution and
writes lockfiles so CLI and API callers can replay deterministic runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:  # pragma: no cover - platform specific availability
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - windows
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no cover - platform specific availability
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-windows
    msvcrt = None  # type: ignore[assignment]

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from .cancellation import CancellationToken, CancellationTokenGroup
from .checksums import ExpectedChecksum, resolve_expected_checksum
from .errors import (
    ConfigurationError,
    DownloadFailure,
    OntologyDownloadError,
    PolicyError,
    ResolverError,
    RetryableValidationError,
    ValidationError,
)

# ============================================================================
# CATALOG BOUNDARIES INTEGRATION (Task 1.1)
# ============================================================================
# Optional DuckDB and catalog boundary imports
try:
    import duckdb

    from DocsToKG.OntologyDownload.catalog.boundaries import (
        download_boundary,
        extraction_boundary,
        set_latest_boundary,
        validation_boundary,
    )

    CATALOG_AVAILABLE = True
except ImportError:  # pragma: no cover
    CATALOG_AVAILABLE = False
    duckdb = None  # type: ignore
    download_boundary = None  # type: ignore
    extraction_boundary = None  # type: ignore
    validation_boundary = None  # type: ignore
    set_latest_boundary = None  # type: ignore

from .io import (
    RDF_MIME_ALIASES,
    RDF_MIME_FORMAT_LABELS,
    download_stream,
    extract_archive_safe,
    generate_correlation_id,
    get_bucket,
    is_retryable_error,
    retry_with_backoff,
    sanitize_filename,
    validate_url_security,
)
from .io.network import _extract_correlation_id, _parse_retry_after, request_with_redirect_audit
from .logging_utils import setup_logging
from .migrations import migrate_manifest
from .net import get_http_client
from .resolvers import (
    RESOLVERS,
    BaseResolver,
    BioPortalResolver,
    DirectResolver,
    FetchPlan,
    LOVResolver,
    OBOResolver,
    OLSResolver,
    OntobeeResolver,
    ResolverCandidate,
    SKOSResolver,
    XBRLResolver,
    normalize_license_to_spdx,
)
from .settings import (
    CACHE_DIR,
    LOCAL_ONTOLOGY_DIR,
    STORAGE,
    ConfigError,
    DefaultsConfig,
    DownloadConfiguration,
    PlannerConfig,
    ResolvedConfig,
    _coerce_sequence,
    ensure_python_version,
    get_default_config,
)
from .validation import ValidationRequest, ValidationResult, run_validators

# ============================================================================
# PLAN CACHING & DATABASE INTEGRATION (Phase 4)
# ============================================================================
# Optional database plan caching for deterministic replays
try:
    from DocsToKG.OntologyDownload.database import (
        PlanRow,
        close_database,
        get_database,
    )

    PLAN_CACHING_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLAN_CACHING_AVAILABLE = False
    get_database = None  # type: ignore
    close_database = None  # type: ignore
    PlanRow = None  # type: ignore


def _log_with_extra(
    logger: logging.Logger,
    level: int,
    message: str,
    extra: Mapping[str, object],
) -> None:
    """Log ``message`` with structured ``extra`` supporting LoggerAdapters."""

    if isinstance(logger, logging.LoggerAdapter):
        adapter_extra = getattr(logger, "extra", None)
        merged: dict[str, object] = dict(adapter_extra or {})
        merged.update(extra)
        logger.logger.log(level, message, extra=merged)
        return
    logger.log(level, message, extra=extra)


MANIFEST_SCHEMA_VERSION = "1.0"

MANIFEST_JSON_SCHEMA: dict[str, Any] = {
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
        "streaming_prefix_sha256": {"type": ["string", "null"]},
        "streaming_content_sha256": {"type": ["string", "null"]},
        "expected_checksum": {
            "type": ["object", "null"],
            "properties": {
                "algorithm": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["algorithm", "value"],
            "additionalProperties": False,
        },
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

# Maintain backwards compatibility for existing internal imports.
_resolve_expected_checksum = resolve_expected_checksum


Draft202012Validator.check_schema(MANIFEST_JSON_SCHEMA)
_MANIFEST_VALIDATOR = Draft202012Validator(MANIFEST_JSON_SCHEMA)


def get_manifest_schema() -> dict[str, Any]:
    """Return a deep copy of the manifest JSON Schema definition.

    Args:
        None

    Returns:
        Dictionary describing the manifest JSON Schema.
    """

    return deepcopy(MANIFEST_JSON_SCHEMA)


def validate_manifest_dict(payload: Mapping[str, Any], *, source: Path | None = None) -> None:
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


@dataclass(slots=True, frozen=True)
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
    extras: dict[str, object]
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

    raw_prefer_source = raw_spec.get("prefer_source")
    prefer_source = _coerce_sequence(raw_prefer_source) or list(defaults.prefer_source)
    normalize_to = _coerce_sequence(raw_spec.get("normalize_to")) or list(defaults.normalize_to)
    extras = raw_spec.get("extras")
    if extras is not None and not isinstance(extras, Mapping):
        raise ConfigError(f"Ontology '{ontology_id}' extras must be a mapping if provided")

    if not allow_missing_resolvers:
        missing = [resolver for resolver in prefer_source if resolver not in RESOLVERS]
        if missing:
            if raw_prefer_source is None:
                prefer_source = [resolver for resolver in prefer_source if resolver in RESOLVERS]
            else:
                raise ConfigError(
                    "Unknown resolver(s) specified: " + ", ".join(sorted(set(missing)))
                )

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
    raw_spec: Mapping[str, object], defaults: DefaultsConfig, *, index: int | None = None
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


@dataclass(slots=True, frozen=True)
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
    expected_checksum: ExpectedChecksum | None = None
    etag: str | None = None
    last_modified: str | None = None
    content_type: str | None = None
    content_length: int | None = None
    cache_status: Mapping[str, object] | None = None


ResolvedConfig.model_rebuild()


class BatchPlanningError(RuntimeError):
    """Raised when ontology planning aborts after a failure."""

    def __init__(
        self,
        *,
        failed_spec: FetchSpec,
        original: BaseException,
        completed: list[PlannedFetch],
        total: int,
    ) -> None:
        self.failed_spec = failed_spec
        self.original = original
        self.completed = completed
        self.total = total
        successful = len(completed)
        message = (
            f"Planning aborted after {successful}/{total} completed when "
            f"{failed_spec.id} failed: {original}"
        )
        super().__init__(message)


class BatchFetchError(RuntimeError):
    """Raised when ontology downloads abort after a failure."""

    def __init__(
        self,
        *,
        failed_spec: FetchSpec,
        original: BaseException,
        completed: list[FetchResult],
        total: int,
    ) -> None:
        self.failed_spec = failed_spec
        self.original = original
        self.completed = completed
        self.total = total
        successful = len(completed)
        message = (
            f"Download aborted after {successful}/{total} completed when "
            f"{failed_spec.id} failed: {original}"
        )
        super().__init__(message)


def _cancel_pending_futures(
    futures: Mapping[Future[Any], tuple[int, FetchSpec]],
    *,
    current: Future[Any] | None = None,
) -> None:
    """Cancel any futures that are still pending execution."""

    for future in futures:
        if future is current:
            continue
        done_fn = getattr(future, "done", None)
        if callable(done_fn) and done_fn():
            continue
        cancel_fn = getattr(future, "cancel", None)
        if callable(cancel_fn):
            cancel_fn()


def _executor_is_shutting_down(executor: ThreadPoolExecutor) -> bool:
    """Return ``True`` when *executor* has begun shutdown processing."""

    return bool(getattr(executor, "_shutdown", False))


def _shutdown_executor_nowait(executor: ThreadPoolExecutor) -> None:
    """Request non-blocking shutdown with cancellation for *executor*."""

    if _executor_is_shutting_down(executor):
        return
    executor.shutdown(wait=False, cancel_futures=True)


@dataclass(slots=True, frozen=True)
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
        streaming_content_sha256: Streaming canonical content hash when available.
        streaming_prefix_sha256: Hash of Turtle prefix header when available.
        downloaded_at: UTC timestamp of the completed download.
        target_formats: Desired conversion targets for normalization.
        validation: Mapping of validator names to their results.
        artifacts: Additional file paths generated during processing.
        resolver_attempts: Ordered record of resolver attempts during download.
        expected_checksum: Optional checksum metadata enforced for the download.

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
    version: str | None
    license: str | None
    status: str
    sha256: str
    normalized_sha256: str | None
    fingerprint: str | None
    etag: str | None
    last_modified: str | None
    downloaded_at: str
    target_formats: Sequence[str]
    validation: dict[str, ValidationResult]
    artifacts: Sequence[str]
    resolver_attempts: Sequence[dict[str, object]]
    content_type: str | None = None
    content_length: int | None = None
    source_media_type_label: str | None = None
    streaming_prefix_sha256: str | None = None
    streaming_content_sha256: str | None = None
    expected_checksum: ExpectedChecksum | None = None

    def to_dict(self) -> dict[str, Any]:
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
            "streaming_prefix_sha256": self.streaming_prefix_sha256,
            "streaming_content_sha256": self.streaming_content_sha256,
            "expected_checksum": (
                self.expected_checksum.to_mapping() if self.expected_checksum else None
            ),
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
    metadata: dict[str, object] = field(default_factory=dict)
    last_modified: str | None = None
    last_modified_at: datetime | None = None
    size: int | None = None


def parse_http_datetime(value: str | None) -> datetime | None:
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
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_iso_datetime(value: str | None) -> datetime | None:
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
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_version_timestamp(value: str | None) -> datetime | None:
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

    candidates: list[str] = []

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
            return naive.replace(tzinfo=UTC)
    return None


def infer_version_timestamp(value: str | None) -> datetime | None:
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
        return naive.replace(tzinfo=UTC)
    return None


def _coerce_datetime(value: str | None) -> datetime | None:
    """Return timezone-aware datetime parsed from HTTP or ISO timestamp."""

    parsed = parse_http_datetime(value)
    if parsed is not None:
        return parsed
    return parse_iso_datetime(value)


def _normalize_timestamp(value: str | None) -> str | None:
    """Return canonical ISO8601 string for HTTP timestamp headers."""

    parsed = _coerce_datetime(value)
    if parsed is None:
        return value
    return parsed.isoformat().replace("+00:00", "Z")


def _canonical_media_type(value: str | None) -> str | None:
    """Return a normalized MIME type without parameters."""

    if not value:
        return None
    return value.split(";", 1)[0].strip().lower() or None


DEFAULT_VALIDATOR_NAMES: tuple[str, ...] = (
    "rdflib",
    "pronto",
    "owlready2",
    "robot",
    "arelle",
)

_MEDIA_VALIDATOR_WHITELIST = {
    "application/zip": {"arelle"},
}


def _select_validators(media_type: str | None) -> list[str]:
    """Return validator names appropriate for ``media_type``."""

    canonical = _canonical_media_type(media_type)
    if canonical and canonical in _MEDIA_VALIDATOR_WHITELIST:
        allowed = set(_MEDIA_VALIDATOR_WHITELIST[canonical])
    else:
        allowed = set(DEFAULT_VALIDATOR_NAMES)
        if canonical and canonical not in RDF_MIME_ALIASES:
            allowed -= {"rdflib", "robot"}
    return [name for name in DEFAULT_VALIDATOR_NAMES if name in allowed]


@dataclass
class PlannerProbeResult:
    """Normalized response metadata produced by planner HTTP probes."""

    url: str
    method: str
    status_code: int
    ok: bool
    headers: Mapping[str, str]
    cache_status: Mapping[str, object] | None = None


def planner_http_probe(
    *,
    url: str,
    http_config: DownloadConfiguration,
    logger: logging.Logger,
    headers: Mapping[str, str] | None = None,
    service: str | None = None,
    context: Mapping[str, object] | None = None,
    planner_config: PlannerConfig | None = None,
) -> PlannerProbeResult | None:
    """Issue a polite planner probe using shared networking primitives."""

    parsed = urlparse(url)
    host = parsed.hostname.lower() if parsed.hostname else None

    base_extra: dict[str, object] = {"stage": "plan", "url": url}
    if service:
        base_extra["service"] = service
    if host:
        base_extra["host"] = host
    if context:
        base_extra.update(context)

    def _requires_head(hostname: str | None) -> bool:
        if not hostname or planner_config is None:
            return False
        candidates = getattr(planner_config, "head_precheck_hosts", []) or []
        hostname_lc = hostname.lower()
        for entry in candidates:
            if not entry:
                continue
            candidate = entry.strip().lower()
            if not candidate:
                continue
            if candidate.startswith("*."):
                suffix = candidate[1:]
                if hostname_lc.endswith(suffix):
                    return True
            elif candidate.startswith("."):
                if hostname_lc.endswith(candidate):
                    return True
            else:
                if hostname_lc == candidate:
                    return True
        return False

    primary_method = "HEAD" if _requires_head(host) else "GET"

    correlation_id = _extract_correlation_id(logger)
    polite_headers = http_config.polite_http_headers(correlation_id=correlation_id)
    merged_headers: dict[str, str] = {str(k): str(v) for k, v in polite_headers.items()}
    if headers:
        for key, value in headers.items():
            merged_headers[str(key)] = str(value)

    timeout = max(1, getattr(http_config, "timeout_sec", 30) or 30)
    bucket = get_bucket(http_config=http_config, service=service, host=host)

    _log_with_extra(
        logger,
        logging.INFO,
        "planner probe start",
        {**base_extra, "method": primary_method, "event": "planner_probe_start"},
    )

    class _RangeRetry(Exception):
        """Internal sentinel to retry GET without a Range header."""

    client = get_http_client(http_config)

    def _issue(current_method: str, *, allow_range: bool = True) -> PlannerProbeResult | None:
        range_header = "bytes=0-0" if current_method == "GET" and allow_range else None

        def _perform_once() -> PlannerProbeResult:
            if bucket is not None:
                bucket.consume()
            request_headers = dict(merged_headers)
            if range_header and "Range" not in request_headers:
                request_headers["Range"] = range_header
            extensions = {
                "config": http_config,
                "headers": request_headers,
                "correlation_id": correlation_id,
            }
            with request_with_redirect_audit(
                client=client,
                method=current_method,
                url=url,
                headers=request_headers,
                timeout=timeout,
                stream=current_method != "HEAD",
                http_config=http_config,
                assume_url_validated=True,
                extensions=extensions,
            ) as response:
                status_code = response.status_code
                if status_code in {429, 503}:
                    retry_delay = _parse_retry_after(response.headers.get("Retry-After"))
                    http_error = httpx.HTTPStatusError(
                        f"HTTP error {status_code}", request=response.request, response=response
                    )
                    if retry_delay is not None and retry_delay > 0:
                        http_error._retry_after_delay = retry_delay
                    raise http_error
                if status_code == 416 and range_header:
                    raise _RangeRetry()

                final_url = getattr(response, "validated_url", str(response.url))
                headers_map = httpx.Headers(response.headers)
                cache_status = response.extensions.get("ontology_cache_status")
                ok = 200 <= status_code < 400
                return PlannerProbeResult(
                    url=final_url,
                    method=current_method,
                    status_code=status_code,
                    ok=ok,
                    headers=headers_map,
                    cache_status=cache_status if isinstance(cache_status, Mapping) else None,
                )

        def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
            retry_extra = dict(base_extra)
            retry_extra.update(
                {
                    "method": current_method,
                    "attempt": attempt,
                    "retry_delay_sec": round(delay, 2),
                    "error": str(exc),
                    "event": "planner_probe_retry",
                }
            )
            _log_with_extra(logger, logging.WARNING, "planner probe retrying", retry_extra)

        try:
            return retry_with_backoff(
                _perform_once,
                retryable=is_retryable_error,
                max_attempts=max(1, http_config.max_retries),
                backoff_base=http_config.backoff_factor,
                jitter=http_config.backoff_factor,
                callback=_on_retry,
                retry_after=lambda exc: getattr(exc, "_retry_after_delay", None),
            )
        except _RangeRetry:
            if not allow_range:
                raise
            return _issue(current_method, allow_range=False)

    try:
        result = _issue(primary_method)
    except Exception as exc:  # pragma: no cover - exercised via tests
        failure_extra = dict(base_extra)
        failure_extra.update(
            {"method": primary_method, "error": str(exc), "event": "planner_probe_failed"}
        )
        _log_with_extra(logger, logging.WARNING, "planner probe failed", failure_extra)
        if isinstance(exc, (PolicyError, ConfigError)):
            raise
        return None

    if result is None:
        return None

    if result.status_code == 405 and primary_method == "HEAD":
        _log_with_extra(
            logger,
            logging.INFO,
            "planner probe fallback",
            {
                **base_extra,
                "from_method": "HEAD",
                "to_method": "GET",
                "event": "planner_probe_fallback",
            },
        )
        try:
            result = _issue("GET")
        except Exception as exc:  # pragma: no cover - exercised via tests
            failure_extra = dict(base_extra)
            failure_extra.update(
                {"method": "GET", "error": str(exc), "event": "planner_probe_failed"}
            )
            _log_with_extra(logger, logging.WARNING, "planner probe failed", failure_extra)
            if isinstance(exc, (PolicyError, ConfigError)):
                raise
            return None

    if result is None:
        return None

    completion_extra = dict(base_extra)
    completion_extra.update(
        {
            "method": result.method,
            "status_code": result.status_code,
            "ok": result.ok,
            "event": "planner_probe_complete",
        }
    )
    _log_with_extra(logger, logging.INFO, "planner probe complete", completion_extra)

    return result


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

    http_config = config.defaults.http
    planner_defaults = getattr(config.defaults, "planner", None)

    try:
        secure_url = validate_url_security(planned.plan.url, http_config)
    except (ConfigError, PolicyError) as exc:
        _log_with_extra(
            adapter,
            logging.ERROR,
            "metadata probe blocked by URL policy",
            {
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "url": planned.plan.url,
                "error": str(exc),
                "event": "planner_probe_blocked",
            },
        )
        raise

    planned.plan.url = secure_url

    # GATE 2: URL Security Gate
    try:
        from DocsToKG.OntologyDownload.policy.errors import PolicyReject
        from DocsToKG.OntologyDownload.policy.gates import url_gate

        url_result = url_gate(
            secure_url,
            allowed_hosts=getattr(http_config, "allowed_hosts", None),
            allowed_ports=getattr(http_config, "allowed_ports", None),
        )
        if isinstance(url_result, PolicyReject):
            _log_with_extra(
                adapter,
                logging.ERROR,
                "url gate rejected",
                {
                    "stage": "plan",
                    "url": secure_url,
                    "error_code": url_result.error_code,
                    "event": "url_gate_rejected",
                },
            )
            raise PolicyError(f"URL policy violation: {url_result.error_code}")
    except PolicyError:
        raise

    probing_enabled = True if planner_defaults is None else planner_defaults.probing_enabled
    if not probing_enabled:
        _log_with_extra(
            adapter,
            logging.INFO,
            "planner probe skipped by configuration",
            {
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "url": secure_url,
                "reason": "planner.probing_enabled=false",
                "event": "planner_probe_skipped",
            },
        )
        return planned

    probe_result = planner_http_probe(
        url=secure_url,
        http_config=http_config,
        logger=adapter,
        headers=planned.plan.headers,
        service=planned.plan.service,
        context={"ontology_id": planned.spec.id, "resolver": planned.resolver},
        planner_config=planner_defaults,
    )
    if probe_result is None:
        return planned

    if not probe_result.ok:
        _log_with_extra(
            adapter,
            logging.WARNING,
            "metadata probe rejected",
            {
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "url": secure_url,
                "status": probe_result.status_code,
                "resolver": planned.resolver,
                "event": "planner_probe_rejected",
            },
        )
        return planned

    if probe_result.cache_status:
        metadata.setdefault("cache_status", {}).update(dict(probe_result.cache_status))

    headers_map = probe_result.headers

    last_modified_value = headers_map.get("Last-Modified")
    if last_modified_value:
        normalized = _normalize_timestamp(last_modified_value)
        planned.last_modified = normalized or last_modified_value
        planned.last_modified_at = _coerce_datetime(normalized or last_modified_value)
        planned.plan.last_modified = normalized or last_modified_value
        metadata["last_modified"] = planned.plan.last_modified

    content_type_value = headers_map.get("Content-Type")
    canonical_type = _canonical_media_type(content_type_value)
    if canonical_type:
        metadata["content_type"] = canonical_type
        planned.plan.media_type = canonical_type
        label = RDF_MIME_FORMAT_LABELS.get(canonical_type)
        if label:
            metadata["source_media_type_label"] = label

    if planned.size is None:
        content_length_value = headers_map.get("Content-Length")
        if content_length_value:
            try:
                parsed_length = int(content_length_value)
            except ValueError:
                parsed_length = None
            if parsed_length is not None:
                planned.size = parsed_length
                planned.plan.content_length = parsed_length
                metadata["content_length"] = parsed_length

    etag = headers_map.get("ETag")
    if etag:
        metadata["etag"] = etag

    return planned


def _read_manifest(manifest_path: Path) -> dict | None:
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
    payload = migrate_manifest(payload)
    validate_manifest_dict(payload, source=manifest_path)
    return payload


def _validate_manifest(manifest: Manifest) -> dict[str, Any]:
    """Check that a manifest instance satisfies structural and type requirements.

    Args:
        manifest: Manifest produced after a download completes.

    Raises:
        ConfigurationError: If required fields are missing or contain invalid types.
    """
    payload = migrate_manifest(manifest.to_dict())
    validate_manifest_dict(payload)

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
    if manifest.streaming_prefix_sha256 is not None and not isinstance(
        manifest.streaming_prefix_sha256, str
    ):
        raise ConfigurationError("Manifest streaming_prefix_sha256 must be a string when provided")
    return payload


def _parse_last_modified(value: str | None) -> datetime | None:
    """Return a timezone-aware datetime parsed from HTTP date headers."""

    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fetch_last_modified(
    plan: FetchPlan, config: ResolvedConfig, logger: logging.Logger
) -> str | None:
    """Probe the upstream plan URL for a Last-Modified header."""

    http_config = config.defaults.http
    planner_defaults = getattr(config.defaults, "planner", None)

    try:
        secure_url = validate_url_security(plan.url, http_config)
    except (ConfigError, PolicyError) as exc:
        _log_with_extra(
            logger,
            logging.ERROR,
            "metadata probe blocked by URL policy",
            {
                "stage": "plan",
                "url": plan.url,
                "resolver": plan.service or plan.url,
                "error": str(exc),
                "event": "planner_probe_blocked",
            },
        )
        raise

    plan.url = secure_url

    probing_enabled = True if planner_defaults is None else planner_defaults.probing_enabled
    if not probing_enabled:
        _log_with_extra(
            logger,
            logging.INFO,
            "planner probe skipped by configuration",
            {
                "stage": "plan",
                "url": secure_url,
                "resolver": plan.service or plan.url,
                "reason": "planner.probing_enabled=false",
                "event": "planner_probe_skipped",
            },
        )
        return None

    result = planner_http_probe(
        url=secure_url,
        http_config=http_config,
        logger=logger,
        headers=plan.headers,
        service=plan.service,
        context={"resolver": plan.service or plan.url},
        planner_config=planner_defaults,
    )
    if result is None:
        return None

    if not result.ok:
        _log_with_extra(
            logger,
            logging.WARNING,
            "metadata probe rejected",
            {
                "stage": "plan",
                "url": secure_url,
                "resolver": plan.service or plan.url,
                "status": result.status_code,
                "event": "planner_probe_rejected",
            },
        )
        return None

    return result.headers.get("Last-Modified")


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace ``path`` with ``content`` to avoid partial writes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except (AttributeError, OSError):
            # Some platforms or filesystems may not support fsync on text handles.
            pass
        temp_name = handle.name
    Path(temp_name).replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    """Serialize ``payload`` to JSON and atomically persist it to ``path``."""

    serialized = json.dumps(payload, indent=2, sort_keys=True)
    _atomic_write_text(path, serialized)


def _write_manifest(manifest_path: Path, manifest: Manifest) -> None:
    """Persist a validated manifest to disk as JSON.

    Args:
        manifest_path: Destination path for the manifest file.
        manifest: Manifest describing the downloaded ontology artifact.
    """
    payload = _validate_manifest(manifest)
    _atomic_write_json(manifest_path, payload)


def _mirror_to_cas_if_enabled(
    *,
    destination: Path,
    digest: str,
    config: ResolvedConfig,
    logger: logging.LoggerAdapter,
) -> Path | None:
    """Mirror ``destination`` into the content-addressable cache when enabled."""

    if not config.defaults.enable_cas_mirror:
        return None
    if not digest:
        return None
    try:
        cas_path = STORAGE.mirror_cas_artifact("sha256", digest, destination)
    except Exception as exc:  # pragma: no cover - defensive guard for filesystem issues
        logger.warning(
            "cas mirror failed",
            extra={
                "stage": "download",
                "error": str(exc),
            },
        )
        return None
    logger.debug(
        "cas mirror created",
        extra={
            "stage": "download",
            "cas_path": str(cas_path),
            "algorithm": "sha256",
        },
    )
    return cas_path


def _cleanup_failed_validation_artifacts(
    *,
    destination: Path,
    extraction_dir: Path | None,
    cas_path: Path | None,
    base_dir: Path,
    logger: logging.LoggerAdapter,
) -> None:
    """Remove artefacts produced before a strict-mode validation failure."""

    cleanup_targets: list[tuple[Path | None, str, bool]] = [
        (destination, "downloaded file", False),
        (extraction_dir, "extracted archive", True),
        (cas_path, "cas mirror", False),
    ]

    for path, description, is_directory in cleanup_targets:
        if path is None:
            continue
        try:
            exists = path.exists()
        except OSError as exc:
            logger.warning(
                "cleanup skipped due to access error",
                extra={"stage": "cleanup", "path": str(path), "error": str(exc)},
            )
            continue
        if not exists:
            continue
        try:
            if is_directory:
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning(
                "cleanup failed after validation error",
                extra={
                    "stage": "cleanup",
                    "path": str(path),
                    "error": str(exc),
                    "resource": description,
                },
            )

    try:
        if base_dir.exists() and not any(entry.is_file() for entry in base_dir.rglob("*")):
            shutil.rmtree(base_dir)
    except OSError as exc:
        logger.warning(
            "cleanup failed for ontology workspace",
            extra={"stage": "cleanup", "path": str(base_dir), "error": str(exc)},
        )


def _build_destination(
    spec: FetchSpec, plan: FetchPlan, config: ResolvedConfig
) -> tuple[Path, str, Path]:
    """Determine the output directory and filename for a download.

    Args:
        spec: Fetch specification identifying the ontology.
        plan: Resolver plan containing URL metadata and optional hints.
        config: Resolved configuration with storage layout parameters.

    Returns:
        Tuple containing the target file path, resolved version, and base directory.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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


def _resolver_candidates(spec: FetchSpec, config: ResolvedConfig) -> list[str]:
    candidates: list[str] = []
    seen = set()

    def _add(name: str | None) -> None:
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
    spec: FetchSpec,
    config: ResolvedConfig,
    adapter: logging.LoggerAdapter,
    *,
    cancellation_token: CancellationToken | None = None,
) -> tuple[ResolverCandidate, Sequence[ResolverCandidate]]:
    attempts: list[str] = []
    candidates: list[ResolverCandidate] = []
    primary: ResolverCandidate | None = None
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
            plan = resolver.plan(spec, config, adapter, cancellation_token=cancellation_token)
        except PolicyError as exc:
            message = str(exc)
            attempts.append(f"{resolver_name}: {message}")
            adapter.warning(
                "download attempt rejected by policy",
                extra={
                    "stage": "download",
                    "resolver": resolver_name,
                    "attempt": attempt_number,
                    "error": message,
                },
            )
            continue
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


# ============================================================================
# CATALOG BOUNDARY HELPERS (Task 1.1)
# ============================================================================


def _get_duckdb_conn(active_config: ResolvedConfig) -> duckdb.DuckDBPyConnection | None:
    """Get or create DuckDB writer connection from config."""
    if not CATALOG_AVAILABLE or duckdb is None:
        return None
    try:
        from DocsToKG.OntologyDownload.catalog.connection import DuckDBConfig, get_writer

        db_cfg = DuckDBConfig(
            path=active_config.defaults.db.path,
            threads=active_config.defaults.db.threads,
            readonly=False,
            writer_lock=active_config.defaults.db.writer_lock,
        )
        return get_writer(db_cfg)
    except Exception:
        return None


def _safe_record_boundary(
    adapter: logging.LoggerAdapter, boundary_name: str, boundary_fn, *args, **kwargs
) -> tuple[bool, Any | None]:
    """Safely call a boundary context manager with error handling.

    Returns: (success: bool, result: Any)
    """
    try:
        with boundary_fn(*args, **kwargs) as result:
            adapter.info(
                f"{boundary_name} recorded in catalog",
                extra={
                    "stage": "catalog",
                    "boundary": boundary_name,
                },
            )
            return True, result
    except Exception as e:
        adapter.warning(
            f"failed to record {boundary_name} in catalog",
            extra={
                "stage": "catalog",
                "boundary": boundary_name,
                "error": str(e),
                "severity": "non-critical",
            },
        )
        return False, None


def fetch_one(
    spec: FetchSpec,
    *,
    config: ResolvedConfig | None = None,
    correlation_id: str | None = None,
    logger: logging.Logger | None = None,
    force: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> FetchResult:
    """Fetch, validate, and persist a single ontology described by *spec*.

    Args:
        spec: Ontology fetch specification describing sources and formats.
        config: Optional resolved configuration overriding global defaults.
        correlation_id: Correlation identifier for structured logging.
        logger: Optional logger to reuse instead of configuring a new one.
        force: When ``True``, bypass local cache checks and redownload artifacts.
        cancellation_token: Optional token for cooperative cancellation.

    Returns:
        FetchResult: Structured result containing manifest metadata and resolver attempts.

    Raises:
        ResolverError: If all resolver candidates fail to retrieve the ontology.
    """

    ensure_python_version()
    active_config = config or get_default_config(copy=True)
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

    # Gate 1: Validate configuration
    try:
        from DocsToKG.OntologyDownload.policy.gates import config_gate

        config_result = config_gate(active_config)
        if not config_result.passed:
            raise ConfigError(f"Configuration validation failed: {config_result.error_code}")
    except Exception as e:
        adapter.error(
            "config gate rejected",
            extra={"stage": "plan", "error": str(e), "event": "config_gate_rejected"},
        )
        raise

    primary, candidates = _resolve_plan_with_fallback(
        spec, active_config, adapter, cancellation_token=cancellation_token
    )
    download_config = active_config.defaults.http
    candidate_list = list(candidates) or [primary]

    resolver_attempts: list[dict[str, object]] = []
    last_error: Exception | None = None

    for attempt_number, candidate in enumerate(candidate_list, start=1):
        attempt_record: dict[str, object] = {
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
                expected_checksum = _resolve_expected_checksum(
                    spec=pending_spec,
                    plan=candidate.plan,
                    download_config=download_config,
                    logger=adapter,
                )
                expected_hash_value = (
                    expected_checksum.to_known_hash() if expected_checksum else None
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
                    expected_hash=expected_hash_value,
                    cancellation_token=cancellation_token,
                    url_already_validated=True,
                )

                # === CATALOG: Record download in DuckDB (Task 1.1) ===
                try:
                    if CATALOG_AVAILABLE and download_boundary is not None:
                        conn = _get_duckdb_conn(active_config)
                        if conn is not None:
                            # Compute relative path from storage root
                            try:
                                fs_relpath = str(
                                    destination.relative_to(
                                        STORAGE.base_path() or destination.parent
                                    )
                                )
                            except (ValueError, AttributeError):
                                fs_relpath = destination.name

                            success, _ = _safe_record_boundary(
                                adapter,
                                "download",
                                download_boundary,
                                conn,
                                artifact_id=result.sha256,
                                version_id=version,
                                fs_relpath=fs_relpath,
                                size=result.content_length or 0,
                                etag=result.etag,
                            )
                except Exception as e:
                    adapter.debug(f"Skipping download boundary: {e}")

                if expected_checksum:
                    attempt_record["expected_checksum"] = expected_checksum.to_mapping()

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
                cas_path = _mirror_to_cas_if_enabled(
                    destination=destination,
                    digest=result.sha256,
                    config=active_config,
                    logger=adapter,
                )
                if cas_path:
                    artifacts.append(str(cas_path))
                extraction_dir: Path | None = None
                if plan.media_type == "application/zip" or destination.suffix.lower() == ".zip":
                    extraction_dir = destination.parent / f"{destination.stem}_extracted"
                    try:
                        extracted_paths = extract_archive_safe(
                            destination,
                            extraction_dir,
                            logger=adapter,
                            max_uncompressed_bytes=active_config.defaults.http.max_uncompressed_bytes(),
                        )
                        artifacts.extend(str(path) for path in extracted_paths)

                        # === CATALOG: Record extracted files in DuckDB (Task 1.1) ===
                        try:
                            if CATALOG_AVAILABLE and extraction_boundary is not None:
                                conn = _get_duckdb_conn(active_config)
                                if conn is not None and extraction_dir and extraction_dir.exists():
                                    with extraction_boundary(conn, result.sha256) as ex_result:
                                        # Insert extracted files into DB via appender
                                        app = conn.appender("extracted_files")
                                        for extracted_path in extracted_paths:
                                            try:
                                                rel_path = extracted_path.relative_to(
                                                    extraction_dir
                                                )
                                            except ValueError:
                                                rel_path = extracted_path.name

                                            if extracted_path.exists():
                                                st = extracted_path.stat()
                                                file_id = f"{result.sha256}:{rel_path}"
                                                app.append(
                                                    [
                                                        file_id,
                                                        result.sha256,
                                                        version,
                                                        str(rel_path),
                                                        extracted_path.suffix.lower() or "unknown",
                                                        st.st_size,
                                                        datetime.fromtimestamp(
                                                            st.st_mtime
                                                        ).isoformat(),
                                                        None,
                                                    ]
                                                )
                                                ex_result.files_inserted += 1
                                                ex_result.total_size += st.st_size

                                        app.close()
                                        ex_result.audit_path = (
                                            extraction_dir / "extraction_audit.json"
                                        )
                                        adapter.info(
                                            "extraction recorded in catalog",
                                            extra={
                                                "stage": "catalog",
                                                "files": ex_result.files_inserted,
                                            },
                                        )
                        except Exception as e:
                            adapter.debug(f"Skipping extraction boundary: {e}")

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
                failed_validators = [
                    name
                    for name, result in validation_results.items()
                    if not getattr(result, "ok", False)
                ]

                # === CATALOG: Record validation results in DuckDB (Task 1.1) ===
                try:
                    if CATALOG_AVAILABLE and validation_boundary is not None:
                        conn = _get_duckdb_conn(active_config)
                        if conn is not None:
                            # Record validations for each extracted file and validator
                            files_for_validation = (
                                extracted_paths
                                if (extraction_dir and extraction_dir.exists())
                                else [destination]
                            )
                            for extracted_path in files_for_validation:
                                try:
                                    file_id = f"{result.sha256}:{extracted_path.relative_to(extraction_dir) if extraction_dir else extracted_path.name}"
                                except (ValueError, AttributeError):
                                    file_id = f"{result.sha256}:{extracted_path.name}"

                                for validator_name, validation_result in validation_results.items():
                                    try:
                                        status = (
                                            "pass"
                                            if getattr(validation_result, "ok", False)
                                            else "fail"
                                        )
                                        details = getattr(validation_result, "details", None)

                                        success, _ = _safe_record_boundary(
                                            adapter,
                                            f"validation({validator_name})",
                                            validation_boundary,
                                            conn,
                                            file_id=file_id,
                                            validator=validator_name,
                                            status=status,
                                            details=details if isinstance(details, dict) else None,
                                        )
                                    except Exception as e:
                                        adapter.debug(
                                            f"Skipping validation boundary for {validator_name}: {e}"
                                        )
                except Exception as e:
                    adapter.debug(f"Skipping validation boundary: {e}")

                if failed_validators:
                    log_payload = {
                        "stage": "validate",
                        "validators": failed_validators,
                        "strict": not active_config.defaults.continue_on_error,
                    }
                    attempt_record["validators"] = list(failed_validators)
                    if active_config.defaults.continue_on_error:
                        adapter.warning("validation failures ignored", extra=log_payload)
                    else:
                        adapter.error("validation failures detected", extra=log_payload)
                        _cleanup_failed_validation_artifacts(
                            destination=destination,
                            extraction_dir=extraction_dir,
                            cas_path=cas_path,
                            base_dir=base_dir,
                            logger=adapter,
                        )
                        has_additional_candidates = attempt_number < len(candidate_list)
                        raise RetryableValidationError(
                            "Validation failed for "
                            f"'{effective_spec.id}' via {', '.join(failed_validators)}",
                            validators=tuple(failed_validators),
                            retryable=has_additional_candidates,
                        )

                normalized_hash = None
                streaming_content_hash = None
                normalization_mode = "none"
                rdflib_result = validation_results.get("rdflib")
                if rdflib_result and isinstance(rdflib_result.details, dict):
                    maybe_hash = rdflib_result.details.get("normalized_sha256")
                    if isinstance(maybe_hash, str):
                        normalized_hash = maybe_hash
                    maybe_mode = rdflib_result.details.get("normalization_mode")
                    if isinstance(maybe_mode, str):
                        normalization_mode = maybe_mode
                    maybe_streaming_content = rdflib_result.details.get("streaming_nt_sha256")
                    if isinstance(maybe_streaming_content, str):
                        streaming_content_hash = maybe_streaming_content

                streaming_prefix_hash = None
                for item in validation_results.values():
                    details = getattr(item, "details", None)
                    if isinstance(details, dict):
                        maybe_stream = details.get("streaming_prefix_sha256")
                        if isinstance(maybe_stream, str):
                            streaming_prefix_hash = maybe_stream
                            break
                        if streaming_content_hash is None:
                            maybe_stream_hash = details.get("streaming_nt_sha256")
                            if isinstance(maybe_stream_hash, str):
                                streaming_content_hash = maybe_stream_hash

                target_formats_sorted = ",".join(sorted(effective_spec.target_formats))

                content_fingerprint_basis = (
                    streaming_content_hash or normalized_hash or result.sha256
                )
                fingerprint_components = [
                    MANIFEST_SCHEMA_VERSION,
                    content_fingerprint_basis or "",
                    streaming_prefix_hash or "",
                    normalization_mode,
                    target_formats_sorted,
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
                    streaming_content_sha256=streaming_content_hash,
                    etag=result.etag,
                    last_modified=result.last_modified,
                    content_type=content_type,
                    content_length=content_length,
                    source_media_type_label=source_media_label,
                    streaming_prefix_sha256=streaming_prefix_hash,
                    expected_checksum=expected_checksum,
                    downloaded_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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
                    "streaming_content_sha256": manifest.streaming_content_sha256,
                    "streaming_prefix_sha256": manifest.streaming_prefix_sha256,
                    "etag": manifest.etag,
                    "size": destination.stat().st_size if destination.exists() else None,
                    "source_url": manifest.url,
                    "content_type": manifest.content_type,
                    "content_length": manifest.content_length,
                    "status": manifest.status,
                }
                if expected_checksum:
                    index_entry["expected_checksum"] = expected_checksum.to_mapping()
                if cas_path:
                    index_entry["cas_path"] = str(cas_path)
                # === CATALOG: Mark as latest in DuckDB (Task 1.1) ===
                try:
                    if CATALOG_AVAILABLE and set_latest_boundary is not None:
                        conn = _get_duckdb_conn(active_config)
                        if conn is not None:
                            # Prepare LATEST.json in temp file
                            latest_json_path = base_dir.parent / "LATEST.json"
                            latest_temp_path = latest_json_path.with_suffix(".json.tmp")
                            latest_temp_path.parent.mkdir(parents=True, exist_ok=True)

                            latest_data = {
                                "version": version,
                                "downloaded_at": datetime.now(UTC).isoformat(),
                                "sha256": result.sha256,
                                "resolver": effective_spec.resolver,
                                "correlation_id": correlation,
                            }

                            with open(latest_temp_path, "w") as f:
                                json.dump(latest_data, f, indent=2)

                            success, _ = _safe_record_boundary(
                                adapter,
                                "set_latest",
                                set_latest_boundary,
                                conn,
                                version_id=version,
                                latest_json_path=latest_json_path,
                            )
                except Exception as e:
                    adapter.debug(f"Skipping set_latest boundary: {e}")

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
                    expected_checksum=expected_checksum,
                    etag=result.etag,
                    last_modified=result.last_modified,
                    content_type=content_type,
                    content_length=content_length,
                    cache_status=result.cache_status,
                )

        try:
            return _execute_candidate()
        except ValidationError as exc:
            last_error = exc
            attempt_record.update({"status": "failed", "error": str(exc)})
            validators = getattr(exc, "validators", None)
            if validators:
                attempt_record["validators"] = list(validators)
            resolver_attempts.append(dict(attempt_record))
            adapter.warning(
                "validation attempt failed",
                extra={
                    "stage": "validate",
                    "resolver": candidate.resolver,
                    "attempt": attempt_number,
                    "error": str(exc),
                },
            )
            retryable = getattr(exc, "retryable", False)
            if retryable:
                adapter.info(
                    "trying fallback resolver",
                    extra={
                        "stage": "validate",
                        "resolver": candidate.resolver,
                        "attempt": attempt_number,
                    },
                )
                continue
            raise OntologyDownloadError(
                f"Validation failed for '{pending_spec.id}': {exc}"
            ) from exc
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
    if isinstance(last_error, (ConfigurationError, PolicyError)):
        raise last_error
    raise OntologyDownloadError(f"Download failed for '{spec.id}': {last_error}") from last_error


def _planned_fetch_to_dict(planned: PlannedFetch) -> dict[str, Any]:
    """Convert a PlannedFetch to a dictionary for database storage.

    Args:
        planned: PlannedFetch object to serialize.

    Returns:
        Dictionary representation suitable for JSON serialization.
    """
    return {
        "spec": {
            "id": planned.spec.id,
            "resolver": planned.spec.resolver,
            "extras": planned.spec.extras,
            "target_formats": (
                list(planned.spec.target_formats) if planned.spec.target_formats else []
            ),
        },
        "resolver": planned.resolver,
        "plan": {
            "url": planned.plan.url,
            "headers": dict(planned.plan.headers),
            "filename_hint": planned.plan.filename_hint,
            "version": planned.plan.version,
            "license": planned.plan.license,
            "media_type": planned.plan.media_type,
            "content_length": planned.plan.content_length,
            "last_modified": planned.plan.last_modified,
        },
        "candidates": [
            {
                "resolver": c.resolver,
                "plan": {
                    "url": c.plan.url,
                    "headers": dict(c.plan.headers),
                    "filename_hint": c.plan.filename_hint,
                    "version": c.plan.version,
                    "license": c.plan.license,
                    "media_type": c.plan.media_type,
                    "content_length": c.plan.content_length,
                    "last_modified": c.plan.last_modified,
                },
            }
            for c in planned.candidates
        ],
        "metadata": planned.metadata or {},
        "last_modified": planned.last_modified,
        "size": planned.size,
    }


def _dict_to_planned_fetch(data: dict[str, Any], spec: FetchSpec) -> PlannedFetch | None:
    """Reconstruct a PlannedFetch from a stored dictionary.

    Args:
        data: Dictionary loaded from database.
        spec: Original FetchSpec for reference.

    Returns:
        PlannedFetch if reconstruction succeeds, else None.
    """
    if not isinstance(data, dict):
        return None

    try:
        plan_data = data.get("plan", {})
        primary_plan = FetchPlan(
            url=plan_data.get("url", ""),
            headers=dict(plan_data.get("headers", {})),
            filename_hint=plan_data.get("filename_hint", ""),
            version=plan_data.get("version"),
            license=plan_data.get("license"),
            media_type=plan_data.get("media_type"),
            content_length=plan_data.get("content_length"),
            last_modified=plan_data.get("last_modified"),
        )

        candidates = []
        for c_data in data.get("candidates", []):
            c_plan_data = c_data.get("plan", {})
            c_plan = FetchPlan(
                url=c_plan_data.get("url", ""),
                headers=dict(c_plan_data.get("headers", {})),
                filename_hint=c_plan_data.get("filename_hint", ""),
                version=c_plan_data.get("version"),
                license=c_plan_data.get("license"),
                media_type=c_plan_data.get("media_type"),
                content_length=c_plan_data.get("content_length"),
                last_modified=c_plan_data.get("last_modified"),
            )
            candidates.append(ResolverCandidate(resolver=c_data.get("resolver", ""), plan=c_plan))

        return PlannedFetch(
            spec=spec,
            resolver=data.get("resolver", ""),
            plan=primary_plan,
            candidates=tuple(candidates),
            metadata=data.get("metadata", {}),
            last_modified=data.get("last_modified"),
            size=data.get("size"),
        )
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def _get_cached_plan(spec: FetchSpec, logger: logging.LoggerAdapter) -> PlannedFetch | None:
    """Retrieve a cached plan for an ontology specification from the database.

    Args:
        spec: FetchSpec to look up.
        logger: Logger for diagnostics.

    Returns:
        Cached PlannedFetch if available, else None.
    """
    if not PLAN_CACHING_AVAILABLE or not get_database:
        return None

    db = get_database()
    plan_row = db.get_current_plan(spec.id)
    if plan_row and plan_row.plan_json:
        plan_data = (
            json.loads(plan_row.plan_json)
            if isinstance(plan_row.plan_json, str)
            else plan_row.plan_json
        )
        planned = _dict_to_planned_fetch(plan_data, spec)
        if planned:
            logger.debug(
                f"Using cached plan for '{spec.id}' from database",
                extra={"ontology_id": spec.id, "cached_at": plan_row.cached_at},
            )
            return planned
    close_database()
    return None


def _save_plan_to_db(spec: FetchSpec, planned: PlannedFetch, logger: logging.LoggerAdapter) -> bool:
    """Save a PlannedFetch to the database for future caching.

    Args:
        spec: FetchSpec for the ontology.
        planned: PlannedFetch to cache.
        logger: Logger for diagnostics.

    Returns:
        True if save succeeds, False otherwise.
    """
    if not PLAN_CACHING_AVAILABLE or not get_database:
        return False

    db = get_database()
    plan_dict = _planned_fetch_to_dict(planned)
    db.upsert_plan(
        ontology_id=spec.id,
        resolver=planned.resolver,
        plan_json=plan_dict,
        is_current=True,
    )
    logger.debug(
        f"Saved plan for '{spec.id}' to database",
        extra={"ontology_id": spec.id},
    )
    close_database()
    return True


def _compare_plans(older_plan: PlannedFetch | None, newer_plan: PlannedFetch) -> dict[str, Any]:
    """Compare two plan versions and return a diff.

    Args:
        older_plan: Previous plan (can be None if this is the first plan).
        newer_plan: Current plan.

    Returns:
        Dictionary containing diff statistics and details.
    """
    diff = {
        "older": older_plan is not None,
        "added": [],
        "removed": [],
        "modified": [],
        "unchanged": 0,
    }

    if older_plan is None:
        diff["added"].append(
            {
                "resolver": newer_plan.resolver,
                "url": newer_plan.plan.url,
                "version": newer_plan.plan.version,
            }
        )
        return diff

    # Compare primary resolver and URL
    if older_plan.resolver != newer_plan.resolver:
        diff["modified"].append(
            {
                "field": "resolver",
                "old": older_plan.resolver,
                "new": newer_plan.resolver,
            }
        )

    if older_plan.plan.url != newer_plan.plan.url:
        diff["modified"].append(
            {
                "field": "url",
                "old": older_plan.plan.url,
                "new": newer_plan.plan.url,
            }
        )

    if older_plan.plan.version != newer_plan.plan.version:
        diff["modified"].append(
            {
                "field": "version",
                "old": older_plan.plan.version,
                "new": newer_plan.plan.version,
            }
        )

    if older_plan.plan.license != newer_plan.plan.license:
        diff["modified"].append(
            {
                "field": "license",
                "old": older_plan.plan.license,
                "new": newer_plan.plan.license,
            }
        )

    if older_plan.plan.media_type != newer_plan.plan.media_type:
        diff["modified"].append(
            {
                "field": "media_type",
                "old": older_plan.plan.media_type,
                "new": newer_plan.plan.media_type,
            }
        )

    if older_plan.size != newer_plan.size:
        diff["modified"].append(
            {
                "field": "size_bytes",
                "old": older_plan.size,
                "new": newer_plan.size,
            }
        )

    # If no differences, mark as unchanged
    if not diff["modified"]:
        diff["unchanged"] = 1

    return diff


def _save_plan_diff_to_db(
    ontology_id: str,
    older_plan: PlannedFetch | None,
    newer_plan: PlannedFetch,
    logger: logging.LoggerAdapter,
) -> bool:
    """Save a plan diff to the database for historical comparison.

    Args:
        ontology_id: ID of the ontology.
        older_plan: Previous plan (can be None).
        newer_plan: Current plan.
        logger: Logger for diagnostics.

    Returns:
        True if save succeeds, False otherwise.
    """
    if not PLAN_CACHING_AVAILABLE or not get_database:
        return False

    db = get_database()
    diff_data = _compare_plans(older_plan, newer_plan)

    # Only save if there are actual changes
    if diff_data.get("modified") or diff_data.get("added") or diff_data.get("removed"):
        db.insert_plan_diff(
            ontology_id=ontology_id,
            older_plan_id=None,
            newer_plan_id=None,
            added_count=len(diff_data.get("added", [])),
            removed_count=len(diff_data.get("removed", [])),
            modified_count=len(diff_data.get("modified", [])),
            diff_json=diff_data,
        )
        logger.debug(
            f"Saved plan diff for '{ontology_id}' to database",
            extra={
                "ontology_id": ontology_id,
                "modified_count": len(diff_data.get("modified", [])),
            },
        )
        close_database()
        return True
    close_database()
    return False


def plan_one(
    spec: FetchSpec,
    *,
    config: ResolvedConfig | None = None,
    correlation_id: str | None = None,
    logger: logging.Logger | None = None,
    cancellation_token: CancellationToken | None = None,
) -> PlannedFetch:
    """Return a resolver plan for a single ontology without performing downloads.

    Args:
        spec: Fetch specification describing the ontology to plan.
        config: Optional resolved configuration providing defaults and limits.
        correlation_id: Correlation identifier reused for logging context.
        logger: Logger instance used to emit resolver telemetry.
        cancellation_token: Optional token for cooperative cancellation.

    Returns:
        PlannedFetch containing the normalized spec, resolver name, and plan.

    Raises:
        ResolverError: If all resolvers fail to produce a plan for ``spec``.
        ConfigurationError: If licence checks reject the planned ontology.
    """

    ensure_python_version()
    active_config = config or get_default_config(copy=True)
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

    # Attempt to retrieve cached plan from database
    cached_plan = _get_cached_plan(spec, adapter)
    if cached_plan is not None:
        adapter.info(
            "using cached plan from database",
            extra={
                "stage": "plan",
                "cached": True,
                "resolver": cached_plan.resolver,
            },
        )
        return cached_plan

    primary, candidates = _resolve_plan_with_fallback(
        spec, active_config, adapter, cancellation_token=cancellation_token
    )
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
    expected_checksum = _resolve_expected_checksum(
        spec=effective_spec,
        plan=primary.plan,
        download_config=active_config.defaults.http,
        logger=adapter,
    )
    if expected_checksum:
        planned.metadata["expected_checksum"] = expected_checksum.to_mapping()
    final_planned = _populate_plan_metadata(planned, active_config, adapter)

    # Save plan to database
    _save_plan_to_db(spec, final_planned, adapter)

    return final_planned


def plan_all(
    specs: Iterable[FetchSpec],
    *,
    config: ResolvedConfig | None = None,
    logger: logging.Logger | None = None,
    since: datetime | None = None,
    total: int | None = None,
    cancellation_token_group: CancellationTokenGroup | None = None,
) -> list[PlannedFetch]:
    """Return resolver plans for a collection of ontologies.

    Args:
        specs: Iterable of fetch specifications to resolve.
        config: Optional resolved configuration reused across plans.
        logger: Logger instance used for annotation-aware logging.
        since: Optional cutoff date; plans older than this timestamp are filtered out.
        total: Optional total number of specifications, used for progress metadata when
            the iterable cannot be sized cheaply.
        cancellation_token_group: Optional group of cancellation tokens for cooperative cancellation.

    Returns:
        List of PlannedFetch entries describing each ontology plan.

    Raises:
        ResolverError: Propagated when fallback planning fails for any spec.
        ConfigurationError: When licence enforcement rejects a planned ontology.
    """

    ensure_python_version()
    active_config = config or get_default_config(copy=True)
    logging_config = active_config.defaults.logging
    log = logger or setup_logging(
        level=logging_config.level,
        retention_days=logging_config.retention_days,
        max_log_size_mb=logging_config.max_log_size_mb,
    )
    correlation = generate_correlation_id()
    adapter = logging.LoggerAdapter(log, extra={"correlation_id": correlation})

    # Create cancellation token group if not provided
    if cancellation_token_group is None:
        cancellation_token_group = CancellationTokenGroup()

    total_hint = total
    if total_hint is None:
        try:
            total_hint = len(specs)  # type: ignore[arg-type]
        except TypeError:
            total_hint = None

    spec_iter = iter(specs)

    try:
        first_item = next(spec_iter)
    except StopIteration:
        return []

    index_counter = 0
    max_workers = max(1, active_config.defaults.http.concurrent_plans)
    progress_payload: dict[str, object] = {"stage": "plan", "workers": max_workers}
    if total_hint is not None:
        progress_payload["progress"] = {"total": total_hint}
    adapter.info("planning batch", extra=progress_payload)

    results: dict[int, PlannedFetch] = {}
    futures: dict[Future[PlannedFetch], tuple[int, FetchSpec]] = {}
    future_tokens: dict[Future[PlannedFetch], CancellationToken] = {}

    exhausted = False

    def _submit(spec: FetchSpec, index: int) -> None:
        # Create a cancellation token for this task
        token = (
            cancellation_token_group.create_token()
            if cancellation_token_group is not None
            else None
        )

        future = executor.submit(
            plan_one,
            spec,
            config=active_config,
            correlation_id=correlation,
            logger=log,
            cancellation_token=token,
        )
        futures[future] = (index, spec)
        if token is not None:
            future_tokens[future] = token

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _submit(first_item, index_counter)
        index_counter += 1

        while futures or not exhausted:
            while not exhausted and len(futures) < max_workers:
                try:
                    spec = next(spec_iter)
                except StopIteration:
                    exhausted = True
                    break
                _submit(spec, index_counter)
                index_counter += 1

            if not futures:
                break

            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                index, spec = futures.pop(future)
                token = future_tokens.pop(future, None)
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
                    # Cancel all pending tasks using cancellation tokens
                    cancellation_token_group.cancel_all()
                    _shutdown_executor_nowait(executor)
                    if token is not None:
                        cancellation_token_group.remove_token(token)
                    for remaining_token in list(future_tokens.values()):
                        cancellation_token_group.remove_token(remaining_token)
                    future_tokens.clear()
                    if isinstance(exc, (ConfigError, ConfigurationError, PolicyError)):
                        raise
                    completed_plans = [results[i] for i in sorted(results)]
                    total_known = total_hint if total_hint is not None else index_counter
                    raise BatchPlanningError(
                        failed_spec=spec,
                        original=exc,
                        completed=completed_plans,
                        total=total_known,
                    ) from exc
                else:
                    results[index] = planned
                    if token is not None:
                        cancellation_token_group.remove_token(token)

    ordered_indices = sorted(results)
    ordered_plans = [results[i] for i in ordered_indices]

    if since is None:
        return ordered_plans

    filtered: list[PlannedFetch] = []
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
    config: ResolvedConfig | None = None,
    logger: logging.Logger | None = None,
    force: bool = False,
    total: int | None = None,
    cancellation_token_group: CancellationTokenGroup | None = None,
) -> list[FetchResult]:
    """Fetch a sequence of ontologies sequentially.

    Args:
        specs: Iterable of fetch specifications to process.
        config: Optional resolved configuration shared across downloads.
        logger: Logger used to emit progress and error events.
        force: When True, skip manifest reuse and download everything again.
        total: Optional total number of specifications for progress metadata when
            the iterable cannot be cheaply materialised.
        cancellation_token_group: Optional group of cancellation tokens for cooperative cancellation.

    Returns:
        List of FetchResult entries corresponding to completed downloads.

    Raises:
        OntologyDownloadError: Propagated when downloads fail and the pipeline
            is configured to stop on error.
    """

    ensure_python_version()
    active_config = config or get_default_config(copy=True)
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

    # Create cancellation token group if not provided
    if cancellation_token_group is None:
        cancellation_token_group = CancellationTokenGroup()

    total_hint = total
    if total_hint is None:
        try:
            total_hint = len(specs)  # type: ignore[arg-type]
        except TypeError:
            total_hint = None

    spec_iter = iter(specs)

    try:
        first_spec = next(spec_iter)
    except StopIteration:
        return []

    max_workers = max(1, active_config.defaults.http.concurrent_downloads)
    batch_extra: dict[str, object] = {"stage": "batch", "workers": max_workers}
    if total_hint is not None:
        batch_extra["progress"] = {"total": total_hint}
    adapter.info("starting batch", extra=batch_extra)

    results_map: dict[int, FetchResult] = {}
    futures: dict[Future[FetchResult], tuple[int, FetchSpec]] = {}
    future_tokens: dict[Future[FetchResult], CancellationToken] = {}
    exhausted = False
    submitted = 0

    def _submit(spec: FetchSpec, index: int) -> None:
        progress: dict[str, object] = {"current": index}
        if total_hint is not None:
            progress["total"] = total_hint
        adapter.info(
            "starting ontology fetch",
            extra={"stage": "start", "ontology_id": spec.id, "progress": progress},
        )

        # Create a cancellation token for this task
        token = (
            cancellation_token_group.create_token()
            if cancellation_token_group is not None
            else None
        )

        future = executor.submit(
            fetch_one,
            spec,
            config=active_config,
            correlation_id=correlation,
            logger=log,
            force=force,
            cancellation_token=token,
        )
        futures[future] = (index, spec)
        if token is not None:
            future_tokens[future] = token

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted += 1
        _submit(first_spec, submitted)

        while futures or not exhausted:
            while not exhausted and len(futures) < max_workers:
                try:
                    spec = next(spec_iter)
                except StopIteration:
                    exhausted = True
                    break
                submitted += 1
                _submit(spec, submitted)

            if not futures:
                break

            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                index, spec = futures.pop(future)
                token = future_tokens.pop(future, None)
                try:
                    result = future.result()
                except Exception as exc:  # pylint: disable=broad-except
                    adapter.error(
                        "ontology fetch failed",
                        extra={"stage": "error", "ontology_id": spec.id, "error": str(exc)},
                    )
                    # Cancel all pending tasks using cancellation tokens
                    cancellation_token_group.cancel_all()
                    _shutdown_executor_nowait(executor)
                    if token is not None:
                        cancellation_token_group.remove_token(token)
                    for remaining_token in list(future_tokens.values()):
                        cancellation_token_group.remove_token(remaining_token)
                    future_tokens.clear()
                    completed_results = [results_map[i] for i in sorted(results_map)]
                    total_known = total_hint if total_hint is not None else submitted
                    raise BatchFetchError(
                        failed_spec=spec,
                        original=exc,
                        completed=completed_results,
                        total=total_known,
                    ) from exc
                else:
                    results_map[index] = result
                    progress = {"current": len(results_map)}
                    if total_hint is not None:
                        progress["total"] = total_hint
                    adapter.info(
                        "progress update",
                        extra={"stage": "progress", "ontology_id": spec.id, "progress": progress},
                    )
                    if token is not None:
                        cancellation_token_group.remove_token(token)

    ordered_results = [results_map[i] for i in sorted(results_map)]
    return ordered_results


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


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_JSON_SCHEMA",
    "FetchSpec",
    "FetchResult",
    "ExpectedChecksum",
    "PlannedFetch",
    "ResolverCandidate",
    "FetchPlan",
    "BaseResolver",
    "OBOResolver",
    "OLSResolver",
    "BioPortalResolver",
    "LOVResolver",
    "SKOSResolver",
    "DirectResolver",
    "XBRLResolver",
    "OntobeeResolver",
    "RESOLVERS",
    "normalize_license_to_spdx",
    "OntologyDownloadError",
    "ResolverError",
    "ValidationError",
    "ConfigurationError",
    "merge_defaults",
    "get_manifest_schema",
    "validate_manifest_dict",
    "_build_destination",
    "fetch_one",
    "fetch_all",
    "plan_one",
    "plan_all",
    "setup_logging",
]
