"""Ontology download orchestration for DocsToKG.

This module plans resolver candidates, enforces license allowlists, performs
fallback-aware downloads, orchestrates streaming normalization, and writes
schema-validated manifests with deterministic fingerprints. It aligns with the
refactored ontology download specification by recording resolver attempt
chains, honoring CLI concurrency overrides, and supporting batch operations for
planning and pull commands.
"""

from __future__ import annotations

import hashlib
import json
import logging
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlparse

import requests

from .config import ConfigError, ResolvedConfig, ensure_python_version
from .download import (
    download_stream,
    extract_archive_safe,
    RDF_MIME_ALIASES,
    sanitize_filename,
    validate_url_security,
)
from .logging_config import generate_correlation_id, setup_logging
from .resolvers import RESOLVERS, FetchPlan, normalize_license_to_spdx
from .storage import CACHE_DIR, LOCAL_ONTOLOGY_DIR, STORAGE
from .validators import ValidationRequest, ValidationResult, run_validators

ONTOLOGY_DIR = LOCAL_ONTOLOGY_DIR

MANIFEST_SCHEMA_VERSION = "1.0"


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

    def to_json(self) -> str:
        """Serialize the manifest to a stable, human-readable JSON string.

        Args:
            None

        Returns:
            JSON document encoding the manifest metadata.
        """
        payload = {
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
            "downloaded_at": self.downloaded_at,
            "target_formats": list(self.target_formats),
            "validation": {name: result.to_dict() for name, result in self.validation.items()},
            "artifacts": list(self.artifacts),
            "resolver_attempts": [dict(entry) for entry in self.resolver_attempts],
        }
        return json.dumps(payload, indent=2, sort_keys=True)


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


def _coerce_datetime(value: Optional[str]) -> Optional[datetime]:
    """Return timezone-aware datetime parsed from HTTP or ISO timestamp."""

    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _normalize_timestamp(value: Optional[str]) -> Optional[str]:
    """Return canonical ISO8601 string for HTTP timestamp headers."""

    parsed = _coerce_datetime(value)
    if parsed is None:
        return value
    return parsed.isoformat().replace("+00:00", "Z")


def _populate_plan_metadata(
    planned: PlannedFetch,
    config: ResolvedConfig,
    adapter: logging.LoggerAdapter,
) -> PlannedFetch:
    """Augment planned fetch with HTTP metadata when available."""

    if planned.plan.content_length is not None and planned.size is None:
        planned.size = planned.plan.content_length
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

    needs_size = planned.size is None
    needs_last_modified = planned.last_modified is None
    if not (needs_size or needs_last_modified):
        return planned

    try:
        validate_url_security(planned.plan.url, config.defaults.http.allowed_hosts)
    except ConfigError as exc:
        adapter.warning(
            "metadata probe skipped",
            extra={
                "stage": "plan",
                "ontology_id": planned.spec.id,
                "error": str(exc),
            },
        )
        return planned

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

    return planned


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
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None


def _validate_manifest(manifest: Manifest) -> None:
    """Check that a manifest instance satisfies structural and type requirements.

    Args:
        manifest: Manifest produced after a download completes.

    Raises:
        ConfigurationError: If required fields are missing or contain invalid types.
    """
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
    if not manifest.url.startswith("https://"):
        raise ConfigurationError("Manifest URL must use https scheme")
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
    base_dir = ONTOLOGY_DIR / sanitize_filename(spec.id) / sanitize_filename(version)
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
        except ConfigError as exc:
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
        spec: Fetch specification outlining resolver selection and target formats.
        config: Optional resolved configuration supplying defaults and limits.
        correlation_id: Identifier used to correlate structured log entries.
        logger: Logger instance reused for download and validation telemetry.
        force: When True, ignore cached manifests and re-download artefacts.

    Returns:
        FetchResult capturing download status, SHA-256 hashes, and manifest path.

    Raises:
        ResolverError: If all resolvers fail to produce a viable FetchPlan.
        OntologyDownloadError: If download, extraction, or validation fails.
        ConfigurationError: If licence checks or manifest validation fail.
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
    selected_candidate: Optional[ResolverCandidate] = None
    effective_spec: Optional[FetchSpec] = None
    destination: Optional[Path] = None
    version: Optional[str] = None
    base_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None
    secure_url: Optional[str] = None
    result: Optional[DownloadResult] = None
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
            resolver_attempts.append(attempt_record)
            last_error = exc
            continue

        if candidate.plan.service:
            adapter.extra["service"] = candidate.plan.service
        else:
            adapter.extra.pop("service", None)

        pending_destination, pending_version, pending_base_dir = _build_destination(
            pending_spec, candidate.plan, active_config
        )
        pending_manifest_path = pending_base_dir / "manifest.json"

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
        try:
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
            )
        except ConfigError as exc:
            attempt_record.update({"status": "failed", "error": str(exc)})
            resolver_attempts.append(attempt_record)
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

        attempt_record["status"] = "success"
        resolver_attempts.append(attempt_record)
        selected_candidate = candidate
        effective_spec = pending_spec
        destination = pending_destination
        version = pending_version
        base_dir = pending_base_dir
        manifest_path = pending_manifest_path
        secure_url = pending_secure_url
        break

    if result is None or selected_candidate is None or effective_spec is None:
        if last_error is None:
            raise OntologyDownloadError(f"All resolver candidates failed for '{spec.id}'")
        if isinstance(last_error, ConfigurationError):
            raise last_error
        raise OntologyDownloadError(
            f"Download failed for '{spec.id}': {last_error}"
        ) from last_error

    assert destination is not None
    assert version is not None
    assert base_dir is not None
    assert manifest_path is not None
    assert secure_url is not None

    plan = selected_candidate.plan

    normalized_dir = base_dir / "normalized"
    validation_dir = base_dir / "validation"
    validation_requests: List[ValidationRequest] = [
        ValidationRequest(
            name="rdflib",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        ),
        ValidationRequest(
            name="pronto",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        ),
        ValidationRequest(
            name="owlready2",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        ),
        ValidationRequest(
            name="robot",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        ),
        ValidationRequest(
            name="arelle",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        ),
    ]

    media_type = (plan.media_type or "").strip().lower()
    if media_type and media_type not in RDF_MIME_ALIASES:
        validation_requests = [
            request for request in validation_requests if request.name not in {"rdflib", "robot"}
        ]
        adapter.info(
            "skipping rdf validators",
            extra={
                "stage": "validate",
                "media_type": media_type,
                "validator": "rdf",
            },
        )

    artifacts = [str(destination)]
    if plan.media_type == "application/zip" or destination.suffix.lower() == ".zip":
        extraction_dir = destination.parent / f"{destination.stem}_extracted"
        try:
            extracted_paths = extract_archive_safe(destination, extraction_dir, logger=adapter)
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
    fingerprint = hashlib.sha256("|".join(fingerprint_components).encode("utf-8")).hexdigest()

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
        downloaded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        target_formats=effective_spec.target_formats,
        validation=validation_results,
        artifacts=artifacts,
        resolver_attempts=resolver_attempts,
    )
    _write_manifest(manifest_path, manifest)
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


__all__ = [
    "FetchSpec",
    "FetchResult",
    "Manifest",
    "ResolverCandidate",
    "PlannedFetch",
    "Resolver",
    "fetch_one",
    "fetch_all",
    "plan_one",
    "plan_all",
    "OntologyDownloadError",
    "ResolverError",
    "ValidationError",
    "ConfigurationError",
]
