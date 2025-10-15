"""
Ontology Download Orchestration

This module coordinates resolver planning, document downloading, validation,
and manifest generation for ontology artifacts. It serves as the main entry
point for fetching ontologies from configured sources and producing provenance
metadata that downstream knowledge graph construction can rely upon.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence
from urllib.parse import urlparse

from .config import ConfigError, ResolvedConfig, ensure_python_version
from .download import (
    download_stream,
    extract_zip_safe,
    sanitize_filename,
    validate_url_security,
)
from .logging_config import generate_correlation_id, setup_logging
from .optdeps import get_pystow
from .resolvers import RESOLVERS, FetchPlan, normalize_license_to_spdx
from .validators import ValidationRequest, ValidationResult, run_validators

pystow = get_pystow()


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

    Examples:
        >>> manifest = Manifest(
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
        ... )
        >>> manifest.resolver
        'obo'
    """

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

    def to_json(self) -> str:
        """Serialize the manifest to a stable, human-readable JSON string.

        Args:
            None

        Returns:
            JSON document encoding the manifest metadata.
        """
        payload = {
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


DATA_ROOT = pystow.join("ontology-fetcher")
CONFIG_DIR = DATA_ROOT / "configs"
CACHE_DIR = DATA_ROOT / "cache"
LOG_DIR = DATA_ROOT / "logs"
ONTOLOGY_DIR = DATA_ROOT / "ontologies"

for directory in (CONFIG_DIR, CACHE_DIR, LOG_DIR, ONTOLOGY_DIR):
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:  # pragma: no cover - dependent on environment
        message = f"Permission denied writing to {directory}. Set PYSTOW_HOME env var"
        logging.getLogger("DocsToKG.OntologyDownload").error(
            "pystow directory permission error",
            extra={"stage": "startup", "path": str(directory)},
        )
        print(message, file=sys.stderr)
        raise SystemExit(1) from exc


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
    if not isinstance(manifest.validation, dict):
        raise ConfigurationError("Manifest validation payload must be a dictionary")
    if not isinstance(manifest.artifacts, Sequence):
        raise ConfigurationError("Manifest artifacts must be a sequence of paths")
    for item in manifest.artifacts:
        if not isinstance(item, str):
            raise ConfigurationError("Manifest artifacts must contain only string paths")
    if manifest.normalized_sha256 is not None and not isinstance(manifest.normalized_sha256, str):
        raise ConfigurationError("Manifest normalized_sha256 must be a string when provided")
    if manifest.fingerprint is not None and not isinstance(manifest.fingerprint, str):
        raise ConfigurationError("Manifest fingerprint must be a string when provided")


def _write_manifest(manifest_path: Path, manifest: Manifest) -> None:
    """Persist a validated manifest to disk as JSON.

    Args:
        manifest_path: Destination path for the manifest file.
        manifest: Manifest describing the downloaded ontology artifact.
    """
    _validate_manifest(manifest)
    manifest_path.write_text(manifest.to_json())


def _build_destination(spec: FetchSpec, plan: FetchPlan, config: ResolvedConfig) -> Path:
    """Determine the output directory and filename for a download.

    Args:
        spec: Fetch specification identifying the ontology.
        plan: Resolver plan containing URL metadata and optional hints.
        config: Resolved configuration with storage layout parameters.

    Returns:
        Path where the raw ontology artifact should be written.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = plan.version or timestamp
    base_dir = ONTOLOGY_DIR / sanitize_filename(spec.id) / sanitize_filename(version)
    for subdir in ("original", "normalized", "validation"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    parsed = urlparse(plan.url)
    candidate = Path(parsed.path).name if parsed.path else f"{spec.id}.owl"
    filename = plan.filename_hint or sanitize_filename(candidate)
    return base_dir / "original" / filename


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
        normalize_license_to_spdx(entry) or entry
        for entry in config.defaults.accept_licenses
    }
    plan_license = normalize_license_to_spdx(plan.license)
    if not allowed or plan_license is None:
        return
    if plan_license not in allowed:
        raise ConfigurationError(
            f"License '{plan.license}' for ontology '{spec.id}' is not in the allowlist: {sorted(allowed)}"
        )


def fetch_one(
    spec: FetchSpec,
    *,
    config: Optional[ResolvedConfig] = None,
    correlation_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> FetchResult:
    """Fetch and validate a single ontology described by *spec*.

    Args:
        spec: Fetch specification outlining resolver and target formats.
        config: Optional resolved configuration; defaults to library values.
        correlation_id: Identifier that groups log entries for observability.
        logger: Optional logger to reuse existing logging infrastructure.
        force: When True, ignore existing manifests and re-download artifacts.

    Returns:
        FetchResult capturing download metadata and produced artifacts.

    Raises:
        ResolverError: If the resolver cannot produce a viable fetch plan.
        OntologyDownloadError: If download or extraction steps fail.
        ConfigurationError: If manifest validation or license checks fail.
    """

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    resolver = RESOLVERS.get(spec.resolver)
    if resolver is None:
        raise ResolverError(f"Unknown resolver '{spec.resolver}' for ontology '{spec.id}'")

    log = logger or setup_logging(active_config.defaults.logging)
    correlation = correlation_id or generate_correlation_id()
    adapter = logging.LoggerAdapter(
        log, extra={"correlation_id": correlation, "ontology_id": spec.id}
    )
    adapter.info("planning fetch", extra={"stage": "plan"})

    try:
        plan = resolver.plan(spec, active_config, adapter)
    except ConfigError as exc:
        raise ResolverError(
            f"Resolver '{spec.resolver}' failed for ontology '{spec.id}': {exc}"
        ) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise ResolverError(f"Unexpected resolver failure for ontology '{spec.id}': {exc}") from exc

    _ensure_license_allowed(plan, active_config, spec)

    if plan.service:
        adapter.extra["service"] = plan.service
    else:
        adapter.extra.pop("service", None)

    destination = _build_destination(spec, plan, active_config)
    manifest_path = destination.parent.parent / "manifest.json"
    previous_manifest = None if force else _read_manifest(manifest_path)

    adapter.info(
        "downloading",
        extra={
            "stage": "download",
            "url": plan.url,
            "destination": str(destination),
            "version": plan.version,
        },
    )

    download_config = active_config.defaults.http
    secure_url = validate_url_security(plan.url, download_config)
    try:
        result = download_stream(
            url=secure_url,
            destination=destination,
            headers=plan.headers,
            previous_manifest=previous_manifest,
            http_config=download_config,
            cache_dir=CACHE_DIR,
            logger=adapter,
            expected_media_type=plan.media_type,
            service=plan.service,
        )
    except ConfigError as exc:
        raise OntologyDownloadError(f"Download failed for '{spec.id}': {exc}") from exc

    validation_requests: List[ValidationRequest] = []
    base_dir = destination.parent.parent
    normalized_dir = base_dir / "normalized"
    validation_dir = base_dir / "validation"
    validation_requests.append(
        ValidationRequest(
            name="rdflib",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        )
    )
    validation_requests.append(
        ValidationRequest(
            name="pronto",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        )
    )
    validation_requests.append(
        ValidationRequest(
            name="owlready2",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        )
    )
    validation_requests.append(
        ValidationRequest(
            name="robot",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        )
    )
    validation_requests.append(
        ValidationRequest(
            name="arelle",
            file_path=destination,
            normalized_dir=normalized_dir,
            validation_dir=validation_dir,
            config=active_config,
        )
    )

    artifacts = [str(destination)]
    if plan.media_type == "application/zip" or destination.suffix.lower() == ".zip":
        extraction_dir = destination.parent / f"{destination.stem}_extracted"
        try:
            extracted_paths = extract_zip_safe(destination, extraction_dir, logger=adapter)
            artifacts.extend(str(path) for path in extracted_paths)
        except ConfigError as exc:
            adapter.error(
                "zip extraction failed",
                extra={"stage": "extract", "error": str(exc)},
            )
            if not active_config.defaults.continue_on_error:
                raise OntologyDownloadError(f"Extraction failed for '{spec.id}': {exc}") from exc

    validation_results = run_validators(validation_requests, adapter)

    normalized_hash = None
    rdflib_result = validation_results.get("rdflib")
    if rdflib_result and isinstance(rdflib_result.details, dict):
        maybe_hash = rdflib_result.details.get("normalized_sha256")
        if isinstance(maybe_hash, str):
            normalized_hash = maybe_hash

    fingerprint_components = [
        spec.id,
        spec.resolver,
        plan.version or "",
        result.sha256,
        normalized_hash or "",
        secure_url,
    ]
    fingerprint = hashlib.sha256("|".join(fingerprint_components).encode("utf-8")).hexdigest()

    manifest = Manifest(
        id=spec.id,
        resolver=spec.resolver,
        url=secure_url,
        filename=destination.name,
        version=plan.version,
        license=plan.license,
        status=result.status,
        sha256=result.sha256,
        normalized_sha256=normalized_hash,
        fingerprint=fingerprint,
        etag=result.etag,
        last_modified=result.last_modified,
        downloaded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        target_formats=spec.target_formats,
        validation=validation_results,
        artifacts=artifacts,
    )
    _write_manifest(manifest_path, manifest)
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
        spec=spec,
        local_path=destination,
        status=result.status,
        sha256=result.sha256,
        manifest_path=manifest_path,
        artifacts=artifacts,
    )


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
            log = setup_logging(active_config.defaults.logging)
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
    "Resolver",
    "fetch_one",
    "fetch_all",
    "OntologyDownloadError",
    "ResolverError",
    "ValidationError",
    "ConfigurationError",
]
