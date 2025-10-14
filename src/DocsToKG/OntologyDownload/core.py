"""
Ontology Download Orchestration

This module coordinates resolver planning, document downloading, validation,
and manifest generation for ontology artifacts. It serves as the main entry
point for fetching ontologies from configured sources and producing provenance
metadata that downstream knowledge graph construction can rely upon.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence
from urllib.parse import urlparse

import pystow

from .config import ConfigError, ResolvedConfig, ensure_python_version
from .download import (
    DownloadResult,
    download_stream,
    extract_zip_safe,
    sanitize_filename,
    validate_url_security,
)
from .logging_config import generate_correlation_id, setup_logging
from .resolvers import RESOLVERS, FetchPlan
from .validators import ValidationRequest, ValidationResult, run_validators


class OntologyDownloadError(RuntimeError):
    """Base exception for ontology download failures."""


class ResolverError(OntologyDownloadError):
    """Raised when resolver planning fails."""


class ValidationError(OntologyDownloadError):
    """Raised when validation encounters unrecoverable issues."""


class ConfigurationError(OntologyDownloadError):
    """Raised when configuration or manifest validation fails."""


@dataclass(slots=True)
class FetchSpec:
    """Specification describing a single ontology download.

    Attributes:
        id: Stable identifier for the ontology to fetch.
        resolver: Name of the resolver strategy used to locate resources.
        extras: Resolver-specific configuration overrides.
        target_formats: Normalized ontology formats that should be produced.
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
        etag: HTTP ETag returned by the upstream server, when provided.
        last_modified: Upstream last-modified timestamp, if supplied.
        downloaded_at: UTC timestamp of the completed download.
        target_formats: Desired conversion targets for normalization.
        validation: Mapping of validator names to their results.
        artifacts: Additional file paths generated during processing.
    """

    id: str
    resolver: str
    url: str
    filename: str
    version: Optional[str]
    license: Optional[str]
    status: str
    sha256: str
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
            "etag": self.etag,
            "last_modified": self.last_modified,
            "downloaded_at": self.downloaded_at,
            "target_formats": list(self.target_formats),
            "validation": {name: result.to_dict() for name, result in self.validation.items()},
            "artifacts": list(self.artifacts),
        }
        return json.dumps(payload, indent=2, sort_keys=True)


class Resolver(Protocol):
    """Protocol describing resolver planning behaviour."""

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
    directory.mkdir(parents=True, exist_ok=True)


def _read_manifest(manifest_path: Path) -> Optional[dict]:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None


def _validate_manifest(manifest: Manifest) -> None:
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


def _write_manifest(manifest_path: Path, manifest: Manifest) -> None:
    _validate_manifest(manifest)
    manifest_path.write_text(manifest.to_json())


def _build_destination(spec: FetchSpec, plan: FetchPlan, config: ResolvedConfig) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    version = plan.version or timestamp
    base_dir = ONTOLOGY_DIR / sanitize_filename(spec.id) / sanitize_filename(version)
    for subdir in ("original", "normalized", "validation"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    parsed = urlparse(plan.url)
    candidate = Path(parsed.path).name if parsed.path else f"{spec.id}.owl"
    filename = plan.filename_hint or sanitize_filename(candidate)
    return base_dir / "original" / filename


def _ensure_license_allowed(plan: FetchPlan, config: ResolvedConfig, spec: FetchSpec) -> None:
    allowed = set(config.defaults.accept_licenses)
    if not allowed or plan.license is None:
        return
    if plan.license not in allowed:
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
    adapter = logging.LoggerAdapter(log, extra={"correlation_id": correlation, "ontology_id": spec.id})
    adapter.info("planning fetch", extra={"stage": "plan"})

    try:
        plan = resolver.plan(spec, active_config, adapter)
    except ConfigError as exc:
        raise ResolverError(f"Resolver '{spec.resolver}' failed for ontology '{spec.id}': {exc}") from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise ResolverError(f"Unexpected resolver failure for ontology '{spec.id}': {exc}") from exc

    _ensure_license_allowed(plan, active_config, spec)

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
    secure_url = validate_url_security(plan.url)
    try:
        result = download_stream(
            url=secure_url,
            destination=destination,
            headers=plan.headers,
            previous_manifest=previous_manifest,
            http_config=download_config,
            cache_dir=CACHE_DIR,
            logger=adapter,
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

    manifest = Manifest(
        id=spec.id,
        resolver=spec.resolver,
        url=secure_url,
        filename=destination.name,
        version=plan.version,
        license=plan.license,
        status=result.status,
        sha256=result.sha256,
        etag=result.etag,
        last_modified=result.last_modified,
        downloaded_at=datetime.utcnow().isoformat() + "Z",
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
    log = logger or setup_logging(active_config.defaults.logging)
    correlation = generate_correlation_id()
    adapter = logging.LoggerAdapter(log, extra={"correlation_id": correlation})

    spec_list = list(specs)
    total = len(spec_list)
    results: List[FetchResult] = []
    for index, spec in enumerate(spec_list, start=1):
        adapter.info(
            "starting ontology fetch",
            extra={
                "stage": "start",
                "ontology_id": spec.id,
                "progress": {"current": index, "total": total},
            },
        )
        try:
            result = fetch_one(
                spec,
                config=active_config,
                correlation_id=correlation,
                logger=log,
                force=force,
            )
            results.append(result)
            adapter.info(
                "progress update",
                extra={
                    "stage": "progress",
                    "ontology_id": spec.id,
                    "progress": {"current": index, "total": total},
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            adapter.error(
                "ontology fetch failed",
                extra={"stage": "error", "ontology_id": spec.id, "error": str(exc)},
            )
            if not active_config.defaults.continue_on_error:
                raise
    return results


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
