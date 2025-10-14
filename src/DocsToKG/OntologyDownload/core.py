"""Core orchestration for ontology downloads."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

import pystow

from .config import ConfigError, ResolvedConfig, ensure_python_version
from .download import DownloadResult, download_stream, sanitize_filename, validate_url_security
from .logging_config import generate_correlation_id, setup_logging
from .resolvers import RESOLVERS, FetchPlan
from .validators import (
    ValidationRequest,
    ValidationResult,
    run_validators,
)


@dataclass(slots=True)
class FetchSpec:
    """Specification describing a single ontology download."""

    id: str
    resolver: str
    extras: Dict[str, object]
    target_formats: Sequence[str]


@dataclass(slots=True)
class FetchResult:
    """Outcome of a single ontology fetch operation."""

    spec: FetchSpec
    local_path: Path
    status: str
    sha256: str
    manifest_path: Path


@dataclass(slots=True)
class Manifest:
    """Provenance information for a downloaded ontology artifact."""

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

    def to_json(self) -> str:
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
        }
        return json.dumps(payload, indent=2, sort_keys=True)


class Resolver(Protocol):
    """Protocol describing resolver planning behaviour."""

    def plan(self, spec: FetchSpec, config: ResolvedConfig, logger: logging.Logger) -> FetchPlan:
        """Return a FetchPlan describing how to obtain the ontology."""


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


def _write_manifest(manifest_path: Path, manifest: Manifest) -> None:
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
        raise ConfigError(
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
    """Fetch and validate a single ontology described by *spec*."""

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    resolver = RESOLVERS.get(spec.resolver)
    if resolver is None:
        raise ConfigError(f"Unknown resolver '{spec.resolver}' for ontology '{spec.id}'")

    log = logger or setup_logging(active_config.defaults.logging)
    correlation = correlation_id or generate_correlation_id()
    adapter = logging.LoggerAdapter(log, extra={"correlation_id": correlation, "ontology_id": spec.id})
    adapter.info("planning fetch", extra={"stage": "plan"})

    plan = resolver.plan(spec, active_config, adapter)
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
    result = download_stream(
        url=secure_url,
        destination=destination,
        headers=plan.headers,
        previous_manifest=previous_manifest,
        http_config=download_config,
        cache_dir=CACHE_DIR,
        logger=adapter,
    )

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

    return FetchResult(spec=spec, local_path=destination, status=result.status, sha256=result.sha256, manifest_path=manifest_path)


def fetch_all(
    specs: Iterable[FetchSpec],
    *,
    config: Optional[ResolvedConfig] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> List[FetchResult]:
    """Fetch a sequence of ontologies sequentially."""

    ensure_python_version()
    active_config = config or ResolvedConfig.from_defaults()
    log = logger or setup_logging(active_config.defaults.logging)
    correlation = generate_correlation_id()
    adapter = logging.LoggerAdapter(log, extra={"correlation_id": correlation})

    results: List[FetchResult] = []
    for spec in specs:
        adapter.info("starting ontology fetch", extra={"stage": "start", "ontology_id": spec.id})
        try:
            result = fetch_one(
                spec,
                config=active_config,
                correlation_id=correlation,
                logger=log,
                force=force,
            )
            results.append(result)
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
]
