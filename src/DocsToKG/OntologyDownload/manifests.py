"""Manifest and lockfile utilities for the OntologyDownload CLI."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .errors import ConfigError
from .planning import (
    FetchResult,
    FetchSpec,
    PlannedFetch,
    infer_version_timestamp,
    parse_iso_datetime,
)
from .settings import CACHE_DIR, LOCAL_ONTOLOGY_DIR, STORAGE, DefaultsConfig

DEFAULT_LOCKFILE_PATH = Path("ontologies.lock.json")
DEFAULT_PLAN_BASELINE = CACHE_DIR / "plans" / "baseline.json"
LOCKFILE_SCHEMA_VERSION = "1.0"
PLAN_DIFF_FIELDS: Tuple[str, ...] = (
    "url",
    "version",
    "license",
    "media_type",
    "service",
    "last_modified",
    "content_length",
)

__all__ = [
    "DEFAULT_LOCKFILE_PATH",
    "DEFAULT_PLAN_BASELINE",
    "LOCKFILE_SCHEMA_VERSION",
    "PLAN_DIFF_FIELDS",
    "collect_version_metadata",
    "compute_plan_diff",
    "ensure_manifest_path",
    "format_plan_diff",
    "load_latest_manifest",
    "load_lockfile_payload",
    "load_manifest",
    "plan_to_dict",
    "resolve_version_metadata",
    "results_to_dict",
    "spec_from_lock_entry",
    "specs_from_lock_payload",
    "write_json_atomic",
    "write_lockfile",
]


def plan_to_dict(plan: PlannedFetch) -> dict:
    """Convert a planned fetch into a JSON-friendly dictionary."""

    candidates: List[dict] = []
    for candidate in getattr(plan, "candidates", ()):
        candidate_payload = {
            "resolver": candidate.resolver,
            "url": candidate.plan.url,
            "service": candidate.plan.service,
            "media_type": candidate.plan.media_type,
            "headers": candidate.plan.headers,
            "version": candidate.plan.version,
            "license": candidate.plan.license,
            "last_modified": candidate.plan.last_modified,
        }
        if candidate.plan.content_length is not None:
            candidate_payload["content_length"] = candidate.plan.content_length
        candidate_metadata = getattr(candidate, "metadata", None)
        if candidate_metadata is None:
            candidate_metadata = getattr(candidate.plan, "metadata", None)
        if isinstance(candidate_metadata, dict):
            etag_value = candidate_metadata.get("etag")
            if etag_value:
                candidate_payload["etag"] = etag_value
        candidates.append(candidate_payload)

    payload = {
        "id": plan.spec.id,
        "resolver": plan.resolver,
        "url": plan.plan.url,
        "version": plan.plan.version,
        "license": plan.plan.license,
        "media_type": plan.plan.media_type,
        "service": plan.plan.service,
        "headers": plan.plan.headers,
        "candidates": candidates,
    }
    if plan.spec.target_formats:
        payload["target_formats"] = list(plan.spec.target_formats)

    metadata = getattr(plan, "metadata", {}) or {}
    last_modified = metadata.get("last_modified")
    if last_modified:
        payload["last_modified"] = last_modified
    size_hint = metadata.get("content_length")
    if size_hint is None:
        size_hint = getattr(plan, "size", None) or plan.plan.content_length
    if size_hint is not None:
        payload["content_length"] = int(size_hint)
    if metadata.get("etag"):
        payload["etag"] = metadata["etag"]
    expected_checksum = metadata.get("expected_checksum")
    if isinstance(expected_checksum, dict):
        payload["expected_checksum"] = expected_checksum

    payload["spec"] = {
        "id": plan.spec.id,
        "resolver": plan.spec.resolver,
        "extras": dict(plan.spec.extras),
        "target_formats": list(plan.spec.target_formats),
    }
    payload["plan"] = {
        "resolver": plan.resolver,
        "url": plan.plan.url,
        "version": plan.plan.version,
        "license": plan.plan.license,
        "media_type": plan.plan.media_type,
        "service": plan.plan.service,
        "headers": plan.plan.headers,
        "candidates": candidates,
    }
    if last_modified:
        payload["plan"]["last_modified"] = last_modified
    if size_hint is not None:
        payload["plan"]["content_length"] = int(size_hint)
    if metadata.get("etag"):
        payload["plan"]["etag"] = metadata["etag"]
    if isinstance(expected_checksum, dict):
        payload["plan"]["expected_checksum"] = expected_checksum

    return payload


def write_json_atomic(path: Path, payload: object) -> Path:
    """Atomically persist ``payload`` as JSON to ``path``."""

    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(resolved.parent), delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except (AttributeError, OSError):
            pass
        temp_name = handle.name
    Path(temp_name).replace(resolved)
    return resolved


def write_lockfile(plans: Sequence[PlannedFetch], path: Path) -> Path:
    """Write lockfile capturing planned resolver outputs."""

    payload = {
        "schema_version": LOCKFILE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "entries": [plan_to_dict(plan) for plan in plans],
    }
    written_path = write_json_atomic(path, payload)
    logging.getLogger("DocsToKG.OntologyDownload").info(
        "lockfile written",
        extra={"stage": "plan", "lock_path": str(written_path), "entries": len(plans)},
    )
    return written_path


def load_lockfile_payload(path: Path) -> Dict[str, object]:
    """Return the parsed lockfile payload."""

    resolved = path.expanduser()
    try:
        payload = json.loads(resolved.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"Lock file not found: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Lock file {resolved} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ConfigError("Lock file must contain a JSON object")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ConfigError("Lock file missing 'entries' array")
    schema_version = payload.get("schema_version")
    if schema_version and schema_version != LOCKFILE_SCHEMA_VERSION:
        logging.getLogger("DocsToKG.OntologyDownload").warning(
            "lockfile schema mismatch",
            extra={
                "stage": "plan",
                "expected": LOCKFILE_SCHEMA_VERSION,
                "found": schema_version,
                "lock_path": str(resolved),
            },
        )
    return payload


def spec_from_lock_entry(entry: Dict[str, object], defaults: DefaultsConfig) -> FetchSpec:
    """Convert a lockfile entry back into a fetch specification."""

    if not isinstance(entry, dict):
        raise ConfigError("Lock file entry must be an object")
    ontology_id = entry.get("id")
    if not isinstance(ontology_id, str) or not ontology_id:
        raise ConfigError("Lock file entry missing 'id'")
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        raise ConfigError(f"Lock file entry for '{ontology_id}' is missing 'url'")

    extras: Dict[str, object] = {"url": url}
    headers = entry.get("headers")
    if isinstance(headers, Mapping):
        extras["headers"] = {str(key): str(value) for key, value in headers.items()}
    media_type = entry.get("media_type")
    if isinstance(media_type, str):
        extras["media_type"] = media_type
    service = entry.get("service")
    if isinstance(service, str):
        extras["service"] = service
    version = entry.get("version")
    if isinstance(version, str):
        extras["version"] = version
    license_value = entry.get("license")
    if isinstance(license_value, str):
        extras["license"] = license_value
    expected_checksum = entry.get("expected_checksum")
    if isinstance(expected_checksum, Mapping):
        algorithm = expected_checksum.get("algorithm")
        value = expected_checksum.get("value")
        if isinstance(algorithm, str) and isinstance(value, str):
            extras["checksum"] = {"algorithm": algorithm, "value": value}
    origin_resolver = entry.get("resolver")
    if isinstance(origin_resolver, str):
        extras.setdefault("origin_resolver", origin_resolver)

    target_formats_raw = entry.get("target_formats")
    if isinstance(target_formats_raw, str):
        target_formats = [target_formats_raw]
    elif isinstance(target_formats_raw, Sequence):
        target_formats = [str(item) for item in target_formats_raw]
    else:
        target_formats = list(defaults.normalize_to)
    if not target_formats:
        target_formats = list(defaults.normalize_to)

    return FetchSpec(
        id=ontology_id,
        resolver="direct",
        extras=extras,
        target_formats=tuple(target_formats),
    )


def specs_from_lock_payload(
    payload: Dict[str, object],
    *,
    defaults: DefaultsConfig,
    requested_ids: Sequence[str],
) -> List[FetchSpec]:
    """Build fetch specifications from lockfile payload."""

    entries = payload.get("entries") or []
    if requested_ids:
        requested = set(requested_ids)
        indexed = {entry.get("id"): entry for entry in entries if isinstance(entry, dict)}
        missing = sorted(requested - set(indexed))
        if missing:
            raise ConfigError(f"Lock file does not contain entries for: {', '.join(missing)}")
        selected = [indexed[oid] for oid in requested_ids]
    else:
        selected = [entry for entry in entries if isinstance(entry, dict)]
    return [spec_from_lock_entry(entry, defaults) for entry in selected]


def resolve_version_metadata(
    ontology_id: str, version: str
) -> Tuple[Path, Optional[datetime], int]:
    """Return path, timestamp, and size metadata for a stored version."""

    path = STORAGE.version_path(ontology_id, version)
    manifest_path = path / "manifest.json"
    timestamp: Optional[datetime] = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
        else:
            timestamp = parse_iso_datetime(manifest.get("downloaded_at"))
    if timestamp is None:
        timestamp = infer_version_timestamp(version)
    if timestamp is None:
        try:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            timestamp = None
    size = STORAGE.directory_size(path)
    return path, timestamp, size


def ensure_manifest_path(ontology_id: str, version: Optional[str]) -> Path:
    """Return the manifest path for a given ontology and version."""

    selected_version = version
    if selected_version:
        local_dir = STORAGE.ensure_local_version(ontology_id, selected_version)
    else:
        versions = STORAGE.available_versions(ontology_id)
        if not versions:
            raise ConfigError(f"No versions stored for ontology '{ontology_id}'")
        selected_version = versions[-1]
        local_dir = STORAGE.ensure_local_version(ontology_id, selected_version)
    manifest_path = local_dir / "manifest.json"
    if not manifest_path.exists():
        raise ConfigError(f"Manifest not found for ontology '{ontology_id}' at {selected_version}")
    return manifest_path


def load_manifest(manifest_path: Path) -> dict:
    """Read and parse a manifest JSON document from disk."""

    return json.loads(manifest_path.read_text())


def collect_version_metadata(ontology_id: str) -> List[Dict[str, object]]:
    """Return sorted metadata entries for stored ontology versions."""

    from .io import sanitize_filename  # Lazy import to avoid circular dependency

    safe_id = sanitize_filename(ontology_id)
    base_dir = LOCAL_ONTOLOGY_DIR / safe_id
    metadata: List[Dict[str, object]] = []
    for version in STORAGE.available_versions(ontology_id):
        if version == "latest":
            continue
        safe_version = sanitize_filename(version)
        version_dir = base_dir / safe_version
        manifest_path = version_dir / "manifest.json"
        timestamp = None
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                manifest_data = {}
            timestamp = parse_iso_datetime((manifest_data or {}).get("downloaded_at"))
            if timestamp is None:
                timestamp = parse_iso_datetime((manifest_data or {}).get("last_modified"))
        if timestamp is None:
            if manifest_path.exists():
                timestamp = datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc)
            elif version_dir.exists():
                timestamp = datetime.fromtimestamp(version_dir.stat().st_mtime, tz=timezone.utc)
        size = STORAGE.directory_size(version_dir)
        metadata.append(
            {
                "id": ontology_id,
                "version": version,
                "timestamp": timestamp,
                "size": size,
                "path": version_dir,
            }
        )

    metadata.sort(
        key=lambda item: item["timestamp"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return metadata


def load_latest_manifest(ontology_id: str) -> Optional[Dict[str, object]]:
    """Return the most recent manifest for ``ontology_id`` when available."""

    for entry in collect_version_metadata(ontology_id):
        version_dir = entry.get("path")
        if not isinstance(version_dir, Path):
            continue
        manifest_path = version_dir / "manifest.json"
        if manifest_path.exists():
            try:
                return json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                continue
    return None


def results_to_dict(result: FetchResult) -> dict:
    """Serialize a :class:`FetchResult` to a JSON-friendly dictionary."""

    return {
        "id": result.spec.id,
        "resolver": result.spec.resolver,
        "status": result.status,
        "sha256": result.sha256,
        "path": str(result.local_path),
        "expected_checksum": (
            result.expected_checksum.to_known_hash()
            if getattr(result, "expected_checksum", None)
            else None
        ),
    }


def compute_plan_diff(baseline: Sequence[dict], current: Sequence[dict]) -> Dict[str, object]:
    """Compute a diff between baseline and current plan payloads."""

    baseline_index = OrderedDict(
        (entry["id"], entry) for entry in baseline if isinstance(entry, dict)
    )
    current_index = OrderedDict(
        (entry["id"], entry) for entry in current if isinstance(entry, dict)
    )

    added = [current_index[key] for key in current_index.keys() - baseline_index.keys()]
    removed = [baseline_index[key] for key in baseline_index.keys() - current_index.keys()]
    modified: List[Dict[str, object]] = []

    for key in baseline_index.keys() & current_index.keys():
        before = baseline_index[key]
        after = current_index[key]
        changes: MutableMapping[str, Dict[str, object]] = {}
        for field in PLAN_DIFF_FIELDS:
            before_value = before.get(field)
            after_value = after.get(field)
            if before_value != after_value:
                changes[field] = {"before": before_value, "after": after_value}
        if changes:
            modified.append({"id": key, "changes": changes})

    return {"added": added, "removed": removed, "modified": modified}


def format_plan_diff(diff: Dict[str, object]) -> List[str]:
    """Render human-readable diff lines from plan comparison."""

    lines: List[str] = []
    for entry in diff.get("added", []):
        version = entry.get("version") or "unknown"
        lines.append(f"+ {entry.get('id')} version={version} url={entry.get('url', '')}")
    for entry in diff.get("removed", []):
        version = entry.get("version") or "unknown"
        lines.append(f"- {entry.get('id')} version={version} url={entry.get('url', '')}")
    for entry in diff.get("modified", []):
        changes = entry.get("changes", {})
        parts = [
            f"{field}: {payload.get('before')} -> {payload.get('after')}"
            for field, payload in changes.items()
        ]
        detail = ", ".join(parts) if parts else "no changes"
        lines.append(f"~ {entry.get('id')} {detail}")
    return lines
