"""Ontology downloader CLI entry points.

The `ontofetch` command exposes planning, pull, validation, diagnostics, and
storage management workflows described in the ontology download refactor.
Operators can override concurrency and host allowlists from the CLI, diff plan
outputs, prune historical versions, and run comprehensive `doctor` diagnostics
without editing configuration files. JSON output modes support automation while
rich ASCII tables summarise resolver fallback chains and validator results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .ontology_download import (
    CACHE_DIR,
    CONFIG_DIR,
    LOCAL_ONTOLOGY_DIR,
    LOG_DIR,
    MANIFEST_SCHEMA_VERSION,
    STORAGE,
    ConfigError,
    ConfigurationError,
    FetchResult,
    FetchSpec,
    OntologyDownloadError,
    PlannedFetch,
    ResolvedConfig,
    ValidationRequest,
    fetch_all,
    get_manifest_schema,
    load_config,
    infer_version_timestamp,
    parse_iso_datetime,
    parse_rate_limit_to_rps,
    parse_version_timestamp,
    plan_all,
    run_validators,
    sanitize_filename,
    setup_logging,
    validate_config,
    validate_manifest_dict,
    _directory_size,
)

ONTOLOGY_DIR = LOCAL_ONTOLOGY_DIR

DEFAULT_PLAN_BASELINE = CACHE_DIR / "plans" / "baseline.json"

PLAN_DIFF_FIELDS = (
    "url",
    "version",
    "license",
    "media_type",
    "service",
    "last_modified",
    "content_length",
)


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a padded ASCII table.

    Args:
        headers: Column headers rendered on the first row.
        rows: Iterable of table rows; each row must match the header length.

    Returns:
        Formatted table as a multi-line string.
    """

    column_widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            column_widths[index] = max(column_widths[index], len(cell))

    def _format_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(column_widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in column_widths)
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def format_plan_rows(plans: Iterable[PlannedFetch]) -> List[Tuple[str, str, str, str, str]]:
    """Convert planner output into rows for table rendering.

    Args:
        plans: Planned fetch objects emitted by the planner.

    Returns:
        List of tuples suitable for :func:`format_table`.
    """

    rows: List[Tuple[str, str, str, str, str]] = []
    for plan in plans:
        rows.append(
            (
                plan.spec.id,
                plan.resolver,
                plan.plan.service or "",
                plan.plan.media_type or "",
                plan.plan.url,
            )
        )
    return rows


def format_results_table(results: Iterable[FetchResult]) -> str:
    """Render download results as a table summarizing status and file paths.

    Args:
        results: Download results produced by :func:`fetch_all`.

    Returns:
        Formatted table summarizing ontology downloads.
    """

    rows = [
        (
            result.spec.id,
            result.spec.resolver,
            result.status,
            result.sha256,
            str(result.local_path),
        )
        for result in results
    ]
    return format_table(("id", "resolver", "status", "sha256", "file"), rows)


def format_validation_summary(results: Dict[str, Dict[str, Any]]) -> str:
    """Summarise validator outcomes in a compact status table.

    Args:
        results: Mapping of validator names to structured result payloads.

    Returns:
        Table highlighting validator status and notable detail messages.
    """

    formatted: List[Tuple[str, str, str]] = []
    for name, payload in results.items():
        status = "ok" if payload.get("ok") else "error"
        details = payload.get("details", {})
        message = ""
        if isinstance(details, dict):
            if "error" in details:
                message = str(details["error"])
            elif details:
                message = ", ".join(f"{key}={value}" for key, value in details.items())
        formatted.append((name, status, message))
    return format_table(("validator", "status", "details"), formatted)


def _build_parser() -> argparse.ArgumentParser:
    """Configure the top-level CLI parser and subcommands.

    Returns:
        Argument parser providing the consolidated ``ontofetch`` CLI.
    """

    parser = argparse.ArgumentParser(
        prog="ontofetch",
        description="Ontology downloader for DocsToKG supporting OBO, OLS, BioPortal, SKOS, and XBRL sources.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging verbosity",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull = subparsers.add_parser(
        "pull",
        help="Download ontologies",
        description="Download ontologies from configuration or ad-hoc specification.",
    )
    pull.add_argument("ids", nargs="*", help="Ontology identifiers to download")
    pull.add_argument(
        "--spec",
        type=Path,
        help="Path to sources.yaml (default: configs/sources.yaml)",
    )
    pull.add_argument("--force", action="store_true", help="Force redownload bypassing cache")
    pull.add_argument("--resolver", help="Resolver type for single ontology")
    pull.add_argument(
        "--target-formats",
        help="Comma-separated formats (e.g., owl,obo)",
    )
    pull.add_argument("--json", action="store_true", help="Emit pull results as JSON")
    pull.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview resolver actions without downloading",
    )
    pull.add_argument(
        "--concurrent-downloads",
        type=_parse_positive_int,
        help="Override maximum concurrent downloads for this invocation",
    )
    pull.add_argument(
        "--concurrent-plans",
        type=_parse_positive_int,
        help="Override maximum concurrent planning workers (useful with --dry-run)",
    )
    pull.add_argument(
        "--allowed-hosts",
        help="Comma-separated list of additional hosts permitted for this run",
    )

    plan_cmd = subparsers.add_parser(
        "plan",
        help="Preview resolver plans without downloading",
        description="Plan resolver actions without performing downloads.",
    )
    plan_cmd.add_argument("ids", nargs="*", help="Ontology identifiers to plan")
    plan_cmd.add_argument(
        "--spec",
        type=Path,
        help="Path to sources.yaml (default: configs/sources.yaml)",
    )
    plan_cmd.add_argument("--resolver", help="Resolver type for single ontology")
    plan_cmd.add_argument("--target-formats", help="Comma-separated formats (e.g., owl,obo)")
    plan_cmd.add_argument(
        "--since",
        type=_parse_since_arg,
        help="Only include ontologies modified on or after YYYY-MM-DD",
    )
    plan_cmd.add_argument(
        "--concurrent-plans",
        type=_parse_positive_int,
        help="Override maximum concurrent resolver planning workers",
    )
    plan_cmd.add_argument(
        "--concurrent-downloads",
        type=_parse_positive_int,
        help="Override concurrent downloads when using --dry-run",
    )
    plan_cmd.add_argument(
        "--allowed-hosts",
        help="Comma-separated list of additional hosts permitted for this run",
    )
    plan_cmd.add_argument("--json", action="store_true", help="Emit plan details as JSON")

    plan_diff = subparsers.add_parser(
        "plan-diff",
        help="Compare current plan output to a previous baseline",
        description="Generate a plan and compare it to a stored baseline to highlight differences.",
    )
    plan_diff.add_argument("ids", nargs="*", help="Ontology identifiers to include in the plan")
    plan_diff.add_argument(
        "--spec",
        type=Path,
        help="Path to sources.yaml (default: configs/sources.yaml)",
    )
    plan_diff.add_argument("--resolver", help="Resolver type for single ontology")
    plan_diff.add_argument("--target-formats", help="Comma-separated formats (e.g., owl,obo)")
    plan_diff.add_argument(
        "--since",
        type=_parse_since_arg,
        help="Only include ontologies modified on or after YYYY-MM-DD",
    )
    plan_diff.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_PLAN_BASELINE,
        help=f"Baseline plan JSON file (default: {DEFAULT_PLAN_BASELINE})",
    )
    plan_diff.add_argument(
        "--concurrent-plans",
        type=_parse_positive_int,
        help="Override maximum concurrent resolver planning workers",
    )
    plan_diff.add_argument(
        "--concurrent-downloads",
        type=_parse_positive_int,
        help="Override concurrent downloads when using --dry-run",
    )
    plan_diff.add_argument(
        "--allowed-hosts",
        help="Comma-separated list of additional hosts permitted for this run",
    )
    plan_diff.add_argument("--json", action="store_true", help="Emit plan diff as JSON")

    show = subparsers.add_parser("show", help="Display ontology metadata")
    show.add_argument("id", help="Ontology identifier")
    show.add_argument("--versions", action="store_true", help="List available versions")
    show.add_argument("--json", action="store_true", help="Output manifest as JSON")

    validate = subparsers.add_parser("validate", help="Re-run validation on downloaded ontologies")
    validate.add_argument("id", help="Ontology identifier")
    validate.add_argument("version", nargs="?", help="Specific version to validate")
    validate.add_argument("--json", action="store_true", help="Emit validation summary as JSON")
    validate.add_argument("--rdflib", action="store_true", help="Run only the RDFLib validator")
    validate.add_argument("--pronto", action="store_true", help="Include Pronto validation")
    validate.add_argument("--owlready2", action="store_true", help="Include Owlready2 validation")
    validate.add_argument("--robot", action="store_true", help="Include ROBOT validation")
    validate.add_argument("--arelle", action="store_true", help="Include Arelle validation")

    init = subparsers.add_parser("init", help="Create example sources.yaml configuration")
    init.add_argument("path", nargs="?", default=Path("sources.yaml"), type=Path)

    config_cmd = subparsers.add_parser("config", help="Configuration utilities")
    config_sub = config_cmd.add_subparsers(dest="config_command", required=True)
    config_validate = config_sub.add_parser("validate", help="Validate a configuration file")
    config_validate.add_argument(
        "--spec",
        type=Path,
        default=CONFIG_DIR / "sources.yaml",
        help="Path to configuration file (default: ~/.data/ontology-fetcher/configs/sources.yaml)",
    )
    config_validate.add_argument(
        "--json",
        action="store_true",
        help="Output validation result as JSON",
    )

    doctor = subparsers.add_parser("doctor", help="Diagnose environment issues")
    doctor.add_argument("--json", action="store_true", help="Output diagnostics as JSON")

    prune = subparsers.add_parser("prune", help="Prune stored ontology versions")
    prune.add_argument(
        "--keep",
        type=_parse_positive_int,
        required=True,
        help="Number of versions to retain per ontology",
    )
    prune.add_argument(
        "--ids",
        nargs="*",
        help="Optional ontology identifiers to limit pruning scope",
    )
    prune.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletions without removing files",
    )
    prune.add_argument(
        "--json",
        action="store_true",
        help="Emit pruning summary as JSON",
    )

    return parser


EXAMPLE_SOURCES_YAML = """# Example configuration for ontology downloader\ndefaults:
  accept_licenses: ["CC-BY-4.0", "CC0-1.0", "OGL-UK-3.0"]
  normalize_to: ["ttl"]
  prefer_source: ["obo", "ols", "bioportal", "direct"]
  http:
    max_retries: 5
    timeout_sec: 30
    backoff_factor: 0.5
    per_host_rate_limit: "4/second"
    max_download_size_gb: 5
    validate_media_type: true
  validation:
    skip_reasoning_if_size_mb: 500
    parser_timeout_sec: 60
  logging:
    level: "INFO"
    max_log_size_mb: 100
    retention_days: 30

ontologies:
  - id: hp
    resolver: obo
    target_formats: [owl, obo]
  - id: efo
    resolver: ols
  - id: ncit
    resolver: bioportal
    extras:
      acronym: NCIT
  - id: eurovoc
    resolver: skos
    extras:
      url: https://op.europa.eu/o/opportal-service/euvoc-download-handler?cellarURI=http%3A%2F%2Fpublications.europa.eu%2Fresource%2Fauthority%2Feurovoc
"""


def _parse_target_formats(value: Optional[str]) -> List[str]:
    """Normalize comma-separated target format strings.

    Args:
        value: Raw CLI argument possibly containing comma-delimited formats.

    Returns:
        List of stripped format identifiers, or an empty list when no formats are supplied.
    """
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_positive_int(value: str) -> int:
    """Parse CLI argument ensuring it is a positive integer.

    Args:
        value: Textual representation of a positive integer.

    Returns:
        Parsed positive integer value.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a positive integer.
    """

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - argparse handles message
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_allowed_hosts(value: Optional[str]) -> List[str]:
    """Split comma-delimited host allowlist argument into unique entries.

    Args:
        value: Raw CLI argument containing comma-delimited hostnames.

    Returns:
        List of unique hostnames preserving input order.
    """

    if not value:
        return []
    entries: List[str] = []
    for host in value.split(","):
        candidate = host.strip()
        if candidate and candidate not in entries:
            entries.append(candidate)
    return entries


def _normalize_plan_args(args: Sequence[str]) -> List[str]:
    """Ensure ``plan`` command defaults to the ``run`` subcommand when omitted.

    Args:
        args: Original CLI argument vector.

    Returns:
        Updated argument list with explicit subcommands injected as needed.
    """

    tokens = list(args)
    try:
        index = tokens.index("plan")
    except ValueError:
        return tokens

    if index + 1 >= len(tokens):
        return tokens + ["run"]

    next_token = tokens[index + 1]
    if next_token in {"run", "diff"}:
        return tokens

    if next_token.startswith("-"):
        return tokens[: index + 1] + ["run", *tokens[index + 1 :]]

    return tokens[: index + 1] + ["run", *tokens[index + 1 :]]


def _parse_since_arg(value: str) -> datetime:
    """Parse ``YYYY-MM-DD`` strings into timezone-aware datetimes.

    Args:
        value: Date string provided on the command line.

    Returns:
        Timezone-aware datetime instance aligned to UTC.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not a valid date.
    """

    parsed = parse_version_timestamp(value)
    if parsed is None:
        raise argparse.ArgumentTypeError("must be YYYY-MM-DD")
    return parsed


def _parse_since(value: Optional[Union[str, datetime]]) -> Optional[datetime]:
    """Parse optional date input into timezone-aware datetimes.

    Args:
        value: Either a string date or a pre-parsed datetime.

    Returns:
        Normalized datetime in UTC, or ``None`` when not supplied.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = value.strip()
    if not text:
        return None
    return _parse_since_arg(text)


def _format_bytes(num: int) -> str:
    """Return human-readable byte count representation.

    Args:
        num: Byte count to format.

    Returns:
        String describing the size using binary units.
    """

    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < step or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PB"


def _apply_cli_overrides(config: ResolvedConfig, args) -> None:
    """Mutate resolved configuration based on CLI override arguments.

    Args:
        config: Resolved configuration subject to mutation.
        args: Parsed CLI namespace containing override values.
    """

    downloads = getattr(args, "concurrent_downloads", None)
    if downloads is not None:
        config.defaults.http.concurrent_downloads = downloads

    plans = getattr(args, "concurrent_plans", None)
    if plans is not None:
        config.defaults.http.concurrent_plans = plans

    merged_hosts = _parse_allowed_hosts(getattr(args, "allowed_hosts", None))
    if merged_hosts:
        existing = list(config.defaults.http.allowed_hosts or [])
        for host in merged_hosts:
            if host not in existing:
                existing.append(host)
        config.defaults.http.allowed_hosts = existing


def _results_to_dict(result: FetchResult) -> dict:
    """Convert a ``FetchResult`` instance into a JSON-friendly mapping.

    Args:
        result: Completed fetch result returned by :func:`fetch_one`.

    Returns:
        Dictionary containing resolver metadata, manifest location, and artifact list.
    """
    return {
        "id": result.spec.id,
        "resolver": result.spec.resolver,
        "status": result.status,
        "file": str(result.local_path),
        "sha256": result.sha256,
        "manifest": str(result.manifest_path),
        "artifacts": list(result.artifacts),
    }


def _compute_plan_diff(baseline: Sequence[dict], current: Sequence[dict]) -> Dict[str, object]:
    """Compute additions, removals, and modifications between plan snapshots."""

    baseline_index = {entry.get("id"): entry for entry in baseline if entry.get("id")}
    current_index = {entry.get("id"): entry for entry in current if entry.get("id")}
    baseline_ids = set(baseline_index)
    current_ids = set(current_index)

    added = [current_index[oid] for oid in sorted(current_ids - baseline_ids)]
    removed = [baseline_index[oid] for oid in sorted(baseline_ids - current_ids)]
    modified = []
    for oid in sorted(baseline_ids & current_ids):
        before = baseline_index[oid]
        after = current_index[oid]
        changes = {}
        for field in PLAN_DIFF_FIELDS:
            if before.get(field) != after.get(field):
                changes[field] = {"before": before.get(field), "after": after.get(field)}
        if changes:
            modified.append({"id": oid, "before": before, "after": after, "changes": changes})

    return {"added": added, "removed": removed, "modified": modified}


def _format_plan_diff(diff: Dict[str, object]) -> List[str]:
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


def _plan_to_dict(plan: PlannedFetch) -> dict:
    """Convert a planned fetch into a JSON-friendly dictionary.

    Args:
        plan: Planned fetch data produced by :func:`plan_one` or :func:`plan_all`.

    Returns:
        Mapping containing resolver metadata and planned download details suitable
        for serialization.
    """

    candidates = []
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
    return payload


def _resolve_version_metadata(
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
    size = _directory_size(path)
    return path, timestamp, size


def _ensure_manifest_path(ontology_id: str, version: Optional[str]) -> Path:
    """Return the manifest path for a given ontology and version.

    Args:
        ontology_id: Identifier for the ontology whose manifest is requested.
        version: Optional version string; when omitted the latest available is used.

    Returns:
        Path to the manifest JSON file on disk.

    Raises:
        ConfigError: If the ontology or manifest cannot be located locally.
    """
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


def _load_manifest(manifest_path: Path) -> dict:
    """Read and parse a manifest JSON document from disk.

    Args:
        manifest_path: Filesystem location of the manifest file.

    Returns:
        Dictionary representation of the manifest contents.
    """
    return json.loads(manifest_path.read_text())


def _collect_version_metadata(ontology_id: str) -> List[Dict[str, object]]:
    """Return sorted metadata entries for stored ontology versions.

    Args:
        ontology_id: Identifier of the ontology to inspect.

    Returns:
        List of metadata dictionaries ordered by most recent timestamp first.
    """

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
        size = _directory_size(version_dir)
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


def _resolve_specs_from_args(
    args,
    base_config: Optional[ResolvedConfig],
    *,
    allow_empty: bool = False,
) -> tuple[ResolvedConfig, List[FetchSpec]]:
    """Return configuration and fetch specifications derived from CLI arguments.

    Args:
        args: Parsed command-line arguments for `pull`/`plan` commands.
        base_config: Optional pre-loaded configuration used when no spec file is supplied.

    Returns:
        Tuple containing the active resolved configuration and the list of fetch specs
        that should be processed.

    Raises:
        ConfigError: If neither explicit IDs nor a configuration file are provided.
    """

    target_formats = _parse_target_formats(getattr(args, "target_formats", None))
    config_path: Optional[Path] = getattr(args, "spec", None)
    ids: Sequence[str] = getattr(args, "ids", [])
    if config_path is None and not ids:
        default_config = CONFIG_DIR / "sources.yaml"
        if default_config.exists():
            config_path = default_config

    if config_path:
        config = load_config(config_path)
    else:
        config = base_config or ResolvedConfig.from_defaults()

    config.defaults.logging.level = getattr(args, "log_level", config.defaults.logging.level)
    _apply_cli_overrides(config, args)

    if ids:
        resolver_name = getattr(args, "resolver", None) or config.defaults.prefer_source[0]
        formats = target_formats or config.defaults.normalize_to
        specs = [
            FetchSpec(id=oid, resolver=resolver_name, extras={}, target_formats=formats)
            for oid in ids
        ]
        return config, specs

    if config.specs:
        return config, config.specs

    if allow_empty:
        return config, []

    raise ConfigError("Please provide ontology IDs or --spec configuration")


def _handle_pull(
    args,
    base_config: Optional[ResolvedConfig],
    *,
    dry_run: bool = False,
):
    """Execute the ``pull`` subcommand workflow."""

    config, specs = _resolve_specs_from_args(args, base_config)
    if dry_run:
        return plan_all(specs, config=config)
    return fetch_all(
        specs,
        config=config,
        force=getattr(args, "force", False),
    )


def _handle_plan(args, base_config: Optional[ResolvedConfig]) -> List[PlannedFetch]:
    """Resolve plans without executing downloads."""

    since = _parse_since(getattr(args, "since", None))
    config, specs = _resolve_specs_from_args(args, base_config)
    plans = plan_all(specs, config=config, since=since)
    return plans


def _handle_plan_diff(args, base_config: Optional[ResolvedConfig]) -> Dict[str, object]:
    """Compare current plan output against a baseline plan file."""

    baseline_path: Path = getattr(args, "baseline", DEFAULT_PLAN_BASELINE).expanduser()
    if not baseline_path.exists():
        raise ConfigError(f"Baseline plan file not found: {baseline_path}")
    try:
        baseline_payload = json.loads(baseline_path.read_text()) or []
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Baseline plan file {baseline_path} is not valid JSON") from exc
    if not isinstance(baseline_payload, list):
        raise ConfigError("Baseline plan must be a JSON array of plan entries")

    since = _parse_since(getattr(args, "since", None))
    config, specs = _resolve_specs_from_args(args, base_config, allow_empty=True)
    plans = plan_all(specs, config=config, since=since)
    current_payload = [_plan_to_dict(plan) for plan in plans]
    diff = _compute_plan_diff(baseline_payload, current_payload)
    diff["baseline"] = str(baseline_path)
    return diff


def _handle_prune(args, logger) -> Dict[str, object]:
    """Delete surplus ontology versions based on ``--keep`` parameter."""

    keep = args.keep
    if keep <= 0:
        raise ConfigError("--keep must be a positive integer")

    target_ids = args.ids or STORAGE.available_ontologies()
    target_ids = sorted(set(target_ids))

    summary: List[Dict[str, object]] = []
    total_reclaimed = 0
    total_deleted = 0
    messages: List[str] = []

    for ontology_id in target_ids:
        raw_metadata = _collect_version_metadata(ontology_id)
        metadata = sorted(
            raw_metadata,
            key=lambda item: item.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        if len(metadata) <= keep:
            continue
        retained = metadata[:keep]
        to_remove = metadata[keep:]
        reclaimed = 0
        deleted_versions: List[str] = []
        for item in to_remove:
            deleted_versions.append(item["version"])
            if args.dry_run:
                reclaimed += int(item.get("size") or 0)
            else:
                reclaimed_bytes = STORAGE.delete_version(ontology_id, item["version"])
                reclaimed += reclaimed_bytes
                total_deleted += 1
                if logger is not None:
                    logger.info(
                        "pruned ontology version",
                        extra={
                            "ontology_id": ontology_id,
                            "version": item["version"],
                            "freed_bytes": reclaimed_bytes,
                        },
                    )
        if not args.dry_run and retained:
            STORAGE.set_latest_version(ontology_id, retained[0]["path"])
        total_reclaimed += reclaimed
        if args.dry_run:
            total_deleted += len(to_remove)
        summary.append(
            {
                "id": ontology_id,
                "deleted": deleted_versions,
                "retained": [entry["version"] for entry in retained],
                "reclaimed_bytes": reclaimed,
            }
        )
        if args.dry_run:
            for item in to_remove:
                messages.append(
                    f"[DRY-RUN] {ontology_id} version {item['version']} would free {_format_bytes(int(item.get('size') or 0))}"
                )
        elif deleted_versions:
            messages.append(
                f"Deleted {len(deleted_versions)} versions for {ontology_id} (freed {_format_bytes(reclaimed)})"
            )

    return {
        "ontologies": summary,
        "total_deleted": total_deleted,
        "total_reclaimed_bytes": total_reclaimed,
        "dry_run": bool(args.dry_run),
        "messages": messages,
    }


def _doctor_report() -> Dict[str, object]:
    """Collect diagnostic information for the ``doctor`` command.

    Returns:
        Mapping capturing disk, dependency, network, and configuration status.
    """

    directories = {}
    for name, path in {
        "configs": CONFIG_DIR,
        "cache": CACHE_DIR,
        "logs": LOG_DIR,
        "ontologies": LOCAL_ONTOLOGY_DIR,
    }.items():
        directories[name] = {
            "path": str(path),
            "exists": path.exists(),
            "writable": os.access(path, os.W_OK),
        }

    disk_usage = shutil.disk_usage(LOCAL_ONTOLOGY_DIR)
    threshold_bytes = max(10 * 1_000_000_000, int(disk_usage.total * 0.1))
    disk_report = {
        "total_bytes": disk_usage.total,
        "free_bytes": disk_usage.free,
        "total_gb": round(disk_usage.total / 1_000_000_000, 2),
        "free_gb": round(disk_usage.free / 1_000_000_000, 2),
        "threshold_bytes": threshold_bytes,
        "warning": disk_usage.free < threshold_bytes,
    }

    dependencies = {
        "rdflib": importlib.util.find_spec("rdflib") is not None,
        "pronto": importlib.util.find_spec("pronto") is not None,
        "owlready2": importlib.util.find_spec("owlready2") is not None,
        "arelle": importlib.util.find_spec("arelle") is not None,
    }

    robot_path = shutil.which("robot")
    robot_info: Dict[str, object] = {"available": bool(robot_path), "path": robot_path}
    if robot_path:
        try:
            completed = subprocess.run(
                [robot_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            output = (completed.stdout or completed.stderr or "").strip()
            match = re.search(r"(\d+\.\d+(?:\.\d+)?)", output)
            if match:
                robot_info["version"] = match.group(1)
            if completed.returncode != 0:
                robot_info["detail"] = output or f"exit code {completed.returncode}"
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - system dependent
            robot_info["error"] = str(exc)

    api_key_path = CONFIG_DIR / "bioportal_api_key.txt"
    bioportal = {
        "path": str(api_key_path),
        "configured": api_key_path.exists() and api_key_path.read_text().strip() != "",
    }

    network_targets = {
        "ols": "https://www.ebi.ac.uk/ols4/api/health",
        "bioportal": "https://data.bioontology.org",
        "bioregistry": "https://bioregistry.io",
    }
    network: Dict[str, Dict[str, object]] = {}
    for name, url in network_targets.items():
        result: Dict[str, object] = {"url": url}
        try:
            response = requests.head(url, timeout=3, allow_redirects=True)
            status = response.status_code
            ok = response.ok
            if status == 405:
                response = requests.get(url, timeout=3, allow_redirects=True)
                status = response.status_code
                ok = response.ok
            result.update({"ok": ok, "status": status})
            if not ok:
                result["detail"] = response.reason
        except requests.RequestException as exc:  # pragma: no cover - network variability
            result.update({"ok": False, "detail": str(exc)})
        network[name] = result

    rate_limits: Dict[str, object] = {
        "effective": ResolvedConfig.from_defaults().defaults.http.rate_limits,
    }
    config_path = CONFIG_DIR / "sources.yaml"
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text()) or {}
        except Exception as exc:  # pragma: no cover - YAML errors depend on file contents
            rate_limits["error"] = f"Failed to parse {config_path}: {exc}"  # type: ignore[assignment]
        else:
            http_section = raw.get("defaults", {}).get("http") if isinstance(raw, dict) else None
            configured = http_section.get("rate_limits") if isinstance(http_section, dict) else None
            if isinstance(configured, dict):
                valid: Dict[str, Dict[str, object]] = {}
                invalid: Dict[str, str] = {}
                for service, limit in configured.items():
                    text_value = str(limit)
                    rps = parse_rate_limit_to_rps(text_value)
                    if rps is None:
                        invalid[service] = text_value
                    else:
                        valid[service] = {
                            "value": text_value,
                            "requests_per_second": rps,
                        }
                if valid:
                    rate_limits["configured"] = valid
                if invalid:
                    rate_limits["invalid"] = invalid

    schema_report: Dict[str, object] = {"version": MANIFEST_SCHEMA_VERSION}
    try:
        Draft202012Validator.check_schema(get_manifest_schema())
        schema_report["schema_valid"] = True
    except SchemaError as exc:
        schema_report["schema_valid"] = False
        schema_report["error"] = str(exc)

    sample_manifest: Optional[Path] = None
    if LOCAL_ONTOLOGY_DIR.exists():
        for candidate in LOCAL_ONTOLOGY_DIR.rglob("manifest.json"):
            sample_manifest = candidate
            break

    if sample_manifest is not None:
        try:
            payload = json.loads(sample_manifest.read_text())
        except json.JSONDecodeError as exc:
            schema_report["sample"] = {
                "path": str(sample_manifest),
                "valid": False,
                "error": f"Invalid JSON: {exc}",
            }
        else:
            try:
                validate_manifest_dict(payload, source=sample_manifest)
            except ConfigurationError as exc:
                detail = str(exc.__cause__) if exc.__cause__ else str(exc)
                schema_report["sample"] = {
                    "path": str(sample_manifest),
                    "valid": False,
                    "error": detail,
                }
            else:
                schema_report["sample"] = {
                    "path": str(sample_manifest),
                    "valid": True,
                }
    else:
        schema_report["sample"] = {"path": None, "valid": None}

    storage_backend = {
        "backend": STORAGE.__class__.__name__,
        "remote": hasattr(STORAGE, "fs"),
    }

    report: Dict[str, object] = {
        "directories": directories,
        "disk": disk_report,
        "dependencies": dependencies,
        "robot": robot_info,
        "bioportal_api_key": bioportal,
        "network": network,
        "rate_limits": rate_limits,
        "manifest_schema": schema_report,
        "storage": storage_backend,
    }
    return report


def _print_doctor_report(report: Dict[str, object]) -> None:
    """Render human-readable diagnostics from :func:`_doctor_report`.

    Args:
        report: Diagnostics mapping generated by :func:`_doctor_report`.
    """

    print("Directories:")
    for name, info in report["directories"].items():
        status = []
        if info["exists"]:
            status.append("exists")
        else:
            status.append("missing")
        if info["writable"]:
            status.append("writable")
        else:
            status.append("read-only")
        print(f"  - {name}: {', '.join(status)} ({info['path']})")

    disk = report["disk"]
    print(
        "Disk space: {free:.2f} GB free / {total:.2f} GB total".format(
            free=disk["free_gb"], total=disk["total_gb"]
        )
    )
    if disk.get("warning"):
        threshold_gb = disk["threshold_bytes"] / 1_000_000_000
        print(f"  Warning: free space below threshold ({threshold_gb:.2f} GB).")

    print("Optional dependencies:")
    for name, available in report["dependencies"].items():
        status = "available" if available else "missing"
        print(f"  - {name}: {status}")

    robot = report["robot"]
    if robot.get("available"):
        version = robot.get("version", "unknown")
        detail = robot.get("detail") or robot.get("error")
        extra = f" (version {version})"
        if detail:
            extra += f" [{detail}]"
        print(f"ROBOT tool: available{extra}")
    else:
        print("ROBOT tool: not found in PATH")

    rate_limits = report["rate_limits"]
    configured = rate_limits.get("configured", {})
    if configured:
        print("Rate limits:")
        for service, info in configured.items():
            rps = info.get("requests_per_second")
            if rps is not None:
                print(f"  - {service}: {info['value']} (~{rps:.2f} req/s)")
            else:
                print(f"  - {service}: {info['value']}")
    if rate_limits.get("invalid"):
        print("  Invalid rate limits detected:")
        for service, value in rate_limits["invalid"].items():
            print(f"    * {service}: '{value}' (expected <number>/<unit>)")
    if rate_limits.get("error"):
        print(f"Rate limit check error: {rate_limits['error']}")

    schema = report.get("manifest_schema", {})
    if schema:
        status = "valid" if schema.get("schema_valid") else "invalid"
        print(
            f"Manifest schema ({schema.get('version', 'unknown')}): {status}"
            + (f" [{schema.get('error')}]" if schema.get("error") else "")
        )
        sample = schema.get("sample", {})
        sample_path = sample.get("path")
        if sample_path:
            sample_status = sample.get("valid")
            if sample_status is True:
                print(f"  Sample manifest: OK ({sample_path})")
            elif sample_status is False:
                error = sample.get("error", "UNKNOWN ERROR")
                print(f"  Sample manifest: invalid ({sample_path}) - {error}")
            else:
                print(f"  Sample manifest: not evaluated ({sample_path})")
        else:
            print("  Sample manifest: none discovered")

    print("Network connectivity:")
    for name, info in report["network"].items():
        status = "ok" if info.get("ok") else "failed"
        detail = info.get("detail") or info.get("status")
        print(f"  - {name}: {status} ({detail})")

    bioportal = report["bioportal_api_key"]
    if bioportal["configured"]:
        print(f"BioPortal API key: configured ({bioportal['path']})")
    else:
        print(f"BioPortal API key: missing ({bioportal['path']})")

    storage = report["storage"]
    backend_desc = storage.get("backend", "unknown")
    if storage.get("remote"):
        backend_desc += " (remote)"
    else:
        backend_desc += " (local)"
    print(f"Storage backend: {backend_desc}")


def _handle_show(args) -> None:
    """Display ontology manifest information for the ``show`` command.

    Args:
        args: Parsed CLI arguments including ontology identifier and output format.

    Returns:
        None

    Raises:
        ConfigError: When the manifest cannot be located.
    """
    if args.versions:
        versions = STORAGE.available_versions(args.id)
        if not versions:
            raise ConfigError(f"No versions stored for ontology '{args.id}'")
        for version in versions:
            print(version)
        return
    manifest_path = _ensure_manifest_path(args.id, args.version)
    manifest = _load_manifest(manifest_path)
    if args.json:
        json.dump(manifest, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for key, value in manifest.items():
            print(f"{key}: {value}")


def _selected_validators(args) -> Sequence[str]:
    """Determine which validators should execute based on CLI flags.

    Args:
        args: Parsed CLI arguments for the ``validate`` command.

    Returns:
        Sequence containing validator names in execution order.
    """
    mapping = {
        "rdflib": args.rdflib,
        "pronto": args.pronto,
        "owlready2": args.owlready2,
        "robot": args.robot,
        "arelle": args.arelle,
    }
    chosen = [name for name, enabled in mapping.items() if enabled]
    return chosen or list(mapping.keys())


def _handle_validate(args, config: ResolvedConfig) -> dict:
    """Run validators for a previously downloaded ontology.

    Args:
        args: Parsed CLI arguments specifying ontology ID, version, and output format.
        config: Resolved configuration supplying validator defaults.

    Returns:
        Mapping of validator names to their structured result payloads.

    Raises:
        ConfigError: If the manifest or downloaded artifacts cannot be located.
    """
    manifest_path = _ensure_manifest_path(args.id, args.version)
    manifest = _load_manifest(manifest_path)
    version_dir = manifest_path.parent
    original_path = version_dir / "original" / manifest["filename"]
    validation_dir = version_dir / "validation"
    normalized_dir = version_dir / "normalized"
    validator_names = _selected_validators(args)
    requests = [
        ValidationRequest(name, original_path, normalized_dir, validation_dir, config)
        for name in validator_names
    ]
    logging_config = config.defaults.logging
    logger = setup_logging(
        level=logging_config.level,
        retention_days=logging_config.retention_days,
        max_log_size_mb=logging_config.max_log_size_mb,
    )
    results = run_validators(requests, logger)
    manifest["validation"] = {name: result.to_dict() for name, result in results.items()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest["validation"]


def _handle_init(path: Path) -> None:
    """Create a starter ``sources.yaml`` file for new installations.

    Args:
        path: Destination path for the generated configuration template.

    Raises:
        ConfigError: If the target file already exists.
    """
    if path.exists():
        raise ConfigError(f"Refusing to overwrite existing file {path}")
    path.write_text(EXAMPLE_SOURCES_YAML)
    print(f"Wrote example configuration to {path}")


def _handle_config_validate(path: Path) -> dict:
    """Validate a configuration file and return a summary report.

    Args:
        path: Filesystem path to the configuration file under validation.

    Returns:
        Dictionary describing validation status, ontology count, and file path.
    """
    config = validate_config(path)
    return {
        "ok": True,
        "ontologies": len(config.specs),
        "path": str(path),
    }


def _normalize_argv(argv: Sequence[str]) -> List[str]:
    """Rewrite legacy aliases to the canonical subcommand syntax.

    Args:
        argv: Raw argument vector supplied by the user.

    Returns:
        Adjusted argument list with legacy ``plan diff`` rewired to ``plan-diff``.
    """

    tokens = list(argv)
    for index in range(len(tokens) - 1):
        if tokens[index] == "plan" and tokens[index + 1] == "diff":
            return tokens[:index] + ["plan-diff"] + tokens[index + 2 :]
    return tokens


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ontology downloader CLI.

    Args:
        argv: Optional argument vector supplied for testing or scripting.

    Returns:
        Process exit code indicating success (`0`) or failure.

    Raises:
        ConfigError: If configuration files are invalid or unsafe to overwrite.
        OntologyDownloadError: If download or validation operations fail.
    """
    raw_args = list(argv or sys.argv[1:])
    arg_list = _normalize_plan_args(_normalize_argv(raw_args))
    parser = _build_parser()
    args = parser.parse_args(arg_list)
    try:
        base_config = ResolvedConfig.from_defaults()
        base_config.defaults.logging.level = args.log_level
        logging_config = base_config.defaults.logging
        logger = setup_logging(
            level=logging_config.level,
            retention_days=logging_config.retention_days,
            max_log_size_mb=logging_config.max_log_size_mb,
        )
        if args.command == "pull":
            if getattr(args, "dry_run", False):
                plans = _handle_pull(args, base_config, dry_run=True)
                if args.json:
                    json.dump([_plan_to_dict(plan) for plan in plans], sys.stdout, indent=2)
                    sys.stdout.write("\n")
                else:
                    if plans:
                        rows = format_plan_rows(plans)
                        print(
                            format_table(
                                ("id", "resolver", "service", "media_type", "url"),
                                rows,
                            )
                        )
                    else:
                        print("No ontologies to process")
            else:
                results = _handle_pull(args, base_config, dry_run=False)
                if args.json:
                    json.dump(
                        [_results_to_dict(result) for result in results], sys.stdout, indent=2
                    )
                    sys.stdout.write("\n")
                else:
                    if results:
                        print(format_results_table(results))
                    else:
                        print("No ontologies to process")
                for result in results:
                    logger.info(
                        "ontology processed",
                        extra={
                            "stage": "complete",
                            "ontology_id": result.spec.id,
                            "status": result.status,
                        },
                    )
        elif args.command == "plan":
            plans = _handle_plan(args, base_config)
            if args.json:
                json.dump([_plan_to_dict(plan) for plan in plans], sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                if plans:
                    rows = format_plan_rows(plans)
                    print(
                        format_table(
                            ("id", "resolver", "service", "media_type", "url"),
                            rows,
                        )
                    )
                else:
                    print("No ontologies to process")
        elif args.command == "plan-diff":
            diff = _handle_plan_diff(args, base_config)
            if args.json:
                json.dump(diff, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                lines = _format_plan_diff(diff)
                if lines:
                    print("\n".join(lines))
                else:
                    print("No plan differences detected")
        elif args.command == "prune":
            summary = _handle_prune(args, logger)
            if args.json:
                json.dump(summary, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                messages = summary.get("messages", [])
                if messages:
                    print("\n".join(messages))
                total = summary.get("total_reclaimed_bytes", 0)
                deleted = summary.get("total_deleted", 0)
                label = "Dry-run" if summary.get("dry_run") else "Pruned"
                print(f"{label}: reclaimed {_format_bytes(total)} across {deleted} versions")
        elif args.command == "show":
            _handle_show(args)
        elif args.command == "validate":
            summary = _handle_validate(args, base_config)
            if args.json:
                json.dump(summary, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                print(format_validation_summary(summary))
        elif args.command == "init":
            _handle_init(args.path)
        elif args.command == "config" and args.config_command == "validate":
            report = _handle_config_validate(args.spec)
            if args.json:
                json.dump(report, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                status = "passed" if report["ok"] else "failed"
                print(
                    f"Configuration {status} ({report['ontologies']} ontologies) -> {report['path']}"
                )
        elif args.command == "doctor":
            report = _doctor_report()
            if args.json:
                json.dump(report, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                _print_doctor_report(report)
        else:  # pragma: no cover - argparse should prevent unknown commands
            parser.error(f"Unsupported command: {args.command}")
        return 0
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OntologyDownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
