"""Command-line orchestration for DocsToKG OntologyDownload."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import requests
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .api import _collect_plugin_details
from .errors import (
    ConfigError,
    ConfigurationError,
    OntologyDownloadError,
    UnsupportedPythonError,
)
from .formatters import (
    PLAN_TABLE_HEADERS,
    format_plan_rows,
    format_results_table,
    format_table,
    format_validation_summary,
)
from .io import format_bytes
from .logging_utils import _cleanup_logs, setup_logging
from .manifests import (
    DEFAULT_LOCKFILE_PATH,
    DEFAULT_PLAN_BASELINE,
    collect_version_metadata,
    compute_plan_diff,
    ensure_manifest_path,
    format_plan_diff,
    load_latest_manifest,
    load_lockfile_payload,
    load_manifest,
    plan_to_dict,
    results_to_dict,
    specs_from_lock_payload,
    write_lockfile,
)
from .planning import (
    MANIFEST_SCHEMA_VERSION,
    BatchFetchError,
    BatchPlanningError,
    FetchSpec,
    PlannedFetch,
    fetch_all,
    get_manifest_schema,
    parse_version_timestamp,
    plan_all,
    validate_manifest_dict,
)
from .settings import (
    CACHE_DIR,
    CONFIG_DIR,
    LOCAL_ONTOLOGY_DIR,
    LOG_DIR,
    STORAGE,
    ResolvedConfig,
    get_default_config,
    load_config,
    parse_rate_limit_to_rps,
    validate_config,
)
from .validation import (
    ValidationRequest,
    run_validators,
)


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
    pull.add_argument(
        "--planner-probes",
        dest="planner_probes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable planner metadata HTTP probes (default: enabled).",
    )
    pull.add_argument(
        "--lock",
        type=Path,
        help="Path to ontologies.lock.json to pin downloads to exact planned inputs",
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
    plan_cmd.add_argument(
        "--planner-probes",
        dest="planner_probes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable planner metadata HTTP probes (default: enabled).",
    )
    plan_cmd.add_argument("--json", action="store_true", help="Emit plan details as JSON")
    plan_cmd.add_argument(
        "--lock-output",
        type=Path,
        default=DEFAULT_LOCKFILE_PATH,
        help=f"Write lockfile (default: {DEFAULT_LOCKFILE_PATH})",
    )
    plan_cmd.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip writing the ontologies.lock.json file",
    )

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
        "--update-baseline",
        action="store_true",
        help="Write the current plan to the baseline file after computing the diff",
    )
    plan_diff.add_argument(
        "--use-manifest",
        action="store_true",
        help="Compare planned metadata against the latest stored manifests",
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
    plan_diff.add_argument(
        "--planner-probes",
        dest="planner_probes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable planner metadata HTTP probes (default: enabled).",
    )
    plan_diff.add_argument("--json", action="store_true", help="Emit plan diff as JSON")
    plan_diff.add_argument(
        "--lock-output",
        type=Path,
        default=DEFAULT_LOCKFILE_PATH,
        help=f"Write lockfile (default: {DEFAULT_LOCKFILE_PATH})",
    )
    plan_diff.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip writing the ontologies.lock.json file",
    )

    plugins_cmd = subparsers.add_parser(
        "plugins",
        help="List available resolver or validator plugins",
        description="Show resolver or validator plugins available in this environment.",
    )
    plugins_cmd.add_argument(
        "--kind",
        choices=("resolver", "validator", "all"),
        default="resolver",
        help="Plugin category to display (default: resolver)",
    )
    plugins_cmd.add_argument(
        "--json",
        action="store_true",
        help="Emit plugin inventory as JSON",
    )

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
    doctor.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to create required directories, rotate logs, and scaffold API key placeholders",
    )

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
        "--older-than",
        type=_parse_since_arg,
        help="Delete versions with timestamps older than YYYY-MM-DD",
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
    max_uncompressed_size_gb: 10
    validate_media_type: true
  validation:
    skip_reasoning_if_size_mb: 500
    parser_timeout_sec: 60
  logging:
    level: "INFO"
    max_log_size_mb: 100
    retention_days: 30
  enable_cas_mirror: false

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

    return list(args)


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

    planner_override = getattr(args, "planner_probes", None)
    if planner_override is not None:
        config.defaults.planner.probing_enabled = bool(planner_override)


def _resolve_specs_from_args(
    args,
    base_config: Optional[ResolvedConfig],
    *,
    allow_empty: bool = False,
) -> tuple[ResolvedConfig, List[FetchSpec]]:
    """Return configuration and fetch specifications derived from CLI arguments."""

    target_formats = _parse_target_formats(getattr(args, "target_formats", None))
    config_path: Optional[Path] = getattr(args, "spec", None)
    ids: List[str] = list(getattr(args, "ids", []))
    if config_path is None and not ids:
        default_config = CONFIG_DIR / "sources.yaml"
        if default_config.exists():
            config_path = default_config

    if config_path:
        config = load_config(config_path)
    else:
        config = base_config or get_default_config(copy=True)

    config.defaults.logging.level = getattr(args, "log_level", config.defaults.logging.level)
    _apply_cli_overrides(config, args)

    lock_path: Optional[Path] = getattr(args, "lock", None)
    if lock_path is not None:
        lock_payload = load_lockfile_payload(lock_path)
        specs = specs_from_lock_payload(
            lock_payload,
            defaults=config.defaults,
            requested_ids=ids,
        )
        if not specs and not allow_empty:
            raise ConfigError(f"No matching entries found in lock file {lock_path}")
        config.defaults.resolver_fallback_enabled = False
        config.defaults.prefer_source = ["direct"]
        config.specs = specs
        logging.getLogger("DocsToKG.OntologyDownload").info(
            "using lockfile",
            extra={
                "stage": "plan",
                "lock_path": str(lock_path.expanduser()),
                "entries": len(specs),
            },
        )
        return config, specs

    if ids:
        resolver_override = getattr(args, "resolver", None)
        default_resolver = None
        if config.defaults.prefer_source:
            default_resolver = config.defaults.prefer_source[0]
        fallback_resolver = resolver_override or default_resolver or "obo"
        fallback_formats = (
            tuple(target_formats)
            if target_formats
            else tuple(config.defaults.normalize_to)
        )

        specs_by_id = {spec.id: spec for spec in config.specs or []}
        resolved_specs: List[FetchSpec] = []
        for oid in ids:
            existing = specs_by_id.get(oid)
            if existing is not None:
                resolved_specs.append(
                    FetchSpec(
                        id=existing.id,
                        resolver=existing.resolver,
                        extras=dict(existing.extras),
                        target_formats=tuple(existing.target_formats),
                    )
                )
            else:
                resolved_specs.append(
                    FetchSpec(
                        id=oid,
                        resolver=fallback_resolver,
                        extras={},
                        target_formats=fallback_formats,
                    )
                )
        return config, resolved_specs

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
    logger: Optional[logging.Logger] = None,
):
    """Execute the ``pull`` subcommand workflow."""

    config, specs = _resolve_specs_from_args(args, base_config)
    if dry_run:
        return plan_all(specs, config=config, logger=logger)
    return fetch_all(
        specs,
        config=config,
        logger=logger,
        force=getattr(args, "force", False),
    )


def _handle_plan(
    args,
    base_config: Optional[ResolvedConfig],
    *,
    logger: Optional[logging.Logger] = None,
) -> List[PlannedFetch]:
    """Resolve plans without executing downloads."""

    since = _parse_since(getattr(args, "since", None))
    config, specs = _resolve_specs_from_args(args, base_config)
    plans = plan_all(specs, config=config, since=since, logger=logger)
    if not getattr(args, "no_lock", False):
        lock_path = getattr(args, "lock_output", DEFAULT_LOCKFILE_PATH)
        write_lockfile(plans, lock_path)
    return plans


def _handle_plan_diff(args, base_config: Optional[ResolvedConfig]) -> Dict[str, object]:
    """Compare current plan output against a baseline plan file."""

    use_manifest = bool(getattr(args, "use_manifest", False))
    update_baseline = bool(getattr(args, "update_baseline", False))
    if use_manifest:
        baseline_payload: List[dict] = []
        baseline_path = None
    else:
        baseline_path = getattr(args, "baseline", DEFAULT_PLAN_BASELINE).expanduser()
        if baseline_path.exists():
            try:
                baseline_payload = json.loads(baseline_path.read_text()) or []
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Baseline plan file {baseline_path} is not valid JSON") from exc
            if not isinstance(baseline_payload, list):
                raise ConfigError("Baseline plan must be a JSON array of plan entries")
        elif update_baseline:
            baseline_payload = []
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise ConfigError(f"Baseline plan file not found: {baseline_path}")

    since = _parse_since(getattr(args, "since", None))
    config, specs = _resolve_specs_from_args(args, base_config, allow_empty=True)
    plans = plan_all(specs, config=config, since=since)
    current_payload = [plan_to_dict(plan) for plan in plans]
    lock_written: Optional[str] = None
    if not getattr(args, "no_lock", False):
        lock_path = getattr(args, "lock_output", DEFAULT_LOCKFILE_PATH)
        lock_written = str(write_lockfile(plans, lock_path))

    if use_manifest:
        baseline_payload = []
        for plan in plans:
            manifest = load_latest_manifest(plan.spec.id)
            if not manifest:
                continue
            baseline_payload.append(
                {
                    "id": manifest.get("id") or plan.spec.id,
                    "resolver": manifest.get("resolver"),
                    "url": manifest.get("url"),
                    "version": manifest.get("version"),
                    "license": manifest.get("license"),
                    "media_type": manifest.get("content_type"),
                    "service": manifest.get("service"),
                    "last_modified": manifest.get("last_modified"),
                    "content_length": manifest.get("content_length"),
                    "sha256": manifest.get("sha256"),
                    "normalized_sha256": manifest.get("normalized_sha256"),
                    "fingerprint": manifest.get("fingerprint"),
                    "streaming_content_sha256": manifest.get("streaming_content_sha256"),
                    "streaming_prefix_sha256": manifest.get("streaming_prefix_sha256"),
                }
            )

    diff = compute_plan_diff(baseline_payload, current_payload)
    diff["baseline"] = "manifests" if use_manifest else str(baseline_path)
    if update_baseline and not use_manifest and baseline_path is not None:
        baseline_path.write_text(json.dumps(current_payload, indent=2))
        diff["baseline_updated"] = True
    else:
        diff["baseline_updated"] = False
    if lock_written:
        diff["lockfile"] = lock_written
    return diff


def _handle_plugins(args) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Return plugin inventory for the requested kind."""

    if args.kind == "all":
        return {
            "resolver": dict(_collect_plugin_details("resolver")),
            "validator": dict(_collect_plugin_details("validator")),
        }
    return {args.kind: dict(_collect_plugin_details(args.kind))}


def _handle_prune(args, logger) -> Dict[str, object]:
    """Delete surplus ontology versions based on ``--keep`` and age filters."""

    keep = args.keep
    if keep <= 0:
        raise ConfigError("--keep must be a positive integer")

    threshold = getattr(args, "older_than", None)
    if threshold is not None and threshold.tzinfo is None:
        threshold = threshold.replace(tzinfo=timezone.utc)

    target_ids = args.ids or STORAGE.available_ontologies()
    target_ids = sorted(set(target_ids))

    summary: List[Dict[str, object]] = []
    total_reclaimed = 0
    total_deleted = 0
    messages: List[str] = []

    for ontology_id in target_ids:
        raw_metadata = collect_version_metadata(ontology_id)
        metadata = sorted(
            raw_metadata,
            key=lambda item: item.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        if not metadata:
            continue

        removal_set = {entry["version"] for entry in metadata[keep:]}
        if threshold is not None:
            for entry in metadata:
                timestamp = entry.get("timestamp")
                if isinstance(timestamp, datetime) and timestamp < threshold:
                    removal_set.add(entry["version"])

        required = min(keep, len(metadata))
        retained_versions: List[str] = []
        for entry in metadata:
            version = entry["version"]
            if version in removal_set:
                continue
            retained_versions.append(version)
            if len(retained_versions) == required:
                break
        if len(retained_versions) < required:
            for entry in metadata:
                version = entry["version"]
                if version in retained_versions:
                    continue
                retained_versions.append(version)
                removal_set.discard(version)
                if len(retained_versions) == required:
                    break

        to_remove = [entry for entry in metadata if entry["version"] in removal_set]
        if not to_remove:
            if args.dry_run and threshold is not None:
                messages.append(f"[DRY-RUN] {ontology_id}: no versions matched prune criteria")
            continue

        retained_entries = [entry for entry in metadata if entry["version"] in retained_versions]

        reclaimed = 0
        deleted_versions: List[str] = []
        if args.dry_run:
            for entry in to_remove:
                reclaimed += int(entry.get("size") or 0)
                deleted_versions.append(entry["version"])
            total_deleted += len(to_remove)
        else:
            for entry in to_remove:
                version = entry["version"]
                reclaimed_bytes = STORAGE.delete_version(ontology_id, version)
                reclaimed += reclaimed_bytes
                total_deleted += 1
                deleted_versions.append(version)
                if logger is not None:
                    logger.info(
                        "pruned ontology version",
                        extra={
                            "ontology_id": ontology_id,
                            "version": version,
                            "freed_bytes": reclaimed_bytes,
                        },
                    )
            if retained_entries:
                STORAGE.set_latest_version(ontology_id, retained_entries[0]["version"])

        total_reclaimed += reclaimed
        keep_summary = ", ".join(entry["version"] for entry in retained_entries) or "none"
        summary.append(
            {
                "id": ontology_id,
                "deleted": deleted_versions,
                "retained": [entry["version"] for entry in retained_entries],
                "reclaimed_bytes": reclaimed,
                "threshold": threshold.isoformat() if threshold else None,
            }
        )
        if args.dry_run:
            age_note = f" older than {threshold.date()}" if threshold else ""
            messages.append(
                f"[DRY-RUN] {ontology_id}: keep {keep_summary}; would delete {len(deleted_versions)} version(s){age_note} freeing {format_bytes(reclaimed)}"
            )
        else:
            messages.append(
                f"Deleted {len(deleted_versions)} versions for {ontology_id} (freed {format_bytes(reclaimed)}; kept {keep_summary})"
            )

    return {
        "ontologies": summary,
        "total_deleted": total_deleted,
        "total_reclaimed_bytes": total_reclaimed,
        "dry_run": bool(args.dry_run),
        "messages": messages,
    }


def _read_api_key_status(path: Path) -> Dict[str, object]:
    """Return diagnostic metadata for an API key file.

    Args:
        path: Target file path to inspect.

    Returns:
        Mapping including the path, whether it exists, configuration status, and
        any error encountered while reading the contents.
    """

    info: Dict[str, object] = {"path": str(path), "exists": path.exists(), "configured": False}
    if not info["exists"]:
        return info

    try:
        info["configured"] = path.read_text().strip() != ""
    except OSError as exc:
        info["error"] = f"{exc.__class__.__name__}: {exc}"

    return info


def _doctor_report() -> Dict[str, object]:
    """Collect diagnostic information for the ``doctor`` command.

    Returns:
        Mapping capturing disk, dependency, network, and configuration status.
    """

    ontology_dir = LOCAL_ONTOLOGY_DIR
    created_for_diagnostics = False
    if not ontology_dir.exists():
        try:
            ontology_dir.mkdir(parents=True, exist_ok=True)
            created_for_diagnostics = True
        except OSError:
            # The directory could not be created (e.g., permissions). We'll
            # continue with fallback disk usage handling below.
            pass

    directories = {}
    for name, path in {
        "configs": CONFIG_DIR,
        "cache": CACHE_DIR,
        "logs": LOG_DIR,
        "ontologies": ontology_dir,
    }.items():
        entry = {
            "path": str(path),
            "exists": path.exists(),
            "writable": os.access(path, os.W_OK),
        }
        if name == "ontologies" and created_for_diagnostics:
            entry["created_for_diagnostics"] = True
        directories[name] = entry

    probe_path: Path = ontology_dir
    if not probe_path.exists():
        for candidate in ontology_dir.parents:
            if candidate.exists():
                probe_path = candidate
                break
        else:
            probe_path = Path("/")

    disk_usage = shutil.disk_usage(LOCAL_ONTOLOGY_DIR)
    default_floor_bytes = 10 * 1_000_000_000
    threshold_bytes = min(
        disk_usage.total,
        max(default_floor_bytes, int(disk_usage.total * 0.1)),
    )
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
    bioportal = _read_api_key_status(api_key_path)

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


def _apply_doctor_fixes(report: Dict[str, object]) -> List[str]:
    """Attempt to remediate common doctor issues and return action notes."""

    actions: List[str] = []

    for info in report.get("directories", {}).values():
        path = Path(info.get("path", ""))
        if not path:
            continue
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            actions.append(f"Created directory {path}")

    if LOG_DIR.exists():
        retention_days = ResolvedConfig.from_defaults().defaults.logging.retention_days
        try:
            rotations = _cleanup_logs(LOG_DIR, retention_days)
            actions.extend(rotations)
        except OSError as exc:
            actions.append(f"Failed to rotate logs in {LOG_DIR}: {exc}")

    placeholders = {
        CONFIG_DIR / "bioportal_api_key.txt": "Add your BioPortal API key here\n",
        CONFIG_DIR / "ols_api_token.txt": "Add your OLS API token here\n",
    }
    for path, content in placeholders.items():
        try:
            if not path.exists() or not path.read_text().strip():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                actions.append(f"Ensured placeholder {path.name}")
        except OSError as exc:
            actions.append(f"Failed to update {path}: {exc}")

    return actions


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
        print(
            "  Warning: free space below threshold "
            f"({threshold_gb:.2f} GB; min(total capacity, max(10 GB, 10% of capacity)))."
        )

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
    manifest_path = ensure_manifest_path(args.id, args.version)
    manifest = load_manifest(manifest_path)
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
    manifest_path = ensure_manifest_path(args.id, args.version)
    manifest = load_manifest(manifest_path)
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


# --- Error handling helpers ---


def _emit_batch_failure(exc: Union[BatchPlanningError, BatchFetchError], args) -> None:
    """Print a concise error message and optionally serialize partial results."""

    message = exc.args[0] if exc.args else str(exc)
    print(f"Error: {message}", file=sys.stderr)
    if not getattr(args, "json", False):
        return

    payload: List[dict]
    if isinstance(exc, BatchPlanningError):
        payload = [plan_to_dict(plan) for plan in getattr(exc, "completed", [])]
    else:
        payload = [results_to_dict(result) for result in getattr(exc, "completed", [])]

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


# --- Module Entry Points ---


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
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
    arg_list = _normalize_plan_args(raw_args)
    parser = _build_parser()
    args = parser.parse_args(arg_list)
    try:
        base_config = get_default_config(copy=True)
        base_config.defaults.logging.level = args.log_level
        logging_config = base_config.defaults.logging
        logger = setup_logging(
            level=logging_config.level,
            retention_days=logging_config.retention_days,
            max_log_size_mb=logging_config.max_log_size_mb,
        )
        if getattr(args, "json", False):
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setStream(sys.stderr)
        if args.command == "pull":
            if getattr(args, "dry_run", False):
                plans = _handle_pull(args, base_config, dry_run=True, logger=logger)
                if args.json:
                    json.dump([plan_to_dict(plan) for plan in plans], sys.stdout, indent=2)
                    sys.stdout.write("\n")
                else:
                    if plans:
                        rows = format_plan_rows(plans)
                        print(format_table(PLAN_TABLE_HEADERS, rows))
                    else:
                        print("No ontologies to process")
            else:
                results = _handle_pull(args, base_config, dry_run=False, logger=logger)
                if args.json:
                    json.dump([results_to_dict(result) for result in results], sys.stdout, indent=2)
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
            plans = _handle_plan(args, base_config, logger=logger)
            if args.json:
                json.dump([plan_to_dict(plan) for plan in plans], sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                if plans:
                    rows = format_plan_rows(plans)
                    print(format_table(PLAN_TABLE_HEADERS, rows))
                else:
                    print("No ontologies to process")
        elif args.command == "plan-diff":
            diff = _handle_plan_diff(args, base_config)
            if args.json:
                json.dump(diff, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                lines = format_plan_diff(diff)
                if lines:
                    print("\n".join(lines))
                else:
                    print("No plan differences detected")
                if diff.get("baseline_updated"):
                    print(f"Updated baseline at {diff['baseline']}")
        elif args.command == "plugins":
            inventory = _handle_plugins(args)
            if args.json:
                json.dump(inventory, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                for kind, plugins in inventory.items():
                    print(f"{kind}:")
                    if not plugins:
                        print("  (none)")
                        continue
                    for name, data in plugins.items():
                        qualified = data.get("qualified")
                        version = data.get("version", "unknown")
                        print(f"  - {name} ({version}): {qualified}")
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
                print(f"{label}: reclaimed {format_bytes(total)} across {deleted} versions")
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
            fixes: List[str] = []
            if getattr(args, "fix", False):
                fixes = _apply_doctor_fixes(report)
                report = _doctor_report()
                if fixes:
                    report["fixes"] = fixes
            if args.json:
                json.dump(report, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                _print_doctor_report(report)
                if fixes:
                    print("\nApplied fixes:")
                    for action in fixes:
                        print(f"  - {action}")
        else:  # pragma: no cover - argparse should prevent unknown commands
            parser.error(f"Unsupported command: {args.command}")
        return 0
    except BatchPlanningError as exc:
        _emit_batch_failure(exc, args)
        return 1
    except BatchFetchError as exc:
        _emit_batch_failure(exc, args)
        return 1
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except UnsupportedPythonError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except OntologyDownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1
