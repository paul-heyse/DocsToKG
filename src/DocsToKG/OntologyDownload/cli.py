"""
Ontology Downloader CLI

This module exposes the `ontofetch` command-line experience for DocsToKG.
It provides entry points for downloading ontologies, inspecting manifests,
re-running validators, and bootstrapping configuration files. The CLI is
designed to support both automated pipelines and human operators by offering
structured JSON output, progress tables, and detailed error reporting.

Key Features:
- Multi-command interface covering pull, show, validate, init, and config tasks
- Seamless integration with resolver planning and validation subsystems
- Support for JSON output to aid automation and downstream tooling
- Logging configuration that aligns with DocsToKG observability standards

Dependencies:
- argparse: command-line parsing
- pathlib: filesystem path handling
- DocsToKG.OntologyDownload.core: download orchestration helpers
- DocsToKG.OntologyDownload.validators: validation pipeline execution

Usage:
    from DocsToKG.OntologyDownload import cli

    if __name__ == "__main__":
        raise SystemExit(cli.main())
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
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
import yaml

from .cli_utils import (
    format_plan_rows,
    format_results_table,
    format_table,
    format_validation_summary,
)
from .config import ConfigError, ResolvedConfig, load_config, validate_config
from .core import (
    FetchResult,
    FetchSpec,
    OntologyDownloadError,
    PlannedFetch,
    fetch_all,
    plan_all,
)
from .logging_config import setup_logging
from .storage import CACHE_DIR, CONFIG_DIR, LOCAL_ONTOLOGY_DIR, LOG_DIR, STORAGE
from .validators import ValidationRequest, run_validators

ONTOLOGY_DIR = LOCAL_ONTOLOGY_DIR


def _build_parser() -> argparse.ArgumentParser:
    """Configure the top-level CLI parser and subcommands.

    Returns:
        Parser instance with sub-commands for pull, show, validate, init, and config.

    Raises:
        None
    """
    parser = argparse.ArgumentParser(
        prog="ontofetch",
        description="Ontology downloader for DocsToKG supporting OBO, OLS, BioPortal, SKOS, XBRL sources.",
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
        "--dry-run", action="store_true", help="Preview resolver actions without downloading"
    )
    pull.add_argument(
        "--concurrent-downloads",
        type=_parse_positive_int,
        help="Override maximum concurrent downloads for this invocation",
    )
    pull.add_argument(
        "--allowed-hosts",
        help="Comma-separated list of additional hosts permitted for this run",
    )

    plan_cmd = subparsers.add_parser("plan", help="Preview resolver plans without downloading")
    plan_cmd.add_argument("ids", nargs="*", help="Ontology identifiers to plan")
    plan_cmd.add_argument(
        "--spec", type=Path, help="Path to sources.yaml (default: configs/sources.yaml)"
    )
    plan_cmd.add_argument("--resolver", help="Resolver type for single ontology")
    plan_cmd.add_argument("--target-formats", help="Comma-separated formats (e.g., owl,obo)")
    plan_cmd.add_argument("--json", action="store_true", help="Emit plan details as JSON")
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
        "--json", action="store_true", help="Output validation result as JSON"
    )

    doctor = subparsers.add_parser("doctor", help="Diagnose environment issues")
    doctor.add_argument("--json", action="store_true", help="Output diagnostics as JSON")

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
    """Parse CLI argument ensuring it is a positive integer."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - argparse handles message
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_allowed_hosts(value: Optional[str]) -> List[str]:
    """Split comma-delimited host allowlist argument into unique entries."""

    if not value:
        return []
    entries: List[str] = []
    for host in value.split(","):
        candidate = host.strip()
        if candidate and candidate not in entries:
            entries.append(candidate)
    return entries


def _apply_cli_overrides(config: ResolvedConfig, args) -> None:
    """Mutate resolved configuration based on CLI override arguments."""

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


_RATE_LIMIT_RE = re.compile(r"^([\d.]+)/(second|sec|s|minute|min|m|hour|h)$")


def _rate_limit_to_rps(value: str) -> Optional[float]:
    """Convert rate limit string into requests-per-second float."""

    match = _RATE_LIMIT_RE.match(value)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    if unit in {"second", "sec", "s"}:
        return amount
    if unit in {"minute", "min", "m"}:
        return amount / 60.0
    if unit in {"hour", "h"}:
        return amount / 3600.0
    return None


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


def _plan_to_dict(plan: PlannedFetch) -> dict:
    """Convert a planned fetch into a JSON-friendly dictionary.

    Args:
        plan: Planned fetch data produced by :func:`plan_one` or :func:`plan_all`.

    Returns:
        Mapping containing resolver metadata and planned download details suitable
        for serialization.
    """

    candidates = [
        {
            "resolver": candidate.resolver,
            "url": candidate.plan.url,
            "service": candidate.plan.service,
            "media_type": candidate.plan.media_type,
            "headers": candidate.plan.headers,
            "version": candidate.plan.version,
            "license": candidate.plan.license,
        }
        for candidate in getattr(plan, "candidates", ())
    ]

    return {
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


def _resolve_specs_from_args(
    args, base_config: Optional[ResolvedConfig]
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

    config, specs = _resolve_specs_from_args(args, base_config)
    return plan_all(specs, config=config)


def _doctor_report() -> Dict[str, object]:
    """Collect diagnostic information for the ``doctor`` command."""

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
            http_section = (
                raw.get("defaults", {}).get("http") if isinstance(raw, dict) else None
            )
            configured = http_section.get("rate_limits") if isinstance(http_section, dict) else None
            if isinstance(configured, dict):
                valid: Dict[str, Dict[str, object]] = {}
                invalid: Dict[str, str] = {}
                for service, limit in configured.items():
                    text_value = str(limit)
                    rps = _rate_limit_to_rps(text_value)
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
        print(
            f"  Warning: free space below threshold ({threshold_gb:.2f} GB)."
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

    Returns:
        None

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
    parser = _build_parser()
    args = parser.parse_args(argv)
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
