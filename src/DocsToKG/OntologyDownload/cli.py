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
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

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

    plan_cmd = subparsers.add_parser("plan", help="Preview resolver plans without downloading")
    plan_cmd.add_argument("ids", nargs="*", help="Ontology identifiers to plan")
    plan_cmd.add_argument(
        "--spec", type=Path, help="Path to sources.yaml (default: configs/sources.yaml)"
    )
    plan_cmd.add_argument("--resolver", help="Resolver type for single ontology")
    plan_cmd.add_argument("--target-formats", help="Comma-separated formats (e.g., owl,obo)")
    plan_cmd.add_argument("--json", action="store_true", help="Emit plan details as JSON")

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
    """Collect diagnostic information for the ``doctor`` command.

    Returns:
        Mapping containing directory health, dependency availability, remote
        service connectivity, and storage backend information.
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

    api_key_path = CONFIG_DIR / "bioportal_api_key.txt"
    bioportal = {
        "path": str(api_key_path),
        "configured": api_key_path.exists() and api_key_path.read_text().strip() != "",
    }

    try:
        response = requests.get(
            "https://www.ebi.ac.uk/ols4/api/health",
            timeout=5,
        )
        ols_status = {
            "ok": response.ok,
            "detail": f"status {response.status_code}",
        }
    except Exception as exc:  # pragma: no cover - network failures vary
        ols_status = {"ok": False, "detail": f"error: {exc}"}

    disk_usage = shutil.disk_usage(LOCAL_ONTOLOGY_DIR)

    module_checks = {
        "rdflib": importlib.util.find_spec("rdflib") is not None,
        "pronto": importlib.util.find_spec("pronto") is not None,
        "owlready2": importlib.util.find_spec("owlready2") is not None,
        "arelle": importlib.util.find_spec("arelle") is not None,
    }
    module_checks["robot"] = shutil.which("robot") is not None

    storage_backend = {
        "backend": STORAGE.__class__.__name__,
        "remote": hasattr(STORAGE, "fs"),
    }

    report: Dict[str, object] = {
        "directories": directories,
        "bioportal_api_key": bioportal,
        "ols_api": ols_status,
        "disk": {
            "total_bytes": disk_usage.total,
            "free_bytes": disk_usage.free,
        },
        "dependencies": module_checks,
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

    bioportal = report["bioportal_api_key"]
    print(
        "BioPortal API key:",
        "configured" if bioportal["configured"] else f"missing ({bioportal['path']})",
    )

    ols = report["ols_api"]
    print("OLS API:", "accessible" if ols["ok"] else f"unreachable ({ols['detail']})")

    disk = report["disk"]
    free_gb = disk["free_bytes"] / (1024**3)
    print(f"Disk free: {free_gb:.2f} GB")

    print("Optional dependencies:")
    for name, available in report["dependencies"].items():
        print(f"  - {name}: {'available' if available else 'missing'}")

    storage = report["storage"]
    backend_desc = storage["backend"]
    if storage["remote"]:
        backend_desc += " (remote)"
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
