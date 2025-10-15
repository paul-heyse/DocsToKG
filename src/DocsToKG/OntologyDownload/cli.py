"""Command-line interface for the ontology downloader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .cli_utils import format_table, format_validation_summary
from .config import ConfigError, ResolvedConfig, load_config, validate_config
from .core import (
    CONFIG_DIR,
    ONTOLOGY_DIR,
    FetchResult,
    FetchSpec,
    OntologyDownloadError,
    fetch_all,
)
from .logging_config import setup_logging
from .validators import ValidationRequest, run_validators


def _build_parser() -> argparse.ArgumentParser:
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
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _results_to_dict(result: FetchResult) -> dict:
    return {
        "id": result.spec.id,
        "resolver": result.spec.resolver,
        "status": result.status,
        "file": str(result.local_path),
        "sha256": result.sha256,
        "manifest": str(result.manifest_path),
        "artifacts": list(result.artifacts),
    }


def _ensure_manifest_path(ontology_id: str, version: Optional[str]) -> Path:
    ontology_dir = ONTOLOGY_DIR / ontology_id
    if not ontology_dir.exists():
        raise ConfigError(f"Ontology '{ontology_id}' has not been downloaded yet")
    if version:
        version_dir = ontology_dir / version
    else:
        versions = sorted((d for d in ontology_dir.iterdir() if d.is_dir()), reverse=True)
        if not versions:
            raise ConfigError(f"No versions stored for ontology '{ontology_id}'")
        version_dir = versions[0]
    manifest_path = version_dir / "manifest.json"
    if not manifest_path.exists():
        raise ConfigError(f"Manifest not found for ontology '{ontology_id}' at {version_dir}")
    return manifest_path


def _load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text())


def _handle_pull(args, base_config: Optional[ResolvedConfig]) -> List[FetchResult]:
    target_formats = _parse_target_formats(args.target_formats)
    config_path = args.spec
    if config_path is None and not args.ids:
        default_config = CONFIG_DIR / "sources.yaml"
        if default_config.exists():
            config_path = default_config
    if config_path:
        config = load_config(config_path)
        config.defaults.logging.level = args.log_level
        results = fetch_all(config.specs, config=config, force=args.force)
        return results
    if not args.ids:
        raise ConfigError("Please provide ontology IDs or --spec configuration")
    config = base_config or ResolvedConfig.from_defaults()
    config.defaults.logging.level = args.log_level
    resolver = args.resolver or config.defaults.prefer_source[0]
    specs = [
        FetchSpec(
            id=oid,
            resolver=resolver,
            extras={},
            target_formats=target_formats or config.defaults.normalize_to,
        )
        for oid in args.ids
    ]
    results = fetch_all(specs, config=config, force=args.force)
    return results


def _handle_show(args) -> None:
    if args.versions:
        versions = sorted(d.name for d in (ONTOLOGY_DIR / args.id).iterdir() if d.is_dir())
        for version in versions:
            print(version)
        return
    manifest_path = _ensure_manifest_path(args.id, None)
    manifest = _load_manifest(manifest_path)
    if args.json:
        json.dump(manifest, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for key, value in manifest.items():
            print(f"{key}: {value}")


def _selected_validators(args) -> Sequence[str]:
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
    logger = setup_logging(config.defaults.logging)
    results = run_validators(requests, logger)
    manifest["validation"] = {name: result.to_dict() for name, result in results.items()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest["validation"]


def _handle_init(path: Path) -> None:
    if path.exists():
        raise ConfigError(f"Refusing to overwrite existing file {path}")
    path.write_text(EXAMPLE_SOURCES_YAML)
    print(f"Wrote example configuration to {path}")


def _handle_config_validate(path: Path) -> dict:
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
        logger = setup_logging(base_config.defaults.logging)
        if args.command == "pull":
            results = _handle_pull(args, base_config)
            if args.json:
                json.dump([_results_to_dict(result) for result in results], sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                if results:
                    table = format_table(
                        ("id", "resolver", "status", "sha256", "file"),
                        [
                            (
                                result.spec.id,
                                result.spec.resolver,
                                result.status,
                                result.sha256[:12],
                                result.local_path.name,
                            )
                            for result in results
                        ],
                    )
                    print(table)
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
