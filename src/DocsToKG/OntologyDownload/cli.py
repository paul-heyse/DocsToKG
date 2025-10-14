"""Command-line interface for the ontology downloader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import ConfigError, ResolvedConfig, load_config
from .core import ONTOLOGY_DIR, FetchResult, FetchSpec, fetch_all
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

    show = subparsers.add_parser("show", help="Display ontology metadata")
    show.add_argument("id", help="Ontology identifier")
    show.add_argument("--versions", action="store_true", help="List available versions")
    show.add_argument("--json", action="store_true", help="Output manifest as JSON")

    validate = subparsers.add_parser("validate", help="Re-run validation on downloaded ontologies")
    validate.add_argument("id", help="Ontology identifier")
    validate.add_argument("version", nargs="?", help="Specific version to validate")

    init = subparsers.add_parser("init", help="Create example sources.yaml configuration")
    init.add_argument("path", nargs="?", default=Path("sources.yaml"), type=Path)

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
    if args.spec:
        config = load_config(args.spec)
        config.defaults.logging.level = args.log_level
        results = fetch_all(config.specs, config=config, force=args.force)
        return results
    if not args.ids:
        raise ConfigError("Please provide ontology IDs or --spec configuration")
    config = base_config or ResolvedConfig.from_defaults()
    config.defaults.logging.level = args.log_level
    resolver = args.resolver or config.defaults.prefer_source[0]
    specs = [FetchSpec(id=oid, resolver=resolver, extras={}, target_formats=target_formats or config.defaults.normalize_to) for oid in args.ids]
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


def _handle_validate(args, config: ResolvedConfig) -> None:
    manifest_path = _ensure_manifest_path(args.id, args.version)
    manifest = _load_manifest(manifest_path)
    version_dir = manifest_path.parent
    original_path = version_dir / "original" / manifest["filename"]
    validation_dir = version_dir / "validation"
    normalized_dir = version_dir / "normalized"
    request = ValidationRequest(
        name="rdflib",
        file_path=original_path,
        normalized_dir=normalized_dir,
        validation_dir=validation_dir,
        config=config,
    )
    additional = [
        ValidationRequest("pronto", original_path, normalized_dir, validation_dir, config),
        ValidationRequest("owlready2", original_path, normalized_dir, validation_dir, config),
        ValidationRequest("robot", original_path, normalized_dir, validation_dir, config),
        ValidationRequest("arelle", original_path, normalized_dir, validation_dir, config),
    ]
    logger = setup_logging(config.defaults.logging)
    requests = [request, *additional]
    results = run_validators(requests, logger)
    manifest["validation"] = {name: result.to_dict() for name, result in results.items()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["validation"], indent=2))


def _handle_init(path: Path) -> None:
    if path.exists():
        raise ConfigError(f"Refusing to overwrite existing file {path}")
    path.write_text(EXAMPLE_SOURCES_YAML)
    print(f"Wrote example configuration to {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        base_config = ResolvedConfig.from_defaults()
        base_config.defaults.logging.level = args.log_level
        logger = setup_logging(base_config.defaults.logging)
        if args.command == "pull":
            results = _handle_pull(args, base_config)
            for result in results:
                logger.info(
                    "ontology processed",
                    extra={"stage": "complete", "ontology_id": result.spec.id, "status": result.status},
                )
        elif args.command == "show":
            _handle_show(args)
        elif args.command == "validate":
            _handle_validate(args, base_config)
        elif args.command == "init":
            _handle_init(args.path)
        return 0
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
