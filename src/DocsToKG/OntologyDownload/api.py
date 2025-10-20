# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.api",
#   "purpose": "CLI entry points for ontology downloads",
#   "sections": [
#     {
#       "id": "results-to-dict",
#       "name": "_results_to_dict",
#       "anchor": "function-results-to-dict",
#       "kind": "function"
#     },
#     {
#       "id": "validator-worker-main",
#       "name": "validator_worker_main",
#       "anchor": "function-validator-worker-main",
#       "kind": "function"
#     },
#     {
#       "id": "validate-url-security",
#       "name": "validate_url_security",
#       "anchor": "function-validate-url-security",
#       "kind": "function"
#     },
#     {
#       "id": "list-plugins",
#       "name": "list_plugins",
#       "anchor": "function-list-plugins",
#       "kind": "function"
#     },
#     {
#       "id": "collect-plugin-details",
#       "name": "_collect_plugin_details",
#       "anchor": "function-collect-plugin-details",
#       "kind": "function"
#     },
#     {
#       "id": "about",
#       "name": "about",
#       "anchor": "function-about",
#       "kind": "function"
#     },
#     {
#       "id": "cli-main",
#       "name": "cli_main",
#       "anchor": "function-cli-main",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""High-level orchestration helpers for ontology download workflows.

This module glues together the planner, resolver, manifest, and validation
subsystems exposed throughout ``DocsToKG.OntologyDownload``.  It provides the
functions that sit behind the CLI, including helpers to:

- translate plan and validation results into table-friendly structures,
- enumerate registered resolvers and validators with their metadata,
- execute downloads and validation pipelines while honouring security checks,
- surface convenience entry points consumed by ``DocsToKG.OntologyDownload.cli``.

External callers typically import from :mod:`DocsToKG.OntologyDownload` which
re-exports the public symbols defined here; keeping this module cohesive
ensures the CLI and programmatic APIs stay in sync.
"""

from __future__ import annotations

import hashlib as _hashlib
import sys
from collections import OrderedDict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Optional, Sequence

from . import plugins as plugin_mod
from .errors import (
    PolicyError,
)
from .exports import PUBLIC_EXPORT_NAMES
from .formatters import (
    PLAN_TABLE_HEADERS,
    RESULT_TABLE_HEADERS,
    VALIDATION_TABLE_HEADERS,
    format_plan_rows,
    format_results_table,
    format_table,
    format_validation_summary,
)
from .io import (
    mask_sensitive_data,
)
from .io import (
    validate_url_security as _validate_url_security,
)
from .manifests import results_to_dict as _manifest_results_to_dict
from .planning import (
    MANIFEST_SCHEMA_VERSION,
    FetchResult,
)
from .planning import (
    FetchSpec as _FetchSpec,
)
from .planning import (
    fetch_all as _fetch_all,
)
from .planning import (
    fetch_one as _fetch_one,
)
from .settings import (
    CACHE_DIR,
    LOCAL_ONTOLOGY_DIR,
    LOG_DIR,
    STORAGE,
    ConfigError,
    DefaultsConfig,
    DownloadConfiguration,
    LoggingConfiguration,
    ValidationConfig,
    get_default_config,
)
from .validation import (
    main as validation_main,
)

hashlib = _hashlib

try:  # pragma: no cover - metadata may be unavailable during development
    _PACKAGE_VERSION = importlib_metadata.version("DocsToKG")
except importlib_metadata.PackageNotFoundError:  # pragma: no cover - local source tree
    _PACKAGE_VERSION = "0.0.0"

__version__ = _PACKAGE_VERSION

DefaultsConfiguration = DefaultsConfig
LoggingConfig = LoggingConfiguration
ValidationConfiguration = ValidationConfig


def _results_to_dict(result: FetchResult) -> dict:
    """Compatibility wrapper returning CLI-oriented fetch result payloads."""

    payload = _manifest_results_to_dict(result)
    payload["manifest"] = str(result.manifest_path) if result.manifest_path else ""
    payload["artifacts"] = [str(path) for path in result.artifacts]
    return payload


def validator_worker_main() -> None:
    """Entry point used by validator worker console scripts."""

    validation_main()


def validate_url_security(url: str, http_config: Optional[DownloadConfiguration] = None) -> str:
    """Wrapper that normalizes PolicyError into ConfigError for the public API."""

    try:
        return _validate_url_security(url, http_config)
    except PolicyError as exc:  # pragma: no cover - thin wrapper
        raise ConfigError(str(exc)) from exc


__all__ = list(PUBLIC_EXPORT_NAMES)
__all__ += [
    "format_table",
    "format_plan_rows",
    "format_results_table",
    "format_validation_summary",
    "PLAN_TABLE_HEADERS",
    "RESULT_TABLE_HEADERS",
    "VALIDATION_TABLE_HEADERS",
    "_results_to_dict",
    "mask_sensitive_data",
    "FetchSpec",
    "fetch_one",
    "fetch_all",
]

FetchSpec = _FetchSpec
fetch_one = _fetch_one
fetch_all = _fetch_all


def list_plugins(kind: str) -> Dict[str, str]:
    """Return a deterministic mapping of registered plugins for ``kind``.

    Args:
        kind: Plugin category (``"resolver"`` or ``"validator"``).

    Returns:
        Mapping of plugin names to import-qualified identifiers.
    """

    try:
        return plugin_mod.list_registered_plugins(kind)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown plugin kind: {kind}") from exc


def _collect_plugin_details(kind: str) -> "OrderedDict[str, Dict[str, str]]":
    """Return plugin metadata including qualified path and version."""

    if kind not in {"resolver", "validator"}:
        raise ValueError(f"Unknown plugin kind: {kind}")

    discovered_meta = plugin_mod.get_registered_plugin_meta(kind)
    details: Dict[str, Dict[str, str]] = {}
    for name, qualified in list_plugins(kind).items():
        meta = discovered_meta.get(name, {})
        resolved_qualified = meta.get("qualified", qualified)
        version = meta.get("version")
        if not version or version == "unknown":
            module_root = resolved_qualified.split(".")[0]
            try:
                version = importlib_metadata.version(module_root)
            except importlib_metadata.PackageNotFoundError:
                if resolved_qualified.startswith("DocsToKG."):
                    version = _PACKAGE_VERSION
                else:
                    version = "unknown"
        details[name] = {
            "qualified": resolved_qualified,
            "version": version,
        }
    return OrderedDict(sorted(details.items()))


def about() -> Dict[str, object]:
    """Return metadata describing the ontology download subsystem."""
    config = get_default_config()
    defaults = config.defaults

    logging_cfg = defaults.logging
    http_cfg = defaults.http
    validation_cfg = defaults.validation

    config_summary = {
        "logging": {
            "level": logging_cfg.level,
            "retention_days": logging_cfg.retention_days,
            "max_log_size_mb": logging_cfg.max_log_size_mb,
        },
        "http": {
            "timeout_sec": http_cfg.timeout_sec,
            "download_timeout_sec": http_cfg.download_timeout_sec,
            "max_retries": http_cfg.max_retries,
            "concurrent_plans": http_cfg.concurrent_plans,
            "concurrent_downloads": http_cfg.concurrent_downloads,
            "strict_dns": http_cfg.strict_dns,
        },
        "validation": {
            "parser_timeout_sec": validation_cfg.parser_timeout_sec,
            "max_concurrent_validators": validation_cfg.max_concurrent_validators,
            "use_process_pool": validation_cfg.use_process_pool,
        },
    }

    rate_limits = {
        "per_host": http_cfg.per_host_rate_limit,
        "services": dict(http_cfg.rate_limits),
    }

    storage_backend = STORAGE
    storage_info: Dict[str, object] = {
        "backend": type(storage_backend).__name__,
    }
    local_root = getattr(storage_backend, "root", None)
    if isinstance(local_root, Path):
        storage_info["local_root"] = str(local_root)
    remote_root = getattr(storage_backend, "base_path", None)
    if remote_root is not None:
        storage_info["remote_root"] = str(remote_root)

    return {
        "package_version": _PACKAGE_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "plugins": {
            "resolver": _collect_plugin_details("resolver"),
            "validator": _collect_plugin_details("validator"),
        },
        "config": config_summary,
        "rate_limits": rate_limits,
        "paths": {
            "cache_dir": str(CACHE_DIR),
            "log_dir": str(LOG_DIR),
            "ontology_dir": str(LOCAL_ONTOLOGY_DIR),
        },
        "storage": storage_info,
    }


# --- Globals ---

ONTOLOGY_DIR = LOCAL_ONTOLOGY_DIR


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ontology downloader CLI."""

    from .cli import cli_main as _cli_main

    return _cli_main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli_main())
