"""Public facade aggregating ontology downloader modules."""

from __future__ import annotations

import hashlib as _hashlib
from collections import OrderedDict
from importlib import metadata as importlib_metadata
from typing import Dict

from .config import (
    ConfigError,
    DefaultsConfig,
    DownloadConfiguration,
    LoggingConfiguration,
    ResolvedConfig,
    ValidationConfig,
    build_resolved_config,
    ensure_python_version,
    get_env_overrides,
    load_config,
    load_raw_yaml,
    parse_rate_limit_to_rps,
    validate_config,
)
from .io_safe import (
    extract_archive_safe,
    generate_correlation_id,
    mask_sensitive_data,
    sanitize_filename,
    validate_url_security,
)
from .net import (
    RDF_MIME_ALIASES,
    DownloadFailure,
    DownloadResult,
    download_stream,
    retry_with_backoff,
)
from .optdeps import get_owlready2, get_pronto, get_pystow, get_rdflib
from .pipeline import (
    MANIFEST_JSON_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    ConfigurationError,
    FetchResult,
    FetchSpec,
    OntologyDownloadError,
    PlannedFetch,
    ResolverCandidate,
    _build_destination,
    fetch_all,
    fetch_one,
    get_manifest_schema,
    infer_version_timestamp,
    merge_defaults,
    parse_http_datetime,
    parse_iso_datetime,
    parse_version_timestamp,
    plan_all,
    plan_one,
    setup_logging,
    validate_manifest_dict,
)
from .storage import CACHE_DIR, CONFIG_DIR, LOCAL_ONTOLOGY_DIR, LOG_DIR, STORAGE
from .resolvers import RESOLVERS
from .validation_core import (
    VALIDATORS,
    ValidationRequest,
    ValidationResult,
    ValidationTimeout,
    ValidatorSubprocessError,
    normalize_streaming,
    run_validators,
    validate_arelle,
    validate_owlready2,
    validate_pronto,
    validate_rdflib,
    validate_robot,
)
from .validation_core import (
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


def main() -> None:
    """CLI entry point delegating to the validation worker."""

    validation_main()


__all__ = [
    "FetchSpec",
    "FetchResult",
    "PlannedFetch",
    "DownloadResult",
    "DownloadFailure",
    "OntologyDownloadError",
    "ValidationRequest",
    "ValidationResult",
    "ValidationTimeout",
    "ValidatorSubprocessError",
    "ResolverCandidate",
    "ResolvedConfig",
    "ConfigError",
    "ConfigurationError",
    "DefaultsConfig",
    "DownloadConfiguration",
    "ValidationConfig",
    "LoggingConfiguration",
    "build_resolved_config",
    "get_env_overrides",
    "load_raw_yaml",
    "merge_defaults",
    "validate_config",
    "CACHE_DIR",
    "CONFIG_DIR",
    "LOG_DIR",
    "LOCAL_ONTOLOGY_DIR",
    "STORAGE",
    "RDF_MIME_ALIASES",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_JSON_SCHEMA",
    "fetch_one",
    "fetch_all",
    "plan_one",
    "plan_all",
    "_build_destination",
    "download_stream",
    "extract_archive_safe",
    "validate_manifest_dict",
    "validate_url_security",
    "run_validators",
    "normalize_streaming",
    "validate_rdflib",
    "validate_pronto",
    "validate_owlready2",
    "validate_robot",
    "validate_arelle",
    "main",
    "parse_iso_datetime",
    "parse_http_datetime",
    "parse_version_timestamp",
    "infer_version_timestamp",
    "setup_logging",
    "load_config",
    "ensure_python_version",
    "retry_with_backoff",
    "sanitize_filename",
    "generate_correlation_id",
    "parse_rate_limit_to_rps",
    "get_pystow",
    "get_rdflib",
    "get_pronto",
    "get_owlready2",
    "get_manifest_schema",
    "mask_sensitive_data",
    "list_plugins",
    "about",
    "__version__",
    "hashlib",
]
def _describe_plugin(obj: object) -> str:
    module = getattr(obj, "__module__", obj.__class__.__module__)
    name = getattr(obj, "__qualname__", obj.__class__.__name__)
    return f"{module}.{name}"


def list_plugins(kind: str) -> Dict[str, str]:
    """Return a deterministic mapping of registered plugins for ``kind``.

    Args:
        kind: Plugin category (``"resolver"`` or ``"validator"``).

    Returns:
        Mapping of plugin names to import-qualified identifiers.
    """

    if kind == "resolver":
        items = {name: _describe_plugin(resolver) for name, resolver in RESOLVERS.items()}
    elif kind == "validator":
        items = {name: _describe_plugin(handler) for name, handler in VALIDATORS.items()}
    else:
        raise ValueError(f"Unknown plugin kind: {kind}")
    return OrderedDict(sorted(items.items()))


def about() -> Dict[str, object]:
    """Return metadata describing the ontology download subsystem."""

    return {
        "package_version": _PACKAGE_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "plugins": {
            "resolver": list_plugins("resolver"),
            "validator": list_plugins("validator"),
        },
    }
