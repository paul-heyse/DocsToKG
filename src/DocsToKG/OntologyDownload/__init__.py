"""Public API for the DocsToKG ontology downloader and resolver pipeline.

This facade exposes the primary fetch utilities used by external callers to
plan resolver fallback chains, download ontologies with hardened validation,
perform stream normalization, and emit schema-compliant manifests with
deterministic fingerprints.
"""

from __future__ import annotations

import importlib
import sys

from .ontology_download import (
    CACHE_DIR,
    CONFIG_DIR,
    LOCAL_ONTOLOGY_DIR,
    LOG_DIR,
    MANIFEST_SCHEMA_VERSION,
    RDF_MIME_ALIASES,
    STORAGE,
    ConfigError,
    DownloadFailure,
    DownloadResult,
    FetchResult,
    FetchSpec,
    OntologyDownloadError,
    PlannedFetch,
    ResolvedConfig,
    ValidationRequest,
    ValidationResult,
    download_stream,
    ensure_python_version,
    extract_archive_safe,
    fetch_all,
    fetch_one,
    generate_correlation_id,
    get_manifest_schema,
    get_owlready2,
    get_pronto,
    get_pystow,
    get_rdflib,
    load_config,
    plan_all,
    plan_one,
    retry_with_backoff,
    run_validators,
    sanitize_filename,
    setup_logging,
    validate_manifest_dict,
    validate_url_security,
)

__all__ = [
    "FetchSpec",
    "FetchResult",
    "PlannedFetch",
    "DownloadResult",
    "DownloadFailure",
    "OntologyDownloadError",
    "ValidationRequest",
    "ValidationResult",
    "ResolvedConfig",
    "ConfigError",
    "CACHE_DIR",
    "CONFIG_DIR",
    "LOG_DIR",
    "LOCAL_ONTOLOGY_DIR",
    "STORAGE",
    "RDF_MIME_ALIASES",
    "MANIFEST_SCHEMA_VERSION",
    "fetch_one",
    "fetch_all",
    "plan_one",
    "plan_all",
    "download_stream",
    "extract_archive_safe",
    "validate_manifest_dict",
    "validate_url_security",
    "run_validators",
    "setup_logging",
    "load_config",
    "ensure_python_version",
    "retry_with_backoff",
    "sanitize_filename",
    "generate_correlation_id",
    "get_pystow",
    "get_rdflib",
    "get_pronto",
    "get_owlready2",
    "get_manifest_schema",
]

_LEGACY_MODULE_MAP = {
    ".core": ".ontology_download",
    ".config": ".ontology_download",
    ".validators": ".ontology_download",
    ".download": ".ontology_download",
    ".storage": ".ontology_download",
    ".optdeps": ".ontology_download",
    ".utils": ".ontology_download",
    ".logging_config": ".ontology_download",
    ".validator_workers": ".ontology_download",
    ".foundation": ".ontology_download",
    ".infrastructure": ".ontology_download",
    ".network": ".ontology_download",
    ".pipeline": ".ontology_download",
    ".settings": ".ontology_download",
    ".validation": ".ontology_download",
    ".cli_utils": ".cli",
}

_package = sys.modules[__name__]

for legacy_suffix, target_suffix in _LEGACY_MODULE_MAP.items():
    legacy_name = __name__ + legacy_suffix
    target_name = __name__ + target_suffix
    module = sys.modules.get(legacy_name)
    if module is None:
        module = importlib.import_module(target_name)
        sys.modules.setdefault(legacy_name, module)
    setattr(_package, legacy_suffix.lstrip("."), module)
