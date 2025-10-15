"""Ontology downloader public API.

Expose the primary fetch utilities used by external callers to plan resolver
fallback chains, download ontologies with hardened validation, stream
normalization, and emit schema-compliant manifests with deterministic
fingerprints.
"""

from __future__ import annotations

import importlib
import sys

from .foundation import generate_correlation_id, retry_with_backoff, sanitize_filename
from .infrastructure import (
    CACHE_DIR,
    CONFIG_DIR,
    LOCAL_ONTOLOGY_DIR,
    LOG_DIR,
    STORAGE,
    get_owlready2,
    get_pronto,
    get_pystow,
    get_rdflib,
)
from .network import (
    DownloadFailure,
    DownloadResult,
    RDF_MIME_ALIASES,
    download_stream,
    extract_archive_safe,
    validate_url_security,
)
from .pipeline import (
    FetchResult,
    FetchSpec,
    MANIFEST_SCHEMA_VERSION,
    OntologyDownloadError,
    PlannedFetch,
    fetch_all,
    fetch_one,
    get_manifest_schema,
    plan_all,
    plan_one,
    validate_manifest_dict,
)
from .settings import ConfigError, ResolvedConfig, ensure_python_version, load_config, setup_logging
from .validation import ValidationRequest, ValidationResult, run_validators

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

# Backwards-compatible module aliases for legacy import paths.
_LEGACY_MODULE_MAP = {
    ".core": ".pipeline",
    ".config": ".settings",
    ".validators": ".validation",
    ".download": ".network",
    ".storage": ".infrastructure",
    ".optdeps": ".infrastructure",
    ".utils": ".foundation",
    ".logging_config": ".settings",
    ".validator_workers": ".validation",
    ".cli_utils": ".cli",
}

for legacy_suffix, target_suffix in _LEGACY_MODULE_MAP.items():
    legacy_name = __name__ + legacy_suffix
    target_name = __name__ + target_suffix
    if legacy_name not in sys.modules:
        module = importlib.import_module(target_name)
        sys.modules.setdefault(legacy_name, module)
