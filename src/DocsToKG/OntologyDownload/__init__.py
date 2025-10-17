# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload",
#   "purpose": "Package initialization for DocsToKG.OntologyDownload",
#   "sections": []
# }
# === /NAVMAP ===

"""Public API for the DocsToKG ontology downloader and resolver pipeline.

This facade exposes the primary fetch utilities used by external callers to
plan resolver fallback chains, download ontologies with hardened validation,
perform stream normalization, and emit schema-compliant manifests with
deterministic fingerprints.
"""

from __future__ import annotations

from .api import (
    DownloadFailure,
    DownloadResult,
    FetchResult,
    FetchSpec,
    OntologyDownloadError,
    PlannedFetch,
    ValidationRequest,
    ValidationResult,
    ValidationTimeout,
    ValidatorSubprocessError,
    __version__,
    fetch_all,
    fetch_one,
    plan_all,
    plan_one,
    run_validators,
    validate_manifest_dict,
)

# --- Globals ---

# --- Globals ---

__all__ = [
    "FetchSpec",
    "PlannedFetch",
    "FetchResult",
    "DownloadResult",
    "DownloadFailure",
    "OntologyDownloadError",
    "ValidationRequest",
    "ValidationResult",
    "ValidationTimeout",
    "ValidatorSubprocessError",
    "plan_all",
    "plan_one",
    "fetch_all",
    "fetch_one",
    "run_validators",
    "validate_manifest_dict",
    "__version__",
    "PUBLIC_API_MANIFEST",
]

PUBLIC_API_MANIFEST = {
    "FetchSpec": "class",
    "PlannedFetch": "class",
    "FetchResult": "class",
    "DownloadResult": "class",
    "DownloadFailure": "class",
    "OntologyDownloadError": "class",
    "ValidationRequest": "class",
    "ValidationResult": "class",
    "ValidationTimeout": "class",
    "ValidatorSubprocessError": "class",
    "plan_all": "function",
    "plan_one": "function",
    "fetch_all": "function",
    "fetch_one": "function",
    "run_validators": "function",
    "validate_manifest_dict": "function",
    "PUBLIC_API_MANIFEST": "const",
    "__version__": "const",
}
