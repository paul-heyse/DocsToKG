# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload",
#   "purpose": "Package initialization for DocsToKG.OntologyDownload",
#   "sections": [
#     {
#       "id": "getattr",
#       "name": "__getattr__",
#       "anchor": "function-getattr",
#       "kind": "function"
#     },
#     {
#       "id": "dir",
#       "name": "__dir__",
#       "anchor": "function-dir",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Public API for the DocsToKG ontology downloader and resolver pipeline.

This facade exposes the primary fetch utilities used by external callers to
plan resolver fallback chains, download ontologies with hardened validation,
perform stream normalization, and emit schema-compliant manifests with
deterministic fingerprints.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

_API_EXPORTS = (
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
)

__all__ = [*_API_EXPORTS, "PUBLIC_API_MANIFEST"]

_LEGACY_EXPORTS = {
    "DefaultsConfig",
    "DownloadConfiguration",
    "ResolvedConfig",
    "ResolverCandidate",
}

if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    from .api import (
        DownloadFailure,
        DownloadResult,
        FetchResult,
        FetchSpec,
        DefaultsConfig,
        DownloadConfiguration,
        OntologyDownloadError,
        PlannedFetch,
        ResolverCandidate,
        ResolvedConfig,
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


def __getattr__(name: str) -> Any:
    """Lazily import API exports to avoid resolver dependencies at import time."""

    if name in __all__ or name in _LEGACY_EXPORTS:
        module = import_module(".api", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Expose lazily-populated attributes in ``dir()`` results."""

    return sorted(set(globals()) | set(__all__) | _LEGACY_EXPORTS)

PUBLIC_API_MANIFEST: dict
