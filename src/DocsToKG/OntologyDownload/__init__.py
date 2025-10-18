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
from typing import TYPE_CHECKING, Any

from .exports import EXPORT_MAP, EXPORTS, PUBLIC_API_MANIFEST

_PUBLIC_EXPORTS = tuple(spec.name for spec in EXPORTS if spec.include_in_manifest)

__all__ = [*_PUBLIC_EXPORTS, "PUBLIC_API_MANIFEST"]

if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    from . import api as _api

    DownloadFailure = _api.DownloadFailure
    DownloadResult = _api.DownloadResult
    FetchResult = _api.FetchResult
    FetchSpec = _api.FetchSpec
    OntologyDownloadError = _api.OntologyDownloadError
    PlannedFetch = _api.PlannedFetch
    ValidationRequest = _api.ValidationRequest
    ValidationResult = _api.ValidationResult
    ValidationTimeout = _api.ValidationTimeout
    ValidatorSubprocessError = _api.ValidatorSubprocessError
    __version__ = _api.__version__
    fetch_all = _api.fetch_all
    fetch_one = _api.fetch_one
    plan_all = _api.plan_all
    plan_one = _api.plan_one
    run_validators = _api.run_validators
    validate_manifest_dict = _api.validate_manifest_dict


def __getattr__(name: str) -> Any:
    """Lazily import API exports to avoid resolver dependencies at import time."""

    if name == "PUBLIC_API_MANIFEST":
        globals()[name] = PUBLIC_API_MANIFEST
        return PUBLIC_API_MANIFEST

    spec = EXPORT_MAP.get(name)
    if spec is not None and spec.name in _PUBLIC_EXPORTS:
        module = import_module(spec.module)
        value = getattr(module, spec.attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Expose lazily-populated attributes in ``dir()`` results."""

    return sorted(set(globals()) | set(__all__))
