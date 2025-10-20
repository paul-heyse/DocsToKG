"""Central definition of public exports for ``DocsToKG.OntologyDownload``.

Import-time side effects are carefully managed in this package so that heavy
dependencies (rdflib, owlready2, etc.) are only loaded when needed.  This
module captures the canonical list of symbols that form the public API,
describes where they live, and feeds the lazy export mechanism in
``__init__.py`` as well as documentation generators.  Keeping the manifest
here ensures CLI users, integrators, and Sphinx builds all see the same surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class ExportSpec:
    """Metadata describing a public export."""

    name: str
    module: str
    attribute: str
    kind: str
    include_in_manifest: bool = True


EXPORTS: Tuple[ExportSpec, ...] = (
    ExportSpec("FetchSpec", "DocsToKG.OntologyDownload.planning", "FetchSpec", "class"),
    ExportSpec("PlannedFetch", "DocsToKG.OntologyDownload.planning", "PlannedFetch", "class"),
    ExportSpec("FetchResult", "DocsToKG.OntologyDownload.planning", "FetchResult", "class"),
    ExportSpec("DownloadResult", "DocsToKG.OntologyDownload.io", "DownloadResult", "class"),
    ExportSpec("DownloadFailure", "DocsToKG.OntologyDownload.errors", "DownloadFailure", "class"),
    ExportSpec(
        "OntologyDownloadError",
        "DocsToKG.OntologyDownload.errors",
        "OntologyDownloadError",
        "class",
    ),
    ExportSpec(
        "ValidationRequest", "DocsToKG.OntologyDownload.validation", "ValidationRequest", "class"
    ),
    ExportSpec(
        "ValidationResult", "DocsToKG.OntologyDownload.validation", "ValidationResult", "class"
    ),
    ExportSpec(
        "ValidationTimeout", "DocsToKG.OntologyDownload.validation", "ValidationTimeout", "class"
    ),
    ExportSpec(
        "ValidatorSubprocessError",
        "DocsToKG.OntologyDownload.validation",
        "ValidatorSubprocessError",
        "class",
    ),
    ExportSpec("plan_all", "DocsToKG.OntologyDownload.planning", "plan_all", "function"),
    ExportSpec("plan_one", "DocsToKG.OntologyDownload.planning", "plan_one", "function"),
    ExportSpec("fetch_all", "DocsToKG.OntologyDownload.planning", "fetch_all", "function"),
    ExportSpec("fetch_one", "DocsToKG.OntologyDownload.planning", "fetch_one", "function"),
    ExportSpec(
        "run_validators", "DocsToKG.OntologyDownload.validation", "run_validators", "function"
    ),
    ExportSpec(
        "validate_manifest_dict",
        "DocsToKG.OntologyDownload.planning",
        "validate_manifest_dict",
        "function",
    ),
    ExportSpec("__version__", "DocsToKG.OntologyDownload.api", "__version__", "const"),
    ExportSpec(
        "cli_main",
        "DocsToKG.OntologyDownload.api",
        "cli_main",
        "function",
        include_in_manifest=False,
    ),
    ExportSpec(
        "validator_worker_main",
        "DocsToKG.OntologyDownload.api",
        "validator_worker_main",
        "function",
        include_in_manifest=False,
    ),
    ExportSpec(
        "PUBLIC_API_MANIFEST",
        "DocsToKG.OntologyDownload.api",
        "PUBLIC_API_MANIFEST",
        "const",
        include_in_manifest=False,
    ),
)

EXPORT_MAP: Dict[str, ExportSpec] = {spec.name: spec for spec in EXPORTS}

PUBLIC_API_MANIFEST: Mapping[str, str] = {
    name: spec.kind for name, spec in EXPORT_MAP.items() if spec.include_in_manifest
}

PUBLIC_EXPORT_NAMES: Tuple[str, ...] = tuple(EXPORT_MAP)


def iter_manifest_exports() -> Iterable[ExportSpec]:
    """Yield export specifications included in the public manifest."""

    for spec in EXPORTS:
        if spec.include_in_manifest:
            yield spec
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.exports",
#   "purpose": "Catalogue and expose the public API surface for the ontology downloader",
#   "sections": [
#     {"id": "spec", "name": "ExportSpec dataclass", "anchor": "SPEC", "kind": "api"},
#     {"id": "manifest", "name": "Export Manifest", "anchor": "MAN", "kind": "constants"}
#   ]
# }
# === /NAVMAP ===
