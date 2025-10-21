"""Export manifest and public API surface.

This module defines the public API and export configuration for OntologyDownload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ExportSpec",
    "EXPORT_MAP",
    "EXPORTS",
    "PUBLIC_API_MANIFEST",
]


@dataclass(frozen=True)
class ExportSpec:
    """Specification for an exported symbol."""

    name: str
    """Name of the symbol."""

    module: str
    """Module where the symbol is defined."""

    include_in_manifest: bool = True
    """Whether to include this symbol in the public API manifest."""

    doc: str = ""
    """Short documentation string."""


# Core API specs
_CORE_SPECS = [
    ExportSpec("plan_all", "planning", doc="Plan a list of FetchSpecs"),
    ExportSpec("fetch_all", "planning", doc="Fetch and validate ontologies"),
    ExportSpec("Manifest", "planning", doc="Manifest schema"),
    ExportSpec("Resolver", "resolvers", doc="Base resolver interface"),
    ExportSpec("cli_main", "cli", doc="Main CLI entry point"),
]

# Export list
EXPORTS: list[ExportSpec] = _CORE_SPECS

# Export map: module name -> exported symbols
EXPORT_MAP: dict[str, list[str]] = {
    spec.module: [spec.name for spec in EXPORTS if spec.module == spec.module]
    for spec in EXPORTS
}

# Public API manifest
PUBLIC_API_MANIFEST: dict[str, Any] = {
    "version": "1.0.0",
    "modules": list(EXPORT_MAP.keys()),
    "symbols": [spec.name for spec in EXPORTS if spec.include_in_manifest],
}

# Public export names (for api.py)
PUBLIC_EXPORT_NAMES: list[str] = [spec.name for spec in EXPORTS if spec.include_in_manifest]
