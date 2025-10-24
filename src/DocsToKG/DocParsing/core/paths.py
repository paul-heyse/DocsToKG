# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.paths",
#   "purpose": "Filesystem helpers for DocParsing core utilities.",
#   "sections": [
#     {
#       "id": "normalize-output-path",
#       "name": "normalize_output_path",
#       "anchor": "function-normalize-output-path",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Filesystem helpers shared across DocParsing core modules.

The functions in this module offer small, well-scoped helpers that avoid
re-implementing common filesystem behaviours across the DocParsing core
package. Keeping them here allows orchestration code to share a consistent
interpretation of user-provided paths and manifest metadata.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["normalize_output_path"]


def normalize_output_path(path: Path | str) -> Path:
    """Return a resolved ``Path`` for manifest and resume checks.

    The helper expands user home markers and resolves symlinks without
    requiring the underlying artifact to exist. This ensures that manifest
    resume logic compares and probes consistent filesystem targets even when
    stages are configured to emit into symlinked directories.
    """

    resolved = Path(path).expanduser().resolve()
    return resolved

