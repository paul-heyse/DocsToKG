# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.concurrency.__init__",
#   "purpose": "Concurrency helpers shared across DocsToKG components.",
#   "sections": []
# }
# === /NAVMAP ===

"""
Concurrency helpers shared across DocsToKG components.

Currently exposes :func:`create_executor` which mirrors the legacy stage
runner semantics (IO → threads, CPU → processes) while keeping executor
implementations out of module trees that forbid local pools.
"""

from .executors import create_executor

__all__ = ["create_executor"]

