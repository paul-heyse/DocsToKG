# === NAVMAP v1 ===
# {
#   "module": "openspec.changes.archive.2025_10_15_refine_docparsing_implementation",
#   "purpose": "Archived OpenSpec helper module openspec.changes.archive.2025_10_15_refine_docparsing_implementation",
#   "sections": []
# }
# === /NAVMAP ===

"""
DocsToKG DocParsing Refinement Documentation Package.

1. Purpose
   The ``refine-docparsing-implementation`` initiative documents production
   hardening work for the DocsToKG DocParsing stack while the underlying code
   evolves. Keeping this narrative in module docstrings ensures the automated
   documentation pipeline exposes the latest OpenSpec intent alongside the
   pending implementation.

2. Scope
   The annotations summarise the production-readiness requirements captured in
   the change proposal so that reviewers, automation agents, and developers
   understand how DocsToKG ingestion behaviour must evolve during the rollout.

3. References
   * Specification deltas: ``openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md``
   * Design discussion: ``openspec/changes/refine-docparsing-implementation/design.md``
   * Implementation playbooks: ``openspec/changes/refine-docparsing-implementation/implementation-patterns.md``
   * Execution roadmap: ``openspec/changes/refine-docparsing-implementation/tasks.md``
"""

from .documentation import RequirementDoc, load_requirement_docs
# --- Globals ---

__all__ = ["RequirementDoc", "load_requirement_docs"]
