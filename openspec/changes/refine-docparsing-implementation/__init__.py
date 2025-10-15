"""Refine DocParsing implementation change package documentation.

This package contains documentation scaffolding that captures the intent of
`refine-docparsing-implementation` while the actual code changes are being
implemented elsewhere in the repository.  The docstrings summarise the
production-readiness requirements that are documented in the OpenSpec change
proposal so that automated documentation generators surface them alongside the
forthcoming implementation work.

References:
    * Specification deltas: ``openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md``
    * Design discussion: ``openspec/changes/refine-docparsing-implementation/design.md``
    * Implementation playbooks: ``openspec/changes/refine-docparsing-implementation/implementation-patterns.md``
"""

from .documentation import RequirementDoc, load_requirement_docs

__all__ = ["RequirementDoc", "load_requirement_docs"]
