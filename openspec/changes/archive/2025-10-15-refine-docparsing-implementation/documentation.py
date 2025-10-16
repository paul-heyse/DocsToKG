# === NAVMAP v1 ===
# {
#   "module": "openspec.changes.archive.2025-10-15-refine-docparsing-implementation.documentation",
#   "purpose": "Archived OpenSpec helper module openspec.changes.archive.2025_10_15_refine_docparsing_implementation.documentation",
#   "sections": [
#     {
#       "id": "requirement_doc",
#       "name": "RequirementDoc",
#       "anchor": "REQU",
#       "kind": "class"
#     },
#     {
#       "id": "load_requirement_docs",
#       "name": "load_requirement_docs",
#       "anchor": "LRD",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
DocsToKG DocParsing Refinement Requirement Registry.

1. Context
   The ``refine-docparsing-implementation`` change documents staged hardening
   work for DocsToKG DocParsing, aligning specification deltas and design notes
   while code evolves in ``src/DocsToKG``. Capturing this context in code keeps
   automated documentation exports authoritative during the rollout.

2. Source Material
   Requirement details are drawn from
   ``openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md``,
   with supporting rationale in ``design.md`` and execution planning in
   ``tasks.md``. Summaries stay synchronized with these files so reviews track
   the latest intent.

3. Automation Role
   Tooling such as ``docs/scripts/generate_api_docs.py`` and Sphinx rely on
   these docstrings to surface behaviour changes in DocsToKG documentation even
   before implementation patches merge. Maintaining the registry here prevents
   drift between specification and observable documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class RequirementDoc:
    """Human-readable description of a DocParsing refinement requirement.

    The dataclass captures the core documentation elements required by the
    OpenSpec change proposal so that they can be surfaced programmatically.  It
    mirrors the information architecture defined in ``tasks.md`` and the
    capability deltas under ``specs/doc-parsing/spec.md``.

    Attributes:
        name: Title of the requirement as recorded in the specification.
        summary: Short narrative describing why the refinement exists.
        spec_reference: Relative path to the authoritative specification text.
        scenarios: Concrete user or system scenarios that illustrate the
            acceptance criteria.  The entries should be aligned with the
            ``#### Scenario`` blocks in the spec delta so that reviewers can
            trace requirements directly to validation steps.
    """

    name: str
    summary: str
    spec_reference: str
    scenarios: List[str]


def load_requirement_docs() -> Iterable[RequirementDoc]:
    """Yield structured documentation for each DocParsing refinement.

    Returns:
        An iterable of :class:`RequirementDoc` entries describing the
        production-readiness improvements enumerated in the OpenSpec change. The
        data is curated manually to remain faithful to
        ``specs/doc-parsing/spec.md`` while providing enough context for
        docstring-driven documentation pipelines.
    """

    return (
        RequirementDoc(
            name="Atomic File Writes",
            summary=(
                "Chunking and embedding stages must write JSONL outputs using an "
                "atomic temporary-file-and-rename pattern to avoid partial files "
                "when processes crash."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Process interruption during chunk write leaves either a complete"
                " file or no file so resume logic can safely retry.",
                "Pipeline resume after crash never encounters truncated JSONL rows"
                " because incomplete outputs are removed atomically.",
            ],
        ),
        RequirementDoc(
            name="UTC Timestamp Correctness",
            summary=(
                "Structured logging must emit genuine UTC timestamps by setting "
                "the formatter's converter to time.gmtime so that monitoring "
                "systems read accurate ISO 8601 values."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Cross-timezone operators can correlate pipeline logs because"
                " timestamps match datetime.utcnow values.",
                "Logged timestamps preserve the expected 'Z' suffix and include"
                " microseconds for precise tracing.",
            ],
        ),
        RequirementDoc(
            name="Content Hash Algorithm Tagging",
            summary=(
                "Manifests must store the hashing algorithm alongside each"
                " content hash so that SHA-1 and SHA-256 runs can coexist without"
                " breaking resume logic."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Operators switching DOCSTOKG_HASH_ALG to sha256 see matching"
                " manifest entries that record the new algorithm.",
                "Mixed-algorithm corpora avoid false resume matches because"
                " algorithms are compared before hashes are evaluated.",
            ],
        ),
        RequirementDoc(
            name="CLI Argument Parsing Simplicity",
            summary=(
                "Command entry points should accept an optional argparse.Namespace"
                " for programmatic invocation while defaulting to parser.parse_args"
                " for CLI use without bespoke merging logic."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Tests can inject a Namespace instance directly without triggering"
                " redundant parsing or mutation steps.",
                "Command-line execution respects user-provided arguments exactly"
                " as typed in the shell.",
            ],
        ),
        RequirementDoc(
            name="Embedding Memory Efficiency",
            summary=(
                "The embeddings pipeline should avoid retaining full corpus text"
                " in memory by streaming chunk contents during pass B and"
                " discarding text once BM25 statistics are computed in pass A."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Processing a 50K document corpus stays under a 16GB peak"
                " memory envelope because text is streamed from disk.",
                "Resume operations recompute BM25 statistics without holding"
                " processed chunk text, enabling efficient restarts.",
            ],
        ),
        RequirementDoc(
            name="Portable Model Paths",
            summary=(
                "Model directories must be configurable via environment variables"
                " and CLI flags so deployments can run in CI, local, or"
                " air-gapped environments without editing code."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Setting HF_HOME ensures downloads flow to the configured cache"
                " instead of hardcoded user directories.",
                "Operators override Qwen and SPLADE model directories per run"
                " to point at custom storage locations.",
            ],
        ),
        RequirementDoc(
            name="Manifest Scalability via Sharding",
            summary=(
                "Resume logic should read stage-specific manifest shards so that"
                " large corpora avoid scanning unrelated processing history."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Chunking resumes by loading only docparse.chunks.manifest.jsonl,"
                " keeping startup latency under five seconds for 100K documents.",
                "Legacy monolithic manifests remain compatible by falling back"
                " when stage shards are missing.",
            ],
        ),
        RequirementDoc(
            name="Legacy Script Quarantine",
            summary=(
                "Deprecated direct-invocation scripts should live under a legacy"
                " namespace with deprecation warnings and shims that forward to"
                " the unified CLI."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Running historical scripts displays the replacement CLI command"
                " and still completes successfully via a shim.",
                "Documentation links in the warning message provide migration"
                " guidance for operators.",
            ],
        ),
        RequirementDoc(
            name="vLLM Service Preflight Telemetry",
            summary=(
                "PDF conversion should emit a service health manifest entry before"
                " processing documents so operators can attribute failures to"
                " infrastructure issues."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Successful runs record served models, version, port, and"
                " readiness metrics using doc_id='__service__'.",
                "Failures reference preflight diagnostics so responders prioritise"
                " infrastructure troubleshooting.",
            ],
        ),
        RequirementDoc(
            name="Offline Operation Support",
            summary=(
                "Embeddings and conversion pipelines must honour an --offline flag"
                " and rely solely on pre-downloaded models when network access is"
                " unavailable."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Air-gapped deployments succeed when required models are cached"
                " locally with no outbound network calls.",
                "Missing model directories surface actionable errors that include"
                " the expected filesystem path before GPU allocation begins.",
            ],
        ),
        RequirementDoc(
            name="Image Metadata Promotion",
            summary=(
                "Chunk outputs should expose high-level image metadata fields to"
                " simplify downstream filtering without parsing nested provenance"
                " structures."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Consumers quickly filter chunks with image captions or"
                " classifications using top-level boolean flags.",
                "Detailed provenance remains available for analytics without"
                " affecting the streamlined metadata fields.",
            ],
        ),
        RequirementDoc(
            name="SPLADE Sparsity Threshold Documentation",
            summary=(
                "Corpus summary manifests must record the SPLADE sparsity warning"
                " threshold so CI alerts remain unambiguous."
            ),
            spec_reference="openspec/changes/refine-docparsing-implementation/specs/doc-parsing/spec.md",
            scenarios=[
                "Automated quality gates inspect sparsity_warn_threshold_pct to"
                " understand alert semantics.",
                "Operators review manifest metadata to confirm the actual"
                " attention backend selected during runs.",
            ],
        ),
    )
