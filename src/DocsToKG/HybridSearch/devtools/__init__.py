# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.devtools.__init__",
#   "purpose": "Developer tooling helpers for DocsToKG hybrid search.",
#   "sections": []
# }
# === /NAVMAP ===

"""Developer tooling helpers for DocsToKG hybrid search.

The ``devtools`` package aggregates utilities that make it easy to spin up a
fully functional hybrid-search environment without external services. It
re-exports the deterministic feature generators plus the in-memory OpenSearch
simulator so tests and notebooks can import from a single namespace while still
conforming to the production interfaces.
"""

from .features import (
    FeatureGenerator,
    sliding_window,
    tokenize,
    tokenize_with_spans,
)
from .opensearch_simulator import (
    OpenSearchIndexTemplate,
    OpenSearchSchemaManager,
    OpenSearchSimulator,
    matches_filters,
)

__all__ = (
    "FeatureGenerator",
    "OpenSearchIndexTemplate",
    "OpenSearchSchemaManager",
    "OpenSearchSimulator",
    "matches_filters",
    "sliding_window",
    "tokenize",
    "tokenize_with_spans",
)
