# 1. Module: opensearch_simulator

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.devtools.opensearch_simulator``.

## 1. Overview

In-memory OpenSearch simulator and schema helpers for development.

Args:
    None

Returns:
    None

Raises:
    None

## 2. Functions

### `matches_filters(chunk, filters)`

Return ``True`` when ``chunk`` satisfies the provided ``filters``.

Args:
chunk: Chunk payload whose metadata and namespace should be evaluated.
filters: Mapping of filter keys to expected values mirroring the production
API contract. Values may be scalars or lists of acceptable values.

Returns:
``True`` if ``chunk`` matches all supplied filters, otherwise ``False``.

### `asdict(self)`

Return a dictionary representation of the template.

Args:
None

Returns:
Dictionary containing serializable template fields.

### `bootstrap_template(self, namespace, chunking)`

Create and cache a template for ``namespace``.

Args:
namespace: Namespace identifier that will own the template.
chunking: Optional chunking configuration. Defaults to ``ChunkingConfig()``.

Returns:
Registered template instance for ``namespace``.

### `get_template(self, namespace)`

Return the template registered for ``namespace`` if it exists.

Args:
namespace: Namespace identifier used during registration.

Returns:
Template for ``namespace`` or ``None`` when the namespace is unknown.

### `list_templates(self)`

Return a shallow copy of the namespace → template mapping.

Args:
None

Returns:
Mapping of namespace names to template definitions.

### `bulk_upsert(self, chunks)`

Insert or update ``chunks`` within the simulator.

Args:
chunks: Iterable of chunk payloads to store or replace.

Returns:
None

Raises:
None

### `bulk_delete(self, vector_ids)`

Remove chunk payloads whose vector identifiers are present in ``vector_ids``.

Args:
vector_ids: Vector identifiers associated with chunks to remove.

Returns:
None

Raises:
None

### `_update_df_on_add(self, terms)`

*No documentation available.*

### `_update_df_on_remove(self, terms)`

*No documentation available.*

### `fetch(self, vector_ids)`

Return the stored payloads matching ``vector_ids``.

Args:
vector_ids: Identifiers of the desired chunks.

Returns:
List of chunk payloads that were previously stored.

### `vector_ids(self)`

Return all vector identifiers currently stored.

Args:
None

Returns:
Vector identifiers known to the simulator.

### `register_template(self, template)`

Associate ``template`` with its namespace for later lookups.

Args:
template: Template definition to register.

Returns:
None

### `template_for(self, namespace)`

Return the registered template for ``namespace`` if present.

Args:
namespace: Namespace identifier to look up.

Returns:
Registered template or ``None`` when no template exists.

### `search_bm25(self, query_weights, filters, top_k, cursor)`

Execute a BM25-like search using ``query_weights``.

Args:
query_weights: Sparse query representation with token weights.
filters: Metadata filters applied to stored chunks.
top_k: Maximum number of results to return in the current page.
cursor: Optional cursor returned from a previous call.

Returns:
Tuple containing hits and the next pagination cursor (or ``None``).

### `search_splade(self, query_weights, filters, top_k, cursor)`

Execute a SPLADE-style sparse search using ``query_weights``.

Args:
query_weights: Sparse query with SPLADE activations.
filters: Metadata filters applied to stored chunks.
top_k: Maximum number of results to return in the current page.
cursor: Optional cursor returned from a previous call.

Returns:
Tuple containing hits and the next pagination cursor (or ``None``).

### `highlight(self, chunk, query_tokens)`

Return naive highlight tokens that appear in ``chunk``.

Args:
chunk: Chunk text that should be scanned for highlights.
query_tokens: Tokens extracted from the query phrase.

Returns:
List of tokens present in both ``chunk`` and ``query_tokens``.

### `stats(self)`

Return summary statistics about the indexed corpus.

Args:
None

Returns:
Mapping containing document counts and average chunk length.

### `_filtered_chunks(self, filters)`

Return stored chunks that satisfy ``filters``.

Args:
filters: Metadata filters applied to stored chunks.

Returns:
Stored chunks matching the provided filters.

### `_bm25_score(self, stored, query_weights)`

Compute a BM25-inspired score for ``stored`` given ``query_weights``.

Args:
stored: Stored chunk candidate being scored.
query_weights: Sparse query representation with token weights.

Returns:
BM25-inspired similarity score for the stored chunk.

### `_paginate(self, results, top_k, cursor)`

Paginate ``results`` according to ``top_k`` and ``cursor``.

Args:
results: Ranked search hits produced by a sparse search.
top_k: Maximum number of results to include in the response.
cursor: Optional offset cursor used for pagination.

Returns:
Tuple containing the sliced page of results and the next cursor.

### `_search_sparse(self, scoring_fn, filters, top_k, cursor)`

Shared sparse search implementation used by BM25 and SPLADE search.

Args:
scoring_fn: Callable that scores stored chunks.
filters: Metadata filters applied to the search corpus.
top_k: Maximum number of results requested by the caller.
cursor: Optional pagination cursor.

Returns:
Tuple containing search hits and a possible pagination cursor.

### `search_bm25_true(self, query_weights, filters, top_k, cursor)`

Execute Okapi BM25 using stored DF statistics.

### `_recompute_avg_length(self)`

Update the cached average chunk length metric.

Args:
None

Returns:
None

### `bm25_score(stored)`

*No documentation available.*

## 3. Classes

### `OpenSearchIndexTemplate`

Representation of a namespace-specific OpenSearch template.

Attributes:
name: Human-readable name of the template (e.g. ``"hybrid-chunks-demo"``).
namespace: DocsToKG namespace that owns the template.
body: Raw OpenSearch template payload including settings and mappings.
chunking: Chunking configuration applied to documents within the namespace.

Examples:
>>> template = OpenSearchIndexTemplate(
...     name="hybrid-chunks-demo",
...     namespace="demo",
...     body={"settings": {}, "mappings": {}},
...     chunking=ChunkingConfig(),
... )
>>> template.asdict()["namespace"]
'demo'

### `OpenSearchSchemaManager`

Manage simulated OpenSearch index templates for tests.

The manager mirrors the minimal subset of the OpenSearch API required by the
validation harness. It stores templates keyed by namespace and exposes helper
methods to create or retrieve definitions.

Attributes:
_templates: Internal mapping of namespace names to template instances.

Examples:
>>> manager = OpenSearchSchemaManager()
>>> template = manager.bootstrap_template("demo")
>>> manager.get_template("demo") is template
True

### `_StoredChunk`

Wrapper used to keep chunk payloads inside the simulator.

Attributes:
payload: Chunk payload stored in memory.

Examples:
>>> from DocsToKG.HybridSearch.types import ChunkFeatures, ChunkPayload
>>> features = ChunkFeatures(bm25_terms={}, splade_weights={}, embedding=[])  # doctest: +SKIP
>>> chunk = ChunkPayload(  # doctest: +SKIP
...     doc_id="d1",
...     chunk_id="c1",
...     vector_id="v1",
...     namespace="demo",
...     text="example",
...     token_count=1,
...     metadata={},
...     features=features,
... )
>>> _StoredChunk(chunk).payload is chunk  # doctest: +SKIP
True

### `OpenSearchSimulator`

Simplified OpenSearch-like index used for development and tests.

The simulator implements the :class:`~DocsToKG.HybridSearch.interfaces.LexicalIndex`
protocol and mimics the behaviour of the production OpenSearch integration.

Attributes:
_chunks: Mapping of vector identifiers to stored chunk payloads.
_avg_length: Average token length across indexed chunks (used for BM25).
_templates: Registry of namespace templates registered via ``register_template``.

Examples:
>>> simulator = OpenSearchSimulator()
>>> simulator.bulk_upsert([])
>>> simulator.stats()["document_count"]
0.0
