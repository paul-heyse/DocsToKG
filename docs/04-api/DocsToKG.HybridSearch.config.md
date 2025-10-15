# Module: config

Configuration models and manager for hybrid search.

This module provides comprehensive configuration management for DocsToKG's
hybrid search capabilities, including chunking, indexing, fusion, and
retrieval parameters with thread-safe configuration management.

## Functions

### `from_dict(payload)`

*No documentation available.*

### `get(self)`

*No documentation available.*

### `reload(self)`

*No documentation available.*

### `_load(self)`

*No documentation available.*

### `_load_yaml(self, raw)`

*No documentation available.*

## Classes

### `ChunkingConfig`

Configuration for document chunking operations.

Defines how documents are split into smaller chunks for processing
and indexing, balancing between context preservation and search
granularity.

Attributes:
max_tokens: Maximum number of tokens per chunk (800 default)
overlap: Number of tokens to overlap between chunks (150 default)

Examples:
>>> config = ChunkingConfig(max_tokens=1000, overlap=200)
>>> # Creates chunks with 1000 tokens max, 200 token overlap

### `DenseIndexConfig`

Configuration for FAISS dense vector indexing.

Controls the behavior of vector similarity search using FAISS,
including index type selection and performance parameters.

Attributes:
index_type: Type of FAISS index ("flat", "ivf_flat", "ivf_pq")
nlist: Number of Voronoi cells for IVF indexes (1024 default)
nprobe: Number of cells to search for IVF indexes (8 default)
pq_m: Number of sub-quantizers for PQ indexes (16 default)
pq_bits: Bits per sub-quantizer for PQ indexes (8 default)
oversample: Oversampling factor for PQ search (2 default)

Examples:
>>> config = DenseIndexConfig(
...     index_type="ivf_pq",
...     nlist=4096,
...     nprobe=32,
...     pq_m=32,
...     pq_bits=8
... )

### `FusionConfig`

Configuration for result fusion and ranking.

Controls how results from different retrieval methods (BM25, SPLADE,
dense vectors) are combined and ranked for optimal relevance.

Attributes:
k0: RRF (Reciprocal Rank Fusion) parameter (60.0 default)
mmr_lambda: MMR diversification parameter (0.6 default)
enable_mmr: Whether to apply MMR diversification (True default)
cosine_dedupe_threshold: Cosine similarity threshold for deduplication (0.98 default)
max_chunks_per_doc: Maximum chunks to return per document (3 default)

Examples:
>>> config = FusionConfig(
...     k0=50.0,
...     mmr_lambda=0.7,
...     enable_mmr=True,
...     max_chunks_per_doc=5
... )

### `RetrievalConfig`

Configuration for individual retrieval methods.

Controls the behavior of each retrieval method (BM25, SPLADE, dense)
including how many candidates each method should return.

Attributes:
bm25_top_k: Number of BM25 candidates to retrieve (50 default)
splade_top_k: Number of SPLADE candidates to retrieve (50 default)
dense_top_k: Number of dense vector candidates to retrieve (50 default)

Examples:
>>> config = RetrievalConfig(
...     bm25_top_k=100,
...     splade_top_k=75,
...     dense_top_k=25
... )

### `HybridSearchConfig`

Complete configuration for hybrid search operations.

This class aggregates all configuration for hybrid search functionality,
providing a single source of truth for all search-related parameters.

Attributes:
chunking: Document chunking configuration
dense: Dense vector indexing configuration
fusion: Result fusion and ranking configuration
retrieval: Individual retrieval method configuration

Examples:
>>> config = HybridSearchConfig(
...     chunking=ChunkingConfig(max_tokens=1000),
...     dense=DenseIndexConfig(index_type="ivf_pq"),
...     fusion=FusionConfig(enable_mmr=True),
...     retrieval=RetrievalConfig(bm25_top_k=100)
... )

### `HybridSearchConfigManager`

File-backed configuration manager with reload support.
