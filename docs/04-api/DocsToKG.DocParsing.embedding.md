# 1. Module: embedding

This reference documents the DocsToKG module ``DocsToKG.DocParsing.embedding``.

## 1. Overview

Hybrid Embedding Pipeline

Generates BM25, SPLADE, and Qwen embeddings for DocsToKG chunk files while
maintaining manifest entries, UUID hygiene, and data quality metrics. The
pipeline runs in two passes: the first ensures chunk UUID integrity and builds
BM25 corpus statistics; the second executes SPLADE and Qwen models to emit
vector JSONL artefacts ready for downstream search.

Key Features:
- Auto-detect DocsToKG data directories and manage resume/force semantics
- Stream SPLADE sparse encoding and Qwen dense embeddings from local caches
- Validate vector schemas, norms, and dimensions before writing outputs
- Record manifest metadata for observability and auditing
- Explain SPLADE attention backend fallbacks (auto→FlashAttention2→SDPA→eager)

Usage:
    python -m DocsToKG.DocParsing.embedding --resume

Dependencies:
- sentence_transformers (optional): Provides SPLADE sparse encoders.
- vllm (optional): Hosts the Qwen embedding model with pooling support.
- tqdm: Surface user-friendly progress bars across pipeline phases.

## 2. Functions

### `_shutdown_llm_instance(llm)`

Best-effort shutdown for a cached Qwen LLM instance.

### `close_all_qwen()`

Release all cached Qwen LLM instances.

### `_qwen_cache_key(cfg)`

Return cache key tuple for Qwen LLM instances.

### `_resolve_qwen_dir(model_root)`

Resolve Qwen model directory with ``DOCSTOKG_QWEN_DIR`` override.

Args:
model_root: Base directory housing DocsToKG models.

Returns:
Absolute path to the Qwen embedding model directory.

### `_resolve_splade_dir(model_root)`

Resolve SPLADE model directory with ``DOCSTOKG_SPLADE_DIR`` override.

Args:
model_root: Base directory housing DocsToKG models.

Returns:
Absolute path to the SPLADE model directory.

### `_expand_optional(path)`

Expand optional :class:`Path` values to absolutes when provided.

Args:
path: Optional path reference supplied by the caller.

Returns:
``None`` when ``path`` is ``None``; otherwise the expanded absolute path.

### `_resolve_cli_path(value, default)`

Resolve a CLI-provided path, falling back to ``default`` when omitted.

Args:
value: Optional user-supplied path.
default: Fallback path used when ``value`` is absent.

Returns:
Absolute path derived from ``value`` or ``default``.

### `_ensure_splade_dependencies()`

Backward-compatible shim that delegates to core.ensure_splade_dependencies.

### `_ensure_qwen_dependencies()`

Backward-compatible shim that delegates to core.ensure_qwen_dependencies.

### `ensure_uuid(rows)`

Validate or assign deterministic chunk UUIDs based on content offsets.

Args:
rows: Chunk dictionaries to normalise. Each row may include ``start_offset``
(preferred); when absent the legacy UUID derivation is applied.

Returns:
``True`` when at least one UUID was added or corrected; otherwise ``False``.

### `_legacy_chunk_uuid(doc_id, source_chunk_idxs, text_value)`

Derive the historical UUID used before deterministic start offsets.

### `ensure_chunk_schema(rows, source)`

Assert that chunk rows declare a compatible schema version.

Args:
rows: Iterable of chunk dictionaries to validate.
source: Path to the originating chunk file, used for error context.

Returns:
None

Raises:
ValueError: Propagated when an incompatible schema version is detected.

### `tokens(text)`

Tokenize normalized text for sparse retrieval features.

Args:
text: Input string to tokenize.

Returns:
Lowercased alphanumeric tokens extracted from the text.

### `print_bm25_summary(stats)`

Print corpus-level BM25 statistics.

Args:
stats: Computed BM25 statistics to log.

Returns:
None: Writes structured logs only.

### `bm25_vector(text, stats, k1, b)`

Generate BM25 term weights for a chunk of text.

Args:
text: Chunk text to convert into a sparse representation.
stats: Precomputed BM25 statistics for the corpus.
k1: Term frequency saturation parameter.
b: Length normalization parameter.

Returns:
Tuple of `(terms, weights)` describing the sparse vector.

### `splade_encode(cfg, texts, batch_size)`

Encode text with SPLADE to obtain sparse lexical vectors.

Args:
cfg: SPLADE configuration describing device, batch size, and cache.
texts: Batch of input strings to encode.
batch_size: Optional override for the encoding batch size.

Returns:
Tuple of token lists and weight lists aligned per input text.

### `_detect_splade_backend(encoder, requested)`

Best-effort detection of the attention backend used by SPLADE.

### `_get_splade_encoder(cfg)`

Retrieve (or create) a cached SPLADE encoder instance.

Args:
cfg: SPLADE configuration describing model location and runtime options.

Returns:
Cached :class:`SparseEncoder` ready for SPLADE inference.

Raises:
ValueError: If the encoder cannot be initialised with the supplied configuration.
ImportError: If required SPLADE dependencies are unavailable.

### `_get_splade_backend_used(cfg)`

Return the backend string recorded for a given SPLADE configuration.

### `_qwen_embed_direct(cfg, texts, batch_size)`

Produce dense embeddings using a local Qwen3 model served by vLLM.

Args:
cfg: Configuration describing model path, dtype, and batching.
texts: Batch of documents to embed.
batch_size: Optional override for inference batch size.

Returns:
List of embedding vectors, one per input text.

### `qwen_embed(cfg, texts, batch_size)`

Public wrapper around the direct Qwen embedding implementation.

### `process_pass_a(files, logger)`

Assign UUIDs and build BM25 statistics (streaming + atomic rewrite).

This implementation streams each JSONL row and writes a temporary file with
normalised schema/UUIDs. The original file is atomically replaced **only**
when changes are detected. This bounds memory on huge shards and prevents
partial writes.

Args:
files: Sequence of chunk file paths to process.
logger: Logger used for structured progress output.

Returns:
Aggregated BM25 statistics for the supplied chunk corpus.

Raises:
OSError: If chunk files cannot be read or written.
json.JSONDecodeError: If a chunk row contains invalid JSON.

### `iter_rows_in_batches(path, batch_size)`

Iterate over JSONL rows in batches to reduce memory usage.

Args:
path: Path to JSONL file to read.
batch_size: Number of rows to yield per batch.

Returns:
Iterator[List[dict]]: Lazy iterator producing batched chunk rows.

Yields:
Lists of row dictionaries, each containing up to batch_size items.

### `iter_chunk_files(directory)`

Deprecated shim that forwards to :func:`iter_chunks`.

Args:
directory: Directory to scan for chunk artifacts.

Returns:
Iterator over chunk files.

### `create_vector_writer(path, fmt)`

Factory returning the appropriate vector writer for ``fmt``.

### `process_chunk_file_vectors(chunk_file, out_path, stats, args, validator, logger)`

Generate vectors for a single chunk file and persist them to disk.

Args:
chunk_file: Chunk JSONL file to process.
out_path: Destination path for vectors.
stats: Precomputed BM25 statistics.
args: Parsed CLI arguments with runtime configuration.
validator: SPLADE validator for sparsity metrics.
logger: Logger for structured output.

Returns:
Tuple of ``(vector_count, splade_nnz_list, qwen_norms)``.

Raises:
ValueError: Propagated if vector dimensions or norms fail validation.

### `write_vectors(writer, uuids, texts, splade_results, qwen_results, stats, args)`

Write validated vector rows to disk with schema enforcement.

Args:
writer: Vector writer responsible for persisting rows.
uuids: Sequence of chunk UUIDs aligned with the other inputs.
texts: Chunk text bodies.
splade_results: SPLADE token and weight pairs per chunk.
qwen_results: Dense embedding vectors per chunk.
stats: BM25 statistics used to generate sparse vectors.
args: Parsed CLI arguments for runtime configuration.
rows: Original chunk row dictionaries.
validator: SPLADE validator capturing sparsity data.
logger: Logger used to emit structured diagnostics.

Returns:
Tuple containing the number of vectors written, SPLADE nnz counts,
and Qwen vector norms.

Raises:
ValueError: If vector lengths are inconsistent or fail validation.

### `_validate_vectors_for_chunks(chunks_dir, vectors_dir, logger)`

Validate vectors associated with chunk files without recomputing models.

Returns:
(files_checked, rows_validated)

### `build_parser()`

Construct the CLI parser for the embedding pipeline.

Args:
None

Returns:
argparse.ArgumentParser: Parser configured for embedding options.

Raises:
None

### `parse_args(argv)`

Parse CLI arguments for standalone embedding execution.

Args:
argv (list[str] | None): Optional CLI argument vector. When ``None`` the
current process arguments are used.

Returns:
argparse.Namespace: Parsed embedding configuration.

Raises:
SystemExit: Propagated if ``argparse`` reports invalid options.

### `main(args)`

CLI entrypoint for chunk UUID cleanup and embedding generation.

Args:
args (argparse.Namespace | None): Optional parsed arguments, primarily
for testing or orchestration.

Returns:
int: Exit code where ``0`` indicates success.

Raises:
ValueError: If invalid runtime parameters (such as batch sizes) are supplied.

### `from_env(cls, defaults)`

Construct configuration from environment variables.

### `from_args(cls, args, defaults)`

Merge CLI arguments, configuration files, and environment variables.

### `finalize(self)`

Normalise paths and casing after all sources have been applied.

### `add_document(self, text)`

Add document to statistics without retaining text.

Args:
text: Document contents used to update running statistics.

Returns:
None

### `finalize(self)`

Compute final statistics.

Args:
None: The accumulator finalises its internal counters without parameters.

Returns:
:class:`BM25Stats` summarising the accumulated corpus.

### `validate(self, uuid, tokens, weights)`

Record sparsity information for a single chunk.

Args:
uuid: Chunk identifier associated with the SPLADE vector.
tokens: Token list produced by the SPLADE encoder.
weights: Weight list aligned with ``tokens``.

Returns:
None

Raises:
None

### `report(self, logger)`

Emit warnings if sparsity metrics exceed thresholds.

Args:
logger: Logger used to emit warnings and metrics.

Returns:
None

### `_worker(self)`

Consume enqueued embedding requests until shutdown is signalled.

### `embed(self, texts, batch_size)`

Queue an embedding request and block until the result is ready.

### `shutdown(self, wait)`

Flush pending requests and terminate the worker thread.

### `write_rows(self, rows)`

Persist a batch of vector rows to the underlying storage medium.

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `write_rows(self, rows)`

Append ``rows`` to the active JSONL artifact created by ``__enter__``.

### `_process_entry(entry)`

Encode vectors for a chunk file and report per-file metrics.

## 3. Classes

### `EmbedCfg`

Stage configuration container for the embedding pipeline.

### `BM25StatsAccumulator`

Streaming accumulator for BM25 corpus statistics.

Attributes:
N: Number of documents processed so far.
total_tokens: Total token count across processed documents.
df: Document frequency map collected to date.

Examples:
>>> acc = BM25StatsAccumulator()
>>> acc.add_document("hybrid search")
>>> acc.N
1

### `SPLADEValidator`

Track SPLADE sparsity metrics across the corpus.

Attributes:
total_chunks: Total number of chunks inspected.
zero_nnz_chunks: UUIDs whose SPLADE vector has zero active terms.
nnz_counts: Non-zero counts per processed chunk.

Examples:
>>> validator = SPLADEValidator()
>>> validator.total_chunks
0

### `QwenEmbeddingQueue`

Serialize Qwen embedding requests across worker threads.

### `EmbeddingProcessingError`

Wrap exceptions raised during per-file embedding with timing metadata.

### `VectorWriter`

Abstract base class for vector writers.

### `JsonlVectorWriter`

Context manager that writes vector rows to JSONL atomically.
