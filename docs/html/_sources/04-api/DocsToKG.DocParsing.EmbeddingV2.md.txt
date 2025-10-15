# 1. Module: EmbeddingV2

This reference documents the DocsToKG module ``DocsToKG.DocParsing.EmbeddingV2``.

EmbedVectors.py

- Reads chunked JSONL from /home/paul/DocsToKG/Data/ChunkedDocTagFiles
- Ensures each chunk has a UUID (writes back to the same files)
- Emits vectors JSONL to /home/paul/DocsToKG/Data/Vectors

Uses local HF cache at /home/paul/hf-cache/:
  - Qwen3-Embedding-4B at   /home/paul/hf-cache/models/Qwen/Qwen3-Embedding-4B
  - SPLADE-v3 at            /home/paul/hf-cache/models/naver/splade-v3

## 1. Functions

### `iter_chunk_files(d)`

Enumerate chunked DocTags JSONL files in a directory.

Args:
d: Directory containing `*.chunks.jsonl` files.

Returns:
Sorted list of chunk file paths.

### `ensure_uuid(rows)`

Populate missing chunk UUIDs in-place.

Args:
rows: Chunk dictionaries that should include a `uuid` key.

Returns:
True when at least one UUID was newly assigned; otherwise False.

### `tokens(text)`

Tokenize normalized text for sparse retrieval features.

Args:
text: Input string to tokenize.

Returns:
Lowercased alphanumeric tokens extracted from the text.

### `build_bm25_stats(chunks)`

Compute corpus statistics required for BM25 weighting.

Args:
chunks: Iterable of text chunks with identifiers.

Returns:
BM25Stats containing document frequency counts and average length.

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

### `_get_splade_encoder(cfg)`

Retrieve (or create) a cached SPLADE encoder instance.

### `qwen_embed(cfg, texts, batch_size)`

Produce dense embeddings using a local Qwen3 model served by vLLM.

Args:
cfg: Configuration describing model path, dtype, and batching.
texts: Batch of documents to embed.
batch_size: Optional override for inference batch size.

Returns:
List of embedding vectors, one per input text.

### `process_pass_a(files, logger)`

Assign UUIDs and build BM25 statistics for a corpus of chunk files.

Args:
files: Sequence of chunk file paths to process.
logger: Logger used for structured progress output.

Returns:
Tuple containing the UUID→Chunk index and aggregated BM25 statistics.

Raises:
ValueError: Propagated when chunk rows are missing required fields.

### `process_chunk_file_vectors(chunk_file, uuid_to_chunk, stats, args, validator, logger)`

Generate vectors for a single chunk file and persist them to disk.

Args:
chunk_file: Chunk JSONL file to process.
uuid_to_chunk: Mapping of chunk UUIDs to chunk metadata.
stats: Precomputed BM25 statistics.
args: Parsed CLI arguments with runtime configuration.
validator: SPLADE validator for sparsity metrics.
logger: Logger for structured output.

Returns:
Tuple of ``(vector_count, splade_nnz_list, qwen_norms)``.

Raises:
ValueError: Propagated if vector dimensions or norms fail validation.

### `write_vectors(path, uuids, texts, splade_results, qwen_results, stats, args)`

Write validated vector rows to disk with schema enforcement.

Args:
path: Destination JSONL path for vector rows.
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

### `build_parser()`

Construct the CLI parser for the embedding pipeline.

Args:
None: Parser creation does not require inputs.

Returns:
:class:`argparse.ArgumentParser` configured for embedding options.

Raises:
None

### `parse_args(argv)`

Parse CLI arguments for standalone embedding execution.

Args:
argv: Optional CLI argument vector. When ``None`` the current process
arguments are used.

Returns:
Namespace containing parsed embedding configuration.

Raises:
SystemExit: Propagated if ``argparse`` reports invalid options.

### `main(args)`

CLI entrypoint for chunk UUID cleanup and embedding generation.

Args:
args: Optional parsed arguments, primarily for testing or orchestration.

Returns:
Exit code where ``0`` indicates success.

Raises:
ValueError: If invalid runtime parameters (such as batch sizes) are supplied.

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

## 2. Classes

### `Chunk`

Minimal representation of a DocTags chunk stored on disk.

Attributes:
uuid: Stable identifier for the chunk.
text: Textual content extracted from the DocTags document.
doc_id: Identifier of the source document for manifest reporting.

Examples:
>>> chunk = Chunk(uuid="chunk-1", text="Hybrid search is powerful.", doc_id="doc")
>>> chunk.uuid
'chunk-1'

### `BM25Stats`

Corpus-wide statistics required for BM25 weighting.

Attributes:
N: Total number of documents (chunks) in the corpus.
avgdl: Average document length in tokens.
df: Document frequency per token.

Examples:
>>> stats = BM25Stats(N=100, avgdl=120.5, df={"hybrid": 5})
>>> stats.df["hybrid"]
5

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

### `SpladeCfg`

Runtime configuration for SPLADE sparse encoding.

Attributes:
model_dir: Path to the SPLADE model directory.
device: Torch device identifier to run inference on.
batch_size: Number of texts encoded per batch.
cache_folder: Directory where transformer weights are cached.
max_active_dims: Optional cap on active sparse dimensions.
attn_impl: Preferred attention implementation override.

Examples:
>>> cfg = SpladeCfg(batch_size=8, device="cuda:1")
>>> cfg.batch_size
8

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

### `QwenCfg`

Configuration for generating dense embeddings with Qwen via vLLM.

Attributes:
model_dir: Path to the local Qwen model.
dtype: Torch dtype used during inference.
tp: Tensor parallelism degree.
gpu_mem_util: Target GPU memory utilization for vLLM.
batch_size: Number of texts processed per embedding batch.
quantization: Optional quantization mode (e.g., `awq`).

Examples:
>>> cfg = QwenCfg(batch_size=64, tp=2)
>>> cfg.tp
2
