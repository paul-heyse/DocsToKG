# Module: EmbeddingV2

EmbedVectors.py

- Reads chunked JSONL from /home/paul/DocsToKG/Data/ChunkedDocTagFiles
- Ensures each chunk has a UUID (writes back to the same files)
- Emits vectors JSONL to /home/paul/DocsToKG/Data/Vectors

Uses local HF cache at /home/paul/hf-cache/:
  - Qwen3-Embedding-4B at   /home/paul/hf-cache/models/Qwen/Qwen3-Embedding-4B
  - SPLADE-v3 at            /home/paul/hf-cache/models/naver/splade-v3

## Functions

### `iter_chunk_files(d)`

Enumerate chunked DocTags JSONL files in a directory.

Args:
d: Directory containing `*.chunks.jsonl` files.

Returns:
Sorted list of chunk file paths.

### `load_rows(p)`

Load JSONL rows from disk into memory.

Args:
p: Path to the `.jsonl` file.

Returns:
List of dictionaries parsed from the file.

Raises:
json.JSONDecodeError: If a line contains malformed JSON.

### `save_rows(p, rows)`

Persist JSONL rows to disk atomically using a temporary file.

Args:
p: Destination path for the chunk file.
rows: Sequence of dictionaries to serialize.

Returns:
None

### `ensure_uuid(rows)`

Populate missing chunk UUIDs in-place.

Args:
rows: Chunk dictionaries that should include a `uuid` key.

Returns:
None

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

### `bm25_vector(text, stats, k1, b)`

Generate BM25 term weights for a chunk of text.

Args:
text: Chunk text to convert into a sparse representation.
stats: Precomputed BM25 statistics for the corpus.
k1: Term frequency saturation parameter.
b: Length normalization parameter.

Returns:
Tuple of `(terms, weights)` describing the sparse vector.

### `splade_encode(cfg, texts)`

Encode text with SPLADE to obtain sparse lexical vectors.

Args:
cfg: SPLADE configuration describing device, batch size, and cache.
texts: Batch of input strings to encode.

Returns:
Tuple of token lists and weight lists aligned per input text.

### `qwen_embed(cfg, texts)`

Produce dense embeddings using a local Qwen3 model served by vLLM.

Args:
cfg: Configuration describing model path, dtype, and batching.
texts: Batch of documents to embed.

Returns:
List of embedding vectors, one per input text.

### `main()`

CLI entrypoint for chunk UUID cleanup and embedding generation.

Args:
None

Returns:
None

## Classes

### `Chunk`

Minimal representation of a DocTags chunk stored on disk.

Attributes:
uuid: Stable identifier for the chunk.
text: Textual content extracted from the DocTags document.

Examples:
>>> chunk = Chunk(uuid="chunk-1", text="Hybrid search is powerful.")
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
