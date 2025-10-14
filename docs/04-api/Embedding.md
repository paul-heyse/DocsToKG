# Module: Embedding

EmbedVectors.py

- Reads chunked JSONL files from /home/paul/DocsToKG/Data/ChunkedDocTagFiles
- Ensures each chunk has a stable UUID (writes back to the chunk files)
- Emits vectors JSONL to   /home/paul/DocsToKG/Data/Vectors

For each chunk record in vectors JSONL:
{
  "UUID": "...",
  "BM25": {
    "terms": [...], "weights": [...],
    "k1": 1.5, "b": 0.75, "avgdl": <float>, "N": <int>
  },
  "SpladeV3": {
    "model_id": "naver/splade-v3",
    "tokens": [...], "weights": [...]
  },
  "Qwen3-4B": {
    "model_id": "Qwen/Qwen3-Embedding-4B",
    "dim": 2048,
    "vector": [...]
  }
}

## Functions

### `iter_chunk_files(in_dir)`

*No documentation available.*

### `load_chunks(path)`

*No documentation available.*

### `save_chunks(path, rows)`

*No documentation available.*

### `ensure_uuid(rows)`

*No documentation available.*

### `tokens(text)`

*No documentation available.*

### `build_bm25_stats(all_chunks)`

*No documentation available.*

### `bm25_vector(text, stats, k1, b)`

Return (terms, weights) sparse BM25 vector for a document, using corpus-wide IDF.

### `splade_encode(cfg, texts)`

Returns per-text tokens & weights (only non-zeros).

### `qwen_embed(cfg, texts)`

*No documentation available.*

### `main()`

*No documentation available.*

## Classes

### `Chunk`

*No documentation available.*

### `BM25Stats`

*No documentation available.*

### `SpladeConfig`

*No documentation available.*

### `QwenCfg`

*No documentation available.*
