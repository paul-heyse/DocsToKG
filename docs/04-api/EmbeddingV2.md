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

*No documentation available.*

### `load_rows(p)`

*No documentation available.*

### `save_rows(p, rows)`

*No documentation available.*

### `ensure_uuid(rows)`

*No documentation available.*

### `tokens(text)`

*No documentation available.*

### `build_bm25_stats(chunks)`

*No documentation available.*

### `bm25_vector(text, stats, k1, b)`

*No documentation available.*

### `splade_encode(cfg, texts)`

*No documentation available.*

### `qwen_embed(cfg, texts)`

*No documentation available.*

### `main()`

*No documentation available.*

## Classes

### `Chunk`

*No documentation available.*

### `BM25Stats`

*No documentation available.*

### `SpladeCfg`

*No documentation available.*

### `QwenCfg`

*No documentation available.*
