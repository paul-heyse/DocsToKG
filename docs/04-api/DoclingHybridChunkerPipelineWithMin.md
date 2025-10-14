# Module: DoclingHybridChunkerPipelineWithMin

## Functions

### `find_doctags_files(in_dir)`

*No documentation available.*

### `read_utf8(p)`

*No documentation available.*

### `build_doc(doc_name, doctags_text)`

*No documentation available.*

### `extract_refs_and_pages(chunk)`

*No documentation available.*

### `merge_rec(a, b, tokenizer)`

*No documentation available.*

### `coalesce_small_runs(records, tokenizer, min_tokens, max_tokens)`

Strategy:
  • Identify contiguous runs where EVERY chunk < min_tokens.
  • Within each run, greedily pack neighbors to form groups that hit >= min_tokens
without exceeding max_tokens (keeps small chunks together).
  • If the final group in a run is still < min_tokens:
- prefer merging into the previous group from the SAME RUN if it fits,
- else merge into the smaller adjacent BIG neighbor (left/right) only if it fits,
- else keep as-is (rare).
  • Chunks >= min_tokens outside runs are left untouched (avoids skewing well-formed 300–500 token chunks).

### `main()`

*No documentation available.*

### `serialize(self)`

*No documentation available.*

### `get_serializer(self, doc)`

*No documentation available.*

### `is_small(idx)`

*No documentation available.*

## Classes

### `CaptionPlusAnnotationPictureSerializer`

*No documentation available.*

### `RichSerializerProvider`

*No documentation available.*

### `Rec`

*No documentation available.*
