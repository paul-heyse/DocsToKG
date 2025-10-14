# Module: DoclingHybridChunkerPipeline

DocTags -> DoclingDocument -> HybridChunker (512 tokens, no overlap) -> JSONL

Input : /home/paul/DocsToKG/Data/DocTagsFiles
Output: /home/paul/DocsToKG/Data/ChunkedDocTagFiles

Behavior:
- No fallback code paths.
- No custom min-token coalescence: rely on HybridChunker split/merge.
- Inject picture captions (and optional annotations) into contextualized text.

## Functions

### `find_doctags_files(in_dir)`

*No documentation available.*

### `read_utf8(path)`

*No documentation available.*

### `load_docling_from_doctags_string(doc_name, doctags_text)`

*No documentation available.*

### `extract_refs_and_pages(chunk)`

*No documentation available.*

### `main()`

*No documentation available.*

### `serialize(self)`

*No documentation available.*

### `get_serializer(self, doc)`

*No documentation available.*

## Classes

### `CaptionPlusAnnotationPictureSerializer`

Emit picture caption and selected annotations into the chunk's text.

### `RichSerializerProvider`

Use Markdown tables + caption-aware pictures; keep default params otherwise.
