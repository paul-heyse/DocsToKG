# 1. Module: DoclingHybridChunkerPipelineWithMin

This reference documents the DocsToKG module ``DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin``.

Docling Hybrid Chunker with Minimum Token Coalescence

Transforms DocTags documents into chunked records with topic-aware coalescence.

Tokenizer Alignment:
    The default tokenizer (``Qwen/Qwen3-Embedding-4B``) aligns with the dense
    embedder used by the embeddings pipeline. When experimenting with other
    tokenizers (for example, legacy BERT models), run the calibration utility
    beforehand to understand token count deltas::

        python scripts/calibrate_tokenizers.py --doctags-dir Data/DocTagsFiles

    The calibration script reports relative token ratios and recommends
    adjustments to ``--min-tokens`` so chunk sizes remain compatible with the
    embedding stage.

## 1. Functions

### `_promote_simple_namespace_modules()`

Convert any SimpleNamespace placeholders in sys.modules to real modules.

Some tests install lightweight SimpleNamespace stubs into sys.modules for
optional dependencies (for example ``trafilatura``). Hypothesis' internal
providers assume module objects are hashable, which SimpleNamespace is not.
Promoting the stubs to ModuleType instances preserves their attributes while
restoring hashability, preventing spurious test failures.

### `read_utf8(p)`

Load text from disk using UTF-8 with replacement for invalid bytes.

Args:
p: Path to the text file.

Returns:
String contents of the file.

### `build_doc(doc_name, doctags_text)`

Construct a Docling document from serialized DocTags text.

Args:
doc_name: Human-readable document identifier for logging.
doctags_text: Serialized DocTags payload.

Returns:
Loaded DoclingDocument ready for chunking.

### `extract_refs_and_pages(chunk)`

Collect self-references and page numbers associated with a chunk.

Args:
chunk: Chunk object produced by the hybrid chunker.

Returns:
Tuple containing a list of reference identifiers and sorted page numbers.

Raises:
None

### `summarize_image_metadata(chunk, text)`

Infer image annotation flags and counts from chunk metadata and text.

Args:
chunk: Chunk metadata object containing image annotations.
text: Chunk text used to detect fallback caption cues.

Returns:
Tuple of ``(has_caption, has_classification, num_images)`` describing
inferred image metadata.

### `merge_rec(a, b, tokenizer)`

Merge two chunk records, updating token counts and provenance metadata.

Args:
a: First record to merge.
b: Second record to merge.
tokenizer: Tokenizer used to recompute token counts for combined text.

Returns:
New `Rec` instance containing fused text, token counts, and metadata.

### `is_structural_boundary(rec)`

Detect whether a chunk begins with a structural heading or caption marker.

Args:
rec: Chunk record to inspect.

Returns:
``True`` when ``rec.text`` starts with a heading indicator (``#``) or a
recognised caption prefix, otherwise ``False``.

Examples:
>>> is_structural_boundary(Rec(text="# Introduction", n_tok=2, src_idxs=[], refs=[], pages=[]))
True
>>> is_structural_boundary(Rec(text="Regular paragraph", n_tok=2, src_idxs=[], refs=[], pages=[]))
False

### `coalesce_small_runs(records, tokenizer, min_tokens, max_tokens)`

Merge contiguous short chunks until they satisfy minimum token thresholds.

Args:
records: Ordered list of chunk records to normalize.
tokenizer: Tokenizer used to recompute token counts for merged chunks.
min_tokens: Target minimum tokens per chunk after coalescing.
max_tokens: Hard ceiling to avoid producing overly large chunks.

Returns:
New list of records where small runs are merged while preserving order.

Note:
Strategy:
• Identify contiguous runs where every chunk has fewer than `min_tokens`.
• Greedily pack neighbors within a run to exceed `min_tokens` without
surpassing `max_tokens`.
• Merge trailing fragments into adjacent groups when possible,
preferring same-run neighbors to maintain topical cohesion.
• Leave chunks outside small runs unchanged.

### `build_parser()`

Construct an argument parser for the chunking pipeline.

Args:
None: Parser construction does not require inputs.

Returns:
:class:`argparse.ArgumentParser` configured with chunking options.

Raises:
None

### `parse_args(argv)`

Parse CLI arguments for standalone chunking execution.

Args:
argv: Optional CLI argument vector. When ``None`` the process arguments
are parsed.

Returns:
Namespace containing parsed CLI options.

Raises:
SystemExit: Propagated if ``argparse`` reports invalid arguments.

### `main(args)`

CLI driver that chunks DocTags files and enforces minimum token thresholds.

Args:
args: Optional CLI namespace supplied during testing or orchestration.

Returns:
Exit code where ``0`` indicates success.

### `is_small(idx)`

Return True when the chunk at `idx` is below the minimum token threshold.

Args:
idx: Index of the chunk under evaluation.

Returns:
True if the chunk length is less than `min_tokens`, else False.

## 2. Classes

### `Rec`

Intermediate record tracking chunk text and provenance.

Attributes:
text: Chunk text content.
n_tok: Token count computed by the tokenizer.
src_idxs: Source chunk indices contributing to this record.
refs: List of inline reference identifiers.
pages: Page numbers associated with the chunk.

Examples:
>>> rec = Rec(text="Example", n_tok=5, src_idxs=[0], refs=[], pages=[1])
>>> rec.n_tok
5
