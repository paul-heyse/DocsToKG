# 1. Module: chunking

This reference documents the DocsToKG module ``DocsToKG.DocParsing.chunking``.

## 1. Overview

Docling Hybrid Chunker with Minimum Token Coalescence

Transforms DocTags documents into chunked records with topic-aware coalescence.
The module exposes a CLI (`python -m DocsToKG.DocParsing.chunking`)
and reusable helpers for other pipelines.

Key Features:
- Token-aware chunk merging that respects structural boundaries and image metadata.
- Shared CLI configuration via :func:`DocsToKG.DocParsing.doctags.add_data_root_option`.
- Manifest logging that records chunk counts, parsing engines, and durations.

Dependencies:
- docling_core: Provides chunkers, serializers, and DocTags parsing.
- transformers: Supplies HuggingFace tokenizers.
- tqdm: Optional progress reporting when imported by callers.

Usage:
    python -m DocsToKG.DocParsing.chunking \
        --data-root /datasets/Data --min-tokens 256 --max-tokens 512

Tokenizer Alignment:
    The default tokenizer (``DEFAULT_TOKENIZER`` from :mod:`DocsToKG.DocParsing.core`) aligns with the dense
    embedder used by the embeddings pipeline. When experimenting with other
    tokenizers (for example, legacy BERT models), run the calibration utility
    beforehand to understand token count deltas::

        python scripts/calibrate_tokenizers.py --doctags-dir Data/DocTagsFiles

    The calibration script reports relative token ratios and recommends
    adjustments to ``--min-tokens`` so chunk sizes remain compatible with the
    embedding stage.

## 2. Functions

### `_resolve_serializer_provider(spec)`

Return the serializer provider class referenced by ``spec``.

### `_validate_chunk_files(files, logger)`

Validate chunk JSONL rows across supplied files.

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

Infer image annotation flags, counts, confidences, and structured metadata.

### `_chunk_worker_initializer(cfg)`

Initialise worker-local tokenizer and chunker state for multiprocessing.

### `_process_chunk_task(task)`

Chunk a single DocTags file inside a worker process.

### `merge_rec(a, b, tokenizer)`

Merge two chunk records, updating token counts and provenance metadata.

Args:
a: First record to merge.
b: Second record to merge.
tokenizer: Tokenizer used to recompute token counts for combined text.
recount: When ``True`` the merged text is re-tokenized; otherwise token
counts are summed from inputs.

Returns:
New `Rec` instance containing fused text, token counts, and metadata.

### `is_structural_boundary(rec, heading_markers, caption_markers)`

Detect whether a chunk begins with a structural heading or caption marker.

Args:
rec: Chunk record to inspect.
heading_markers: Optional prefixes treated as section headings.
caption_markers: Optional prefixes treated as caption markers.

Returns:
``True`` when ``rec.text`` starts with a heading indicator (``#``) or a
recognised caption prefix, otherwise ``False``.

Examples:
>>> is_structural_boundary(Rec(text="# Introduction", n_tok=2, src_idxs=[], refs=[], pages=[]))
True
>>> is_structural_boundary(Rec(text="Regular paragraph", n_tok=2, src_idxs=[], refs=[], pages=[]))
False

### `coalesce_small_runs(records, tokenizer, min_tokens, max_tokens, soft_barrier_margin, heading_markers, caption_markers)`

Merge contiguous short chunks until they satisfy minimum token thresholds.

Args:
records: Ordered list of chunk records to normalize.
tokenizer: Tokenizer used to recompute token counts for merged chunks.
min_tokens: Target minimum tokens per chunk after coalescing.
max_tokens: Hard ceiling to avoid producing overly large chunks.
soft_barrier_margin: Margin applied when respecting structural boundaries.
heading_markers: Optional heading prefixes treated as structural boundaries.
caption_markers: Optional caption prefixes treated as structural boundaries.

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
None

Returns:
argparse.ArgumentParser: Parser configured with chunking options.

Raises:
None

### `parse_args(argv)`

Parse CLI arguments for standalone chunking execution.

Args:
argv (list[str] | None): Optional CLI argument vector. When ``None`` the
process arguments are parsed.

Returns:
argparse.Namespace: Parsed CLI options.

Raises:
SystemExit: Propagated if ``argparse`` reports invalid arguments.

### `main(args)`

CLI driver that chunks DocTags files and enforces minimum token thresholds.

Args:
args (argparse.Namespace | None): Optional CLI namespace supplied during
testing or orchestration.

Returns:
int: Exit code where ``0`` indicates success.

### `from_env(cls, defaults)`

Instantiate configuration derived solely from environment variables.

### `from_args(cls, args, defaults)`

Create a configuration by layering env vars, config files, and CLI args.

### `finalize(self)`

Normalise paths, casing, and defaults after all inputs are merged.

### `_maybe_add_conf(value)`

Collect numeric confidence scores from metadata sources.

### `_normalise_meta(payload, doc_item)`

*No documentation available.*

### `is_small(idx)`

Return True when the chunk at `idx` is below the minimum token threshold.

Args:
idx: Index of the chunk under evaluation.

Returns:
True if the chunk length is less than `min_tokens`, else False.

### `handle_result(result)`

Persist manifest information and raise on worker failure.

Args:
result: Structured outcome emitted by the chunking worker.

## 3. Classes

### `ChunkerCfg`

Configuration values for the chunking stage.

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
