# 1. Module: runtime

This reference documents the DocsToKG module ``DocsToKG.DocParsing.chunking.runtime``.

## 1. Overview

Docling Hybrid Chunker with Minimum Token Coalescence

Transforms DocTags documents into chunked records with topic-aware coalescence.
The module exposes a CLI (``python -m DocsToKG.DocParsing.core chunk``)
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
    python -m DocsToKG.DocParsing.core chunk \
        --data-root /datasets/Data --min-tokens 256 --max-tokens 512

Tokenizer Alignment:
    The default tokenizer (``DEFAULT_TOKENIZER`` from :mod:`DocsToKG.DocParsing.core`) aligns with the dense
    embedder used by the embeddings pipeline. When experimenting with other
    tokenizers (for example, legacy BERT models), run the calibration utility
    beforehand to understand token count deltas::

        python -m DocsToKG.DocParsing.token_profiles --doctags-dir Data/DocTagsFiles

    The calibration script reports relative token ratios and recommends
    adjustments to ``--min-tokens`` so chunk sizes remain compatible with the
    embedding stage.

## 2. Functions

### `read_utf8(path)`

Load UTF-8 text from ``path`` replacing undecodable bytes.

### `_hash_doctags_text(text)`

Return a normalised content hash for DocTags ``text``.

### `build_doc(doc_name, doctags_text)`

Construct a Docling document from serialized DocTags markup.

### `extract_refs_and_pages(chunk)`

Extract inline references and page numbers from a Docling chunk.

### `is_structural_boundary(record, heading_markers, caption_markers)`

Return True when ``record`` begins with a structural marker.

### `summarize_image_metadata(chunk, text)`

Summarise image annotations associated with ``chunk``.

### `_extract_chunk_start(chunk)`

Attempt to extract the starting character offset for ``chunk``.

### `merge_rec(a, b, tokenizer)`

Merge two chunk records into a single aggregate record.

### `coalesce_small_runs(records, tokenizer)`

Merge contiguous undersized chunks while respecting structural boundaries.

### `_chunk_worker_initializer(cfg)`

Initialise shared tokenizer/chunker state for worker processes.

### `_process_chunk_task(task)`

Chunk a single DocTags file using worker-local state.

### `_process_indexed_chunk_task(payload)`

Execute ``_process_chunk_task`` and preserve submission ordering.

### `_ordered_results(results)`

Yield chunk results in their original submission order.

### `_resolve_serializer_provider(spec)`

Return the serializer provider class referenced by ``spec``.

### `_validate_chunk_files(files, logger)`

Validate chunk JSONL rows across supplied files.

Returns a dictionary summarising file, row, and quarantine counts. Detailed
log events for individual errors are emitted within the function; callers
are responsible for logging the aggregate summary so they can attach
run-specific context.

### `_main_inner(args)`

CLI driver that chunks DocTags files and enforces minimum token thresholds.

Args:
args (argparse.Namespace | None): Optional CLI namespace supplied during
testing or orchestration.

Returns:
int: Exit code where ``0`` indicates success.

### `_run_validate_only()`

Validate chunk inputs and report statistics without writing outputs.

### `main(args)`

Wrapper that normalises CLI validation failures for the chunk stage.

### `iter_chunk_tasks()`

Generate chunk tasks for processing, respecting resume/force settings.

### `handle_result(result)`

Persist manifest information and raise on worker failure.

Args:
result: Structured outcome emitted by the chunking worker.

## 3. Classes

### `Rec`

Chunk record used by coalescence helpers during validation.
