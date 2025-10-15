# 1. Module: cli

This reference documents the DocsToKG module ``DocsToKG.DocParsing.cli``.

Unified DocParsing command-line interface.

This module consolidates the individual DocParsing CLIs into a single entry
point with subcommands. Invoke it with:

    python -m DocsToKG.DocParsing.cli <command> [options...]

Available commands:
    - chunk:     Run the Docling hybrid chunker.
    - embed:     Generate BM25, SPLADE, and dense vectors for chunks.
    - doctags:   Convert HTML/PDF corpora into DocTags.

## 1. Functions

### `_run_chunk(argv)`

*No documentation available.*

### `_run_embed(argv)`

*No documentation available.*

### `_build_doctags_parser(prog)`

*No documentation available.*

### `_detect_mode(input_dir)`

*No documentation available.*

### `_merge_args(parser, overrides)`

*No documentation available.*

### `_run_doctags(argv)`

*No documentation available.*

### `main(argv)`

Dispatch to one of the DocParsing subcommands.

### `chunk(argv)`

Programmatic helper mirroring ``docparse chunk``.

### `embed(argv)`

Programmatic helper mirroring ``docparse embed``.

### `doctags(argv)`

Programmatic helper mirroring ``docparse doctags``.

## 2. Classes

### `_Command`

*No documentation available.*
