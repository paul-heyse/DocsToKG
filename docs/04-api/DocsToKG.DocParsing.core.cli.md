# 1. Module: cli

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core.cli``.

## 1. Overview

Unified CLI entry points for DocParsing stages.

## 2. Functions

### `_run_stage(handler, argv)`

Execute a stage handler while normalising CLI validation errors.

### `build_doctags_parser(prog)`

Create an :mod:`argparse` parser configured for DocTags conversion.

### `_resolve_doctags_paths(args)`

Resolve DocTags input/output directories and mode.

### `doctags(argv)`

Execute the DocTags conversion subcommand.

### `chunk(argv)`

Execute the Docling chunker subcommand.

### `embed(argv)`

Execute the embedding pipeline subcommand.

### `token_profiles(argv)`

Execute the tokenizer profiling subcommand.

### `plan(argv)`

Display the doctags → chunk → embed plan without executing.

### `manifest(argv)`

Inspect pipeline manifest artifacts via CLI.

### `_manifest_main(argv)`

Implementation for the ``docparse manifest`` command.

### `_build_stage_args(args)`

Construct argument lists for doctags/chunk/embed stages.

### `run_all(argv)`

Execute DocTags conversion, chunking, and embedding sequentially.

### `_command(handler, help_text)`

Package a handler and help text into a command descriptor.

### `main(argv)`

Dispatch to one of the DocParsing subcommands.

### `_split_lines(self, text, width)`

Wrap help text without breaking on intra-stage hyphens.

## 3. Classes

### `_ManifestHelpFormatter`

Help formatter that avoids hyphenated aliases being split across lines.

### `_Command`

Callable wrapper storing handler metadata for subcommands.
