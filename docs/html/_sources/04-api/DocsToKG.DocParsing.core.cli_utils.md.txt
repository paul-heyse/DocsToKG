# 1. Module: cli_utils

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core.cli_utils``.

## 1. Overview

Reusable CLI assembly helpers for DocParsing.

## 2. Functions

### `build_subcommand(parser, options)`

Attach CLI options described by ``options`` to ``parser``.

### `preview_list(items, limit)`

Return a truncated preview list with remainder hint.

### `merge_args(parser, overrides)`

Merge override values into the default parser namespace.

### `scan_pdf_html(input_dir)`

Return booleans indicating whether PDFs or HTML files exist beneath ``input_dir``.

### `directory_contains_suffixes(directory, suffixes)`

Return True when ``directory`` contains at least one file ending with ``suffixes``.

### `detect_mode(input_dir)`

Infer conversion mode based on the contents of ``input_dir``.

## 3. Classes

### `CLIOption`

Declarative CLI argument specification used by ``build_subcommand``.
