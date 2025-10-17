# 1. Module: token_profiles

This reference documents the DocsToKG module ``DocsToKG.DocParsing.token_profiles``.

## 1. Overview

Print simple token ratio stats for DocTags samples.

## 2. Functions

### `_clean_text(text, max_chars)`

Strip DocTags markup and collapse whitespace.

### `_load_samples(root, sample_size, max_chars)`

Read DocTags files from ``root`` and return cleaned text samples.

### `_count_tokens(name, texts)`

Return token counts for ``texts`` using the HuggingFace tokenizer ``name``.

### `_mean(values)`

*No documentation available.*

### `_mean_ratio(candidate, baseline)`

*No documentation available.*

### `_scale(window, ratio)`

*No documentation available.*

### `_render_table(tokenizer_ids, counts, baseline_name, baseline_counts, window_min, window_max)`

Render a table summarising token statistics.

### `build_parser()`

Create the CLI parser for tokenizer profiling.

### `parse_args(argv)`

Parse CLI arguments for tokenizer profiling.

### `main(args)`

Entry point for the tokenizer profiling CLI.

### `from_env(cls, defaults)`

Instantiate configuration using environment overlays.

### `from_args(cls, args, defaults)`

Layer CLI arguments, config files, and env vars into a configuration.

### `finalize(self)`

Normalise derived attributes after overlays are applied.

### `tokenizer_ids(self)`

Return the ordered tokenizer identifiers to profile.

## 3. Classes

### `TokenProfilesCfg`

Structured configuration for tokenizer ratio profiling.
