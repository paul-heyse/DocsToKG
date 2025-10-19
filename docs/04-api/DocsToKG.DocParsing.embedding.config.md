# 1. Module: config

This reference documents the DocsToKG module ``DocsToKG.DocParsing.embedding.config``.

## 1. Overview

Configuration objects and presets for the embedding stage.

## 2. Functions

### `from_env(cls, defaults)`

Construct configuration from environment variables.

### `from_args(cls, args, defaults)`

Merge CLI arguments, configuration files, and environment variables.

### `finalize(self)`

Normalise paths and casing after all sources have been applied.

## 3. Classes

### `EmbedCfg`

Stage configuration container for the embedding pipeline.
