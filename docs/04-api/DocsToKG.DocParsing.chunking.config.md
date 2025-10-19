# 1. Module: config

This reference documents the DocsToKG module ``DocsToKG.DocParsing.chunking.config``.

## 1. Overview

Configuration objects for the chunking stage.

## 2. Functions

### `from_env(cls, defaults)`

Instantiate configuration derived solely from environment variables.

### `from_args(cls, args, defaults)`

Create a configuration by layering env vars, config files, and CLI args.

### `finalize(self)`

Normalise paths and derived values after merging all sources.

## 3. Classes

### `ChunkerCfg`

Configuration values for the chunking stage.
