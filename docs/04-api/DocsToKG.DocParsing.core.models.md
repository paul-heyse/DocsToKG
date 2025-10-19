# 1. Module: models

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core.models``.

## 1. Overview

Lightweight data containers shared across DocParsing stages.

## 2. Classes

### `BM25Stats`

Corpus-level statistics required for BM25 weighting.

### `SpladeCfg`

Runtime configuration for SPLADE sparse encoding.

### `QwenCfg`

Configuration for generating dense embeddings with Qwen via vLLM.

### `ChunkWorkerConfig`

Lightweight configuration shared across chunker worker processes.

### `ChunkTask`

Work unit describing a single DocTags file to chunk.

### `ChunkResult`

Result envelope emitted by chunker workers.
