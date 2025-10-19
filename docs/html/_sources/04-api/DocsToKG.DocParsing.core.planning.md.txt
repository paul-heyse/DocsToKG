# 1. Module: planning

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core.planning``.

## 1. Overview

Planner orchestration utilities for DocParsing stages.

## 2. Functions

### `_new_bucket()`

Return a new mutable bucket for tracking plan membership.

### `_record_bucket(bucket, doc_id)`

Update ``bucket`` with ``doc_id`` while respecting preview bounds.

### `_bucket_counts(entry, key)`

Return ``(count, preview)`` for ``key`` within ``entry``.

### `_manifest_hash_requirements(manifest_entry)`

Return manifest hash metadata when both entry and hash are present.

### `_render_preview(preview, count)`

Render a preview string that includes remainder hints when applicable.

### `plan_doctags(argv)`

Compute which DocTags inputs would be processed.

### `plan_chunk(argv)`

Compute which DocTags files the chunk stage would touch.

### `plan_embed(argv)`

Compute which chunk files the embed stage would process or validate.

### `display_plan(plans, stream)`

Pretty-print plan summaries and return the rendered lines.
