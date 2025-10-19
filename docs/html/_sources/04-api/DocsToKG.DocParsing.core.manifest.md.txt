# 1. Module: manifest

This reference documents the DocsToKG module ``DocsToKG.DocParsing.core.manifest``.

## 1. Overview

Manifest bookkeeping utilities shared across DocParsing stages.

## 2. Functions

### `should_skip_output(output_path, manifest_entry, input_hash, resume, force)`

Return ``True`` when resume/skip conditions indicate work can be skipped.

### `entry(self, doc_id)`

Return the manifest entry associated with ``doc_id`` when available.

### `can_skip_without_hash(self, doc_id, output_path)`

Return ``True`` when manifest metadata alone justifies skipping.

The predicate implements a fast path for planners and other tooling that
only need to confirm that successful outputs already exist. It honours
the resume/force flags and checks that the manifest recorded a
successful or skipped status without touching the input payload.

### `should_skip(self, doc_id, output_path, input_hash)`

Return ``True`` when work for ``doc_id`` can be safely skipped.

### `should_process(self, doc_id, output_path, input_hash)`

Return ``True`` when ``doc_id`` requires processing.

## 3. Classes

### `ResumeController`

Centralize resume/force decisions using manifest metadata.
