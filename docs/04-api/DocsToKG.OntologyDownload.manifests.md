# 1. Module: manifests

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.manifests``.

## 1. Overview

Manifest and lockfile utilities for the OntologyDownload CLI.

## 2. Functions

### `plan_to_dict(plan)`

Convert a planned fetch into a JSON-friendly dictionary.

### `write_json_atomic(path, payload)`

Atomically persist ``payload`` as JSON to ``path``.

### `write_lockfile(plans, path)`

Write lockfile capturing planned resolver outputs.

### `load_lockfile_payload(path)`

Return the parsed lockfile payload.

### `spec_from_lock_entry(entry, defaults)`

Convert a lockfile entry back into a fetch specification.

### `specs_from_lock_payload(payload)`

Build fetch specifications from lockfile payload.

### `resolve_version_metadata(ontology_id, version)`

Return path, timestamp, and size metadata for a stored version.

### `ensure_manifest_path(ontology_id, version)`

Return the manifest path for a given ontology and version.

### `load_manifest(manifest_path)`

Read and parse a manifest JSON document from disk.

### `collect_version_metadata(ontology_id)`

Return sorted metadata entries for stored ontology versions.

### `load_latest_manifest(ontology_id)`

Return the most recent manifest for ``ontology_id`` when available.

### `results_to_dict(result)`

Serialize a :class:`FetchResult` to a JSON-friendly dictionary.

### `compute_plan_diff(baseline, current)`

Compute a diff between baseline and current plan payloads.

### `format_plan_diff(diff)`

Render human-readable diff lines from plan comparison.
