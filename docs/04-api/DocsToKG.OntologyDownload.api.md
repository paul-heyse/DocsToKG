# 1. Module: api

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.api``.

## 1. Overview

Public facade aggregating ontology downloader modules.

## 2. Functions

### `_results_to_dict(result)`

Compatibility wrapper returning CLI-oriented fetch result payloads.

### `validator_worker_main()`

Entry point used by validator worker console scripts.

### `validate_url_security(url, http_config)`

Wrapper that normalizes PolicyError into ConfigError for the public API.

### `list_plugins(kind)`

Return a deterministic mapping of registered plugins for ``kind``.

Args:
kind: Plugin category (``"resolver"`` or ``"validator"``).

Returns:
Mapping of plugin names to import-qualified identifiers.

### `_collect_plugin_details(kind)`

Return plugin metadata including qualified path and version.

### `about()`

Return metadata describing the ontology download subsystem.

### `cli_main(argv)`

Entry point for the ontology downloader CLI.
