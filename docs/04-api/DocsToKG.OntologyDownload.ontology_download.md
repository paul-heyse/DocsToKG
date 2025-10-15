# 1. Module: ontology_download

This reference documents the consolidated module
``DocsToKG.OntologyDownload.ontology_download`` which now houses the full
ontology downloader stack: foundation helpers, configuration models, storage
backends, networking utilities, validation adapters, and pipeline orchestration.

## 1. Key Sections

- **Foundation utilities**: `retry_with_backoff`, `sanitize_filename`,
  `generate_correlation_id`, and `mask_sensitive_data` supply cross-cutting
  helpers used throughout the downloader.
- **Configuration & settings**: `ConfigError`, `DefaultsConfig`,
  `DownloadConfiguration`, `ResolvedConfig`, `load_config`,
  `validate_config`, and `setup_logging` define and manage CLI / YAML-driven
  configuration.
- **Storage infrastructure**: `CACHE_DIR`, `CONFIG_DIR`, `LOG_DIR`,
  `LOCAL_ONTOLOGY_DIR`, `StorageBackend`, `LocalStorageBackend`,
  `FsspecStorageBackend`, and `get_storage_backend` provide durable artifact
  storage with optional fsspec mirroring.
- **Network utilities**: `DownloadFailure`, `DownloadResult`,
  `TokenBucket`, `download_stream`, `extract_archive_safe`,
  `validate_url_security`, and `sha256_file` implement hardened downloading,
  polite throttling, archive safety, and checksum helpers.
- **Validation utilities**: `ValidationRequest`, `ValidationResult`,
  `normalize_streaming`, `validate_rdflib`, `validate_pronto`,
  `validate_owlready2`, `validate_robot`, `validate_arelle`, and
  `run_validators` ensure ontologies are parsed, normalized, and checked with
  deterministic outputs.
- **Download pipeline**: `FetchSpec`, `FetchResult`, `PlannedFetch`,
  `Manifest`, `fetch_one`, `fetch_all`, `plan_one`, `plan_all`,
  `MANIFEST_SCHEMA_VERSION`, `MANIFEST_JSON_SCHEMA`, and
  `validate_manifest_dict` orchestrate resolver planning, download execution,
  manifest creation, and batch workflows.
- **Resolvers integration**: The module collaborates with
  ``DocsToKG.OntologyDownload.resolvers`` via `RESOLVERS`, `FetchPlan`, and
  `normalize_license_to_spdx` to map planner specifications onto concrete
  download candidates.

## 2. Usage Example

```python
from DocsToKG.OntologyDownload import (
    FetchSpec,
    fetch_one,
    load_config,
)

config = load_config("configs/sources.yaml")
spec = FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])
result = fetch_one(spec, config=config)
print(result.status, result.sha256)
```
