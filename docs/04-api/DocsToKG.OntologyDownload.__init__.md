# 1. Module: __init__

This reference documents ``DocsToKG.OntologyDownload.__init__``. The package now
re-exports its public API from the consolidated
``DocsToKG.OntologyDownload.ontology_download`` module and retains backward
compatible aliases for legacy import paths (e.g., ``core``, ``settings``,
``validation``).

Key exports include:

- Configuration helpers such as `DefaultsConfig`, `DownloadConfiguration`, and
  `load_config`
- Storage primitives (`StorageBackend`, `LOCAL_ONTOLOGY_DIR`, `STORAGE`)
- Networking utilities (`download_stream`, `validate_url_security`)
- Validation adapters (`ValidationRequest`, `validate_rdflib`, `run_validators`)
- Pipeline orchestration (`FetchSpec`, `fetch_one`, `plan_all`)

For a detailed breakdown of each area, see
``DocsToKG.OntologyDownload.ontology_download``.
