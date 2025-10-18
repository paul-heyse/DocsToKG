# 1. Troubleshooting Guide

Common issues and recovery steps when operating DocsToKG.

## 2. Hybrid Search

- **`400 Bad Request` responses** – Missing `query` or invalid payload types. Confirm payloads match `HybridSearchRequest` schema and enable diagnostics logging via `HybridSearchAPI.post_hybrid_search` to inspect errors.
- **Empty results for known documents** – Namespace mismatches or stale FAISS snapshots. Verify namespace registration, then run `serialize_state` / `restore_state` with the latest ingestion outputs.
- **Slow queries (>500 ms)** – Low FAISS `nprobe`, overloaded hardware, or cold caches. Increase `config.dense.nprobe`, scale compute resources, and warm caches by replaying representative queries.
- **Divergent ranking between releases** – Configuration drift. Compare `hybrid_config.json` (fusion weights, filters) and re-run `python -m DocsToKG.HybridSearch.validation`.
- **API timeouts** – FastAPI deployment not reusing the service instance. Ensure `HybridSearchService` is initialised once at startup and injected into the router.

## 3. Document Parsing

- **Chunking errors for specific DocTags** – Execute `direnv exec . docparse chunk --log-level DEBUG --validate-only` to capture granular logs without writing outputs. Review telemetry JSONL for the failing doc ID.
- **Embedding throughput low** – Confirm CUDA 12.9 drivers match the bundled `torch`/`xformers` wheels and tune `--batch-size-qwen` or `--execution-backend` (CPU vs VLLM) to fit available VRAM.
- **`serializer provider` import failures** – Ensure custom providers are importable modules and derive from `ChunkingSerializerProvider`. Pass the dotted path via `--serializer-provider package.module:Provider`.
- **NAVMAP/docstring violations** – Run `direnv exec . python docs/scripts/validate_code_annotations.py` to surface missing docstrings or misordered navigation metadata.

## 4. Ontology Download

- **Authentication errors** – Renew BioPortal/OBO tokens in a secrets manager; export them before running the CLI (`PA_ALEX_KEY`, `BIOPORTAL_API_KEY`).
- **Validation crashes** – Scope validators while debugging (`--only rdflib,pronto`) then re-enable the full suite prior to production runs.
- **Incomplete manifests** – Re-run `direnv exec . python -m DocsToKG.OntologyDownload.cli pull <id> --force --json` to regenerate metadata and inspect the resulting JSON for failures.
- **Slow downloads** – Check resolver rate limits in `settings.defaults.http` and adjust concurrency (`--defaults.http.max_concurrent_downloads`) cautiously.

## 5. Documentation Tooling

- `validate_docs.py` warns about missing sections – ensure required headings follow templates in `docs/05-development/index.md`.
- `validate_code_annotations.py` reports NAVMAP mismatches – update the JSON block at the top of the module so sections mirror top-level class/function order.
- `check_links.py` timeouts – retry with `--timeout 20` or add flaky domains to the allowlist in `docs/scripts/check_links.py`.
- Sphinx build failures – install dependencies from `docs/build/sphinx/requirements.txt` and confirm `direnv exec . python docs/scripts/build_docs.py --format html` runs with `PYTHONPATH=src`.

Escalate unresolved issues in `CONTRIBUTING.md` guidance and document solutions for future playbooks.
