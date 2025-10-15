# 1. Troubleshooting Guide

Common issues and recovery steps when operating DocsToKG.

## 2. Hybrid Search

- **`400 Bad Request` responses** – Missing `query` or invalid payload types. Validate
  client payloads and enable the diagnostics flag to inspect validation errors.
- **Empty results for known documents** – Namespace misconfiguration or stale FAISS
  snapshots. Verify namespace registration, then rebuild and reload the FAISS index.
- **Slow queries (>500 ms)** – Low `nprobe`, overloaded hardware, or stale caches.
  Increase `nprobe`, scale compute resources, and warm caches via popular queries.
- **Divergent scores between releases** – Configuration drift. Compare
  `FusionConfig` versions and re-run the validation harness.

## 3. Document Parsing

- **Parsing fails on specific PDFs** – Run
  `python -m DocsToKG.DocParsing.run_docling_html_to_doctags_parallel --debug` to
  capture detailed failure logs. Consider upgrading Docling or adjusting chunking
  parameters.
- **Embedding throughput low** – Ensure GPU drivers align with `requirements.in`
  versions and batching flags use available VRAM efficiently.

## 4. Ontology Download

- **Authentication errors** – Confirm BioPortal or OBO credentials via a secrets
  manager; tokens may expire.
- **Validation crashes** – Use `--rdflib` or `--pronto` options to scope validators
  when debugging. Re-enable the full suite before production publication.
- **Missing manifests** – Re-run `python -m DocsToKG.OntologyDownload.cli pull <id>
  --force` to regenerate metadata.

## 5. Documentation Tooling

- `validate_docs.py` warns about missing sections: confirm required headings exist (see
  Style Guide).
- `check_links.py` reports timeouts: re-run with a larger `--timeout` or whitelist
  intermittent domains.
- Sphinx builds fail on missing modules: install dependencies from
  `docs/build/sphinx/requirements.txt` and set `PYTHONPATH=src`.

Escalate unresolved issues in `CONTRIBUTING.md` guidance and document solutions for future playbooks.
