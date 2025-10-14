# Troubleshooting Guide

Common issues and recovery steps when operating DocsToKG.

## Hybrid Search

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `400 Bad Request` from `/v1/hybrid-search` | Missing `query` or invalid payload types | Validate client payloads, enable diagnostics flag to inspect validation errors |
| Empty results for known documents | Namespace misconfiguration or stale FAISS snapshot | Verify namespace registration, rebuild and reload FAISS index |
| Slow queries (>500 ms) | Low `nprobe`, overloaded hardware, or stale caches | Increase `nprobe`, scale compute, warm caches via popular queries |
| Divergent scores between releases | Fusion config drift | Compare `FusionConfig` versions, re-run validation harness |

## Document Parsing

- **Parsing fails on specific PDFs**: Run `python -m DocsToKG.DocParsing.run_docling_html_to_doctags_parallel --debug` to capture detailed failure logs. Consider upgrading Docling or adjusting chunking parameters.
- **Embedding throughput low**: Ensure GPU drivers align with `requirements.in` versions and batching flags use available VRAM efficiently.

## Ontology Download

- **Authentication errors**: Confirm BioPortal or OBO credentials via secrets manager; tokens may expire.
- **Validation crashes**: Use `--rdflib` or `--pronto` options to scope validators when debugging; re-enable full suite after issue is resolved.
- **Missing manifests**: Re-run `python -m DocsToKG.OntologyDownload.cli pull <id> --force` to regenerate metadata.

## Documentation Tooling

- `validate_docs.py` warns about missing sections: confirm required headings exist (see Style Guide).
- `check_links.py` reports timeouts: re-run with larger `--timeout` or whitelist intermittent domains.
- Sphinx builds fail on missing modules: install dependencies from `docs/build/sphinx/requirements.txt` and set `PYTHONPATH=src`.

Escalate unresolved issues in `CONTRIBUTING.md` guidance and document solutions for future playbooks.
