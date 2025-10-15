# Performance Optimization

Optimising DocsToKG involves tuning the retrieval stack, document processing throughput, and ontology ingest latency.

## Hybrid Search

- **Dense index parameters**: Adjust `DenseIndexConfig` (see `DocsToKG.HybridSearch.config`) for `nlist`, `nprobe`, and PQ settings. Use the FAISS scaffold to evaluate recall/latency trade-offs before rolling updates.
- **Fusion weights**: Tweak `FusionConfig.k0` and `FusionConfig.mmr_lambda` to balance lexical vs dense dominance. Validate with `tests/test_hybrid_search.py`.
- **Oversampling**: Increase `DenseIndexConfig.oversample` when accuracy drifts; monitor GPU memory impact via observability metrics.

## Document Parsing

- **Batching**: Use the `--batch-size` flags on DocParsing pipelines to balance throughput and memory consumption.
- **vLLM workers**: Enable multi-worker mode when running large-scale extraction with GPU acceleration.
- **Caching**: Cache intermediate DocTags outputs to avoid repeated parsing of static corpora.

## Ontology Downloads

- **Concurrency**: Configure HTTP rate limits and retry policies within `sources.yaml` to avoid provider throttling.
- **Validation scope**: Skip heavyweight validators (`--robot`, `--arelle`) when quick iteration is required, re-enabling them before production publication.

## Monitoring Signals

Track the following metrics via `DocsToKG.HybridSearch.observability` or external dashboards:

- Query latency percentiles by namespace
- Recall@k on benchmark datasets
- FAISS index load/restore durations
- Ontology download success rates and average transfer time

Use `docs/hybrid_search_runbook.md` for remediation actions when metrics breach guardrails.
