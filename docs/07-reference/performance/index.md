# 1. Performance Optimization

Optimising DocsToKG involves tuning the retrieval stack, document processing throughput,
and ontology ingest latency.

## 2. Hybrid Search

- **Dense index parameters**: Adjust `DenseIndexConfig` (see
  `DocsToKG.HybridSearch.config`) for `nlist`, `nprobe`, and PQ settings. Use the FAISS
  scaffold to evaluate recall/latency trade-offs before rolling updates.
- **Fusion weights**: Tweak `FusionConfig.k0` and `FusionConfig.mmr_lambda` to balance lexical vs dense dominance. Validate with `tests/test_hybrid_search.py`.
- **Oversampling**: Increase `DenseIndexConfig.oversample` when accuracy drifts; monitor GPU memory impact via observability metrics.

## 3. Document Parsing

- **Batching**: Use the `--batch-size` flags on DocParsing pipelines to balance throughput and memory consumption.
- **vLLM workers**: Enable multi-worker mode when running large-scale extraction with GPU acceleration.
- **Caching**: Cache intermediate DocTags outputs to avoid repeated parsing of static corpora.

## 4. Ontology Downloads

- **Concurrency**: Configure HTTP rate limits and retry policies within `sources.yaml` to avoid provider throttling.
- **Validation scope**: Skip heavyweight validators (`--robot`, `--arelle`) when quick iteration is required, re-enabling them before production publication.

## 5. Monitoring Signals

Track the following metrics via `DocsToKG.HybridSearch.observability` or external dashboards:

- Query latency percentiles by namespace
- Recall@k on benchmark datasets
- FAISS index load/restore durations
- Ontology download success rates and average transfer time

Use `docs/hybrid_search_runbook.md` for remediation actions when metrics breach guardrails.

## 6. Content Download Cache Validation

- The downloader now skips SHA-256 recomputation when cached files match recorded size and mtime. This keeps 304 validations effectively free even for 100 MB artifacts.
- When regulatory or audit workflows require full digests, enable `--verify-cache-digest` (or set `DownloadOptions.verify_cache_digest=True`). Expect ~55 ms per 100 MB artifact while the in-process LRU prevents repeat hashing within a run.
- Remember to reset the flag once investigations conclude; always-on digest verification materially increases CPU and I/O on large corpora.
