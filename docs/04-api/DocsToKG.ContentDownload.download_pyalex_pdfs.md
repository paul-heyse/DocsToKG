# Module: download_pyalex_pdfs

Download PDFs for OpenAlex works with a configurable resolver stack.

## Functions

### `_has_pdf_eof(path)`

*No documentation available.*

### `slugify(text, keep)`

*No documentation available.*

### `ensure_dir(path)`

*No documentation available.*

### `_make_session(headers)`

Create a retry-enabled :class:`requests.Session` with polite headers.

The session mounts an :class:`urllib3.util.retry.Retry` adapter that retries
transient HTTP failures (429/502/503/504) up to five times with
exponential backoff and `Retry-After` support. Callers should pass the
polite header set returned by :func:`load_resolver_config` so that each
worker advertises the required contact information.

### `_make_session_for_worker(headers)`

Factory helper for per-worker sessions.

### `load_previous_manifest(path)`

Load manifest JSONL entries indexed by work identifier.

### `build_manifest_entry(artifact, resolver, url, outcome, html_paths)`

*No documentation available.*

### `classify_payload(head_bytes, content_type, url)`

Return 'pdf', 'html', or None if undecided.

### `_build_download_outcome()`

Create a :class:`DownloadOutcome` applying PDF validation rules.

The helper normalises classification labels, performs the terminal ``%%EOF``
check for PDFs (skipping when running in ``--dry-run`` mode), and attaches
bookkeeping metadata such as digests and conditional request headers.

### `_normalize_pmid(pmid)`

*No documentation available.*

### `_normalize_arxiv(arxiv_id)`

*No documentation available.*

### `_collect_location_urls(work)`

*No documentation available.*

### `build_query(args)`

*No documentation available.*

### `resolve_topic_id_if_needed(topic_text)`

*No documentation available.*

### `create_artifact(work, pdf_dir, html_dir)`

*No documentation available.*

### `download_candidate(session, artifact, url, referer, timeout, context)`

*No documentation available.*

### `read_resolver_config(path)`

*No documentation available.*

### `apply_config_overrides(config, data, resolver_names)`

*No documentation available.*

### `load_resolver_config(args, resolver_names, resolver_order_override)`

*No documentation available.*

### `iterate_openalex(query, per_page, max_results)`

*No documentation available.*

### `attempt_openalex_candidates(session, artifact, logger, metrics, context)`

*No documentation available.*

### `process_one_work(work, session, pdf_dir, html_dir, pipeline, logger, metrics)`

*No documentation available.*

### `_write(self, payload)`

*No documentation available.*

### `log_attempt(self, record)`

*No documentation available.*

### `log(self, record)`

*No documentation available.*

### `log_manifest(self, entry)`

*No documentation available.*

### `log_summary(self, summary)`

*No documentation available.*

### `close(self)`

*No documentation available.*

### `log_attempt(self, record)`

*No documentation available.*

### `log_manifest(self, entry)`

*No documentation available.*

### `log_summary(self, summary)`

*No documentation available.*

### `log(self, record)`

*No documentation available.*

### `close(self)`

*No documentation available.*

### `__post_init__(self)`

*No documentation available.*

### `append_location(loc)`

*No documentation available.*

### `main()`

*No documentation available.*

### `record_result(res)`

*No documentation available.*

### `submit_work(work_item)`

*No documentation available.*

### `runner()`

*No documentation available.*

## Classes

### `ManifestEntry`

*No documentation available.*

### `JsonlLogger`

Structured logger that emits attempt, manifest, and summary JSONL records.

### `CsvAttemptLoggerAdapter`

Adapter that mirrors attempt records to CSV for backward compatibility.

### `WorkArtifact`

*No documentation available.*

### `DownloadState`

*No documentation available.*
