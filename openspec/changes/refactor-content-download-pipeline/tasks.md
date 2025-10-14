## 1. Shared Utilities Extraction

- [ ] 1.1 Create `src/DocsToKG/ContentDownload/utils.py` with exact implementation:
  ```python
  """Shared utility functions for ContentDownload module."""
  from __future__ import annotations
  import re
  from typing import List, Optional
  
  def normalize_doi(doi: Optional[str]) -> Optional[str]:
      """Normalize DOI by stripping https://doi.org/ prefix and whitespace.
      
      Examples:
          normalize_doi("https://doi.org/10.1234/abc") -> "10.1234/abc"
          normalize_doi("  10.1234/abc  ") -> "10.1234/abc"
          normalize_doi(None) -> None
      """
      if not doi:
          return None
      doi = doi.strip()
      if doi.lower().startswith("https://doi.org/"):
          doi = doi[16:]
      return doi.strip()
  
  def normalize_pmcid(pmcid: Optional[str]) -> Optional[str]:
      """Normalize PMCID ensuring PMC prefix.
      
      Examples:
          normalize_pmcid("PMC123456") -> "PMC123456"
          normalize_pmcid("123456") -> "PMC123456"
          normalize_pmcid("pmc123456") -> "PMC123456"
          normalize_pmcid(None) -> None
      """
      if not pmcid:
          return None
      pmcid = pmcid.strip()
      match = re.search(r"PMC?(\d+)", pmcid, flags=re.I)
      if match:
          return f"PMC{match.group(1)}"
      return None
  
  def strip_prefix(value: Optional[str], prefix: str) -> Optional[str]:
      """Strip case-insensitive prefix from value.
      
      Examples:
          strip_prefix("arxiv:2301.12345", "arxiv:") -> "2301.12345"
          strip_prefix("ARXIV:2301.12345", "arxiv:") -> "2301.12345"
          strip_prefix(None, "prefix") -> None
      """
      if not value:
          return None
      value = value.strip()
      if value.lower().startswith(prefix.lower()):
          return value[len(prefix):]
      return value
  
  def dedupe(items: List[str]) -> List[str]:
      """Remove duplicates while preserving first occurrence order.
      
      Examples:
          dedupe(['b', 'a', 'b', 'c']) -> ['b', 'a', 'c']
          dedupe([]) -> []
      """
      seen = set()
      result = []
      for item in items:
          if item and item not in seen:
              result.append(item)
              seen.add(item)
      return result
  ```
- [ ] 1.2 Add unit tests in `tests/test_content_download_utils.py` with minimum coverage:
  ```python
  import pytest
  from DocsToKG.ContentDownload.utils import normalize_doi, normalize_pmcid, strip_prefix, dedupe
  
  def test_normalize_doi_with_https_prefix():
      assert normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"
  
  def test_normalize_doi_without_prefix():
      assert normalize_doi("10.1234/abc") == "10.1234/abc"
  
  def test_normalize_doi_with_whitespace():
      assert normalize_doi("  10.1234/abc  ") == "10.1234/abc"
  
  def test_normalize_doi_none():
      assert normalize_doi(None) is None
  
  def test_normalize_pmcid_with_pmc_prefix():
      assert normalize_pmcid("PMC123456") == "PMC123456"
  
  def test_normalize_pmcid_without_prefix():
      assert normalize_pmcid("123456") is None  # requires PMC in input
  
  def test_normalize_pmcid_lowercase():
      assert normalize_pmcid("pmc123456") == "PMC123456"
  
  def test_strip_prefix_case_insensitive():
      assert strip_prefix("ARXIV:2301.12345", "arxiv:") == "2301.12345"
  
  def test_dedupe_preserves_order():
      assert dedupe(['b', 'a', 'b', 'c']) == ['b', 'a', 'c']
  ```
- [ ] 1.3 In `download_pyalex_pdfs.py`, delete lines 159-166 (old `_normalize_doi`) and add import at top: `from DocsToKG.ContentDownload.utils import normalize_doi, normalize_pmcid, strip_prefix, dedupe`, then replace all calls to `_normalize_doi(doi)` with `normalize_doi(doi)` (lines 287, 765, etc.)
- [ ] 1.4 In `resolvers/__init__.py`, delete lines 408-433 (old `_normalize_doi`, `_normalize_pmcid`, `_strip_prefix`) and add import at top: `from .utils import normalize_doi, normalize_pmcid, strip_prefix, dedupe`, then replace all calls (lines 454, 534, 705, 814, 856, 902, 949)
- [ ] 1.5 Update `_collect_location_urls` in `download_pyalex_pdfs.py` to use `utils.dedupe()` instead of inline dedupe logic
- [ ] 1.6 Update resolver URL collection in `resolvers/__init__.py` (Unpaywall, Crossref) to use `utils.dedupe()`

## 2. HTTP Retry Infrastructure

- [ ] 2.1 In `download_pyalex_pdfs.py`, create `_make_session(headers: Dict[str, str]) -> requests.Session` that constructs Session, mounts HTTPAdapter with Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 502, 503, 504], respect_retry_after_header=True) on http:// and https://
- [ ] 2.2 Replace `session = requests.Session()` in `main()` with `session = _make_session(config.polite_headers)`
- [ ] 2.3 Add integration test in `tests/test_download_retries.py` that mocks sequence [503, 503, 200] and verifies download succeeds after retries
- [ ] 2.4 Add test verifying Retry-After header is respected (mock 429 with Retry-After: 2, assert delay >=2s)
- [ ] 2.5 Add test verifying 4xx errors (401, 404) do NOT retry (fail immediately)

## 3. Atomic File Writes and Digests

- [ ] 3.1 In `download_candidate()`, change file write pattern: open `dest_path.with_suffix(dest_path.suffix + '.part')` instead of `dest_path` directly
- [ ] 3.2 After closing file handle, compute SHA-256 digest: `sha256 = hashlib.sha256(); with open(part_path, 'rb') as f: for chunk in iter(lambda: f.read(1<<20), b''): sha256.update(chunk); digest = sha256.hexdigest()`
- [ ] 3.3 Get file size: `content_length = part_path.stat().st_size`
- [ ] 3.4 Perform atomic rename: `os.replace(part_path, dest_path)`
- [ ] 3.5 Add `sha256` and `content_length` fields to `DownloadOutcome` dataclass
- [ ] 3.6 Update `DownloadOutcome` return statement to include `sha256=digest, content_length=content_length`
- [ ] 3.7 Add test in `tests/test_atomic_writes.py` that kills download mid-stream (using signal or mock exception), verifies .part file exists and final file does NOT exist
- [ ] 3.8 Add test verifying SHA-256 digest matches expected value for known test file

## 4. Rate Limit Configuration Clarity

- [ ] 4.1 In `resolvers/__init__.py` ResolverConfig dataclass, rename field: `resolver_rate_limits: Dict[str, float]` → `resolver_min_interval_s: Dict[str, float]`
- [ ] 4.2 Update `ResolverPipeline._respect_rate_limit()` to read `config.resolver_min_interval_s.get(resolver_name)`
- [ ] 4.3 In `load_resolver_config()` in `download_pyalex_pdfs.py`, add deprecation handling: if `resolver_rate_limits` in config_data and `resolver_min_interval_s` not in config_data, copy value and log warning "resolver_rate_limits deprecated, use resolver_min_interval_s"
- [ ] 4.4 Update `apply_config_overrides()` to handle both field names (old and new) for backward compatibility
- [ ] 4.5 Update example resolver config YAML in docs/ to use `resolver_min_interval_s` with inline comment "# seconds between calls (e.g., 1.0 = max 1 QPS)"
- [ ] 4.6 Update comment in `load_resolver_config()` where Unpaywall default is set: `# Unpaywall recommends 1 request per second`
- [ ] 4.7 Add test verifying deprecation warning is logged when old field name used

## 5. LRU Cache for Resolver APIs

- [ ] 5.1 In `resolvers/__init__.py`, add at module level: `from functools import lru_cache`
- [ ] 5.2 Create cached helper for Unpaywall: `@lru_cache(maxsize=1000)` decorator on new function `_fetch_unpaywall_data(doi: str, email: str, timeout: float, headers: Dict) -> Optional[Dict]` that makes API call and returns JSON
- [ ] 5.3 Update `UnpaywallResolver.iter_urls()` to call `_fetch_unpaywall_data(doi, config.unpaywall_email, config.get_timeout(self.name), ...)`
- [ ] 5.4 Create cached helper for Crossref: `@lru_cache(maxsize=1000)` decorator on `_fetch_crossref_data(doi: str, mailto: Optional[str], timeout: float, headers: Dict) -> Optional[Dict]`
- [ ] 5.5 Update `CrossrefResolver.iter_urls()` to call `_fetch_crossref_data(...)`
- [ ] 5.6 Create cached helper for Semantic Scholar: `@lru_cache(maxsize=1000)` decorator on `_fetch_s2_data(doi: str, api_key: Optional[str], timeout: float, headers: Dict) -> Optional[Dict]`
- [ ] 5.7 Update `SemanticScholarResolver.iter_urls()` to call `_fetch_s2_data(...)`
- [ ] 5.8 Add `clear_resolver_caches()` function that calls `.cache_clear()` on all three cached functions
- [ ] 5.9 In `main()`, call `clear_resolver_caches()` if `--resume-from` flag is set (to force fresh lookups)
- [ ] 5.10 Add test in `tests/test_resolver_caching.py` that calls resolver twice with same DOI, verifies second call does not make HTTP request (mock session)

## 6. Conditional Requests Support

- [ ] 6.1 In `Manifest` dataclass in `download_pyalex_pdfs.py`, add fields: `etag: Optional[str] = None`, `last_modified: Optional[str] = None`
- [ ] 6.2 Update manifest.jsonl write to include etag and last_modified from FetchResult
- [ ] 6.3 In `download_candidate()`, add parameter `previous_etag: Optional[str] = None`, `previous_last_modified: Optional[str] = None`
- [ ] 6.4 Before making GET request in `download_candidate()`, add to headers dict: if `previous_etag`: `headers['If-None-Match'] = previous_etag`; if `previous_last_modified`: `headers['If-Modified-Since'] = previous_last_modified`
- [ ] 6.5 After receiving response, if `response.status_code == 304`: return `DownloadOutcome(classification='cached', path=str(existing_path), http_status=304, ...)`
- [ ] 6.6 In `main()`, before calling `download_candidate()`, read manifest.jsonl, lookup (work_id, url) to get previous etag/last_modified
- [ ] 6.7 Pass etag/last_modified to `download_candidate()` call
- [ ] 6.8 Add test in `tests/test_conditional_requests.py` that mocks 304 response, verifies cached outcome returned and no file written
- [ ] 6.9 Add test verifying ETag/Last-Modified recorded in manifest after successful download

## 7. Unified JSONL Logging

- [ ] 7.1 Create `JsonlLogger` class in `download_pyalex_pdfs.py` with methods: `log_attempt(record: AttemptRecord)`, `log_summary(summary: Dict)`
- [ ] 7.2 Implement `log_attempt()` to write JSON line with schema: `{"timestamp": ISO8601, "record_type": "attempt", "work_id": ..., "resolver_name": ..., ...}`
- [ ] 7.3 Implement `log_summary()` to write JSON line with schema: `{"timestamp": ISO8601, "record_type": "summary", "total_works": ..., "success_count": ..., ...}`
- [ ] 7.4 Replace `CsvAttemptLogger` instantiation in `main()` with `JsonlLogger(args.log_jsonl or (pdf_dir / "attempts.jsonl"))`
- [ ] 7.5 Replace `ManifestLogger` instantiation with same `JsonlLogger` instance (unify)
- [ ] 7.6 Update `AttemptRecord` logging calls to add `sha256` and `content_length` fields
- [ ] 7.7 Create CSV export script `scripts/export_attempts_csv.py` that reads JSONL and writes CSV with same columns as old CsvAttemptLogger
- [ ] 7.8 Add CLI flag `--log-format [jsonl|csv]` (default: jsonl); if csv, wrap JsonlLogger with CSV adapter
- [ ] 7.9 Add test in `tests/test_jsonl_logging.py` verifying JSONL lines are valid JSON and contain required fields
- [ ] 7.10 Add test verifying CSV export script produces correct columns and values

## 8. Refactor Download State Machine

- [ ] 8.1 In `download_candidate()`, add `from enum import Enum` and define: `class DownloadState(Enum): PENDING = "pending"; WRITING = "writing"`
- [ ] 8.2 Replace `detected: Optional[str] = None` with `state = DownloadState.PENDING; detected: Optional[str] = None`
- [ ] 8.3 Refactor chunk iteration loop: `if state == DownloadState.PENDING: ... if detected: state = DownloadState.WRITING; elif state == DownloadState.WRITING: ...`
- [ ] 8.4 Extract outcome building to helper function: `def _build_download_outcome(detected: Optional[str], dest_path: Optional[Path], response: requests.Response, elapsed_ms: float, flagged_unknown: bool, sha256: str, content_length: int) -> DownloadOutcome: ...`
- [ ] 8.5 Replace multiple `return DownloadOutcome(...)` statements with calls to `_build_download_outcome(...)`
- [ ] 8.6 Add docstring to `_build_download_outcome()` explaining classification logic and EOF check
- [ ] 8.7 Run existing test suite (`tests/test_download_pyalex_pdfs.py`, `tests/test_resolvers.py`) to verify no regressions

## 9. Parallel Execution with ThreadPoolExecutor

- [ ] 9.1 Add CLI argument: `parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1 for sequential)')`
- [ ] 9.2 In `main()`, extract work processing into function: `def process_one_work(work: Dict, session: requests.Session, pdf_dir: Path, html_dir: Path, pipeline: ResolverPipeline, logger: JsonlLogger, metrics: ResolverMetrics) -> Dict: ...`
- [ ] 9.3 Refactor main loop: if `args.workers == 1`: keep existing sequential loop; else: use `with ThreadPoolExecutor(max_workers=args.workers) as executor: ...`
- [ ] 9.4 Create session factory: `def _make_session_for_worker() -> requests.Session: return _make_session(config.polite_headers)`
- [ ] 9.5 Submit work to executor: `futures = {executor.submit(process_one_work, work, _make_session_for_worker(), ...): work for work in batch}`
- [ ] 9.6 Add `threading.Lock` to `ResolverPipeline._last_invocation` dict access in `_respect_rate_limit()` to make thread-safe
- [ ] 9.7 Update `ResolverPipeline.__init__()` to initialize: `self._lock = threading.Lock()`
- [ ] 9.8 Wrap `self._last_invocation[resolver_name] = time.monotonic()` in `with self._lock:` block
- [ ] 9.9 Add test in `tests/test_parallel_execution.py` with --workers=3, verify rate limiting still enforced (mock time, assert minimum intervals)
- [ ] 9.10 Document in README: "Use --workers=3-5 for production; higher values may violate rate limits"

## 10. Dry Run and Resume Modes

- [ ] 10.1 Add CLI flag: `parser.add_argument('--dry-run', action='store_true', help='Measure resolver coverage without writing files')`
- [ ] 10.2 Add CLI flag: `parser.add_argument('--resume-from', type=Path, default=None, help='Resume from manifest.jsonl, only process missed works')`
- [ ] 10.3 In `main()`, if `args.resume_from`: read manifest.jsonl, collect set of work_ids with status 'success' or 'cached'
- [ ] 10.4 In work iteration loop, if `args.resume_from` and `work_id in completed_work_ids`: skip with log message "Skipping work_id (already completed)"
- [ ] 10.5 If `args.dry_run`: in `download_candidate()`, after classification, return immediately without writing file (but still compute metrics)
- [ ] 10.6 Update logger to include `dry_run: bool` field in records
- [ ] 10.7 In final summary, if `args.dry_run`: print "DRY RUN: no files written, resolver coverage: ..."
- [ ] 10.8 Add test in `tests/test_dry_run.py` verifying no files written when --dry-run, but logs generated
- [ ] 10.9 Add test in `tests/test_resume.py` verifying works in manifest are skipped on --resume-from

## 11. HTML Text Extraction

- [ ] 11.1 Add CLI flag: `parser.add_argument('--extract-html-text', action='store_true', help='Extract plaintext from HTML fallbacks (requires trafilatura)')`
- [ ] 11.2 In `download_candidate()`, after writing HTML file, if `extract_html_text` flag: `try: import trafilatura; text = trafilatura.extract(html_content); except ImportError: log warning`
- [ ] 11.3 If extraction succeeds, write text to `dest_path.with_suffix('.html.txt')`
- [ ] 11.4 Add `extracted_text_path` field to DownloadOutcome
- [ ] 11.5 Update manifest to include extracted_text_path
- [ ] 11.6 Add trafilatura to optional dependencies in pyproject.toml: `[tool.poetry.group.extract]`
- [ ] 11.7 Document in README: "For HTML text extraction: pip install trafilatura && use --extract-html-text"
- [ ] 11.8 Add test in `tests/test_html_extraction.py` (requires trafilatura) verifying .html.txt created with plaintext

## 12. Additional Resolvers

- [ ] 12.1 In `resolvers/__init__.py`, create `OpenAireResolver` class following pattern: `name = "openaire"`, `is_enabled()` checks DOI, `iter_urls()` calls <https://api.openaire.eu/search/publications?doi=>... and parses bestlicense/instances
- [ ] 12.2 Create `HalResolver` class: `name = "hal"`, checks DOI, calls <https://api.archives-ouvertes.fr/search/?q=doiId_s>:... and parses file_s field
- [ ] 12.3 Create `OsfResolver` class: `name = "osf"`, checks DOI, calls <https://api.osf.io/v2/preprints/?filter[doi]=>... and parses links.download
- [ ] 12.4 Add all three to `default_resolvers()` list
- [ ] 12.5 Set all three to disabled by default in DEFAULT_RESOLVER_ORDER config: `resolver_toggles = {"openaire": False, "hal": False, "osf": False, ...}`
- [ ] 12.6 Document in README: "Enable additional resolvers for EU OA / preprints: --enable-resolver openaire"
- [ ] 12.7 Add unit tests for each resolver mocking API responses

## 13. Enhanced User-Agent for Crossref

- [ ] 13.1 In `load_resolver_config()` where User-Agent is constructed, update format: `user_agent = f"DocsToKGDownloader/1.0 (+{config.mailto}; mailto:{config.mailto})"`
- [ ] 13.2 Ensure mailto is included directly in UA string even if polite_headers already has separate mailto header
- [ ] 13.3 Add test verifying User-Agent includes mailto in correct format

## 14. Testing Gaps

- [ ] 14.1 Add test in `tests/test_edge_cases.py` for corrupt HTML with Content-Type: application/pdf, verify sniff overrides Content-Type
- [ ] 14.2 Add test for Wayback resolver when `archived_snapshots.closest.available == False`, verify skip
- [ ] 14.3 Add test verifying manifest and attempts have exactly one success row per saved PDF with matching work_id and path
- [ ] 14.4 Add test for polite headers propagation: verify OpenAlex candidate attempts use same headers as pipeline
- [ ] 14.5 Add test for retry budget exhaustion: mock 10 URLs all returning 503, verify max_attempts_per_work honored

## 15. Documentation and Migration

- [ ] 15.1 Create MIGRATION.md in docs/ with sections: Config Changes (resolver_rate_limits → resolver_min_interval_s), Logging Changes (CSV → JSONL with export script), CLI Additions (--workers, --dry-run, --resume-from, --extract-html-text)
- [ ] 15.2 Update README.md with new CLI flags examples and recommended --workers values
- [ ] 15.3 Document rate limit semantics: "resolver_min_interval_s: 1.0 means minimum 1 second between calls (max ~1 QPS)"
- [ ] 15.4 Add troubleshooting section for common issues: partial files (check .part), rate limit violations (reduce --workers), memory issues (reduce --workers)
- [ ] 15.5 Document CSV export: "To convert JSONL to CSV: jq -r '[.timestamp, .work_id, ...] | @csv' attempts.jsonl > attempts.csv" or use scripts/export_attempts_csv.py
- [ ] 15.6 Update docstrings in `download_candidate()`, `_make_session()`, `process_one_work()` with parameter descriptions and examples
- [ ] 15.7 Add example resolver config YAML in docs/examples/ showing all new fields

## 16. End-to-End Validation

- [ ] 16.1 Run smoke test: 100 works from OpenAlex, --workers=3, verify success rate >=90%, no .part files left, all digests valid
- [ ] 16.2 Run smoke test with --dry-run, verify no files written but logs complete
- [ ] 16.3 Run smoke test with --resume-from, verify skipped works not re-attempted
- [ ] 16.4 Run full test suite: `pytest tests/ -v --cov=src/DocsToKG/ContentDownload --cov-report=html`
- [ ] 16.5 Verify coverage >=85% for ContentDownload modules
- [ ] 16.6 Run linter: `ruff check src/DocsToKG/ContentDownload/` and fix any issues
- [ ] 16.7 Run type checker: `mypy src/DocsToKG/ContentDownload/` and resolve any type errors
- [ ] 16.8 Performance benchmark: compare single-threaded vs --workers=5 throughput on 500 works, document speedup
- [ ] 16.9 Verify atomic writes: kill process mid-download (SIGKILL), verify no corrupt files, only .part files
- [ ] 16.10 Verify rate limiting under parallel load: --workers=10, monitor actual request rates per resolver, assert <=1/min_interval_s
