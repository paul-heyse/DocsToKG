## 1. HTTP Retry Infrastructure

- [ ] 1.1 Create `src/DocsToKG/ContentDownload/http.py` module skeleton
  - Import `requests`, `time`, `random`, `Optional`, `Set`, `Any` from typing
  - Add module docstring: "Unified HTTP request utilities with retry and backoff support."

- [ ] 1.2 Implement `parse_retry_after_header(response: requests.Response) -> Optional[float]` function
  - Parse `Retry-After` header returning seconds as float
  - Handle both integer (delta-seconds) and HTTP-date formats
  - Return `None` if header missing or unparseable
  - Add docstring with examples

- [ ] 1.3 Implement `request_with_retries()` function with signature:
  ```python
  def request_with_retries(
      session: requests.Session,
      method: str,
      url: str,
      *,
      max_retries: int = 3,
      retry_statuses: Optional[Set[int]] = None,
      backoff_factor: float = 0.75,
      respect_retry_after: bool = True,
      **kwargs: Any,
  ) -> requests.Response:
  ```

- [ ] 1.4 Implement retry loop logic in `request_with_retries()`
  - Default `retry_statuses` to `{429, 500, 502, 503, 504}` if None
  - Attempt request up to `max_retries + 1` times
  - On retry-eligible status code, parse `Retry-After` if `respect_retry_after=True`
  - Use exponential backoff: `backoff_factor * (2 ** attempt_num) + random.random() * 0.1`
  - Override backoff with `Retry-After` value when present and larger
  - Re-raise `requests.RequestException` on final retry exhaustion

- [ ] 1.5 Add comprehensive docstring to `request_with_retries()`
  - Document all parameters with types and defaults
  - Document return type and exceptions raised
  - Add usage example for HEAD and GET requests
  - Note thread-safety guarantees (session must be thread-safe)

- [ ] 1.6 Update `src/DocsToKG/ContentDownload/download_pyalex_pdfs.py` imports
  - Add: `from DocsToKG.ContentDownload.http import request_with_retries`
  - Replace `session.head(url, ...)` call at line ~970 with `request_with_retries(session, "HEAD", url, max_retries=1, ...)`
  - Replace `session.get(url, stream=True, ...)` call at line ~980 with `request_with_retries(session, "GET", url, ...)`
  - Remove HEAD error suppression `contextlib.suppress` since retries handle transients

- [ ] 1.7 Update `src/DocsToKG/ContentDownload/resolvers/__init__.py` to use new utility
  - Replace `_request_with_retries()` calls (lines ~1199, 1344, 1404, 1482, etc.) with `from DocsToKG.ContentDownload.http import request_with_retries`
  - Remove local `_request_with_retries()` definition (lines ~158-196)
  - Update `_sleep_backoff()` references to use backoff logic from `http.request_with_retries()`

- [ ] 1.8 Add unit tests in `tests/test_http_retry.py`
  - Test successful request (no retries)
  - Test transient 503 with exponential backoff (verify timing)
  - Test `Retry-After` header compliance (integer and date formats)
  - Test exhausted retries raising exception
  - Test custom `retry_statuses` parameter
  - Test `respect_retry_after=False` ignoring header

## 2. Conditional Request Helper

- [ ] 2.1 Create `src/DocsToKG/ContentDownload/conditional.py` module skeleton
  - Import `dataclass`, `Optional`, `Dict`, `requests`
  - Add module docstring: "Conditional HTTP request helpers for ETag and Last-Modified caching."

- [ ] 2.2 Define `CachedResult` dataclass
  ```python
  @dataclass
  class CachedResult:
      """Represents HTTP 304 Not Modified response with prior metadata."""
      path: str
      sha256: str
      content_length: int
      etag: Optional[str]
      last_modified: Optional[str]
  ```

- [ ] 2.3 Define `ModifiedResult` dataclass
  ```python
  @dataclass
  class ModifiedResult:
      """Represents HTTP 200 response requiring fresh download."""
      etag: Optional[str]
      last_modified: Optional[str]
  ```

- [ ] 2.4 Implement `ConditionalRequestHelper` class initialization
  ```python
  class ConditionalRequestHelper:
      def __init__(
          self,
          prior_etag: Optional[str] = None,
          prior_last_modified: Optional[str] = None,
          prior_sha256: Optional[str] = None,
          prior_content_length: Optional[int] = None,
          prior_path: Optional[str] = None,
      ):
          self.prior_etag = prior_etag
          self.prior_last_modified = prior_last_modified
          self.prior_sha256 = prior_sha256
          self.prior_content_length = prior_content_length
          self.prior_path = prior_path
  ```

- [ ] 2.5 Implement `build_headers()` method
  ```python
  def build_headers(self) -> Dict[str, str]:
      """Generate conditional request headers from prior metadata."""
      headers = {}
      if self.prior_etag:
          headers["If-None-Match"] = self.prior_etag
      if self.prior_last_modified:
          headers["If-Modified-Since"] = self.prior_last_modified
      return headers
  ```

- [ ] 2.6 Implement `interpret_response()` method
  ```python
  def interpret_response(
      self, response: requests.Response
  ) -> Union[CachedResult, ModifiedResult]:
      """Interpret response status and headers as cached or modified result."""
      if response.status_code == 304:
          if not all([self.prior_path, self.prior_sha256, self.prior_content_length]):
              raise ValueError("304 response requires complete prior metadata")
          return CachedResult(
              path=self.prior_path,
              sha256=self.prior_sha256,
              content_length=self.prior_content_length,
              etag=self.prior_etag,
              last_modified=self.prior_last_modified,
          )
      return ModifiedResult(
          etag=response.headers.get("ETag"),
          last_modified=response.headers.get("Last-Modified"),
      )
  ```

- [ ] 2.7 Update `download_candidate()` in `download_pyalex_pdfs.py` to use helper
  - Import `ConditionalRequestHelper`, `CachedResult`, `ModifiedResult`
  - At line ~950, create helper instance: `cond_helper = ConditionalRequestHelper(prior_etag=previous_etag, ...)`
  - Replace manual header building (lines ~958-961) with `headers.update(cond_helper.build_headers())`
  - Replace 304 handling block (lines ~988-999) with: `result = cond_helper.interpret_response(response); if isinstance(result, CachedResult): return DownloadOutcome(...)`

- [ ] 2.8 Add unit tests in `tests/test_conditional_requests.py`
  - Test `build_headers()` with no prior data (empty dict)
  - Test `build_headers()` with ETag only
  - Test `build_headers()` with Last-Modified only
  - Test `build_headers()` with both headers
  - Test `interpret_response()` with 304 status returning `CachedResult`
  - Test `interpret_response()` with 200 status returning `ModifiedResult`
  - Test `interpret_response()` with 304 but missing prior metadata raising ValueError
  - Test header extraction from 200 response (ETag and Last-Modified present)

## 3. Resolver Module Restructuring - Foundation

- [ ] 3.1 Create directory structure
  - `mkdir -p src/DocsToKG/ContentDownload/resolvers/providers`
  - `touch src/DocsToKG/ContentDownload/resolvers/providers/__init__.py`

- [ ] 3.2 Create `src/DocsToKG/ContentDownload/resolvers/types.py`
  - Copy dataclass definitions from `resolvers/__init__.py`:
    - `ResolverResult` (lines ~198-232)
    - `ResolverConfig` (lines ~234-302)
    - `AttemptRecord` (lines ~304-348)
    - `DownloadOutcome` (lines ~378-422)
    - `PipelineResult` (lines ~424-444)
    - `ResolverMetrics` (lines ~501-571)
  - Copy `Resolver` protocol (lines ~447-498)
  - Copy `AttemptLogger` protocol (lines ~350-376)
  - Copy `DownloadFunc` type alias (line ~574)
  - Add imports: `dataclass`, `field`, `Protocol`, `Dict`, `List`, `Optional`, `Any`, `Counter`, `defaultdict`
  - Add module docstring: "Type definitions and protocols for the resolver pipeline."

- [ ] 3.3 Create `src/DocsToKG/ContentDownload/resolvers/pipeline.py`
  - Copy `ResolverPipeline` class (lines ~594-840)
  - Copy helper functions: `_callable_accepts_argument` (lines ~577-591)
  - Import types from `.types`: `ResolverConfig`, `AttemptRecord`, `DownloadOutcome`, `PipelineResult`, `ResolverResult`, `Resolver`, `AttemptLogger`, `DownloadFunc`, `ResolverMetrics`
  - Import standard library: `threading`, `time`, `random`, `defaultdict`, `Dict`, `List`, `Optional`, `Any`, `Set`, `Sequence`
  - Import `requests`
  - Add module docstring: "Resolver pipeline orchestration and execution logic."

- [ ] 3.4 Update `resolvers/pipeline.py` imports for new structure
  - Change `from DocsToKG.ContentDownload.utils import ...` to relative import if needed
  - Ensure all type references point to `types` module

- [ ] 3.5 Create `src/DocsToKG/ContentDownload/resolvers/providers/__init__.py` with registry
  ```python
  """Resolver provider implementations and registry."""
  from typing import List
  from ..types import Resolver
  
  def default_resolvers() -> List[Resolver]:
      """Return default resolver instances in priority order."""
      # Will be populated in subsequent tasks
      return []
  
  __all__ = ["default_resolvers"]
  ```

## 4. Resolver Module Restructuring - Individual Providers

- [ ] 4.1 Create `src/DocsToKG/ContentDownload/resolvers/providers/unpaywall.py`
  - Copy `UnpaywallResolver` class (lines ~851-988 from `resolvers/__init__.py`)
  - Copy `_fetch_unpaywall_data` helper and LRU cache (lines ~85-101)
  - Copy `_headers_cache_key` helper (lines ~81-82)
  - Import necessary types from `..types`: `Resolver`, `ResolverConfig`, `ResolverResult`
  - Import utilities: `normalize_doi`, `dedupe` from `DocsToKG.ContentDownload.utils`
  - Import `requests`, `lru_cache`, `quote`, `Iterable`, `Dict`, `List`, `Tuple`, `Any`
  - Add module docstring: "Unpaywall API resolver for open access PDFs."

- [ ] 4.2 Create `src/DocsToKG/ContentDownload/resolvers/providers/crossref.py`
  - Copy `CrossrefResolver` class (lines ~990-1145)
  - Copy `_fetch_crossref_data` helper and LRU cache (lines ~104-121)
  - Import necessary types and utilities
  - Add module docstring: "Crossref API resolver for publisher-hosted PDFs."

- [ ] 4.3 Create `src/DocsToKG/ContentDownload/resolvers/providers/landing_page.py`
  - Copy `LandingPageResolver` class (lines ~1148-1256)
  - Copy `_absolute_url` helper (lines ~844-848)
  - Import optional `BeautifulSoup` dependency with try/except
  - Import necessary types and utilities
  - Add module docstring: "Landing page scraper resolver using BeautifulSoup."

- [ ] 4.4 Create `src/DocsToKG/ContentDownload/resolvers/providers/arxiv.py`
  - Copy `ArxivResolver` class (lines ~1258-1309)
  - Import `strip_prefix` from utils
  - Add module docstring: "arXiv preprint resolver."

- [ ] 4.5 Create `src/DocsToKG/ContentDownload/resolvers/providers/pmc.py`
  - Copy `PmcResolver` class (lines ~1312-1433)
  - Import necessary utilities including `normalize_pmcid`, `normalize_doi`, `dedupe`
  - Add module docstring: "PubMed Central resolver using NCBI utilities."

- [ ] 4.6 Create `src/DocsToKG/ContentDownload/resolvers/providers/europe_pmc.py`
  - Copy `EuropePmcResolver` class (lines ~1436-1504)
  - Add module docstring: "Europe PMC resolver for European open access articles."

- [ ] 4.7 Create `src/DocsToKG/ContentDownload/resolvers/providers/core.py`
  - Copy `CoreResolver` class (lines ~1507-1581)
  - Add module docstring: "CORE API resolver for aggregated open access content."

- [ ] 4.8 Create `src/DocsToKG/ContentDownload/resolvers/providers/doaj.py`
  - Copy `DoajResolver` class (lines ~1584-1658)
  - Add module docstring: "DOAJ (Directory of Open Access Journals) resolver."

- [ ] 4.9 Create `src/DocsToKG/ContentDownload/resolvers/providers/semantic_scholar.py`
  - Copy `SemanticScholarResolver` class (lines ~1661-1721)
  - Copy `_fetch_semantic_scholar_data` helper and LRU cache (lines ~124-143)
  - Add module docstring: "Semantic Scholar Graph API resolver."

- [ ] 4.10 Create `src/DocsToKG/ContentDownload/resolvers/providers/openaire.py`
  - Copy `OpenAireResolver` class (lines ~1724-1793)
  - Copy `_collect_candidate_urls` helper (lines ~146-155)
  - Add module docstring: "OpenAIRE research infrastructure resolver."

- [ ] 4.11 Create `src/DocsToKG/ContentDownload/resolvers/providers/hal.py`
  - Copy `HalResolver` class (lines ~1796-1873)
  - Add module docstring: "HAL (Hyper Articles en Ligne) open archive resolver."

- [ ] 4.12 Create `src/DocsToKG/ContentDownload/resolvers/providers/osf.py`
  - Copy `OsfResolver` class (lines ~1876-1954)
  - Add module docstring: "Open Science Framework preprints resolver."

- [ ] 4.13 Create `src/DocsToKG/ContentDownload/resolvers/providers/wayback.py`
  - Copy `WaybackResolver` class (lines ~1957-2022)
  - Add module docstring: "Internet Archive Wayback Machine fallback resolver."

- [ ] 4.14 Update `resolvers/providers/__init__.py` with complete registry
  ```python
  from .unpaywall import UnpaywallResolver
  from .crossref import CrossrefResolver
  from .landing_page import LandingPageResolver
  from .arxiv import ArxivResolver
  from .pmc import PmcResolver
  from .europe_pmc import EuropePmcResolver
  from .core import CoreResolver
  from .doaj import DoajResolver
  from .semantic_scholar import SemanticScholarResolver
  from .openaire import OpenAireResolver
  from .hal import HalResolver
  from .osf import OsfResolver
  from .wayback import WaybackResolver
  
  def default_resolvers():
      return [
          UnpaywallResolver(),
          CrossrefResolver(),
          LandingPageResolver(),
          ArxivResolver(),
          PmcResolver(),
          EuropePmcResolver(),
          CoreResolver(),
          DoajResolver(),
          SemanticScholarResolver(),
          OpenAireResolver(),
          HalResolver(),
          OsfResolver(),
          WaybackResolver(),
      ]
  ```

- [ ] 4.15 Create `src/DocsToKG/ContentDownload/resolvers/__init__.py` with backward-compatible exports
  ```python
  """Resolver pipeline and provider implementations.
  
  This module maintains backward compatibility by re-exporting all public APIs.
  New code should import from submodules (pipeline, types, providers) directly.
  """
  
  from .types import (
      AttemptLogger,
      AttemptRecord,
      DownloadOutcome,
      PipelineResult,
      Resolver,
      ResolverConfig,
      ResolverMetrics,
      ResolverResult,
  )
  from .pipeline import ResolverPipeline
  from .providers import default_resolvers
  
  # Preserve legacy exports
  DEFAULT_RESOLVER_ORDER = [
      "unpaywall", "crossref", "landing_page", "arxiv", "pmc",
      "europe_pmc", "core", "doaj", "semantic_scholar",
      "openaire", "hal", "osf", "wayback",
  ]
  
  def clear_resolver_caches():
      """Clear resolver-level LRU caches."""
      from .providers.unpaywall import _fetch_unpaywall_data
      from .providers.crossref import _fetch_crossref_data
      from .providers.semantic_scholar import _fetch_semantic_scholar_data
      _fetch_unpaywall_data.cache_clear()
      _fetch_crossref_data.cache_clear()
      _fetch_semantic_scholar_data.cache_clear()
  
  __all__ = [
      "AttemptRecord", "AttemptLogger", "DownloadOutcome",
      "PipelineResult", "Resolver", "ResolverConfig",
      "ResolverPipeline", "ResolverResult", "ResolverMetrics",
      "default_resolvers", "DEFAULT_RESOLVER_ORDER",
      "clear_resolver_caches",
  ]
  ```

- [ ] 4.16 Verify all existing imports still work
  - Run: `python -c "from DocsToKG.ContentDownload.resolvers import ResolverPipeline, default_resolvers, ResolverConfig"`
  - Run: `python -c "from DocsToKG.ContentDownload.resolvers import AttemptRecord, DownloadOutcome"`
  - Verify no ImportError exceptions

## 5. OpenAlex Virtual Resolver

- [ ] 5.1 Create `src/DocsToKG/ContentDownload/resolvers/providers/openalex.py`
  ```python
  """OpenAlex direct URL resolver (position 0 in pipeline)."""
  from typing import Iterable
  import requests
  from ..types import Resolver, ResolverConfig, ResolverResult
  from DocsToKG.ContentDownload.utils import dedupe
  
  class OpenAlexResolver:
      """Resolver for PDF URLs directly provided by OpenAlex metadata."""
      
      name = "openalex"
      
      def is_enabled(self, config: ResolverConfig, artifact) -> bool:
          """Enable when artifact has pdf_urls or open_access_url."""
          return bool(artifact.pdf_urls or artifact.open_access_url)
      
      def iter_urls(self, session: requests.Session, config: ResolverConfig, artifact) -> Iterable[ResolverResult]:
          """Yield all PDF URLs from OpenAlex work metadata."""
          candidates = list(artifact.pdf_urls)
          if artifact.open_access_url:
              candidates.append(artifact.open_access_url)
          
          for url in dedupe(candidates):
              if url:
                  yield ResolverResult(
                      url=url,
                      metadata={"source": "openalex_metadata"}
                  )
  ```

- [ ] 5.2 Register `OpenAlexResolver` in providers registry
  - Update `resolvers/providers/__init__.py` to import `OpenAlexResolver`
  - Insert at **position 0** in `default_resolvers()` return list (before Unpaywall)

- [ ] 5.3 Update `DEFAULT_RESOLVER_ORDER` in `resolvers/__init__.py`
  - Prepend `"openalex"` to list: `["openalex", "unpaywall", "crossref", ...]`

- [ ] 5.4 Remove `attempt_openalex_candidates()` function from `download_pyalex_pdfs.py`
  - Delete function definition (lines ~1332-1402)
  - Remove all calls to `attempt_openalex_candidates()` in `process_one_work()` (lines ~1509-1530)

- [ ] 5.5 Simplify `process_one_work()` to single pipeline execution
  - Remove `openalex_result` variable and conditional return (lines ~1509-1530)
  - Keep only `pipeline_result = pipeline.run(...)` logic
  - Remove `openalex_html_paths` aggregation logic (now handled by pipeline)
  - Remove `html_paths_total = list(artifact.metadata.get("openalex_html_paths", []))` (line ~1532)
  - Update `html_paths` to come solely from `pipeline_result.html_paths`

- [ ] 5.6 Update tests to reflect OpenAlexResolver integration
  - In `tests/test_pipeline_behaviour.py`, add test verifying OpenAlex resolver executes first
  - Verify rate limiting applies to OpenAlex URLs
  - Verify metrics track OpenAlex attempts correctly

## 6. HEAD-Based Content Filtering

- [ ] 6.1 Add `enable_head_precheck` field to `ResolverConfig` dataclass
  ```python
  # In resolvers/types.py, ResolverConfig class
  enable_head_precheck: bool = True
  resolver_head_precheck: Dict[str, bool] = field(default_factory=dict)
  ```

- [ ] 6.2 Implement HEAD pre-check helper in `resolvers/pipeline.py`
  ```python
  def _should_attempt_head_check(
      self, resolver_name: str
  ) -> bool:
      """Determine if HEAD pre-check should be performed for this resolver."""
      # Resolver-specific override takes precedence
      if resolver_name in self.config.resolver_head_precheck:
          return self.config.resolver_head_precheck[resolver_name]
      return self.config.enable_head_precheck
  
  def _head_precheck_url(
      self, session: requests.Session, url: str, timeout: float
  ) -> bool:
      """Return True if URL passes HEAD pre-check, False to skip."""
      try:
          from DocsToKG.ContentDownload.http import request_with_retries
          response = request_with_retries(
              session, "HEAD", url,
              max_retries=1,
              timeout=min(timeout, 5.0)
          )
          if response.status_code not in {200, 302, 304}:
              return False
          
          content_type = response.headers.get("Content-Type", "").lower()
          content_length = response.headers.get("Content-Length", "")
          
          # Skip obvious HTML or empty responses
          if "text/html" in content_type:
              return False
          if content_length == "0":
              return False
          
          return True
      except Exception:
          # On HEAD failure, allow GET attempt anyway
          return True
  ```

- [ ] 6.3 Integrate HEAD pre-check into pipeline URL iteration (in `ResolverPipeline.run()`)
  - After `result.url` extraction (line ~756), add:
  ```python
  if self._should_attempt_head_check(resolver_name):
      if not self._head_precheck_url(session, url, self.config.get_timeout(resolver_name)):
          self.logger.log(
              AttemptRecord(
                  ...,
                  status="skipped",
                  reason="head-precheck-failed",
              )
          )
          self.metrics.record_skip(resolver_name, "head-precheck-failed")
          continue
  ```

- [ ] 6.4 Add HEAD pre-check tests in `tests/test_resolver_pipeline.py`
  - Test HEAD returning HTML content-type skips GET
  - Test HEAD returning zero content-length skips GET
  - Test HEAD returning 404 skips GET
  - Test HEAD returning 200 with PDF content-type allows GET
  - Test HEAD failure (timeout/error) still allows GET attempt
  - Test `enable_head_precheck=False` disables all checks
  - Test resolver-specific override: `resolver_head_precheck: {wayback: false}`

## 7. Bounded Intra-Work Concurrency

- [ ] 7.1 Add concurrency fields to `ResolverConfig` dataclass
  ```python
  # In resolvers/types.py
  max_concurrent_resolvers: int = 1  # Sequential by default
  ```

- [ ] 7.2 Add thread-safe rate limit tracking in `ResolverPipeline.__init__()`
  - Ensure `_last_invocation` dict and `_lock` are used for thread safety
  - Already present in current implementation; verify thread-safety annotations

- [ ] 7.3 Implement concurrent resolver execution in `ResolverPipeline.run()`
  - At start of method, check `if self.config.max_concurrent_resolvers == 1:` to use existing sequential path
  - For concurrent mode (`> 1`), implement:
  ```python
  from concurrent.futures import ThreadPoolExecutor, as_completed
  
  with ThreadPoolExecutor(max_workers=self.config.max_concurrent_resolvers) as executor:
      futures_map = {}
      
      for order_index, resolver_name in enumerate(self.config.resolver_order, start=1):
          # ... existing is_enabled checks ...
          
          def _resolver_task(res_name, res_instance, order_idx):
              self._respect_rate_limit(res_name)
              results = []
              for result in res_instance.iter_urls(session, self.config, artifact):
                  results.append((res_name, order_idx, result))
              return results
          
          future = executor.submit(_resolver_task, resolver_name, resolver, order_index)
          futures_map[future] = (resolver_name, order_index)
      
      for future in as_completed(futures_map):
          resolver_name, order_index = futures_map[future]
          for result in future.result():
              # ... existing URL attempt logic ...
              if outcome.is_pdf:
                  executor.shutdown(wait=False, cancel_futures=True)
                  return PipelineResult(success=True, ...)
  ```

- [ ] 7.4 Ensure thread-safe metrics recording
  - Verify `ResolverMetrics` uses `Counter` which is thread-safe
  - Verify `self.logger.log()` calls are thread-safe (they are, since file writes use OS-level locks)

- [ ] 7.5 Add configuration validation
  - In `ResolverConfig.__post_init__()`, add:
  ```python
  if self.max_concurrent_resolvers < 1:
      raise ValueError("max_concurrent_resolvers must be >= 1")
  if self.max_concurrent_resolvers > 10:
      warnings.warn("max_concurrent_resolvers > 10 may violate rate limits")
  ```

- [ ] 7.6 Add concurrent execution tests in `tests/test_bounded_concurrency.py`
  - Test `max_concurrent_resolvers=1` uses sequential path
  - Test `max_concurrent_resolvers=3` executes up to 3 resolvers concurrently
  - Test rate limits are enforced despite concurrency
  - Test early-stop on first PDF cancels remaining futures
  - Test resolver failure doesn't crash concurrent execution
  - Mock slow resolvers to verify wall-time improvement

## 8. Zenodo Resolver

- [ ] 8.1 Create `src/DocsToKG/ContentDownload/resolvers/providers/zenodo.py`
  ```python
  """Zenodo repository resolver for DOI-indexed research outputs."""
  from typing import Iterable
  import requests
  from urllib.parse import quote
  from ..types import Resolver, ResolverConfig, ResolverResult
  from DocsToKG.ContentDownload.utils import normalize_doi
  from DocsToKG.ContentDownload.http import request_with_retries
  
  class ZenodoResolver:
      """Resolver for Zenodo open access repository."""
      
      name = "zenodo"
      
      def is_enabled(self, config: ResolverConfig, artifact) -> bool:
          """Enable when artifact has a DOI."""
          return artifact.doi is not None
      
      def iter_urls(
          self, session: requests.Session, config: ResolverConfig, artifact
      ) -> Iterable[ResolverResult]:
          """Query Zenodo API by DOI and yield PDF file URLs."""
          doi = normalize_doi(artifact.doi)
          if not doi:
              yield ResolverResult(
                  url=None, event="skipped", event_reason="no-doi"
              )
              return
          
          try:
              response = request_with_retries(
                  session,
                  "GET",
                  "https://zenodo.org/api/records/",
                  params={"q": f'doi:"{doi}"', "size": 3, "sort": "mostrecent"},
                  timeout=config.get_timeout(self.name),
                  headers=config.polite_headers,
              )
          except requests.RequestException as exc:
              yield ResolverResult(
                  url=None,
                  event="error",
                  event_reason="request-error",
                  metadata={"message": str(exc)},
              )
              return
          
          if response.status_code != 200:
              yield ResolverResult(
                  url=None,
                  event="error",
                  event_reason="http-error",
                  http_status=response.status_code,
              )
              return
          
          try:
              data = response.json()
          except ValueError:
              yield ResolverResult(
                  url=None, event="error", event_reason="json-error"
              )
              return
          
          for record in data.get("hits", {}).get("hits", []):
              for file_entry in record.get("files", []):
                  file_type = (file_entry.get("type") or "").lower()
                  file_key = (file_entry.get("key") or "").lower()
                  
                  if file_type == "pdf" or file_key.endswith(".pdf"):
                      url = file_entry.get("links", {}).get("self")
                      if url:
                          yield ResolverResult(
                              url=url,
                              metadata={
                                  "source": "zenodo",
                                  "record_id": record.get("id"),
                                  "filename": file_entry.get("key"),
                              },
                          )
  ```

- [ ] 8.2 Add Zenodo to resolver registry
  - Import `ZenodoResolver` in `resolvers/providers/__init__.py`
  - Insert in `default_resolvers()` after `CoreResolver`, before `DoajResolver`

- [ ] 8.3 Update `DEFAULT_RESOLVER_ORDER` constant
  - Insert `"zenodo"` after `"core"` in list

- [ ] 8.4 Add Zenodo resolver tests in `tests/test_zenodo_resolver.py`
  - Test DOI query with successful response containing PDF file
  - Test DOI query with multiple files, filter to PDF only
  - Test DOI query with no matches (empty hits)
  - Test DOI query with HTTP 404
  - Test DOI query with malformed JSON
  - Test DOI query with network error
  - Use `responses` library for mock HTTP fixtures

- [ ] 8.5 Add test fixture for Zenodo API response
  - Create `tests/data/zenodo_response_sample.json` with realistic API response
  - Include record with PDF file and non-PDF files

## 9. Figshare Resolver

- [ ] 9.1 Create `src/DocsToKG/ContentDownload/resolvers/providers/figshare.py`
  ```python
  """Figshare repository resolver for DOI-indexed research outputs."""
  from typing import Iterable
  import requests
  from ..types import Resolver, ResolverConfig, ResolverResult
  from DocsToKG.ContentDownload.utils import normalize_doi
  from DocsToKG.ContentDownload.http import request_with_retries
  
  class FigshareResolver:
      """Resolver for Figshare repository."""
      
      name = "figshare"
      
      def is_enabled(self, config: ResolverConfig, artifact) -> bool:
          """Enable when artifact has a DOI."""
          return artifact.doi is not None
      
      def iter_urls(
          self, session: requests.Session, config: ResolverConfig, artifact
      ) -> Iterable[ResolverResult]:
          """Search Figshare API by DOI and yield PDF file URLs."""
          doi = normalize_doi(artifact.doi)
          if not doi:
              yield ResolverResult(
                  url=None, event="skipped", event_reason="no-doi"
              )
              return
          
          try:
              response = request_with_retries(
                  session,
                  "POST",
                  "https://api.figshare.com/v2/articles/search",
                  json={
                      "search_for": f':doi: "{doi}"',
                      "page": 1,
                      "page_size": 3,
                  },
                  timeout=config.get_timeout(self.name),
                  headers=dict(config.polite_headers, **{"Content-Type": "application/json"}),
              )
          except requests.RequestException as exc:
              yield ResolverResult(
                  url=None,
                  event="error",
                  event_reason="request-error",
                  metadata={"message": str(exc)},
              )
              return
          
          if response.status_code != 200:
              yield ResolverResult(
                  url=None,
                  event="error",
                  event_reason="http-error",
                  http_status=response.status_code,
              )
              return
          
          try:
              articles = response.json()
          except ValueError:
              yield ResolverResult(
                  url=None, event="error", event_reason="json-error"
              )
              return
          
          for article in articles:
              for file_entry in article.get("files", []):
                  filename = (file_entry.get("name") or "").lower()
                  download_url = file_entry.get("download_url")
                  
                  if filename.endswith(".pdf") and download_url:
                      yield ResolverResult(
                          url=download_url,
                          metadata={
                              "source": "figshare",
                              "article_id": article.get("id"),
                              "filename": file_entry.get("name"),
                          },
                      )
  ```

- [ ] 9.2 Add Figshare to resolver registry
  - Import `FigshareResolver` in `resolvers/providers/__init__.py`
  - Insert in `default_resolvers()` after `ZenodoResolver`, before `DoajResolver`

- [ ] 9.3 Update `DEFAULT_RESOLVER_ORDER` constant
  - Insert `"figshare"` after `"zenodo"` in list

- [ ] 9.4 Add Figshare resolver tests in `tests/test_figshare_resolver.py`
  - Test DOI search with successful response containing PDF file
  - Test DOI search with multiple files, filter to PDF only
  - Test DOI search with no matches (empty array)
  - Test DOI search with HTTP 404
  - Test DOI search with malformed JSON
  - Test DOI search with network error
  - Use `responses` library for mock HTTP fixtures

- [ ] 9.5 Add test fixture for Figshare API response
  - Create `tests/data/figshare_response_sample.json` with realistic search response
  - Include article with PDF file and non-PDF files

## 10. DownloadOutcome Completeness

- [ ] 10.1 Audit `download_candidate()` for outcome completeness
  - Verify 304 cached path (line ~989) populates all fields: `sha256`, `content_length`, `etag`, `last_modified`
  - Verify HTTP error path (line ~1002) sets fields to `None` explicitly
  - Verify PDF/HTML success path (line ~1114) populates all fields
  - Verify dry-run path (line ~1060) populates available fields or `None`

- [ ] 10.2 Standardize `None` vs missing field handling
  - In `_build_download_outcome()` (line ~694), ensure all optional fields default to `None` rather than being omitted
  - Update `DownloadOutcome` dataclass initialization defaults to `None` explicitly

- [ ] 10.3 Audit `build_manifest_entry()` for completeness
  - Verify function at line ~526 extracts all `DownloadOutcome` fields
  - Ensure `ManifestEntry` includes all fields: `sha256`, `content_length`, `etag`, `last_modified`, `extracted_text_path`

- [ ] 10.4 Add validation tests in `tests/test_download_outcomes.py`
  - Test successful download has all metadata fields populated
  - Test 304 cached response has all prior metadata fields
  - Test HTTP error response has explicit `None` for unavailable fields
  - Test HTML download with text extraction has `extracted_text_path` populated
  - Test dry-run mode has correct field availability

## 11. Extended Logging and Observability

- [ ] 11.1 Add `resolver_wall_time_ms` field to `AttemptRecord` dataclass
  ```python
  # In resolvers/types.py, AttemptRecord
  resolver_wall_time_ms: Optional[float] = None
  ```

- [ ] 11.2 Track resolver wall time in `ResolverPipeline.run()`
  - Before `_respect_rate_limit()` call, capture start time: `resolver_start = time.monotonic()`
  - After all URLs from resolver attempted, compute: `resolver_wall = (time.monotonic() - resolver_start) * 1000.0`
  - Pass `resolver_wall_time_ms=resolver_wall` to relevant `AttemptRecord` instances

- [ ] 11.3 Update JSONL logger to include new field
  - In `JsonlLogger.log_attempt()` method (line ~262 of `download_pyalex_pdfs.py`), add `"resolver_wall_time_ms"` to output dict

- [ ] 11.4 Update CSV logger adapter to include new field
  - Add `"resolver_wall_time_ms"` to `CsvAttemptLoggerAdapter.HEADER` list (line ~380)
  - Add field to `_writer.writerow()` dict (line ~417)

- [ ] 11.5 Add logging tests in `tests/test_structured_logging.py`
  - Test attempt record includes `resolver_wall_time_ms` when resolver completes
  - Test CSV export includes new column
  - Test JSONL export includes new field

## 12. Configuration Enhancements

- [ ] 12.1 Document new configuration options in `ResolverConfig` docstring
  - Add description for `enable_head_precheck`
  - Add description for `resolver_head_precheck` per-resolver override
  - Add description for `max_concurrent_resolvers`

- [ ] 12.2 Add configuration examples to documentation
  - Create `docs/resolver-configuration.md` with YAML examples
  - Include example enabling bounded concurrency:
  ```yaml
  max_concurrent_resolvers: 3
  resolver_min_interval_s:
    unpaywall: 1.0
    crossref: 0.5
  ```
  - Include example disabling HEAD pre-check for specific resolvers:
  ```yaml
  enable_head_precheck: true
  resolver_head_precheck:
    wayback: false
  ```

- [ ] 12.3 Add CLI help for new options (if exposed)
  - Currently config is file-based; document in README

## 13. Integration Testing

- [ ] 13.1 Create comprehensive pipeline integration test in `tests/test_full_pipeline_integration.py`
  - Mock OpenAlex work metadata with DOI, PMCID, arXiv ID
  - Mock responses for all resolver types (Unpaywall, Crossref, PMC, Zenodo, Figshare, etc.)
  - Verify resolver order: OpenAlex → Unpaywall → ... → Zenodo → Figshare → ... → Wayback
  - Verify early-stop on first PDF success
  - Verify rate limiting enforced
  - Verify metrics aggregation correct

- [ ] 13.2 Add end-to-end test with real network calls (marked as integration)
  - Use `pytest.mark.integration` decorator
  - Test downloading a known open-access DOI (e.g., "10.1371/journal.pone.0000001")
  - Verify PDF downloaded successfully
  - Verify manifest entry contains all metadata fields

- [ ] 13.3 Add performance benchmark test
  - Create `tests/benchmarks/test_resolver_performance.py`
  - Benchmark sequential vs concurrent resolver execution (mock resolvers with sleep)
  - Verify concurrent mode reduces wall time by expected factor
  - Use `pytest-benchmark` plugin for structured results

## 14. Migration and Compatibility

- [ ] 14.1 Create migration guide in `docs/migration-modularize-resolvers.md`
  - Document old import paths: `from DocsToKG.ContentDownload.resolvers import ResolverPipeline`
  - Document new import paths (recommended): `from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline`
  - Note that old paths continue working via re-exports
  - List new configuration options and defaults

- [ ] 14.2 Add deprecation warnings for internal APIs (if any are exposed)
  - Add `warnings.warn()` for any deprecated internal functions
  - Document in CHANGELOG.md

- [ ] 14.3 Verify backward compatibility with existing tests
  - Run full test suite: `pytest tests/`
  - Ensure all existing tests pass without modification
  - Ensure no new import errors

## 15. Documentation

- [ ] 15.1 Update main README.md
  - Add section on new Zenodo and Figshare resolver support
  - Document bounded concurrency option
  - Document HEAD pre-check optimization

- [ ] 15.2 Create developer guide for adding custom resolvers
  - Create `docs/adding-custom-resolvers.md`
  - Provide resolver template:
  ```python
  class MyResolver:
      name = "my_resolver"
      
      def is_enabled(self, config, artifact):
          return True  # or conditional logic
      
      def iter_urls(self, session, config, artifact):
          yield ResolverResult(url="https://example.org/file.pdf")
  ```
  - Document registration process in `providers/__init__.py`
  - Document configuration options (timeouts, rate limits, toggles)

- [ ] 15.3 Update API documentation
  - Regenerate Sphinx/MkDocs API docs if applicable
  - Ensure new modules appear in documentation

- [ ] 15.4 Create architecture diagram
  - Visualize resolver pipeline flow: Work → Pipeline → Resolvers → Download → Outcome
  - Show modular structure: `pipeline.py`, `types.py`, `providers/*`
  - Include in `docs/architecture.md`

## 16. Testing Coverage

- [ ] 16.1 Achieve 95%+ branch coverage for new modules
  - Run: `pytest --cov=src/DocsToKG/ContentDownload/http --cov-report=html`
  - Run: `pytest --cov=src/DocsToKG/ContentDownload/conditional --cov-report=html`
  - Run: `pytest --cov=src/DocsToKG/ContentDownload/resolvers --cov-report=html`
  - Review HTML report and add tests for uncovered branches

- [ ] 16.2 Add missing edge case tests
  - Test `Retry-After` with date format (currently only integer tested)
  - Test HEAD pre-check with redirects (302 → 200)
  - Test concurrent execution with resolver exception handling
  - Test Zenodo/Figshare pagination (if implemented)

- [ ] 16.3 Add property-based tests with `hypothesis`
  - Test `ConditionalRequestHelper` with arbitrary ETag/Last-Modified values
  - Test `request_with_retries` backoff timing properties
  - Test dedupe utility with various input sequences

## 17. Cleanup and Finalization

- [ ] 17.1 Remove unused code from `download_pyalex_pdfs.py`
  - Remove `attempt_openalex_candidates()` function (replaced by OpenAlexResolver)
  - Remove any dead code paths from refactoring

- [ ] 17.2 Run linters and formatters
  - Run: `black src/DocsToKG/ContentDownload/`
  - Run: `isort src/DocsToKG/ContentDownload/`
  - Run: `mypy src/DocsToKG/ContentDownload/` (if type checking enabled)
  - Fix any linter errors or warnings

- [ ] 17.3 Update CHANGELOG.md
  - Add entry for modularization changes
  - List new features: Zenodo, Figshare, bounded concurrency, HEAD pre-check
  - Note backward compatibility maintained

- [ ] 17.4 Verify all tasks complete and tests pass
  - Run full test suite: `pytest tests/`
  - Verify no regressions
  - Verify new features work as expected

- [ ] 17.5 Create summary of changes for pull request
  - List files created (13 new provider modules, 3 new utility modules)
  - List files modified (download_pyalex_pdfs.py, tests)
  - Document test coverage metrics
  - Include performance benchmarks (wall-time improvements)

