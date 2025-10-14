## 1. OpenAlex Intake & Data Modeling
- [ ] 1.1 Update `download_pyalex_pdfs.build_query` (and any helper that wraps pyalex `Works`) to include a `select` clause that fetches `ids`, `locations.landing_page_url`, `locations.source.display_name`, `primary_location`, `best_oa_location`, and `open_access.oa_url` alongside the existing fields.
- [ ] 1.2 Introduce a `WorkArtifact` (dataclass or typed dict) that normalizes the identifiers (DOI, PMID, PMCID, arXiv), landing-page URLs, OpenAlex-provided PDF URLs, and human-readable context (title, year, venue) so downstream resolvers and loggers can consume a stable shape.
- [ ] 1.3 Thread the enriched `WorkArtifact` through the download loop, ensuring existing sniffing logic receives the same base file stem, and update any CSV logging helpers to read from this artifact instead of raw OpenAlex records.
- [ ] 1.4 Capture the namespace keyed directories (`pdf`, `html`) in the artifact or config so resolvers know where to persist outputs without duplicating path logic.

## 2. Resolver Stack Implementation
- [ ] 2.1 Create a `resolvers/` module that defines a `Resolver` protocol (`name`, `is_enabled(config, artifact)`, `iter_urls(...)`) and central `ResolverPipeline` that enforces ordering, dedupes candidate URLs, applies referer headers when needed, and records attempt metadata.
- [ ] 2.2 Implement `UnpaywallResolver` that requires a configured email, calls `/v2/<doi>`, extracts `best_oa_location.url_for_pdf` plus `oa_locations[].url_for_pdf`, applies polite rate limiting, and surfaces soft failures (e.g., HTTP 404) to the pipeline.
- [ ] 2.3 Implement `CrossrefResolver` that hits `/works/<doi>` once per DOI, parses `message.link[]`, prioritizes `content-type == application/pdf`, and appends `content-version`/`intended-application` context to the attempt metadata.
- [ ] 2.4 Implement `LandingPageResolver` that fetches each landing page with the shared HTTP session, parses `<meta name="citation_pdf_url">` and `<link rel="alternate" type="application/pdf">` via `BeautifulSoup`, normalizes relative URLs, and returns at most one candidate per landing page to avoid combinatorial explosions.
- [ ] 2.5 Implement repository resolvers: `ArxivResolver`, `PmcResolver`, `EuropePmcResolver`, `CoreResolver`, `DoajResolver`, and `SemanticScholarResolver`, each using official APIs/patterns, attaching necessary headers (e.g., API keys), and performing DOI/identifier normalization (e.g., stripping `arXiv:` prefixes, upper-casing `PMC`).
- [ ] 2.6 Implement `WaybackResolver` that only fires when a previously-known PDF URL failed, calls the availability API, and returns archived URLs tagged with capture timestamp metadata.
- [ ] 2.7 Ensure each resolver respects global timeout/backoff rules, reuses the shared `requests.Session`, and yields URLs in a generator-friendly fashion so the pipeline can short-circuit on the first confirmed PDF.
- [ ] 2.8 Update the main download loop to call the pipeline after OpenAlex candidates, pass the artifact + configuration, reuse the existing `download_pdf` streaming helper, and emit resolver provenance (resolver name, URL, status) on every attempt.

## 3. Configuration, Politeness, and Logging
- [ ] 3.1 Introduce a `ResolverConfig` loader (e.g., from `download_config.yaml` or CLI JSON) that contains per-resolver toggles, max attempts per work, global timeout, sleep jitter, and credential slots (`unpaywall_email`, `core_api_key`, `semantic_scholar_api_key`, `doaj_api_key`).
- [ ] 3.2 Extend the CLI to accept `--resolver-config` plus individual overrides (`--unpaywall-email`, `--disable-resolver=core`) while keeping backward compatibility for existing flags; document environment variable fallbacks.
- [ ] 3.3 Add polite-pool headers (`mailto`, `User-Agent`) and per-resolver rate policies (e.g., Unpaywall 1 QPS, Crossref back-off on 429) configurable via the new config.
- [ ] 3.4 Augment the CSV log (and any JSONL output) with columns for `resolver_source`, `resolver_order`, `attempt_url`, `status`, `http_status`, `content_type`, and `elapsed_ms`, and ensure misses capture the reason (`no-doi`, `resolver-disabled`, `max-attempts-reached`).
- [ ] 3.5 Emit structured INFO logs summarizing resolver success rate per run and aggregate counts (OpenAlex hits vs. each fallback source) to help calibrate the stack.

## 4. Validation, Testing, and Operational Hardening
- [ ] 4.1 Add unit tests for `classify_payload`, URL deduplication, resolver enable/disable logic, and ordering guarantees using pytest + `responses` (or `requests-mock`) to simulate HTTP interactions.
- [ ] 4.2 Write resolver-specific tests that confirm metadata parsing (e.g., Unpaywall JSON, Crossref link filtering, landing-page HTML cases, PMC OA XML) and ensure each resolver stops emitting URLs when required fields are missing.
- [ ] 4.3 Create an integration test fixture (JSONL or recorded responses) that drives the CLI against 3–5 synthetic works covering DOI-only, arXiv-only, PMC-only, and total miss scenarios, asserting outputs land in the correct directories and log rows match expectations.
- [ ] 4.4 Document a manual smoke test checklist (e.g., run against 10 live DOIs with rate limit safe pacing, verify compliance headers, inspect logs) and attach it to the change artifacts for future operators.
- [ ] 4.5 Establish metrics/alert expectations (e.g., acceptable resolver miss rate, HTTP error thresholds) and feed them into the ops documentation referenced by the change.
