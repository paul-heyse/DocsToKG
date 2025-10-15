## ADDED Requirements

### Requirement: Enrich OpenAlex Work Intake
`download_pyalex_pdfs.py` MUST request and capture OpenAlex `ids` (DOI, PMID, PMCID, arXiv) along with `locations.landing_page_url`, `locations.source`, and existing OA fields so downstream resolvers have complete context.

#### Scenario: Work metadata includes resolver-ready identifiers
- **GIVEN** the downloader fetches works from OpenAlex
- **WHEN** a work is parsed
- **THEN** the work record made available to resolvers includes DOI, PMID/PMCID (when present), arXiv ID, any landing-page URLs, and OpenAlex-provided PDF URLs.

### Requirement: Resolver Stack Executes in Priority Order
When no OpenAlex-provided URL yields a confirmed PDF, the downloader MUST attempt resolvers in this order and stop after the first successful PDF: Unpaywall → Crossref → landing-page metadata scrape → arXiv → PubMed Central → Europe PMC → CORE → DOAJ → Semantic Scholar → Wayback Machine.

#### Scenario: PDF found via Crossref after OpenAlex miss
- **GIVEN** OpenAlex returns no working PDF URL for a work with a DOI
- **WHEN** the resolver stack runs
- **THEN** the Crossref resolver is attempted after Unpaywall and before landing-page scraping, and the first confirmed PDF halts further resolvers.

#### Scenario: No DOI but arXiv identifier present
- **GIVEN** a work contains only an arXiv ID and OpenAlex-provided URLs fail
- **WHEN** the resolver stack executes
- **THEN** the arXiv PDF endpoint is tried in its designated position and provides the download if reachable.

### Requirement: Downloads Respect Format Classification
All resolver downloads MUST reuse the existing sniffing logic to classify payloads, saving confirmed PDFs into the PDF output directory and HTML/landing copies into the HTML directory while recording the resolver source.

#### Scenario: Landing page returns HTML
- **GIVEN** the landing-page scraper identifies an HTML-only response
- **WHEN** the download completes
- **THEN** the HTML is stored in the HTML directory with the work stem and the attempt is logged as a non-PDF resolver outcome.

### Requirement: Resolver Configuration and Telemetry
The downloader MUST expose configuration for resolver toggles, per-source credentials (e.g., Unpaywall email, API keys), timeout/backoff controls, and MUST log the resolver provenance, final URL, and status for every attempt in its CSV/log output.

#### Scenario: Resolver disabled via configuration
- **GIVEN** a resolver is toggled off in configuration
- **WHEN** a work would otherwise reach that resolver in the stack
- **THEN** the downloader skips that resolver, records the skip in logs, and proceeds to the next enabled resolver.
