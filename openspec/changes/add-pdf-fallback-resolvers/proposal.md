## Why

DocsToKG currently relies on the OpenAlex/pyalex payload for locating downloadable PDFs. Works that surface without a populated `pdf_url` are skipped entirely, even when other public sources provide an accessible copy. This leaves large gaps in downstream processing and forces manual recovery, despite there being safe, well-documented resolver APIs we can automate against.

## What Changes

- Extend the OpenAlex query to request richer identifier and location fields (DOI, PMID/PMCID, arXiv, landing pages) so downstream resolvers have the right context.
- Introduce a resolver module/pipeline interface so each source (Unpaywall → Crossref → landing-page metadata scrape → arXiv → PubMed Central → Europe PMC → CORE → DOAJ → Semantic Scholar → Wayback) can be independently enabled, share retry/backoff policy, and short-circuit on the first confirmed PDF.
- Normalize download handling so every resolver streams through the existing sniffing logic, confirms `%PDF-` signatures or `Content-Type`, and saves PDFs and HTML fallbacks into their dedicated directories without duplicating path logic.
- Make resolver behavior configurable (per-source toggles, API credentials, polite pool email, rate limits) via a lightweight YAML resolver config file (PyYAML-backed) plus CLI overrides while keeping compatibility with current usage.
- Capture resolver provenance and outcomes in structured logs/CSV (resolver name, URL, status, elapsed time) and provide aggregate metrics so operators can monitor hit rates, retry cost, and any ToS-related failures across sources.
- Cover the resolver stack with unit tests (per-source parsing), integration fixtures, and a manual smoke checklist so AI agents and maintainers have a clear validation path.

## Impact

- Affected specs: new `content-download` capability (PDF resolver stack and storage)
- Affected code: `src/DocsToKG/ContentDownload/download_pyalex_pdfs.py`, new resolver utilities/modules, configuration helpers, logging/reporting
- External services: Unpaywall, Crossref, publisher landing pages, arXiv, PubMed Central, Europe PMC, CORE, DOAJ, Semantic Scholar, Internet Archive (Wayback)
