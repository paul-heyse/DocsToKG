## Context

DocsToKG processes scientific documents and builds knowledge graphs from their content. Ontologies provide controlled vocabularies, taxonomies, and semantic relationships essential for entity recognition, concept linking, and knowledge graph quality. The project requires access to biomedical ontologies (GO, HP, UBERON, EFO, NCIT), legal/financial thesauri (EuroVoc, XBRL taxonomies), and domain-specific vocabularies from multiple sources (OBO Library, OLS, BioPortal, SKOS registries). Current workflows rely on manual downloads or ad-hoc scripts that lack caching, provenance, validation, and error recovery.

## Goals / Non-Goals

- Goals:
  - Automate discovery and download of ontologies from OBO Library, OLS, BioPortal, SKOS endpoints, and XBRL registries
  - Provide robust HTTP handling with conditional requests (ETag/Last-Modified), resume support, checksums, retries, and rate limiting
  - Validate downloaded artifacts using domain-appropriate parsers (RDFLib, Pronto, Owlready2, Arelle)
  - Normalize ontologies to consistent formats (Turtle, OBO Graph JSON) while preserving originals
  - Record comprehensive provenance (source URL, version, license, checksums, timestamps) in manifest files
  - Expose declarative configuration (YAML) for ontology lists, resolver preferences, and HTTP policies
  - Support idempotent operations (cached downloads, incremental updates)
  - Enable observability through structured logs and validation reports
- Non-Goals:
  - Building ontology editors or reasoners (use existing tools: Protégé, HermiT, ELK)
  - Implementing custom RDF stores or SPARQL endpoints (downstream systems handle storage)
  - Supporting proprietary ontologies with restricted access beyond credential configuration
  - Real-time ontology update notifications (polling-based updates suffice)
  - GUI for ontology browsing (CLI and programmatic API only)

## Decisions

- Decision: **Use battle-tested Python libraries instead of custom implementations** for discovery (Bioregistry, OAK), access (ols-client, ontoportal-client), parsing (RDFLib, Pronto, Owlready2), and domain-specific validation (Arelle for XBRL, optional ROBOT for conversions).  
  Alternatives: custom implementations would duplicate well-maintained community tools and increase maintenance burden. Rationale: leverages expertise and testing from OBO, EMBL-EBI, NCBO, and RDF communities.

- Decision: **Adopt pystow for standardized data directories and secrets management**, aligning with OAK and ontoportal-client conventions.  
  Alternatives: custom directory structures would diverge from ecosystem conventions. Rationale: users get predictable paths (~/.data/ontology-fetcher), env-var overrides, and unified configuration with related tools.

- Decision: **Use pooch (not cached_path) for download plumbing** with hash verification, caching, and registry support.  
  Alternatives: custom HTTP logic or cached_path. Rationale: pooch provides built-in SHA-256 verification, registry patterns, and cleaner API for our use case; cached_path is simpler but lacks hash verification features.

- Decision: **Implement resolver registry pattern** with protocol-based interfaces (`Resolver.plan(spec) → FetchPlan`) allowing independent resolvers for OBO, OLS, BioPortal, SKOS, and XBRL.  
  Alternatives: monolithic resolver with conditional logic would be harder to test and extend. Rationale: enables per-source unit testing, clean addition of new sources, and isolated credential handling.

- Decision: **Support format preferences and fallback chains** (e.g., prefer OWL → OBO → RDF) per ontology, with first-available-wins semantics.  
  Alternatives: hardcode single format per source. Rationale: some ontologies publish multiple serializations with varying quality; flexibility improves success rates.

- Decision: **Store original artifacts and normalized versions separately** in `original/` and `normalized/` subdirectories with validation reports in `validation/`.  
  Alternatives: only normalized versions or single mixed directory. Rationale: preserves bit-exact originals for reproducibility while providing tooling-friendly normalized formats (Turtle) for downstream pipelines.

- Decision: **Record comprehensive provenance in manifest.json** including URL, resolver, version, license, ETag, Last-Modified, SHA-256, timestamps, and validation status.  
  Alternatives: minimal metadata or external database. Rationale: self-contained manifests enable auditing, cache invalidation decisions, and offline reproducibility without external dependencies.

- Decision: **Support ETag/Last-Modified conditional requests and HTTP Range resume** for idempotent operations and efficient incremental updates.  
  Alternatives: always redownload. Rationale: reduces bandwidth, respects rate limits, and enables graceful recovery from network interruptions.

- Decision: **Implement exponential backoff retry (5 attempts, 0.5s base) and per-host rate limiting (4 req/sec default)** to avoid overwhelming source services.  
  Alternatives: no retry or unlimited rate. Rationale: improves reliability against transient failures while being a respectful HTTP client (follows robots.txt spirit).

- Decision: **Use ols-client (not ebi-ols-client) for OLS4 access** as specified by user requirements.  
  Alternatives: ebi-ols-client or direct REST calls. Rationale: user preference stated explicitly; ols-client is actively maintained.

- Decision: **Expose CLI via `ontofetch` command** with subcommands (pull, show, validate) for both single-ontology and batch operations.  
  Alternatives: Python API only or separate scripts. Rationale: CLI provides immediate usability for operators while Python API supports programmatic integration.

- Decision: **Configure ontology lists and policies via sources.yaml** with defaults section for global settings and per-ontology overrides.  
  Alternatives: JSON, TOML, or code-based configuration. Rationale: YAML is human-friendly for lists, supports comments, and aligns with common DevOps practices.

- Decision: **Validate using multiple parsers per format** (RDFLib + Pronto for OWL/OBO) to catch format-specific issues and gather richer metadata.  
  Alternatives: single parser per format. Rationale: different parsers catch different issues (RDFLib for RDF syntax, Pronto for OBO semantics, Owlready2 for reasoning); redundancy improves quality assurance.

- Decision: **Make ROBOT integration optional** (subprocess-based, only when installed) for conversions and QC reports.  
  Alternatives: mandatory ROBOT or skip entirely. Rationale: ROBOT requires Java runtime and separate install; optional support maximizes compatibility while enabling power users to leverage OBO community's QC tooling.

- Decision: **Treat XBRL and RF2 (SNOMED) as native formats** without forced conversion, using domain-specific tools (Arelle for XBRL, external Snowstorm/FHIR for RF2).  
  Alternatives: force RDF conversion. Rationale: these formats have specialized semantics and tooling; premature conversion loses fidelity.

## Architecture Overview

- **Configuration Layer**: YAML parser loads `sources.yaml` into structured config objects (FetchSpec list, defaults) with validation of required fields and allowlist checks.

- **Resolver Registry**: Protocol-based registry maps resolver names (`obo`, `ols`, `bioportal`, `skos`, `xbrl`) to concrete implementations:
  - **OBOResolver**: uses Bioregistry functions (`get_owl_download`, `get_obo_download`, `get_rdf_download`) to resolve OBO Foundry prefixes to PURLs; falls back through format preferences.
  - **OLSResolver**: queries OLS4 API via ols-client to find ontology by ID/acronym and retrieve canonical artifact URL with version metadata.
  - **BioPortalResolver**: uses ontoportal-client to fetch ontology metadata and latest submission download URL; includes API key from pystow config in headers.
  - **SKOSResolver**: returns configured direct URL for SKOS thesauri (EuroVoc, ECB vocabularies); validation deferred to RDFLib.
  - **XBRLResolver**: returns configured ZIP URL for XBRL taxonomy packages; validation deferred to Arelle post-extraction.

- **Download Manager**: Wraps pooch for robust fetching with:
  - Stream-to-`.part` file pattern to avoid corrupted partial downloads.
  - ETag/Last-Modified conditional headers from previous manifest; on HTTP 304, returns cached status.
  - Range request support for resume on connection interruption.
  - SHA-256 computation on complete files; verification against known hashes when available.
  - Exponential backoff retry with per-host rate limiting (token bucket).
  - Structured logging of HTTP status, elapsed time, retry count, and final outcome.

- **Validation Pipeline**: Post-download, runs format-appropriate validators:
  - **RDFLib**: `Graph.parse()` for RDF/OWL/SKOS; records triple count or error; optionally serializes to normalized Turtle.
  - **Pronto**: `Ontology()` load for OBO/OWL; records term count, relationships, or error; optionally exports OBO Graph JSON.
  - **Owlready2**: `get_ontology().load()` for OWL; enables optional reasoning checks.
  - **ROBOT** (if installed): subprocess calls for `robot convert` (format conversions) and `robot report` (SPARQL-based QC); captures stdout/stderr and TSV reports.
  - **Arelle** (XBRL): loads taxonomy package, runs validation, records results as JSON.
  - Results written to `validation/<parser>_parse.json` or `validation/robot_report.tsv`.

- **Storage Layout**: pystow-managed root (~/.data/ontology-fetcher) contains:
  ```
  configs/sources.yaml        # ontology specs
  cache/                      # HTTP cache, ETags, partial downloads
  logs/                       # structured JSON logs per run
  ontologies/<id>/<version>/
    original/                 # bit-exact downloaded files
    normalized/               # Turtle, OBO Graph JSON
    validation/               # parse results, QC reports
    manifest.json             # provenance metadata
  ```

- **CLI (`ontofetch`)**: Command-line interface with subcommands:
  - `ontofetch pull [--spec FILE | ID...]`: download ontologies; updates existing or fetches new.
  - `ontofetch show ID [--versions]`: display manifest, versions, or validation status.
  - `ontofetch validate ID[@VERSION] [--robot] [--rdflib] [--pronto]`: re-run validators.
  - Structured output (JSON for machine, formatted for human) based on `--json` flag.

- **Observability**: Structured JSON logs capture:
  - Resolver: source, ID, resolver type, plan URL, planning time.
  - Download: URL, status code, ETag, content length, SHA-256, elapsed time, retries, cache hit/miss.
  - Validation: parser, outcome (success/failure), error messages, metrics (triples, terms).
  - Aggregation: per-run summary with success/failure/cached counts, total bandwidth, total time.

## Risks / Trade-offs

- Risk: External service rate limits or transient failures during batch downloads. Mitigation: exponential backoff, per-host rate limiting, graceful failure recording in logs; operators can resume failed items.

- Risk: OLS, BioPortal, or other APIs change response formats or authentication. Mitigation: use well-maintained client libraries (ols-client, ontoportal-client) that abstract API details; monitor upstream releases for breaking changes.

- Risk: Large ontologies (SNOMED, NCIT, ChEBI) exhaust memory during parsing/reasoning. Mitigation: stream-based download, optional validation (skip reasoning for large files), document memory requirements per ontology class.

- Risk: License restrictions on some ontologies (SNOMED, LOINC) require explicit credentials or acceptance. Mitigation: fail closed with clear error messages when restricted ontologies lack credentials; document credential configuration in pystow paths.

- Risk: XBRL ZIP files may contain malicious paths (zip-slip vulnerability). Mitigation: use safe extraction with path validation (Python's `zipfile` with `is_zipfile` check and member path filtering).

- Risk: ROBOT dependency on Java runtime may not be available. Mitigation: make ROBOT integration optional with runtime detection (`shutil.which('robot')`); skip conversion/QC steps if absent.

- Risk: Divergent tokenization between Pronto/RDFLib/Owlready2 on same OWL file. Mitigation: treat parsers as independent validators rather than requiring consensus; log discrepancies for manual review.

- Risk: Stale cached ontologies if ETag/Last-Modified not supported by source. Mitigation: allow force-refresh flag (`--force`) to bypass cache; document staleness detection limits.

- Risk: Configuration drift if sources.yaml manually edited without validation. Mitigation: validate YAML schema on load; provide `ontofetch config validate` command.

## Migration Plan

1. Scaffold module structure under `src/DocsToKG/OntologyDownload/`: `core.py`, `resolvers.py`, `validators.py`, `cli.py`.
2. Install dependencies via `requirements.txt` or `pyproject.toml` with version pins: `bioregistry`, `oaklib`, `ols-client`, `ontoportal-client`, `rdflib`, `pronto`, `owlready2`, `arelle`, `pystow`, `pooch`.
3. Implement core data models (FetchSpec, FetchPlan, FetchResult) and resolver protocol.
4. Implement resolvers one at a time (OBO → OLS → BioPortal → SKOS → XBRL) with unit tests using mocked library responses.
5. Build download manager wrapping pooch with ETag/Range/SHA-256 logic; test with local HTTP server fixture supporting conditional requests and Range headers.
6. Implement validation pipeline with RDFLib and Pronto; add small test fixtures (5-10 term mini-ontologies in Turtle, OBO, OWL).
7. Add ROBOT and Arelle integrations with runtime detection and subprocess management.
8. Build pystow-based storage layer with directory creation, manifest read/write, and version resolution.
9. Implement CLI using argparse or Click with pull/show/validate subcommands; add structured logging (JSON) via Python `logging` module with custom formatter.
10. Create example `sources.yaml` with representative ontologies (HP, EFO, NCIT, EuroVoc, small XBRL) and document configuration schema.
11. Add integration smoke tests that fetch 2-3 small public ontologies (e.g., PATO from OBO, BFO from OLS) and verify manifests and validation outputs.
12. Document installation, configuration (pystow env vars, API keys), CLI usage, and troubleshooting in README or docs/.
13. Optional: Add `openspec archive` after deployment to move change to archive/ and create canonical spec in `openspec/specs/ontology-downloader/`.

## Open Questions

- Should we support parallel downloads for batch operations, or serialize to respect global rate limits?
- What is the preferred handling for ontologies with multiple versions (keep all, prune old, configurable retention)?
- Should validation failures block downstream processing, or just log warnings?
- Do we need integration with OAK's adapter system for post-download ontology querying, or keep downloader focused solely on acquisition?
- Should we expose programmatic Python API alongside CLI, or CLI-first with API added later?
- What authentication patterns should we support for restricted ontologies (API keys in pystow, OAuth, certificate-based)?

