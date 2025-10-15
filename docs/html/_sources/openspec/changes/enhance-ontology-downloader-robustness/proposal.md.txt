# Enhance Ontology Downloader: Robustness, Code Quality, and Extended Capabilities

## Why

The ontology downloader implementation in `src/DocsToKG/OntologyDownload` is functional and well-architected but carries technical debt and missing hardening that limits its production readiness and maintainability. A comprehensive code review identified 15 high-ROI improvements spanning three categories:

1. **Code Quality & Maintenance Burden**: ~150 lines of custom YAML fallback parser, duplicated optional dependency stubs across modules, and scattered CLI formatting utilities increase maintenance surface and testing complexity.

2. **Robustness & Security Gaps**: Missing media-type validation before downloads, coarse per-host rate limiting (vs. per-service), insufficient archive format coverage (tar.gz/tar.xz), non-deterministic normalization output, memory fragmentation from heavy validators, raw license string matching, and basic SSRF protection without IDN handling.

3. **Limited Capabilities**: Single-resolver fallback requires manual config changes, missing polite API headers increase throttling, coverage gaps for LOV/Ontobee sources, local-only storage limits team workflows, and basic CLI lacks planning/diagnostics commands.

These issues create operational friction, increase failure rates for edge cases, and require more manual intervention than necessary for a production pipeline component.

## What Changes

### Quick Wins (Code Cleanup - delete boilerplate, DRY it up)

- **Replace custom YAML parser with Pydantic v2 models** (`config.py`)
  - Delete ~150 lines of fallback YAML parser implementation
  - Replace `DefaultsConfig`, `DownloadConfiguration`, `ValidationConfig`, `LoggingConfiguration` dataclasses with Pydantic `BaseModel` subclasses
  - Use Pydantic's `model_validate`, environment merging, and JSON Schema generation
  - Keep `yaml.safe_load` as the only YAML parser

- **Centralize optional dependency handling** (new `optdeps.py`)
  - Create single module with `get_pystow()`, `get_rdflib()`, `get_pronto()`, `get_owlready2()` functions
  - Move stub implementations from `core.py`, `resolvers.py`, `validators.py` to `optdeps.py`
  - Reduce duplication from 3 separate stub implementations to 1 centralized module

- **Extract CLI formatting utilities** (new `cli_utils.py`)
  - Move `_format_table()` and `_format_row()` from `cli.py` to reusable module
  - Add `_format_validation_summary()` to same module for consistency

### Robustness & Safety (minimal code, maximum impact)

- **Add HEAD request with media-type validation** (`download.py`)
  - Modify `StreamingDownloader.__call__` to issue HEAD request before full GET
  - Check `Content-Type` header matches `FetchPlan.media_type` when specified
  - Retrieve `Content-Length` early to fail fast on oversized files
  - Log warnings for media-type mismatches, allow override via config flag

- **Implement per-service rate limits** (`download.py`, `config.py`)
  - Extend `TokenBucket` to support service keys ("obo", "ols", "bioportal")
  - Add `defaults.http.rate_limits` config section with per-service overrides
  - Extend rate limit parser to accept `/min`, `/hour` units (e.g., "2/s", "5/min", "1/hour")
  - Default to per-host limit when service-specific limit not configured

- **Harden URL validation with allowlist and IDN handling** (`download.py`)
  - Add optional `allowed_hosts` parameter to `validate_url_security()`
  - Implement punycode normalization for Internationalized Domain Names
  - Check resolved host against allowlist when provided
  - Reject URLs not meeting allowlist criteria with clear error message

- **Safe tar.gz/tar.xz extraction** (`download.py`)
  - Create `extract_tar_safe()` function mirroring `extract_zip_safe()` logic
  - Validate all tar member paths for traversal attacks before extraction
  - Apply same compression ratio checks as ZIP extraction
  - Support XBRL and regulatory bundles that ship as tarballs

- **Deterministic normalization with stable hashing** (`validators.py`, `core.py`)
  - Implement TTL canonicalization: sort prefixes and triples before serialization
  - Compute SHA-256 hash of normalized (canonical) TTL content
  - Add `normalized_sha256` and `fingerprint` fields to `Manifest` dataclass
  - Write these fields during `_write_manifest()` for cache correctness

- **Subprocess isolation for memory-intensive validators** (`validators.py`)
  - Modify `validate_pronto()` and `validate_owlready2()` to execute in subprocess
  - Pass file path and temp output directory as subprocess arguments
  - Read validation results back from JSON written by subprocess
  - Preserve existing timeout and memory monitoring in subprocess context

- **SPDX license normalization** (`resolvers.py`, `core.py`)
  - Create `normalize_license_to_spdx()` function mapping common variants to SPDX IDs
  - Update `_ensure_license_allowed()` to compare normalized SPDX identifiers
  - Support common variations: "CC-BY" → "CC-BY-4.0", "CC0" → "CC0-1.0", etc.

### Capability Upgrades (extend coverage without breaking changes)

- **Automatic multi-resolver fallback** (`core.py`)
  - Create `FallbackResolver` class wrapping existing resolvers
  - Implement automatic fallback through `prefer_source` order on `ResolverError`
  - Integrate into `fetch_one()` transparently without config changes
  - Log fallback attempts with source sequence and success/failure per attempt

- **Polite API headers for resolver requests** (`resolvers.py`, `config.py`)
  - Add `defaults.http.polite_headers` config section
  - Include `User-Agent`, `From`, `X-Request-ID` headers in OLS/BioPortal API calls
  - Reduce throttling and improve reproducibility of API interactions
  - Default to project-appropriate headers when not configured

- **LOV and Ontobee resolver implementations** (`resolvers.py`)
  - Implement `LOVResolver` for Linked Open Vocabularies SKOS/TTL sources
  - Implement `OntobeeResolver` for OBO domain ontology PURLs
  - Register both in `RESOLVERS` dictionary
  - Document resolver usage in example `sources.yaml`

- **fsspec-backed remote storage support** (`core.py`, `config.py`)
  - Add `ONTOFETCH_STORAGE_URL` environment variable (e.g., `s3://bucket/prefix`)
  - Implement `fsspec`-based manifest and artifact read/write when URL provided
  - Maintain backward compatibility: default to local `pystow` storage
  - Enable team-wide cache sharing for multi-user pipelines

- **Enhanced CLI with plan, doctor, dry-run** (`cli.py`)
  - Add `ontofetch plan <id>` subcommand: print `FetchPlan` JSON without downloading
  - Add `ontofetch doctor` subcommand: check credentials, filesystem permissions, emit diagnostics
  - Add `--dry-run` flag to `pull` subcommand: log planned actions without execution
  - Improve operational visibility and troubleshooting

## Impact

### Affected Specs

- `ontology-downloader` (multiple requirements modified and added)

### Affected Code

- `src/DocsToKG/OntologyDownload/config.py` - **MODIFIED**: Pydantic models, rate limit config, polite headers
- `src/DocsToKG/OntologyDownload/core.py` - **MODIFIED**: FallbackResolver, fsspec storage, manifest fields
- `src/DocsToKG/OntologyDownload/download.py` - **MODIFIED**: HEAD requests, per-service rate limits, tar extraction, URL validation
- `src/DocsToKG/OntologyDownload/validators.py` - **MODIFIED**: subprocess isolation, deterministic normalization
- `src/DocsToKG/OntologyDownload/resolvers.py` - **MODIFIED**: LOV/Ontobee resolvers, polite headers, SPDX normalization
- `src/DocsToKG/OntologyDownload/cli.py` - **MODIFIED**: plan/doctor commands, dry-run, extracted formatting
- `src/DocsToKG/OntologyDownload/optdeps.py` - **NEW**: centralized optional dependency handling
- `src/DocsToKG/OntologyDownload/cli_utils.py` - **NEW**: extracted CLI formatting utilities
- `tests/ontology_download/*.py` - **MODIFIED/NEW**: comprehensive test coverage for all new behaviors

### Breaking Changes

None. All changes are backward compatible:

- Existing `sources.yaml` files continue to work
- New config fields are optional with sensible defaults
- CLI commands remain unchanged, new subcommands are additive
- Manifest schema is extended (added fields), not modified
- Resolver interface unchanged, new resolvers are additions

### Migration Path

No migration required. Optional enhancements can be adopted incrementally:

1. Users can add `rate_limits` section to config when needed
2. Teams can set `ONTOFETCH_STORAGE_URL` to enable remote storage
3. New resolvers and CLI commands available immediately without config changes
4. Pydantic validation provides clearer error messages automatically
