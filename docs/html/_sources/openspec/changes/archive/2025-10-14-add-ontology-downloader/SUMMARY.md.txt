# OpenSpec Change Proposal: Ontology Downloader

## Status

✅ **Validated** - Change proposal passes `openspec validate --strict`

## Overview

This change proposal creates a comprehensive, production-ready ontology downloader for DocsToKG that can fetch, validate, and normalize ontologies from multiple heterogeneous sources using battle-tested Python libraries.

## Change ID

`add-ontology-downloader`

## Metrics

- **Requirements**: 20 ADDED requirements
- **Scenarios**: 96 test scenarios
- **Tasks**: 151 implementation tasks across 20 phases
- **Affected Capabilities**: 1 new capability (`ontology-downloader`)

### Gap Analysis Enhancements

Following comprehensive gap analysis, added 7 requirements with 44 scenarios covering:

- Comprehensive error handling and recovery (10 scenarios)
- Security and integrity validation (7 scenarios)
- Performance constraints and resource management (5 scenarios)
- Enhanced configuration schema (6 scenarios)
- Multi-level logging with sensitive data protection (6 scenarios)
- Cross-platform compatibility (5 scenarios)
- User documentation and help (5 scenarios)

Added 57 implementation tasks (sections 13-20) with explicit instructions for AI agents including function signatures, error message formats, platform-specific code, and test data requirements.

## Key Features

### 1. Source-Agnostic Resolver Registry

Supports 5 resolver types:

- **OBO Library** (via Bioregistry) - for OBO Foundry ontologies (GO, HP, UBERON, etc.)
- **OLS4** (via ols-client) - for EMBL-EBI Ontology Lookup Service
- **BioPortal/OntoPortal** (via ontoportal-client) - for NCBO ontologies
- **SKOS/RDF** (direct URLs) - for legal/financial thesauri (EuroVoc, etc.)
- **XBRL** (taxonomy packages) - for financial reporting standards

### 2. Robust HTTP Handling

- ETag/Last-Modified conditional requests (HTTP 304 caching)
- Range header resume for interrupted downloads
- SHA-256 checksum verification
- Exponential backoff retry (5 attempts, 0.5s base)
- Per-host rate limiting (4 req/sec default)
- Stream-to-.part pattern to avoid corrupted partials

### 3. Multi-Parser Validation

- **RDFLib**: RDF/OWL/SKOS parsing, Turtle normalization
- **Pronto**: OBO/OWL validation, OBO Graph JSON export
- **Owlready2**: OWL reasoning checks
- **ROBOT** (optional): format conversions and SPARQL QC
- **Arelle**: XBRL taxonomy validation

### 4. Comprehensive Provenance

Each downloaded ontology gets a `manifest.json` with:

- Source URL, resolver, version, license
- ETag, Last-Modified, SHA-256 hash
- Timestamps, file sizes, validation status
- Target formats and validation outcomes

### 5. Declarative Configuration

YAML-based `sources.yaml` with:

- Global defaults (license allowlists, format preferences, HTTP params)
- Per-ontology specifications (ID, resolver, formats, extras)
- License compliance enforcement (fail-closed for restricted)

### 6. pystow-Based Storage

Organized directory structure at `~/.data/ontology-fetcher`:

```
configs/sources.yaml
cache/                      # HTTP cache, ETags
logs/                       # Structured JSON logs
ontologies/<id>/<version>/
  original/                 # Bit-exact downloaded files
  normalized/               # Turtle, OBO Graph JSON
  validation/               # Parser results, QC reports
  manifest.json             # Provenance metadata
```

### 7. CLI Interface

`ontofetch` command with subcommands:

- `pull [--spec FILE | ID...]` - Download ontologies
- `show ID [--versions]` - Display manifests
- `validate ID[@VERSION]` - Re-run validators
- `--json` flag for machine-readable output
- `--force` flag to bypass cache

### 8. Observability

Structured JSON logging captures:

- Resolver planning time and URLs
- HTTP status, elapsed time, retries, cache hits
- Validation outcomes and error messages
- Per-run summaries (success/failure/cached counts)

## Additional Requirements from Gap Analysis

### 7. Comprehensive Error Handling and Recovery

- Network failures with retry logic and exponential backoff
- Invalid credentials for restricted ontologies with remediation paths
- Resolver API unavailability with graceful degradation
- Disk space exhaustion with cleanup and clear errors
- Corrupted downloads detected via checksum with automatic retry
- Parser failures with detailed error capture
- Configuration file errors with helpful messages
- Permission errors with PYSTOW_HOME guidance
- Timeouts on slow APIs with retry
- Memory limit exceeded during large file parsing

### 8. Security and Integrity Validation

- HTTPS enforcement with upgrade from HTTP
- TLS certificate verification enabled by default
- URL validation preventing SSRF to private IPs
- API key masking in all logs
- Safe filename sanitization preventing directory traversal
- Maximum file size limit enforcement
- ZIP bomb detection for XBRL taxonomies

### 9. Performance Constraints and Resource Management

- HTTP request timeouts (30s API, 300s download)
- Concurrent download limits with semaphore
- Memory limit configuration for parsers
- Streaming downloads for large files (>100MB)
- Parser timeouts to prevent hanging (60s default)

### 10. Enhanced Configuration Schema

- All HTTP parameters configurable (retries, timeouts, backoff, rate limits)
- Default values with explicit documentation
- Environment variable overrides (ONTOFETCH_* pattern)
- Per-ontology timeout overrides
- Configuration validation with clear error messages
- Required resolver parameters validation

### 11. Multi-Level Logging with Sensitive Data Protection

- Log levels (DEBUG, INFO, WARNING, ERROR) with appropriate verbosity
- Structured JSON logs with parseable fields
- Sensitive data masking (API keys, tokens, passwords)
- Correlation IDs for batch operation tracing
- Log rotation with compression and retention policies
- Machine-readable logs for aggregators (ELK, Splunk)

### 12. Cross-Platform Compatibility

- Python 3.9+ requirement check at startup
- pathlib.Path for platform-agnostic paths
- Platform-specific timeout implementations (signal vs threading)
- Dependency version constraints enforced
- Tested on Linux, macOS, Windows

### 13. User Documentation and Help

- Comprehensive CLI help text with examples
- Subcommand-specific help
- Error messages with remediation steps
- `ontofetch init` command for example configuration
- Troubleshooting documentation
- Configuration schema documentation with inline comments
- YAML parse errors with line numbers

## Implementation Phases

1. **Setup and Dependencies** (4 tasks)
2. **Core Data Models** (5 tasks)
3. **Resolver Implementations** (7 tasks)
4. **Download Manager** (10 tasks)
5. **Validation Pipeline** (11 tasks)
6. **Storage and Manifests** (6 tasks)
7. **Configuration** (6 tasks)
8. **Orchestration** (5 tasks)
9. **CLI** (10 tasks)
10. **Testing** (13 tasks)
11. **Documentation** (8 tasks)
12. **Integration and Polish** (10 tasks)
13. **Error Handling and Recovery** (10 tasks)
14. **Security Implementation** (8 tasks)
15. **Performance and Resource Management** (6 tasks)
16. **Enhanced Configuration** (5 tasks)
17. **Enhanced Logging System** (6 tasks)
18. **Cross-Platform Compatibility** (5 tasks)
19. **User Documentation and Help** (7 tasks)
20. **Additional Testing for New Requirements** (10 tasks)

## Dependencies (All Available via pip)

### Discovery/Access

- `bioregistry` - Prefix resolution and download URLs
- `oaklib` - Universal ontology adapter
- `ols-client` - OLS4 API client
- `ontoportal-client` - BioPortal/OntoPortal API client

### Parsing/Normalization

- `rdflib` - RDF/OWL/SKOS parsing
- `pronto` - OBO/OWL validation
- `owlready2` - OWL reasoning
- `robot` (optional, Java) - Conversions and QC

### Specialized Domains

- `arelle` - XBRL taxonomy validation

### Infrastructure

- `pystow` - Data directory management
- `pooch` - Download caching and hashing

## Design Decisions

### Key Architectural Choices

1. **Use battle-tested libraries** over custom implementations (leverage community expertise)
2. **Protocol-based resolver registry** for extensibility and testability
3. **Store original + normalized** formats for reproducibility and tooling
4. **pooch for downloads** (hash verification built-in)
5. **ols-client over ebi-ols-client** (per user specification)
6. **ROBOT optional** (requires Java runtime)
7. **Fail-closed license enforcement** (explicit allowlist required)

### Trade-offs

- **Memory**: Large ontologies (SNOMED, NCIT) may exhaust memory during reasoning → document limits, skip optional steps
- **API Changes**: External services may break → use maintained clients, monitor upstream
- **Java Dependency**: ROBOT requires JVM → make optional with runtime detection
- **Rate Limits**: Batch downloads may hit limits → exponential backoff, per-host throttling

## Files Created

```
openspec/changes/add-ontology-downloader/
├── proposal.md              # Why, what, impact (this change)
├── design.md                # Technical decisions and architecture
├── tasks.md                 # 151 implementation tasks
├── specs/
│   └── ontology-downloader/
│       └── spec.md          # 20 requirements, 96 scenarios
└── SUMMARY.md               # This file
```

## Next Steps

### Before Implementation (Required)

1. ⚠️ **Request approval** - Do not start implementation until proposal is reviewed and approved
2. Review open questions in `design.md`
3. Clarify any ambiguous requirements

### During Implementation

1. Follow tasks sequentially in `tasks.md`
2. Mark tasks complete immediately with `- [x]`
3. Run `openspec validate add-ontology-downloader --strict` periodically
4. Write tests as you implement features (not after)

### After Implementation

1. Run integration smoke tests with real APIs
2. Create separate PR to archive change:
   - Move to `changes/archive/YYYY-MM-DD-add-ontology-downloader/`
   - Update `specs/ontology-downloader/spec.md` with final spec
   - Use `openspec archive add-ontology-downloader`

## Validation

```bash
# Validate change proposal
openspec validate add-ontology-downloader --strict

# View change details
openspec show add-ontology-downloader

# View delta JSON
openspec show add-ontology-downloader --json --deltas-only
```

## Reference Materials

The design is based on the detailed implementation blueprint provided, which includes:

- Concrete code skeletons using the specified libraries
- Example workflows for HP (OBO), EFO (OLS), NCIT (BioPortal), EuroVoc (SKOS), IFRS (XBRL)
- Testing strategy with unit/integration/smoke tests
- Security considerations (zip-slip, license compliance, safe extraction)
- Observability patterns (structured logs, provenance manifests)

## Questions/Clarifications

See **Open Questions** section in `design.md` for items requiring stakeholder input:

- Parallel vs. serial downloads for batch operations
- Version retention policies
- Validation failure handling (block vs. warn)
- OAK adapter integration
- Programmatic API exposure timing
- Authentication patterns for restricted ontologies

---

**Generated**: 2025-10-14
**Validated**: ✅ Passes `openspec validate --strict`
**Status**: Awaiting approval to begin implementation
