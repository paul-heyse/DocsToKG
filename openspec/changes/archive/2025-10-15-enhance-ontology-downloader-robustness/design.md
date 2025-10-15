# Design Document: Ontology Downloader Enhancements

## Context

The ontology downloader is a critical component of the DocsToKG pipeline responsible for fetching, validating, and caching ontology documents from multiple sources (OBO, OLS, BioPortal, SKOS, XBRL). The current implementation has proven functional but carries technical debt in configuration parsing, lacks hardening for production edge cases, and misses capabilities that would improve operational success rates.

This enhancement initiative addresses 15 identified improvements across code quality, robustness, and capabilities without breaking existing workflows or requiring user migration.

### Stakeholders

- **Pipeline Operators**: Need reliable downloads, clear diagnostics, and graceful failure handling
- **Development Team**: Need maintainable code, clear error messages, and testable components
- **End Users**: Need fast, cached downloads with clear progress and error reporting

### Constraints

- **Backward Compatibility**: Existing `sources.yaml` files and manifest.json files must continue working
- **Optional Dependencies**: Cannot require new hard dependencies (fsspec, s3fs remain optional)
- **Cross-Platform**: Must work on Linux, macOS, Windows with Python 3.9+
- **Performance**: Cannot regress download or validation performance

## Goals / Non-Goals

### Goals

1. Reduce code maintenance burden by eliminating custom parsers and consolidating stubs
2. Harden download security with better validation and safe archive handling
3. Improve operational success rates with fallback resolvers and better rate limiting
4. Enable team workflows with remote storage support
5. Improve troubleshooting with diagnostic CLI commands
6. Maintain 100% backward compatibility

### Non-Goals

1. Rewriting core download logic (existing implementation is solid)
2. Changing resolver API contracts (keep protocol stable)
3. Modifying manifest schema in breaking ways (only additions)
4. Adding new required dependencies (keep them optional)
5. Supporting non-Python configuration formats

## Decisions

### 1. Configuration: Pydantic v2 vs. Custom Dataclasses

**Decision**: Migrate to Pydantic v2 models

**Rationale**:

- Eliminates ~150 lines of custom YAML fallback parser
- Provides superior validation with clear error messages
- Enables JSON Schema generation for documentation
- Handles environment variable merging elegantly
- Industry standard with excellent type inference support

**Alternatives Considered**:

- Keep custom dataclasses: Maintains status quo but keeps maintenance burden
- Use attrs with cattrs: Less common, weaker ecosystem
- Use marshmallow: Heavier dependency, more verbose schemas

**Implementation**:

```python
from pydantic import BaseModel, Field, field_validator

class DownloadConfiguration(BaseModel):
    max_retries: int = Field(default=5, ge=0)
    timeout_sec: int = Field(default=30, gt=0)
    per_host_rate_limit: str = Field(default="4/second", pattern=r"^\d+/(second|sec|s|minute|min|m|hour|h)$")

    @field_validator('per_host_rate_limit')
    def validate_rate_limit(cls, v):
        # Parse and validate rate limit format
        pass
```

### 2. Optional Dependencies: Centralized Module vs. Inline Stubs

**Decision**: Create centralized `optdeps.py` module

**Rationale**:

- Eliminates duplication across 3 modules (core, resolvers, validators)
- Single source of truth for optional dependency behavior
- Easier to test stub behavior in isolation
- Cleaner import statements in consuming modules

**Alternatives Considered**:

- Keep inline stubs: Maintains status quo, higher duplication
- Use try/except at import sites: More scattered, harder to test
- Make all dependencies required: Breaks lightweight installations

**Implementation**:

```python
# optdeps.py
_pystow = None

def get_pystow():
    global _pystow
    if _pystow is None:
        try:
            import pystow
            _pystow = pystow
        except ModuleNotFoundError:
            _pystow = _PystowFallback()
    return _pystow
```

### 3. Rate Limiting: Per-Host vs. Per-Service

**Decision**: Support both per-host (default) and per-service (opt-in)

**Rationale**:

- Different services have different rate limit policies
- OBO PURLs, OLS API, BioPortal API all share hosts but need separate limits
- Backward compatible: default to per-host when service not specified
- Prevents one noisy service from starving others

**Alternatives Considered**:

- Per-host only: Simpler but insufficient for multi-service hosts
- Per-service only: Breaks backward compatibility
- Global rate limit: Too coarse, would harm performance

**Implementation**:

```python
# config.yaml
defaults:
  http:
    per_host_rate_limit: "4/second"  # fallback
    rate_limits:
      obo: "2/second"
      ols: "5/second"
      bioportal: "1/second"

# download.py
def _get_bucket(host: str, service: Optional[str], config):
    key = f"{service}:{host}" if service else host
    if key not in _TOKEN_BUCKETS:
        rate = config.defaults.http.rate_limits.get(service) or config.defaults.http.per_host_rate_limit
        _TOKEN_BUCKETS[key] = TokenBucket(rate)
    return _TOKEN_BUCKETS[key]
```

### 4. Validator Isolation: Subprocess vs. Separate Process Pool

**Decision**: Use subprocess.run for heavy validators (pronto, owlready2)

**Rationale**:

- Prevents memory fragmentation in long-lived batch processes
- Subprocess lifecycle matches validator execution scope
- Simpler than maintaining a separate process pool
- Compatible with existing timeout mechanisms

**Alternatives Considered**:

- Keep in-process: Status quo, memory fragmentation issues
- Multiprocessing.Pool: Overkill, adds complexity for worker management
- Docker containers: Too heavy, requires container runtime

**Implementation**:

```python
def _run_pronto_in_subprocess(file_path: Path, output_dir: Path, timeout: int):
    script = f"""
import json
from pronto import Ontology
ontology = Ontology('{file_path}')
result = {{"ok": True, "terms": len(list(ontology.terms()))}}
(Path('{output_dir}') / 'result.json').write_text(json.dumps(result))
"""
    subprocess.run([sys.executable, '-c', script], timeout=timeout, check=True)
    return json.loads((output_dir / 'result.json').read_text())
```

### 5. Storage Backend: Direct fsspec vs. Abstraction Layer

**Decision**: Create thin abstraction layer with Local and Fsspec backends

**Rationale**:

- Keeps fsspec optional (not everyone needs remote storage)
- Provides clean migration path from pystow
- Allows testing with mock backends
- Enables future backends (e.g., HTTP, GCS) without changing call sites

**Alternatives Considered**:

- Direct fsspec everywhere: Makes it a hard dependency
- Keep pystow-only: Doesn't solve team-wide cache problem
- Build custom S3 client: Reinventing wheels

**Implementation**:

```python
class StorageBackend(Protocol):
    def read_manifest(self, ontology_id: str, version: str) -> dict: ...
    def write_manifest(self, ontology_id: str, version: str, manifest: Manifest): ...
    def read_artifact(self, path: str) -> bytes: ...
    def write_artifact(self, path: str, content: bytes): ...

class LocalStorageBackend:
    # Wraps current pystow-based logic

class FsspecStorageBackend:
    # Uses fsspec for remote URLs
```

### 6. Resolver Fallback: Wrapper vs. Modified fetch_one

**Decision**: Create `FallbackResolver` wrapper class

**Rationale**:

- Cleaner separation of concerns
- Testable in isolation
- Doesn't pollute fetch_one with retry logic
- Allows selective enable/disable via config

**Alternatives Considered**:

- Modify fetch_one directly: Adds complexity to already-complex function
- Client-side retry: Pushes responsibility to callers
- No fallback: Status quo, lower success rates

**Implementation**:

```python
class FallbackResolver:
    def __init__(self, resolvers: List[BaseResolver], spec: FetchSpec, config: ResolvedConfig):
        self.resolvers = resolvers
        self.spec = spec
        self.config = config

    def plan(self, logger: logging.Logger) -> FetchPlan:
        for resolver in self.resolvers:
            try:
                return resolver.plan(self.spec, self.config, logger)
            except (ResolverError, ConfigError) as e:
                logger.warning(f"Resolver {resolver} failed, trying next: {e}")
        raise ResolverError("All resolvers exhausted")
```

### 7. License Normalization: Inline Mapping vs. External SPDX Library

**Decision**: Inline mapping table for common variants

**Rationale**:

- Limited set of licenses used in ontology domain
- Avoids dependency on full SPDX library
- Faster and more predictable
- Easy to extend for domain-specific variants

**Alternatives Considered**:

- Use license-expression library: Heavyweight for simple use case
- Use spdx library: Overkill, slow parsing
- No normalization: Status quo, fragile string matching

**Implementation**:

```python
LICENSE_SPDX_MAPPING = {
    "CC-BY": "CC-BY-4.0",
    "CC BY 4.0": "CC-BY-4.0",
    "CC0": "CC0-1.0",
    "Public Domain": "CC0-1.0",
    "Apache": "Apache-2.0",
    "Apache License 2.0": "Apache-2.0",
    # ... domain-specific variants
}

def normalize_license_to_spdx(license_str: Optional[str]) -> Optional[str]:
    if not license_str:
        return None
    normalized = LICENSE_SPDX_MAPPING.get(license_str.strip())
    return normalized or license_str  # Return original if no mapping
```

### 8. CLI Commands: Subcommands vs. Flags

**Decision**: Add plan and doctor as new subcommands

**Rationale**:

- Consistent with existing CLI structure (pull, show, validate)
- Clear command intent and help text
- Easier to add command-specific options later
- Standard argparse pattern

**Alternatives Considered**:

- Flags on pull command: Clutters pull options, unclear semantics
- Separate scripts: Fragmented UX, harder to discover
- Interactive menu: Overkill for CLI tool

## Risks / Trade-offs

### Risk: Pydantic v2 Breaking Changes

**Impact**: Medium - Users on older Pydantic v1 installations may see conflicts
**Mitigation**: Pin pydantic>=2.0.0 in requirements, document upgrade path
**Likelihood**: Low - Pydantic v2 has been stable since 2023

### Risk: Subprocess Overhead for Validators

**Impact**: Low - Slight performance overhead (process spawn)
**Mitigation**: Only use subprocess for heavy validators (pronto, owlready2), keep rdflib in-process
**Likelihood**: High - Overhead will occur but is acceptable trade-off for memory safety

### Risk: fsspec Optional Dependency Confusion

**Impact**: Low - Users may not realize they need fsspec for remote storage
**Mitigation**: Clear error message when ONTOFETCH_STORAGE_URL set but fsspec missing
**Likelihood**: Medium - Will affect users trying remote storage

### Risk: Backward Compatibility Edge Cases

**Impact**: Low - Existing manifest.json files with unexpected structure
**Mitigation**: Thorough testing with real-world manifests, graceful field addition
**Likelihood**: Low - Manifest schema is stable

### Trade-off: Complexity vs. Capability

Adding 15 improvements increases overall code complexity. We mitigate by:

- Keeping each feature modular and independently testable
- Maintaining clear separation of concerns
- Documenting each feature thoroughly
- Making advanced features opt-in

### Trade-off: Performance vs. Safety

Subprocess isolation adds overhead but prevents memory leaks. We mitigate by:

- Only using subprocess for validators known to leak (pronto, owlready2)
- Keeping fast validators (rdflib) in-process
- Caching subprocess results to avoid re-execution

## Migration Plan

### Phase 1: Internal Testing (Week 1)

1. Merge code changes to feature branch
2. Run full test suite on CI (Linux/macOS/Windows)
3. Test with representative sources.yaml files
4. Benchmark performance against baseline

### Phase 2: Alpha Testing (Week 2)

1. Deploy to staging environment
2. Run batch pull on 50+ real ontologies
3. Monitor logs for Pydantic validation issues
4. Monitor memory usage with subprocess validators
5. Test new CLI commands (plan, doctor)

### Phase 3: Beta Release (Week 3)

1. Tag release candidate
2. Deploy to production with canary group
3. Monitor error rates and fallback resolver usage
4. Collect user feedback on new features

### Phase 4: General Availability (Week 4)

1. Release to all users
2. Update documentation
3. Monitor adoption of new config options
4. Provide support for migration issues

### Rollback Plan

If critical issues discovered:

1. Revert to previous version via git tag
2. Restore old configuration parsing if Pydantic causes issues
3. Disable subprocess validators via config flag
4. Document known issues and workarounds

## Open Questions

1. **Q**: Should we make rate limit parsing support decimal values (e.g., "0.5/second")?
   **A**: Yes, useful for very slow APIs. Add float parsing to rate limit parser.

2. **Q**: Should doctor command auto-fix detected issues?
   **A**: No, only diagnose and suggest. Auto-fix is risky and surprising.

3. **Q**: Should we support multiple storage backends simultaneously?
   **A**: No, single backend per run. Users can switch via env var.

4. **Q**: Should fallback resolver be opt-out or opt-in?
   **A**: Opt-out (enabled by default). Users can disable via config flag if needed.

5. **Q**: Should we validate SPDX identifiers against official SPDX list?
   **A**: Not initially. Simple mapping sufficient for v1, can enhance later.

6. **Q**: Should tar extraction support encrypted archives?
   **A**: No, out of scope. Focus on safety, not advanced tar features.

7. **Q**: Should HEAD requests be retried with same backoff as GET?
   **A**: Yes, for consistency. But log separately to distinguish HEAD vs GET failures.

8. **Q**: Should we add telemetry for fallback success rates?
   **A**: Yes, via structured logging. No external telemetry service.

## Testing Strategy

### Unit Testing Approach

**Coverage Goals**:

- Target: >85% line coverage for all modified/new modules
- Critical paths (download, validation) require >95% coverage
- All public APIs must have unit tests

**Test Organization**:

```
tests/ontology_download/
├── test_optdeps.py              # Optional dependency stubs
├── test_cli_utils.py            # CLI formatting
├── test_config_pydantic.py      # Pydantic models
├── test_download_head.py        # HEAD request validation
├── test_download_rate_limit.py  # Per-service rate limits
├── test_download_security.py    # URL validation, archive safety
├── test_validators_subprocess.py # Subprocess isolation
├── test_normalization.py        # Deterministic TTL
├── test_resolvers_fallback.py   # Fallback resolver
├── test_resolvers_new.py        # LOV, Ontobee
├── test_storage_backends.py     # fsspec storage
└── test_cli_commands.py         # plan, doctor, dry-run
```

**Test Data Requirements**:

- Sample ontology files (10KB-1MB) for validation tests
- Malformed ontologies for error handling tests
- Mock archives (ZIP, tar.gz) with safe and unsafe paths
- Configuration files with valid/invalid schemas

**Mocking Strategy**:

- Use `requests_mock` for HTTP interactions
- Use `pytest.monkeypatch` for filesystem operations
- Mock subprocess.run for validator isolation tests
- Use temporary directories (`pytest tmp_path`) for file operations

### Integration Testing

**End-to-End Scenarios**:

1. Full download pipeline with all new features enabled
2. Multi-resolver fallback with simulated failures
3. Remote storage (S3) with mocked filesystem
4. Batch pull with concurrent downloads and rate limiting
5. Validation pipeline with subprocess isolation

**Integration Test Fixtures**:

- Real ontology samples (HP, EFO from OBO)
- Mock OLS/BioPortal API responses
- Test S3 bucket (localstack or moto)

**Performance Benchmarks**:

- Download 50 ontologies and compare time vs baseline
- Measure memory usage during batch validation
- Benchmark subprocess vs in-process validator overhead
- Validate config parsing time <100ms

### Regression Testing

**Backward Compatibility Tests**:

- Existing sources.yaml files parse correctly
- Old manifest.json files readable with new code
- CLI commands work with existing data directories
- No API contract changes for programmatic usage

**Upgrade Testing**:

- Upgrade from previous version preserves data
- No data loss during Pydantic migration
- Environment variables still override config

### Security Testing

**Threat Scenarios**:

1. SSRF attempts with private IPs, localhost, cloud metadata URLs
2. Path traversal in ZIP/tar archives
3. IDN homograph attacks
4. Compression bombs (ZIP/tar)
5. XXE attacks in XML parsers
6. Injection attacks in subprocess calls

**Security Test Cases**:

```python
def test_ssrf_private_ip_blocked():
    """Test URL validation rejects private IP addresses."""
    with pytest.raises(ConfigError) as exc_info:
        validate_url_security("https://192.168.1.1/ontology.owl")
    assert "private" in str(exc_info.value).lower()

def test_ssrf_localhost_blocked():
    """Test URL validation rejects localhost."""
    with pytest.raises(ConfigError):
        validate_url_security("https://localhost/ontology.owl")

def test_ssrf_cloud_metadata_blocked():
    """Test URL validation rejects cloud metadata endpoints."""
    with pytest.raises(ConfigError):
        validate_url_security("https://169.254.169.254/latest/meta-data/")

def test_zip_path_traversal_blocked():
    """Test ZIP extraction rejects path traversal."""
    # Create ZIP with member path ../../etc/passwd
    # Attempt extraction
    # Assert: raises ConfigError
```

**Fuzzing**:

- Fuzz configuration parser with malformed YAML
- Fuzz URL validator with edge cases
- Fuzz archive extractors with malicious archives

## Security Model

### Threat Model

**Assets**:

- Downloaded ontology data (integrity, availability)
- System resources (CPU, memory, disk, network)
- Credentials (API keys for BioPortal, OLS)
- Host system (prevent arbitrary code execution, file access)

**Threat Actors**:

- Malicious ontology providers (supply chain attack)
- Compromised resolver APIs (serving malicious content)
- Internal misconfigurations (accidental exposure)

**Attack Vectors**:

1. **Server-Side Request Forgery (SSRF)**: Attacker controls ontology URL, attempts to scan internal network
2. **Path Traversal**: Malicious archive extracts files outside intended directory
3. **Denial of Service**: Compression bombs, infinite archives, memory exhaustion
4. **Code Injection**: XXE in XML parsers, command injection in subprocess calls
5. **Credential Theft**: Log file exposure, error message leakage

### Security Controls

**Authentication & Authorization**:

- API keys stored in pystow directory with restrictive permissions (0600)
- Keys never logged (masked with ***masked***)
- Keys not passed via command line (only env vars or files)

**Input Validation**:

- All URLs validated before use (HTTPS only, no private IPs)
- Host allowlist support for restricted environments
- IDN punycode normalization
- Configuration schema validation (Pydantic)
- File size limits enforced early (HEAD request)

**Secure Coding Practices**:

- No shell=True in subprocess calls
- Path traversal checks before file operations
- Compression ratio limits
- Memory limits for parsers
- Timeouts on all network operations

**Logging & Monitoring**:

- Sensitive data masked in logs
- Security events logged at WARNING level
- Structured logs enable SIEM integration
- Correlation IDs for attack tracking

**Compliance**:

- HTTPS certificate verification enabled (verify=True)
- Follows OWASP secure coding guidelines
- No hardcoded secrets
- Secure defaults (fail closed)

## Performance Optimization

### Current Performance Baseline

**Measured Metrics** (to establish before implementation):

- Average download time per ontology: ~X seconds
- Peak memory usage during batch: ~Y MB
- Config parsing time: ~Z ms
- Validation time per ontology: ~W seconds

### Optimization Techniques

**1. Lazy Loading**:

- Optional dependencies loaded on first use (not at import)
- Pystow initialized only when needed
- Validator imports deferred until validation requested

**2. Caching**:

- DNS resolution cached (already handled by requests)
- TokenBucket instances cached per host/service
- Pydantic model validation results cached where safe

**3. Subprocess Optimization**:

- Subprocess spawn overhead measured (~10-50ms acceptable)
- Reuse interpreter for multiple validations if beneficial
- JSON serialization for subprocess communication (faster than pickle)

**4. Network Optimization**:

- HEAD request used to avoid unnecessary downloads
- HTTP/2 connection reuse (via session)
- Chunked streaming (1MB chunks) for large files
- Conditional requests (ETag, Last-Modified)

**5. I/O Optimization**:

- Asynchronous writes where possible
- Buffered I/O for large files
- Compression for log rotation

**6. Parallelization**:

- Concurrent downloads (configurable workers)
- Independent validation tasks can run in parallel
- Per-service rate limits prevent bottlenecks

### Performance Monitoring

**Instrumentation Points**:

- Download start/end with elapsed time
- Validation start/end per validator
- Subprocess spawn/exit timing
- Memory usage samples (before/after heavy operations)

**Metrics to Track**:

- Downloads per second
- Average download time
- P95/P99 download latency
- Memory high-water mark
- Subprocess overhead percentage
- Config parsing time
- Fallback resolver success rate

## Standards Compliance

### HTTP Standards (RFC 7230-7235)

**Compliance Requirements**:

- Proper HTTP method usage (HEAD, GET)
- Standard headers (User-Agent, Accept, Authorization)
- Status code handling (2xx, 3xx, 4xx, 5xx)
- Connection reuse (Keep-Alive)
- Conditional requests (If-None-Match, If-Modified-Since)

**Testing**:

- Validate headers match RFC specifications
- Test redirect handling (301, 302, 307, 308)
- Verify timeout behavior matches RFC recommendations

### RDF Standards (W3C)

**Compliance Requirements**:

- Turtle serialization follows W3C Turtle spec
- RDF/XML parsing compatible with RDF 1.1
- Namespaces handled correctly
- Blank nodes serialized consistently

**Testing**:

- Validate output with RDF validators
- Test round-trip (parse → serialize → parse)
- Verify namespace preservation

### OBO Format Standards

**Compliance Requirements**:

- OBO format parsing follows spec
- OBO Graphs JSON matches schema
- Cross-references preserved
- Ontology metadata captured

**Testing**:

- Parse reference OBO files
- Validate against OBO format validator
- Check metadata extraction completeness

### Security Standards

**OWASP Top 10 Coverage**:

- Injection: Prevented via input validation, no shell=True
- Broken Authentication: Keys stored securely, never logged
- Sensitive Data Exposure: Masking in logs, restricted file permissions
- XXE: rdflib configured with safe defaults
- Security Misconfiguration: Secure defaults, explicit validation
- SSRF: URL validation, private IP blocking

**TLS/SSL**:

- TLS 1.2+ required
- Certificate validation enabled
- No insecure ciphers

## Error Handling Patterns

### Error Classification

**Transient Errors** (retry with backoff):

- Network timeouts
- HTTP 429 (rate limit), 503 (service unavailable)
- Connection resets
- DNS resolution failures

**Permanent Errors** (fail immediately):

- HTTP 404 (not found), 403 (forbidden), 401 (unauthorized)
- Invalid configuration
- Malformed URLs
- Missing required fields

**Resource Errors** (graceful degradation):

- Disk space exhausted
- Memory limits exceeded
- File permission denied

**Validation Errors** (log and continue):

- Parser failures
- Malformed ontologies
- Validation timeouts

### Error Handling Strategy

**At Download Layer**:

```python
try:
    result = download_stream(...)
except ConfigError as e:
    # Permanent error, log and fail
    logger.error("Download configuration error", extra={"error": str(e)})
    raise
except requests.Timeout as e:
    # Transient, retry with backoff
    if attempt < max_retries:
        time.sleep(backoff_factor * (2 ** attempt))
        continue
    raise
except OSError as e:
    # Resource error, check specific cause
    if "No space left" in str(e):
        logger.critical("Disk full", extra={"path": str(destination)})
        raise SystemExit(1)
    raise
```

**At Validation Layer**:

```python
try:
    result = validate_rdflib(...)
except ValidationTimeout:
    # Log timeout, record in manifest, continue
    logger.warning("Validation timeout", extra={"validator": "rdflib"})
    return ValidationResult(ok=False, details={"error": "timeout"})
except MemoryError:
    # Log memory issue, record in manifest, continue
    logger.warning("Memory limit exceeded", extra={"validator": "rdflib"})
    return ValidationResult(ok=False, details={"error": "memory"})
except Exception as e:
    # Unexpected error, log full traceback, continue
    logger.exception("Validation error", extra={"validator": "rdflib"})
    return ValidationResult(ok=False, details={"error": str(e)})
```

**At Batch Layer**:

- Collect errors for all ontologies
- Continue processing remaining items
- Report summary at end
- Return non-zero exit code if any failures

### Error Messages

**Requirements**:

- Clear description of what went wrong
- Context (URL, file, ontology ID)
- Actionable remediation steps
- No sensitive data (URLs ok, API keys not ok)

**Examples**:

```
Good: "BioPortal API key not found. Configure at ~/.data/ontology-fetcher/configs/bioportal_api_key.txt"
Bad:  "API error"

Good: "File exceeds size limit: 6.2GB > 5.0GB. Increase max_download_size_gb in config."
Bad:  "File too large"

Good: "URL validation failed: host 192.168.1.1 is private. Set allowed_hosts to override."
Bad:  "Invalid URL"
```

## Scalability and Future-Proofing

### Scalability Considerations

**Horizontal Scaling**:

- Remote storage (S3) enables multiple workers sharing cache
- Stateless design (no global state) allows parallelization
- Independent per-ontology operations

**Vertical Scaling**:

- Configurable concurrent downloads
- Memory limits prevent resource exhaustion
- Subprocess isolation bounds memory growth

**Data Volume**:

- Designed for 100s-1000s of ontologies
- Streaming download for large files (>1GB)
- Incremental validation (cache validation results)

### Extensibility Points

**Adding New Resolvers**:

```python
class MyResolver(BaseResolver):
    def plan(self, spec, config, logger):
        # Custom resolution logic
        return self._build_plan(url="...")

# Register in resolvers.py
RESOLVERS["my_resolver"] = MyResolver()
```

**Adding New Validators**:

```python
def validate_custom(request, logger):
    # Custom validation logic
    return ValidationResult(ok=True, details={})

# Register in validators.py
VALIDATORS["custom"] = validate_custom
```

**Adding New Storage Backends**:

```python
class MyStorageBackend:
    def read_manifest(...): ...
    def write_manifest(...): ...
    def read_artifact(...): ...
    def write_artifact(...): ...

# Use in core.py
if STORAGE_URL.startswith("my://"):
    storage = MyStorageBackend(STORAGE_URL)
```

### API Stability

**Versioning**:

- Public API (`fetch_one`, `fetch_all`) remains stable
- Internal APIs can change between minor versions
- Configuration schema follows semantic versioning
- Manifest format additions only (no removals)

**Deprecation Policy**:

- Deprecated features marked in docs for 2 minor versions
- Breaking changes only in major versions
- Migration guides provided for all breaking changes

### Future Enhancements (Out of Scope)

**Not in This Change**:

- GraphQL API for ontology search
- Web UI for configuration
- Automatic ontology discovery
- Ontology merging/alignment
- Semantic search over ontologies
- Real-time update subscriptions

**Potential Future Work**:

- Support for additional formats (JSON-LD, N-Triples)
- Ontology versioning and diff
- Dependency resolution (import chains)
- Quality metrics (completeness, consistency)
- Integration with ontology registries

## User Feedback and Continuous Improvement

### Feedback Channels

**Issue Tracking**:

- GitHub issues for bug reports
- Feature request template
- Bug report template with environment info

**Telemetry** (opt-in only):

- Anonymous usage statistics (ontology counts, resolver usage)
- Error rate by resolver
- Performance metrics (download speed, validation time)
- **Privacy**: No ontology IDs, no URLs, no user data

**Community Input**:

- Monthly triage of GitHub issues
- Quarterly roadmap updates
- User surveys for feature prioritization

### Continuous Improvement Process

**Performance Monitoring**:

- Weekly review of CI performance benchmarks
- Alert on >10% regression
- Track adoption of new features (plan, doctor commands)

**Error Analysis**:

- Monthly review of error logs
- Identify common failure patterns
- Improve error messages and documentation

**Documentation Feedback**:

- "Was this helpful?" on each doc page
- Track most-visited and least-visited pages
- Update FAQs based on support questions

**Release Cadence**:

- Patch releases: Monthly (bug fixes)
- Minor releases: Quarterly (new features)
- Major releases: Annually (breaking changes)

## Success Metrics

1. **Code Quality**:
   - Lines of code reduced by ~150 (YAML parser removal)
   - Test coverage maintained above 85%
   - Cyclomatic complexity unchanged or improved
   - Zero high-severity linter warnings

2. **Robustness**:
   - Download success rate increases by 5-10% (via fallback)
   - Zero SSRF incidents (via hardened URL validation)
   - Memory usage stable in batch operations (via subprocess isolation)
   - Zero security vulnerabilities in dependency scan

3. **Usability**:
   - Doctor command used in >20% of support cases
   - Plan command used in >10% of workflow development
   - Remote storage adopted by >5% of team installations
   - Average support issue resolution time reduced by 25%

4. **Performance**:
   - Download performance within 5% of baseline
   - Validation performance within 10% of baseline
   - Config parsing <100ms (Pydantic overhead acceptable)
   - Subprocess overhead <5% of total validation time

5. **Adoption**:
   - 90% of users upgrade within 3 months
   - Zero critical bugs reported within first month
   - Positive feedback on new CLI commands (plan, doctor)
   - Documentation viewed by 80% of users
