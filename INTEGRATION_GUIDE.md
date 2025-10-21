# ContentDownload Catalog v2.0 - Integration Guide

**Purpose**: Wire the Catalog into the ContentDownload pipeline for seamless artifact tracking, deduplication, and lifecycle management.

**Status**: Ready for implementation

---

## 1. Architecture Integration

The Catalog integrates at the **finalization stage** of the ContentDownload pipeline:

```
Raw Artifact
    ↓
[Resolver Pipeline] ← existing
    ↓
[Network/Hishel Cache] ← existing
    ↓
[Download/Stream] ← existing
    ↓
[FINALIZATION] ← NEW integration point
    ├─ Compute SHA-256
    ├─ Choose final path (CAS or policy)
    ├─ Atomic move
    ├─ Register in catalog ← NEW
    ├─ Record telemetry
    └─ Return outcome
    ↓
[Download Complete]
```

---

## 2. Code Integration Points

### 2.1 Bootstrap (Initialization)

**File**: `src/DocsToKG/ContentDownload/bootstrap.py`

```python
from DocsToKG.ContentDownload.catalog import SQLiteCatalog, PostgresCatalogStore
from DocsToKG.ContentDownload.config.models import CatalogConfig, StorageConfig

def bootstrap_catalog(config: ContentDownloadConfig):
    """Initialize catalog based on configuration."""
    
    # Build catalog store
    if config.catalog.backend == "sqlite":
        catalog = SQLiteCatalog(
            path=config.catalog.path,
            wal_mode=config.catalog.wal_mode
        )
    elif config.catalog.backend == "postgres":
        catalog = PostgresCatalogStore(
            connection_string=config.catalog.path
        )
    else:
        raise ValueError(f"Unknown catalog backend: {config.catalog.backend}")
    
    return catalog

# Usage in runner.py:
catalog = bootstrap_catalog(config)
```

### 2.2 Pipeline Integration

**File**: `src/DocsToKG/ContentDownload/download_pipeline.py`

```python
from DocsToKG.ContentDownload.catalog.finalize import finalize_artifact

def process_artifact(self, artifact: Artifact) -> DownloadOutcome:
    """Process artifact with catalog integration."""
    
    # ... existing download logic ...
    
    # NEW: Finalize with catalog
    outcome = finalize_artifact(
        artifact=artifact,
        stream=download_stream,
        catalog=self.catalog,  # injected from bootstrap
        storage_config=self.config.storage,
        catalog_config=self.config.catalog,
        run_id=self.run_id,
        logger=self.logger
    )
    
    return outcome
```

### 2.3 CLI Integration

**File**: `src/DocsToKG/ContentDownload/cli_v2.py`

```python
import typer
from DocsToKG.ContentDownload.catalog.cli import catalog_app

# Wire catalog commands into main CLI
app.add_typer(catalog_app, name="catalog")

# Usage: 
# contentdownload catalog show <artifact_id>
# contentdownload catalog dedup-report
# contentdownload catalog gc --dry-run
```

---

## 3. Configuration

### 3.1 Add to Config File

**Example**: `config/contentdownload.yaml`

```yaml
# Existing config...
resolvers:
  # ...

# NEW: Catalog configuration
catalog:
  backend: sqlite              # or "postgres"
  path: state/catalog.sqlite   # SQLite file path or Postgres connection URL
  wal_mode: true               # WAL mode for SQLite
  compute_sha256: true         # Compute SHA-256 on successful downloads
  verify_on_register: false    # Re-verify after move (set true for S3)
  retention_days: 0            # Retention policy (0 = disabled)
  orphan_ttl_days: 7          # Days before orphan files are eligible for GC

# NEW: Storage configuration
storage:
  backend: fs                  # "fs" or "s3"
  root_dir: data/artifacts     # Base directory for final artifacts
  layout: cas                  # "policy_path" or "cas" (content-addressable)
  cas_prefix: sha256           # Prefix for CAS paths
  hardlink_dedup: true         # Use hardlinks for dedup (POSIX only)
  s3_bucket: null              # S3 bucket name (if backend="s3")
  s3_prefix: docs/             # S3 key prefix
  s3_storage_class: STANDARD   # S3 storage class
```

### 3.2 Environment Variables

```bash
# Override config values via environment
export DOCSTOKG_CATALOG_BACKEND="sqlite"
export DOCSTOKG_CATALOG_PATH="state/catalog.sqlite"
export DOCSTOKG_CATALOG_COMPUTE_SHA256="true"
export DOCSTOKG_STORAGE_LAYOUT="cas"
export DOCSTOKG_STORAGE_HARDLINK_DEDUP="true"
```

---

## 4. Integration Checklist

### Phase 1: Bootstrap (2 hours)

- [ ] Add `bootstrap_catalog()` to bootstrap.py
- [ ] Inject catalog into DownloadPipeline
- [ ] Add config models to ContentDownloadConfig
- [ ] Test initialization with SQLite backend
- [ ] Test initialization with Postgres backend
- [ ] Verify no breaking changes to existing code

**Validation**:
```bash
python -c "from DocsToKG.ContentDownload.bootstrap import bootstrap_catalog; \
  from DocsToKG.ContentDownload.config import ContentDownloadConfig; \
  cfg = ContentDownloadConfig(); \
  cat = bootstrap_catalog(cfg); \
  print('✅ Bootstrap works')"
```

### Phase 2: Finalization (4 hours)

- [ ] Implement `finalize_artifact()` function
- [ ] Integrate SHA-256 computation
- [ ] Implement CAS path selection
- [ ] Implement hardlink deduplication
- [ ] Register successful outcomes in catalog
- [ ] Handle errors gracefully
- [ ] Add comprehensive logging
- [ ] Write unit tests

**Validation**:
```bash
pytest tests/content_download/test_catalog_integration.py -v
```

### Phase 3: CLI Integration (2 hours)

- [ ] Wire catalog_app into main CLI
- [ ] Test each command:
  - `contentdownload catalog show <artifact_id>`
  - `contentdownload catalog where <sha256>`
  - `contentdownload catalog dedup-report`
  - `contentdownload catalog verify <record_id>`
  - `contentdownload catalog gc --dry-run`
  - `contentdownload catalog import-manifest <path>`

**Validation**:
```bash
contentdownload catalog --help
contentdownload catalog show test:001
contentdownload catalog dedup-report
```

### Phase 4: End-to-End Testing (3 hours)

- [ ] Test with 10 sample artifacts
- [ ] Verify catalog records created
- [ ] Verify dedup hits tracked
- [ ] Verify metrics emitted
- [ ] Test failure scenarios
- [ ] Test recovery procedures

**Validation**:
```bash
# Run end-to-end test
python tests/content_download/test_catalog_e2e.py -v

# Verify catalog has records
sqlite3 state/catalog.sqlite "SELECT COUNT(*) FROM documents;"

# Check metrics
python -c "from DocsToKG.ContentDownload.catalog.metrics import CatalogMetrics; \
  m = CatalogMetrics(); \
  print('✅ Metrics initialized')"
```

### Phase 5: Documentation (2 hours)

- [ ] Update README with catalog features
- [ ] Document CLI commands
- [ ] Add integration examples
- [ ] Document troubleshooting
- [ ] Add performance tuning guide

**Validation**:
```bash
ls -lh README.md INTEGRATION_GUIDE.md
```

---

## 5. Testing Strategy

### Unit Tests

```python
# tests/content_download/test_catalog_integration.py

def test_bootstrap_catalog_sqlite():
    """Test catalog bootstrap with SQLite backend."""
    config = ContentDownloadConfig(
        catalog=CatalogConfig(backend="sqlite", path=":memory:")
    )
    catalog = bootstrap_catalog(config)
    assert catalog is not None
    catalog.close()

def test_finalize_artifact_creates_record():
    """Test artifact finalization creates catalog record."""
    catalog = SQLiteCatalog(":memory:")
    stream = MockDownloadStream(bytes=1000, content_type="application/pdf")
    artifact = Artifact(id="test:001", url="http://example.com/file.pdf")
    
    outcome = finalize_artifact(
        artifact=artifact,
        stream=stream,
        catalog=catalog,
        storage_config=StorageConfig(),
        catalog_config=CatalogConfig(),
        run_id="run-001"
    )
    
    assert outcome.ok
    records = catalog.get_by_artifact("test:001")
    assert len(records) == 1
    assert records[0].sha256 is not None

def test_dedup_hardlink():
    """Test hardlink deduplication."""
    # Create first file
    outcome1 = finalize_artifact(...)
    
    # Create identical file
    outcome2 = finalize_artifact(...)
    
    # Verify hardlink created
    assert os.stat(outcome1.path).st_ino == os.stat(outcome2.path).st_ino
```

### Integration Tests

```python
# tests/content_download/test_catalog_e2e.py

def test_end_to_end_with_pipeline():
    """Test full pipeline with catalog integration."""
    pipeline = build_pipeline(config)
    
    artifacts = [
        Artifact(id="test:001", url="http://example.com/a.pdf"),
        Artifact(id="test:002", url="http://example.com/b.pdf"),
    ]
    
    results = pipeline.process_artifacts(artifacts)
    
    # Verify catalog records
    for result in results:
        assert result.ok
        records = pipeline.catalog.get_by_artifact(result.artifact_id)
        assert len(records) > 0
```

---

## 6. Deployment Workflow

### Pre-Deployment

1. **Review Integration**
   ```bash
   git diff --name-only  # See changed files
   git log -1 --stat    # See changes summary
   ```

2. **Run Tests**
   ```bash
   pytest tests/content_download/ -v
   ```

3. **Check Type Safety**
   ```bash
   mypy src/DocsToKG/ContentDownload/
   ```

4. **Check Linting**
   ```bash
   ruff check src/DocsToKG/ContentDownload/
   ```

### Deployment

1. **Dev Environment**
   ```bash
   # Test with SQLite
   export DOCSTOKG_CATALOG_BACKEND="sqlite"
   contentdownload download --config config/dev.yaml
   ```

2. **Staging Environment**
   ```bash
   # Test with Postgres
   export DOCSTOKG_CATALOG_BACKEND="postgres"
   contentdownload download --config config/staging.yaml
   ```

3. **Production Environment**
   ```bash
   # Deploy with chosen backend
   export DOCSTOKG_CATALOG_BACKEND="postgres"  # or "sqlite"
   contentdownload download --config config/production.yaml
   ```

### Post-Deployment

1. **Verify Integration**
   ```bash
   # Check catalog has records
   sqlite3 state/catalog.sqlite "SELECT COUNT(*) FROM documents;"
   
   # Test CLI commands
   contentdownload catalog dedup-report
   contentdownload catalog gc --dry-run
   ```

2. **Monitor Operations**
   ```bash
   # Watch logs
   tail -f Data/Logs/contentdownload-*.jsonl
   
   # Monitor metrics
   curl http://localhost:8000/metrics
   ```

3. **Verify Backups**
   ```bash
   contentdownload catalog backup
   ls -lh Data/Backups/
   ```

---

## 7. Rollback Plan

### If Issues Occur

1. **Immediate Rollback**
   ```bash
   # Revert integration commits
   git revert HEAD~3..HEAD  # Assuming 3 integration commits
   
   # Restore from backup
   contentdownload catalog restore --from-backup Data/Backups/latest.sqlite
   
   # Restart with previous version
   systemctl restart contentdownload
   ```

2. **Investigation**
   ```bash
   # Check error logs
   tail -100 Data/Logs/contentdownload-error.jsonl
   
   # Run diagnostics
   contentdownload catalog consistency check
   
   # Verify integrity
   contentdownload catalog verify
   ```

3. **Documentation**
   - Record issue description
   - Document root cause
   - Add prevention to runbook

---

## 8. Validation Checklist

- [ ] Code changes reviewed
- [ ] All tests passing (99/99)
- [ ] Type checking clean (mypy)
- [ ] Lint checking clean (ruff)
- [ ] Bootstrap integration tested
- [ ] Finalization integration tested
- [ ] CLI commands tested
- [ ] End-to-end test passed
- [ ] Documentation updated
- [ ] Backup strategy tested
- [ ] Rollback procedure tested
- [ ] Team trained on operations

---

## 9. Operations Guide

### Daily Operations

```bash
# Check catalog health
contentdownload catalog stats

# Monitor deduplication
contentdownload catalog dedup-report

# Check for orphaned files
contentdownload catalog gc --dry-run
```

### Weekly Operations

```bash
# Verify integrity
contentdownload catalog verify

# Run consistency check
contentdownload catalog consistency check

# Backup catalog
contentdownload catalog backup
```

### Monthly Operations

```bash
# Garbage collection (if enabled)
contentdownload catalog gc --apply

# Analyze performance
contentdownload catalog analytics

# Review retention policies
contentdownload catalog retention review
```

---

## 10. Success Metrics

After integration, verify:

| Metric | Target | Method |
|--------|--------|--------|
| Catalog Records | 100% of successful downloads | `SELECT COUNT(*) FROM documents;` |
| Dedup Hits | >10% for repeated resolvers | `catalog dedup-report` |
| Verification Failures | 0% (ideally) | `catalog verify` |
| Mean Registration Time | <100ms per record | Monitor logs |
| Storage Savings | Varies by dedup rate | `catalog analytics` |

---

## 11. Next Steps

After integration is complete and validated:

1. **Monitor in Production** (1 week)
   - Watch metrics and logs
   - Verify dedup/GC operations
   - Confirm backup procedures

2. **Optimize** (ongoing)
   - Tune batch sizes based on performance
   - Adjust retention policies as needed
   - Add custom metadata extractors

3. **Extend** (optional)
   - Implement Phase 12 features
   - Add multi-region support
   - Integrate with search/indexing

---

## 12. Questions & Support

### Common Questions

**Q: Will this break existing functionality?**  
A: No. Catalog integration is purely additive. Existing code paths remain unchanged.

**Q: Do I need Postgres?**  
A: No. SQLite works fine for development and moderate scale. Postgres is recommended for >10M records.

**Q: How much storage will the catalog use?**  
A: ~1KB per record. For 1M records, expect ~1GB of database overhead.

**Q: Can I use S3 instead of local filesystem?**  
A: Yes. Set `storage.backend="s3"` in config and provide bucket credentials.

### Support Resources

- Documentation: `src/DocsToKG/ContentDownload/ARCHITECTURE_catalog.md`
- Examples: `tests/content_download/test_catalog_*.py`
- Git History: `git log --oneline` (18 commits, one per feature)
- Architecture: `DO NOT DELETE docs-instruct/.../# PR #9...`

---

## Summary

Integration takes **~13 hours total** across 5 phases and enables:

✅ Automatic artifact tracking  
✅ Deduplication across resolvers  
✅ Content verification  
✅ Lifecycle management  
✅ Operational visibility  

Follow this guide, run the validation checklist, and you'll have a production-ready integrated system!

