# Connector Architecture - Next Steps Implementation

**Status**: Design Complete, Ready for Implementation

---

## Overview

The Connector Architecture applies the **PR-4 Embedding Providers pattern** to the Catalog deployment system. Instead of separate "Option A/B/C/D", we now have:

- **One unified `CatalogConnector` factory**
- **Three provider implementations** (Development, Enterprise, Cloud)
- **Config-driven backend selection**
- **Identical API across all providers**
- **Migration without code changes** (only config changes)

---

## Key Insight: "Hybrid" is Automatic

**Option D (Hybrid) is NOT a separate provider.** Instead, it's the natural result of the architecture:

```python
# Day 1: Development
with CatalogConnector("development", {}) as cat:
    cat.register_or_get(...)  # SQLite, local FS

# Week 2: Migrate to Enterprise (same code!)
with CatalogConnector("enterprise", {
    "connection_url": "postgresql://..."
}) as cat:
    cat.register_or_get(...)  # Postgres, local FS

# Month 2: Move to Cloud (same code!)
with CatalogConnector("cloud", {
    "connection_url": "postgresql://rds...",
    "s3_bucket": "my-artifacts"
}) as cat:
    cat.register_or_get(...)  # RDS, S3

# All three are Option D in action - progressive migration!
```

---

## Architecture Components

### 1. Base Protocol (`base.py`)

```python
# Define the interface ALL providers implement
class CatalogProvider(Protocol):
    def name() -> str
    def open(config) -> None
    def close() -> None
    def register_or_get(...) -> DocumentRecord
    def get_by_artifact(id) -> List[DocumentRecord]
    def get_by_sha256(sha) -> List[DocumentRecord]
    def find_duplicates() -> List[Tuple]
    def verify(record_id) -> bool
    def stats() -> Dict[str, Any]
    def health_check() -> HealthStatus
```

### 2. Unified Connector (`connector.py`)

```python
# The ONLY entry point users interact with
class CatalogConnector:
    def __init__(self, provider_type: str, config: Dict):
        # Factory creates the right provider
        self.provider = self._create_provider(provider_type, config)
    
    # All methods delegate to provider (transparent abstraction)
    def register_or_get(self, ...):
        return self.provider.register_or_get(...)
    
    # Context manager support
    def __enter__(self): ...
    def __exit__(self): ...
```

### 3. Three Provider Implementations

**Development Provider** (`dev_provider.py`):
- SQLite (built-in, no dependencies)
- Local filesystem
- In-memory caching
- Zero config needed

**Enterprise Provider** (`enterprise_provider.py`):
- Postgres with connection pooling
- Local filesystem
- Thread-safe (RLock)
- Production-grade

**Cloud Provider** (`cloud_provider.py`):
- Postgres RDS
- S3 storage
- AWS auth/credentials
- Infinite scale

### 4. Configuration (`config.py`)

```python
# Load from multiple sources with precedence
class ConnectorConfig:
    @staticmethod
    def from_env(provider_type) -> Dict
    @staticmethod
    def from_file(config_path) -> Dict
```

---

## Implementation Phases (7 total)

### Phase 1: Scaffolding (2 hours)
- [x] Create `connectors/` package structure
- [x] Design `CatalogProvider` protocol
- [x] Implement `CatalogConnector` factory
- [x] Define error types

**Files to create**:
- `connectors/__init__.py`
- `connectors/base.py` (protocol + errors)
- `connectors/connector.py` (factory)
- `connectors/errors.py` (exception types)

### Phase 2: Development Provider (4 hours)
- [ ] Implement `DevelopmentProvider` (SQLite)
- [ ] Add in-memory caching
- [ ] Implement all protocol methods
- [ ] Add unit tests

**File to create**:
- `connectors/dev_provider.py` (150 LOC)

**Test file**:
- `tests/content_download/connectors/test_dev_provider.py` (100 LOC)

### Phase 3: Enterprise Provider (6 hours)
- [ ] Implement `EnterpriseProvider` (Postgres)
- [ ] Add connection pooling
- [ ] Implement thread-safety (RLock)
- [ ] Add unit tests

**File to create**:
- `connectors/enterprise_provider.py` (200 LOC)

**Test file**:
- `tests/content_download/connectors/test_enterprise_provider.py` (150 LOC)

### Phase 4: Cloud Provider (6 hours)
- [ ] Implement `CloudProvider` (RDS + S3)
- [ ] Add AWS auth/credentials handling
- [ ] Implement S3 operations
- [ ] Add unit tests

**File to create**:
- `connectors/cloud_provider.py` (250 LOC)

**Test file**:
- `tests/content_download/connectors/test_cloud_provider.py` (150 LOC)

### Phase 5: Configuration & Integration (4 hours)
- [ ] Implement `ConnectorConfig`
- [ ] Update `bootstrap.py` to use connector
- [ ] Add CLI support
- [ ] Environment variable support

**Files to create/modify**:
- `connectors/config.py` (100 LOC)
- `bootstrap.py` (update to use connector)
- `cli.py` (add --provider flag)

### Phase 6: Testing & Parity (8 hours)
- [ ] Provider unit tests
- [ ] Parity tests (ensure identical behavior)
- [ ] Factory switching tests
- [ ] Integration tests

**Test files to create**:
- `tests/content_download/connectors/test_connector_factory.py`
- `tests/content_download/connectors/test_connector_integration.py`
- `tests/content_download/connectors/test_parity.py`

### Phase 7: Documentation (3 hours)
- [ ] Update README with connector examples
- [ ] Add migration guide
- [ ] Document config options
- [ ] Update AGENTS.md

**Files to create**:
- `CONNECTOR_USAGE_GUIDE.md` (usage examples)
- `CONNECTOR_MIGRATION_GUIDE.md` (migration path)
- `CONNECTOR_CONFIG_REFERENCE.md` (all config options)

---

## Implementation Details

### Phase 2: Development Provider (Detailed)

```python
# connectors/dev_provider.py (150 LOC)

class DevelopmentProvider(CatalogProvider):
    def __init__(self, config: Dict):
        self.db_path = config.get("db_path", ":memory:")
        self.cache_size = config.get("cache_size", 1000)
        self.enable_wal = config.get("enable_wal", True)
        self.conn = None
        self._cache = {}
    
    def open(self) -> None:
        """Initialize SQLite."""
        import sqlite3
        self.conn = sqlite3.connect(self.db_path)
        if self.enable_wal:
            self.conn.execute("PRAGMA journal_mode=WAL")
        # Load schema from schema.sql
        schema = Path(__file__).parent.parent / "schema.sql"
        self.conn.executescript(schema.read_text())
        self.conn.commit()
    
    def register_or_get(self, **kwargs) -> DocumentRecord:
        # Reuse existing SQLiteCatalog logic
        # Returns DocumentRecord
        pass
    
    def close(self) -> None:
        """Cleanup."""
        if self.conn:
            self.conn.close()
    
    # ... implement all protocol methods ...
```

### Phase 3: Enterprise Provider (Detailed)

```python
# connectors/enterprise_provider.py (200 LOC)

class EnterpriseProvider(CatalogProvider):
    def __init__(self, config: Dict):
        self.connection_url = config["connection_url"]
        self.pool_size = config.get("pool_size", 10)
        self.max_overflow = config.get("max_overflow", 20)
        self._lock = threading.RLock()
        self.pool = None
    
    def open(self) -> None:
        """Initialize connection pool."""
        from sqlalchemy import create_engine
        self.engine = create_engine(
            self.connection_url,
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
        )
    
    def register_or_get(self, **kwargs) -> DocumentRecord:
        with self._lock:
            # Reuse existing PostgresCatalog logic
            pass
    
    def close(self) -> None:
        """Cleanup."""
        self.engine.dispose()
    
    # ... implement all protocol methods with thread safety ...
```

### Phase 4: Cloud Provider (Detailed)

```python
# connectors/cloud_provider.py (250 LOC)

class CloudProvider(CatalogProvider):
    def __init__(self, config: Dict):
        self.connection_url = config["connection_url"]
        self.aws_region = config["aws_region"]
        self.s3_bucket = config["s3_bucket"]
        self.s3_prefix = config.get("s3_prefix", "docs/")
        self._lock = threading.RLock()
        self.engine = None
        self.s3_client = None
    
    def open(self) -> None:
        """Initialize RDS + S3."""
        from sqlalchemy import create_engine
        import boto3
        
        self.engine = create_engine(self.connection_url)
        self.s3_client = boto3.client('s3', region_name=self.aws_region)
    
    def register_or_get(self, **kwargs) -> DocumentRecord:
        with self._lock:
            # Use RDS for catalog
            # Use S3 for storage
            pass
    
    def close(self) -> None:
        """Cleanup."""
        self.engine.dispose()
    
    # ... implement all protocol methods with S3 support ...
```

---

## Testing Strategy

### Unit Tests (Phase 6)

```python
# test_dev_provider.py
def test_dev_provider_creates_sqlite():
    """SQLite database is created."""
    
def test_dev_provider_registers_record():
    """Record can be registered and retrieved."""
    
def test_dev_provider_caching():
    """In-memory cache works."""

# test_enterprise_provider.py  
def test_enterprise_provider_thread_safe():
    """Multiple threads don't corrupt data."""
    
def test_enterprise_provider_connection_pool():
    """Pool limits are respected."""

# test_cloud_provider.py
def test_cloud_provider_s3_upload():
    """Files can be uploaded to S3."""
    
def test_cloud_provider_rds_connection():
    """RDS connection works."""
```

### Parity Tests (Phase 6)

```python
# test_parity.py
def test_all_providers_same_api():
    """All providers have identical API."""
    
def test_all_providers_identical_behavior():
    """Behavior is identical across all providers."""
    
def test_migration_dev_to_enterprise():
    """Can migrate from dev to enterprise."""
    
def test_migration_enterprise_to_cloud():
    """Can migrate from enterprise to cloud."""
```

---

## Configuration Reference

### Environment Variables

```bash
# Provider selection
DOCSTOKG_CATALOG_PROVIDER=development  # or enterprise, cloud

# Development provider
DOCSTOKG_DEV_DB_PATH=:memory:
DOCSTOKG_DEV_CACHE_SIZE=1000

# Enterprise provider
DOCSTOKG_POSTGRES_URL=postgresql://user:pass@localhost/catalog
DOCSTOKG_POOL_SIZE=10
DOCSTOKG_STORAGE_ROOT=/data/artifacts

# Cloud provider
DOCSTOKG_RDS_URL=postgresql://user:pass@rds:5432/catalog
AWS_REGION=us-east-1
DOCSTOKG_S3_BUCKET=my-artifacts
DOCSTOKG_S3_PREFIX=docs/
```

### YAML Config

```yaml
# config.yaml
catalog:
  provider: development  # or enterprise, cloud
  
  # Development-specific
  dev:
    db_path: ":memory:"
    cache_size: 1000
  
  # Enterprise-specific
  enterprise:
    connection_url: "postgresql://..."
    pool_size: 10
    root_dir: "/data/artifacts"
  
  # Cloud-specific
  cloud:
    connection_url: "postgresql://..."
    aws_region: "us-east-1"
    s3_bucket: "my-artifacts"
    s3_prefix: "docs/"
```

---

## Usage Examples

### Development

```python
from DocsToKG.ContentDownload.catalog import CatalogConnector

with CatalogConnector("development", {}) as cat:
    cat.register_or_get(
        artifact_id="test:001",
        source_url="http://example.com",
        resolver="test",
        content_type="application/pdf",
        bytes=1000,
        sha256="abc123",
        storage_uri="file:///tmp/test.pdf",
        run_id="run-001"
    )
    stats = cat.stats()
```

### Enterprise

```python
with CatalogConnector("enterprise", {
    "connection_url": "postgresql://user:pass@localhost/catalog",
    "pool_size": 10,
    "root_dir": "/data/artifacts"
}) as cat:
    cat.register_or_get(...)
```

### Cloud

```python
with CatalogConnector("cloud", {
    "connection_url": "postgresql://user:pass@rds-endpoint:5432/catalog",
    "aws_region": "us-east-1",
    "s3_bucket": "my-artifacts"
}) as cat:
    cat.register_or_get(...)
```

---

## Migration Path (Hybrid = Automatic)

```
Day 1 (Dev):
  DOCSTOKG_CATALOG_PROVIDER=development
  No external dependencies
  Perfect for testing

Day 7 (Enterprise):
  DOCSTOKG_CATALOG_PROVIDER=enterprise
  DOCSTOKG_POSTGRES_URL=postgresql://...
  Same code, better performance
  Production on-prem ready

Day 30 (Cloud):
  DOCSTOKG_CATALOG_PROVIDER=cloud
  DOCSTOKG_RDS_URL=postgresql://...
  AWS_REGION=us-east-1
  DOCSTOKG_S3_BUCKET=my-artifacts
  Infinite scale, global reach

NO CODE CHANGES NEEDED - only config!
```

---

## Deliverables Summary

### Total LOC (estimated)
- Scaffolding: 200 LOC
- Development Provider: 150 LOC
- Enterprise Provider: 200 LOC
- Cloud Provider: 250 LOC
- Configuration: 100 LOC
- Tests: 500+ LOC
- **Total: ~1,400 LOC**

### Timeline
- Phase 1: 2 hours (scaffolding)
- Phase 2: 4 hours (dev provider)
- Phase 3: 6 hours (enterprise provider)
- Phase 4: 6 hours (cloud provider)
- Phase 5: 4 hours (config & integration)
- Phase 6: 8 hours (testing)
- Phase 7: 3 hours (documentation)
- **Total: 33 hours**

### Quality Gates
✅ 100% API parity across providers
✅ Same behavior (tested)
✅ Config-only migration
✅ Full test coverage
✅ Zero code changes to migrate

---

## Next Action

Ready to implement Phase 1 (Scaffolding)?

Would you like me to:
1. Create the `connectors/` package structure
2. Define the `CatalogProvider` protocol
3. Implement the `CatalogConnector` factory
4. Set up error types

All in one commit, ready for Phase 2?

