# Catalog Connector Architecture - Provider Pattern

**Objective**: One unified connector with pluggable provider backends for development, enterprise, and cloud deployments.

**Pattern**: Mirrors PR-4 Embedding Providers approach - small interfaces + concrete implementations.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│         CatalogConnector (unified entry point)      │
│  - Factory pattern                                   │
│  - Config-driven backend selection                   │
│  - Transparent provider abstraction                  │
└────────┬────────────────────────────────────────────┘
         │
         ├─────────────────┬──────────────────┬──────────────────┐
         ▼                 ▼                  ▼                  ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐
    │ Development │  │  Enterprise  │  │    Cloud     │  │  Provider  │
    │  Provider   │  │   Provider   │  │   Provider   │  │  Factory   │
    ├─────────────┤  ├──────────────┤  ├──────────────┤  └────────────┘
    │ SQLite      │  │ Postgres     │  │ Postgres RDS │
    │ Local FS    │  │ Local FS     │  │ S3           │
    │ In-memory   │  │ Connection   │  │ AWS Creds    │
    │             │  │ pooling      │  │ CDN-ready    │
    └─────────────┘  └──────────────┘  └──────────────┘
```

---

## 2. Provider Interface (Base Protocol)

```
CatalogProvider Protocol:
├── name() → str                          # "development" | "enterprise" | "cloud"
├── open(config) → None                   # Initialize backend
├── register_or_get(...) → DocumentRecord # Core CRUD
├── get_by_artifact(id) → List[Record]
├── get_by_sha256(sha) → List[Record]
├── find_duplicates() → List[Tuple]
├── verify(record_id) → bool
├── stats() → Dict[str, Any]
├── close() → None                        # Cleanup
└── health_check() → HealthStatus
```

---

## 3. Concrete Providers

### Development Provider

**File**: `src/DocsToKG/ContentDownload/catalog/connectors/dev_provider.py`

```
Features:
  • SQLite backend (no external deps)
  • Local filesystem storage
  • In-memory caching (thread-local)
  • Single-process safe
  • Zero config needed
  
When to use:
  • Local development
  • Testing
  • Prototyping
  • CI/CD
  
Config:
  • db_path: str (default: ":memory:" or "./catalog.sqlite")
  • cache_size: int (default: 1000)
  • enable_wal: bool (default: True)
```

### Enterprise Provider

**File**: `src/DocsToKG/ContentDownload/catalog/connectors/enterprise_provider.py`

```
Features:
  • Postgres backend (connection pooling)
  • Local filesystem storage
  • Thread-safe operations (RLock)
  • ACID compliance
  • Production-grade config
  
When to use:
  • On-premises production
  • Compliance requirements
  • Multi-process deployment
  • High reliability needed
  
Config:
  • connection_url: str (postgresql://...)
  • pool_size: int (default: 10)
  • max_overflow: int (default: 20)
  • echo_sql: bool (default: False)
  • root_dir: str (filesystem root)
  • wal_mode: bool (default: True)
```

### Cloud Provider

**File**: `src/DocsToKG/ContentDownload/catalog/connectors/cloud_provider.py`

```
Features:
  • Postgres RDS backend (AWS managed)
  • S3 storage backend
  • Multi-region ready
  • Infinite scalability
  • Auto-failover support
  
When to use:
  • AWS/cloud deployments
  • Global scale
  • Unlimited growth
  • CDN-ready distribution
  
Config:
  • connection_url: str (RDS endpoint)
  • pool_size: int (default: 20)
  • aws_region: str
  • s3_bucket: str
  • s3_prefix: str (default: "docs/")
  • enable_encryption: bool (default: True)
  • storage_class: str (default: "STANDARD")
```

---

## 4. Unified Connector (Factory)

**File**: `src/DocsToKG/ContentDownload/catalog/connectors/connector.py`

```python
class CatalogConnector:
    """Unified entry point for all catalog backends."""
    
    def __init__(self, provider_type: str, config: Dict[str, Any]):
        """
        Args:
            provider_type: "development" | "enterprise" | "cloud"
            config: Provider-specific configuration
        """
        self.provider_type = provider_type
        self.provider = self._create_provider(provider_type, config)
    
    def _create_provider(self, provider_type: str, config: Dict) -> CatalogProvider:
        """Factory method - creates appropriate provider."""
        if provider_type == "development":
            from .dev_provider import DevelopmentProvider
            return DevelopmentProvider(config)
        elif provider_type == "enterprise":
            from .enterprise_provider import EnterpriseProvider
            return EnterpriseProvider(config)
        elif provider_type == "cloud":
            from .cloud_provider import CloudProvider
            return CloudProvider(config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    def open(self) -> None:
        """Initialize the backend."""
        self.provider.open()
    
    def close(self) -> None:
        """Cleanup."""
        self.provider.close()
    
    def __enter__(self):
        """Context manager support."""
        self.open()
        return self
    
    def __exit__(self, *args):
        """Context manager cleanup."""
        self.close()
    
    # Delegate all catalog operations to provider
    def register_or_get(self, **kwargs) -> DocumentRecord:
        return self.provider.register_or_get(**kwargs)
    
    def get_by_artifact(self, artifact_id: str) -> List[DocumentRecord]:
        return self.provider.get_by_artifact(artifact_id)
    
    # ... other methods delegate to provider ...
```

---

## 5. File Structure

```
src/DocsToKG/ContentDownload/catalog/connectors/
├── __init__.py                  # Exports CatalogConnector, ProviderFactory
├── base.py                      # CatalogProvider protocol, HealthStatus
├── connector.py                 # CatalogConnector factory
├── dev_provider.py              # Development (SQLite + local FS)
├── enterprise_provider.py       # Enterprise (Postgres + local FS)
├── cloud_provider.py            # Cloud (RDS + S3)
└── errors.py                    # Provider-specific exceptions
```

---

## 6. Usage Examples

### Development

```python
from DocsToKG.ContentDownload.catalog import CatalogConnector

# Option A: Default in-memory
connector = CatalogConnector("development", {})
connector.open()
rec = connector.register_or_get("test:001", ...)
connector.close()

# Option B: With context manager
with CatalogConnector("development", {"db_path": "./dev.sqlite"}) as cat:
    records = cat.get_by_artifact("test:001")
```

### Enterprise

```python
# On-premises Postgres
with CatalogConnector("enterprise", {
    "connection_url": "postgresql://user:pass@localhost/catalog",
    "pool_size": 10,
    "root_dir": "/data/artifacts"
}) as cat:
    cat.register_or_get(...)
    stats = cat.stats()
```

### Cloud

```python
# AWS RDS + S3
with CatalogConnector("cloud", {
    "connection_url": "postgresql://user:pass@rds-endpoint:5432/catalog",
    "aws_region": "us-east-1",
    "s3_bucket": "my-artifacts",
    "s3_prefix": "docs/",
    "enable_encryption": True
}) as cat:
    cat.register_or_get(...)
    cat.backup()
```

---

## 7. Configuration Loading

**File**: `src/DocsToKG/ContentDownload/catalog/connectors/config.py`

```python
from typing import Dict, Any
import os
import json

class ConnectorConfig:
    """Load connector config from multiple sources."""
    
    @staticmethod
    def from_env(provider_type: str = None) -> Dict[str, Any]:
        """Load from environment variables."""
        if not provider_type:
            provider_type = os.getenv("DOCSTOKG_CATALOG_PROVIDER", "development")
        
        if provider_type == "development":
            return {
                "db_path": os.getenv("DOCSTOKG_DEV_DB_PATH", ":memory:"),
                "cache_size": int(os.getenv("DOCSTOKG_DEV_CACHE_SIZE", "1000"))
            }
        elif provider_type == "enterprise":
            return {
                "connection_url": os.getenv("DOCSTOKG_POSTGRES_URL"),
                "pool_size": int(os.getenv("DOCSTOKG_POOL_SIZE", "10")),
                "root_dir": os.getenv("DOCSTOKG_STORAGE_ROOT", "data/artifacts")
            }
        elif provider_type == "cloud":
            return {
                "connection_url": os.getenv("DOCSTOKG_RDS_URL"),
                "aws_region": os.getenv("AWS_REGION", "us-east-1"),
                "s3_bucket": os.getenv("DOCSTOKG_S3_BUCKET"),
                "s3_prefix": os.getenv("DOCSTOKG_S3_PREFIX", "docs/")
            }
    
    @staticmethod
    def from_file(config_path: str) -> Dict[str, Any]:
        """Load from YAML/JSON config file."""
        with open(config_path) as f:
            if config_path.endswith(".json"):
                return json.load(f)
            else:  # YAML
                import yaml
                return yaml.safe_load(f)
```

---

## 8. Integration with Existing Code

### Bootstrap Integration

**File**: `src/DocsToKG/ContentDownload/catalog/bootstrap.py`

```python
from .connectors import CatalogConnector, ConnectorConfig

def build_catalog_connector(
    provider_type: str = None,
    config_path: str = None,
    cli_overrides: Dict[str, Any] = None
) -> CatalogConnector:
    """Build connector using configuration precedence."""
    # 1. CLI overrides (highest priority)
    # 2. Config file
    # 3. Environment (default)
    
    if not provider_type:
        provider_type = os.getenv("DOCSTOKG_CATALOG_PROVIDER", "development")
    
    if config_path:
        config = ConnectorConfig.from_file(config_path)
    else:
        config = ConnectorConfig.from_env(provider_type)
    
    if cli_overrides:
        config.update(cli_overrides)
    
    return CatalogConnector(provider_type, config)
```

---

## 9. Testing Strategy

```
tests/content_download/connectors/
├── test_dev_provider.py          # SQLite-specific tests
├── test_enterprise_provider.py   # Postgres-specific tests
├── test_cloud_provider.py        # RDS/S3-specific tests
├── test_connector_factory.py     # Factory switching
└── test_connector_integration.py # Cross-provider parity
```

Each test file:
- Unit tests for provider-specific logic
- Mocked external services (Postgres, S3)
- Parity tests ensuring identical behavior across providers
- Configuration loading tests

---

## 10. Benefits of This Approach

### Clean Separation
- One interface, three implementations
- Zero coupling between providers
- Each backend owns its dependencies

### Developer Experience
- Same API regardless of deployment
- Config-driven switching (no code changes)
- Environment-based selection

### Production Ready
- Connection pooling (enterprise/cloud)
- ACID compliance (Postgres)
- Auto-failover ready (RDS)
- Infinite scale (S3)

### Testing
- Mock one provider at a time
- Full integration tests per backend
- Parity tests ensure consistency

### Migration Friendly
- Start with development (SQLite)
- Migrate to enterprise (Postgres) without code changes
- Move to cloud (AWS) by changing config only

---

## 11. Implementation Sequence

### Phase 1: Scaffolding
- Create `connectors/` package
- Define base protocol and errors
- Implement factory

### Phase 2: Development Provider
- SQLite implementation
- Local filesystem operations
- In-memory caching

### Phase 3: Enterprise Provider
- Postgres implementation
- Connection pooling
- Thread-safety

### Phase 4: Cloud Provider
- RDS support
- S3 integration
- AWS auth/credentials

### Phase 5: Integration
- Wire into bootstrap
- Config loading
- CLI support

### Phase 6: Testing
- Provider unit tests
- Integration tests
- Parity validation

### Phase 7: Documentation
- Usage examples
- Config reference
- Migration guide

---

## 12. Hybrid Deployment (Optional D)

Option D (Hybrid) uses this architecture transparently:

```python
# Start with development (no external services)
with CatalogConnector("development", {}) as cat:
    # ... accumulate data ...
    pass

# Migrate to enterprise (add Postgres)
with CatalogConnector("enterprise", {
    "connection_url": "postgresql://...",
    "root_dir": "data/artifacts"  # Keep local FS initially
}) as cat:
    # ... same API, improved performance ...
    pass

# Move to cloud (add S3)
with CatalogConnector("cloud", {
    "connection_url": "postgresql://rds...",
    "s3_bucket": "my-bucket"
}) as cat:
    # ... infinite scale ...
    pass

# All three use identical API - migration is config-only!
```

---

## Acceptance Criteria

✅ One unified `CatalogConnector` with three providers  
✅ Same public API across all providers  
✅ Config-driven provider selection  
✅ Development provider requires zero external dependencies  
✅ Enterprise provider supports production Postgres  
✅ Cloud provider supports AWS RDS + S3  
✅ All three providers have identical behavior (tested)  
✅ Migration between providers requires config change only  
✅ Full test coverage per provider  
✅ Documentation with examples for each option

