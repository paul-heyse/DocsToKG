# Resolver Migration Guide: Legacy → Modern Pattern

## Overview

All 15 resolvers need to migrate from legacy inheritance pattern to modern standalone pattern. This guide shows the exact changes needed.

## Migration Template

### BEFORE (Current Legacy Pattern)

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
import httpx

from DocsToKG.ContentDownload.core import dedupe
from DocsToKG.ContentDownload.urls import canonical_for_index

# ❌ LEGACY: Importing from base.py
from .base import RegisteredResolver, ResolverEvent, ResolverEventReason, ResolverResult
# ❌ LEGACY: Importing from pipeline.py
from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    # ❌ LEGACY: ResolverConfig from pipeline.py
    from DocsToKG.ContentDownload.pipeline import ResolverConfig


# ❌ LEGACY: Inheriting from RegisteredResolver base class
@register_v2("example")
class ExampleResolver(RegisteredResolver):
    """Example resolver."""

    name = "example"

    def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:
        """Check if resolver should run."""
        return True

    # ❌ LEGACY: Uses iter_urls() method signature
    def iter_urls(
        self,
        client: httpx.Client,
        config: "ResolverConfig",
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs."""
        yield ResolverResult(url="https://example.com/pdf")
```

### AFTER (Modern Pattern)

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Any
import httpx

from DocsToKG.ContentDownload.core import dedupe
from DocsToKG.ContentDownload.urls import canonical_for_index

# ✅ MODERN: Only import types for annotations
from .registry_v2 import register_v2

if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact


# Simple data class for resolver result (no inheritance needed)
class ResolverResult:
    """Result from resolver attempt."""
    def __init__(self, url: str | None = None, **kwargs):
        self.url = url
        for k, v in kwargs.items():
            setattr(self, k, v)


# ✅ MODERN: No inheritance, standalone class
@register_v2("example")
class ExampleResolver:
    """Example resolver - modern pattern."""

    name = "example"

    def is_enabled(self, config: Any, artifact: "WorkArtifact") -> bool:
        """Check if resolver should run.
        
        Args:
            config: Resolver configuration (any type - no inheritance needed)
            artifact: Work artifact to process
            
        Returns:
            True if resolver should attempt to resolve
        """
        return True

    # ✅ MODERN: Compatible with both iter_urls() and resolve() patterns
    def iter_urls(
        self,
        client: httpx.Client,
        config: Any,
        artifact: "WorkArtifact",
    ) -> Iterable[ResolverResult]:
        """Yield candidate URLs.
        
        This method will be called by the modern DownloadPipeline.
        Keep implementation as-is (no changes needed to logic).
        
        Args:
            client: HTTPX client for making requests
            config: Resolver configuration
            artifact: Work metadata
            
        Yields:
            ResolverResult with URL or skip/error event
        """
        # Implementation stays the same - just remove base class dependency
        yield ResolverResult(url="https://example.com/pdf")

    # ✅ OPTIONAL: Modern resolve() pattern (can coexist with iter_urls)
    def resolve(self, artifact: "WorkArtifact") -> list[str]:
        """Modern resolve pattern - optional.
        
        Return list of URLs instead of yielding ResolverResult.
        The DownloadPipeline will handle both patterns.
        
        Args:
            artifact: Work metadata
            
        Returns:
            List of candidate URLs
        """
        return ["https://example.com/pdf"]
```

## Key Changes

### 1. **Remove Inheritance**
```python
# ❌ BEFORE
class ExampleResolver(RegisteredResolver):
    pass

# ✅ AFTER
class ExampleResolver:
    pass
```

### 2. **Remove base.py Import**
```python
# ❌ BEFORE
from .base import RegisteredResolver, ResolverEvent, ResolverEventReason, ResolverResult

# ✅ AFTER
# (Just remove it - ResolverResult can be a simple class or inline)
```

### 3. **Move ResolverConfig to TYPE_CHECKING**
```python
# ❌ BEFORE
from DocsToKG.ContentDownload.pipeline import ResolverConfig

# ✅ AFTER
if TYPE_CHECKING:  # pragma: no cover
    from DocsToKG.ContentDownload.core import WorkArtifact
    # ResolverConfig no longer imported - use Any for config parameter
```

### 4. **Update Method Signatures**
```python
# ❌ BEFORE
def is_enabled(self, config: "ResolverConfig", artifact: "WorkArtifact") -> bool:

# ✅ AFTER
def is_enabled(self, config: Any, artifact: "WorkArtifact") -> bool:
```

### 5. **Keep iter_urls() Implementation Unchanged**

The method signature and implementation stay the same - only the class structure changes.

```python
# Logic stays identical - only remove dependency on RegisteredResolver base class
def iter_urls(
    self,
    client: httpx.Client,
    config: Any,  # Changed from "ResolverConfig"
    artifact: "WorkArtifact",
) -> Iterable[ResolverResult]:
    # Implementation unchanged
```

## Migration Checklist

For each resolver, verify:

- [ ] Remove inheritance from RegisteredResolver or ApiResolverBase
- [ ] Remove `from .base import ...` statement
- [ ] Change `config: "ResolverConfig"` → `config: Any`
- [ ] Keep `@register_v2("name")` decorator
- [ ] Keep `name = "resolver_name"` class variable
- [ ] Keep `is_enabled()` method logic unchanged
- [ ] Keep `iter_urls()` method logic unchanged
- [ ] Run tests: `pytest tests/ -k resolver_name`
- [ ] Verify pipeline can instantiate and use resolver

## Testing Each Migration

```bash
# Test single resolver
.venv/bin/pytest tests/content_download/ -k "test_example_resolver" -v

# Test all resolvers still work
.venv/bin/pytest tests/content_download/test_resolvers.py -v

# Test pipeline integration
.venv/bin/pytest tests/content_download/test_download_pipeline.py -v

# Type check
.venv/bin/mypy src/DocsToKG/ContentDownload/resolvers/
```

## Batch Migration Plan

### Batch 1 (5 resolvers) - Simplest
- arxiv.py - RegisteredResolver, simple iter_urls
- unpaywall.py - RegisteredResolver, simple iter_urls
- crossref.py - ApiResolverBase, HTTP calls
- core.py - ApiResolverBase, HTTP calls
- doaj.py - ApiResolverBase, HTTP calls

### Batch 2 (5 resolvers) - Moderate
- europe_pmc.py - ApiResolverBase, XML parsing
- landing_page.py - RegisteredResolver, with __init__
- semantic_scholar.py - RegisteredResolver, HTTP calls
- wayback.py - RegisteredResolver, with __init__, close()
- openalex.py - RegisteredResolver, simple

### Batch 3 (5 resolvers) - Complex
- zenodo.py - ApiResolverBase
- osf.py - ApiResolverBase
- openaire.py - RegisteredResolver
- hal.py - ApiResolverBase
- figshare.py - ApiResolverBase

## After All Resolvers Migrated

1. Delete `src/DocsToKG/ContentDownload/resolvers/base.py`
   - RegisteredResolver no longer used
   - ApiResolverBase no longer used
   - Helper types (ResolverEvent, etc.) can be moved to resolvers/__init__.py if needed

2. Delete `src/DocsToKG/ContentDownload/pipeline.py`
   - ResolverConfig no longer referenced at runtime
   - ResolverPipeline replaced by download_pipeline.py
   - All dependent code updated

3. Update imports in remaining files
   - Check for any remaining imports from base.py or pipeline.py
   - Update telemetry.py, streaming_schema.py, etc. if needed

4. Final verification
   - All tests passing
   - mypy clean
   - Lint clean
   - End-to-end system test passing

## ResolverResult Handling

Since we're removing ResolverResult from base.py, we have options:

### Option A: Simple inline definition (Recommended)
```python
class ResolverResult:
    def __init__(self, url=None, referer=None, metadata=None, 
                 event=None, event_reason=None, **kwargs):
        self.url = url
        self.referer = referer
        self.metadata = metadata or {}
        self.event = event
        self.event_reason = event_reason
        for k, v in kwargs.items():
            setattr(self, k, v)
```

### Option B: Typed dataclass
```python
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class ResolverResult:
    url: Optional[str] = None
    referer: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event: Optional[str] = None
    event_reason: Optional[str] = None
```

### Option C: Keep simple dict return (if DownloadPipeline supports it)

## Questions?

Refer back to the modern DownloadPipeline in `download_pipeline.py`:
- See how it calls `iter_urls()` method
- See how it handles ResolverResult objects
- Both old and new patterns are supported

