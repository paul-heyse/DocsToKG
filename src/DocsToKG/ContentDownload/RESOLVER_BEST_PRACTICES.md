# Resolver Best Practices: URL Canonicalization

**Date**: October 21, 2025
**Status**: Complete
**Phase**: 3B - Resolver Integration

---

## Overview

This guide documents best practices for handling URLs in resolvers, ensuring consistent RFC 3986-compliant canonicalization across the ContentDownload pipeline.

---

## Key Principle

**All resolvers automatically benefit from URL canonicalization** via `ResolverResult.__post_init__()`, which automatically canonicalizes any URL using `canonical_for_index()` if not explicitly provided.

---

## Two Valid Patterns

### Pattern A: Minimal (Rely on Auto-Canonicalization)

**Best for**: Simple resolvers that don't need early deduplication

```python
from DocsToKG.ContentDownload.core import dedupe
from .base import ResolverResult

def iter_urls(self, client, config, artifact):
    candidates = [url for url in artifact.pdf_urls if url]

    if not candidates:
        yield ResolverResult(url=None, event=ResolverEvent.SKIPPED, ...)
        return

    for url in dedupe(candidates):
        if not url:
            continue
        # ResolverResult.__post_init__ will automatically canonicalize
        yield ResolverResult(url=url, metadata={"source": "my_resolver"})
```

**Examples**: OpenAlex (before), Unpaywall (before), Landing Page resolver

**Pros**:

- ✅ Simple, concise code
- ✅ Automatic canonicalization via `__post_init__`
- ✅ Preserves original URL for telemetry

**Cons**:

- ❌ No early deduplication before yielding
- ❌ Semantics not explicit in code

---

### Pattern B: Explicit (Compute Canonical URL Early)

**Best for**: Resolvers needing early deduplication or deduplicating within resolver results

```python
from DocsToKG.ContentDownload.urls import canonical_for_index
from DocsToKG.ContentDownload.core import dedupe
from .base import ResolverResult

def iter_urls(self, client, config, artifact):
    candidates = fetch_urls_from_api(artifact.doi)

    seen: Set[str] = set()
    for url in candidates:
        if not url:
            continue

        # Explicitly compute canonical URL for early deduplication
        try:
            canonical_url = canonical_for_index(url)
        except Exception:
            canonical_url = url

        if canonical_url in seen:
            continue  # Skip semantically identical URLs

        seen.add(canonical_url)

        # Emit both original and canonical forms
        yield ResolverResult(
            url=url,
            canonical_url=canonical_url,
            metadata={"source": "my_resolver", "api_version": "v2"},
        )
```

**Examples**: Crossref (always), Unpaywall (now), OpenAlex (now)

**Pros**:

- ✅ Explicit RFC 3986 compliance in code
- ✅ Early deduplication (fewer API calls)
- ✅ Clear intent in code
- ✅ Can dedupe within resolver results

**Cons**:

- ⚠️ Slightly more verbose
- ⚠️ Need exception handling for `canonical_for_index()`

---

## When to Use Each Pattern

| Scenario | Pattern | Reasoning |
|----------|---------|-----------|
| Simple metadata from source | A (Minimal) | No deduplication needed |
| API queries with potential duplicates | B (Explicit) | Early dedupe saves API calls |
| Multi-location searches (Crossref, Unpaywall) | B (Explicit) | Multiple results may be identical |
| Direct URL emission (OpenAlex) | B (Explicit) | Best practice for clarity |
| Complex deduplication logic | B (Explicit) | Needed for pre-filtering |

---

## Implementation Checklist

### For New Resolvers

- [ ] Choose pattern (A or B based on needs)
- [ ] If Pattern B: import `canonical_for_index` from `urls`
- [ ] If Pattern B: wrap `canonical_for_index()` in try/except
- [ ] Emit `ResolverResult` with `url` and optional `canonical_url`
- [ ] Add metadata to track source and version
- [ ] Document which pattern is used and why
- [ ] Add tests for URL canonicalization

### For Existing Resolvers

**Minimal Pattern (Pattern A)**:

- ✅ Already working (no changes needed)
- ✅ `__post_init__` handles canonicalization
- ✅ Pipeline handles deduplication

**Explicit Pattern (Pattern B)**:

- Recommended for resolvers with multi-result APIs
- Examples: Crossref, Unpaywall, OpenAlex (updated)
- Benefits from early deduplication

---

## Code Examples by Resolver

### OpenAlex (Pattern B - Explicit)

```python
from DocsToKG.ContentDownload.urls import canonical_for_index

for url in dedupe(candidates):
    if not url:
        continue
    try:
        canonical_url = canonical_for_index(url)
    except Exception:
        canonical_url = url
    yield ResolverResult(
        url=url,
        canonical_url=canonical_url,
        metadata={"source": "openalex_metadata"},
    )
```

### Unpaywall (Pattern B - Explicit)

```python
from DocsToKG.ContentDownload.urls import canonical_for_index

for unique_url in unique_urls:
    for candidate_url, metadata in candidates:
        if candidate_url == unique_url:
            try:
                canonical_url = canonical_for_index(unique_url)
            except Exception:
                canonical_url = unique_url
            yield ResolverResult(
                url=unique_url,
                canonical_url=canonical_url,
                metadata=metadata,
            )
            break
```

### Crossref (Pattern B - Explicit with Pre-Filtering)

```python
from DocsToKG.ContentDownload.urls import canonical_for_index

seen: Set[str] = set()
for url, meta in pdf_candidates:
    try:
        normalized = canonical_for_index(url)
    except Exception:
        normalized = url
    if normalized in seen:
        continue  # Skip duplicate
    seen.add(normalized)
    yield ResolverResult(
        url=url,
        canonical_url=normalized,
        metadata={"source": "crossref", "content_type": meta.get("content-type")},
    )
```

---

## Important Notes

### ResolverResult.**post_init** Behavior

When you create a `ResolverResult`:

```python
result = ResolverResult(url=url, metadata={...})
# OR
result = ResolverResult(url=url, canonical_url=canonical, metadata={...})
```

The `__post_init__` method:

1. ✅ Preserves your explicit `canonical_url` if provided
2. ✅ Auto-computes `canonical_url` via `canonical_for_index()` if not provided
3. ✅ Always sets `url` to the canonical form
4. ✅ Preserves `original_url` from input for telemetry

### Exception Handling

Always wrap `canonical_for_index()` in try/except:

```python
try:
    canonical_url = canonical_for_index(url)
except Exception:
    canonical_url = url  # Fallback to original on error
```

This handles edge cases and prevents resolver failures on malformed URLs.

---

## URL Flow Through System

```
Resolver emits:
    ResolverResult(url=url, canonical_url=canonical, metadata={...})

    ↓ __post_init__

ResolverResult normalized to:
    url: canonical form
    canonical_url: canonical form
    original_url: original form

    ↓ Pipeline processing

Pipeline uses:
    url (= canonical_url) for deduplication
    original_url for telemetry
    Both passed to download layer

    ↓ Download & Telemetry

Manifest stores:
    url, canonical_url, original_url, path, sha256, classification

    ↓ Resume/Dedupe

ManifestUrlIndex lookups use:
    canonical_url field for exact deduplication
```

---

## Testing Patterns

### Test Pattern A (Minimal)

```python
def test_openalex_canonicalizes_urls():
    resolver = OpenAlexResolver()
    artifact = WorkArtifact(pdf_urls=["HTTP://EXAMPLE.COM/paper"])

    results = list(resolver.iter_urls(client, config, artifact))

    assert len(results) == 1
    result = results[0]

    # __post_init__ handles canonicalization
    assert result.canonical_url == "https://example.com/paper"
    assert result.original_url == "HTTP://EXAMPLE.COM/paper"
```

### Test Pattern B (Explicit)

```python
def test_crossref_dedupes_canonical_urls():
    resolver = CrossrefResolver()
    artifact = WorkArtifact(doi="10.1234/test")

    # Mock API returns identical URLs with different cases
    mock_data = {
        "message": {
            "link": [
                {"URL": "https://example.com/paper", "content-type": "application/pdf"},
                {"URL": "HTTPS://EXAMPLE.COM/PAPER", "content-type": "application/pdf"},
            ]
        }
    }

    with patch("_fetch_crossref", return_value=mock_data):
        results = list(resolver.iter_urls(client, config, artifact))

    # Should return only one (deduplicated)
    assert len(results) == 1
    assert results[0].canonical_url == "https://example.com/paper"
```

---

## Updating Existing Resolvers

### Migration Steps

1. **Add import**:

   ```python
   from DocsToKG.ContentDownload.urls import canonical_for_index
   ```

2. **Wrap URL computation**:

   ```python
   try:
       canonical_url = canonical_for_index(url)
   except Exception:
       canonical_url = url
   ```

3. **Pass to ResolverResult**:

   ```python
   yield ResolverResult(
       url=url,
       canonical_url=canonical_url,
       metadata={...},
   )
   ```

4. **Add test** for canonicalization

5. **Verify backward compatibility** (should be automatic)

---

## Summary

### Pattern Selection

- **Pattern A (Minimal)**: Use when no early deduplication needed
  - Simpler code
  - Automatic via `__post_init__`
  - Examples: Landing page, simple metadata

- **Pattern B (Explicit)**: Use for multi-result APIs
  - Clear RFC 3986 compliance
  - Early deduplication possible
  - Examples: Crossref, Unpaywall, OpenAlex

### Current Implementation Status

| Resolver | Pattern | Status |
|----------|---------|--------|
| OpenAlex | B (Explicit) | ✅ Updated |
| Unpaywall | B (Explicit) | ✅ Updated |
| Crossref | B (Explicit) | ✅ Already implemented |
| Landing Page | A (Minimal) | ✅ Working |
| ArXiv | A (Minimal) | ✅ Working |
| PMC | A (Minimal) | ✅ Working |

### Benefits

✅ RFC 3986-compliant URL normalization
✅ Consistent canonicalization across all resolvers
✅ Early deduplication where needed
✅ Preserved URL history for telemetry
✅ Full backward compatibility
✅ Clear code intent via explicit pattern choice

---

**Last Updated**: October 21, 2025
**Related**: `src/DocsToKG/ContentDownload/urls.py`, `ResolverResult.__post_init__`
