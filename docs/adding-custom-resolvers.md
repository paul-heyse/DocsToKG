# Adding Custom Content Download Resolvers

Use this guide to implement and register bespoke resolver providers within the
modular content download pipeline.

## 1. Resolver Template

Create a new module under
`src/DocsToKG/ContentDownload/resolvers/providers/your_resolver.py`:

```python
from DocsToKG.ContentDownload.resolvers.types import ResolverConfig, ResolverResult


class MyResolver:
    """Resolve organisation-specific repositories into download URLs."""

    name = "my_resolver"

    def is_enabled(self, config: ResolverConfig, artifact) -> bool:
        # Gate the resolver behind optional toggles or metadata availability.
        return True

    def iter_urls(self, session, config: ResolverConfig, artifact):
        # Yield ResolverResult instances for each candidate URL.
        yield ResolverResult(url="https://example.org/file.pdf")
```

Key points:

* `name` must be unique and referenced in configuration toggles.
* `iter_urls` should yield `ResolverResult` objects with either URLs or events.
* Use `DocsToKG.ContentDownload.http.request_with_retries` for HTTP calls to
  benefit from retry handling.

## 2. Register the Resolver

Add the resolver to the provider registry in
`src/DocsToKG/ContentDownload/resolvers/providers/__init__.py`:

```python
from .my_resolver import MyResolver

PROVIDERS = {
    # ... existing resolvers ...
    "my_resolver": MyResolver(),
}
```

Include the resolver in `default_resolvers()` if it should run by default.

## 3. Configuration Options

Custom resolvers automatically inherit `ResolverConfig` behaviour:

* `resolver_toggles["my_resolver"]`: Enable/disable the resolver.
* `resolver_timeouts["my_resolver"]`: Override request timeouts (seconds).
* `resolver_min_interval_s["my_resolver"]`: Enforce per-resolver rate limits.
* `resolver_head_precheck["my_resolver"]`: Opt out of HEAD preflight checks
  when upstream servers reject HEAD.

Document the resolver-specific options alongside your configuration files to
help operators adopt the new behaviour.

## 4. Testing Checklist

1. Write unit tests for happy-path URL extraction and error handling.
2. Add integration coverage in `tests/test_resolver_pipeline.py` if the resolver
   introduces unique behaviours.
3. Validate manifests via `python scripts/export_attempts_csv.py` to ensure
   attempt logging remains consistent.
4. Run `pytest tests/ -q` to exercise the full resolver suite.
