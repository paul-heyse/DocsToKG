# 1. Adding Custom Content Download Resolvers

This guide explains how to extend the DocsToKG content download pipeline with
project-specific resolver providers. The numbered sections walk through the
steps required to implement, register, and validate a new resolver.

## 1. Create the Resolver Template

Create a new module under
`src/DocsToKG/ContentDownload/resolvers/providers/your_resolver.py` and
implement the public surface expected by `ResolverConfig`:

```python
from __future__ import annotations

from typing import Iterable

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.resolvers.types import ResolverConfig, ResolverResult


class MyResolver:
    """Resolve organisation-specific repositories into download URLs."""

    name = "my_resolver"

    def is_enabled(self, config: ResolverConfig, artifact) -> bool:
        # Gate the resolver behind optional toggles or metadata availability.
        return config.toggles.get(self.name, True)

    def iter_urls(
        self,
        session: requests.Session,
        config: ResolverConfig,
        artifact,
    ) -> Iterable[ResolverResult]:
        try:
            response = request_with_retries(
                session,
                "get",
                "https://example.org/api",
                timeout=config.get_timeout(self.name),
                headers=config.polite_headers,
            )
        except requests.RequestException as exc:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="request-error",
                metadata={"error": str(exc)},
            )
            return

        if response.status_code != 200:
            yield ResolverResult(
                url=None,
                event="error",
                event_reason="http-error",
                http_status=response.status_code,
            )
            return

        data = response.json()
        # Extract relevant URLs from the response payload before yielding results.
        for candidate in data.get("files", []):
            yield ResolverResult(url=candidate["download_url"])
```

Key requirements:

- Implement `name`, `is_enabled`, and `iter_urls` as defined in
  `DocsToKG.ContentDownload.resolvers.types.Resolver`.
- Surface errors by yielding `ResolverResult` events instead of raising
  exceptions.
- Use `request_with_retries` for outbound HTTP to benefit from retry policies
  and structured logging.

## 2. Register the Resolver

Add the resolver to the provider registry in
`src/DocsToKG/ContentDownload/resolvers/providers/__init__.py` so the DocsToKG
pipeline can instantiate it:

```python
from .my_resolver import MyResolver


def default_resolvers() -> List[Resolver]:
    return [
        OpenAlexResolver(),
        # ... existing resolvers ...
        MyResolver(),
    ]
```

Order determines fallback precedence—place the resolver alongside providers
with similar behaviour.

## 3. Configure Resolver Options

All resolvers automatically honour standard configuration keys:

```yaml
resolver_toggles:
  my_resolver: true
resolver_timeouts:
  my_resolver: 20.0
resolver_min_interval_s:
  my_resolver: 0.5
resolver_head_precheck:
  my_resolver: true
```

Extend `ResolverConfig` only when new settings are unavoidable; most resolvers
can rely on the shared options above.

## 4. Validate the Resolver

1. Add unit tests in `tests/` covering success cases, HTTP errors, and malformed
   responses (see the Figshare and Zenodo tests for reference).
2. Update pipeline integration tests if resolver ordering changes.
3. Run `pytest tests/` to confirm all DocsToKG scenarios pass.

## 5. Document and Instrument

- Document new configuration keys or environment variables in the DocsToKG
  documentation set.
- Log unexpected conditions using `logging.getLogger(__name__)` inside the
  resolver to aid troubleshooting.

Following these steps keeps the resolver ecosystem consistent and ensures
observability, retries, and configuration toggles work out of the box.
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
