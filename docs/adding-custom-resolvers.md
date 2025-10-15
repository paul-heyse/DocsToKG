# Adding Custom Content Download Resolvers

This guide explains how to add new resolver providers to the modular
DocsToKG Content Download pipeline.

## 1. Resolver Skeleton

Create a new module under
`src/DocsToKG/ContentDownload/resolvers/providers/<name>.py`:

```python
from __future__ import annotations

from typing import Iterable

import requests

from DocsToKG.ContentDownload.http import request_with_retries
from DocsToKG.ContentDownload.resolvers.types import ResolverConfig, ResolverResult


class MyResolver:
    """Resolve PDFs for a custom repository."""

    name = "my_resolver"

    def is_enabled(self, config: ResolverConfig, artifact) -> bool:
        return True  # adjust to inspect artifact metadata

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
                metadata={"error_detail": "Unexpected status"},
            )
            return

        data = response.json()
        # TODO: Extract URLs from `data`
        yield ResolverResult(url="https://example.org/file.pdf")
```

Key requirements:

- Implement `name`, `is_enabled`, and `iter_urls` following the
  `Resolver` protocol documented in `resolvers/types.py`.
- Surface errors by yielding `ResolverResult` events instead of raising
  exceptions.
- Use `request_with_retries` for outbound HTTP to benefit from retry
  policies and logging.

## 2. Registering the Resolver

Update `src/DocsToKG/ContentDownload/resolvers/providers/__init__.py`
so that `default_resolvers()` instantiates your resolver in the desired
position.

```python
from .my_resolver import MyResolver


def default_resolvers() -> List[Resolver]:
    return [
        OpenAlexResolver(),
        # ... existing resolvers ...
        MyResolver(),
    ]
```

## 3. Configuration Options

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

You can add resolver-specific settings by extending `ResolverConfig` if
needed, but most providers can rely on the shared options above.

## 4. Testing Checklist

1. Add unit tests in `tests/` covering success, HTTP errors, and malformed
   responses. Use fixtures similar to the Figshare and Zenodo tests.
2. Update the full pipeline integration test if resolver ordering changes.
3. Run `pytest tests/` to confirm all scenarios pass.

## 5. Documentation and Logging

- Document new configuration keys or environment variables in
  `docs/` as appropriate.
- Log unexpected conditions using `logging.getLogger(__name__)` inside your
  resolver to aid troubleshooting.

Following these steps keeps the resolver ecosystem consistent and ensures
observability, retries, and configuration toggles work out of the box.
