# 1. Module: rate_limit

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.io.rate_limit``.

## 1. Overview

Rate limiting primitives for ontology downloads.

## 2. Functions

### `_shared_bucket_path(http_config, key)`

Return the filesystem path for the shared token bucket state.

### `get_bucket()`

Return a registry-managed bucket.

### `apply_retry_after()`

Adjust bucket capacity after receiving a Retry-After hint.

### `reset()`

Clear all buckets (testing hook).

### `consume(self, tokens)`

Consume tokens from the bucket, sleeping until capacity is available.

### `_acquire_file_lock(self, handle)`

*No documentation available.*

### `_release_file_lock(self, handle)`

*No documentation available.*

### `_read_state(self, handle)`

*No documentation available.*

### `_write_state(self, handle, state)`

*No documentation available.*

### `_try_consume(self, tokens)`

*No documentation available.*

### `consume(self, tokens)`

Consume tokens from the shared bucket, waiting when insufficient.

### `_qualify(self, service, host)`

*No documentation available.*

### `_normalize_rate(self)`

*No documentation available.*

### `get_bucket(self)`

Return a token bucket for ``service``/``host`` using shared registry.

### `apply_retry_after(self)`

Reduce available tokens to honor server-provided retry-after hints.

### `reset(self)`

Clear all registered buckets (used in tests).

## 3. Classes

### `TokenBucket`

Token bucket used to enforce per-host and per-service rate limits.

### `SharedTokenBucket`

Token bucket backed by a filesystem state file for multi-process usage.

### `_BucketEntry`

*No documentation available.*

### `RateLimiterRegistry`

Manage shared token buckets keyed by (service, host).
