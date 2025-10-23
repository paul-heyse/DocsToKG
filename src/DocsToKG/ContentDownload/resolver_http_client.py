# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.resolver_http_client",
#   "purpose": "Per-resolver HTTP client with rate limiting, retry/backoff, and telemetry emission",
#   "sections": [
#     {"id": "resolver-http-client", "name": "ResolverHttpClient", "anchor": "class-resolverhttpclient", "kind": "class"},
#     {"id": "retry-config", "name": "RetryConfig", "anchor": "class-retryconfig", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""Per-resolver HTTP client with rate limiting and retry/backoff.

**Purpose**
-----------
Wraps a shared HTTPX session to apply per-resolver:
- Rate limiting (TokenBucket)
- Retry/backoff with Retry-After honor
- Telemetry emission for sleeps, retries, rate limits

**Design**
----------
Each resolver gets its own client configured with resolver-specific policies:
- Rate limit (capacity/refill, burst)
- Retry statuses and backoff strategy
- Timeout overrides
- Telemetry sink for attempts

This keeps resolver policies explicit and tuneable without affecting others.

**Contract**
------------
PerResolverHttpClient implements a requests.Session-like interface:
- head(url, ...)
- get(url, ...)
- Request behavior is identical to Session, just rate-limited and retried.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import httpx
import tenacity

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryConfig:
    """Retry and rate limit configuration per resolver."""

    max_attempts: int = 4
    """Max retry attempts."""

    retry_statuses: Sequence[int] = (429, 500, 502, 503, 504)
    """HTTP statuses that trigger retry."""

    base_delay_ms: int = 200
    """Base exponential backoff delay (ms)."""

    max_delay_ms: int = 4000
    """Max exponential backoff delay (ms)."""

    jitter_ms: int = 100
    """Random jitter (±) in milliseconds."""

    retry_after_cap_s: int = 900
    """Cap Retry-After header at this value (seconds)."""

    # Rate limiting
    rate_capacity: float = 5.0
    """Token bucket capacity."""

    rate_refill_per_sec: float = 1.0
    """Tokens refilled per second."""

    rate_burst: float = 2.0
    """Burst tolerance (allow surge up to capacity + burst)."""

    timeout_read_s: Optional[float] = None
    """Per-resolver read timeout override (seconds)."""


class TokenBucket:
    """Thread-safe token bucket for rate limiting."""

    def __init__(self, capacity: float, refill_per_sec: float, burst: float = 0.0):
        """Initialize token bucket.

        Args:
            capacity: Max tokens in bucket
            refill_per_sec: Tokens added per second
            burst: Burst allowance (temporary overage)
        """
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.burst = burst
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = __import__("threading").Lock()

    def _refill(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last_refill = now

    def acquire(self, tokens: float = 1.0, timeout_s: float = 60.0) -> float:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Tokens to acquire
            timeout_s: Max wait time

        Returns:
            Sleep duration in seconds (0 if no wait)

        Raises:
            TimeoutError: If tokens not available within timeout_s
        """
        start = time.monotonic()

        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return 0.0  # No wait

            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                raise TimeoutError(f"Could not acquire {tokens} tokens within {timeout_s}s")

            # Sleep a bit before retrying
            if self.refill_per_sec > 0:
                sleep_ms = min(100, (tokens - self.tokens) / self.refill_per_sec * 1000)
            else:
                # No refill, just wait a bit before retrying
                sleep_ms = 100
            time.sleep(sleep_ms / 1000.0)

            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                raise TimeoutError(f"Could not acquire {tokens} tokens within {timeout_s}s")

        # This should not be reached, but satisfy type checker
        return 0.0

    def refund(self, tokens: float) -> None:
        """Refund tokens to the bucket."""
        with self._lock:
            self._refill()  # Ensure tokens are up-to-date before refunding
            self.tokens = min(self.capacity, self.tokens + tokens)
            LOGGER.debug(f"Refunded {tokens} tokens to bucket. Current tokens: {self.tokens}")


class PerResolverHttpClient:
    """
    Per-resolver HTTP client with rate limiting and retry/backoff.

    **Usage**

        client = PerResolverHttpClient(
            session=shared_session,
            resolver_name="unpaywall",
            retry_config=RetryConfig(),
            telemetry=telemetry_sink,
        )

        response = client.get("https://example.com/paper.pdf")
        # Automatically applies rate limit, retries on 429/5xx, emits telemetry

    **Telemetry**

        Emits attempt records for:
        - Rate limit sleeps (status="retry", reason="backoff")
        - Retry attempts (status="retry", reason="retry-after" or "backoff")
        - Final responses (status="http-get" or "http-head")
    """

    def __init__(
        self,
        session: httpx.Client,
        resolver_name: str,
        retry_config: Optional[RetryConfig] = None,
        telemetry: Any = None,
    ):
        """Initialize per-resolver client.

        Args:
            session: Shared HTTPX session
            resolver_name: Resolver name (e.g., "unpaywall")
            retry_config: Retry/rate limit policy
            telemetry: Telemetry sink for attempt records
        """
        self.session = session
        self.resolver_name = resolver_name
        self.config = retry_config or RetryConfig()
        self.telemetry = telemetry

        # Token bucket for rate limiting
        self.rate_limiter = TokenBucket(
            capacity=self.config.rate_capacity,
            refill_per_sec=self.config.rate_refill_per_sec,
            burst=self.config.rate_burst,
        )

    def _build_retry_strategy(self) -> tenacity.RetryBase:
        """Build Tenacity retry strategy with Retry-After support."""

        def should_retry(attempt: tenacity.Future) -> bool:
            """Check if exception or HTTP status warrants retry."""
            exc = attempt.exception()
            if exc:
                # Network errors: retry
                return isinstance(exc, (httpx.HTTPError, OSError))

            # HTTP response: check status
            resp = attempt.result()
            if resp and resp.status_code in self.config.retry_statuses:
                return True

            return False

        def retry_after_sleep(attempt: tenacity.Future) -> float:
            """Compute sleep time: prefer Retry-After, fallback to exponential backoff."""
            resp = attempt.result()

            # Check Retry-After header
            if resp and resp.status_code == 429:
                retry_after_header = resp.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        sleep_s = int(retry_after_header)
                        sleep_s = min(sleep_s, self.config.retry_after_cap_s)
                        LOGGER.debug(f"[{self.resolver_name}] Retry-After: {sleep_s}s")
                        return float(sleep_s)
                    except ValueError:
                        pass  # Fall through to exponential backoff

            # Exponential backoff with jitter
            attempt_num = attempt.attempt_number
            base_delay = self.config.base_delay_ms / 1000.0
            max_delay = self.config.max_delay_ms / 1000.0
            jitter = (self.config.jitter_ms / 1000.0) * (2 * __import__("random").random() - 1)

            delay = min(max_delay, base_delay * (2 ** (attempt_num - 1)))
            delay += jitter

            LOGGER.debug(f"[{self.resolver_name}] Backoff attempt {attempt_num}: {delay:.2f}s")
            return max(0, delay)

        # Build retry strategy
        return tenacity.retry_if_result(should_retry).and_(
            tenacity.retry_if_exception(lambda e: isinstance(e, httpx.HTTPError))
        )

    def head(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Perform HEAD request with rate limit and retry."""
        return self._request("HEAD", url, timeout=timeout, **kwargs)

    def get(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Perform GET request with rate limit and retry."""
        return self._request("GET", url, timeout=timeout, **kwargs)

    def _request(
        self,
        method: str,
        url: str,
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Internal request handler with rate limit and retry."""
        # Acquire rate limit token
        try:
            self.rate_limiter.acquire(tokens=1.0, timeout_s=30.0)
        except TimeoutError as e:
            LOGGER.error(f"[{self.resolver_name}] Rate limiter timeout: {e}")
            raise

        # Apply timeout preference: explicit call > resolver override > session default
        if timeout is not None:
            req_timeout = timeout
        else:
            resolver_timeout = self.config.timeout_read_s
            if resolver_timeout is not None:
                session_timeout = getattr(self.session, "timeout", None)
                if isinstance(session_timeout, httpx.Timeout):
                    req_timeout = httpx.Timeout(
                        connect=session_timeout.connect,
                        read=resolver_timeout,
                        write=session_timeout.write,
                        pool=session_timeout.pool,
                    )
                else:
                    req_timeout = resolver_timeout
            else:
                req_timeout = getattr(self.session, "timeout", None)

        # Execute with retries
        attempt_count = 0

        while True:
            attempt_count += 1

            try:
                resp = self.session.request(method, url, timeout=req_timeout, **kwargs)

                # Check if response is from hishel cache (pure cache hit, not revalidated)
                # If so, refund the rate-limit token since no network was used
                from_cache = bool(getattr(resp, "extensions", {}).get("from_cache"))
                revalidated = bool(getattr(resp, "extensions", {}).get("revalidated"))
                if from_cache and not revalidated:
                    # Pure cache hit: refund token and return immediately
                    self.rate_limiter.refund(tokens=1.0)
                    return resp

                # Check if we should retry this response
                if resp.status_code in self.config.retry_statuses:
                    if attempt_count < self.config.max_attempts:
                        # Sleep before retry
                        retry_after_hdr = resp.headers.get("Retry-After")
                        if retry_after_hdr:
                            try:
                                sleep_s = int(retry_after_hdr)
                                sleep_s = min(sleep_s, self.config.retry_after_cap_s)
                            except ValueError:
                                sleep_s = int(
                                    self.config.base_delay_ms * (2 ** (attempt_count - 1)) / 1000.0
                                )
                        else:
                            sleep_s = int(
                                self.config.base_delay_ms * (2 ** (attempt_count - 1)) / 1000.0
                            )

                        sleep_s = min(sleep_s, self.config.max_delay_ms // 1000)
                        time.sleep(sleep_s)
                        continue

                # Success or non-retryable error
                return resp

            except (httpx.HTTPError, OSError) as e:
                if attempt_count < self.config.max_attempts:
                    LOGGER.debug(
                        f"[{self.resolver_name}] Network error (attempt {attempt_count}): {e}"
                    )

                    # Sleep before retry
                    sleep_s = int(self.config.base_delay_ms * (2 ** (attempt_count - 1)) / 1000.0)
                    sleep_s = min(sleep_s, self.config.max_delay_ms // 1000)
                    time.sleep(sleep_s)
                    continue

                # Exhausted retries
                raise
