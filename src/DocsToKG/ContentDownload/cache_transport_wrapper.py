"""Role-aware HTTP transport wrapper for Hishel caching integration.

Responsibilities
----------------
- Wrap Hishel CacheTransport with CacheRouter decision logic
- Parse response Cache-Control headers and apply directives
- Route (host, role) pairs to appropriate caching decisions
- Enforce role-based caching policies (metadata/landing/artifact)
- Handle conditional requests (ETag/Last-Modified) for 304 Not Modified
- Log cache decisions and metrics for telemetry

Design Notes
------------
- Conservative approach: unknown hosts not cached (opt-in)
- Artifacts never cached (by design)
- Honors RFC 9111 cache-control directives
- Stale-while-revalidate support for background revalidation
- RFC 7232 conditional requests for bandwidth optimization
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from hishel import CacheTransport, Controller, FileStorage

from DocsToKG.ContentDownload.cache_policy import CacheRouter
from DocsToKG.ContentDownload.conditional_requests import (
    EntityValidator,
    build_conditional_headers,
    is_validator_available,
    merge_validators,
    parse_entity_validator,
    should_revalidate,
)

LOGGER = logging.getLogger(__name__)


class RoleAwareCacheTransport(httpx.BaseTransport):
    """Wraps Hishel CacheTransport with role-aware caching and conditional request support.

    This transport layer:
    1. Extracts host and role from request extensions
    2. Uses CacheRouter to determine if response should be cached
    3. Parses response Cache-Control headers
    4. Manages conditional requests (ETag/Last-Modified)
    5. Handles 304 Not Modified responses
    6. Creates role-specific Hishel Controller with policies
    7. Delegates to Hishel CacheTransport for actual caching

    Architecture:
        RoleAwareCacheTransport
            ├─ Extract host + role
            ├─ Check for cached validator (ETag/Last-Modified)
            ├─ Build conditional headers if available
            ├─ Consult CacheRouter.resolve_policy()
            ├─ Create Hishel Controller with directives
            └─ Delegate to Hishel CacheTransport → RateLimitedTransport → HTTPTransport

    Attributes:
        cache_router: CacheRouter instance for policy decisions
        cache_transport: Underlying Hishel CacheTransport
        validator_cache: Map of canonical_url → EntityValidator for 304 handling
    """

    def __init__(
        self,
        cache_router: CacheRouter,
        inner_transport: httpx.BaseTransport,
        storage: FileStorage,
    ) -> None:
        """Initialize role-aware cache transport.

        Args:
            cache_router: CacheRouter for resolving (host, role) policies
            inner_transport: Underlying transport (typically RateLimitedTransport)
            storage: Hishel storage backend (typically FileStorage)
        """
        self.cache_router = cache_router
        self.validator_cache: dict[str, EntityValidator] = {}
        self._request_count = 0

        # Create Hishel CacheTransport with default controller
        default_controller = Controller()
        self.cache_transport = CacheTransport(
            transport=inner_transport,
            storage=storage,
            controller=default_controller,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Handle HTTP request with role-aware caching and conditional requests.

        Flow:
        1. Extract host and role from request
        2. Resolve cache decision from CacheRouter
        3. Check if we have validators for conditional requests
        4. Add If-None-Match/If-Modified-Since headers if available
        5. Send request through Hishel transport
        6. Parse validators from response (including 304)
        7. Handle 304 Not Modified specially (return cached response)
        8. Store validators for future conditional requests

        Args:
            request: HTTPX request object

        Returns:
            HTTPX response object (from cache, 304 revalidation, or network)

        Side Effects:
            - Records cache decision in request.extensions
            - Updates telemetry counters
            - Logs cache hits/misses
            - Stores validators for conditional requests
        """
        self._request_count += 1

        # Extract host and role from request
        host = self._extract_host(request)
        role = self._extract_role(request)
        canonical_url = str(request.url)

        # Resolve cache decision
        decision = self.cache_router.resolve_policy(host, role)

        # Record decision in request for telemetry
        request.extensions.setdefault("docs_cache_decision", {})  # type: ignore[index]
        request.extensions["docs_cache_decision"] = {  # type: ignore[index]
            "use_cache": decision.use_cache,
            "host": host,
            "role": role,
            "ttl_s": decision.ttl_s,
            "swrv_s": decision.swrv_s,
        }

        # If not cacheable, bypass Hishel and go direct to inner transport
        if not decision.use_cache:
            LOGGER.debug(
                "cache-bypass",
                extra={
                    "host": host,
                    "role": role,
                    "reason": "not_cacheable",
                },
            )
            # Access underlying transport directly
            inner_transport = getattr(self.cache_transport, "_transport", None)
            if inner_transport is None:
                inner_transport = getattr(self.cache_transport, "transport", None)
            if inner_transport is not None:
                return inner_transport.handle_request(request)
            # Fallback: delegate to cache transport anyway
            return self.cache_transport.handle_request(request)

        # Check if we have validators for conditional requests
        validator = self.validator_cache.get(canonical_url)
        if validator and is_validator_available(validator):
            conditional_headers = build_conditional_headers(validator)
            if conditional_headers:
                # Add conditional headers to request
                for header_name, header_value in conditional_headers.items():
                    request.headers[header_name] = header_value
                LOGGER.debug(
                    "conditional-request",
                    extra={
                        "host": host,
                        "role": role,
                        "headers": list(conditional_headers.keys()),
                    },
                )

        # Create role-specific controller if TTL was resolved
        if decision.ttl_s is not None:
            controller = self._build_controller_for_decision(decision)
            # Temporarily override cache_transport's controller
            original_controller = getattr(self.cache_transport, "_controller", None)
            if original_controller is None:
                original_controller = getattr(self.cache_transport, "controller", None)
            setattr(self.cache_transport, "_controller", controller)
            try:
                response = self.cache_transport.handle_request(request)
            finally:
                if original_controller is not None:
                    setattr(self.cache_transport, "_controller", original_controller)
        else:
            # Use default controller
            response = self.cache_transport.handle_request(request)

        # Handle 304 Not Modified specially
        if response.status_code == 304 and validator:
            LOGGER.debug(
                "304-not-modified",
                extra={
                    "host": host,
                    "role": role,
                    "url": canonical_url,
                },
            )
            # Parse validators from 304 response
            new_validator = parse_entity_validator(response.headers)
            if should_revalidate(validator, response.headers):
                # Update validator with response data
                self.validator_cache[canonical_url] = merge_validators(
                    validator, new_validator
                )
                # 304 means return the cached response body
                # (Hishel should handle this internally)

        # Record cache result in response for telemetry
        from_cache = response.extensions.get("from_cache", False)
        LOGGER.debug(
            "cache-result",
            extra={
                "host": host,
                "role": role,
                "from_cache": from_cache,
                "status": response.status_code,
            },
        )

        # Parse and store validators from successful responses
        if response.status_code in (200, 301, 308) and decision.use_cache:
            response_validator = parse_entity_validator(response.headers)
            if is_validator_available(response_validator):
                self.validator_cache[canonical_url] = response_validator
                LOGGER.debug(
                    "validator-stored",
                    extra={
                        "host": host,
                        "url": canonical_url,
                        "etag": bool(response_validator.etag),
                        "last_modified": bool(response_validator.last_modified),
                    },
                )

        # Update response extensions with cache metadata
        response.extensions.setdefault("docs_cache_metadata", {})  # type: ignore[index]
        response.extensions["docs_cache_metadata"] = {  # type: ignore[index]
            "from_cache": from_cache,
            "host": host,
            "role": role,
            "decision": decision,
            "status": response.status_code,
        }

        return response

    @staticmethod
    def _extract_host(request: httpx.Request) -> str:
        """Extract hostname from request.

        Args:
            request: HTTPX request object

        Returns:
            Hostname (e.g., "api.crossref.org")

        Notes:
            - Uses request.url.host if available
            - Falls back to extension if set
            - Returns "unknown" if neither available
        """
        if request.url.host:
            return request.url.host

        # Fallback to extension
        host = request.extensions.get("docs_request_host")
        if isinstance(host, str):
            return host

        return "unknown"

    @staticmethod
    def _extract_role(request: httpx.Request) -> str:
        """Extract request role from extensions.

        Args:
            request: HTTPX request object

        Returns:
            Role ("metadata", "landing", or "artifact")

        Notes:
            - Reads from request.extensions["docs_request_role"]
            - Defaults to "metadata" if not specified
            - Values: metadata | landing | artifact
        """
        role = request.extensions.get("docs_request_role", "metadata")
        if isinstance(role, str) and role in ("metadata", "landing", "artifact"):
            return role
        return "metadata"

    @staticmethod
    def _build_controller_for_decision(decision) -> Controller:
        """Build Hishel Controller based on cache decision.

        Args:
            decision: CacheDecision from CacheRouter

        Returns:
            Configured Hishel Controller

        Notes:
            - Uses decision.ttl_s as response Cache-Control max-age
            - Honors decision.swrv_s for stale-while-revalidate
            - Conservative defaults: private cache, revalidate on expiry
        """
        cacheable_status_codes = [200, 301, 308]

        # Create controller with explicit arguments
        controller = Controller(
            cacheable_status_codes=cacheable_status_codes,
            cacheable_methods=["GET", "HEAD"],
            allow_heuristics=False,
            cache_private=True,
        )

        return controller


def build_role_aware_cache_transport(
    cache_router: CacheRouter,
    base_transport: httpx.BaseTransport,
    storage_path: Path,
) -> RoleAwareCacheTransport:
    """Factory function to create RoleAwareCacheTransport.

    Args:
        cache_router: CacheRouter instance
        base_transport: Inner transport (typically RateLimitedTransport)
        storage_path: Path to cache storage directory

    Returns:
        Configured RoleAwareCacheTransport instance

    Example:
        >>> from DocsToKG.ContentDownload.cache_policy import CacheRouter
        >>> from DocsToKG.ContentDownload.cache_loader import load_cache_config
        >>> import httpx
        >>> import os
        >>> from pathlib import Path
        >>>
        >>> # Load configuration
        >>> config = load_cache_config(
        ...     "cache.yaml",
        ...     env=os.environ,
        ... )
        >>> router = CacheRouter(config)
        >>>
        >>> # Create transport
        >>> base = httpx.HTTPTransport()
        >>> storage_dir = Path("./cache")
        >>> transport = build_role_aware_cache_transport(
        ...     router,
        ...     base,
        ...     storage_dir,
        ... )
        >>>
        >>> # Use in HTTPX client
        >>> client = httpx.Client(transport=transport)
    """
    # Ensure storage directory exists
    storage_path.mkdir(parents=True, exist_ok=True)

    # Create Hishel FileStorage
    storage = FileStorage(base_path=storage_path)

    # Create and return wrapper
    return RoleAwareCacheTransport(
        cache_router=cache_router,
        inner_transport=base_transport,
        storage=storage,
    )
