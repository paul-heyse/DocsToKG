"""Shared HTTPX client factory with Hishel caching for ContentDownload.

Responsibilities
----------------
- Construct a process-wide :class:`httpx.Client` configured with HTTP/2 (when
  available), connection pooling limits, timeout budgets, and a Certifi-backed
  SSL context.
- Wrap the underlying transport with :class:`hishel.CacheTransport` so metadata
  requests benefit from RFC-9111 caching while still allowing callers to inject
  custom transports (e.g., :class:`httpx.MockTransport`) for tests.
- Expose helpers (:func:`configure_http_client`, :func:`get_http_client`,
  :func:`reset_http_client_for_tests`, :func:`purge_http_cache`) that make it
  easy to override transports, event hooks, or proxy mounts without touching
  module globals directly.
- Gracefully fall back to HTTP/1.1 when the optional ``h2`` dependency is not
  installed, ensuring production and test environments remain operational.

Design Notes
------------
- Event hooks annotate each request/response with telemetry metadata used by
  the downloader and resolver pipeline for logging and cache introspection.
- Cache directories honour ``DOCSTOKG_DATA_ROOT`` to keep test runs isolated
  while defaulting to ``Data/cache/http/ContentDownload`` during development.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import ssl
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import certifi
import httpx

from DocsToKG.ContentDownload.cache_loader import load_cache_config
from DocsToKG.ContentDownload.cache_policy import CacheRouter
from DocsToKG.ContentDownload.cache_transport_wrapper import (
    RoleAwareCacheTransport,
    build_role_aware_cache_transport,
)
from DocsToKG.ContentDownload.ratelimit import (
    RateLimitedTransport,
    get_rate_limiter_manager,
)

LOGGER = logging.getLogger("DocsToKG.ContentDownload.network")

_CLIENT_LOCK = threading.RLock()
_HTTP_CLIENT: Optional[httpx.Client] = None
_CURRENT_OVERRIDES: Dict[str, object] = {}
_CACHE_ROUTER: Optional[CacheRouter] = None

_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
_DEFAULT_LIMITS = httpx.Limits(
    max_connections=128, max_keepalive_connections=32, keepalive_expiry=15.0
)


def _resolve_data_root() -> Path:
    """Return the root directory used for storing cache artefacts."""

    env_value = os.environ.get("DOCSTOKG_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "Data"


def _resolve_cache_dir() -> Path:
    cache_dir = _resolve_data_root() / "cache" / "http" / "ContentDownload"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _build_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.load_verify_locations(certifi.where())
    return context


def _request_hook(request: httpx.Request) -> None:
    canonical_url = request.extensions.get("docs_canonical_url")
    if isinstance(canonical_url, str) and canonical_url and canonical_url != str(request.url):
        request.url = httpx.URL(canonical_url)

    meta: MutableMapping[str, object] = request.extensions.setdefault("docs_network_meta", {})  # type: ignore[assignment]
    meta["client"] = "httpx"
    meta["start_time"] = time.perf_counter()
    if canonical_url:
        meta["canonical_url"] = canonical_url
    original_url = request.extensions.get("docs_original_url")
    if isinstance(original_url, str) and original_url:
        meta["original_url"] = original_url
    canonical_index = request.extensions.get("docs_canonical_index")
    if isinstance(canonical_index, str) and canonical_index:
        meta["canonical_index"] = canonical_index
    attempt = meta.get("attempt")
    if isinstance(attempt, int):
        meta["attempt"] = attempt + 1
    else:
        meta["attempt"] = 1


def _response_hook(response: httpx.Response) -> None:
    meta: MutableMapping[str, object] = response.request.extensions.setdefault(  # type: ignore[assignment]
        "docs_network_meta", {}
    )
    start_time = meta.get("start_time")
    if isinstance(start_time, (int, float)):
        meta["elapsed"] = time.perf_counter() - start_time
    cache_status = bool(response.extensions.get("from_cache"))
    meta["from_cache"] = cache_status
    if cache_status:
        metadata = response.extensions.get("cache_metadata")
        if isinstance(metadata, dict):
            meta["cache_metadata"] = metadata
    LOGGER.debug(
        "httpx-response",
        extra={
            "url": str(response.request.url),
            "status": response.status_code,
            "cache_hit": cache_status,
        },
    )


def _build_event_hooks(extra_hooks: Optional[Mapping[str, Iterable]]) -> Dict[str, list]:
    hooks: Dict[str, list] = {
        "request": [_request_hook],
        "response": [_response_hook],
    }
    if extra_hooks:
        for name, values in extra_hooks.items():
            if not values:
                continue
            target = hooks.setdefault(name, [])
            target.extend(values)
    return hooks


def configure_http_client(
    *,
    proxy_mounts: Optional[Mapping[str, httpx.BaseTransport]] = None,
    transport: Optional[httpx.BaseTransport] = None,
    event_hooks: Optional[Mapping[str, Iterable]] = None,
) -> None:
    """Override HTTPX client configuration and rebuild the singleton on next use."""

    with _CLIENT_LOCK:
        _CURRENT_OVERRIDES["proxy_mounts"] = dict(proxy_mounts) if proxy_mounts else None
        _CURRENT_OVERRIDES["transport"] = transport
        _CURRENT_OVERRIDES["event_hooks"] = dict(event_hooks) if event_hooks else None
        _close_client_unlocked()


def reset_http_client_for_tests() -> None:
    """Clear overrides and dispose of the cached client."""

    with _CLIENT_LOCK:
        _CURRENT_OVERRIDES.clear()
        _close_client_unlocked()


def purge_http_cache() -> None:
    """Remove all cached HTTP artefacts."""

    cache_dir = _resolve_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _CLIENT_LOCK:
        _close_client_unlocked()


def get_http_client() -> httpx.Client:
    """Return the shared HTTPX client instance wrapped with Hishel caching."""

    with _CLIENT_LOCK:
        if _HTTP_CLIENT is None:
            _create_client_unlocked()
        assert _HTTP_CLIENT is not None
        return _HTTP_CLIENT


def _close_client_unlocked() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        with contextlib.suppress(Exception):
            _HTTP_CLIENT.close()
    _HTTP_CLIENT = None


def _load_or_create_cache_router() -> CacheRouter:
    """Load cache configuration and create CacheRouter instance.

    Strategy:
    1. Load from cache.yaml if it exists in ContentDownload/config/
    2. Apply environment variable overrides
    3. Create CacheRouter with loaded config

    Returns:
        CacheRouter instance for role-aware cache decisions

    Notes:
        - Caches router in module-level _CACHE_ROUTER
        - Conservative defaults: unknown hosts not cached
        - Graceful fallback if config missing
    """
    global _CACHE_ROUTER

    if _CACHE_ROUTER is not None:
        return _CACHE_ROUTER

    # Try to load cache.yaml from ContentDownload/config/
    cache_yaml = Path(__file__).parent / "config" / "cache.yaml"
    cache_yaml_path = str(cache_yaml) if cache_yaml.exists() else None

    try:
        config = load_cache_config(
            cache_yaml_path,
            env=os.environ,
        )
        _CACHE_ROUTER = CacheRouter(config)
        LOGGER.info("Cache configuration loaded from %s", cache_yaml_path or "defaults")
    except Exception as e:
        LOGGER.warning("Failed to load cache configuration: %s; using defaults", e)
        # Create minimal config with conservative defaults
        from DocsToKG.ContentDownload.cache_loader import (
            CacheConfig,
            CacheControllerDefaults,
            CacheDefault,
            CacheStorage,
            StorageKind,
        )

        minimal_config = CacheConfig(
            storage=CacheStorage(kind=StorageKind.FILE, path=str(_resolve_cache_dir())),
            controller=CacheControllerDefaults(default=CacheDefault.DO_NOT_CACHE),
            hosts={},
        )
        _CACHE_ROUTER = CacheRouter(minimal_config)

    return _CACHE_ROUTER


def initialize_rate_limiter_from_config(rate_config: Optional[object] = None) -> None:
    """Initialize the global rate limiter with Phase 7 configuration.

    Args:
        rate_config: Optional RateConfig from args.resolve_config (Phase 7).
                    If None, uses defaults from get_rate_limiter_manager().
    """
    if rate_config is None:
        # Just ensure the manager is initialized with defaults
        _manager = get_rate_limiter_manager()
        LOGGER.info("Rate limiter initialized with defaults")
    else:
        # Initialize with provided config
        try:
            # rate_config is a RateConfig from ratelimits_loader
            _manager = get_rate_limiter_manager(cfg=rate_config)
            LOGGER.info(
                "Rate limiter initialized with Phase 7 config (backend: %s, inflight: %s, AIMD: %s)",
                rate_config.backend.kind,
                rate_config.global_max_inflight,
                rate_config.aimd.enabled,
            )
        except Exception as e:
            LOGGER.warning("Failed to initialize rate limiter from config: %s; using defaults", e)
            get_rate_limiter_manager()


def _create_client_unlocked() -> None:
    global _HTTP_CLIENT

    cache_dir = _resolve_cache_dir()

    limiter_manager = get_rate_limiter_manager()
    base_transport: Optional[httpx.BaseTransport] = _CURRENT_OVERRIDES.get("transport")  # type: ignore[assignment]

    # Load cache router for role-aware caching
    cache_router = _load_or_create_cache_router()

    if isinstance(base_transport, RoleAwareCacheTransport):
        # Already wrapped, use as-is
        cache_transport = base_transport
    else:
        # Create role-aware cache transport with nested rate limiter
        base_transport = base_transport or httpx.HTTPTransport(retries=0)
        if not isinstance(base_transport, RateLimitedTransport):
            # Wrap with rate limiter
            base_transport = RateLimitedTransport(base_transport, registry=limiter_manager)

        # Wrap with role-aware cache transport
        cache_transport = build_role_aware_cache_transport(
            cache_router,
            base_transport,
            cache_dir,
        )

    event_hooks = _build_event_hooks(_CURRENT_OVERRIDES.get("event_hooks"))  # type: ignore[arg-type]

    mounts: Optional[Mapping[str, httpx.BaseTransport]]
    override_mounts = _CURRENT_OVERRIDES.get("proxy_mounts")  # type: ignore[assignment]
    if override_mounts:
        mounts = dict(override_mounts)  # type: ignore[arg-type]
    else:
        mounts = None

    client_kwargs = dict(
        transport=cache_transport,
        timeout=_DEFAULT_TIMEOUT,
        limits=_DEFAULT_LIMITS,
        verify=_build_ssl_context(),
        trust_env=True,
        event_hooks=event_hooks,
        mounts=mounts,
    )

    try:
        _HTTP_CLIENT = httpx.Client(http2=True, **client_kwargs)
    except ImportError as exc:  # pragma: no cover - graceful fallback when h2 missing
        if "http2" in str(exc) and "h2" in str(exc):
            LOGGER.warning(
                "HTTP/2 support unavailable (missing 'h2' package); falling back to HTTP/1.1 transport."
            )
            _HTTP_CLIENT = httpx.Client(http2=False, **client_kwargs)
        else:
            raise
