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
from hishel import CacheTransport, FileStorage

from DocsToKG.ContentDownload.core import normalize_url

LOGGER = logging.getLogger("DocsToKG.ContentDownload.network")

_CLIENT_LOCK = threading.RLock()
_HTTP_CLIENT: Optional[httpx.Client] = None
_CURRENT_OVERRIDES: Dict[str, object] = {}

_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
_DEFAULT_LIMITS = httpx.Limits(max_connections=128, max_keepalive_connections=32, keepalive_expiry=15.0)


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
    # Normalise the target URL for cache key stability.
    try:
        canonical = normalize_url(str(request.url))
    except Exception:  # pragma: no cover - defensive guard
        canonical = str(request.url)
    if canonical != str(request.url):
        request.url = httpx.URL(canonical)

    meta: MutableMapping[str, object] = request.extensions.setdefault("docs_network_meta", {})  # type: ignore[assignment]
    meta["client"] = "httpx"
    meta["start_time"] = time.perf_counter()
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


def _create_client_unlocked() -> None:
    global _HTTP_CLIENT

    cache_dir = _resolve_cache_dir()
    storage = FileStorage(base_path=cache_dir)

    base_transport = _CURRENT_OVERRIDES.get("transport")
    if isinstance(base_transport, CacheTransport):
        cache_transport = base_transport
    else:
        base_transport = base_transport or httpx.HTTPTransport(retries=0)
        cache_transport = CacheTransport(transport=base_transport, storage=storage)

    event_hooks = _build_event_hooks(_CURRENT_OVERRIDES.get("event_hooks"))  # type: ignore[arg-type]

    mounts: Optional[Mapping[str, httpx.BaseTransport]]
    override_mounts = _CURRENT_OVERRIDES.get("proxy_mounts")
    if override_mounts:
        mounts = dict(override_mounts)  # type: ignore[arg-type]
    else:
        mounts = None

    _HTTP_CLIENT = httpx.Client(
        http2=True,
        transport=cache_transport,
        timeout=_DEFAULT_TIMEOUT,
        limits=_DEFAULT_LIMITS,
        verify=_build_ssl_context(),
        trust_env=True,
        event_hooks=event_hooks,
        mounts=mounts,
    )
