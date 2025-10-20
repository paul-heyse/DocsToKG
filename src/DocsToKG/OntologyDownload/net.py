# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.net",
#   "purpose": "Provide a shared HTTPX client with Hishel caching for ontology networking",
#   "sections": [
#     {"id": "constants", "name": "Constants & globals", "anchor": "CONST", "kind": "constants"},
#     {"id": "helpers", "name": "Client construction helpers", "anchor": "HELP", "kind": "helpers"},
#     {"id": "api", "name": "Public API", "anchor": "API", "kind": "api"}
#   ]
# }
# === /NAVMAP ===

"""Shared HTTPX + Hishel client used across OntologyDownload networking."""

from __future__ import annotations

import contextlib
import logging
import ssl
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional

import certifi
import httpx
from hishel import CacheTransport, Controller, FileStorage

from .settings import CACHE_DIR, DownloadConfiguration

LOGGER = logging.getLogger("DocsToKG.OntologyDownload.net")

# --- Constants & globals -------------------------------------------------------

HTTP_CACHE_DIR: Path = CACHE_DIR / "http" / "ontology"
_CLIENT_LOCK = threading.RLock()
_HTTP_CLIENT: Optional[httpx.Client] = None
_CLIENT_FACTORY: Optional[Callable[[], Optional[httpx.Client]]] = None
_DEFAULT_CONFIG = DownloadConfiguration()
_EXTRA_REQUEST_HOOKS: Dict[str, Iterable] = {}
_EXTRA_RESPONSE_HOOKS: Dict[str, Iterable] = {}

# --- Client construction helpers ----------------------------------------------


def _ensure_cache_dir() -> Path:
    HTTP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return HTTP_CACHE_DIR


def _build_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.load_verify_locations(certifi.where())
    return context


def _controller() -> Controller:
    return Controller(
        cacheable_methods=["GET", "HEAD"],
        cacheable_status_codes=[200, 203, 300, 301, 308, 404, 410, 416],
        cache_private=True,
        allow_heuristics=False,
        always_revalidate=True,
    )


def _request_hook(request: httpx.Request) -> None:
    extension_payload = request.extensions.pop("ontology_headers", None)
    config: Optional[DownloadConfiguration] = None
    extra_headers: Dict[str, str] = {}
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None

    if isinstance(extension_payload, Mapping):
        cfg = extension_payload.get("config")
        if isinstance(cfg, DownloadConfiguration):
            config = cfg
        headers = extension_payload.get("headers")
        if isinstance(headers, Mapping):
            extra_headers = {str(k): str(v) for k, v in headers.items()}
        corr = extension_payload.get("correlation_id")
        if isinstance(corr, str):
            correlation_id = corr
        req_id = extension_payload.get("request_id")
        if isinstance(req_id, str):
            request_id = req_id

    cfg = config or _DEFAULT_CONFIG
    polite = cfg.polite_http_headers(correlation_id=correlation_id, request_id=request_id)

    merged: Dict[str, str] = {}
    merged.update(polite)
    merged.update(extra_headers)
    for header, value in merged.items():
        request.headers.setdefault(header, value)

    meta: MutableMapping[str, object] = request.extensions.setdefault("ontology_meta", {})  # type: ignore[assignment]
    meta["correlation_id"] = correlation_id or polite.get("X-Request-ID")
    meta["start_time"] = time.perf_counter()
    meta["attempt"] = int(meta.get("attempt", 0)) + 1

    extra = _EXTRA_REQUEST_HOOKS.get("request")
    if extra:
        for hook in extra:
            try:
                hook(request)
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("request hook failed")


def _response_hook(response: httpx.Response) -> None:
    response.raise_for_status()

    meta: MutableMapping[str, object] = response.request.extensions.setdefault(  # type: ignore[assignment]
        "ontology_meta", {}
    )
    start = meta.get("start_time")
    if isinstance(start, (int, float)):
        meta["elapsed_sec"] = time.perf_counter() - start

    cache_hit = bool(response.extensions.get("from_cache"))
    cache_metadata = response.extensions.get("cache_metadata")
    cache_info: Dict[str, object] = {"from_cache": cache_hit}
    if isinstance(cache_metadata, Mapping):
        revalidated = cache_metadata.get("revalidated")
        if isinstance(revalidated, bool):
            cache_info["revalidated"] = revalidated
        cache_key = cache_metadata.get("cache_key")
        if isinstance(cache_key, str):
            cache_info["cache_key"] = cache_key
    response.extensions["ontology_cache_status"] = cache_info

    LOGGER.debug(
        "ontology-http-response",
        extra={
            "url": str(response.request.url),
            "status": response.status_code,
            "cache_hit": cache_hit,
        },
    )

    extra = _EXTRA_RESPONSE_HOOKS.get("response")
    if extra:
        for hook in extra:
            try:
                hook(response)
            except Exception:  # pragma: no cover
                LOGGER.exception("response hook failed")


def _timeout_for(config: DownloadConfiguration) -> httpx.Timeout:
    connect = getattr(config, "connect_timeout_sec", 5.0)
    pool = getattr(config, "pool_timeout_sec", connect)
    read = getattr(config, "timeout_sec", 30.0)
    write = getattr(config, "timeout_sec", 30.0)
    return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)


def _limits_for(config: DownloadConfiguration) -> httpx.Limits:
    max_conn = getattr(config, "max_httpx_connections", 100)
    keepalive_conn = getattr(config, "max_keepalive_connections", 20)
    keepalive_expiry = getattr(config, "keepalive_expiry_sec", 30.0)
    return httpx.Limits(
        max_connections=max_conn,
        max_keepalive_connections=keepalive_conn,
        keepalive_expiry=keepalive_expiry,
    )


def _build_http_client(cache_root: Path, config: Optional[DownloadConfiguration]) -> httpx.Client:
    cfg = config or _DEFAULT_CONFIG
    ssl_context = _build_ssl_context()
    transport = CacheTransport(
        transport=httpx.HTTPTransport(retries=0),
        storage=FileStorage(base_path=cache_root),
        controller=_controller(),
    )
    return httpx.Client(
        http2=bool(getattr(cfg, "http2_enabled", True)),
        transport=transport,
        timeout=_timeout_for(cfg),
        limits=_limits_for(cfg),
        verify=ssl_context,
        trust_env=True,
        follow_redirects=False,
        event_hooks={"request": [_request_hook], "response": [_response_hook]},
    )


def _close_client_unlocked() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        with contextlib.suppress(Exception):
            _HTTP_CLIENT.close()
    _HTTP_CLIENT = None


# --- Public API ----------------------------------------------------------------

def configure_http_client(
    client: Optional[httpx.Client] = None,
    *,
    factory: Optional[Callable[[], Optional[httpx.Client]]] = None,
    request_hooks: Optional[Mapping[str, Iterable]] = None,
    response_hooks: Optional[Mapping[str, Iterable]] = None,
    default_config: Optional[DownloadConfiguration] = None,
) -> None:
    """Override the shared HTTPX client or register a factory for tests."""

    if client is not None and factory is not None:
        raise ValueError("provide either a client or factory, not both")

    with _CLIENT_LOCK:
        global _HTTP_CLIENT, _CLIENT_FACTORY, _EXTRA_REQUEST_HOOKS, _EXTRA_RESPONSE_HOOKS, _DEFAULT_CONFIG

        if default_config is not None:
            _DEFAULT_CONFIG = default_config

        _EXTRA_REQUEST_HOOKS = dict(request_hooks or {})
        _EXTRA_RESPONSE_HOOKS = dict(response_hooks or {})

        if client is None:
            _close_client_unlocked()
        else:
            if _HTTP_CLIENT is not client:
                _close_client_unlocked()
            _HTTP_CLIENT = client

        _CLIENT_FACTORY = factory


def reset_http_client() -> None:
    """Reset the shared HTTPX client to its default configuration (test helper)."""

    with _CLIENT_LOCK:
        global _CLIENT_FACTORY, _EXTRA_REQUEST_HOOKS, _EXTRA_RESPONSE_HOOKS
        _CLIENT_FACTORY = None
        _EXTRA_REQUEST_HOOKS = {}
        _EXTRA_RESPONSE_HOOKS = {}
        _close_client_unlocked()


def get_http_client(config: Optional[DownloadConfiguration] = None) -> httpx.Client:
    """Return the shared HTTPX client, creating it if necessary."""

    with _CLIENT_LOCK:
        if _HTTP_CLIENT is not None:
            return _HTTP_CLIENT

        cfg = config or _DEFAULT_CONFIG

        factory = _CLIENT_FACTORY
        if factory is None and config is not None:
            factory = config.get_session_factory()

        if factory is not None:
            candidate = factory()
            if candidate is not None and not isinstance(candidate, httpx.Client):
                raise TypeError("session_factory must return an httpx.Client or None")
            if candidate is not None:
                factory_name = getattr(factory, "__qualname__", getattr(factory, "__name__", repr(factory)))
                LOGGER.info(
                    "using custom httpx client",
                    extra={
                        "factory": factory_name,
                        "factory_module": getattr(factory, "__module__", None),
                    },
                )
                _set_client_unlocked(candidate)
                return candidate

        cache_root = _ensure_cache_dir()
        client = _build_http_client(cache_root, cfg)
        _set_client_unlocked(client)
        return client


def _set_client_unlocked(client: httpx.Client) -> None:
    global _HTTP_CLIENT
    _HTTP_CLIENT = client
