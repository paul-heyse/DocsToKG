"""Testing utilities for exercising the ontology downloader end-to-end.

Provides a harness that fakes HTTP servers and supplies deterministic test
limiters, plus helpers for temporarily registering resolvers/validators during
unit tests.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import logging
import os
import socket
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.parse import urljoin

import httpx

from ..io import rate_limit as rate_mod
from ..io import sanitize_filename
from ..plugins import (
    register_resolver,
    register_validator,
    unregister_resolver,
    unregister_validator,
)
from ..resolvers import BaseResolver, FetchPlan
from ..settings import (
    DefaultsConfig,
    DownloadConfiguration,
    ResolvedConfig,
    invalidate_default_config_cache,
)
from ..settings import StorageBackend as _StorageBackend

__all__ = [
    "ResponseSpec",
    "RequestRecord",
    "TestingEnvironment",
    "temporary_resolver",
    "temporary_validator",
    "use_mock_http_client",
]


@contextlib.contextmanager
def use_mock_http_client(transport: "httpx.BaseTransport", **client_kwargs):
    """Temporarily install an HTTPX client backed by ``transport``."""

    import httpx

    from ..net import configure_http_client, reset_http_client

    default_config = client_kwargs.pop("default_config", None)
    client = httpx.Client(transport=transport, **client_kwargs)
    configure_http_client(client=client, default_config=default_config)
    try:
        yield client
    finally:
        reset_http_client()
        client.close()


@dataclass
class ResponseSpec:
    """HTTP response definition served by the loopback test server."""

    status: int = 200
    body: Union[bytes, str] = b""
    headers: Mapping[str, str] = field(default_factory=dict)
    method: str = "GET"
    stream: Optional[Iterable[Union[bytes, str]]] = None
    delay_sec: Optional[float] = None

    def serialise_body(self) -> bytes:
        if isinstance(self.body, bytes):
            return self.body
        if isinstance(self.body, str):
            return self.body.encode("utf-8")
        return json.dumps(self.body).encode("utf-8")


@dataclass
class RequestRecord:
    """Captured HTTP request emitted by the downloader during tests."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes


class _ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, *, env) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.env = env


class _RequestHandler(http.server.BaseHTTPRequestHandler):
    server_version = "OntologyDownloadTestServer/1.0"

    def log_message(self, format, *args):  # noqa: D401  (silence default logging)
        return

    def _handle(self) -> None:
        env: TestingEnvironment = self.server.env  # type: ignore[attr-defined]
        path = self.path.split("?", 1)[0]
        record = RequestRecord(
            method=self.command,
            path=path,
            headers={key: value for key, value in self.headers.items()},
            body=self.rfile.read(int(self.headers.get("Content-Length", "0") or "0")),
        )
        env._request_log.append(record)
        response = env._dequeue_response(method=self.command, path=path)
        if response is None:
            self.send_error(404, "No response queued for path")
            return
        if response.delay_sec:
            time.sleep(response.delay_sec)
        body = response.serialise_body()
        headers = dict(response.headers)
        if response.stream is None and "Content-Length" not in headers:
            headers["Content-Length"] = str(len(body))
        self.send_response(response.status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            if response.stream is not None:
                for chunk in response.stream:
                    self.wfile.write(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
                    self.wfile.flush()
            else:
                self.wfile.write(body)

    def do_HEAD(self):  # noqa: D401
        self._handle()

    def do_GET(self):  # noqa: D401
        self._handle()

    def do_POST(self):  # pragma: no cover - seldom used
        self._handle()


def _find_free_port(host: str = "127.0.0.1") -> Tuple[str, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    addr, port = sock.getsockname()
    sock.close()
    return addr, port


class TestingEnvironment(contextlib.AbstractContextManager["TestingEnvironment"]):
    """Context manager provisioning an isolated runtime for ontology download tests."""

    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger("OntologyDownload.testing")
        self._tmp = TemporaryDirectory(prefix="ontofetch-test-")
        self.root = Path(self._tmp.name)
        self.cache_dir = self.root / "cache"
        self.config_dir = self.root / "configs"
        self.log_dir = self.root / "logs"
        self.ontology_dir = self.root / "ontologies"
        for directory in (
            self.cache_dir,
            self.config_dir,
            self.log_dir,
            self.ontology_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._request_log: List[RequestRecord] = []
        self._responses: Dict[Tuple[str, str], Deque[ResponseSpec]] = defaultdict(deque)
        self._http_thread: Optional[threading.Thread] = None
        self._http_server: Optional[_ThreadedHTTPServer] = None
        self._http_root: Optional[str] = None
        self._http_host: Optional[str] = None
        self._http_port: Optional[int] = None
        self._original_paths: Dict[str, Dict[str, object]] = {}
        self._storage: Optional[_StorageBackend] = None
        self._bucket_state: Dict[
            Tuple[Optional[str], Optional[str]], rate_mod.RateLimiterHandle
        ] = {}
        self._env_overrides: Dict[str, Optional[str]] = {}
        self._registered = False

    # --- Context manager protocol -------------------------------------------------

    def __enter__(self) -> "TestingEnvironment":
        self._install_runtime_paths()
        self._start_http_server()
        self._reset_network_primitives()
        self._apply_environment_overrides()
        invalidate_default_config_cache()
        self._registered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_http_server()
        self._restore_runtime_paths()
        self._reset_network_primitives()
        self._restore_environment_overrides()
        invalidate_default_config_cache()
        self._registered = False
        self._tmp.cleanup()

    # --- Runtime path management --------------------------------------------------

    _PATH_TARGETS: Mapping[str, Sequence[str]] = {
        "DocsToKG.OntologyDownload.settings": (
            "DATA_ROOT",
            "CACHE_DIR",
            "CONFIG_DIR",
            "LOG_DIR",
            "LOCAL_ONTOLOGY_DIR",
            "STORAGE",
        ),
        "DocsToKG.OntologyDownload.planning": ("CACHE_DIR", "LOCAL_ONTOLOGY_DIR", "STORAGE"),
        "DocsToKG.OntologyDownload.api": (
            "CACHE_DIR",
            "LOCAL_ONTOLOGY_DIR",
            "LOG_DIR",
            "ONTOLOGY_DIR",
            "STORAGE",
        ),
        "DocsToKG.OntologyDownload.cli": ("CACHE_DIR", "LOCAL_ONTOLOGY_DIR", "LOG_DIR", "STORAGE"),
        "DocsToKG.OntologyDownload.manifests": ("CACHE_DIR", "LOCAL_ONTOLOGY_DIR", "STORAGE"),
        "DocsToKG.OntologyDownload.logging_utils": ("LOG_DIR",),
    }

    def _install_runtime_paths(self) -> None:
        mapping = {
            "DATA_ROOT": self.root,
            "CACHE_DIR": self.cache_dir,
            "CONFIG_DIR": self.config_dir,
            "LOG_DIR": self.log_dir,
            "LOCAL_ONTOLOGY_DIR": self.ontology_dir,
        }
        storage_shim = _StorageShim(self.ontology_dir)
        self._storage = storage_shim

        for module_name, attrs in self._PATH_TARGETS.items():
            module = import_module(module_name)
            store = self._original_paths.setdefault(module_name, {})
            for attr in attrs:
                if attr == "STORAGE":
                    original = getattr(module, attr, None)
                    store[attr] = original
                    setattr(module, attr, storage_shim)
                    continue
                value = mapping.get(attr)
                if value is None and hasattr(module, attr):
                    value = getattr(module, attr)
                if value is None:
                    continue
                store[attr] = getattr(module, attr, None)
                setattr(module, attr, value)

    def _restore_runtime_paths(self) -> None:
        for module_name, attrs in self._original_paths.items():
            module = import_module(module_name)
            for attr, value in attrs.items():
                setattr(module, attr, value)
        self._original_paths.clear()
        self._storage = None
        self._bucket_state.clear()

    def _reset_network_primitives(self) -> None:
        """Reset network caches, rate limits, and DNS stubs between tests."""

        # ``clear_dns_stubs`` also evicts any stubbed cache entries ensuring subsequent
        # lookups re-evaluate against the patched or real resolver.
        from ..io import network as network_mod  # Local import to avoid cycles
        from ..io import rate_limit as rate_mod
        from ..net import reset_http_client

        reset_http_client()
        rate_mod.reset()
        network_mod.clear_dns_stubs()
        self._bucket_state.clear()

    def _apply_environment_overrides(self) -> None:
        env_root = self.root / "env"
        env_root.mkdir(parents=True, exist_ok=True)
        pystow_home = env_root / "pystow"
        pystow_home.mkdir(parents=True, exist_ok=True)

        desired = {
            "PYSTOW_HOME": str(pystow_home),
            "BIOPORTAL_API_KEY": "test-bioportal-key",
        }

        for key, value in desired.items():
            if key not in self._env_overrides:
                self._env_overrides[key] = os.environ.get(key)
            os.environ[key] = value

    def _restore_environment_overrides(self) -> None:
        for key, previous in self._env_overrides.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self._env_overrides.clear()

    # --- HTTP server --------------------------------------------------------------

    def _start_http_server(self) -> None:
        host, port = _find_free_port()
        server = _ThreadedHTTPServer((host, port), _RequestHandler, env=self)
        thread = threading.Thread(target=server.serve_forever, name="OntologyDownloadTestServer")
        thread.daemon = True
        thread.start()
        self._http_server = server
        self._http_thread = thread
        self._http_root = f"http://{host}:{port}/"
        self._http_host = host
        self._http_port = port

    def _stop_http_server(self) -> None:
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)
            self._http_thread = None
        self._responses.clear()
        self._http_host = None
        self._http_port = None

    def http_url(self, path: str) -> str:
        """Return the absolute URL served by the harness for ``path``."""

        if not self._http_root:
            raise RuntimeError("TestingEnvironment must be entered before requesting URLs")
        normalized = path.lstrip("/")
        return urljoin(self._http_root, normalized)

    def queue_response(self, path: str, response: ResponseSpec) -> None:
        """Enqueue ``response`` for the specified ``path``."""

        key = (response.method.upper(), "/" + path.lstrip("/"))
        responses = self._responses[key]
        responses.clear()
        responses.append(response)

    def _dequeue_response(self, *, method: str, path: str) -> Optional[ResponseSpec]:
        key = (method.upper(), path)
        responses = self._responses.get(key)
        if not responses:
            return None
        return responses.popleft()

    def build_httpx_transport(self) -> "httpx.MockTransport":
        """Return an HTTPX transport that serves responses from this environment."""

        import httpx

        env = self

        def _should_return_not_modified(
            request: httpx.Request, headers: Mapping[str, str]
        ) -> bool:
            response_etag = headers.get("ETag")
            if_none_match = request.headers.get("if-none-match")
            if if_none_match and response_etag:
                candidates = [token.strip() for token in if_none_match.split(",") if token.strip()]
                normalized = response_etag.strip()
                weak = normalized[2:] if normalized.startswith("W/") else normalized
                for token in candidates:
                    if token == "*":
                        return True
                    unquoted = token.strip('"')
                    if unquoted == normalized.strip('"') or unquoted == weak.strip('"'):
                        return True

            if_modified_since = request.headers.get("if-modified-since")
            last_modified = headers.get("Last-Modified")
            if if_modified_since and last_modified:
                try:
                    request_time = parsedate_to_datetime(if_modified_since)
                    response_time = parsedate_to_datetime(last_modified)
                except (TypeError, ValueError, IndexError):
                    return if_modified_since.strip() == last_modified.strip()
                if request_time and response_time and response_time <= request_time:
                    return True
            return False

        def _handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path or "/"
            normalized_headers = {
                "-".join(part.capitalize() for part in key.split("-")): value
                for key, value in request.headers.items()
            }
            record = RequestRecord(
                method=request.method,
                path=path,
                headers=normalized_headers,
                body=request.content or b"",
            )
            env._request_log.append(record)

            spec = env._dequeue_response(method=request.method.upper(), path=path)
            if spec is None:
                return httpx.Response(404, request=request, content=b"")

            if spec.delay_sec:
                time.sleep(spec.delay_sec)

            headers = dict(spec.headers)
            cache_key = (request.method.upper(), path)
            if _should_return_not_modified(request, headers):
                env._responses[cache_key].appendleft(spec)
                limited_headers = {
                    key: value
                    for key, value in headers.items()
                    if key in {"ETag", "Last-Modified", "Cache-Control"}
                }
                return httpx.Response(
                    304,
                    headers=limited_headers,
                    request=request,
                )
            if spec.stream:

                def iterator() -> Iterable[bytes]:
                    for chunk in spec.stream or []:
                        yield chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")

                return httpx.Response(
                    spec.status,
                    headers=headers,
                    stream=iterator(),
                    request=request,
                )

            body = spec.serialise_body()
            return httpx.Response(
                spec.status,
                headers=headers,
                content=body,
                request=request,
            )

        return httpx.MockTransport(_handler)

    # --- Fixture helpers ----------------------------------------------------------

    def register_fixture(
        self,
        name: str,
        data: Union[bytes, str, Path],
        *,
        media_type: str = "application/octet-stream",
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
        repeats: int = 1,
    ) -> str:
        """Register a fixture served over HTTP and return its URL."""

        if isinstance(data, (bytes, str)):
            path = self.root / "fixtures" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = data.encode("utf-8") if isinstance(data, str) else data
            path.write_bytes(content)
        else:
            path = Path(data)
            content = path.read_bytes()

        headers = {
            "Content-Type": media_type,
            "ETag": etag or f'W/"{hash(content)}"',
            "Last-Modified": last_modified or "Wed, 01 Jan 2025 00:00:00 GMT",
        }

        rel_path = f"fixtures/{name}"
        head_spec = ResponseSpec(status=200, body=b"", headers=headers, method="HEAD")
        get_spec = ResponseSpec(status=200, body=content, headers=headers, method="GET")
        key_head = ("HEAD", "/" + rel_path)
        key_get = ("GET", "/" + rel_path)
        for _ in range(max(1, repeats)):
            self._responses[key_head].append(head_spec)
            self._responses[key_get].append(get_spec)
        return self.http_url(rel_path)

    # --- Configuration helpers ----------------------------------------------------

    def build_download_config(self) -> DownloadConfiguration:
        """Return a download configuration bound to this harness.

        The returned configuration uses the harness-managed bucket provider and
        uses a harness-controlled stub limiter so tests can exercise
        deterministic behaviour.
        """

        config = DownloadConfiguration()
        config.set_bucket_provider(self._bucket_provider)
        config.rate_limiter = "pyrate"
        if self._http_host:
            allowed = [self._http_host]
            if self._http_port is not None:
                allowed.append(f"{self._http_host}:{self._http_port}")
            config.allowed_hosts = allowed
            config.allowed_ports = list(
                {80, 443, self._http_port} if self._http_port else {80, 443}
            )
        return config

    def build_resolved_config(self) -> ResolvedConfig:
        """Return a resolved config using defaults rooted at the harness directories."""

        defaults = DefaultsConfig()
        defaults.http = self.build_download_config()
        config = ResolvedConfig(defaults=defaults, specs=[])
        return config

    def seed_manifest(
        self,
        *,
        ontology_id: str,
        version: str,
        manifest: Optional[Mapping[str, object]] = None,
    ) -> Path:
        """Create a manifest on disk for ``ontology_id``/``version`` and return its path."""

        safe_id = sanitize_filename(ontology_id)
        safe_version = sanitize_filename(version)
        manifest_dir = self.ontology_dir / safe_id / safe_version
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        payload = dict(manifest or {})
        if "id" not in payload:
            payload["id"] = ontology_id
        if "version" not in payload:
            payload["version"] = version
        if "schema_version" not in payload:
            payload["schema_version"] = "1.0"
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return manifest_path

    # --- Resolver helpers ---------------------------------------------------------

    def static_resolver(
        self,
        *,
        name: str,
        fixture_url: str,
        filename: str,
        media_type: str = "application/rdf+xml",
        service: str = "test",
    ) -> BaseResolver:
        """Return a resolver that always resolves to ``fixture_url``."""

        class _StaticResolver(BaseResolver):
            NAME = name

            def plan(self_inner, spec, config, logger, *, cancellation_token=None):
                headers = {"Accept": media_type}
                return FetchPlan(
                    url=fixture_url,
                    headers=headers,
                    filename_hint=filename,
                    version="test-version",
                    license="CC0-1.0",
                    media_type=media_type,
                    service=service,
                )

        return _StaticResolver()

    # --- Request inspection -------------------------------------------------------

    @property
    def requests(self) -> Sequence[RequestRecord]:
        """Return the list of captured requests."""

        return list(self._request_log)

    # --- Custom session/bucket providers -----------------------------------------

    def _bucket_provider(
        self,
        service: Optional[str],
        config: DownloadConfiguration,
        host: Optional[str],
    ) -> rate_mod.RateLimiterHandle:
        key = (service, host)
        bucket = self._bucket_state.get(key)
        if bucket is None:
            bucket = _TestLimiter()
            self._bucket_state[key] = bucket
        return bucket


class _TestLimiter:
    """Simple limiter used in testing harness to avoid touching global manager."""

    def consume(self, tokens: float = 1.0) -> None:  # noqa: D401 - trivial stub
        return


class _StorageShim(_StorageBackend):
    """Minimal storage backend used during testing."""

    def __init__(self, root: Path) -> None:
        from ..settings import LocalStorageBackend

        self._delegate = LocalStorageBackend(root)

    def __getattr__(self, item):
        return getattr(self._delegate, item)

    def __dir__(self):
        return dir(self._delegate)


@contextlib.contextmanager
def temporary_resolver(name: str, resolver: BaseResolver):
    """Context manager registering ``resolver`` and restoring the previous entry."""

    previous: Optional[BaseResolver] = None
    existed = False
    try:
        previous = unregister_resolver(name)
        existed = True
    except KeyError:
        pass
    register_resolver(name, resolver, overwrite=True)
    try:
        yield resolver
    finally:
        unregister_resolver(name)
        if existed and previous is not None:
            register_resolver(name, previous, overwrite=True)


@contextlib.contextmanager
def temporary_validator(name: str, validator: Callable[..., object]):
    """Context manager registering ``validator`` temporarily."""

    previous: Optional[Callable[..., object]] = None
    existed = False
    try:
        previous = unregister_validator(name)
        existed = True
    except KeyError:
        pass
    register_validator(name, validator, overwrite=True)
    try:
        yield validator
    finally:
        unregister_validator(name)
        if existed and previous is not None:
            register_validator(name, previous, overwrite=True)
# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.testing",
#   "purpose": "Test utilities providing isolated environments, mock clients, and resolver stubs",
#   "sections": [
#     {"id": "fixtures", "name": "TestingEnvironment", "anchor": "ENV", "kind": "api"},
#     {"id": "mock-http", "name": "HTTP Client Helpers", "anchor": "HTTP", "kind": "helpers"},
#     {"id": "responses", "name": "Loopback Server & Response Specs", "anchor": "SRV", "kind": "helpers"},
#     {"id": "plugins", "name": "Temporary Plugin Registration", "anchor": "PLG", "kind": "helpers"}
#   ]
# }
# === /NAVMAP ===
