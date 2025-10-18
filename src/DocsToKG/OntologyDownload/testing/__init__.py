"""Testing utilities for exercising the ontology downloader end-to-end."""

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
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.parse import urljoin

from ..plugins import (
    register_resolver,
    register_validator,
    unregister_resolver,
    unregister_validator,
)
from ..resolvers import BaseResolver, FetchPlan
from ..settings import (
    CACHE_DIR,
    CONFIG_DIR,
    LOG_DIR,
    LOCAL_ONTOLOGY_DIR,
    DefaultsConfig,
    DownloadConfiguration,
    ResolvedConfig,
    STORAGE,
    get_default_config,
    invalidate_default_config_cache,
)
from ..settings import StorageBackend as _StorageBackend

__all__ = [
    "ResponseSpec",
    "RequestRecord",
    "TestingEnvironment",
    "temporary_resolver",
    "temporary_validator",
]


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
        self._original_paths: Dict[str, Dict[str, object]] = {}
        self._storage: Optional[_StorageBackend] = None
        self._registered = False

    # --- Context manager protocol -------------------------------------------------

    def __enter__(self) -> "TestingEnvironment":
        self._install_runtime_paths()
        self._start_http_server()
        invalidate_default_config_cache()
        self._registered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_http_server()
        self._restore_runtime_paths()
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
        for module_name, attrs in self._PATH_TARGETS.items():
            module = import_module(module_name)
            store = self._original_paths.setdefault(module_name, {})
            for attr in attrs:
                if attr == "STORAGE":
                    original = getattr(module, attr, None)
                    store[attr] = original
                    # Recreate storage backend pointing at new ontology dir
                    storage = _StorageShim(self.ontology_dir)
                    setattr(module, attr, storage)
                    if module_name == "DocsToKG.OntologyDownload.settings":
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

    def _stop_http_server(self) -> None:
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)
            self._http_thread = None
        self._responses.clear()

    def http_url(self, path: str) -> str:
        """Return the absolute URL served by the harness for ``path``."""

        if not self._http_root:
            raise RuntimeError("TestingEnvironment must be entered before requesting URLs")
        normalized = path.lstrip("/")
        return urljoin(self._http_root, normalized)

    def queue_response(self, path: str, response: ResponseSpec) -> None:
        """Enqueue ``response`` for the specified ``path``."""

        key = (response.method.upper(), "/" + path.lstrip("/"))
        self._responses[key].append(response)

    def _dequeue_response(self, *, method: str, path: str) -> Optional[ResponseSpec]:
        key = (method.upper(), path)
        responses = self._responses.get(key)
        if not responses:
            return None
        return responses.popleft()

    # --- Fixture helpers ----------------------------------------------------------

    def register_fixture(
        self,
        name: str,
        data: Union[bytes, str, Path],
        *,
        media_type: str = "application/octet-stream",
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> str:
        """Register a fixture served over HTTP and return its URL."""

        path = Path(data)
        if isinstance(data, (bytes, str)):
            path = self.root / "fixtures" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = data.encode("utf-8") if isinstance(data, str) else data
            path.write_bytes(content)
        else:
            content = path.read_bytes()

        headers = {
            "Content-Type": media_type,
            "ETag": etag or f"W/\"{hash(content)}\"",
            "Last-Modified": last_modified or "Wed, 01 Jan 2025 00:00:00 GMT",
        }

        rel_path = f"fixtures/{name}"
        self.queue_response(
            rel_path,
            ResponseSpec(status=200, body=content, headers=headers, method="GET"),
        )
        self.queue_response(
            rel_path,
            ResponseSpec(status=200, body=b"", headers=headers, method="HEAD"),
        )
        return self.http_url(rel_path)

    # --- Configuration helpers ----------------------------------------------------

    def build_download_config(self) -> DownloadConfiguration:
        """Return a download configuration bound to this harness."""

        config = DownloadConfiguration()
        config.set_session_factory(None)
        config.set_bucket_provider(None)
        return config

    def build_resolved_config(self) -> ResolvedConfig:
        """Return a resolved config using defaults rooted at the harness directories."""

        defaults = DefaultsConfig()
        defaults.http = defaults.http.model_copy()
        config = ResolvedConfig(defaults=defaults, specs=[])
        return config

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

            def plan(self_inner, spec, config, logger):
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
