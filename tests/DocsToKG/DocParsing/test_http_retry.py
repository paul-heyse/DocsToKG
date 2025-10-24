import importlib.util
import pathlib
import sys

import httpx
import pytest


@pytest.fixture
def http_module():
    package_root = pathlib.Path(__file__).resolve().parents[3] / "src"
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

    module_path = package_root / "DocsToKG" / "DocParsing" / "core" / "http.py"
    module_name = "docparsing_http_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load http module for testing")
    http_module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    sys.modules[module_name] = http_module
    loader.exec_module(http_module)
    try:
        yield http_module
    finally:
        # Ensure globals are reset between tests.
        if hasattr(http_module, "_HTTP_SESSION_LOCK"):
            with http_module._HTTP_SESSION_LOCK:  # type: ignore[attr-defined]
                http_module._HTTP_SESSION = None  # type: ignore[attr-defined]
                http_module._HTTP_SESSION_TIMEOUT = http_module.DEFAULT_HTTP_TIMEOUT  # type: ignore[attr-defined]
                http_module._HTTP_SESSION_POLICY = http_module._DEFAULT_RETRY_POLICY  # type: ignore[attr-defined]
        sys.modules.pop(module_name, None)


def test_get_http_session_header_clone_isolated(http_module):
    shared_session, _ = http_module.get_http_session()
    override_session, _ = http_module.get_http_session(base_headers={"X-Test": "value"})

    assert override_session is not shared_session
    assert override_session.headers["X-Test"] == "value"
    assert "X-Test" not in shared_session.headers

    override_session.headers["X-Extra"] = "2"
    assert "X-Extra" not in shared_session.headers


def test_get_http_session_retry_override_does_not_mutate_global(http_module):
    shared_session, _ = http_module.get_http_session()
    assert shared_session.retry_policy.retry_total == http_module.DEFAULT_RETRY_TOTAL

    override_session, _ = http_module.get_http_session(
        retry_total=1,
        status_forcelist=(599,),
    )

    assert override_session is not shared_session
    assert override_session.retry_policy.retry_total == 1
    assert override_session.retry_policy.status_forcelist == (599,)

    shared_again, _ = http_module.get_http_session()
    assert shared_again is shared_session
    assert shared_again.retry_policy.retry_total == http_module.DEFAULT_RETRY_TOTAL
    assert shared_again.retry_policy.status_forcelist == http_module.DEFAULT_STATUS_FORCELIST


def test_request_with_retries_respects_override_policy(http_module, monkeypatch):
    shared_session, _ = http_module.get_http_session()

    captured: dict[str, object] = {}

    def fake_request(self, method, url, **kwargs):
        captured["client"] = self
        captured["retry_total"] = self.retry_policy.retry_total
        captured["status_forcelist"] = self.retry_policy.status_forcelist
        captured["headers"] = dict(self.headers)
        return httpx.Response(200, request=httpx.Request(method, url))

    monkeypatch.setattr(http_module.TenacityClient, "request", fake_request)

    response = http_module.request_with_retries(
        None,
        "GET",
        "https://example.com",
        retry_total=1,
        retry_backoff=0.1,
        status_forcelist=(599,),
        base_headers={"X-Test": "abc"},
    )

    assert response.status_code == 200
    assert captured["retry_total"] == 1
    assert captured["status_forcelist"] == (599,)
    headers = {str(key).lower(): value for key, value in captured["headers"].items()}
    assert headers["x-test"] == "abc"
    assert captured["client"] is not shared_session

    shared_again, _ = http_module.get_http_session()
    assert shared_again is shared_session
    assert "X-Test" not in shared_session.headers
    assert shared_again.retry_policy.retry_total == http_module.DEFAULT_RETRY_TOTAL


def test_request_with_retries_clones_provided_session(http_module, monkeypatch):
    shared_session, _ = http_module.get_http_session()

    captured_clients: list[object] = []

    def fake_request(self, method, url, **kwargs):
        captured_clients.append(self)
        return httpx.Response(204, request=httpx.Request(method, url))

    monkeypatch.setattr(http_module.TenacityClient, "request", fake_request)

    response = http_module.request_with_retries(
        shared_session,
        "POST",
        "https://example.com/post",
        retry_total=2,
        base_headers={"X-Token": "1"},
    )

    assert response.status_code == 204
    assert captured_clients[0] is not shared_session
    assert shared_session.retry_policy.retry_total == http_module.DEFAULT_RETRY_TOTAL
    assert "X-Token" not in shared_session.headers
