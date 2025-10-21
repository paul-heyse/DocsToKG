# === NAVMAP v1 ===
# {
#   "module": "tests.fixtures.http_mocking",
#   "purpose": "HTTP mocking fixtures for hermetic network testing",
#   "sections": [
#     {"id": "imports", "name": "Imports & Types", "anchor": "imports", "kind": "section"},
#     {"id": "mock-response-builder", "name": "MockResponseBuilder", "anchor": "class-mock-response-builder", "kind": "class"},
#     {"id": "http-mock-fixture", "name": "http_mock", "anchor": "fixture-http-mock", "kind": "fixture"},
#     {"id": "http-client-fixture", "name": "mocked_http_client", "anchor": "fixture-mocked-http-client", "kind": "fixture"}
#   ]
# }
# === /NAVMAP ===

"""
HTTP mocking fixtures for hermetic network testing.

Provides HTTPX MockTransport and mock response builders to test HTTP client behavior
without real network access. All responses are deterministic and reproducible.
"""

from __future__ import annotations

from typing import Any, Callable, Generator

import httpx
import pytest


class MockResponseBuilder:
    """Builder for constructing mock HTTP responses with fluent API."""

    def __init__(self, status_code: int = 200, content: bytes = b""):
        """Initialize response builder with defaults."""
        self.status_code = status_code
        self.content = content
        self.headers: dict[str, str] = {}
        self.cookies: dict[str, str] = {}

    def with_status(self, code: int) -> MockResponseBuilder:
        """Set response status code."""
        self.status_code = code
        return self

    def with_content(self, content: bytes | str) -> MockResponseBuilder:
        """Set response content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.content = content
        return self

    def with_json(self, data: dict[str, Any]) -> MockResponseBuilder:
        """Set response content as JSON."""
        import json

        self.content = json.dumps(data).encode("utf-8")
        self.headers["content-type"] = "application/json"
        return self

    def with_header(self, name: str, value: str) -> MockResponseBuilder:
        """Add response header."""
        self.headers[name] = value
        return self

    def with_headers(self, headers: dict[str, str]) -> MockResponseBuilder:
        """Set multiple response headers."""
        self.headers.update(headers)
        return self

    def with_cookie(self, name: str, value: str) -> MockResponseBuilder:
        """Add response cookie."""
        self.cookies[name] = value
        return self

    def build(self) -> httpx.Response:
        """Build the final response object."""
        return httpx.Response(
            status_code=self.status_code,
            content=self.content,
            headers=self.headers,
        )


@pytest.fixture
def http_mock() -> Generator[Callable[[int, bytes], httpx.Response], None, None]:
    """
    Provide a mock HTTP response builder factory.

    Returns a callable that creates responses with fluent API:
        response = http_mock(200, b"content")
        json_response = http_mock(200).with_json({"key": "value"}).build()

    Yields:
        Callable: MockResponseBuilder factory

    Example:
        def test_http_client(http_mock):
            builder = http_mock(200, b"Hello")
            response = builder.build()
            assert response.status_code == 200
            assert response.content == b"Hello"

            json_resp = http_mock(200).with_json({"id": 1}).build()
            assert json_resp.headers["content-type"] == "application/json"
    """

    def _mock_response(
        status_code: int = 200,
        content: bytes | str = b"",
    ) -> MockResponseBuilder:
        """Create a mock response builder."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return MockResponseBuilder(status_code=status_code, content=content)

    yield _mock_response


@pytest.fixture
def mocked_http_client() -> Generator[dict[str, Any], None, None]:
    """
    Provide a mocked HTTPX client with interceptor support.

    Yields a dict with:
    - client: httpx.Client with MockTransport
    - responses: dict to map (method, url_pattern) -> response
    - register: function to register mock responses
    - reset: function to clear registered responses

    Example:
        def test_with_mock_client(mocked_http_client):
            mc = mocked_http_client
            mc['register']('GET', 'https://api.example.com/users', 200, {'id': 1})
            response = mc['client'].get('https://api.example.com/users')
            assert response.status_code == 200
    """
    responses: dict[tuple[str, str], httpx.Response] = {}

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        """Handle mocked HTTP requests."""
        # Try exact match first
        key = (request.method, str(request.url))
        if key in responses:
            return responses[key]

        # Try pattern match (treat as prefix)
        for (method, pattern), response in responses.items():
            if method == request.method and str(request.url).startswith(pattern):
                return response

        # Default 404
        return httpx.Response(
            status_code=404,
            content=b'{"error": "Not mocked"}',
            headers={"content-type": "application/json"},
        )

    transport = httpx.MockTransport(mock_transport_handler)
    client = httpx.Client(transport=transport)

    def register(
        method: str,
        url_or_pattern: str,
        status_code: int = 200,
        content: bytes | str | dict[str, Any] = b"",
    ) -> None:
        """Register a mock response."""
        if isinstance(content, dict):
            import json

            content = json.dumps(content).encode("utf-8")
        elif isinstance(content, str):
            content = content.encode("utf-8")

        response = httpx.Response(
            status_code=status_code,
            content=content,
        )
        responses[(method, url_or_pattern)] = response

    def reset() -> None:
        """Clear all registered responses."""
        responses.clear()

    yield {
        "client": client,
        "responses": responses,
        "register": register,
        "reset": reset,
    }

    # Cleanup
    client.close()
