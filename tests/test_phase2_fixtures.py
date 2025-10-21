# === NAVMAP v1 ===
# {
#   "module": "tests.test_phase2_fixtures",
#   "purpose": "Unit tests for Phase 2 core fixtures",
#   "sections": [
#     {"id": "http-mock-tests", "name": "HTTP Mocking Tests", "anchor": "http-mock-tests", "kind": "section"},
#     {"id": "duckdb-tests", "name": "DuckDB Tests", "anchor": "duckdb-tests", "kind": "section"},
#     {"id": "telemetry-tests", "name": "Telemetry Tests", "anchor": "telemetry-tests", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Unit tests for Phase 2 core fixtures (HTTP mocking, DuckDB, telemetry).

Tests verify that fixtures work correctly in isolation and integration.
"""

from __future__ import annotations

import json

import pytest


# --- HTTP Mocking Tests ---


@pytest.mark.unit
def test_http_mock_builder_basic(http_mock):
    """Test basic mock response builder."""
    builder = http_mock(200, b"Hello World")
    response = builder.build()
    assert response.status_code == 200
    assert response.content == b"Hello World"


@pytest.mark.unit
def test_http_mock_builder_with_json(http_mock):
    """Test mock response builder with JSON."""
    builder = http_mock(200).with_json({"id": 1, "name": "test"})
    response = builder.build()
    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["id"] == 1
    assert data["name"] == "test"
    assert response.headers["content-type"] == "application/json"


@pytest.mark.unit
def test_http_mock_builder_with_status(http_mock):
    """Test setting custom status code."""
    builder = http_mock(200).with_status(404).with_content(b"Not found")
    response = builder.build()
    assert response.status_code == 404
    assert response.content == b"Not found"


@pytest.mark.unit
def test_mocked_http_client_basic(mocked_http_client):
    """Test mocked HTTP client basic registration and retrieval."""
    mc = mocked_http_client
    mc["register"]("GET", "https://api.example.com/users", 200, {"users": []})

    response = mc["client"].get("https://api.example.com/users")
    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["users"] == []


@pytest.mark.unit
def test_mocked_http_client_pattern_matching(mocked_http_client):
    """Test URL pattern matching in mock client."""
    mc = mocked_http_client
    mc["register"]("GET", "https://api.example.com", 200, {"version": "1.0"})

    response = mc["client"].get("https://api.example.com/v1/resource")
    assert response.status_code == 200


@pytest.mark.unit
def test_mocked_http_client_404_fallback(mocked_http_client):
    """Test 404 fallback for unregistered URLs."""
    mc = mocked_http_client
    response = mc["client"].get("https://unmocked.example.com/path")
    assert response.status_code == 404


@pytest.mark.unit
def test_mocked_http_client_reset(mocked_http_client):
    """Test reset functionality."""
    mc = mocked_http_client
    mc["register"]("GET", "https://api.example.com", 200, {})
    assert len(mc["responses"]) == 1

    mc["reset"]()
    assert len(mc["responses"]) == 0


# --- DuckDB Tests ---


@pytest.mark.unit
def test_ephemeral_duckdb_basic(ephemeral_duckdb):
    """Test ephemeral DuckDB connection."""
    db = ephemeral_duckdb
    db["cursor"].execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    db["cursor"].execute("INSERT INTO test VALUES (1, 'Alice')")

    result = db["query"]("SELECT * FROM test WHERE id = 1")
    assert result[0] == (1, "Alice")


@pytest.mark.unit
def test_ephemeral_duckdb_query_helper(ephemeral_duckdb):
    """Test query helper function."""
    db = ephemeral_duckdb
    db["cursor"].execute("CREATE TABLE test (value INTEGER)")
    db["cursor"].execute("INSERT INTO test VALUES (42)")

    result = db["query"]("SELECT * FROM test")
    assert len(result) == 1
    assert result[0][0] == 42


@pytest.mark.unit
def test_duckdb_with_test_data_schema(duckdb_with_test_data):
    """Test DuckDB with test data has proper schema."""
    db = duckdb_with_test_data
    tables = db["query"](
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
    )
    table_names = [t[0] for t in tables]
    assert "ontologies" in table_names
    assert "versions" in table_names
    assert "artifacts" in table_names


@pytest.mark.unit
def test_duckdb_with_test_data_content(duckdb_with_test_data):
    """Test DuckDB with test data contains expected data."""
    db = duckdb_with_test_data
    ontologies = db["query"]("SELECT COUNT(*) FROM ontologies")
    assert ontologies[0][0] == 3

    versions = db["query"]("SELECT COUNT(*) FROM versions")
    assert versions[0][0] == 3

    artifacts = db["query"]("SELECT COUNT(*) FROM artifacts")
    assert artifacts[0][0] == 4


@pytest.mark.unit
def test_duckdb_migrations_available(duckdb_migrations):
    """Test migration helper provides all versions."""
    mig = duckdb_migrations
    assert "v1" in mig["schemas"]
    assert "v2" in mig["schemas"]
    assert "v3" in mig["schemas"]
    assert mig["current_version"] == "v3"


@pytest.mark.unit
def test_duckdb_migrations_apply(duckdb_migrations, ephemeral_duckdb):
    """Test applying migrations."""
    db = ephemeral_duckdb
    mig = duckdb_migrations
    mig["apply"](db["cursor"], "v1")

    tables = db["query"](
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
    )
    table_names = [t[0] for t in tables]
    assert "ontologies" in table_names
    assert "versions" in table_names


# --- Telemetry Tests ---


@pytest.mark.unit
def test_event_sink_emit(event_sink):
    """Test event emission."""
    sink = event_sink
    sink["emit"]({"type": "test", "value": 42})
    assert sink["assertions"].count() == 1


@pytest.mark.unit
def test_event_sink_by_type(event_sink):
    """Test filtering events by type."""
    sink = event_sink
    sink["emit"]({"type": "request", "status": 200})
    sink["emit"]({"type": "response", "size": 1024})
    sink["emit"]({"type": "request", "status": 404})

    requests = sink["assertions"].by_type("request")
    assert len(requests) == 2
    assert requests[0]["status"] == 200


@pytest.mark.unit
def test_event_sink_has_event(event_sink):
    """Test checking for event type."""
    sink = event_sink
    sink["emit"]({"type": "request"})
    assert sink["assertions"].has_event("request")
    assert not sink["assertions"].has_event("response")


@pytest.mark.unit
def test_event_sink_has_error(event_sink):
    """Test error detection."""
    sink = event_sink
    sink["emit"]({"level": "info", "msg": "normal"})
    assert not sink["assertions"].has_error()

    sink["emit"]({"level": "error", "msg": "oops"})
    assert sink["assertions"].has_error()


@pytest.mark.unit
def test_event_sink_filter(event_sink):
    """Test arbitrary field filtering."""
    sink = event_sink
    sink["emit"]({"type": "request", "method": "GET", "url": "/"})
    sink["emit"]({"type": "request", "method": "POST", "url": "/"})
    sink["emit"]({"type": "response", "method": "GET", "status": 200})

    get_events = sink["assertions"].filter(method="GET")
    assert len(get_events) == 2


@pytest.mark.unit
def test_event_sink_assertions_count(event_sink):
    """Test count assertions."""
    sink = event_sink
    sink["emit"]({"type": "a"})
    sink["emit"]({"type": "b"})
    sink["emit"]({"type": "c"})

    sink["assertions"].assert_count(3)
    sink["assertions"].assert_min_count(2)
    sink["assertions"].assert_max_count(5)


@pytest.mark.unit
def test_event_sink_reset(event_sink):
    """Test event sink reset."""
    sink = event_sink
    sink["emit"]({"type": "test"})
    assert sink["assertions"].count() == 1

    sink["reset"]()
    assert sink["assertions"].count() == 0


@pytest.mark.unit
def test_mock_event_emitter_basic(mock_event_emitter):
    """Test mock event emitter."""
    emitter = mock_event_emitter
    emitter["emit_event"]("net.request", {"url": "https://example.com"})
    emitter["emit_event"]("net.response", {"status": 200})

    assert emitter["call_count"]() == 2


@pytest.mark.unit
def test_mock_event_emitter_by_type(mock_event_emitter):
    """Test filtering emitted events by type."""
    emitter = mock_event_emitter
    emitter["emit_event"]("http.get", {"url": "/"})
    emitter["emit_event"]("http.post", {"url": "/"})
    emitter["emit_event"]("http.get", {"url": "/api"})

    gets = emitter["by_type"]("http.get")
    assert len(gets) == 2
    assert gets[0]["url"] == "/"


@pytest.mark.unit
def test_mock_event_emitter_reset(mock_event_emitter):
    """Test mock event emitter reset."""
    emitter = mock_event_emitter
    emitter["emit_event"]("test", {"data": 1})
    assert emitter["call_count"]() == 1

    emitter["reset"]()
    assert emitter["call_count"]() == 0


# --- Integration Tests ---


@pytest.mark.component
def test_http_and_event_sink_integration(mocked_http_client, event_sink):
    """Test HTTP mock and event sink together."""
    mc = mocked_http_client
    sink = event_sink

    mc["register"]("GET", "https://api.example.com", 200, {"success": True})
    response = mc["client"].get("https://api.example.com")

    sink["emit"]({"type": "http", "method": "GET", "status": response.status_code})
    assert sink["assertions"].count() == 1
    assert sink["assertions"].by_type("http")[0]["status"] == 200


@pytest.mark.component
def test_duckdb_and_event_sink_integration(duckdb_with_test_data, event_sink):
    """Test DuckDB and event sink together."""
    db = duckdb_with_test_data
    sink = event_sink

    ontologies = db["query"]("SELECT COUNT(*) FROM ontologies")
    count = ontologies[0][0]

    sink["emit"]({"type": "db", "table": "ontologies", "count": count})
    assert sink["assertions"].by_type("db")[0]["count"] == 3
