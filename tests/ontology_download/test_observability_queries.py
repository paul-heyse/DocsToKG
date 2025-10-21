"""Tests for observability stock SQL queries.

Covers:
- Query registry and access
- Query descriptions
- Query completeness
- SQL syntax validation (basic)
"""

import pytest

from DocsToKG.OntologyDownload.observability.queries import (
    QUERY_429_RESPONSES,
    QUERY_BYTES_WRITTEN_BY_SERVICE,
    QUERY_CACHE_HIT_RATIO,
    QUERY_ERROR_RATE_BY_SERVICE,
    QUERY_EXTRACTION_FAILURES,
    QUERY_EXTRACTION_STATS,
    QUERY_NET_REQUEST_P95_LATENCY,
    QUERY_POLICY_REJECTIONS,
    QUERY_RATE_LIMIT_PRESSURE,
    QUERY_SLOW_REQUESTS,
    QUERY_ZIP_BOMB_CANDIDATES,
    describe_query,
    get_query,
    get_stock_queries,
    list_queries,
    query_summary,
)


# ============================================================================
# Query Registry Tests
# ============================================================================


class TestQueryRegistry:
    """Test the stock query registry."""

    def test_get_stock_queries(self):
        """get_stock_queries() returns all queries."""
        queries = get_stock_queries()
        assert isinstance(queries, dict)
        assert len(queries) > 0

    def test_registry_has_expected_queries(self):
        """Registry contains expected query categories."""
        queries = get_stock_queries()
        categories = [
            "net_request_p95_latency",
            "error_rate_by_service",
            "cache_hit_ratio",
            "rate_limit_pressure",
            "http_429_responses",
            "policy_rejections",
            "extraction_failures",
            "bytes_written_by_service",
            "extraction_stats",
            "zip_bomb_candidates",
            "slow_requests",
        ]
        for category in categories:
            assert category in queries

    def test_list_queries(self):
        """list_queries() returns sorted list of names."""
        names = list_queries()
        assert isinstance(names, list)
        assert len(names) > 0
        # Verify it's sorted
        assert names == sorted(names)

    def test_query_names_are_strings(self):
        """All query names are strings."""
        names = list_queries()
        assert all(isinstance(name, str) for name in names)


# ============================================================================
# Query Access Tests
# ============================================================================


class TestQueryAccess:
    """Test accessing individual queries."""

    def test_get_query_by_name(self):
        """get_query() retrieves query by name."""
        query = get_query("net_request_p95_latency")
        assert isinstance(query, str)
        assert "SELECT" in query
        assert "service" in query

    def test_get_query_invalid_name(self):
        """get_query() raises KeyError for invalid name."""
        with pytest.raises(KeyError):
            get_query("nonexistent_query")

    def test_get_query_error_message(self):
        """get_query() error lists available queries."""
        try:
            get_query("invalid")
        except KeyError as e:
            error_msg = str(e)
            # Should list some available queries
            assert "Available" in error_msg or "available" in error_msg

    def test_describe_query(self):
        """describe_query() returns description."""
        desc = describe_query("net_request_p95_latency")
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "p95" in desc.lower() or "latency" in desc.lower()

    def test_describe_query_invalid_name(self):
        """describe_query() raises KeyError for invalid name."""
        with pytest.raises(KeyError):
            describe_query("nonexistent_query")


# ============================================================================
# Query Summary Tests
# ============================================================================


class TestQuerySummary:
    """Test query summary functionality."""

    def test_query_summary(self):
        """query_summary() returns all queries with descriptions."""
        summary = query_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_summary_has_descriptions(self):
        """All summary entries have non-empty descriptions."""
        summary = query_summary()
        for name, description in summary.items():
            assert isinstance(name, str)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_summary_matches_registry(self):
        """Summary names match registry."""
        summary = query_summary()
        registry = get_stock_queries()
        assert set(summary.keys()) == set(registry.keys())


# ============================================================================
# Query Syntax Tests (Basic)
# ============================================================================


class TestQuerySyntax:
    """Test that queries have valid basic SQL syntax."""

    def test_all_queries_have_select(self):
        """All queries contain SELECT keyword."""
        for name in list_queries():
            query = get_query(name)
            assert "SELECT" in query, f"Query {name} missing SELECT"

    def test_all_queries_have_from(self):
        """All queries contain FROM keyword."""
        for name in list_queries():
            query = get_query(name)
            assert "FROM" in query, f"Query {name} missing FROM"

    def test_query_descriptions_are_unique(self):
        """All query descriptions are unique."""
        summary = query_summary()
        descriptions = list(summary.values())
        assert len(descriptions) == len(set(descriptions))


# ============================================================================
# Specific Query Tests
# ============================================================================


class TestSpecificQueries:
    """Test specific queries for correctness."""

    def test_net_request_p95_query(self):
        """net_request_p95_latency query is valid."""
        query = QUERY_NET_REQUEST_P95_LATENCY
        assert "p95" in query.lower() or "APPROX_QUANTILE" in query
        assert "net.request" in query
        assert "elapsed_ms" in query

    def test_cache_hit_ratio_query(self):
        """cache_hit_ratio query is valid."""
        query = QUERY_CACHE_HIT_RATIO
        assert "cache" in query.lower()
        assert "hit" in query.lower() or "revalidated" in query
        assert "net.request" in query

    def test_rate_limit_pressure_query(self):
        """rate_limit_pressure query is valid."""
        query = QUERY_RATE_LIMIT_PRESSURE
        assert "ratelimit.acquire" in query
        assert "blocked" in query.lower()

    def test_policy_rejections_query(self):
        """policy_rejections query is valid."""
        query = QUERY_POLICY_REJECTIONS
        assert "error_code" in query.lower()
        assert "E_" in query

    def test_zip_bomb_query(self):
        """zip_bomb_candidates query is valid."""
        query = QUERY_ZIP_BOMB_CANDIDATES
        assert "ratio" in query.lower()
        assert "10.0" in query
        assert "extract.done" in query

    def test_slow_requests_query(self):
        """slow_requests query is valid."""
        query = QUERY_SLOW_REQUESTS
        assert "30000" in query or "30" in query
        assert "elapsed" in query.lower()

    def test_error_rate_query(self):
        """error_rate_by_service query is valid."""
        query = QUERY_ERROR_RATE_BY_SERVICE
        assert "ERROR" in query
        assert "error" in query.lower()

    def test_extraction_stats_query(self):
        """extraction_stats query is valid."""
        query = QUERY_EXTRACTION_STATS
        assert "extract.done" in query
        assert "AVG" in query

    def test_bytes_written_query(self):
        """bytes_written_by_service query is valid."""
        query = QUERY_BYTES_WRITTEN_BY_SERVICE
        assert "bytes_written" in query
        assert "extract.done" in query

    def test_http_429_query(self):
        """http_429_responses query is valid."""
        query = QUERY_429_RESPONSES
        assert "429" in query
        assert "net.request" in query

    def test_extraction_failures_query(self):
        """extraction_failures query is valid."""
        query = QUERY_EXTRACTION_FAILURES
        assert "extract.error" in query
        assert "error_code" in query


# ============================================================================
# Query Constant Tests
# ============================================================================


class TestQueryConstants:
    """Test that query constants match registry."""

    def test_all_constants_in_registry(self):
        """All query constants are in registry."""
        constants = [
            ("QUERY_NET_REQUEST_P95_LATENCY", "net_request_p95_latency"),
            ("QUERY_ERROR_RATE_BY_SERVICE", "error_rate_by_service"),
            ("QUERY_CACHE_HIT_RATIO", "cache_hit_ratio"),
            ("QUERY_RATE_LIMIT_PRESSURE", "rate_limit_pressure"),
            ("QUERY_429_RESPONSES", "http_429_responses"),
            ("QUERY_POLICY_REJECTIONS", "policy_rejections"),
            ("QUERY_EXTRACTION_FAILURES", "extraction_failures"),
            ("QUERY_BYTES_WRITTEN_BY_SERVICE", "bytes_written_by_service"),
            ("QUERY_EXTRACTION_STATS", "extraction_stats"),
            ("QUERY_ZIP_BOMB_CANDIDATES", "zip_bomb_candidates"),
            ("QUERY_SLOW_REQUESTS", "slow_requests"),
        ]

        queries = get_stock_queries()
        for const_name, registry_name in constants:
            assert registry_name in queries
