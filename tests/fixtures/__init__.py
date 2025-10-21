"""
Pytest fixtures for DocsToKG test suite.

Fixtures are organized by subsystem:
- determinism: Global determinism and environment controls
- http_fixtures: HTTP mocking and transport
- filesystem_fixtures: Temporary storage and encapsulation
- database_fixtures: DuckDB catalog and connectivity
- ratelimit_fixtures: Rate limiter management
- telemetry_fixtures: Event capture and assertion helpers
"""
