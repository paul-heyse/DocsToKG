# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.network.policy",
#   "purpose": "HTTP policy constants and defaults.",
#   "sections": []
# }
# === /NAVMAP ===

"""HTTP policy constants and defaults.

Defines RFC 9111-compliant caching policies, timeout budgets, and connection pooling
parameters for the unified HTTPX + Hishel + Tenacity client stack.

All values are tuned for reliable downloads from scientific data providers with
conservative timeouts and bounded concurrency to respect provider rate limits.
"""

# ============================================================================
# Timeout Budgets (seconds)
# ============================================================================

#: Connection establishment timeout (initial TCP 3-way handshake)
HTTP_CONNECT_TIMEOUT = 5.0

#: Read timeout (time between data packets on established connection)
#: Generous to accommodate slow providers and large file streaming
HTTP_READ_TIMEOUT = 30.0

#: Write timeout (time to send request body)
HTTP_WRITE_TIMEOUT = 15.0

#: Pool timeout (acquiring a connection from the pool)
HTTP_POOL_TIMEOUT = 5.0


# ============================================================================
# Connection Pooling
# ============================================================================

#: Maximum concurrent connections (total across all hosts)
#: Conservative to avoid provider blocks; tunable per-deployment
MAX_CONNECTIONS = 100

#: Maximum connections to keep open per host (reuse benefit vs resource cost)
MAX_KEEPALIVE_CONNECTIONS = 20

#: How long to keep idle connections alive (seconds)
#: Reuses connections for ~5 seconds of inactivity
KEEPALIVE_EXPIRY = 5.0


# ============================================================================
# HTTP/2 Settings
# ============================================================================

#: Enable HTTP/2 for multiplexing on concurrent requests to same host
#: Safe on all modern providers; fallback to HTTP/1.1 if unsupported
HTTP2_ENABLED = True


# ============================================================================
# Hishel RFC 9111 Cache Settings
# ============================================================================

#: Cache storage TTL (garbage collection interval, not freshness)
#: Files older than this are eligible for eviction; separate from RFC freshness
CACHE_STORAGE_TTL_SECONDS = 7 * 24 * 3600  # 7 days

#: How often to scan cache for expired entries (seconds)
CACHE_STORAGE_CHECK_INTERVAL_SECONDS = 24 * 3600  # once per day

#: Cacheable HTTP methods (GET and HEAD are safe; no POST caching)
CACHEABLE_METHODS = ["GET", "HEAD"]

#: Cacheable HTTP status codes (200, 301 permanent redirect, 308 permanent redirect)
#: Note: 301/308 are rare but some providers use for content versioning
CACHEABLE_STATUS_CODES = [200, 301, 308]

#: Allow heuristic caching (RFC 9111 Section 4.2.3)
#: If false (strict): only cache responses with explicit Cache-Control/Expires
#: If true: may cache responses without explicit directives (not recommended for APIs)
ALLOW_HEURISTIC_CACHING = False

#: Cache scope for this process
#: "shared": persistent across runs (default)
#: "run": temp dir cleared between runs (useful for tests)
CACHE_SCOPE = "shared"

#: Maximum cache size before automatic eviction (bytes)
#: 2 GB is reasonable for most deployments; tune based on available disk
CACHE_MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

#: Maximum cache entry age before eviction (seconds)
#: 90 days is a conservative refresh interval for stable content
CACHE_MAX_AGE_SECONDS = 90 * 24 * 3600  # 90 days

#: Budgeted time (milliseconds) for GC on startup (0 = disabled)
#: If > 0, sample eviction runs for at most this duration at startup
#: Use CLI `ontofetch cache gc` for full cleanup
CACHE_STARTUP_GC_MS = 0


# ============================================================================
# User-Agent Construction
# ============================================================================

#: User-Agent template format
#: Includes product name, version, project URL for polite client identification
#: Example: "ontofetch/0.1.0 (+https://github.com/…) run-id:abc123"
USER_AGENT_TEMPLATE = "ontofetch/{version} (+{project_url}) {run_id}"

#: Project URL for user-agent (where to report issues, get docs)
PROJECT_URL = "https://github.com/jgm/docstokg"


# ============================================================================
# Security & Compliance
# ============================================================================

#: Require TLS for all HTTPS connections (no self-signed or weak certs)
TLS_VERIFY_ENABLED = True

#: Minimum TLS version (1.2 is stable; 1.3 recommended for new deployments)
#: Note: This is controlled via SSLContext; not directly exposed here
TLS_MIN_VERSION_NAME = "TLSv1_2"

#: Disable automatic redirects (all redirects must be audited explicitly)
#: This prevents accidental access to unintended hosts via server-side redirects
FOLLOW_REDIRECTS = False

#: Maximum number of redirect hops to follow (with explicit audit)
MAX_REDIRECT_HOPS = 5


__all__ = [
    # Timeouts
    "HTTP_CONNECT_TIMEOUT",
    "HTTP_READ_TIMEOUT",
    "HTTP_WRITE_TIMEOUT",
    "HTTP_POOL_TIMEOUT",
    # Connection pooling
    "MAX_CONNECTIONS",
    "MAX_KEEPALIVE_CONNECTIONS",
    "KEEPALIVE_EXPIRY",
    # HTTP/2
    "HTTP2_ENABLED",
    # Caching
    "CACHE_STORAGE_TTL_SECONDS",
    "CACHE_STORAGE_CHECK_INTERVAL_SECONDS",
    "CACHEABLE_METHODS",
    "CACHEABLE_STATUS_CODES",
    "ALLOW_HEURISTIC_CACHING",
    "CACHE_SCOPE",
    "CACHE_MAX_SIZE_BYTES",
    "CACHE_MAX_AGE_SECONDS",
    "CACHE_STARTUP_GC_MS",
    # User-Agent
    "USER_AGENT_TEMPLATE",
    "PROJECT_URL",
    # Security
    "TLS_VERIFY_ENABLED",
    "TLS_MIN_VERSION_NAME",
    "FOLLOW_REDIRECTS",
    "MAX_REDIRECT_HOPS",
]
