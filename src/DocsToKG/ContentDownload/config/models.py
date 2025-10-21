"""
Pydantic v2 Configuration Models for ContentDownload

Provides strict, typed configuration for all ContentDownload subsystems:
- HTTP client settings (timeouts, TLS, proxies)
- Retry and backoff policies
- Rate limiting policies
- Robots.txt policies
- Download policies (atomic write, content verification)
- Telemetry configuration
- Resolver-specific overrides
- Top-level ContentDownloadConfig as single source of truth

All models use extra="forbid" for strict validation. Environment variables
and CLI overrides follow: file < env < CLI precedence.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================================
# Shared Policy Models
# ============================================================================


class RetryPolicy(BaseModel):
    """Configuration for HTTP request retry behavior."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    retry_statuses: List[int] = Field(
        default=[429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retry",
    )
    max_attempts: int = Field(default=4, description="Maximum retry attempts")
    base_delay_ms: int = Field(default=200, description="Base delay in ms")
    max_delay_ms: int = Field(default=4000, description="Maximum delay in ms")
    jitter_ms: int = Field(default=100, description="Jitter range in ms")

    @field_validator("max_attempts")
    @classmethod
    def validate_max_attempts(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_attempts must be >= 1")
        return v

    @field_validator("base_delay_ms", "max_delay_ms", "jitter_ms")
    @classmethod
    def validate_delays(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Delay values must be >= 0")
        return v


class BackoffPolicy(BaseModel):
    """Configuration for backoff strategy."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    strategy: Literal["exponential", "constant"] = Field(
        default="exponential", description="Backoff strategy"
    )
    factor: float = Field(default=2.0, description="Backoff factor")

    @field_validator("factor")
    @classmethod
    def validate_factor(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("factor must be > 0")
        return v


class RateLimitPolicy(BaseModel):
    """Configuration for rate limiting (token bucket)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    capacity: int = Field(default=5, description="Token bucket capacity")
    refill_per_sec: float = Field(default=1.0, description="Tokens/sec refill rate")
    burst: int = Field(default=2, description="Burst allowance")

    @field_validator("capacity", "burst")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be > 0")
        return v

    @field_validator("refill_per_sec")
    @classmethod
    def validate_refill(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Must be > 0")
        return v


class RobotsPolicy(BaseModel):
    """Configuration for robots.txt handling."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable robots.txt checks")
    ttl_seconds: int = Field(default=3600, description="Cache TTL")

    @field_validator("ttl_seconds")
    @classmethod
    def validate_ttl(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("ttl_seconds must be > 0")
        return v


class DownloadPolicy(BaseModel):
    """Configuration for download safety and integrity."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    atomic_write: bool = Field(default=True, description="Use atomic writes")
    verify_content_length: bool = Field(default=True, description="Verify Content-Length matches")
    chunk_size_bytes: int = Field(default=1 << 20, description="Stream chunk size")
    max_bytes: Optional[int] = Field(
        default=None, description="Maximum download size (None = unlimited)"
    )

    @field_validator("chunk_size_bytes")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size_bytes must be > 0")
        return v

    @field_validator("max_bytes")
    @classmethod
    def validate_max_bytes(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("max_bytes must be > 0 or None")
        return v


class HttpClientConfig(BaseModel):
    """Configuration for HTTP client behavior."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    user_agent: str = Field(default="DocsToKG/ContentDownload", description="User-Agent string")
    mailto: Optional[str] = Field(default=None, description="Email for robots.txt compliance")
    timeout_connect_s: float = Field(default=10.0, description="Connection timeout in seconds")
    timeout_read_s: float = Field(default=60.0, description="Read timeout in seconds")
    verify_tls: bool = Field(default=True, description="Verify TLS certificates")
    proxies: Dict[str, str] = Field(default_factory=dict, description="Proxy configuration")
    polite_headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "DocsToKG/ContentDownload (+mailto:research@example.com)"
        },
        description="Polite HTTP headers (User-Agent, referer, etc.)",
    )

    @field_validator("timeout_connect_s", "timeout_read_s")
    @classmethod
    def validate_timeouts(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Timeouts must be > 0")
        return v


class TelemetryConfig(BaseModel):
    """Configuration for telemetry and manifest output."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    sinks: List[str] = Field(
        default_factory=lambda: ["csv"],
        description="Output sinks (csv, jsonl, console, otlp)",
    )
    csv_path: str = Field(default="attempts.csv", description="CSV output path")
    manifest_path: str = Field(default="manifest.jsonl", description="Manifest path")

    @field_validator("sinks")
    @classmethod
    def validate_sinks(cls, v: List[str]) -> List[str]:
        valid = {"csv", "jsonl", "console", "otlp"}
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f"Invalid sinks: {invalid}. Must be in {valid}")
        return v


# ============================================================================
# Resolver-Specific Config Models
# ============================================================================


class ResolverCommonConfig(BaseModel):
    """Common configuration options for all resolvers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable this resolver")
    retry: RetryPolicy = Field(default_factory=RetryPolicy, description="Retry policy")
    rate_limit: RateLimitPolicy = Field(
        default_factory=RateLimitPolicy, description="Rate limit policy"
    )
    timeout_read_s: Optional[float] = Field(default=None, description="Override HTTP read timeout")


class UnpaywallConfig(ResolverCommonConfig):
    """Unpaywall resolver configuration."""

    email: Optional[str] = Field(default=None, description="Email for Unpaywall API")


class CrossrefConfig(ResolverCommonConfig):
    """Crossref resolver configuration."""

    mailto: Optional[str] = Field(default=None, description="Email for Crossref API")


class ArxivConfig(ResolverCommonConfig):
    """arXiv resolver configuration."""

    pass


class EuropePmcConfig(ResolverCommonConfig):
    """Europe PMC resolver configuration."""

    pass


class CoreConfig(ResolverCommonConfig):
    """CORE resolver configuration."""

    pass


class DoajConfig(ResolverCommonConfig):
    """DOAJ resolver configuration."""

    pass


class SemanticScholarConfig(ResolverCommonConfig):
    """Semantic Scholar resolver configuration."""

    pass


class LandingPageConfig(ResolverCommonConfig):
    """Landing page resolver configuration."""

    pass


class WaybackConfig(ResolverCommonConfig):
    """Wayback Machine resolver configuration."""

    pass


class OpenAlexConfig(ResolverCommonConfig):
    """OpenAlex resolver configuration."""

    pass


class ZenodoConfig(ResolverCommonConfig):
    """Zenodo resolver configuration."""

    pass


class OsfsConfig(ResolverCommonConfig):
    """OSF resolver configuration."""

    pass


class OpenAireConfig(ResolverCommonConfig):
    """OpenAIRE resolver configuration."""

    pass


class HalConfig(ResolverCommonConfig):
    """HAL resolver configuration."""

    pass


class FigshareConfig(ResolverCommonConfig):
    """Figshare resolver configuration."""

    pass


# ============================================================================
# Top-Level Resolvers Configuration
# ============================================================================


class ResolversConfig(BaseModel):
    """Configuration for resolver system and ordering."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    order: List[str] = Field(
        default_factory=lambda: [
            "unpaywall",
            "crossref",
            "arxiv",
            "europe_pmc",
            "core",
            "doaj",
            "semantic_scholar",
            "landing_page",
            "wayback",
            "openalex",
            "zenodo",
            "osf",
            "openaire",
            "hal",
            "figshare",
        ],
        description="Resolver execution order",
    )
    unpaywall: UnpaywallConfig = Field(
        default_factory=UnpaywallConfig, description="Unpaywall config"
    )
    crossref: CrossrefConfig = Field(default_factory=CrossrefConfig, description="Crossref config")
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig, description="arXiv config")
    europe_pmc: EuropePmcConfig = Field(
        default_factory=EuropePmcConfig, description="Europe PMC config"
    )
    core: CoreConfig = Field(default_factory=CoreConfig, description="CORE config")
    doaj: DoajConfig = Field(default_factory=DoajConfig, description="DOAJ config")
    semantic_scholar: SemanticScholarConfig = Field(
        default_factory=SemanticScholarConfig, description="Semantic Scholar config"
    )
    landing_page: LandingPageConfig = Field(
        default_factory=LandingPageConfig, description="Landing page config"
    )
    wayback: WaybackConfig = Field(default_factory=WaybackConfig, description="Wayback config")
    openalex: OpenAlexConfig = Field(default_factory=OpenAlexConfig, description="OpenAlex config")
    zenodo: ZenodoConfig = Field(default_factory=ZenodoConfig, description="Zenodo config")
    osf: OsfsConfig = Field(default_factory=OsfsConfig, description="OSF config")
    openaire: OpenAireConfig = Field(default_factory=OpenAireConfig, description="OpenAIRE config")
    hal: HalConfig = Field(default_factory=HalConfig, description="HAL config")
    figshare: FigshareConfig = Field(default_factory=FigshareConfig, description="Figshare config")

    @field_validator("order")
    @classmethod
    def validate_order(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("order must not be empty")
        return v


# ============================================================================
# Storage & Catalog Configuration (PR #9: Artifact Catalog & Storage Index)
# ============================================================================


class StorageConfig(BaseModel):
    """Storage backend and layout configuration for downloaded artifacts."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    backend: Literal["fs", "s3"] = Field(
        default="fs",
        description="Storage backend: filesystem or S3",
    )
    root_dir: str = Field(
        default="data/docs",
        description="Root directory for final artifacts",
    )
    layout: Literal["policy_path", "cas"] = Field(
        default="policy_path",
        description="Layout strategy: policy_path (human-friendly) or cas (content-addressable)",
    )
    cas_prefix: str = Field(
        default="sha256",
        description="CAS prefix (e.g., 'sha256')",
    )
    hardlink_dedup: bool = Field(
        default=True,
        description="Enable hardlink deduplication on POSIX systems",
    )
    s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket name (required if backend='s3')",
    )
    s3_prefix: str = Field(
        default="docs/",
        description="S3 object key prefix",
    )
    s3_storage_class: str = Field(
        default="STANDARD",
        description="S3 storage class (STANDARD, INTELLIGENT_TIERING, GLACIER, etc.)",
    )


class CatalogConfig(BaseModel):
    """Artifact catalog database and retention configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    backend: Literal["sqlite", "postgres"] = Field(
        default="sqlite",
        description="Database backend (sqlite or postgres)",
    )
    path: str = Field(
        default="state/catalog.sqlite",
        description="Database file path (for SQLite) or connection URL",
    )
    wal_mode: bool = Field(
        default=True,
        description="Enable WAL mode for SQLite",
    )
    compute_sha256: bool = Field(
        default=True,
        description="Compute SHA-256 on successful downloads",
    )
    verify_on_register: bool = Field(
        default=False,
        description="Verify SHA-256 after finalization before registering",
    )
    retention_days: int = Field(
        default=0,
        ge=0,
        description="Retention policy: 0 = disabled; N > 0 = delete records older than N days",
    )
    orphan_ttl_days: int = Field(
        default=7,
        ge=1,
        description="GC eligibility: files not referenced for N days are candidates",
    )


# ============================================================================
# Top-Level Configuration
# ============================================================================


class ContentDownloadConfig(BaseModel):
    """
    Single source of truth for ContentDownload configuration.

    Loaded from file (YAML/JSON), overlaid with environment variables,
    and finally overridden by CLI arguments. Precedence: file < env < CLI.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: Optional[str] = Field(
        default=None, description="Unique run identifier for traceability"
    )
    http: HttpClientConfig = Field(
        default_factory=HttpClientConfig, description="HTTP client configuration"
    )
    robots: RobotsPolicy = Field(default_factory=RobotsPolicy, description="Robots.txt policy")
    download: DownloadPolicy = Field(
        default_factory=DownloadPolicy, description="Download safety policy"
    )
    telemetry: TelemetryConfig = Field(
        default_factory=TelemetryConfig, description="Telemetry configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage backend configuration",
    )
    catalog: CatalogConfig = Field(
        default_factory=CatalogConfig,
        description="Artifact catalog configuration",
    )
    resolvers: ResolversConfig = Field(
        default_factory=ResolversConfig, description="Resolver configuration"
    )

    def config_hash(self) -> str:
        """
        Compute deterministic SHA256 hash of config for reproducibility.

        Returns:
            Hex-encoded SHA256 hash of normalized config JSON.
        """
        import hashlib
        import json

        normalized = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode()).hexdigest()
