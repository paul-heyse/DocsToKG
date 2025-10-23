# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.models",
#   "purpose": "Pydantic v2 Models for Telemetry & Dashboard Integration.",
#   "sections": [
#     {
#       "id": "attemptstatus",
#       "name": "AttemptStatus",
#       "anchor": "class-attemptstatus",
#       "kind": "class"
#     },
#     {
#       "id": "tiername",
#       "name": "TierName",
#       "anchor": "class-tiername",
#       "kind": "class"
#     },
#     {
#       "id": "metrictype",
#       "name": "MetricType",
#       "anchor": "class-metrictype",
#       "kind": "class"
#     },
#     {
#       "id": "telemetryattemptrecord",
#       "name": "TelemetryAttemptRecord",
#       "anchor": "class-telemetryattemptrecord",
#       "kind": "class"
#     },
#     {
#       "id": "telemetrybatchrecord",
#       "name": "TelemetryBatchRecord",
#       "anchor": "class-telemetrybatchrecord",
#       "kind": "class"
#     },
#     {
#       "id": "storageconfig",
#       "name": "StorageConfig",
#       "anchor": "class-storageconfig",
#       "kind": "class"
#     },
#     {
#       "id": "dashboardconfig",
#       "name": "DashboardConfig",
#       "anchor": "class-dashboardconfig",
#       "kind": "class"
#     },
#     {
#       "id": "metricsthreshold",
#       "name": "MetricsThreshold",
#       "anchor": "class-metricsthreshold",
#       "kind": "class"
#     },
#     {
#       "id": "telemetryconfig",
#       "name": "TelemetryConfig",
#       "anchor": "class-telemetryconfig",
#       "kind": "class"
#     },
#     {
#       "id": "performancemetrics",
#       "name": "PerformanceMetrics",
#       "anchor": "class-performancemetrics",
#       "kind": "class"
#     },
#     {
#       "id": "tiermetrics",
#       "name": "TierMetrics",
#       "anchor": "class-tiermetrics",
#       "kind": "class"
#     },
#     {
#       "id": "sourcemetrics",
#       "name": "SourceMetrics",
#       "anchor": "class-sourcemetrics",
#       "kind": "class"
#     },
#     {
#       "id": "dashboardpanel",
#       "name": "DashboardPanel",
#       "anchor": "class-dashboardpanel",
#       "kind": "class"
#     },
#     {
#       "id": "dashboarddefinition",
#       "name": "DashboardDefinition",
#       "anchor": "class-dashboarddefinition",
#       "kind": "class"
#     },
#     {
#       "id": "alertrule",
#       "name": "AlertRule",
#       "anchor": "class-alertrule",
#       "kind": "class"
#     },
#     {
#       "id": "exporttarget",
#       "name": "ExportTarget",
#       "anchor": "class-exporttarget",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Pydantic v2 Models for Telemetry & Dashboard Integration

Provides validated, typed data models for:
  - Telemetry records
  - Dashboard configurations
  - Metrics definitions
  - Performance thresholds
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ============================================================================
# Enums for Type Safety
# ============================================================================


class AttemptStatus(str, Enum):
    """Status of a resolution attempt."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TierName(str, Enum):
    """Fallback tier names."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class MetricType(str, Enum):
    """Type of metric."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


# ============================================================================
# Telemetry Record Models
# ============================================================================


class TelemetryAttemptRecord(BaseModel):
    """Validated telemetry record for a single resolution attempt."""

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=False,
        extra="forbid",
        use_enum_values=True,
    )

    # Core identification
    run_id: str = Field(..., description="Unique run identifier")
    attempt_id: str = Field(..., description="Unique attempt identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When attempt occurred"
    )

    # Resolution context
    tier: TierName = Field(..., description="Which tier this attempt was in")
    host: str = Field(..., description="Target host/source")
    url: str = Field(..., description="URL being attempted")

    # Outcome
    status: AttemptStatus = Field(..., description="Resolution outcome")
    reason: str | None = Field(None, description="Why it succeeded/failed")

    # Metrics
    elapsed_ms: int = Field(..., ge=0, description="Elapsed time in milliseconds")
    http_status: int | None = Field(None, ge=100, le=599, description="HTTP status code")

    # Extended telemetry
    retry_count: int = Field(default=0, ge=0, description="Number of retries")
    rate_limiter_wait_ms: int | None = Field(None, ge=0, description="Rate limit delay")
    breaker_state: str | None = Field(None, description="Circuit breaker state")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> TelemetryAttemptRecord:
        """Validate record consistency."""
        if self.status == AttemptStatus.SUCCESS and self.http_status != 200:
            raise ValueError("Success status requires HTTP 200")
        return self


class TelemetryBatchRecord(BaseModel):
    """Batch of telemetry records."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(..., description="Batch identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    records: list[TelemetryAttemptRecord] = Field(..., min_length=1)
    count: int = Field(..., ge=1, description="Number of records")

    @field_validator("count")
    @classmethod
    def validate_count(cls, v: int, info) -> int:
        """Validate count matches records length."""
        if info.data.get("records") and len(info.data["records"]) != v:
            raise ValueError("Count must match number of records")
        return v


# ============================================================================
# Configuration Models
# ============================================================================


class StorageConfig(BaseModel):
    """Storage configuration with validation."""

    model_config = ConfigDict(extra="forbid")

    storage_type: Literal["sqlite", "jsonl", "hybrid"] = Field(
        default="sqlite", description="Storage backend type"
    )
    path: str = Field(default="Data/Manifests/manifest.sqlite3", description="Storage file path")
    backup_path: str | None = Field(None, description="Backup file path")
    auto_backup: bool = Field(default=True, description="Automatic backups enabled")
    batch_size: int = Field(default=100, ge=1, le=10000, description="Batch size for operations")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate storage path."""
        if not v or len(v) < 3:
            raise ValueError("Path must be valid")
        return v


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Dashboard enabled")
    export_formats: list[Literal["grafana", "prometheus", "json"]] = Field(
        default=["json"], description="Export formats"
    )
    update_interval_s: int = Field(
        default=60, ge=5, le=3600, description="Update interval in seconds"
    )
    retention_days: int = Field(
        default=30, ge=1, le=365, description="Data retention period in days"
    )

    @field_validator("export_formats")
    @classmethod
    def validate_formats(cls, v: list[str]) -> list[str]:
        """Validate export formats."""
        valid = {"grafana", "prometheus", "json"}
        if not all(f in valid for f in v):
            raise ValueError(f"Invalid format. Must be one of {valid}")
        return v


class MetricsThreshold(BaseModel):
    """Thresholds for metrics alerting."""

    model_config = ConfigDict(extra="forbid")

    success_rate_min: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Minimum acceptable success rate"
    )
    latency_p95_max_ms: int = Field(
        default=5000, ge=100, description="Maximum acceptable P95 latency"
    )
    latency_p99_max_ms: int = Field(
        default=10000, ge=100, description="Maximum acceptable P99 latency"
    )
    error_rate_max: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Maximum acceptable error rate"
    )


class TelemetryConfig(BaseModel):
    """Complete telemetry configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Telemetry enabled")
    storage: StorageConfig = Field(default_factory=StorageConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    thresholds: MetricsThreshold = Field(default_factory=MetricsThreshold)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )


# ============================================================================
# Performance Models
# ============================================================================


class PerformanceMetrics(BaseModel):
    """Validated performance metrics."""

    model_config = ConfigDict(frozen=True)

    total_attempts: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    avg_latency_ms: float = Field(..., ge=0)
    p50_latency_ms: float = Field(..., ge=0)
    p95_latency_ms: float = Field(..., ge=0)
    p99_latency_ms: float = Field(..., ge=0)
    error_count: int = Field(..., ge=0)
    timeout_count: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_percentiles(self) -> PerformanceMetrics:
        """Validate percentile ordering."""
        if not (self.p50_latency_ms <= self.p95_latency_ms <= self.p99_latency_ms):
            raise ValueError("Percentiles must be in ascending order")
        if self.success_count + self.error_count + self.timeout_count > self.total_attempts:
            raise ValueError("Outcome counts cannot exceed total attempts")
        return self


class TierMetrics(BaseModel):
    """Metrics for a specific tier."""

    model_config = ConfigDict(frozen=True)

    tier: TierName = Field(...)
    metrics: PerformanceMetrics = Field(...)
    sources: list[str] = Field(..., min_length=1)


class SourceMetrics(BaseModel):
    """Metrics for a specific source."""

    model_config = ConfigDict(frozen=True)

    host: str = Field(...)
    metrics: PerformanceMetrics = Field(...)
    tier: TierName = Field(...)


# ============================================================================
# Dashboard Models
# ============================================================================


class DashboardPanel(BaseModel):
    """Dashboard panel definition."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(...)
    title: str = Field(...)
    type: Literal["gauge", "graph", "table", "stat"] = Field(...)
    metrics: list[str] = Field(...)
    thresholds: dict[str, float] | None = Field(None)


class DashboardDefinition(BaseModel):
    """Complete dashboard definition."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(...)
    title: str = Field(...)
    version: int = Field(default=1, ge=1)
    panels: list[DashboardPanel] = Field(...)
    refresh_interval: str = Field(default="30s")
    tags: list[str] = Field(default_factory=list)


# ============================================================================
# Alert Models
# ============================================================================


class AlertRule(BaseModel):
    """Alert rule definition."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(...)
    name: str = Field(...)
    metric: str = Field(...)
    condition: Literal["above", "below", "equals"] = Field(...)
    threshold: float = Field(...)
    duration_s: int = Field(default=300, ge=60)
    enabled: bool = Field(default=True)

    @field_validator("duration_s")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        """Validate alert duration."""
        if v < 60:
            raise ValueError("Duration must be at least 60 seconds")
        return v


# ============================================================================
# Export Models
# ============================================================================


class ExportTarget(BaseModel):
    """Export target configuration."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["grafana", "prometheus", "json", "csv"] = Field(...)
    destination: str = Field(..., description="Export destination path or URL")
    schedule: str | None = Field(None, description="Cron schedule for exports")
    enabled: bool = Field(default=True)


__all__ = [
    "AttemptStatus",
    "TierName",
    "MetricType",
    "TelemetryAttemptRecord",
    "TelemetryBatchRecord",
    "StorageConfig",
    "DashboardConfig",
    "MetricsThreshold",
    "TelemetryConfig",
    "PerformanceMetrics",
    "TierMetrics",
    "SourceMetrics",
    "DashboardPanel",
    "DashboardDefinition",
    "AlertRule",
    "ExportTarget",
]
