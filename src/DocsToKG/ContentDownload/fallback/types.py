"""
Core types for fallback & resiliency strategy.

Defines the data contracts for tiered PDF resolution, including budgets,
policies, attempt results, and resolution outcomes.

Type Contracts:
- ResolutionOutcome: Literal result types (success, no_pdf, timeout, etc.)
- AttemptPolicy: Per-source configuration (timeout, retries, robots policy)
- AttemptResult: Single resolution attempt outcome
- TierPlan: Tier definition (sources, parallelism)
- FallbackPlan: Complete resolution strategy (budgets, tiers, policies, gates)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional

# ============================================================================
# OUTCOME TYPES
# ============================================================================

ResolutionOutcome = Literal[
    "success",  # PDF URL found and validated
    "no_pdf",  # Source returned non-PDF (e.g., HTML, text)
    "nonretryable",  # Permanent failure (404, auth, etc.)
    "retryable",  # Transient failure (429, 5xx, timeout)
    "timeout",  # Exceeded per-source timeout
    "skipped",  # Skipped due to health gate
    "error",  # Unexpected error
]
"""
Literals for resolution outcomes.

- success: PDF URL found and available
- no_pdf: Source responded but no PDF (e.g., landing page only)
- nonretryable: Permanent error (auth, policy, not found)
- retryable: Transient error (rate limit, server error)
- timeout: Exceeded per-source timeout
- skipped: Skipped due to breaker state, offline, rate limiter
- error: Unexpected error (invalid state, etc.)
"""


# ============================================================================
# POLICY & CONFIGURATION
# ============================================================================


@dataclass(frozen=True, slots=True)
class AttemptPolicy:
    """
    Per-source configuration for resolution attempts.

    Attributes:
        name: Source name (e.g., 'unpaywall_pdf', 'arxiv_pdf')
        timeout_ms: Maximum time for this source (milliseconds)
        retries_max: Maximum retry attempts for transient errors
        robots_respect: Whether to respect robots.txt for this source
        metadata: Optional custom metadata (extensible)
    """

    name: str
    timeout_ms: int
    retries_max: int
    robots_respect: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate policy invariants."""
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")
        if self.retries_max < 0:
            raise ValueError(f"retries_max must be non-negative, got {self.retries_max}")


@dataclass(frozen=True, slots=True)
class TierPlan:
    """
    Tier definition for tiered resolution.

    Sources within a tier are executed in parallel up to `parallel` concurrency.
    Tiers are executed sequentially (tier 1 → tier 2 → ...).

    Attributes:
        name: Tier name (e.g., 'direct_oa', 'doi_follow', 'web_scrape')
        parallel: Number of concurrent sources within this tier
        sources: List of source names in this tier
    """

    name: str
    parallel: int
    sources: list[str]

    def __post_init__(self) -> None:
        """Validate tier invariants."""
        if self.parallel < 1:
            raise ValueError(f"parallel must be >= 1, got {self.parallel}")
        if not self.sources:
            raise ValueError("sources list cannot be empty")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("sources list contains duplicates")


@dataclass(frozen=True, slots=True)
class FallbackPlan:
    """
    Complete fallback resolution strategy configuration.

    Contains budgets, tier ordering, per-source policies, and health gates.

    Attributes:
        budgets: Global execution constraints
            - total_timeout_ms: Total time budget (milliseconds)
            - total_attempts: Maximum attempts across all sources
            - max_concurrent: Maximum concurrent threads
            - per_source_timeout_ms: Default timeout per source
        tiers: List of tier plans (executed sequentially)
        policies: Per-source configuration policies (dict[source_name, AttemptPolicy])
        gates: Optional health gate configuration (breaker, offline, rate limiter awareness)
    """

    budgets: Dict[str, Any]
    tiers: list[TierPlan]
    policies: Dict[str, AttemptPolicy]
    gates: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate plan invariants."""
        # Validate budgets
        if not self.budgets.get("total_timeout_ms", 0) > 0:
            raise ValueError("budgets.total_timeout_ms must be positive")
        if not self.budgets.get("total_attempts", 0) > 0:
            raise ValueError("budgets.total_attempts must be positive")
        if not self.budgets.get("max_concurrent", 0) > 0:
            raise ValueError("budgets.max_concurrent must be positive")

        # Validate tiers exist
        if not self.tiers:
            raise ValueError("tiers list cannot be empty")

        # Validate all tier sources have policies
        all_sources = set()
        for tier in self.tiers:
            all_sources.update(tier.sources)

        missing_policies = all_sources - set(self.policies.keys())
        if missing_policies:
            raise ValueError(f"Missing policies for sources: {missing_policies}")

        # Validate no duplicate sources across tiers
        seen_sources: set[str] = set()
        for tier in self.tiers:
            duplicates = seen_sources & set(tier.sources)
            if duplicates:
                raise ValueError(f"Source(s) appear in multiple tiers: {duplicates}")
            seen_sources.update(tier.sources)


# ============================================================================
# ATTEMPT RESULTS
# ============================================================================


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """
    Single resolution attempt outcome.

    Represents the result of attempting resolution from one source.
    Used internally by FallbackOrchestrator and emitted as telemetry events.

    Attributes:
        outcome: Literal outcome type (success, no_pdf, timeout, etc.)
        url: PDF URL if found (only populated when outcome='success')
        reason: Short reason code for failure (e.g., 'robots', 'timeout')
        elapsed_ms: Time spent on this attempt (milliseconds)
        status: HTTP status code if applicable
        host: Hostname/domain of the source
        retry_count: Number of retries attempted
        meta: Optional additional metadata (extensible)
    """

    outcome: ResolutionOutcome
    url: Optional[str] = None
    reason: Optional[str] = None
    elapsed_ms: int = 0
    status: Optional[int] = None
    host: Optional[str] = None
    retry_count: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result invariants."""
        # Success must have URL
        if self.outcome == "success" and not self.url:
            raise ValueError("outcome=success requires url to be set")

        # Non-success should not have URL
        if self.outcome != "success" and self.url:
            raise ValueError(f"outcome={self.outcome} should not have url set")

        # Validate elapsed time
        if self.elapsed_ms < 0:
            raise ValueError(f"elapsed_ms must be non-negative, got {self.elapsed_ms}")

        # Validate retry count
        if self.retry_count < 0:
            raise ValueError(f"retry_count must be non-negative, got {self.retry_count}")

    def is_success(self) -> bool:
        """Check if this result is a success."""
        return self.outcome == "success" and self.url is not None

    def is_terminal(self) -> bool:
        """Check if this outcome is terminal (no retry possible)."""
        return self.outcome in ("success", "no_pdf", "nonretryable", "timeout")


__all__ = [
    "ResolutionOutcome",
    "AttemptPolicy",
    "AttemptResult",
    "TierPlan",
    "FallbackPlan",
]
