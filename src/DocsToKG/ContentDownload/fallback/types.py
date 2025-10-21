"""Core types for fallback & resiliency strategy.

This module defines the core dataclasses used by the fallback orchestration
system:

- ResolutionOutcome: Enumeration of possible attempt outcomes
- AttemptPolicy: Configuration for a single source attempt
- AttemptResult: Result of a single attempt
- TierPlan: Configuration for a resolution tier
- FallbackPlan: Complete resolution plan with budgets and tiers

All types are frozen dataclasses for immutability and hashability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

# ============================================================================
# Resolution Outcomes
# ============================================================================

ResolutionOutcome = Literal[
    "success",  # Found valid PDF
    "no_pdf",  # Target exists but no PDF found
    "nonretryable",  # Error that won't improve with retry (e.g., 404)
    "retryable",  # Transient error (e.g., 429, 503)
    "timeout",  # Attempt exceeded timeout
    "skipped",  # Skipped due to health gate or cancellation
    "error",  # Unexpected exception
]

# ============================================================================
# AttemptPolicy: Configuration for a single source
# ============================================================================


@dataclass(frozen=True)
class AttemptPolicy:
    """Configuration policy for a single source within a tier.

    Attributes:
        name: Source identifier (e.g., "unpaywall_pdf", "arxiv_pdf")
        timeout_ms: Maximum time for this attempt (milliseconds)
        retries_max: Maximum retries for this source within Tenacity
        robots_respect: Whether to respect robots.txt for this source (landing pages)

    Example:
        ```python
        policy = AttemptPolicy(
            name="unpaywall_pdf",
            timeout_ms=6000,
            retries_max=3,
            robots_respect=False
        )
        ```
    """

    name: str = field()
    timeout_ms: int = field()
    retries_max: int = field()
    robots_respect: bool = field(default=False)

    def __post_init__(self) -> None:
        """Validate policy parameters."""
        if self.timeout_ms <= 0:
            msg = f"timeout_ms must be positive, got {self.timeout_ms}"
            raise ValueError(msg)
        if self.retries_max < 0:
            msg = f"retries_max must be non-negative, got {self.retries_max}"
            raise ValueError(msg)


# ============================================================================
# AttemptResult: Result of a single attempt
# ============================================================================


@dataclass(frozen=True)
class AttemptResult:
    """Result of a single source attempt.

    Attributes:
        outcome: ResolutionOutcome indicating success/failure/skip
        reason: Short reason code (e.g., "oa_pdf", "breaker_open", "api_error")
        elapsed_ms: Wall-clock time for this attempt
        url: Candidate PDF URL if outcome is "success"
        status: HTTP status code if applicable
        host: Hostname of the source
        meta: Additional metadata (content-type, redirect chain, etc.)

    Example:
        ```python
        # Successful outcome
        result = AttemptResult(
            outcome="success",
            reason="oa_pdf",
            elapsed_ms=1234,
            url="https://example.org/paper.pdf",
            status=200,
            host="example.org",
            meta={"content_type": "application/pdf"}
        )

        # Skipped outcome
        result = AttemptResult(
            outcome="skipped",
            reason="breaker_open",
            elapsed_ms=5,
            url=None,
            status=None,
            host="api.example.org",
            meta={"msg": "breaker is open"}
        )
        ```
    """

    outcome: ResolutionOutcome = field()
    reason: str = field()
    elapsed_ms: int = field()
    url: Optional[str] = field(default=None)
    status: Optional[int] = field(default=None)
    host: Optional[str] = field(default=None)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result integrity."""
        if self.outcome == "success" and self.url is None:
            msg = "outcome='success' requires url to be set"
            raise ValueError(msg)
        if self.elapsed_ms < 0:
            msg = f"elapsed_ms must be non-negative, got {self.elapsed_ms}"
            raise ValueError(msg)

    @property
    def is_success(self) -> bool:
        """Check if this attempt was successful."""
        return self.outcome == "success"

    @property
    def is_retryable(self) -> bool:
        """Check if failure is retryable (vs. terminal)."""
        return self.outcome in ("retryable", "timeout")

    @property
    def is_terminal(self) -> bool:
        """Check if failure is terminal (won't improve with retry)."""
        return self.outcome in ("success", "no_pdf", "nonretryable", "skipped")


# ============================================================================
# TierPlan: Configuration for a single tier
# ============================================================================


@dataclass(frozen=True)
class TierPlan:
    """Configuration for a single resolution tier.

    Tiers are executed sequentially. Within a tier, sources run in parallel
    up to the `parallel` limit. Execution stops as soon as one source succeeds.

    Attributes:
        name: Tier identifier (e.g., "direct_oa", "doi_follow")
        parallel: Maximum concurrent attempts within this tier
        sources: List of source names to attempt in this tier

    Example:
        ```python
        tier = TierPlan(
            name="direct_oa",
            parallel=2,
            sources=["unpaywall_pdf", "arxiv_pdf", "pmc_pdf"]
        )
        ```
    """

    name: str = field()
    parallel: int = field()
    sources: tuple[str, ...] = field()

    def __post_init__(self) -> None:
        """Validate tier configuration."""
        if self.parallel <= 0:
            msg = f"parallel must be positive, got {self.parallel}"
            raise ValueError(msg)
        if len(self.sources) == 0:
            msg = "sources list cannot be empty"
            raise ValueError(msg)
        if self.parallel > len(self.sources):
            msg = f"parallel ({self.parallel}) cannot exceed len(sources) ({len(self.sources)})"
            raise ValueError(msg)


# ============================================================================
# FallbackPlan: Complete resolution plan
# ============================================================================


@dataclass(frozen=True)
class FallbackPlan:
    """Complete fallback resolution plan.

    This is the top-level configuration object that orchestrates the entire
    PDF resolution process. It specifies:
    - Overall budgets (time, attempts, concurrency)
    - Tier ordering and parallelism
    - Per-source policies
    - Health gates

    Attributes:
        budgets: Dict with keys:
            - "total_timeout_ms": Hard cap for entire resolution
            - "total_attempts": Max attempts across all sources
            - "max_concurrent": Max concurrent threads across all tiers
            - "per_source_timeout_ms": Default timeout per source
        tiers: Ordered list of tier plans
        policies: Mapping of source name to AttemptPolicy
        gates: Dict with health gate configuration:
            - "skip_if_breaker_open": Skip if breaker open
            - "skip_if_http2_denied": Skip if HTTP/2 denied (rarely needed)
            - "offline_behavior": "metadata_only" | "block_all" | "cache_only"

    Example:
        ```python
        plan = FallbackPlan(
            budgets={
                "total_timeout_ms": 120000,
                "total_attempts": 20,
                "max_concurrent": 3,
                "per_source_timeout_ms": 10000,
            },
            tiers=(
                TierPlan("direct_oa", 2, ("unpaywall_pdf", "arxiv_pdf")),
                TierPlan("doi_follow", 1, ("doi_redirect_pdf",)),
            ),
            policies={
                "unpaywall_pdf": AttemptPolicy(...),
                "arxiv_pdf": AttemptPolicy(...),
            },
            gates={
                "skip_if_breaker_open": True,
                "offline_behavior": "metadata_only",
            }
        )
        ```
    """

    budgets: Dict[str, int] = field()
    tiers: tuple[TierPlan, ...] = field()
    policies: Dict[str, AttemptPolicy] = field()
    gates: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate plan integrity."""
        # Validate budgets
        required_budget_keys = {
            "total_timeout_ms",
            "total_attempts",
            "max_concurrent",
            "per_source_timeout_ms",
        }
        if not required_budget_keys.issubset(self.budgets.keys()):
            missing = required_budget_keys - set(self.budgets.keys())
            msg = f"budgets missing required keys: {missing}"
            raise ValueError(msg)

        if self.budgets["total_timeout_ms"] <= 0:
            msg = "budgets[total_timeout_ms] must be positive"
            raise ValueError(msg)
        if self.budgets["total_attempts"] <= 0:
            msg = "budgets[total_attempts] must be positive"
            raise ValueError(msg)
        if self.budgets["max_concurrent"] <= 0:
            msg = "budgets[max_concurrent] must be positive"
            raise ValueError(msg)

        # Validate tiers
        if len(self.tiers) == 0:
            msg = "tiers list cannot be empty"
            raise ValueError(msg)

        # Validate that all sources in tiers have policies
        all_sources = set()
        for tier in self.tiers:
            all_sources.update(tier.sources)

        missing_policies = all_sources - set(self.policies.keys())
        if missing_policies:
            msg = f"sources missing policies: {missing_policies}"
            raise ValueError(msg)

    def get_policy(self, source_name: str) -> AttemptPolicy:
        """Get policy for a source (with fallback to per_source_timeout_ms).

        Args:
            source_name: Name of the source

        Returns:
            AttemptPolicy for the source

        Raises:
            KeyError: If source not found in policies
        """
        return self.policies[source_name]

    @property
    def total_sources(self) -> int:
        """Total number of unique sources across all tiers."""
        sources = set()
        for tier in self.tiers:
            sources.update(tier.sources)
        return len(sources)

    @property
    def offline_behavior(self) -> str:
        """Get configured offline behavior."""
        return self.gates.get("offline_behavior", "metadata_only")

    @property
    def skip_if_breaker_open(self) -> bool:
        """Check if breaker should cause skips."""
        return self.gates.get("skip_if_breaker_open", True)
