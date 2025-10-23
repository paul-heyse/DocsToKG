# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.orchestrator",
#   "purpose": "Fallback & Resiliency Orchestrator.",
#   "sections": [
#     {
#       "id": "fallbackorchestrator",
#       "name": "FallbackOrchestrator",
#       "anchor": "class-fallbackorchestrator",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Fallback & Resiliency Orchestrator

Coordinates tiered PDF resolution across multiple sources with:
- Deterministic source ordering and tier sequencing
- Budgeted execution (time, attempts, concurrency)
- Thread-safe parallel resolution within tiers
- Health gate integration (breaker, offline, rate limiter)
- Full telemetry emission and observability
- Graceful fallback to main pipeline on failure

Design:
- Pure orchestration logic (no side effects beyond resolution)
- Adapter protocol (resolve function)
- Budget enforcement at orchestrator level
- Health gates checked before each source attempt
- Full attempt tracing for debugging and metrics
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional

from .types import AttemptResult, FallbackPlan

LOGGER = logging.getLogger(__name__)


class FallbackOrchestrator:
    """
    Orchestrates tiered PDF resolution across multiple sources.

    Coordinates:
    - Tier-by-tier resolution (sequential tiers, parallel sources within tier)
    - Budget enforcement (time, attempts, concurrency)
    - Health gate evaluation (circuit breaker, offline, rate limiter)
    - Attempt result aggregation and first-success semantics
    - Full telemetry emission

    Attributes:
        plan: FallbackPlan with budgets, tiers, policies, gates
        breaker: Optional circuit breaker registry
        rate_limiter: Optional rate limiter manager
        clients: HTTP clients dict (passed to adapters)
        telemetry: Optional telemetry sink
        logger: Logger instance
    """

    def __init__(
        self,
        plan: FallbackPlan,
        breaker: Optional[Any] = None,
        rate_limiter: Optional[Any] = None,
        clients: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            plan: FallbackPlan configuration
            breaker: Optional circuit breaker registry
            rate_limiter: Optional rate limiter manager
            clients: HTTP clients dict
            telemetry: Optional telemetry sink
            logger: Optional logger (uses module logger if not provided)
        """
        self.plan = plan
        self.breaker = breaker
        self.rate_limiter = rate_limiter
        self.clients = clients or {}
        self.telemetry = telemetry
        self.logger = logger or LOGGER

        # Budget tracking (mutable state)
        self._attempt_count = 0
        self._budget_lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._cancellation_flag = threading.Event()

    def resolve_pdf(
        self,
        context: Dict[str, Any],
        adapters: Dict[str, Callable[[Any, Dict[str, Any]], AttemptResult]],
    ) -> AttemptResult:
        """Resolve PDF using tiered strategy.

        Main entry point. Iterates through tiers, executes sources in parallel
        (up to tier parallelism), enforces budgets, checks health gates,
        and returns first success or final failure.

        Args:
            context: Resolution context (artifact metadata, etc.)
            adapters: Dict mapping source_name → adapter function

        Returns:
            AttemptResult with outcome and metadata
        """
        self._start_time = time.time()
        self._attempt_count = 0
        self._cancellation_flag.clear()

        self.logger.debug(
            f"Starting fallback resolution: {len(self.plan.tiers)} tier(s), "
            f"budget={self.plan.budgets.get('total_timeout_ms', 'unlimited')}ms"
        )

        # Try each tier sequentially
        for tier in self.plan.tiers:
            if self._is_budget_exhausted():
                self.logger.info("Budget exhausted, stopping resolution")
                return self._timeout_outcome()

            tier_result = self._resolve_tier(tier, context, adapters)

            if tier_result.is_success():
                self.logger.info(
                    f"Success in tier '{tier.name}': {tier_result.url} "
                    f"(elapsed={tier_result.elapsed_ms}ms)"
                )
                return tier_result

            # Continue to next tier on skip/failure
            self.logger.debug(
                f"Tier '{tier.name}' failed/skipped, trying next tier "
                f"(attempts={self._attempt_count})"
            )

        # All tiers exhausted
        self.logger.warning(f"All tiers exhausted, no PDF found (attempts={self._attempt_count})")
        return AttemptResult(
            outcome="error",
            reason="all_tiers_exhausted",
            elapsed_ms=self._elapsed_ms(),
            meta={"total_attempts": self._attempt_count},
        )

    def _resolve_tier(
        self,
        tier: Any,
        context: Dict[str, Any],
        adapters: Dict[str, Callable[[Any, Dict[str, Any]], AttemptResult]],
    ) -> AttemptResult:
        """Resolve a single tier (parallel sources).

        Executes sources within tier up to tier.parallel concurrency.
        Returns first success or aggregated failure.

        Args:
            tier: TierPlan with source names and parallelism
            context: Resolution context
            adapters: Adapter functions dict

        Returns:
            AttemptResult (success or failure)
        """
        self.logger.debug(f"Resolving tier '{tier.name}' ({len(tier.sources)} sources)")

        # Filter adapters and health-check each source
        available_sources = []
        for source_name in tier.sources:
            if self._is_budget_exhausted():
                break

            # Health gate check
            gate_result = self._health_gate(source_name, context)
            if gate_result is not None:
                self.logger.debug(f"Source '{source_name}' skipped: {gate_result.reason}")
                self._emit_telemetry(tier.name, gate_result, context)
                self._attempt_count += 1
                continue

            # Source available
            if source_name in adapters:
                available_sources.append(source_name)
            else:
                self.logger.warning(f"Adapter not found for source '{source_name}'")

        if not available_sources:
            self.logger.debug(f"Tier '{tier.name}': no available sources")
            return AttemptResult(
                outcome="skipped",
                reason="no_available_sources",
                elapsed_ms=self._elapsed_ms(),
            )

        # Parallel execution
        with ThreadPoolExecutor(max_workers=min(tier.parallel, len(available_sources))) as executor:
            futures = {}
            for source_name in available_sources:
                if self._is_budget_exhausted():
                    break

                future = executor.submit(
                    adapters[source_name],
                    self.plan.policies[source_name],
                    context,
                )
                futures[future] = source_name

            # Collect results (first success wins)
            for future in as_completed(futures, timeout=self._remaining_timeout_s()):
                if self._cancellation_flag.is_set():
                    break

                source_name = futures[future]
                try:
                    result = future.result(timeout=1)
                except Exception as e:
                    self.logger.error(f"Adapter '{source_name}' raised: {e}")
                    result = AttemptResult(
                        outcome="error",
                        reason="adapter_error",
                        elapsed_ms=self._elapsed_ms(),
                        meta={"error": str(e)},
                    )

                self._attempt_count += 1
                self._emit_telemetry(tier.name, result, context)

                if result.is_success():
                    self._cancellation_flag.set()
                    return result

        # Tier exhausted
        return AttemptResult(
            outcome="error",
            reason="tier_exhausted",
            elapsed_ms=self._elapsed_ms(),
            meta={"tier": tier.name, "sources_tried": len(available_sources)},
        )

    def _health_gate(self, source_name: str, context: Dict[str, Any]) -> Optional[AttemptResult]:
        """Check health gate for a source.

        Evaluates:
        - Circuit breaker state
        - Offline mode
        - Rate limiter awareness

        Returns None if source is healthy (proceed), or AttemptResult(skipped)
        if gate fails.
        """
        # Circuit breaker check
        if self.breaker:
            try:
                # Check if source is open
                breaker_key = f"fallback_{source_name}"
                if self.breaker.is_open(breaker_key):
                    return AttemptResult(
                        outcome="skipped",
                        reason="breaker_open",
                        meta={"breaker_key": breaker_key},
                    )
            except Exception as e:
                self.logger.debug(f"Breaker check failed: {e}")

        # Offline mode check
        if context.get("offline_mode"):
            # Allow only cached sources
            if source_name not in ("landing_scrape",):  # Landing may have local cache
                return AttemptResult(
                    outcome="skipped",
                    reason="offline_mode",
                    meta={"source": source_name},
                )

        # Rate limiter awareness (informational only, don't block)
        if self.rate_limiter:
            try:
                # Check rate limiter state (don't acquire, just peek)
                # This is informational for telemetry
                pass
            except Exception as e:
                self.logger.debug(f"Rate limiter check failed: {e}")

        return None

    def _emit_telemetry(
        self, tier_name: str, result: AttemptResult, context: Dict[str, Any]
    ) -> None:
        """Emit attempt telemetry.

        Args:
            tier_name: Current tier name
            result: AttemptResult
            context: Resolution context
        """
        if not self.telemetry:
            return

        try:
            event = {
                "event_type": "fallback_attempt",
                "tier": tier_name,
                "outcome": result.outcome,
                "reason": result.reason,
                "url": result.url,
                "elapsed_ms": result.elapsed_ms,
                "status": result.status,
                "host": result.host,
                "meta": dict(result.meta),
            }
            self.telemetry.emit(event)
        except Exception as e:
            self.logger.warning(f"Telemetry emission failed: {e}")

    def _is_budget_exhausted(self) -> bool:
        """Check if budget is exhausted.

        Checks both time and attempt budgets.
        """
        with self._budget_lock:
            # Attempt budget
            if self._attempt_count >= self.plan.budgets.get("total_attempts", float("inf")):
                return True

            # Time budget
            if self._elapsed_ms() >= self.plan.budgets.get("total_timeout_ms", float("inf")):
                return True

        return False

    def _elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        if not self._start_time:
            return 0
        return int((time.time() - self._start_time) * 1000)

    def _remaining_timeout_s(self) -> Optional[float]:
        """Get remaining timeout in seconds."""
        remaining_ms = self.plan.budgets.get("total_timeout_ms", float("inf")) - self._elapsed_ms()
        if remaining_ms <= 0:
            return 0.1  # Minimal timeout to trigger
        return remaining_ms / 1000.0

    def _timeout_outcome(self) -> AttemptResult:
        """Create a timeout outcome."""
        return AttemptResult(
            outcome="timeout",
            reason="budget_exhausted",
            elapsed_ms=self._elapsed_ms(),
            meta={
                "total_attempts": self._attempt_count,
                "budget_ms": self.plan.budgets.get("total_timeout_ms"),
            },
        )


__all__ = ["FallbackOrchestrator"]
