"""Fallback orchestration: deterministic, tiered PDF resolution.

This module implements the core orchestration logic for the fallback strategy:
- Tiered execution with per-tier parallelism
- Budget enforcement (time, attempts, concurrency)
- Health gates (breaker, offline, rate limiter awareness)
- Automatic cancellation when success is found
- Full telemetry emission

The orchestrator uses threading for parallelism within tiers and a queue for
result collection. Execution stops immediately upon success.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

from DocsToKG.ContentDownload.fallback.types import (
    AttemptResult,
    FallbackPlan,
)

# Type alias for adapter functions
AttemptFn = Callable[[Any, Dict[str, Any]], AttemptResult]


class FallbackOrchestrator:
    """Orchestrates deterministic, budgeted PDF resolution across tiers.

    The orchestrator executes tiers sequentially. Within each tier, sources run
    in parallel up to the tier's `parallel` limit. Execution stops immediately
    when one attempt succeeds. Budget constraints (time, attempts, concurrency)
    are enforced throughout.

    Attributes:
        plan: FallbackPlan with configuration
        breaker: BreakerRegistry for health checking
        rate: Rate limiter for politeness
        head_client: Cached HTTP client for metadata
        raw_client: Raw HTTP client for artifacts
        telemetry: Telemetry sink for structured events
        logger: Logger instance

    Example:
        ```python
        orchestrator = FallbackOrchestrator(
            plan=plan,
            breaker=breaker_registry,
            rate=rate_limiter,
            head_client=cached_client,
            raw_client=raw_client,
            telemetry=telemetry_sink,
            logger=logger
        )

        result = orchestrator.resolve_pdf(
            context={"work_id": "123", "artifact_id": "abc", "doi": "..."},
            adapters={
                "unpaywall_pdf": adapter_unpaywall,
                "arxiv_pdf": adapter_arxiv,
                # ... etc
            }
        )

        if result.outcome == "success":
            print(f"Found: {result.url}")
        else:
            print(f"Failed: {result.reason}")
        ```
    """

    def __init__(
        self,
        *,
        plan: FallbackPlan,
        breaker: Any,
        rate: Any,
        head_client: Any,
        raw_client: Any,
        telemetry: Any,
        logger: logging.Logger,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            plan: Complete FallbackPlan with budgets and tiers
            breaker: BreakerRegistry instance
            rate: Rate limiter manager
            head_client: Cached HTTP client for metadata
            raw_client: Raw HTTP client for artifacts
            telemetry: Telemetry sink
            logger: Logger instance
        """
        self.plan = plan
        self.breaker = breaker
        self.rate = rate
        self.head = head_client
        self.raw = raw_client
        self.tele = telemetry
        self.log = logger

    def resolve_pdf(
        self,
        *,
        context: Dict[str, Any],
        adapters: Dict[str, AttemptFn],
    ) -> AttemptResult:
        """Resolve a PDF URL using tiered fallback strategy.

        Executes tiers sequentially, parallelizing within each tier. Stops
        immediately upon success or when budgets are exhausted.

        Args:
            context: Request context (work_id, artifact_id, doi, etc.)
            adapters: Mapping of source_name to adapter function

        Returns:
            AttemptResult indicating success/failure and details

        Raises:
            KeyError: If a required adapter is missing
        """
        budgets = self.plan.budgets
        t0 = time.monotonic()
        attempts_used = 0

        # Cancellation flag (shared across threads)
        cancel_flag = threading.Event()

        # Result queue (thread-safe)
        result_queue: queue.Queue[AttemptResult] = queue.Queue()

        def run_attempt(source_name: str) -> None:
            """Run a single source attempt in a thread.

            Args:
                source_name: Name of the source to attempt
            """
            policy = self.plan.get_policy(source_name)
            try:
                # Check cancellation flag
                if cancel_flag.is_set():
                    res = AttemptResult(
                        outcome="skipped",
                        reason="cancelled",
                        elapsed_ms=0,
                    )
                    result_queue.put(res)
                    return

                # Check health gates
                gate_result = self._health_gate(source_name, context)
                if gate_result is not None:
                    result_queue.put(gate_result)
                    return

                # Execute adapter
                started = time.monotonic()
                res = adapters[source_name](policy, context)
                res_with_time = AttemptResult(
                    outcome=res.outcome,
                    reason=res.reason,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    url=res.url,
                    status=res.status,
                    host=res.host,
                    meta=res.meta,
                )
                result_queue.put(res_with_time)
            except KeyError as e:
                result_queue.put(
                    AttemptResult(
                        outcome="error",
                        reason="missing_adapter",
                        elapsed_ms=0,
                        meta={"msg": str(e)},
                    )
                )
            except Exception as e:  # pylint: disable=broad-except
                result_queue.put(
                    AttemptResult(
                        outcome="error",
                        reason="exception",
                        elapsed_ms=0,
                        meta={"msg": str(e)},
                    )
                )

        # ====================================================================
        # Tiered scheduling loop
        # ====================================================================
        for tier in self.plan.tiers:
            # Check if we've exhausted attempt budget
            if attempts_used >= budgets["total_attempts"]:
                self.log.info(
                    f"Stopping: exhausted attempt budget "
                    f"({attempts_used}/{budgets['total_attempts']})"
                )
                break

            # Check if we've exhausted time budget
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            if elapsed_ms >= budgets["total_timeout_ms"]:
                self.log.info(
                    f"Stopping: exhausted time budget "
                    f"({elapsed_ms}/{budgets['total_timeout_ms']}ms)"
                )
                break

            self.log.debug(f"Starting tier: {tier.name}")

            # Build list of sources to try
            names = list(tier.sources)
            inflight: list[threading.Thread] = []
            launched = 0

            # Launch sources up to tier parallelism limit
            while names and launched < tier.parallel and attempts_used < budgets["total_attempts"]:
                name = names.pop(0)
                th = threading.Thread(
                    target=run_attempt,
                    args=(name,),
                    daemon=True,
                    name=f"fallback-{name}",
                )
                th.start()
                inflight.append(th)
                launched += 1
                attempts_used += 1

            # Collect outcomes for this tier
            done = 0
            tier_results: list[AttemptResult] = []

            while done < launched:
                # Check time budget on each iteration
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                if elapsed_ms >= budgets["total_timeout_ms"]:
                    self.log.info("Time budget exhausted, cancelling tier")
                    cancel_flag.set()
                    break

                try:
                    # Wait with timeout to check budget periodically
                    res = result_queue.get(timeout=0.25)
                    done += 1
                    tier_results.append(res)

                    # Emit telemetry
                    self._emit_attempt_telemetry(tier.name, res, context)

                    # Check for success
                    if res.is_success:
                        self.log.info(f"Success in tier {tier.name}: {res.reason}")
                        cancel_flag.set()
                        # Clean up remaining threads
                        for th in inflight:
                            th.join(timeout=0.5)
                        # Emit summary
                        elapsed_ms = int((time.monotonic() - t0) * 1000)
                        summary_event = {
                            "outcome": "success",
                            "reason": res.reason,
                            "tier": tier.name,
                            "total_elapsed_ms": elapsed_ms,
                            "attempts_used": attempts_used,
                            "work_id": context.get("work_id"),
                            "artifact_id": context.get("artifact_id"),
                        }
                        if hasattr(self.tele, "log_fallback_summary"):
                            self.tele.log_fallback_summary(summary_event)
                        return res

                except queue.Empty:
                    # Timeout - check budget and continue waiting
                    continue

            # Join threads for this tier
            for th in inflight:
                th.join(timeout=0.5)

            # Log tier summary
            successes = sum(1 for r in tier_results if r.is_success)
            self.log.debug(f"Tier {tier.name} complete: {successes}/{launched} success")

        # ====================================================================
        # No success - return exhausted outcome
        # ====================================================================
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self.log.warning(f"No PDF found after {attempts_used} attempts in {elapsed_ms}ms")
        # Emit summary for exhausted outcome
        summary_event = {
            "outcome": "exhausted",
            "reason": "no_pdf_found",
            "total_elapsed_ms": elapsed_ms,
            "attempts_used": attempts_used,
            "tiers_attempted": len(self.plan.tiers),
            "work_id": context.get("work_id"),
            "artifact_id": context.get("artifact_id"),
        }
        if hasattr(self.tele, "log_fallback_summary"):
            self.tele.log_fallback_summary(summary_event)
        return AttemptResult(
            outcome="no_pdf",
            reason="exhausted",
            elapsed_ms=elapsed_ms,
            meta={
                "attempts": attempts_used,
                "tiers": len(self.plan.tiers),
            },
        )

    def _health_gate(
        self,
        source_name: str,
        context: Dict[str, Any],
    ) -> Optional[AttemptResult]:
        """Check health gates for a source attempt.

        Gates include:
        - Breaker status (skip if open)
        - Offline mode (skip artifact attempts)

        Args:
            source_name: Name of the source
            context: Request context

        Returns:
            AttemptResult (skipped) if gate blocks, None otherwise
        """
        # Check offline behavior
        if context.get("offline"):
            if self.plan.offline_behavior != "metadata_only" and self._is_artifact_source(
                source_name
            ):
                return AttemptResult(
                    outcome="skipped",
                    reason="offline_block",
                    elapsed_ms=0,
                    meta={"behavior": self.plan.offline_behavior},
                )

        # Breaker gate would be checked by adapter before network attempt
        # (adapters call breaker.allow() internally)
        return None

    def _is_artifact_source(self, source_name: str) -> bool:
        """Check if a source is for artifact (vs. metadata) attempts.

        Artifact sources: any that directly fetch PDFs (as opposed to
        metadata APIs). For now, conservative: mark none as metadata-only.

        Args:
            source_name: Name of the source

        Returns:
            True if artifact source, False if metadata-only
        """
        # In this simple implementation, all sources are artifact attempts
        # (they fetch actual PDFs or landing pages, not metadata APIs only).
        # Adapters like landing_scrape still download HTML, so they're
        # "artifact-tier" in this context.
        return True

    def _emit_attempt_telemetry(
        self,
        tier: str,
        result: AttemptResult,
        context: Dict[str, Any],
    ) -> None:
        """Emit structured telemetry for an attempt.

        Args:
            tier: Tier name
            result: AttemptResult
            context: Request context
        """
        event = {
            "event": "fallback_attempt",
            "tier": tier,
            "outcome": result.outcome,
            "reason": result.reason,
            "elapsed_ms": result.elapsed_ms,
            "host": result.host,
            "status": result.status,
            "work_id": context.get("work_id"),
            "artifact_id": context.get("artifact_id"),
        }

        # Add source from metadata if available
        if result.meta and "source" in result.meta:
            event["source"] = result.meta["source"]

        # Log via telemetry sink (assuming it has log_fallback_attempt)
        if hasattr(self.tele, "log_fallback_attempt"):
            self.tele.log_fallback_attempt(event)

        # Also log to standard logger
        self.log.debug(f"Fallback attempt: {event}")
