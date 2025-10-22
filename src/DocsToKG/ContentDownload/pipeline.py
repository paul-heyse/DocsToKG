"""
Pipeline Orchestration - Canonical Implementation

Orchestrates the complete download flow:
1. Iterate resolvers in priority order
2. Per resolver, collect DownloadPlans from ResolverResult
3. Try each plan: prepare → stream → finalize
4. Catch exceptions, convert to outcomes
5. Stop on first success or collect all failures
6. Record manifest entry

Design:
- Pure coordination (no side effects beyond resolution/download)
- Exception handling: SkipDownload/DownloadError → DownloadOutcome
- Telemetry: Structured logging at each stage
- Idempotent: Can retry on any failure
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Optional, Sequence

from DocsToKG.ContentDownload.api import (
    DownloadOutcome,
    DownloadPlan,
    ResolverResult,
)
from DocsToKG.ContentDownload.api.exceptions import DownloadError, SkipDownload
from DocsToKG.ContentDownload.download_execution import (
    finalize_candidate_download,
    prepare_candidate_download,
    stream_candidate_payload,
)

LOGGER = logging.getLogger(__name__)


class ResolverPipeline:
    """
    Orchestrates resolver-driven download pipeline.

    Coordinates:
    - Multiple resolvers (tried in order)
    - Multiple plans per resolver (tried in order)
    - Three-stage download (prepare → stream → finalize)
    - Exception handling and outcome recording
    """

    def __init__(
        self,
        resolvers: Sequence[Any],
        session: Any,
        telemetry: Optional[Any] = None,
        run_id: Optional[str] = None,
        client_map: Optional[dict[str, Any]] = None,
        **policy_knobs: Any,
        **policy_overrides: Any,
    ):
        """
        Initialize pipeline.

        Args:
            resolvers: Sequence of Resolver instances (in priority order)
            session: HTTP session/client
            telemetry: Optional telemetry sink
            run_id: Optional run ID for correlation
            client_map: Optional dict mapping resolver_name → per-resolver HTTP client
            **policy_overrides: Additional policy knobs (download, robots, etc.)
        """
        self._resolvers = list(resolvers)
        self._session = session
        self._telemetry = telemetry
        self._run_id = run_id
        self._client_map = client_map or {}
        self._policy_knobs = policy_knobs
        self._policy_overrides = policy_overrides

    def run(self, artifact: Any, ctx: Any) -> DownloadOutcome:
        """
        Run pipeline for a single artifact.

        Algorithm:
        1. For each resolver in order:
           a. Call resolver.resolve(artifact, session, ctx, telemetry, run_id)
           b. Get ResolverResult with plans
           c. For each plan in plans:
              - try: prepare → stream → finalize
              - catch SkipDownload → DownloadOutcome(classification="skip")
              - catch DownloadError → DownloadOutcome(classification="error")
              - if success: return immediately (first win)
           d. If all plans skipped/failed: continue to next resolver

        2. If all resolvers exhausted: return error outcome

        Args:
            artifact: Work artifact to download
            ctx: Context with config, caches, state

        Returns:
            DownloadOutcome (success, skip, or error)
        """
        if ctx is None and self._policy_knobs:
            ctx = SimpleNamespace(**self._policy_knobs)

        outcomes = []

        # Try each resolver in order
        for resolver in self._resolvers:
            LOGGER.debug(f"Trying resolver '{resolver.name}' for artifact {artifact.work_id}")

            try:
                # Call resolver
                result: ResolverResult = resolver.resolve(
                    artifact, self._session, ctx, self._telemetry, self._run_id
                )

                # If resolver returned no plans, try next resolver
                if not result.plans:
                    LOGGER.debug(
                        f"Resolver '{resolver.name}' returned zero plans for "
                        f"artifact {artifact.work_id}"
                    )
                    continue

                # Try each plan from this resolver
                for plan in result.plans:
                    outcome = self._try_plan(plan, artifact, ctx)
                    outcomes.append(outcome)

                    # Stop on first success
                    if outcome.ok:
                        LOGGER.info(
                            f"Downloaded artifact {artifact.work_id} via "
                            f"resolver '{resolver.name}' "
                            f"(path: {outcome.path})"
                        )
                        return outcome

                    # Log skip/error but continue to next plan
                    LOGGER.debug(
                        f"Plan failed (url={plan.url}, reason={outcome.reason}), "
                        f"trying next plan..."
                    )

            except Exception as e:  # pylint: disable=broad-except
                LOGGER.warning(
                    f"Resolver '{resolver.name}' raised exception: {e}",
                    exc_info=True,
                )
                continue

        # All resolvers exhausted
        LOGGER.warning(
            f"All resolvers exhausted for artifact {artifact.work_id}. "
            f"Tried {len(outcomes)} plans, all failed/skipped."
        )

        # Return final error outcome
        return DownloadOutcome(
            ok=False,
            classification="error",
            path=None,
            reason="download-error",
            meta={"attempted": len(outcomes), "outcomes": [o.classification for o in outcomes]},
        )

    def _try_plan(self, plan: DownloadPlan, artifact: Any, ctx: Any) -> DownloadOutcome:
        """
        Try a single plan: prepare → stream → finalize.

        Catches exceptions and converts to outcomes.

        Args:
            plan: DownloadPlan to execute
            artifact: Work artifact (for context)
            ctx: Context with config, caches, state

        Returns:
            DownloadOutcome (success, skip, or error)
        """
        # Select per-resolver client (fallback to shared session if not available)
        client = self._client_map.get(plan.resolver_name, self._session)

        try:
            # Stage 1: Prepare (preflight validation)
            try:
                adj_plan = prepare_candidate_download(
                    plan,
                    session=client,
                    ctx=ctx,
                    telemetry=self._telemetry,
                    run_id=self._run_id,
                )
            except SkipDownload as e:
                LOGGER.debug(f"Skipped plan (url={plan.url}, reason={e.reason}): {e}")
                return DownloadOutcome(
                    ok=False,
                    classification="skip",
                    path=None,
                    reason=e.reason,
                    meta={"error": str(e)},
                )
            except DownloadError as e:
                LOGGER.debug(f"Preflight error (url={plan.url}, reason={e.reason}): {e}")
                return DownloadOutcome(
                    ok=False,
                    classification="error",
                    path=None,
                    reason=e.reason,
                    meta={"error": str(e)},
                )

            # Stage 2: Stream (HTTP GET + write to temp)
            try:
                stream = stream_candidate_payload(
                    adj_plan,
                    session=client,
                    timeout_s=getattr(ctx, "timeout_s", None),
                    max_bytes=getattr(ctx, "max_bytes", None),
                    telemetry=self._telemetry,
                    run_id=self._run_id,
                )
            except SkipDownload as e:
                LOGGER.debug(f"Skipped stream (url={plan.url}, reason={e.reason}): {e}")
                return DownloadOutcome(
                    ok=False,
                    classification="skip",
                    path=None,
                    reason=e.reason,
                    meta={"error": str(e)},
                )
            except DownloadError as e:
                LOGGER.debug(f"Stream error (url={plan.url}, reason={e.reason}): {e}")
                return DownloadOutcome(
                    ok=False,
                    classification="error",
                    path=None,
                    reason=e.reason,
                    meta={"error": str(e)},
                )

            # Stage 3: Finalize (move to final, manifest record)
            try:
                storage_settings = getattr(ctx, "storage", None)
                if storage_settings is None:
                    config = getattr(ctx, "config", None)
                    if config is not None:
                        storage_settings = getattr(config, "storage", None)
                storage_root = getattr(ctx, "storage_root", None)
                outcome = finalize_candidate_download(
                    adj_plan,
                    stream,
                    final_path=getattr(artifact, "final_path", None),
                    storage_settings=storage_settings,
                    storage_root=storage_root,
                    telemetry=self._telemetry,
                    run_id=self._run_id,
                )
                LOGGER.info(
                    f"Successfully finalized download for "
                    f"artifact {artifact.work_id} (path: {outcome.path})"
                )
                return outcome
            except DownloadError as e:
                LOGGER.debug(f"Finalization error (url={plan.url}, reason={e.reason}): {e}")
                return DownloadOutcome(
                    ok=False,
                    classification="error",
                    path=None,
                    reason=e.reason,
                    meta={"error": str(e)},
                )

        except Exception as e:  # pylint: disable=broad-except
            LOGGER.error(
                f"Unexpected error in _try_plan (url={plan.url}): {e}",
                exc_info=True,
            )
            return DownloadOutcome(
                ok=False,
                classification="error",
                path=None,
                reason="download-error",
                meta={"error": str(e)},
            )


__all__ = [
    "ResolverPipeline",
]
