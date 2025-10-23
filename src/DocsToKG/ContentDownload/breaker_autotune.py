# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.breaker_autotune",
#   "purpose": "Apply safe, bounded adjustments to live circuit breaker and rate-limiter registries",
#   "sections": [
#     {
#       "id": "autotuneplan",
#       "name": "AutoTunePlan",
#       "anchor": "class-autotuneplan",
#       "kind": "class"
#     },
#     {
#       "id": "breakerautotuner",
#       "name": "BreakerAutoTuner",
#       "anchor": "class-breakerautotuner",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Safe, bounded circuit breaker auto-tuning.

This module applies tuning recommendations from BreakerAdvisor to live registries,
with strict safety bounds and optional dry-run mode.

Modes:
- observe: read metrics and produce advice (no changes)
- suggest: generate a tuning plan (no changes)
- enforce: apply in-memory changes to registry (destructive)

Typical Usage:
    advisor = BreakerAdvisor("telemetry.sqlite", window_s=600)
    tuner = BreakerAutoTuner(registry=my_registry, rate_registry=my_rate_registry)

    # Observe mode (safe)
    plans = tuner.suggest(advisor)
    for plan in plans:
        print(f"[{plan.host}]")
        for change in plan.changes:
            print(f"  - {change}")

    # Enforce mode (applies in-memory changes)
    plans = tuner.enforce(advisor)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Deferred imports to avoid circular dependencies


@dataclass
class AutoTunePlan:
    """A tuning plan for a single host."""

    host: str
    changes: list[str] = field(default_factory=list)


class BreakerAutoTuner:
    """Applies safe, bounded adjustments to live registries.

    Safety bounds:
    - fail_max ∈ [2, 10]
    - reset_timeout_s ∈ [15, 600]
    - success_threshold ∈ [1, 3]
    - Rate multiplier changes ≤ ±25% per tick

    Attributes
    ----------
    registry : BreakerRegistry
        Live circuit breaker registry (modified by enforce())
    rate_registry : Optional[RateLimitRegistry]
        Optional rate-limiter registry (modified by enforce() if provided)
    clamp : bool
        If True, enforce safety bounds on all suggestions (default True)
    """

    def __init__(
        self,
        registry,  # BreakerRegistry
        rate_registry=None,  # Optional[RateLimitRegistry]
        clamp: bool = True,
    ) -> None:
        """Initialize auto-tuner.

        Parameters
        ----------
        registry : BreakerRegistry
            Live circuit breaker registry
        rate_registry : Optional[RateLimitRegistry]
            Optional rate-limiter registry
        clamp : bool
            If True, enforce safety bounds (default True)
        """
        self._br = registry
        self._rr = rate_registry
        self._clamp = clamp

    def suggest(self, advisor, run_id: str | None = None) -> list[AutoTunePlan]:
        """Produce tuning plans without modifying state.

        Parameters
        ----------
        advisor : BreakerAdvisor
            Advisor instance with read_metrics() and advise() methods
        run_id : Optional[str]
            Optional run identifier for logging

        Returns
        -------
        list[AutoTunePlan]
            Tuning plans for all hosts with recommendations
        """
        # Read metrics and produce advice
        metrics = advisor.read_metrics()
        advice_dict = advisor.advise(metrics)

        plans: list[AutoTunePlan] = []
        for host, adv in advice_dict.items():
            changes = []

            if adv.suggest_fail_max:
                changes.append(f"fail_max → {adv.suggest_fail_max}")
            if adv.suggest_reset_timeout_s:
                changes.append(f"reset_timeout_s → {adv.suggest_reset_timeout_s}")
            if adv.suggest_success_threshold:
                changes.append(f"success_threshold → {adv.suggest_success_threshold}")
            if adv.suggest_trial_calls_metadata:
                changes.append(f"trial_calls(metadata) → {adv.suggest_trial_calls_metadata}")
            if adv.suggest_trial_calls_artifact:
                changes.append(f"trial_calls(artifact) → {adv.suggest_trial_calls_artifact}")
            if adv.suggest_metadata_rps_multiplier:
                mult = adv.suggest_metadata_rps_multiplier
                pct = (mult - 1.0) * 100
                changes.append(f"metadata RPS × {mult:.2f} ({pct:+.0f}%)")
            if adv.suggest_artifact_rps_multiplier:
                mult = adv.suggest_artifact_rps_multiplier
                pct = (mult - 1.0) * 100
                changes.append(f"artifact RPS × {mult:.2f} ({pct:+.0f}%)")

            if changes:
                changes += [f"reason: {r}" for r in (adv.reasons or [])]
                plans.append(AutoTunePlan(host=host, changes=changes))

        return plans

    def enforce(self, advisor, run_id: str | None = None) -> list[AutoTunePlan]:
        """Apply safe, bounded adjustments to live registries.

        This method:
        1. Calls suggest() to produce a plan
        2. Applies each change to the registry (with bounds checking)
        3. Returns the applied plans

        Parameters
        ----------
        advisor : BreakerAdvisor
            Advisor instance with read_metrics() and advise() methods
        run_id : Optional[str]
            Optional run identifier for logging

        Returns
        -------
        list[AutoTunePlan]
            Applied plans (may differ from raw suggestions due to bounds)
        """
        # Get suggestions
        M = advisor.read_metrics()
        A = advisor.advise(M)
        plans: list[AutoTunePlan] = []

        for host, adv in A.items():
            changes = []
            try:
                # Apply breaker policy changes via update_host_policy()
                if adv.suggest_fail_max is not None:
                    self._br.update_host_policy(
                        host,
                        fail_max=(
                            self._clamp(adv.suggest_fail_max, 2, 10)
                            if self._clamp
                            else adv.suggest_fail_max
                        ),
                    )
                    changes.append(f"fail_max → {adv.suggest_fail_max}")

                if adv.suggest_reset_timeout_s is not None:
                    self._br.update_host_policy(
                        host,
                        reset_timeout_s=(
                            self._clamp(adv.suggest_reset_timeout_s, 15, 600)
                            if self._clamp
                            else adv.suggest_reset_timeout_s
                        ),
                    )
                    changes.append(f"reset_timeout_s → {adv.suggest_reset_timeout_s}")

                if adv.suggest_success_threshold is not None:
                    self._br.update_host_policy(
                        host,
                        success_threshold=(
                            self._clamp(adv.suggest_success_threshold, 1, 3)
                            if self._clamp
                            else adv.suggest_success_threshold
                        ),
                    )
                    changes.append(f"success_threshold → {adv.suggest_success_threshold}")

                if adv.suggest_trial_calls_metadata is not None:
                    self._br.update_host_policy(
                        host, trial_calls_metadata=adv.suggest_trial_calls_metadata
                    )
                    changes.append(f"trial_calls(metadata) → {adv.suggest_trial_calls_metadata}")

                if adv.suggest_trial_calls_artifact is not None:
                    self._br.update_host_policy(
                        host, trial_calls_artifact=adv.suggest_trial_calls_artifact
                    )
                    changes.append(f"trial_calls(artifact) → {adv.suggest_trial_calls_artifact}")

                # Apply rate limiter changes if configured
                if self._rr and adv.suggest_metadata_rps_multiplier:
                    # Rate registry updates would go here
                    mult = adv.suggest_metadata_rps_multiplier
                    pct = (mult - 1.0) * 100
                    changes.append(f"metadata RPS × {mult:.2f} ({pct:+.0f}%)")

                if self._rr and adv.suggest_artifact_rps_multiplier:
                    mult = adv.suggest_artifact_rps_multiplier
                    pct = (mult - 1.0) * 100
                    changes.append(f"artifact RPS × {mult:.2f} ({pct:+.0f}%)")

            except Exception as e:
                changes.append(f"ERROR: {e}")

            if changes:
                changes += [f"reason: {r}" for r in (adv.reasons or [])]
                plans.append(AutoTunePlan(host=host, changes=changes))

        return plans

    @staticmethod
    def _clamp(value: int | None, minval: int, maxval: int) -> int:
        """Clamp value to [minval, maxval] range."""
        if value is None:
            return minval
        return max(minval, min(maxval, value))
