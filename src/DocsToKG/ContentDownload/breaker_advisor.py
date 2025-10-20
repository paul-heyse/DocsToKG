# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.breaker_advisor",
#   "purpose": "Telemetry-driven analysis and breaker tuning recommendations",
#   "sections": [
#     {
#       "id": "hostmetrics",
#       "name": "HostMetrics",
#       "anchor": "class-hostmetrics",
#       "kind": "class"
#     },
#     {
#       "id": "hostadvice",
#       "name": "HostAdvice",
#       "anchor": "class-hostadvice",
#       "kind": "class"
#     },
#     {
#       "id": "breakeradvisor",
#       "name": "BreakerAdvisor",
#       "anchor": "class-breakeradvisor",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Telemetry-driven circuit breaker advisor.

This module analyzes per-request and breaker transition telemetry to detect noisy hosts
and recommend safe, bounded adjustments to breaker and rate-limiter policies.

Typical Usage:
    advisor = BreakerAdvisor("telemetry.sqlite", window_s=600)
    metrics = advisor.read_metrics()  # Aggregate over sliding window
    advice = advisor.advise(metrics)  # Produce recommendations

    for host, adv in advice.items():
        if adv.suggest_fail_max:
            print(f"{host}: lower fail_max to {adv.suggest_fail_max}")
        if adv.suggest_reset_timeout_s:
            print(f"{host}: increase reset_timeout to {adv.suggest_reset_timeout_s}s")
"""

from __future__ import annotations

import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class HostMetrics:
    """Aggregated metrics for a single host over a sliding window."""

    host: str
    window_s: int
    calls_total: int
    calls_cache_hits: int
    calls_net: int  # non-cache network calls
    e429: int  # 429 Too Many Requests
    e5xx: int  # 5xx errors (all)
    e503: int  # 503 Service Unavailable specifically
    timeouts: int  # network timeouts/exceptions
    retry_after_samples: List[float] = field(default_factory=list)
    open_events: int = 0
    open_durations_s: List[float] = field(default_factory=list)
    half_open_success_trials: int = 0
    half_open_fail_trials: int = 0
    max_consecutive_failures: int = 0


@dataclass
class HostAdvice:
    """Tuning recommendations for a single host."""

    host: str
    # Circuit breaker knobs
    suggest_fail_max: Optional[int] = None
    suggest_reset_timeout_s: Optional[int] = None
    suggest_success_threshold: Optional[int] = None
    suggest_trial_calls_metadata: Optional[int] = None
    suggest_trial_calls_artifact: Optional[int] = None
    # Rate limiting knobs
    suggest_metadata_rps_multiplier: Optional[float] = None
    suggest_artifact_rps_multiplier: Optional[float] = None
    # Reasoning snippets
    reasons: List[str] = field(default_factory=list)


class BreakerAdvisor:
    """Analyzes telemetry and produces safe, bounded breaker/rate-limiter tuning advice."""

    # Safety bounds for suggested adjustments
    DEFAULTS = {
        "min_fail_max": 2,
        "max_fail_max": 10,
        "min_reset_timeout": 15,
        "max_reset_timeout": 600,
        "min_success_threshold": 1,
        "max_success_threshold": 3,
        "max_rps_multiplier_change": 0.25,  # ±25% per tick
    }

    def __init__(self, db_path: str | Path, window_s: int = 600) -> None:
        """Initialize advisor for a telemetry database.

        Parameters
        ----------
        db_path : str | Path
            Path to telemetry SQLite database
        window_s : int
            Sliding window in seconds for aggregation (default 600)
        """
        self.db_path = str(db_path)
        self.window_s = window_s

    def _conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path, timeout=2.0)

    def read_metrics(self, now: Optional[float] = None) -> Dict[str, HostMetrics]:
        """Read and aggregate metrics from telemetry in a sliding window.

        Returns
        -------
        dict[str, HostMetrics]
            Metrics keyed by hostname
        """
        now = now or time.time()
        since = now - self.window_s
        metrics: Dict[str, HostMetrics] = {}

        try:
            with self._conn() as cx:
                # Per-request HTTP events
                try:
                    rows = cx.execute(
                        """
                        SELECT host,
                               COUNT(*) FILTER (WHERE from_cache=1) as cache_hits,
                               COUNT(*) as calls_total,
                               COUNT(*) FILTER (WHERE status=429) as e429,
                               COUNT(*) FILTER (WHERE status BETWEEN 500 AND 599) as e5xx,
                               COUNT(*) FILTER (WHERE status=503) as e503
                        FROM http_events
                        WHERE ts >= ? AND (role='metadata' OR role='landing' OR role='artifact')
                        GROUP BY host
                        """,
                        (since,),
                    ).fetchall()

                    for host, cache_hits, calls_total, e429, e5xx, e503 in rows:
                        metrics[host] = HostMetrics(
                            host=host,
                            window_s=self.window_s,
                            calls_total=calls_total or 0,
                            calls_cache_hits=cache_hits or 0,
                            calls_net=(calls_total or 0) - (cache_hits or 0),
                            e429=e429 or 0,
                            e5xx=e5xx or 0,
                            e503=e503 or 0,
                            timeouts=0,
                        )
                except (sqlite3.OperationalError, sqlite3.DatabaseError):
                    # Table may not exist yet; skip gracefully
                    pass

                # Breaker transitions
                try:
                    trans = cx.execute(
                        """
                        SELECT host, ts, old_state, new_state, reset_timeout_s
                        FROM breaker_transitions WHERE ts >= ?
                        ORDER BY ts
                        """,
                        (since,),
                    ).fetchall()

                    open_started: Dict[str, float] = {}
                    for host, ts, old_s, new_s, reset_s in trans:
                        if host not in metrics:
                            metrics[host] = HostMetrics(host, self.window_s, 0, 0, 0, 0, 0, 0, 0)

                        if new_s and new_s.endswith("OPEN"):
                            open_started[host] = ts
                            metrics[host].open_events += 1
                        elif old_s and old_s.endswith("OPEN"):
                            start = open_started.pop(host, None)
                            if start:
                                metrics[host].open_durations_s.append(max(0.0, ts - start))
                except (sqlite3.OperationalError, sqlite3.DatabaseError):
                    # Table may not exist yet; skip gracefully
                    pass

                # Half-open trial outcomes
                try:
                    half = cx.execute(
                        """
                        SELECT host,
                               SUM(CASE WHEN breaker_recorded='success' THEN 1 ELSE 0 END) as ok,
                               SUM(CASE WHEN breaker_recorded='failure' THEN 1 ELSE 0 END) as ko
                        FROM http_events
                        WHERE ts >= ? AND breaker_host_state LIKE '%half_open%'
                        GROUP BY host
                        """,
                        (since,),
                    ).fetchall()

                    for host, ok, ko in half:
                        if host not in metrics:
                            metrics[host] = HostMetrics(host, self.window_s, 0, 0, 0, 0, 0, 0, 0)
                        metrics[host].half_open_success_trials += ok or 0
                        metrics[host].half_open_fail_trials += ko or 0
                except (sqlite3.OperationalError, sqlite3.DatabaseError):
                    # Table may not exist yet; skip gracefully
                    pass

        except Exception:
            # Database unavailable or schema mismatch; return empty metrics
            pass

        return metrics

    def advise(self, metrics: Dict[str, HostMetrics]) -> Dict[str, HostAdvice]:
        """Produce tuning advice based on metrics.

        Heuristics:
        - High 429 ratio (≥5%) → reduce metadata RPS 20%
        - Retry-After samples or open durations → estimate reset_timeout
        - Multiple opens in window → lower fail_max to 3
        - Half-open failures ≥50% → raise success_threshold to 2, limit trial_calls to 1

        Returns
        -------
        dict[str, HostAdvice]
            Recommendations keyed by hostname
        """
        advice: Dict[str, HostAdvice] = {}

        for host, h in metrics.items():
            a = HostAdvice(host=host, reasons=[])

            if h.calls_net <= 0:
                advice[host] = a
                continue

            # (1) 429 handling: prefer rate-limiter changes over breaker tuning
            r429 = h.e429 / max(1, h.calls_net)
            if r429 >= 0.05:  # 5%+ 429s
                a.suggest_metadata_rps_multiplier = 0.80  # -20%
                a.reasons.append(f"High 429 ratio {r429:.1%}: reduce metadata RPS 20%")

            # (2) Estimate unhealthy cooldown period
            est_cool_s: Optional[int] = None
            if h.retry_after_samples:
                est_cool_s = int(min(900, max(15, statistics.median(h.retry_after_samples))))
            elif h.open_durations_s:
                est_cool_s = int(min(600, max(15, statistics.median(h.open_durations_s))))
            if est_cool_s:
                a.suggest_reset_timeout_s = est_cool_s
                a.reasons.append(
                    f"Reset timeout → ~{est_cool_s}s (based on Retry-After/open durations)"
                )

            # (3) fail_max based on open frequency
            if h.open_events >= 3:  # multiple opens in window
                a.suggest_fail_max = 3
                a.reasons.append("Multiple breaker opens observed: suggest fail_max=3")

            # (4) half-open outcomes → success_threshold & trial_calls
            total_trials = h.half_open_success_trials + h.half_open_fail_trials
            if total_trials >= 2 and (h.half_open_fail_trials / total_trials) >= 0.5:
                a.suggest_success_threshold = 2
                a.suggest_trial_calls_metadata = 1
                a.reasons.append(
                    "Half-open failures ≥50%: raise success_threshold to 2, trial_calls=1"
                )

            advice[host] = a

        return advice
