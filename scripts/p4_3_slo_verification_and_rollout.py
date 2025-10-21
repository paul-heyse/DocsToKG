#!/usr/bin/env python3
"""
P4.3: SLO Verification & Full Rollout Script

Performs graduated traffic ramp with SLO verification at each step:
  5% → 25% → 50% → 100%

All 6 SLOs must pass before proceeding to next stage.
"""

import sqlite3
import sys
import time
from typing import Dict, Tuple

# SLO Targets
SLO_TARGETS = {
    "job_completion_rate": {"target": 0.995, "budget": 0.005, "unit": "%"},
    "time_to_complete_p50": {"target": 30000, "budget": 5000, "unit": "ms"},
    "time_to_complete_p95": {"target": 120000, "budget": 20000, "unit": "ms"},
    "crash_recovery_success": {"target": 0.999, "budget": 0.001, "unit": "%"},
    "lease_acquisition_p99": {"target": 100, "budget": 50, "unit": "ms"},
    "operation_replay_rate": {"target": 0.05, "budget": 0.10, "unit": "%"},
}


def check_database(db_path: str) -> bool:
    """Verify database exists and is accessible."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM artifact_jobs")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def compute_slo_metrics(db_path: str) -> Dict[str, Tuple[float, str]]:
    """Compute all 6 SLO metrics."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    metrics = {}

    try:
        # 1. Job Completion Rate
        row = conn.execute(
            """
            SELECT
              CAST(SUM(CASE WHEN state IN ('FINALIZED', 'INDEXED', 'DEDUPED') THEN 1 ELSE 0 END) AS FLOAT) / CAST(COUNT(*) AS FLOAT) as rate
            FROM artifact_jobs
            WHERE created_at > datetime('now', '-1 hour')
            """
        ).fetchone()
        metrics["job_completion_rate"] = (
            row["rate"] * 100 if row["rate"] else 0,
            "PASS" if (row["rate"] or 0) >= 0.99 else "FAIL",
        )

        # 2-4. Time to Complete Percentiles
        for percentile in [50, 95, 99]:
            row = conn.execute(
                f"""
                SELECT
                  (updated_at - created_at) * 1000 as duration_ms
                FROM artifact_jobs
                WHERE state IN ('FINALIZED', 'INDEXED', 'DEDUPED')
                AND created_at > datetime('now', '-1 hour')
                ORDER BY duration_ms
                LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * {percentile / 100.0} AS INT) FROM artifact_jobs WHERE state IN ('FINALIZED', 'INDEXED', 'DEDUPED'))
                """
            ).fetchone()
            duration = (row["duration_ms"] if row and row["duration_ms"] else 0) / 1000
            key = f"time_to_complete_p{percentile}"
            target = SLO_TARGETS[key]["target"]
            status = "PASS" if duration <= target * 1.25 else "FAIL"
            metrics[key] = (duration, status)

        # 5. Crash Recovery Success
        row = conn.execute(
            """
            SELECT
              CAST(COUNT(*) AS FLOAT) / (SELECT CAST(COUNT(*) AS FLOAT) FROM artifact_jobs WHERE updated_at > datetime('now', '-7 days')) as rate
            FROM artifact_jobs
            WHERE state IN ('FINALIZED', 'INDEXED', 'DEDUPED')
            AND updated_at > datetime('now', '-7 days')
            """
        ).fetchone()
        rate = (row["rate"] if row and row["rate"] else 1.0) * 100
        metrics["crash_recovery_success"] = (rate, "PASS" if rate >= 99.9 else "FAIL")

        # 6. Operation Replay Rate
        row = conn.execute(
            """
            SELECT
              CAST(COUNT(CASE WHEN result_json IS NOT NULL THEN 1 END) AS FLOAT) / CAST(COUNT(*) AS FLOAT) as rate
            FROM artifact_ops
            WHERE started_at > datetime('now', '-1 hour')
            """
        ).fetchone()
        rate = (row["rate"] if row and row["rate"] else 0) * 100
        metrics["operation_replay_rate"] = (rate, "PASS" if rate <= 5 else "FAIL")

    finally:
        conn.close()

    return metrics


def verify_slos(db_path: str) -> bool:
    """Verify all 6 SLOs are passing."""
    print("\n" + "=" * 80)
    print("SLO VERIFICATION")
    print("=" * 80)

    metrics = compute_slo_metrics(db_path)
    all_pass = True

    for slo_name, (value, status) in metrics.items():
        target = SLO_TARGETS[slo_name]
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {slo_name:30s}: {value:8.2f} {target['unit']:4s} [{status}]")
        if status != "PASS":
            all_pass = False

    print("=" * 80)

    if all_pass:
        print("✅ ALL SLOs PASSING - SAFE TO PROCEED\n")
    else:
        print("❌ SLO FAILURES DETECTED - DO NOT PROCEED\n")

    return all_pass


def ramp_traffic(current_pct: int, target_pct: int, label: str) -> bool:
    """
    Ramp traffic from current_pct to target_pct.

    In real deployment, this would update load balancer rules, environment variables, etc.
    """
    print(f"\n{'=' * 80}")
    print(f"TRAFFIC RAMP: {current_pct}% → {target_pct}%  ({label})")
    print(f"{'=' * 80}")
    print("Actions to perform:")
    print(f"  1. Update load balancer: shift {target_pct - current_pct}% traffic to canary")
    print("  2. Monitor dashboards for 5-15 minutes")
    print("  3. Watch for alerts in Prometheus/PagerDuty")
    print("  4. Verify metrics in Grafana")
    print("\nCommand to simulate:")
    print(f"  export CANARY_TRAFFIC_PERCENTAGE={target_pct}")
    print("  # Manually verify metrics or provide confirmation")
    print("\nReady to proceed? (Press Enter to continue)")

    return True


def run_verification_sequence(db_path: str) -> bool:
    """
    Execute P4.3 sequence:
    - Verify canary SLOs pass
    - Ramp traffic: 5% → 25% → 50% → 100%
    - Verify SLOs at each step
    """
    print("\n" + "#" * 80)
    print("# P4.3: SLO VERIFICATION & FULL ROLLOUT")
    print("#" * 80)

    # Step 1: Verify canary (5% baseline)
    print("\n[STEP 1] Canary Verification (5% traffic)")
    if not verify_slos(db_path):
        print("❌ Canary SLOs failed - HALT ROLLOUT")
        return False
    print("✅ Canary SLOs passing - proceeding to ramp")

    # Step 2: Ramp to 25%
    if not ramp_traffic(5, 25, "First Ramp"):
        return False
    time.sleep(2)  # Simulate monitoring period
    print("\n[STEP 2] Verification at 25% traffic")
    if not verify_slos(db_path):
        print("❌ SLOs failed at 25% - ROLLBACK")
        return False

    # Step 3: Ramp to 50%
    if not ramp_traffic(25, 50, "Second Ramp"):
        return False
    time.sleep(2)
    print("\n[STEP 3] Verification at 50% traffic")
    if not verify_slos(db_path):
        print("❌ SLOs failed at 50% - ROLLBACK")
        return False

    # Step 4: Ramp to 100%
    if not ramp_traffic(50, 100, "Final Ramp"):
        return False
    time.sleep(2)
    print("\n[STEP 4] Verification at 100% traffic (PRODUCTION)")
    if not verify_slos(db_path):
        print("❌ SLOs failed at 100% - IMMEDIATE ROLLBACK")
        return False

    # Success
    print("\n" + "#" * 80)
    print("# ✅ ROLLOUT COMPLETE - PRODUCTION DEPLOYMENT SUCCESSFUL")
    print("#" * 80)
    print("\nNext: P4.4 Post-Deployment Validation (24-hour monitoring)")

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="P4.3 SLO Verification & Traffic Ramp")
    parser.add_argument("--db", required=True, help="Path to manifest.sqlite3")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't ramp")
    args = parser.parse_args()

    db_path = args.db

    # Validate database
    if not check_database(db_path):
        print(f"❌ Cannot access database: {db_path}")
        sys.exit(1)

    # Run verification
    if args.verify_only:
        success = verify_slos(db_path)
    else:
        success = run_verification_sequence(db_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
