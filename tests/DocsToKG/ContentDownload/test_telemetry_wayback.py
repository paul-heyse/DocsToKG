from __future__ import annotations

import pytest

from DocsToKG.ContentDownload.telemetry_wayback_legacy import WaybackTelemetry


@pytest.fixture
def telemetry() -> WaybackTelemetry:
    # Sample at most two events per attempt to exercise the budgeting logic.
    return WaybackTelemetry(cdx_sample_budget=2)


def test_candidate_counters_reset_each_attempt(telemetry: WaybackTelemetry) -> None:
    first_attempt = telemetry.start_attempt("attempt-1")
    events = [telemetry.emit_candidate(first_attempt) for _ in range(3)]

    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert [event["is_cdx_sample"] for event in events] == [True, True, False]

    second_attempt = telemetry.start_attempt("attempt-2")
    follow_up = telemetry.emit_candidate(second_attempt)

    assert follow_up["sequence"] == 1
    # Sampling budget must be replenished for the new attempt.
    assert follow_up["is_cdx_sample"] is True


def test_discovery_counters_reset_each_attempt(telemetry: WaybackTelemetry) -> None:
    first_attempt = telemetry.start_attempt("attempt-1")
    events = [telemetry.emit_discovery(first_attempt) for _ in range(2)]

    assert [event["sequence"] for event in events] == [1, 2]

    second_attempt = telemetry.start_attempt("attempt-2")
    follow_up = telemetry.emit_discovery(second_attempt)

    assert follow_up["sequence"] == 1


def test_sampling_budget_shared_within_attempt(telemetry: WaybackTelemetry) -> None:
    attempt = telemetry.start_attempt("attempt-1")

    candidate_event = telemetry.emit_candidate(attempt)
    discovery_event = telemetry.emit_discovery(attempt)
    overflow_event = telemetry.emit_candidate(attempt)

    assert candidate_event["is_cdx_sample"] is True
    assert discovery_event["is_cdx_sample"] is True
    # Budget exhausted after two events; the third should not be sampled.
    assert overflow_event["is_cdx_sample"] is False

    # Starting a new attempt replenishes the shared sampling budget.
    new_attempt = telemetry.start_attempt("attempt-2")
    assert telemetry.emit_candidate(new_attempt)["is_cdx_sample"] is True
