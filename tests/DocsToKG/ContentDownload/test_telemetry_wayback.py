from __future__ import annotations

from DocsToKG.ContentDownload.telemetry_wayback import (
    AttemptMeasurement,
    TelemetryWayback,
)


class FakeMonotonic:
    def __init__(self, start: float = 0.0) -> None:
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, delta: float) -> None:
        self.current += delta


def test_attempt_context_uses_injected_monotonic_for_elapsed() -> None:
    fake_monotonic = FakeMonotonic()
    telemetry = TelemetryWayback(monotonic_fn=fake_monotonic)

    with telemetry.attempt("attempt-1") as attempt:
        fake_monotonic.advance(1.5)
        assert attempt.elapsed_ms_so_far() == 1500
        fake_monotonic.advance(0.5)

    assert len(telemetry.attempts) == 1
    measurement = telemetry.attempts[0]
    assert isinstance(measurement, AttemptMeasurement)
    assert measurement.start_monotonic == 0.0
    assert measurement.end_monotonic == 2.0
    assert measurement.elapsed_ms == 2000


def test_attempt_context_finish_reuses_monotonic_clock() -> None:
    fake_monotonic = FakeMonotonic(10.0)
    telemetry = TelemetryWayback(monotonic_fn=fake_monotonic)
    attempt = telemetry.attempt("attempt-2", resolver="wayback")

    fake_monotonic.advance(0.75)
    measurement = attempt.finish(outcome="success")

    assert measurement.metadata["resolver"] == "wayback"
    assert measurement.metadata["outcome"] == "success"
    assert measurement.start_monotonic == 10.0
    assert measurement.end_monotonic == 10.75
    assert measurement.elapsed_ms == 750

    # Subsequent calls should not mutate measurements and should not call the
    # monotonic function a second time.
    fake_monotonic.advance(25.0)
    same_measurement = attempt.finish()
    assert same_measurement is measurement
    assert same_measurement.end_monotonic == 10.75
    assert same_measurement.elapsed_ms == 750
