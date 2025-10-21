"""
Telemetry Record Types - Extended Data Contracts

Defines the rich attempt record used internally by telemetry for logging
and manifest recording.
"""

from .records import PipelineResult, TelemetryAttemptRecord

__all__ = [
    "TelemetryAttemptRecord",
    "PipelineResult",
]
