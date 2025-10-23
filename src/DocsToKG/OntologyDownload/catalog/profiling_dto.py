# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.profiling_dto",
#   "purpose": "Data Transfer Objects for query profiling.",
#   "sections": [
#     {
#       "id": "planstep",
#       "name": "PlanStep",
#       "anchor": "class-planstep",
#       "kind": "class"
#     },
#     {
#       "id": "queryprofile",
#       "name": "QueryProfile",
#       "anchor": "class-queryprofile",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Data Transfer Objects for query profiling.

Represents query performance metrics and optimization information from
EXPLAIN ANALYZE and cost estimation.

NAVMAP:
  - QueryProfile: Complete query profile with plan and metrics
  - PlanStep: Single step in query execution plan
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanStep:
    """Single step in query execution plan.

    Attributes:
        step_name: Name of the plan step (e.g., "Seq Scan", "Hash Join")
        rows_estimated: Estimated number of rows
        rows_actual: Actual number of rows
        startup_cost: Cost to produce first row
        total_cost: Total cost for all rows
        duration_ms: Execution time in milliseconds
    """

    step_name: str
    rows_estimated: int
    rows_actual: int
    startup_cost: float
    total_cost: float
    duration_ms: float


@dataclass(frozen=True)
class QueryProfile:
    """Complete query profile from EXPLAIN ANALYZE.

    Attributes:
        query: The SQL query text
        plan_steps: List of execution plan steps
        total_rows: Total rows processed
        total_duration_ms: Total query execution time
        total_cost: Total query cost estimate
        planning_time_ms: Query planning time
        effective_cache_size_bytes: Estimated cache size
        suggestions: List of optimization suggestions
    """

    query: str
    plan_steps: list[PlanStep]
    total_rows: int
    total_duration_ms: float
    total_cost: float
    planning_time_ms: float
    effective_cache_size_bytes: int
    suggestions: list[str]

    @property
    def is_expensive(self) -> bool:
        """Check if query is expensive (>1000 cost units)."""
        return self.total_cost > 1000.0

    @property
    def is_slow(self) -> bool:
        """Check if query is slow (>100ms)."""
        return self.total_duration_ms > 100.0

    @property
    def critical_steps(self) -> list[PlanStep]:
        """Get steps with high cost or duration."""
        return [
            step for step in self.plan_steps if step.total_cost > 500.0 or step.duration_ms > 50.0
        ]

    @property
    def efficiency_ratio(self) -> float:
        """Calculate ratio of estimated to actual rows (efficiency)."""
        if self.total_rows == 0:
            return 1.0
        return self.plan_steps[0].rows_estimated / max(self.total_rows, 1)
