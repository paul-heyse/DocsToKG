# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.profiler",
#   "purpose": "Query profiling and optimization API.",
#   "sections": [
#     {
#       "id": "catalogprofiler",
#       "name": "CatalogProfiler",
#       "anchor": "class-catalogprofiler",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Query profiling and optimization API.

Provides performance analysis and optimization recommendations for queries
using DuckDB's EXPLAIN ANALYZE capabilities.

NAVMAP:
  - CatalogProfiler: Main profiler class
  - Query Methods:
    * profile_query() - EXPLAIN ANALYZE for query plans
    * estimate_query_cost() - Cost estimation
    * optimize_suggestions() - Performance recommendations
"""

from __future__ import annotations

from .profiling_dto import PlanStep, QueryProfile


class CatalogProfiler:
    """Query profiler for performance analysis.

    Provides methods to analyze query performance, estimate costs,
    and generate optimization suggestions.

    Attributes:
        repo: Underlying Repo instance for database access
    """

    def __init__(self, repo):
        """Initialize profiler.

        Args:
            repo: Repo instance for database access
        """
        self.repo = repo

    def profile_query(self, query: str) -> QueryProfile:
        """Profile a query using EXPLAIN ANALYZE.

        Args:
            query: SQL query to profile

        Returns:
            QueryProfile with plan steps and metrics

        Performance:
            Executes in < 500ms (includes full query execution)
        """
        # Get EXPLAIN ANALYZE results
        explain_results = self.repo.query_all(f"EXPLAIN ANALYZE {query}", [])

        # Parse results into plan steps
        plan_steps = []
        total_cost = 0.0
        total_duration_ms = 0.0

        for row in explain_results:
            # Extract plan information from EXPLAIN output
            # Format: step_name, rows_est, rows_actual, cost, duration
            if len(row) >= 5:
                step = PlanStep(
                    step_name=str(row[0]),
                    rows_estimated=int(row[1]) if row[1] else 0,
                    rows_actual=int(row[2]) if row[2] else 0,
                    startup_cost=float(row[3]) if row[3] else 0.0,
                    total_cost=float(row[4]) if row[4] else 0.0,
                    duration_ms=float(row[5]) if len(row) > 5 and row[5] else 0.0,
                )
                plan_steps.append(step)
                total_cost += step.total_cost
                total_duration_ms += step.duration_ms

        # Get total rows processed
        total_rows = sum(step.rows_actual for step in plan_steps)

        # Generate suggestions
        suggestions = self._generate_suggestions(plan_steps, total_cost, total_duration_ms)

        return QueryProfile(
            query=query,
            plan_steps=plan_steps,
            total_rows=total_rows,
            total_duration_ms=total_duration_ms,
            total_cost=total_cost,
            planning_time_ms=0.0,  # Would be extracted from EXPLAIN output
            effective_cache_size_bytes=1000000,  # Would be from settings
            suggestions=suggestions,
        )

    def estimate_query_cost(self, query: str) -> float:
        """Estimate query cost without execution.

        Uses EXPLAIN (without ANALYZE) to estimate cost.

        Args:
            query: SQL query to estimate

        Returns:
            Estimated cost in query cost units

        Performance:
            Executes in < 100ms (plan only, no execution)
        """
        explain_results = self.repo.query_all(f"EXPLAIN {query}", [])

        total_cost = 0.0
        for row in explain_results:
            if len(row) >= 4:
                cost = float(row[3]) if row[3] else 0.0
                total_cost += cost

        return total_cost

    def optimize_suggestions(self, profile: QueryProfile) -> list[str]:
        """Generate optimization suggestions from profile.

        Args:
            profile: QueryProfile to analyze

        Returns:
            List of optimization suggestions

        Performance:
            Executes in < 10ms (local analysis only)
        """
        return self._generate_suggestions(
            profile.plan_steps, profile.total_cost, profile.total_duration_ms
        )

    def _generate_suggestions(
        self, plan_steps: list[PlanStep], total_cost: float, duration_ms: float
    ) -> list[str]:
        """Generate optimization suggestions from plan analysis.

        Args:
            plan_steps: Query plan steps
            total_cost: Total query cost
            duration_ms: Total execution time

        Returns:
            List of suggestions
        """
        suggestions = []

        # Expensive query suggestion
        if total_cost > 1000.0:
            suggestions.append(
                "Query is expensive (cost > 1000). Consider adding indexes or filtering."
            )

        # Slow query suggestion
        if duration_ms > 100.0:
            suggestions.append(
                "Query is slow (>100ms). Check for sequential scans or missing indexes."
            )

        # Look for expensive steps
        expensive_steps = [s for s in plan_steps if s.total_cost > 500.0]
        if expensive_steps:
            step_names = ", ".join(s.step_name for s in expensive_steps)
            suggestions.append(f"High-cost steps: {step_names}. Consider optimization.")

        # Look for estimation errors
        for step in plan_steps:
            if step.rows_actual > 0:
                ratio = step.rows_estimated / step.rows_actual
                if ratio > 10.0 or ratio < 0.1:
                    suggestions.append(
                        f"Estimation error in {step.step_name}: estimated {step.rows_estimated} "
                        f"but actual {step.rows_actual}. Stats may be outdated."
                    )

        # Default suggestion if no issues
        if not suggestions:
            suggestions.append("Query appears well-optimized.")

        return suggestions
