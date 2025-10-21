"""Fallback & Resiliency Strategy for artifact resolution.

This package implements a deterministic, tiered fallback strategy for PDF
resolution across multiple sources. It provides:

- Tiered source resolution (Unpaywall → arXiv → PMC → DOI → Landing → EPMC → Wayback)
- Budgeted execution (time, attempts, concurrency limits)
- Health gates (circuit breaker, offline mode, rate limiter awareness)
- Automatic cancellation (stop as soon as valid PDF found)
- Full observability (per-attempt telemetry, SLO tracking)
- Zero-configuration defaults with YAML/env/CLI tuning

Key modules:
  - types: Core dataclasses (AttemptPolicy, AttemptResult, TierPlan, FallbackPlan)
  - orchestrator: Main orchestration logic (FallbackOrchestrator)
  - adapters: 7 source-specific adapters
  - loader: Configuration loading and merging
  - cli_fallback: CLI commands for operational control

Example:
    ```python
    from DocsToKG.ContentDownload.fallback import FallbackOrchestrator
    from DocsToKG.ContentDownload.fallback.loader import load_fallback_plan

    # Load configuration
    plan = load_fallback_plan(Path("config/fallback.yaml"))

    # Create orchestrator
    orchestrator = FallbackOrchestrator(
        plan=plan,
        breaker=breaker_registry,
        rate=rate_limiter,
        head_client=cached_client,
        raw_client=raw_client,
        telemetry=telemetry_sink,
        logger=logger
    )

    # Resolve PDF
    result = orchestrator.resolve_pdf(context, adapters)
    if result.outcome == "success":
        print(f"Found PDF at: {result.url}")
    ```
"""

__version__ = "1.0.0"

__all__ = [
    "FallbackOrchestrator",
    "FallbackPlan",
    "TierPlan",
    "AttemptPolicy",
    "AttemptResult",
    "ResolutionOutcome",
]
