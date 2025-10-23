# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.__init__",
#   "purpose": "Fallback & Resiliency Strategy.",
#   "sections": []
# }
# === /NAVMAP ===

"""
Fallback & Resiliency Strategy

Provides tiered, budgeted PDF resolution across multiple sources with:
- Deterministic resolver ordering (Unpaywall → arXiv → PMC → DOI → Landing → Europe PMC → Wayback)
- Time and attempt budgets
- Concurrent tier execution
- Circuit breaker awareness
- Full telemetry integration
- Optional feature gate for gradual rollout

Public API:
  FallbackOrchestrator - Main orchestrator class
  AttemptResult - Per-attempt outcome
  FallbackPlan - Configuration
  ResolutionOutcome - Success/failure literals
"""

from .types import (
    AttemptPolicy,
    AttemptResult,
    FallbackPlan,
    ResolutionOutcome,
    TierPlan,
)

__all__ = [
    "AttemptPolicy",
    "AttemptResult",
    "FallbackPlan",
    "ResolutionOutcome",
    "TierPlan",
]
