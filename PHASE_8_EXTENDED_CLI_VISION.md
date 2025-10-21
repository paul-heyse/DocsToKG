# Phase 8.10.3: Extended CLI Commands - Complete Vision

**Date**: October 21, 2025  
**Status**: Detailed Specification (Ready for Implementation)  
**Scope**: 4 additional commands to extend operational tooling

---

## Overview

The extended CLI commands provide operators with deep visibility into fallback strategy performance, configuration effectiveness, and operational health. These commands enable data-driven tuning and quick troubleshooting.

---

## Command 1: `fallback stats` - Telemetry Analysis

### Purpose
Parse production telemetry and provide actionable statistics about fallback performance.

### Implementation Details

**Command Syntax**:
```bash
python -m DocsToKG.ContentDownload.cli fallback stats \
  --manifest <path>           # Path to manifest.jsonl or manifest.sqlite3
  --period <duration>         # e.g., "24h", "7d", "30d"
  --tier <name>              # Optional: filter by tier
  --source <name>            # Optional: filter by source
  --resolution <granularity> # hourly, daily, weekly
  --format <output>          # text (default), json, csv
```

**Output Structure** (text format):
```
═══════════════════════════════════════════════════════════════════
                    FALLBACK STRATEGY STATISTICS
═══════════════════════════════════════════════════════════════════

Period: 2025-10-21 to 2025-10-22 (24 hours)
Total Work Items: 10,432
Total Attempts: 28,456

OVERALL METRICS:
  Success Rate: 78.5% (8,180/10,432)
  Average Attempts per Work: 2.73
  Average Elapsed Time: 1,245 ms
  Success Rate by Outcome:
    - success (PDF): 78.5% (8,180)
    - no_pdf (HTML): 12.3% (1,283)
    - error: 9.2% (959)

TIER PERFORMANCE:
  Tier 1 (Direct OA):
    - Attempts: 10,432
    - Success Rate: 45.2% (4,714)
    - Avg Elapsed: 850 ms
    - Sources: unpaywall, arxiv, pmc

  Tier 2 (DOI Resolution):
    - Attempts: 5,718 (55% of tier 1)
    - Success Rate: 32.1% (1,834)
    - Avg Elapsed: 1,520 ms
    - Sources: doi_redirect, landing_scrape

  Tier 3 (Web Archive):
    - Attempts: 948 (16.6% of tier 2)
    - Success Rate: 8.4% (80)
    - Avg Elapsed: 2,150 ms
    - Sources: wayback

SOURCE BREAKDOWN:
  unpaywall:
    - Attempts: 10,432
    - Success Rate: 42.1% (4,389)
    - Avg Elapsed: 820 ms
    - Timeout Rate: 0.2%
    - Error Rate: 3.1%

  arxiv:
    - Attempts: 4,156
    - Success Rate: 35.8% (1,489)
    - Avg Elapsed: 650 ms
    - Timeout Rate: 0.1%
    - Error Rate: 2.8%

  [... additional sources ...]

TIME DISTRIBUTION:
  Peak Hour (14:00-15:00): 1,243 attempts (11.9% of total)
  Quiet Hour (02:00-03:00): 310 attempts (3.0% of total)

TOP FAILURES (by reason):
  1. robots_blocked: 2,134 (22.3%)
  2. timeout: 1,832 (19.1%)
  3. not_found: 1,456 (15.2%)
  4. rate_limited: 987 (10.3%)
  5. [... more ...]

BUDGET EFFICIENCY:
  Total Time Budget Used: 45.2% (4h 32m / 10h total)
  Total Attempts Budget Used: 28,456 / 100,000 (28.5%)
  Concurrent Threads Peak: 8/10
  Cancellations: 0
═══════════════════════════════════════════════════════════════════
```

**JSON Output** (structured for dashboards):
```json
{
  "period": {
    "start": "2025-10-21T00:00:00Z",
    "end": "2025-10-22T00:00:00Z",
    "duration_hours": 24
  },
  "overall": {
    "work_items": 10432,
    "attempts": 28456,
    "success_rate": 0.785,
    "avg_attempts_per_work": 2.73,
    "avg_elapsed_ms": 1245
  },
  "tier_performance": [
    {
      "tier_name": "tier_1",
      "tier_number": 1,
      "attempts": 10432,
      "success_rate": 0.452,
      "avg_elapsed_ms": 850,
      "sources": ["unpaywall", "arxiv", "pmc"]
    }
  ],
  "source_breakdown": [
    {
      "source_name": "unpaywall",
      "attempts": 10432,
      "success_rate": 0.421,
      "avg_elapsed_ms": 820,
      "timeout_rate": 0.002,
      "error_rate": 0.031
    }
  ],
  "budget_efficiency": {
    "time_used_percent": 45.2,
    "attempts_used_percent": 28.5,
    "concurrent_peak": 8,
    "cancellations": 0
  }
}
```

**Implementation**:
```python
def cmd_fallback_stats(args: Any) -> None:
    """Display fallback strategy statistics from telemetry."""
    manifest_path = args.manifest or "Data/Manifests/manifest.sqlite3"
    period = args.period or "24h"
    tier_filter = getattr(args, 'tier', None)
    source_filter = getattr(args, 'source', None)
    output_format = args.format or "text"
    
    # Parse manifest (JSONL or SQLite)
    records = load_telemetry_records(manifest_path, period=period)
    
    # Filter by tier/source if provided
    if tier_filter:
        records = [r for r in records if r.get('tier') == tier_filter]
    if source_filter:
        records = [r for r in records if r.get('source') == source_filter]
    
    # Compute statistics
    stats = compute_statistics(records)
    
    # Format output
    if output_format == "text":
        print(format_stats_text(stats))
    elif output_format == "json":
        print(json.dumps(stats, indent=2))
    elif output_format == "csv":
        print(format_stats_csv(stats))
```

---

## Command 2: `fallback tune` - Configuration Recommendations

### Purpose
Analyze telemetry and suggest configuration improvements based on observed performance patterns.

### Implementation Details

**Command Syntax**:
```bash
python -m DocsToKG.ContentDownload.cli fallback tune \
  --manifest <path>          # Path to telemetry
  --period <duration>        # Analysis period
  --config <yaml>            # Current config (to show recommendations)
  --show-alternatives        # Show alternative configurations
  --simulation               # Simulate changes without applying
```

**Output Structure**:
```
═══════════════════════════════════════════════════════════════════
                    CONFIGURATION TUNING RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════

Based on 24-hour analysis (10,432 work items):

IMMEDIATE RECOMMENDATIONS:

1. ⚠️  TIER 3 (Wayback) Low Engagement
   Current: 948 attempts (9.1% of tier 2)
   Problem: Very low success rate (8.4%), high latency (2,150ms avg)
   
   Recommendation A (CONSERVATIVE):
     - Reduce tier 3 parallel from 5 to 2
     - Increase timeout from 5s to 10s (more lenient)
     - Impact: Save ~40% of tier 3 budget, similar success
   
   Recommendation B (AGGRESSIVE):
     - Move wayback to tier 4 (separate tier)
     - Tier 3 attempts only if tier 2 yields no PDF
     - Impact: 35% fewer tier 3 attempts, faster average resolution
   
   Recommendation C (DISABLE):
     - Disable wayback for this cohort
     - Impact: 9% reduction in overall latency

2. ✅ TIER 1 Performing Well
   Current: Success rate 45.2%, very stable
   Recommendation: Keep as-is, well-tuned

3. 🔧 RATE LIMITER OPTIMIZATION
   Current: Unpaywall hitting 10/sec limit frequently
   Problem: Creating bottleneck in tier 1
   
   Recommendation:
     - Increase unpaywall rate from 10/sec to 15/sec
     - Impact: Expected 12% faster tier 1 completion
     - Risk: May trigger rate limit from provider (low probability)

4. ⏱️  TIMEOUT TUNING
   Analysis shows arxiv rarely exceeds 2s:
   Current: 10s timeout
   Recommendation:
     - Reduce arxiv timeout from 10s to 3s
     - Impact: Fail fast on unresponsive endpoints, 18% faster tier 1
     - Risk: May miss slow but successful requests (0.5% impact estimated)

5. 🎯 BUDGET OPTIMIZATION
   Current usage: 45.2% of time budget, 28.5% of attempt budget
   Analysis: Time budget is constraint, attempt budget is slack
   
   Recommendation:
     - Increase total_timeout_ms from 30s to 40s (+33%)
     - Keep attempt budget at current level
     - Impact: Better success rate (+5-7%), no coordination overhead
     - Rationale: Time is the bottleneck, not attempts

PERFORMANCE PROJECTIONS:

Current Configuration:
  Success Rate: 78.5%
  Avg Latency: 1,245 ms
  Tier 3 Usage: 9.1% of tier 2 attempts

If Recommendation B Applied:
  Projected Success Rate: 80.1% (+1.6pp)
  Projected Avg Latency: 1,120 ms (-125ms, -10%)
  Tier 3 Usage: 3.1% of tier 2 attempts (-66%)

If All Recommendations Applied:
  Projected Success Rate: 83.4% (+4.9pp)
  Projected Avg Latency: 980 ms (-265ms, -21%)
  Budget Efficiency: 52.1% time, 31.2% attempts

ALTERNATIVE CONFIGURATIONS:

Configuration A: "Speed Optimized"
  - Aggressive timeouts (2-5s)
  - Higher rate limits
  - Fewer tier attempts
  - Expected: 15% faster, 2-3% lower success

Configuration B: "Reliability Optimized"
  - Conservative timeouts (10-15s)
  - Lower rate limits (respect providers)
  - More tier attempts (more fallback)
  - Expected: 5% slower, 8-12% higher success

Configuration C: "Balanced" (current recommendation)
  - Moderate timeouts (3-8s)
  - Medium rate limits
  - Tiered strategy
  - Expected: Best of both worlds

═══════════════════════════════════════════════════════════════════
```

**Implementation**:
```python
def cmd_fallback_tune(args: Any) -> None:
    """Analyze telemetry and recommend configuration improvements."""
    manifest_path = args.manifest or "Data/Manifests/manifest.sqlite3"
    period = args.period or "24h"
    config_path = getattr(args, 'config', None)
    
    # Load telemetry
    records = load_telemetry_records(manifest_path, period=period)
    
    # Current config
    current_config = load_fallback_plan(config_path) if config_path else load_fallback_plan()
    
    # Analyze patterns
    analysis = analyze_performance_patterns(records, current_config)
    
    # Generate recommendations
    recommendations = generate_tuning_recommendations(analysis)
    
    # Project impacts
    projections = project_configuration_impacts(recommendations, records)
    
    # Display
    print(format_tuning_recommendations(recommendations, projections))
    
    if args.show_alternatives:
        alternatives = generate_alternative_configs(analysis)
        print(format_alternatives(alternatives))
    
    if args.simulation:
        print(simulate_configuration_changes(recommendations, records))
```

---

## Command 3: `fallback explain` - Strategy Clarity

### Purpose
Explain the current fallback strategy in human-readable format.

### Implementation Details

**Command Syntax**:
```bash
python -m DocsToKG.ContentDownload.cli fallback explain \
  --config <yaml>            # Optional: custom config
  --detail <level>           # summary (default), detailed, full
  --show-timings            # Include timing estimates
  --show-flow               # ASCII flow diagram
```

**Output Structure**:
```
═══════════════════════════════════════════════════════════════════
                    FALLBACK STRATEGY EXPLANATION
═══════════════════════════════════════════════════════════════════

STRATEGY OVERVIEW:
This strategy attempts to resolve PDFs using a tiered approach with
parallel sources within each tier. It stops on first success.

TIER STRUCTURE:

┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Direct Open Access (Parallel: 5)                  │
│ ├─ unpaywall (10/sec, timeout: 10s)                       │
│ ├─ arxiv (no rate limit, timeout: 10s)                    │
│ └─ pmc (5/sec, timeout: 10s)                              │
│ └─ RESULT: Try next tier if no PDF found                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: DOI Resolution (Parallel: 3)                       │
│ ├─ doi_redirect (no rate limit, timeout: 8s)             │
│ ├─ landing_scrape (robots-aware, timeout: 8s)            │
│ └─ europe_pmc (5/sec, timeout: 8s)                       │
│ └─ RESULT: Try next tier if no PDF found                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Web Archive (Parallel: 1)                          │
│ └─ wayback (2/sec, timeout: 10s)                         │
│ └─ RESULT: Fallback to resolver pipeline if no PDF       │
└─────────────────────────────────────────────────────────────┘

BUDGETS (Global):
  - Total Timeout: 30 seconds
  - Total Attempts: 100
  - Max Concurrent: 10 threads

HEALTH GATES:
  ✓ Circuit breaker: OFF
  ✓ Offline mode: OFF
  ✓ Rate limiter: ACTIVE (per-source)

FLOW DIAGRAM (Actual Execution):

Start
  │
  ├─→ Load plan ✓
  │
  ├─→ Tier 1 (Parallel: 5 sources, 10s timeout)
  │   ├─→ unpaywall     (attempt 1)  ─→ PDF? ✓ SUCCESS [stop]
  │   ├─→ arxiv         (attempt 2)  ─→ PDF? ✗ continue
  │   ├─→ pmc           (attempt 3)  ─→ PDF? ✗ continue
  │   │
  │   └─→ Tier 1 Result: 2/3 successful, avg 1.2s
  │       PDF found in unpaywall → RESOLVED
  │
  ├─→ [If tier 1 failed, proceed to tier 2]
  │
  └─→ Return result with telemetry

TIMING ESTIMATES:

Best Case (Tier 1 Success, First Source):
  unpaywall resolves in 0.8s → Immediate success
  Total: ~1 second

Average Case (Tier 1 Fallback):
  Try 2-3 sources in tier 1, fallback to tier 2
  Tier 1: 1-2 seconds (parallel, fast fail)
  Tier 2: 1.5 seconds
  Total: ~2.5 seconds

Worst Case (All Tiers):
  All three tiers attempted
  Tier 1: 2 seconds
  Tier 2: 2 seconds
  Tier 3: 3 seconds (slowest)
  Total: ~7 seconds (budget: 30 seconds, OK)

INTEGRATION:

The strategy runs BEFORE the main resolver pipeline:
  1. Attempt fallback strategy (30s timeout)
  2. If successful, return PDF and skip pipeline
  3. If failed, fall back to resolver pipeline
  4. Return first successful result from pipeline

This dual-strategy approach ensures:
  • Fast success path (often 1-2 seconds)
  • Robust fallback (resolver pipeline as backup)
  • No production risk (can disable via feature gate)

KEY DESIGN DECISIONS:

1. Parallel within tiers (faster for multi-source availability)
2. Sequential across tiers (respect provider rate limits)
3. First-success semantics (stop on first PDF)
4. Budget enforcement (time and attempt caps)
5. Health gates (aware of breaker/rate limiter/offline)
6. Comprehensive telemetry (every decision tracked)

═══════════════════════════════════════════════════════════════════
```

**Implementation**:
```python
def cmd_fallback_explain(args: Any) -> None:
    """Explain the fallback strategy in detail."""
    config_path = getattr(args, 'config', None)
    detail_level = args.detail or "summary"
    
    plan = load_fallback_plan(config_path) if config_path else load_fallback_plan()
    
    explanation = ExplanationBuilder(plan, detail_level)
    
    if args.show_flow:
        print(explanation.render_flow_diagram())
    else:
        print(explanation.render_overview())
    
    if detail_level in ("detailed", "full"):
        print(explanation.render_detailed_breakdown())
    
    if args.show_timings:
        print(explanation.render_timing_estimates())
```

---

## Command 4: `fallback config` - Configuration Introspection

### Purpose
Show the effective configuration after merging YAML/env/CLI overrides.

### Implementation Details

**Command Syntax**:
```bash
python -m DocsToKG.ContentDownload.cli fallback config \
  --show-yaml               # YAML format (default)
  --show-json              # JSON format
  --show-effective         # After all merges (default: yes)
  --show-defaults          # Show hardcoded defaults
  --show-sources           # Where each value came from (YAML/env/CLI)
  --diff <path>            # Show diff vs another config
  --validate               # Validate without applying
```

**Output (YAML)**:
```yaml
fallback_strategy:
  enabled: true
  source: environment_variable (DOCSTOKG_ENABLE_FALLBACK_STRATEGY=1)
  
  budgets:
    total_timeout_ms: 30000        # from: defaults
    total_attempts: 100            # from: defaults
    max_concurrent: 10             # from: defaults
  
  tiers:
    - name: "tier_1_direct_oa"
      source: "config.yaml"
      parallel: 5
      sources:
        - "unpaywall"
        - "arxiv"
        - "pmc"
    
    - name: "tier_2_doi_resolution"
      source: "config.yaml"
      parallel: 3
      sources:
        - "doi_redirect"
        - "landing_scrape"
        - "europe_pmc"
    
    - name: "tier_3_archive"
      source: "config.yaml"
      parallel: 1
      sources:
        - "wayback"
  
  policies:
    unpaywall:
      name: "unpaywall"
      source: "config.yaml"
      timeout_ms: 10000
      retries_max: 3
      robots_respect: false
      rate_limit: "10/second"
    
    arxiv:
      name: "arxiv"
      source: "defaults"
      timeout_ms: 10000
      retries_max: 2
      robots_respect: true
      rate_limit: "unlimited"
    
    # ... more policies ...
  
  gates:
    circuit_breaker:
      enabled: false
      source: "defaults"
    
    offline_mode:
      enabled: false
      source: "runtime"
    
    rate_limiter:
      aware: true
      source: "defaults"
```

**Output (with sources)**:
```
Configuration Effective Values and Their Sources:

Key                          | Value              | Source            | Precedence
──────────────────────────────────────────────────────────────────────────
budgets.total_timeout_ms     | 30000              | defaults          | DEFAULT
budgets.total_attempts       | 100                | defaults          | DEFAULT
budgets.max_concurrent       | 10                 | defaults          | DEFAULT
tier_1.name                  | "tier_1_direct_oa" | config.yaml       | YAML
tier_1.parallel              | 5                  | config.yaml       | YAML
tier_2.parallel              | 3                  | environment (env) | ENV (override)
tier_3.parallel              | 1                  | config.yaml       | YAML
unpaywall.rate_limit         | "10/second"        | config.yaml       | YAML
arxiv.timeout_ms             | 10000              | defaults          | DEFAULT
──────────────────────────────────────────────────────────────────────────

Precedence Order (highest to lowest):
  1. CLI arguments (--tier-1-parallel 6)
  2. Environment variables (DOCSTOKG_FALLBACK_*)
  3. YAML config file
  4. Hardcoded defaults
```

**Implementation**:
```python
def cmd_fallback_config(args: Any) -> None:
    """Display effective fallback configuration."""
    # Load configuration with source tracking
    config, sources = load_fallback_plan_with_sources()
    
    if args.show_json:
        output = format_config_json(config, sources if args.show_sources else None)
    else:  # YAML
        output = format_config_yaml(config, sources if args.show_sources else None)
    
    print(output)
    
    if args.show_defaults:
        print("\nHardcoded Defaults:")
        print(format_config_yaml(DEFAULT_PLAN))
    
    if args.diff:
        other_config = load_fallback_plan_from_file(args.diff)
        print("\nConfiguration Diff:")
        print(generate_config_diff(config, other_config))
    
    if args.validate:
        validation_result = validate_plan(config)
        if validation_result.is_valid():
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration has issues:")
            for issue in validation_result.issues:
                print(f"  - {issue}")
```

---

## Summary of Extended CLI Commands

| Command | Purpose | Use Case |
|---------|---------|----------|
| `fallback stats` | Performance analytics | "How is fallback performing?" |
| `fallback tune` | Auto-tuning recommendations | "How can I improve this?" |
| `fallback explain` | Strategy documentation | "How does this work?" |
| `fallback config` | Configuration introspection | "What are the effective settings?" |

---

## Implementation Priority

1. **High Priority** (Immediate):
   - `fallback config` - Essential for troubleshooting
   - `fallback stats` - Key for performance monitoring

2. **Medium Priority** (Week 1-2):
   - `fallback explain` - Great for documentation and onboarding
   - `fallback tune` - Value-add for optimization

---

## Integration Points

All commands integrate with:
- **Telemetry storage**: Load from `manifest.sqlite3` or `manifest.jsonl`
- **Configuration system**: Load from YAML/env/defaults
- **Logging**: Structured output for dashboards
- **CLI framework**: Standard Typer integration

---

## Benefits

✅ **Operational Visibility**: See exactly how fallback is performing  
✅ **Data-Driven Tuning**: Recommendations based on real telemetry  
✅ **Self-Service Documentation**: Operators understand the strategy  
✅ **Configuration Confidence**: Know where each setting comes from  
✅ **Troubleshooting**: Quick diagnosis of performance issues  

