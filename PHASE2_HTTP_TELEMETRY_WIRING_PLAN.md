# Phase 2: HTTP Telemetry Wiring Implementation Plan

**Status**: Ready for implementation (architecture validated)

## Objective

Wire `telemetry` + `run_id` through the download pipeline so HTTP request/response events are emitted to the telemetry database.

## Current State

✅ **Already Complete**:
- Rate limiter emitting `rate_events` (ratelimit.py lines 307, 326)
- Breaker listener emitting state transitions (networking_breaker_listener.py)
- Phase 1 HTTP helpers complete (8 extraction functions in networking.py)

⚠️ **Missing**:
- HTTP telemetry NOT wired into download pipeline
- `download_candidate()` and helpers don't receive/emit telemetry
- `process_one_work()` doesn't pass run_id to pipeline

## Implementation Changes Required

### 1) Modify DownloadPreflightPlan (download.py:417-442)

**Add fields**:
```python
@dataclass
class DownloadPreflightPlan:
    # ... existing fields ...
    telemetry: Optional[Any] = None          # RunTelemetry sink
    run_id: Optional[str] = None             # Run identifier
```

### 2) Update prepare_candidate_download() (download.py:458-469)

**Add parameters**:
```python
def prepare_candidate_download(
    client: Optional[httpx.Client],
    artifact: WorkArtifact,
    url: str,
    referer: Optional[str],
    timeout: float,
    ctx: DownloadContext,
    *,
    head_precheck_passed: bool = False,
    original_url: Optional[str] = None,
    origin_host: Optional[str] = None,
    telemetry: Optional[Any] = None,         # NEW
    run_id: Optional[str] = None,            # NEW
) -> DownloadPreflightPlan:
```

**Propagate to plan**:
```python
    return DownloadPreflightPlan(
        # ... existing fields ...
        telemetry=telemetry,                 # NEW
        run_id=run_id,                       # NEW
    )
```

### 3) Update stream_candidate_payload() (download.py:605)

**Add parameters**:
```python
def stream_candidate_payload(
    plan: DownloadPreflightPlan,
    telemetry: Optional[Any] = None,        # NEW (or extract from plan)
    run_id: Optional[str] = None,           # NEW (or extract from plan)
) -> DownloadStreamResult:
```

**Emit HTTP events**:
At line 646 (request_with_retries call), add:
```python
                response_cm = request_with_retries(
                    client,
                    "GET",
                    url,
                    role="artifact",
                    stream=True,
                    allow_redirects=True,
                    timeout=timeout,
                    headers=headers,
                    retry_after_cap=retry_after_cap,
                    content_policy=content_policy,
                    original_url=plan.original_url,
                    origin_host=plan.origin_host,
                    telemetry=plan.telemetry,    # NEW
                    run_id=plan.run_id,          # NEW
                )
```

### 4) Update finalize_candidate_download() (download.py)

**Add parameters**:
```python
def finalize_candidate_download(
    plan: DownloadPreflightPlan,
    stream_result: DownloadStreamResult,
    telemetry: Optional[Any] = None,        # NEW
    run_id: Optional[str] = None,           # NEW
) -> DownloadOutcome:
```

**Use for manifest emission** (at end):
```python
    if telemetry and run_id:
        telemetry.record_pipeline_result(
            plan.artifact,
            result=outcome,
            run_id=run_id,
        )
    
    return outcome
```

### 5) Wire from ResolverPipeline._process_result() (pipeline.py)

**Find where download_candidate is called**:
```python
    outcome = download_candidate(
        client=client,
        artifact=artifact,
        url=url,
        referer=None,
        timeout=timeout,
        context=resolved_context,
        original_url=original_url,
        origin_host=origin_host,
        telemetry=self.logger,              # NEW
        run_id=self._run_id,                # NEW
    )
```

## Test Plan

### Test 1: HTTP events emitted with run_id

**File**: `tests/content_download/test_telemetry_integration_phase2.py`

```python
def test_http_get_event_includes_run_id():
    # Arrange: Mock RunTelemetry with capture sink
    events = []
    mock_telemetry = Mock()
    mock_telemetry.record_pipeline_result = lambda **kw: events.append(kw)
    
    # Act: download_candidate with telemetry
    outcome = download_candidate(
        client=mock_client,
        artifact=work_artifact,
        url="https://example.org/test.pdf",
        referer=None,
        timeout=30,
        context=ctx,
        telemetry=mock_telemetry,
        run_id="test-run-123",
    )
    
    # Assert: record_pipeline_result called with run_id
    assert len(events) == 1
    assert events[0]["run_id"] == "test-run-123"
```

### Test 2: Robots-disallowed short-circuit

**Setup**: RobotsCache returns False
**Assert**: Outcome reason is `ROBOTS_DISALLOWED`, still emitted with run_id

### Test 3: Conditional 304 path

**Setup**: Response is 304 Not Modified
**Assert**: Outcome reason is cached/not-modified, telemetry emitted

## Acceptance Criteria

- [ ] `DownloadPreflightPlan` has `telemetry` + `run_id` fields
- [ ] `prepare_candidate_download()` accepts and stores telemetry + run_id
- [ ] `stream_candidate_payload()` passes telemetry/run_id to `request_with_retries()`
- [ ] `finalize_candidate_download()` uses `telemetry.record_pipeline_result()`
- [ ] `ResolverPipeline._process_result()` wires `self.logger` and `self._run_id`
- [ ] 3 integration tests pass
- [ ] No linting/type errors
- [ ] HTTP events appear in telemetry database during smoke test

## Estimated Effort

- Code changes: 2-3 hours (6 edits across 2 files)
- Testing: 1-2 hours (3 integration tests)
- Total: **4-5 hours** → **Phase 2 COMPLETE**

