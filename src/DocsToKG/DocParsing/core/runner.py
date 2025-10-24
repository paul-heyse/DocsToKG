# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.runner",
#   "purpose": "Stage runner scaffolding shared by DocParsing pipelines.",
#   "sections": [
#     {
#       "id": "itemfingerprint",
#       "name": "ItemFingerprint",
#       "anchor": "class-itemfingerprint",
#       "kind": "class"
#     },
#     {
#       "id": "workitem",
#       "name": "WorkItem",
#       "anchor": "class-workitem",
#       "kind": "class"
#     },
#     {
#       "id": "stageplan",
#       "name": "StagePlan",
#       "anchor": "class-stageplan",
#       "kind": "class"
#     },
#     {
#       "id": "stageoptions",
#       "name": "StageOptions",
#       "anchor": "class-stageoptions",
#       "kind": "class"
#     },
#     {
#       "id": "stageerror",
#       "name": "StageError",
#       "anchor": "class-stageerror",
#       "kind": "class"
#     },
#     {
#       "id": "itemoutcome",
#       "name": "ItemOutcome",
#       "anchor": "class-itemoutcome",
#       "kind": "class"
#     },
#     {
#       "id": "stageoutcome",
#       "name": "StageOutcome",
#       "anchor": "class-stageoutcome",
#       "kind": "class"
#     },
#     {
#       "id": "stagehooks",
#       "name": "StageHooks",
#       "anchor": "class-stagehooks",
#       "kind": "class"
#     },
#     {
#       "id": "stagecontext",
#       "name": "StageContext",
#       "anchor": "class-stagecontext",
#       "kind": "class"
#     },
#     {
#       "id": "submission",
#       "name": "_Submission",
#       "anchor": "class-submission",
#       "kind": "class"
#     },
#     {
#       "id": "workerpayload",
#       "name": "_WorkerPayload",
#       "anchor": "class-workerpayload",
#       "kind": "class"
#     },
#     {
#       "id": "now",
#       "name": "_now",
#       "anchor": "function-now",
#       "kind": "function"
#     },
#     {
#       "id": "call-worker",
#       "name": "_call_worker",
#       "anchor": "function-call-worker",
#       "kind": "function"
#     },
#     {
#       "id": "should-skip",
#       "name": "_should_skip",
#       "anchor": "function-should-skip",
#       "kind": "function"
#     },
#     {
#       "id": "apply-backoff",
#       "name": "_apply_backoff",
#       "anchor": "function-apply-backoff",
#       "kind": "function"
#     },
#     {
#       "id": "percentile",
#       "name": "_percentile",
#       "anchor": "function-percentile",
#       "kind": "function"
#     },
#     {
#       "id": "create-executor",
#       "name": "_create_executor",
#       "anchor": "function-create-executor",
#       "kind": "function"
#     },
#     {
#       "id": "run-stage",
#       "name": "run_stage",
#       "anchor": "function-run-stage",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Stage runner scaffolding shared by DocParsing pipelines.

Stages describe the work they need to perform via :class:`StagePlan` and the
runner takes care of concurrency, retries, resume/force semantics, and progress
reporting.  The implementation mirrors the PR-5 design notes so doctags,
chunking, and embedding can rely on a single execution core.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import random
import statistics
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
)

from DocsToKG.concurrency.executors import create_executor
from DocsToKG.DocParsing.logging import get_logger, log_event

__all__ = [
    "ItemFingerprint",
    "ItemOutcome",
    "StageContext",
    "StageError",
    "StageHooks",
    "StageOptions",
    "StageOutcome",
    "StagePlan",
    "WorkItem",
    "run_stage",
]

ItemStatus = str  # Allowed: "success", "skip", "failure"


@dataclass(slots=True, frozen=True)
class ItemFingerprint:
    """Resume fingerprint recorded alongside stage outputs."""

    path: Path
    input_sha256: str | None = None
    cfg_hash: str | None = None

    def matches(self) -> bool:
        """Return ``True`` when the fingerprint on disk matches expectations."""

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        expected_input = (self.input_sha256 or "").strip()
        expected_cfg = (self.cfg_hash or "").strip()
        observed_input = str(payload.get("input_sha256", "")).strip()
        observed_cfg = str(payload.get("cfg_hash", "")).strip()
        if expected_input and expected_input != observed_input:
            return False
        if expected_cfg and expected_cfg != observed_cfg:
            return False
        return True


@dataclass(slots=True, frozen=True)
class WorkItem:
    """Immutable description of a single unit of work."""

    item_id: str
    inputs: Mapping[str, Path]
    outputs: Mapping[str, Path]
    cfg_hash: str
    cost_hint: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: ItemFingerprint | None = None

    def materialize(self) -> WorkItem:
        """Return a version with plain dictionaries suitable for pickling."""

        inputs = {key: Path(value) for key, value in dict(self.inputs).items()}
        outputs = {key: Path(value) for key, value in dict(self.outputs).items()}
        metadata = dict(self.metadata)
        fingerprint = self.fingerprint
        return WorkItem(
            item_id=self.item_id,
            inputs=inputs,
            outputs=outputs,
            cfg_hash=self.cfg_hash,
            cost_hint=float(self.cost_hint),
            metadata=metadata,
            fingerprint=fingerprint,
        )


@dataclass(slots=True, frozen=True)
class StagePlan:
    """Deterministic plan enumerating the items a stage must execute."""

    stage_name: str
    items: Sequence[WorkItem]
    total_items: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))

    def __iter__(self) -> Iterator[WorkItem]:
        return iter(self.items)


@dataclass(slots=True)
class StageOptions:
    """Shared execution knobs across DocParsing stages."""

    policy: str = "io"
    workers: int = 1
    per_item_timeout_s: float = 0.0
    retries: int = 0
    retry_backoff_s: float = 1.0
    error_budget: int = 0
    max_queue: int = 0
    resume: bool = False
    force: bool = False
    diagnostics_interval_s: float = 30.0
    seed: int | None = None
    dry_run: bool = False


@dataclass(slots=True)
class StageError(Exception):
    """Structured error surfaced by stage workers."""

    stage: str
    item_id: str
    category: str
    message: str
    retryable: bool = False
    detail: Mapping[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover - human readable fallback
        return f"[{self.category}] {self.message}"


@dataclass(slots=True)
class ItemOutcome:
    """Worker outcome normalised for runner bookkeeping."""

    status: ItemStatus
    duration_s: float
    manifest: Mapping[str, Any] = field(default_factory=dict)
    result: Mapping[str, Any] = field(default_factory=dict)
    error: StageError | None = None


@dataclass(slots=True)
class StageOutcome:
    """Summary returned by :func:`run_stage`."""

    scheduled: int
    skipped: int
    succeeded: int
    failed: int
    cancelled: bool
    wall_ms: float
    queue_p50_ms: float
    queue_p95_ms: float
    exec_p50_ms: float
    exec_p95_ms: float
    exec_p99_ms: float
    cpu_time_total_ms: float
    errors: Sequence[StageError]


@dataclass(slots=True)
class StageHooks:
    """Optional lifecycle hooks invoked around execution."""

    before_stage: Callable[[StageContext], None] | None = None
    after_stage: Callable[[StageOutcome, StageContext], None] | None = None
    before_item: Callable[[WorkItem, StageContext], None] | None = None
    after_item: Callable[[WorkItem, ItemOutcome | StageError, StageContext], None] | None = None


@dataclass(slots=True)
class StageContext:
    """Mutable context surfaced to hooks."""

    plan: StagePlan
    options: StageOptions
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stage_name(self) -> str:
        return self.plan.stage_name


@dataclass(slots=True)
class _Submission:
    """Internal metadata describing an in-flight worker execution."""

    item: WorkItem
    attempt: int
    enqueue_time: float
    future: Future | None = None
    started_at: float | None = None
    deadline: float | None = None


@dataclass(slots=True)
class _WorkerPayload:
    """Result wrapper returned by worker processes/threads."""

    outcome: ItemOutcome | None
    error: BaseException | None
    started_at: float
    finished_at: float


def _now() -> float:
    return time.perf_counter()


def _call_worker(worker: Callable[[WorkItem], ItemOutcome], item: WorkItem) -> _WorkerPayload:
    started = _now()
    try:
        outcome = worker(item)
        finished = _now()
        if not isinstance(outcome, ItemOutcome):
            raise StageError(
                stage="unknown",
                item_id=item.item_id,
                category="contract",
                message=f"Worker for {item.item_id} returned {type(outcome).__name__}",
                retryable=False,
            )
        if outcome.duration_s <= 0.0:
            duration = finished - started
            outcome = ItemOutcome(
                status=outcome.status,
                duration_s=duration,
                manifest=dict(outcome.manifest),
                result=dict(outcome.result),
                error=outcome.error,
            )
        return _WorkerPayload(outcome=outcome, error=None, started_at=started, finished_at=finished)
    except BaseException as exc:  # pragma: no cover - defensive
        finished = _now()
        return _WorkerPayload(outcome=None, error=exc, started_at=started, finished_at=finished)


def _should_skip(item: WorkItem, options: StageOptions) -> bool:
    """Return ``True`` when resume semantics allow skipping ``item``."""

    if options.force or not options.resume:
        return False

    for path in item.outputs.values():
        try:
            stat = Path(path).stat()
            if stat.st_size <= 0:
                return False
        except FileNotFoundError:
            return False

    fingerprint = item.fingerprint
    if fingerprint is None:
        return False
    return fingerprint.matches()


def _apply_backoff(
    base: float, attempt: int, rng: random.Random | None = None
) -> float:
    if base <= 0:
        return 0.0
    exponent = max(0, attempt - 1)
    delay = base * (2**exponent)
    jitter = delay * (rng.uniform(0.0, 0.25) if rng is not None else 0.0)
    return delay + jitter


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pct = float(pct)
    if pct <= 0.0:
        return float(min(values))
    if pct >= 100.0:
        return float(max(values))
    try:
        quantiles = statistics.quantiles(values, n=100, method="inclusive")
        index = max(1, min(99, int(math.ceil(pct)))) - 1
        return float(quantiles[index])
    except (ValueError, statistics.StatisticsError):
        ordered = sorted(values)
        index = int(math.ceil(pct / 100.0 * len(ordered))) - 1
        index = max(0, min(index, len(ordered) - 1))
        return float(ordered[index])


def _create_executor(options: StageOptions) -> tuple[cf.Executor | None, bool]:
    workers = max(1, int(options.workers))
    if workers <= 1:
        return None, False
    return create_executor(options.policy or "io", workers)


def run_stage(
    plan: StagePlan,
    worker: Callable[[WorkItem], ItemOutcome],
    options: StageOptions | None = None,
    hooks: StageHooks | None = None,
) -> StageOutcome:
    """Execute ``plan`` using the provided ``worker`` and options."""

    options = options or StageOptions()
    hooks = hooks or StageHooks()

    backoff_rng: random.Random | None = None
    if options.retry_backoff_s > 0 and options.seed is not None:
        backoff_rng = random.Random(options.seed)

    logger = get_logger(
        f"DocsToKG.DocParsing.core.runner.{plan.stage_name}",
        base_fields={"stage": plan.stage_name},
    )
    context = StageContext(plan=plan, options=options)

    cpu_start = time.process_time()
    items = [item.materialize() for item in plan]
    total_items = plan.total_items if plan.total_items >= 0 else len(items)

    if hooks.before_stage:
        try:
            hooks.before_stage(context)
        except Exception as exc:  # pragma: no cover - defensive logging
            log_event(
                logger.logger,
                "warning",
                "before_stage hook failed",
                stage=plan.stage_name,
                doc_id="__system__",
                error=str(exc),
            )

    scheduled = 0
    skipped = 0
    succeeded = 0
    failed = 0
    cancelled = False
    queue_ms: list[float] = []
    exec_ms: list[float] = []
    errors: list[StageError] = []

    diagnostics_interval = max(1.0, float(options.diagnostics_interval_s or 30.0))
    last_diag = _now()
    wall_start = last_diag

    submission_queue: deque[tuple[WorkItem, int]] = deque()

    for item in items:
        if options.dry_run:
            skipped += 1
            continue

        if hooks.before_item:
            try:
                hooks.before_item(item, context)
            except Exception:  # pragma: no cover - defensive
                pass

        if _should_skip(item, options):
            skipped += 1
            skip_outcome = ItemOutcome(
                status="skip",
                duration_s=0.0,
                manifest={},
                result={"reason": "resume-satisfied"},
            )
            if hooks.after_item:
                try:
                    hooks.after_item(item, skip_outcome, context)
                except Exception:  # pragma: no cover - defensive
                    pass
            continue

        submission_queue.append((item, 1))

    total_to_run = len(submission_queue)

    if options.dry_run:
        wall_ms = (_now() - wall_start) * 1000.0
        cpu_total_ms = max(0.0, (time.process_time() - cpu_start) * 1000.0)
        outcome = StageOutcome(
            scheduled=0,
            skipped=skipped,
            succeeded=0,
            failed=0,
            cancelled=False,
            wall_ms=wall_ms,
            queue_p50_ms=0.0,
            queue_p95_ms=0.0,
            exec_p50_ms=0.0,
            exec_p95_ms=0.0,
            exec_p99_ms=0.0,
            cpu_time_total_ms=cpu_total_ms,
            errors=tuple(),
        )
        if hooks.after_stage:
            try:
                hooks.after_stage(outcome, context)
            except Exception as hook_exc:  # pragma: no cover - defensive
                log_event(
                    logger.logger,
                    "warning",
                    "after_stage hook raised",
                    stage=plan.stage_name,
                    doc_id="__system__",
                    error=str(hook_exc),
                )
        return outcome

    executor, _ = _create_executor(options)
    pending: dict[Future, _Submission] = {}

    def _handle_worker_payload(
        item: WorkItem,
        attempt: int,
        payload: _WorkerPayload,
        enqueue_time: float,
    ) -> StageError | None:
        nonlocal succeeded, failed

        queue_ms.append(max(0.0, (payload.started_at - enqueue_time) * 1000.0))
        exec_ms.append(max(0.0, (payload.finished_at - payload.started_at) * 1000.0))

        if payload.error is not None:
            exc = payload.error
            if isinstance(exc, StageError):
                err = exc
            else:
                err = StageError(
                    stage=plan.stage_name,
                    item_id=item.item_id,
                    category="runtime",
                    message=str(exc),
                    retryable=False,
                )
            errors.append(err)
            if hooks.after_item:
                try:
                    hooks.after_item(item, err, context)
                except Exception as hook_exc:  # pragma: no cover - defensive
                    log_event(
                        logger.logger,
                        "warning",
                        "after_item hook raised",
                        stage=plan.stage_name,
                        doc_id=item.item_id,
                        error=str(hook_exc),
                    )
            return err

        outcome = payload.outcome
        if outcome is None:
            err = StageError(
                stage=plan.stage_name,
                item_id=item.item_id,
                category="runtime",
                message="Worker returned no outcome",
                retryable=False,
            )
            errors.append(err)
            if hooks.after_item:
                try:
                    hooks.after_item(item, err, context)
                except Exception as hook_exc:
                    log_event(
                        logger.logger,
                        "warning",
                        "after_item hook raised",
                        stage=plan.stage_name,
                        doc_id=item.item_id,
                        error=str(hook_exc),
                    )
            return err

        if outcome.status == "success":
            succeeded += 1
        elif outcome.status == "skip":
            # Skip results triggered from worker; maintain counts.
            skipped += 1
        else:
            failed += 1
            if outcome.error is not None:
                errors.append(outcome.error)
            else:
                errors.append(
                    StageError(
                        stage=plan.stage_name,
                        item_id=item.item_id,
                        category="runtime",
                        message="Worker reported failure without error",
                        retryable=False,
                    )
                )

        if hooks.after_item:
            try:
                hooks.after_item(item, outcome, context)
            except Exception as hook_exc:  # pragma: no cover - defensive
                log_event(
                    logger.logger,
                    "warning",
                    "after_item hook raised",
                    stage=plan.stage_name,
                    doc_id=item.item_id,
                    error=str(hook_exc),
                )
        return outcome.error

    class _BudgetExceeded(RuntimeError):
        """Signal that the error budget has been exhausted."""

    def _submit_with_retry(item: WorkItem, attempt: int, err: StageError | None) -> None:
        nonlocal failed
        if err is not None and err.retryable and attempt <= options.retries:
            failed = max(0, failed - 1)
            delay = _apply_backoff(options.retry_backoff_s, attempt, backoff_rng)
            if delay > 0:
                time.sleep(delay)
            submission_queue.appendleft((item, attempt + 1))

    def _submit(item: WorkItem, attempt: int) -> None:
        nonlocal scheduled, failed
        enqueue_time = _now()
        if executor is None:
            scheduled += 1
            payload = _call_worker(worker, item)
            error = _handle_worker_payload(item, attempt, payload, enqueue_time)
            _submit_with_retry(item, attempt, error)
            if options.error_budget and len(errors) > options.error_budget:
                raise _BudgetExceeded()
            return

        submission = _Submission(item=item, attempt=attempt, enqueue_time=enqueue_time)

        timeout_s = options.per_item_timeout_s if options.per_item_timeout_s > 0 else None

        def _execute_item() -> _WorkerPayload:
            submission.started_at = _now()
            if timeout_s is not None:
                submission.deadline = submission.started_at + timeout_s
            else:
                submission.deadline = None
            return _call_worker(worker, item)

        future = executor.submit(_execute_item)
        submission.future = future
        pending[future] = submission
        scheduled += 1

    def _expire_timeouts(now: float) -> None:
        nonlocal failed
        if options.per_item_timeout_s <= 0:
            return
        for future, submission in list(pending.items()):
            deadline = submission.deadline
            if deadline is None or submission.future is None:
                continue
            if future.done() or submission.started_at is None:
                continue
            if now < deadline:
                continue
            pending.pop(future, None)
            future.cancel()
            started = submission.started_at
            queue_ms.append(max(0.0, (started - submission.enqueue_time) * 1000.0))
            exec_ms.append(max(0.0, (now - started) * 1000.0))
            failed += 1
            err = StageError(
                stage=plan.stage_name,
                item_id=submission.item.item_id,
                category="timeout",
                message=f"Operation exceeded {options.per_item_timeout_s}s",
                retryable=True,
            )
            errors.append(err)
            if hooks.after_item:
                try:
                    hooks.after_item(submission.item, err, context)
                except Exception as hook_exc:
                    log_event(
                        logger.logger,
                        "warning",
                        "after_item hook raised",
                        stage=plan.stage_name,
                        doc_id=submission.item.item_id,
                        error=str(hook_exc),
                    )
            _submit_with_retry(submission.item, submission.attempt, err)
            if options.error_budget and len(errors) > options.error_budget:
                raise _BudgetExceeded()

    def _drain_completed(blocking: bool) -> None:
        nonlocal failed
        if not pending:
            return
        now = _now()
        _expire_timeouts(now)
        if not pending:
            return

        futures = tuple(pending.keys())
        timeout = None if blocking else 0.0
        if blocking and options.per_item_timeout_s > 0:
            deadlines = [
                submission.deadline
                for submission in pending.values()
                if submission.deadline is not None
                and submission.future is not None
                and not submission.future.done()
            ]
            if deadlines:
                min_deadline = max(0.0, min(deadlines) - now)
                timeout = min_deadline if timeout is None else min(timeout, min_deadline)

        done, _ = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
        for future in done:
            submission = pending.pop(future, None)
            if submission is None:
                continue
            try:
                payload = future.result()
            except CancelledError:
                # Already handled via timeout cancellation.
                continue
            except Exception as exc:  # pragma: no cover - defensive
                payload = _WorkerPayload(
                    outcome=None,
                    error=exc,
                    started_at=_now(),
                    finished_at=_now(),
                )

            error = _handle_worker_payload(
                submission.item,
                submission.attempt,
                payload,
                submission.enqueue_time,
            )
            _submit_with_retry(submission.item, submission.attempt, error)
            if options.error_budget and len(errors) > options.error_budget:
                raise _BudgetExceeded()

        if pending:
            _expire_timeouts(_now())

    try:
        while submission_queue or pending:
            while submission_queue:
                if (
                    executor is not None
                    and options.max_queue > 0
                    and len(pending) >= options.max_queue
                ):
                    _drain_completed(blocking=True)
                    continue
                item, attempt = submission_queue.popleft()
                _submit(item, attempt)
                break

            _drain_completed(blocking=False)

            now = _now()
            if now - last_diag >= diagnostics_interval:
                last_diag = now
                log_event(
                    logger.logger,
                    "info",
                    "stage progress",
                    stage=plan.stage_name,
                    doc_id="__system__",
                    scheduled=scheduled,
                    completed=succeeded + failed + skipped,
                    total=total_items,
                    succeeded=succeeded,
                    failed=failed,
                    skipped=skipped,
                    pending=len(pending),
                )
    except KeyboardInterrupt:  # pragma: no cover - interactive safety
        cancelled = True
        for future in list(pending.keys()):
            submission = pending.pop(future)
            future.cancel()
            err = StageError(
                stage=plan.stage_name,
                item_id=submission.item.item_id,
                category="cancelled",
                message="Cancelled by user",
                retryable=False,
            )
            errors.append(err)
            if hooks.after_item:
                try:
                    hooks.after_item(submission.item, err, context)
                except Exception as hook_exc:
                    log_event(
                        logger.logger,
                        "warning",
                        "after_item hook raised",
                        stage=plan.stage_name,
                        doc_id=submission.item.item_id,
                        error=str(hook_exc),
                    )
    except _BudgetExceeded:
        cancelled = True
        for future in list(pending.keys()):
            submission = pending.pop(future)
            future.cancel()
            err = StageError(
                stage=plan.stage_name,
                item_id=submission.item.item_id,
                category="budget",
                message="Stage error budget exceeded",
                retryable=False,
            )
            errors.append(err)
            if hooks.after_item:
                try:
                    hooks.after_item(submission.item, err, context)
                except Exception as hook_exc:
                    log_event(
                        logger.logger,
                        "warning",
                        "after_item hook raised",
                        stage=plan.stage_name,
                        doc_id=submission.item.item_id,
                        error=str(hook_exc),
                    )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    wall_ms = (_now() - wall_start) * 1000.0
    cpu_total_ms = max(0.0, (time.process_time() - cpu_start) * 1000.0)
    outcome = StageOutcome(
        scheduled=total_to_run,
        skipped=skipped,
        succeeded=succeeded,
        failed=failed,
        cancelled=cancelled,
        wall_ms=wall_ms,
        queue_p50_ms=_percentile(queue_ms, 50.0),
        queue_p95_ms=_percentile(queue_ms, 95.0),
        exec_p50_ms=_percentile(exec_ms, 50.0),
        exec_p95_ms=_percentile(exec_ms, 95.0),
        exec_p99_ms=_percentile(exec_ms, 99.0),
        cpu_time_total_ms=cpu_total_ms,
        errors=tuple(errors),
    )

    if hooks.after_stage:
        try:
            hooks.after_stage(outcome, context)
        except Exception as hook_exc:  # pragma: no cover - defensive
            log_event(
                logger.logger,
                "warning",
                "after_stage hook raised",
                stage=plan.stage_name,
                doc_id="__system__",
                error=str(hook_exc),
            )

    return outcome
