from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from DocsToKG.DocParsing.core import (
    ItemFingerprint,
    ItemOutcome,
    StageError,
    StageOptions,
    StagePlan,
    WorkItem,
    run_stage,
)


def _make_item(tmp_path: Path, idx: int, *, cfg_hash: str = "cfg") -> WorkItem:
    input_path = tmp_path / f"input-{idx}.txt"
    input_path.write_text(f"payload-{idx}", encoding="utf-8")
    output_path = tmp_path / f"output-{idx}.txt"
    metadata = {"doc_id": f"item-{idx}", "input_path": str(input_path), "output_path": str(output_path)}
    return WorkItem(
        item_id=f"item-{idx}",
        inputs={"source": input_path},
        outputs={"sink": output_path},
        cfg_hash=cfg_hash,
        metadata=metadata,
    )


def _success_worker(item: WorkItem) -> ItemOutcome:
    output = Path(item.outputs["sink"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("done", encoding="utf-8")
    manifest = {
        "input_path": str(item.inputs["source"]),
        "input_hash": "hash",
        "output_path": str(output),
        "schema_version": "docparse/1.0.0",
        "hash_alg": "sha256",
    }
    return ItemOutcome(status="success", duration_s=0.01, manifest=manifest, result={"ok": True})


def test_run_stage_success(tmp_path: Path) -> None:
    items = tuple(_make_item(tmp_path, idx) for idx in range(3))
    plan = StagePlan(stage_name="test-stage", items=items, total_items=len(items))
    outcome = run_stage(plan, _success_worker, StageOptions(workers=1))

    assert outcome.succeeded == len(items)
    assert outcome.failed == 0
    assert outcome.skipped == 0
    for item in items:
        assert Path(item.outputs["sink"]).read_text(encoding="utf-8") == "done"


def test_run_stage_resume_with_fingerprint(tmp_path: Path) -> None:
    item = _make_item(tmp_path, 0)
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text(json.dumps({"input_sha256": "abc", "cfg_hash": item.cfg_hash}), encoding="utf-8")
    output = Path(item.outputs["sink"])
    output.write_text("existing", encoding="utf-8")
    planned = WorkItem(
        item_id=item.item_id,
        inputs=item.inputs,
        outputs=item.outputs,
        cfg_hash=item.cfg_hash,
        metadata=item.metadata,
        fingerprint=ItemFingerprint(path=fingerprint, input_sha256="abc", cfg_hash=item.cfg_hash),
    )
    plan = StagePlan(stage_name="stage", items=(planned,), total_items=1)
    outcome = run_stage(plan, _success_worker, StageOptions(workers=1, resume=True))
    assert outcome.succeeded == 0
    assert outcome.skipped == 1
    assert outcome.failed == 0
    assert output.read_text(encoding="utf-8") == "existing"


def test_run_stage_retries_retryable_errors(tmp_path: Path) -> None:
    item = _make_item(tmp_path, 0)
    plan = StagePlan(stage_name="retry", items=(item,), total_items=1)
    attempts: defaultdict[str, int] = defaultdict(int)

    def flaky_worker(work_item: WorkItem) -> ItemOutcome:
        attempts[work_item.item_id] += 1
        if attempts[work_item.item_id] == 1:
            err = StageError(
                stage="retry",
                item_id=work_item.item_id,
                category="runtime",
                message="transient",
                retryable=True,
            )
            return ItemOutcome(status="failure", duration_s=0.01, manifest={}, result={}, error=err)
        return _success_worker(work_item)

    outcome = run_stage(plan, flaky_worker, StageOptions(workers=1, retries=2, retry_backoff_s=0.0))
    assert attempts[item.item_id] == 2
    assert outcome.succeeded == 1
    assert outcome.failed == 0
