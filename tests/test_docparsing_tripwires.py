"""Trip-wire tests for DocParsing pipeline stability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (
    Rec,
    coalesce_small_runs,
)

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
st = hypothesis.strategies

GOLDEN_DIR = Path("tests/data/docparsing/golden")
GOLDEN_CHUNKS = GOLDEN_DIR / "sample.chunks.jsonl"
GOLDEN_VECTORS = GOLDEN_DIR / "sample.vectors.jsonl"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def test_golden_chunk_count_and_hash():
    """Golden chunk fixtures should maintain deterministic order and count."""

    rows = list(load_jsonl(GOLDEN_CHUNKS))
    assert len(rows) == 2
    hashes = [hashlib.sha1(row["text"].encode("utf-8")).hexdigest() for row in rows]
    assert hashes == [
        "9d40a282aefb81ec15147275d8d490b40c334694",
        "4baf2d39299bd11c643b7c398248aae3b80765ae",
    ]


class _DummyTokenizer:
    def count_tokens(self, text: str) -> int:
        return len([tok for tok in text.split() if tok])


@given(st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=6))
def test_coalescer_invariants(token_counts):
    """Coalesced records must obey size and ordering invariants."""

    tokenizer = _DummyTokenizer()
    records = [
        Rec(
            text=" ".join(f"tok{i}_{j}" for j in range(count)),
            n_tok=count,
            src_idxs=[i],
            refs=[],
            pages=[],
        )
        for i, count in enumerate(token_counts)
    ]
    min_tokens = 5
    max_tokens = 10
    result = coalesce_small_runs(records, tokenizer, min_tokens=min_tokens, max_tokens=max_tokens)

    if len(result) > 1:
        for idx, rec in enumerate(result[:-1]):
            if rec.n_tok < min_tokens:
                left = result[idx - 1] if idx > 0 else None
                right = result[idx + 1]
                if left is not None:
                    assert left.n_tok + rec.n_tok > max_tokens
                if right is not None:
                    assert rec.n_tok + right.n_tok > max_tokens
            else:
                assert rec.n_tok >= min_tokens
    for rec in result:
        if rec.n_tok > max_tokens:
            assert any(records[idx].n_tok > max_tokens for idx in rec.src_idxs)

    flattened = [idx for rec in result for idx in rec.src_idxs]
    assert flattened == sorted(flattened)


def test_embedding_shapes_tripwire():
    """Hybrid vectors must satisfy schema shape invariants."""

    rows = list(load_jsonl(GOLDEN_VECTORS))
    assert rows, "golden vectors fixture is empty"
    for row in rows:
        qwen = row["Qwen3-4B"]
        splade = row["SPLADEv3"]
        bm25 = row["BM25"]

        assert len(qwen["vector"]) == 2560
        assert qwen.get("dimension") == 2560
        assert len(splade["tokens"]) == len(splade["weights"]) > 0
        assert all(weight >= 0 for weight in splade["weights"])
        assert len(bm25["terms"]) == len(bm25["weights"]) > 0
