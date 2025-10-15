"""Trip-wire tests for DocParsing pipeline stability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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
        "a63d6a8f9a76873180c41b1a086e4047f8b8880a",
        "5b3d13f3fd1a552d3d1c28f814063936aecc42c7",
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
        for rec in result[:-1]:
            assert rec.n_tok >= min_tokens
    for rec in result:
        assert rec.n_tok <= max_tokens

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
