# === NAVMAP v1 ===
# {
#   "module": "tests.test_topic_aware_coalescence",
#   "purpose": "Pytest coverage for topic aware coalescence scenarios",
#   "sections": [
#     {
#       "id": "dummytokenizer",
#       "name": "_DummyTokenizer",
#       "anchor": "class-dummytokenizer",
#       "kind": "class"
#     },
#     {
#       "id": "test-is-structural-boundary-detects-headings",
#       "name": "test_is_structural_boundary_detects_headings",
#       "anchor": "function-test-is-structural-boundary-detects-headings",
#       "kind": "function"
#     },
#     {
#       "id": "test-is-structural-boundary-detects-captions",
#       "name": "test_is_structural_boundary_detects_captions",
#       "anchor": "function-test-is-structural-boundary-detects-captions",
#       "kind": "function"
#     },
#     {
#       "id": "test-is-structural-boundary-handles-non-markers",
#       "name": "test_is_structural_boundary_handles_non_markers",
#       "anchor": "function-test-is-structural-boundary-handles-non-markers",
#       "kind": "function"
#     },
#     {
#       "id": "test-topic-aware-coalescence-respects-section-boundaries",
#       "name": "test_topic_aware_coalescence_respects_section_boundaries",
#       "anchor": "function-test-topic-aware-coalescence-respects-section-boundaries",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

from pathlib import Path
from typing import List

import pytest

pytest.importorskip("transformers")

from DocsToKG.DocParsing.chunking import (
    Rec,
    coalesce_small_runs,
    is_structural_boundary,
)


class _DummyTokenizer:
    def count_tokens(self, text: str) -> int:
        return len([tok for tok in text.replace("\n", " ").split() if tok])


@pytest.mark.parametrize(
    "text",
    ["# Heading", "## Subheading", "### Deep", "#### Outline"],
)
# --- Test Cases ---


def test_is_structural_boundary_detects_headings(text: str) -> None:
    rec = Rec(text=text, n_tok=1, src_idxs=[], refs=[], pages=[])
    assert is_structural_boundary(rec)


@pytest.mark.parametrize(
    "text",
    [
        "Figure caption: Example figure",
        "Table: Sample data",
        "Picture description: Structure",
        "<!-- image --> Inline alt",
    ],
)
def test_is_structural_boundary_detects_captions(text: str) -> None:
    rec = Rec(text=text, n_tok=1, src_idxs=[], refs=[], pages=[])
    assert is_structural_boundary(rec)


@pytest.mark.parametrize("text", ["", "   ", "Regular paragraph", "Intro text"])
def test_is_structural_boundary_handles_non_markers(text: str) -> None:
    rec = Rec(text=text, n_tok=1, src_idxs=[], refs=[], pages=[])
    assert not is_structural_boundary(rec)


def test_is_structural_boundary_supports_custom_markers() -> None:
    rec = Rec(text="Article 1. Scope", n_tok=2, src_idxs=[], refs=[], pages=[])
    assert not is_structural_boundary(rec)
    assert is_structural_boundary(rec, heading_markers=("Article ",))


def test_topic_aware_coalescence_respects_section_boundaries() -> None:
    sample_path = Path("tests/data/docparsing/topic_aware_sample.doctags")
    raw = sample_path.read_text(encoding="utf-8")
    segments = [segment.strip() for segment in raw.split("\n\n---\n\n") if segment.strip()]

    tokenizer = _DummyTokenizer()
    records = [
        Rec(
            text=segment,
            n_tok=tokenizer.count_tokens(segment),
            src_idxs=[index],
            refs=[],
            pages=[],
        )
        for index, segment in enumerate(segments)
    ]

    result = coalesce_small_runs(records, tokenizer, min_tokens=100, max_tokens=180)

    assert len(result) == 2
    first_chunk, second_chunk = result

    assert "# Section One" in first_chunk.text
    assert "intro0" in first_chunk.text

    assert "# Section Two" in second_chunk.text
    assert second_chunk.text.lstrip().startswith("# Section Two")
    assert not second_chunk.text.split("# Section Two")[0].strip()


def test_coalesce_small_runs_soft_barrier_override() -> None:
    tokenizer = _DummyTokenizer()
    body_text = " ".join(["body"] * 35)
    heading_text = "# Section" + " " + " ".join(["item"] * 10)

    def _records() -> List[Rec]:
        return [
            Rec(
                text=body_text,
                n_tok=tokenizer.count_tokens(body_text),
                src_idxs=[0],
                refs=[],
                pages=[],
            ),
            Rec(
                text=heading_text,
                n_tok=tokenizer.count_tokens(heading_text),
                src_idxs=[1],
                refs=[],
                pages=[],
            ),
        ]

    default_result = coalesce_small_runs(
        records=_records(),
        tokenizer=tokenizer,
        min_tokens=40,
        max_tokens=70,
    )
    assert len(default_result) == 2

    relaxed_result = coalesce_small_runs(
        records=_records(),
        tokenizer=tokenizer,
        min_tokens=40,
        max_tokens=70,
        soft_barrier_margin=0,
    )
    assert len(relaxed_result) == 1
