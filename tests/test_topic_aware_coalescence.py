# === NAVMAP v1 ===
# {
#   "module": "tests.test_topic_aware_coalescence",
#   "purpose": "Pytest coverage for topic aware coalescence scenarios",
#   "sections": [
#     {
#       "id": "_dummy_tokenizer",
#       "name": "_DummyTokenizer",
#       "anchor": "DUMM",
#       "kind": "class"
#     },
#     {
#       "id": "test_is_structural_boundary_detects_headings",
#       "name": "test_is_structural_boundary_detects_headings",
#       "anchor": "TISBD",
#       "kind": "function"
#     },
#     {
#       "id": "test_is_structural_boundary_detects_captions",
#       "name": "test_is_structural_boundary_detects_captions",
#       "anchor": "ISBD1",
#       "kind": "function"
#     },
#     {
#       "id": "test_is_structural_boundary_handles_non_markers",
#       "name": "test_is_structural_boundary_handles_non_markers",
#       "anchor": "TISBH",
#       "kind": "function"
#     },
#     {
#       "id": "test_topic_aware_coalescence_respects_section_boundaries",
#       "name": "test_topic_aware_coalescence_respects_section_boundaries",
#       "anchor": "TTACR",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

from pathlib import Path

import pytest

pytest.importorskip("transformers")

from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (
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
