#!/usr/bin/env python3
"""
Docling Hybrid Chunker with Minimum Token Coalescence

Transforms DocTags documents into chunked records while ensuring short runs of
chunks are merged to satisfy minimum token thresholds required by downstream
embedding pipelines.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

# Third-party imports
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import (
    DoclingDocument,
    DocTagsDocument,
    PictureClassificationData,
    PictureDescriptionData,
    PictureItem,
    PictureMoleculeData,
)
from docling_core.types.doc.document import (
    DoclingDocument as _Doc,
)
from transformers import AutoTokenizer
from typing_extensions import override

# ---------- I/O ----------
DEFAULT_IN_DIR = Path("/home/paul/DocsToKG/Data/DocTagsFiles")
DEFAULT_OUT_DIR = Path("/home/paul/DocsToKG/Data/ChunkedDocTagFiles")


# ---------- Picture serializer: inject CAPTIONS (+ optional annotations) ----------
class CaptionPlusAnnotationPictureSerializer(MarkdownPictureSerializer):
    """Serialize picture items with captions and rich annotation metadata."""

    @override
    def serialize(
        self, *, item: PictureItem, doc_serializer: BaseDocSerializer, doc: _Doc, **_: Any
    ) -> SerializationResult:
        """Render picture metadata into Markdown-friendly text.

        Args:
            item: Picture element emitted by Docling.
            doc_serializer: Parent serializer responsible for post-processing.
            doc: Full Docling document containing the picture context.

        Returns:
            SerializationResult capturing the rendered string and provenance.
        """
        parts: List[str] = []
        try:
            cap = (item.caption_text(doc) or "").strip()
            if cap:
                parts.append(f"Figure caption: {cap}")
        except Exception:
            pass
        try:
            for ann in item.annotations or []:
                if isinstance(ann, PictureDescriptionData) and ann.text:
                    parts.append(f"Picture description: {ann.text}")
                elif isinstance(ann, PictureClassificationData) and ann.predicted_classes:
                    parts.append(f"Picture type: {ann.predicted_classes[0].class_name}")
                elif isinstance(ann, PictureMoleculeData) and ann.smi:
                    parts.append(f"SMILES: {ann.smi}")
        except Exception:
            pass
        if not parts:
            parts.append("<!-- image -->")
        text = doc_serializer.post_process(text="\n".join(parts))
        return create_ser_result(text=text, span_source=item)


class RichSerializerProvider(ChunkingSerializerProvider):
    """Provide a serializer that augments tables and pictures with Markdown."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Construct a ChunkingDocSerializer tailored for DocTags documents.

        Args:
            doc: Docling document that will be serialized into chunk text.

        Returns:
            Configured ChunkingDocSerializer instance.
        """
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # tables -> Markdown
            picture_serializer=CaptionPlusAnnotationPictureSerializer(),  # pictures -> caption/annots
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )


# ---------- Helpers ----------
def find_doctags_files(in_dir: Path) -> List[Path]:
    """Discover `.doctags` artifacts within a directory.

    Args:
        in_dir: Directory containing DocTags outputs.

    Returns:
        Sorted list of unique DocTags file paths.
    """
    files = []
    for pat in ("*.doctags", "*.doctag"):
        files.extend(in_dir.glob(pat))
    return sorted({p.resolve() for p in files})


def read_utf8(p: Path) -> str:
    """Load text from disk using UTF-8 with replacement for invalid bytes.

    Args:
        p: Path to the text file.

    Returns:
        String contents of the file.
    """
    return p.read_text(encoding="utf-8", errors="replace")


def build_doc(doc_name: str, doctags_text: str) -> DoclingDocument:
    """Construct a Docling document from serialized DocTags text.

    Args:
        doc_name: Human-readable document identifier for logging.
        doctags_text: Serialized DocTags payload.

    Returns:
        Loaded DoclingDocument ready for chunking.
    """
    dt = DocTagsDocument.from_doctags_and_image_pairs([doctags_text], [None])
    return DoclingDocument.load_from_doctags(dt, document_name=doc_name)


def extract_refs_and_pages(chunk: BaseChunk) -> Tuple[List[str], List[int]]:
    """Collect self-references and page numbers associated with a chunk.

    Args:
        chunk: Chunk object produced by the hybrid chunker.

    Returns:
        Tuple containing a list of reference identifiers and sorted page numbers.
    """
    refs, pages = [], set()
    try:
        for it in chunk.meta.doc_items:
            if getattr(it, "self_ref", None):
                refs.append(it.self_ref)
            for pv in getattr(it, "prov", []) or []:
                pn = getattr(pv, "page_no", None)
                if pn is not None:
                    pages.add(int(pn))
    except Exception:
        pass
    return refs, sorted(pages)


@dataclass
class Rec:
    """Intermediate record tracking chunk text and provenance.

    Attributes:
        text: Chunk text content.
        n_tok: Token count computed by the tokenizer.
        src_idxs: Source chunk indices contributing to this record.
        refs: List of inline reference identifiers.
        pages: Page numbers associated with the chunk.
    """

    text: str
    n_tok: int
    src_idxs: List[int]
    refs: List[str]
    pages: List[int]


def merge_rec(a: Rec, b: Rec, tokenizer: HuggingFaceTokenizer) -> Rec:
    """Merge two chunk records, updating token counts and provenance metadata.

    Args:
        a: First record to merge.
        b: Second record to merge.
        tokenizer: Tokenizer used to recompute token counts for combined text.

    Returns:
        New `Rec` instance containing fused text, token counts, and metadata.
    """
    text = a.text + "\n\n" + b.text
    n_tok = tokenizer.count_tokens(text=text)
    refs = a.refs + [r for r in b.refs if r not in a.refs]
    pages = sorted(set(a.pages).union(b.pages))
    return Rec(text=text, n_tok=n_tok, src_idxs=a.src_idxs + b.src_idxs, refs=refs, pages=pages)


# ---------- Smart coalescence of SMALL-RUNS (< min_tokens) ----------
def coalesce_small_runs(
    records: List[Rec],
    tokenizer: HuggingFaceTokenizer,
    min_tokens: int = 256,
    max_tokens: int = 512,
) -> List[Rec]:
    """Merge contiguous short chunks until they satisfy minimum token thresholds.

    Args:
        records: Ordered list of chunk records to normalize.
        tokenizer: Tokenizer used to recompute token counts for merged chunks.
        min_tokens: Target minimum tokens per chunk after coalescing.
        max_tokens: Hard ceiling to avoid producing overly large chunks.

    Returns:
        New list of records where small runs are merged while preserving order.

    Note:
        Strategy:
            • Identify contiguous runs where every chunk has fewer than `min_tokens`.
            • Greedily pack neighbors within a run to exceed `min_tokens` without
              surpassing `max_tokens`.
            • Merge trailing fragments into adjacent groups when possible,
              preferring same-run neighbors to maintain topical cohesion.
            • Leave chunks outside small runs unchanged.
    """
    out: List[Rec] = []
    i, N = 0, len(records)

    def is_small(idx: int) -> bool:
        """Return True when the chunk at `idx` is below the minimum token threshold.

        Args:
            idx: Index of the chunk under evaluation.

        Returns:
            True if the chunk length is less than `min_tokens`, else False.
        """
        return records[idx].n_tok < min_tokens

    while i < N:
        if not is_small(i):
            out.append(records[i])
            i += 1
            continue

        # find small-run [s, e)
        s = i
        while i < N and is_small(i):
            i += 1
        e = i

        # pack within [s, e)
        groups: List[Rec] = []
        j = s
        while j < e:
            g = records[j]
            k = j + 1
            while k < e and g.n_tok < min_tokens and (g.n_tok + records[k].n_tok) <= max_tokens:
                g = merge_rec(g, records[k], tokenizer)
                k += 1
            groups.append(g)
            j = k

        # handle tiny trailing group if present
        trailing_small = len(groups) >= 1 and groups[-1].n_tok < min_tokens
        # add all but possibly the last (we may re-route it)
        for g in groups[:-1] if trailing_small else groups:
            out.append(g)

        if trailing_small:
            tail = groups[-1]
            # first try merge into previous group FROM SAME RUN if it fits
            if (
                out
                and out[-1].src_idxs
                and (out[-1].n_tok + tail.n_tok) <= max_tokens
                and max(out[-1].src_idxs) < s
            ):  # previous output is from before this run, not same run
                # The condition above prevents merging into a big chunk; we WANT same-run,
                # so we check if the last out element is from this run by src idx range.
                # If it's not, we won't merge here; instead, try intra-run previous if available:
                pass

            # better: try intra-run previous explicitly if exists
            if len(groups) >= 2 and (groups[-2].n_tok + tail.n_tok) <= max_tokens:
                merged = merge_rec(groups[-2], tail, tokenizer)
                # replace the last emitted group (groups[-2]) in 'out'
                if out and out[-1].src_idxs == groups[-2].src_idxs:
                    out[-1] = merged
                else:
                    # if not last in out (e.g., unusual ordering), append
                    out.append(merged)

            else:
                # prefer merging with the smaller big neighbor (left or right) if it FITS
                left_can = len(out) >= 1
                right_can = e < N  # next big chunk exists
                left_ok = left_can and (out[-1].n_tok + tail.n_tok) <= max_tokens
                right_ok = right_can and (records[e].n_tok + tail.n_tok) <= max_tokens

                if left_ok and right_ok:
                    # choose the neighbor with smaller size to minimize skew
                    if out[-1].n_tok <= records[e].n_tok:
                        out[-1] = merge_rec(out[-1], tail, tokenizer)
                    else:
                        records[e] = merge_rec(tail, records[e], tokenizer)  # pre-merge right
                elif left_ok:
                    out[-1] = merge_rec(out[-1], tail, tokenizer)
                elif right_ok:
                    records[e] = merge_rec(tail, records[e], tokenizer)
                else:
                    # no good fit—emit as-is rather than exceed max_tokens
                    out.append(tail)

    return out


# ---------- Main ----------
def main():
    """CLI driver that chunks DocTags files and enforces minimum token thresholds.

    Args:
        None

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--min-tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    in_dir, out_dir = args.in_dir, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_doctags_files(in_dir)
    if not files:
        print(f"[WARN] No *.doctags files found in {in_dir}")
        return

    # Tokenizer (BERT family) → 512 cap applied to contextualized text
    hf = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf, max_tokens=args.max_tokens)

    # HybridChunker: token-aware split + peer-merge; no overlap
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        serializer_provider=RichSerializerProvider(),
    )

    for path in files:
        name = path.stem
        doctags_text = read_utf8(path)
        doc = build_doc(doc_name=name, doctags_text=doctags_text)

        # Stage 1: Docling chunking
        chunks = list(chunker.chunk(dl_doc=doc))

        # Stage 2: materialize contextualized text + metadata
        recs: List[Rec] = []
        for idx, ch in enumerate(chunks):
            text = chunker.contextualize(ch)
            n_tok = tokenizer.count_tokens(text=text)
            refs, pages = extract_refs_and_pages(ch)
            recs.append(Rec(text=text, n_tok=n_tok, src_idxs=[idx], refs=refs, pages=pages))

        # Stage 3: smart coalescence of contiguous small runs
        final_recs = coalesce_small_runs(
            records=recs,
            tokenizer=tokenizer,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )

        # Stage 4: write JSONL
        out_path = out_dir / f"{name}.chunks.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for cid, r in enumerate(final_recs):
                obj = {
                    "doc_id": name,
                    "source_path": str(path),
                    "chunk_id": cid,
                    "source_chunk_idxs": r.src_idxs,
                    "num_tokens": r.n_tok,
                    "text": r.text,
                    "doc_items_refs": r.refs,
                    "page_nos": r.pages,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[OK] {name}: {len(final_recs)} chunks  →  {out_path.name}")


if __name__ == "__main__":
    main()
