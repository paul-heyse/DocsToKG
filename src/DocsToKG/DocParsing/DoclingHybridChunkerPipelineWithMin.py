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
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Third-party imports
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument, DocTagsDocument
from transformers import AutoTokenizer

from DocsToKG.DocParsing._common import (
    compute_content_hash,
    data_chunks,
    data_doctags,
    data_manifests,
    detect_data_root,
    get_logger,
    iter_doctags,
    load_manifest_index,
    manifest_append,
)
from DocsToKG.DocParsing.serializers import RichSerializerProvider

# ---------- Defaults ----------
DEFAULT_DATA_ROOT = detect_data_root()
DEFAULT_IN_DIR = data_doctags(DEFAULT_DATA_ROOT)
DEFAULT_OUT_DIR = data_chunks(DEFAULT_DATA_ROOT)
MANIFEST_STAGE = "chunks"

# ---------- Helpers ----------
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

    Raises:
        None
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

    Examples:
        >>> rec = Rec(text="Example", n_tok=5, src_idxs=[0], refs=[], pages=[1])
        >>> rec.n_tok
        5
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
    """CLI driver that chunks DocTags files and enforces minimum token thresholds."""

    logger = get_logger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Override DocsToKG Data directory. Defaults to auto-detection or "
            "$DOCSTOKG_DATA_ROOT."
        ),
    )
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--min-tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip DocTags whose chunk outputs already exist with matching hash",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even when resume criteria are satisfied",
    )
    args = ap.parse_args()

    data_root_override = args.data_root
    resolved_data_root = (
        detect_data_root(data_root_override)
        if data_root_override is not None
        else DEFAULT_DATA_ROOT
    )

    if data_root_override is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved_data_root)

    data_manifests(resolved_data_root)

    in_dir = (
        data_doctags(resolved_data_root)
        if args.in_dir == DEFAULT_IN_DIR and data_root_override is not None
        else args.in_dir
    )
    out_dir = (
        data_chunks(resolved_data_root)
        if args.out_dir == DEFAULT_OUT_DIR and data_root_override is not None
        else args.out_dir
    )

    logger.info(
        "Chunking configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_data_root),
                "input_dir": str(in_dir),
                "output_dir": str(out_dir),
                "min_tokens": args.min_tokens,
                "max_tokens": args.max_tokens,
            }
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_doctags(in_dir))
    if not files:
        logger.warning(
            "No .doctags files found",
            extra={"extra_fields": {"input_dir": str(in_dir)}},
        )
        return

    if args.force:
        logger.info("Force mode: reprocessing all DocTags files")
    elif args.resume:
        logger.info("Resume mode enabled: unchanged inputs will be skipped")

    manifest_index = load_manifest_index(MANIFEST_STAGE, resolved_data_root) if args.resume else {}

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
        rel_id = path.relative_to(in_dir).as_posix()
        name = path.stem
        out_path = out_dir / f"{name}.chunks.jsonl"
        input_hash = compute_content_hash(path)
        manifest_entry = manifest_index.get(rel_id)

        if (
            args.resume
            and not args.force
            and out_path.exists()
            and manifest_entry
            and manifest_entry.get("input_hash") == input_hash
        ):
            logger.info("Skipping %s: output exists and input unchanged", rel_id)
            manifest_append(
                stage=MANIFEST_STAGE,
                doc_id=rel_id,
                status="skip",
                duration_s=0.0,
                schema_version="docparse/1.1.0",
                input_path=str(path),
                input_hash=input_hash,
                output_path=str(out_path),
            )
            continue

        start = time.perf_counter()
        try:
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

            duration = time.perf_counter() - start
            logger.info(
                "Chunk file written",
                extra={
                    "extra_fields": {
                        "doc_id": name,
                        "chunks": len(final_recs),
                        "output_file": out_path.name,
                    }
                },
            )
            manifest_append(
                stage=MANIFEST_STAGE,
                doc_id=rel_id,
                status="success",
                duration_s=round(duration, 3),
                schema_version="docparse/1.1.0",
                input_path=str(path),
                input_hash=input_hash,
                output_path=str(out_path),
                chunk_count=len(final_recs),
            )
        except Exception as exc:
            duration = time.perf_counter() - start
            manifest_append(
                stage=MANIFEST_STAGE,
                doc_id=rel_id,
                status="failure",
                duration_s=round(duration, 3),
                schema_version="docparse/1.1.0",
                input_path=str(path),
                input_hash=input_hash,
                output_path=str(out_path),
                error=str(exc),
            )
            raise


if __name__ == "__main__":
    main()
