#!/usr/bin/env python3
"""
Docling Hybrid Chunker with Minimum Token Coalescence

Transforms DocTags documents into chunked records with topic-aware coalescence.

Tokenizer Alignment:
    The default tokenizer (``Qwen/Qwen3-Embedding-4B``) aligns with the dense
    embedder used by the embeddings pipeline. When experimenting with other
    tokenizers (for example, legacy BERT models), run the calibration utility
    beforehand to understand token count deltas::

        python scripts/calibrate_tokenizers.py --doctags-dir Data/DocTagsFiles

    The calibration script reports relative token ratios and recommends
    adjustments to ``--min-tokens`` so chunk sizes remain compatible with the
    embedding stage.
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

SOFT_BARRIER_MARGIN = 64

# ---------- Defaults ----------
DEFAULT_DATA_ROOT = detect_data_root()
DEFAULT_IN_DIR = data_doctags(DEFAULT_DATA_ROOT)
DEFAULT_OUT_DIR = data_chunks(DEFAULT_DATA_ROOT)
MANIFEST_STAGE = "chunks"

_LOGGER = get_logger(__name__)

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


# ---------- Topic-aware boundary detection ----------
def is_structural_boundary(rec: Rec) -> bool:
    """Detect whether a chunk begins with a structural heading or caption marker.

    Args:
        rec: Chunk record to inspect.

    Returns:
        ``True`` when ``rec.text`` starts with a heading indicator (``#``) or a
        recognised caption prefix, otherwise ``False``.

    Examples:
        >>> is_structural_boundary(Rec(text="# Introduction", n_tok=2, src_idxs=[], refs=[], pages=[]))
        True
        >>> is_structural_boundary(Rec(text="Regular paragraph", n_tok=2, src_idxs=[], refs=[], pages=[]))
        False
    """

    text = rec.text.lstrip()
    if text.startswith("#"):
        return True

    caption_markers = (
        "Figure caption:",
        "Table:",
        "Picture description:",
        "<!-- image -->",
    )
    return any(text.startswith(marker) for marker in caption_markers)


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
            while k < e and g.n_tok < min_tokens:
                next_rec = records[k]
                combined_size = g.n_tok + next_rec.n_tok
                threshold = max_tokens - SOFT_BARRIER_MARGIN

                if is_structural_boundary(next_rec) and combined_size > threshold:
                    _LOGGER.debug(
                        "Soft barrier at chunk %s: boundary detected, combined size %s > %s",
                        k,
                        combined_size,
                        threshold,
                        extra={
                            "extra_fields": {
                                "combined_tokens": combined_size,
                                "threshold": threshold,
                                "max_tokens": max_tokens,
                                "context": "intra_run",
                            }
                        },
                    )
                    break

                if combined_size <= max_tokens:
                    g = merge_rec(g, next_rec, tokenizer)
                    k += 1
                else:
                    break
            groups.append(g)
            j = k

        # handle tiny trailing group if present
        trailing_small = len(groups) >= 1 and groups[-1].n_tok < min_tokens
        # add all but possibly the last (we may re-route it)
        for g in groups[:-1] if trailing_small else groups:
            out.append(g)

        if trailing_small:
            tail = groups[-1]
            threshold = max_tokens - SOFT_BARRIER_MARGIN

            if len(groups) >= 2:
                combined_size = groups[-2].n_tok + tail.n_tok
                if is_structural_boundary(tail) and combined_size > threshold:
                    _LOGGER.debug(
                        "Soft barrier prevented intra-run merge: combined size %s > %s",
                        combined_size,
                        threshold,
                        extra={
                            "extra_fields": {
                                "combined_tokens": combined_size,
                                "threshold": threshold,
                                "max_tokens": max_tokens,
                                "context": "intra_run_tail",
                            }
                        },
                    )
                elif combined_size <= max_tokens:
                    merged = merge_rec(groups[-2], tail, tokenizer)
                    if out and out[-1].src_idxs == groups[-2].src_idxs:
                        out[-1] = merged
                    else:
                        out.append(merged)
                    continue

            left_can = len(out) >= 1
            right_can = e < N
            left_ok = False
            right_ok = False

            if left_can:
                combined_size = out[-1].n_tok + tail.n_tok
                if is_structural_boundary(tail) and combined_size > threshold:
                    _LOGGER.debug(
                        "Soft barrier prevented left merge: combined size %s > %s",
                        combined_size,
                        threshold,
                        extra={
                            "extra_fields": {
                                "combined_tokens": combined_size,
                                "threshold": threshold,
                                "max_tokens": max_tokens,
                                "context": "left_neighbor",
                            }
                        },
                    )
                else:
                    left_ok = combined_size <= max_tokens

            if right_can:
                combined_size = records[e].n_tok + tail.n_tok
                if is_structural_boundary(tail) and combined_size > threshold:
                    _LOGGER.debug(
                        "Soft barrier prevented right merge: combined size %s > %s",
                        combined_size,
                        threshold,
                        extra={
                            "extra_fields": {
                                "combined_tokens": combined_size,
                                "threshold": threshold,
                                "max_tokens": max_tokens,
                                "context": "right_neighbor",
                            }
                        },
                    )
                else:
                    right_ok = combined_size <= max_tokens

            if left_ok and right_ok:
                if out[-1].n_tok <= records[e].n_tok:
                    out[-1] = merge_rec(out[-1], tail, tokenizer)
                else:
                    records[e] = merge_rec(tail, records[e], tokenizer)
            elif left_ok:
                out[-1] = merge_rec(out[-1], tail, tokenizer)
            elif right_ok:
                records[e] = merge_rec(tail, records[e], tokenizer)
            else:
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
        "--tokenizer-model",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="HuggingFace tokenizer model (default aligns with dense embedder)",
    )
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

    tokenizer_model = args.tokenizer_model
    logger.info(
        "Loading tokenizer",
        extra={"extra_fields": {"tokenizer_model": tokenizer_model}},
    )
    hf = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
    if "bert" in tokenizer_model.lower():
        logger.warning(
            "BERT tokenizer may not align with Qwen embedder. Consider running "
            "scripts/calibrate_tokenizers.py or using --tokenizer-model "
            "Qwen/Qwen3-Embedding-4B.",
            extra={"extra_fields": {"tokenizer_model": tokenizer_model}},
        )
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
