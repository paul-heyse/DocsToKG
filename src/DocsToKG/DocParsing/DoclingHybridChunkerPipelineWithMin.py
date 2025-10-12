#!/usr/bin/env python3
# /home/paul/DocsToKG/src/DocsToKG/DocParsing/DoclingHybridChunkerPipeline.py

from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any

# ---------- I/O ----------
DEFAULT_IN_DIR = Path("/home/paul/DocsToKG/Data/DocTagsFiles")
DEFAULT_OUT_DIR = Path("/home/paul/DocsToKG/Data/ChunkedDocTagFiles")

# ---------- Docling imports ----------
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownTableSerializer,
    MarkdownPictureSerializer,
    MarkdownParams,
)
from docling_core.types.doc.document import (
    DoclingDocument as _Doc,
    PictureItem,
    PictureDescriptionData,
    PictureClassificationData,
    PictureMoleculeData,
)
from typing_extensions import override

from transformers import AutoTokenizer


# ---------- Picture serializer: inject CAPTIONS (+ optional annotations) ----------
class CaptionPlusAnnotationPictureSerializer(MarkdownPictureSerializer):
    @override
    def serialize(
        self, *, item: PictureItem, doc_serializer: BaseDocSerializer, doc: _Doc, **_: Any
    ) -> SerializationResult:
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
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # tables -> Markdown
            picture_serializer=CaptionPlusAnnotationPictureSerializer(),  # pictures -> caption/annots
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )


# ---------- Helpers ----------
def find_doctags_files(in_dir: Path) -> List[Path]:
    files = []
    for pat in ("*.doctags", "*.doctag"):
        files.extend(in_dir.glob(pat))
    return sorted({p.resolve() for p in files})


def read_utf8(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def build_doc(doc_name: str, doctags_text: str) -> DoclingDocument:
    dt = DocTagsDocument.from_doctags_and_image_pairs([doctags_text], [None])
    return DoclingDocument.load_from_doctags(dt, document_name=doc_name)


def extract_refs_and_pages(chunk: BaseChunk) -> Tuple[List[str], List[int]]:
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
    text: str
    n_tok: int
    src_idxs: List[int]
    refs: List[str]
    pages: List[int]


def merge_rec(a: Rec, b: Rec, tokenizer: HuggingFaceTokenizer) -> Rec:
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
    """
    Strategy:
      • Identify contiguous runs where EVERY chunk < min_tokens.
      • Within each run, greedily pack neighbors to form groups that hit >= min_tokens
        without exceeding max_tokens (keeps small chunks together).
      • If the final group in a run is still < min_tokens:
           - prefer merging into the previous group from the SAME RUN if it fits,
           - else merge into the smaller adjacent BIG neighbor (left/right) only if it fits,
           - else keep as-is (rare).
      • Chunks >= min_tokens outside runs are left untouched (avoids skewing well-formed 300–500 token chunks).
    """
    out: List[Rec] = []
    i, N = 0, len(records)

    def is_small(idx: int) -> bool:
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
