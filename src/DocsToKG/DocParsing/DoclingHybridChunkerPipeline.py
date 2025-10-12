#!/usr/bin/env python3
"""
DocTags -> DoclingDocument -> HybridChunker (512 tokens, no overlap) -> JSONL

Input : /home/paul/DocsToKG/Data/DocTagsFiles
Output: /home/paul/DocsToKG/Data/ChunkedDocTagFiles

Behavior:
- No fallback code paths.
- No custom min-token coalescence: rely on HybridChunker split/merge.
- Inject picture captions (and optional annotations) into contextualized text.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import List, Tuple, Any

# ----- I/O roots -----
DEFAULT_IN_DIR = Path("/home/paul/DocsToKG/Data/DocTagsFiles")
DEFAULT_OUT_DIR = Path("/home/paul/DocsToKG/Data/ChunkedDocTagFiles")

# ----- Docling imports -----
from docling_core.types.doc.document import (
    DocTagsDocument,
    DoclingDocument,
)  # load_from_doctags, Doc model
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

# HF tokenizer for token counting + alignment to 512
from transformers import AutoTokenizer


# ---------- Custom picture serializer: puts CAPTION (+ optional annotations) into text ----------
class CaptionPlusAnnotationPictureSerializer(MarkdownPictureSerializer):
    """Emit picture caption and selected annotations into the chunk's text."""

    @override
    def serialize(
        self, *, item: PictureItem, doc_serializer: BaseDocSerializer, doc: _Doc, **_: Any
    ) -> SerializationResult:
        parts: List[str] = []

        # 1) Figure caption (if present)
        try:
            cap = (item.caption_text(doc) or "").strip()
            if cap:
                parts.append(f"Figure caption: {cap}")
        except Exception:
            pass

        # 2) Optional annotations (if your pipeline added them)
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
            parts.append("<!-- image -->")  # keep pictures visible in text stream

        text = doc_serializer.post_process(text="\n".join(parts))
        return create_ser_result(text=text, span_source=item)


class RichSerializerProvider(ChunkingSerializerProvider):
    """Use Markdown tables + caption-aware pictures; keep default params otherwise."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # tables -> Markdown
            picture_serializer=CaptionPlusAnnotationPictureSerializer(),  # pictures -> caption/annotations
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )


# ---------- Helpers ----------
def find_doctags_files(in_dir: Path) -> List[Path]:
    patterns = ("*.doctags", "*.doctag")
    files: List[Path] = []
    for pat in patterns:
        files.extend(in_dir.glob(pat))
    return sorted({p.resolve() for p in files})


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_docling_from_doctags_string(doc_name: str, doctags_text: str) -> DoclingDocument:
    # Build a DocTagsDocument, then load to DoclingDocument (official path)
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags_text], [None])
    return DoclingDocument.load_from_doctags(doctags_doc, document_name=doc_name)


def extract_refs_and_pages(chunk: BaseChunk) -> Tuple[List[str], List[int]]:
    refs: List[str] = []
    pages = set()
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


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_doctags_files(in_dir)
    if not files:
        print(f"[WARN] No *.doctags files found in {in_dir}")
        return

    # Tokenizer aligned with BERT family; cap contextualized text at 512 tokens
    hf_tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=512)

    # HybridChunker: token-aware; relies on built-in merge of undersized peers (no overlap)
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        serializer_provider=RichSerializerProvider(),
    )

    for path in files:
        name = path.stem
        text = read_utf8(path)

        # DocTags -> DoclingDocument
        doc = load_docling_from_doctags_string(doc_name=name, doctags_text=text)

        # Chunk and serialize (context-enriched text with headings/captions/tables)
        chunks = list(chunker.chunk(dl_doc=doc))

        out_path = out_dir / f"{name}.chunks.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for idx, ch in enumerate(chunks):
                ctx = chunker.contextualize(chunk=ch)
                n_tok = tokenizer.count_tokens(text=ctx)  # counts contextualized form
                refs, pages = extract_refs_and_pages(ch)
                rec = {
                    "doc_id": name,
                    "source_path": str(path),
                    "chunk_id": idx,
                    "num_tokens": n_tok,
                    "text": ctx,
                    "doc_items_refs": refs,
                    "page_nos": pages,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[OK] {name}: wrote {len(chunks)} chunks → {out_path}")


if __name__ == "__main__":
    main()
