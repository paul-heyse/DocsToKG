#!/usr/bin/env python3
"""
EmbedVectors.py

- Reads chunked JSONL from /home/paul/DocsToKG/Data/ChunkedDocTagFiles
- Ensures each chunk has a UUID (writes back to the same files)
- Emits vectors JSONL to /home/paul/DocsToKG/Data/Vectors

Uses local HF cache at /home/paul/hf-cache/:
  - Qwen3-Embedding-4B at   /home/paul/hf-cache/models/Qwen/Qwen3-Embedding-4B
  - SPLADE-v3 at            /home/paul/hf-cache/models/naver/splade-v3
"""

from __future__ import annotations
import argparse, json, math, os, re, uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

# ---- Fixed locations ----
HF_HOME = Path("/home/paul/hf-cache")
MODEL_ROOT = HF_HOME
QWEN_DIR = MODEL_ROOT / "Qwen" / "Qwen3-Embedding-4B"
SPLADE_DIR = MODEL_ROOT / "naver" / "splade-v3"

CHUNKS_DIR = Path("/home/paul/DocsToKG/Data/ChunkedDocTagFiles")
VECTORS_DIR = Path("/home/paul/DocsToKG/Data/Vectors")

# Make sure every lib (Transformers / HF Hub / Sentence-Transformers) honors this cache.
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_ROOT))

# ---- SPLADE (Sentence-Transformers SparseEncoder) ----
from sentence_transformers import (
    SparseEncoder,
)  # loads from local dir if given (cache_folder supported)

# ---- Qwen (vLLM offline API) ----
from vllm import (
    LLM,
    PoolingParams,
)  # PoolingParams(dimensions=...) selects output dim if model supports MRL

# ---- simple tokenizer for BM25 ----
import re

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


@dataclass
class Chunk:
    uuid: str
    text: str


def iter_chunk_files(d: Path) -> List[Path]:
    return sorted(d.glob("*.chunks.jsonl"))


def load_rows(p: Path) -> List[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_rows(p: Path, rows: List[dict]) -> None:
    tmp = p.with_suffix(".chunks.jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def ensure_uuid(rows: List[dict]) -> None:
    for r in rows:
        if not r.get("uuid"):
            r["uuid"] = str(uuid.uuid4())


def tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


# ---- BM25 (global) ----
@dataclass
class BM25Stats:
    N: int
    avgdl: float
    df: Dict[str, int]


def build_bm25_stats(chunks: Iterable[Chunk]) -> BM25Stats:
    N = 0
    total = 0
    df = Counter()
    for ch in chunks:
        N += 1
        toks = tokens(ch.text)
        total += len(toks)
        df.update(set(toks))
    avgdl = total / max(N, 1)
    return BM25Stats(N=N, avgdl=avgdl, df=dict(df))


def bm25_vector(
    text: str, stats: BM25Stats, k1: float = 1.5, b: float = 0.75
) -> Tuple[List[str], List[float]]:
    toks = tokens(text)
    dl = len(toks) or 1
    tf = Counter(toks)
    terms, weights = [], []
    for t, f in tf.items():
        n_qi = stats.df.get(t, 0)
        idf = math.log((stats.N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)  # RSJ-smoothed IDF
        denom = f + k1 * (1.0 - b + b * (dl / stats.avgdl))
        w = idf * (f * (k1 + 1.0)) / denom
        if w > 0:
            terms.append(t)
            weights.append(w)
    return terms, weights


# ---- SPLADE-v3 (GPU) ----
@dataclass
class SpladeCfg:
    model_dir: Path = SPLADE_DIR
    device: str = "cuda"
    batch_size: int = 32
    cache_folder: Path = MODEL_ROOT
    max_active_dims: int | None = None
    # Leave None to let HF/torch pick (usually SDPA); set to "flash_attention_2" if available.
    attn_impl: str | None = None


def splade_encode(cfg: SpladeCfg, texts: List[str]) -> Tuple[List[List[str]], List[List[float]]]:
    model_kwargs = {"attn_implementation": cfg.attn_impl} if cfg.attn_impl else {}
    try:
        enc = SparseEncoder(
            str(cfg.model_dir),
            device=cfg.device,
            cache_folder=str(cfg.cache_folder),
            model_kwargs=model_kwargs,
            local_files_only=True,  # use local paths only
        )
    except (ValueError, ImportError) as exc:
        if cfg.attn_impl == "flash_attention_2" and "Flash Attention 2" in str(exc):
            print("[SPLADE] FlashAttention 2 unavailable; retrying with standard attention.")
            enc = SparseEncoder(
                str(cfg.model_dir),
                device=cfg.device,
                cache_folder=str(cfg.cache_folder),
                model_kwargs={"attn_implementation": "sdpa"},
                local_files_only=True,
            )
        else:
            raise
    token_lists, weight_lists = [], []
    for i in range(0, len(texts), cfg.batch_size):
        batch = texts[i : i + cfg.batch_size]
        # returns a torch.sparse tensor (rows = batch)
        s = enc.encode(batch)  # shape: (B, |vocab|)
        for r in range(s.shape[0]):
            # decode all non-zeros for this row
            # decodes to list[(token, weight)] sorted by weight
            nnz = s[r].coalesce().values().numel()
            decoded = enc.decode(s[r], top_k=int(nnz))
            toks, wts = zip(*decoded) if decoded else ([], [])
            token_lists.append(list(toks))
            weight_lists.append([float(w) for w in wts])
    return token_lists, weight_lists


# ---- Qwen3-Embedding-4B via vLLM (2560-d) ----
@dataclass
class QwenCfg:
    model_dir: Path = QWEN_DIR
    dtype: str = "bfloat16"  # good default on Ada/Hopper
    tp: int = 1
    gpu_mem_util: float = 0.60
    batch_size: int = 32
    quantization: str | None = None  # 'awq' if you have an AWQ checkpoint


def qwen_embed(cfg: QwenCfg, texts: List[str]) -> List[List[float]]:
    llm = LLM(
        model=str(cfg.model_dir),  # local path
        task="embed",
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tp,
        gpu_memory_utilization=cfg.gpu_mem_util,
        quantization=cfg.quantization,  # None or 'awq' (if a matching AWQ checkpoint exists)
        download_dir=str(HF_HOME),  # belt & suspenders: keep any aux files in your cache
    )
    pool = PoolingParams(normalize=True)
    out: List[List[float]] = []
    for i in range(0, len(texts), cfg.batch_size):
        batch = texts[i : i + cfg.batch_size]
        res = llm.embed(batch, pooling_params=pool)
        for r in res:
            out.append([float(x) for x in r.outputs.embedding])
    return out


# ---- Main driver ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=VECTORS_DIR)
    # BM25 knobs
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    # SPLADE knobs
    ap.add_argument("--splade-batch", type=int, default=32)
    ap.add_argument("--splade-max-active-dims", type=int, default=None)
    ap.add_argument(
        "--splade-attn",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        help="Attention backend for SPLADE transformer (default: auto/SDPA).",
    )
    # Qwen knobs
    ap.add_argument("--qwen-dtype", type=str, default="bfloat16")
    ap.add_argument("--qwen-quant", type=str, default=None, choices=[None, "awq"])
    ap.add_argument("--qwen-batch", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_chunk_files(args.chunks_dir)
    if not files:
        print(f"[WARN] No *.chunks.jsonl in {args.chunks_dir}")
        return

    # ---- Pass 1: ensure UUIDs + assemble for global BM25 stats
    all_chunks: List[Chunk] = []
    print("[1/4] Ensuring UUIDs + global BM25 stats...")
    for f in tqdm(files, ncols=98):
        rows = load_rows(f)
        ensure_uuid(rows)
        save_rows(f, rows)
        for r in rows:
            all_chunks.append(Chunk(uuid=r["uuid"], text=r["text"]))
    stats = build_bm25_stats(all_chunks)
    print(f"    BM25: N={stats.N} avgdl={stats.avgdl:.2f}")

    texts = [c.text for c in all_chunks]

    # ---- Pass 2: SPLADE on GPU
    print("[2/4] SPLADE-v3 encoding...")
    attn_impl = None if args.splade_attn == "auto" else args.splade_attn
    spl_cfg = SpladeCfg(
        batch_size=args.splade_batch,
        max_active_dims=args.splade_max_active_dims,
        attn_impl=attn_impl,
    )
    spl_tokens, spl_weights = splade_encode(spl_cfg, texts)

    # ---- Pass 3: Qwen3-Embedding-4B via vLLM (MRL 2048d)
    print("[3/4] Qwen3-Embedding-4B encoding...")
    qcfg = QwenCfg(
        dtype=args.qwen_dtype,
        tp=int(args.tp),
        batch_size=int(args.qwen_batch),
        quantization=args.qwen_quant,
    )
    qvecs = qwen_embed(qcfg, texts)

    assert len(all_chunks) == len(spl_tokens) == len(qvecs)

    # ---- Pass 4: write vectors per source file
    print("[4/4] Writing vectors JSONL...")
    uuid2idx = {c.uuid: i for i, c in enumerate(all_chunks)}
    for f in tqdm(files, ncols=98):
        rows = load_rows(f)  # now with UUIDs
        out_path = out_dir / f"{f.stem.replace('.chunks', '')}.vectors.jsonl"
        with out_path.open("w", encoding="utf-8") as vf:
            for r in rows:
                i = uuid2idx[r["uuid"]]
                terms, weights = bm25_vector(
                    all_chunks[i].text, stats, k1=float(args.bm25_k1), b=float(args.bm25_b)
                )
                obj = {
                    "UUID": r["uuid"],
                    "BM25": {
                        "terms": terms,
                        "weights": weights,
                        "k1": float(args.bm25_k1),
                        "b": float(args.bm25_b),
                        "avgdl": stats.avgdl,
                        "N": stats.N,
                    },
                    "SpladeV3": {
                        "model_id": "naver/splade-v3",
                        "tokens": spl_tokens[i],
                        "weights": spl_weights[i],
                    },
                    "Qwen3-4B": {
                        "model_id": "Qwen/Qwen3-Embedding-4B",
                        "vector": qvecs[i],
                    },
                }
                vf.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[OK] {out_path.name}")


if __name__ == "__main__":
    main()
