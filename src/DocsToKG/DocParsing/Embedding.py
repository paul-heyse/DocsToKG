#!/usr/bin/env python3
"""
EmbedVectors.py

- Reads chunked JSONL files from /home/paul/DocsToKG/Data/ChunkedDocTagFiles
- Ensures each chunk has a stable UUID (writes back to the chunk files)
- Emits vectors JSONL to   /home/paul/DocsToKG/Data/Vectors

For each chunk record in vectors JSONL:
{
  "UUID": "...",
  "BM25": {
    "terms": [...], "weights": [...],
    "k1": 1.5, "b": 0.75, "avgdl": <float>, "N": <int>
  },
  "SpladeV3": {
    "model_id": "naver/splade-v3",
    "tokens": [...], "weights": [...]
  },
  "Qwen3-4B": {
    "model_id": "Qwen/Qwen3-Embedding-4B",
    "dim": 2048,
    "vector": [...]
  }
}
"""

from __future__ import annotations

import argparse, json, math, os, uuid, re, sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import torch
from tqdm import tqdm

# ---------- Paths ----------
CHUNKS_DIR = Path("/home/paul/DocsToKG/Data/ChunkedDocTagFiles")
VECTORS_DIR = Path("/home/paul/DocsToKG/Data/Vectors")

# ---------- SPLADE (Sentence-Transformers SparseEncoder) ----------
from sentence_transformers import SparseEncoder

# ---------- Qwen (vLLM, embedding task + matryoshka dims) ----------
from vllm import LLM, PoolingParams

# ---------- Tokenization for BM25 ----------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


# ---------- Records ----------
@dataclass
class Chunk:
    uuid: str
    text: str


# ---------- Helpers ----------
def iter_chunk_files(in_dir: Path) -> List[Path]:
    return sorted(in_dir.glob("*.chunks.jsonl"))


def load_chunks(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_chunks(path: Path, rows: List[dict]) -> None:
    tmp = path.with_suffix(".chunks.jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def ensure_uuid(rows: List[dict]) -> None:
    for r in rows:
        if "uuid" not in r or not r["uuid"]:
            r["uuid"] = str(uuid.uuid4())


def tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


# ---------- BM25 corpus stats ----------
@dataclass
class BM25Stats:
    N: int
    avgdl: float
    df: Dict[str, int]


def build_bm25_stats(all_chunks: Iterable[Chunk]) -> BM25Stats:
    N = 0
    total_len = 0
    df = Counter()
    for ch in all_chunks:
        N += 1
        toks = tokens(ch.text)
        total_len += len(toks)
        df.update(set(toks))  # doc frequency
    avgdl = total_len / max(N, 1)
    return BM25Stats(N=N, avgdl=avgdl, df=dict(df))


def bm25_vector(
    text: str, stats: BM25Stats, k1: float = 1.5, b: float = 0.75
) -> Tuple[List[str], List[float]]:
    """Return (terms, weights) sparse BM25 vector for a document, using corpus-wide IDF."""
    toks = tokens(text)
    dl = len(toks) or 1
    tf = Counter(toks)

    terms, weights = [], []
    for t, f in tf.items():
        n_qi = stats.df.get(t, 0)
        # IDF with +0.5 smoothing (Robertson–Spärck Jones)
        idf = math.log((stats.N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
        denom = f + k1 * (1.0 - b + b * (dl / stats.avgdl))
        w = idf * (f * (k1 + 1.0)) / denom
        if w > 0:
            terms.append(t)
            weights.append(w)
    return terms, weights


# ---------- SPLADE v3 ----------
@dataclass
class SpladeConfig:
    model_id: str = "naver/splade-v3"
    device: str = "cuda"
    batch_size: int = 32
    max_active_dims: int | None = None  # e.g., 4096 to cap nnz per vector; None = no cap


def splade_encode(cfg: SpladeConfig, texts: List[str]) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Returns per-text tokens & weights (only non-zeros).
    """
    enc = SparseEncoder(cfg.model_id, device=cfg.device)
    # Optionally cap the number of active dims per vector
    kwargs = {}
    if cfg.max_active_dims is not None:
        kwargs["max_active_dims"] = int(cfg.max_active_dims)

    token_lists: List[List[str]] = []
    weight_lists: List[List[float]] = []

    for i in range(0, len(texts), cfg.batch_size):
        batch = texts[i : i + cfg.batch_size]
        sparse_emb = enc.encode(batch, **kwargs)  # torch.sparse_coo_tensor
        # Map indices -> tokens/weights for each row
        for row in range(sparse_emb.shape[0]):
            row_coo = sparse_emb[row].coalesce()
            idx = row_coo.indices().squeeze(0).tolist()
            val = row_coo.values().tolist()
            # decode indices to tokens (vectorized decode is available via enc.decode, but we want all nnz)
            # enc.decode(...) returns top_k only; here we invert tokenizer vocab:
            # For full mapping, get tokenizer vocab once:
            pass
        # Efficient per-batch mapping using encoder.decode with large top_k:
        # get nnz counts to choose top_k
        stats = enc.sparsity(sparse_emb)
        active = int(stats["active_dims"]) if isinstance(stats, dict) else None  # average per batch
        top_k = None
        # fallback: for each row, choose its nnz length
        for row in range(sparse_emb.shape[0]):
            row_coo = sparse_emb[row].coalesce()
            nnz = row_coo.values().numel()
            top_k = nnz
            decoded = enc.decode(sparse_emb[row], top_k=top_k)  # List[(token, weight)]
            toks, wts = zip(*decoded) if decoded else ([], [])
            token_lists.append(list(toks))
            weight_lists.append([float(w) for w in wts])

    return token_lists, weight_lists


# ---------- Qwen3-Embedding-4B via vLLM ----------
@dataclass
class QwenCfg:
    model_id: str = "Qwen/Qwen3-Embedding-4B"
    dim: int = 2048
    dtype: str = "bfloat16"  # or "float16"
    tensor_parallel_size: int = 1
    gpu_mem_util: float = 0.90
    batch_size: int = 64
    quantization: str | None = None  # e.g., "awq"


def qwen_embed(cfg: QwenCfg, texts: List[str]) -> List[List[float]]:
    # vLLM offline API
    llm = LLM(
        model=cfg.model_id,
        task="embed",
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_mem_util,
        quantization=cfg.quantization,  # None or "awq"
    )
    pool = PoolingParams(normalize=True, dimensions=cfg.dim)
    vecs: List[List[float]] = []
    for i in range(0, len(texts), cfg.batch_size):
        batch = texts[i : i + cfg.batch_size]
        outs = llm.embed(batch, pooling_params=pool)
        for o in outs:
            vecs.append([float(x) for x in o.outputs.embedding])
    return vecs


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=VECTORS_DIR)
    # BM25
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    # SPLADE
    ap.add_argument("--splade-model", type=str, default="naver/splade-v3")
    ap.add_argument("--splade-batch-size", type=int, default=32)
    ap.add_argument("--splade-max-active-dims", type=int, default=None)
    # Qwen
    ap.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--qwen-dim", type=int, default=2048)
    ap.add_argument("--qwen-dtype", type=str, default="bfloat16")
    ap.add_argument(
        "--qwen-quant",
        type=str,
        default=None,
        choices=[None, "awq", "fp8"],
        help="vLLM quantization",
    )
    ap.add_argument("--qwen-batch-size", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    chunks_dir: Path = args.chunks_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_chunk_files(chunks_dir)
    if not files:
        print(f"[WARN] No *.chunks.jsonl in {chunks_dir}")
        return

    # ---------- Pass 1: ensure UUIDs & collect corpus stats for BM25 ----------
    all_chunks: List[Chunk] = []
    print("[1/4] Ensuring UUIDs + collecting BM25 stats...")
    for f in tqdm(files, ncols=100):
        rows = load_chunks(f)
        ensure_uuid(rows)
        save_chunks(f, rows)  # overwrite with uuid added
        for r in rows:
            all_chunks.append(Chunk(uuid=r["uuid"], text=r["text"]))

    bm25_stats = build_bm25_stats(all_chunks)
    print(f"    BM25: N={bm25_stats.N}, avgdl={bm25_stats.avgdl:.2f}")

    # ---------- Pass 2: SPLADE on GPU ----------
    print("[2/4] SPLADE-v3 (GPU) encoding...")
    splade_cfg = SpladeConfig(
        model_id=args.splade_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.splade_batch_size,
        max_active_dims=args.splade_max_active_dims,
    )
    spl_tokens, spl_weights = splade_encode(splade_cfg, [c.text for c in all_chunks])

    # ---------- Pass 3: Qwen3-Embedding-4B via vLLM (2048 dims) ----------
    print("[3/4] Qwen3-Embedding-4B (vLLM) encoding...")
    qcfg = QwenCfg(
        model_id=args.qwen_model,
        dim=int(args.qwen_dim),
        dtype=args.qwen_dtype,
        tensor_parallel_size=int(args.tp),
        quantization=args.qwen_quant,
        batch_size=args.qwen_batch_size,
    )
    qwen_vecs = qwen_embed(qcfg, [c.text for c in all_chunks])

    assert len(spl_tokens) == len(all_chunks) == len(qwen_vecs)

    # ---------- Pass 4: Write vectors per source file ----------
    print("[4/4] Writing vectors JSONL...")
    # Map UUID -> index in all_chunks
    uuid2idx = {c.uuid: i for i, c in enumerate(all_chunks)}

    for f in tqdm(files, ncols=100):
        rows = load_chunks(f)  # now include uuids
        out_path = out_dir / f"{f.stem.replace('.chunks', '')}.vectors.jsonl"
        with out_path.open("w", encoding="utf-8") as vf:
            for r in rows:
                u = r["uuid"]
                i = uuid2idx[u]
                # BM25 sparse vector
                terms, weights = bm25_vector(
                    text=all_chunks[i].text,
                    stats=bm25_stats,
                    k1=args.bm25_k1,
                    b=args.bm25_b,
                )
                # SPLADE sparse (token, weight)
                stoks, swts = spl_tokens[i], spl_weights[i]
                # Qwen dense (2048 dims)
                qvec = qwen_vecs[i]

                obj = {
                    "UUID": u,
                    "BM25": {
                        "terms": terms,
                        "weights": weights,
                        "k1": args.bm25_k1,
                        "b": args.bm25_b,
                        "avgdl": bm25_stats.avgdl,
                        "N": bm25_stats.N,
                    },
                    "SpladeV3": {
                        "model_id": splade_cfg.model_id,
                        "tokens": stoks,
                        "weights": swts,
                    },
                    "Qwen3-4B": {
                        "model_id": qcfg.model_id,
                        "dim": qcfg.dim,
                        "vector": qvec,
                    },
                }
                vf.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"[OK] {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
