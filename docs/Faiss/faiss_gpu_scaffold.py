# faiss_gpu_scaffold.py
import os
import sys
import time
import argparse
from typing import Optional, Tuple, List

import numpy as np
from tqdm import tqdm

import faiss

from datasets import generate_synthetic, load_npy_pair, maybe_normalize_cosine, ensure_ids
from tuner import recall_at_k, time_block, write_sweep_csv

# ------------------------------
# Metric helpers
# ------------------------------
def metric_from_str(metric: str) -> int:
    metric = metric.lower()
    if metric in ("l2", "euclidean", "sqeuclidean"):
        return faiss.METRIC_L2
    if metric in ("ip", "inner_product", "dot"):
        return faiss.METRIC_INNER_PRODUCT
    if metric in ("cos", "cosine", "cos_sim"):
        # We implement cosine by L2-normalizing and using IP
        return faiss.METRIC_INNER_PRODUCT
    raise ValueError(f"Unsupported metric: {metric}")

def need_cosine_norm(metric: str) -> bool:
    return metric.lower() in ("cos", "cosine", "cos_sim")

# ------------------------------
# Index builders
# ------------------------------
def make_cpu_index(d: int, index_kind: str, metric: int, nlist: int = 0, M: int = 0, nbits: int = 8) -> faiss.Index:
    kind = index_kind.lower()
    if kind == "flat":
        return faiss.index_factory(d, "Flat", metric)
    elif kind == "ivfflat":
        assert nlist > 0, "ivfflat requires --nlist > 0"
        return faiss.index_factory(d, f"IVF{nlist},Flat", metric)
    elif kind == "ivfpq":
        assert nlist > 0 and M > 0, "ivfpq requires --nlist > 0 and --M > 0"
        return faiss.index_factory(d, f"IVF{nlist},PQ{M}x{nbits}", metric)
    elif kind == "ivfsq8":
        assert nlist > 0, "ivfsq8 requires --nlist > 0"
        return faiss.index_factory(d, f"IVF{nlist},SQ8", metric)
    else:
        raise ValueError(f"Unknown index kind: {index_kind}")

def maybe_train(index: faiss.Index, xb_train: np.ndarray):
    if index.is_trained:
        return
    print(f"[train] Training index on {xb_train.shape[0]} vectors...")
    index.train(xb_train)
    print("[train] Done.")

def add_in_chunks(index: faiss.Index, xb: np.ndarray, ids: Optional[np.ndarray] = None, add_bs: int = 100_000):
    nb = xb.shape[0]
    if ids is not None:
        assert ids.shape[0] == nb and ids.dtype == np.int64
    for i0 in tqdm(range(0, nb, add_bs), desc="[add]"):
        i1 = min(nb, i0 + add_bs)
        if ids is None:
            index.add(xb[i0:i1])
        else:
            index.add_with_ids(xb[i0:i1], ids[i0:i1])

def to_gpu(index_cpu: faiss.Index, device: int = 0) -> faiss.Index:
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, device, index_cpu)
    return index_gpu

def to_cpu(index_any: faiss.Index) -> faiss.Index:
    # If this is a GPU index, clone to CPU for serialization
    try:
        return faiss.index_gpu_to_cpu(index_any)
    except Exception:
        return index_any

def save_index(index_any: faiss.Index, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    index_cpu = to_cpu(index_any)
    faiss.write_index(index_cpu, path)
    print(f"[io] Index saved to {path}")

def load_index(path: str) -> faiss.Index:
    idx = faiss.read_index(path)
    print(f"[io] Loaded index from {path} (ntotal={idx.ntotal})")
    return idx

# ------------------------------
# Search utilities
# ------------------------------
def search_batches(index: faiss.Index, xq: np.ndarray, k: int, search_bs: int = 50_000) -> Tuple[np.ndarray, np.ndarray, float]:
    nq = xq.shape[0]
    D = np.empty((nq, k), dtype="float32")
    I = np.empty((nq, k), dtype="int64")
    t0 = time.perf_counter()
    o = 0
    for i0 in tqdm(range(0, nq, search_bs), desc="[search]"):
        i1 = min(nq, i0 + search_bs)
        Di, Ii = index.search(xq[i0:i1], k)
        D[o:o+(i1-i0)] = Di
        I[o:o+(i1-i0)] = Ii
        o += (i1 - i0)
    t_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = t_ms / max(1, nq)
    return D, I, avg_ms

def exact_ground_truth(xb: np.ndarray, xq: np.ndarray, k: int, metric: int, gpu_first: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build an exact Flat index and search for ground truth. Uses GPU Flat if available and allowed.
    """
    d = xb.shape[1]
    if gpu_first and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        print("[gt] Using GPU Flat for exact ground truth")
        res = faiss.StandardGpuResources()
        if metric == faiss.METRIC_L2:
            index = faiss.GpuIndexFlatL2(res, d)
        else:
            index = faiss.GpuIndexFlatIP(res, d)
    else:
        print("[gt] Using CPU Flat for exact ground truth")
        index = faiss.IndexFlatL2(d) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(d)
    index.add(xb)
    return search_batches(index, xq, k)

# ------------------------------
# CLI workflows
# ------------------------------
def run_quickstart(args):
    # Data
    xb, xq = generate_or_load_data(args)
    if need_cosine_norm(args.metric):
        xb = maybe_normalize_cosine(xb, True)
        xq = maybe_normalize_cosine(xq, True)
    metric = metric_from_str(args.metric)

    # Build CPU index first (uniform path), then optionally move to GPU
    index_cpu = make_cpu_index(d=xb.shape[1], index_kind=args.index, metric=metric, nlist=args.nlist, M=args.M, nbits=args.nbits)

    # Training
    train_size = min(args.train_size, xb.shape[0])
    xb_train = xb[:train_size].copy()
    maybe_train(index_cpu, xb_train)

    # GPU?
    index = index_cpu
    if args.gpu and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        index = to_gpu(index_cpu, device=args.device)
        print(f"[gpu] Using GPU device {args.device}")

    # IVF tuning
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe

    # Add
    ids = ensure_ids(xb.shape[0], args.use_ids)
    add_in_chunks(index, xb, ids, add_bs=args.add_bs)
    print(f"[info] ntotal={index.ntotal}")

    # Search
    D, I, avg_ms = search_batches(index, xq, args.k, search_bs=args.search_bs)
    print(f"[search] avg latency per query: {avg_ms:.3f} ms")

    # Ground truth (optional small sample if very large)
    if args.eval:
        nb_gt = min(args.nb_gt, xb.shape[0])
        nq_gt = min(args.nq_gt, xq.shape[0])
        xb_gt = xb[:nb_gt].copy()
        xq_gt = xq[:nq_gt].copy()
        Dgt, Igt, _ = exact_ground_truth(xb_gt, xq_gt, args.k, metric, gpu_first=True)
        # For fair compare, restrict predictions to first nb_gt database points
        mask = (Igt >= 0)
        I_pred_sub = I[:nq_gt].copy()
        I_pred_sub[mask] = np.where(I_pred_sub[mask] < nb_gt, I_pred_sub[mask], -1)
        r_at_k = recall_at_k(I_pred_sub, Igt, args.k)
        print(f"[eval] recall@{args.k}: {r_at_k:.4f}  (on nq={nq_gt}, nb={nb_gt} for GT)")

    # Save index (optional)
    if args.save_index:
        save_index(index, args.save_index)

def run_build(args):
    xb, xq = generate_or_load_data(args, only_base=True)
    if need_cosine_norm(args.metric):
        xb = maybe_normalize_cosine(xb, True)
    metric = metric_from_str(args.metric)
    index_cpu = make_cpu_index(d=xb.shape[1], index_kind=args.index, metric=metric, nlist=args.nlist, M=args.M, nbits=args.nbits)
    xb_train = xb[:min(args.train_size, xb.shape[0])].copy()
    maybe_train(index_cpu, xb_train)
    index = index_cpu
    if args.gpu and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        index = to_gpu(index_cpu, device=args.device)
        print(f"[gpu] Using GPU device {args.device}")
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe
    ids = ensure_ids(xb.shape[0], args.use_ids)
    add_in_chunks(index, xb, ids, add_bs=args.add_bs)
    if args.save_index:
        save_index(index, args.save_index)

def run_search(args):
    assert args.index_path, "--index_path is required"
    # Loading always yields a CPU index; optionally move to GPU
    index = load_index(args.index_path)
    # Metric matters for cosine normalization; but at search time we only affect queries
    xq = load_or_generate_queries(args)
    if need_cosine_norm(args.metric):
        xq = maybe_normalize_cosine(xq, True)
    # GPU?
    if args.gpu and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        index = to_gpu(index, device=args.device)
        print(f"[gpu] Using GPU device {args.device}")

    rows = []
    if args.nprobe_sweep:
        # IVF-only parameter; safe to try on Flat (it will be ignored or absent)
        for npv in [int(s) for s in args.nprobe_sweep.split(",")]:
            if hasattr(index, "nprobe"):
                index.nprobe = npv
            D, I, avg_ms = search_batches(index, xq, args.k, search_bs=args.search_bs)
            rows.append({"nprobe": npv, "avg_ms_per_query": avg_ms})
            print(f"[sweep] nprobe={npv}  avg_ms={avg_ms:.3f}")
        if args.out_csv:
            write_sweep_csv(rows, args.out_csv)
            print(f"[sweep] wrote {args.out_csv}")
    else:
        if hasattr(index, "nprobe"):
            index.nprobe = args.nprobe
        D, I, avg_ms = search_batches(index, xq, args.k, search_bs=args.search_bs)
        print(f"[search] avg latency per query: {avg_ms:.3f} ms")

def run_eval(args):
    xb, xq = generate_or_load_data(args)
    if need_cosine_norm(args.metric):
        xb = maybe_normalize_cosine(xb, True)
        xq = maybe_normalize_cosine(xq, True)
    metric = metric_from_str(args.metric)

    # Build candidate index
    index_cpu = make_cpu_index(d=xb.shape[1], index_kind=args.index, metric=metric, nlist=args.nlist, M=args.M, nbits=args.nbits)
    maybe_train(index_cpu, xb[:min(args.train_size, xb.shape[0])].copy())

    # GPU?
    index = index_cpu
    if args.gpu and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        index = to_gpu(index_cpu, device=args.device)
        print(f"[gpu] Using GPU device {args.device}")
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe

    # Add base
    ids = ensure_ids(xb.shape[0], args.use_ids)
    add_in_chunks(index, xb, ids, add_bs=args.add_bs)

    # Candidate search
    D, I, avg_ms = search_batches(index, xq, args.k, search_bs=args.search_bs)
    print(f"[candidate] avg latency per query: {avg_ms:.3f} ms")

    # Ground truth on subset
    nb_gt = min(args.nb_gt, xb.shape[0])
    nq_gt = min(args.nq_gt, xq.shape[0])
    xb_gt = xb[:nb_gt].copy()
    xq_gt = xq[:nq_gt].copy()
    Dgt, Igt, _ = exact_ground_truth(xb_gt, xq_gt, args.k, metric, gpu_first=True)

    # Compare (masking predictions outside nb_gt)
    mask = (Igt >= 0)
    I_pred_sub = I[:nq_gt].copy()
    I_pred_sub[mask] = np.where(I_pred_sub[mask] < nb_gt, I_pred_sub[mask], -1)
    r_at_k = recall_at_k(I_pred_sub, Igt, args.k)
    print(f"[eval] recall@{args.k}: {r_at_k:.4f}  (on nq={nq_gt}, nb={nb_gt} for GT)")

# ------------------------------
# Data input helpers
# ------------------------------
def generate_or_load_data(args, only_base=False):
    if args.xb and args.xq:
        xb, xq = load_npy_pair(args.xb, args.xq)
    else:
        xb, xq = generate_synthetic(args.nb, args.d, args.nq, seed=args.seed)
    if only_base:
        return xb, None
    return xb, xq

def load_or_generate_queries(args):
    if args.xq:
        xq = np.load(args.xq)
        xq = np.ascontiguousarray(xq.astype(np.float32))
        return xq
    # Synthetic queries only
    _, xq = generate_synthetic(args.nb, args.d, args.nq, seed=args.seed)
    return xq

# ------------------------------
# Argparse
# ------------------------------
def add_common_args(p):
    p.add_argument("--index", choices=["flat","ivfflat","ivfpq","ivfsq8"], default="flat")
    p.add_argument("--metric", choices=["l2","ip","cosine"], default="l2")
    p.add_argument("--d", type=int, default=256, help="Dimensionality")
    p.add_argument("--nb", type=int, default=200_000, help="Number of base vectors")
    p.add_argument("--nq", type=int, default=10_000, help="Number of queries")
    p.add_argument("--k", type=int, default=10, help="Top-k")
    p.add_argument("--xb", type=str, default=None, help="Path to base .npy (float32)")
    p.add_argument("--xq", type=str, default=None, help="Path to query .npy (float32)")
    p.add_argument("--seed", type=int, default=123)

    # IVF / PQ params
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--M", type=int, default=32, help="Number of PQ subquantizers")
    p.add_argument("--nbits", type=int, default=8, help="Bits per PQ subvector")

    # Compute & batching
    p.add_argument("--gpu", type=int, default=1, help="Use GPU if available (1=yes, 0=no)")
    p.add_argument("--device", type=int, default=0, help="GPU device id")
    p.add_argument("--add_bs", type=int, default=100_000, help="Add() batch size")
    p.add_argument("--search_bs", type=int, default=50_000, help="Search() batch size")

    # IDs and training
    p.add_argument("--use_ids", action="store_true", help="Use explicit int64 IDs")
    p.add_argument("--train_size", type=int, default=300_000, help="Training set size (for IVF/PQ)")

def main():
    ap = argparse.ArgumentParser(description="FAISS GPU Scaffold (single GPU friendly)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_qs = sub.add_parser("quickstart", help="End-to-end demo: build+train+add+search+eval (synthetic by default)")
    add_common_args(p_qs)
    p_qs.add_argument("--eval", action="store_true", help="Compute recall@k vs exact GT on subset")
    p_qs.add_argument("--nb_gt", type=int, default=100_000, help="Base size for ground truth subset")
    p_qs.add_argument("--nq_gt", type=int, default=10_000, help="Query size for ground truth subset")
    p_qs.add_argument("--save_index", type=str, default=None, help="Path to save the index")
    p_qs.set_defaults(func=run_quickstart)

    p_build = sub.add_parser("build", help="Build/train/add and optionally save index")
    add_common_args(p_build)
    p_build.add_argument("--save_index", type=str, default=None, help="Path to save the index")
    p_build.set_defaults(func=run_build)

    p_search = sub.add_parser("search", help="Load an index and search queries; supports nprobe sweeps")
    p_search.add_argument("--index_path", type=str, required=True, help="Path to a saved index (CPU-serialized)")
    p_search.add_argument("--metric", choices=["l2","ip","cosine"], default="l2")
    p_search.add_argument("--d", type=int, default=256)
    p_search.add_argument("--nq", type=int, default=10_000)
    p_search.add_argument("--k", type=int, default=10)
    p_search.add_argument("--xq", type=str, default=None)
    p_search.add_argument("--seed", type=int, default=123)
    p_search.add_argument("--gpu", type=int, default=1)
    p_search.add_argument("--device", type=int, default=0)
    p_search.add_argument("--search_bs", type=int, default=50_000)
    p_search.add_argument("--nprobe", type=int, default=32)
    p_search.add_argument("--nprobe_sweep", type=str, default=None, help="Comma-separated list of nprobe values")
    p_search.add_argument("--out_csv", type=str, default=None, help="Write sweep results to CSV")
    p_search.set_defaults(func=run_search)

    p_eval = sub.add_parser("eval", help="Build an index and compute recall@k vs exact ground truth")
    add_common_args(p_eval)
    p_eval.add_argument("--nb_gt", type=int, default=100_000)
    p_eval.add_argument("--nq_gt", type=int, default=10_000)
    p_eval.set_defaults(func=run_eval)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
