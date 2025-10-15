"""
FAISS GPU factories and cloning helpers for HybridSearch.

This module centralises creation of GPU-backed FAISS indexes and provides
strict cloning utilities that prevent silent CPU fallbacks. Downstream callers
can request specific index layouts (flat, IVF, IVFPQ) while keeping control
over GPU placement and tuning parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import faiss  # type: ignore


@dataclass(frozen=True, slots=True)
class GPUOpts:
    """Runtime options controlling GPU index behaviour."""

    device: int = 0
    ivfpq_use_precomputed: bool = True
    ivfpq_float16_lut: bool = True


def gpu_index_factory(
    dim: int,
    *,
    index_type: str,
    nlist: int,
    nprobe: int,
    pq_m: int,
    pq_bits: int,
    resources: faiss.StandardGpuResources,
    opts: Optional[GPUOpts] = None,
) -> faiss.IndexIDMap2:
    """Return an IndexIDMap2 wrapping a GPU-resident FAISS index.

    Args:
        dim: Embedding dimensionality.
        index_type: One of {"flat", "ivf_flat", "ivf_pq"}.
        nlist: IVF coarse centroid count (ignored for flat).
        nprobe: IVF search breadth (ignored for flat).
        pq_m: PQ sub-vector count (IVFPQ only).
        pq_bits: PQ bits per sub-vector (IVFPQ only).
        resources: Shared StandardGpuResources instance.
        opts: Additional GPU options (device selection, PQ tuning).

    Returns:
        IndexIDMap2 wrapping the requested GPU FAISS index.
    """

    opts = opts or GPUOpts()
    device = int(opts.device)
    metric = faiss.METRIC_INNER_PRODUCT

    if index_type == "flat":
        if hasattr(faiss, "GpuIndexFlatConfig"):
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = device
            base = faiss.GpuIndexFlatIP(resources, dim, cfg)
        else:
            base = faiss.GpuIndexFlatIP(resources, dim)
        return faiss.IndexIDMap2(base)

    if index_type == "ivf_flat":
        if not hasattr(faiss, "GpuIndexIVFFlat"):
            raise RuntimeError("FAISS build is missing GpuIndexIVFFlat")
        quantizer = faiss.IndexFlatIP(dim)
        cfg = faiss.GpuIndexIVFFlatConfig()
        cfg.device = device
        base = faiss.GpuIndexIVFFlat(resources, quantizer, dim, int(nlist), metric, cfg)
        base.nprobe = int(nprobe)
        idx = faiss.IndexIDMap2(base)
        setattr(idx, "_cpu_quantizer_ref", quantizer)
        return idx

    if index_type == "ivf_pq":
        if not hasattr(faiss, "GpuIndexIVFPQ"):
            raise RuntimeError("FAISS build is missing GpuIndexIVFPQ")
        cfg = faiss.GpuIndexIVFPQConfig()
        cfg.device = device
        cfg.usePrecomputedTables = bool(opts.ivfpq_use_precomputed)
        cfg.useFloat16LookupTables = bool(opts.ivfpq_float16_lut)
        base = faiss.GpuIndexIVFPQ(
            resources,
            dim,
            int(nlist),
            int(pq_m),
            int(pq_bits),
            metric,
            cfg,
        )
        base.nprobe = int(nprobe)
        return faiss.IndexIDMap2(base)

    raise ValueError(f"Unsupported GPU index_type: {index_type}")


def maybe_clone_to_gpu(
    index_cpu: "faiss.Index",
    *,
    device: int,
    resources: faiss.StandardGpuResources,
) -> "faiss.Index":
    """Clone a CPU FAISS index onto a GPU with strict cloner options."""

    # Allow callers to pass an already GPU-backed index.
    if hasattr(index_cpu, "getDevice"):
        try:
            index_cpu.getDevice()  # pyright: ignore[reportUnusedExpression]
            return index_cpu
        except Exception:
            pass

    if not hasattr(faiss, "index_cpu_to_gpu"):
        raise RuntimeError("FAISS build lacks index_cpu_to_gpu helper")

    if hasattr(faiss, "GpuClonerOptions"):
        co = faiss.GpuClonerOptions()
        co.device = int(device)
        co.allowCpuCoarseQuantizer = False
        co.verbose = True
        return faiss.index_cpu_to_gpu(resources, int(device), index_cpu, co)

    return faiss.index_cpu_to_gpu(resources, int(device), index_cpu)
