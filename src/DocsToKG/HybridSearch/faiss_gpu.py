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
    """Runtime options controlling GPU index behaviour.

    Attributes:
        device: GPU device identifier used for FAISS operations.
        ivfpq_use_precomputed: Whether to enable IVFPQ precomputed tables.
        ivfpq_float16_lut: Whether to enable float16 lookup tables for IVFPQ.

    Examples:
        >>> opts = GPUOpts(device=1, ivfpq_use_precomputed=False)
        >>> opts.device
        1
    """

    device: int = 0
    ivfpq_use_precomputed: bool = True
    ivfpq_float16_lut: bool = True
    interleaved_layout: bool = True
    flat_use_fp16: bool = False


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

    Raises:
        RuntimeError: When the requested index type is unavailable in the FAISS build.
        ValueError: If an unsupported index_type is provided.

    Examples:
        >>> resources = faiss.StandardGpuResources()  # doctest: +SKIP
        >>> index = gpu_index_factory(128, index_type="flat", nlist=1, nprobe=1,
        ...                           pq_m=16, pq_bits=8, resources=resources)  # doctest: +SKIP
        >>> isinstance(index, faiss.IndexIDMap2)  # doctest: +SKIP
        True
    """

    opts = opts or GPUOpts()
    device = int(opts.device)
    metric = faiss.METRIC_INNER_PRODUCT

    if index_type == "flat":
        if hasattr(faiss, "GpuIndexFlatConfig"):
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = device
            if hasattr(cfg, "useFloat16"):
                cfg.useFloat16 = bool(opts.flat_use_fp16)
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
        if hasattr(cfg, "interleavedLayout"):
            cfg.interleavedLayout = bool(opts.interleaved_layout)
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
        if hasattr(cfg, "interleavedLayout"):
            cfg.interleavedLayout = bool(opts.interleaved_layout)
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
    indices_32_bits: bool = True,
) -> "faiss.Index":
    """Clone a CPU FAISS index onto a GPU with strict cloner options.

    Args:
        index_cpu: CPU-backed FAISS index to clone.
        device: GPU device identifier to host the cloned index.
        resources: Shared FAISS GPU resources.

    Returns:
        faiss.Index: GPU-resident index (or the original if already on GPU).

    Raises:
        RuntimeError: If FAISS lacks GPU cloning support.
    """

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
        if indices_32_bits and hasattr(faiss, "INDICES_32_BIT"):
            co.indicesOptions = faiss.INDICES_32_BIT
        return faiss.index_cpu_to_gpu(resources, int(device), index_cpu, co)

    return faiss.index_cpu_to_gpu(resources, int(device), index_cpu)
