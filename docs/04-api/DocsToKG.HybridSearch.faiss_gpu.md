# Module: faiss_gpu

FAISS GPU factories and cloning helpers for HybridSearch.

This module centralises creation of GPU-backed FAISS indexes and provides
strict cloning utilities that prevent silent CPU fallbacks. Downstream callers
can request specific index layouts (flat, IVF, IVFPQ) while keeping control
over GPU placement and tuning parameters.

## Functions

### `gpu_index_factory(dim)`

Return an IndexIDMap2 wrapping a GPU-resident FAISS index.

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

### `maybe_clone_to_gpu(index_cpu)`

Clone a CPU FAISS index onto a GPU with strict cloner options.

Args:
index_cpu: CPU-backed FAISS index to clone.
device: GPU device identifier to host the cloned index.
resources: Shared FAISS GPU resources.

Returns:
faiss.Index: GPU-resident index (or the original if already on GPU).

Raises:
RuntimeError: If FAISS lacks GPU cloning support.

## Classes

### `GPUOpts`

Runtime options controlling GPU index behaviour.

Attributes:
device: GPU device identifier used for FAISS operations.
ivfpq_use_precomputed: Whether to enable IVFPQ precomputed tables.
ivfpq_float16_lut: Whether to enable float16 lookup tables for IVFPQ.

Examples:
>>> opts = GPUOpts(device=1, ivfpq_use_precomputed=False)
>>> opts.device
1
