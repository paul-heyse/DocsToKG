Below is a **programmer‑oriented reference** to the **GPU‑enabled FAISS wheel** you attached (`faiss-1.12.0-py3-none-any.whl`). I generated it by inspecting the Python files and SWIG layer inside the wheel (e.g., `faiss/swigfaiss.py`, `faiss/gpu_wrappers.py`, `faiss/extra_wrappers.py`, `faiss/class_wrappers.py`, `faiss/contrib/*`). The goal is to give an AI programming agent everything it needs—APIs, conventions, and working recipes—to use this build effectively.

---

## 0) What’s in this wheel (at a glance)

**Package name / version**

* Distribution metadata reports **`faiss==1.12.0`**.
* Python package `faiss/` includes a compiled extension **`_swigfaiss.so`** and a SWIG wrapper **`swigfaiss.py`** exposing both **CPU and GPU** APIs.
* GPU helpers are in **`faiss/gpu_wrappers.py`**, with high-level functions like `knn_gpu`, `pairwise_distance_gpu`, `index_cpu_to_all_gpus`, etc.
* Convenience wrappers, array conversions, factory helpers, and contrib modules (IVF tools, on‑disk merging, RPC, datasets, torch interop) are present under `faiss/*.py` and `faiss/contrib/*.py`.

**GPU classes present in this build (SWIG-exposed)**

* Resources / parameters: `StandardGpuResources`, `GpuResources`, `GpuResourcesVector`, `GpuParameterSpace`, `GpuClonerOptions`, `GpuMultipleClonerOptions`, `GpuIndexConfig`, `GpuIndexFlatConfig`, `GpuIndexIVFFlatConfig`, `GpuIndexIVFPQConfig`, `GpuIndexIVFScalarQuantizerConfig`.
* Indexes: `GpuIndex`, `GpuIndexFlat`, `GpuIndexFlatL2`, `GpuIndexFlatIP`, `GpuIndexIVF`, `GpuIndexIVFFlat`, `GpuIndexIVFPQ`, `GpuIndexIVFScalarQuantizer`, `GpuIndexBinaryFlat`.
* Utility: `GpuResourcesProvider`, `GpuResourcesProviderFromInstance`, `GpuProgressiveDimIndexFactory`, etc.

**CPU index families available**

* **Flat**: `IndexFlat`, `IndexFlatL2`, `IndexFlatIP`
* **IVF family**: `IndexIVF*` (`IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer`, `IndexIVFPQR`, fast‑scan variants)
* **HNSW**: `IndexHNSW`, `IndexHNSWFlat`, `IndexHNSWSQ`
* **PQ / Additive / Residual quantizers**: `IndexPQ`, `IndexAdditiveQuantizer`, `IndexResidualQuantizer`
* **Binary**: `IndexBinaryFlat`, `IndexBinaryIVF`, `IndexBinaryHNSW`
* **Meta/compose**: `IndexPreTransform`, `IndexIDMap`, `IndexIDMap2`, `IndexShards`, `IndexReplicas`, `IndexRefine`, `IndexRefineFlat`, `Index2Layer`
* **Transforms**: `PCAMatrix`, `OPQMatrix`, `ITQMatrix`, `RandomRotationMatrix`
* **Other**: `IndexLSH`, `IndexRowwiseMinMax*`, many specialized subtypes

**Enums & constants commonly used**

* Metrics: `METRIC_L2`, `METRIC_INNER_PRODUCT`, `METRIC_L1`, `METRIC_GOWER`
* GPU indices storage: `INDICES_CPU`, `INDICES_IVF`, `INDICES_32_BIT`, `INDICES_64_BIT`
* Scalar quantizer kinds (on CPU side): `ScalarQuantizer.QT_8bit`, `QT_4bit`, `QT_6bit`, `QT_8bit_uniform`, `QT_4bit_uniform`, `QT_8bit_direct`, `QT_8bit_direct_signed`, `QT_fp16`, `QT_bf16`

---

## 1) Core data model and conventions

**Array shapes & dtypes**

* Vectors are **row‑major** numpy arrays of shape `(n, d)` and **dtype=float32**.
* IDs are **int64** (aka `idx_t` in FAISS). When you don’t supply IDs, FAISS assigns `[0..ntotal-1]`.
* Results:

  * `D` distances: shape `(nq, k)`, `float32`
  * `I` labels/ids: shape `(nq, k)`, `int64`, with `-1` for “no neighbor”
* For **cosine similarity**, normalize inputs once, then use **`METRIC_INNER_PRODUCT`**:

  ```python
  faiss.normalize_L2(xb)     # in-place, requires float32
  faiss.normalize_L2(xq)
  index = faiss.IndexFlatIP(d)
  ```

**Training vs. adding**

* **Flat** indexes do **not** require training.
* **IVF / PQ / SQ** families **must be trained** (e.g., on a representative subset) before `add`/`add_with_ids`. Attempting `add` before `train` raises.

**Serialization & I/O**

* To files: `faiss.write_index(index, path)`, `faiss.read_index(path)`.
* In‑memory bytes: `faiss.serialize_index(index) -> np.uint8 array`, `faiss.deserialize_index(buf) -> Index`.
  Binary‑index variants also exist: `serialize_index_binary` / `deserialize_index_binary`.

**Parallelism**

* CPU threading: `faiss.omp_set_num_threads(n)`.
* GPU stream interop with PyTorch: `faiss.contrib.torch_utils.using_stream(...)`.

---

## 2) Distances and utilities

**Exact kNN without an index (CPU & GPU)**

```python
# CPU
D, I = faiss.knn(xq, xb, k, metric=faiss.METRIC_L2)

# GPU
res = faiss.StandardGpuResources()
D, I = faiss.knn_gpu(res, xq, xb, k, metric=faiss.METRIC_L2, device=0)
```

* `knn_gpu` accepts `vectorsMemoryLimit` / `queriesMemoryLimit` for chunking large runs.

**Pairwise distance matrices**

```python
# CPU
D = faiss.pairwise_distances(xq, xb)  # shape (nq, nb)

# GPU
res = faiss.StandardGpuResources()
D = faiss.pairwise_distance_gpu(res, xq, xb, metric=faiss.METRIC_L2, device=0)
```

Other handy helpers:

* `faiss.normalize_L2(x)` (in‑place), `faiss.kmin/kmax`, `faiss.merge_knn_results`, `faiss.bucket_sort`

---

## 3) Index families and when to use them

> All classes below also exist in CPU form; GPU coverage is called out explicitly.

### **Flat (exact search)**

* **CPU**: `IndexFlatL2`, `IndexFlatIP`
* **GPU**: `GpuIndexFlatL2`, `GpuIndexFlatIP`
* **Use when**: Small to mid‑sized db, or need exact results.
* **Complexity**: O(N·d) per query; memory ~ `4 * d * N` bytes (+ IDs).

**Minimal GPU example**

```python
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, d)       # or GpuIndexFlatIP
index.add(xb)
D, I = index.search(xq, k)
```

### **IVF + Flat (coarse quantizer + exact in lists)**

* **CPU**: `IndexIVFFlat`
* **GPU**: `GpuIndexIVFFlat` (or clone CPU -> GPU)
* **Train** on `xb_train` (representative sample)
* **Tunable**: `nlist` (number of coarse centroids), `nprobe` (lists scanned at query)
* **Use when**: Large dbs; want recall/speed trade‑off without PQ loss.

**Minimal GPU example (clone from CPU)**

```python
index_cpu = faiss.index_factory(d, "IVF4096,Flat", faiss.METRIC_L2)
index_cpu.train(xb_train)
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index.nprobe = 32
index.add(xb)
D, I = index.search(xq, k)
```

### **IVF + PQ (product quantization in lists)**

* **CPU**: `IndexIVFPQ`
* **GPU**: `GpuIndexIVFPQ` (or clone from CPU)
* **Train** required. Tunables: `nlist`, `M` (subquantizers), `nbits` (bits/code), `nprobe`.
* **Memory**: code size ≈ `M * nbits/8` bytes / vector (+ IDs).
* **Use when**: Very large dbs; need compact memory with good recall.

**GPU config knobs**

* `GpuIndexIVFPQConfig.usePrecomputedTables`
* `GpuIndexIVFPQConfig.useFloat16LookupTables`
* `GpuIndexIVFPQConfig.interleavedLayout`

**Example (direct GPU)**

```python
res = faiss.StandardGpuResources()
# GpuIndexIVFPQ(res, d, nlist, M, nbits, metric, config)
cfg = faiss.GpuIndexIVFPQConfig()
cfg.usePrecomputedTables = True
index = faiss.GpuIndexIVFPQ(res, d, 4096, 32, 8, faiss.METRIC_L2, cfg)
index.train(xb_train)
index.add(xb)
index.nprobe = 32
D, I = index.search(xq, k)
```

### **IVF + ScalarQuantizer (SQ)**

* **CPU**: `IndexIVFScalarQuantizer`
* **GPU**: `GpuIndexIVFScalarQuantizer`
* **Quantizers**: `ScalarQuantizer.QT_8bit`, `QT_4bit`, `QT_fp16`, etc.
* **Use when**: Lighter compression than PQ with simpler compute.

### **HNSW (graph ANN)**

* **CPU**: `IndexHNSWFlat`, `IndexHNSWSQ`
* **GPU**: **not** a native `GpuIndexHNSW*` class in this build (search takes place on CPU).
  You can still: train on CPU, or search CPU; use `IndexRefineFlat` to refine HNSW hits.

### **Binary (Hamming space)**

* **CPU**: `IndexBinaryFlat`, `IndexBinaryIVF`, `IndexBinaryHNSW`
* **GPU**: `GpuIndexBinaryFlat`
* **Use when**: Inputs are binary codebooks; metric is Hamming distance.
  (See also `faiss.knn_hamming` and `faiss.pack_bitstrings`/`unpack_bitstrings`.)

### **Meta indexes**

* `IndexIDMap` / `IndexIDMap2`: keep your own IDs (`add_with_ids`); supports removals with `remove_ids` (use `DirectMap` for faster deletes).
* `IndexPreTransform`: chain transforms (e.g., `PCAMatrix`, `OPQMatrix`) before an index.
* `IndexShards`, `IndexReplicas`: distribute / replicate across devices or processes; GPU helpers below automate multi‑GPU clones.

---

## 4) GPU workflow essentials

### 4.1 Resources and memory

* Create resources once and reuse:

  ```python
  res = faiss.StandardGpuResources()
  # Optional: res.noTempMemory(), res.setTempMemory(bytes), res.getMemoryInfo()
  ```

* Choose device / memory space via `GpuIndexConfig` and derived configs:

  * `device`: GPU id (int)
  * `memorySpace`: memory placement (device vs unified/host; subject to build support)
  * `use_cuvs`: allow dispatch to cuVS implementations when available

### 4.2 Build on CPU, run on GPU (cloning)

* **Single GPU**: `faiss.index_cpu_to_gpu(res, device, index_cpu)`
* **All GPUs**: `faiss.index_cpu_to_all_gpus(index_cpu)` (convenience wrapper)
* **Explicit list**: `faiss.index_cpu_to_gpus_list(index_cpu, gpus=[0,1], co=faiss.GpuMultipleClonerOptions())`
* `GpuMultipleClonerOptions` fields you will likely use:

  * `shard` (bool): shard vs replicate across GPUs
  * `indicesOptions`: one of `INDICES_CPU`, `INDICES_IVF`, `INDICES_32_BIT`, `INDICES_64_BIT`
  * `useFloat16`, `useFloat16CoarseQuantizer`, `usePrecomputed`, `reserveVecs`, `verbose`, `use_cuvs`, `allowCpuCoarseQuantizer`

### 4.3 Flat vs IVF on GPU

* **Flat**: build and use directly on GPU (`GpuIndexFlatL2/IP`).
* **IVF/PQ/SQ**: either build directly on GPU (`GpuIndexIVF*`) or train on CPU then clone to GPU.
* Tune search using

  * `index.nprobe` (IVF)
  * `GpuParameterSpace().set_index_parameter(index, "nprobe", value)`

### 4.4 Direct GPU kNN without an index

```python
D, I = faiss.knn_gpu(res, xq, xb, k, metric=faiss.METRIC_L2, device=0)
```

Good for one‑off or small data; for repeated queries use an index.

### 4.5 Multi‑GPU patterns

* **Replicated** (better recall/latency when each GPU scans full data in parallel); set `co.shard=False`.
* **Sharded** (scale to very large datasets, each GPU owns a subset); set `co.shard=True`.

---

## 5) Typical recipes (copy‑paste ready)

### 5.1 Exact cosine search on GPU

```python
import faiss
import numpy as np

d, k = 768, 10
xb = np.random.rand(1_000_000, d).astype('float32')
xq = np.random.rand(10_000, d).astype('float32')
faiss.normalize_L2(xb); faiss.normalize_L2(xq)

res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatIP(res, d)
index.add(xb)                   # ntotal grows; IDs are 0..ntotal-1
D, I = index.search(xq, k)
```

### 5.2 IVF‑Flat on GPU (clone from CPU)

```python
index_cpu = faiss.index_factory(d, "IVF4096,Flat", faiss.METRIC_L2)
index_cpu.train(xb[:200_000])   # representative sample
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index.nprobe = 32               # scan 32 lists at query
index.add(xb)
D, I = index.search(xq, k)
```

### 5.3 IVFPQ on GPU (direct)

```python
cfg = faiss.GpuIndexIVFPQConfig()
cfg.usePrecomputedTables = True
res = faiss.StandardGpuResources()

index = faiss.GpuIndexIVFPQ(res, d, 4096, 32, 8, faiss.METRIC_L2, cfg)
index.train(xb[:300_000])
index.add(xb)
index.nprobe = 32
D, I = index.search(xq, k)
```

### 5.4 Multi‑GPU (replicated)

```python
co = faiss.GpuMultipleClonerOptions()
co.shard = False    # replicate on all GPUs
index = faiss.index_cpu_to_all_gpus(index_cpu, co=co)
```

### 5.5 Keep your own IDs and support deletes

```python
base = faiss.IndexFlatL2(d)
index = faiss.IndexIDMap2(base)     # wrap
ids = np.arange(xb.shape[0], dtype='int64')
index.add_with_ids(xb, ids)

# delete a subset of ids
to_remove = faiss.IDSelectorArray(np.array([5, 42, 99], dtype='int64'))
index.remove_ids(to_remove)
```

### 5.6 Pre‑transform (PCA) + IVF

```python
pca = faiss.PCAMatrix(d, 256)                       # output dim 256
pre = faiss.IndexPreTransform(pca, faiss.IndexFlatL2(256))
index = faiss.IndexIVFFlat(pre, 256, 4096, faiss.METRIC_L2)
index.train(xb)
index.add(xb)
```

### 5.7 Range search

```python
# All neighbors within a radius (e.g., L2 radius)
radius = 1.23
res = faiss.RangeSearchResult(xq.shape[0])
index.range_search(xq, radius, res)
# Then extract result from res, or use helpers in contrib modules
```

---

## 6) Parameter tuning

**General guidance**

* **IVF**:

  * `nlist` ~ O(√N) to O(N^(2/3)) (choose larger for bigger databases)
  * `nprobe` ~ 1–10% of `nlist` for recall target (increase for recall)
* **IVFPQ**:

  * Increase `M` (subquantizers) and/or `nbits` (bits per code) for quality at the cost of memory and compute.
  * Try `GpuIndexIVFPQConfig.usePrecomputedTables=True` for speed if memory allows.
* **Flat**: no tuning; scale via multi‑GPU replication.

**Programmatic tuning**

```python
ps = faiss.ParameterSpace()
ps.set_index_parameter(index, "nprobe", 64)
# For GPU:
gps = faiss.GpuParameterSpace()
gps.set_index_parameter(index, "nprobe", 64)
```

---

## 7) Big data / systems integrations (contrib)

* **On‑disk IVF merge** (`faiss/contrib/ondisk.py`): build many IVF shards and merge into on‑disk lists using `OnDiskInvertedLists`. High‑level helper `merge_ondisk(trained_index, shard_fnames, ivfdata_fname, shift_ids=False)`.
* **IVF tools** (`faiss/contrib/ivf_tools.py`): `add_preassigned` (add with precomputed coarse assignments), `search_preassigned`, `range_search_preassigned`, plus list‑size inspection functions.
* **Datasets / IO** (`faiss/contrib/vecs_io.py`): read/write `fvecs`, `bvecs`, `ivecs` formats.
* **RPC / client‑server** (`faiss/contrib/client_server.py`, `faiss/contrib/rpc.py`): simple multi‑process sharding + RPC.
* **Torch interop** (`faiss/contrib/torch_utils.py`):

  * `using_stream(...)`: make FAISS use the current PyTorch CUDA stream
  * Pointers from tensors: `swig_ptr_from_FloatTensor`, etc.
  * Drop‑in GPU kNN for torch tensors: `torch_replacement_knn_gpu`

---

## 8) API reference cheat‑sheet (most used)

**Building**

* `faiss.IndexFlatL2(d)`, `faiss.IndexFlatIP(d)`
* `faiss.index_factory(d, "IVF4096,Flat", metric)`
* `faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)` (CPU)
* GPU direct: `faiss.GpuIndexFlatL2(res, d)`, `faiss.GpuIndexIVFFlat(res, d, nlist, metric, cfg)`, `faiss.GpuIndexIVFPQ(res, d, nlist, M, nbits, metric, cfg)`, `faiss.GpuIndexIVFScalarQuantizer(...)`

**Training / adding / searching**

* `index.train(xb_train)` *(if required)*
* `index.add(xb)` / `index.add_with_ids(xb, ids)`
* `D, I = index.search(xq, k)`
* `index.range_search(xq, radius, result)`

**Tuning & params**

* IVF: `index.nprobe = 16` (or via `ParameterSpace`)
* HNSW: `index.hnsw.efSearch = ...`, `index.hnsw.efConstruction = ...`, `index.hnsw.M = ...`
* Quantizers: `ScalarQuantizer.QT_*` enums

**Serialization**

* Files: `faiss.write_index(index, path)`, `faiss.read_index(path)`
* Bytes: `faiss.serialize_index(index)`, `faiss.deserialize_index(buf)`

**Multi‑GPU / GPU helpers**

* `faiss.get_num_gpus()`
* `faiss.index_cpu_to_gpu(res, device, index_cpu)`
* `faiss.index_cpu_to_all_gpus(index_cpu, co=None)`
* `faiss.index_cpu_to_gpus_list(index_cpu, gpus=[...], co=None)`
* `faiss.knn_gpu(res, xq, xb, k, metric=..., device=...)`
* `faiss.pairwise_distance_gpu(res, xq, xb, metric=..., device=...)`

**Housekeeping**

* `faiss.omp_set_num_threads(n)`
* `faiss.normalize_L2(x)` (in place)

---

## 9) Practical tips & gotchas

* **Always use `float32` and contiguous arrays.** The wrappers will copy/transpose if needed, but you pay for it.
* **Cosine similarity**: L2‑normalize and use `METRIC_INNER_PRODUCT`.
* **Train once, then add**. Attempting to `add` to untrained IVF/PQ indexes raises.
* **IDs**: If you need stable, user‑visible IDs, wrap your base index in `IndexIDMap2` and use `add_with_ids`.
* **Deletes**: use `remove_ids` (with `IDSelector*`). For large delete workloads, a `DirectMap` improves performance.
* **Large builds**: build `IVF*` offline; use `contrib.ivf_tools.add_preassigned` or `contrib.ondisk.merge_ondisk` to scale.
* **GPU memory**: For huge datasets, consider IVF+PQ/SQ and/or multi‑GPU sharding. Use `StandardGpuResources.setTempMemory`, `getMemoryInfo` to manage temp buffers.
* **PyTorch streams**: wrap calls in `using_stream(...)` when mixing FAISS GPU with PyTorch to avoid sync stalls.
* **Cloning options**: `GpuClonerOptions.allowCpuCoarseQuantizer=False` by default—indices not implemented on GPU will raise rather than silently fallback.
* **cuVS integration**: some configs expose `use_cuvs=True`; set it only if your runtime has the corresponding support (or you’ll get an error).

---

## 10) Minimal decision flow for an agent

1. **Metric**: `METRIC_L2` vs `METRIC_INNER_PRODUCT` (cosine → normalize + IP).
2. **Scale**:

   * ≤ few million vectors → try `GpuIndexFlat*` (exact) or `IVF-Flat`.
   * Tens/hundreds of millions → `IVF‑PQ` (or `IVF‑SQ`) + multi‑GPU sharding.
3. **Train** if using IVF/PQ/SQ (sample 100k–1M vectors).
4. **Build** CPU or GPU index.
5. **Clone** to GPU(s) if needed; choose replicate vs shard.
6. **Tune** `nprobe` (and PQ code size) to meet recall/latency targets.
7. **Persist**: write/read via `write_index` / `read_index` (or `serialize_*`).

---

## 11) Environment & runtime checks

* **GPU availability**: `faiss.get_num_gpus()` → integer count.
* **Resources**: create `StandardGpuResources()` once and reuse.
* **No GPU?** Build and use CPU indexes; the API is largely symmetric.
* **Runtime libs**: GPU features require a CUDA‑capable environment. If unavailable, GPU constructors/methods will raise—catch and fallback to CPU.

---

## 12) Pointers to specific symbols present in this wheel

* **Top‑level metrics**: `faiss.METRIC_L2`, `faiss.METRIC_INNER_PRODUCT`, `faiss.METRIC_L1`, `faiss.METRIC_GOWER`
* **GPU configs**:

  * `GpuIndexFlatConfig`: `useFloat16`, `storeTransposed` (deprecated)
  * `GpuIndexIVFFlatConfig`: `interleavedLayout`
  * `GpuIndexIVFPQConfig`: `usePrecomputedTables`, `useFloat16LookupTables`, `interleavedLayout`
  * `GpuIndexIVFScalarQuantizerConfig`: `interleavedLayout`
  * `GpuIndexConfig`: `device`, `memorySpace`, `use_cuvs`
  * `GpuClonerOptions`: `indicesOptions`, `useFloat16CoarseQuantizer`, `useFloat16`, `usePrecomputed`, `reserveVecs`, `storeTransposed`, `verbose`, `use_cuvs`, `allowCpuCoarseQuantizer`
* **Contrib modules**:

  * `faiss.contrib.ivf_tools` (`add_preassigned`, `search_preassigned`, …)
  * `faiss.contrib.ondisk` (`merge_ondisk`, `OnDiskInvertedLists`)
  * `faiss.contrib.vecs_io` (`fvecs_read/write`, `bvecs_read/write`, `ivecs_read/write`)
  * `faiss.contrib.torch_utils` (`using_stream`, tensor pointer helpers)
  * `faiss.contrib.client_server`/`rpc` (simple RPC/sharding)
  * `faiss.contrib.inspect_tools` (inspect IVF lists, PQ centroids, etc.)

---

If you want, I can now generate **scaffold code** (scripts/notebooks) for your exact use case (dataset size, metric, latency/recall targets, number of GPUs).
