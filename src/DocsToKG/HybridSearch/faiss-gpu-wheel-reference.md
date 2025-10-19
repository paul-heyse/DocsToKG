# Custom FAISS GPU Wheel — Essential Reference for AI Agents
**Wheel**: `faiss-1.12.0-py3-none-any.whl`  
**FAISS version label**: `1.12.0`  
**Generated**: 2025-10-19 01:46 UTC

This document summarizes the *actual* GPU-enabled surface area exposed by your custom FAISS wheel, plus practical guidance for using it safely and efficiently from autonomous or tool-using AI systems. HybridSearch bundles FAISS alongside the NVIDIA cuVS Python bindings (`cuvs`, version 25.10.00) and their loader (`libcuvs`). Even though the shipped FAISS binary cannot enable cuVS kernels (`use_cuvs=True` raises “cuVS has not been compiled into the current version”), the interfaces are present and HybridSearch code interacts with them. The cuVS sections below explain how discovery, configuration, and fallbacks work so you can reason about current behaviour and future upgrades.

> ⚠️ **Important**: The wheel is tagged as `py3-none-any` (pure Python), but it **does contain native code** (`_swigfaiss.so`). Treat it as platform-specific. See **Runtime prerequisites** for required shared libraries.

---

## 1) Runtime prerequisites (from the built extension)

These are extracted by running `ldd` on the installed extension:

```
/home/sandbox/.local/lib/python3.11/site-packages/faiss/_swigfaiss.so: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by /home/sandbox/.local/lib/python3.11/site-packages/faiss/_swigfaiss.so)
/home/sandbox/.local/lib/python3.11/site-packages/faiss/_swigfaiss.so: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /home/sandbox/.local/lib/python3.11/site-packages/faiss/_swigfaiss.so)
	linux-vdso.so.1 (0x00007eb996f59000)
	/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 (0x00007eb992800000)
	libopenblas.so.0 => not found
	libcudart.so.12 => not found
	libcublas.so.12 => not found
	libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007eb992bb8000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007eb992400000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007eb992721000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007eb996f27000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007eb99221f000)
	/lib64/ld-linux-x86-64.so.2 (0x0000559bff138000)
```

**Implications**

- **CUDA 12 runtime** is required (`libcudart.so.12`, `libcublas.so.12`).
- **OpenBLAS** is required at runtime (`libopenblas.so.0`).
- Also linked: `libjemalloc.so.2`, `libgomp.so.1`, standard C/C++ libs.
- The extension is an **ELF64 x86-64** shared object (Linux). Do not install on macOS/Windows.

> If you see `ImportError: libopenblas.so.0: cannot open shared object file`, install OpenBLAS runtime; missing `libcudart.so.12` / `libcublas.so.12` means your CUDA toolkit/driver is too old or not on the loader path.

---

## 2) Installation & environment

```bash
pip install /path/to/faiss-1.12.0-py3-none-any.whl
```

**Environment variables used by the package loader** (see `faiss/loader.py`):

- `FAISS_OPT_LEVEL` — optional selector for CPU instruction set variant (`generic`, `avx2`, `avx512`, etc). If set, the loader **skips CPU feature detection**. This is useful to avoid import-time issues with some NumPy versions.
- `FAISS_DISABLE_CPU_FEATURES` — comma-separated features to disable (advanced).

> **Note on NumPy ≥ 1.24**: Some FAISS loaders probe NumPy’s *private* module paths for CPU feature detection. If you encounter an import error, set `FAISS_OPT_LEVEL=avx2` **before** importing `faiss` to bypass autodetection.

---

## 3) Package layout (what this wheel ships)

Top-level modules included:

- `faiss/_swigfaiss.so` — compiled C++/CUDA extension with GPU & CPU implementations.
- `faiss/swigfaiss.py` — SWIG-generated Python bindings around the extension (exposes classes/functions).
- `faiss/gpu_wrappers.py` — *Python* helpers for GPU workflows (multi-GPU cloning, no-index KNN, pairwise distances).
- `faiss/class_wrappers.py`, `faiss/extra_wrappers.py`, `faiss/array_conversions.py` — ergonomic wrappers, NumPy interop, parameter spaces, IO helpers.
- `faiss/contrib/torch_utils.py` and `faiss/contrib/torch/*` — PyTorch tensor interop and utilities.
- `faiss/contrib/*` — clustering, big-batch search, evaluation, datasets, ivf tools, on-disk helpers.

---

## 4) GPU resource model & streams

**Key classes**

- `faiss.StandardGpuResources` — user-facing handle that manages cuBLAS handle(s), CUDA streams and temporary/pinned memory.
  - `setTempMemory(bytes)` — cap temporary GPU memory; set to `0` to avoid internal temp buffers.
  - `setPinnedMemory(bytes)` — pin host memory for faster H2D/D2H transfers.
  - `setDefaultNullStreamAllDevices()` / `setDefaultNullStream(device)` — opt into using the default/null stream (interoperability).
  - `getDefaultStreamCurrentDevice()` / `getDefaultStream(device)` — retrieve streams for coordination.
  - `getMemoryInfo()` — structure describing current allocations (useful for agents to adapt batch sizes).

- `faiss.get_num_gpus()` — how many CUDA devices FAISS can see.

**Tip**: In multi-library scenarios (PyTorch, CuPy), call `setDefaultNullStreamAllDevices()` or explicitly set the FAISS stream to the one your framework uses to avoid unintended synchronization.

---

## 5) Single‑GPU index types available

These classes are present in the wheel (from `swigfaiss.py`):

- Flat (brute force): `GpuIndexFlatL2`, `GpuIndexFlatIP` (+ `GpuIndexFlat` & `GpuIndexFlatConfig`)
- IVF family: `GpuIndexIVFFlat`, `GpuIndexIVFPQ`, `GpuIndexIVFScalarQuantizer` (+ their `*Config` classes)
- Binary: `GpuIndexBinaryFlat` (+ `GpuIndexBinaryFlatConfig`)
- Progressive dimension builder: `GpuProgressiveDimIndexFactory` (incrementally expand dimensionality on GPU; useful for ANN warm starts).
- Progressive dimension builder: `GpuProgressiveDimIndexFactory` (incrementally expand dimensionality on GPU; useful for warm-starting ANN structures).

**Common patterns**

```python
import faiss, numpy as np

d = 128
res = faiss.StandardGpuResources()

# Flat L2
idx = faiss.GpuIndexFlatL2(res, d)
xb = np.random.randn(100_000, d).astype('float32')
xq = np.random.randn(10, d).astype('float32')
idx.add(xb)
D, I = idx.search(xq, 10)  # (10,10) distances+ids

# IVF-PQ
nlist, m, nbits = 4096, 16, 8
ivfpq = faiss.GpuIndexIVFPQ(res, d, nlist, m, nbits, faiss.METRIC_L2)
ivfpq.train(xb)            # train codebooks on GPU
ivfpq.add(xb)
ps = faiss.GpuParameterSpace(); ps.initialize(ivfpq)
ps.set_index_parameter(ivfpq, "nprobe", 64)  # probe lists at search
D2, I2 = ivfpq.search(xq, 10)
```

**Precision & memory knobs** (via `GpuClonerOptions` or `*Config` objects):

- `useFloat16` / `useFloat16CoarseQuantizer` — speed/footprint vs precision.
- `usePrecomputedTables` — precompute LUTs (IVFPQ) for higher throughput.
- `storeTransposed` — data layout tweak for `GpuIndexFlat*`.
- `indicesOptions` — choose ID storage (`INDICES_32_BIT`, `INDICES_64_BIT`, `INDICES_CPU`, `INDICES_IVF`).
- `reserveVecs` — pre-reserve inverted list capacity to reduce reallocs.
- `flatHashed` / `directMapType` — enable hashed flat indexes or direct-map variants when mixing CPU+GPU fleets.
- `flatHashed` / `directMapType` — enable hashed flat indexes or direct-map variants when mixing CPU/GPU fleets.

Supported metrics include METRIC_GOWER, METRIC_INNER_PRODUCT, METRIC_L1, METRIC_L2.

Distance input types for low-level ops include DistanceDataType_BF16, DistanceDataType_F16, DistanceDataType_F32.

---

## 6) Multi‑GPU: replication, sharding, and conversion utilities

**High-level helpers (Python):** from `faiss.gpu_wrappers`

- `index_cpu_to_all_gpus(index, co=None, ngpu=-1)` — replicate across visible GPUs (or first `ngpu` devices). Returns an `IndexReplicas`.
- `index_cpu_to_gpus_list(index, co=None, gpus=[0,1,...], ngpu=-1)` — explicit device list.
- `index_cpu_to_gpu_multiple(provider, devices, index[, options])` — SWIG binding for multi-device cloning.

**Control behavior with `faiss.GpuMultipleClonerOptions`:**

- `shard: bool` — `False` = replicate full index to each GPU (`IndexReplicas`); `True` = shard data across GPUs (IVF uses `IndexShardsIVF`, others use `IndexShards`).
- `common_ivf_quantizer: bool` — share a single coarse quantizer when sharding IVF (saves memory; improves consistency).
- `shard_type` — choose subset mapping for IVF (`IndexIVF::copy_subset_to` semantics).

**Example (2 GPUs, sharded IVF-PQ):**

```python
res = faiss.StandardGpuResources()
cpu = faiss.index_factory(d, "IVF4096,PQ16")   # build/train CPU or read from disk
# ... cpu.train(xb) ; cpu.add(xb)
co = faiss.GpuMultipleClonerOptions()
co.shard = True
co.common_ivf_quantizer = True
gpu = faiss.index_cpu_to_gpus_list(cpu, co=co, gpus=[0,1])   # sharded multi-GPU index
```

---

## 7) No‑index GPU routines (brute‑force search / pairwise distance)

These functions run directly on a GPU **without constructing an index** (from `gpu_wrappers.py`):

- `knn_gpu(res, xq, xb, k, D=None, I=None, metric=faiss.METRIC_L2, device=-1, use_cuvs=False, vectorsMemoryLimit=0, queriesMemoryLimit=0)`
- `pairwise_distance_gpu(res, xq, xb, D=None, metric=faiss.METRIC_L2, device=-1)`
- `pairwise_index_gpu_multiple(...)` — dispatch brute-force searches across multiple GPU indexes (see `gpu_wrappers.py`).
- `pairwise_index_gpu_multiple(...)` — orchestrate multi-index brute-force queries (see `gpu_wrappers.py` for signature).

**Notes for agents**

- Inputs must be contiguous `float32` or `float16` arrays (NumPy), shapes `(nq, d)` and `(nb, d)`.
- Set `device` to a specific GPU id, or `-1` to use the current CUDA device.
- `use_cuvs=True` currently raises `RuntimeError: cuVS has not been compiled into the current version so it cannot be used`. HybridSearch guards the flag through `resolve_cuvs_state` and will record `cuvs_enabled=False`/`cuvs_available=False` in `AdapterStats`.
- Use `vectorsMemoryLimit` / `queriesMemoryLimit` or `bfKnn_tiling(...)` to cap workspace footprint on large problems.
- If/when a cuVS-enabled build is shipped, diagnostics/logging live under `.venv/lib/python3.13/site-packages/cuvs/common`.
- When cuVS kernels execute, diagnostics and additional helpers live in `.venv/lib/python3.13/site-packages/cuvs/common`; enable logging if you need kernel-level visibility.

**Example**

```python
res = faiss.StandardGpuResources()
D, I = faiss.knn_gpu(res, xq, xb, k=10, metric=faiss.METRIC_INNER_PRODUCT, device=0, use_cuvs=False)
```

---

## 8) PyTorch tensor interoperability

Importing `faiss.contrib.torch_utils` enables **direct use of `torch.Tensor`** (CPU or GPU) in place of NumPy arrays; the module will handle pointer extraction and stream synchronization when FAISS GPU is available.

```python
import torch, faiss
import faiss.contrib.torch_utils  # enable tensor interop

res = faiss.StandardGpuResources()
idx = faiss.GpuIndexFlatIP(res, d)

xb = torch.randn(100_000, d, device="cuda", dtype=torch.float32)
xq = torch.randn(10, d, device="cuda", dtype=torch.float32)
idx.add(xb)                 # torch -> Faiss (GPU)
D, I = idx.search(xq, 10)   # returns torch-backed arrays via SWIG pointers
```

---

## 9) Persistence & device transfers

- **CPU↔GPU**: `index_gpu_to_cpu(gpu_index)` / `index_cpu_to_gpu(res, device, cpu_index, options=None)`
- **Multi-GPU**: `index_cpu_to_gpu_multiple(...)` and helpers in `gpu_wrappers`.
- **Disk IO**: FAISS saves **CPU** indexes. To persist a GPU index, first convert to CPU:

```python
cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index, "/path/to/index.faiss")
# later
cpu2 = faiss.read_index("/path/to/index.faiss")
gpu2 = faiss.index_cpu_to_gpu(res, 0, cpu2)
```

Binary indexes use `read_index_binary` / `write_index_binary`.

---

## 10) Tuning checklist for agents

- **Recall vs speed (IVF)**: Increase `nprobe` via `GpuParameterSpace` (e.g., 16 → 256). Initialize first to set reasonable defaults.
  ```python
  ps = faiss.GpuParameterSpace(); ps.initialize(index); ps.set_index_parameter(index, "nprobe", 64)
  ```
- **Precision**: Try `useFloat16=True` (or BF16 if supported) for throughput; validate quality on your data.
- **Memory**: `StandardGpuResources.setTempMemory(…); setPinnedMemory(…)`; for brute force, use tiling limits.
- **Multi-GPU strategy**: *Replicate* small/medium datasets; *Shard* very large datasets. Consider `common_ivf_quantizer=True` for IVF sharding.
- **Interoperability**: Align CUDA streams with other frameworks; consider `setDefaultNullStreamAllDevices()`.
- **IDs**: Choose `INDICES_32_BIT` unless you expect more than 2^31-1 vectors; otherwise use 64-bit.
- **cuVS flag**: HybridSearch exposes the `use_cuvs` knob through `DenseIndexConfig` and tracks outcomes via `AdapterStats`. With the shipped wheel, `use_cuvs=True` triggers a runtime guard (`resolve_cuvs_state` forces it off and records `cuvs_enabled=False`) because the binary was compiled without cuVS kernels.

---

## 11) cuVS bindings & current limitations

The repository includes NVIDIA cuVS Python packages even though the bundled FAISS binary cannot execute cuVS kernels:

- `.venv/lib/python3.13/site-packages/cuvs` (version **25.10.00**) exposes Python wrappers for ANN algorithms (CAGRA, IVF-Flat/PQ, Vamana, NN-Descent), clustering, distance utilities, and preprocessing modules. Only the pure-Python pieces are present—the expected compiled extensions (`*.so`) for subpackages such as `cuvs.neighbors.vamana` are missing, so importing them raises `ModuleNotFoundError`.
- `.venv/lib/python3.13/site-packages/libcuvs` provides the loader shim (`load.py`) and metadata (`VERSION`, `GIT_COMMIT`) but no `lib64/libcuvs.so` suitable for dispatch. Consequently, FAISS’s `should_use_cuvs(...)` returns `False` when given a `GpuIndexConfig`, and calling `knn_gpu(..., use_cuvs=True)` raises:

```
RuntimeError: cuVS has not been compiled into the current version so it cannot be used.
```

Example guard (mirrors HybridSearch behaviour):

```python
cfg = faiss.GpuIndexFlatConfig()
if not faiss.should_use_cuvs(cfg):
    # safe: cuVS kernels unavailable, stay on FAISS implementation
    D, I = faiss.knn_gpu(res, xq, xb, k, use_cuvs=False)
else:
    D, I = faiss.knn_gpu(res, xq, xb, k, use_cuvs=True)
```

- HybridSearch’s `resolve_cuvs_state()` (see `store.py`) checks for `faiss.knn_gpu`, calls `faiss.should_use_cuvs(...)`, records telemetry (`AdapterStats.cuvs_*`), and disables cuVS when unavailable. `ManagedFaissAdapter._apply_use_cuvs_parameter()` echoes the applied value back into stats/logs so operators know whether requests took effect.
- Future wheel updates could ship the missing shared objects (`libcuvs.so` plus per-operator extensions). When that happens, re-run smoke tests with `use_cuvs=True`, monitor `AdapterStats`/logs, and update this reference to document behaviour changes.

---

## 12) What’s actually present in this wheel (symbol inventory)

**GPU index classes**
- `GpuIndex`
- `GpuIndexBinaryFlat`
- `GpuIndexBinaryFlatConfig`
- `GpuIndexConfig`
- `GpuIndexFlat`
- `GpuIndexFlatConfig`
- `GpuIndexFlatIP`
- `GpuIndexFlatL2`
- `GpuIndexIVF`
- `GpuIndexIVFConfig`
- `GpuIndexIVFFlat`
- `GpuIndexIVFFlatConfig`
- `GpuIndexIVFPQ`
- `GpuIndexIVFPQConfig`
- `GpuIndexIVFScalarQuantizer`
- `GpuIndexIVFScalarQuantizerConfig`

**Other GPU classes / utilities**
- `GpuClonerOptions`
- `GpuDistanceParams`
- `GpuIcmEncoder`
- `GpuIcmEncoderFactory`
- `GpuMultipleClonerOptions`
- `GpuParameterSpace`
- `GpuProgressiveDimIndexFactory`
- `GpuResources`
- `GpuResourcesProvider`
- `GpuResourcesProviderFromInstance`
- `GpuResourcesVector`

**GPU-related functions (top-level)**
- `bfKnn()`
- `bfKnn_tiling()`
- `bruteForceKnn()`
- `gpu_profiler_start()`
- `gpu_profiler_stop()`
- `gpu_sync_all_devices()`
- `index_binary_cpu_to_gpu_multiple()`
- `index_binary_gpu_to_cpu()`
- `index_cpu_to_gpu()`
- `index_cpu_to_gpu_multiple()`
- `index_gpu_to_cpu()`
- `should_use_cuvs()`

**Constants**
- Metrics: METRIC_GOWER, METRIC_INNER_PRODUCT, METRIC_L1, METRIC_L2
- Distance data types: DistanceDataType_BF16, DistanceDataType_F16, DistanceDataType_F32
- Indices options: INDICES_32_BIT, INDICES_64_BIT, INDICES_CPU, INDICES_IVF

---

## 13) Quick smoke test (from your build script)

This is the same check your script performs after building the wheel:

```python
import faiss, numpy as np
print("Faiss:", getattr(faiss, "__version__", "<no __version__>"),
      "GPUs:", faiss.get_num_gpus())
r = faiss.StandardGpuResources()
idx = faiss.GpuIndexFlatL2(r, 128)
x = np.random.RandomState(0).randn(1000,128).astype('float32')
idx.add(x); D,I = idx.search(x[:3], 3)
print("OK. top-1:", float(D[0,0]))
```

---

## 14) Notes from the provided build script

```
#!/usr/bin/env bash
set -Eeuo pipefail

### ── config (override via env) ──────────────────────────────────────────────
: "${FAISS_VERSION:=v1.12.0}"     # git tag/branch (e.g., v1.12.0)
: "${REPO_URL:=https://github.com/facebookresearch/faiss.git}"
: "${WORKDIR:=${PWD}/faiss-src}"  # where to clone/build
: "${WHEELS_DIR:=${HOME}/wheels}" # where to archive the built wheel
: "${FAISS_OPT_LEVEL:=avx2}"      # good default for Ryzen
: "${CUDA_HOME:=}"                # auto-detected if empty
: "${CMAKE_GENERATOR:=Ninja}"     # falls back to default if Ninja missing

### ── guard: must be in an activated venv ───────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: please activate the target virtualenv first (e.g., 'source .venv/bin/activate')"
  exit 1
fi

PYTHON_EXE="$(command -v python)"
PIP_EXE="$(command -v python)-m pip"

### ── detect CUDA_HOME if not provided ──────────────────────────────────────
if [[ -z "${CUDA_HOME}" ]]; then
  for C in /usr/local/cuda-13.0 /usr/local/cuda-12.9 /usr/local/cuda; do
    [[ -x "$C/bin/nvcc" ]] && CUDA_HOME="$C" && break
  done
fi
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "ERROR: nvcc not found. Set CUDA_HOME (e.g., /usr/local/cuda-12.9) and re-run."
  exit 1
fi

### ── detect compute capability (arch) ──────────────────────────────────────
ARCH_DEFAULT=90
if command -v nvidia-smi >/dev/null 2>&1; then
  CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')"
  if [[ "$CAP" =~ ^[0-9]+$ ]]; then ARCH_DEFAULT="$CAP"; fi
fi
: "${CMAKE_CUDA_ARCHITECTURES:=${ARCH_DEFAULT}}"

...
  -DPython_EXECUTABLE="$PYTHON_EXE" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  -DCMAKE_INSTALL_PREFIX="$("$PYTHON_EXE" - <<'PY'
import sys; print(sys.prefix)
PY
)"

### ── build & install C++ core and SWIG module ──────────────────────────────
cmake --build build -j"$(nproc)" --target faiss swigfaiss
cmake --install build

### ── build a wheel from the *build tree* (safe) ────────────────────────────
pushd build/faiss/python >/dev/null
rm -rf dist build *.egg-info contrib
# copy contrib (not symlink) so setup.py can see it in CWD
rsync -a --delete ../../../contrib/ ./contrib/
$PIP_EXE install -U setuptools wheel
python setup.py bdist_wheel
WHEEL="$(ls -1 dist/faiss-*.whl | tail -n1)"
popd >/dev/null

### ── install the wheel into this venv & archive it ─────────────────────────
$PIP_EXE install "$WHEEL"
mkdir -p "$WHEELS_DIR"
cp -v "$WHEEL" "$WHEELS_DIR/"

### ── smoke test ────────────────────────────────────────────────────────────
python - <<'PY'
import faiss, numpy as np
print("Faiss:", getattr(faiss, "__version__", "<no __version__>"),
      "GPUs:", faiss.get_num_gpus())
r = faiss.StandardGpuResources()
idx = faiss.GpuIndexFlatL2(r, 128)
x = np.random.RandomState(0).randn(1000,128).astype('float32')
idx.add(x); D,I = idx.search(x[:3], 3)
print("OK. top-1:", float(D[0,0]))
PY

echo "✅ Done. Wheel archived in: ${WHEELS_DIR}"
```

**Highlights inferred**

- Builds from `1.12.0` with GPU enabled, producing a wheel and running a GPU smoke test.
- Supports environment overrides: `FAISS_VERSION`, `REPO_URL`, `WORKDIR`, `WHEELS_DIR`, `FAISS_OPT_LEVEL`, `CUDA_HOME`, `CMAKE_GENERATOR`.
- Uses a virtualenv and standard `python setup.py bdist_wheel` (per the excerpt).
- Post-build smoke test constructs `StandardGpuResources`, builds a `GpuIndexFlatL2`, and runs a small query batch.

---

## 15) Common pitfalls & remedies

- **`ImportError: libopenblas.so.0 / libcudart.so.12 / libcublas.so.12`** → Install matching OpenBLAS and CUDA 12 runtimes; ensure they are on `LD_LIBRARY_PATH` / default linker path.
- **NumPy import during `faiss` import** → set `FAISS_OPT_LEVEL=avx2` to bypass CPU feature probing.
- **“CUDA device not found” / `get_num_gpus()==0`** → check driver version, `CUDA_VISIBLE_DEVICES`, and device permissions; validate with `nvidia-smi`.
- **“not enough memory” on large searches** → lower batch size, enable tiling (`vectorsMemoryLimit`, `queriesMemoryLimit`), reduce `nprobe`, or use sharding across GPUs.
- **Saving GPU indexes** → always convert to CPU with `index_gpu_to_cpu` before writing to disk.

---

## 16) Minimal recipes (copy‑paste)

**A. Build an IVFPQ index entirely on GPU and query it**
```python
import faiss, numpy as np
d, nlist, m, nbits = 256, 8192, 32, 8
res = faiss.StandardGpuResources()
idx = faiss.GpuIndexIVFPQ(res, d, nlist, m, nbits, faiss.METRIC_L2)
xb = np.random.randn(2_000_000, d).astype('float32')
idx.train(xb)              # train centroids + PQ on GPU
idx.add(xb)
ps = faiss.GpuParameterSpace(); ps.initialize(idx)
ps.set_index_parameter(idx, "nprobe", 64)
xq = np.random.randn(1000, d).astype('float32')
D, I = idx.search(xq, 10)
```

**B. Convert an existing CPU IVF-PQ index to 4 GPUs (sharded)**
```python
cpu = faiss.index_factory(384, "IVF16384,PQ32x8", faiss.METRIC_INNER_PRODUCT)
# ... cpu.train(xb); cpu.add(xb)
co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.common_ivf_quantizer = True
gpu = faiss.index_cpu_to_gpus_list(cpu, co=co, gpus=[0,1,2,3])
```

**C. Run brute‑force kNN on a single GPU without building an index**
```python
res = faiss.StandardGpuResources()
D, I = faiss.knn_gpu(res, xq, xb, k=100, metric=faiss.METRIC_L2, device=0,
                     vectorsMemoryLimit=512*1024*1024, queriesMemoryLimit=256*1024*1024)
```

---

### Appendix — snippets from `gpu_wrappers.py` (signatures)

```
def knn_gpu(res, xq, xb, k, D=None, I=None, metric=METRIC_L2, device=-1, use_cuvs=False, vectorsMemoryLimit=0, queriesMemoryLimit=0):
    """
    Compute the k nearest neighbors of a vector on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    xq : array_like
        Query vectors, shape (nq, d) where d is appropriate for the index.
        `dtype` must be float32.
    xb : array_like
        Database vectors, shape (nb, d) where d is appropriate for the index.
        `dtype` must be float32.


def pairwise_distance_gpu(res, xq, xb, D=None, metric=METRIC_L2, device=-1):
    """
    Compute all pairwise distances between xq and xb on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    xq : array_like
        Query vectors, shape (nq, d) where d is appropriate for the index.
        `dtype` must be float32.
    xb : array_like
        Database vectors, shape (nb, d) where d is appropriate for the index.
        `dtype` must be float32.
    D : array_like, optional
        Output array fo
```

---

If you’d like, I can tailor additional examples to your exact GPU count, memory budget, and target recall/latency envelope.
