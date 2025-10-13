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

### ── prep python build tools in this venv ──────────────────────────────────
$PIP_EXE install -U pip setuptools wheel numpy packaging

### ── optional: avoid mixing prior faiss installs ───────────────────────────
$PIP_EXE uninstall -y faiss faiss-cpu faiss-gpu faiss-gpu-cu12 || true

### ── clone or refresh repo ─────────────────────────────────────────────────
mkdir -p "$(dirname "$WORKDIR")"
if [[ -d "$WORKDIR/.git" ]]; then
  git -C "$WORKDIR" fetch --tags --prune
  git -C "$WORKDIR" reset --hard
  git -C "$WORKDIR" checkout "$FAISS_VERSION"
  git -C "$WORKDIR" pull --ff-only || true
else
  git clone --depth 1 --branch "$FAISS_VERSION" "$REPO_URL" "$WORKDIR"
fi

cd "$WORKDIR"

### ── configure build ───────────────────────────────────────────────────────
rm -rf build
GEN_ARGS=()
if command -v ninja >/dev/null 2>&1; then GEN_ARGS=(-G "$CMAKE_GENERATOR"); fi

cmake -S . -B build "${GEN_ARGS[@]}" \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=OFF \
  -DFAISS_ENABLE_PERF_TESTS=OFF \
  -DFAISS_ENABLE_BENCHMARK=OFF \
  -DFAISS_ENABLE_EXAMPLES=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_gflags=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_glog=ON \
  -DBLA_VENDOR=OpenBLAS \
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
