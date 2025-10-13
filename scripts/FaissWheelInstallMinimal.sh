#prerequisites
sudo apt update
sudo apt install -y build-essential cmake ninja-build swig git \
                    libopenblas-dev libomp-dev python3-dev rsync


# from repo root of the faiss repo
export CUDA_HOME=/usr/local/cuda-12.9     # or 13.0
export VENV_PREFIX="$(python -c 'import sys; print(sys.prefix)')"

rm -rf build
cmake -S . -B build -G Ninja \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=OFF -DFAISS_ENABLE_PERF_TESTS=OFF \
  -DFAISS_ENABLE_BENCHMARK=OFF -DFAISS_ENABLE_EXAMPLES=OFF \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_INSTALL_PREFIX="$VENV_PREFIX"

cmake --build build -j"$(nproc)" --target faiss swigfaiss
cmake --install build

# make wheel from build tree (safe)
cd build/faiss/python
rm -rf dist build *.egg-info contrib
rsync -a --delete ../../../contrib/ ./contrib/
python -m pip install -U setuptools wheel
python setup.py bdist_wheel
python -m pip install dist/faiss-*.whl

#sanity check after install

python - <<'PY'
import faiss, os, sys
print("Faiss", faiss.__version__, "from", faiss.__file__)
print("GPUs:", faiss.get_num_gpus())
PY

# if you ever see a loader error for libfaiss*.so
export LD_LIBRARY_PATH="$(python - <<'PY'
import sys,sysconfig; print(sys.prefix + '/' + (sysconfig.get_config_var('LIBDIR') or 'lib'))
PY
):${LD_LIBRARY_PATH:-}"
