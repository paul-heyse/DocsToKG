# Toolchain & Torch aligned to CUDA 12.9
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"


# Restrict ALL arch selection points to Blackwell
export TORCH_CUDA_ARCH_LIST="12.0"         # PyTorch extension arch list (numeric form)
export CMAKE_CUDA_ARCHITECTURES=120        # CMake-native arch  # feeds the cmake var QuTLASS inspects
export VLLM_TARGET_CUDA_ARCHS="120"
# Disable MoE (so nothing references sm100 kernels)
export VLLM_USE_MOE=0                      # if your tree honors it
export VLLM_BUILD_MOE=0
export VLLM_BUILD_FA=1
export VLLM_BUILD_FA3=1

export CMAKE_ARGS="$CMAKE_ARGS \
  -DCUDA_ARCHS=12.0a \
  -DMARLIN_ARCHS=12.0a \
  -DBUILD_MARLIN_MOE:BOOL=OFF \
  -DENABLE_MOE:BOOL=OFF \
  -DBUILD_FLASH_ATTN:BOOL=ON \
  -DFA2_ARCHS=12.0a \
  -DBUILD_FLASH_ATTN_V2:BOOL=ON \
  -DBUILD_FLASH_ATTN_V3:BOOL=OFF"

export VLLM_ATTENTION_BACKEND=FLASHINFER
# (If not honored, quick surgical fallback:)
# sed -i '/moe/Id' vllm/csrc/CMakeLists.txt

# Clean & build
#pip uninstall -y vllm
#rm -rf build/ dist/ *.egg-info
#pip install . --no-build-isolation --no-deps --force-reinstall -v
