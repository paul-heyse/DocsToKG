export USE_NINJA=1                                  # let CMake/setuptools pick Ninja
export MAX_JOBS=$(( $(nproc) - 4 ))                 # leave a little headroom (e.g., 28 on a 32-thread box)
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
export NVCC_THREADS=1                               # avoid double-oversubscription
