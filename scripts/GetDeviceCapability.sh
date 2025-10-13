python - <<'PY'
import torch
maj, min = torch.cuda.get_device_capability(0)
print(f"{maj}.{min}")
PY
# Suppose this prints e.g. 10.0  (whatever your 5090 reports)

export TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch; m,n=torch.cuda.get_device_capability(0); print(f"{m}.{n}")
PY
)"
