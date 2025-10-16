# === NAVMAP v1 ===
# {
#   "module": "scripts.torchcheck",
#   "purpose": "Utility script for torchcheck workflows",
#   "sections": [
#     {
#       "id": "module_body",
#       "name": "Module Body",
#       "anchor": "MB",
#       "kind": "infra"
#     },
#     {
#       "id": "module_body",
#       "name": "Module Body",
#       "anchor": "MB1",
#       "kind": "infra"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Torch Environment Diagnostics

This script reports the installed PyTorch version along with the CUDA
architectures that the runtime recognizes. It is primarily used during
CI setup and developer onboarding to confirm GPU compatibility before
running hybrid search benchmarks or embedding jobs.

Usage:
    python scripts/torchcheck.py

Dependencies:
- torch: Used for runtime introspection of CUDA capabilities
"""
import torch

print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_arch_list())
