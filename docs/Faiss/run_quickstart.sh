#!/usr/bin/env bash
set -euo pipefail

# Example runs (adjust to your GPU memory and patience)

# 1) Flat, cosine
python faiss_gpu_scaffold.py quickstart --index flat --metric cosine --d 256 --nb 200000 --nq 10000 --k 10 --gpu 1 --device 0 --eval

# 2) IVF-Flat, L2
python faiss_gpu_scaffold.py quickstart --index ivfflat --metric l2 --d 256 --nb 500000 --nq 10000 --k 10 --nlist 4096 --nprobe 32 --gpu 1 --device 0 --eval

# 3) IVF-PQ, IP
python faiss_gpu_scaffold.py quickstart --index ivfpq --metric ip --d 256 --nb 500000 --nq 10000 --k 10 --nlist 4096 --M 32 --nbits 8 --nprobe 32 --gpu 1 --device 0 --eval
