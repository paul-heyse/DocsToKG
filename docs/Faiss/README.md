# FAISS GPU Scaffold (Single-GPU friendly)

End-to-end, **educational** scaffolding for building, tuning, evaluating, and persisting FAISS indexes
on **one GPU** (e.g., your RTX 5090) with CPU fallback. It aims to be broadly useful across datasets,
metrics (L2 / cosine / inner product), and index families (Flat, IVF-Flat, IVF-PQ).

> Tested against the wheel you provided (`faiss==1.12.0`), which exposes GPU and CPU APIs via SWIG.
> Adapt as needed—this is a teaching scaffold, not a one-size-fits-all benchmark harness.

## Contents

- `faiss_gpu_scaffold.py` — the main script: build/train/add/search, quick evaluation, optional parameter sweeps.
- `datasets.py` — data utilities: synthetic generation and convenient loaders.
- `tuner.py` — light utilities for `nprobe` sweeps and tabular reporting.
- `requirements.txt` — minimal Python deps (you will install your custom FAISS wheel manually).
- `run_quickstart.sh` — copy-paste commands that run end-to-end with synthetic data.

## Install

1) Create a fresh environment (recommended):
```bash
python -m venv .venv && source .venv/bin/activate
# or conda/mamba/etc.
```

2) Install Python deps (FAISS itself is not listed here because you have a custom wheel):
```bash
pip install -r requirements.txt
```

3) Install your custom FAISS wheel (path will differ for you):
```bash
pip install /path/to/faiss-1.12.0-py3-none-any.whl
```

4) Verify FAISS + GPU are visible:
```bash
python -c "import faiss; print('FAISS version OK'); print('num_gpus=', getattr(faiss,'get_num_gpus',lambda:0)())"
```

## Quickstart (synthetic demo)

The commands below generate synthetic data, build an index, search, and report basic metrics.
They run in a few seconds with modest sizes—tweak `--nb`, `--nq`, `--d` to scale.

```bash
# Exact (Flat) with cosine similarity on GPU (falls back to CPU if GPU unavailable)
python faiss_gpu_scaffold.py quickstart --index flat --metric cosine --d 256 --nb 200000 --nq 10000 --k 10 --gpu 1 --device 0

# IVF-Flat (L2) on GPU
python faiss_gpu_scaffold.py quickstart --index ivfflat --metric l2 --d 256 --nb 500000 --nq 10000 --k 10 --nlist 4096 --nprobe 32 --gpu 1 --device 0

# IVF-PQ (IP) on GPU, cosine equivalent if you normalize inputs
python faiss_gpu_scaffold.py quickstart --index ivfpq --metric ip --d 256 --nb 500000 --nq 10000 --k 10 --nlist 4096 --M 32 --nbits 8 --nprobe 32 --gpu 1 --device 0
```

## Parameter sweeps (recall vs latency)

Once you have an index on disk, you can run an `nprobe` sweep on your queries to see recall/latency trade-offs:

```bash
# Build and save an IVF-Flat index
python faiss_gpu_scaffold.py build --index ivfflat --metric l2 --d 256 --nb 500000 --nlist 4096 --gpu 1 --device 0 --save_index ivfflat.faiss

# Search with sweeps (loads the saved index)
python faiss_gpu_scaffold.py search --index_path ivfflat.faiss --metric l2 --nq 10000 --k 10 --nprobe_sweep 4,8,16,32,64,128 --gpu 1 --device 0 --out_csv ivfflat_sweep.csv
```

Open the CSV to compare **avg latency/query (ms)** vs **recall@k** at different `nprobe` values.

## Tips

- **Cosine similarity**: L2-normalize your vectors and use `METRIC_INNER_PRODUCT` (IP). The script handles this when `--metric cosine` is set.
- **Train then add**: IVF/PQ/SQ indices require `train()` before `add()`.
- **Saving GPU indexes**: FAISS cannot directly serialize a GPU index. The script clones to CPU internally before saving.
- **Large data**: Use `--add_bs` and `--search_bs` to control chunking. Consider PQ (`ivfpq`) or scalar quantizers for big datasets.
- **IDs**: Use `--use_ids` to add your own int64 IDs (synthetic demo auto-generates contiguous IDs).

## Files

- `faiss_gpu_scaffold.py` — CLI entrypoint.
- `datasets.py` — dataset loading & generation utilities.
- `tuner.py` — helper to sweep parameters and write CSVs.
- `run_quickstart.sh` — example invocations.
- `requirements.txt` — `numpy`, `tqdm`, `pandas` (for CSV handling).

