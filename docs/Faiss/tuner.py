# tuner.py
import time
import numpy as np
import pandas as pd

def recall_at_k(I_pred: np.ndarray, I_true: np.ndarray, k: int) -> float:
    """
    Mean recall@k across queries, comparing predicted indices to exact ground truth.
    """
    assert I_pred.shape == I_true.shape
    nq, kk = I_pred.shape
    assert kk >= k
    I_pred_k = I_pred[:, :k]
    I_true_k = I_true[:, :k]
    # Compute intersection sizes per row efficiently
    hits = 0
    for i in range(nq):
        hits += len(set(I_pred_k[i]).intersection(set(I_true_k[i])))
    return hits / float(nq * k)

def time_block(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    dur = (time.perf_counter() - start) * 1000.0  # ms
    return out, dur

def write_sweep_csv(rows, out_csv: str):
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df
