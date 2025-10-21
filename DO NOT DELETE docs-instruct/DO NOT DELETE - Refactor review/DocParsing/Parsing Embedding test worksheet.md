Absolutely — here’s a **comprehensive “parity test worksheet”** you can hand directly to your team/agents. It defines what to compare, how to run both sides, what metrics to compute, hard gates to pass/fail a build, and how to plug it into CI. It assumes your **baseline** is the legacy embedding path and your **candidate** is the providers path (same models/backends, just routed through the new provider interfaces).

---

# 0) Objective & scope

**Goal:** Prove the **providers implementation** is a drop-in replacement for legacy embedding logic by showing **functional parity** (vectors & files) and **operational parity** (timings, manifests, footers) on controlled datasets.

**Families covered:** **Dense** (Qwen/vLLM, ST, TEI), **Sparse** (SPLADE), **Lexical** (BM25).
**Outputs compared:** Parquet/JSONL vectors, manifests, and Parquet footers (provenance).
**Non-goals:** Cross-backend parity (e.g., TEI vs vLLM) — only compare **same backend** legacy vs provider.

---

# 1) Preconditions (freeze everything that affects outputs)

* **Config lock:** Use `docparse config show --format yaml > .docparse.lock.yaml` on both runs. Make sure:

  * `embedding.normalize_l2=true` for dense (or both sides disabled consistently).
  * **Same** model ids & revisions (dense ST/vLLM/SPLADE), same tokenizer revisions, same BM25 params (k1, b, stopwords, tokenizer, df thresholds).
  * TEI: same server URL & model; stable version.
* **Offline determinism:** If possible, run with `embedding.offline=true` for local models. TEI is networked (see §8).
* **Dtypes:** Prefer identical weight dtypes (e.g. FP16/BF16 inside the model, but vectors serialized as **float32**).
* **Normalization:** Decide once (on/off) — parity tolerances below assume **on** for dense.
* **Chunk inputs identical:** Chunks files (JSONL/Parquet) must match byte-for-byte source of both runs.
* **Seed/stability:** Set `random_seed` in config (even if not used, document it).

---

# 2) Test datasets (three tiers)

1. **Smoke** (≈ 3–5 docs; mixed HTML/PDF; tiny tables): validate wiring fast (<30s).
2. **Mini** (≈ 80–120 docs; varied lengths; has formulas, tables, headings): primary parity gate.
3. **Hetero** (optional; ≈ 300–500 docs): ensures tail-latency/transient issues don’t mask mismatches.

For each: preserve a **document list file** (`doclist.txt`) and an **expected chunk count** range.

---

# 3) Execution plan (two runs, two roots)

We’ll write to two separate data roots to avoid accidental overwrites:

* **Baseline (legacy):**

  ```bash
  export DOCSTOKG_EMBED_PROVIDERS=off
  python -m DocsToKG.DocParsing.cli embed run \
    --chunks-dir Data/Chunks \
    --out-dir Data_legacy/Vectors \
    --vector-format parquet \
    --enable-dense --enable-sparse --enable-lexical
  ```

* **Candidate (providers):**

  ```bash
  export DOCSTOKG_EMBED_PROVIDERS=on
  python -m DocsToKG.DocParsing.cli embed run \
    --chunks-dir Data/Chunks \
    --out-dir Data_providers/Vectors \
    --vector-format parquet \
    --enable-dense --enable-sparse --enable-lexical
  ```

**Notes**

* Keep the same `workers`, `policy`, and per-family batch sizes.
* For TEI, keep the same `max_inflight` and `batch_size` on both runs.
* Save manifests from both runs.

---

# 4) Alignment rules (what rows match)

* Join on `(doc_id, chunk_id, family)` for vector rows.
* Restrict to **intersection** of keys in baseline and candidate; any missing from one side is a **hard fail** (except documented skip due to quarantine — then it must match on both sides).
* Sort order **within files** must match chunk order (flag mismatch if out-of-order and counts differ).

---

# 5) Metrics & thresholds (by family)

## 5.1 Dense (Qwen/vLLM, ST, TEI — same backend both sides)

**Per vector**

* **Cosine similarity** between `v_legacy` and `v_prov`:

  * **Gate #1 (strict)**: `cos ≥ 0.9990`
  * **Gate #2 (distribution)**: `p01(cos) ≥ 0.9980`, `median(cos) ≥ 0.9997`
* **Norm check** (if L2 on): `||v|| ∈ [0.9995, 1.0005]` for both; **max deviation** ≤ 5e-4
* **Dimension equality**: `dim_legacy == dim_prov`

**Aggregate**

* Fraction of vectors passing Gate #1 ≥ **0.995** (99.5%)
* **Hard fail** if any doc has > 1% of its vectors with `cos < 0.995`

**Explanations**

* FP differences (e.g., different matmul kernels) can cause microscopic drift; thresholds above are conservative but strict enough to catch real errors.

## 5.2 Sparse (SPLADE)

Let `S` be sparse vector as `(indices, weights)` with `nnz`.

**Per vector**

* **NNZ exact**: `nnz_legacy == nnz_prov` (preferred). If post-proc differs in tie-breakers, allow:

  * **Soft gate**: `abs(nnz_legacy − nnz_prov) ≤ 2` *and* `|mean(weights_legacy) − mean(weights_prov)| ≤ 1e-6`
* **Overlap**: Weighted **Jaccard** on top-K (K = min(100, nnz)):
  `J = Σ_min(w_i)/Σ_max(w_i)` over the union (by token id)

  * **Gate**: `J ≥ 0.995` (top-100 very similar)
* **Rank correlation** (top-K ids by weight):

  * **Spearman ρ ≥ 0.995**

**Aggregate**

* Mean `J` across vectors ≥ 0.998; p01 `J` ≥ 0.990
* **Hard fail** if any vector has `J < 0.980` or Spearman < 0.980

**Explanations**

* SPLADE should be near-deterministic given same weights/device; tiny tie-breaking or pruning edges may vary.

## 5.3 Lexical (BM25)

**Deterministic expectation** with identical tokenizer, k1, b, stopwords, and df policy.

**Per vector**

* **Exact idx set**: `indices_legacy == indices_prov` (or `terms` if term-space).
* **Weights tolerance**: `max_abs_diff(weights) ≤ 1e-8` *and* `mean_abs_diff ≤ 1e-10`.

**Aggregate**

* 100% of vectors pass exact/equivalent checks.
* **Hard fail** if any per-vector check fails.

---

# 6) Filesystem parity (footers, manifests, formats)

## 6.1 Parquet footers (each vectors file)

* Required keys (already spec’d in your footers contract):

  * `docparse.family` matches family under the path.
  * `docparse.provider` (candidate must set; legacy may be missing — warn only).
  * `docparse.model_id` identical (id@rev).
  * `docparse.dim` identical for dense.
  * `docparse.dtype` identical for all families.
  * `docparse.cfg_hash` identical (if config truly identical).
* **Hard fail**: missing family/dim/dtype on candidate; mismatch in `model_id`.

## 6.2 Manifests

* For each stage success row (per file):

  * **Counts parity**: vectors written per family equal.
  * **Vector format**: both `parquet` (or your chosen format).
  * **Duration parity** (non-blocking check): per-file `duration_s` p50 within ±20%, p95 within ±30% (informational).

**Differences to ignore**

* Wall clock timestamps; in-file row order as long as `(doc_id, chunk_id)` mapping persists and counts equal.

---

# 7) Report schema (single JSON artifact from the parity job)

Write one JSON file per run that CI can archive and humans can read. Suggested top-level keys:

```json
{
  "dataset": "mini",
  "families": ["dense", "sparse", "lexical"],
  "summary": {
    "dense": { "count": 5342, "fail": 7, "cos_min": 0.9987, "cos_p01": 0.9991, "cos_median": 0.9999 },
    "sparse": { "count": 5342, "fail": 0, "jaccard_mean": 0.9993, "jaccard_p01": 0.9958, "spearman_p01": 0.996 },
    "lexical": { "count": 5342, "fail": 0 }
  },
  "failures": [
    {
      "family": "dense",
      "doc_id": "Papers/Nature/2025_10_gene-editing",
      "chunk_id": 42,
      "cos": 0.9941,
      "norm_legacy": 1.0003,
      "norm_prov": 0.9998,
      "dim": 1024
    }
  ],
  "footer_mismatches": [
    { "path": ".../family=dense/...parquet", "key": "docparse.model_id", "legacy": "Qwen2-7B@A", "provider": "Qwen2-7B@B" }
  ],
  "manifest_deltas": {
    "dense": { "legacy_vectors": 5342, "provider_vectors": 5342, "delta": 0 }
  }
}
```

This gets attached to CI (Artifacts) and can be pretty-printed in logs.

---

# 8) Special notes per backend

* **TEI:** The legacy path may not be TEI-based. Only assert parity when **both** legacy and provider use **TEI**. If not available, run **self-consistency**: two provider runs against the same TEI server must match thresholds above (cosine ≥ 0.999, etc.). Network variance is tolerated only through the retry layer; content differences should not occur for the same inputs.
* **vLLM:** Disable autotuning knobs that change determinism (e.g., random parallelism affecting tokenization buffers). Keep the same `dtype`, `TP`, and `max_model_len`.
* **Sentence-Transformers:** Pin `revision` for the model & tokenizer; set `max_seq_length` the same; set identical `device` and `dtype`; use memory-map option consistently.

---

# 9) CI wiring (minimal)

* **Job matrix**: `smoke` (fast), `mini` (gate), `hetero` (optional, nightly).
* **Env toggles**:

  * `DOCSTOKG_EMBED_PROVIDERS=off` for baseline, `on` for candidate.
  * `DOCSTOKG_STRICT_CONFIG=true` so unknown/deprecated keys fail fast.
  * TEI URL/KEY secrets provided via CI secrets if testing TEI.
* **Markers**:

  * `@pytest.mark.gpu` for vLLM/ST GPU tests (skip if no CUDA).
  * `@pytest.mark.network` for TEI (skip on offline CI).
* **Fail conditions** (build red):

  * Any family exceeds its **hard fail** thresholds in §5.
  * Any footer **hard fail** in §6.1.
  * Any manifest count mismatch in §6.2.

Produce the **report JSON** and echo a compact “gate summary” to the console at the end.

---

# 10) Drilldown & triage playbook (when a gate fails)

1. **Missing row keys** → Check chunk alignment, intersection filter, quarantine/skip parity.
2. **Dense cosine outliers** → Verify L2 normalization on both sides; check `dim` equality; confirm same model revision/dtype.
3. **SPLADE Jaccard < 0.98** → Check pruning & top-k knobs; ensure same tokenizer and attn backend.
4. **BM25 mismatch** → Confirm stopwords/tokenizer/min_df/max_df_ratio identical; confirm corpus stats policy (per-file vs corpus) matches.
5. **Footer mismatch** → Check ProviderFactory passed the right metadata; verify `model_id@rev` values.

Add the top 10 worst offenders to the CI log with `(doc_id, chunk_id)` and metric values.

---

# 11) Optional performance parity (informational)

* Capture per-file `duration_s` from manifests for both runs.
* Report per-family p50 & p95 deltas:

  * Dense p95 within **±30%**, p50 within **±20%** (TEI may vary with network; don’t gate unless stable).
  * SPLADE & BM25 similar ballpark.
* Use this to spot obvious regressions, but don’t fail the build until you’ve stabilized providers.

---

# 12) Minimal test scaffolds (what to implement)

* `tests/parity/test_dense_parity.py`

  * Build aligned key set; compute cosine, norms; assert thresholds & counts.
* `tests/parity/test_sparse_parity.py`

  * Compute nnz deltas, top-K weighted Jaccard & Spearman; assert gates.
* `tests/parity/test_lexical_parity.py`

  * Exact index set equality + weight tolerances.
* `tests/parity/test_footers_and_manifests.py`

  * Validate footer keys & manifest count parity.
* `tests/parity/test_report_assembly.py`

  * Assemble the JSON report; ensure fields present and values sensible.

Each test accepts **paths for baseline and candidate** (e.g., via env vars) so the same code can run on smoke/mini/hetero.

---

# 13) One-page acceptance summary (copy into the PR template)

* **Dense:** `cos_min ≥ 0.9990`, `cos_p01 ≥ 0.9980`, `median ≥ 0.9997`; ≥ 99.5% vectors pass strict gate.
* **Sparse:** mean **Weighted Jaccard (top-100)** ≥ 0.998; p01 ≥ 0.990; Spearman p01 ≥ 0.996; **nnz parity** unless documented post-proc difference.
* **Lexical:** **exact index set** & weights within 1e-8 max abs diff (1e-10 mean abs diff).
* **Footers:** provider metadata present; `model_id`, `dim`, `dtype`, `family` correct.
* **Manifests:** vector counts equal per-file and family.
* **Report:** JSON artifact attached; hard gates all green.

---

This worksheet gives you a **repeatable, objective parity gate**. If you want, I can turn it into a short **README for the `tests/parity/` folder** and a **Typer subcommand (`docparse parity run`)** that automates baseline/candidate runs and emits the JSON report in one shot.
