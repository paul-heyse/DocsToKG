Below is a practical, end‑to‑end **field guide to Polars (Python)** aimed at an AI refactoring agent. It focuses on *how the library actually works*, the idioms you’ll use most, and what to watch out for when replacing custom dataframe/ETL code with Polars.

---

## 0) Snapshot (what you’re targeting)

* **Latest (PyPI):** `polars 1.34.0` released Oct 2, 2025. Requires **Python ≥ 3.9**. Install with `pip install polars`. Extras include `[gpu]`, `[excel]`, `[fsspec]`, `[adbc]`, `[deltalake]`, `[iceberg]`, `[all]`, etc. ([PyPI][1])
* **Key capabilities:** Eager & Lazy execution; optimizer; multi‑threaded; streaming for larger‑than‑RAM; Arrow memory model; SQL interface; optional **GPU engine** (open beta) powered by RAPIDS cuDF. ([PyPI][1])

---

## 1) Mental model (how Polars “thinks”)

1. **Data structures:** `DataFrame` (eager), `LazyFrame` (deferred). You write **expressions** (via `pl.col`, `pl.lit`, `pl.when`) that Polars composes into a **logical plan** and optimizes (predicate/projection pushdown, etc.). You *materialize* with `.collect()`. See optimizer/plan docs. ([Polars User Guide][2])
2. **Two modes:**

   * **Eager:** immediate execution (small/interactive tasks).
   * **Lazy:** build a plan, then `.collect()` — enables optimization and **streaming**. Streaming is triggered by `collect(engine="streaming")` (newer API). ([Polars User Guide][3])
3. **Columnar + Arrow:** zero‑/low‑copy interchange with PyArrow and other Arrow tools. Use `.to_arrow()` / `pl.from_arrow()`. ([Polars User Guide][4])
4. **SQL front‑end:** `pl.sql()` and `SQLContext` let you query registered frames (Polars, pandas, PyArrow) using SQL; mixing SQL and expressions is supported. ([Polars User Guide][5])

---

## 2) Install & runtime knobs

* **CPU install:** `pip install polars` (consider `polars[all]` for convenience). ([Polars User Guide][2])
* **GPU (Open Beta):** `pip install -U polars[gpu] --extra-index-url=https://pypi.nvidia.com`, then `.collect(engine="gpu")`. CPU fallback occurs when GPU cannot handle an operation. ([Polars User Guide][6])
* **Threads:** Polars is multi‑threaded. Limit threads by setting **`POLARS_MAX_THREADS` *before* importing** Polars. Query with `pl.thread_pool_size()`. ([Polars User Guide][7])
* **Notebook/printing config:** `pl.Config.set_tbl_rows/cols/width_chars/formatting` to change display. ([Polars User Guide][8])

---

## 3) Quick start (eager vs lazy + streaming)

```python
import polars as pl

# Eager
df = pl.read_csv("sales.csv")
res = (df
       .filter(pl.col("region") == "EMEA")
       .with_columns(rev_eur = pl.col("rev_usd") * 0.93)
       .group_by("account")
       .agg(pl.sum("rev_eur"))
)

# Lazy + pushdown + streaming
lf = pl.scan_parquet("s3://bucket/transactions/*.parquet")     # lazy 'scan_*' never loads all rows up front
res2 = (lf
        .filter(pl.col("ts") >= pl.datetime(2025,1,1))
        .group_by("customer_id")
        .agg(total=pl.col("amount").sum())
        .sort("total", descending=True)
        .limit(100)
        .collect(engine="streaming"))                           # execute as streaming pipeline
```

Scanning (`scan_csv`, `scan_parquet`, `scan_ndjson`, etc.) plus `collect(engine="streaming")` gives you optimized, larger‑than‑RAM pipelines. ([Polars User Guide][9])

---

## 4) I/O that replaces ad‑hoc readers

### Local & remote

* **CSV:** `pl.read_csv(...)` or `pl.scan_csv(...)` (glob patterns supported lazily). `read_csv_batched` for batch iteration. Remote paths use `fsspec` if installed. ([Polars User Guide][10])
* **Parquet / IPC / JSON / NDJSON:** `read_parquet/scan_parquet`, `read_ipc/scan_ipc`, `read_json`, `read_ndjson/scan_ndjson`. ([Polars User Guide][9])
* **Excel:** `pl.read_excel` (via external engines; prefer *calamine*). ([Polars User Guide][11])
* **Cloud storage:** S3, Azure Blob, GCS — same API; install `fsspec` and the relevant filesystem (e.g., `s3fs`). ([Polars User Guide][12])
* **Data lakes:** **Delta Lake** (`read_delta/scan_delta`, `write_delta`) and **Iceberg** (`scan_iceberg`) supported (with extras). ([Polars User Guide][13])

**Tip:** Prefer **`scan_*`** + lazy transforms for pushdown, parallel IO, and streaming; avoid `read_*().lazy()` anti‑pattern for NDJSON. ([Polars User Guide][14])

---

## 5) Selecting & transforming (expressions)

Everything revolves around **expressions**:

```python
df.select(
    pl.col("a", "b") * 2,
    pl.when(pl.col("qty") > 10).then("bulk").otherwise("std").alias("tier"),
    pl.col("item").str.to_lowercase().alias("item_lower"),
)
```

* **`select`** returns *only* the new columns. **`with_columns`** adds/overwrites while preserving others. ([Polars User Guide][2])
* **Selectors** (`pl.selectors`) pick columns by dtype, name, etc., and broadcast expressions over them, e.g. `pl.selectors.numeric() * 1.2`. ([Polars User Guide][15])
* **String/list/struct ops:** rich namespaces (`.str`, `.list`, `pl.struct`). For horizontal string concatenation across columns use `pl.concat_str([..], separator="-")`. ([Polars User Guide][16])

---

## 6) Grouping, aggregation & windows

```python
out = (df
  .group_by("customer")
  .agg(
      total = pl.col("amount").sum(),
      mean_last_30d = pl.col("amount").filter(pl.col("date") > pl.date("2025-09-20")).mean()
  )
)
```

* API is `group_by` (renamed from `groupby` in 0.19); `GroupBy.apply` became `map_groups`, and `GroupBy.count()` → `len()`. Update old code accordingly. ([pola.rs][17])
* **Time windows:**

  * **Dynamic:** `group_by_dynamic(index_column="ts", every="1d", period="1d")` for fixed windows (daily/weekly, etc.). ([Polars User Guide][18])
  * **Rolling:** `df.rolling(index_column="ts", period="7d").agg(...)` (replaces `group_by_rolling`). ([Polars User Guide][19])
* **Window functions:** `pl.col("x").rank().over("group")`, `pl.sum("x").over("g1","g2")`. ([Polars User Guide][20])

---

## 7) Joins & set‑like operations

```python
df.join(dim, on="key", how="left")           # inner|left|right|full|semi|anti|cross
df.join_asof(quotes, on="ts", strategy="nearest", tolerance="5m")
```

* Join types include **semi**/**anti** and **cross** joins; `join_asof` for nearest‑key time joins. Left join preserves left order. Validate multiplicity via `validate="1:m"`, etc. ([Polars User Guide][21])
* For non‑equi conditions that aren’t directly supported, compose `cross` + `filter`. ([Stack Overflow][22])
* **Categoricals:** joining on `pl.Categorical` often requires a **global StringCache** (`with pl.StringCache(): ...`). Use only when needed. ([Polars User Guide][23])

---

## 8) Reshaping

```python
# Wide -> Long
long = df.unpivot(index="id", on=["q1","q2"], variable_name="quarter", value_name="score")

# Long -> Wide
wide = long.pivot(index="id", columns="quarter", values="score", aggregate_function="first")
```

Use `unpivot` (preferred) / `melt` for long format; `pivot` to widen back. `pl.concat` supports vertical/diagonal concatenations; `DataFrame.vstack()` appends rows. ([Polars User Guide][24])

---

## 9) Dates/times, types & time zones

* **Datetime dtype:** `pl.Datetime(time_unit="us|ns|ms", time_zone=...)`. Use `.dt.convert_time_zone()` to convert and `.dt.replace_time_zone()` to set/replace. Time‑zone‑aware processing is first‑class. ([Polars User Guide][25])
* **Decimals:** 128‑bit fixed‑point `pl.Decimal(precision, scale)` (marked unstable — check for edge cases). ([Polars User Guide][26])
* **Categorical/Enum:** efficient string handling; consider `StringCache` when joining across frames. ([Polars User Guide][27])

---

## 10) SQL layer (use when refactoring SQL‑ish custom code)

```python
import polars as pl
df = pl.DataFrame({"a":[1,2,3], "b":[6,7,8]})
res = pl.sql("SELECT a, b*2 AS b2 FROM df", eager=True)
```

`pl.sql()` executes against objects in the *global namespace* (Polars frames, **pandas** DataFrame/Series, **PyArrow** Table/RecordBatch). Return a `LazyFrame` (default) or collect eagerly. You can also register tables with `SQLContext` for finer control. Mix SQL → native expressions. ([Polars User Guide][5])

---

## 11) User‑defined functions (UDFs) — use sparingly

* Element‑wise: `Expr.map_elements(fn, return_dtype=...)` — **much slower** than native expressions. Prefer built‑ins. ([Polars User Guide][28])
* Batch/series‑wise: `Expr.map_batches(fn, returns_scalar=...)` where your function handles a full Series/batch. ([Polars User Guide][29])
* Row‑wise: `DataFrame.map_rows` exists but is the slowest path; it materializes rows — avoid for performance. ([Polars User Guide][30])

**Rule of thumb:** if you can express it with Polars expressions, do that; only drop to Python UDFs for last‑mile logic. ([Polars User Guide][31])

---

## 12) Performance, scaling & streaming

* **Prefer `scan_*`** + lazy transforms for optimizer pushdown.
* **Streaming engine:** `lf.collect(engine="streaming")` executes in batches (lower memory, often faster on very large inputs). Tune with `pl.Config.set_streaming_chunk_size(n)`. ([Polars User Guide][3])
* **Threads:** set `POLARS_MAX_THREADS` before import to cap CPU parallelism (handy in oversubscribed environments or when embedding inside other engines). ([Polars User Guide][7])
* **GPU engine (open beta):** `.collect(engine="gpu")` for acceleration; falls back to CPU when needed. Newer releases add streaming on GPU (experimental). ([Polars User Guide][6])

---

## 13) Interoperability

* **Arrow:** `df.to_arrow()` ↔ `pl.from_arrow(table)` (mostly zero‑copy). ([Polars User Guide][4])
* **pandas:** convert freely (esp. with pandas’ PyArrow dtypes). ([Python⇒Speed][32])
* **DuckDB:** can read/write Polars via Arrow; install PyArrow for the bridge. ([DuckDB][33])

---

## 14) Common refactors (from custom/pandas-ish code)

**Selecting multiple numeric columns to scale**

```python
# ❌ custom loop over df.columns
# ✅ selectors broadcast
df = df.with_columns(pl.selectors.numeric() * 1.05)
```

(Selectors: choose by dtype/name/pattern.) ([Polars User Guide][15])

**Large CSVs with filtering during read**

```python
# ❌ read then filter (loads everything)
# ✅ pushdown via scan + lazy
lf = pl.scan_csv("s3://bucket/*.csv")
top = (lf.filter(pl.col("country")=="US")
        .select("user_id","amount")
        .group_by("user_id").agg(total=pl.sum("amount"))
        .sort("total", descending=True).limit(100)
        .collect(engine="streaming"))
```

(Cloud storage via `fsspec`/S3/Azure/GCS.) ([Polars User Guide][12])

**As‑of join on time**

```python
trades.join_asof(quotes, on="ts", strategy="nearest", tolerance="1m")
```

(Requires both frames sorted on the join key.) ([Polars User Guide][34])

**Pivot/unpivot**

```python
long = df.unpivot(index="id", on=["m1","m2"], variable_name="metric", value_name="val")
wide = long.pivot(index="id", columns="metric", values="val", aggregate_function="first")
```

([Polars User Guide][24])

**Windowed ranks per group**

```python
df.select(
  "grp", "score",
  pl.col("score").rank("dense", descending=True).over("grp").alias("rank_in_grp")
)
```

([Polars User Guide][20])

---

## 15) Gotchas & best practices

* **Use `group_by` (not `groupby`)**; update old APIs (e.g., `GroupBy.count()` → `len()`). ([pola.rs][17])
* **Categorical joins:** enable `StringCache` around both frames when needed. Don’t leave it on globally unless necessary. ([Polars User Guide][23])
* **Avoid per‑row Python code** (`iter_rows`, `map_rows`) and per‑element Python UDFs (`map_elements`) unless unavoidable. Use expressions. ([Polars User Guide][30])
* **Time windows:** `group_by_dynamic` needs the index column sorted; `rolling()` differs from dynamic windows (value‑driven vs fixed interval). ([Polars User Guide][18])
* **Streaming:** not every plan can stream; Polars will fall back to the in‑memory engine when needed. Prefer scans + projections/filters early. ([Polars User Guide][35])

---

## 16) API map (high‑value functions you’ll call a lot)

* **I/O:** `read_csv/scan_csv`, `read_parquet/scan_parquet`, `read_json/read_ndjson/scan_ndjson`, `read_excel`, `read_delta/scan_delta`, `scan_iceberg`. ([Polars User Guide][9])
* **Core transforms:** `select`, `with_columns`, `filter`, `sort`, `unique`, `drop_nulls`, `fill_null`. ([Polars User Guide][36])
* **Grouping & windows:** `group_by().agg(...)`, `group_by_dynamic`, `rolling`, `Expr.over`, `Expr.rolling`. ([Polars User Guide][37])
* **Joins:** `DataFrame.join`, `join_asof` (+ `LazyFrame.join` for lazy plans). ([Polars User Guide][21])
* **Selectors:** `pl.selectors.numeric/string/matches/...`. ([Polars User Guide][15])
* **Reshaping & concat:** `unpivot/melt`, `pivot`, `pl.concat`, `vstack`. ([Polars User Guide][24])
* **SQL:** `pl.sql(...)`, `SQLContext`. ([Polars User Guide][5])
* **Streaming/GPU execution:** `LazyFrame.collect(engine="streaming"|"gpu")`, `pl.Config.set_streaming_chunk_size`, `pl.GPUEngine(...)`. ([Polars User Guide][3])

---

## 17) Minimal recipes (copy/paste)

**1) Top N per group with window**

```python
(df
 .with_columns(
   rank=pl.col("metric").rank("dense", descending=True).over("group")
 )
 .filter(pl.col("rank") <= 3)
)
```

([Polars User Guide][20])

**2) Rolling 7‑day sum**

```python
(df
 .rolling(index_column="ts", period="7d")
 .agg(total=pl.col("y").sum())
)
```

([Polars User Guide][19])

**3) Dynamic (daily) aggregates**

```python
(df
 .group_by_dynamic("ts", every="1d", period="1d")
 .agg(total=pl.col("y").sum())
)
```

([Polars User Guide][18])

**4) As‑of join (nearest quote for each trade)**

```python
trades.join_asof(quotes, on="timestamp", strategy="nearest", tolerance="5m")
```

([Polars User Guide][34])

**5) SQL + native mixing**

```python
lf = pl.LazyFrame({"a":[1,2,3],"b":[6,7,8]})
out = (pl.sql("SELECT a, b*2 AS b2 FROM lf")          # returns LazyFrame
         .filter(pl.col("b2") < 16)
         .collect())
```

([Polars User Guide][5])

---

## 18) When/why to prefer Polars over custom code

* **Streaming + pushdown** replace bespoke chunked readers and ad‑hoc filters. Use `scan_*` + `collect(engine="streaming")`. ([Polars User Guide][3])
* **Time‑aware joins/windows** (`join_asof`, `group_by_dynamic`, `rolling`) replace hand‑rolled time bucketing and nearest‑neighbor logic. ([Polars User Guide][34])
* **Selectors & expression API** replace column loops, dtype checks, and per‑column broadcasting you might be doing manually. ([Polars User Guide][15])
* **SQL gateway** absorbs legacy SQL fragments without keeping a separate SQL engine in your app. ([Polars User Guide][5])

---

## 19) References you’ll consult most

* **Getting started & concepts:** user guide sections on expressions, lazy, IO, SQL, streaming. ([Polars User Guide][2])
* **Python API index:** browse all classes/functions. ([Polars User Guide][38])
* **Streaming:** concept + `collect(engine="streaming")`. ([Polars User Guide][35])
* **Joins:** join types + as‑of. ([Polars User Guide][21])
* **Selectors:** reference. ([Polars User Guide][15])
* **GPU engine:** overview & usage. ([Polars User Guide][39])

---

### Final notes for an AI refactoring agent

* Default to **Lazy** pipelines with `scan_*`.
* Aggressively **prefer expressions** over Python UDFs.
* Use **`collect(engine="streaming")`** for big data; fall back to CPU engine automatically on GPU when unsupported. ([Polars User Guide][3])
* Be mindful of API renames (e.g., `groupby` → `group_by`). ([pola.rs][17])
* For categorical joins across frames, **wrap in `pl.StringCache()`**. ([Polars User Guide][23])

If you want a deeper dive on a specific transformation you’re refactoring, tell me what the current custom code does (input shape → output shape + rules), and I’ll translate it into an idiomatic, optimized Polars pipeline.

[1]: https://pypi.org/project/polars/ "polars · PyPI"
[2]: https://docs.pola.rs/user-guide/getting-started/ "Getting started - Polars user guide"
[3]: https://docs.pola.rs/api/python/dev/reference/lazyframe/api/polars.LazyFrame.collect.html "polars.LazyFrame.collect — Polars  documentation"
[4]: https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.to_arrow.html?utm_source=chatgpt.com "polars.DataFrame.to_arrow — Polars documentation"
[5]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.sql.html "polars.sql — Polars  documentation"
[6]: https://docs.pola.rs/user-guide/lazy/gpu/?utm_source=chatgpt.com "GPU Support - Polars user guide"
[7]: https://docs.pola.rs/py-polars/html/reference/api/polars.thread_pool_size.html?utm_source=chatgpt.com "polars.thread_pool_size — Polars documentation"
[8]: https://docs.pola.rs/py-polars/html/reference/api/polars.Config.set_tbl_rows.html?utm_source=chatgpt.com "polars.Config.set_tbl_rows — Polars documentation"
[9]: https://docs.pola.rs/py-polars/html/reference/io.html?utm_source=chatgpt.com "Input/output — Polars documentation"
[10]: https://docs.pola.rs/py-polars/html/reference/api/polars.read_csv.html?utm_source=chatgpt.com "polars.read_csv — Polars documentation"
[11]: https://docs.pola.rs/user-guide/io/excel/?utm_source=chatgpt.com "Excel - Polars user guide"
[12]: https://docs.pola.rs/user-guide/io/cloud-storage/?utm_source=chatgpt.com "Cloud storage - Polars user guide"
[13]: https://docs.pola.rs/py-polars/html/reference/api/polars.read_delta.html?utm_source=chatgpt.com "polars.read_delta — Polars documentation"
[14]: https://docs.pola.rs/py-polars/html/reference/api/polars.read_ndjson.html?utm_source=chatgpt.com "polars.read_ndjson — Polars documentation"
[15]: https://docs.pola.rs/py-polars/html/reference/selectors.html?utm_source=chatgpt.com "Selectors — Polars documentation"
[16]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.concat_str.html?utm_source=chatgpt.com "polars.concat_str — Polars documentation"
[17]: https://pola.rs/posts/polars-0-19-upgrade-guide/?utm_source=chatgpt.com "Polars 0.19 upgrade guide"
[18]: https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.group_by_dynamic.html?utm_source=chatgpt.com "polars.DataFrame.group_by_dynamic"
[19]: https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.rolling.html?utm_source=chatgpt.com "polars.DataFrame.rolling — Polars documentation"
[20]: https://docs.pola.rs/user-guide/expressions/window-functions/?utm_source=chatgpt.com "Window functions"
[21]: https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.join.html?utm_source=chatgpt.com "polars.DataFrame.join — Polars documentation"
[22]: https://stackoverflow.com/questions/78922047/non-equi-join-in-polars?utm_source=chatgpt.com "Non-equi join in polars - python"
[23]: https://docs.pola.rs/py-polars/html/reference/api/polars.StringCache.html?utm_source=chatgpt.com "polars.StringCache — Polars documentation"
[24]: https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.unpivot.html?utm_source=chatgpt.com "polars.DataFrame.unpivot — Polars documentation"
[25]: https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Datetime.html?utm_source=chatgpt.com "polars.datatypes.Datetime — Polars documentation"
[26]: https://docs.pola.rs/py-polars/html/reference/api/polars.datatypes.Decimal.html?utm_source=chatgpt.com "polars.datatypes.Decimal — Polars documentation"
[27]: https://docs.pola.rs/user-guide/expressions/categorical-data-and-enums/?utm_source=chatgpt.com "Categorical data and enums"
[28]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.map_elements.html?utm_source=chatgpt.com "polars.Expr.map_elements — Polars documentation"
[29]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.map_batches.html?utm_source=chatgpt.com "polars.Expr.map_batches — Polars documentation"
[30]: https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.map_rows.html?utm_source=chatgpt.com "polars.DataFrame.map_rows — Polars documentation"
[31]: https://docs.pola.rs/user-guide/expressions/user-defined-python-functions/?utm_source=chatgpt.com "User-defined Python functions"
[32]: https://pythonspeed.com/articles/polars-pandas-interopability/?utm_source=chatgpt.com "Using Polars in a Pandas world"
[33]: https://duckdb.org/docs/stable/guides/python/polars.html?utm_source=chatgpt.com "Integration with Polars"
[34]: https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.join_asof.html?utm_source=chatgpt.com "polars.DataFrame.join_asof — Polars documentation"
[35]: https://docs.pola.rs/user-guide/concepts/streaming/?utm_source=chatgpt.com "Streaming - Polars user guide"
[36]: https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.select.html?utm_source=chatgpt.com "polars.DataFrame.select — Polars documentation"
[37]: https://docs.pola.rs/py-polars/html/reference/dataframe/group_by.html?utm_source=chatgpt.com "GroupBy — Polars documentation"
[38]: https://docs.pola.rs/py-polars/html/reference/?utm_source=chatgpt.com "Python API reference — Polars documentation"
[39]: https://docs.pola.rs/user-guide/gpu-support/?utm_source=chatgpt.com "GPU Support [Open Beta]"
