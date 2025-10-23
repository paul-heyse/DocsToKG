# DocsToKG • End‑to‑End North‑Star & Acceptance Gates

> **Goal:** Ship a domain‑focused, reproducible **Docs → Chunks → Hybrid (sparse+dense) → Ontology‑aligned KG → RAG** system that *agents and humans* can query with **low latency**, **high faithfulness**, and **operational SLOs**. This file is the single source of truth for success measures, test plans, and guardrails. It complements the Level‑2 specs already in the repo (e.g., Observability/SLOs and Security/Data Handling). fileciteturn1file8 fileciteturn1file9

---

## 0) “North‑Star” Outcomes (what “done right” looks like)

**E2E user experience (UX/SRE mix):**

- **Latency:** Query → *fully rendered answer with citations* in **≤ 1.2s p50 / ≤ 3.0s p95** (no text generation), **≤ 2.5s p50 / ≤ 6.0s p95** (with 1,024‑token generation).  
- **Answer quality:** **Faithfulness ≥ 0.85**, **Answer Relevance ≥ 0.80**, **Context Precision ≥ 0.80** on our eval suites (RAGAS‑style metrics). citeturn6search1turn6search0  
- **Retrieval quality:** **nDCG@10 ≥ 0.55** on in‑domain holdouts **and** no regression vs. baseline BM25/SPLADE on BEIR‑style task mixes. citeturn4search10turn3search2  
- **Hybrid retrieval robustness:** Combining dense, BM25 and/or SPLADE with **RRF** improves nDCG@10 **by ≥ +5%** vs. the best single retriever on internal eval; default **MMR** enforces diversity in reranking. citeturn1search0turn1search9turn2search2  
- **Ontology alignment:** **≥ 90%** of chunks that carry mappable semantics get **at least one** ontology concept edge; **SHACL errors = 0** on release builds. citeturn7search0  
- **SLO compliance:** Page‑level alerts only when multi‑window burn‑rate thresholds are violated; <1% of days with “false‑page” per service. citeturn15search0  
- **Safety & supply chain:** SBOM present; images pinned by digest; CI fails on Critical vulns; secrets never appear in logs; data directories 0640; processes non‑root. (See Security L2.) fileciteturn1file9

**Why these are realistic:** MTEB/MMTEB show broad variance across tasks; we hold ourselves to **task‑appropriate** targets and require non‑regression when swapping models. citeturn0search5turn0search2turn0search4

---

## 1) Acceptance Gates by Stage

Each stage must pass **functional**, **quality**, **latency/throughput**, and **operability** gates before merging to `main` for production namespaces.

### 1.1 Content Acquisition (OpenAlex, arXiv; optional others)

**Functional**  
- Must pull works by domain filters (e.g., OpenAlex `topics`, `sources`, `institutions`) and metadata (DOI, authors, abstract, year).  
- arXiv query API used for gaps and PDFs where allowed; respect ToU.

**Quality**  
- **Coverage:** ≥ 95% of in‑scope seed DOIs/IDs present.  
- **Freshness:** Δ(t_ingest − t_source) ≤ 24h for deltas.  
- **Dedup:** < 1% duplicate works after DOI/normalized‑title dedup.  
- **Licenses:** Only ingest works that are open‑licensed or within permitted use.

**Throughput & Limits (hard gates)**  
- **OpenAlex:** **≤ 100k/day** and **≤ 10 rps** overall; for LLM/agent workflows, use **polite pool with email** and keep global rps within limits. citeturn9search0turn9search1turn9search2  
- **arXiv:** **≤ 1 request / 3 seconds** and single connection when batching; client defaults must enforce `delay_seconds ≥ 3`. citeturn12search4turn14search2

**Operability**  
- Emit `content_http_requests_total`, `content_http_latency_ms_bucket`, `content_yield_ratio`, `content_breaker_open_total`. Ship dashboards and alerts tied to 429/5xx and backoff. fileciteturn1file8

**Exit criteria**  
- Dry‑run of a 48‑hour incremental sync completes with zero rate‑limit violations and **yield ≥ 85%** (fetched/attempted). fileciteturn1file8

---

### 1.2 Ontology & Vocab Ingestion

**Functional**  
- Fetch and cache OWL/RDF ontologies; maintain version graph; optional ROBOT “validate‑profile” (EL/DL/RL/QL) on ingest. citeturn7search3

**Quality**  
- **SHACL validation:** **0 violations** against shapes; **≥ 95%** of classes/properties with labels/descriptions indexed for UI/hints. citeturn7search0

**Operability**  
- Checksums on downloads; **HTTPS only**; provenance stored; integrity failures trigger quarantine. (See Security L2.) fileciteturn1file9

---

### 1.3 Document Parsing & Chunking

**Functional**  
- Robust PDF/HTML parsing; language detection; citation & section headers; chunking target **~512–1,024 tokens** with adaptive overlap.

**Quality**  
- **Parse success rate ≥ 98%** on PDFs; **title/abstract fidelity ≥ 99%** vs. source.  
- **Chunk health:** length distribution within spec (p95 ≤ 1,200 tokens); **entity yield** (NER/keyphrases) tracked by domain.  
- **Long‑context guardrail:** order chunks to counter “lost‑in‑the‑middle”; relevant evidence is **front‑loaded** for readers. citeturn5search0turn5search2

**Operability**  
- Metrics: `docparse_doctags_pages_total`, `docparse_embed_rows_total`, `docparse_embed_dim_mismatch_total`, `docparse_stage_latency_ms_bucket{stage}`. fileciteturn1file8

---

### 1.4 Embeddings (Dense)

**Functional**  
- Pluggable encoders (e5/BGE/etc.); enforce dimension compatibility; normalize vectors.

**Quality**  
- **MTEB/MMTEB gates:** On an in‑domain task basket, selected model must be **≥ P50** of public leaderboard for the task family *or* show **≥ +3%** vs. baseline on our internal set. citeturn0search5turn0search2  
- **Non‑regression:** When swapping models, **no drop** in R@10 / nDCG@10 on internal eval.

**Operability**  
- Offline throughput ≥ 8k chunks/s (CPU) or ≥ 40k/s (GPU) per worker (reference hardware); backpressure when queue>threshold.  
- Dim mismatch counter **= 0** in release. fileciteturn1file8

---

### 1.5 Embeddings (Sparse / Lexical)

**Functional**  
- BM25 available by default; **SPLADE** optional for learned sparse vectors and inverted index build. citeturn3search2turn3search4

**Quality**  
- On BEIR or in‑domain analogs, **SPLADE ≥ BM25** on nDCG@10 in ≥70% of datasets (known result trend) and **hybrid dense+sparse ≥ best single**. citeturn4search10

**Operability**  
- Index build completes within budget; memory use recorded; snapshotting and rollbacks tested.

---

### 1.6 Hybrid Retrieval & Fusion

**Functional**  
- **RRF** to combine ranks; **MMR** for diversity in rerank; reciprocal fusion `k=60` (tunable). citeturn1search9turn1search0

**Quality**  
- **Hybrid uplift gate:** **≥ +5% nDCG@10** vs. best single retriever on our eval; **ablation** keeps uplift.  
- **Windowing guardrail:** limit reader context to top‑M chunks with **query‑aware ordering** to avoid mid‑context failures. citeturn5search0

**Latency**  
- **p50 ≤ 150ms / p99 ≤ 600ms** for retrieval only (namespace‑scoped). fileciteturn1file8

**Operability**  
- Metrics: `hybrid_ingest_rows_total{ns}`, `hybrid_query_latency_ms_bucket{ns}`, `hybrid_snapshot_age_seconds{ns}`, `hybrid_gpu_memory_bytes{device}`. fileciteturn1file8  
- GPU indexes use **FAISS StandardGpuResources**/**cuVS IVF‑PQ** with pre‑allocated temp memory for predictable latency. citeturn8search6turn8search0turn8search2

---

### 1.7 KG Construction & Alignment

**Functional**  
- Build entity/statement graph from chunks; align to ontology using dense+sparse signals; write edges with provenance.  
- **Validation:** **pySHACL** + optional ROBOT profile checks on every write batch. citeturn7search4turn7search3

**Quality**  
- **Alignment yield:** ≥ 70% of eligible chunks produce ≥1 concept link; **precision ≥ 0.90** on sampled audit; **SHACL failures = 0**.  
- Metrics: `kg_alignment_yield{ns}`, `kg_shacl_failures_total{shape,ns}`, `kg_tx_retries_total{ns}`. fileciteturn1file8

---

### 1.8 RAG Service

**Functional**  
- Retrieval‑only and Retrieval+Generation endpoints; citations with passage offsets; streaming supported.

**Quality & Eval**  
- **RAGAS**: Faithfulness, Answer Relevance, Context Precision tracked; **faithfulness ≥ 0.85** for gated domains; dataset‑backed regression tests. citeturn6search1

**Latency**  
- Retrieval+scoring **p50 ≤ 300ms** (no gen). Generation budgets configurable.

**Operability**  
- Metrics: `rag_latency_ms_bucket{path}`, `rag_degraded_total{reason}`. fileciteturn1file8

---

### 1.9 Agent Gateway & Budgets

**Functional**  
- API keys + RBAC; per‑tenant budgets (RPM, TPM, GPU‑seconds) with policy denials surfaced.  
- **Backoff & retries** honored on upstream 429/5xx; idempotent write patterns.

**Operability**  
- Metric: `gateway_budget_denials_total{reason}`; SLO: error_rate < 1%. fileciteturn1file8

**Security**  
- HTTPS everywhere; keys stored as secrets; request/response redaction for PII; follow Security L2. fileciteturn1file9

---

## 2) System‑wide SLOs, Alerting, and Runbooks

**Golden signals by subsystem** are defined in *Observability & SLOs (Level‑2)*. Adopt **multi‑window, multi‑burn‑rate** SLO alerts as recommended in the SRE Workbook: **page** at 2%/1h (14.4x) and 5%/6h (6x); **ticket** at 10%/3d (1x). PromQL examples are provided in the companion file. fileciteturn1file8 citeturn15search0

**Dashboards**: per‑namespace service dashboards + global E2E. **Tracing**: OpenTelemetry spans carry `run_id`, `config_hash`, `namespace`. fileciteturn1file8

---

## 3) Model‑Swap & Fine‑Tuning Policy

- All encoder swaps must pass: (a) **internal retrieval non‑regression** (R@k, nDCG@k), (b) **MTEB/MMTEB parity** for task family, and (c) **no latency regression** >10%. citeturn0search5turn0search4  
- If fine‑tuned on domain corpus: must beat base model on **in‑domain BEIR‑like tasks** and sustain **RAGAS faithfulness** with no drop. citeturn4search10turn6search1

---

## 4) Packaging, Security & Compliance

- Reproducible builds, **SBOM (Syft)**, scan (Grype/Trivy), **pinned digests**, non‑root containers, 0640 data perms, encrypted backups, checksum‑verified ontologies. (See Security L2 for the complete checklist.) fileciteturn1file9

---

## 5) E2E Acceptance Scenarios (must pass before “production‑ready” label)

1) **Cold‑start ingest** of a small domain (e.g., `oncology‑trial‑design`): no rate‑limit violations; yield ≥ 85%; SHACL = 0; hybrid uplift ≥ +5%; end‑to‑end query returns an answer with ≥ 0.85 faithfulness on the eval set. fileciteturn1file8 citeturn6search1  
2) **Daily delta ingest**: freshness ≤ 24h; OpenAlex polite‑pool honored; arXiv delay≥3s; no duplicate works introduced. citeturn9search2turn9search0 citeturn12search4  
3) **Model swap (dense)**: passes policy in §3 with no regression; hybrid uplift retained. citeturn0search5  
4) **Load test**: 50 concurrent queries, p95 within targets; GPU memory stable (no eviction thrash) with FAISS/cuVS configs. citeturn8search6turn8search2  
5) **Fault drill**: content API returns 429—client backs off, alerts fire at correct burn rates; no paging for flapping conditions. citeturn15search0  
6) **Security drill**: secret rotation & redact logs verified; containers non‑root; SBOM attached to release. fileciteturn1file9

---

## 6) Appendix A — Source‑Side Limits & Etiquette (for agents)

- **OpenAlex:** 100k/day; **10 rps**; use email/polite pool; batch lookups with OR to reduce calls. citeturn9search0turn9search2  
- **arXiv:** **1 req / 3s**; single connection; the canonical Python client defaults to `delay_seconds=3`. citeturn12search4turn14search2  
- **Crossref & others:** honor documented 429 semantics and `Retry‑After`. (General practice.)

---

## 7) Appendix B — References

- **Fusion & Diversification:** RRF (SIGIR’09); MMR (TIPSTER’98). citeturn1search0turn2search2  
- **Sparse retrieval:** SPLADE v2. citeturn3search2  
- **Benchmarks:** BEIR; MTEB/MMTEB. citeturn4search10turn0search2turn0search5  
- **Long‑context effects:** “Lost in the Middle,” TACL 2024. citeturn5search0  
- **KG validation:** SHACL (W3C Rec); pySHACL; ROBOT validate‑profile. citeturn7search0turn7search4turn7search3  
- **ANN infra:** FAISS StandardGpuResources; RAPIDS/cuVS IVF‑PQ. citeturn8search6turn8search0turn8search2  
- **SRE:** Multi‑window, multi‑burn‑rate alerting. citeturn15search0

---

**Document status:** Living design. Owners: @you + collaborators. Update alongside code changes that affect behavior, metrics, or targets.

