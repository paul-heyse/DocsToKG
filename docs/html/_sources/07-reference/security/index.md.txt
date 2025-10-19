# 1. Security Considerations

DocsToKG pipelines touch external APIs, locally cached artifacts, and hybrid search services that may expose sensitive metadata. This page outlines safeguards aligned with the current codebase.

## 2. Data Handling

- **Storage segregation** – keep `$DOCSTOKG_DATA_ROOT` partitioned per environment (for example `/srv/docstokg/{staging,prod}`) and encrypt those volumes with provider-managed or customer-managed keys.
- **Manifest hygiene** – JSONL manifests written by `DocsToKG.ContentDownload.telemetry` include hashes and file paths. Do not expose these raw logs externally; export redacted views with `tools/manifest_to_csv.py`.
- **Chunk and embedding scoping** – the chunking pipeline stores doc IDs and references. Use namespace prefixes (`DocumentInput.namespace`) to prevent cross-tenant FAISS ingestion.
- **Telemetry minimisation** – configure sinks (`RotatingJsonlSink`, `SqliteSink`) to log only operational fields; avoid injecting PII via custom metadata or resolver overrides.

## 3. Access Control

- **Hybrid Search API** – `DocsToKG.HybridSearch.service.HybridSearchAPI` is unauthenticated by default. Front it with an API gateway (JWT, OAuth2) or embed within a service implementing auth middleware.
- **Secrets** – manage `PA_ALEX_KEY`, ontology credentials, and storage keys in a secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault). The CLI modules read environment variables at runtime; never commit `.env` files.
- **Artifact writes** – guard FAISS snapshot locations and ontology storage buckets with write-only roles for CI/CD principals. Audit rotations via bucket policies or object-locking rules.

## 4. Supply Chain Controls

- Dependencies are pinned in `requirements.txt` and `pyproject.toml`; the bootstrap script installs bundled wheels (`ci/wheels/torch…whl`, `faiss…whl`, `vllm…whl`). Verify these artifacts with `sha256sum` before mirroring to private registries.
- Enable `pip install --require-hashes -r requirements.txt` in deployment CI so supply chain tampering fails closed.
- Run `direnv exec . python docs/scripts/validate_code_annotations.py` and `validate_docs.py` in CI to surface undocumented or stale modules before release.
- Git LFS stores large wheels; enforce `git lfs install` and `git lfs pull` in build pipelines to avoid truncated dependencies.

## 5. Operational Hardening

- **Principle of least privilege** – run ingestion (`DocsToKG.ContentDownload.cli`) and ontology pipelines with service identities that only access required buckets, queues, or credential scopes.
- **Network hygiene** – outbound resolver requests use `requests` + `tenacity` retry logic. Configure egress rules (proxy, firewall) to restrict unfamiliar domains; maintain allowlists in resolver config.
- **TLS everywhere** – terminate TLS at the gateway and use mTLS or service mesh policies for internal calls. Validate upstream certificates when custom resolvers are added.
- **Logging & monitoring** – ship manifest and telemetry logs to central observability. Leverage `DocsToKG.HybridSearch.observability.Observability` hooks to forward metrics. Set alerts for unusual query rates or ontology validation failures.

## 6. Incident Response & Recovery

- **Snapshots** – create and version FAISS snapshots (`serialize_state`) and ontology manifests after each deployment. Store them in immutable buckets for 30+ days.
- **Playbooks** – rehearse the rollback steps documented in `docs/hybrid_search_runbook.md`, including restoring snapshots and revoking leaked credentials.
- **Forensics** – retain raw download manifests for the duration required by policy, but rotate access tokens immediately after suspicious resolver activity.
- **Contact chain** – keep the escalation roster in `docs/06-operations/index.md` current so responders can coordinate cross-team fixes.
