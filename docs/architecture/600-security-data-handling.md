# DocsToKG • Security, Data Handling & Compliance

## Data Classification
- **Source Artifacts** (PDF/HTML/XML): Restricted internal.
- **Derived Data** (DocTags, chunks, vectors): Restricted internal; never exposed raw.
- **Metadata/Manifests**: Internal; shared between services.
- **Logs/Telemetry**: Internal; may contain URLs and IDs (no secrets).

## Principles
- Least privilege (per-service roles, separate read vs write).
- Deterministic outputs (reproducibility, lockfiles).
- No secret leakage in logs (mask & hash).
- Signed releases and SBOMs for images.

## Secrets
- Stored via environment or file mounts; never in repos.
- Rotatable keys; per-namespace scoping.

## Storage & Transport
- HTTPS for all fetches; checksum enforcement for ontologies.
- At-rest encryption for backups/snapshots; OS disk encryption recommended.

## Access Controls
- Gateway API keys + quotas; RBAC with namespaces.
- Neo4j: dedicated writer role; read replicas for serving.

## Supply Chain
- Build images with pinned digests; generate SBOM (Syft).
- Scan images (Grype/Trivy); fail CI on high/Crit.

## Privacy & Safety
- No PII by default; add DLP gates if later required.
- LLM prompts exclude raw embeddings; strict grounding prompts.

## Incident Response
- Isolate namespace, rotate secrets, audit snapshots, root cause, and backfill if needed.

## Tests
- Secret linting in CI; SBOM presence; containers run as non-root; file perms 0640.
