# DocsToKG • Security, Data Handling & Compliance (Level-2 Spec)

## Data Classes
Artifacts (restricted), Derived (restricted), Metadata (internal), Secrets (confidential).

## Controls
Secrets via env/file mounts; rotation policy; mask secrets in logs. HTTPS everywhere; checksums for ontologies. Encrypted backups; CAS optional. Writer vs reader roles; Gateway API keys with RBAC.

## Supply Chain
Pinned image digests; SBOM (Syft); vulnerability scanning (Grype/Trivy); fail CI on Critical.

## Privacy & DLP
No PII; add DLP gates if needed. Redact snippets in logs; dev-mode reveal via flag.

## Incident Response
Quarantine namespace → rotate creds → snapshot audit → root cause → backfill → post-mortem.

## Tests
Secret-lint, SBOM presence, non-root containers, 0640 perms for data dirs.
