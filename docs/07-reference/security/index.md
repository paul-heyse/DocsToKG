# Security Considerations

DocsToKG processes potentially sensitive documents. Follow these guidelines to keep deployments secure.

## Data Handling

- Encrypt document storage (at rest) with provider-managed keys or customer-managed keys (CMK).
- Use dedicated namespaces per tenant and avoid cross-tenant embeddings in shared indexes.
- Scrub PII before propagating chunks to shared analytics destinations.

## Access Control

- Restrict Hybrid Search REST endpoints behind an API gateway with JWT or OAuth2 authentication.
- Gate ontology download credentials and API keys via a secrets manager (`AWS Secrets Manager`, `HashiCorp Vault`, etc.).
- Limit write access to FAISS snapshot storage and configuration files.

## Supply Chain

- Pin Python dependencies through `pyproject.toml` and `requirements.in`; review updates quarterly.
- Verify checksums of the FAISS wheel before installation (provided in `docs/07-reference/faiss/resources/`).
- Enable `pip install --require-hashes` for deployment pipelines when possible.

## Operational Hardening

- Run ingestion and search services with least-privilege IAM roles.
- Enable TLS end-to-end; if running locally, use HTTPS proxies for test environments.
- Monitor access logs for unusual query patterns and rate-limit high-volume callers.

## Incident Response

- Maintain weekly FAISS and ontology snapshots for rollback.
- Automate restoration drills (see `docs/hybrid_search_runbook.md`) and keep validation datasets available offline.
- Document contact points for escalation in the operations playbook (`docs/06-operations/index.md`).
