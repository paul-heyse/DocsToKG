# 1. DocsToKG Documentation

DocsToKG is a comprehensive system for transforming documents into knowledge graphs using vector search, machine learning, and AI technologies.

## 2. Documentation Structure

This documentation is organized hierarchically to help you quickly find the information you need:

### 📋 [01. Overview](./01-overview/index.md)

Project introduction, goals, and high-level architecture overview.

### ⚙️ [02. Setup](./02-setup/index.md)

Installation guides, configuration, and development environment setup.

### 🏗️ [03. Architecture](./03-architecture/index.md)

System design, component relationships, and data flow diagrams.

### 🔌 [04. API Reference](./04-api/index.md)

Complete API documentation, endpoints, and data models.

### 👥 [05. Development](./05-development/index.md)

Contributing guidelines, coding standards, and development workflows.

### 🚀 [06. Operations](./06-operations/index.md)

Deployment instructions, monitoring, and troubleshooting guides.

### 📚 [07. Reference](./07-reference/index.md)

Glossary, FAISS integration details, external dependencies, and deep dives like the ContentDownload artifact catalog & storage index.

### 📂 Package Guides & Agent Playbooks

Each DocsToKG package ships a README (deep technical overview) and an AGENTS guide (execution guardrails and operational tips for AI assistants). Use the links below to access the canonical references:

#### ContentDownload
- **README** – Covers resolver architecture, CLI usage, manifest formats, rate limiting, storage layout, and operational playbooks. → [`src/DocsToKG/ContentDownload/README.md`](../src/DocsToKG/ContentDownload/README.md)
- **AGENTS** – No-install runbook, guardrails, canonical commands, and performance objectives for agents executing the content acquisition pipeline. → [`src/DocsToKG/ContentDownload/AGENTS.md`](../src/DocsToKG/ContentDownload/AGENTS.md)

#### DocParsing
- **README** – Details DocTags generation, chunking heuristics, embedding runtimes, configuration, schemas, and observability. → [`src/DocsToKG/DocParsing/README.md`](../src/DocsToKG/DocParsing/README.md)
- **AGENTS** – Environment expectations, stage-by-stage command matrix, resume/determinism guidance, and testing checklist for parsing workflows. → [`src/DocsToKG/DocParsing/AGENTS.md`](../src/DocsToKG/DocParsing/AGENTS.md)

#### OntologyDownload
- **README** – Explains resolver planning, download/runtime safety, validator orchestration, configuration schemas, and artifact outputs. → [`src/DocsToKG/OntologyDownload/README.md`](../src/DocsToKG/OntologyDownload/README.md)
- **AGENTS** – Operational guardrails, quick command references, troubleshooting cues, and change-management guidance for ontology fetches. → [`src/DocsToKG/OntologyDownload/AGENTS.md`](../src/DocsToKG/OntologyDownload/AGENTS.md)

#### HybridSearch
- **README** – Documents FAISS GPU setup, ingestion/search flows, configuration surfaces, and observability for the hybrid retrieval service. → [`src/DocsToKG/HybridSearch/README.md`](../src/DocsToKG/HybridSearch/README.md)
- **AGENTS** – Describes no-install expectations, module architecture summary, ingestion/search workflow overview, test matrix, and troubleshooting cues. → [`src/DocsToKG/HybridSearch/AGENTS.md`](../src/DocsToKG/HybridSearch/AGENTS.md)

## 3. Core Pipeline Overview

DocsToKG is composed of four coordinated pipelines that transform raw content into search-ready knowledge assets:

- **Content acquisition (`DocsToKG.ContentDownload`)** – Resolves and downloads PDFs/HTML/XML using provider-specific resolver pipelines with polite rate limits, resumable manifests (JSONL + SQLite), and storage layout guidance. Outputs canonical artifact directories plus manifests enumerating every attempt.
- **Document parsing & enrichment (`DocsToKG.DocParsing`)** – Converts documents into DocTags via Docling and vLLM-hosted Qwen models, chunkifies them with structural + token-aware heuristics, and generates dense (Qwen), sparse (SPLADE), and lexical features. Produces chunk/embedding JSONL files and stage manifests to support deterministic resume behaviour.
- **Ontology acquisition (`DocsToKG.OntologyDownload`)** – Plans, downloads, and validates third-party ontologies (HP, GO, etc.) using hardened networking, checksum verification, and pluggable validators (ROBOT, rdflib, Arelle). Ensures ontology artifacts and lockfiles remain versioned and replayable for downstream ingestion.
- **Hybrid retrieval (`DocsToKG.HybridSearch`)** – Loads DocParsing outputs into GPU-accelerated FAISS indexes, fuses dense and lexical scores (RRF + MMR), manages namespace routing/snapshots, and exposes synchronous search APIs with observability and diagnostics.

Cross-cutting conventions include manifest-driven idempotency (SHA-256 hashes, JSONL + SQLite indexes), environment-driven configuration (`DOCSTOKG_*`), documentation-first development, and GPU requirements for high-throughput DocTags/embedding and FAISS workloads.

## 4. Quick Start

1. **New to DocsToKG?** Start with [Overview](./01-overview/index.md) to understand what we do
2. **Ready to contribute?** Check [Development](./05-development/index.md) for guidelines
3. **Need to deploy?** See [Operations](./06-operations/index.md) for deployment guides
4. **Using the API?** Go directly to [API Reference](./04-api/index.md)

## 5. Documentation Standards

This documentation follows established standards for clarity and consistency:

- All content is written in Markdown for maximum compatibility
- API documentation is auto-generated from code comments using Sphinx
- Regular automated checks ensure documentation quality and link integrity
- All documentation is version-controlled alongside the codebase

## 6. Specifications & Processes

- **Documentation-first development** – Update READMEs/AGENTS before landing behavioural changes; specs live under `openspec/` (e.g., [`openspec/project.md`](../openspec/project.md)) to capture goals, constraints, and architecture notes.
- **Style & annotation standards** – Follow [`docs/STYLE_GUIDE.md`](./STYLE_GUIDE.md) and [`docs/CODE_ANNOTATION_STANDARDS.md`](./CODE_ANNOTATION_STANDARDS.md); NAVMAP headers at the top of modules keep navigation tools in sync.
- **Automated validation** – Use `docs/scripts/validate_docs.py`, `check_links.py`, and `generate_all_docs.py` to keep documentation accurate; CI enforces linting (`ruff`), formatting (`black`, `isort`), typing (`mypy`), and tests (`pytest`).
- **Agent guardrails** – Each package’s AGENTS guide outlines no-install expectations, canonical commands, troubleshooting paths, and performance objectives for AI-driven execution.

## 7. Getting Help

- 📖 **Read the docs** - Most questions are answered here
- 🐛 **Found an issue?** Check existing documentation first, then file an issue
- 💬 **Need clarification?** Open a discussion or reach out to the team

---

*This documentation is automatically maintained and validated as part of our development workflow.*
