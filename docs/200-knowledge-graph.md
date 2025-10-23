# DocsToKG • KnowledgeGraph — Subsystem Architecture

## Purpose
Align parsed content to ontology concepts and persist a **provenance-rich knowledge graph** (Neo4j) with **SHACL**-gated validity.

## Scope
- **In**: concept mention detection, concept alignment/scoring, triple extraction, graph persistence, SHACL validation, provenance.
- **Out**: PDF/HTML parsing, embedding generation, hybrid vector search serving.

## Data Model (v0)
**Nodes**
- `Document(doc_id, source, title, year, doi, url, hash, created_at)`
- `Chunk(uuid, doc_id, start_offset, end_offset, text_hash, token_len, section, page)`
- `Concept(curie, label, ontology, ontology_version, synonyms[], iri)`
- `Mention(id, uuid, span_start, span_end, surface_form, confidence)`
- `Triple(id, subj_curie, pred_curie, obj_curie, extractor, confidence)`
- `Ontology(id, name)` and `OntologyVersion(id, name, version, url, checksum)`
- `Provenance(id, stage, run_id, config_hash, timestamp)`

**Relationships**
- `(Document)-[:HAS_CHUNK]->(Chunk)`
- `(Chunk)-[:MENTIONS {score}]->(Concept)`
- `(Concept)-[:SAME_AS]->(Concept)` (cross-ontology equivalence, optional)
- `(Triple)-[:SUBJECT]->(Concept)`, `(Triple)-[:PREDICATE]->(Concept)`, `(Triple)-[:OBJECT]->(Concept)`
- `(Chunk)-[:EVIDENCE_FOR]->(Triple)`
- `(Concept)-[:FROM_ONTOLOGY_VERSION]->(OntologyVersion)`
- `(:Provenance)-[:WAS_GENERATED_BY]->(Chunk|Mention|Triple)`

## Alignment Pipeline
1. Dictionary & synonym candidates.
2. Lexical (BM25/SPLADE) + Dense (optional) candidates.
3. Fuse (RRF) and disambiguate with context priors.
4. Threshold; emit `Mention` and `MENTIONS` edges.
5. Optional triple extraction; attach `EVIDENCE_FOR`.

## SHACL Validations (outline)
- Every `Chunk` must carry `doc_id`; every `Mention` span must be within `Chunk.text` bounds.
- `SAME_AS` only across different ontologies; prevent cycles.
- `Triple` requires ≥1 `EVIDENCE_FOR` link.
- Provenance required on all writes.

## Interfaces
- Batch build, upsert APIs, and internal query helpers.

## Observability
- Alignment yield, avg candidates/chunk, p50 alignment latency, validation failure rates.

## Failure Modes & Recovery
- Ontology drift → lockfile mismatch; halt build.
- Graph write partials → batch transactions + retries.
- SHACL failures → mark run invalid; keep artifacts for forensics.
