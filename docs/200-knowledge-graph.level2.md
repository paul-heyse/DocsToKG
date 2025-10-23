# DocsToKG • KnowledgeGraph (Level-2 Spec)

## Purpose & Non-Goals
**Purpose:** Align chunks to ontology concepts, create provenance-rich nodes/edges, validate with SHACL, and persist to Neo4j with **atomic alias promotion**.  
**Non-Goals:** Retrieval or answer generation.

## Model (Neo4j) — v1
Nodes: Document, Chunk, Concept, Mention, Triple, OntologyVersion, Provenance.  
Rels: HAS_CHUNK, MENTIONS{score}, SUBJECT/PREDICATE/OBJECT, EVIDENCE_FOR, FROM_ONTOLOGY_VERSION, WAS_GENERATED_BY.

### Indexes & Constraints (Cypher)
```cypher
CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;
CREATE CONSTRAINT chunk_uuid_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.uuid IS UNIQUE;
CREATE CONSTRAINT concept_curie_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.curie IS UNIQUE;
CREATE INDEX concept_label IF NOT EXISTS FOR (c:Concept) ON (c.label);
```

## Alignment Pipeline
- Candidate gen: dictionary + lexical (BM25/SPLADE) + dense (optional).
- Fusion: per-channel z-score → **RRF** (`k=60`); tie-break via synonym exactness.
- Disambiguation: section/page priors; stoplist; context window.
- Emission: `Mention` + `MENTIONS{score}`; optional triples with evidence threshold.

## SHACL (excerpt)
See `/docs/architecture/shapes/ChunkShape.ttl` and `TripleShape.ttl` for constraints.

## Batch Upsert (internal)
```py
class KGWriter:
    def upsert_concepts(...): ...
    def upsert_mentions(...): ...
    def upsert_triples(...): ...
```
Writes to temp DB, validates, then promotes alias on success.

## Observability
- `kg_alignment_yield`, `kg_shacl_failures_total{shape}`, `kg_tx_retries_total`; logs include failing examples.

## Performance Budgets
- Alignment ≥ 5k chunks/s (dictionary path); writes ≥ 50k nodes/s, 100k rels/s in batch.

## Failure Modes
- Lock mismatch → stop; tx deadlocks → retry; SHACL failures → no alias switch.

## Security
- Separate writer/readers; encrypted backups; credential rotation on promotion.

## Tests
- Candidate gen & fusion math, SHACL shapes, E2E upsert & alias switch.
