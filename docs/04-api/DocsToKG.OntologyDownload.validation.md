# 1. Module: validation

Validation helpers are now implemented in ``DocsToKG.OntologyDownload.validation_core``
and re-exported via ``DocsToKG.OntologyDownload.ontology_download`` for
backwards compatibility. Consult ``validation_core`` for the authoritative
implementations of `ValidationRequest`, `ValidationResult`, streaming
normalization, and validator subprocess orchestration.
