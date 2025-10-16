# 1. Module: retrieval

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.retrieval``.

> **Deprecated:** ``DocsToKG.HybridSearch.retrieval`` is now a thin shim that re-exports
> the public interface from ``DocsToKG.HybridSearch.service``. Import
> :class:`HybridSearchService`, :class:`HybridSearchAPI`, and supporting helpers from the
> service module directly. The shim emits a :class:`DeprecationWarning` on import.

## 1. Re-exported symbols

| Symbol | New home |
| ------ | -------- |
| ``HybridSearchService`` | ``DocsToKG.HybridSearch.service`` |
| ``HybridSearchAPI`` | ``DocsToKG.HybridSearch.service`` |
| ``RequestValidationError`` | ``DocsToKG.HybridSearch.service`` |
| ``ChannelResults`` | ``DocsToKG.HybridSearch.service`` |
| ``PaginationCheckResult`` | ``DocsToKG.HybridSearch.service`` |
| ``verify_pagination`` | ``DocsToKG.HybridSearch.service`` |
| ``build_stats_snapshot`` | ``DocsToKG.HybridSearch.service`` |
| ``should_rebuild_index`` | ``DocsToKG.HybridSearch.service`` |

The service implementation remains unchanged—only the import location has moved. Update
integrations to import from ``DocsToKG.HybridSearch.service`` to avoid future breakage.

Examples:
>>> service = HybridSearchService(
...     config_manager=config_manager,
...     feature_generator=feature_generator,
...     faiss_index=faiss_index,
...     opensearch=opensearch,
...     registry=registry
... )
>>> results = service.search(request)
