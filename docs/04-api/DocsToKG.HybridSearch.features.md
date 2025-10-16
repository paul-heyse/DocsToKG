# 1. Module: features

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.features``.

## 1. Overview

Compatibility layer for the retired ``DocsToKG.HybridSearch.features`` module.

This shim is gated behind the ``DOCSTOKG_HYBRID_FEATURES_SHIM`` environment
variable. By default the module raises an ImportError instructing callers to
switch to ``DocsToKG.HybridSearch.devtools.features``. Setting the environment
variable to a truthy value temporarily re-enables the shim (and emits a
DeprecationWarning) so teams can smoke-test outstanding imports.
