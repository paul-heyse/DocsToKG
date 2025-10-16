# 1. Module: network

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.network``.

## 1. Overview

Deprecated shim for the legacy ``DocsToKG.ContentDownload.network`` module.

The networking utilities were consolidated into :mod:`DocsToKG.ContentDownload.networking`.
Re-export the public surface so older imports keep working while the rest of the
codebase consistently uses the new module.
