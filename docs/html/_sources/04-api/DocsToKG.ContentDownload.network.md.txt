# 1. Module: network

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.network``.

## 1. Overview

Compatibility shim exposing networking helpers.

The canonical implementations live in :mod:`DocsToKG.ContentDownload.networking`.
This module re-exports that surface so existing imports continue to function
while downstream code migrates to the new module name.
