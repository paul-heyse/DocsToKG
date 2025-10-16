# 1. Module: http

The legacy module ``DocsToKG.ContentDownload.http`` has been consolidated into
``DocsToKG.ContentDownload.networking``. All retry helpers now live in the unified
module. Update imports to::

    from DocsToKG.ContentDownload.networking import request_with_retries

See :doc:`DocsToKG.ContentDownload.networking <DocsToKG.ContentDownload.networking>`
for the consolidated API reference.
