# 1. Module: http

The legacy module ``DocsToKG.ContentDownload.http`` has been consolidated into
``DocsToKG.ContentDownload.network``. All retry helpers now live in the unified
module. Update imports to::

    from DocsToKG.ContentDownload.network import request_with_retries

See :doc:`DocsToKG.ContentDownload.network <DocsToKG.ContentDownload.network>`
for the consolidated API reference.
