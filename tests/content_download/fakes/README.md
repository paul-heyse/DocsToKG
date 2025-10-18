# Content Download Fake Dependencies

The modules under this package provide importable shims for optional
third-party libraries that the content download test suite references. They
mirror the runtime API surface closely enough that the tests can exercise the
code paths without hitting missing-import failures while also giving mypy a
static view of the available attributes.

For shared dependencies (``docling_core``, ``transformers``, ``tqdm``) the
content download suite reuses the fake implementations from
``tests.docparsing.fake_deps``. Only content-download-specific affordances live
here.
