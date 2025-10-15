# 1. Module: pipeline

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.pipeline``.

Resolver pipeline orchestration and execution logic.

## 1. Functions

### `_callable_accepts_argument(func, name)`

*No documentation available.*

### `_respect_rate_limit(self, resolver_name)`

*No documentation available.*

### `_jitter_sleep(self)`

*No documentation available.*

### `_should_attempt_head_check(self, resolver_name)`

*No documentation available.*

### `_head_precheck_url(self, session, url, timeout)`

*No documentation available.*

### `run(self, session, artifact, context)`

Execute resolvers until a PDF is obtained or resolvers are exhausted.

### `_run_sequential(self, session, artifact, context_data, state)`

*No documentation available.*

### `_run_concurrent(self, session, artifact, context_data, state)`

*No documentation available.*

### `_prepare_resolver(self, resolver_name, order_index, artifact, state)`

*No documentation available.*

### `_collect_resolver_results(self, resolver_name, resolver, session, artifact)`

*No documentation available.*

### `_process_result(self, session, artifact, resolver_name, order_index, result, context_data, state)`

*No documentation available.*

### `submit_next(executor, start_index)`

Queue additional resolvers until reaching concurrency limits.

Args:
executor: Thread pool responsible for executing resolver calls.
start_index: Index in ``resolver_order`` where submission should resume.

Returns:
Updated index pointing to the next resolver candidate that has not been submitted.

## 2. Classes

### `_RunState`

Mutable pipeline execution state shared across resolvers.

### `ResolverPipeline`

Executes resolvers in priority order until a PDF download succeeds.

Attributes:
config: Resolver configuration containing ordering and rate limits.
download_func: Callable responsible for downloading resolved URLs.
logger: Structured attempt logger capturing resolver telemetry.
metrics: Metrics collector tracking resolver performance.

Examples:
>>> pipeline = ResolverPipeline([], ResolverConfig(), lambda *args, **kwargs: None, None)  # doctest: +SKIP
>>> isinstance(pipeline.metrics, ResolverMetrics)  # doctest: +SKIP
True
