Below is a practical, end‑to‑end guide to the **Writer** Python SDK—the official client for the Writer (writer.com) API—written for an AI programming agent that may be refactoring from custom HTTP wrappers. It focuses on what you’ll actually call from code, how to compose those calls into robust agent loops, and the gotchas that matter in production.

> **What this covers.** The SDK you install with `pip install writer-sdk` and import as `writerai`. It exposes chat completions, structured outputs, streaming, tool calling (custom + prebuilt tools), Knowledge Graph (graph‑RAG), file management, no‑code agents (“applications”), and more. ([PyPI][1])

> **Note on naming.** Despite the name “writer,” the Python import path is `writerai` and the PyPI package is `writer-sdk` (latest: **2.3.2**, released Oct 3, 2025 at the time of writing). ([PyPI][1])

---

## 1) Install & authenticate

```bash
pip install writer-sdk
export WRITER_API_KEY="your_api_key"  # recommended via env var / .env
```

```python
from writerai import Writer, AsyncWriter

client = Writer()                 # infers WRITER_API_KEY from env
# or: client = Writer(api_key="...")  # avoid hard-coding in real apps
```

* Requires Python ≥ 3.8. The SDK auto-detects your key from `WRITER_API_KEY`. ([PyPI][1])

---

## 2) Models you’ll use

Most agent use-cases use **chat** models:

* General: `palmyra-x5` (latest flagships) or `palmyra-x4`.
* Domain: `palmyra-fin`, `palmyra-med`, `palmyra-creative`.
* Legacy: `palmyra-x-003-instruct` (still supported in API).
  You specify these in `model="..."` when calling chat. ([Writer Developer Portal][2])

---

## 3) Chat completions (sync + streaming)

**Synchronous, single turn**

```python
resp = client.chat.chat(
    model="palmyra-x5",
    messages=[{"role": "user", "content": "Give me a one-line summary of Rust vs Go"}],
)
text = resp.choices[0].message.content
```

**Low-latency streaming (two styles)**
A. **Simple streaming**: iterate chunks and print deltas

```python
stream = client.chat.chat(
    model="palmyra-x5",
    messages=[{"role": "user", "content": "Stream me a haiku about compilers"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

B. **Streaming helper**: event‑based with `client.chat.stream(...)`

```python
with client.chat.stream(
    model="palmyra-x5",
    messages=[{"role": "user", "content": "Explain backprop step-by-step"}],
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)
    final = stream.get_final_completion()
```

Chat + streaming behavior, return shapes (`choices[0].delta.content`) and helper usage are documented; the helper provides a higher‑level event stream on top of SSE. ([Writer Developer Portal][3])

---

## 4) Structured outputs (JSON Schema)

You can require well‑formed JSON that conforms to a schema via `response_format`.

```python
schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "ProductSpec",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "features": {"type": "array", "items": {"type": "string"}},
                "price_usd": {"type": "number"},
            },
            "required": ["name", "features"]
        }
    }
}

resp = client.chat.chat(
    model="palmyra-x5",
    messages=[{"role": "user", "content": "Return JSON spec for a hiking backpack"}],
    response_format=schema,
)
payload = resp.choices[0].message.content  # JSON string (or a parsed field if exposed)
```

Writer’s chat API supports `response_format` with JSON Schema, which is enforced by Palmyra X4/X5 to reduce post‑hoc parsing failures. Validate with Pydantic if you need hard guarantees. ([Writer Developer Portal][4])

---

## 5) Tool calling (function calls & prebuilt tools)

### 5.1 Custom function tools

Define JSON‑schema’d functions and let the model call them when needed:

```python
tools = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get weather by city",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"]
    }
  }
}]

resp = client.chat.chat(
    model="palmyra-x5",
    messages=[{"role": "user", "content": "Is it raining in Seattle?"}],
    tools=tools,
    tool_choice="auto",  # 'auto'|'none'|'required'|force specific tool via JSON
)
msg = resp.choices[0].message
if msg.tool_calls:
    for call in msg.tool_calls:
        if call.function.name == "get_weather":
            tool_result = {"temp_c": 12, "raining": True}   # <-- your function output
            # Send a follow-up turn with the tool result for the model to use:
            follow = client.chat.chat(
                model="palmyra-x5",
                messages=[
                  {"role": "user", "content": "Is it raining in Seattle?"},
                  {"role": "assistant", "tool_calls": msg.tool_calls},  # echo tool call
                  {"role": "tool", "tool_call_id": call.id, "content": tool_result}
                ],
            )
            print(follow.choices[0].message.content)
```

* `tools` accept JSON‑Schema definitions; `tool_choice` controls whether/which tool is called (`auto`, `none`, `required`, or a JSON selector). The response includes `message.tool_calls` when a function should be invoked. ([Writer Developer Portal][2])

### 5.2 Prebuilt, server‑side tools

Only **one** prebuilt tool can be present in `tools` per request (you may include multiple custom `function` tools). The prebuilt set includes:

* **Knowledge Graph** (`type: "graph"`): give graph IDs + optional `query_config` and `subqueries` to add grounded, source‑cited RAG to a chat; extra data returns in `graph_data`. ([Writer Developer Portal][5])
* **Model Delegation (LLM tool)** (`type: "llm"`): let Palmyra delegate a subtask to a specialty model like `palmyra-fin` or `palmyra-med`; extra data returns in `llm_data`. ([Writer Developer Portal][6])
* **Translation** (`type: "translation"`): call `palmyra-translate` with options (formality, length control, profanity masking); metadata in `translation_data`. ([Writer Developer Portal][7])
* **Vision** (`type: "vision"`): analyze images from uploaded file IDs inside chat. ([Writer Developer Portal][8])
* **Web search** (`type: "web_search"`): fetch fresh sources inline; results arrive as `web_search_data` (query, sources, optional raw content). The tool call itself is not streamed; once complete, normal chat streaming continues. ([Writer Developer Portal][9])

The “only one prebuilt tool at a time” rule and the overall `tools` schema are specified in the Chat Completions API. ([Writer Developer Portal][2])

---

## 6) Knowledge Graph (graph‑RAG)

**Create + manage a graph**

```python
# Upload files first
from pathlib import Path

file = client.files.upload(
    content=Path("/path/to/ACME-Q4.pdf"),
    content_disposition="attachment; filename='ACME-Q4.pdf'",
    content_type="application/pdf",
)

# Create a graph
graph = client.graphs.create(name="ACME Reports", description="2024 filings")

# Attach a file to a graph (endpoint: POST /v1/graphs/{graph_id}/file)
client.graphs.file.add(graph_id=graph.id, file_id=file.id)  # method name varies by SDK; API supports it
```

* Files persist in your account; upload supports many MIME types (txt, pdf, docx, images, etc.). ([Writer Developer Portal][10])
* Graph management endpoints include create/list/update/delete, add/remove files, and even attaching web connectors/URLs. ([Writer Developer Portal][11])

**Ask the graph directly (outside chat)**

```python
answer = client.graphs.question.create(
    graph_ids=[graph.id],
    question="What was Q4 revenue and YoY growth?",
    # Optional: subqueries=True, stream=True, query_config={...}
)
```

The **Question** endpoint returns `answer`, `sources`, and optional `subqueries`, and supports streaming. ([Writer Developer Portal][12])

**Use the graph in a chat (prebuilt tool)**

```python
tools = [{
  "type": "graph",
  "function": {
    "graph_ids": [graph.id],
    "description": "ACME financial reports",
    "subqueries": True,
    "query_config": {"inline_citations": True, "max_snippets": 12, "search_weight": 60}
  }
}]
resp = client.chat.chat(
  model="palmyra-x5",
  messages=[{"role": "user", "content": "Which products include both food coloring and chocolate?"}],
  tools=tools, tool_choice="auto", stream=True
)
```

Knowledge Graph query configuration parameters let you tune retrieval behavior and citation styles. ([Writer Developer Portal][5])

---

## 7) Files (upload, list, delete)

```python
# Upload (Path, bytes, or (filename, bytes, mime))
client.files.upload(
    content=Path("handbook.pdf"),
    content_disposition="attachment; filename='handbook.pdf'",
    content_type="application/pdf",
)

# List (auto‑pagination supported)
for f in client.files.list():
    print(f.id, f.name, f.status)

# Delete
client.files.delete("file_id")
```

Upload/list/get/delete are covered by the Files API; the SDK supports iterating paginated results and allows filtering (e.g., by graph or status). ([Writer Developer Portal][10])

---

## 8) No‑code agents (“applications”) via API

Business users can design agents in AI Studio; developers can **invoke them programmatically** (these endpoints use the term “applications” for compatibility). Two capability types are supported: **text generation** and **research** (research supports streaming and stages). ([Writer Developer Portal][13])

**Invoke an agent**:

```python
out = client.applications.generate(
  application_id="app_123",
  inputs=[{"id": "Product description", "value": ["Terra running shoe"]}],
  # stream=True for research agents
)
```

**Async jobs**: For long‑running research agents, trigger and later retrieve/ retry with jobs API:

```python
job = client.applications.jobs.create(
  application_id="app_123",
  inputs=[{"id": "query", "value": ["Hotels near Union Square under $200"]}],
)
# ... later
job_result = client.applications.jobs.retrieve(job.id)
```

SDK 2.0 introduced application retrieval, async jobs, and model delegation; it also lets you associate graphs with chat applications programmatically. ([WRITER][14])

---

## 9) Vision and other task‑specific endpoints

* **Vision** (`POST /v1/vision`): analyze one or more uploaded images with a prompt (e.g., compare images or OCR). You pass variables of `{name, file_id}` mapped into the prompt like `{{name}}`. Max image size: 7 MB. ([Writer Developer Portal][8])
* **Web search** (standalone endpoint) and **web-search tool** (inside chat): pull fresh sources and optionally raw text into the response. ([Writer Developer Portal][15])
* **Translation** (standalone endpoint) or translation tool inside chat with `palmyra-translate`. ([Writer Developer Portal][7])

These can be combined into agent loops the same way as custom function calls—by adding the appropriate prebuilt tool into `tools` (remember: at most one prebuilt tool per call). ([Writer Developer Portal][2])

---

## 10) Error handling, retries, timeouts, logging

* Errors subclass `writerai.APIError`. Connection issues raise `APIConnectionError`; 4xx/5xx raise `APIStatusError` (e.g., `RateLimitError` for 429). The SDK retries **2 times by default** on connection errors, 408/409/429, and 5xx. ([GitHub][16])
* Default timeout is **3 minutes**; configure globally or per‑request—can pass a float seconds value or an `httpx.Timeout`. On timeout, `APITimeoutError` is raised (and timeouts are retried by default). ([GitHub][16])
* Enable SDK logs by setting `WRITER_LOG=info` (or `debug`). ([GitHub][16])

**Example**

```python
import httpx, writerai

client = Writer(timeout=httpx.Timeout(60, read=5, write=10, connect=2),
                max_retries=2)

try:
    client.chat.chat(model="palmyra-x5",
                     messages=[{"role": "user", "content": "Hello"}])
except writerai.RateLimitError:
    # jittered backoff, queue, etc.
    ...
```

---

## 11) Pagination, raw responses, and low‑level access

* **Auto‑pagination**: list endpoints (e.g., `client.graphs.list()`, `client.files.list()`) are iterable; you can also inspect `.has_next_page()` and `.get_next_page()`. ([GitHub][16])
* **Raw responses**: prepend `.with_raw_response` (non‑streaming) or `.with_streaming_response` (streaming) to inspect headers/bytes directly; useful for debugging or custom telemetry. ([GitHub][16])

---

## 12) Async + performance tuning

* Use **`AsyncWriter`** for concurrent workloads. The async client defaults to **httpx**; for higher concurrency you can install `writer-sdk[aiohttp]` and pass `DefaultAioHttpClient()` to the async client. ([PyPI][1])
* You can replace or tune the underlying **HTTP client** (proxies, custom transports, etc.) via `DefaultHttpxClient(...)`, and even override `base_url` (or set `WRITER_BASE_URL`). ([GitHub][16])

---

## 13) Putting it together: a minimal “tools + RAG” agent loop

```python
from writerai import Writer

client = Writer()

TOOLS = [
  {  # 1) prebuilt Knowledge Graph tool
    "type": "graph",
    "function": {
      "graph_ids": ["<GRAPH_ID>"],
      "description": "ACME product knowledge",
      "subqueries": True,
      "query_config": {"inline_citations": True, "max_snippets": 10}
    }
  },
  {  # 2) custom function
    "type": "function",
    "function": {
      "name": "lookup_sku",
      "description": "Look up SKU info in ERP",
      "parameters": {
        "type": "object",
        "properties": {"sku": {"type": "string"}},
        "required": ["sku"]
      }
    }
  }
]

messages = [{"role": "user", "content": "Compare SKUs 1234 and 5678 for allergens."}]
while True:
    resp = client.chat.chat(model="palmyra-x5", messages=messages,
                            tools=TOOLS, tool_choice="auto")
    msg = resp.choices[0].message

    if not msg.tool_calls:
        print(msg.content)  # final answer
        break

    # Execute tool calls and feed results back
    for call in msg.tool_calls:
        if call.type == "function" and call.function.name == "lookup_sku":
            args = call.function.arguments
            result = erp_lookup(args["sku"])  # your system
            messages.append({"role": "assistant", "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        else:
            # Prebuilt graph tool is executed server-side; just keep the messages
            messages.append({"role": "assistant", "tool_calls": [call]})
```

* The **graph** tool runs server‑side (no client function to call). Its citations/subqueries are returned in `graph_data`. Custom function tools require you to call your function and then add a `tool` role message with the result. One prebuilt tool per call. ([Writer Developer Portal][5])

---

## 14) Useful endpoints beyond chat

* **Text generation** (`/v1/completions`) is a prompt‑in → text‑out endpoint; supports streaming too. ([Writer Developer Portal][3])
* **AI content detection**, **PDF parsing**, **context‑aware text splitting**, **medical entity extraction**, and **search the web** are exposed under the Tools API. These are handy as building blocks in pipelines before/after a chat step. ([Writer Developer Portal][2])

---

## 15) Production checklist (for an agent refactor)

* **Replace custom HTTP** with the SDK’s clients (`Writer` / `AsyncWriter`) to get built‑in retries, timeouts, structured typing, and streaming helpers. ([GitHub][16])
* **Move hard‑coded “RAG”** to **Knowledge Graph** APIs/tools for higher‑accuracy, grounded answers and citations (optionally add URLs to graphs). ([Writer Developer Portal][11])
* **Consolidate function‑calling**: migrate your schema’d functions to the `tools` array and drive a standard tool‑loop (honor `tool_choice`). ([Writer Developer Portal][2])
* **Use prebuilt tools** (web search, translation, vision, LLM‑delegation) instead of maintaining separate services where possible. ([Writer Developer Portal][9])
* **Prefer streaming** for UI latency; adopt `client.chat.stream(...)` when you need event granularity. ([Writer Developer Portal][3])
* **Tune reliability**: set `timeout`, `max_retries`, add backoff on 429, and centralize error handling on `writerai.APIError` subclasses. ([GitHub][16])
* **Go async** for high concurrency; consider `writer-sdk[aiohttp]` and custom HTTP client to add proxies/transports. ([PyPI][1])

---

## 16) Quick API surface map (common namespaces)

* `client.chat.chat(...)` – chat completions (+ tools, streaming, JSON schema). ([Writer Developer Portal][2])
* `client.completions.create(...)` – text generation. ([Writer Developer Portal][3])
* `client.files.upload/list/get/delete(...)` – file lifecycle. ([Writer Developer Portal][10])
* `client.graphs.create/list/retrieve/update/delete(...)` and `.../file` ops – graph‑RAG admin; `client.graphs.question.create(...)` – direct graph Q&A. ([Writer Developer Portal][11])
* `client.applications.generate(...)`, `client.applications.jobs.*`, `client.applications.graphs.*` – no‑code agents, async jobs, associating graphs. ([WRITER][14])

---

## 17) Versioning & diagnostics

* Check runtime version:

  ```python
  import writerai; print(writerai.__version__)
  ```

* Inspect raw responses / headers with `.with_raw_response` and stream bodies with `.with_streaming_response`. Useful for tracing and bespoke telemetry. ([GitHub][16])

---

### References

* Official Python SDK repo + README (install, sync/async usage, streaming helpers, pagination, errors, retries, timeouts, logging, HTTP customization). ([GitHub][16])
* PyPI: package name, version, Python requirement, `aiohttp` extra. ([PyPI][1])
* Developer docs: **Chat Completions** (models list, tools schema, tool choice, response format, streaming overview). ([Writer Developer Portal][2])
* Developer docs: **Structured outputs** (JSON Schema). ([Writer Developer Portal][4])
* Developer docs: **Knowledge Graph** (create/manage graphs, question endpoint, chat with KG). ([Writer Developer Portal][11])
* Developer docs: **Files API**. ([Writer Developer Portal][10])
* Developer docs: **Web search tool**, **Vision**, **Translation tool**. ([Writer Developer Portal][9])
* Engineering blog: **SDK 2.0** features (async jobs, model delegation, apps/graphs). ([WRITER][14])

---

If you meant a *different* Python package named “writer” (e.g., an unrelated framework on PyPI), tell me which one and I’ll tailor the guide to that specific library.

[1]: https://pypi.org/project/writer-sdk/ "writer-sdk · PyPI"
[2]: https://dev.writer.com/api-reference/completion-api/chat-completion "Chat completion - Writer AI Studio"
[3]: https://dev.writer.com/home/streaming "Stream responses from the API - Writer AI Studio"
[4]: https://dev.writer.com/home/models "Models - Writer AI Studio"
[5]: https://dev.writer.com/home/kg-chat "Use a Knowledge Graph in a chat - Writer AI Studio"
[6]: https://dev.writer.com/home/model-delegation "Use another LLM as a tool - Writer AI Studio"
[7]: https://dev.writer.com/home/translation-tool "Translate text in a chat - Writer AI Studio"
[8]: https://dev.writer.com/home/analyze-images "Analyze images - Writer AI Studio"
[9]: https://dev.writer.com/home/web-search-tool "Web search in a chat - Writer AI Studio"
[10]: https://dev.writer.com/home/files "Manage files - Writer AI Studio"
[11]: https://dev.writer.com/home/knowledge-graph "Create and manage a Knowledge Graph - Writer AI Studio"
[12]: https://dev.writer.com/home/kg-query "Ask questions to Knowledge Graphs - Writer AI Studio"
[13]: https://dev.writer.com/home/applications "Invoke no-code agents via the API - Writer AI Studio"
[14]: https://writer.com/engineering/sdk-2-0/ "Writer SDK 2.0 release: async jobs, model delegation, and more - WRITER"
[15]: https://dev.writer.com/home/web-search "Search the web - Writer AI Studio"
[16]: https://github.com/writer/writer-python "GitHub - writer/writer-python: The official Python library for the Writer API"
