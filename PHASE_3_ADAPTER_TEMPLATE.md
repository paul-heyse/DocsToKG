# Phase 3: Source Adapters - Implementation Template

**Status**: READY TO IMPLEMENT
**Files to Create**: 7
**Estimated LOC**: 500-600
**Estimated Time**: 6 hours

## Overview

Each adapter follows the same pattern:

```python
def adapter_XXX(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
    """Adapter for XXX source.

    Args:
        policy: AttemptPolicy with timeout, retries, robots_respect
        context: Dict with work_id, artifact_id, doi, URL, offline flag

    Returns:
        AttemptResult with outcome, url (if success), status, host, reason, meta
    """
    # 1. Extract needed context (DOI, URL, etc.)
    # 2. Call appropriate API/service (cached client for metadata, raw for artifacts)
    # 3. Validate response
    # 4. Construct candidate PDF URL
    # 5. HEAD validate the PDF URL
    # 6. Return AttemptResult(outcome, reason, url, status, host, ...)
```

## Adapter Checklist (for each of 7)

### 1. unpaywall.py
```python
def adapter_unpaywall_pdf(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
    doi = context.get("doi")
    if not doi:
        return AttemptResult("skipped", reason="no_doi", elapsed_ms=0)

    # GET https://api.unpaywall.org/v2/{doi}?email=user@example.com
    # Look for .best_oa_location.url_for_pdf
    # HEAD validate the PDF URL
    # Return success if valid
```

### 2. arxiv.py
```python
def adapter_arxiv_pdf(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
    # Extract arXiv ID from context or DOI
    # Build https://arxiv.org/pdf/{id}.pdf
    # HEAD validate
    # Return success if valid
```

### 3. pmc.py
```python
def adapter_pmc_pdf(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
    # Extract PMCID/PMID from context
    # Query E-utilities API or EPMC API
    # Extract PDF URL
    # HEAD validate
    # Return success if valid
```

### 4. doi_redirect.py
```python
def adapter_doi_redirect_pdf(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
    doi = context.get("doi")
    if not doi:
        return AttemptResult("skipped", reason="no_doi", elapsed_ms=0)

    # GET https://doi.org/{doi} with redirects
    # Follow chain until .pdf or HTML landing page
    # HEAD validate final URL
    # Return success if valid
```

### 5. landing_scrape.py
```python
def adapter_landing_scrape_pdf(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
    landing_url = context.get("landing_url") or context.get("url")
    if not landing_url:
        return AttemptResult("skipped", reason="no_landing_url", elapsed_ms=0)

    # GET landing page (cached client, metadata role)
    # Parse HTML for PDF URLs:
    #   - <meta name="citation_pdf_url" content="...">
    #   - <link rel="alternate" type="application/pdf" href="...">
    #   - <a href="...pdf">Download PDF</a>
    # For each candidate, HEAD validate
    # Return first valid
```

### 6. europe_pmc.py
```python
def adapter_europe_pmc_pdf(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
    # Extract DOI/PMID from context
    # Query Europe PMC API
    # Extract PDF URL
    # HEAD validate
    # Return success if valid
```

### 7. wayback.py
```python
def adapter_wayback_pdf(policy: AttemptPolicy, context: Dict[str, Any]) -> AttemptResult:
    # Use WaybackResolver (existing code)
    # Query CDX API for snapshots
    # Find PDF or landing page
    # HEAD validate
    # Return success if valid
```

## Common Patterns

### Pattern 1: API Metadata → PDF URL
```python
# Call metadata API (cached client)
resp = context["head_client"].get(
    f"https://api.example.com/query?doi={doi}",
    timeout=(5, policy.timeout_ms / 1000),
    extensions={"role": "metadata"}  # Helps with caching
)
if resp.status_code != 200:
    return AttemptResult("retryable" if resp.status_code in (429, 503) else "nonretryable",
                        reason="api_error", elapsed_ms=0, status=resp.status_code, host="api.example.com")

# Extract PDF URL from JSON
pdf_url = resp.json().get("pdf_url")
if not pdf_url:
    return AttemptResult("no_pdf", reason="no_pdf_field", elapsed_ms=0)

# HEAD validate
ok, status, ct, reason = head_pdf(pdf_url, context["raw_client"], timeout_s=policy.timeout_ms/1000)
if ok:
    return AttemptResult("success", reason="api_pdf", elapsed_ms=0, url=pdf_url, status=200, host=extract_host(pdf_url))
else:
    return AttemptResult("nonretryable", reason=reason, elapsed_ms=0, status=status)
```

### Pattern 2: Redirect Following
```python
# Follow redirects
resp = context["raw_client"].get(url, follow_redirects=True, timeout=(5, policy.timeout_ms/1000))
if resp.status_code == 200 and resp.url.endswith(".pdf"):
    return AttemptResult("success", reason="redirect_pdf", elapsed_ms=0, url=str(resp.url), status=200)
else:
    return AttemptResult("nonretryable", reason="no_pdf_redirect", elapsed_ms=0, status=resp.status_code)
```

### Pattern 3: HTML Parsing
```python
from html.parser import HTMLParser

class PDFLinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pdf_urls = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "meta" and attrs_dict.get("name") == "citation_pdf_url":
            self.pdf_urls.append(attrs_dict.get("content"))
        elif tag == "link" and "pdf" in attrs_dict.get("type", ""):
            self.pdf_urls.append(attrs_dict.get("href"))
        elif tag == "a" and ".pdf" in attrs_dict.get("href", ""):
            self.pdf_urls.append(attrs_dict.get("href"))

# Parse HTML and extract URLs
parser = PDFLinkExtractor()
parser.feed(html_content)

# Validate each URL
for pdf_url in parser.pdf_urls:
    ok, status, ct, reason = head_pdf(pdf_url, context["raw_client"], ...)
    if ok:
        return AttemptResult("success", reason="scraped_pdf", url=pdf_url, status=200, ...)
```

## Error Handling

All adapters should handle:
- Missing required context (e.g., no DOI) → "skipped" / "missing_context"
- API errors → "retryable" (429, 503) or "nonretryable" (400, 401, etc.)
- Invalid responses → "nonretryable" / "no_pdf"
- Network errors → "error" / "network_error"
- Timeout → "timeout" (handled by Tenacity integration)

## Telemetry

Each adapter should set:
- `result.meta["source"] = "adapter_name"`
- `result.host = extracted_hostname`
- `result.status = http_status` if applicable

## Testing Template

```python
def test_adapter_unpaywall_success():
    policy = AttemptPolicy("unpaywall_pdf", 6000, 3)
    context = {
        "doi": "10.1234/example",
        "head_client": mock_cached_client,
        "raw_client": mock_raw_client,
    }
    result = adapter_unpaywall_pdf(policy, context)
    assert result.is_success
    assert result.url.endswith(".pdf")

def test_adapter_unpaywall_no_doi():
    policy = AttemptPolicy("unpaywall_pdf", 6000, 3)
    context = {"head_client": mock_cached_client, "raw_client": mock_raw_client}
    result = adapter_unpaywall_pdf(policy, context)
    assert result.outcome == "skipped"
    assert result.reason == "no_doi"
```

## Implementation Order

Recommended order (easy to hard):
1. unpaywall.py (simple API)
2. arxiv.py (simple URL construction)
3. europe_pmc.py (similar to unpaywall)
4. pmc.py (similar to europe_pmc)
5. doi_redirect.py (redirect following)
6. landing_scrape.py (HTML parsing)
7. wayback.py (uses existing WaybackResolver)

## Files to Create

```
src/DocsToKG/ContentDownload/fallback/adapters/
├── __init__.py (shared utilities - DONE)
├── unpaywall.py (80 LOC)
├── arxiv.py (60 LOC)
├── pmc.py (90 LOC)
├── doi_redirect.py (80 LOC)
├── landing_scrape.py (120 LOC)
├── europe_pmc.py (80 LOC)
└── wayback.py (90 LOC)
```

## Next Steps After Phase 3

1. Phase 4: Create config/fallback.yaml (YAML template)
2. Phase 5: Create fallback/loader.py (YAML+env+CLI parsing)
3. Phase 6: CLI commands for operational control
4. Phase 7: Telemetry integration
5. Phase 8: Integration into download.py
6. Phase 9: Comprehensive tests
7. Phase 10: Documentation
