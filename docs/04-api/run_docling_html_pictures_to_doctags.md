# Module: run_docling_html_pictures_to_doctags

HTML → DocTags with robust picture handling + Docling captioning & classification.

- Input :  Data/HTML  (recurses)
- Output:  Data/DocTagsFiles/*.doctags
- Models:  Built-in DocumentFigureClassifier + (choose) Granite Vision, SmolVLM, or a remote vLLM API
- Notes :
    * We normalize HTML to surface every kind of image representation (img/src, data-src, srcset,
      CSS background-image, og:image, data: URIs), download to a local images dir, and rewrite the HTML.
    * For captioning, select one of:
        --caption-model granite|smolvlm|api
      When using --caption-model api, set --api-url and optional --api-headers/--api-model.

## Functions

### `default_data_root(start)`

*No documentation available.*

### `sha1_name(url_or_bytes, ext_hint)`

*No documentation available.*

### `pick_src_from_srcset(srcset_val)`

*No documentation available.*

### `guess_ext_from_mime(m)`

*No documentation available.*

### `download_or_decode_image(src, base_url, out_dir, timeout)`

Returns a local file path (str) to the downloaded/decoded image.
Handles: absolute/relative URLs, data: URIs.

### `find_base_url(soup, html_path)`

*No documentation available.*

### `promote_og_images(soup)`

*No documentation available.*

### `extract_bg_images(soup)`

*No documentation available.*

### `normalize_html_images(html_path, images_dir)`

*No documentation available.*

### `convert_one(html_path, out_dir, caption_model, use_classes, api_url, api_model, api_headers, timeout)`

*No documentation available.*

### `main()`

*No documentation available.*
