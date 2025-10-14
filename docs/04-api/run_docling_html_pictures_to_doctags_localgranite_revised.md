# Module: run_docling_html_pictures_to_doctags_localgranite_revised

HTML → DocTags with robust image handling + captioning & classification (Docling)
and Hugging Face authentication (env or token file).

- Input :  Data/HTML/        (recurses; excludes *.normalized.html)
- Output:  Data/DocTagsFiles/*.doctags
- Images:  Data/ImagesCache/ (downloaded/decoded assets)

Captioning:
  --caption-model granite|smolvlm|api
  --hf-cache /home/paul/hf-cache   --offline
  --api-url/--api-model/--api-headers for remote OpenAI-style endpoints

Classification:
  --classify

Accelerator:
  --device auto|cpu|cuda|mps  --threads N

## Functions

### `detect_data_root(start)`

*No documentation available.*

### `ensure_hf_auth(token_file)`

Load HF token from env or token file and:
  - set common env vars (HUGGING_FACE_HUB_TOKEN, HF_TOKEN, HUGGINGFACEHUB_API_TOKEN)
  - persist via HfFolder.save_token() under HF_HOME

### `make_http_session()`

*No documentation available.*

### `is_ad_or_tracker(url)`

*No documentation available.*

### `sha1_name(payload, ext_hint)`

*No documentation available.*

### `pick_src_from_srcset(srcset_val)`

*No documentation available.*

### `guess_ext(mime)`

*No documentation available.*

### `find_base_url(soup, html_path)`

*No documentation available.*

### `extract_bg_images(soup)`

*No documentation available.*

### `promote_og_images(soup)`

*No documentation available.*

### `list_htmls(root)`

*No documentation available.*

### `resolve_candidate_url(src, base_url)`

*No documentation available.*

### `download_or_decode_image(src, base_url, out_dir, timeout)`

Returns absolute filesystem path (string) to the local copy, or None if skipped.

### `normalize_html_images(html_path, images_dir)`

*No documentation available.*

### `make_accel(device, threads)`

*No documentation available.*

### `make_vlm_options_for_granite(hf_cache, prompt, area)`

*No documentation available.*

### `make_vlm_options_for_smol(prompt, area)`

*No documentation available.*

### `make_api_options(url, model, headers, prompt, area, timeout)`

*No documentation available.*

### `convert_one(html_path, out_dir, caption_model, classify, hf_cache, api_url, api_model, api_headers, timeout, device, threads, caption_prompt, min_pic_area, overwrite)`

*No documentation available.*

### `main()`

*No documentation available.*
