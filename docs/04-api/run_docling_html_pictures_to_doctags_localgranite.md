# Module: run_docling_html_pictures_to_doctags_localgranite

HTML → DocTags with robust image handling + captioning & classification (Docling).

- Input :  Data/HTML/        (recurses)
- Output:  Data/DocTagsFiles/*.doctags
- Images:  Data/ImagesCache/ (downloaded/decoded assets for stable local references)

Captioning options:
  --caption-model granite     # local Granite Vision 3.3-2b via Transformers (no vLLM)
  --granite-path /path/to/local/model/dir
  --hf-cache /home/paul/hf-cache  --offline

  --caption-model smolvlm     # local tiny captioner

  --caption-model api         # remote OpenAI-style endpoint (e.g., your vLLM)
  --api-url http://localhost:8000/v1/chat/completions --api-model ibm-granite/granite-vision-3.3-2b
  --api-headers '{"Authorization":"Bearer ..."}'

Classification:
  --classify                  # enable DocumentFigureClassifier (charts/diagrams/photos, etc.)

Accelerator:
  --device auto|cpu|cuda|mps  # default: auto
  --threads N                 # CPU worker threads for models

Notes
- This uses Docling’s HTML backend + Simple/Convert pipeline; picture enrichment is configured
  via ConvertPipelineOptions (do_picture_description / do_picture_classification +
  PictureDescription{Vlm,Api}Options). :contentReference[oaicite:1]{index=1}
- HTML is normalized to ensure ALL images are visible to the pipeline (lazy-load, srcset,
  CSS backgrounds, OpenGraph figures, base64 data URIs).
- HF offline/caching envs are honored when --offline/--hf-cache are set. :contentReference[oaicite:2]{index=2}

## Functions

### `detect_data_root(start)`

*No documentation available.*

### `sha1_name(payload, ext_hint)`

*No documentation available.*

### `pick_src_from_srcset(srcset_val)`

*No documentation available.*

### `guess_ext(mime)`

*No documentation available.*

### `download_or_decode_image(src, base_url, out_dir, timeout)`

*No documentation available.*

### `find_base_url(soup, html_path)`

*No documentation available.*

### `promote_og_images(soup)`

*No documentation available.*

### `extract_bg_images(soup)`

*No documentation available.*

### `normalize_html_images(html_path, images_dir)`

*No documentation available.*

### `make_accel(device, threads)`

*No documentation available.*

### `make_vlm_options_for_granite(local_path, hf_cache, prompt, area)`

*No documentation available.*

### `make_vlm_options_for_smol(prompt, area)`

*No documentation available.*

### `make_api_options(url, model, headers, prompt, area, timeout)`

*No documentation available.*

### `list_htmls(root)`

*No documentation available.*

### `convert_one(html_path, out_dir, caption_model, classify, granite_path, hf_cache, offline, api_url, api_model, api_headers, timeout, device, threads, caption_prompt, min_pic_area, overwrite)`

*No documentation available.*

### `main()`

*No documentation available.*
