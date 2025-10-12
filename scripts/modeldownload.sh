# cache location (as you’re using)
export HF_HOME=/home/paul/hf-cache

# paste your token into an env var (safer than typing it inline)
export HF_TOKEN="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"

# download into your cache folders
hf download naver/splade-v3 \
  --local-dir "$HF_HOME/naver/splade-v3" \
  --token "$HF_TOKEN"

hf download Qwen/Qwen3-Embedding-4B \
  --local-dir "$HF_HOME/Qwen/Qwen3-Embedding-4B" \
  --token "$HF_TOKEN"
