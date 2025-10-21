import glob
import json
import sys

import pyarrow.parquet as pq

errors = []

# A) Vectors parquet footers must have provider provenance
for p in glob.glob("Data/Vectors/family=*/fmt=parquet/*/*/*.parquet"):
    meta = pq.ParquetFile(p).metadata.metadata or {}
    m = {k.decode(): v.decode() for k, v in meta.items()}
    for k in [
        "docparse.family",
        "docparse.provider",
        "docparse.model_id",
        "docparse.dtype",
        "docparse.cfg_hash",
        "docparse.created_at",
    ]:
        if k not in m:
            errors.append(("FOOTER_MISSING", p, k))
    if m.get("docparse.family") == "dense" and "docparse.dim" not in m:
        errors.append(("FOOTER_MISSING", p, "docparse.dim"))


# B) Manifests should include provider extras (vectors) & chunks_format (chunks)
def tail(path, n=5000):
    try:
        with open(path) as f:
            return f.readlines()[-n:]
    except FileNotFoundError:
        return []


rows = [json.loads(x) for x in tail("Data/Manifests/docparse.embeddings.manifest.jsonl")]
if rows:
    for r in rows[-100:]:
        if r.get("status") == "success" and r.get("doc_id") not in ("__config__", "__corpus__"):
            for k in ["provider_name", "model_id", "vector_format"]:
                if k not in r:
                    errors.append(("MANIFEST_MISSING", "embeddings", k))
rows = [json.loads(x) for x in tail("Data/Manifests/docparse.chunk.manifest.jsonl")]
if rows:
    for r in rows[-100:]:
        if r.get("status") == "success" and r.get("doc_id") != "__config__":
            for k in ["chunks_format"]:
                if k not in r:
                    errors.append(("MANIFEST_MISSING", "chunk", k))

print("OK" if not errors else json.dumps(errors, indent=2))
sys.exit(1 if errors else 0)
