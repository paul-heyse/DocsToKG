#!/usr/bin/env bash
set -euo pipefail

###############################################
# Config (overridable via environment vars)
###############################################
ROOT_DIR="${ROOT_DIR:-$(pwd)}"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"

# MinIO wheelhouse controls
ENABLE_MINIO_WHEELHOUSE="${ENABLE_MINIO_WHEELHOUSE:-1}"   # set 0 to skip MinIO entirely
WHEELHOUSE="${WHEELHOUSE:-$ROOT_DIR/.wheelhouse}"
WHEELHOUSE_STRICT="${WHEELHOUSE_STRICT:-0}"               # set 1 to install wheels ONLY (no PyPI fallback)

# Where your wheels live in MinIO
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://127.0.0.1:9000}"
MINIO_BUCKET="${MINIO_BUCKET:-docs2kg-wheels}"
MINIO_PREFIX="${MINIO_PREFIX:-cp313/}"

# MinIO auth: choose ONE of these approaches:
# 1) Use an existing mc alias name (e.g., "myminio")
MINIO_ALIAS="${MINIO_ALIAS:-}"   # if set, we'll use this alias
# 2) Or, provide explicit access/secret here (exports in your shell), and we'll make a temp alias
MINIO_ACCESS_KEY_ID="${MINIO_ACCESS_KEY_ID:-}"
MINIO_SECRET_ACCESS_KEY="${MINIO_SECRET_ACCESS_KEY:-}"

# Optionally point to your mc binary (defaults try: mc in PATH, then $HOME/mc)
MC_BIN="${MC_BIN:-}"

# Extras to detect/use when installing the project
PROJECT_EXTRAS_DEFAULT="${PROJECT_EXTRAS_DEFAULT:-gpu12x}"


###############################################
# Python interpreter detection (unchanged)
###############################################
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "[bootstrap] error: could not find a Python interpreter (expected python3.13)" >&2
        exit 1
    fi
    echo "[bootstrap] warning: falling back to $PYTHON_BIN (set PYTHON_BIN to override)" >&2
fi

###############################################
# Resolve mc binary
###############################################
if [[ -z "${MC_BIN}" ]]; then
    if command -v mc >/dev/null 2>&1; then
        MC_BIN="mc"
    elif [[ -x "$HOME/mc" ]]; then
        MC_BIN="$HOME/mc"
    else
        MC_BIN=""  # not available; we'll skip MinIO if ENABLE_MINIO_WHEELHOUSE=1 but no mc present
    fi
fi

###############################################
# Pull wheels from MinIO into local wheelhouse
###############################################
DID_SYNC=0
if [[ "$ENABLE_MINIO_WHEELHOUSE" == "1" ]]; then
    if [[ -z "$MC_BIN" ]]; then
        echo "[bootstrap] note: 'mc' (MinIO client) not found; skipping wheel sync"
    else
        echo "[bootstrap] Preparing wheelhouse at $WHEELHOUSE"
        mkdir -p "$WHEELHOUSE"

        # Choose alias to use/create
        ALIAS_TO_USE="$MINIO_ALIAS"
        if [[ -z "$ALIAS_TO_USE" ]]; then
            if [[ -n "$MINIO_ACCESS_KEY_ID" && -n "$MINIO_SECRET_ACCESS_KEY" ]]; then
                ALIAS_TO_USE="bootstrap_minio"
                # Set/overwrite a private alias for this script run
                "$MC_BIN" alias set "$ALIAS_TO_USE" "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY_ID" "$MINIO_SECRET_ACCESS_KEY" >/dev/null
            elif "$MC_BIN" alias list 2>/dev/null | grep -q '^myminio'; then
                ALIAS_TO_USE="myminio"
            fi
        fi

        if [[ -n "$ALIAS_TO_USE" ]]; then
            echo "[bootstrap] Syncing wheels from $ALIAS_TO_USE/$MINIO_BUCKET/$MINIO_PREFIX -> $WHEELHOUSE"
            # Copy wheels (and SHA256SUMS if present)
            "$MC_BIN" cp --recursive "$ALIAS_TO_USE/$MINIO_BUCKET/$MINIO_PREFIX" "$WHEELHOUSE/" || {
                echo "[bootstrap] warning: failed to pull from MinIO; continuing without wheelhouse" >&2
            }

            if [[ -f "$WHEELHOUSE/SHA256SUMS" ]]; then
                echo "[bootstrap] Verifying wheel checksums"
                if ! (cd "$WHEELHOUSE" && sha256sum -c SHA256SUMS); then
                    echo "[bootstrap] warning: checksum file invalid or mismatched; continuing without verification" >&2
                fi
            fi

            DID_SYNC=1
        else
            echo "[bootstrap] note: no MinIO alias/creds available; skipping wheel sync"
        fi
    fi
fi

###############################################
# Create / reuse virtual environment (unchanged)
###############################################
if [[ ! -d "$VENV_PATH" ]]; then
    echo "[bootstrap] Creating virtual environment at $VENV_PATH"
    "$PYTHON_BIN" -m venv "$VENV_PATH"
else
    echo "[bootstrap] Reusing existing virtual environment at $VENV_PATH"
fi

echo "[bootstrap] Upgrading pip inside the virtual environment"
"$VENV_PATH/bin/pip" install --upgrade pip

###############################################
# Build pip args to prefer (or require) local wheelhouse
###############################################
PIP_ARGS=()
if [[ -d "$WHEELHOUSE" && $(find "$WHEELHOUSE" -maxdepth 1 -name '*.whl' | wc -l) -gt 0 ]]; then
    echo "[bootstrap] Using wheelhouse at $WHEELHOUSE"
    PIP_ARGS+=( --find-links "$WHEELHOUSE" )
    if [[ "$WHEELHOUSE_STRICT" == "1" ]]; then
        PIP_ARGS+=( --no-index )
        echo "[bootstrap] Wheelhouse STRICT mode: PyPI disabled"
    fi
elif [[ "$ENABLE_MINIO_WHEELHOUSE" == "1" && "$DID_SYNC" == "1" ]]; then
    echo "[bootstrap] warning: no wheels found in $WHEELHOUSE (check MinIO path/policy)"
fi

###############################################
# Install requirements (prefer wheelhouse if present)
###############################################
if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
    echo "[bootstrap] Installing project requirements"
    "$VENV_PATH/bin/pip" install --no-cache-dir "${PIP_ARGS[@]}" -r "$ROOT_DIR/requirements.txt"
else
    echo "[bootstrap] warning: requirements.txt not found; skipping dependency install" >&2
fi

###############################################
# HARD FAIL GUARD: CuPy must not be a namespace package
###############################################
"$VENV_PATH/bin/python" - <<'PY'
import importlib.util, os, sys, subprocess
def is_broken():
    spec = importlib.util.find_spec("cupy")
    return not (spec and spec.origin and spec.origin.endswith("__init__.py"))

if is_broken():
    print("[bootstrap] ERROR: 'cupy' looks like a namespace (missing __init__.py). Attempting repair...", flush=True)
    py = sys.executable
    # Uninstall any CuPy distributions that might be half-installed
    subprocess.run([py, "-m", "pip", "uninstall", "-y", "cupy", "cupy-cuda12x"], check=False)
    # Reinstall strictly from the wheelhouse
    wh = os.environ.get("WHEELHOUSE", ".wheelhouse")
    subprocess.check_call([py, "-m", "pip", "install", "--no-cache-dir", "--no-index",
                           f"--find-links={wh}", "cupy-cuda12x==14.0.0a1"])
    if is_broken():
        raise SystemExit("[bootstrap] FATAL: CuPy still broken after reinstall. Replace the wheel in your wheelhouse.")
print("[bootstrap] CuPy OK.")
PY

###############################################
# Editable install (skip if requirements already did -e .[…])
###############################################
if [[ -f "$ROOT_DIR/requirements.txt" ]] && grep -Eq '^-e[[:space:]]+\.\[' "$ROOT_DIR/requirements.txt"; then
    echo "[bootstrap] Editable install done via requirements.txt (detected -e .[…]); skipping duplicate"
else
    echo "[bootstrap] Installing DocsToKG in editable mode"
    "$VENV_PATH/bin/pip" install "${PIP_ARGS[@]}" -e .
fi

###############################################
# Optional docs hint (unchanged)
###############################################
if [[ -f "docs/build/sphinx/requirements.txt" ]]; then
    echo "[bootstrap] (Optional) Install documentation dependencies with:"
    echo "  $VENV_PATH/bin/pip install -r docs/build/sphinx/requirements.txt"
fi

###############################################
# Legacy LFS pointer check (kept, harmless)
###############################################
if command -v head >/dev/null 2>&1; then
    for wheel in "$ROOT_DIR"/ci/wheels/*.whl; do
        [[ -f "$wheel" ]] || continue
        if head -c 32 "$wheel" | grep -q "version https://git-lfs"; then
            echo "[bootstrap] warning: $wheel is a Git LFS pointer; run 'git lfs install && git lfs pull'" >&2
        fi
    done
fi

echo
echo "[bootstrap] Done. Activate the virtualenv with:"
echo "  source $VENV_PATH/bin/activate"
