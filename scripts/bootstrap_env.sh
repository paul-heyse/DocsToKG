#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DocsToKG bootstrap (uv-based)
# -----------------------------------------------------------------------------
# Creates (or reuses) .venv using uv and installs project dependencies.
# Optional flag --gpu (or AGENT_MODE=gpu) installs the [project.optional-dependencies].gpu extra.
# -----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_VERSION="3.13"
GPU_MODE="${AGENT_MODE:-cpu}"
INSTALL_GPU=0

if [[ "$GPU_MODE" == "gpu" ]]; then
  INSTALL_GPU=1
fi

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_env.sh [--gpu|--cpu] [--python <version>]

Creates/updates the uv-managed virtual environment at .venv and installs project deps.

Options:
  --gpu              Install GPU extras (same as AGENT_MODE=gpu)
  --cpu              Force CPU-only install (default)
  --python VERSION   Target Python runtime (default: 3.13)
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      INSTALL_GPU=1
      ;;
    --cpu)
      INSTALL_GPU=0
      ;;
    --python)
      shift
      if [[ $# -eq 0 ]]; then
        echo "[bootstrap] error: --python requires an argument" >&2
        exit 1
      fi
      PYTHON_VERSION="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[bootstrap] unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "[bootstrap] uv not found; attempting installation via https://astral.sh/uv/install.sh"
  local installer="https://astral.sh/uv/install.sh"

  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$installer" | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$installer" | sh
  else
    echo "[bootstrap] error: need curl or wget to download uv installer" >&2
    exit 1
  fi

  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    echo "[bootstrap] error: uv installation failed or PATH not updated" >&2
    exit 1
  fi
}

require_gpu_wheel() {
  local wheel_path="$1"
  if [[ ! -f "$wheel_path" ]]; then
    echo "[bootstrap] error: required wheel missing: $wheel_path" >&2
    echo "[bootstrap] tip: populate .wheelhouse/ or rerun without --gpu" >&2
    exit 1
  fi
}

ensure_uv

VENV_PATH="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "[bootstrap] creating uv virtual environment at $VENV_PATH (python $PYTHON_VERSION)"
  uv venv --python "$PYTHON_VERSION" "$VENV_PATH"
else
  echo "[bootstrap] reusing existing environment at $VENV_PATH"
fi

if [[ $INSTALL_GPU -eq 1 ]]; then
  require_gpu_wheel ".wheelhouse/cupy_cuda12x-14.0.0a1-cp313-cp313-manylinux2014_x86_64.whl"
  require_gpu_wheel ".wheelhouse/vllm-0.11.0rc2.dev449+g134f70b3e.d20251014.cu129-cp313-cp313-linux_x86_64.whl"
  require_gpu_wheel ".wheelhouse/faiss-1.12.0-py3-none-any.whl"
fi

INSTALL_SPEC="."
if [[ $INSTALL_GPU -eq 1 ]]; then
  INSTALL_SPEC=".[gpu]"
  echo "[bootstrap] installing project in GPU mode (extras: gpu)"
else
  echo "[bootstrap] installing project in CPU mode"
fi

FIND_LINKS_ARGS=()
if [[ -d "$REPO_ROOT/.wheelhouse" ]]; then
  FIND_LINKS_ARGS+=(--find-links "$REPO_ROOT/.wheelhouse")
fi

uv pip install --python "$VENV_PATH" "${FIND_LINKS_ARGS[@]}" -e "$INSTALL_SPEC"

if command -v direnv >/dev/null 2>&1; then
  direnv allow "$REPO_ROOT" >/dev/null 2>&1 || true
fi

echo "[bootstrap] environment ready."
echo "[bootstrap] activate with: source \"$VENV_PATH/bin/activate\""
