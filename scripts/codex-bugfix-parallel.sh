#!/usr/bin/env bash
# Codex 0.47+ — simple parallel bug-fixer
# - 4 parallel workers (one per folder)
# - iterative passes; find + fix real bugs
# - commits & pushes to main (PR fallback)
# - edits are bounded to each folder
# - logs to codex_runs/<folder>/*.log and <folder>/CODEX_REVIEW_LOG.md

set -euo pipefail

# ===== CONFIG =====
SUBDIRS=(
  "src/DocsToKG/ContentDownload"
  "src/DocsToKG/DocParsing"
  "src/DocsToKG/HybridSearch"
  "src/DocsToKG/OntologyDownload"
)

BATCH_SIZE="${BATCH_SIZE:-40}"          # seed files per Codex run (explore remainder on demand)
MAX_PASSES="${MAX_PASSES:-10}"            # start small; bump after you’re comfy
INCLUDE_UNTRACKED="${INCLUDE_UNTRACKED:-true}"  # include new files
RECENT_COMMITS="${RECENT_COMMITS:-30}"   # prioritise files touched in the last N commits

BASE_BRANCH="${BASE_BRANCH:-main}"
PUSH_TO_MAIN="${PUSH_TO_MAIN:-true}"     # false -> always PR
PR_TOOL="${PR_TOOL:-gh}"                 # needs GitHub CLI for PR fallback

# Safety rails
MAX_CHANGED_FILES="${MAX_CHANGED_FILES:-500}"    # cap to avoid runaway diffs
EXEC_TIMEOUT="${EXEC_TIMEOUT:-1500}"             # retained for backwards compatibility; Codex CLI ignores per-run timeout overrides

# Logging
VERBOSE="${VERBOSE:-true}"
LOG_DIR="${LOG_DIR:-codex_runs}"

# Excludes (use long-form pathspec so Git doesn’t choke)
EXCLUDES=(
  ':(exclude,glob)**/*.png'  ':(exclude,glob)**/*.jpg'  ':(exclude,glob)**/*.jpeg'
  ':(exclude,glob)**/*.gif'  ':(exclude,glob)**/*.svg'  ':(exclude,glob)**/*.pdf'
  ':(exclude,glob)**/*.zip'  ':(exclude,glob)**/*.tar'  ':(exclude,glob)**/*.gz'
  ':(exclude,glob)**/*.bz2'  ':(exclude,glob)**/*.xz'   ':(exclude,glob)**/*.7z'
  ':(exclude,glob)**/*.parquet' ':(exclude,glob)**/*.onnx' ':(exclude,glob)**/*.pt'
  ':(exclude,glob)**/*.bin'  ':(exclude,glob)**/*.so'   ':(exclude,glob)**/*.dylib'
  ':(exclude,glob)**/*.a'    ':(exclude,glob)**/*.whl'  ':(exclude,glob)**/*.ipynb'
  ':(exclude,glob)**/node_modules/**'  ':(exclude,glob)**/dist/**' ':(exclude,glob)**/build/**'
  ':(exclude,glob)**/__pycache__/**'   ':(exclude,glob)**/.venv/**'
  ':(exclude,glob)**/.mypy_cache/**'   ':(exclude,glob)**/.ruff_cache/**'
  ':(exclude,glob)**/wheels/**'
)

# ===== HELPERS =====
need() { command -v "$1" >/dev/null 2>&1 || { echo "✖ Missing: $1"; exit 1; }; }
slugify() { echo "$1" | sed 's#[/ ]#-#g'; }

append_log() {
  local log="$1"; shift
  {
    echo
    echo "<!-- $(date -u +"%Y-%m-%d %H:%M:%SZ") UTC -->"
    printf "%s\n" "$@"
  } >> "$log"
}

git_commit_push_with_retry() {
  local scope="$1" msg="$2" attempt=0
  git fetch origin "$BASE_BRANCH" --quiet || true
  git checkout "$BASE_BRANCH" --quiet
  while (( attempt < 3 )); do
    git pull --rebase --autostash --quiet || true
    git add "$scope" || true
    git add "$scope/CODEX_REVIEW_LOG.md" 2>/dev/null || true
    if git diff --cached --quiet; then return 0; fi
    git commit -m "$msg" || true
    if git push origin "$BASE_BRANCH" 2>/dev/null; then return 0; fi
    attempt=$((attempt+1)); sleep 1
  done
  return 1
}

list_candidates() {
  # prints NUL-separated file list for a subdir (tracked + optional untracked) with excludes
  local subdir="$1"
  local subdir_clean="${subdir%/}"

  local exclude_args=()
  for pattern in "${EXCLUDES[@]}"; do
    if [[ "${pattern:0:1}" == ":" ]]; then
      local trimmed="${pattern#:(exclude,glob)}"
      trimmed="${trimmed#/}"
      exclude_args+=( ":(exclude,glob)${subdir_clean}/${trimmed}" )
    else
      exclude_args+=( "${subdir_clean}/${pattern}" )
    fi
  done

  if [[ "${INCLUDE_UNTRACKED}" == "true" ]]; then
    git ls-files -z -c -o --exclude-standard -- "$subdir_clean" "${exclude_args[@]}"
  else
    git ls-files -z -c -- "$subdir_clean" "${exclude_args[@]}"
  fi
}

# ===== WORKER =====
bugfix_one_subdir() {
  local subdir="$1"
  local subdir_clean="${subdir%/}"
  local lock=".codex-lock.$(slugify "$subdir_clean")"
  local log="$subdir_clean/CODEX_REVIEW_LOG.md"

  [[ -d "$subdir_clean" ]] || { echo "⚠ Skipping missing $subdir"; return 0; }

  exec 9> "$lock"; if ! flock -n 9; then echo "⏭  Already running: $subdir"; return 0; fi

  [[ -f "$log" ]] || cat > "$log" <<EOF
# Codex Bugfix Log — \`$subdir\`
This log is maintained by scripts/codex-bugfix-parallel.sh
EOF

  # Collect files
  mapfile -d '' files < <(list_candidates "$subdir_clean")
  if [[ ${#files[@]} -eq 0 ]]; then
    append_log "$log" "No eligible files after excludes."
    rm -f "$lock"; return 0
  fi

  if (( RECENT_COMMITS > 0 )); then
    declare -A remaining=()
    for f in "${files[@]}"; do
      remaining["$f"]=1
    done
    local recent_paths=()
    while IFS= read -r path; do
      [[ -z "$path" ]] && continue
      if [[ -n "${remaining["$path"]:-}" ]]; then
        recent_paths+=( "$path" )
        unset "remaining[$path]"
      fi
    done < <(git log --name-only --pretty=format: -n "$RECENT_COMMITS" -- "$subdir_clean" 2>/dev/null || true)
    if (( ${#recent_paths[@]} > 0 )); then
      local ordered=( "${recent_paths[@]}" )
      for f in "${files[@]}"; do
        if [[ -n "${remaining["$f"]:-}" ]]; then
          ordered+=( "$f" )
        fi
      done
      files=( "${ordered[@]}" )
    fi
  fi

  local files_len=${#files[@]}
  local offset=0
  local pass=1
  while (( pass <= MAX_PASSES )); do
    local tmpdir; tmpdir="$(mktemp -d)"
    append_log "$log" "## Pass $pass — find and fix real bugs"

    local focus_count="$BATCH_SIZE"
    if ! [[ "$focus_count" =~ ^[0-9]+$ ]]; then
      focus_count=40
    fi
    focus_count=$((focus_count))
    local chunk=()
    if (( focus_count > 0 && files_len > 0 )); then
      local end=$(( offset + focus_count ))
      if (( end > files_len )); then
        end=$files_len
      fi
      for (( idx=offset; idx < end; idx++ )); do
        chunk+=( "${files[$idx]}" )
      done
    fi

    local batch=0
    local pf="$tmpdir/prompt-pass${pass}.txt"
    {
      echo "You are a senior Python engineer. Goal: **find a real bug and fix it** in the files listed below, limited to '$subdir'."
      echo
      echo "Prioritize concrete, high-signal issues over style:"
      echo "- exceptions that can be thrown (None/KeyError/IndexError/ValueError),"
      echo "- incorrect edge-case logic or off-by-one,"
      echo "- resource leaks (files, processes),"
      echo "- concurrency/async misuse,"
      echo "- incorrect API usage (httpx/requests, DuckDB, FAISS, etc.),"
      echo "- invalid types/return values, bad default mutables."
      echo
      echo "Rules:"
      echo "- **Stay strictly inside** '$subdir' (read/write only those files)."
      echo "- Prioritise the focus files below first. If you need more context, explore other files under '$subdir' using repo-friendly commands (e.g. rg, ls, fd) before editing."
      echo "- **Do NOT run network commands** (git push/pull/fetch, curl, pip, apt). Local edits only."
      echo "- Keep changes minimal and reversible; preserve public behavior unless the fix demands it."
      echo "- Add/adjust targeted unit tests only if trivial; otherwise leave TODOs."
      echo
      echo "FILES:"
      if (( ${#chunk[@]} > 0 )); then
        printf ' - %s\n' "${chunk[@]}"
      else
        echo " - (no seeded focus files; explore within '$subdir')"
      fi
      echo
      echo "OUTPUT:"
      echo "1) Apply the fix directly to the workspace files above (no confirmations)."
      echo "2) Append a short section '### Batch ${batch} (Pass ${pass})' to $log summarizing:"
      echo "   - what was broken,"
      echo "   - how you fixed it (1–3 bullets),"
      echo "   - any follow-ups as TODOs."
    } > "$pf"

    local run_log="$LOG_DIR/$(slugify "$subdir")/pass${pass}-batch${batch}.log"
    mkdir -p "$(dirname "$run_log")"
    local cmd=(codex exec --full-auto -s workspace-write -c 'approval_policy="never"')
    if [[ "$VERBOSE" == "true" ]]; then
      echo "[codex][$subdir][pass $pass][batch $batch] starting…"
      "${cmd[@]}" -- "$(cat "$pf")" 2>&1 | tee -a "$run_log" || true
    else
      "${cmd[@]}" -- "$(cat "$pf")" >"$run_log" 2>&1 || true
    fi

    if (( focus_count > 0 && files_len > 0 )); then
      offset=$(( (offset + focus_count) % files_len ))
    fi

    # Optional local gates (never fail the script)
    command -v ruff  >/dev/null && ruff check --fix "$subdir" || true
    command -v black >/dev/null && black "$subdir"           || true
    

    # Commit & push if there were changes inside this folder
    local changed_count
    changed_count=$(git status --porcelain "$subdir" | wc -l | tr -d ' ')
    if (( changed_count > 0 )); then
      if (( changed_count > MAX_CHANGED_FILES )); then
        append_log "$log" "**ABORT COMMIT:** too many changed files ($changed_count > $MAX_CHANGED_FILES)."
        rm -rf "$tmpdir"; break
      fi
      local msg="Codex bugfix pass $pass ($subdir): $(date -u +"%Y-%m-%d %H:%M:%SZ")"
      if [[ "$PUSH_TO_MAIN" == "true" ]]; then
        git_commit_push_with_retry "$subdir" "$msg" || {
          local branch="codex/bugfix-$(slugify "$subdir")-p$pass-$(date +%Y%m%d-%H%M%S)"
          git checkout -b "$branch"
          git add "$subdir" "$log" || true
          git commit -m "$msg" || true
          git push -u origin "$branch" || true
          if command -v "$PR_TOOL" >/dev/null 2>&1; then
            "$PR_TOOL" pr create --title "$msg" --body "Automated bugfix for \`$subdir\`." \
              --base "$BASE_BRANCH" --head "$branch" || true
          fi
          append_log "$log" "Opened branch \`$branch\` with PR (main likely protected)."
        }
      else
        local branch="codex/bugfix-$(slugify "$subdir")-p$pass-$(date +%Y%m%d-%H%M%S)"
        git checkout -b "$branch"
        git add "$subdir" "$log" || true
        git commit -m "$msg" || true
        git push -u origin "$branch" || true
        if command -v "$PR_TOOL" >/dev/null 2>&1; then
          "$PR_TOOL" pr create --title "$msg" --body "Automated bugfix for \`$subdir\`." \
            --base "$BASE_BRANCH" --head "$branch" || true
        fi
        append_log "$log" "Pushed to branch \`$branch\` with PR."
      fi
    else
      append_log "$log" "No edits on pass $pass — stopping."
      rm -rf "$tmpdir"; break
    fi

    rm -rf "$tmpdir"
    pass=$((pass+1))
  done

  rm -f "$lock"
  echo "✔ $subdir done."
}

# ===== PRECHECKS & RUN =====
need git; need codex
cd "$(git rev-parse --show-toplevel)"

pids=()
for d in "${SUBDIRS[@]}"; do bugfix_one_subdir "$d" & pids+=( "$!" ); done
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit "$fail"
