#!/usr/bin/env bash
# Codex 0.47+ local automation
# - 4 parallel workers (one per folder)
# - per-folder "Architectural Kickoff" (RFC + tasks) on first cycle
# - iterative passes; pushes to main (PR fallback)
# - edits are bounded to the folder by prompt + staging scope
# - appends to <folder>/CODEX_REVIEW_LOG.md

set -euo pipefail

# ===== CONFIG =====
SUBDIRS=(
  "src/DocsToKG/ContentDownload"
  "src/DocsToKG/DocParsing"
  "src/DocsToKG/HybridSearch"
  "src/DocsToKG/OntologyDownload"
)

BATCH_SIZE="${BATCH_SIZE:-200}"        # files per pass per codex run
MAX_PASSES="${MAX_PASSES:-3}"          # keep this small while you warm up
BASE_BRANCH="${BASE_BRANCH:-main}"
PUSH_TO_MAIN="${PUSH_TO_MAIN:-true}"   # set false to always open PRs
PR_TOOL="${PR_TOOL:-gh}"               # for PR fallback

# safety rails
MAX_CHANGED_FILES="${MAX_CHANGED_FILES:-500}"   # refuse mega-diffs early on
EXEC_TIMEOUT="${EXEC_TIMEOUT:-0}"               # 0 = let codex decide
RISK_MODE="${RISK_MODE:-balanced}"              # balanced|aggressive|conservative

# optional local standards file you may add later
RULES_FILE="${RULES_FILE:-.codex/rules.yml}"

# exclude noisy/binary stuff from review sets

EXCLUDES=(
  ':(exclude,glob)**/*.png' ':(exclude,glob)**/*.jpg' ':(exclude,glob)**/*.jpeg' ':(exclude,glob)**/*.gif' ':(exclude,glob)**/*.svg'
  ':(exclude,glob)**/*.pdf' ':(exclude,glob)**/*.zip' ':(exclude,glob)**/*.tar' ':(exclude,glob)**/*.gz'  ':(exclude,glob)**/*.bz2' ':(exclude,glob)**/*.xz' ':(exclude,glob)**/*.7z'
  ':(exclude,glob)**/*.parquet' ':(exclude,glob)**/*.onnx' ':(exclude,glob)**/*.pt' ':(exclude,glob)**/*.bin' ':(exclude,glob)**/*.so' ':(exclude,glob)**/*.dylib' ':(exclude,glob)**/*.a' ':(exclude,glob)**/*.whl'
  ':(exclude,glob)**/*.ipynb'
  ':(exclude,glob)**/node_modules/**' ':(exclude,glob)**/dist/**' ':(exclude,glob)**/build/**'
  ':(exclude,glob)**/__pycache__/**' ':(exclude,glob)**/.venv/**' ':(exclude,glob)**/.mypy_cache/**' ':(exclude,glob)**/.ruff_cache/**'
  ':(exclude,glob)**/wheels/**'
)

# ===== helpers =====
need() { command -v "$1" >/dev/null 2>&1 || { echo "✖ Missing: $1"; exit 1; }; }

risk_text() {
  case "$RISK_MODE" in
    aggressive)   echo "Be proactive: apply safe, high-confidence refactors (modularity, interfaces, typed models, small API shims).";;
    conservative) echo "Prefer minimal, high-confidence fixes; leave riskier items as TODOs.";;
    *)            echo "Balance improvement and caution; apply safe mechanical changes and straightforward refactors.";;
  esac
}

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
  git fetch origin "$BASE_BRANCH" --quiet
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

open_pr() {
  local branch="$1" title="$2" body="$3"
  if command -v "$PR_TOOL" >/dev/null 2>&1; then
    "$PR_TOOL" pr create --title "$title" --body "$body" --base "$BASE_BRANCH" --head "$branch" || true
  fi
}

# ===== one worker per folder =====
review_one_subdir() {
  local subdir="$1"
  local lock=".codex-lock.$(echo "$subdir" | sed 's#[/ ]#-#g')"
  local log="$subdir/CODEX_REVIEW_LOG.md"
  local agents="$subdir/agents.md"

  [[ -d "$subdir" ]] || { echo "⚠ Skipping missing $subdir"; return 0; }

  # ensure single worker per folder
  exec 9> "$lock"; if ! flock -n 9; then echo "⏭  Already running: $subdir"; return 0; fi

  # create log if absent
  [[ -f "$log" ]] || cat > "$log" <<EOF
# Codex Review Log — \`$subdir\`
This log is maintained by \`scripts/codex-review-parallel.sh\`.
EOF

  # collect reviewable files
  mapfile -d '' files < <(git ls-files -z "$subdir" "${EXCLUDES[@]}")
  if [[ ${#files[@]} -eq 0 ]]; then
    append_log "$log" "No eligible files after excludes."
    rm -f "$lock"; return 0
  fi

  # ===== Architectural Kickoff (first cycle) =====
  # produce/update ARCHITECTURE_RFC.md and ARCHITECTURE_TASKS.md guiding this folder
  if [[ ! -f "$subdir/ARCHITECTURE_RFC.md" || ! -f "$subdir/ARCHITECTURE_TASKS.md" ]]; then
    local tdir; tdir="$(mktemp -d)"
    cat > "$tdir/kickoff.txt" <<PROMPT
You are a principal Python architect focusing on '$subdir'.

Goal: think through local architectural improvements and write TWO files:
1) ARCHITECTURE_RFC.md
   - current state (modules, data flow), risks/debts
   - target architecture for this folder (layers, boundaries, interfaces/Protocols)
   - cross-cutting standards (typing, logging, errors, config, testing)
   - migration plan (ordered phases, rollbacks)

2) ARCHITECTURE_TASKS.md
   - a numbered checklist of small, reversible tasks with concrete file paths
   - tag each: [SAFE] [MEDIUM] [RISKY]
   - prioritize many [SAFE]/[MEDIUM] automation-friendly items

Rules:
- Stay strictly inside '$subdir'.
- Prefer Protocols/ABCs + dependency inversion.
- Extract pure functions; isolate I/O.
- Tighten typing/docstrings; remove obvious dead code.
- Do NOT change files outside '$subdir'.
PROMPT
    local cmd=(codex exec --full-auto)
    (( EXEC_TIMEOUT > 0 )) && cmd+=(--timeout "$EXEC_TIMEOUT")
    "${cmd[@]}" -- "$(cat "$tdir/kickoff.txt")" >/dev/null 2>&1 || true
    rm -rf "$tdir"
    append_log "$log" "Initialized ARCHITECTURE_RFC.md and ARCHITECTURE_TASKS.md."
  fi

  # inline agents.md (if present) for extra context
  local agents_inline=""
  [[ -f "$agents" ]] && agents_inline=$'\n\n---- BEGIN agents.md ----\n'"$(cat "$agents")"$'\n---- END agents.md ----\n'

  # ===== iterative passes =====
  local pass=1
  while (( pass <= MAX_PASSES )); do
    local tmpdir; tmpdir="$(mktemp -d)"
    local risk; risk="$(risk_text)"
    append_log "$log" "## Pass $pass — mode: $RISK_MODE"

    # batch the file set to keep tokens in check
    local idx=0 batch=0
    while (( idx < ${#files[@]} )); do
      local chunk=( )
      for (( i=0; i < BATCH_SIZE && idx < ${#files[@]}; i++, idx++ )); do
        chunk+=( "${files[$idx]}" )
      done

      cat > "$tmpdir/prompt-${batch}.txt" <<PROMPT
You are a senior reviewer/refactorer for '$subdir'.

1) Use ARCHITECTURE_TASKS.md — execute any applicable [SAFE]/[MEDIUM] items for these files first.
2) Then perform a thorough review focused on modularity, reuse, typing/docstrings, correctness,
   clear boundaries (Protocols/ABCs), and small, safe refactors. ${risk}
3) Keep edits *strictly inside* '$subdir'.

FILES:
$(printf ' - %s\n' "${chunk[@]}")

OUTPUT:
- Apply edits directly (no confirmations).
- Append a succinct section "### Batch ${batch} (Pass ${pass})" to $log summarizing what changed and TODOs.

Context:
- ARCHITECTURE_RFC.md
- ARCHITECTURE_TASKS.md
- $( [[ -f "$RULES_FILE" ]] && echo "$RULES_FILE" || echo "(no repo rules file yet)" )
$agents_inline
PROMPT

      local cmd=(codex exec --full-auto)
      (( EXEC_TIMEOUT > 0 )) && cmd+=(--timeout "$EXEC_TIMEOUT")
      "${cmd[@]}" -- "$(cat "$tmpdir/prompt-${batch}.txt")" >/dev/null 2>&1 || true
      batch=$((batch+1))
    done

    # optional quality gates (run if present; never fail the script)
    command -v ruff  >/dev/null && ruff check --fix "$subdir" || true
    command -v black >/dev/null && black "$subdir"           || true
    command -v mypy  >/dev/null && mypy "$subdir"            || true

    # commit only changes inside this folder
    local changed inside_count
    inside_count=$(git status --porcelain "$subdir" | wc -l | tr -d ' ')
    if (( inside_count > 0 )); then
      if (( inside_count > MAX_CHANGED_FILES )); then
        append_log "$log" "**ABORTING COMMIT:** changed files $inside_count > $MAX_CHANGED_FILES (raise limit to allow)."
        rm -rf "$tmpdir"; break
      fi
      local msg="Codex pass $pass ($subdir): $(date -u +"%Y-%m-%d %H:%M:%SZ")"
      if [[ "$PUSH_TO_MAIN" == "true" ]]; then
        git_commit_push_with_retry "$subdir" "$msg" || {
          local branch="codex/review-$(echo "$subdir" | sed 's#[/ ]#-#g')-p$pass-$(date +%Y%m%d-%H%M%S)"
          git checkout -b "$branch"; git add "$subdir" "$log" || true; git commit -m "$msg" || true
          git push -u origin "$branch" || true
          open_pr "$branch" "$msg" "Automated Codex review for \`$subdir\`."
          append_log "$log" "Opened branch \`$branch\` with PR (main likely protected)."
        }
      else
        local branch="codex/review-$(echo "$subdir" | sed 's#[/ ]#-#g')-p$pass-$(date +%Y%m%d-%H%M%S)"
        git checkout -b "$branch"; git add "$subdir" "$log" || true; git commit -m "$msg" || true
        git push -u origin "$branch" || true
        open_pr "$branch" "$msg" "Automated Codex review for \`$subdir\`."
        append_log "$log" "Pushed to branch \`$branch\` with PR."
      fi
    else
      append_log "$log" "No further edits on pass $pass — stopping."
      rm -rf "$tmpdir"; break
    fi

    rm -rf "$tmpdir"
    pass=$((pass+1))
  done

  rm -f "$lock"
  echo "✔ $subdir done."
}

# ===== prechecks & run =====
need git; need codex

pids=()
for d in "${SUBDIRS[@]}"; do
  review_one_subdir "$d" &
  pids+=( "$!" )
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit "$fail"
