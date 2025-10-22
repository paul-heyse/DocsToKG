#!/usr/bin/env bash
# Codex parallel reviewer (Codex 0.47+)
# - 4 parallel workers (one per SUBDIR)
# - reads agents.md per folder
# - iterative passes until no changes (or MAX_PASSES)
# - only edits inside the folder (guarded by prompt + git apply scope)
# - commits & pushes to main (PR fallback)
# - appends to <subdir>/CODEX_REVIEW_LOG.md

set -euo pipefail

# ====== CONFIG ======
SUBDIRS=(
  "src/DocsToKG/ContentDownload"
  "src/DocsToKG/DocParsing"
  "src/DocsToKG/HybridSearch"
  "src/DocsToKG/OntologyDownload"
)

BATCH_SIZE="${BATCH_SIZE:-200}"       # files per codex exec batch
MAX_PASSES="${MAX_PASSES:-3}"         # iterations per folder
BASE_BRANCH="${BASE_BRANCH:-main}"
PUSH_TO_MAIN="${PUSH_TO_MAIN:-true}"  # false -> always open a PR
PR_TOOL="${PR_TOOL:-gh}"              # needs 'gh' if PR fallback used

# Safety rails
MAX_CHANGED_FILES="${MAX_CHANGED_FILES:-500}"    # refuse mega-changes
EXEC_TIMEOUT="${EXEC_TIMEOUT:-0}"                # seconds; 0=let codex decide
RISK_MODE="${RISK_MODE:-balanced}"               # balanced | aggressive | conservative

# Excludes (token control / non-text files)
EXCLUDES=(
  ':!*.png' ':!*.jpg' ':!*.jpeg' ':!*.gif' ':!*.svg'
  ':!*.pdf' ':!*.zip' ':!*.tar' ':!*.gz' ':!*.bz2' ':!*.xz' ':!*.7z'
  ':!*.parquet' ':!*.onnx' ':!*.pt' ':!*.bin' ':!*.so' ':!*.dylib' ':!*.a' ':!*.whl'
  ':!*.ipynb' ':!node_modules/**' ':!dist/**' ':!build/**' ':!__pycache__/**'
  ':!.venv/**' ':!.mypy_cache/**' ':!.ruff_cache/**' ':!wheels/**'
)

RULES_FILE="${RULES_FILE:-.codex/rules.yml}"  # optional; referenced in prompt if present

# ====== HELPERS ======
need() { command -v "$1" >/dev/null 2>&1 || { echo "✖ Missing: $1"; exit 1; }; }

# Build a risk description for the prompt
risk_text() {
  case "$RISK_MODE" in
    aggressive)
      echo "Be proactive: apply all safe, mechanical and stylistic improvements (readability, type hints, docstrings, dead code removal, small refactors) when confidence is high."
      ;;
    conservative)
      echo "Prefer minimal, high-confidence changes only. Defer anything risky to TODOs."
      ;;
    *)
      echo "Balance fixes and caution: apply safe, mechanical improvements and straightforward refactors; leave risky items as TODOs."
      ;;
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
  # Args: subdir, msg
  local subdir="$1" msg="$2" attempt=0 max_tries=3
  git fetch origin "$BASE_BRANCH" --quiet
  git checkout "$BASE_BRANCH" --quiet
  while (( attempt < max_tries )); do
    git pull --rebase --autostash --quiet || true
    git add "$subdir" || true
    git add "$subdir/CODEX_REVIEW_LOG.md" || true
    if git diff --cached --quiet; then
      return 0
    fi
    git commit -m "$msg" || true
    if git push origin "$BASE_BRANCH" 2>/dev/null; then
      return 0
    fi
    attempt=$((attempt+1))
    sleep 1
  done
  return 1
}

open_pr() {
  local branch="$1" title="$2" body="$3"
  if command -v "$PR_TOOL" >/dev/null 2>&1; then
    "$PR_TOOL" pr create --title "$title" --body "$body" --base "$BASE_BRANCH" --head "$branch" || true
  fi
}

# ====== WORKER ======
review_one_subdir() {
  local subdir="$1"
  local lock=".codex-lock.$(echo "$subdir" | sed 's#[/ ]#-#g')"
  local log="$subdir/CODEX_REVIEW_LOG.md"
  local agents="$subdir/agents.md"

  [[ -d "$subdir" ]] || { echo "⚠ Skipping missing $subdir"; return 0; }

  # Single worker protection for this subdir
  exec 9> "$lock"
  if ! flock -n 9; then
    echo "⏭  Already running: $subdir"
    return 0
  fi

  mkdir -p "$(dirname "$lock")"

  # Ensure log exists
  if [[ ! -f "$log" ]]; then
    cat > "$log" <<EOF
# Codex Review Log — \`$subdir\`

This file is maintained by \`scripts/codex-review-parallel.sh\`.
Each run appends a new entry with:
- inputs (batch size, passes, risk mode)
- a pass-by-pass summary
- links to diffs (commit SHAs)
- raw Codex logs (collapsed)
EOF
  fi

  # List tracked files in subdir
  mapfile -d '' files < <(git ls-files -z "$subdir" "${EXCLUDES[@]}")
  if [[ ${#files[@]} -eq 0 ]]; then
    append_log "$log" "### Run $(date -u +"%Y-%m-%d %H:%M:%SZ") — No eligible files after excludes."
    rm -f "$lock"
    return 0
  fi

  # Read agents.md content once (inline to prompt so it always exists)
  local agents_inline=""
  if [[ -f "$agents" ]]; then
    agents_inline="$(printf "\n\n---- BEGIN agents.md (%s) ----\n%s\n---- END agents.md ----\n" "$agents" "$(cat "$agents")")"
  fi

  local pass=1 changed_any="false"
  while (( pass <= MAX_PASSES )); do
    local tmpdir; tmpdir="$(mktemp -d)"
    local summary_file="$tmpdir/summary.md"; : > "$summary_file"
    local risk; risk="$(risk_text)"

    append_log "$log" "## Pass $pass
- **Batch size:** \`$BATCH_SIZE\`
- **Risk mode:** \`$RISK_MODE\`
- **Codex:** \`$(codex --version 2>/dev/null || echo unknown)\`
- **Files under review:** ${#files[@]}"

    local batch_idx=0 idx=0
    while (( idx < ${#files[@]} )); do
      local chunk=( )
      for (( i=0; i < BATCH_SIZE && idx < ${#files[@]}; i++, idx++ )); do
        chunk+=( "${files[$idx]}" )
      done

      # Build prompt
      local prompt_file="$tmpdir/prompt-${batch_idx}.txt"
      {
        echo "You are a senior code reviewer and refactorer."
        echo
        echo "GOAL: Perform a comprehensive review of the files listed below **only** within '$subdir'."
        echo "Apply all safe, high-confidence improvements you recommend (formatting, docstrings,"
        echo "type hints, readability, obvious bug fixes, perf micro-optimizations, small refactors,"
        echo "security best-practices). $risk"
        echo
        [[ -f "$RULES_FILE" ]] && echo "Repo rules are in '$RULES_FILE' — follow them when applicable."
        echo "CRITICAL BOUNDARY: Do not read, write, move, or delete files outside '$subdir'."
        echo "Do not introduce new external services or break public APIs. Keep changes atomic."
        echo
        echo "FILES:"
        printf ' - %s\n' "${chunk[@]}"
        echo
        echo "OUTPUT REQUIREMENTS:"
        echo "1) Make edits directly to the workspace files listed above (no approval prompts)."
        echo "2) Append a concise Markdown report to: '$log', headed '### Batch $batch_idx (Pass $pass)'."
        echo "   - Summarize key issues fixed and any TODOs deliberately left."
        echo "3) Keep edits within '$subdir' only."
        echo "4) Execute any applicable items from ARCHITECTURE_TASKS.md for these files before general review."
        echo
        echo "CONTEXT:"
        echo "Consider the folder's agents.md guidance if present."
        printf "%s\n" "$agents_inline"
      } > "$prompt_file"

      # Build command
      # Prefer fully automatic, workspace-scoped writes if supported.
      # We don't assume flags; keep the command simple and robust on 0.47+.
      cmd=(codex exec --full-auto)
      # Limit runtime if requested
      (( EXEC_TIMEOUT > 0 )) && cmd+=(--timeout "$EXEC_TIMEOUT")
      # Run
      "${cmd[@]}" -- "$(cat "$prompt_file")" > "$tmpdir/batch-${batch_idx}.log" 2>&1 || true

      {
        echo "- Batch $batch_idx: ${#chunk[@]} files processed."
      } >> "$summary_file"

      batch_idx=$((batch_idx+1))
    done

    # Detect changes in this subdir
    local changed_files
    changed_files=$(git status --porcelain "$subdir" | wc -l | tr -d ' ')
    append_log "$log" "### Summary (Pass $pass)
$(cat "$summary_file")
- Changed files (detected by git): \`$changed_files\`"

    # If edits happened, commit & push (or PR)
    if (( changed_files > 0 )); then
      changed_any="true"
      if (( changed_files > MAX_CHANGED_FILES )); then
        append_log "$log" "**ABORTING COMMIT:** too many changed files ($changed_files > $MAX_CHANGED_FILES). Review manually."
        rm -rf "$tmpdir"; break
      fi
      local msg="Codex pass $pass ($subdir): $(date -u +"%Y-%m-%d %H:%M:%SZ")"
      if [[ "$PUSH_TO_MAIN" == "true" ]]; then
        if git_commit_push_with_retry "$subdir" "$msg"; then
          append_log "$log" "✅ Committed & pushed to \`$BASE_BRANCH\`: $msg"
        else
          # fallback: PR
          local branch="codex/review-$(echo "$subdir" | sed 's#[/ ]#-#g')-pass$pass-$(date +%Y%m%d-%H%M%S)"
          git checkout -b "$branch"
          git add "$subdir" "$log" || true
          git commit -m "$msg" || true
          git push -u origin "$branch" || true
          open_pr "$branch" "$msg" "Automated Codex review for \`$subdir\` (pass $pass). See \`$log\`."
          append_log "$log" "ℹ️ Opened branch \`$branch\` with PR (main likely protected)."
        fi
      else
        local branch="codex/review-$(echo "$subdir" | sed 's#[/ ]#-#g')-pass$pass-$(date +%Y%m%d-%H%M%S)"
        git checkout -b "$branch"
        git add "$subdir" "$log" || true
        git commit -m "$msg" || true
        git push -u origin "$branch" || true
        open_pr "$branch" "$msg" "Automated Codex review for \`$subdir\` (pass $pass). See \`$log\`."
        append_log "$log" "✅ Pushed to branch \`$branch\` with PR requested."
      fi
    fi

    rm -rf "$tmpdir"

    # Stop if no changes this pass
    if (( changed_files == 0 )); then
      append_log "$log" "No further edits detected on pass $pass — stopping."
      break
    fi

    pass=$((pass+1))
  done

  [[ "$changed_any" == "false" ]] && append_log "$log" "No changes were required."

  rm -f "$lock"
  echo "✔ $subdir done."
}

# ====== PRECHECKS ======
need git
need codex
if [[ "$PUSH_TO_MAIN" != "true" ]] && ! command -v "$PR_TOOL" >/dev/null 2>&1; then
  echo "⚠ PR tool '$PR_TOOL' not found; set PUSH_TO_MAIN=true or install GitHub CLI 'gh'."
fi


# ---- ARCHITECTURE BOOTSTRAP (one-time per run) ----
local rfc="$subdir/ARCHITECTURE_RFC.md"
local plan="$subdir/ARCHITECTURE_TASKS.md"

cat > "$tmpdir/arch-prompt.txt" <<'PROMPT'
You are a principal Python architect. Produce a concise but actionable plan to raise this folder to
a modern, modular, high-performance architecture suitable for a long-lived codebase.

Deliver TWO files in this folder:
1) ARCHITECTURE_RFC.md
   - Current-state map (packages, key modules, data flow)
   - Target architecture (layers, boundaries, interfaces/Protocols, plugin points)
   - Cross-cutting standards (typing strictness, logging/telemetry, errors, config, testing)
   - Migration plan (ordered phases, risks, rollbacks)

2) ARCHITECTURE_TASKS.md
   - A numbered checklist of concrete edits with file paths, each small and reversible
   - Mark tasks as [SAFE], [MEDIUM], or [RISKY]
   - Prefer many [SAFE]/[MEDIUM] tasks that can be automated

Rules:
- Stay inside this folder.
- Prefer Protocols/ABCs + dependency inversion over implicit coupling.
- Favor small modules; avoid god-objects; extract pure functions.
- Keep public APIs stable unless the RFC explains and migrates them.
PROMPT

codex exec --full-auto -- "$(cat "$tmpdir/arch-prompt.txt")" >/dev/null 2>&1 || true


# ====== RUN PARALLEL ======
pids=()
for d in "${SUBDIRS[@]}"; do
  review_one_subdir "$d" &
  pids+=( "$!" )
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
exit "$fail"

