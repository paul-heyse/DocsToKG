#!/usr/bin/env bash
# Configure Codex to auto-approve edits in your workspace
# Run this script to enable auto-approval for your DocsToKG repository
# This is useful for continuous development and testing
# It allows you to make edits without manual approval
# and helps streamline the development process

# Set the sandbox mode to allow workspace writes
# This allows Codex to make changes to your workspace
# and commit them to your repository
cat > ~/.codex/config.toml <<'TOML'
sandbox_mode = "workspace-write"
approval_mode = "auto"          # <- make edits without asking
model_reasoning_effort = "high"

[[trusted_workspaces]]
path = "/home/paul/DocsToKG"    # absolute path to your repo
TOML

codex /status   # should now show Approval: auto (or equivalent)