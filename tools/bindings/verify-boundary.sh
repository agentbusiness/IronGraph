#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

# Product documentation is allowed; repository guidance remains exclusively AGENTS.md.
test "$(rg --files --hidden -g 'AGENTS.md' -g '!target/**' -g '!.git/**' -g '!.unlazy/**' -g '!**/node_modules/**')" = "AGENTS.md"
python3 tools/release/check_boundary.py

browser_data_paths="$(rg -o '\.route\("/[^"]+", post' \
  crates/server/src/server/http.rs crates/server/src/server/remote.rs \
  | sed -E 's/.*\.route\("([^"]+)", post/\1/' | sort -u)"
test "$browser_data_paths" = "/api/query"
! rg -n '\.route\("/v[0-9]+/' crates/server/src/server

test "$(rg -o "'/web[^']*'" web/src/App.tsx | sort -u | tr '\n' ' ')" = "'/web/' '/web/docs' '/web/documents' '/web/settings' '/web/streams' '/web/training' "

backend_variants="$(sed -n '/pub enum BackendKind {/,/^}/p' crates/execution/src/backend.rs \
  | rg '^    [A-Z][A-Za-z0-9_]*,' | sed -E 's/^    ([A-Za-z0-9_]+),/\1/' | tr '\n' ' ')"
test "$backend_variants" = "Cpu Metal Cuda "

generated_material="$(git status --short --untracked-files=all | \
  rg '(^|/)(dist|node_modules|__pycache__|target)/|\.(node|whl|tgz|pyc)$' || true)"
test -z "$generated_material"
echo 'boundary verification passed'
