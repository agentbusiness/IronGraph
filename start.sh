#!/usr/bin/env bash
# Build what changed and run IronGraph.
#
# One script. There were two — this and a near-identical `start-workspace.sh` — differing only in
# which data directory they used and which Bolt port they picked, which is not a reason for two
# scripts. Both are options here.
#
# Release, always. A debug build runs the model's tensor maths tens of times slower, which turns
# loading the weights into minutes; the compile is paid once, the wait would be paid every start.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export IRONGRAPH_ROLE=node
export IRONGRAPH_DATA_DIR="${IRONGRAPH_DATA_DIR:-${HOME:?HOME must be set}/.irongraph/data}"
export IRONGRAPH_HTTP_ADDR="${IRONGRAPH_HTTP_ADDR:-127.0.0.1:18484}"
export IRONGRAPH_MCP_URL="${IRONGRAPH_MCP_URL:-http://$IRONGRAPH_HTTP_ADDR}"
export IRONGRAPH_MCP_ADDR="${IRONGRAPH_MCP_ADDR:-127.0.0.1:18488}"
export IRONGRAPH_BOLT_ADDR="${IRONGRAPH_BOLT_ADDR:-127.0.0.1:18485}"
export IRONGRAPH_STREAM_ADDR="${IRONGRAPH_STREAM_ADDR:-127.0.0.1:18486}"
export IRONGRAPH_QUEUE_ADDR="${IRONGRAPH_QUEUE_ADDR:-127.0.0.1:18487}"
export RUST_LOG="${RUST_LOG:-info}"
unset IRONGRAPH_JOIN_INVITATION
unset IRONGRAPH_ROUTER_CONNECT_ENDPOINT

# `RustEmbed` tracks `web/dist`. Vite removes and recreates that directory on every invocation,
# which makes Cargo rebuild the server crate based on timestamps even when every output byte is
# identical. Hash the actual UI inputs first and do not invoke Vite unless one changed.
UI_SOURCE_STAMP="$ROOT_DIR/target/.ui-source-stamp"
UI_SOURCE_HASH="$(
  find "$ROOT_DIR/web" -type f \
    ! -path "$ROOT_DIR/web/node_modules/*" \
    ! -path "$ROOT_DIR/web/dist/*" \
    ! -path "$ROOT_DIR/web/coverage/*" \
    ! -path "$ROOT_DIR/web/.vite/*" \
    ! -name '*.tsbuildinfo' \
    ! -name '.DS_Store' \
    -print0 \
    | sort -z \
    | xargs -0 shasum -a 256 \
    | shasum -a 256 \
    | cut -d' ' -f1
)"
if [ ! -d "$ROOT_DIR/web/dist" ] \
  || [ ! -f "$UI_SOURCE_STAMP" ] \
  || [ "$(cat "$UI_SOURCE_STAMP")" != "$UI_SOURCE_HASH" ]; then
  printf 'Building the web UI...\n'
  (cd "$ROOT_DIR/web" && npm run --silent build)
  mkdir -p "$ROOT_DIR/target"
  printf '%s' "$UI_SOURCE_HASH" > "$UI_SOURCE_STAMP"
else
  printf 'Web UI unchanged; using the existing bundle.\n'
fi

printf 'Building the binary...\n'
cargo build --release --package irongraph --package irongraph-mcp --manifest-path "$ROOT_DIR/Cargo.toml"

# Refresh only integrations that IronGraph installed previously. New hosts are never injected
# implicitly; the explicit setup command performs the first install.
if ! "$ROOT_DIR/target/release/irongraph-mcp" integrations update --all; then
  printf 'Integration refresh failed; IronGraph will still start. Run irongraph-mcp integrations status --all for details.\n' >&2
fi

printf '\nWeb:     http://%s/web/\n' "$IRONGRAPH_HTTP_ADDR"
printf 'Streams: http://%s/web/streams\n' "$IRONGRAPH_HTTP_ADDR"
printf 'API:     POST http://%s/api/query\n' "$IRONGRAPH_HTTP_ADDR"
printf 'MCP:     http://%s/mcp\n' "$IRONGRAPH_MCP_ADDR"
printf 'Data:   %s\n\n' "$IRONGRAPH_DATA_DIR"
printf 'Hard-reload (Cmd-Shift-R) after a UI change — the browser caches the bundle.\n\n'

cd "$ROOT_DIR"
exec "$ROOT_DIR/target/release/irongraph" "$@"
