#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/irongraph-opencypher.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT INT TERM

commit=677cbafabb8c3c5eed458fd3b1ec0daec8d67d23
git clone --quiet --filter=blob:none --no-checkout https://github.com/opencypher/openCypher "$work_dir/openCypher"
git -C "$work_dir/openCypher" checkout --quiet "$commit"

case "${1:-cpu}" in
  cpu)
    OPENCYPHER_TCK_DIR="$work_dir/openCypher/tck/features" \
      cargo test --manifest-path "$repo_root/Cargo.toml" \
        --test cypher full_opencypher_tck_canonical_cpu_conformance \
        -- --ignored --nocapture
    ;;
  metal)
    OPENCYPHER_TCK_DIR="$work_dir/openCypher/tck/features" \
      cargo test --manifest-path "$repo_root/Cargo.toml" \
        --test cypher --features legacy-graph \
        full_opencypher_tck_gpu_conformance -- --ignored --nocapture
    ;;
  *)
    echo "usage: $0 [cpu|metal]" >&2
    exit 2
    ;;
esac
