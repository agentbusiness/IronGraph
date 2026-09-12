#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/irongraph-opencypher.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT INT TERM

commit=677cbafabb8c3c5eed458fd3b1ec0daec8d67d23
git clone --quiet --filter=blob:none --no-checkout https://github.com/opencypher/openCypher "$work_dir/openCypher"
git -C "$work_dir/openCypher" checkout --quiet "$commit"

OPENCYPHER_TCK_DIR="$work_dir/openCypher/tck/features" \
  cargo test --manifest-path "$repo_root/Cargo.toml" \
    --test cypher --features accelerator \
    full_opencypher_tck_gpu_conformance -- --ignored --nocapture
