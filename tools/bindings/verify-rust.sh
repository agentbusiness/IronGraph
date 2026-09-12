#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

cargo fmt --all -- --check
cargo check --workspace
cargo check --workspace --no-default-features
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
