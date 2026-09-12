#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
verification_environment="$(mktemp -d /tmp/irongraph-python-verification.XXXXXX)"
cleanup() {
  rm -rf "$verification_environment"
}
trap cleanup EXIT

python3 -m venv "$verification_environment"
"$verification_environment/bin/python" -m pip install --quiet 'maturin>=1.15,<2'
export VIRTUAL_ENV="$verification_environment"
export PATH="$verification_environment/bin:$PATH"

cd "$repository_root"
maturin build --profile dev --manifest-path bindings/python/Cargo.toml \
  --no-default-features --out "$verification_environment/smoke-wheels"
"$verification_environment/bin/python" -m pip install --quiet --force-reinstall \
  "$verification_environment"/smoke-wheels/*.whl
"$verification_environment/bin/python" bindings/python/test_smoke.py
maturin build --release --manifest-path bindings/python/Cargo.toml \
  --out "$verification_environment/wheels"
"$verification_environment/bin/python" -m pip install --quiet --force-reinstall \
  "$verification_environment"/wheels/*.whl
"$verification_environment/bin/python" -c \
  'from irongraph import Client, EmbeddedDatabase; assert Client and EmbeddedDatabase'
