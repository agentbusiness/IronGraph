#!/usr/bin/env bash
# One entrypoint for local versioning, native builds, packaging, and publishing.
set -euo pipefail
set +x
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$repository_root/tools/release/release.py" "$@"
