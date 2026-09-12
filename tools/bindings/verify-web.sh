#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root/web"

npm ci
npm test
npm run lint
npm run build
test -f dist/index.html
rg -q '/web/' dist/index.html
