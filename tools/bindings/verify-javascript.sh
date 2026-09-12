#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
javascript_directory="$repository_root/bindings/javascript"
node_directory="$repository_root/bindings/node"
cleanup() {
  rm -rf "$javascript_directory/dist"
  rm -f \
    "$node_directory/irongraph.darwin-arm64.node" \
    "$node_directory/irongraph.darwin-x64.node" \
    "$node_directory/irongraph.linux-arm64-gnu.node" \
    "$node_directory/irongraph.linux-x64-gnu.node" \
    "$node_directory/irongraph.win32-x64-msvc.node" \
    "$node_directory/generated.d.ts" \
    "$node_directory/npm/darwin-arm64/irongraph.darwin-arm64.node" \
    "$node_directory/npm/darwin-x64/irongraph.darwin-x64.node" \
    "$node_directory/npm/linux-arm64-gnu/irongraph.linux-arm64-gnu.node" \
    "$node_directory/npm/linux-x64-gnu/irongraph.linux-x64-gnu.node" \
    "$node_directory/npm/win32-x64-msvc/irongraph.win32-x64-msvc.node"
}
trap cleanup EXIT
cleanup

cd "$javascript_directory"
npm ci --legacy-peer-deps
npm test
npm run build
npm run pack:check

cd "$node_directory"
npm ci
npx napi build --platform --release --dts generated.d.ts
npm test
npm run pack:check

case "$(node -p 'process.platform + "-" + process.arch')" in
  darwin-arm64) platform_package="darwin-arm64" ;;
  darwin-x64) platform_package="darwin-x64" ;;
  linux-arm64) platform_package="linux-arm64-gnu" ;;
  linux-x64) platform_package="linux-x64-gnu" ;;
  win32-x64) platform_package="win32-x64-msvc" ;;
  *) echo "unsupported Node package verification platform" >&2; exit 1 ;;
esac
native_binary="$(find "$node_directory" -maxdepth 1 -type f -name 'irongraph.*.node' -print -quit)"
test -n "$native_binary"
cp "$native_binary" "$node_directory/npm/$platform_package/"
cd "$node_directory/npm/$platform_package"
npm pack --dry-run
