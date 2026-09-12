"""Check the repository contract without rejecting ordinary public documentation."""
import json
from pathlib import Path
import re
import subprocess
import tomllib

root = Path(__file__).resolve().parents[2]
files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root).decode().split("\0")
guidance = []
for name in set(files) - {""}:
    path = Path(name)
    if path.suffix.lower() != ".md" or not (root / path).is_file():
        continue
    # Packaged host integration instructions govern the receiving host, not this repository.
    if path.parts[:2] == ("integrations", "agent-plugins"):
        continue
    text = (root / path).read_text()
    if path.name in {"AGENTS.md", "CLAUDE.md", "GEMINI.md", "CONTRIBUTING.md"} or re.search(
        r"^#+ (?:IronGraph repository contract|Repository instructions|Agent instructions|Editing and verification)$", text, re.M
    ):
        guidance.append(name)
assert guidance == ["AGENTS.md"], f"Unexpected repository guidance: {guidance}"
for path in [root / "Cargo.toml", *root.glob("crates/*/Cargo.toml"), root / "bindings/python/Cargo.toml", root / "bindings/node/Cargo.toml"]:
    assert tomllib.loads(path.read_text())["package"].get("publish") is False, f"Internal crate publishable: {path}"
sdk = tomllib.loads((root / "bindings/rust/Cargo.toml").read_text())
for section in ("dependencies", "build-dependencies"):
    assert not any(name.startswith("irongraph-") or isinstance(value, dict) and "path" in value
                   for name, value in sdk[section].items()), "Public Cargo package depends on internal source"
node = json.loads((root / "bindings/node/package.json").read_text())
assert set(node["napi"]["targets"]) == {"aarch64-apple-darwin", "aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"}
assert len(node["optionalDependencies"]) == 3
cli = json.loads((root / "bindings/cli/package.json").read_text())
assert cli["name"] == "irongraph" and cli["bin"] == {"irongraph": "cli.cjs"}
assert cli["optionalDependencies"] == {
    "@irongraph/cli-" + target: cli["version"]
    for target in ("darwin-arm64", "linux-arm64-gnu", "linux-x64-gnu")
}
assert not {"preinstall", "install", "postinstall"} & set(cli.get("scripts", {}))
for path in [root / "bindings/node/package.json", root / "bindings/javascript/package.json",
             root / "bindings/cli/package.json", *root.glob("bindings/cli/npm/*/package.json")]:
    package = json.loads(path.read_text())
    assert package["license"] == "Apache-2.0"
    if path.parent.parent.name == "npm":
        assert set(package["files"]) == {"bin/irongraph", "bin/irongraph-mcp", "README.md", "LICENSE.txt", "THIRD_PARTY_NOTICES.txt"}
        assert package["version"] == cli["version"]
for path in (root / "bindings/node/npm/darwin-x64/package.json", root / "bindings/node/npm/win32-x64-msvc/package.json"):
    package = json.loads(path.read_text())
    assert package.get("private") is True and "publishConfig" not in package
assert subprocess.check_output(["git", "check-ignore", ".env.publish"], cwd=root, text=True).strip() == ".env.publish"
assert not subprocess.check_output(["git", "ls-files", ".env.publish"], cwd=root)
print("package and guidance boundary checks passed")
