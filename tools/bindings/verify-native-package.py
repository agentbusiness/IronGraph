#!/usr/bin/env python3
"""Build a packaged Rust consumer outside the workspace and deny runtime networking on macOS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="irongraph-native-package-"))
    output.mkdir(parents=True, exist_ok=True)
    source_files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    ).decode().split("\0")
    report = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "source_files_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sorted(set(source_files)) if name and (ROOT / name).is_file()
        },
        "commands": [],
    }

    def run(command, cwd=ROOT, env=None, log="command.log"):
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
        (output / log).write_text(result.stdout + result.stderr)
        report["commands"].append({"command": command, "cwd": str(cwd), "exit_status": result.returncode, "log": log})
        (output / "report.json").write_text(json.dumps(report, indent=2))
        if result.returncode:
            raise RuntimeError(f"{command[0]} failed; see {output / log}")
        return result.stdout

    run(["cargo", "build", "-p", "irongraph-ffi", "--no-default-features"], log="native-build.log")
    run(["cargo", "package", "--manifest-path", "bindings/rust/Cargo.toml", "--allow-dirty", "--no-verify"], log="package.log")
    version = tomllib.loads((ROOT / "bindings/rust/Cargo.toml").read_text())["package"]["version"]
    artifact = output / f"irongraph-sdk-{version}.crate"
    shutil.copy2(ROOT / "bindings/rust/target/package" / artifact.name, artifact)
    native = output / "libirongraph_ffi.a"
    shutil.copy2(ROOT / "target/debug/libirongraph_ffi.a", native)
    with tarfile.open(artifact) as archive:
        archive.extractall(output / "package", filter="data")
    consumer = output / "consumer"
    (consumer / "tests").mkdir(parents=True, exist_ok=True)
    package = output / "package" / f"irongraph-sdk-{version}"
    (consumer / "Cargo.toml").write_text(
        '[package]\nname="irongraph-artifact-consumer"\nversion="0.1.0"\nedition="2024"\n'
        '[workspace]\n[dependencies]\nirongraph-sdk={path=' + json.dumps(str(package)) + '}\nserde_json="1.0"\ntempfile="3"\n')
    shutil.copy2(ROOT / "bindings/rust/tests/embedded.rs", consumer / "tests/embedded.rs")
    env = os.environ.copy()
    env["IRONGRAPH_NATIVE_DIR"] = str(output)
    if sys.platform == "darwin":
        env["RUSTFLAGS"] = "-C strip=none"
    events = run(["cargo", "test", "--no-run", "--message-format=json"], cwd=consumer, env=env, log="consumer-build.log")
    executable = next(item["executable"] for line in events.splitlines() if line.startswith("{")
                      for item in [json.loads(line)] if item.get("reason") == "compiler-artifact"
                      and item.get("executable") and item["target"]["name"] == "embedded")
    command = [executable, "--exact", "packaged_sdk_preserves_documents_and_integer_precision", "--nocapture"]
    if sys.platform == "darwin":
        command = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)(deny process-fork)", *command]
    run(command, cwd=consumer, env=env, log="consumer-test.log")
    report.update({"version": version, "network_and_child_processes_denied": sys.platform == "darwin",
                   "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [artifact, native]}})
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Packaged native consumer passed: {output / 'report.json'}")


if __name__ == "__main__":
    main()
