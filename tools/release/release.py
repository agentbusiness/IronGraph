#!/usr/bin/env python3
"""Local-only release driver. Builds never receive publication credentials."""
from __future__ import annotations

import argparse
import base64
import email.parser
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "aarch64-apple-darwin": "darwin-arm64",
    "aarch64-unknown-linux-gnu": "linux-arm64-gnu",
    "x86_64-unknown-linux-gnu": "linux-x64-gnu",
}
SECRETS = ("NPM_TOKEN", "MATURIN_PYPI_TOKEN", "CARGO_REGISTRY_TOKEN", "GITHUB_TOKEN")
CONFIG_KEYS = set(SECRETS) | {
    "IRONGRAPH_BINARY_REPO", "IRONGRAPH_RELEASE_REVIEWED",
    "IRONGRAPH_MANYLINUX_ARM64", "IRONGRAPH_MANYLINUX_AMD64",
    "MACOSX_DEPLOYMENT_TARGET",
}
VERSION_RE = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
PUBLIC_FILES = {"README.md", "LICENSE.txt", "THIRD_PARTY_NOTICES.txt"}
# Keep private audit markers out of the public source text itself.
PRIVATE_BYTES = re.compile(b"|".join(bytes.fromhex(value) for value in (
    "6174686c657261", "6e657879726f6e", "676c6f62616c5b205f2d5d3f636f72746578",
    "73696d616e6c616369", "6c61737a6c6f", "6cc3a1737a6cc3b3",
    "5c626c6163695c62", "5c6273696d616e5c62", "5c62696e74656c6c6967656e63655c62",
    "2f55736572732f",
)), re.I)


class ReleaseError(Exception):
    pass


def load_config(path: Path) -> dict[str, str]:
    """A dotenv file is data, never executable shell input."""
    result = {}
    if path.exists():
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise ReleaseError(f"{path.name}: require an ordinary file with mode 600 (chmod 600).")
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator or key not in CONFIG_KEYS:
                raise ReleaseError(f"{path.name}:{number}: unknown configuration key or invalid assignment.")
            value = value.strip()
            if value.startswith(('"', "'")):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ReleaseError(f"{path.name}:{number}: unclosed quote.")
                value = value[1:-1]
            if key in result or any(c in value for c in ("\x00", "\r", "\n")):
                raise ReleaseError(f"{path.name}:{number}: duplicate key or invalid value.")
            result[key] = value
    for key in CONFIG_KEYS:
        if key in os.environ:
            result[key] = os.environ[key]
    for key, value in result.items():
        if any(c in value for c in ("\x00", "\r", "\n")) or key in SECRETS and any(ord(c) < 33 or ord(c) > 126 for c in value):
            raise ReleaseError(f"{key}: invalid characters in configuration value.")
    return result


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items()
           if not any(word in k.upper() for word in ("TOKEN", "PASSWORD", "SECRET", "CREDENTIAL"))
           and k not in {"NODE_OPTIONS", "NODE_PATH", "PYTHONPATH", "PYTHONHOME", "BASH_ENV", "ENV", "NPM_CONFIG_USERCONFIG",
                         "CARGO_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTUP_TOOLCHAIN",
                         "IRONGRAPH_NATIVE_DIR", "NAPI_RS_NATIVE_LIBRARY_PATH", "DOCS_RS"}}
    env.update(extra or {})
    return env


def run(argv, cwd=ROOT, env=None, capture=False):
    command = [str(a) for a in argv]
    print("Running " + " ".join(command[:5]), flush=True)
    result = subprocess.run(command, cwd=cwd, env=clean_env(env), text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        # Captured output may contain registry diagnostics; don't print authentication material.
        if capture and not env:
            print((result.stderr or "")[-3000:], file=sys.stderr)
        raise ReleaseError(f"{command[0]} failed with exit {result.returncode}; retry the same staged version.")
    return result.stdout if capture else ""


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def current_version(root=ROOT):
    return tomllib.loads((root / "Cargo.toml").read_text())["package"]["version"]


def next_version(version, bump):
    match = VERSION_RE.fullmatch(version)
    if not match:
        raise ReleaseError("Version must be MAJOR.MINOR.PATCH with no leading zeroes.")
    values = list(map(int, match.groups()))
    index = {"major": 0, "minor": 1, "patch": 2}[bump]
    values[index] += 1
    values[index + 1:] = [0] * (2 - index)
    return ".".join(map(str, values))


def replace_one(text, pattern, replacement, label):
    updated, count = re.subn(pattern, replacement, text, flags=re.M)
    if count != 1:
        raise ReleaseError(f"{label}: expected one version anchor, found {count}.")
    return updated


def version_updates(root, version):
    """Calculate every edit before writing any; never edit by line range."""
    if not VERSION_RE.fullmatch(version):
        raise ReleaseError("Invalid release version.")
    files = {}
    manifests = [root / "Cargo.toml", *root.glob("crates/*/Cargo.toml"),
                 *root.glob("bindings/*/Cargo.toml")]
    own_names = set()
    for path in manifests:
        text = path.read_text()
        package = tomllib.loads(text)["package"]
        own_names.add(package["name"])
        files[path] = replace_one(text, r'^(version\s*=\s*)"[^"\n]+"',
                                  lambda m: m[1] + json.dumps(version), str(path))
    path = root / "bindings/python/pyproject.toml"
    files[path] = replace_one(path.read_text(), r'^(version\s*=\s*)"[^"\n]+"',
                              lambda m: m[1] + json.dumps(version), str(path))
    for path in [*root.glob("bindings/*/package.json"), *root.glob("bindings/node/npm/*/package.json"),
                 *root.glob("bindings/cli/npm/*/package.json"),
                 root / "web/package.json"]:
        data = read_json(path)
        data["version"] = version
        for name in data.get("optionalDependencies", {}):
            if name.startswith("@irongraph/"):
                data["optionalDependencies"][name] = version
        files[path] = json.dumps(data, indent=2) + "\n"
        lock = path.with_name("package-lock.json")
        if lock.exists():
            data = read_json(lock)
            data["version"] = version
            data["packages"][""]["version"] = version
            for name in data["packages"][""].get("optionalDependencies", {}):
                if name.startswith("@irongraph/"):
                    data["packages"][""]["optionalDependencies"][name] = version
            # Platform dependencies are unpublished at build time and intentionally optional.
            for name in list(data.get("packages", {})):
                if name.startswith("node_modules/@irongraph/"):
                    del data["packages"][name]
            files[lock] = json.dumps(data, indent=2) + "\n"
    for lock in (root / "Cargo.lock", root / "bindings/rust/Cargo.lock"):
        if lock.exists():
            sections = lock.read_text().split("[[package]]")
            for i, section in enumerate(sections[1:], 1):
                name = re.search(r'^name = "([^"]+)"', section, re.M)
                if name and name[1] in own_names and not re.search(r'^source = ', section, re.M):
                    sections[i] = replace_one(section, r'^(version = )"[^"\n]+"',
                                             lambda m: m[1] + json.dumps(version), str(lock))
            files[lock] = "[[package]]".join(sections)
    loader = root / "bindings/node/index.js"
    old = current_version(root)
    if loader.exists():
        files[loader] = loader.read_text().replace(f"'{old}'", f"'{version}'").replace(
            f"expected {old} but got", f"expected {version} but got")
    return files


def write_versions(root, version):
    updates = version_updates(root, version)
    for path, text in updates.items():
        path.write_text(text)
    return updates


def source_files(root=ROOT):
    output = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
    paths = sorted(set(output.decode().split("\0")) - {""})
    result = []
    for name in paths:
        path = Path(name)
        if any(part in {".git", ".unlazy", "node_modules", "target", "dist", "__pycache__"} for part in path.parts):
            continue
        if any(part.startswith(".env") and not part.endswith(".example") for part in path.parts):
            continue
        if path.suffix in {".whl", ".tgz", ".node", ".pyc"}:
            continue
        if path.parts[:3] == ("bindings", "cli", "npm") and "bin" in path.parts[3:]:
            continue
        if (root / path).is_symlink():
            raise ReleaseError(f"Source staging rejects symlinks: {name}")
        if (root / path).is_file():
            result.append(path)
    return result


def fingerprint(root=ROOT):
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(str(path).encode() + b"\0" + (root / path).read_bytes())
    return digest.hexdigest()


def stage_source(destination, version, root=ROOT):
    destination.mkdir(parents=True)
    for path in source_files(root):
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / path, target)
    write_versions(destination, version)


def finish_staging(stage, state):
    if state.get("staged"):
        return
    source = stage / "source"
    partial = stage / "source.partial"
    if not source.exists():
        if partial.exists():
            if partial.is_symlink():
                raise ReleaseError("Refusing symlinked release staging directory.")
            shutil.rmtree(partial)  # Only this release's incomplete generated copy.
        stage_source(partial, state["version"])
        if fingerprint() != state["source_fingerprint"]:
            raise ReleaseError("Source changed while staging; restore the reviewed source before resuming.")
        partial.rename(source)
    state["staged"] = True
    write_json(stage / "state.json", state)


def finalize_versions(stage, state, root=ROOT):
    """Journal before/after text so interruption never strands a partially bumped checkout."""
    journal = stage / "version-journal.json"
    if not journal.exists():
        if fingerprint(root) != state["source_fingerprint"]:
            raise ReleaseError("Packages uploaded, but source changed. Local versions were not overwritten.")
        updates = version_updates(root, state["version"])
        write_json(journal, {str(path.relative_to(root)): {"before": path.read_text(), "after": text}
                             for path, text in updates.items()})
    changes = read_json(journal)
    for name, change in changes.items():
        if (root / name).read_text() not in (change["before"], change["after"]):
            raise ReleaseError(f"Local version file has unrelated changes: {name}. Preserve them before resuming finalization.")
    for name, change in changes.items():
        path = root / name
        temporary = path.with_name(path.name + ".release-tmp")
        temporary.write_text(change["after"])
        temporary.replace(path)
    state["complete"] = True
    write_json(stage / "state.json", state)


def tool_environment(source, target):
    flags = [f"--remap-path-prefix={source}=/irongraph",
             f"--remap-path-prefix={Path.home()}=/build-home", "-C", "debuginfo=0"]
    # Encoded flags override .cargo/config.toml. Preserve procedural-macro symbol tables
    # on macOS even in the isolated release build.
    if target == "aarch64-apple-darwin":
        flags.extend(["-C", "strip=none"])
    env = {"CARGO_TARGET_DIR": str(source.parent / "build" / target),
           "CARGO_ENCODED_RUSTFLAGS": "\x1f".join(flags),
           "RUSTFLAGS": "", "PYO3_PYTHON": sys.executable}
    if target == "aarch64-apple-darwin":
        env["CARGO_PROFILE_RELEASE_STRIP"] = "none"
    return env


def llvm_objcopy(source):
    sysroots = [Path(run(["rustc", "--print", "sysroot"], source, capture=True).strip())]
    channel = tomllib.loads((source / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    rustup_sysroot = Path(run(["rustup", "run", channel, "rustc", "--print", "sysroot"],
                              source, capture=True).strip())
    if rustup_sysroot not in sysroots:
        sysroots.append(rustup_sysroot)
    for sysroot in sysroots:
        candidate = sysroot / "lib/rustlib" / host_target() / "bin/llvm-objcopy"
        if candidate.is_file():
            return candidate
    raise ReleaseError("llvm-objcopy is missing; install rustup component llvm-tools-preview for the selected toolchain.")


def venv_python(directory):
    python = directory / "bin/python"
    if not python.exists():
        run([sys.executable, "-m", "venv", directory])
    return python


def license_notices(source, env):
    from notices import NoticesError, collect
    metadata = json.loads(run(["cargo", "metadata", "--format-version", "1", "--locked",
                               "--filter-platform", host_target()], source, env, True))
    try:
        return collect(metadata, source.parent / "license-cache", clean_env(env))
    except NoticesError as error:
        raise ReleaseError(str(error)) from None


def package_legal(source, notices):
    directories = [source / "bindings" / name for name in ("python", "node", "javascript", "rust", "cli")]
    directories += [source / "bindings" / binding / "npm" / name
                    for binding in ("node", "cli") for name in TARGETS.values()]
    for directory in directories:
        shutil.copy2(source / "LICENSE.txt", directory / "LICENSE.txt")
        (directory / "THIRD_PARTY_NOTICES.txt").write_text(notices)
        if directory.parent.name == "npm":
            shutil.copy2(directory.parent.parent / "README.md", directory / "README.md")


def npm_notices(directory):
    notices = []
    lock = read_json(directory / "package-lock.json")
    # Package artifacts bundle only our emitted JS. React is an external peer dependency.
    # Preserve notices for every installed dependency too, including build-tool outputs.
    for name in sorted(lock.get("packages", {})):
        path = directory / name
        if not name or not path.is_dir():
            continue
        for item in sorted(path.iterdir()):
            if item.is_file() and item.name.upper().startswith(("LICENSE", "LICENCE", "NOTICE", "COPYING")):
                notices.append(f"\n{name}\n{item.read_text(errors='replace')}\n")
    return "\n".join(notices)


def npm_pack(directory, output):
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads(run(["npm", "pack", "--json", "--ignore-scripts", "--pack-destination", output], directory, capture=True))
    return output / result[0]["filename"]


def host_target():
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine == "arm64":
        return "aarch64-apple-darwin"
    if sys.platform == "linux" and machine in {"aarch64", "arm64"}:
        return "aarch64-unknown-linux-gnu"
    if sys.platform == "linux" and machine in {"x86_64", "amd64"}:
        return "x86_64-unknown-linux-gnu"
    raise ReleaseError("Unsupported build host. Use macOS ARM64 for the complete local matrix.")


def ensure_console(source):
    if not (source / "web/dist/index.html").exists():
        run(["bash", "tools/bindings/verify-web.sh"], source)


def standalone_names(target, version):
    return {f"irongraph-{version}.tgz", f"irongraph-cli-{TARGETS[target]}-{version}.tgz",
            f"irongraph-{version}-{target}.tar.gz"}


def build_standalone(source, output, target, config=None):
    """Build the server and MCP executable using the same native build cache as the SDKs."""
    if target != host_target():
        raise ReleaseError("Standalone builds must execute on their target architecture.")
    config = config or {}
    output.mkdir(parents=True, exist_ok=True)
    ensure_console(source)
    env = tool_environment(source, target)
    if sys.platform == "darwin":
        env["MACOSX_DEPLOYMENT_TARGET"] = config.get("MACOSX_DEPLOYMENT_TARGET", "15.0")
    run(["rustup", "component", "add", "rust-docs", "llvm-tools-preview"], source)
    run(["cargo", "build", "--release", "--locked", "--target", target,
         "-p", "irongraph", "--bin", "irongraph", "-p", "irongraph-mcp", "--bin", "irongraph-mcp"], source, env)
    notices, _ = license_notices(source, env)
    package_legal(source, notices)
    cli = source / "bindings/cli"
    native = cli / "npm" / TARGETS[target]
    (native / "bin").mkdir(exist_ok=True)
    objcopy = llvm_objcopy(source)
    for name in ("irongraph", "irongraph-mcp"):
        run([objcopy, "--strip-all", Path(env["CARGO_TARGET_DIR"]) / target / "release" / name,
             native / "bin" / name], source)
        (native / "bin" / name).chmod(0o755)
        # Mach-O stripping changes the bytes covered by Rust's ad-hoc signature.
        if sys.platform == "darwin":
            run(["codesign", "--force", "--sign", "-", native / "bin" / name], source)
    version = current_version(source)
    for package in (npm_pack(native, output), npm_pack(cli, output)):
        audit_archive(package, "npm", version)
    archive = output / f"irongraph-{version}-{target}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in sorted(PUBLIC_FILES | {"bin/irongraph", "bin/irongraph-mcp"}):
            bundle.add(native / name, arcname=f"irongraph-{version}/{name}", recursive=False)
    audit_archive(archive, "standalone", version)


def verify_standalone(source, output, target):
    """Exercise the installed launcher, never a checkout-relative executable."""
    if target != host_target():
        raise ReleaseError("Standalone packages must be verified on their target architecture.")
    version = current_version(source)
    names = standalone_names(target, version)
    for name in sorted(names):
        audit_archive(output / name, "npm" if name.endswith(".tgz") else "standalone", version)
    # The direct-download bundle must contain the exact executables exercised through npm.
    native_package = output / f"irongraph-cli-{TARGETS[target]}-{version}.tgz"
    bundle = output / f"irongraph-{version}-{target}.tar.gz"
    binaries = {PurePosixPath(name).name: hashlib.sha256(data).hexdigest()
                for name, data in archive_files(native_package) if "/bin/" in name}
    bundled = {PurePosixPath(name).name: hashlib.sha256(data).hexdigest()
               for name, data in archive_files(bundle) if "/bin/" in name}
    if binaries != bundled:
        raise ReleaseError("Standalone archive executables differ from the installed native package.")
    with tempfile.TemporaryDirectory(prefix="irongraph-standalone-consumer-") as temporary:
        consumer = Path(temporary)
        write_json(consumer / "package.json", {"name": "irongraph-standalone-check", "private": True})
        run(["npm", "install", "--ignore-scripts", "--no-audit", "--no-fund",
             *[output / name for name in sorted(names) if name.endswith(".tgz")]], consumer)
        for name in ("irongraph", "irongraph-mcp"):
            installed = consumer / "node_modules/@irongraph" / ("cli-" + TARGETS[target]) / "bin" / name
            actual = run([installed, "--version"], consumer, capture=True).strip()
            if actual != f"{name} {version}":
                raise ReleaseError(f"Installed standalone executable version mismatch: {name}")
        run(["node", source / "tools/release/verify-standalone.cjs", consumer, version], consumer)
    write_json(output / "standalone-qualification.json", {
        "target": target, "version": version, "installed_lifecycle": True,
        "artifacts": {name: sha256(output / name) for name in sorted(names)},
    })
    print(f"standalone installed verification passed: {target}", flush=True)


def build_target(source, output, target, qualify=True, config=None):
    if target != host_target():
        raise ReleaseError("Native qualification must execute on its target architecture (Docker emulation is supported).")
    config = config or {}
    output.mkdir(parents=True, exist_ok=True)
    ensure_console(source)
    env = tool_environment(source, target)
    if sys.platform == "darwin":
        env["MACOSX_DEPLOYMENT_TARGET"] = config.get("MACOSX_DEPLOYMENT_TARGET", "15.0")
    python = venv_python(source.parent / "python-tools" / target)
    run(["rustup", "component", "add", "rust-docs", "llvm-tools-preview"], source)
    run([python, "-m", "pip", "install", "maturin>=1.15,<2", "twine>=6,<7"], source)
    env["PYO3_PYTHON"] = str(python)
    run(["cargo", "build", "--release", "--locked", "--target", target, "-p", "irongraph-ffi"], source, env)
    notices, inventory = license_notices(source, env)
    package_legal(source, notices)
    run(["npm", "ci", "--ignore-scripts"], source / "bindings/node")
    notices += npm_notices(source / "bindings/node")
    package_legal(source, notices)
    (output / "THIRD_PARTY_NOTICES.txt").write_text(notices)
    native_dir = Path(env["CARGO_TARGET_DIR"]) / target / "release"
    archive = output / f"libirongraph_ffi-{target}.a"
    objcopy = llvm_objcopy(source)
    # Static archives otherwise retain dependency LLVM bitcode in addition to machine code.
    run([objcopy, "--strip-debug", "--remove-section=__LLVM,__bitcode", "--remove-section=__LLVM,__cmdline",
         "--remove-section=.llvmbc", "--remove-section=.llvmcmd", native_dir / "libirongraph_ffi.a", archive], source)
    # NAPI's build step is intentional, while publish/install lifecycle hooks remain disabled.
    run(["npx", "--no-install", "napi", "build", "--platform", "--release", "--target", target,
         "--dts", "generated.d.ts"], source / "bindings/node", env)
    binary_name = f"irongraph.{TARGETS[target]}.node"
    node_dir = source / "bindings/node"
    if sys.platform == "darwin":
        # Mach-O's LC_ID_DYLIB otherwise records the absolute local build path.
        run(["install_name_tool", "-id", f"@rpath/{binary_name}", node_dir / binary_name], source)
        run(["codesign", "--force", "--sign", "-", node_dir / binary_name], source)
    shutil.copy2(node_dir / binary_name, node_dir / "npm" / TARGETS[target] / binary_name)
    wheels = output / "wheels"
    wheels.mkdir(exist_ok=True)
    run([python, "-m", "maturin", "build", "--release", "--locked", "--target", target,
         "--manifest-path", "bindings/python/Cargo.toml", "--out", wheels,
         *([] if sys.platform == "darwin" else ["--compatibility", "manylinux_2_28"])], source, env)
    npm_pack(node_dir / "npm" / TARGETS[target], output)
    npm_pack(node_dir, output)
    build_standalone(source, output, target, config)
    qualify_target(source, output, target, qualify, config)


def qualify_target(source, output, target, qualify=True, config=None):
    """Verify existing artifacts without recompiling the already built database."""
    if target != host_target():
        raise ReleaseError("Installed packages must be verified on their target architecture.")
    config = config or {}
    verify_standalone(source, output, target)
    env = tool_environment(source, target)
    if sys.platform == "darwin":
        env["MACOSX_DEPLOYMENT_TARGET"] = config.get("MACOSX_DEPLOYMENT_TARGET", "15.0")
    python = venv_python(source.parent / "python-tools" / target)
    native_dir = source.parent / "qualified-native" / target
    native_dir.mkdir(parents=True, exist_ok=True)
    archive = output / f"libirongraph_ffi-{target}.a"
    shutil.copy2(archive, native_dir / "libirongraph_ffi.a")
    node_dir = source / "bindings/node"
    binary_name = f"irongraph.{TARGETS[target]}.node"
    version = current_version(source)
    root_package = output / f"irongraph-node-{version}.tgz"
    native_package = output / f"irongraph-node-{TARGETS[target]}-{version}.tgz"
    wheels = output / "wheels"
    for path in [root_package, native_package, *wheels.glob("*.whl")]:
        audit_archive(path, "wheel" if path.suffix == ".whl" else "npm", version)
    # Full tests and installation use only the staged archives, outside binding source directories.
    with tempfile.TemporaryDirectory(prefix="irongraph-consumer-") as temp:
        consumer = Path(temp)
        wheel_python = venv_python(consumer / "python")
        wheel_files = list(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise ReleaseError("Expected exactly one ABI3 wheel for each target.")
        run([python, "-m", "twine", "check", *wheel_files], source)
        run([wheel_python, "-m", "pip", "install", "--no-deps", *wheel_files], consumer)
        shutil.copy2(source / "bindings/python/test_smoke.py", consumer / "test_smoke.py")
        qualification_env = {"IRONGRAPH_QUALIFY_EMBEDDINGS": "1"} if qualify else {}
        run([wheel_python, "test_smoke.py"], consumer, qualification_env)
        if qualify and sys.platform == "darwin":
            run([wheel_python, "test_smoke.py"], consumer,
                {**qualification_env, "IRONGRAPH_QUALIFY_DEVICE": "metal"})
        write_json(consumer / "package.json", {"name": "irongraph-installed-check", "private": True})
        run(["npm", "install", "--ignore-scripts", "--no-audit", "--no-fund", root_package, native_package], consumer)
        test = (source / "bindings/node/test.cjs").read_text()
        test = test.replace("require('./index.js')", "require('@irongraph/node')")
        (consumer / "test.cjs").write_text(test)
        run(["node", "test.cjs"], consumer, qualification_env)
        if qualify and sys.platform == "darwin":
            run(["node", "test.cjs"], consumer,
                {**qualification_env, "IRONGRAPH_QUALIFY_DEVICE": "metal"})
    sdk_env = {**env, "IRONGRAPH_NATIVE_DIR": str(native_dir)}
    run(["cargo", "test", "--manifest-path", "bindings/rust/Cargo.toml"], source, sdk_env)
    if qualify:
        semantic_test = ["cargo", "test", "--manifest-path", "bindings/rust/Cargo.toml",
                         "--test", "embedded", "automatic_semantic_native_and_remote", "--", "--ignored"]
        run(semantic_test, source, sdk_env)
        if sys.platform == "darwin":
            run(semantic_test, source, {**sdk_env, "IRONGRAPH_QUALIFY_DEVICE": "metal"})
    # Record linked system libraries and toolchain versions for later support qualification.
    inspection = run(["otool", "-L", node_dir / binary_name] if sys.platform == "darwin" else
                     ["ldd", node_dir / binary_name], source, capture=True)
    write_json(output / "qualification.json", {
        "target": target, "version": current_version(source), "embedding_search": qualify,
        "os_version": platform.mac_ver()[0] if sys.platform == "darwin" else platform.release(),
        "backends": (["cpu", "metal"] if sys.platform == "darwin" else ["cpu"]) if qualify else [],
        "native_sha256": sha256(archive), "rustc": run(["rustc", "--version"], capture=True).strip(),
        "system_libraries": inspection,
        "artifacts": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                      if p.is_file() and p.name != "qualification.json"},
    })
    print(f"installed package verification passed: {target}", flush=True)


def archive_files(path):
    """Read contents without extracting paths or following links."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ReleaseError(f"Symlink in wheel: {info.filename}")
                if not info.is_dir():
                    yield info.filename, archive.read(info)
    else:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ReleaseError(f"Non-regular package entry: {member.name}")
                file = archive.extractfile(member)
                if file is None:
                    raise ReleaseError("Unreadable archive entry.")
                yield member.name, file.read()


def audit_archive(path, kind, version):
    names = []
    package_manifest = None
    for name, data in archive_files(path):
        if PRIVATE_BYTES.search(data):
            raise ReleaseError(f"Private identity or brand found in {path.name}: {name}")
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts:
            raise ReleaseError(f"Unsafe archive path: {name}")
        names.append(name)
        relative = "/".join(parts.parts[1:])
        if kind == "npm" and parts.parts[0] != "package":
            raise ReleaseError("npm archive must have a single package root.")
        if kind == "cargo":
            allowed = relative in PUBLIC_FILES | {"Cargo.toml", "Cargo.toml.orig", "Cargo.lock", "build.rs", "native-manifest.json", "src/lib.rs"}
        elif kind == "npm":
            allowed = (relative in PUBLIC_FILES | {"package.json", "index.js", "index.d.ts", "cli.cjs", "bin/irongraph", "bin/irongraph-mcp"}
                       or relative.startswith("dist/") and relative.endswith((".js", ".d.ts"))
                       or "/" not in relative and relative.startswith("irongraph.") and relative.endswith(".node"))
        elif kind == "standalone":
            allowed = (parts.parts[0] == f"irongraph-{version}"
                       and relative in PUBLIC_FILES | {"bin/irongraph", "bin/irongraph-mcp"})
        else:
            allowed = (name in {"irongraph/__init__.py", "irongraph/__init__.pyi", "irongraph/py.typed"}
                       or name.startswith("irongraph/_native") and name.endswith(".so")
                       or ".dist-info/" in name and (parts.name in PUBLIC_FILES | {"METADATA", "WHEEL", "RECORD", "INSTALLER", "entry_points.txt", "top_level.txt"})
                       or name.endswith(".dist-info/sboms/auditwheel.cdx.json")
                       or name.startswith("irongraph.libs/") and ".so" in parts.name)
        if not allowed:
            raise ReleaseError(f"Unexpected file in {path.name}: {name}")
        if kind == "cargo" and relative in {"Cargo.toml", "Cargo.toml.orig"}:
            manifest = tomllib.loads(data.decode())
            for section in ("dependencies", "build-dependencies", "dev-dependencies"):
                for dependency, value in manifest.get(section, {}).items():
                    if dependency.startswith("irongraph-") or isinstance(value, dict) and ("path" in value or "git" in value):
                        raise ReleaseError("Cargo package exposes an internal implementation dependency.")
        if parts.name == "package.json":
            manifest = json.loads(data)
            audit_npm_metadata(manifest)
            package_manifest = manifest
            if manifest.get("version") != version or manifest.get("license") != "Apache-2.0":
                raise ReleaseError("npm version/license mismatch.")
            if any(value != version for name, value in manifest.get("optionalDependencies", {}).items()
                   if name.startswith("@irongraph/")):
                raise ReleaseError("npm optional native dependencies must match the release version exactly.")
        if parts.name == "METADATA":
            metadata = email.parser.Parser().parsestr(data.decode())
            if metadata["Version"] != version or metadata["License-Expression"] != "Apache-2.0":
                raise ReleaseError("Wheel version/license mismatch.")
    if len(names) != len(set(names)):
        raise ReleaseError("Duplicate archive entries are not allowed.")
    if kind == "standalone":
        expected = {f"irongraph-{version}/{name}" for name in PUBLIC_FILES | {"bin/irongraph", "bin/irongraph-mcp"}}
        if set(names) != expected:
            raise ReleaseError("Standalone archive must contain exactly both executables and public documentation.")
    if kind == "npm":
        if package_manifest is None:
            raise ReleaseError("npm package manifest is missing.")
        package_name = package_manifest.get("name", "")
        cli_names = {f"@irongraph/cli-{suffix}" for suffix in TARGETS.values()}
        relative_names = {name.removeprefix("package/") for name in names}
        if package_name == "irongraph" or package_name in cli_names:
            if path.name != f"{package_name.removeprefix('@').replace('/', '-')}-{version}.tgz":
                raise ReleaseError("Standalone npm artifact name does not match its package metadata.")
            expected = PUBLIC_FILES | {"package.json"} | (
                {"cli.cjs"} if package_name == "irongraph" else {"bin/irongraph", "bin/irongraph-mcp"})
            lifecycle_hooks = {"preinstall", "install", "postinstall", "prepare", "prepublish", "prepublishOnly",
                               "prepack", "postpack", "publish", "postpublish"}
            if relative_names != expected or lifecycle_hooks & set(package_manifest.get("scripts", {})):
                raise ReleaseError("Standalone npm package inventory or lifecycle hooks violate the distribution boundary.")
            if package_name == "irongraph" and (
                package_manifest.get("bin") != {"irongraph": "cli.cjs"}
                or package_manifest.get("optionalDependencies") != {name: version for name in cli_names}
            ):
                raise ReleaseError("Standalone launcher must expose the CLI and exact-version native matrix.")
            if package_name in cli_names:
                suffix = package_name.removeprefix("@irongraph/cli-")
                expected_os = "darwin" if suffix == "darwin-arm64" else "linux"
                expected_cpu = "x64" if suffix == "linux-x64-gnu" else "arm64"
                if (package_manifest.get("os") != [expected_os]
                        or package_manifest.get("cpu") != [expected_cpu]
                        or expected_os == "linux" and package_manifest.get("libc") != ["glibc"]):
                    raise ReleaseError("Standalone native package platform metadata does not match its target.")
        elif relative_names & {"cli.cjs", "bin/irongraph", "bin/irongraph-mcp"}:
            raise ReleaseError("Standalone executable files are only allowed in the official CLI packages.")
    if kind in {"npm", "standalone"}:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if ("/bin/" in member.name or member.name == "package/cli.cjs") and not member.mode & 0o111:
                    raise ReleaseError(f"Standalone command is not executable: {member.name}")
    for required in PUBLIC_FILES:
        if not any(PurePosixPath(name).name == required for name in names):
            # Python renders the README into its metadata rather than including README.md.
            if kind == "wheel" and required == "README.md":
                continue
            raise ReleaseError(f"{path.name}: missing {required}.")
    if kind == "cargo" and path.stat().st_size > 10_000_000:
        raise ReleaseError("Cargo wrapper exceeds the crates.io default 10 MB limit.")
    if kind == "wheel" and path.stat().st_size > 100_000_000:
        raise ReleaseError("Wheel exceeds PyPI's default 100 MB limit; request an increase before release.")


def verify_standalone_report(directory, target, version):
    if not (directory / "standalone-qualification.json").is_file():
        raise ReleaseError(f"{target}: standalone installed lifecycle qualification required; SDK-only reports cannot be reused.")
    report = read_json(directory / "standalone-qualification.json")
    if (report.get("target") != target or report.get("version") != version
            or report.get("installed_lifecycle") is not True
            or set(report.get("artifacts", {})) != standalone_names(target, version)):
        raise ReleaseError(f"{target}: standalone installed lifecycle qualification required.")
    for name, expected in report["artifacts"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ReleaseError(f"{target}: qualified standalone artifact changed: {name}")


def verify_qualification(output, version):
    for target in TARGETS:
        directory = output / target
        report = read_json(directory / "qualification.json")
        if report["target"] != target or report["version"] != version or report["embedding_search"] is not True:
            raise ReleaseError(f"{target}: complete installed-package and embedding-search qualification required.")
        verify_standalone_report(directory, target, version)
        wheels = list((directory / "wheels").glob("*.whl"))
        required = {f"libirongraph_ffi-{target}.a", f"irongraph-node-{version}.tgz",
                    f"irongraph-node-{TARGETS[target]}-{version}.tgz", "THIRD_PARTY_NOTICES.txt"}
        required.update(str(path.relative_to(directory)) for path in wheels)
        if len(wheels) != 1 or not required.issubset(report.get("artifacts", {})):
            raise ReleaseError(f"{target}: qualification must cover every current embedded SDK artifact.")
        for name, expected in report["artifacts"].items():
            path = directory / name
            if not path.is_file() or sha256(path) != expected:
                raise ReleaseError(f"{target}: verified artifact changed: {name}")


def verify_cargo_consumer(crate, env):
    """Link and run an independent application against the exact packaged public SDK."""
    with tempfile.TemporaryDirectory(prefix="irongraph-cargo-consumer-") as temporary:
        directory = Path(temporary)
        for name, data in archive_files(crate):
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        sdk = directory / crate.name.removesuffix(".crate")
        consumer = directory / "consumer"
        (consumer / "src").mkdir(parents=True)
        (consumer / "Cargo.toml").write_text(
            '[package]\nname="irongraph-consumer-check"\nversion="0.0.0"\nedition="2024"\n'
            '[dependencies]\nirongraph-sdk={path=' + json.dumps(str(sdk)) + '}\n')
        (consumer / "src/main.rs").write_text('''use irongraph_sdk::{EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice, Query};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = EmbeddedOptions::new("data")
        .with_execution_device(ExecutionDevice::Cpu)
        .with_embedding_policy(EmbeddingPolicy::Disabled);
    let database = EmbeddedDatabase::open(options.clone())?;
    database.query(Query::new("CREATE PROJECT packaged"))?;
    database.query(Query::new("USE packaged CREATE (:Document {body: 'From the crate archive'})"))?;
    database.snapshot()?;
    database.close()?;
    let reopened = EmbeddedDatabase::open(options)?;
    let result = reopened.query(Query::new("USE packaged MATCH (d:Document) RETURN d.body"))?;
    assert_eq!(result.rows[0][0]["value"].as_str(), Some("From the crate archive"));
    reopened.close()?;
    println!("packaged Cargo consumer passed");
    Ok(())
}
''')
        run(["cargo", "run", "--quiet", "--manifest-path", consumer / "Cargo.toml"], consumer, env)


def package_all(stage, config):
    source, output = stage / "source", stage / "artifacts"
    version = current_version(source)
    verify_qualification(output, version)
    # Cargo can select any native target, so its legal bundle covers the entire matrix.
    notices = "\n".join(f"IronGraph native distribution: {target}\n\n" +
                        (output / target / "THIRD_PARTY_NOTICES.txt").read_text()
                        for target in TARGETS)
    package_legal(source, notices)
    publication = stage / "publish"
    publication.mkdir(exist_ok=True)
    expected_files = {"README.md", "LICENSE.txt"}
    manifest = {"version": version, "targets": {}}
    for target, suffix in TARGETS.items():
        directory = output / target
        archive = directory / f"libirongraph_ffi-{target}.a"
        manifest["targets"][target] = {
            "url": f"https://github.com/{config['IRONGRAPH_BINARY_REPO']}/releases/download/v{version}/{archive.name}",
            "sha256": sha256(archive),
        }
        shutil.copy2(archive, publication / archive.name)
        expected_files.add(archive.name)
        expected_npm = {f"irongraph-node-{version}.tgz", f"irongraph-node-{suffix}-{version}.tgz",
                        f"irongraph-{version}.tgz", f"irongraph-cli-{suffix}-{version}.tgz"}
        if {path.name for path in directory.glob("*.tgz")} != expected_npm:
            raise ReleaseError(f"{target}: unexpected or missing native npm release package.")
        for wheel in (directory / "wheels").glob("*.whl"):
            audit_archive(wheel, "wheel", version)
            shutil.copy2(wheel, publication / wheel.name)
            expected_files.add(wheel.name)
        for package in directory.glob("*.tgz"):
            if package.name in {f"irongraph-node-{version}.tgz", f"irongraph-{version}.tgz"} and target != "aarch64-apple-darwin":
                continue
            audit_archive(package, "npm", version)
            shutil.copy2(package, publication / package.name)
            expected_files.add(package.name)
        standalone = directory / f"irongraph-{version}-{target}.tar.gz"
        audit_archive(standalone, "standalone", version)
        shutil.copy2(standalone, publication / standalone.name)
        expected_files.add(standalone.name)
    javascript = source / "bindings/javascript"
    run(["npm", "ci", "--ignore-scripts"], javascript)
    run(["npm", "test"], javascript)
    run(["npm", "run", "build"], javascript)
    # The remote browser client carries its JavaScript notices; it contains no native engine.
    (javascript / "THIRD_PARTY_NOTICES.txt").write_text(npm_notices(javascript))
    javascript_package = npm_pack(javascript, publication)
    audit_archive(javascript_package, "npm", version)
    expected_files.add(javascript_package.name)
    sdk = source / "bindings/rust"
    write_json(sdk / "native-manifest.json", manifest)
    env = {"IRONGRAPH_NATIVE_DIR": str(stage / "qualified-native/aarch64-apple-darwin"),
           "CARGO_TARGET_DIR": str(stage / "sdk-package"),
           "RUSTUP_TOOLCHAIN": tomllib.loads((source / "rust-toolchain.toml").read_text())["toolchain"]["channel"]}
    # Package outside the private Git checkout so Cargo cannot attach private VCS provenance.
    with tempfile.TemporaryDirectory(prefix="irongraph-cargo-package-") as temporary:
        public_sdk = Path(temporary) / "irongraph-sdk"
        shutil.copytree(sdk, public_sdk, ignore=shutil.ignore_patterns("target", "native", ".git"))
        run(["cargo", "package", "--manifest-path", public_sdk / "Cargo.toml"], public_sdk, env)
    crate = stage / "sdk-package/package" / f"irongraph-sdk-{version}.crate"
    audit_archive(crate, "cargo", version)
    verify_cargo_consumer(crate, env)
    shutil.copy2(crate, publication / crate.name)
    expected_files.add(crate.name)
    shutil.copy2(source / "README.md", publication / "README.md")
    shutil.copy2(source / "LICENSE.txt", publication / "LICENSE.txt")
    if {p.name for p in publication.iterdir()} != expected_files:
        raise ReleaseError("Publication directory has unexpected or missing files; no files will be uploaded.")
    return {p.name: sha256(p) for p in publication.iterdir() if p.is_file()}


def github(config, route, method="GET", value=None):
    request = urllib.request.Request("https://api.github.com" + route,
        data=json.dumps(value).encode() if value is not None else None, method=method,
        headers={"Authorization": "Bearer " + config["GITHUB_TOKEN"], "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "IronGraph-release"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise ReleaseError(f"GitHub {method} request failed (HTTP {error.code}); check repository permissions.") from None


def validate_binary_repo(config, allow_missing=False):
    repo = config.get("IRONGRAPH_BINARY_REPO", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ReleaseError("Set IRONGRAPH_BINARY_REPO to the public source repository OWNER/REPO.")
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True).strip()
    remote_name = re.sub(r"\.git$", "", re.sub(r".*github\.com[:/]", "", remote))
    if remote_name.lower() != repo.lower():
        raise ReleaseError("GitHub releases must be created in the configured source repository.")
    metadata = github(config, f"/repos/{repo}")
    if metadata is None and allow_missing:
        return None
    if not metadata or metadata.get("private"):
        raise ReleaseError("The source repository must exist and be public before publishing packages.")
    return metadata


def prepare_binary_repository(config, publication):
    del publication
    validate_binary_repo(config)


def upload_asset(config, release, path):
    existing = next((asset for asset in release.get("assets", []) if asset["name"] == path.name), None)
    digest = sha256(path)
    if existing:
        if existing.get("digest") == "sha256:" + digest and existing["size"] == path.stat().st_size:
            return
        raise ReleaseError(f"Existing GitHub asset differs or has no verifiable digest: {path.name}. Refusing overwrite.")
    parsed = urllib.parse.urlsplit(release["upload_url"].split("{")[0])
    if parsed.scheme != "https" or parsed.hostname != "uploads.github.com":
        raise ReleaseError("Unexpected GitHub upload host.")
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=300)
    try:
        connection.putrequest("POST", parsed.path + "?" + urllib.parse.urlencode({"name": path.name}))
        connection.putheader("Authorization", "Bearer " + config["GITHUB_TOKEN"])
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(path.stat().st_size))
        connection.putheader("User-Agent", "IronGraph-release")
        connection.endheaders()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        body = response.read()
        if response.status != 201:
            raise ReleaseError(f"GitHub asset upload failed (HTTP {response.status}): {path.name}")
        asset = json.loads(body)
        if asset.get("digest") != "sha256:" + digest:
            raise ReleaseError(f"GitHub checksum did not match: {path.name}")
    finally:
        connection.close()


def public_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "IronGraph-release"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise ReleaseError(f"Registry lookup failed (HTTP {error.code}).") from None


def existing_package(path, version):
    if path.suffix == ".whl":
        metadata = public_json(f"https://pypi.org/pypi/irongraph/{version}/json")
        existing = next((item for item in (metadata or {}).get("urls", []) if item["filename"] == path.name), None)
        expected = existing and existing["digests"]["sha256"]
    elif path.suffix == ".crate":
        metadata = public_json(f"https://crates.io/api/v1/crates/irongraph-sdk/{version}")
        expected = metadata and metadata["version"]["checksum"]
    else:
        with tarfile.open(path) as archive:
            metadata = json.load(archive.extractfile("package/package.json"))
        package = urllib.parse.quote(metadata["name"], safe="")
        remote = public_json(f"https://registry.npmjs.org/{package}/{version}")
        if not remote:
            return False
        audit_npm_metadata(remote)
        expected_integrity = "sha512-" + base64.b64encode(hashlib.sha512(path.read_bytes()).digest()).decode()
        if remote.get("dist", {}).get("integrity") != expected_integrity:
            raise ReleaseError(f"Published npm version differs from staged archive: {metadata['name']} {version}")
        return True
    if expected and expected != sha256(path):
        raise ReleaseError(f"Published version differs from staged archive: {path.name}")
    return bool(expected)


def audit_npm_metadata(metadata, secrets=()):
    """Check the registry manifest as well as the manifest inside the archive."""
    forbidden = {"_from", "_resolved", "_where", "_args", "_location", "_requested"}
    local_path = re.compile(r"(?:(?:^|[\s\"'=(:])(?:file:\S|/(?:Users|home|Volumes|private|tmp|var/folders)/|[A-Za-z]:[\\/])|\\\\[^\\\s]+\\)", re.I)

    def check(value):
        if isinstance(value, dict):
            if forbidden & value.keys():
                raise ReleaseError("npm metadata contains local installation fields.")
            for key, child in value.items():
                check(key)
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str):
            for _ in range(4):
                if PRIVATE_BYTES.search(value.encode()) or local_path.search(value) or any(secret and secret in value for secret in secrets):
                    raise ReleaseError("npm metadata contains private information or a local filesystem path.")
                decoded = urllib.parse.unquote(value)
                if decoded == value:
                    break
                value = decoded

    check(metadata)


def publish_npm(path, env, registry="https://registry.npmjs.org/"):
    """Publish directory metadata, retaining exactly the reviewed archive bytes."""
    path = Path(path).resolve()
    with tarfile.open(path) as archive:
        manifest = json.load(archive.extractfile("package/package.json"))
    audit_archive(path, "npm", manifest["version"])
    audit_npm_metadata(manifest, tuple(env.values()))
    for name, contents in archive_files(path):
        if PurePosixPath(name).name.lower().startswith("readme"):
            audit_npm_metadata(contents.decode(), tuple(env.values()))
    # A tarball spec makes npm inject its absolute pathname into public metadata.
    # Directory specs avoid that path, while the CLI retains browser/OTP support.
    with tempfile.TemporaryDirectory(prefix="irongraph-npm-", dir="/tmp") as temporary:
        root = Path(temporary).resolve()
        package = root / "package"
        output = root / "packed"
        output.mkdir()
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isdir():
                    continue
                destination = root / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())
                destination.chmod(member.mode & 0o777)
        isolated_env = {**env, "NPM_CONFIG_CACHE": str(root / "cache")}
        packed = json.loads(run(["npm", "pack", ".", "--json", "--ignore-scripts", "--registry", registry,
                                 "--pack-destination", output], package, isolated_env, capture=True))
        if len(packed) != 1 or Path(packed[0]["filename"]).name != packed[0]["filename"]:
            raise ReleaseError("Unexpected npm repack output.")
        if sha256(output / packed[0]["filename"]) != sha256(path):
            raise ReleaseError("npm repack differs from reviewed archive; refusing publication.")
        run(["npm", "publish", ".", "--access", "public", "--ignore-scripts", "--registry", registry],
            package, isolated_env)


def npm_publish_order(publication, version):
    parents = {f"irongraph-node-{version}.tgz", f"irongraph-{version}.tgz"}
    return sorted(publication.glob("*.tgz"), key=lambda path: (path.name in parents, path.name))


def wait_for_public_package(path, version):
    deadline = time.monotonic() + 1200
    while not existing_package(path, version):
        if time.monotonic() >= deadline:
            raise ReleaseError(f"Registry publication is still pending: {path.name}; resume after it becomes public.")
        print(f"Waiting for public registry checksum: {path.name}", flush=True)
        time.sleep(15)


def publish(stage, config, state):
    publication = stage / "publish"
    if {p.name for p in publication.iterdir()} != set(state["checksums"]):
        raise ReleaseError("Publication directory inventory changed; refusing unreviewed uploads.")
    for name, digest in state["checksums"].items():
        if (publication / name).is_symlink() or not (publication / name).is_file() or sha256(publication / name) != digest:
            raise ReleaseError("Staged publication artifacts changed; refusing upload.")
    prepare_binary_repository(config, publication)
    repo = config["IRONGRAPH_BINARY_REPO"]
    base = f"/repos/{repo}"
    tag = "v" + state["version"]
    release = github(config, base + f"/releases/tags/{tag}")
    if release is None:
        # Draft releases are not found by releases/tags on every GitHub deployment.
        releases = github(config, base + "/releases?per_page=100") or []
        release = next((item for item in releases if item["tag_name"] == tag), None)
    if release is None:
        release = github(config, base + "/releases", "POST", {
            "tag_name": tag, "name": "IronGraph " + tag, "draft": True,
            "body": (publication / "README.md").read_text(),
        })
    if release is None:
        raise ReleaseError("Could not create binary release.")
    # Registry packages are published to their registries, not duplicated on GitHub.
    # The SDK downloads one matching native library at build time.
    github_assets = {f"libirongraph_ffi-{target}.a" for target in TARGETS}
    if not github_assets.issubset(state["checksums"]):
        raise ReleaseError("Missing required GitHub binary assets.")
    if {asset["name"] for asset in release.get("assets", [])} - github_assets:
        raise ReleaseError("GitHub release contains assets outside the three required native libraries.")
    for name in sorted(github_assets):
        upload_asset(config, release, publication / name)
    if release.get("draft"):
        github(config, base + f"/releases/{release['id']}", "PATCH", {"draft": False})
    # Cargo's build verifier now has public access to the immutable native artifacts.
    python = venv_python(stage / "publish-tools")
    run([python, "-m", "pip", "install", "twine>=6,<7"])
    npmrc = stage / "publish.npmrc"
    fd = os.open(npmrc, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write("registry=https://registry.npmjs.org/\n@irongraph:registry=https://registry.npmjs.org/\n//registry.npmjs.org/:_authToken=${NPM_TOKEN}\n")
    try:
        # Native npm children must exist before the root optional dependencies are published.
        packages = npm_publish_order(publication, state["version"])
        packages += sorted(publication.glob("*.whl"))
        packages += sorted(publication.glob("*.crate"))
        for path in packages:
            if existing_package(path, state["version"]):
                print(f"Already published, checksum verified: {path.name}", flush=True)
                continue
            if path.suffix == ".tgz":
                publish_npm(path, {"NPM_TOKEN": config["NPM_TOKEN"], "NPM_CONFIG_USERCONFIG": str(npmrc)})
            elif path.suffix == ".whl":
                run([python, "-m", "twine", "upload", "--non-interactive", "--disable-progress-bar", path], stage,
                    {"TWINE_USERNAME": "__token__", "TWINE_PASSWORD": config["MATURIN_PYPI_TOKEN"], "TWINE_REPOSITORY_URL": "https://upload.pypi.org/legacy/"})
            else:
                # Metadata and upload bytes both come from the exact reviewed archive.
                from cargo_upload import CargoPublishError, publish_crate
                try:
                    publish_crate(path, config["CARGO_REGISTRY_TOKEN"])
                except CargoPublishError as error:
                    raise ReleaseError(str(error)) from None
            wait_for_public_package(path, state["version"])
            state.setdefault("published", []).append(path.name)
            write_json(stage / "state.json", state)
    finally:
        npmrc.unlink(missing_ok=True)


def prerequisites(config, upload):
    if config.get("MACOSX_DEPLOYMENT_TARGET", "15.0") != "15.0":
        raise ReleaseError("This release targets macOS 15.0; change the support matrix and package descriptions together to change that floor.")
    if host_target() != "aarch64-apple-darwin":
        raise ReleaseError("Run the complete release on macOS ARM64; Linux targets build in local Docker.")
    for command in ("cargo", "rustup", "node", "npm", "docker", "git", "xcrun", "rg"):
        if not shutil.which(command):
            raise ReleaseError(f"Install required local tool: {command}")
    run(["xcrun", "--find", "clang"], capture=True)
    run(["docker", "info", "--format", "{{.OSType}}"], capture=True)
    if upload:
        missing = [key for key in SECRETS if not config.get(key)]
        if missing:
            raise ReleaseError("Fill .env.publish: " + ", ".join(missing))
        if config.get("IRONGRAPH_RELEASE_REVIEWED") != "yes":
            raise ReleaseError("Review LICENSE.txt and package READMEs, then set IRONGRAPH_RELEASE_REVIEWED=yes.")
        validate_binary_repo(config, allow_missing=True)
    elif not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", config.get("IRONGRAPH_BINARY_REPO", "")):
        raise ReleaseError("Set IRONGRAPH_BINARY_REPO for the native artifact URLs before packaging.")


def build_matrix(stage, config):
    source, output = stage / "source", stage / "artifacts"
    cache, shared_cache = stage / "license-cache", ROOT / "target/release-license-cache"
    cache.mkdir(parents=True, exist_ok=True)
    if shared_cache.exists():
        shutil.copytree(shared_cache, cache, dirs_exist_ok=True)
    ensure_console(source)
    host_env = tool_environment(source, "aarch64-apple-darwin")
    if not (stage / "rust-verified.json").exists():
        run(["node", "--test", "bindings/cli/test/lifecycle.test.cjs"], source)
        run([sys.executable, "tools/bindings/verify-ports.py"], source)
        run(["bash", "tools/bindings/verify-rust.sh"], source, host_env)
        write_json(stage / "rust-verified.json", {"version": current_version(source)})
    for target, suffix in TARGETS.items():
        directory = output / target
        if (directory / "qualification.json").exists():
            verify_standalone_report(directory, target, current_version(source))
            print(f"Reusing completed target: {target}", flush=True)
            continue
        if target == "aarch64-apple-darwin":
            build_target(source, directory, target, config=config)
            shutil.copytree(cache, shared_cache, dirs_exist_ok=True)
            continue
        arch = "arm64" if target.startswith("aarch64") else "amd64"
        image = config.get("IRONGRAPH_MANYLINUX_" + arch.upper(),
                           "quay.io/pypa/manylinux_2_28_" + ("aarch64" if arch == "arm64" else "x86_64") + ":latest")
        if not re.fullmatch(r"[A-Za-z0-9./_:@-]+", image):
            raise ReleaseError("Invalid manylinux image reference.")
        tag = "irongraph-release-" + arch
        run(["docker", "build", "--platform", "linux/" + arch, "--build-arg", "MANYLINUX_IMAGE=" + image,
             "--tag", tag, source / "tools/release"])
        # Containers receive a fresh source copy; host node_modules/venvs never cross architectures.
        container_stage = stage / ("linux-" + arch)
        container_source = container_stage / "source"
        if not container_source.exists():
            partial = container_stage / "source.partial"
            if partial.is_symlink():
                raise ReleaseError("Refusing symlinked container staging directory.")
            if partial.exists():
                shutil.rmtree(partial)
            shutil.copytree(source, partial, ignore=shutil.ignore_patterns("node_modules", "dist", "target", "*.node", "generated.d.ts"))
            partial.rename(container_source)
        directory.mkdir(parents=True, exist_ok=True)
        run(["docker", "run", "--rm", "--platform", "linux/" + arch,
             "--env", "CARGO_BUILD_JOBS=2",
             "--mount", f"type=bind,source={container_stage},target=/build",
             "--mount", f"type=bind,source={directory},target=/out",
             "--mount", f"type=bind,source={cache},target=/build/license-cache",
             "--mount", "type=volume,source=irongraph-release-models,target=/root/.irongraph/models",
             "--mount", "type=volume,source=irongraph-release-registry,target=/root/.cargo/registry",
             "--workdir", "/build/source", tag, "--source", "/build/source", "--output", "/out", "--target", target])
        shutil.copytree(cache, shared_cache, dirs_exist_ok=True)


def release(args, config):
    version = (args.resume.removeprefix("v") if args.resume else
               next_version(current_version(), args.bump) if args.bump else current_version())
    if not VERSION_RE.fullmatch(version):
        raise ReleaseError("Use vMAJOR.MINOR.PATCH for --resume.")
    if args.dry_run:
        print(f"Plan: {current_version()} -> v{version}; local macOS ARM64 + Docker Linux ARM64/AMD64; "
              "wheels -> PyPI; embedded/React/standalone launcher and native executables -> npm; "
              "wrapper -> crates.io; only the three native SDK libraries -> GitHub release.\n"
              "No files changed, builds started, credentials inspected remotely, or uploads performed.")
        return
    if args.check:
        prerequisites(config, not args.build_only)
        print("release prerequisites passed")
        return
    prerequisites(config, not args.build_only)
    base = ROOT / "target/releases"
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ReleaseError("Another local release is running.") from None
        stage = base / ("v" + version)
        if not args.resume:
            pending = [p.parent.name for p in base.glob("v*/state.json") if not read_json(p).get("complete")]
            if pending:
                raise ReleaseError("An unfinished release exists; use ./release.sh --resume " + pending[0])
            dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True)
            if dirty.strip() and not args.build_only:
                raise ReleaseError("Commit the reviewed source changes before releasing; versioning uses a clean snapshot.")
            if stage.exists():
                raise ReleaseError("Release stage already exists; use --resume.")
            stage.mkdir()
            state = {"version": version, "source_fingerprint": fingerprint(), "complete": False, "staged": False,
                     "binary_repo": config["IRONGRAPH_BINARY_REPO"]}
            write_json(stage / "state.json", state)
        else:
            if not (stage / "state.json").exists():
                raise ReleaseError("No staged release with that version.")
            state = read_json(stage / "state.json")
            if state.get("complete"):
                print(f"Release v{version} is already complete.")
                return
            if state["binary_repo"] != config["IRONGRAPH_BINARY_REPO"]:
                raise ReleaseError("Cannot change binary repository while resuming a release.")
            if state.get("uploaded"):
                finalize_versions(stage, state)
                print(f"Published v{version}; local version finalization completed.")
                return
            if state["source_fingerprint"] != fingerprint():
                raise ReleaseError("Source changed since staging; restore the reviewed release source before resuming.")
        finish_staging(stage, state)
        if not state.get("checksums"):
            build_matrix(stage, config)
            state["checksums"] = package_all(stage, config)
            write_json(stage / "state.json", state)
        if args.build_only:
            print(f"Built and verified v{version}. Review {stage / 'publish'}; publish with ./release.sh --publish --resume v{version}")
            return
        publish(stage, config, state)
        state["uploaded"] = True
        write_json(stage / "state.json", state)
        finalize_versions(stage, state)
        print(f"Published v{version}. Local package versions updated; review and commit those version changes.")


def main():
    parser = argparse.ArgumentParser(description="Build and verify IronGraph locally. Publishing requires an explicit --publish flag.")
    parser.add_argument("command", nargs="?", choices=("release", "build-target", "verify-target", "verify-local", "build-standalone", "verify-standalone"), default="release")
    version = parser.add_mutually_exclusive_group()
    version.add_argument("--bump", choices=("patch", "minor", "major"))
    version.add_argument("--resume", metavar="vMAJOR.MINOR.PATCH")
    parser.add_argument("--dry-run", action="store_true", help="Describe next release without changes or uploads.")
    parser.add_argument("--check", action="store_true", help="Check local tools and publishing prerequisites only.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", dest="build_only", action="store_false", help="Publish the reviewed artifacts after every verification passes.")
    mode.add_argument("--build-only", dest="build_only", action="store_true", help="Build and verify; leave all registries untouched (default).")
    parser.set_defaults(build_only=True)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", choices=tuple(TARGETS))
    args = parser.parse_args()
    if args.command in {"build-target", "verify-target", "build-standalone", "verify-standalone"}:
        if not args.output or not args.target:
            parser.error(f"{args.command} needs --output and --target")
        operation = {"build-target": build_target, "verify-target": qualify_target,
                     "build-standalone": build_standalone, "verify-standalone": verify_standalone}[args.command]
        operation(args.source.resolve(), args.output.resolve(), args.target)
    elif args.command == "verify-local":
        stage = ROOT / "target/release-local-verification" / fingerprint()[:16]
        # Persistent isolated staging supports expensive native builds without deleting shared work.
        if not (stage / "source").exists():
            stage_source(stage / "source", current_version())
        build_target(stage / "source", stage / "artifacts", host_target(), qualify=False)
    else:
        config = load_config(ROOT / ".env.publish")
        if not config.get("IRONGRAPH_BINARY_REPO"):
            remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True).strip()
            match = re.fullmatch(r"(?:https://github.com/|git@github.com:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?", remote)
            if match:
                config["IRONGRAPH_BINARY_REPO"] = match[1]
        release(args, config)


if __name__ == "__main__":
    try:
        main()
    except (ReleaseError, OSError, ValueError) as error:
        print(f"Release stopped: {error}", file=sys.stderr)
        sys.exit(1)
