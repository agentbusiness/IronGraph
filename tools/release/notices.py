"""Collect attributable license texts; this does not approve their distribution terms."""

import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tomllib
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


class NoticesError(RuntimeError):
    pass


ROOT_PACKAGES = {"irongraph-ffi", "irongraph-node", "irongraph-python"}
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
LICENSE_NAME = re.compile(r"^(?:unlicense|licen[cs]e|copying|notice|copyright)(?:$|[._-])", re.I)
SOURCE_SUFFIXES = {".rs", ".py", ".js", ".c", ".h", ".toml", ".json", ".cmake"}

# These exact published manifests were inspected with their upstream release trees. They
# declare the licenses below but contain no separate license file. This is deliberately
# version- and content-scoped, not a blanket SPDX permission rule for new dependencies.
DECLARATIONS = {
    ("block", "0.1.6"): ("Cargo.toml", "24df2f33b48c9e756a4f864f882c1ab07a159b6c2b790f45af2e8162499b6719", "MIT"),
    ("malloc_buf", "0.0.6"): ("Cargo.toml", "d87624d9d117489bd49dbaeb70077a0721e00ac2a76eb371faec818a92da5661", "MIT"),
    ("crc32c", "0.6.8"): ("Cargo.toml.orig", "99342c77e539bf3d549262dadd7d7ed646ce309b6dabde67a5a35b7c0faaca31", "Apache-2.0"),
    ("usize_cast", "1.1.0"): ("Cargo.toml.orig", "ece3b36fb104424fb82357383277aea2e5d9f757e965a64cc15bccffc157d585", "Apache-2.0"),
}
SPDX_COMMIT = "d46e94e2c78ceede1cfc63cfa0396472d2798d4c"  # SPDX license-list-data v3.27.0
SPDX_TEXT_SHA256 = {
    "MIT": "b05785f9f18e6716bab63424b11454513b9943a222595b70411009202fc592b5",
    "Apache-2.0": "074e6e32c86a4c0ef8b3ed25b721ca23aca83df277cd88106ef7177c354615ff",
    "Zlib": "bfb1112d49db5b1daecdfef24bd7e2f3ea0bafb33aa67aa0ab51e2bf8407c03d",
}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _license_name(path):
    return bool(LICENSE_NAME.match(path.name)) and path.suffix.lower() not in SOURCE_SUFFIXES


def _download(url):
    request = Request(url, headers={"User-Agent": "IronGraph-release-notices", "Accept": "application/vnd.github+json"})
    try:
        with urlopen(request, timeout=30) as response:
            final = urlsplit(response.url)
            if final.scheme != "https" or final.hostname not in {"api.github.com", "raw.githubusercontent.com"}:
                raise NoticesError("Upstream license response left the official HTTPS source.")
            data = response.read(MAX_DOWNLOAD_BYTES + 1)
    except OSError as error:
        raise NoticesError(f"Cannot retrieve pinned upstream license resource {url}: {error}") from error
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise NoticesError(f"Upstream license resource exceeds the bounded download size: {url}")
    return data


def _cached(url, cache, git_blob=None):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (_digest(url.encode()) + ".json")
    if path.exists():
        try:
            record = json.loads(path.read_text())
            data = base64.b64decode(record["body"], validate=True)
            if record["url"] != url or record["sha256"] != _digest(data):
                raise ValueError("checksum mismatch")
        except (ValueError, KeyError, TypeError) as error:
            raise NoticesError(f"Pinned notices cache is corrupt: {path.name}") from error
    else:
        data = _download(url)
        record = {"url": url, "sha256": _digest(data), "body": base64.b64encode(data).decode()}
    if git_blob:
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != git_blob:
            raise NoticesError(f"License bytes differ from the pinned Git tree blob: {url}")
    if not path.exists():
        # One collector owns its per-target cache; incomplete writes never become valid entries.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(path)
    return data


def _selected_packages(metadata):
    packages = {package["id"]: package for package in metadata["packages"]}
    roots = {identifier for identifier, package in packages.items() if package["name"] in ROOT_PACKAGES}
    resolve = metadata.get("resolve")
    if not roots or not resolve:
        return list(packages.values())
    nodes = {node["id"]: node for node in resolve["nodes"]}
    pending, seen = list(roots), set()
    while pending:
        identifier = pending.pop()
        if identifier in seen:
            continue
        seen.add(identifier)
        if identifier not in nodes:
            raise NoticesError(f"Cargo dependency resolution is incomplete for {identifier}")
        for dependency in nodes[identifier].get("deps", []):
            kinds = dependency.get("dep_kinds", [{}])
            if any(kind.get("kind") != "dev" for kind in kinds):
                pending.append(dependency["pkg"])
    return [packages[identifier] for identifier in seen]


def _relative(value):
    path = PurePosixPath(value or "")
    if path.is_absolute() or ".." in path.parts or "\\" in str(path):
        raise NoticesError("Invalid path in published package VCS metadata.")
    return path


def _upstream(package, directory, cache):
    repository = urlsplit(package.get("repository") or "")
    # Repository metadata may identify a subtree; the Cargo VCS path selects the exact package.
    parts = repository.path.strip("/").split("/")
    if repository.hostname != "github.com" or len(parts) < 2 or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts[:2]):
        raise NoticesError("Missing license text has no supported official GitHub repository provenance.")
    owner, repo = parts[:2]
    repo = repo.removesuffix(".git")
    vcs_path = directory / ".cargo_vcs_info.json"
    if not vcs_path.is_file():
        raise NoticesError("Missing license text has no commit-pinned .cargo_vcs_info.json; supply verified upstream provenance before publishing.")
    vcs = json.loads(vcs_path.read_text())
    commit = vcs.get("git", {}).get("sha1", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise NoticesError("Published package VCS metadata has no full Git commit SHA.")
    package_path = _relative(vcs.get("path_in_vcs", ""))
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{commit}?recursive=1"
    tree = json.loads(_cached(tree_url, cache))
    if tree.get("truncated") or not isinstance(tree.get("tree"), list):
        raise NoticesError(f"Pinned repository tree is incomplete: {owner}/{repo}@{commit}")
    ancestors = {package_path, *package_path.parents}
    candidates = []
    for item in tree["tree"]:
        path = _relative(item.get("path", ""))
        if item.get("type") == "blob" and path.parent in ancestors and _license_name(path):
            if not re.fullmatch(r"[0-9a-f]{40}", item.get("sha", "")):
                raise NoticesError("Pinned license tree entry is missing its blob checksum.")
            candidates.append((path, item["sha"]))
    if not candidates:
        raise NoticesError(f"No license text exists at the published package path or its ancestors in {owner}/{repo}@{commit}.")
    result = []
    for path, blob in sorted(candidates):
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{quote(str(path), safe='/')}"
        data = _cached(url, cache, git_blob=blob)
        result.append(_text_record(str(path), url, data, git_commit=commit, git_blob=blob))
    return result


class _PlainHtml(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"br", "p", "div", "li", "h1", "h2", "h3", "pre", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "div", "li", "h1", "h2", "h3", "pre", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _text_record(name, source, data, **provenance):
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise NoticesError(f"License resource is not valid UTF-8; review its encoding: {source}") from error
    if not text.strip():
        raise NoticesError(f"License resource is empty: {source}")
    if name.lower().endswith((".html", ".htm")):
        parser = _PlainHtml()
        parser.feed(text)
        text = "".join(parser.parts)
    return {"path": name, "source": source, "sha256": _digest(data), "text": text, **provenance}


def _local(package, directory):
    checksum_path = directory / ".cargo-checksum.json"
    checksum = json.loads(checksum_path.read_text()) if checksum_path.is_file() else {}
    files = checksum.get("files")
    paths = {path for path in directory.rglob("*") if path.is_file() and not path.is_symlink() and _license_name(path)}
    explicit = package.get("license_file")
    if explicit:
        explicit = Path(explicit)
        path = explicit if explicit.is_absolute() else directory / explicit
        if path.is_file() and path.resolve().is_relative_to(directory.resolve()):
            paths.add(path)
    records = []
    for path in sorted(paths):
        relative = path.relative_to(directory).as_posix()
        if files is not None and relative not in files:
            # Cargo caches can contain generated build artifacts; they are not registry sources.
            continue
        data = path.read_bytes()
        if files is not None and files[relative] != _digest(data):
            raise NoticesError(f"Registry license checksum mismatch: {package['name']} {relative}")
        source = f"{package['source']}#{package['name']}@{package['version']}/{relative}"
        records.append(_text_record(relative, source, data))
    return records, checksum.get("package")


def _standard_terms(license_id, cache):
    url = f"https://raw.githubusercontent.com/spdx/license-list-data/{SPDX_COMMIT}/text/{license_id}.txt"
    data = _cached(url, cache)
    if _digest(data) != SPDX_TEXT_SHA256[license_id]:
        raise NoticesError(f"Canonical {license_id} text differs from its pinned checksum.")
    record = _text_record(license_id + ".txt", url, data, git_commit=SPDX_COMMIT)
    if license_id == "MIT":
        # SPDX's generic template is not an attribution from this upstream author. Preserve
        # the complete permission/warranty terms and identify the published author separately.
        record["text"] = record["text"].replace("Copyright (c) <year> <copyright holders>\n\n", "", 1)
        record["text_transform"] = "Removed generic SPDX copyright placeholder; no copyright year or holder was invented."
    return record


def _declared_terms(package, directory, cache):
    rule = DECLARATIONS.get((package["name"], package["version"]))
    if rule is None:
        return None
    filename, expected, selected = rule
    data = (directory / filename).read_bytes()
    if _digest(data) != expected:
        raise NoticesError("Inspected upstream license declaration changed; review the published manifest before packaging.")
    declaration = tomllib.loads(data.decode())["package"]
    if declaration["name"] != package["name"] or declaration["version"] != package["version"] or declaration["license"] != package["license"]:
        raise NoticesError("Cargo metadata differs from the inspected upstream license declaration.")
    note = ("The published package declares the license shown here and supplies no separate license file. "
            "Published author names are preserved as attribution; no copyright year or additional copyright statement is inferred. "
            "Canonical terms are reproduced from the pinned SPDX license text. Review this declaration-based attribution with the release notices.")
    record = _text_record(filename, f"{package['source']}#{package['name']}@{package['version']}/{filename}", data)
    record["text"] = "\n".join([
        f"Published package: {declaration['name']} {declaration['version']}",
        "Published authors: " + "; ".join(declaration.get("authors", [])),
        "Declared license: " + declaration["license"],
        "Selected license: " + selected,
        note,
    ])
    record["text_transform"] = "Extracted the exact package identity, author list, and license declaration from the published manifest."
    records = [record, _standard_terms(selected, cache)]
    if package["name"] == "crc32c":
        # The crate retains a separate attribution for its translated zlib combine function.
        source = directory / "src/combine.rs"
        original = source.read_bytes()
        lines = original.decode().splitlines()
        header = []
        for line in lines:
            if not line.startswith("//!"):
                break
            header.append(line.removeprefix("//!").removeprefix(" "))
        if not any("Copyright (C) 1995-2006, 2010, 2011, 2012, 2016 Mark Adler" in line for line in header):
            raise NoticesError("crc32c's original zlib attribution is missing; review before packaging.")
        attribution = _text_record("src/combine.rs notice", f"{package['source']}#crc32c@{package['version']}/src/combine.rs", original)
        attribution["text"] = "\n".join(header)
        attribution["text_transform"] = "Preserved the complete original leading attribution comment, removing only Rust comment markers."
        records.extend([attribution, _standard_terms("Zlib", cache)])
    return records, {"selected_license": selected, "attribution_note": note, "declaration_based": True}


def _stdlib(env=None):
    try:
        identity = subprocess.check_output(["rustc", "--version", "--verbose"], text=True, env=env)
        sysroot = Path(subprocess.check_output(["rustc", "--print", "sysroot"], text=True, env=env).strip())
    except (OSError, subprocess.CalledProcessError) as error:
        raise NoticesError("Cannot identify the Rust toolchain for standard-library notices.") from error
    fields = dict(line.split(": ", 1) for line in identity.splitlines() if ": " in line)
    commit = fields.get("commit-hash", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise NoticesError("Rust toolchain does not identify a full source commit for its library notices.")
    documents = [sysroot / "share/doc/rust/COPYRIGHT-library.html",
                 sysroot / "share/doc/rustc/COPYRIGHT-library.html"]
    document = next((candidate for candidate in documents if candidate.is_file()), None)
    if document is None:
        raise NoticesError("Rust standard-library license inventory is missing; install rustup component rust-docs for the selected toolchain.")
    record = _text_record(document.name, f"rust-toolchain:{fields.get('release')}@{commit}/{document.name}", document.read_bytes(), git_commit=commit)
    return {"name": "Rust standard library and bundled library dependencies", "version": fields.get("release"),
            "license": "See component-specific license texts", "source": "https://github.com/rust-lang/rust/tree/" + commit,
            "files": [record]}


def collect(metadata: dict, cache: Path, env=None) -> tuple[str, list]:
    """Collect normal/build dependency notices from target-filtered Cargo metadata.

    All declarations and alternative licenses are retained verbatim. No license expression is
    interpreted as permission, and no generic SPDX text replaces missing upstream attribution.
    Inventory file checksums describe the original bytes, including original HTML for rust-docs.
    """
    components, failures = [], []
    for package in sorted(_selected_packages(metadata), key=lambda item: (item["name"], item["version"], item["id"])):
        if package.get("source") is None:
            continue
        directory = Path(package["manifest_path"]).parent
        try:
            records, package_checksum = _local(package, directory)
            attribution = {}
            explicit = Path(package["license_file"]).name if package.get("license_file") else None
            has_terms = any(Path(record["path"]).name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "UNLICENSE"))
                            or Path(record["path"]).name == explicit for record in records)
            if not has_terms:
                declared = _declared_terms(package, directory, cache)
                if declared is not None:
                    declared_records, attribution = declared
                    records.extend(declared_records)
                else:
                    records.extend(_upstream(package, directory, cache))
            components.append({"name": package["name"], "version": package["version"], "license": package.get("license"),
                               "source": package["source"], "package_sha256": package_checksum, "files": records, **attribution})
        except (NoticesError, OSError, ValueError, KeyError) as error:
            failures.append(f"{package['name']} {package['version']}: {error}")
    if failures:
        raise NoticesError("Cannot produce complete attributable third-party notices:\n" + "\n".join(failures))
    components.append(_stdlib(env))
    parts = ["Third-party notices for this IronGraph distribution.\n"
             "These components retain their respective licenses and copyright notices.\n"
             "This inventory preserves the upstream terms; it does not relicense these components.\n"]
    inventory = []
    for component in components:
        parts.append(f"\n{'=' * 72}\n{component['name']} {component['version']}\nDeclared license: {component.get('license') or 'see license text'}\n")
        for record in component["files"]:
            parts.append(f"\n{record['path']}\nSource: {record['source']}\nSHA256: {record['sha256']}\n\n{record['text']}\n")
        inventory.append({**component, "files": [{key: value for key, value in record.items() if key != "text"} for record in component["files"]]})
    return "\n".join(parts), inventory
