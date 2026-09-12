"""Publish reviewed .crate bytes using Cargo's documented registry upload protocol.

Protocol: https://doc.rust-lang.org/cargo/reference/registry-web-api.html#publish
No Cargo subprocess runs here; metadata comes from the normalized manifest in the archive.
"""
from __future__ import annotations

import http.client
import json
from pathlib import Path, PurePosixPath
import struct
import tarfile
import tomllib


class CargoPublishError(Exception):
    """Safe publication error whose text never includes tokens or registry response bodies."""


def _relative_path(value: str) -> str:
    if not isinstance(value, str):
        raise CargoPublishError("Cargo archive contains an invalid relative path.")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise CargoPublishError("Cargo archive contains an unsafe relative path.")
    return str(path)


def _dependencies(table: dict, target: str | None = None) -> list[dict]:
    result = []
    for section, kind in (("dependencies", "normal"), ("build-dependencies", "build"),
                          ("dev-dependencies", "dev")):
        for alias, raw in sorted(table.get(section, {}).items()):
            dependency = {"version": raw} if isinstance(raw, str) else raw
            if not isinstance(dependency, dict) or any(
                key in dependency for key in ("path", "git", "workspace")
            ) or not isinstance(dependency.get("version"), str):
                raise CargoPublishError("Cargo upload requires normalized registry dependencies with versions.")
            registry = dependency.get("registry-index")
            if dependency.get("registry") and not registry:
                raise CargoPublishError("Cargo dependency registry must be resolved to its index URL.")
            original = dependency.get("package", alias)
            result.append({
                "name": original,
                "version_req": dependency["version"],
                "features": dependency.get("features", []),
                "optional": dependency.get("optional", False),
                "default_features": dependency.get("default-features", True),
                "target": target,
                "kind": kind,
                "registry": registry,
                "explicit_name_in_toml": alias if original != alias else None,
            })
    return result


def metadata_from_archive(file) -> dict:
    """Read metadata without extracting archive members to the filesystem."""
    file.seek(0)
    with tarfile.open(fileobj=file, mode="r:gz") as archive:
        members = {}
        roots = set()
        for member in archive:
            name = _relative_path(member.name)
            if name in members:
                raise CargoPublishError("Cargo archive contains duplicate entries.")
            if not (member.isdir() or member.isfile()):
                raise CargoPublishError("Cargo archive contains a link or special file.")
            members[name] = member
            roots.add(PurePosixPath(name).parts[0])
            if len(members) > 10_000:
                raise CargoPublishError("Cargo archive contains too many entries.")
        if len(roots) != 1:
            raise CargoPublishError("Cargo archive must contain exactly one package root.")
        root = roots.pop()

        def text_file(relative: str, maximum: int = 4 * 1024 * 1024) -> str:
            name = root + "/" + _relative_path(relative)
            member = members.get(name)
            if member is None or not member.isfile() or member.size > maximum:
                raise CargoPublishError("Cargo package metadata file is missing or exceeds its size limit.")
            source = archive.extractfile(member)
            if source is None:
                raise CargoPublishError("Cargo package metadata file cannot be read.")
            return source.read().decode("utf-8")

        manifest = tomllib.loads(text_file("Cargo.toml", 1024 * 1024))
        package = manifest["package"]
        name, version = package["name"], package["version"]
        if root != f"{name}-{version}":
            raise CargoPublishError("Cargo archive root and package name/version differ.")
        readme_file = package.get("readme")
        if readme_file is True:
            readme_file = "README.md"
        elif readme_file is False:
            readme_file = None
        readme = text_file(readme_file) if readme_file is not None else None
        license_file = package.get("license-file")
        if license_file is not None:
            text_file(license_file)
        dependencies = _dependencies(manifest)
        for target, table in sorted(manifest.get("target", {}).items()):
            dependencies.extend(_dependencies(table, target))
        return {
            "name": name, "vers": version, "deps": dependencies,
            "features": manifest.get("features", {}), "authors": package.get("authors", []),
            "description": package.get("description"), "documentation": package.get("documentation"),
            "homepage": package.get("homepage"), "readme": readme, "readme_file": readme_file,
            "keywords": package.get("keywords", []), "categories": package.get("categories", []),
            "license": package.get("license"), "license_file": license_file,
            "repository": package.get("repository"), "badges": manifest.get("badges", {}),
            "links": package.get("links"), "rust_version": package.get("rust-version"),
        }


def publish_crate(path: Path, token: str) -> None:
    """PUT the exact reviewed archive to crates.io; callers own retry/checksum reconciliation."""
    if not token or any(ord(character) < 33 or ord(character) > 126 for character in token):
        raise CargoPublishError("Cargo registry token is empty or contains invalid header characters.")
    connection = None
    try:
        with Path(path).open("rb") as file:
            file.seek(0, 2)
            size = file.tell()
            if size == 0 or size > 0xFFFFFFFF:
                raise CargoPublishError("Cargo archive length cannot be represented by the registry protocol.")
            metadata = json.dumps(metadata_from_archive(file), separators=(",", ":"),
                                  ensure_ascii=False).encode("utf-8")
            if len(metadata) > 0xFFFFFFFF:
                raise CargoPublishError("Cargo publication metadata exceeds the protocol limit.")
            file.seek(0)
            connection = http.client.HTTPSConnection("crates.io", timeout=300)
            connection.putrequest("PUT", "/api/v1/crates/new")
            connection.putheader("Authorization", token)
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Accept", "application/json")
            connection.putheader("User-Agent", "IronGraph-release")
            connection.putheader("Content-Length", str(8 + len(metadata) + size))
            connection.endheaders()
            connection.send(struct.pack("<I", len(metadata)))
            connection.send(metadata)
            connection.send(struct.pack("<I", size))
            remaining = size
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise CargoPublishError("Cargo archive changed during upload; reconcile registry state before retrying.")
                connection.send(chunk)
                remaining -= len(chunk)
            response = connection.getresponse()
            body = response.read(1024 * 1024 + 1)
            if not 200 <= response.status < 300:
                raise CargoPublishError(f"Cargo registry upload failed (HTTP {response.status}); reconcile the staged version before retrying.")
            if len(body) > 1024 * 1024:
                raise CargoPublishError("Cargo registry response exceeded its limit; reconcile the staged version before retrying.")
            result = json.loads(body)
            if not isinstance(result, dict) or result.get("errors"):
                raise CargoPublishError("Cargo registry reported a publication error; reconcile the staged version before retrying.")
    except CargoPublishError:
        raise
    except Exception:
        raise CargoPublishError("Cargo publication could not complete; reconcile registry state before retrying the same archive.") from None
    finally:
        if connection is not None:
            connection.close()
