import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("release", Path(__file__).parents[1] / "release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_semver_rollover_and_no_padded_numbers(self):
        self.assertEqual(release.next_version("0.1.9", "patch"), "0.1.10")
        self.assertEqual(release.next_version("2.19.99", "minor"), "2.20.0")
        self.assertEqual(release.next_version("2.19.99", "major"), "3.0.0")
        for version in ("v0.1.0", "0.01.0", "1.2", "1.2.3-rc.1"):
            with self.assertRaises(release.ReleaseError):
                release.next_version(version, "patch")

    def test_env_is_data_not_shell_and_permissions_are_enforced(self):
        path = self.root / ".env.publish"
        path.write_text("NPM_TOKEN='$(touch${IFS}stolen)'\nMATURIN_PYPI_TOKEN=some$literal`value`\n")
        path.chmod(0o600)
        with patch.dict(os.environ, {}, clear=True):
            result = release.load_config(path)
        self.assertEqual(result["NPM_TOKEN"], "$(touch${IFS}stolen)")
        self.assertFalse((self.root / "stolen").exists())
        path.chmod(0o644)
        with self.assertRaises(release.ReleaseError):
            release.load_config(path)

    def test_env_duplicates_unknown_fields_rejected_without_value_leak(self):
        path = self.root / ".env.publish"
        path.touch(mode=0o600)
        for contents in ("NPM_TOKEN=sensitive\nNPM_TOKEN=secret", "BAD=sensitive"):
            path.write_text(contents)
            with self.assertRaises(release.ReleaseError) as failure:
                release.load_config(path)
            self.assertNotIn("sensitive", str(failure.exception))

    def test_build_env_removes_tokens_and_native_overrides(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "secret", "NPM_TOKEN": "secret",
                                    "UNRELATED_PASSWORD": "secret", "CARGO_TARGET_DIR": "/elsewhere",
                                    "NAPI_RS_NATIVE_LIBRARY_PATH": "/wrong.node", "NODE_PATH": "/wrong"}, clear=True):
            self.assertEqual(release.clean_env(), {})
            self.assertEqual(release.clean_env({"NPM_TOKEN": "publish-only"}), {"NPM_TOKEN": "publish-only"})

    def test_isolated_macos_flags_preserve_proc_macro_symbols(self):
        macos = release.tool_environment(self.root, "aarch64-apple-darwin")
        linux = release.tool_environment(self.root, "aarch64-unknown-linux-gnu")
        self.assertIn("strip=none", macos["CARGO_ENCODED_RUSTFLAGS"].split("\x1f"))
        self.assertNotIn("strip=none", linux["CARGO_ENCODED_RUSTFLAGS"].split("\x1f"))
        self.assertEqual(macos["CARGO_PROFILE_RELEASE_STRIP"], "none")
        self.assertNotIn("CARGO_PROFILE_RELEASE_STRIP", linux)

    def fixture_source(self):
        for folder in ("bindings/python", "bindings/node", "bindings/javascript", "bindings/rust", "bindings/cli", "web", "crates/ffi"):
            (self.root / folder).mkdir(parents=True)
        (self.root / "Cargo.toml").write_text('[package]\nname="irongraph"\nversion="0.1.0"\n')
        (self.root / "crates/ffi/Cargo.toml").write_text('[package]\nname="irongraph-ffi"\nversion="0.1.0"\n')
        (self.root / "bindings/python/pyproject.toml").write_text('[project]\nversion="0.1.0"\n')
        (self.root / "Cargo.lock").write_text('version = 4\n[[package]]\nname = "irongraph-ffi"\nversion = "0.1.0"\n[[package]]\nname = "other"\nversion = "0.1.0"\nsource = "registry+https://example.org"\n')
        for path in ("web", "bindings/node", "bindings/javascript", "bindings/cli"):
            package = {"name": path, "version": "0.1.0", "optionalDependencies": {"@irongraph/node-linux-arm64-gnu": "0.1.0"}}
            release.write_json(self.root / path / "package.json", package)
            release.write_json(self.root / path / "package-lock.json", {"version": "0.1.0", "packages": {"": package}})
        for suffix in release.TARGETS.values():
            release.write_json(self.root / "bindings/cli/npm" / suffix / "package.json",
                               {"name": "@irongraph/cli-" + suffix, "version": "0.1.0"})

    def test_version_update_keeps_external_dependencies_and_syncs_optional_packages(self):
        self.fixture_source()
        release.write_versions(self.root, "0.2.0")
        self.assertEqual(release.current_version(self.root), "0.2.0")
        lock = (self.root / "Cargo.lock").read_text()
        self.assertIn('name = "irongraph-ffi"\nversion = "0.2.0"', lock)
        self.assertIn('name = "other"\nversion = "0.1.0"', lock)
        package = release.read_json(self.root / "bindings/node/package-lock.json")
        self.assertEqual(package["packages"][""]["optionalDependencies"]["@irongraph/node-linux-arm64-gnu"], "0.2.0")
        for suffix in release.TARGETS.values():
            self.assertEqual(release.read_json(self.root / "bindings/cli/npm" / suffix / "package.json")["version"], "0.2.0")
        self.assertEqual(release.read_json(self.root / "bindings/cli/package-lock.json")["version"], "0.2.0")

    def test_ambiguous_version_anchor_has_no_partial_writes(self):
        self.fixture_source()
        path = self.root / "crates/ffi/Cargo.toml"
        path.write_text(path.read_text() + '[other]\nversion="bad"\n')
        with self.assertRaises(release.ReleaseError):
            release.write_versions(self.root, "0.2.0")
        self.assertEqual(release.current_version(self.root), "0.1.0")

    def make_npm(self, extra=None):
        path = self.root / "package.tgz"
        contents = {"package/" + name: b"public license or description" for name in release.PUBLIC_FILES}
        contents["package/package.json"] = json.dumps({"version": "0.1.0", "license": "Apache-2.0"}).encode()
        contents["package/index.js"] = b"module.exports = {}"
        contents.update(extra or {})
        with tarfile.open(path, "w:gz") as archive:
            for name, data in contents.items():
                item = tarfile.TarInfo(name)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
        return path

    def test_archive_allowlist_has_positive_control_and_rejects_source_secrets_and_maps(self):
        release.audit_archive(self.make_npm(), "npm", "0.1.0")
        for name in ("package/src/lib.rs", "package/.env.publish", "package/dist/index.js.map", "../../source.rs"):
            with self.subTest(name=name), self.assertRaises(release.ReleaseError):
                release.audit_archive(self.make_npm({name: b"private"}), "npm", "0.1.0")

    def test_archive_symlink_rejected(self):
        path = self.make_npm()
        with tarfile.open(path, "w:gz") as archive:
            item = tarfile.TarInfo("package/LICENSE.txt")
            item.type = tarfile.SYMTYPE
            item.linkname = "/private"
            archive.addfile(item)
        with self.assertRaises(release.ReleaseError):
            list(release.archive_files(path))

    def test_wheel_rejects_private_workspace_sbom(self):
        path = self.root / "irongraph.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("irongraph/__init__.py", "")
            archive.writestr("irongraph-0.1.0.dist-info/METADATA", "Version: 0.1.0\nLicense-Expression: Apache-2.0\n")
            for name in ("LICENSE.txt", "THIRD_PARTY_NOTICES.txt"):
                archive.writestr("irongraph-0.1.0.dist-info/licenses/" + name, "terms")
        release.audit_archive(path, "wheel", "0.1.0")
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("irongraph-0.1.0.dist-info/sboms/irongraph-python.cyclonedx.json", '{"private": "workspace paths"}')
        with self.assertRaises(release.ReleaseError):
            release.audit_archive(path, "wheel", "0.1.0")

    def test_changed_qualified_artifact_rejected(self):
        version = "0.1.0"
        for target in release.TARGETS:
            folder = self.root / target
            folder.mkdir()
            (folder / "binary.a").write_bytes(b"binary")
            (folder / "wheels").mkdir()
            artifacts = {"binary.a": release.sha256(folder / "binary.a")}
            for name in (f"libirongraph_ffi-{target}.a", "irongraph-node-0.1.0.tgz",
                         f"irongraph-node-{release.TARGETS[target]}-0.1.0.tgz", "THIRD_PARTY_NOTICES.txt",
                         "wheels/irongraph.whl"):
                (folder / name).write_bytes(b"sdk artifact")
                artifacts[name] = release.sha256(folder / name)
            release.write_json(folder / "qualification.json", {"target": target, "version": version,
                "embedding_search": True, "artifacts": artifacts})
            self.make_standalone_report(folder, target, version)
        release.verify_qualification(self.root, version)
        (self.root / next(iter(release.TARGETS)) / "binary.a").write_bytes(b"different")
        with self.assertRaises(release.ReleaseError):
            release.verify_qualification(self.root, version)
        (self.root / next(iter(release.TARGETS)) / "binary.a").write_bytes(b"binary")
        report_path = self.root / next(iter(release.TARGETS)) / "qualification.json"
        report = release.read_json(report_path)
        report["artifacts"].pop("irongraph-node-0.1.0.tgz")
        release.write_json(report_path, report)
        with self.assertRaisesRegex(release.ReleaseError, "every current embedded SDK artifact"):
            release.verify_qualification(self.root, version)

    def make_standalone_report(self, folder, target, version="0.1.0"):
        artifacts = {}
        for name in release.standalone_names(target, version):
            (folder / name).write_bytes(b"standalone artifact")
            artifacts[name] = release.sha256(folder / name)
        release.write_json(folder / "standalone-qualification.json", {
            "target": target, "version": version, "installed_lifecycle": True, "artifacts": artifacts,
        })

    def test_standalone_qualification_rejects_sdk_only_and_missing_or_changed_artifact(self):
        target = next(iter(release.TARGETS))
        with self.assertRaisesRegex(release.ReleaseError, "SDK-only"):
            release.verify_standalone_report(self.root, target, "0.1.0")
        self.make_standalone_report(self.root, target)
        release.verify_standalone_report(self.root, target, "0.1.0")
        path = self.root / next(iter(release.standalone_names(target, "0.1.0")))
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(release.ReleaseError, "changed"):
            release.verify_standalone_report(self.root, target, "0.1.0")
        self.make_standalone_report(self.root, target)
        report = release.read_json(self.root / "standalone-qualification.json")
        report["artifacts"].pop(path.name)
        release.write_json(self.root / "standalone-qualification.json", report)
        with self.assertRaisesRegex(release.ReleaseError, "qualification required"):
            release.verify_standalone_report(self.root, target, "0.1.0")

    def make_cli_archive(self, native=False, bundle=False, extra=None, manifest_edit=None, executable=True):
        version = "0.1.0"
        path = self.root / ("standalone.tar.gz" if bundle else
                            "irongraph-cli-darwin-arm64-0.1.0.tgz" if native else "irongraph-0.1.0.tgz")
        root = f"irongraph-{version}" if bundle else "package"
        files = {name: b"public terms" for name in release.PUBLIC_FILES}
        files.update({"bin/irongraph": b"binary", "bin/irongraph-mcp": b"mcp"} if native or bundle else {"cli.cjs": b"#!/usr/bin/env node\n"})
        if not bundle:
            manifest = {"name": "@irongraph/cli-darwin-arm64" if native else "irongraph",
                        "version": version, "license": "Apache-2.0"}
            if native:
                manifest.update({"os": ["darwin"], "cpu": ["arm64"]})
            if not native:
                manifest.update({"bin": {"irongraph": "cli.cjs"}, "optionalDependencies": {
                    "@irongraph/cli-" + suffix: version for suffix in release.TARGETS.values()}})
            manifest.update(manifest_edit or {})
            files["package.json"] = json.dumps(manifest).encode()
        files.update(extra or {})
        with tarfile.open(path, "w:gz") as archive:
            for name, data in files.items():
                member = tarfile.TarInfo(root + "/" + name)
                member.size = len(data)
                member.mode = 0o755 if executable else 0o644
                archive.addfile(member, io.BytesIO(data))
        return path

    def test_cli_archive_exact_inventory_versions_and_lifecycle_hooks(self):
        for native in (False, True):
            release.audit_archive(self.make_cli_archive(native=native), "npm", "0.1.0")
            for extra in ({"src/main.rs": b"private"}, {"index.js": b"unexpected"}):
                with self.assertRaises(release.ReleaseError):
                    release.audit_archive(self.make_cli_archive(native=native, extra=extra), "npm", "0.1.0")
            for edit in ({"version": "0.2.0"}, {"scripts": {"postinstall": "download-private-source"}}):
                with self.assertRaises(release.ReleaseError):
                    release.audit_archive(self.make_cli_archive(native=native, manifest_edit=edit), "npm", "0.1.0")
        for edit in ({"optionalDependencies": {}}, {"optionalDependencies": {"@irongraph/cli-darwin-arm64": "^0.1.0"}},
                     {"bin": {"irongraph": "wrong.cjs"}}):
            with self.assertRaises(release.ReleaseError):
                release.audit_archive(self.make_cli_archive(manifest_edit=edit), "npm", "0.1.0")
        release.audit_archive(self.make_cli_archive(manifest_edit={"scripts": {"test": "node --test test"}}), "npm", "0.1.0")
        for edit in ({"cpu": ["x64"]}, {"os": ["linux"]}, {"name": "@irongraph/cli-linux-arm64-gnu"}):
            with self.assertRaises(release.ReleaseError):
                release.audit_archive(self.make_cli_archive(native=True, manifest_edit=edit), "npm", "0.1.0")

    def test_standalone_bundle_exact_root_inventory_and_executable_modes(self):
        release.audit_archive(self.make_cli_archive(bundle=True), "standalone", "0.1.0")
        for options in ({"extra": {"Cargo.toml": b"private"}}, {"executable": False}):
            with self.assertRaises(release.ReleaseError):
                release.audit_archive(self.make_cli_archive(bundle=True, **options), "standalone", "0.1.0")
        with self.assertRaises(release.ReleaseError):
            release.audit_archive(self.make_cli_archive(bundle=True), "standalone", "0.2.0")
        with self.assertRaises(release.ReleaseError):
            release.audit_archive(self.make_cli_archive(executable=False), "npm", "0.1.0")

    def test_all_native_npm_children_publish_before_both_parents(self):
        parents = {"irongraph-0.1.0.tgz", "irongraph-node-0.1.0.tgz"}
        children = {f"irongraph-{binding}-{suffix}-0.1.0.tgz"
                    for binding in ("node", "cli") for suffix in release.TARGETS.values()}
        for name in parents | children | {"irongraph-client-0.1.0.tgz"}:
            (self.root / name).touch()
        ordered = [path.name for path in release.npm_publish_order(self.root, "0.1.0")]
        self.assertEqual(set(ordered[-2:]), parents)
        self.assertTrue(children.issubset(set(ordered[:-2])))

    def test_release_preflight_runs_launcher_and_port_checks_once_before_rust_marker(self):
        self.fixture_source()
        stage = self.root / "stage"
        source = stage / "source"
        source.mkdir(parents=True)
        (source / "Cargo.toml").write_text((self.root / "Cargo.toml").read_text())
        with patch.object(release, "TARGETS", {}), patch.object(release, "ROOT", self.root), \
             patch.object(release, "ensure_console"), patch.object(release, "run") as execute:
            release.build_matrix(stage, {})
            release.build_matrix(stage, {})
        commands = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(commands, [
            ["node", "--test", "bindings/cli/test/lifecycle.test.cjs"],
            [release.sys.executable, "tools/bindings/verify-ports.py"],
            ["bash", "tools/bindings/verify-rust.sh"],
        ])
        self.assertEqual(release.read_json(stage / "rust-verified.json"), {"version": "0.1.0"})

    def test_publish_rejects_unlisted_file_before_any_network(self):
        folder = self.root / "publish"
        folder.mkdir()
        (folder / "private-source.tar.gz").write_bytes(b"private")
        with patch.object(release, "github") as network, self.assertRaises(release.ReleaseError):
            release.publish(self.root, {}, {"checksums": {}})
        network.assert_not_called()

    def test_publication_waits_for_verified_registry_visibility(self):
        archive = self.root / "package.tgz"
        with patch.object(release, "existing_package", side_effect=[False, False, True]) as lookup, \
             patch.object(release.time, "sleep") as sleep:
            release.wait_for_public_package(archive, "0.1.2")
        self.assertEqual(lookup.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_publication_wait_is_bounded_and_rejects_checksum_mismatch(self):
        archive = self.root / "package.tgz"
        with patch.object(release, "existing_package", return_value=False), \
             patch.object(release.time, "monotonic", side_effect=[0, 1201]), \
             self.assertRaisesRegex(release.ReleaseError, "still pending"):
            release.wait_for_public_package(archive, "0.1.2")
        with patch.object(release, "existing_package", side_effect=release.ReleaseError("checksum mismatch")), \
             self.assertRaisesRegex(release.ReleaseError, "checksum mismatch"):
            release.wait_for_public_package(archive, "0.1.2")

    def test_release_repository_must_match_origin_and_be_public(self):
        config = {"IRONGRAPH_BINARY_REPO": "owner/engine"}
        with patch.object(release.subprocess, "check_output", return_value="git@github.com:owner/engine.git"), \
             patch.object(release, "github", return_value={"private": False}) as request:
            self.assertEqual(release.validate_binary_repo(config), {"private": False})
            request.assert_called_once_with(config, "/repos/owner/engine")
        with patch.object(release.subprocess, "check_output", return_value="git@github.com:owner/other.git"), \
             self.assertRaises(release.ReleaseError):
            release.validate_binary_repo(config)
        with patch.object(release.subprocess, "check_output", return_value="git@github.com:owner/engine.git"), \
             patch.object(release, "github", return_value={"private": True}), \
             self.assertRaises(release.ReleaseError):
            release.validate_binary_repo(config)

    def test_version_finalization_resumes_partial_write(self):
        self.fixture_source()
        stage = self.root / "stage"
        stage.mkdir()
        changes = release.version_updates(self.root, "0.2.0")
        release.write_json(stage / "version-journal.json", {str(p.relative_to(self.root)): {"before": p.read_text(), "after": t}
                                                          for p, t in changes.items()})
        path = self.root / "Cargo.toml"
        path.write_text(changes[path])
        state = {"version": "0.2.0"}
        release.finalize_versions(stage, state, self.root)
        self.assertTrue(state["complete"])
        self.assertEqual(release.current_version(self.root), "0.2.0")

    def test_unrelated_finalization_change_is_preserved(self):
        self.fixture_source()
        stage = self.root / "stage"
        stage.mkdir()
        release.write_json(stage / "version-journal.json", {"Cargo.toml": {"before": "before", "after": "after"}})
        original = (self.root / "Cargo.toml").read_text()
        with self.assertRaises(release.ReleaseError):
            release.finalize_versions(stage, {}, self.root)
        self.assertEqual((self.root / "Cargo.toml").read_text(), original)

    def test_staging_rejects_concurrent_source_edits_and_recovers_partial_copy(self):
        stage = self.root / "stage"
        stage.mkdir()
        state = {"version": "0.2.0", "source_fingerprint": "reviewed"}

        def copy(destination, version):
            destination.mkdir()
            (destination / "version").write_text(version)

        with patch.object(release, "stage_source", side_effect=copy), patch.object(release, "fingerprint", return_value="changed"):
            with self.assertRaises(release.ReleaseError):
                release.finish_staging(stage, state)
        self.assertFalse((stage / "source").exists())
        self.assertFalse(state.get("staged", False))
        with patch.object(release, "stage_source", side_effect=copy), patch.object(release, "fingerprint", return_value="reviewed"):
            release.finish_staging(stage, state)
        self.assertTrue(state["staged"])
        self.assertEqual((stage / "source/version").read_text(), "0.2.0")
        self.assertFalse((stage / "source.partial").exists())


if __name__ == "__main__":
    unittest.main()
