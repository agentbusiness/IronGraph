import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

SPEC = importlib.util.spec_from_file_location("release", Path(__file__).parents[1] / "release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class PublishAutomationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"GITHUB_TOKEN": "fixture-github-token", "NPM_TOKEN": "npm_abcd12345678wxyz",
                       "IRONGRAPH_BINARY_REPO": "agentbusiness/IronGraph"}

    def token(self, **changes):
        value = {"token": "npm_abcd...wxyz", "bypass_2fa": True, "revoked": None,
                 "permissions": [{"name": "package", "action": "write"}],
                 "scopes": [{"type": "package", "name": None}]}
        return {**value, **changes}

    def check_auth(self, token, login="agentbusiness", username="businessagent"):
        with patch.object(release, "github", return_value={"login": login}) as github, \
             patch.object(release, "validate_binary_repo", return_value={"permissions": {"push": True}}), \
             patch.object(release, "npm_auth_json", side_effect=[{"username": username}, {"objects": [token]}]) as npm:
            release.check_publish_auth(self.config)
        self.assertEqual(github.call_args.args[1], "/user")
        self.assertEqual([call.args[1] for call in npm.call_args_list], ["/-/whoami", "/-/npm/v1/tokens"])

    def test_auth_uses_read_only_endpoints_and_official_accounts(self):
        self.check_auth(self.token())
        with self.assertRaisesRegex(release.ReleaseError, "official agentbusiness"):
            self.check_auth(self.token(), login="different-account")
        with self.assertRaisesRegex(release.ReleaseError, "official businessagent"):
            self.check_auth(self.token(), username="different-account")

    def test_npm_token_must_allow_direct_write_bypass_and_unscoped_launcher(self):
        for changes, message in [({"bypass_2fa": False}, "Bypass 2FA"),
                                 ({"permissions": [{"name": "package", "action": "read"}]}, "write access"),
                                 ({"scopes": [{"type": "package", "name": "@irongraph"}]}, "irongraph"),
                                 ({"token": "npm_other...abcd"}, "uniquely verify")]:
            with self.subTest(changes=changes), self.assertRaisesRegex(release.ReleaseError, message):
                self.check_auth(self.token(**changes))
        self.check_auth(self.token(scopes=[{"type": "package", "name": "@irongraph"},
                                          {"type": "package", "name": "irongraph"}]))

    def test_token_pagination_cannot_forward_secret_to_another_host(self):
        with patch.object(release, "github", return_value={"login": "agentbusiness"}), \
             patch.object(release, "validate_binary_repo", return_value={"permissions": {"push": True}}), \
             patch.object(release, "npm_auth_json", side_effect=[{"username": "businessagent"},
                 {"objects": [], "urls": {"next": "https://untrusted.example/-/npm/v1/tokens"}}]) as request, \
             self.assertRaisesRegex(release.ReleaseError, "Unexpected npm"):
            release.check_publish_auth(self.config)
        self.assertEqual(request.call_count, 2)

    def test_check_auth_cannot_start_build_or_publish(self):
        with patch.object(release.sys, "argv", ["release.py", "--check-auth"]), \
             patch.object(release, "load_config", return_value=self.config), \
             patch.object(release, "check_publish_auth") as auth, \
             patch.object(release, "prerequisites") as prerequisites, \
             patch.object(release, "build_matrix") as build, patch.object(release, "publish") as publish:
            release.main()
        auth.assert_called_once_with(self.config)
        prerequisites.assert_not_called()
        build.assert_not_called()
        publish.assert_not_called()

    def test_packaging_only_never_runs_runtime_qualification(self):
        (self.root / "Cargo.toml").write_text('[package]\nversion="0.1.11"\n')
        output = self.root / "artifacts"
        output.mkdir()
        with patch.object(release, "host_target", return_value="aarch64-apple-darwin"), \
             patch.object(release, "verify_standalone") as standalone, \
             patch.object(release, "verify_packaging") as packaging, patch.object(release, "run") as command:
            release.qualify_target(self.root, output, "aarch64-apple-darwin", config={"packaging_only": True})
        standalone.assert_not_called()
        command.assert_not_called()
        packaging.assert_called_once()
        self.assertFalse(release.read_json(output / "packaging.json")["tests_run"])

    def test_packaging_only_matrix_skips_product_test_commands(self):
        source = self.root / "source"
        source.mkdir()
        with patch.object(release, "TARGETS", {}), patch.object(release, "ROOT", self.root), \
             patch.object(release, "ensure_console"), patch.object(release, "run") as command:
            release.build_matrix(self.root, {"packaging_only": True})
        command.assert_not_called()

    def test_worktree_uses_primary_ignored_config_without_copying_secrets(self):
        common = self.root / "primary/.git"
        common.mkdir(parents=True)
        shared = common.parent / ".env.publish"
        shared.touch(mode=0o600)
        worktree = self.root / "worktree"
        worktree.mkdir()
        with patch.object(release, "ROOT", worktree), \
             patch.object(release.subprocess, "check_output", return_value=str(common)):
            self.assertEqual(release.release_config_path(), shared)
            self.assertFalse((worktree / ".env.publish").exists())

    def test_noninteractive_command_closes_stdin_and_captures_errors(self):
        result = subprocess.CompletedProcess([], 1, "", "fixture-github-token")
        with patch.object(release.subprocess, "run", return_value=result) as command, \
             self.assertRaises(release.ReleaseError) as error:
            release.run(["npm", "publish"], env={"NPM_TOKEN": "fixture-github-token"}, capture=True, noninteractive=True)
        self.assertIs(command.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(command.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertNotIn("fixture-github-token", str(error.exception))

    def test_missing_token_never_falls_back_to_browser_credentials(self):
        with patch.object(release, "run") as command, self.assertRaisesRegex(release.ReleaseError, "NPM_TOKEN"):
            release.publish_npm(self.root / "missing.tgz", {"NPM_CONFIG_USERCONFIG": "browser-session"})
        command.assert_not_called()

    def test_accepted_upload_survives_failed_verification_without_republication(self):
        path = self.root / "fixture.tgz"
        path.write_bytes(b"reviewed")
        state = {"version": "0.1.10", "checksums": {path.name: release.sha256(path)}}
        with patch.object(release, "existing_package", return_value=False), \
             patch.object(release, "publish_npm") as publish, \
             patch.object(release, "wait_for_public_package", side_effect=release.RegistryPending("temporary")), \
             self.assertRaises(release.RegistryPending):
            release.publish_registry_file(self.root, self.config, state, path, "python")
        publish.assert_called_once()
        saved = release.read_json(self.root / "state.json")
        self.assertEqual(saved["submitted"][path.name], release.sha256(path))
        with patch.object(release, "existing_package", return_value=False), \
             patch.object(release, "publish_npm") as publish, patch.object(release, "wait_for_public_package"):
            release.publish_registry_file(self.root, self.config, saved, path, "python")
        publish.assert_not_called()
        self.assertEqual(saved["published"], [path.name])

    def test_transient_registry_errors_retry_but_checksum_errors_do_not(self):
        with patch.object(release, "existing_package", side_effect=[release.RegistryPending("connection"), False, True]) as lookup, \
             patch.object(release.time, "sleep") as sleep:
            release.wait_for_public_package(self.root / "fixture.tgz", "0.1.10")
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(lookup.call_args.kwargs["verify_download"])

    def test_public_download_checks_actual_bytes(self):
        with patch.object(release.urllib.request, "urlopen", return_value=io.BytesIO(b"public bytes")):
            release.verify_public_download("https://registry.npmjs.org/package.tgz", hashlib.sha256(b"public bytes").hexdigest())
        with patch.object(release.urllib.request, "urlopen", return_value=io.BytesIO(b"changed")), \
             self.assertRaisesRegex(release.ReleaseError, "checksum differs"):
            release.verify_public_download("https://registry.npmjs.org/package.tgz", "expected")
        error = urllib.error.HTTPError("https://registry.npmjs.org/package.tgz", 404, "pending", {}, None)
        self.addCleanup(error.close)
        with patch.object(release.urllib.request, "urlopen", side_effect=error), self.assertRaises(release.RegistryPending):
            release.verify_public_download(error.url, "expected")

    def test_github_write_404_is_an_error_not_missing_release(self):
        error = urllib.error.HTTPError("https://api.github.com/repos/fixture", 404, "denied", {}, None)
        self.addCleanup(error.close)
        with patch.object(release.urllib.request, "urlopen", side_effect=error), self.assertRaisesRegex(release.ReleaseError, "GitHub POST"):
            release.github(self.config, "/repos/fixture/releases", "POST", {})

    def test_github_publishes_only_after_all_three_assets_match(self):
        publication = self.root / "publish"
        publication.mkdir()
        (publication / "README.md").write_text("IronGraph release")
        assets = []
        for target in release.TARGETS:
            path = publication / f"libirongraph_ffi-{target}.a"
            path.write_bytes(b"!<arch>\n")
            assets.append({"name": path.name, "size": path.stat().st_size, "digest": "sha256:" + release.sha256(path),
                           "browser_download_url": "https://github.com/agentbusiness/IronGraph/releases/download/v0.1.10/" + path.name})
        state = {"version": "0.1.10", "checksums": {p.name: release.sha256(p) for p in publication.iterdir()}}
        events = []
        def github(config, route, method="GET", value=None):
            if method == "PATCH":
                events.append("publish")
                return {"draft": False}
            return {"id": 7, "draft": True, "assets": [] if "/tags/" in route else assets}
        with patch.object(release, "prepare_binary_repository"), patch.object(release, "github", side_effect=github), \
             patch.object(release, "upload_asset", side_effect=lambda *args: events.append("upload")), \
             patch.object(release, "public_json", return_value={"author": {"login": "agentbusiness"}, "assets": assets}), \
             patch.object(release, "verify_public_download", side_effect=lambda *args: events.append("download")), \
             patch.object(release, "venv_python", return_value="python"), patch.object(release, "run"):
            release.publish(self.root, self.config, state)
        self.assertEqual(events, ["upload"] * 3 + ["publish"] + ["download"] * 3)
        assets[0]["digest"] = "sha256:wrong"
        events.clear()
        with patch.object(release, "prepare_binary_repository"), patch.object(release, "github", side_effect=github), \
             patch.object(release, "upload_asset"), self.assertRaisesRegex(release.ReleaseError, "remains a draft"):
            release.publish(self.root, self.config, state)
        self.assertNotIn("publish", events)

    def test_privacy_rechecks_native_files_and_encoded_credentials(self):
        path = self.root / "libirongraph_ffi.a"
        for secret in (b"fixture-github-token", base64.b64encode(b"fixture-github-token"), b"fixture-github-token".hex().encode()):
            path.write_bytes(b"native" + secret)
            with self.subTest(secret=secret), self.assertRaisesRegex(release.ReleaseError, "Private data"):
                release.audit_publication(self.root, self.config, "0.1.10")

    def test_native_archive_rejects_local_numeric_owner(self):
        path = self.root / "libirongraph_ffi.a"
        header = b"member.o/".ljust(16) + b"0".ljust(12) + b"501".ljust(6) + b"20".ljust(6) + b"100644".ljust(8) + b"0".ljust(10) + b"`\n"
        path.write_bytes(b"!<arch>\n" + header)
        with self.assertRaisesRegex(release.ReleaseError, "ownership metadata"):
            release.audit_publication(self.root, self.config, "0.1.10")

    def test_macos_wheel_stripping_repairs_record_and_removes_zip_metadata(self):
        wheel = self.root / "fixture.whl"
        binary = "irongraph/_native.so"
        record = "irongraph-0.1.10.dist-info/RECORD"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(binary, b"debug paths")
            archive.writestr(record, f"{binary},sha256=old,11\n{record},,\n")
            archive.comment = b"local metadata"
        def transform(argv):
            if argv[0] == "strip":
                Path(argv[-1]).write_bytes(b"stripped")
            elif argv[0] == "codesign":
                Path(argv[-1]).write_bytes(Path(argv[-1]).read_bytes() + b" signed")
        with patch.object(release, "run", side_effect=transform):
            release.clean_macos_wheel(wheel)
        with zipfile.ZipFile(wheel) as archive:
            data = archive.read(binary)
            self.assertEqual(data, b"stripped signed")
            rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
            self.assertEqual(rows[0], [binary, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(), str(len(data))])
            self.assertEqual(archive.comment, b"")
