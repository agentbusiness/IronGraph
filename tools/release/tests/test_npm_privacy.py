import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("release", Path(__file__).parents[1] / "release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class NpmPrivacyTests(unittest.TestCase):
    def test_metadata_rejects_paths_and_install_fields(self):
        for value in ("file:../source.tgz", "/home/private-owner/source.tgz", "%2Ftmp%2Fsource.tgz",
                      "C:\\private-owner\\source.tgz", "\\\\host\\private-owner", "secret-value",
                      "/Volumes/DeveloperHome/private-owner/source.tgz", "source:(/home/private-owner/source.tgz)"):
            with self.subTest(value=value), self.assertRaises(release.ReleaseError):
                release.audit_npm_metadata({"nested": [{"value": value}]}, ("secret-value",))
        for field in ("_from", "_resolved", "_where", "_args", "_location", "_requested"):
            with self.subTest(field=field), self.assertRaises(release.ReleaseError):
                release.audit_npm_metadata({field: "anything"})
        release.audit_npm_metadata({"name": "@irongraph/client", "repository": "https://github.com/agentbusiness/IronGraph",
                                    "dist": {"tarball": "https://registry.npmjs.org/package/-/package.tgz"},
                                    "readme": "Run it from the directory containing the file:\ncommand"})

    def test_real_npm_upload_preserves_bytes_without_local_metadata(self):
        requests = []

        class Registry(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{}')

            def do_PUT(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(201)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

        server = ThreadingHTTPServer(("127.0.0.1", 0), Registry)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        registry = f"http://127.0.0.1:{server.server_port}/"
        with tempfile.TemporaryDirectory(prefix="private-owner-") as temporary:
            root = Path(temporary)
            config = root / "npmrc"
            config.write_text(f"registry={registry}\n//127.0.0.1:{server.server_port}/:_authToken=loopback-test-only\n")
            env = {"NPM_CONFIG_USERCONFIG": str(config), "NPM_CONFIG_GLOBALCONFIG": str(root / "empty-global"),
                   "NPM_CONFIG_CACHE": str(root / "cache"), "NPM_CONFIG_BROWSER": "false"}
            source = root / "source"
            source.mkdir()
            packed = root / "packed"
            packed.mkdir()
            for name in release.PUBLIC_FILES:
                (source / name).write_text("Public package documentation. Set the option to false.\n")
            (source / "index.js").write_text("module.exports = 42;\n")
            for name in ("@irongraph/privacy-fixture", "irongraph-privacy-fixture"):
                manifest = {"name": name, "version": "0.1.3", "license": "Apache-2.0", "main": "index.js"}
                (source / "package.json").write_text(json.dumps(manifest))
                result = subprocess.run(["npm", "pack", "--json", "--ignore-scripts", "--pack-destination", str(packed)],
                                        cwd=source, env=release.clean_env(env), check=True, capture_output=True, text=True)
                archive = packed / json.loads(result.stdout)[0]["filename"]
                # Positive reproducer: the old tarball invocation really emits the private path.
                subprocess.run(["npm", "publish", str(archive), "--ignore-scripts", "--registry", registry],
                               cwd=source, env=release.clean_env(env), check=True, capture_output=True)
                leaking = requests.pop()
                self.assertIn(str(archive), json.dumps(leaking))
                with self.assertRaises(release.ReleaseError):
                    release.audit_npm_metadata(leaking["versions"]["0.1.3"])
                release.publish_npm(archive, env, registry)
                uploaded = requests.pop()
                release.audit_npm_metadata(uploaded["versions"]["0.1.3"])
                self.assertNotIn(str(root), json.dumps(uploaded))
                self.assertNotIn("private-owner", json.dumps(uploaded))
                attachment, = uploaded["_attachments"].values()
                self.assertEqual(base64.b64decode(attachment["data"]), archive.read_bytes())
                self.assertEqual(uploaded["dist-tags"], {"latest": "0.1.3"})
                self.assertEqual(uploaded["versions"]["0.1.3"]["name"], name)
                # npm's output archive must match before any real upload is attempted.
                with patch.object(release, "sha256", side_effect=["changed", "reviewed"]):
                    with self.assertRaisesRegex(release.ReleaseError, "repack differs"):
                        release.publish_npm(archive, env, registry)
                self.assertEqual(requests, [])

    def test_public_checksum_does_not_override_privacy_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            import io
            import tarfile
            archive = Path(temporary) / "fixture.tgz"
            data = json.dumps({"name": "@irongraph/client"}).encode()
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo("package/package.json")
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
            with patch.object(release, "public_json", return_value={"_resolved": "/tmp/fixture.tgz"}):
                with self.assertRaisesRegex(release.ReleaseError, "installation fields"):
                    release.existing_package(archive, "0.1.3")
