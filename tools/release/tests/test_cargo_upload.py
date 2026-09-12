import contextlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import threading
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("cargo_upload", Path(__file__).parents[1] / "cargo_upload.py")
cargo_upload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cargo_upload)

MANIFEST = '''[package]
name = "irongraph-sdk"
version = "1.2.3"
description = "Embedded database"
readme = "README.md"
license-file = "LICENSE.txt"
homepage = "https://github.com/agentbusiness/IronGraph"
documentation = "https://docs.rs/irongraph-sdk"
repository = "https://github.com/agentbusiness/IronGraph"
authors = ["IronGraph"]
keywords = ["database"]
categories = ["database"]
links = "irongraph_ffi"
rust-version = "1.94"
[dependencies.renamed]
package = "original"
version = "1.2"
optional = true
default-features = false
features = ["one"]
registry-index = "https://example.test/index"
[dependencies.serde]
version = "1.0"
[build-dependencies.sha2]
version = "0.10"
[dev-dependencies.tempfile]
version = "3"
[target.'cfg(unix)'.dependencies.libc]
version = "0.2"
[target.'cfg(windows)'.build-dependencies.cc]
version = "1"
[target.'cfg(unix)'.dev-dependencies.test_helper]
version = "2"
[features]
default = ["serde/std"]
extra = ["dep:renamed", "renamed?/one"]
[badges.maintenance]
status = "actively-developed"
'''


def crate_bytes(manifest=MANIFEST, extras=()):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in [
            ("irongraph-sdk-1.2.3/Cargo.toml", manifest.encode()),
            ("irongraph-sdk-1.2.3/README.md", "# IronGraph\nComplete documentation — commercial use.\n".encode()),
            ("irongraph-sdk-1.2.3/LICENSE.txt", b"Apache license terms\n"),
            *extras,
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


@contextlib.contextmanager
def registry(response_status=200, response_body=b'{"warnings":{"other":[]}}'):
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            captured["method"] = self.command
            captured["path"] = self.path
            captured["headers"] = self.headers
            captured["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def local_connection(host, timeout):
            captured["host"] = host
            return http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=timeout)

        with mock.patch.object(cargo_upload.http.client, "HTTPSConnection", side_effect=local_connection):
            yield captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class CargoUploadTests(unittest.TestCase):
    def test_metadata_uses_normalized_archive_and_maps_every_dependency_kind(self):
        metadata = cargo_upload.metadata_from_archive(io.BytesIO(crate_bytes()))
        self.assertEqual(metadata["name"], "irongraph-sdk")
        self.assertEqual(metadata["vers"], "1.2.3")
        self.assertIn("commercial use", metadata["readme"])
        self.assertEqual(metadata["readme_file"], "README.md")
        self.assertEqual(metadata["license_file"], "LICENSE.txt")
        self.assertIsNone(metadata["license"])
        self.assertEqual(metadata["rust_version"], "1.94")
        self.assertEqual(metadata["links"], "irongraph_ffi")
        self.assertEqual(metadata["features"]["extra"], ["dep:renamed", "renamed?/one"])
        self.assertEqual(metadata["badges"], {"maintenance": {"status": "actively-developed"}})
        deps = {(d["name"], d["kind"], d["target"]): d for d in metadata["deps"]}
        self.assertEqual(len(deps), 7)
        self.assertEqual(deps[("original", "normal", None)], {
            "name": "original", "version_req": "1.2", "features": ["one"],
            "optional": True, "default_features": False, "target": None, "kind": "normal",
            "registry": "https://example.test/index", "explicit_name_in_toml": "renamed",
        })
        self.assertTrue(deps[("serde", "normal", None)]["default_features"])
        self.assertFalse(deps[("serde", "normal", None)]["optional"])
        self.assertIn(("sha2", "build", None), deps)
        self.assertIn(("tempfile", "dev", None), deps)
        self.assertIn(("libc", "normal", "cfg(unix)"), deps)
        self.assertIn(("cc", "build", "cfg(windows)"), deps)
        self.assertIn(("test_helper", "dev", "cfg(unix)"), deps)

    def test_exact_archive_protocol_over_mock_http(self):
        payload = crate_bytes()
        with tempfile.TemporaryDirectory() as directory, registry() as captured:
            path = Path(directory) / "reviewed.crate"
            path.write_bytes(payload)
            cargo_upload.publish_crate(path, "test-token")
        self.assertEqual(captured["host"], "crates.io")
        self.assertEqual(captured["method"], "PUT")
        self.assertEqual(captured["path"], "/api/v1/crates/new")
        self.assertEqual(captured["headers"]["Authorization"], "test-token")
        self.assertEqual(captured["headers"]["Content-Type"], "application/octet-stream")
        body = captured["body"]
        metadata_size = struct.unpack("<I", body[:4])[0]
        metadata = json.loads(body[4:4 + metadata_size])
        archive_size = struct.unpack("<I", body[4 + metadata_size:8 + metadata_size])[0]
        self.assertEqual(metadata["vers"], "1.2.3")
        self.assertEqual(archive_size, len(payload))
        self.assertEqual(body[8 + metadata_size:], payload)
        self.assertEqual(int(captured["headers"]["Content-Length"]), len(body))

    def test_registry_errors_are_safe_even_with_success_status(self):
        for status in (200, 403):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "reviewed.crate"
                path.write_bytes(crate_bytes())
                with registry(status, b'{"errors":[{"detail":"sensitive-test-token"}]}'):
                    with self.assertRaises(cargo_upload.CargoPublishError) as caught:
                        cargo_upload.publish_crate(path, "sensitive-test-token")
                self.assertNotIn("sensitive-test-token", str(caught.exception))

    def test_invalid_token_cannot_trigger_network(self):
        with mock.patch.object(cargo_upload.http.client, "HTTPSConnection") as connection:
            with self.assertRaises(cargo_upload.CargoPublishError):
                cargo_upload.publish_crate(Path("unused.crate"), "token\r\nInjected: value")
            connection.assert_not_called()

    def test_unsafe_duplicate_and_unresolved_dependency_are_rejected(self):
        invalid_archives = [
            crate_bytes(extras=[("irongraph-sdk-1.2.3/../outside", b"outside")]),
            crate_bytes(extras=[("irongraph-sdk-1.2.3/Cargo.toml", MANIFEST.encode())]),
            crate_bytes(MANIFEST.replace('registry-index = "https://example.test/index"', 'registry = "private"')),
            crate_bytes(MANIFEST.replace('[dependencies.serde]\nversion = "1.0"', '[dependencies.serde]\npath = "../private"\nversion = "1.0"')),
        ]
        for payload in invalid_archives:
            with self.subTest(length=len(payload)), self.assertRaises(cargo_upload.CargoPublishError):
                cargo_upload.metadata_from_archive(io.BytesIO(payload))


if __name__ == "__main__":
    unittest.main()
