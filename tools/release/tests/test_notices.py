import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("notices", Path(__file__).parents[1] / "notices.py")
notices = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(notices)


class NoticesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.package_dir = self.root / "package"
        self.package_dir.mkdir()
        self.package = {"id": "example@1.2.3", "name": "example", "version": "1.2.3", "license": "MIT OR Apache-2.0",
                        "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": str(self.package_dir / "Cargo.toml"),
                        "repository": "https://github.com/vendor/project/tree/main/crates/example"}
        self.metadata = {"packages": [self.package]}
        self.cache = self.root / "cache"
        self.commit = "a" * 40
        self.std = {"name": "Rust std", "version": "1.94.1", "license": "component terms", "files": []}
        self.std_patch = patch.object(notices, "_stdlib", return_value=self.std)
        self.std_patch.start()
        self.addCleanup(self.std_patch.stop)

    def vcs(self):
        (self.package_dir / ".cargo_vcs_info.json").write_text(json.dumps({"git": {"sha1": self.commit}, "path_in_vcs": "crates/example"}))

    def test_preserves_bundled_native_licenses_and_registry_checksums(self):
        (self.package_dir / "vendor/native").mkdir(parents=True)
        files = {"LICENSE": "Copyright author\nPermission terms\n", "vendor/native/NOTICE.txt": "Native attribution\n"}
        for name, text in files.items():
            (self.package_dir / name).write_text(text)
        (self.package_dir / ".cargo-checksum.json").write_text(json.dumps({"package": "b" * 64,
            "files": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}}))
        with patch.object(notices, "_download", side_effect=AssertionError("must use bundled texts")):
            text, inventory = notices.collect(self.metadata, self.cache)
        self.assertIn("Native attribution", text)
        self.assertEqual(len(inventory[0]["files"]), 2)
        self.assertNotIn("text", inventory[0]["files"][0])
        (self.package_dir / "LICENSE").write_text("tampered")
        with self.assertRaisesRegex(notices.NoticesError, "checksum mismatch"):
            notices.collect(self.metadata, self.cache)

    def test_pinned_upstream_parent_license_and_cache(self):
        self.vcs()
        body = b"Copyright upstream author\nPermission is hereby granted.\n"
        blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        tree = {"tree": [{"path": "LICENSE-MIT", "type": "blob", "sha": blob},
                         {"path": "other/LICENSE", "type": "blob", "sha": blob}]}
        def download(url):
            self.assertIn(self.commit, url)
            if "api.github.com" in url:
                return json.dumps(tree).encode()
            self.assertTrue(url.endswith("/LICENSE-MIT"))
            return body
        with patch.object(notices, "_download", side_effect=download) as network:
            text, inventory = notices.collect(self.metadata, self.cache)
            self.assertEqual(network.call_count, 2)
        self.assertIn("Copyright upstream author", text)
        self.assertEqual(inventory[0]["files"][0]["git_commit"], self.commit)
        with patch.object(notices, "_download", side_effect=AssertionError("cache must be offline")):
            self.assertEqual(notices.collect(self.metadata, self.cache), (text, inventory))

    def test_git_blob_mismatch_and_cache_tampering_fail_closed(self):
        self.vcs()
        tree = {"tree": [{"path": "LICENSE", "type": "blob", "sha": "0" * 40}]}
        with patch.object(notices, "_download", side_effect=[json.dumps(tree).encode(), b"wrong"]):
            with self.assertRaisesRegex(notices.NoticesError, "pinned Git tree blob"):
                notices.collect(self.metadata, self.cache)
        path = next(self.cache.glob("*.json"))
        record = json.loads(path.read_text())
        record["sha256"] = "0" * 64
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(notices.NoticesError, "cache is corrupt"):
            notices.collect(self.metadata, self.cache)

    def test_missing_provenance_never_substitutes_spdx_license(self):
        with self.assertRaisesRegex(notices.NoticesError, "commit-pinned"):
            notices.collect(self.metadata, self.cache)
        self.vcs()
        with patch.object(notices, "_download", return_value=b'{"tree":[]}'):
            with self.assertRaisesRegex(notices.NoticesError, "No license text"):
                notices.collect(self.metadata, self.cache)

    def test_notice_file_alone_does_not_satisfy_missing_license_terms(self):
        (self.package_dir / "NOTICE").write_text("Attribution only")
        with self.assertRaisesRegex(notices.NoticesError, "commit-pinned"):
            notices.collect(self.metadata, self.cache)

    def test_dev_only_dependencies_are_excluded(self):
        root = {"id": "root", "name": "irongraph-ffi", "source": None, "version": "0.1.0"}
        self.metadata["packages"].append(root)
        self.metadata["resolve"] = {"nodes": [{"id": "root", "deps": [{"pkg": self.package["id"], "dep_kinds": [{"kind": "dev"}]}]}]}
        text, inventory = notices.collect(self.metadata, self.cache)
        self.assertEqual(len(inventory), 1)
        self.assertNotIn("example 1.2.3", text)

    def test_copyleft_terms_are_retained_without_permission_inference(self):
        self.package["license"] = "GPL-3.0-only"
        (self.package_dir / "COPYING").write_text("Exact upstream GPL terms and copyright.")
        text, inventory = notices.collect(self.metadata, self.cache)
        self.assertIn("Exact upstream GPL terms and copyright.", text)
        self.assertEqual(inventory[0]["license"], "GPL-3.0-only")
        self.assertNotIn("approved", inventory[0])

    def test_exact_declaration_fallback_preserves_authors_and_selected_license(self):
        manifest = ('[package]\nname="example"\nversion="1.2.3"\n'
                    'authors=["Original Author"]\nlicense="MIT OR Apache-2.0"\n').encode()
        (self.package_dir / "Cargo.toml").write_bytes(manifest)
        canonical = b"Apache License\nVersion 2.0, January 2004\nExact canonical terms fixture\n"
        rules = {("example", "1.2.3"): ("Cargo.toml", hashlib.sha256(manifest).hexdigest(), "Apache-2.0")}
        with patch.object(notices, "DECLARATIONS", rules), \
             patch.dict(notices.SPDX_TEXT_SHA256, {"Apache-2.0": hashlib.sha256(canonical).hexdigest()}), \
             patch.object(notices, "_download", return_value=canonical):
            text, inventory = notices.collect(self.metadata, self.cache)
            self.assertIn("Published authors: Original Author", text)
            self.assertIn("Exact canonical terms fixture", text)
            self.assertEqual(inventory[0]["selected_license"], "Apache-2.0")
            self.assertTrue(inventory[0]["declaration_based"])
            self.assertIn("no copyright year", inventory[0]["attribution_note"])
            (self.package_dir / "Cargo.toml").write_bytes(manifest + b"# changed\n")
            with self.assertRaisesRegex(notices.NoticesError, "declaration changed"):
                notices.collect(self.metadata, self.cache)

    def test_mit_standard_terms_do_not_invent_template_copyright(self):
        canonical = b"MIT License\n\nCopyright (c) <year> <copyright holders>\n\nPermission terms remain verbatim.\n"
        with patch.dict(notices.SPDX_TEXT_SHA256, {"MIT": hashlib.sha256(canonical).hexdigest()}), \
             patch.object(notices, "_download", return_value=canonical):
            record = notices._standard_terms("MIT", self.cache)
        self.assertNotIn("<year>", record["text"])
        self.assertIn("Permission terms remain verbatim.", record["text"])
        self.assertIn("no copyright year", record["text_transform"])

    def test_stdlib_notices_are_from_the_selected_toolchain(self):
        self.std_patch.stop()
        docs = self.root / "share/doc/rust"
        docs.mkdir(parents=True)
        (docs / "COPYRIGHT-library.html").write_text("<h1>Rust standard library</h1><pre>Exact component terms</pre>")
        identity = "rustc 1.94.1\ncommit-hash: " + self.commit + "\nrelease: 1.94.1\n"
        with patch.object(notices.subprocess, "check_output", side_effect=[identity, str(self.root)]):
            record = notices._stdlib()
        self.assertEqual(record["version"], "1.94.1")
        self.assertIn(self.commit, record["source"])
        self.assertIn("Exact component terms", record["files"][0]["text"])

    def test_html_text_preserves_entities_and_discards_style(self):
        record = notices._text_record("COPYRIGHT-library.html", "toolchain", b"<style>hide</style><h1>Copyright A &amp; B</h1><pre>License\n  intact</pre>")
        self.assertIn("Copyright A & B", record["text"])
        self.assertIn("License\n  intact", record["text"])
        self.assertNotIn("hide", record["text"])


if __name__ == "__main__":
    unittest.main()
