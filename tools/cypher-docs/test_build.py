"""Checks that result refreshes preserve the authored reference and omit unstable output."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("reference_build", Path(__file__).with_name("build.py"))
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class ReferenceBuildTests(unittest.TestCase):
    def test_typed_values_are_readable(self):
        self.assertEqual(BUILD.cell({"type": "date", "value": 0}), "1970-01-01")
        self.assertEqual(BUILD.cell({"type": "time", "value": {"nanos": 45_000_000_000_000, "offset_seconds": 3600}}), "12:30:00+01:00")
        self.assertEqual(BUILD.cell([{"type": "integer", "value": 1}, {"type": "null"}]), "[1, null]")

    def test_unstable_output_is_never_published(self):
        for name in ["functions/numeric/rand", "functions/entity/id", "functions/entity/revision", "functions/temporal/date-realtime", "statements/projects/show-projects", "statements/brokers/show-topics"]:
            self.assertFalse(BUILD.stable_page("docs/cypher/" + name + ".md"), name)
        self.assertTrue(BUILD.stable_page("docs/cypher/functions/numeric/abs.md"))

    def test_refresh_only_adds_verified_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "docs/cypher/functions/numeric/abs.md"
            target = root / relative
            target.parent.mkdir(parents=True)
            target.write_text("# `abs`\n\nAuthored prose remains here.\n\n```cypher\nRETURN abs(-1) AS result\n```\n\n[Navigation](./README.md)\n")
            frames = [{"type": "schema", "columns": [{"name": "result"}]}, {"type": "batch", "row_count": 1, "columns": [{"values": [{"type": "integer", "value": 1}]}]}]
            evidence = root / "results.json"
            evidence.write_text(json.dumps([{"file": relative, "query": "RETURN abs(-1) AS result", "frames": frames}]))
            with patch.object(BUILD, "ROOT", root):
                result = BUILD.refresh_patch(evidence)
            self.assertIn("+Result:", result)
            self.assertNotIn("-Authored prose", result)
            self.assertNotIn("-[Navigation]", result)
            self.assertEqual(target.read_text().count("Result:"), 0, "refresh emits a patch rather than editing prose")

    def test_stale_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "docs/cypher/functions/numeric/abs.md"
            target = root / relative
            target.parent.mkdir(parents=True)
            target.write_text("```cypher\nRETURN abs(-2) AS result\n```\n")
            evidence = root / "results.json"
            evidence.write_text(json.dumps([{"file": relative, "query": "RETURN abs(-1) AS result", "frames": []}]))
            with patch.object(BUILD, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
                    BUILD.refresh_patch(evidence)


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("CYPHER_BUILD_TESTS_OK")
