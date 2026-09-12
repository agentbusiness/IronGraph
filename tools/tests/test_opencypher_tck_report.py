from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import opencypher_tck_report as report  # noqa: E402


def scenario(
    name: str,
    *,
    cpu: bool,
    metal: bool,
    shared_failures: list[str] | None = None,
    cpu_failures: list[str] | None = None,
    metal_failures: list[str] | None = None,
    divergences: list[str] | None = None,
    matched: bool | None = None,
    path: str = "/checkout/tck/features/example/Feature.feature",
) -> dict[str, object]:
    matched = cpu == metal if matched is None else matched
    return {
        "path": path,
        "name": name,
        "cpu_passed": cpu,
        "metal_passed": metal,
        "cpu_metal_matched": matched,
        "fully_conformant": cpu and metal and matched,
        "operation_count": 1,
        "shared_failures": shared_failures or [],
        "cpu_failures": cpu_failures or [],
        "metal_failures": metal_failures or [],
        "divergences": divergences or [],
    }


def complete_report(scenarios: list[dict[str, object]]) -> dict[str, object]:
    return {
        "total": len(scenarios),
        "cpu_passed": sum(bool(item["cpu_passed"]) for item in scenarios),
        "metal_passed": sum(bool(item["metal_passed"]) for item in scenarios),
        "matched": sum(bool(item["cpu_metal_matched"]) for item in scenarios),
        "fully_conformant": sum(
            bool(item["fully_conformant"]) for item in scenarios
        ),
        "scenarios": scenarios,
    }


class OutcomeTests(unittest.TestCase):
    def test_outcome_buckets_use_boolean_fields_only(self) -> None:
        scenarios = [
            scenario("pass", cpu=True, metal=True),
            scenario(
                "CPU-only failure",
                cpu=False,
                metal=True,
                cpu_failures=["query unexpectedly failed but text says succeeded"],
            ),
            scenario(
                "Metal-only failure",
                cpu=True,
                metal=False,
                metal_failures=["query unexpectedly failed"],
            ),
            scenario(
                "shared failure",
                cpu=False,
                metal=False,
                shared_failures=["fixture failed"],
            ),
            scenario(
                "independently passing divergence",
                cpu=True,
                metal=True,
                matched=False,
                divergences=["CPU/GPU semantic divergence"],
            ),
        ]

        self.assertEqual(
            report.derive_outcomes(scenarios),
            {
                "total": 5,
                "cpu_passed": 3,
                "metal_passed": 3,
                "fully_conformant": 1,
                "cpu_only_failures": 1,
                "metal_only_failures": 1,
                "shared_failures": 1,
                "backend_divergences": 1,
            },
        )

    def test_validation_rejects_summary_count_drift(self) -> None:
        payload = complete_report([scenario("pass", cpu=True, metal=True)])
        payload["cpu_passed"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                report.ReportValidationError, "scenario-derived count"
            ):
                report.load_and_validate_report(path)

    def test_validation_accepts_false_full_flag_for_mismatched_two_passes(self) -> None:
        item = scenario(
            "strict divergence",
            cpu=True,
            metal=True,
            matched=False,
            divergences=["CPU/GPU semantic divergence"],
        )
        payload = complete_report([item])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = report.load_and_validate_report(path)
            self.assertFalse(loaded["scenarios"][0]["fully_conformant"])

    def test_validation_rejects_false_full_flag_for_matching_two_passes(self) -> None:
        item = scenario("inconsistent", cpu=True, metal=True)
        item["fully_conformant"] = False
        payload = complete_report([item])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                report.ReportValidationError, "cpu_passed AND metal_passed AND"
            ):
                report.load_and_validate_report(path)


class ClassificationTests(unittest.TestCase):
    def test_known_root_cause_families(self) -> None:
        examples = {
            (
                "query unexpectedly failed at Runtime with GpuAdmissionFailure: "
                "active GPU execution class has no complete resident implementation "
                "for this query plan",
                "metal",
            ): "gpu-admission/no-complete-resident-plan",
            (
                "query unexpectedly failed at Compile with QueryType: function is "
                "not a supported built-in",
                "cpu",
            ): "compile/unsupported-builtin-function",
            (
                "TCK fixture procedures are not yet registered: [\"test.proc\"]",
                "shared",
            ): "fixture/procedure-not-registered",
            (
                "expected 3 rows, got 1",
                "cpu",
            ): "result/row-count-mismatch",
            (
                "expected columns [\"a\"], got [\"b\"]",
                "cpu",
            ): "result/column-name-mismatch",
            (
                "CPU/GPU semantic divergence: CPU=Success(...), "
                "GPU=Error(GpuAdmissionFailure)",
                "divergence",
            ): "divergence/gpu-admission",
            (
                "expected SyntaxError error but query succeeded with QueryResult "
                "{ schema: [], batches: [] }",
                "cpu",
            ): "expected-error/query-succeeded/syntaxerror",
        }
        for (message, channel), expected in examples.items():
            with self.subTest(message=message):
                self.assertEqual(
                    report.normalize_failure(message, channel), expected
                )

    def test_feature_order_is_deterministic(self) -> None:
        scenarios = [
            scenario(
                "b",
                cpu=False,
                metal=False,
                shared_failures=["b"],
                path="/x/features/b/B.feature",
            ),
            scenario(
                "a",
                cpu=False,
                metal=False,
                shared_failures=["a"],
                path="/x/features/a/A.feature",
            ),
        ]
        rows = report.aggregate_features(scenarios)
        self.assertEqual(
            [row["feature"] for row in rows], ["a/A.feature", "b/B.feature"]
        )

    def test_every_failed_scenario_gets_family_coverage(self) -> None:
        scenarios = [
            scenario("diagnosed", cpu=True, metal=False, metal_failures=["failure"]),
            scenario("missing", cpu=False, metal=False),
        ]
        families, coverage, divergence_families = report.aggregate_families(
            scenarios, sample_limit=1
        )
        self.assertEqual(coverage["failing_scenarios"], 2)
        self.assertEqual(coverage["failing_scenarios_with_family"], 2)
        self.assertEqual(divergence_families, [])
        self.assertIn(
            "diagnostic/missing-for-failed-scenario",
            {item["family"] for item in families},
        )

    def test_two_passing_backends_with_different_results_are_ranked(self) -> None:
        scenarios = [
            scenario(
                "different",
                cpu=True,
                metal=True,
                matched=False,
                divergences=["CPU/GPU semantic divergence: results differ"],
            )
        ]
        rows = report.aggregate_features(scenarios)
        self.assertEqual(rows[0]["backend_divergences"], 1)
        families, coverage, _ = report.aggregate_families(scenarios, sample_limit=1)
        self.assertEqual(coverage["failing_scenarios_with_family"], 1)
        self.assertIn(
            "outcome/backend-semantic-divergence",
            {item["family"] for item in families},
        )


if __name__ == "__main__":
    unittest.main()
