#!/usr/bin/env python3
"""Summarize IronGraph openCypher TCK reports by root-cause family.

Pass/fail outcomes are read only from the report's boolean outcome fields.  Error
normalization is deliberately downstream of that decision: it groups diagnostics
for investigation, but can never turn a failed scenario into a passing one.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


FAILURE_CHANNELS = (
    ("shared", "shared_failures"),
    ("cpu", "cpu_failures"),
    ("metal", "metal_failures"),
)

DIVERGENCE_CHANNELS = (
    ("divergence", "divergences"),
)

ALL_DIAGNOSTIC_CHANNELS = FAILURE_CHANNELS + DIVERGENCE_CHANNELS

REQUIRED_TOP_LEVEL = {
    "total": int,
    "cpu_passed": int,
    "metal_passed": int,
    "fully_conformant": int,
    "scenarios": list,
}

REQUIRED_SCENARIO = {
    "path": str,
    "name": str,
    "cpu_passed": bool,
    "metal_passed": bool,
    "cpu_metal_matched": bool,
    "fully_conformant": bool,
    "shared_failures": list,
    "cpu_failures": list,
    "metal_failures": list,
    "divergences": list,
}

OPERATION_PREFIX_RE = re.compile(
    r"^(?:primary|control) operation \d+ `.*?`: ", re.DOTALL
)


class ReportValidationError(ValueError):
    """Raised when a report cannot be counted without changing its meaning."""


@dataclass
class FamilyAggregate:
    key: str
    scenario_ids: set[int] = field(default_factory=set)
    occurrences: int = 0
    channels: Counter[str] = field(default_factory=Counter)
    samples: list[dict[str, str]] = field(default_factory=list)


def _is_exact_type(value: Any, expected: type) -> bool:
    # bool is a subclass of int; report counters must not accept true/false.
    return type(value) is expected


def _require_fields(
    value: Mapping[str, Any],
    required: Mapping[str, type],
    location: str,
) -> None:
    for key, expected_type in required.items():
        if key not in value:
            raise ReportValidationError(f"{location}: missing required field {key!r}")
        if not _is_exact_type(value[key], expected_type):
            raise ReportValidationError(
                f"{location}.{key}: expected {expected_type.__name__}, "
                f"got {type(value[key]).__name__}"
            )


def load_and_validate_report(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportValidationError(f"cannot read {path}: {error}") from error

    if not isinstance(report, dict):
        raise ReportValidationError("report root must be a JSON object")
    _require_fields(report, REQUIRED_TOP_LEVEL, "report")

    scenarios = report["scenarios"]
    for index, scenario in enumerate(scenarios):
        location = f"report.scenarios[{index}]"
        if not isinstance(scenario, dict):
            raise ReportValidationError(f"{location}: expected object")
        _require_fields(scenario, REQUIRED_SCENARIO, location)

        for _, field_name in ALL_DIAGNOSTIC_CHANNELS:
            messages = scenario[field_name]
            if any(not isinstance(message, str) for message in messages):
                raise ReportValidationError(
                    f"{location}.{field_name}: every diagnostic must be a string"
                )

        expected_full = (
            scenario["cpu_passed"]
            and scenario["metal_passed"]
            and scenario["cpu_metal_matched"]
        )
        if scenario["fully_conformant"] != expected_full:
            raise ReportValidationError(
                f"{location}: fully_conformant must equal cpu_passed AND "
                "metal_passed AND cpu_metal_matched"
            )

        if scenario["cpu_passed"] and (
            scenario["cpu_failures"] or scenario["shared_failures"]
        ):
            raise ReportValidationError(
                f"{location}: CPU is marked passed but has CPU/shared failures"
            )
        if scenario["metal_passed"] and (
            scenario["metal_failures"] or scenario["shared_failures"]
        ):
            raise ReportValidationError(
                f"{location}: Metal is marked passed but has Metal/shared failures"
            )

    derived = derive_outcomes(scenarios)
    for field_name in ("total", "cpu_passed", "metal_passed", "fully_conformant"):
        if report[field_name] != derived[field_name]:
            raise ReportValidationError(
                f"report.{field_name}={report[field_name]} does not match "
                f"scenario-derived count {derived[field_name]}"
            )

    if "matched" in report:
        if not _is_exact_type(report["matched"], int):
            raise ReportValidationError("report.matched must be an integer")
        derived_matched = sum(
            1 for scenario in scenarios if scenario["cpu_metal_matched"]
        )
        if report["matched"] != derived_matched:
            raise ReportValidationError(
                f"report.matched={report['matched']} does not match "
                f"scenario-derived count {derived_matched}"
            )

    return report


def derive_outcomes(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    outcomes = {
        "total": len(scenarios),
        "cpu_passed": 0,
        "metal_passed": 0,
        "fully_conformant": 0,
        "cpu_only_failures": 0,
        "metal_only_failures": 0,
        "shared_failures": 0,
        "backend_divergences": 0,
    }
    for scenario in scenarios:
        cpu_passed = scenario["cpu_passed"]
        metal_passed = scenario["metal_passed"]
        outcomes["cpu_passed"] += int(cpu_passed)
        outcomes["metal_passed"] += int(metal_passed)
        outcomes["fully_conformant"] += int(scenario["fully_conformant"])
        if cpu_passed and metal_passed and not scenario["cpu_metal_matched"]:
            outcomes["backend_divergences"] += 1
        elif not cpu_passed and metal_passed:
            outcomes["cpu_only_failures"] += 1
        elif cpu_passed and not metal_passed:
            outcomes["metal_only_failures"] += 1
        elif not cpu_passed and not metal_passed:
            outcomes["shared_failures"] += 1
    return outcomes


def feature_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "/features/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return Path(normalized).name or normalized


def failure_bucket(scenario: Mapping[str, Any]) -> str | None:
    cpu_passed = scenario["cpu_passed"]
    metal_passed = scenario["metal_passed"]
    if cpu_passed and metal_passed:
        return None if scenario["fully_conformant"] else "backend_divergences"
    if not cpu_passed and metal_passed:
        return "cpu_only_failures"
    if cpu_passed and not metal_passed:
        return "metal_only_failures"
    return "shared_failures"


def aggregate_features(
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    aggregates: dict[str, Counter[str]] = defaultdict(Counter)
    for scenario in scenarios:
        feature = feature_path(scenario["path"])
        counts = aggregates[feature]
        counts["total"] += 1
        counts["cpu_passed"] += int(scenario["cpu_passed"])
        counts["metal_passed"] += int(scenario["metal_passed"])
        counts["fully_conformant"] += int(scenario["fully_conformant"])
        bucket = failure_bucket(scenario)
        if bucket is not None:
            counts[bucket] += 1

    rows = []
    for feature, counts in aggregates.items():
        row = {
            "feature": feature,
            "total": counts["total"],
            "cpu_passed": counts["cpu_passed"],
            "metal_passed": counts["metal_passed"],
            "fully_conformant": counts["fully_conformant"],
            "failed": counts["total"] - counts["fully_conformant"],
            "cpu_only_failures": counts["cpu_only_failures"],
            "metal_only_failures": counts["metal_only_failures"],
            "shared_failures": counts["shared_failures"],
            "backend_divergences": counts["backend_divergences"],
        }
        if row["failed"]:
            rows.append(row)

    return sorted(
        rows,
        key=lambda row: (
            -row["failed"],
            -row["shared_failures"],
            -row["metal_only_failures"],
            -row["backend_divergences"],
            row["feature"],
        ),
    )


def clean_diagnostic(message: str) -> str:
    return OPERATION_PREFIX_RE.sub("", message).strip()


def _slug(value: str, *, words: int = 10) -> str:
    value = re.sub(r"`[^`]*`", " identifier ", value)
    value = re.sub(r"'(?:[^'\\]|\\.)*'", " value ", value)
    value = re.sub(r'"(?:[^"\\]|\\.)*"', " value ", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", " number ", value)
    tokens = re.findall(r"[a-z]+", value.lower())[:words]
    return "-".join(tokens) or "unclassified"


def _error_code(message: str) -> str:
    match = re.search(r"\b(?:with|got) ([A-Za-z][A-Za-z0-9_]*):", message)
    return match.group(1).lower() if match else "unknown"


def normalize_failure(message: str, channel: str) -> str:
    """Return a stable root-cause key without affecting scenario outcomes."""
    cleaned = clean_diagnostic(message)
    lowered = cleaned.lower()

    if channel == "divergence":
        if "gpuadmissionfailure" in lowered:
            return "divergence/gpu-admission"
        return "divergence/semantic-outcome"

    if "tck fixture procedures are not yet registered" in lowered:
        return "fixture/procedure-not-registered"

    if "gpuadmissionfailure" in lowered:
        if "no complete resident implementation" in lowered:
            return "gpu-admission/no-complete-resident-plan"
        if "optional undirected resident expansion" in lowered:
            return "gpu-admission/optional-undirected-expansion"
        detail = cleaned.split("GpuAdmissionFailure:", 1)[-1]
        return f"gpu-admission/{_slug(detail)}"

    if "function is not a supported built-in" in lowered:
        return "compile/unsupported-builtin-function"
    if "undefinedvariable" in lowered or "variable `" in lowered and "not defined" in lowered:
        return "compile/undefined-variable"
    if "unexpectedsyntax" in lowered:
        return "compile/parser-unexpected-syntax"

    if re.search(r"expected \d+ rows?, got \d+", lowered):
        return "result/row-count-mismatch"
    if lowered.startswith("expected columns "):
        return "result/column-name-mismatch"
    if lowered.startswith("unordered result differs"):
        return "result/unordered-value-mismatch"
    if lowered.startswith("ordered result differs"):
        return "result/ordered-value-mismatch"

    if re.search(r"expected \w+ error but query succeeded", cleaned, re.IGNORECASE):
        expected = re.search(r"expected (\w+) error", cleaned, re.IGNORECASE)
        suffix = expected.group(1).lower() if expected else "error"
        return f"expected-error/query-succeeded/{suffix}"
    if lowered.startswith("expected side effects") or lowered.startswith(
        "side effects differ"
    ):
        return "result/side-effect-mismatch"
    if lowered.startswith("expected schema") or lowered.startswith("schema differs"):
        return "result/schema-mismatch"
    if lowered.startswith("expected tck error detail"):
        expected = re.search(r"detail `([^`]+)`", cleaned)
        suffix = _slug(expected.group(1)) if expected else "unknown"
        return f"expected-error/detail-mismatch/{suffix}"
    if lowered.startswith("expected compile error phase") or (
        lowered.startswith("expected runtime error phase")
    ):
        return "expected-error/phase-mismatch"
    if lowered.startswith("expected ") and ", got " in lowered:
        return f"expected-error/category-mismatch/got-{_error_code(cleaned)}"

    if "unterminated tck path" in lowered:
        return "tck-adapter/unterminated-path"
    if "unexpected tck value suffix" in lowered:
        return "tck-adapter/value-suffix"

    type_families = (
        ("string argument required", "runtime/type/string-argument-required"),
        ("integer value required", "runtime/type/integer-value-required"),
        ("numeric value required", "runtime/type/numeric-value-required"),
        (
            "values of these types cannot be compared with an ordering operator",
            "runtime/type/ordering-comparison",
        ),
        ("ordering is not defined between", "runtime/type/ordering-comparison"),
        (
            "list index requires list and integer operands",
            "runtime/type/list-index-operands",
        ),
        (
            "property access requires node, relationship, or map",
            "runtime/type/property-access",
        ),
        (
            "delete expression must resolve to a node or relationship",
            "runtime/type/delete-target",
        ),
        (
            "node pattern variable has a non-node value",
            "runtime/type/node-pattern-binding",
        ),
        (
            "property value does not match statement-local column type",
            "runtime/type/property-column-type-mismatch",
        ),
        ("string predicate requires string", "runtime/type/string-predicate"),
        (
            "relationship path maximum is below its minimum",
            "runtime/type/relationship-path-bounds",
        ),
    )
    for fragment, family in type_families:
        if fragment in lowered:
            return family

    invalid_argument = re.search(
        r"InvalidArgument(Value|Type):\s*([A-Za-z][A-Za-z0-9.]*)?",
        cleaned,
        re.IGNORECASE,
    )
    if invalid_argument:
        kind = invalid_argument.group(1).lower()
        subject = _slug(invalid_argument.group(2) or "generic", words=3)
        return f"runtime/type/invalid-argument-{kind}/{subject}"

    unexpected = re.match(
        r"query unexpectedly failed at (Compile|Runtime) with "
        r"([A-Za-z][A-Za-z0-9_]*):\s*(.*)",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if unexpected:
        phase, code, detail = unexpected.groups()
        return (
            f"unexpected-failure/{phase.lower()}/{code.lower()}/"
            f"{_slug(detail)}"
        )

    return f"other/{_slug(cleaned)}"


def _short_message(message: str, limit: int = 220) -> str:
    cleaned = " ".join(clean_diagnostic(message).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _ordered_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
) -> list[tuple[int, Mapping[str, Any]]]:
    return sorted(
        enumerate(scenarios),
        key=lambda item: (
            feature_path(item[1]["path"]),
            item[1]["name"],
            item[0],
        ),
    )


def _aggregate_message_channels(
    scenarios: Sequence[Mapping[str, Any]],
    channels: Sequence[tuple[str, str]],
    sample_limit: int,
) -> tuple[dict[str, FamilyAggregate], set[int], int]:
    families: dict[str, FamilyAggregate] = {}
    scenarios_with_family: set[int] = set()
    message_count = 0

    for scenario_id, scenario in _ordered_scenarios(scenarios):
        for channel, field_name in channels:
            for message in sorted(scenario[field_name]):
                message_count += 1
                family_key = normalize_failure(message, channel)
                aggregate = families.setdefault(
                    family_key, FamilyAggregate(key=family_key)
                )
                aggregate.scenario_ids.add(scenario_id)
                aggregate.occurrences += 1
                aggregate.channels[channel] += 1
                scenarios_with_family.add(scenario_id)
                if len(aggregate.samples) < sample_limit:
                    aggregate.samples.append(
                        {
                            "feature": feature_path(scenario["path"]),
                            "scenario": scenario["name"],
                            "channel": channel,
                            "message": _short_message(message),
                        }
                    )
    return families, scenarios_with_family, message_count


def _family_rows(families: Mapping[str, FamilyAggregate]) -> list[dict[str, Any]]:
    rows = [
        {
            "family": aggregate.key,
            "scenarios": len(aggregate.scenario_ids),
            "occurrences": aggregate.occurrences,
            "channels": dict(sorted(aggregate.channels.items())),
            "samples": aggregate.samples,
        }
        for aggregate in families.values()
    ]
    rows.sort(
        key=lambda row: (-row["scenarios"], -row["occurrences"], row["family"])
    )
    return rows


def aggregate_families(
    scenarios: Sequence[Mapping[str, Any]], sample_limit: int
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    failing_scenarios = {
        scenario_id
        for scenario_id, scenario in enumerate(scenarios)
        if not scenario["fully_conformant"]
    }
    families, scenarios_with_family, failure_message_count = (
        _aggregate_message_channels(scenarios, FAILURE_CHANNELS, sample_limit)
    )

    for scenario_id, scenario in _ordered_scenarios(scenarios):
        if scenario_id not in failing_scenarios or scenario_id in scenarios_with_family:
            continue

        has_divergence = bool(scenario["divergences"])
        family_key = (
            "outcome/backend-semantic-divergence"
            if has_divergence
            else "diagnostic/missing-for-failed-scenario"
        )
        aggregate = families.setdefault(
            family_key, FamilyAggregate(key=family_key)
        )
        aggregate.scenario_ids.add(scenario_id)
        aggregate.occurrences += 1
        channel = "divergence" if has_divergence else "missing"
        aggregate.channels[channel] += 1
        scenarios_with_family.add(scenario_id)
        if len(aggregate.samples) < sample_limit:
            aggregate.samples.append(
                {
                    "feature": feature_path(scenario["path"]),
                    "scenario": scenario["name"],
                    "channel": channel,
                    "message": (
                        _short_message(scenario["divergences"][0])
                        if has_divergence
                        else "failed scenario contains no CPU/Metal/shared diagnostic"
                    ),
                }
            )

    divergence_families, divergence_scenarios, divergence_message_count = (
        _aggregate_message_channels(scenarios, DIVERGENCE_CHANNELS, sample_limit)
    )
    coverage = {
        "failing_scenarios": len(failing_scenarios),
        "failing_scenarios_with_family": len(failing_scenarios & scenarios_with_family),
        "failure_messages": failure_message_count,
        "normalized_failure_messages": failure_message_count,
        "divergence_scenarios": len(divergence_scenarios),
        "divergence_messages": divergence_message_count,
    }
    return _family_rows(families), coverage, _family_rows(divergence_families)


def _limit(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return list(rows if count == 0 else rows[:count])


def build_summary(
    report: Mapping[str, Any],
    source: Path,
    *,
    top_features: int,
    top_families: int,
    samples: int,
) -> dict[str, Any]:
    scenarios = report["scenarios"]
    outcomes = derive_outcomes(scenarios)
    features = aggregate_features(scenarios)
    families, coverage, divergence_families = aggregate_families(scenarios, samples)
    reported_counts = {
        key: report[key]
        for key in ("total", "cpu_passed", "metal_passed", "fully_conformant")
    }
    if "matched" in report:
        reported_counts["matched"] = report["matched"]

    return {
        "source": str(source),
        "integrity": {
            "status": "ok",
            "outcome_source": "scenario boolean fields",
            "reported_counts": reported_counts,
            "derived_counts": {
                key: outcomes[key]
                for key in ("total", "cpu_passed", "metal_passed", "fully_conformant")
            },
        },
        "outcomes": outcomes,
        "classification_coverage": coverage,
        "top_features": _limit(features, top_features),
        "failure_families": _limit(families, top_families),
        "divergence_families": divergence_families,
        "available": {
            "failing_features": len(features),
            "failure_families": len(families),
        },
    }


def _percentage(count: int, total: int) -> str:
    return f"{(100.0 * count / total):6.2f}%" if total else "  0.00%"


def _format_channels(channels: Mapping[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in channels.items())


def render_text(summary: Mapping[str, Any]) -> str:
    outcomes = summary["outcomes"]
    total = outcomes["total"]
    lines = [
        "openCypher TCK conformance classification",
        f"source: {summary['source']}",
        "integrity: OK; pass/fail comes only from scenario boolean fields",
        "",
        "Outcome summary",
        f"  total scenarios       {total:6d}",
        f"  CPU passed            {outcomes['cpu_passed']:6d}  "
        f"{_percentage(outcomes['cpu_passed'], total)}",
        f"  Metal passed          {outcomes['metal_passed']:6d}  "
        f"{_percentage(outcomes['metal_passed'], total)}",
        f"  fully conformant      {outcomes['fully_conformant']:6d}  "
        f"{_percentage(outcomes['fully_conformant'], total)}",
        f"  CPU-only failures     {outcomes['cpu_only_failures']:6d}  "
        "(CPU failed, Metal passed)",
        f"  Metal-only failures   {outcomes['metal_only_failures']:6d}  "
        "(CPU passed, Metal failed)",
        f"  shared failures       {outcomes['shared_failures']:6d}  "
        "(both failed)",
        f"  backend divergences   {outcomes['backend_divergences']:6d}  "
        "(both passed TCK, outcomes differ)",
        "",
    ]

    coverage = summary["classification_coverage"]
    lines.extend(
        [
            "Classification coverage",
            f"  failing scenarios     {coverage['failing_scenarios']:6d}",
            f"  with a family         "
            f"{coverage['failing_scenarios_with_family']:6d}",
            f"  failure messages      {coverage['failure_messages']:6d}",
            f"  divergence messages   {coverage['divergence_messages']:6d}  "
            "(reported separately)",
            "",
            "Top failing feature files",
            "  failed  CPU  Metal  full  shared  M-only  C-only  D-only  feature",
        ]
    )
    for row in summary["top_features"]:
        lines.append(
            f"  {row['failed']:6d} {row['cpu_passed']:4d} "
            f"{row['metal_passed']:6d} {row['fully_conformant']:5d} "
            f"{row['shared_failures']:7d} {row['metal_only_failures']:7d} "
            f"{row['cpu_only_failures']:7d} {row['backend_divergences']:7d}  "
            f"{row['feature']}"
        )

    lines.extend(["", "Normalized error/admission families"])
    for row in summary["failure_families"]:
        lines.append(
            f"  {row['scenarios']:4d} scenarios / {row['occurrences']:4d} messages  "
            f"{row['family']}  [{_format_channels(row['channels'])}]"
        )
        for sample in row["samples"]:
            lines.append(
                f"    - {sample['feature']} :: {sample['scenario']} "
                f"({sample['channel']})"
            )
            lines.append(f"      {sample['message']}")

    lines.extend(["", "Divergence diagnostics (excluded from root-cause ranking)"])
    for row in summary["divergence_families"]:
        lines.append(
            f"  {row['scenarios']:4d} scenarios / {row['occurrences']:4d} messages  "
            f"{row['family']}"
        )
        for sample in row["samples"]:
            lines.append(
                f"    - {sample['feature']} :: {sample['scenario']} "
                f"({sample['channel']})"
            )
            lines.append(f"      {sample['message']}")

    available = summary["available"]
    lines.extend(
        [
            "",
            f"Showing {len(summary['top_features'])} of "
            f"{available['failing_features']} failing features and "
            f"{len(summary['failure_families'])} of "
            f"{available['failure_families']} failure families.",
        ]
    )
    return "\n".join(lines)


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a IronGraph openCypher TCK JSON report without "
            "changing its pass/fail outcomes. Use 0 for an unlimited section."
        )
    )
    parser.add_argument("report", type=Path, help="path to the TCK JSON report")
    parser.add_argument(
        "--top-features", type=non_negative_int, default=20, help="feature rows (default: 20)"
    )
    parser.add_argument(
        "--top-families", type=non_negative_int, default=20, help="family rows (default: 20)"
    )
    parser.add_argument(
        "--samples", type=non_negative_int, default=2, help="samples per family (default: 2)"
    )
    parser.add_argument(
        "--format", choices=("text", "json"), default="text", help="output format"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = load_and_validate_report(args.report)
        summary = build_summary(
            report,
            args.report,
            top_features=args.top_features,
            top_families=args.top_families,
            samples=args.samples,
        )
    except ReportValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.format == "json":
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(render_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
