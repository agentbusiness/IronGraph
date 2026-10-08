#!/usr/bin/env python3
"""Check unchanged saved targets against complete, provenance-bound CPU queries."""

import argparse
import hashlib
import json
from pathlib import Path


BASELINE = Path("performance-results/metal-final/current-after-broker/results.json")
BASELINE_SHA = "7c25f39cdfc2824669c4ea831b36f45bc4c537aa403bb4e7ffceca740b9e9089"
WORKLOADS = (
    "two_hop", "one_hop", "adaptive_triangle_count", "avg", "group_count", "sum",
    "adaptive_clustering", "top_k", "adaptive_k_core", "adaptive_shortest_path",
    "variable_1_3", "adaptive_scc", "count_nodes",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline():
    assert digest(BASELINE) == BASELINE_SHA, "saved targets changed"
    data = json.loads(BASELINE.read_text())
    return {
        row["operation"]: row for row in data["measurements"]
        if row["nodes"] == 100000 and row["backend"] == "cpu"
        and row["operation"] in WORKLOADS
    }


def verify_dataset(path, full):
    data = json.loads(path.read_text())
    manifest = json.loads((path.parent / "build-manifest.json").read_text())
    assert digest(path) == manifest["results_sha256"], "measured file changed"
    assert digest(manifest["binary"]) == manifest["binary_sha256"], "executable changed"
    for key in ("source_files_sha256", "embedded_web_assets_sha256"):
        for filename, expected in manifest[key].items():
            assert digest(filename) == expected, f"source changed: {filename}"
    assert all(not Path(p).exists() for p in manifest["deleted_source_files"])
    config = data["config"]
    assert config["fanout"] == 4 and config["dirty_body_bytes"] == 2048
    assert config["samples"] == 5 and config["warmups"] == 2
    assert config["batch_rows"] == 256 and config["operations"] == []
    assert config["backends"] == ["cpu-canonical"]
    assert "QueryEngine" in data["timing_boundaries"]["queries"]
    assert data["cpu_model"] == "Apple M4" and data["parallelism"] == 10
    assert data["physical_memory_bytes"] == 17179869184
    sizes = {100, 10000, 100000, 1000000, 2000000} if full else {100000}
    assert set(config["sizes"]) == sizes
    assert len(data["measurements"]) == 33 * len(sizes)
    assert all(row["status"] == "ok" and row["error"] is None
               for row in data["measurements"])
    assert all(len({r["operation"] for r in data["measurements"] if r["nodes"] == size}) == 33
               for size in sizes)
    return {r["operation"]: r for r in data["measurements"] if r["nodes"] == 100000}


def check(workload, current, old):
    new, saved = current[workload], old[workload]
    assert new["unit"] == saved["unit"] == "microseconds"
    assert new["samples"] == saved["samples"] == 5
    assert new["warmups"] == saved["warmups"] == 2
    assert new["result_rows"] == saved["result_rows"], "result cardinality changed"
    assert new["p50"] < saved["p50"], (
        f"{workload}: current {new['p50']:.6f}us must be below old {saved['p50']:.6f}us"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--results", type=Path,
                        default=Path("performance-results/cpu-recovery/current/results.json"))
    args = parser.parse_args()
    old = baseline()
    assert set(old) == set(WORKLOADS)
    if args.self_test:
        existing = json.loads(Path("performance-results/cpu-canonical/current/results.json").read_text())
        current = {r["operation"]: r for r in existing["measurements"] if r["nodes"] == 100000}
        rejected = []
        for workload in WORKLOADS:
            try:
                check(workload, current, old)
            except AssertionError:
                rejected.append(workload)
        assert tuple(rejected) == WORKLOADS, "positive regression control was accepted"
        print("PASS performance oracle: rejects all 13 known regressions")
        return
    assert args.all != bool(args.workload), "choose --all or --workload"
    current = verify_dataset(args.results, args.full)
    for workload in WORKLOADS if args.all else (args.workload,):
        check(workload, current, old)
    print("PASS performance recovery: " + ("all" if args.all else args.workload))


if __name__ == "__main__":
    main()
