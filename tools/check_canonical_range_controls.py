#!/usr/bin/env python3
"""Validate paired canonical range measurements without mixing timing boundaries."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        default=Path("performance-results/cpu-canonical/range-controls/qualified"))
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    binary = Path("target/release/examples/performance_cpu_canonical")
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest["new_binary_sha256"], \
        "control dataset does not measure the current executable"
    ratios = []
    for pair in (1, 2):
        old = json.loads((args.directory / f"old-stable-{pair}.json").read_text())
        new = json.loads((args.directory / f"final-stable-{pair}.json").read_text())
        assert old["config"] == new["config"], "different fixture or operation sequence"
        assert old["cpu_model"] == new["cpu_model"], "different hardware"
        assert old["parallelism"] == new["parallelism"], "different host thread count"
        assert old["physical_memory_bytes"] == new["physical_memory_bytes"]
        assert old["config"]["samples"] == 50 and old["config"]["warmups"] == 2
        assert old["config"]["fanout"] == 4 and old["config"]["dirty_body_bytes"] == 2048
        assert old["config"]["sizes"] == [1000000]
        assert old["config"]["operations"] == [
            "range_count", "count_nodes", "canonical_scalar_point"]
        for data in (old, new):
            assert len(data["measurements"]) == 5
            assert all(row["status"] == "ok" and row["error"] is None
                       for row in data["measurements"])
        before = next(row for row in old["measurements"]
                      if row["operation"] == "range_count")
        after = next(row for row in new["measurements"]
                     if row["operation"] == "range_count")
        assert before["result_rows"] == after["result_rows"] == 1
        assert before["unit"] == after["unit"] == "microseconds"
        ratios.append(after["p50"] / before["p50"])
    ratio = median(ratios)
    assert ratio <= 1.20, f"material paired range regression: p50 ratio {ratio:.3f}"
    print("PASS: bounded canonical range controls preserve baseline latency")


if __name__ == "__main__":
    main()
