#!/usr/bin/env python3
"""Validate the authored Cypher reference and refresh verified result tables.

Markdown under docs/cypher is the sole source for prose, navigation and example queries.
The source-derived audit checks the complete supported vocabulary. Live execution defaults
to self-contained examples; --all-examples additionally needs each dataset page's prerequisites.

  python3 tools/cypher-docs/build.py --check
  python3 tools/cypher-docs/build.py --endpoint http://127.0.0.1:28484/api/query --check
  python3 tools/cypher-docs/build.py --refresh > results.patch

--refresh emits an apply_patch patch from the most recent successful live run. It never
regenerates prose or navigation and omits operational, identity, clock and random output.
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "docs/cypher"
EVIDENCE = ROOT / ".unlazy/semantic-scope/doc-results.json"


def cell(value: object) -> str:
    """Render typed Query API values without publishing request or project identities."""
    if isinstance(value, dict) and "type" in value:
        kind = value["type"]
        if kind == "null":
            return "null"
        payload = value.get("value")
        if kind == "date":
            return (datetime.date(1970, 1, 1) + datetime.timedelta(days=int(payload))).isoformat()
        if kind == "date_time" and isinstance(payload, dict):
            instant = datetime.datetime.fromtimestamp(payload["seconds"], datetime.timezone.utc)
            rendered = instant.strftime("%Y-%m-%dT%H:%M:%S")
            if payload.get("nanos"):
                rendered += "." + f'{payload["nanos"]:09d}'.rstrip("0")
            return rendered + ("Z" if payload.get("timezone") == "UTC" else "")
        if kind == "time" and isinstance(payload, dict):
            seconds, nanos = divmod(payload["nanos"], 1_000_000_000)
            hours, rest = divmod(seconds, 3600)
            minutes, seconds = divmod(rest, 60)
            rendered = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            if nanos:
                rendered += "." + f"{nanos:09d}".rstrip("0")
            offset = payload.get("offset_seconds")
            if offset is not None:
                sign = "+" if offset >= 0 else "-"
                h, m = divmod(abs(offset) // 60, 60)
                rendered += f"{sign}{h:02d}:{m:02d}"
            return rendered
        return cell(payload)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    if isinstance(value, list):
        return "[" + ", ".join(cell(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def result_table(frames: list[dict]) -> str:
    schema = next((frame for frame in frames if frame["type"] == "schema"), {})
    columns = [column["name"] for column in schema.get("columns", [])]
    if not columns:
        return "No graph rows returned."
    rows = []
    for frame in frames:
        if frame["type"] == "batch":
            rows.extend([[cell(column["values"][index]) for column in frame["columns"]]
                         for index in range(frame["row_count"])])
    shown = rows[:12]
    widths = [max([len(name)] + [len(row[index]) for row in shown])
              for index, name in enumerate(columns)]
    lines = [" | ".join(name.ljust(widths[index]) for index, name in enumerate(columns)),
             "-+-".join("-" * width for width in widths)]
    lines.extend(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
                 for row in shown)
    lines += ["", f"{len(rows)} row" + ("" if len(rows) == 1 else "s")]
    if len(rows) > len(shown):
        lines.append(f"{len(rows) - len(shown)} additional rows omitted from this display.")
    return "\n".join(line.rstrip() for line in lines)


def stable_page(name: str) -> bool:
    if "/functions/" in name:
        return not re.search(r"/(rand|id|revision|[^/]+-(?:transaction|statement|realtime))\.md$", name)
    return name.endswith("/expressions.md") or bool(re.search(
        r"/clauses/core/(where|unwind|for|let|filter|create|set|remove|delete|with|return|order-by|skip|limit|union|finish|paths)\.md$", name))


def refresh_patch(evidence: Path) -> str:
    results = json.loads(evidence.read_text())
    by_file: dict[str, list[dict]] = {}
    for result in results:
        if stable_page(result["file"]):
            by_file.setdefault(result["file"], []).append(result)
    patch = ["*** Begin Patch"]
    changes = 0
    for name, examples in by_file.items():
        source = (ROOT / name).read_text()
        revised = source
        for example in examples:
            block = "```cypher\n" + example["query"] + "\n```"
            if source.count(block) != 1:
                raise ValueError(f"{name}: evidence query is absent or ambiguous; rerun live checks")
            table = "\n\nResult:\n\n```\n" + result_table(example["frames"]) + "\n```"
            pattern = re.escape(block) + r"(?:\n\nResult:\n\n```\n[\s\S]*?\n```)?"
            revised, count = re.subn(pattern, lambda _match: block + table, revised, count=1)
            if count != 1:
                raise ValueError(f"{name}: cannot uniquely refresh example")
        if revised != source:
            patch += ["*** Update File: " + name]
            diff = list(difflib.unified_diff(source.splitlines(), revised.splitlines(), n=5))
            patch.extend("@@" if line.startswith("@@") else line for line in diff[2:])
            changes += 1
    if not changes:
        return ""
    return "\n".join(patch + ["*** End Patch", ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="validate without modifying Markdown")
    parser.add_argument("--endpoint", help="execute examples against an explicitly disposable loopback Query API")
    parser.add_argument("--only", help="limit execution to a reference-relative path prefix")
    parser.add_argument("--all-examples", action="store_true", help="include dataset examples; load every selected page's prerequisites first")
    parser.add_argument("--refresh", action="store_true", help="emit an apply_patch patch for verified stable results")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    if args.refresh:
        if args.check or args.endpoint or args.only or args.all_examples:
            parser.error("--refresh uses existing evidence and cannot be combined with execution options")
        sys.stdout.write(refresh_patch(args.evidence))
        return 0
    command = ["node", str(ROOT / "scripts/verify-cypher-docs.mjs")]
    if args.endpoint:
        command += ["--execute", args.endpoint]
    if args.only:
        command += ["--only", args.only]
    if args.all_examples:
        command += ["--all-examples"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
