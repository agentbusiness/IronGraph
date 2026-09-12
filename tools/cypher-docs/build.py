#!/usr/bin/env python3
"""Builds the Cypher reference documentation under `docs/cypher/`.

Every example on every page is executed against a running IronGraph node that holds the reference
datasets, and the result it actually produced is embedded beneath it. A page cannot be published
with an example that does not run.

    python3 tools/cypher-docs/build.py             # build every page
    python3 tools/cypher-docs/build.py --check      # run every example, write nothing
    python3 tools/cypher-docs/build.py --only functions/aggregation

Load the datasets first with `datasets/download.sh` and `datasets/load.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from irongraph_client import DEFAULT_ENDPOINT, QueryResult, run, table  # noqa: E402
from model import KIND_LABEL, STANDARD_LABEL, Example, Family, Page  # noqa: E402

OUTPUT = ROOT / "docs" / "cypher"
MANIFEST = ROOT / "datasets" / "manifest.json"

ENDPOINT = DEFAULT_ENDPOINT
FAILURES: list[tuple[str, str, str]] = []


# --------------------------------------------------------------------------------------------
# example execution
# --------------------------------------------------------------------------------------------


def execute(page: Page, example: Example, label: str) -> str:
    """Runs one example and renders the block the page publishes beneath it."""
    for statement in example.setup:
        setup = run(statement, endpoint=ENDPOINT)
        if not setup.ok:
            FAILURES.append((page.path, f"{label} setup", setup.error or "unknown"))
            return "_Setup for this example did not run._"

    started = time.time()
    result = run(example.query, endpoint=ENDPOINT)
    elapsed = time.time() - started

    for statement in example.teardown:
        run(statement, endpoint=ENDPOINT)

    if example.expect_error:
        if result.ok:
            FAILURES.append((page.path, label, "expected an error, the statement succeeded"))
            return "_This example was expected to be rejected and was not._"
        return f"```\n{result.error}\n```"

    if not result.ok:
        FAILURES.append((page.path, label, result.error or "unknown"))
        return f"```\nFAILED: {result.error}\n```"

    if not result.columns:
        updates = result.statistics.get("updates", 0)
        noun = "change" if updates == 1 else "changes"
        return f"```\nNo rows returned. {updates} {noun} committed.\n```"

    rendered = table(result, limit=example.rows)
    footer = f"{len(result.rows)} row{'s' if len(result.rows) != 1 else ''}"
    if elapsed >= 0.05:
        footer += f", {elapsed * 1000:.0f} ms"
    return f"```\n{rendered}\n\n{footer}\n```"


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------


def relative(from_family: str, target: str) -> str:
    """Builds a link from a page inside `from_family` to a repository-relative documentation path."""
    depth = len(from_family.split("/"))
    return "../" * depth + target


def bullet_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_page(page: Page, datasets: dict[str, dict]) -> str:
    dataset = datasets.get(page.dataset, {})
    up = relative(page.family, "")
    lines: list[str] = [f"# {page.title}", "", f"> {page.summary}", ""]

    facts = [("Kind", KIND_LABEL.get(page.kind, page.kind))]
    if page.signature:
        # A pipe inside a cell ends it; escape before the table is written.
        facts.append(("Signature", f"`{page.signature}`".replace("|", "\\|")))
    facts.append(("Relationship to standard Cypher", STANDARD_LABEL[page.standard]))
    facts.append(
        (
            "Reference dataset",
            f"[`{page.dataset}`]({up}datasets.md#{page.dataset}) — {dataset.get('title', page.dataset)}",
        )
    )
    lines += ["| | |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in facts]
    lines += ["", "## What it does", "", page.what.strip(), ""]

    if page.detail:
        lines += ["## How it behaves", "", page.detail.strip(), ""]

    lines += ["## When to use it", "", page.when.strip(), ""]

    if page.differs:
        lines += ["## How it differs from its neighbours", "", page.differs.strip(), ""]

    lines += ["## Simple example", ""]
    if page.simple.note:
        lines += [page.simple.note.strip(), ""]
    lines += [f"```cypher\n{page.simple.query.strip()}\n```", "", "Result:", "", page.simple.rendered, ""]

    lines += ["## Advanced example", ""]
    if page.advanced.note:
        lines += [page.advanced.note.strip(), ""]
    lines += [
        f"```cypher\n{page.advanced.query.strip()}\n```",
        "",
        "Result:",
        "",
        page.advanced.rendered,
        "",
    ]

    for heading, body in page.sections:
        lines += [f"## {heading}", "", body.strip(), ""]

    if page.use_cases:
        lines += ["## Where it earns its place", "", bullet_list(page.use_cases), ""]
    if page.limits:
        lines += ["## Limitations and trade-offs", "", bullet_list(page.limits), ""]
    if page.see_also:
        lines += ["## See also", "", bullet_list(page.see_also), ""]

    return "\n".join(lines).rstrip() + "\n"


def render_family_index(family: Family) -> str:
    lines = [f"# {family.title}", "", family.blurb.strip(), "", "| Page | Summary | Standard |", "| --- | --- | --- |"]
    for page in family.pages:
        label = {"standard": "standard", "extended": "extended", "extension": "extension"}[page.standard]
        summary = page.summary.replace("|", "\\|")
        lines.append(f"| [{page.title}](./{page.slug}.md) | {summary} | {label} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------


def load_families() -> list[Family]:
    import spec

    return spec.families()


def main() -> int:
    global ENDPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--only", help="build only families whose path starts with this prefix")
    parser.add_argument("--check", action="store_true", help="run every example without writing pages")
    arguments = parser.parse_args()
    ENDPOINT = arguments.endpoint

    probe = run("SHOW PROJECTS", endpoint=ENDPOINT)
    if not probe.ok:
        print(f"cannot reach an IronGraph node at {ENDPOINT}: {probe.error}", file=sys.stderr)
        return 1
    projects = {row[1] for row in probe.rows}

    datasets = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    missing = sorted({name for name in datasets} - projects)
    if missing:
        print(f"warning: dataset project(s) not loaded: {', '.join(missing)}", file=sys.stderr)

    families = load_families()
    if arguments.only:
        families = [family for family in families if family.path.startswith(arguments.only)]
        if not families:
            print(f"no family matches {arguments.only}", file=sys.stderr)
            return 2

    total = 0
    for family in families:
        for page in family.pages:
            page.simple.rendered = execute(page, page.simple, "simple example")
            page.advanced.rendered = execute(page, page.advanced, "advanced example")
            total += 2
            if not arguments.check:
                target = OUTPUT / page.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(render_page(page, datasets))
        if not arguments.check:
            index = OUTPUT / family.path / "README.md"
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(render_family_index(family))
        print(f"{family.path}: {len(family.pages)} pages", flush=True)

    if not arguments.check and not arguments.only:
        import section
        from spec import overview

        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "README.md").write_text(overview.readme())
        (OUTPUT / "datasets.md").write_text(overview.datasets_page())
        for path, page in section.indexes(families).items():
            target = OUTPUT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(page)
        print("entry pages: README.md, datasets.md, section indexes", flush=True)

    print(f"\n{total} examples executed")
    if FAILURES:
        print(f"{len(FAILURES)} example(s) failed:", file=sys.stderr)
        for path, label, error in FAILURES:
            print(f"  {path} [{label}]: {error}", file=sys.stderr)
        return 1
    print("every example ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
