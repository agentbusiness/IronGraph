"""Section indexes: the page a reader lands on for `procedures/` or `functions/`."""

from __future__ import annotations

from model import Family

BLURBS = {
    "procedures": (
        "Twelve built-in graph algorithms. Every one is an IronGraph extension: standard Cypher has "
        "no procedure catalogue, and these run inside an ordinary Cypher pipeline rather than over "
        "a separately projected graph.\n\n"
        "They divide by the question they answer. Traversal asks what is reachable. Routing asks "
        "how to get there and what it costs. Centrality asks which nodes matter. Community asks how "
        "the graph divides. Structure asks how tightly it is knit."
    ),
    "functions": (
        "Aggregates and scalar functions. The aggregates are documented first because they are "
        "where a graph query becomes a measurement, and because the dispersion and percentile "
        "aggregates go beyond what standard Cypher offers."
    ),
    "clauses": (
        "The pieces a query pipeline is built from. Documented here are the clauses IronGraph adds "
        "to Cypher rather than the ones it shares with it: the three that put time into a query, "
        "the two that choose which layers a query sees and writes to, and the one that ranks rows "
        "by similarity."
    ),
    "statements": (
        "Statements that change what the database holds or how it is organised, as opposed to "
        "clauses that shape a query. Administration is Cypher here: there is no second language "
        "for creating a project, declaring an index or making a property remember its past."
    ),
}


def indexes(families: list[Family]) -> dict[str, str]:
    sections: dict[str, list[Family]] = {}
    for family in families:
        sections.setdefault(family.path.split("/")[0], []).append(family)

    pages: dict[str, str] = {}
    for section, members in sections.items():
        lines = [f"# {section.capitalize()}", "", BLURBS.get(section, ""), ""]
        for family in members:
            depth = len(family.path.split("/")) - 1
            relative = "/".join(family.path.split("/")[1:]) or "."
            lines += [
                f"## [{family.title}](./{relative}/README.md)",
                "",
                family.blurb.strip(),
                "",
                "| Page | Summary |",
                "| --- | --- |",
            ]
            for page in family.pages:
                summary = page.summary.replace("|", "\\|")
                lines.append(f"| [{page.title}](./{relative}/{page.slug}.md) | {summary} |")
            lines.append("")
            _ = depth
        pages[f"{section}/README.md"] = "\n".join(lines).rstrip() + "\n"
    return pages
