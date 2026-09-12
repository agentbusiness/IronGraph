"""Content for the Cypher reference documentation.

Each module returns the families it owns. `build.py` executes every example in these families
against a live node and writes the pages.
"""

from __future__ import annotations

from model import Family

from . import aggregation, indexes, procedures, projects, search, temporal


def families() -> list[Family]:
    collected: list[Family] = []
    for module in (procedures, aggregation, temporal, projects, search, indexes):
        collected.extend(module.families())
    return collected
