"""Page model for the Cypher reference documentation.

A page describes one Cypher surface item: a clause, a function, or a procedure. Every page carries
two runnable examples. The builder executes both against a live IronGraph node loaded with the
reference datasets and embeds the real result, so a documented expected result is never written by
hand and never drifts silently away from the product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Standard = Literal["standard", "extended", "extension"]

STANDARD_LABEL: dict[Standard, str] = {
    "standard": "Standard Cypher",
    "extended": "Standard Cypher, extended by IronGraph",
    "extension": "IronGraph extension",
}

KIND_LABEL = {
    "clause": "Clause",
    "statement": "Statement",
    "aggregate": "Aggregate function",
    "function": "Scalar function",
    "procedure": "Procedure",
}


@dataclass
class Example:
    """One runnable example and the note that frames it."""

    query: str
    note: str = ""
    #: Rows to show in the embedded result. A large result is truncated with a counted footer.
    rows: int = 12
    #: Set when the example is expected to fail, and the error is the point being made.
    expect_error: bool = False
    #: Statements that must run before the example, and are not shown on the page.
    setup: list[str] = field(default_factory=list)
    #: Statements that run after the example to restore the dataset.
    teardown: list[str] = field(default_factory=list)


@dataclass
class Page:
    """One documented item."""

    slug: str
    title: str
    family: str
    kind: str
    summary: str
    dataset: str
    simple: Example
    advanced: Example
    what: str
    when: str
    standard: Standard = "extension"
    signature: str = ""
    #: Prose contrasting this item with the neighbours a reader would otherwise confuse it with.
    differs: str = ""
    #: Longer mechanical detail: evaluation order, row grain, null and type behaviour.
    detail: str = ""
    use_cases: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)
    #: Extra named sections rendered after the examples, as (heading, body) pairs.
    sections: list[tuple[str, str]] = field(default_factory=list)

    @property
    def path(self) -> str:
        return f"{self.family}/{self.slug}.md"


@dataclass
class Family:
    """One directory of pages."""

    path: str
    title: str
    blurb: str
    pages: list[Page] = field(default_factory=list)
