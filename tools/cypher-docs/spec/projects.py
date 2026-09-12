"""Projects and graph layers: the two scopes every query is evaluated in.

Neither exists in standard Cypher. A project is a named, isolated graph, and every query names one —
there is no implicit default. A layer is a semantic partition inside a project, and every query
chooses which layers it reads and which one it writes to.

Together they decide what a query can see before a single pattern is matched, which is why they are
worth understanding before anything else in this reference.
"""

from __future__ import annotations

from model import Example, Family, Page

LAYER_MEANINGS = (
    "The three layers carry fixed meanings.\n\n"
    "- `OBSERVED` — facts captured from source activity. What happened.\n"
    "- `KNOWLEDGE` — curated understanding. What has been concluded.\n"
    "- `WORKSPACE` — provisional or application working state. What is being tried.\n\n"
    "`OBSERVED` and `KNOWLEDGE` together form the default read view. `WORKSPACE` is never in it "
    "unless a query asks, which is what keeps scratch data out of results that did not request it."
)


def families() -> list[Family]:
    return [projects(), layers()]


def projects() -> Family:
    return Family(
        path="statements/projects",
        title="Project statements",
        blurb=(
            "A project is a named, isolated graph. Every query names one, and nothing falls through "
            "to a default — an application cannot accidentally read or write the wrong graph "
            "because it forgot to say which."
        ),
        pages=[
            Page(
                slug="create-project",
                title="`CREATE PROJECT`",
                family="statements/projects",
                kind="statement",
                signature="CREATE PROJECT [IF NOT EXISTS] <name>",
                summary="Creates a named, isolated graph.",
                dataset="trust",
                standard="extension",
                what=(
                    "`CREATE PROJECT` creates an empty graph under a name. The project is the "
                    "isolation boundary: labels, relationship types, properties, indexes, "
                    "constraints, temporal declarations and layers all belong to one project and "
                    "are invisible from any other.\n\n"
                    "`IF NOT EXISTS` makes the statement idempotent, which is what a start-up path "
                    "or a migration wants."
                ),
                detail=(
                    "A project has a stable identity and a display name. `SHOW PROJECTS` returns "
                    "both; the identity is what the database uses and the display name is what "
                    "`USE` matches. Renaming changes the display name and leaves the identity "
                    "alone, so a rename does not orphan anything.\n\n"
                    "Creating a project is a schema operation, not a data one. It commits through "
                    "the same durability path as a write and is visible to the next statement."
                ),
                when=(
                    "Create a project per graph that should not see another: one per tenant, per "
                    "environment, per dataset. Reference datasets in this documentation are one "
                    "project each, which is why an example that says `USE trust` cannot "
                    "accidentally read `epinions`."
                ),
                differs=(
                    "It is not a label or a namespace inside one graph. Two projects share no "
                    "storage, no schema and no index, and no single query can read across them — "
                    "which is stronger isolation than a label prefix and cheaper than a separate "
                    "process."
                ),
                simple=Example(
                    note="Creating a project idempotently, the way a start-up path should.",
                    setup=["DROP PROJECT IF EXISTS worked_example CASCADE"],
                    query="CREATE PROJECT IF NOT EXISTS worked_example",
                ),
                advanced=Example(
                    note=(
                        "Isolation demonstrated rather than asserted. The same label and the same "
                        "property name exist in two projects with different contents, and a query "
                        "in one sees only its own."
                    ),
                    setup=[
                        "DROP PROJECT IF EXISTS worked_example_other CASCADE",
                        "CREATE PROJECT worked_example_other",
                        "USE worked_example CREATE (:Shared {origin: 'first project'})",
                        "USE worked_example_other CREATE (:Shared {origin: 'second project'}), "
                        "(:Shared {origin: 'second project again'})",
                    ],
                    query=(
                        "USE worked_example\n"
                        "MATCH (node:Shared)\n"
                        "RETURN count(node) AS visible_here,\n"
                        "       collect(node.origin) AS origins"
                    ),
                ),
                use_cases=[
                    "One graph per tenant, environment or dataset.",
                    "Keeping an experiment from touching production data.",
                    "Making the graph a query runs against explicit in the statement itself.",
                ],
                limits=[
                    "No query reads across projects. Combining two means reading both and joining "
                    "in the client.",
                    "There is no implicit default project; a graph query without `USE` has nothing "
                    "to run against.",
                    "Every project admitted to an accelerator holds its own resident copy, so "
                    "project count is a capacity decision.",
                ],
                see_also=[
                    "[`SHOW PROJECTS`](./show-projects.md)",
                    "[`DROP PROJECT`](./drop-project.md)",
                ],
            ),
            Page(
                slug="show-projects",
                title="`SHOW PROJECTS`",
                family="statements/projects",
                kind="statement",
                signature="SHOW PROJECTS",
                summary="Lists every project with its stable identity and display name.",
                dataset="trust",
                standard="extension",
                what=(
                    "`SHOW PROJECTS` returns one row per project: `project_id`, the stable identity "
                    "the database uses, and `display_name`, the name `USE` matches. It is the one "
                    "statement that runs without naming a project, because it is about the set of "
                    "them."
                ),
                detail=(
                    "The identity survives a rename and the display name does not, so anything "
                    "that needs to refer to a project across time should hold the identity.\n\n"
                    "The statement stands alone. It cannot be followed by `YIELD`, `WITH` or "
                    "`WHERE`, and cannot be combined with another statement in the same execution, "
                    "so filtering and ordering the listing happens in the client."
                ),
                when=(
                    "Use it to discover what exists on a node, to confirm a create or a rename "
                    "landed, and to resolve a display name to an identity."
                ),
                differs=(
                    "`SHOW INDEXES` and `SHOW CONSTRAINTS` describe one project's schema and "
                    "require a `USE`. `SHOW PROJECTS` describes the node."
                ),
                simple=Example(
                    note="Every project on this node, in name order.",
                    query="SHOW PROJECTS",
                    rows=20,
                ),
                advanced=Example(
                    note=(
                        "The identity is stable and the display name is not. This listing is taken "
                        "after a rename: the project appears under its new name, carrying the "
                        "identity it was created with."
                    ),
                    setup=[
                        "DROP PROJECT IF EXISTS naming_example CASCADE",
                        "DROP PROJECT IF EXISTS naming_example_renamed CASCADE",
                        "CREATE PROJECT naming_example",
                        "ALTER PROJECT naming_example RENAME TO naming_example_renamed",
                    ],
                    query="SHOW PROJECTS",
                    rows=20,
                    teardown=["DROP PROJECT IF EXISTS naming_example_renamed CASCADE"],
                ),
                use_cases=[
                    "Discovering what a node holds.",
                    "Confirming a create or rename.",
                    "Resolving a display name to a stable identity.",
                ],
                limits=[
                    "Names and identities only; it reports nothing about size or contents.",
                    "Every project is listed, with no filtering by permission at this layer.",
                ],
                see_also=["[`CREATE PROJECT`](./create-project.md)"],
            ),
            Page(
                slug="drop-project",
                title="`DROP PROJECT`",
                family="statements/projects",
                kind="statement",
                signature="DROP PROJECT [IF EXISTS] <name> [CASCADE]",
                summary="Removes a project and, with `CASCADE`, everything inside it.",
                dataset="trust",
                standard="extension",
                what=(
                    "`DROP PROJECT` removes a project. `CASCADE` removes its contents with it — "
                    "nodes, relationships, indexes, constraints, temporal declarations and "
                    "rollups. `IF EXISTS` makes the statement idempotent.\n\n"
                    "This is the most destructive statement in the language. There is no undo and "
                    "no recycle bin."
                ),
                detail=(
                    "Without `CASCADE` a project that still holds data is not dropped, which is "
                    "the guard against removing a populated graph by mistake. With `CASCADE` the "
                    "guard is gone by request.\n\n"
                    "The reference dataset loader never runs this statement implicitly. It checks "
                    "`SHOW PROJECTS` and skips an existing dataset project; replacing sample data "
                    "requires an explicit destructive choice."
                ),
                when=(
                    "Use it to reclaim a tenant, retire an environment, or rebuild a dataset from "
                    "scratch. In anything scripted, pair `IF EXISTS` with `CASCADE` so a re-run "
                    "behaves the same as a first run."
                ),
                differs=(
                    "`DELETE` removes matched graph entities and leaves the project, its schema and "
                    "its indexes standing. `DROP PROJECT` removes the container."
                ),
                simple=Example(
                    note=(
                        "Dropping a scratch project idempotently. Nothing is returned; the effect "
                        "is the removal."
                    ),
                    query="DROP PROJECT IF EXISTS worked_example_other CASCADE",
                ),
                advanced=Example(
                    note=(
                        "The rebuild pattern the dataset loader uses. Dropping and recreating "
                        "before loading makes the result depend on the source data alone, not on "
                        "whatever the project happened to contain before."
                    ),
                    setup=[
                        "DROP PROJECT IF EXISTS worked_example CASCADE",
                        "CREATE PROJECT worked_example",
                        "USE worked_example CREATE (:Row {value: 1}), (:Row {value: 2})",
                    ],
                    query=(
                        "USE worked_example\n"
                        "MATCH (row:Row)\n"
                        "RETURN count(row) AS rows_after_rebuild, sum(row.value) AS total"
                    ),
                    teardown=["DROP PROJECT IF EXISTS worked_example CASCADE"],
                ),
                use_cases=[
                    "Reclaiming a tenant or environment.",
                    "Making a data load deterministic by rebuilding rather than appending.",
                    "Removing an experiment completely.",
                ],
                limits=[
                    "Irreversible. There is no undo.",
                    "Without `CASCADE`, a project holding data is not dropped.",
                    "Dropping a project invalidates anything holding its identity.",
                ],
                see_also=["[`CREATE PROJECT`](./create-project.md)"],
            ),
        ],
    )


def layers() -> Family:
    return Family(
        path="clauses/layers",
        title="Layer clauses",
        blurb=(
            "Every project is partitioned into three layers, and every query chooses which it "
            "reads and which it writes. The choice is made once, ahead of the query body, and "
            "applies to the whole statement.\n\n" + LAYER_MEANINGS
        ),
        pages=[
            Page(
                slug="use-layer",
                title="`USE LAYER`",
                family="clauses/layers",
                kind="clause",
                signature="USE LAYER <layer> [, <layer> …]  (before the query body)",
                summary="Chooses which layers the query reads.",
                dataset="trust",
                standard="extension",
                what=(
                    "`USE LAYER` names the layers a query may read. Without it the query reads the "
                    "default view, `OBSERVED` and `KNOWLEDGE` together. Naming layers replaces that "
                    "view rather than adding to it, so `USE LAYER OBSERVED` reads observed facts "
                    "and nothing else.\n\n" + LAYER_MEANINGS
                ),
                detail=(
                    "The choice is total. A node in a layer the query did not select does not "
                    "exist for that query: it cannot be matched, counted, traversed through, or "
                    "reached by an algorithm. That is what makes a layer a scope rather than a "
                    "filter — there is no way for unselected data to leak into a result.\n\n"
                    "Because the graph procedures read the project graph under the query's selected "
                    "layers, `USE LAYER` is also how an algorithm is scoped."
                ),
                when=(
                    "Name layers when the distinction matters: reading only what was observed "
                    "before a conclusion was drawn, keeping curated data out of a raw count, or "
                    "including workspace data that the default view deliberately excludes."
                ),
                differs=(
                    "`USE LAYER` chooses what is visible; `WRITE LAYER` chooses where new data "
                    "lands. The two are linked: a write layer must be among the layers selected for "
                    "reading, so a query that writes to the workspace has to select it."
                ),
                simple=Example(
                    note=(
                        "The same count under three views. Every account in this dataset was "
                        "written to `OBSERVED`, so the knowledge-only view is empty and the "
                        "default view matches the observed one."
                    ),
                    query=(
                        "USE trust\n"
                        "USE LAYER OBSERVED\n"
                        "MATCH (account:Account)\n"
                        "RETURN count(account) AS observed_accounts"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Workspace data is invisible to the default view. A draft is written to "
                        "`WORKSPACE`, counted there, and then counted again without naming the "
                        "layer — where it does not appear at all."
                    ),
                    setup=[
                        "USE trust USE LAYER WORKSPACE WRITE LAYER WORKSPACE "
                        "MATCH (draft:Draft) DELETE draft",
                        "USE trust USE LAYER WORKSPACE WRITE LAYER WORKSPACE "
                        "CREATE (:Draft {note: 'candidate merge'})",
                    ],
                    query=(
                        "USE trust\n"
                        "USE LAYER WORKSPACE\n"
                        "MATCH (draft:Draft)\n"
                        "RETURN count(draft) AS drafts_in_workspace,\n"
                        "       collect(draft.note) AS notes"
                    ),
                ),
                use_cases=[
                    "Separating what was observed from what was concluded.",
                    "Keeping provisional work out of ordinary results by default.",
                    "Scoping a graph algorithm to one layer of the graph.",
                ],
                limits=[
                    "Naming layers replaces the default view rather than extending it.",
                    "Unselected data is invisible, not filtered: it cannot be traversed through "
                    "either.",
                    "One layer selection per query.",
                ],
                see_also=["[`WRITE LAYER`](./write-layer.md)"],
            ),
            Page(
                slug="write-layer",
                title="`WRITE LAYER`",
                family="clauses/layers",
                kind="clause",
                signature="WRITE LAYER <layer>  (before the query body)",
                summary="Chooses which layer the query's writes land in.",
                dataset="trust",
                standard="extension",
                what=(
                    "`WRITE LAYER` names the layer new nodes and relationships are created in. "
                    "Without it, writes go to `OBSERVED`.\n\n"
                    "It is not independent of `USE LAYER`: a query may only write to a layer it "
                    "also reads. Writing to `WORKSPACE` therefore means selecting it, either alone "
                    "or alongside the layers the query reads from — `USE LAYER OBSERVED, WORKSPACE "
                    "WRITE LAYER WORKSPACE` is the shape for deriving provisional data from "
                    "authoritative data without mixing the two. A write layer outside the read set "
                    "is rejected with `write layer is not visible`."
                ),
                detail=(
                    "The layer is a property of where an entity lives, fixed when it is created. "
                    "Writing to `WORKSPACE` then reading without naming that layer will not find "
                    "what was just written — not a fault, but the most common surprise. Read back "
                    "with `USE LAYER WORKSPACE`.\n\n"
                    "A query that only reads may still name a write layer; it simply has no "
                    "effect, and the visibility rule is not enforced against it."
                ),
                when=(
                    "Name a write layer whenever new data should not join the authoritative view: "
                    "a candidate merge, a suggested link, an application's own working state. It "
                    "is what lets a pipeline stage its output where a later stage can find it and "
                    "an ordinary reader cannot."
                ),
                differs=(
                    "`USE LAYER` controls visibility and `WRITE LAYER` controls placement, but the "
                    "second is constrained by the first. A derivation that reads authoritative data "
                    "and writes provisional data names both: the layers it reads from, and the "
                    "workspace among them."
                ),
                simple=Example(
                    note=(
                        "Writing to the workspace while reading the default view. The statement "
                        "returns no rows; its effect is the write."
                    ),
                    setup=[
                        "USE trust USE LAYER WORKSPACE WRITE LAYER WORKSPACE "
                        "MATCH (candidate:Candidate) DELETE candidate"
                    ],
                    query=(
                        "USE trust\n"
                        "USE LAYER OBSERVED, WORKSPACE\n"
                        "WRITE LAYER WORKSPACE\n"
                        "MATCH (account:Account)\n"
                        "WHERE account.account_id = 35\n"
                        "CREATE (:Candidate {account_id: account.account_id, reason: 'high volume'})"
                    ),
                ),
                advanced=Example(
                    note=(
                        "A derivation staged in the workspace: the busiest raters, read from the "
                        "authoritative view and written where they will not disturb it. The result "
                        "reads them back from the layer they landed in."
                    ),
                    setup=[
                        "USE trust USE LAYER WORKSPACE WRITE LAYER WORKSPACE "
                        "MATCH (candidate:Candidate) DELETE candidate",
                        "USE trust USE LAYER OBSERVED, WORKSPACE WRITE LAYER WORKSPACE "
                        "MATCH (rater:Account)-[rating:RATED]->() "
                        "WITH rater.account_id AS account, count(rating) AS given "
                        "ORDER BY given DESC LIMIT 5 "
                        "CREATE (:Candidate {account_id: account, ratings_given: given})",
                    ],
                    query=(
                        "USE trust\n"
                        "USE LAYER WORKSPACE\n"
                        "MATCH (candidate:Candidate)\n"
                        "RETURN candidate.account_id AS account,\n"
                        "       candidate.ratings_given AS ratings_given\n"
                        "ORDER BY ratings_given DESC, account"
                    ),
                    teardown=[
                        "USE trust USE LAYER WORKSPACE WRITE LAYER WORKSPACE "
                        "MATCH (candidate:Candidate) DELETE candidate"
                    ],
                ),
                use_cases=[
                    "Staging a derivation where ordinary readers will not see it.",
                    "Recording curated conclusions in `KNOWLEDGE` beside the observations they came "
                    "from.",
                    "Giving an application its own working state inside the same graph.",
                ],
                limits=[
                    "Data written to a layer is invisible until a query selects that layer.",
                    "An entity's layer is fixed at creation.",
                    "One write layer per query.",
                ],
                see_also=["[`USE LAYER`](./use-layer.md)"],
            ),
        ],
    )
