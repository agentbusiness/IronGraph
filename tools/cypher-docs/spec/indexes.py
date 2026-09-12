"""Index and constraint statements.

Indexes decide how a query reaches its rows, and on a graph of any size that decision is the
difference between a lookup and a scan. Four kinds are declared through one statement shape.
Constraints are a different thing wearing similar syntax: an index is an access path, a constraint
is schema authority that rejects data.

The `flights` dataset backs these pages. It carries all four index kinds, so the examples describe
declarations the reference data actually has.
"""

from __future__ import annotations

from model import Example, Family, Page

SCRATCH = "index_example"

# Sixteen short texts on deliberately unrelated subjects: twice the corpus size at which the
# vector index's approximation still agrees with exact search, and therefore enough to make it
# miss its publication floor.
UNRELATED_NOTES = "(:Note {note_id: 1, body: 'A graph database stores nodes and relationships as first-class records.', embedding: [0.0]}), (:Note {note_id: 2, body: 'Dense vector retrieval ranks documents by the cosine distance between embeddings.', embedding: [0.0]}), (:Note {note_id: 3, body: 'Sourdough needs flour, water, salt and time, and rewards patience over technique.', embedding: [0.0]}), (:Note {note_id: 4, body: 'Bitemporal records separate the time an event happened from the time it was recorded.', embedding: [0.0]}), (:Note {note_id: 5, body: 'Breadth-first search finds the fewest-hop route between two nodes in a network.', embedding: [0.0]}), (:Note {note_id: 6, body: 'The standard deviation describes how far values sit from their own mean.', embedding: [0.0]}), (:Note {note_id: 7, body: 'Migratory terns cross from the Arctic to the Antarctic and back each year.', embedding: [0.0]}), (:Note {note_id: 8, body: 'A write-ahead log makes a crash recoverable by recording intent before effect.', embedding: [0.0]}), (:Note {note_id: 9, body: 'Espresso extraction depends on grind size, pressure, and water temperature.', embedding: [0.0]}), (:Note {note_id: 10, body: 'Modularity optimisation groups a network into communities denser than chance.', embedding: [0.0]}), (:Note {note_id: 11, body: 'Volcanic basalt cools quickly and forms characteristic hexagonal columns.', embedding: [0.0]}), (:Note {note_id: 12, body: 'A hash index answers exact lookups but cannot answer ordered comparisons.', embedding: [0.0]}), (:Note {note_id: 13, body: 'The doppler shift of a receding source moves its light toward the red.', embedding: [0.0]}), (:Note {note_id: 14, body: 'Sailing upwind requires tacking, since no boat sails directly into the wind.', embedding: [0.0]}), (:Note {note_id: 15, body: 'Garbage collection reclaims memory that a program can no longer reach.', embedding: [0.0]}), (:Note {note_id: 16, body: 'Tidal ranges are largest when the sun and moon pull along the same line.', embedding: [0.0]})"


def _scratch_setup() -> list[str]:
    return [
        f"DROP PROJECT IF EXISTS {SCRATCH} CASCADE",
        f"CREATE PROJECT {SCRATCH}",
        f"USE {SCRATCH} CREATE "
        "(:Book {isbn: '0262033844', title: 'Introduction to Algorithms', year: 2009}), "
        "(:Book {isbn: '0201896834', title: 'The Art of Computer Programming', year: 1997}), "
        "(:Book {isbn: '1449373321', title: 'Designing Data-Intensive Applications', year: 2017})",
    ]


def families() -> list[Family]:
    return [indexes(), constraints()]


def indexes() -> Family:
    return Family(
        path="statements/indexes",
        title="Index statements",
        blurb=(
            "Four index kinds, one statement shape. An equality index answers exact lookups, a "
            "range index answers ordered comparisons, a text index answers word matching, and a "
            "vector index answers similarity. Declaring the right one is what turns a scan into a "
            "lookup; declaring the wrong one costs maintenance and buys nothing."
        ),
        pages=[
            Page(
                slug="create-index",
                title="`CREATE INDEX`",
                family="statements/indexes",
                kind="statement",
                signature=(
                    "CREATE [RANGE|TEXT|VECTOR] INDEX <name> FOR (<var>:<Label>) "
                    "ON (<var>.<property> [, <var>.<property> …])"
                ),
                summary="Declares an access path over a label's property or properties.",
                dataset="flights",
                standard="extension",
                what=(
                    "`CREATE INDEX` declares an index on one label and one or more of its "
                    "properties. The bare form builds an equality index; `RANGE`, `TEXT` and "
                    "`VECTOR` build the other three kinds.\n\n"
                    "- **Equality** — exact lookup. `MATCH (a:Airport {iata: 'LHR'})`.\n"
                    "- **Range** — ordered comparison. `WHERE a.latitude > 60`.\n"
                    "- **Text** — matching within text.\n"
                    "- **Vector** — similarity over stored vectors.\n\n"
                    "Several properties may be named, which builds one composite index rather than "
                    "several."
                ),
                detail=(
                    "The label and every property must already exist in the project's schema. A "
                    "property that has never been written is not in the catalogue and the statement "
                    "is rejected with `index property is not declared` — so an index is declared "
                    "after the first write, not before it.\n\n"
                    "Index names are unique within a project; re-declaring a name is rejected "
                    "rather than replacing the index.\n\n"
                    "An index changes cost, never answers. Every query returns the same rows with "
                    "or without one. That also means an index can be dropped to test whether it was "
                    "earning its keep."
                ),
                when=(
                    "Declare an equality index on whatever identifies an entity — the property a "
                    "loader matches on and an application looks up by. On the reference datasets "
                    "that is what makes bulk relationship loading feasible at all: without one, "
                    "every relationship written costs a scan.\n\n"
                    "Add a range index when queries compare rather than match, and a text index "
                    "when they search inside strings."
                ),
                differs=(
                    "A constraint also builds an index, but its purpose is to reject data rather "
                    "than to speed a lookup. Declare an index when you want a fast path and a "
                    "constraint when duplicates are a bug."
                ),
                simple=Example(
                    note=(
                        "The four indexes the flights loader declares, as `SHOW INDEXES` reports "
                        "them. Three access paths and one ordered comparison over coordinates."
                    ),
                    query="USE flights\nSHOW INDEXES",
                ),
                advanced=Example(
                    note=(
                        "All four kinds declared on one label in a scratch project, then read back. "
                        "The properties are written first so that they exist in the schema when the "
                        "declarations run."
                    ),
                    setup=_scratch_setup()
                    + [
                        f"USE {SCRATCH} CREATE INDEX book_by_isbn FOR (b:Book) ON (b.isbn)",
                        f"USE {SCRATCH} CREATE RANGE INDEX book_by_year FOR (b:Book) ON (b.year)",
                        f"USE {SCRATCH} CREATE TEXT INDEX book_title_text FOR (b:Book) ON (b.title)",
                        f"USE {SCRATCH} CREATE INDEX book_by_year_and_title "
                        "FOR (b:Book) ON (b.year, b.title)",
                    ],
                    query=f"USE {SCRATCH}\nSHOW INDEXES",
                ),
                use_cases=[
                    "Making a bulk load feasible by indexing the property it matches on.",
                    "Turning an ordered comparison into a range scan.",
                    "Supporting text and similarity retrieval.",
                ],
                limits=[
                    "The label and properties must already exist; declare after the first write.",
                    "Index names are unique per project and re-declaring is rejected.",
                    "An index is maintained on every write to the property it covers, so an unused "
                    "one is pure cost.",
                    "Indexes are per project. A second project needs its own.",
                ],
                see_also=[
                    "[`SHOW INDEXES`](./show-indexes.md)",
                    "[`DROP INDEX`](./drop-index.md)",
                    "[`CREATE CONSTRAINT`](../constraints/create-constraint.md)",
                ],
            ),
            Page(
                slug="show-indexes",
                title="`SHOW INDEXES`",
                family="statements/indexes",
                kind="statement",
                signature="SHOW INDEXES",
                summary="Lists a project's indexes with their kind, state and diagnostic.",
                dataset="citations",
                standard="extension",
                what=(
                    "`SHOW INDEXES` returns one row per index in the current project: `name`, "
                    "`kind`, `state`, and a `diagnostic` that is null unless something went wrong.\n\n"
                    "`state` is the column that matters. `ONLINE` means the index is being used. "
                    "`FAILED` means it exists but is not usable, and the diagnostic says why."
                ),
                detail=(
                    "A failed index is not a silent degradation. A query that would have used it is "
                    "rejected rather than answered more slowly from a broken structure — so "
                    "`SHOW INDEXES` is the first thing to check when a retrieval query starts "
                    "failing rather than starting to crawl.\n\n"
                    "The most common failure is a vector index that missed its recall floor: the "
                    "diagnostic reports the measured figure against the required one.\n\n"
                    "Rollups are derived state but are not indexes, and do not appear here."
                ),
                when=(
                    "Use it to confirm a declaration landed, to check an index is usable before "
                    "depending on it, and to read the diagnostic when one is not."
                ),
                differs=(
                    "`SHOW CONSTRAINTS` lists schema authority rather than access paths; the two "
                    "listings do not overlap even though a constraint is backed by an index."
                ),
                simple=Example(
                    note="The indexes on the citation dataset: one lookup path and two text indexes.",
                    query="USE citations\nSHOW INDEXES",
                ),
                advanced=Example(
                    note=(
                        "A failed index and its diagnostic, produced rather than described. "
                        "Sixteen short texts on deliberately unrelated subjects are embedded — "
                        "twice the corpus size at which the approximation still agrees with exact "
                        "search — so the index is built, misses the 90% recall floor, and reports "
                        "`FAILED` with the agreement it measured. The equality index declared "
                        "beside it is unaffected."
                    ),
                    setup=[
                        "DROP PROJECT IF EXISTS failed_index_example CASCADE",
                        "CREATE PROJECT failed_index_example",
                        f"USE failed_index_example CREATE {UNRELATED_NOTES}",
                        "USE failed_index_example CREATE INDEX note_by_id "
                        "FOR (n:Note) ON (n.note_id)",
                        "USE failed_index_example CREATE EMBEDDING INDEX note_semantic "
                        "FOR (n:Note) FROM n.body INTO n.embedding "
                        "USING MODEL default SIMILARITY COSINE",
                    ],
                    query="USE failed_index_example\nSHOW INDEXES",
                    teardown=["DROP PROJECT IF EXISTS failed_index_example CASCADE"],
                ),
                use_cases=[
                    "Confirming a declaration took effect.",
                    "Diagnosing a retrieval query that fails rather than slows.",
                    "Auditing what a project maintains.",
                ],
                limits=[
                    "Current project only.",
                    "Cannot be composed with `YIELD`, `WITH` or `WHERE`; filter in the client.",
                    "Rollups are not listed.",
                ],
                see_also=["[`CREATE INDEX`](./create-index.md)"],
            ),
            Page(
                slug="rebuild-index",
                title="`REBUILD INDEX`",
                family="statements/indexes",
                kind="statement",
                signature="REBUILD INDEX <name>",
                summary="Rebuilds an index from the current graph.",
                dataset="flights",
                standard="extension",
                what=(
                    "`REBUILD INDEX` discards an index's derived contents and builds them again "
                    "from the graph as it now stands. The index's declaration is unchanged; only "
                    "what it holds is recomputed."
                ),
                detail=(
                    "It does not change any answer. An index is derived state, so a rebuild "
                    "produces the same query results at possibly different cost.\n\n"
                    "A rebuild cannot fix a structural failure. A vector index that missed its "
                    "recall floor will miss it again on the same data, because the measurement is "
                    "of the data and the parameters rather than of a stale build."
                ),
                when=(
                    "Rebuild after a change in the shape of the data that the incremental path "
                    "would leave a poor structure for — a bulk load, a large deletion — or when "
                    "investigating whether an index's contents explain a performance change."
                ),
                differs=(
                    "Dropping and re-declaring achieves the same contents and briefly leaves the "
                    "project without the index. `REBUILD INDEX` keeps the declaration throughout."
                ),
                simple=Example(
                    note="Rebuilding a declared index. Nothing is returned; the effect is the rebuild.",
                    query="USE flights\nREBUILD INDEX airport_by_iata",
                ),
                advanced=Example(
                    note=(
                        "A rebuild changes no answer. The same lookup is run after the rebuild and "
                        "returns exactly what it did before, which is the property that makes an "
                        "index safe to rebuild at any time."
                    ),
                    setup=["USE flights REBUILD INDEX airport_by_iata"],
                    query=(
                        "USE flights\n"
                        "MATCH (airport:Airport {iata: 'LHR'})\n"
                        "RETURN airport.iata AS iata, airport.name AS name,\n"
                        "       airport.city AS city, airport.country AS country"
                    ),
                ),
                use_cases=[
                    "Recovering index quality after a bulk load or large deletion.",
                    "Isolating whether index contents explain a performance change.",
                ],
                limits=[
                    "Changes cost, never answers.",
                    "Cannot repair a failure that is a property of the data or the parameters.",
                    "The index is being rebuilt while the statement runs.",
                ],
                see_also=["[`SHOW INDEXES`](./show-indexes.md)"],
            ),
            Page(
                slug="drop-index",
                title="`DROP INDEX`",
                family="statements/indexes",
                kind="statement",
                signature="DROP INDEX [IF EXISTS] <name>",
                summary="Removes an index declaration and everything it maintained.",
                dataset="flights",
                standard="extension",
                what=(
                    "`DROP INDEX` removes an index. Queries continue to return the same rows, "
                    "reaching them by scan instead of by lookup. `IF EXISTS` makes the statement "
                    "idempotent, which is what a teardown or migration script needs."
                ),
                detail=(
                    "Because an index changes cost and not answers, dropping one is safe in the "
                    "sense that nothing becomes wrong — and unsafe in the sense that something may "
                    "become far slower. On a large graph the difference between a lookup and a scan "
                    "is the difference between milliseconds and minutes.\n\n"
                    "Without `IF EXISTS`, dropping an index that is not there fails with `index "
                    "does not exist`. With it, the statement succeeds and does nothing."
                ),
                when=(
                    "Drop an index that is not being used, or one whose maintenance cost outweighs "
                    "what it saves. Use `IF EXISTS` in anything that might run twice."
                ),
                differs=(
                    "`REBUILD INDEX` keeps the declaration and recomputes contents; `DROP INDEX` "
                    "removes both. Dropping a constraint takes `DROP CONSTRAINT`, even though a "
                    "constraint is index-backed."
                ),
                simple=Example(
                    note=(
                        "The idempotent form. Naming an index that does not exist succeeds and "
                        "changes nothing, so a teardown script can run twice."
                    ),
                    query=f"USE flights\nDROP INDEX IF EXISTS an_index_that_was_never_declared",
                ),
                advanced=Example(
                    note=(
                        "Dropped and re-declared, with the same query run afterwards. The rows are "
                        "identical to the ones the indexed lookup returned — the index was never "
                        "part of the answer."
                    ),
                    setup=_scratch_setup()
                    + [
                        f"USE {SCRATCH} CREATE INDEX book_by_isbn FOR (b:Book) ON (b.isbn)",
                        f"USE {SCRATCH} DROP INDEX IF EXISTS book_by_isbn",
                    ],
                    query=(
                        f"USE {SCRATCH}\n"
                        "MATCH (book:Book {isbn: '1449373321'})\n"
                        "RETURN book.isbn AS isbn, book.title AS title, book.year AS year"
                    ),
                    teardown=[f"DROP PROJECT IF EXISTS {SCRATCH} CASCADE"],
                ),
                use_cases=[
                    "Removing an index that is not earning its maintenance cost.",
                    "Teardown and migration scripts that must be re-runnable.",
                    "Measuring what an index was actually buying.",
                ],
                limits=[
                    "Without `IF EXISTS`, a missing index is an error.",
                    "Queries stay correct and can become dramatically slower.",
                    "An index backing a constraint is not dropped this way.",
                ],
                see_also=["[`CREATE INDEX`](./create-index.md)"],
            ),
        ],
    )


def constraints() -> Family:
    return Family(
        path="statements/constraints",
        title="Constraint statements",
        blurb=(
            "A unique constraint is schema authority, not an access path. It rejects a write that "
            "would duplicate a value, which is what makes a property safe to treat as identity."
        ),
        pages=[
            Page(
                slug="create-constraint",
                title="`CREATE CONSTRAINT`",
                family="statements/constraints",
                kind="statement",
                signature=(
                    "CREATE CONSTRAINT <name> FOR (<var>:<Label>) REQUIRE <var>.<property> IS UNIQUE"
                ),
                summary="Requires a property to be unique across a label, and enforces it on write.",
                dataset="flights",
                standard="extension",
                what=(
                    "`CREATE CONSTRAINT` declares that one property must hold a distinct value "
                    "across every node with a label. A write that would break it is rejected.\n\n"
                    "The constraint is validated against the data that already exists. If the graph "
                    "already contains duplicates the declaration fails and nothing changes — a "
                    "constraint cannot be declared over data that would violate it."
                ),
                detail=(
                    "Enforcement is what distinguishes it from an index. Both make a lookup fast; "
                    "only a constraint makes a duplicate impossible, which is what lets an "
                    "application treat the property as identity and use `MERGE` on it without "
                    "racing.\n\n"
                    "One property per constraint. There is no composite uniqueness."
                ),
                when=(
                    "Declare one on any property an application treats as identity: a business key, "
                    "an external identifier, anything a `MERGE` matches on. Without a constraint, "
                    "uniqueness is a convention the database will not defend."
                ),
                differs=(
                    "An equality index makes a lookup fast and permits duplicates. A constraint "
                    "does both — but fails at declaration time if the data does not already comply, "
                    "which an index never does."
                ),
                simple=Example(
                    note=(
                        "A constraint over a property whose values are already distinct. Nothing is "
                        "returned; `SHOW CONSTRAINTS` is how you see it."
                    ),
                    setup=_scratch_setup(),
                    query=(
                        f"USE {SCRATCH}\n"
                        "CREATE CONSTRAINT book_isbn_unique FOR (b:Book) REQUIRE b.isbn IS UNIQUE"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The constraint refusing a duplicate. The write below names an ISBN that "
                        "already exists, and is rejected rather than accepted — this example is "
                        "expected to fail, and the error is the point."
                    ),
                    query=(
                        f"USE {SCRATCH}\n"
                        "CREATE (:Book {isbn: '1449373321', title: 'A second copy', year: 2020})"
                    ),
                    expect_error=True,
                ),
                use_cases=[
                    "Defending a business key the application treats as identity.",
                    "Making `MERGE` on a key safe rather than merely conventional.",
                    "Catching a duplicate at the write that causes it.",
                ],
                limits=[
                    "One property per constraint; no composite uniqueness.",
                    "Declaration fails if existing data already violates it.",
                    "Node labels only.",
                ],
                see_also=[
                    "[`DROP CONSTRAINT`](./drop-constraint.md)",
                    "[`CREATE INDEX`](../indexes/create-index.md)",
                ],
            ),
            Page(
                slug="drop-constraint",
                title="`DROP CONSTRAINT`",
                family="statements/constraints",
                kind="statement",
                signature="DROP CONSTRAINT [IF EXISTS] <name>",
                summary="Removes a uniqueness requirement and the index that enforced it.",
                dataset="flights",
                standard="extension",
                what=(
                    "`DROP CONSTRAINT` removes a uniqueness requirement. Writes that were rejected "
                    "before are accepted afterwards, so this statement changes what the database "
                    "will store. `IF EXISTS` makes it idempotent."
                ),
                detail=(
                    "Unlike dropping an index, this is not cost-only. Duplicates become possible "
                    "the moment the constraint is gone, and re-declaring it later will fail if any "
                    "arrived in the meantime — so dropping a constraint on a live system is a "
                    "decision about data, not about performance."
                ),
                when=(
                    "Drop one when the uniqueness rule is genuinely no longer true: a key that has "
                    "become non-unique by design, or a migration that must temporarily hold both "
                    "old and new values."
                ),
                differs=(
                    "`DROP INDEX` removes an access path and cannot change what is storable. This "
                    "removes a rule, and can."
                ),
                simple=Example(
                    note="The idempotent form, on a constraint that does not exist.",
                    query=f"USE {SCRATCH}\nDROP CONSTRAINT IF EXISTS never_declared",
                ),
                advanced=Example(
                    note=(
                        "The write that the constraint rejected, accepted once it is gone. Both "
                        "books now carry the same ISBN, which is exactly what the constraint "
                        "existed to prevent."
                    ),
                    setup=[
                        f"USE {SCRATCH} DROP CONSTRAINT IF EXISTS book_isbn_unique",
                        f"USE {SCRATCH} CREATE "
                        "(:Book {isbn: '1449373321', title: 'A second copy', year: 2020})",
                    ],
                    query=(
                        f"USE {SCRATCH}\n"
                        "MATCH (book:Book {isbn: '1449373321'})\n"
                        "RETURN count(book) AS books_sharing_that_isbn,\n"
                        "       collect(book.title) AS titles"
                    ),
                    teardown=[f"DROP PROJECT IF EXISTS {SCRATCH} CASCADE"],
                ),
                use_cases=[
                    "Retiring a uniqueness rule that no longer holds.",
                    "Migrations that must hold old and new keys at once.",
                ],
                limits=[
                    "Changes what the database will accept, unlike dropping an index.",
                    "Re-declaring later fails if duplicates arrived while it was gone.",
                    "Without `IF EXISTS`, a missing constraint is an error.",
                ],
                see_also=["[`CREATE CONSTRAINT`](./create-constraint.md)"],
            ),
        ],
    )
