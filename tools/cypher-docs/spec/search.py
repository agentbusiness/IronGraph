"""Vectors and semantic search.

None of this is standard Cypher. A vector is an ordinary list-valued property; an embedding index
turns a text property into vectors and keeps them searchable; the `SEARCH` clause filters and ranks
an already-bound variable by similarity; and four `vector.*` functions do the arithmetic directly.

The `library` dataset backs these pages. It holds eight arXiv papers with real abstracts, and it is
eight rather than eight thousand for a reason the pages state plainly: the vector index measures its
own approximation against exact search and refuses to publish below 90% recall, which on real
embeddings is not reached above that size.
"""

from __future__ import annotations

from model import Example, Family, Page

RECALL_FLOOR = (
    "The index validates itself before publishing. It runs a sample of queries through both the "
    "approximate index and exact search, and refuses to come online if agreement falls below 90%. "
    "On the real abstracts in this dataset that threshold is met at eight rows and not above it — "
    "measured agreement was 80% at 16 rows, 82% at 1,000 and 86% at 5,000. A vector index that "
    "misses the floor reports `FAILED` in `SHOW INDEXES` with the measured figure, and searching "
    "against it is rejected rather than silently answered from a worse index. There is no automatic "
    "fall back to exact search."
)


def families() -> list[Family]:
    return [search(), vector_functions()]


def search() -> Family:
    return Family(
        path="clauses/search",
        title="Search",
        blurb=(
            "One clause and one statement. `CREATE EMBEDDING INDEX` turns a text property into "
            "searchable vectors; `SEARCH` uses them to filter and rank rows that a pattern has "
            "already bound."
        ),
        pages=[
            Page(
                slug="create-embedding-index",
                title="`CREATE EMBEDDING INDEX`",
                family="clauses/search",
                kind="statement",
                signature=(
                    "CREATE EMBEDDING INDEX <name> FOR (<var>:<Label>) "
                    "FROM <var>.<source> INTO <var>.<target> USING MODEL default SIMILARITY "
                    "COSINE|DOT|EUCLIDEAN"
                ),
                summary=(
                    "Encodes a text property into vectors with the local model and keeps them "
                    "searchable."
                ),
                dataset="library",
                standard="extension",
                what=(
                    "This statement declares that a text property on a label should be encoded into "
                    "vectors by the database's own embedding model, and that those vectors should "
                    "be maintained and searchable. `FROM` names the text, `INTO` names the vector "
                    "property, and `SIMILARITY` fixes how distance is measured.\n\n"
                    "It is what makes `SEARCH … FOR TEXT` possible: a query supplies a phrase, the "
                    "database encodes it with the same model, and the comparison is meaningful "
                    "because both sides came from one encoder."
                ),
                detail=(
                    "Both properties must already exist in the project's schema when the statement "
                    "runs. The source is the text you already have; the target has to be brought "
                    "into existence first, which in practice means writing a placeholder vector to "
                    "every row. The reference loader writes `embedding: [0.0]`.\n\n"
                    "The vectors live in the index, not in the target property. That placeholder "
                    "keeps whatever value it was given — reading it back shows the placeholder, not "
                    "a 768-element vector — so the target property names the index's slot rather "
                    "than storing its contents.\n\n"
                    "`MODEL` accepts only `default`: the model is the one verified artifact the "
                    "database loads at start-up, and the similarity must match the profile that "
                    "artifact was activated with.\n\n" + RECALL_FLOOR
                ),
                when=(
                    "Declare one when retrieval should follow meaning rather than wording — finding "
                    "the paper about horizon thermodynamics when the query says nothing about "
                    "horizons. Where the words themselves are the query, a text index is the right "
                    "tool and is far cheaper."
                ),
                differs=(
                    "A `TEXT` index matches the words that are present. An embedding index matches "
                    "what the text is about, and will rank a document that shares no vocabulary "
                    "with the query above one that shares several words. A plain `VECTOR` index "
                    "searches vectors you supply and computed yourself; an embedding index computes "
                    "them for you and keeps them current."
                ),
                simple=Example(
                    note=(
                        "The index in the reference dataset, as the loader declared it. The "
                        "statement returns no rows; `SHOW INDEXES` is how you see the result."
                    ),
                    query="USE library\nSHOW INDEXES",
                ),
                advanced=Example(
                    note=(
                        "The whole cycle on a fresh project: text written, a placeholder vector "
                        "written so the target property exists, the index declared, and the index "
                        "state read back. The corpus is four documents, comfortably inside the "
                        "recall floor."
                    ),
                    setup=[
                        "DROP PROJECT IF EXISTS embedding_example CASCADE",
                        "CREATE PROJECT embedding_example",
                        "USE embedding_example CREATE "
                        "(:Note {title: 'Graph storage', body: 'A graph database stores nodes and "
                        "relationships as first-class records rather than as rows in a join table.', "
                        "embedding: [0.0]}), "
                        "(:Note {title: 'Similarity search', body: 'Dense vector retrieval compares "
                        "embeddings by cosine distance to rank documents by meaning.', "
                        "embedding: [0.0]}), "
                        "(:Note {title: 'Sourdough', body: 'A slow ferment needs only flour, water, "
                        "salt and time, and rewards patience over technique.', embedding: [0.0]}), "
                        "(:Note {title: 'Bitemporal records', body: 'Separating event time from "
                        "write time lets a correction arrive late without rewriting history.', "
                        "embedding: [0.0]})",
                        "USE embedding_example CREATE EMBEDDING INDEX note_semantic FOR (n:Note) "
                        "FROM n.body INTO n.embedding USING MODEL default SIMILARITY COSINE",
                    ],
                    query=(
                        "USE embedding_example\n"
                        "MATCH (note:Note)\n"
                        "SEARCH note IN (EMBEDDING INDEX note_semantic\n"
                        "                FOR TEXT 'storing facts that change over time' LIMIT 4)\n"
                        "  SCORE AS score\n"
                        "RETURN note.title AS title,\n"
                        "       round(score * 10000) / 10000.0 AS score,\n"
                        "       size(note.embedding) AS stored_property_size\n"
                        "ORDER BY score DESC"
                    ),
                    teardown=["DROP PROJECT IF EXISTS embedding_example CASCADE"],
                ),
                use_cases=[
                    "Retrieval by meaning rather than by shared vocabulary.",
                    "Ranking documents against a phrase a user typed.",
                    "Combining a semantic ranking with ordinary graph filters in one query.",
                ],
                limits=[
                    "Both the source and target properties must exist before the statement runs.",
                    "The target property stores its placeholder, not the vectors; the index holds "
                    "those.",
                    "Only `MODEL default` is accepted, and the similarity must match the active "
                    "profile.",
                    "The index refuses to publish below 90% measured recall, which on real "
                    "embeddings bounds the practical corpus size severely.",
                ],
                see_also=[
                    "[`SEARCH`](./search.md) to query the index",
                    "[`vector.cosine`](../../functions/vector/vector-cosine.md) for the arithmetic "
                    "it is built on",
                ],
            ),
            Page(
                slug="search",
                title="`SEARCH`",
                family="clauses/search",
                kind="clause",
                signature=(
                    "SEARCH <variable> IN (EMBEDDING INDEX <name> FOR TEXT|VECTOR <input> "
                    "LIMIT <n>) SCORE AS <alias>"
                ),
                summary="Filters and ranks already-bound rows by similarity, binding the score.",
                dataset="library",
                standard="extension",
                what=(
                    "`SEARCH` takes a variable a preceding `MATCH` has bound, asks a vector index "
                    "for the most similar entities to an input, keeps the rows whose variable is "
                    "among them, and binds the similarity as a new variable.\n\n"
                    "`FOR TEXT` supplies a phrase, which the database encodes with the same model "
                    "the index was built from. `FOR VECTOR` supplies a vector directly, for when "
                    "you already have one — a stored embedding, or an average of several."
                ),
                detail=(
                    "The variable must already be bound. `SEARCH` is a filter over existing rows, "
                    "not a source of them, so it always follows a `MATCH` — which is exactly what "
                    "lets an ordinary graph filter and a similarity ranking apply to the same "
                    "query.\n\n"
                    "`LIMIT` inside the parentheses bounds how many candidates the index returns, "
                    "and is not the same as a `LIMIT` on the query. It is the search depth: raise "
                    "it when a graph filter after the search would otherwise discard most "
                    "candidates and leave too few rows.\n\n"
                    "The score's meaning follows the index's declared similarity. Under cosine it "
                    "runs from `1` for identical direction down through `0` for unrelated to "
                    "negative for opposed, so a negative score is a real signal rather than an "
                    "error. Scores are comparable within one result and not across indexes."
                ),
                when=(
                    "Use it when the question is \"which of these is most like that\" and the "
                    "candidates are already narrowed by the graph: the most relevant paper among "
                    "those a person cited, the closest document among those a team owns."
                ),
                differs=(
                    "It resembles a `WHERE` clause that also ranks. Unlike `WHERE` it consults an "
                    "index rather than evaluating a predicate per row, and unlike `ORDER BY` it "
                    "removes rows as well as ordering them. The `vector.*` functions compute "
                    "similarity without any index, which is right for a handful of rows and wrong "
                    "for a corpus."
                ),
                simple=Example(
                    note=(
                        "The papers most about black hole thermodynamics, ranked. Nothing in the "
                        "query mentions the words in any title — the ranking comes from the "
                        "abstracts' meaning."
                    ),
                    query=(
                        "USE library\n"
                        "MATCH (paper:Paper)\n"
                        "SEARCH paper IN (EMBEDDING INDEX abstract_semantic\n"
                        "                 FOR TEXT 'thermodynamics of black hole horizons' LIMIT 4)\n"
                        "  SCORE AS score\n"
                        "RETURN paper.title AS title,\n"
                        "       round(score * 10000) / 10000.0 AS score\n"
                        "ORDER BY score DESC"
                    ),
                ),
                advanced=Example(
                    note=(
                        "A semantic ranking narrowed by an ordinary predicate. The search supplies "
                        "eight candidates and a score; the `WHERE` that follows keeps only those "
                        "scoring above a threshold and carrying a submission date, so a similarity "
                        "filter and an ordinary one compose in a single statement rather than in "
                        "two round trips."
                    ),
                    query=(
                        "USE library\n"
                        "MATCH (paper:Paper)\n"
                        "SEARCH paper IN (EMBEDDING INDEX abstract_semantic\n"
                        "                 FOR TEXT 'geometry of spacetime at short distances'\n"
                        "                 LIMIT 8)\n"
                        "  SCORE AS score\n"
                        "WITH paper, score\n"
                        "WHERE paper.submitted IS NOT NULL AND score > 0.25\n"
                        "RETURN paper.arxiv_id AS arxiv_id,\n"
                        "       paper.submitted AS submitted,\n"
                        "       paper.title AS title,\n"
                        "       round(score * 10000) / 10000.0 AS score\n"
                        "ORDER BY score DESC, arxiv_id"
                    ),
                ),
                use_cases=[
                    "Ranking a graph-narrowed candidate set by meaning.",
                    "Retrieval where the query phrase and the documents share no vocabulary.",
                    "Combining similarity with ordinary predicates in a single statement.",
                ],
                limits=[
                    "The variable must already be bound; `SEARCH` filters rows rather than "
                    "producing them.",
                    "The inner `LIMIT` is search depth, not result size. A later filter can leave "
                    "fewer rows than expected.",
                    "Scores are comparable within one result, not between indexes or similarities.",
                    "A vector index that failed its recall floor rejects the search outright.",
                ],
                see_also=[
                    "[`CREATE EMBEDDING INDEX`](./create-embedding-index.md)",
                    "[`vector.cosine`](../../functions/vector/vector-cosine.md)",
                ],
            ),
        ],
    )


def _vector_page(
    slug: str,
    title: str,
    signature: str,
    summary: str,
    what: str,
    when: str,
    differs: str,
    simple: Example,
    advanced: Example,
    use_cases: list[str],
    limits: list[str],
    see_also: list[str],
) -> Page:
    return Page(
        slug=slug,
        title=title,
        family="functions/vector",
        kind="function",
        signature=signature,
        summary=summary,
        dataset="library",
        standard="extension",
        what=what,
        detail=(
            "The arguments are lists of numbers. Both must be the same length; a length mismatch is "
            "an error rather than a silently truncated comparison. A null argument makes the result "
            "null, as with every other scalar function.\n\n"
            "These functions do the arithmetic in the query and consult no index, so they are "
            "exactly as fast as the number of rows they run over. That makes them right for "
            "comparing a handful of vectors and wrong for scanning a corpus, which is what an "
            "embedding index and `SEARCH` exist for."
        ),
        when=when,
        differs=differs,
        simple=simple,
        advanced=advanced,
        use_cases=use_cases,
        limits=limits,
        see_also=see_also,
    )


def vector_functions() -> Family:
    return Family(
        path="functions/vector",
        title="Vector functions",
        blurb=(
            "Four functions over list-valued numbers. They compute similarity and distance directly "
            "in a query, without an index — useful for comparing a few vectors, and the wrong tool "
            "for searching a corpus."
        ),
        pages=[
            _vector_page(
                slug="vector-cosine",
                title="`vector.cosine`",
                signature="vector.cosine(a, b)",
                summary="Cosine similarity: how aligned two vectors are, ignoring their lengths.",
                what=(
                    "`vector.cosine` returns the cosine of the angle between two vectors: `1` when "
                    "they point the same way, `0` when they are perpendicular, `-1` when opposed. "
                    "Magnitude is divided out, so it measures direction alone."
                ),
                when=(
                    "Use it whenever the vectors are embeddings. Direction is what an encoder "
                    "carries meaning in, and length mostly reflects incidental things like document "
                    "size, so dividing it out is what makes two documents comparable."
                ),
                differs=(
                    "`vector.dot` keeps magnitude, so a long vector scores higher regardless of "
                    "direction. `vector.distance` measures separation in space, where cosine "
                    "measures angle: two vectors far apart in length can be perfectly aligned."
                ),
                simple=Example(
                    note=(
                        "The three defining cases: identical direction, perpendicular, and "
                        "opposed."
                    ),
                    query=(
                        "USE library\n"
                        "RETURN vector.cosine([1.0, 0.0], [1.0, 0.0]) AS identical,\n"
                        "       vector.cosine([1.0, 0.0], [0.0, 1.0]) AS perpendicular,\n"
                        "       vector.cosine([1.0, 0.0], [-1.0, 0.0]) AS opposed,\n"
                        "       round(vector.cosine([1.0, 0.0], [1.0, 1.0]) * 10000) / 10000.0\n"
                        "         AS forty_five_degrees"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Cosine ignores scale where the dot product does not. The same pair of "
                        "directions is compared at three magnitudes: the cosine is identical every "
                        "time and the dot product grows with the vectors."
                    ),
                    query=(
                        "USE library\n"
                        "UNWIND [1.0, 10.0, 100.0] AS scale\n"
                        "WITH scale, [3.0 * scale, 4.0 * scale] AS scaled, [4.0, 3.0] AS fixed\n"
                        "RETURN scale,\n"
                        "       round(vector.cosine(scaled, fixed) * 10000) / 10000.0 AS cosine,\n"
                        "       vector.dot(scaled, fixed) AS dot,\n"
                        "       round(vector.distance(scaled, fixed) * 100) / 100.0 AS distance\n"
                        "ORDER BY scale"
                    ),
                ),
                use_cases=[
                    "Comparing embeddings, where direction carries the meaning.",
                    "Reproducing the score an index returned, to check it by hand.",
                    "Ranking a small candidate set without declaring an index.",
                ],
                limits=[
                    "Undefined for a zero vector, which has no direction.",
                    "Ignores magnitude entirely, which is wrong when magnitude is the signal.",
                    "No index is consulted; cost is linear in rows.",
                ],
                see_also=[
                    "[`vector.dot`](./vector-dot.md)",
                    "[`SEARCH`](../../clauses/search/search.md) for the indexed form",
                ],
            ),
            _vector_page(
                slug="vector-dot",
                title="`vector.dot`",
                signature="vector.dot(a, b)",
                summary="Dot product: alignment scaled by both vectors' magnitudes.",
                what=(
                    "`vector.dot` returns the sum of the element-wise products. It grows with "
                    "alignment and with the length of either vector, so it mixes direction and "
                    "magnitude into one number."
                ),
                when=(
                    "Use it when magnitude is part of the signal — a weight, a count, a confidence "
                    "baked into the vector's length — or when the vectors are already normalised, "
                    "in which case it equals the cosine and costs less."
                ),
                differs=(
                    "It is `vector.cosine` before the division by both magnitudes. On unit vectors "
                    "the two agree exactly; on anything else the dot product rewards length, which "
                    "is either the point or a bug depending on what the vectors mean."
                ),
                simple=Example(
                    note="A dot product, and the same vectors normalised so it equals the cosine.",
                    query=(
                        "USE library\n"
                        "WITH [3.0, 4.0] AS a, [4.0, 3.0] AS b\n"
                        "RETURN vector.dot(a, b) AS raw_dot,\n"
                        "       round(vector.dot(vector.normalize(a), vector.normalize(b)) * 10000)\n"
                        "         / 10000.0 AS normalised_dot,\n"
                        "       round(vector.cosine(a, b) * 10000) / 10000.0 AS cosine"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Where the dot product misleads. A vector that is only weakly aligned but "
                        "much longer outscores a shorter, better-aligned one — which is why cosine "
                        "is the default for embeddings."
                    ),
                    query=(
                        "USE library\n"
                        "WITH [1.0, 0.0] AS query_vector\n"
                        "UNWIND [{name: 'aligned but short', v: [0.9, 0.1]},\n"
                        "        {name: 'weak but long', v: [4.0, 6.0]}] AS candidate\n"
                        "RETURN candidate.name AS candidate,\n"
                        "       vector.dot(query_vector, candidate.v) AS dot,\n"
                        "       round(vector.cosine(query_vector, candidate.v) * 10000) / 10000.0\n"
                        "         AS cosine\n"
                        "ORDER BY dot DESC"
                    ),
                ),
                use_cases=[
                    "Vectors already normalised, where it is the cheaper cosine.",
                    "Scoring where magnitude legitimately carries weight.",
                    "Building a custom similarity from parts.",
                ],
                limits=[
                    "Unbounded in both directions; there is no scale to compare against.",
                    "A long vector outranks a well-aligned one, which is rarely what an embedding "
                    "comparison wants.",
                ],
                see_also=["[`vector.cosine`](./vector-cosine.md)"],
            ),
            _vector_page(
                slug="vector-distance",
                title="`vector.distance`",
                signature="vector.distance(a, b)",
                summary="Euclidean distance: how far apart two vectors are in space.",
                what=(
                    "`vector.distance` returns the straight-line distance between two points: the "
                    "square root of the summed squared differences. It is `0` for identical vectors "
                    "and grows without bound as they separate."
                ),
                when=(
                    "Use it when the vectors are positions rather than directions — coordinates, "
                    "measurements, anything where being far apart is the thing you want to "
                    "measure. It is also the right similarity for an index declared `EUCLIDEAN`."
                ),
                differs=(
                    "Distance falls as similarity rises, the opposite of the other three, so an "
                    "ordering by distance is ascending where an ordering by cosine is descending. "
                    "On normalised vectors distance and cosine agree in ranking; on raw vectors "
                    "they can disagree completely."
                ),
                simple=Example(
                    note="The classic right triangle, and a distance of zero for identical vectors.",
                    query=(
                        "USE library\n"
                        "RETURN vector.distance([0.0, 0.0], [3.0, 4.0]) AS three_four_five,\n"
                        "       vector.distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) AS identical,\n"
                        "       round(vector.distance([1.0, 0.0], [0.0, 1.0]) * 10000) / 10000.0\n"
                        "         AS perpendicular_unit_vectors"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Distance and cosine ranking the same candidates differently. Normalising "
                        "first makes them agree, which is the practical reason embeddings are "
                        "compared by angle rather than by position."
                    ),
                    query=(
                        "USE library\n"
                        "WITH [1.0, 1.0] AS query_vector\n"
                        "UNWIND [{name: 'same direction, far', v: [5.0, 5.0]},\n"
                        "        {name: 'different direction, near', v: [1.4, 0.2]}] AS candidate\n"
                        "RETURN candidate.name AS candidate,\n"
                        "       round(vector.distance(query_vector, candidate.v) * 100) / 100.0\n"
                        "         AS distance,\n"
                        "       round(vector.cosine(query_vector, candidate.v) * 10000) / 10000.0\n"
                        "         AS cosine,\n"
                        "       round(vector.distance(vector.normalize(query_vector),\n"
                        "                             vector.normalize(candidate.v)) * 10000)\n"
                        "         / 10000.0 AS normalised_distance\n"
                        "ORDER BY distance"
                    ),
                ),
                use_cases=[
                    "Positions and measurements rather than directions.",
                    "Indexes declared with `EUCLIDEAN` similarity.",
                    "Thresholding on a bounded neighbourhood in space.",
                ],
                limits=[
                    "Lower is more similar, inverting the ordering the other functions use.",
                    "Unbounded above, so a threshold has to be chosen for the data.",
                    "Sensitive to magnitude, which for embeddings is usually noise.",
                ],
                see_also=[
                    "[`vector.cosine`](./vector-cosine.md)",
                    "[`vector.normalize`](./vector-normalize.md)",
                ],
            ),
            _vector_page(
                slug="vector-normalize",
                title="`vector.normalize`",
                signature="vector.normalize(a)",
                summary="Scales a vector to length one, keeping its direction.",
                what=(
                    "`vector.normalize` divides a vector by its own magnitude and returns a vector "
                    "of length `1` pointing the same way. It is the one function here that returns "
                    "a vector rather than a number."
                ),
                when=(
                    "Normalise when you want to compare directions with tools that are sensitive to "
                    "magnitude — before a dot product, or before a Euclidean distance — or to store "
                    "vectors in a form where the cheaper comparison is also the correct one."
                ),
                differs=(
                    "It changes the vector rather than comparing two. Normalising both sides makes "
                    "`vector.dot` equal `vector.cosine`, and makes `vector.distance` rank the same "
                    "way as cosine, which is why it usually appears as a preparation step rather "
                    "than as an answer."
                ),
                simple=Example(
                    note="A 3-4-5 vector reduced to unit length, and the proof that it is one.",
                    query=(
                        "USE library\n"
                        "WITH vector.normalize([3.0, 4.0]) AS unit\n"
                        "RETURN unit,\n"
                        "       round(vector.distance([0.0, 0.0], unit) * 10000) / 10000.0\n"
                        "         AS its_length,\n"
                        "       round(vector.dot(unit, unit) * 10000) / 10000.0 AS dot_with_itself"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Normalisation makes three different measures agree. Once both sides are "
                        "unit length, the dot product equals the cosine and the distance is a "
                        "monotone function of it, so any of the three gives the same ranking."
                    ),
                    query=(
                        "USE library\n"
                        "UNWIND [[5.0, 5.0], [1.4, 0.2], [0.1, 3.0]] AS raw\n"
                        "WITH [1.0, 1.0] AS query_vector, raw,\n"
                        "     vector.normalize([1.0, 1.0]) AS unit_query,\n"
                        "     vector.normalize(raw) AS unit_candidate\n"
                        "RETURN raw,\n"
                        "       round(vector.cosine(query_vector, raw) * 10000) / 10000.0 AS cosine,\n"
                        "       round(vector.dot(unit_query, unit_candidate) * 10000) / 10000.0\n"
                        "         AS normalised_dot,\n"
                        "       round(vector.distance(unit_query, unit_candidate) * 10000) / 10000.0\n"
                        "         AS normalised_distance\n"
                        "ORDER BY cosine DESC"
                    ),
                ),
                use_cases=[
                    "Preparing vectors so a dot product means cosine similarity.",
                    "Making Euclidean distance rank the same way as cosine.",
                    "Storing vectors in a comparable form.",
                ],
                limits=[
                    "Undefined for a zero vector.",
                    "Discards magnitude, which is a loss when magnitude carries meaning.",
                    "Returns a new vector; it does not modify a stored property.",
                ],
                see_also=["[`vector.cosine`](./vector-cosine.md)"],
            ),
        ],
    )
