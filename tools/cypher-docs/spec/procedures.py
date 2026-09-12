"""Graph procedures: the twelve built-in algorithms.

Every algorithm is an IronGraph extension. Standard Cypher has no procedure catalogue of its own,
and these run inside an ordinary Cypher pipeline rather than over a separately projected graph.
"""

from __future__ import annotations

from model import Example, Family, Page

SHARED_LIMITS = [
    "An algorithm reads the whole project graph under the query's selected layers. There is no "
    "separate projection step and no way to restrict an algorithm to a label or relationship type; "
    "filter the rows it yields, or keep the data you want analysed in its own project.",
    "Scores and components are query results. They are not written back to the graph, so nothing "
    "derived becomes a canonical node or relationship unless you write it yourself.",
    "A procedure runs once per input row. Put `CALL` after a clause that produces exactly the rows "
    "you want it driven by, or the algorithm runs again for each one.",
]

TRAVERSAL_NOTE = (
    "The source argument is a bound node or a non-negative stable node identifier. A node that is "
    "not visible under the query's selected layers is rejected rather than silently skipped."
)


def families() -> list[Family]:
    return [traversal(), routing(), centrality(), community(), structure()]


# --------------------------------------------------------------------------------------------
# traversal
# --------------------------------------------------------------------------------------------


def traversal() -> Family:
    return Family(
        path="procedures/traversal",
        title="Traversal procedures",
        blurb=(
            "Breadth-first and depth-first expansion from one source node. Both follow outgoing "
            "relationships only, and both report reachability rather than a route: use the routing "
            "procedures when you need the path itself."
        ),
        pages=[
            Page(
                slug="graph-bfs",
                title="`graph.bfs`",
                family="procedures/traversal",
                kind="procedure",
                signature="graph.bfs(source) YIELD node, distance",
                summary="Every node reachable from one source, with its hop distance.",
                dataset="flights",
                what=(
                    "`graph.bfs` expands outward from one source node along outgoing relationships, "
                    "level by level, and yields each node it reaches together with the number of "
                    "hops taken to reach it. The source itself is yielded at distance `0`. Nodes "
                    "that cannot be reached are not yielded at all, so the row count is the size of "
                    "the reachable set rather than the size of the graph."
                ),
                detail=(
                    "Distance counts relationships, not weight. Two airports one flight apart are "
                    "at distance `1` whether that flight is 90 kilometres or 9,000. Because the "
                    "expansion is breadth-first, the first time a node is reached is by a shortest "
                    "hop count, and the distance yielded is final.\n\n"
                    "Direction is not optional. `graph.bfs` follows relationships in the direction "
                    "they were created, so in a graph of one-way routes the reachable set from an "
                    "airport is what you can fly *to*, never what can fly *in*."
                ),
                when=(
                    "Use `graph.bfs` to answer reachability and hop-count questions: what is within "
                    "two connections of here, how far away is the furthest thing I can still get "
                    "to, is that node reachable at all. It is the cheapest way to bound a "
                    "neighbourhood before doing more expensive work on it."
                ),
                differs=(
                    "`graph.dfs` visits the same set of nodes and yields a visit order rather than a "
                    "distance; use it when the shape of the descent matters and the distance does "
                    "not. `graph.shortestpath` returns one route between two named nodes instead of "
                    "the whole reachable set. `graph.dijkstra` answers the same question as "
                    "`graph.bfs` but measures cost with a relationship property instead of hops."
                ),
                simple=Example(
                    note=(
                        "How far the rest of the world is from London Heathrow, counted in flights. "
                        f"{TRAVERSAL_NOTE}"
                    ),
                    query=(
                        "USE flights\n"
                        "MATCH (origin:Airport {iata: 'LHR'})\n"
                        "CALL graph.bfs(origin) YIELD node, distance\n"
                        "RETURN distance, count(node) AS airports\n"
                        "ORDER BY distance"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The airports that are exactly three flights from Heathrow and cannot be "
                        "reached in fewer. Because breadth-first distance is final on first "
                        "arrival, filtering on it is enough to express \"no shorter route exists\"."
                    ),
                    query=(
                        "USE flights\n"
                        "MATCH (origin:Airport {iata: 'LHR'})\n"
                        "CALL graph.bfs(origin) YIELD node, distance\n"
                        "WITH node, distance WHERE distance = 3\n"
                        "RETURN node.iata AS iata, node.name AS airport, node.country AS country\n"
                        "ORDER BY country, iata\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Bounding a neighbourhood before running something expensive over it.",
                    "Answering \"how many connections away\" questions without materialising paths.",
                    "Finding the reachable set from a node to test whether the graph is connected "
                    "in the direction you care about.",
                ],
                limits=SHARED_LIMITS
                + [
                    "Only outgoing relationships are followed. To expand both ways, keep the "
                    "reciprocal relationship in the graph.",
                    "Hop distance ignores relationship properties entirely. Reach for "
                    "`graph.dijkstra` when the cost of an edge matters.",
                ],
                see_also=[
                    "[`graph.dfs`](./graph-dfs.md) for visit order rather than distance",
                    "[`graph.dijkstra`](../shortest-path/graph-dijkstra.md) for weighted cost",
                    "[`graph.shortestpath`](../shortest-path/graph-shortestpath.md) for one route",
                ],
            ),
            Page(
                slug="graph-dfs",
                title="`graph.dfs`",
                family="procedures/traversal",
                kind="procedure",
                signature="graph.dfs(source) YIELD node, order",
                summary="Every node reachable from one source, in depth-first visit order.",
                dataset="flights",
                what=(
                    "`graph.dfs` explores as far as it can along one branch before backtracking, "
                    "and yields each reached node with the position at which it was visited. The "
                    "source is visited at order `0`. As with `graph.bfs`, unreachable nodes are not "
                    "yielded, so the row count is the size of the reachable set."
                ),
                detail=(
                    "`order` is a visit sequence, not a distance and not a ranking. Two nodes with "
                    "adjacent order values are adjacent in the descent, which usually means one is "
                    "the other's neighbour, but a node reached late can still be a direct neighbour "
                    "of the source. Never read `order` as \"how far away\".\n\n"
                    "The order depends on the order relationships are stored in, so it is stable "
                    "for a given graph state and changes when the graph changes. Treat it as a "
                    "deterministic traversal trace rather than a property of the data."
                ),
                when=(
                    "Use `graph.dfs` when the question is about the shape of a descent rather than "
                    "the distance to a node: tracing one dependency chain to its end, or walking a "
                    "reachable set in an order where a branch is finished before the next begins."
                ),
                differs=(
                    "`graph.bfs` visits the identical set of nodes and yields a hop distance, which "
                    "is the more useful number in almost every reachability question. Prefer "
                    "`graph.bfs` unless the descent order is specifically what you want."
                ),
                simple=Example(
                    note="The first ten airports a depth-first descent from Heathrow visits.",
                    query=(
                        "USE flights\n"
                        "MATCH (origin:Airport {iata: 'LHR'})\n"
                        "CALL graph.dfs(origin) YIELD node, order\n"
                        "RETURN order, node.iata AS iata, node.name AS airport\n"
                        "ORDER BY order\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Depth-first order set against breadth-first distance for the same source. "
                        "The two disagree sharply, which is the point: a node visited late in the "
                        "descent can be one hop away."
                    ),
                    query=(
                        "USE flights\n"
                        "MATCH (origin:Airport {iata: 'LHR'})\n"
                        "CALL graph.bfs(origin) YIELD node, distance\n"
                        "WITH origin, node AS reached, distance\n"
                        "WHERE distance = 1\n"
                        "WITH origin, collect(reached.iata) AS neighbours\n"
                        "CALL graph.dfs(origin) YIELD node, order\n"
                        "WITH neighbours, node, order WHERE node.iata IN neighbours\n"
                        "RETURN min(order) AS first_neighbour_visited,\n"
                        "       max(order) AS last_neighbour_visited,\n"
                        "       count(node) AS direct_neighbours"
                    ),
                ),
                use_cases=[
                    "Tracing one chain of dependencies to its end.",
                    "Producing a deterministic walk of a reachable set for diffing or replay.",
                ],
                limits=SHARED_LIMITS
                + [
                    "`order` is not a distance and not a rank. Reading it as one is the most common "
                    "mistake with this procedure.",
                    "Only outgoing relationships are followed.",
                ],
                see_also=["[`graph.bfs`](./graph-bfs.md) for hop distance"],
            ),
        ],
    )


# --------------------------------------------------------------------------------------------
# shortest path and routing
# --------------------------------------------------------------------------------------------


def routing() -> Family:
    return Family(
        path="procedures/shortest-path",
        title="Shortest-path procedures",
        blurb=(
            "One route between two nodes, or the cheapest cost to everywhere from one node. The "
            "difference between them is what \"shortest\" is measured in: relationships, or the "
            "value of a relationship property."
        ),
        pages=[
            Page(
                slug="graph-shortestpath",
                title="`graph.shortestpath`",
                family="procedures/shortest-path",
                kind="procedure",
                signature="graph.shortestpath(source, target) YIELD path, cost",
                summary="One path with the fewest relationships between two nodes, and its length.",
                dataset="flights",
                what=(
                    "`graph.shortestpath` finds a route from `source` to `target` following "
                    "outgoing relationships and yields it as a path value together with `cost`, the "
                    "number of relationships on it. When no route exists the procedure yields no "
                    "rows at all, which is how absence is reported."
                ),
                detail=(
                    "`cost` here is a hop count, not a weight. Several routes may tie on hop count; "
                    "the procedure yields one of them, chosen deterministically for a given graph "
                    "state, not all of them.\n\n"
                    "The yielded `path` is an ordinary Cypher path, so `nodes(path)`, "
                    "`relationships(path)` and `length(path)` all apply to it and can be projected "
                    "in the same query. `length(path)` and `cost` agree by construction."
                ),
                when=(
                    "Use it when you need the actual route between two known nodes and every "
                    "relationship counts the same — connections in a journey, hops in a referral "
                    "chain, steps in a dependency chain."
                ),
                differs=(
                    "`graph.dijkstra` measures cost with a numeric relationship property and "
                    "reports the cheapest cost to *every* reachable node rather than one route to "
                    "one node. `graph.bfs` gives the same hop distances as this procedure's `cost` "
                    "but never materialises a path. Cypher's own `shortestPath` pattern selector "
                    "expresses the same idea inside a `MATCH`; this procedure is the form that "
                    "composes with the other algorithms."
                ),
                simple=Example(
                    note="The fewest-flight route from Heathrow to Wellington, New Zealand.",
                    query=(
                        "USE flights\n"
                        "MATCH (origin:Airport {iata: 'LHR'}), (destination:Airport {iata: 'WLG'})\n"
                        "CALL graph.shortestpath(origin, destination) YIELD path, cost\n"
                        "RETURN cost AS flights,\n"
                        "       [airport IN nodes(path) | airport.iata] AS route"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The same route, priced. The hop-count route is not the shortest route in "
                        "kilometres, and putting the two side by side is what makes that visible: "
                        "summing the `km` property along the returned path gives the distance "
                        "actually flown by the fewest-flight itinerary."
                    ),
                    query=(
                        "USE flights\n"
                        "MATCH (origin:Airport {iata: 'LHR'}), (destination:Airport {iata: 'WLG'})\n"
                        "CALL graph.shortestpath(origin, destination) YIELD path, cost\n"
                        "UNWIND relationships(path) AS leg\n"
                        "RETURN cost AS flights,\n"
                        "       count(leg) AS legs,\n"
                        "       round(sum(leg.km)) AS kilometres,\n"
                        "       round(max(leg.km)) AS longest_leg_km"
                    ),
                ),
                use_cases=[
                    "Producing a concrete route to show a person, not just a distance.",
                    "Measuring separation between two named entities in hops.",
                    "Feeding a path into further projection with `nodes`, `relationships` and "
                    "`length`.",
                ],
                limits=SHARED_LIMITS
                + [
                    "No route means no rows. A query that assumes one row per pair will silently "
                    "lose the pair instead of reporting it; use `OPTIONAL MATCH`-style reasoning or "
                    "check the row count.",
                    "Ties are broken deterministically but arbitrarily. Do not read the returned "
                    "route as \"the\" route when several are equally short.",
                    "Every relationship costs one. Use `graph.dijkstra` when they should not.",
                ],
                see_also=[
                    "[`graph.dijkstra`](./graph-dijkstra.md) for weighted cost",
                    "[`graph.bfs`](../traversal/graph-bfs.md) for distances without paths",
                ],
            ),
            Page(
                slug="graph-dijkstra",
                title="`graph.dijkstra`",
                family="procedures/shortest-path",
                kind="procedure",
                signature="graph.dijkstra(source [, weightProperty]) YIELD node, cost, predecessor",
                summary=(
                    "The cheapest cost from one source to every reachable node, measured by a "
                    "numeric relationship property."
                ),
                dataset="flights",
                what=(
                    "`graph.dijkstra` computes the cheapest total cost from `source` to every node "
                    "reachable along outgoing relationships. It yields one row per reachable node "
                    "with the accumulated `cost` and the `predecessor` node on the cheapest route, "
                    "which is what lets you rebuild the route itself.\n\n"
                    "The optional second argument names a relationship property to use as the "
                    "weight. Omit it and every relationship weighs `1`, which makes the cost a hop "
                    "count."
                ),
                detail=(
                    "The weight property must be present and numeric on every relationship the "
                    "search traverses. A missing or non-numeric weight is an error, not a skipped "
                    "edge — the query fails rather than quietly returning a wrong cheapest cost. "
                    "That is deliberate: a silently dropped edge changes the answer without "
                    "changing its shape.\n\n"
                    "`predecessor` is `null` for the source and for nothing else. Following "
                    "`predecessor` backwards from any node reconstructs its cheapest route. The "
                    "procedure yields only reachable nodes, so a node absent from the result has no "
                    "route from the source at all."
                ),
                when=(
                    "Use it whenever the cost of crossing a relationship differs between "
                    "relationships: distance, duration, price, latency, risk. It is also the right "
                    "procedure for one-to-many questions, because a single call prices every "
                    "destination at once."
                ),
                differs=(
                    "`graph.shortestpath` returns one materialised route to one target and counts "
                    "hops. `graph.dijkstra` returns costs to everything and counts weight — and "
                    "with the weight argument omitted the two agree on cost while still differing "
                    "in shape. `graph.bfs` is the unweighted one-to-many form and is cheaper when "
                    "hops are genuinely what you want."
                ),
                simple=Example(
                    note=(
                        "The ten airports closest to Heathrow by total kilometres flown, rather "
                        "than by number of flights. `km` is a numeric property on every `ROUTE` "
                        "relationship in this dataset, computed from the two airports' coordinates."
                    ),
                    query=(
                        "USE flights\n"
                        "MATCH (origin:Airport {iata: 'LHR'})\n"
                        "CALL graph.dijkstra(origin, 'km') YIELD node, cost\n"
                        "WHERE cost > 0\n"
                        "RETURN node.iata AS iata, node.city AS city, round(cost) AS km\n"
                        "ORDER BY cost, iata\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Where the cheapest route is not the most direct one. For each destination "
                        "this compares the cheapest total distance against the great-circle "
                        "distance from Heathrow, and reports the destinations whose best itinerary "
                        "is furthest from a straight line — the detour cost of the route network."
                    ),
                    query=(
                        "USE flights\n"
                        "MATCH (origin:Airport {iata: 'LHR'})\n"
                        "CALL graph.dijkstra(origin, 'km') YIELD node, cost, predecessor\n"
                        "WITH origin, node, cost, predecessor WHERE cost > 2000\n"
                        "WITH node, cost, predecessor,\n"
                        "     6371.0088 * 2 * asin(sqrt(\n"
                        "       sin(radians(node.latitude - origin.latitude) / 2)^2 +\n"
                        "       cos(radians(origin.latitude)) * cos(radians(node.latitude)) *\n"
                        "       sin(radians(node.longitude - origin.longitude) / 2)^2)) AS direct\n"
                        "WHERE direct > 0\n"
                        "RETURN node.iata AS iata, node.city AS city,\n"
                        "       round(cost) AS route_km, round(direct) AS direct_km,\n"
                        "       round(100.0 * cost / direct) AS percent_of_direct,\n"
                        "       predecessor.iata AS arrives_from\n"
                        "ORDER BY percent_of_direct DESC, iata\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Pricing every destination from one origin in a single call.",
                    "Rebuilding the cheapest route to any node by following `predecessor`.",
                    "Comparing network cost against an ideal cost to find where a network detours.",
                ],
                limits=SHARED_LIMITS
                + [
                    "A missing or non-numeric weight on a traversed relationship fails the query.",
                    "Negative weights are not meaningful to this algorithm; costs must be "
                    "non-negative for the result to be the cheapest route.",
                    "The result has one row per reachable node, which for a well-connected graph is "
                    "most of it. Filter inside the query rather than in the client.",
                ],
                see_also=[
                    "[`graph.shortestpath`](./graph-shortestpath.md) for one route by hop count",
                    "[`graph.bfs`](../traversal/graph-bfs.md) for unweighted one-to-many distance",
                ],
            ),
        ],
    )


# --------------------------------------------------------------------------------------------
# centrality
# --------------------------------------------------------------------------------------------


def centrality() -> Family:
    return Family(
        path="procedures/centrality",
        title="Centrality procedures",
        blurb=(
            "Two different answers to \"which nodes matter\". Degree counts relationships. PageRank "
            "weighs an endorsement by the standing of whoever gave it. They disagree often, and "
            "where they disagree is usually the interesting part."
        ),
        pages=[
            Page(
                slug="graph-degree",
                title="`graph.degree`",
                family="procedures/centrality",
                kind="procedure",
                signature="graph.degree() YIELD node, outDegree, inDegree, degree",
                summary="Relationship counts per node, split by direction.",
                dataset="epinions",
                what=(
                    "`graph.degree` yields one row for every node in the graph with three counts: "
                    "`outDegree`, the relationships leaving it; `inDegree`, the relationships "
                    "arriving; and `degree`, their sum. Isolated nodes are yielded too, with zeros, "
                    "so the row count is the node count."
                ),
                detail=(
                    "The split matters more than the total in a directed graph. In a trust network "
                    "`outDegree` is how many people this account trusts and `inDegree` is how many "
                    "trust it — two quite different things that the combined `degree` averages "
                    "away. Reach for the total only when direction genuinely carries no meaning.\n\n"
                    "Every relationship is counted, including parallel relationships between the "
                    "same pair and self-relationships. Degree is a count of relationships, not of "
                    "distinct neighbours."
                ),
                when=(
                    "Use it as the first measurement on any unfamiliar graph. Degree is cheap, it "
                    "needs no parameters, and its distribution tells you immediately whether the "
                    "graph is broadly even or dominated by a few hubs — which decides whether the "
                    "more expensive algorithms will tell you anything."
                ),
                differs=(
                    "Degree is local: it sees only a node's own relationships. `graph.pagerank` is "
                    "recursive and asks who those relationships come *from*. A node can have high "
                    "degree and low PageRank when its many endorsements come from nowhere in "
                    "particular, and the reverse when a handful come from the centre of the graph."
                ),
                simple=Example(
                    note="The ten most-trusted accounts in the Epinions network, by inbound trust.",
                    query=(
                        "USE epinions\n"
                        "CALL graph.degree() YIELD node, inDegree, outDegree\n"
                        "RETURN node.user_id AS user, inDegree AS trusted_by, outDegree AS trusts\n"
                        "ORDER BY inDegree DESC, user\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The shape of the whole degree distribution, bucketed by order of "
                        "magnitude. This is the measurement worth taking before any other "
                        "algorithm: it shows how heavily the graph is concentrated in a few nodes."
                    ),
                    query=(
                        "USE epinions\n"
                        "CALL graph.degree() YIELD node, degree\n"
                        "WITH CASE\n"
                        "       WHEN degree = 0 THEN 0\n"
                        "       ELSE toInteger(floor(log10(toFloat(degree))))\n"
                        "     END AS magnitude, degree\n"
                        "RETURN magnitude,\n"
                        "       count(*) AS accounts,\n"
                        "       min(degree) AS lowest,\n"
                        "       max(degree) AS highest,\n"
                        "       round(avg(degree) * 10) / 10.0 AS mean\n"
                        "ORDER BY magnitude"
                    ),
                ),
                use_cases=[
                    "The first look at an unfamiliar graph.",
                    "Separating who acts from who is acted upon in a directed graph.",
                    "Providing a baseline that a recursive score has to beat to be worth running.",
                ],
                limits=SHARED_LIMITS
                + [
                    "One row per node in the graph, including isolated ones. On a large graph, "
                    "aggregate or filter inside the query.",
                    "Parallel and self relationships are counted individually; degree is not a "
                    "count of distinct neighbours.",
                ],
                see_also=["[`graph.pagerank`](./graph-pagerank.md) for recursive importance"],
            ),
            Page(
                slug="graph-pagerank",
                title="`graph.pagerank`",
                family="procedures/centrality",
                kind="procedure",
                signature="graph.pagerank([damping, tolerance, maxIterations]) YIELD node, score",
                summary="Recursive importance: a score that weighs who points at you, not how many.",
                dataset="epinions",
                what=(
                    "`graph.pagerank` yields one row per node with a score expressing how much of "
                    "the graph's attention settles on it. Importance is recursive: a relationship "
                    "from a node that is itself important contributes more than one from a node "
                    "that is not. Scores across the graph sum to one, so a score is a share of the "
                    "whole rather than an absolute quantity.\n\n"
                    "Called with no arguments the algorithm uses its default damping, tolerance and "
                    "iteration ceiling. All three may be supplied together: damping and tolerance as "
                    "numbers, the iteration ceiling as an integer. Supplying some but not all is "
                    "rejected."
                ),
                detail=(
                    "Damping is the probability that the walk follows a relationship rather than "
                    "restarting somewhere at random. Lower damping concentrates score near "
                    "well-connected regions and converges faster; higher damping lets influence "
                    "travel further from its source. Tolerance and the iteration ceiling bound the "
                    "work: iteration stops when scores stop moving by more than the tolerance, or "
                    "when the ceiling is reached, whichever comes first.\n\n"
                    "Because scores are a share of one, they shrink as the graph grows. Never "
                    "compare a raw score between two graphs of different sizes, and never read a "
                    "score as a probability of anything in the domain. Ranks compare; scores do "
                    "not."
                ),
                when=(
                    "Use PageRank when you want influence rather than volume, and when the "
                    "relationships in your graph genuinely mean endorsement — a citation, a trust "
                    "declaration, a link, a recommendation. On a graph whose relationships mean "
                    "\"happened near\" or \"belongs to\", the recursion has no meaning to propagate."
                ),
                differs=(
                    "`graph.degree` counts relationships and stops there. PageRank asks where they "
                    "came from, which is why the two rankings differ and why running both is more "
                    "informative than running either. For grouping rather than ranking, the "
                    "community procedures answer a different question entirely."
                ),
                simple=Example(
                    note="The ten most influential accounts in the Epinions trust network.",
                    query=(
                        "USE epinions\n"
                        "CALL graph.pagerank() YIELD node, score\n"
                        "RETURN node.user_id AS user, round(score * 1000000) / 1000000.0 AS score\n"
                        "ORDER BY score DESC, user\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Where influence and volume disagree. This ranks accounts by PageRank and "
                        "by inbound degree in the same query and reports the accounts whose "
                        "influence is least explained by how many people trust them — endorsement "
                        "arriving from the centre of the network rather than in bulk."
                    ),
                    query=(
                        "USE epinions\n"
                        "CALL graph.pagerank() YIELD node, score\n"
                        "WITH node, score ORDER BY score DESC LIMIT 100\n"
                        "MATCH (node)<-[trust:TRUSTS]-()\n"
                        "WITH node, score, count(trust) AS trusted_by\n"
                        "RETURN node.user_id AS user,\n"
                        "       round(score * 1000000) / 1000000.0 AS score,\n"
                        "       trusted_by,\n"
                        "       round(score * 100000000 / trusted_by) / 100.0 AS score_per_endorsement\n"
                        "ORDER BY score_per_endorsement DESC, user\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Ranking influence in citation, trust, link and recommendation graphs.",
                    "Prioritising review, moderation or crawling effort.",
                    "Finding nodes whose standing is not explained by their raw connection count.",
                ],
                limits=SHARED_LIMITS
                + [
                    "Scores are a share of one and shrink with graph size. Compare ranks between "
                    "graphs, never raw scores.",
                    "A high score is structural importance, not business value. The graph knows "
                    "nothing about which nodes matter to you.",
                    "Damping, tolerance and the iteration ceiling are supplied together or not at "
                    "all; a partial argument list is rejected.",
                    "Direction is meaning here. Reversing the relationships reverses what the score "
                    "says.",
                ],
                see_also=[
                    "[`graph.degree`](./graph-degree.md) for the local comparison",
                    "[`graph.louvain`](../community/graph-louvain.md) for grouping rather than ranking",
                ],
            ),
        ],
    )


# --------------------------------------------------------------------------------------------
# community and components
# --------------------------------------------------------------------------------------------


def community() -> Family:
    return Family(
        path="procedures/community",
        title="Community and component procedures",
        blurb=(
            "Three ways of cutting a graph into groups. Two are structural facts about "
            "connectivity; the third is an optimisation whose answer depends on the algorithm as "
            "much as on the data."
        ),
        pages=[
            Page(
                slug="graph-wcc",
                title="`graph.wcc`",
                family="procedures/community",
                kind="procedure",
                signature="graph.wcc() YIELD node, component",
                summary="Weakly connected components: the islands of the graph, ignoring direction.",
                dataset="epinions",
                what=(
                    "`graph.wcc` assigns every node an integer component identifier such that two "
                    "nodes share an identifier exactly when a path connects them if relationship "
                    "direction is ignored. It yields one row per node. The identifiers themselves "
                    "carry no meaning beyond grouping — only equality between them does."
                ),
                detail=(
                    "Weak connectivity is a fact about the graph, not an estimate. Run it twice on "
                    "unchanged data and the grouping is identical, although the numbering is not "
                    "something to depend on.\n\n"
                    "Almost every real network has one component holding the large majority of "
                    "nodes and a long tail of tiny ones. That shape is the useful output: the size "
                    "of the largest component tells you how much of the graph is actually one "
                    "connected object, and everything outside it is unreachable from everything "
                    "inside it."
                ),
                when=(
                    "Run it early, before anything expensive. Algorithms that assume connectivity "
                    "produce misleading output when the graph is really several disconnected "
                    "pieces, and this is the cheapest way to find that out."
                ),
                differs=(
                    "`graph.scc` requires a path in *both* directions and therefore cuts the same "
                    "graph much more finely. `graph.louvain` does not answer a connectivity "
                    "question at all: it looks for densely connected groups *within* what is "
                    "already connected."
                ),
                simple=Example(
                    note="How the Epinions network divides into disconnected islands.",
                    query=(
                        "USE epinions\n"
                        "CALL graph.wcc() YIELD node, component\n"
                        "WITH component, count(node) AS members\n"
                        "RETURN members AS component_size, count(*) AS components\n"
                        "ORDER BY component_size DESC\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The share of the graph held by its largest component, computed in one "
                        "query. A number close to 100 means connectivity questions can be treated "
                        "as global; a much lower one means every later algorithm is really being "
                        "run over several unrelated graphs at once."
                    ),
                    query=(
                        "USE epinions\n"
                        "CALL graph.wcc() YIELD node, component\n"
                        "WITH component, count(node) AS members\n"
                        "RETURN count(*) AS components,\n"
                        "       sum(members) AS nodes,\n"
                        "       max(members) AS largest_component,\n"
                        "       min(members) AS smallest_component,\n"
                        "       round(10000.0 * max(members) / sum(members)) / 100.0 AS percent_in_largest"
                    ),
                ),
                use_cases=[
                    "Checking that a graph is one object before trusting a global measurement.",
                    "Separating a main network from imported fragments and orphans.",
                    "Sizing the reachable universe a later algorithm will actually operate on.",
                ],
                limits=SHARED_LIMITS
                + [
                    "Component identifiers are grouping labels. Do not store them as stable "
                    "identity or compare them across runs.",
                    "Direction is discarded. A component says two nodes are connected somehow, not "
                    "that either can reach the other.",
                ],
                see_also=[
                    "[`graph.scc`](./graph-scc.md) for directed connectivity",
                    "[`graph.louvain`](./graph-louvain.md) for density rather than connectivity",
                ],
            ),
            Page(
                slug="graph-scc",
                title="`graph.scc`",
                family="procedures/community",
                kind="procedure",
                signature="graph.scc() YIELD node, component",
                summary=(
                    "Strongly connected components: groups where every node can reach every other, "
                    "following direction."
                ),
                dataset="epinions",
                what=(
                    "`graph.scc` assigns every node an integer component identifier such that two "
                    "nodes share an identifier exactly when each can reach the other by following "
                    "relationship direction. It yields one row per node. A node with no reciprocal "
                    "route to anything forms a component of its own."
                ),
                detail=(
                    "The requirement is mutual reachability, which is far stronger than "
                    "connectivity. In a directed network most nodes end up alone: they can be "
                    "reached, or they can reach others, but not both. The result is usually one "
                    "large mutually-reachable core plus a very large number of singletons, and the "
                    "size of that core is the number worth reading.\n\n"
                    "That core is the part of the graph where influence can circulate. Outside it, "
                    "everything flows one way and never returns."
                ),
                when=(
                    "Use it when direction carries obligation or flow and cycles matter: mutual "
                    "trust, circular dependencies, feedback loops, money moving in a circle. On an "
                    "undirected graph it degenerates to the same answer as `graph.wcc`."
                ),
                differs=(
                    "`graph.wcc` ignores direction and produces far fewer, far larger groups. The "
                    "gap between the two results is itself informative: it measures how one-way the "
                    "graph is. `graph.louvain` optimises for density and will happily group nodes "
                    "with no reciprocal route at all."
                ),
                simple=Example(
                    note=(
                        "The strongly connected components of the Epinions trust network, by size. "
                        "The long tail of size-one components is the expected shape."
                    ),
                    query=(
                        "USE epinions\n"
                        "CALL graph.scc() YIELD node, component\n"
                        "WITH component, count(node) AS members\n"
                        "RETURN members AS component_size, count(*) AS components\n"
                        "ORDER BY component_size DESC\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "How much smaller directed connectivity is than undirected connectivity on "
                        "the same graph. The ratio between the largest strongly connected component "
                        "and the largest weakly connected one measures how much of the network's "
                        "apparent cohesion survives once direction is respected."
                    ),
                    query=(
                        "USE epinions\n"
                        "CALL graph.scc() YIELD node, component\n"
                        "WITH component, count(node) AS members\n"
                        "WITH count(*) AS components, sum(members) AS nodes,\n"
                        "     max(members) AS largest,\n"
                        "     sum(CASE WHEN members = 1 THEN 1 ELSE 0 END) AS singletons\n"
                        "RETURN components, nodes, largest AS largest_strong_component, singletons,\n"
                        "       round(10000.0 * largest / nodes) / 100.0 AS percent_in_largest,\n"
                        "       round(10000.0 * singletons / components) / 100.0 AS percent_singletons"
                    ),
                ),
                use_cases=[
                    "Finding circular dependencies in a directed graph.",
                    "Isolating the mutually reachable core where influence can circulate.",
                    "Measuring how one-way a network is by comparing against weak components.",
                ],
                limits=SHARED_LIMITS
                + [
                    "Component identifiers are grouping labels only.",
                    "On an undirected graph the result is the same as `graph.wcc` at higher cost.",
                    "The result is dominated by singletons in most real directed graphs; aggregate "
                    "rather than listing rows.",
                ],
                see_also=["[`graph.wcc`](./graph-wcc.md) for undirected connectivity"],
            ),
            Page(
                slug="graph-louvain",
                title="`graph.louvain`",
                family="procedures/community",
                kind="procedure",
                signature="graph.louvain() YIELD node, community",
                summary=(
                    "Communities found by modularity optimisation: groups that are denser inside "
                    "than the graph is on average."
                ),
                dataset="email",
                what=(
                    "`graph.louvain` partitions the graph into communities by repeatedly moving "
                    "nodes between groups to increase modularity — the degree to which "
                    "relationships fall inside groups rather than between them. It yields one row "
                    "per node with an integer community identifier.\n\n"
                    "Unlike the component procedures, this is an optimisation and not a fact. The "
                    "partition it returns is a good one, not the only good one."
                ),
                detail=(
                    "Every node receives a community, including nodes that belong nowhere in "
                    "particular. The algorithm does not report confidence, so a community "
                    "assignment carries no claim that the node really belongs there. Judge the "
                    "partition as a whole, by whether communities line up with something you "
                    "already know about the data, rather than trusting any individual assignment.\n\n"
                    "Community identifiers have no meaning beyond grouping, and modularity "
                    "optimisation has a known resolution limit: below a certain size, genuinely "
                    "distinct groups get merged because splitting them does not improve the global "
                    "score. Small communities in the output are less trustworthy than large ones."
                ),
                when=(
                    "Use it to find structure you have no labels for — the natural groupings in a "
                    "network nobody has categorised. Where labels already exist, the interesting "
                    "use is comparison: agreement confirms the labels describe real structure, and "
                    "disagreement points at where they do not."
                ),
                differs=(
                    "`graph.wcc` and `graph.scc` answer a connectivity question with one correct "
                    "answer. Louvain answers a density question with a good answer. Two nodes in "
                    "different Louvain communities are usually still connected; two nodes in "
                    "different weakly connected components never are."
                ),
                simple=Example(
                    note=(
                        "The communities Louvain finds in the email network of a European research "
                        "institution, by size."
                    ),
                    query=(
                        "USE email\n"
                        "CALL graph.louvain() YIELD node, community\n"
                        "WITH community, count(node) AS members\n"
                        "RETURN community, members\n"
                        "ORDER BY members DESC, community\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The same communities checked against ground truth. Every member of this "
                        "network has a known department, which the algorithm never sees. For each "
                        "community this reports its dominant department and what share of the "
                        "community that department accounts for — a direct measure of whether the "
                        "structure found matches the structure that exists."
                    ),
                    query=(
                        "USE email\n"
                        "CALL graph.louvain() YIELD node, community\n"
                        "WITH community, node.department AS department, count(*) AS members\n"
                        "ORDER BY community, members DESC\n"
                        "WITH community, collect(department) AS departments,\n"
                        "     collect(members) AS counts, sum(members) AS size\n"
                        "WHERE size >= 20\n"
                        "RETURN community, size,\n"
                        "       head(departments) AS dominant_department,\n"
                        "       head(counts) AS from_that_department,\n"
                        "       round(1000.0 * head(counts) / size) / 10.0 AS purity_percent\n"
                        "ORDER BY size DESC, community"
                    ),
                ),
                use_cases=[
                    "Finding structure in a network that has never been categorised.",
                    "Testing whether existing labels describe real structural groups.",
                    "Reducing a large graph to a manageable number of groups before analysis.",
                ],
                limits=SHARED_LIMITS
                + [
                    "The partition is an optimisation result, not a fact about the data. A "
                    "different run over changed data can reorganise groups substantially.",
                    "Every node gets a community, including nodes that belong to none. There is no "
                    "confidence output.",
                    "Modularity has a resolution limit: small genuine communities are merged into "
                    "larger ones. Treat small communities with suspicion.",
                    "Community identifiers are grouping labels only and are not stable identity.",
                ],
                see_also=[
                    "[`graph.wcc`](./graph-wcc.md) for connectivity as a fact",
                    "[`graph.kcore`](../structure/graph-kcore.md) for cohesion by depth rather than grouping",
                ],
            ),
        ],
    )


# --------------------------------------------------------------------------------------------
# local structure
# --------------------------------------------------------------------------------------------


def structure() -> Family:
    return Family(
        path="procedures/structure",
        title="Local structure procedures",
        blurb=(
            "How tightly knit the graph is, measured three ways: closed triangles across the whole "
            "graph, the same idea per node, and how deep into a densely connected core each node "
            "survives."
        ),
        pages=[
            Page(
                slug="graph-trianglecount",
                title="`graph.trianglecount`",
                family="procedures/structure",
                kind="procedure",
                signature="graph.trianglecount() YIELD triangleCount",
                summary="One number: how many closed triangles the whole graph contains.",
                dataset="social",
                what=(
                    "`graph.trianglecount` yields a single row with the total number of triangles "
                    "in the graph — sets of three nodes each connected to the other two. It is the "
                    "only procedure here that produces one row rather than one row per node."
                ),
                detail=(
                    "A triangle is the smallest possible evidence of clustering: it means two of a "
                    "node's neighbours are themselves connected. A graph rich in triangles has "
                    "genuine communities; a graph with almost none is a tree, a star, or a chain, "
                    "whatever its size.\n\n"
                    "The count on its own is hard to interpret, because it grows steeply with "
                    "degree. It becomes meaningful when compared: against another graph of similar "
                    "size, against the same graph at another time, or against the relationship "
                    "count as a crude density ratio."
                ),
                when=(
                    "Use it as a single summary statistic for how clustered a graph is — a quick "
                    "check before deciding whether community detection has anything to find."
                ),
                differs=(
                    "`graph.clusteringcoefficient` computes the same underlying idea per node and "
                    "normalises it, which makes individual nodes comparable. This procedure gives "
                    "the graph-wide total and nothing else."
                ),
                simple=Example(
                    note=(
                        "The triangle count of a dense friendship network. Friendship graphs are "
                        "triangle-rich because friends of friends are frequently friends."
                    ),
                    query="USE social\nCALL graph.trianglecount() YIELD triangleCount\nRETURN triangleCount",
                ),
                advanced=Example(
                    note=(
                        "Triangle density compared across two graphs of very different character: a "
                        "friendship network, where mutual connection is the norm, and a citation "
                        "network, where it is nearly impossible because papers cite backwards in "
                        "time. Normalising by relationship count makes the two comparable."
                    ),
                    query=(
                        "USE social\n"
                        "CALL graph.trianglecount() YIELD triangleCount\n"
                        "MATCH ()-[relationship]->()\n"
                        "RETURN 'social' AS graph,\n"
                        "       triangleCount AS triangles,\n"
                        "       count(relationship) AS relationships,\n"
                        "       round(1000.0 * triangleCount / count(relationship) * 100) / 100.0\n"
                        "         AS triangles_per_1000_relationships"
                    ),
                ),
                use_cases=[
                    "A one-number summary of how clustered a graph is.",
                    "Deciding whether community detection is worth running.",
                    "Tracking clustering of one graph over time.",
                ],
                limits=SHARED_LIMITS
                + [
                    "A bare count is not comparable between graphs of different sizes. Normalise "
                    "before comparing.",
                    "Triangle counting is quadratic in the degree of the densest nodes and is the "
                    "most expensive of the structure procedures on a hub-heavy graph.",
                ],
                see_also=[
                    "[`graph.clusteringcoefficient`](./graph-clusteringcoefficient.md) for the "
                    "per-node, normalised form"
                ],
            ),
            Page(
                slug="graph-clusteringcoefficient",
                title="`graph.clusteringcoefficient`",
                family="procedures/structure",
                kind="procedure",
                signature="graph.clusteringcoefficient() YIELD node, coefficient",
                summary=(
                    "Per node, the share of its neighbours that are connected to each other."
                ),
                dataset="social",
                what=(
                    "`graph.clusteringcoefficient` yields one row per node with a coefficient "
                    "between `0` and `1`: the fraction of the possible connections among that "
                    "node's neighbours that actually exist. A node whose neighbours all know each "
                    "other scores `1`; a node at the centre of a star scores `0`."
                ),
                detail=(
                    "Because it is normalised by degree, the coefficient is comparable across nodes "
                    "of very different sizes — which is exactly what a raw triangle count is not. "
                    "That normalisation has a sharp edge: a node with one neighbour has no possible "
                    "connections among its neighbours and scores `0`, which looks identical to a "
                    "genuinely unclustered hub. Filter by degree before ranking on this value.\n\n"
                    "High coefficient and high degree together is the rare and interesting "
                    "combination: a node with many neighbours who nonetheless mostly know each "
                    "other sits inside a dense community rather than bridging between them."
                ),
                when=(
                    "Use it to distinguish nodes embedded in a tight group from nodes that connect "
                    "otherwise separate groups. The low-coefficient, high-degree nodes are the "
                    "bridges, and they are usually the ones worth looking at."
                ),
                differs=(
                    "`graph.trianglecount` gives one number for the whole graph and cannot say "
                    "anything about individual nodes. `graph.kcore` measures cohesion by asking how "
                    "deep a node survives repeated peeling, which finds dense regions rather than "
                    "dense neighbourhoods."
                ),
                simple=Example(
                    note=(
                        "The most tightly embedded people in the friendship network, restricted to "
                        "those with enough neighbours for the coefficient to mean something."
                    ),
                    query=(
                        "USE social\n"
                        "MATCH (person:Person)-[friendship:FRIEND]-()\n"
                        "WITH person, count(friendship) AS degree WHERE degree >= 50\n"
                        "WITH collect(person.person_id) AS dense\n"
                        "CALL graph.clusteringcoefficient() YIELD node, coefficient\n"
                        "WITH dense, node, coefficient WHERE node.person_id IN dense\n"
                        "RETURN node.person_id AS person,\n"
                        "       round(coefficient * 10000) / 10000.0 AS coefficient\n"
                        "ORDER BY coefficient DESC, person\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "The bridges. These are the people with many connections whose connections "
                        "do not know each other — the opposite of the previous example, and the "
                        "structurally significant one: remove a bridge and otherwise separate parts "
                        "of the network lose their link."
                    ),
                    query=(
                        "USE social\n"
                        "MATCH (person:Person)-[friendship:FRIEND]-()\n"
                        "WITH person, count(friendship) AS degree WHERE degree >= 50\n"
                        "WITH collect(person.person_id) AS dense\n"
                        "CALL graph.clusteringcoefficient() YIELD node, coefficient\n"
                        "WITH dense, node, coefficient WHERE node.person_id IN dense\n"
                        "MATCH (node)-[friendship:FRIEND]-()\n"
                        "WITH node, coefficient, count(friendship) AS connections\n"
                        "RETURN node.person_id AS person, connections,\n"
                        "       round(coefficient * 10000) / 10000.0 AS coefficient,\n"
                        "       round(connections * (1 - coefficient)) AS unclustered_connections\n"
                        "ORDER BY coefficient, person\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Separating nodes inside a community from nodes bridging between communities.",
                    "Finding structurally critical nodes whose removal would disconnect groups.",
                    "Comparing local cohesion across nodes of very different degree.",
                ],
                limits=SHARED_LIMITS
                + [
                    "A node with fewer than two neighbours scores `0` for want of any possible "
                    "connection, not because it is unclustered. Filter by degree first.",
                    "The coefficient describes a node's immediate neighbourhood only and says "
                    "nothing about the wider graph.",
                ],
                see_also=[
                    "[`graph.trianglecount`](./graph-trianglecount.md) for the graph-wide total",
                    "[`graph.kcore`](./graph-kcore.md) for cohesion by depth",
                ],
            ),
            Page(
                slug="graph-kcore",
                title="`graph.kcore`",
                family="procedures/structure",
                kind="procedure",
                signature="graph.kcore() YIELD node, core",
                summary=(
                    "How deep into the graph's densely connected interior each node survives."
                ),
                dataset="social",
                what=(
                    "`graph.kcore` yields one row per node with its core number: the largest `k` "
                    "for which the node belongs to a subgraph where every node has at least `k` "
                    "neighbours. It is computed by repeatedly removing the least connected nodes "
                    "and recording when each one falls away."
                ),
                detail=(
                    "The peeling is what makes the number meaningful. A node with a hundred "
                    "relationships to nodes that have one each is removed early and gets a low core "
                    "number, because its neighbours do not survive. A node with ten relationships "
                    "to nodes that also have ten survives deep. Core number therefore measures the "
                    "density of the region a node sits in, not the node's own count.\n\n"
                    "The highest core number in a graph is a property of the graph itself, and the "
                    "nodes that reach it form its densest interior."
                ),
                when=(
                    "Use it to find the resilient centre of a network, or to strip away a "
                    "periphery cheaply. Filtering to nodes above a core threshold is one of the "
                    "most effective ways to reduce a large graph before an expensive algorithm, "
                    "because it removes the sparse edges without breaking the dense middle."
                ),
                differs=(
                    "`graph.degree` counts a node's own relationships and is fooled by hubs "
                    "attached to nothing. `graph.clusteringcoefficient` looks only at a node's "
                    "immediate neighbours. `graph.kcore` is the one that accounts for the "
                    "connectedness of the neighbours' neighbours, through peeling."
                ),
                simple=Example(
                    note="How the friendship network's population is distributed across core depths.",
                    query=(
                        "USE social\n"
                        "CALL graph.kcore() YIELD node, core\n"
                        "RETURN core, count(node) AS people\n"
                        "ORDER BY core DESC\n"
                        "LIMIT 10"
                    ),
                ),
                advanced=Example(
                    note=(
                        "Where core depth and raw degree disagree. This reports, for each core "
                        "depth, the range of degrees found there — showing that a high relationship "
                        "count does not put a node in the dense interior, because a node is only as "
                        "deep as the company it keeps."
                    ),
                    query=(
                        "USE social\n"
                        "CALL graph.kcore() YIELD node, core\n"
                        "MATCH (node)-[friendship:FRIEND]-()\n"
                        "WITH core, node, count(friendship) AS degree\n"
                        "RETURN core,\n"
                        "       count(node) AS people,\n"
                        "       min(degree) AS lowest_degree,\n"
                        "       round(avg(degree) * 10) / 10.0 AS mean_degree,\n"
                        "       max(degree) AS highest_degree\n"
                        "ORDER BY core DESC\n"
                        "LIMIT 10"
                    ),
                ),
                use_cases=[
                    "Extracting the resilient core of a network.",
                    "Cheaply reducing a large graph before an expensive algorithm.",
                    "Distinguishing genuinely central nodes from hubs attached to a sparse fringe.",
                ],
                limits=SHARED_LIMITS
                + [
                    "The core number is coarse. Many nodes share a value, so it groups rather than "
                    "ranks.",
                    "Direction is ignored; a node's core number is computed from its combined "
                    "relationships.",
                ],
                see_also=[
                    "[`graph.degree`](../centrality/graph-degree.md) for the local count it corrects",
                    "[`graph.louvain`](../community/graph-louvain.md) for grouping rather than depth",
                ],
            ),
        ],
    )
