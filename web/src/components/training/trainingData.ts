export interface TrainingLesson {
  id: string;
  dataset: string;
  eyebrow: string;
  title: string;
  question: string;
  query: string;
  explanation: string[];
}

export const DATASETS = [
  { slug: 'flights', title: 'OpenFlights routes', scale: '13,859 nodes · 66,770 routes' },
  { slug: 'trust', title: 'Bitcoin OTC trust', scale: '5,881 accounts · 35,592 ratings' },
  { slug: 'epinions', title: 'Epinions trust network', scale: '75,879 users · 508,837 links' },
  { slug: 'email', title: 'European institution email', scale: '1,005 members · 24,929 messages' },
  { slug: 'social', title: 'Facebook ego networks', scale: '4,039 people · 88,234 friendships' },
  { slug: 'library', title: 'arXiv abstract library', scale: '8 papers · semantic index' },
] as const;

export const TRAINING_LESSONS: TrainingLesson[] = [
  {
    id: 'closest-from-heathrow', dataset: 'flights', eyebrow: '01 · Weighted paths',
    title: 'How far can Heathrow reach?',
    question: 'Find the ten airports closest to Heathrow by total route kilometres, not by hop count.',
    query: `USE flights
MATCH (origin:Airport {iata: 'LHR'})
CALL graph.dijkstra(origin, 'km') YIELD node, cost
WHERE cost > 0
RETURN node.iata AS iata, node.city AS city, round(cost) AS km
ORDER BY cost, iata
LIMIT 10`,
    explanation: [
      '`MATCH` binds Heathrow as the starting node using its IATA identifier.',
      '`graph.dijkstra` expands routes and minimizes the `km` property carried by each relationship.',
      '`YIELD node, cost` exposes every reached airport and the accumulated distance; the final projection keeps the first ten.',
    ],
  },
  {
    id: 'route-detours', dataset: 'flights', eyebrow: '02 · Derived measures',
    title: 'Which routes make the largest detour?',
    question: 'Compare network distance with great-circle distance to expose circuitous connections.',
    query: `USE flights
MATCH (origin:Airport {iata: 'LHR'})
CALL graph.dijkstra(origin, 'km') YIELD node, cost, predecessor
WITH origin, node, cost, predecessor WHERE cost > 2000
WITH node, cost, predecessor,
     6371.0088 * 2 * asin(sqrt(
       sin(radians(node.latitude - origin.latitude) / 2)^2 +
       cos(radians(origin.latitude)) * cos(radians(node.latitude)) *
       sin(radians(node.longitude - origin.longitude) / 2)^2)) AS direct
WHERE direct > 0
RETURN node.iata AS iata, node.city AS city,
       round(cost) AS route_km, round(direct) AS direct_km,
       round(100.0 * cost / direct) AS percent_of_direct,
       predecessor.iata AS arrives_from
ORDER BY percent_of_direct DESC, iata
LIMIT 10`,
    explanation: [
      'The first `WITH` keeps only substantial journeys and carries the origin into the next calculation.',
      'The Haversine expression derives straight-line distance from stored coordinates without adding graph state.',
      'Dividing route distance by direct distance turns the two measures into a comparable detour percentage.',
    ],
  },
  {
    id: 'rating-years', dataset: 'trust', eyebrow: '03 · Temporal windows',
    title: 'How did trust change by year?',
    question: 'Group real rating events into tumbling 365-day windows and compare volume with sentiment.',
    query: `USE trust
MATCH ()-[rating:RATED]->()
WINDOW TUMBLING duration('P365D') ON rating.at AS year
WITH year, rating
RETURN datetime.fromepoch(year.start / 1000000000, 0) AS window_start,
       count(rating) AS ratings,
       round(avg(rating.rating) * 1000) / 1000.0 AS mean_rating
ORDER BY year.start`,
    explanation: [
      '`WINDOW TUMBLING` assigns every relationship to exactly one fixed-width time bucket.',
      '`ON rating.at` uses the event timestamp rather than load time.',
      'The aggregate projection returns one row per window: activity and mean signed rating side by side.',
    ],
  },
  {
    id: 'rating-spread', dataset: 'trust', eyebrow: '04 · Distribution',
    title: 'What does a typical rating look like?',
    question: 'Compare discrete and interpolated percentiles across the signed rating scale.',
    query: `USE trust
MATCH ()-[rating:RATED]->()
RETURN percentiledisc(rating.rating, 0.25) AS q1_discrete,
       percentilecont(rating.rating, 0.25) AS q1_continuous,
       percentiledisc(rating.rating, 0.5) AS median_discrete,
       percentilecont(rating.rating, 0.5) AS median_continuous,
       percentiledisc(rating.rating, 0.9) AS p90_discrete,
       percentilecont(rating.rating, 0.9) AS p90_continuous`,
    explanation: [
      'The pattern binds every `RATED` relationship, making its integer `rating` the input series.',
      '`percentiledisc` always returns a rating that exists; `percentilecont` may interpolate between two observations.',
      'Putting both forms in one projection makes the modelling choice visible rather than implicit.',
    ],
  },
  {
    id: 'page-rank', dataset: 'epinions', eyebrow: '05 · Centrality',
    title: 'Who carries structural trust?',
    question: 'Rank Epinions users by the authority passed through incoming trust links.',
    query: `USE epinions
CALL graph.pagerank() YIELD node, score
RETURN node.user_id AS user, round(score * 1000000) / 1000000.0 AS score
ORDER BY score DESC, user
LIMIT 10`,
    explanation: [
      '`graph.pagerank` runs over the admitted project graph and yields one score per node.',
      'The query projects the domain identifier instead of exposing an internal graph id.',
      'The secondary user sort makes ties deterministic, which keeps repeated investigations comparable.',
    ],
  },
  {
    id: 'strong-components', dataset: 'epinions', eyebrow: '06 · Components',
    title: 'How reciprocal is the trust network?',
    question: 'Measure strongly connected groups: users who can all reach one another along directed links.',
    query: `USE epinions
CALL graph.scc() YIELD node, component
WITH component, count(node) AS members
WITH count(*) AS components, sum(members) AS nodes,
     max(members) AS largest,
     sum(CASE WHEN members = 1 THEN 1 ELSE 0 END) AS singletons
RETURN components, nodes, largest AS largest_strong_component, singletons,
       round(10000.0 * largest / nodes) / 100.0 AS percent_in_largest,
       round(10000.0 * singletons / components) / 100.0 AS percent_singletons`,
    explanation: [
      '`graph.scc` labels nodes that share mutual directed reachability.',
      'The first `WITH` changes the grain from one row per user to one row per component.',
      'The second aggregation summarizes that derived component table without writing component ids back into the graph.',
    ],
  },
  {
    id: 'email-communities', dataset: 'email', eyebrow: '07 · Communities',
    title: 'Do email groups resemble departments?',
    question: 'Discover communication communities, then compare each with the known department labels.',
    query: `USE email
CALL graph.louvain() YIELD node, community
WITH community, node.department AS department, count(*) AS members
ORDER BY community, members DESC
WITH community, collect(department) AS departments,
     collect(members) AS counts, sum(members) AS size
WHERE size >= 20
RETURN community, size,
       head(departments) AS dominant_department,
       head(counts) AS from_that_department,
       round(1000.0 * head(counts) / size) / 10.0 AS purity_percent
ORDER BY size DESC, community`,
    explanation: [
      'Louvain derives communities from connectivity alone; it never sees the department property.',
      'Ordering before `collect` puts the most common department first inside each community.',
      'Purity is the share of members from that dominant department, an interpretable check against ground truth.',
    ],
  },
  {
    id: 'social-triangles', dataset: 'social', eyebrow: '08 · Closure',
    title: 'How much friendship closes into triangles?',
    question: 'Count closed triples and normalize them against relationship volume.',
    query: `USE social
CALL graph.trianglecount() YIELD triangleCount
MATCH ()-[relationship]->()
RETURN 'social' AS graph,
       triangleCount AS triangles,
       count(relationship) AS relationships,
       round(1000.0 * triangleCount / count(relationship) * 100) / 100.0
         AS triangles_per_1000_relationships`,
    explanation: [
      '`graph.trianglecount` produces a project-wide structural measurement.',
      'The following `MATCH` counts canonical relationships in the same project.',
      'Normalization makes the triangle count easier to compare with another graph of a different size.',
    ],
  },
  {
    id: 'social-core', dataset: 'social', eyebrow: '09 · Cohesion',
    title: 'Where is the dense social core?',
    question: 'Group people by the deepest k-core in which they remain connected.',
    query: `USE social
CALL graph.kcore() YIELD node, core
MATCH (node)-[friendship:FRIEND]-()
WITH core, node, count(friendship) AS degree
RETURN core,
       count(node) AS people,
       min(degree) AS lowest_degree,
       round(avg(degree) * 10) / 10.0 AS mean_degree,
       max(degree) AS highest_degree
ORDER BY core DESC
LIMIT 10`,
    explanation: [
      'K-core repeatedly removes nodes below a degree threshold and records the deepest surviving shell.',
      'The relationship match calculates observed degree for each yielded person.',
      'Grouping by `core` contrasts the algorithmic shell with its members’ actual degree distribution.',
    ],
  },
  {
    id: 'semantic-papers', dataset: 'library', eyebrow: '10 · Semantic retrieval',
    title: 'Which papers discuss black-hole horizons?',
    question: 'Search complete abstract text through the declared embedding index.',
    query: `USE library
MATCH (paper:Paper)
SEARCH paper IN (EMBEDDING INDEX abstract_semantic
                 FOR TEXT 'thermodynamics of black hole horizons' LIMIT 4)
  SCORE AS score
RETURN paper.title AS title,
       round(score * 10000) / 10000.0 AS score
ORDER BY score DESC`,
    explanation: [
      '`MATCH` defines the candidate label while `SEARCH` chooses the declared derived index.',
      'The query text is embedded locally; complete abstracts remain on their owning `Paper` nodes.',
      '`SCORE AS score` makes relevance explicit and available for projection, filtering, and ordering.',
    ],
  },
];
