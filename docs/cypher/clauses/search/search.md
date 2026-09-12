# `SEARCH`

> Filters and ranks already-bound rows by similarity, binding the score.

| | |
| --- | --- |
| Kind | Clause |
| Signature | `SEARCH <variable> IN (EMBEDDING INDEX <name> FOR TEXT\|VECTOR <input> LIMIT <n>) SCORE AS <alias>` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`library`](../../datasets.md#library) — Eight arXiv papers with embedded abstracts |

## What it does

`SEARCH` takes a variable a preceding `MATCH` has bound, asks a vector index for the most similar entities to an input, keeps the rows whose variable is among them, and binds the similarity as a new variable.

`FOR TEXT` supplies a phrase, which the database encodes with the same model the index was built from. `FOR VECTOR` supplies a vector directly, for when you already have one — a stored embedding, or an average of several.

## How it behaves

The variable must already be bound. `SEARCH` is a filter over existing rows, not a source of them, so it always follows a `MATCH` — which is exactly what lets an ordinary graph filter and a similarity ranking apply to the same query.

`LIMIT` inside the parentheses bounds how many candidates the index returns, and is not the same as a `LIMIT` on the query. It is the search depth: raise it when a graph filter after the search would otherwise discard most candidates and leave too few rows.

The score's meaning follows the index's declared similarity. Under cosine it runs from `1` for identical direction down through `0` for unrelated to negative for opposed, so a negative score is a real signal rather than an error. Scores are comparable within one result and not across indexes.

## When to use it

Use it when the question is "which of these is most like that" and the candidates are already narrowed by the graph: the most relevant paper among those a person cited, the closest document among those a team owns.

## How it differs from its neighbours

It resembles a `WHERE` clause that also ranks. Unlike `WHERE` it consults an index rather than evaluating a predicate per row, and unlike `ORDER BY` it removes rows as well as ordering them. The `vector.*` functions compute similarity without any index, which is right for a handful of rows and wrong for a corpus.

## Simple example

The papers most about black hole thermodynamics, ranked. Nothing in the query mentions the words in any title — the ranking comes from the abstracts' meaning.

```cypher
USE library
MATCH (paper:Paper)
SEARCH paper IN (EMBEDDING INDEX abstract_semantic
                 FOR TEXT 'thermodynamics of black hole horizons' LIMIT 4)
  SCORE AS score
RETURN paper.title AS title,
       round(score * 10000) / 10000.0 AS score
ORDER BY score DESC
```

Result:

```
title                                                                  | score 
-----------------------------------------------------------------------+-------
Quantum extreme black holes at finite temperature and exactly solvable | 0.3982
Lorentzian and Euclidean Quantum Gravity - Analytical and Numerical    | 0.1777
A Cosmological Mechanism for Stabilizing Moduli                        | 0.1581
The String Uncertainty Relations follow from the New Relativity        | 0.1443

4 rows
```

## Advanced example

A semantic ranking narrowed by an ordinary predicate. The search supplies eight candidates and a score; the `WHERE` that follows keeps only those scoring above a threshold and carrying a submission date, so a similarity filter and an ordinary one compose in a single statement rather than in two round trips.

```cypher
USE library
MATCH (paper:Paper)
SEARCH paper IN (EMBEDDING INDEX abstract_semantic
                 FOR TEXT 'geometry of spacetime at short distances'
                 LIMIT 8)
  SCORE AS score
WITH paper, score
WHERE paper.submitted IS NOT NULL AND score > 0.25
RETURN paper.arxiv_id AS arxiv_id,
       paper.submitted AS submitted,
       paper.title AS title,
       round(score * 10000) / 10000.0 AS score
ORDER BY score DESC, arxiv_id
```

Result:

```
arxiv_id       | submitted  | title                                                               | score 
---------------+------------+---------------------------------------------------------------------+-------
hep-th/0001124 | 2000-01-14 | Lorentzian and Euclidean Quantum Gravity - Analytical and Numerical | 0.3113
hep-th/0001027 | 2000-01-04 | Bi-local Fields in Noncommutative Field Theory                      | 0.2949
hep-th/0001023 | 2000-01-04 | The String Uncertainty Relations follow from the New Relativity     | 0.2681

3 rows
```

## Where it earns its place

- Ranking a graph-narrowed candidate set by meaning.
- Retrieval where the query phrase and the documents share no vocabulary.
- Combining similarity with ordinary predicates in a single statement.

## Limitations and trade-offs

- The variable must already be bound; `SEARCH` filters rows rather than producing them.
- The inner `LIMIT` is search depth, not result size. A later filter can leave fewer rows than expected.
- Scores are comparable within one result, not between indexes or similarities.
- A vector index that failed its recall floor rejects the search outright.

## See also

- [`CREATE EMBEDDING INDEX`](./create-embedding-index.md)
- [`vector.cosine`](../../functions/vector/vector-cosine.md)
