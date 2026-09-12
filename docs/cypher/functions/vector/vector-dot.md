# `vector.dot`

> Dot product: alignment scaled by both vectors' magnitudes.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `vector.dot(a, b)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`library`](../../datasets.md#library) — Eight arXiv papers with embedded abstracts |

## What it does

`vector.dot` returns the sum of the element-wise products. It grows with alignment and with the length of either vector, so it mixes direction and magnitude into one number.

## How it behaves

The arguments are lists of numbers. Both must be the same length; a length mismatch is an error rather than a silently truncated comparison. A null argument makes the result null, as with every other scalar function.

These functions do the arithmetic in the query and consult no index, so they are exactly as fast as the number of rows they run over. That makes them right for comparing a handful of vectors and wrong for scanning a corpus, which is what an embedding index and `SEARCH` exist for.

## When to use it

Use it when magnitude is part of the signal — a weight, a count, a confidence baked into the vector's length — or when the vectors are already normalised, in which case it equals the cosine and costs less.

## How it differs from its neighbours

It is `vector.cosine` before the division by both magnitudes. On unit vectors the two agree exactly; on anything else the dot product rewards length, which is either the point or a bug depending on what the vectors mean.

## Simple example

A dot product, and the same vectors normalised so it equals the cosine.

```cypher
USE library
WITH [3.0, 4.0] AS a, [4.0, 3.0] AS b
RETURN vector.dot(a, b) AS raw_dot,
       round(vector.dot(vector.normalize(a), vector.normalize(b)) * 10000)
         / 10000.0 AS normalised_dot,
       round(vector.cosine(a, b) * 10000) / 10000.0 AS cosine
```

Result:

```
raw_dot | normalised_dot | cosine
--------+----------------+-------
24      | 0.96           | 0.96  

1 row
```

## Advanced example

Where the dot product misleads. A vector that is only weakly aligned but much longer outscores a shorter, better-aligned one — which is why cosine is the default for embeddings.

```cypher
USE library
WITH [1.0, 0.0] AS query_vector
UNWIND [{name: 'aligned but short', v: [0.9, 0.1]},
        {name: 'weak but long', v: [4.0, 6.0]}] AS candidate
RETURN candidate.name AS candidate,
       vector.dot(query_vector, candidate.v) AS dot,
       round(vector.cosine(query_vector, candidate.v) * 10000) / 10000.0
         AS cosine
ORDER BY dot DESC
```

Result:

```
candidate         | dot | cosine
------------------+-----+-------
weak but long     | 4   | 0.5547
aligned but short | 0.9 | 0.9939

2 rows
```

## Where it earns its place

- Vectors already normalised, where it is the cheaper cosine.
- Scoring where magnitude legitimately carries weight.
- Building a custom similarity from parts.

## Limitations and trade-offs

- Unbounded in both directions; there is no scale to compare against.
- A long vector outranks a well-aligned one, which is rarely what an embedding comparison wants.

## See also

- [`vector.cosine`](./vector-cosine.md)
