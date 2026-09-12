# `vector.distance`

> Euclidean distance: how far apart two vectors are in space.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `vector.distance(a, b)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`library`](../../datasets.md#library) — Eight arXiv papers with embedded abstracts |

## What it does

`vector.distance` returns the straight-line distance between two points: the square root of the summed squared differences. It is `0` for identical vectors and grows without bound as they separate.

## How it behaves

The arguments are lists of numbers. Both must be the same length; a length mismatch is an error rather than a silently truncated comparison. A null argument makes the result null, as with every other scalar function.

These functions do the arithmetic in the query and consult no index, so they are exactly as fast as the number of rows they run over. That makes them right for comparing a handful of vectors and wrong for scanning a corpus, which is what an embedding index and `SEARCH` exist for.

## When to use it

Use it when the vectors are positions rather than directions — coordinates, measurements, anything where being far apart is the thing you want to measure. It is also the right similarity for an index declared `EUCLIDEAN`.

## How it differs from its neighbours

Distance falls as similarity rises, the opposite of the other three, so an ordering by distance is ascending where an ordering by cosine is descending. On normalised vectors distance and cosine agree in ranking; on raw vectors they can disagree completely.

## Simple example

The classic right triangle, and a distance of zero for identical vectors.

```cypher
USE library
RETURN vector.distance([0.0, 0.0], [3.0, 4.0]) AS three_four_five,
       vector.distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) AS identical,
       round(vector.distance([1.0, 0.0], [0.0, 1.0]) * 10000) / 10000.0
         AS perpendicular_unit_vectors
```

Result:

```
three_four_five | identical | perpendicular_unit_vectors
----------------+-----------+---------------------------
5               | 0         | 1.4142                    

1 row
```

## Advanced example

Distance and cosine ranking the same candidates differently. Normalising first makes them agree, which is the practical reason embeddings are compared by angle rather than by position.

```cypher
USE library
WITH [1.0, 1.0] AS query_vector
UNWIND [{name: 'same direction, far', v: [5.0, 5.0]},
        {name: 'different direction, near', v: [1.4, 0.2]}] AS candidate
RETURN candidate.name AS candidate,
       round(vector.distance(query_vector, candidate.v) * 100) / 100.0
         AS distance,
       round(vector.cosine(query_vector, candidate.v) * 10000) / 10000.0
         AS cosine,
       round(vector.distance(vector.normalize(query_vector),
                             vector.normalize(candidate.v)) * 10000)
         / 10000.0 AS normalised_distance
ORDER BY distance
```

Result:

```
candidate                 | distance | cosine | normalised_distance
--------------------------+----------+--------+--------------------
different direction, near | 0.89     | 0.8    | 0.6325             
same direction, far       | 5.66     | 1      | 0                  

2 rows
```

## Where it earns its place

- Positions and measurements rather than directions.
- Indexes declared with `EUCLIDEAN` similarity.
- Thresholding on a bounded neighbourhood in space.

## Limitations and trade-offs

- Lower is more similar, inverting the ordering the other functions use.
- Unbounded above, so a threshold has to be chosen for the data.
- Sensitive to magnitude, which for embeddings is usually noise.

## See also

- [`vector.cosine`](./vector-cosine.md)
- [`vector.normalize`](./vector-normalize.md)
