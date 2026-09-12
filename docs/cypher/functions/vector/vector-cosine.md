# `vector.cosine`

> Cosine similarity: how aligned two vectors are, ignoring their lengths.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `vector.cosine(a, b)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`library`](../../datasets.md#library) — Eight arXiv papers with embedded abstracts |

## What it does

`vector.cosine` returns the cosine of the angle between two vectors: `1` when they point the same way, `0` when they are perpendicular, `-1` when opposed. Magnitude is divided out, so it measures direction alone.

## How it behaves

The arguments are lists of numbers. Both must be the same length; a length mismatch is an error rather than a silently truncated comparison. A null argument makes the result null, as with every other scalar function.

These functions do the arithmetic in the query and consult no index, so they are exactly as fast as the number of rows they run over. That makes them right for comparing a handful of vectors and wrong for scanning a corpus, which is what an embedding index and `SEARCH` exist for.

## When to use it

Use it whenever the vectors are embeddings. Direction is what an encoder carries meaning in, and length mostly reflects incidental things like document size, so dividing it out is what makes two documents comparable.

## How it differs from its neighbours

`vector.dot` keeps magnitude, so a long vector scores higher regardless of direction. `vector.distance` measures separation in space, where cosine measures angle: two vectors far apart in length can be perfectly aligned.

## Simple example

The three defining cases: identical direction, perpendicular, and opposed.

```cypher
USE library
RETURN vector.cosine([1.0, 0.0], [1.0, 0.0]) AS identical,
       vector.cosine([1.0, 0.0], [0.0, 1.0]) AS perpendicular,
       vector.cosine([1.0, 0.0], [-1.0, 0.0]) AS opposed,
       round(vector.cosine([1.0, 0.0], [1.0, 1.0]) * 10000) / 10000.0
         AS forty_five_degrees
```

Result:

```
identical | perpendicular | opposed | forty_five_degrees
----------+---------------+---------+-------------------
1         | 0             | -1      | 0.7071            

1 row
```

## Advanced example

Cosine ignores scale where the dot product does not. The same pair of directions is compared at three magnitudes: the cosine is identical every time and the dot product grows with the vectors.

```cypher
USE library
UNWIND [1.0, 10.0, 100.0] AS scale
WITH scale, [3.0 * scale, 4.0 * scale] AS scaled, [4.0, 3.0] AS fixed
RETURN scale,
       round(vector.cosine(scaled, fixed) * 10000) / 10000.0 AS cosine,
       vector.dot(scaled, fixed) AS dot,
       round(vector.distance(scaled, fixed) * 100) / 100.0 AS distance
ORDER BY scale
```

Result:

```
scale | cosine | dot  | distance
------+--------+------+---------
1     | 0.96   | 24   | 1.41    
10    | 0.96   | 240  | 45.22   
100   | 0.96   | 2400 | 495.2   

3 rows
```

## Where it earns its place

- Comparing embeddings, where direction carries the meaning.
- Reproducing the score an index returned, to check it by hand.
- Ranking a small candidate set without declaring an index.

## Limitations and trade-offs

- Undefined for a zero vector, which has no direction.
- Ignores magnitude entirely, which is wrong when magnitude is the signal.
- No index is consulted; cost is linear in rows.

## See also

- [`vector.dot`](./vector-dot.md)
- [`SEARCH`](../../clauses/search/search.md) for the indexed form
