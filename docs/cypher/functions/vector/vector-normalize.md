# `vector.normalize`

> Scales a vector to length one, keeping its direction.

| | |
| --- | --- |
| Kind | Scalar function |
| Signature | `vector.normalize(a)` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`library`](../../datasets.md#library) — Eight arXiv papers with embedded abstracts |

## What it does

`vector.normalize` divides a vector by its own magnitude and returns a vector of length `1` pointing the same way. It is the one function here that returns a vector rather than a number.

## How it behaves

The arguments are lists of numbers. Both must be the same length; a length mismatch is an error rather than a silently truncated comparison. A null argument makes the result null, as with every other scalar function.

These functions do the arithmetic in the query and consult no index, so they are exactly as fast as the number of rows they run over. That makes them right for comparing a handful of vectors and wrong for scanning a corpus, which is what an embedding index and `SEARCH` exist for.

## When to use it

Normalise when you want to compare directions with tools that are sensitive to magnitude — before a dot product, or before a Euclidean distance — or to store vectors in a form where the cheaper comparison is also the correct one.

## How it differs from its neighbours

It changes the vector rather than comparing two. Normalising both sides makes `vector.dot` equal `vector.cosine`, and makes `vector.distance` rank the same way as cosine, which is why it usually appears as a preparation step rather than as an answer.

## Simple example

A 3-4-5 vector reduced to unit length, and the proof that it is one.

```cypher
USE library
WITH vector.normalize([3.0, 4.0]) AS unit
RETURN unit,
       round(vector.distance([0.0, 0.0], unit) * 10000) / 10000.0
         AS its_length,
       round(vector.dot(unit, unit) * 10000) / 10000.0 AS dot_with_itself
```

Result:

```
unit      | its_length | dot_with_itself
----------+------------+----------------
[0.6,0.8] | 1          | 1              

1 row
```

## Advanced example

Normalisation makes three different measures agree. Once both sides are unit length, the dot product equals the cosine and the distance is a monotone function of it, so any of the three gives the same ranking.

```cypher
USE library
UNWIND [[5.0, 5.0], [1.4, 0.2], [0.1, 3.0]] AS raw
WITH [1.0, 1.0] AS query_vector, raw,
     vector.normalize([1.0, 1.0]) AS unit_query,
     vector.normalize(raw) AS unit_candidate
RETURN raw,
       round(vector.cosine(query_vector, raw) * 10000) / 10000.0 AS cosine,
       round(vector.dot(unit_query, unit_candidate) * 10000) / 10000.0
         AS normalised_dot,
       round(vector.distance(unit_query, unit_candidate) * 10000) / 10000.0
         AS normalised_distance
ORDER BY cosine DESC
```

Result:

```
raw                                                         | cosine | normalised_dot | normalised_distance
------------------------------------------------------------+--------+----------------+--------------------
[{"type":"float","value":5.0},{"type":"float","value":5.0}] | 1      | 1              | 0                  
[{"type":"float","value":1.4},{"type":"float","value":0.2}] | 0.8    | 0.8            | 0.6325             
[{"type":"float","value":0.1},{"type":"float","value":3.0}] | 0.7303 | 0.7303         | 0.7345             

3 rows
```

## Where it earns its place

- Preparing vectors so a dot product means cosine similarity.
- Making Euclidean distance rank the same way as cosine.
- Storing vectors in a comparable form.

## Limitations and trade-offs

- Undefined for a zero vector.
- Discards magnitude, which is a loss when magnitude carries meaning.
- Returns a new vector; it does not modify a stored property.

## See also

- [`vector.cosine`](./vector-cosine.md)
