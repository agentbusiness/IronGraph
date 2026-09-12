# Vector functions

Four functions over list-valued numbers. They compute similarity and distance directly in a query, without an index — useful for comparing a few vectors, and the wrong tool for searching a corpus.

| Page | Summary | Standard |
| --- | --- | --- |
| [`vector.cosine`](./vector-cosine.md) | Cosine similarity: how aligned two vectors are, ignoring their lengths. | extension |
| [`vector.dot`](./vector-dot.md) | Dot product: alignment scaled by both vectors' magnitudes. | extension |
| [`vector.distance`](./vector-distance.md) | Euclidean distance: how far apart two vectors are in space. | extension |
| [`vector.normalize`](./vector-normalize.md) | Scales a vector to length one, keeping its direction. | extension |
