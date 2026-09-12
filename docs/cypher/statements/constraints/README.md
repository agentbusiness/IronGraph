# Constraint statements

A unique constraint is schema authority, not an access path. It rejects a write that would duplicate a value, which is what makes a property safe to treat as identity.

| Page | Summary | Standard |
| --- | --- | --- |
| [`CREATE CONSTRAINT`](./create-constraint.md) | Requires a property to be unique across a label, and enforces it on write. | extension |
| [`DROP CONSTRAINT`](./drop-constraint.md) | Removes a uniqueness requirement and the index that enforced it. | extension |
