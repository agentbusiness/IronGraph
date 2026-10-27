# Expressions and value semantics

Build predicates and computed values inside Cypher. This page covers literals, parameters, operators,
conditional expressions, comprehensions and temporal fields used by the clause and function pages.

## Values, names and parameters

Values include `null`, booleans, integers, floating-point numbers, strings, lists, maps, graph entities,
paths, dates, clocks, datetimes and durations. Lists and maps may contain nested values. Write a list
as `[1, 2, null]` and a map as `{name: 'Ada', active: true}`. Use `entity.property`, `map.key` or
`value[index]` to select a value. List indexing is zero-based and supports negative offsets;
`list[start..end]` selects a slice with an exclusive end and optional bounds.

Use `$name` for a request parameter. Pass values through the supported Query API or SDK parameter
argument; parameters are not string substitution. A missing parameter is an error. Backticks quote
identifiers such as `` `display name` ``; double a backtick inside a quoted identifier to escape it.
Keywords and function names are case-insensitive; property, label and relationship-type names retain
their spelling.

SET and REMOVE node-label operations accept literal names, `$(expression)` for one string and
`$all(expression)` for a list of strings. They also accept the `IS` spelling with `&` between labels,
such as `SET node IS Active & Reviewed`. Invalid value types report errors. Node and relationship
patterns use literal label and type names; dynamic pattern labels and relationship types are not
supported.

## Operators and nulls

| Operators | Meaning |
| --- | --- |
| Unary `+`, `-` | Numeric sign |
| `+`, `-`, `*`, `/`, `%`, `^` | Arithmetic; `+` also concatenates supported strings and lists |
| `\|\|` | Concatenate two strings or two lists |
| `=`, `<>`, `!=`, `<`, `<=`, `>`, `>=` | Equality and ordered comparisons |
| `AND`, `OR`, `XOR`, `NOT` | Three-valued boolean logic |
| `IN` | Membership in a list |
| `STARTS WITH`, `ENDS WITH`, `CONTAINS` | Case-sensitive string predicates |
| `=~` | Regular-expression match |
| `IS NULL`, `IS NOT NULL` | Test null explicitly |
| `entity:Label`, `relationship:TYPE` | Test entity label or relationship type |

Use parentheses to make evaluation order clear. Comparisons involving null generally produce null;
`null = null` is not true. A filter keeps only true, so both false and null discard a row. For list
membership, a null list element can make an otherwise unmatched membership test unknown. Boolean
operators preserve three-valued logic: `false AND null` is false and `true OR null` is true.

Integer arithmetic checks overflow. Invalid types, division by zero and invalid regular expressions
report errors. Regular expressions match according to the supported expression syntax; they do not
perform text-index retrieval. Use a text or embedding index through the documented search surface
when that is your intent.

## Simple example

Prerequisite: create `worked_example` with `CREATE PROJECT IF NOT EXISTS worked_example`.

```cypher
USE worked_example
RETURN 2 + 3 * 4 AS arithmetic,
       null IS NULL AS missing,
       false AND null AS known_false,
       'graph database' CONTAINS 'graph' AS contains
```

Result:

```
arithmetic | missing | known_false | contains
-----------+---------+-------------+---------
14         | true    | false       | true

1 row
```

Expected result: `arithmetic` is 14, `missing` is true, `known_false` is false and `contains` is true.

## CASE, lists and maps

`CASE expression WHEN value THEN result ... ELSE fallback END` selects by equality. The searched
form `CASE WHEN predicate THEN result ... ELSE fallback END` selects the first true predicate.
Without a matching branch or ELSE, the result is null.

`[x IN list WHERE predicate | projection]` filters and maps a list. WHERE and projection may be
omitted when not needed. `reduce(acc = initial, x IN list | expression)` folds its list into one value.
List predicates use `all`, `any`, `none` or `single`, as in `all(x IN list WHERE x > 0)`. On an empty
list, all and none are true; any and single are false. Null list items or predicates can produce an
unknown result when they prevent a decisive answer.

`map{.name, .* , computed: expression}` projects properties and computed entries into a map.
`map{variable}` also includes the named in-scope variable under its own key. Pattern
comprehensions, such as `[(a)-[:KNOWS]->(b) | b.name]`, return a list from a correlated graph pattern.
An existential subquery, `EXISTS { MATCH ... WHERE ... }`, returns whether the correlated read has
at least one result. It is a read expression; writes and UNION are not permitted inside it. Pattern
predicates may also appear in predicate contexts such as WHERE.

## Advanced example

```cypher
USE worked_example
WITH [1, 2, 3, 4] AS values
RETURN [x IN values WHERE x > 2 | x * 10] AS selected,
       reduce(total = 0, x IN values | total + x) AS total,
       all(x IN values WHERE x > 0) AS positive,
       CASE WHEN size(values) > 2 THEN 'many' ELSE 'few' END AS category
```

Result:

```
selected | total | positive | category
---------+-------+----------+---------
[30, 40] | 10    | true     | many

1 row
```

Expected result: `selected` is `[30, 40]`, `total` is 10, `positive` is true and `category` is `many`.

## Dynamic label examples

These examples create their own nodes in the disposable example project:

```cypher
USE worked_example
WITH 'ReferenceDynamic' AS label
CREATE (node {name: 'Ada'})
SET node:$(label)
RETURN labels(node) AS labels
```

Result:

```
labels
------------------
[ReferenceDynamic]

1 row
```

Expected result: one node labelled `ReferenceDynamic`.

```cypher
USE worked_example
CREATE (node)
SET node:$all(['ReferenceDynamic', 'Active'])
RETURN size(labels(node)) AS label_count
```

Result:

```
label_count
-----------
2

1 row
```

Expected result: `label_count` is 2.

## Temporal and duration fields

Accessors are case-insensitive and apply only to compatible temporal types. An unsupported field is
an error, rather than an implicitly converted value. Use the
[temporal constructors](./functions/temporal/README.md) before reading fields from text.

| Value | Fields |
| --- | --- |
| Date-bearing value | `year`, `quarter`, `month`, `week`, `weekyear`, `day`, `ordinalday`, `weekday` / `dayofweek`, `dayofquarter` |
| Clock-bearing value | `hour`, `minute`, `second`, `millisecond`, `microsecond`, `nanosecond` |
| Zoned value | `timezone`, `offset`, `offsetminutes`, `offsetseconds` |
| Zoned datetime | `epochseconds`, `epochmillis` |
| Duration calendar parts | `years`, `quarters`, `months`, `weeks`, `days`, `quartersofyear`, `monthsofquarter`, `monthsofyear`, `daysofweek` |
| Duration time parts | `hours`, `minutes`, `seconds`, `milliseconds`, `microseconds`, `nanoseconds`, `minutesofhour`, `secondsofminute`, `millisecondsofsecond`, `microsecondsofsecond`, `nanosecondsofsecond` |

Duration calendar components and elapsed-time components remain separate. For example, its `hours`
field derives from the stored seconds component; it does not assume every calendar day has 24 hours.
Adding a duration to a compatible temporal value uses calendar-aware arithmetic. Duration scaling
and division require compatible numeric operands.

Next, use [functions](./functions/README.md) for individual signatures and examples.
