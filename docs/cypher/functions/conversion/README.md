# Conversion functions

Use these scalar functions to transform values within a query. Every page states accepted arguments,
null behavior, limits and two self-contained examples.

| Function | Signature | Purpose |
| --- | --- | --- |
| [`toString`](./tostring.md) | `toString(value)` | Convert a supported scalar value to a string. |
| [`toInteger`](./tointeger.md) | `toInteger(value)` | Convert a boolean, number or numeric string to an integer. |
| [`toFloat`](./tofloat.md) | `toFloat(value)` | Convert a number or numeric string to a floating-point value. |
| [`toBoolean`](./toboolean.md) | `toBoolean(value)` | Convert a boolean or a boolean string to a boolean. |
| [`toList`](./tolist.md) | `toList(value)` | Keep a list, extract nodes from a path, or wrap another non-null value in a one-element list. |
| [`coalesce`](./coalesce.md) | `coalesce(value [, value ...])` | Return the first non-null value, or null when no argument supplies one. |

