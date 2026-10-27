# String functions

Use these scalar functions to transform values within a query. Every page states accepted arguments,
null behavior, limits and two self-contained examples.

| Function | Signature | Purpose |
| --- | --- | --- |
| [`lower`](./lower.md) | `lower(text)` | Convert a string to lowercase. |
| [`toLower`](./tolower.md) | `toLower(text)` | Convert a string to lowercase. |
| [`upper`](./upper.md) | `upper(text)` | Convert a string to uppercase. |
| [`trim`](./trim.md) | `trim(text)` | Remove whitespace from both ends of a string. |
| [`ltrim`](./ltrim.md) | `ltrim(text)` | Remove whitespace from the beginning of a string. |
| [`rtrim`](./rtrim.md) | `rtrim(text)` | Remove whitespace from the end of a string. |
| [`left`](./left.md) | `left(text, count)` | Return at most count characters from the beginning of a string. |
| [`right`](./right.md) | `right(text, count)` | Return at most count characters from the end of a string. |
| [`substring`](./substring.md) | `substring(text, start [, length])` | Return a substring beginning at a zero-based character offset. |
| [`replace`](./replace.md) | `replace(text, search, replacement)` | Replace occurrences of a substring. |
| [`split`](./split.md) | `split(text, separator)` | Split a string into a list using a literal separator. |

