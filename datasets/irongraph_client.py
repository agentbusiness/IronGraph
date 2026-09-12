"""Minimal Query API client used by the dataset loader and the documentation example checker.

The database exposes exactly one browser-facing data endpoint, `POST /api/query`, and this module
speaks only that endpoint. Responses arrive as newline-delimited frames: one `schema` frame, zero or
more `batch` frames, and one terminal `summary` or `error` frame.
"""

from __future__ import annotations

import datetime
import json
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

DEFAULT_ENDPOINT = "http://127.0.0.1:18484/api/query"


@dataclass
class QueryResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def scalar(self) -> Any:
        return self.rows[0][0] if self.rows and self.rows[0] else None

    def dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]


def _unwrap(cell: Any) -> Any:
    """Unwraps one result cell into a plain Python value.

    Scalars arrive as `{"type": ..., "value": ...}`. Nodes, relationships, paths, lists and maps
    arrive as structured objects and are returned unchanged so callers can inspect them.
    """
    if isinstance(cell, dict) and set(cell) == {"type", "value"}:
        value = cell["value"]
        if cell["type"] == "integer" and isinstance(value, str):
            return int(value)
        return value
    # Some columns carry a bare type envelope with no value, which is how a null arrives.
    if isinstance(cell, dict) and cell.get("type") == "null":
        return None
    return cell


def run(
    query: str,
    parameters: dict[str, Any] | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 3600.0,
) -> QueryResult:
    """Runs one Cypher statement and returns its columns, rows and statistics."""
    body: dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "project_id": None,
        "query": query,
    }
    if parameters:
        body["parameters"] = parameters
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode()
    except urllib.error.HTTPError as error:
        return QueryResult(ok=False, error=f"HTTP {error.code}: {error.read().decode()[:2000]}")
    except OSError as error:
        return QueryResult(ok=False, error=f"{type(error).__name__}: {error}")

    columns: list[str] = []
    rows: list[list[Any]] = []
    statistics: dict[str, Any] = {}
    for line in payload.splitlines():
        if not line.strip():
            continue
        frame = json.loads(line)
        kind = frame.get("type")
        if kind == "schema":
            columns = [column["name"] for column in frame["columns"]]
        elif kind == "batch":
            values = [[_unwrap(cell) for cell in column["values"]] for column in frame["columns"]]
            for index in range(frame["row_count"]):
                rows.append([column[index] for column in values])
        elif kind == "summary":
            statistics = frame.get("statistics", {})
        elif kind == "error":
            return QueryResult(ok=False, error=frame.get("message") or json.dumps(frame))
    return QueryResult(ok=True, columns=columns, rows=rows, statistics=statistics)


def must(query: str, parameters: dict[str, Any] | None = None, **kwargs: Any) -> QueryResult:
    """Runs one statement and raises when the database rejects it."""
    result = run(query, parameters, **kwargs)
    if not result.ok:
        raise RuntimeError(f"{result.error}\n  statement: {query[:400]}")
    return result


def table(result: QueryResult, limit: int = 30) -> str:
    """Renders a result the way the documentation prints an expected result."""
    if not result.ok:
        return f"ERROR: {result.error}"
    if not result.columns:
        return f"(no rows returned) updates={result.statistics.get('updates', 0)}"

    def cell(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            text = f"{value:.6f}".rstrip("0").rstrip(".")
            return text or "0"
        if isinstance(value, dict) and "seconds" in value and "nanos" in value:
            # Render a temporal value the way a reader expects to see it rather than as the
            # structured envelope the wire carries.
            moment = datetime.datetime.fromtimestamp(value["seconds"], datetime.timezone.utc)
            text = moment.strftime("%Y-%m-%dT%H:%M:%S")
            if value["nanos"]:
                text += f".{value['nanos']:09d}".rstrip("0")
            return text + "Z"
        if isinstance(value, (dict, list)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)

    body = [[cell(value) for value in row] for row in result.rows[:limit]]
    widths = [
        max([len(name)] + [len(row[index]) for row in body]) for index, name in enumerate(result.columns)
    ]
    lines = [
        " | ".join(name.ljust(widths[index]) for index, name in enumerate(result.columns)),
        "-+-".join("-" * width for width in widths),
    ]
    lines += [" | ".join(row[index].ljust(widths[index]) for index in range(len(result.columns))) for row in body]
    if len(result.rows) > limit:
        lines.append(f"... {len(result.rows) - limit} more rows")
    return "\n".join(lines)
