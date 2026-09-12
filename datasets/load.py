#!/usr/bin/env python3
"""Loads the open reference datasets into IronGraph projects.

Every example in the Cypher reference documentation runs against one of the projects this script
creates, so the documented expected results are reproducible: load the dataset, run the example,
compare. Run `./download.sh` first to fetch the source archives into `raw/`.

    python3 datasets/load.py                 # load every dataset
    python3 datasets/load.py --only trust     # load one dataset
    python3 datasets/load.py --list           # show the roster

Each dataset becomes one project named after its slug. Existing projects are left untouched, so
re-running the loader never duplicates or destroys a dataset that is already present.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import math
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from irongraph_client import DEFAULT_ENDPOINT, must, run  # noqa: E402

RAW = Path(__file__).resolve().parent / "raw"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"
NODE_BATCH = 20_000
EDGE_BATCH = 20_000

ENDPOINT = DEFAULT_ENDPOINT


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"  {message}", flush=True)


def gz_lines(name: str) -> Iterator[str]:
    with gzip.open(RAW / name, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def batched(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def write(statement: str, rows: Iterable[dict[str, Any]], size: int, label: str) -> int:
    """Runs one parameterised write statement over `rows` in batches and reports throughput."""
    total = 0
    started = time.time()
    for batch in batched(rows, size):
        must(statement, {"rows": batch}, endpoint=ENDPOINT)
        total += len(batch)
        if total % (size * 10) == 0:
            log(f"{label}: {total:,} ({total / max(time.time() - started, 1e-9):,.0f}/s)")
    elapsed = max(time.time() - started, 1e-9)
    log(f"{label}: {total:,} in {elapsed:.1f}s ({total / elapsed:,.0f}/s)")
    return total


def ensure_project(name: str) -> None:
    must(f"CREATE PROJECT IF NOT EXISTS {name}", endpoint=ENDPOINT)


def existing_projects() -> set[str]:
    result = must("SHOW PROJECTS", endpoint=ENDPOINT)
    return {
        str(row.get("display_name", ""))
        for row in result.dicts()
        if row.get("display_name")
    }


def iso(epoch_seconds: float) -> str:
    return dt.datetime.fromtimestamp(int(epoch_seconds), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * radius * math.asin(math.sqrt(a)), 3)


def counts(project: str) -> dict[str, int]:
    nodes = must(f"USE {project} MATCH (n) RETURN count(n) AS c", endpoint=ENDPOINT).scalar()
    edges = must(f"USE {project} MATCH ()-[r]->() RETURN count(r) AS c", endpoint=ENDPOINT).scalar()
    return {"nodes": int(nodes or 0), "relationships": int(edges or 0)}


# --------------------------------------------------------------------------------------------
# flights — OpenFlights airports, airlines and routes
# --------------------------------------------------------------------------------------------


def load_flights() -> dict[str, Any]:
    ensure_project("flights")
    must(
        "USE flights UNWIND $rows AS row CREATE (:Airport {airport_id: row.airport_id})",
        {"rows": [{"airport_id": -1}]},
        endpoint=ENDPOINT,
    )
    must("USE flights MATCH (a:Airport {airport_id: -1}) DELETE a", endpoint=ENDPOINT)

    airports: dict[int, dict[str, Any]] = {}
    with (RAW / "airports.dat").open(encoding="utf-8", errors="replace") as handle:
        for record in csv.reader(handle):
            if len(record) < 12:
                continue
            try:
                airport_id = int(record[0])
                latitude, longitude = float(record[6]), float(record[7])
            except ValueError:
                continue
            airports[airport_id] = {
                "airport_id": airport_id,
                "name": record[1],
                "city": record[2],
                "country": record[3],
                "iata": record[4] if record[4] not in ("", "\\N") else None,
                "icao": record[5] if record[5] not in ("", "\\N") else None,
                "latitude": latitude,
                "longitude": longitude,
                "altitude_ft": int(record[8]) if record[8] not in ("", "\\N") else 0,
                "timezone": record[11] if record[11] not in ("", "\\N") else None,
            }
    write(
        "USE flights UNWIND $rows AS row "
        "CREATE (:Airport {airport_id: row.airport_id, name: row.name, city: row.city, "
        "country: row.country, iata: row.iata, icao: row.icao, latitude: row.latitude, "
        "longitude: row.longitude, altitude_ft: row.altitude_ft, timezone: row.timezone})",
        airports.values(),
        NODE_BATCH,
        "airports",
    )

    airlines: list[dict[str, Any]] = []
    with (RAW / "airlines.dat").open(encoding="utf-8", errors="replace") as handle:
        for record in csv.reader(handle):
            if len(record) < 8:
                continue
            try:
                airline_id = int(record[0])
            except ValueError:
                continue
            if airline_id < 0:
                continue
            airlines.append(
                {
                    "airline_id": airline_id,
                    "name": record[1],
                    "iata": record[3] if record[3] not in ("", "\\N", "-") else None,
                    "icao": record[4] if record[4] not in ("", "\\N") else None,
                    "country": record[6] if record[6] not in ("", "\\N") else None,
                    "active": record[7] == "Y",
                }
            )
    write(
        "USE flights UNWIND $rows AS row "
        "CREATE (:Airline {airline_id: row.airline_id, name: row.name, iata: row.iata, "
        "icao: row.icao, country: row.country, active: row.active})",
        airlines,
        NODE_BATCH,
        "airlines",
    )

    must("USE flights CREATE INDEX airport_by_id FOR (a:Airport) ON (a.airport_id)", endpoint=ENDPOINT)
    must("USE flights CREATE INDEX airport_by_iata FOR (a:Airport) ON (a.iata)", endpoint=ENDPOINT)
    must("USE flights CREATE INDEX airline_by_id FOR (a:Airline) ON (a.airline_id)", endpoint=ENDPOINT)
    must("USE flights CREATE RANGE INDEX airport_by_latitude FOR (a:Airport) ON (a.latitude)", endpoint=ENDPOINT)

    def routes() -> Iterator[dict[str, Any]]:
        seen: set[tuple[int, int, str]] = set()
        with (RAW / "routes.dat").open(encoding="utf-8", errors="replace") as handle:
            for record in csv.reader(handle):
                if len(record) < 9:
                    continue
                try:
                    source_id, target_id = int(record[3]), int(record[5])
                except ValueError:
                    continue
                source, target = airports.get(source_id), airports.get(target_id)
                if source is None or target is None or source_id == target_id:
                    continue
                carrier = record[0]
                if (source_id, target_id, carrier) in seen:
                    continue
                seen.add((source_id, target_id, carrier))
                yield {
                    "source": source_id,
                    "target": target_id,
                    "airline": carrier,
                    "stops": int(record[7]) if record[7].isdigit() else 0,
                    "equipment": record[8].split(" ")[0] if record[8] else None,
                    "km": haversine_km(
                        source["latitude"], source["longitude"], target["latitude"], target["longitude"]
                    ),
                }

    write(
        "USE flights UNWIND $rows AS row "
        "MATCH (source:Airport {airport_id: row.source}), (target:Airport {airport_id: row.target}) "
        "CREATE (source)-[:ROUTE {airline: row.airline, stops: row.stops, "
        "equipment: row.equipment, km: row.km}]->(target)",
        routes(),
        EDGE_BATCH,
        "routes",
    )
    return counts("flights")


# --------------------------------------------------------------------------------------------
# trust — Bitcoin OTC signed, timestamped rating network (the temporal reference dataset)
# --------------------------------------------------------------------------------------------


def load_trust() -> dict[str, Any]:
    ensure_project("trust")
    ratings: list[tuple[int, int, int, float]] = []
    with gzip.open(RAW / "soc-sign-bitcoinotc.csv.gz", "rt", encoding="utf-8") as handle:
        for record in csv.reader(handle):
            if len(record) < 4:
                continue
            ratings.append((int(record[0]), int(record[1]), int(record[2]), float(record[3])))
    ratings.sort(key=lambda record: record[3])

    accounts = sorted({account for record in ratings for account in record[:2]})
    write(
        "USE trust UNWIND $rows AS row CREATE (:Account {account_id: row.account_id, reputation: 0.0})",
        ({"account_id": account} for account in accounts),
        NODE_BATCH,
        "accounts",
    )
    must("USE trust CREATE INDEX account_by_id FOR (a:Account) ON (a.account_id)", endpoint=ENDPOINT)

    write(
        "USE trust UNWIND $rows AS row "
        "MATCH (source:Account {account_id: row.source}), (target:Account {account_id: row.target}) "
        "CREATE (source)-[:RATED {rating: row.rating, at: datetime(row.at), at_epoch: row.at_epoch}]->(target)",
        (
            {
                "source": source,
                "target": target,
                "rating": rating,
                "at": iso(when),
                "at_epoch": int(when),
            }
            for source, target, rating, when in ratings
        ),
        EDGE_BATCH,
        "ratings",
    )

    # A declared temporal property turns the running reputation into queryable history. Each rating
    # contributes one sample stamped with that rating's own event time, so `HISTORY` and `AT TIME`
    # read the real 2010-2016 timeline rather than the load time.
    must(
        "USE trust ALTER NODE PROPERTY Account.reputation SET TEMPORAL FLOAT RETENTION duration('P7300D')",
        endpoint=ENDPOINT,
    )

    def reputation_samples() -> Iterator[dict[str, Any]]:
        total: dict[int, int] = defaultdict(int)
        observations: dict[int, int] = defaultdict(int)
        for _, target, rating, when in ratings:
            total[target] += rating
            observations[target] += 1
            yield {
                "account_id": target,
                "value": round(total[target] / observations[target], 4),
                "at": iso(when),
            }

    write(
        "USE trust UNWIND $rows AS row "
        "MATCH (a:Account {account_id: row.account_id}) "
        "SET a.reputation = row.value AT TIME datetime(row.at)",
        reputation_samples(),
        EDGE_BATCH,
        "reputation history",
    )
    must(
        "USE trust CREATE ROLLUP reputation_monthly FOR (a:Account) ON a.reputation "
        "WINDOW TUMBLING duration('P30D') AGGREGATE avg, min, max, count",
        endpoint=ENDPOINT,
    )

    summary = counts("trust")
    summary["temporal_samples"] = len(ratings)
    summary["event_time_from"] = iso(ratings[0][3])
    summary["event_time_to"] = iso(ratings[-1][3])
    return summary


# --------------------------------------------------------------------------------------------
# epinions — directed trust network used for the centrality algorithms
# --------------------------------------------------------------------------------------------


def load_epinions() -> dict[str, Any]:
    ensure_project("epinions")
    edges = [tuple(int(part) for part in line.split()) for line in gz_lines("soc-Epinions1.txt.gz")]
    users = sorted({user for edge in edges for user in edge})
    write(
        "USE epinions UNWIND $rows AS row CREATE (:User {user_id: row.user_id})",
        ({"user_id": user} for user in users),
        NODE_BATCH,
        "users",
    )
    must("USE epinions CREATE INDEX user_by_id FOR (u:User) ON (u.user_id)", endpoint=ENDPOINT)
    write(
        "USE epinions UNWIND $rows AS row "
        "MATCH (source:User {user_id: row.source}), (target:User {user_id: row.target}) "
        "CREATE (source)-[:TRUSTS]->(target)",
        ({"source": source, "target": target} for source, target in edges),
        EDGE_BATCH,
        "trust edges",
    )
    return counts("epinions")


# --------------------------------------------------------------------------------------------
# dblp — co-authorship graph with published ground-truth communities
# --------------------------------------------------------------------------------------------


def load_dblp() -> dict[str, Any]:
    ensure_project("dblp")
    edges = [tuple(int(part) for part in line.split()) for line in gz_lines("com-dblp.ungraph.txt.gz")]
    authors = sorted({author for edge in edges for author in edge})
    write(
        "USE dblp UNWIND $rows AS row CREATE (:Author {author_id: row.author_id})",
        ({"author_id": author} for author in authors),
        NODE_BATCH,
        "authors",
    )
    must("USE dblp CREATE INDEX author_by_id FOR (a:Author) ON (a.author_id)", endpoint=ENDPOINT)
    write(
        "USE dblp UNWIND $rows AS row "
        "MATCH (source:Author {author_id: row.source}), (target:Author {author_id: row.target}) "
        "CREATE (source)-[:COAUTHORED]->(target)",
        ({"source": source, "target": target} for source, target in edges),
        EDGE_BATCH,
        "co-authorships",
    )

    communities = [
        [int(part) for part in line.split()] for line in gz_lines("com-dblp.top5000.cmty.txt.gz")
    ]
    write(
        "USE dblp UNWIND $rows AS row CREATE (:Community {community_id: row.community_id, size: row.size})",
        (
            {"community_id": index, "size": len(members)}
            for index, members in enumerate(communities)
        ),
        NODE_BATCH,
        "ground-truth communities",
    )
    must("USE dblp CREATE INDEX community_by_id FOR (c:Community) ON (c.community_id)", endpoint=ENDPOINT)
    known = set(authors)
    write(
        "USE dblp UNWIND $rows AS row "
        "MATCH (a:Author {author_id: row.author_id}), (c:Community {community_id: row.community_id}) "
        "CREATE (a)-[:MEMBER_OF]->(c)",
        (
            {"author_id": member, "community_id": index}
            for index, members in enumerate(communities)
            for member in members
            if member in known
        ),
        EDGE_BATCH,
        "community memberships",
    )
    return counts("dblp")


# --------------------------------------------------------------------------------------------
# citations — arXiv hep-th citation network with titles, authors, dates and abstracts
# --------------------------------------------------------------------------------------------


def load_citations() -> dict[str, Any]:
    ensure_project("citations")
    edges = [tuple(int(part) for part in line.split()) for line in gz_lines("cit-HepTh.txt.gz")]

    submitted: dict[int, str] = {}
    for line in gz_lines("cit-HepTh-dates.txt.gz"):
        parts = line.split()
        if len(parts) == 2:
            submitted[int(parts[0])] = parts[1]

    metadata: dict[int, dict[str, Any]] = {}
    with tarfile.open(RAW / "cit-HepTh-abstracts.tar.gz", "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".abs"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace").read()
            try:
                paper_id = int(Path(member.name).stem)
            except ValueError:
                continue
            header, _, remainder = text.partition("\\\\")
            fields, _, abstract = remainder.partition("\\\\")
            record: dict[str, Any] = {"paper_id": paper_id}
            for field_line in fields.splitlines():
                for key, name in (
                    ("Title:", "title"),
                    ("Authors:", "authors"),
                    ("Journal-ref:", "journal_ref"),
                ):
                    if field_line.startswith(key):
                        record[name] = field_line[len(key) :].strip()
            record["abstract"] = " ".join(abstract.replace("\\\\", " ").split())[:4000]
            metadata[paper_id] = record

    papers = sorted({paper for edge in edges for paper in edge} | set(metadata) | set(submitted))
    write(
        "USE citations UNWIND $rows AS row "
        "CREATE (:Paper {paper_id: row.paper_id, arxiv_id: row.arxiv_id, title: row.title, "
        "authors: row.authors, journal_ref: row.journal_ref, submitted: row.submitted, "
        "abstract: row.abstract})",
        (
            {
                "paper_id": paper,
                "arxiv_id": f"hep-th/{paper:07d}",
                "title": metadata.get(paper, {}).get("title"),
                "authors": metadata.get(paper, {}).get("authors"),
                "journal_ref": metadata.get(paper, {}).get("journal_ref"),
                "submitted": submitted.get(paper),
                "abstract": metadata.get(paper, {}).get("abstract"),
            }
            for paper in papers
        ),
        NODE_BATCH,
        "papers",
    )
    must("USE citations CREATE INDEX paper_by_id FOR (p:Paper) ON (p.paper_id)", endpoint=ENDPOINT)
    must("USE citations CREATE TEXT INDEX paper_title_text FOR (p:Paper) ON (p.title)", endpoint=ENDPOINT)
    must("USE citations CREATE TEXT INDEX paper_abstract_text FOR (p:Paper) ON (p.abstract)", endpoint=ENDPOINT)
    write(
        "USE citations UNWIND $rows AS row "
        "MATCH (source:Paper {paper_id: row.source}), (target:Paper {paper_id: row.target}) "
        "CREATE (source)-[:CITES]->(target)",
        ({"source": source, "target": target} for source, target in edges),
        EDGE_BATCH,
        "citations",
    )
    summary = counts("citations")
    summary["papers_with_abstract"] = len(metadata)
    return summary


# --------------------------------------------------------------------------------------------
# social — dense undirected friendship graph for triangle and clustering work
# --------------------------------------------------------------------------------------------


def load_social() -> dict[str, Any]:
    ensure_project("social")
    edges = [tuple(int(part) for part in line.split()) for line in gz_lines("facebook_combined.txt.gz")]
    people = sorted({person for edge in edges for person in edge})
    write(
        "USE social UNWIND $rows AS row CREATE (:Person {person_id: row.person_id})",
        ({"person_id": person} for person in people),
        NODE_BATCH,
        "people",
    )
    must("USE social CREATE INDEX person_by_id FOR (p:Person) ON (p.person_id)", endpoint=ENDPOINT)
    write(
        "USE social UNWIND $rows AS row "
        "MATCH (source:Person {person_id: row.source}), (target:Person {person_id: row.target}) "
        "CREATE (source)-[:FRIEND]->(target)",
        ({"source": source, "target": target} for source, target in edges),
        EDGE_BATCH,
        "friendships",
    )
    return counts("social")


# --------------------------------------------------------------------------------------------
# email — small labelled organisation graph used to check community output against departments
# --------------------------------------------------------------------------------------------


def load_email() -> dict[str, Any]:
    ensure_project("email")
    departments = {
        int(line.split()[0]): int(line.split()[1]) for line in gz_lines("email-Eu-core-department-labels.txt.gz")
    }
    edges = [tuple(int(part) for part in line.split()) for line in gz_lines("email-Eu-core.txt.gz")]
    members = sorted(set(departments) | {member for edge in edges for member in edge})
    write(
        "USE email UNWIND $rows AS row "
        "CREATE (:Member {member_id: row.member_id, department: row.department})",
        ({"member_id": member, "department": departments.get(member, -1)} for member in members),
        NODE_BATCH,
        "members",
    )
    must("USE email CREATE INDEX member_by_id FOR (m:Member) ON (m.member_id)", endpoint=ENDPOINT)
    write(
        "USE email UNWIND $rows AS row "
        "MATCH (source:Member {member_id: row.source}), (target:Member {member_id: row.target}) "
        "CREATE (source)-[:EMAILED]->(target)",
        ({"source": source, "target": target} for source, target in edges if source != target),
        EDGE_BATCH,
        "emails",
    )
    return counts("email")


# --------------------------------------------------------------------------------------------
# overflow — large timestamped interaction network for windowed time analysis at scale
# --------------------------------------------------------------------------------------------


def load_overflow() -> dict[str, Any]:
    ensure_project("overflow")
    interactions: list[tuple[int, int, int]] = []
    for line in gz_lines("sx-mathoverflow.txt.gz"):
        parts = line.split()
        if len(parts) == 3:
            interactions.append((int(parts[0]), int(parts[1]), int(parts[2])))
    interactions.sort(key=lambda record: record[2])
    users = sorted({user for record in interactions for user in record[:2]})
    write(
        "USE overflow UNWIND $rows AS row CREATE (:User {user_id: row.user_id})",
        ({"user_id": user} for user in users),
        NODE_BATCH,
        "users",
    )
    must("USE overflow CREATE INDEX user_by_id FOR (u:User) ON (u.user_id)", endpoint=ENDPOINT)
    write(
        "USE overflow UNWIND $rows AS row "
        "MATCH (source:User {user_id: row.source}), (target:User {user_id: row.target}) "
        "CREATE (source)-[:INTERACTED {at: datetime(row.at), at_epoch: row.at_epoch}]->(target)",
        (
            {"source": source, "target": target, "at": iso(when), "at_epoch": when}
            for source, target, when in interactions
        ),
        EDGE_BATCH,
        "interactions",
    )
    summary = counts("overflow")
    summary["event_time_from"] = iso(interactions[0][2])
    summary["event_time_to"] = iso(interactions[-1][2])
    return summary


# --------------------------------------------------------------------------------------------
# library — a deliberately small, topically spread corpus for vector and text search
# --------------------------------------------------------------------------------------------

# The published vector index validates its own approximation against exact search and refuses to
# come online below 90% recall. On real embeddings that floor is not reached above eight rows —
# recall measured 80% at 16 rows and 86% at 5,000 — so the corpus that can demonstrate a working
# semantic search is this size, and no larger. The papers are chosen for subject spread rather than
# for count, so a query's ranking is legible.
LIBRARY_PAPERS = [1008, 1023, 1027, 1090, 1112, 1124, 2209, 3064]


def load_library() -> dict[str, Any]:
    ensure_project("library")
    rows = must(
        "USE citations MATCH (p:Paper) WHERE p.paper_id IN $ids "
        "RETURN p.paper_id AS paper_id, p.arxiv_id AS arxiv_id, p.title AS title, "
        "p.authors AS authors, p.submitted AS submitted, p.abstract AS abstract",
        {"ids": LIBRARY_PAPERS},
        endpoint=ENDPOINT,
    ).dicts()
    if not rows:
        raise RuntimeError("load the `citations` dataset before `library`")
    write(
        "USE library UNWIND $rows AS row "
        "CREATE (:Paper {paper_id: row.paper_id, arxiv_id: row.arxiv_id, title: row.title, "
        "authors: row.authors, submitted: row.submitted, abstract: row.abstract, "
        "embedding: [0.0]})",
        rows,
        NODE_BATCH,
        "papers",
    )
    must("USE library CREATE INDEX library_paper_by_id FOR (p:Paper) ON (p.paper_id)", endpoint=ENDPOINT)
    must("USE library CREATE TEXT INDEX library_title_text FOR (p:Paper) ON (p.title)", endpoint=ENDPOINT)
    must(
        "USE library CREATE EMBEDDING INDEX abstract_semantic FOR (p:Paper) "
        "FROM p.abstract INTO p.embedding USING MODEL default SIMILARITY COSINE",
        endpoint=ENDPOINT,
    )
    state = must("USE library SHOW INDEXES", endpoint=ENDPOINT).dicts()
    summary = counts("library")
    summary["vector_index_state"] = next(
        (row["state"] for row in state if row["kind"] == "VECTOR"), "absent"
    )
    return summary


# --------------------------------------------------------------------------------------------
# roster
# --------------------------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "flights": {
        "project": "flights",
        "title": "OpenFlights airports, airlines and routes",
        "source": "https://github.com/jpatokal/openflights",
        "licence": "Open Database License (ODbL) 1.0",
        "role": "Weighted routing: shortest paths, Dijkstra, traversal, numeric and string functions.",
        "load": load_flights,
    },
    "trust": {
        "project": "trust",
        "title": "Bitcoin OTC signed trust ratings",
        "source": "https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html",
        "licence": "Public research dataset (SNAP), cite Kumar et al. 2016",
        "role": "The temporal reference dataset: real 2010-2016 event times, declared temporal "
        "properties, HISTORY, AT TIME, WINDOW, rollups and every aggregate.",
        "load": load_trust,
    },
    "epinions": {
        "project": "epinions",
        "title": "Epinions directed trust network",
        "source": "https://snap.stanford.edu/data/soc-Epinions1.html",
        "licence": "Public research dataset (SNAP), cite Richardson et al. 2003",
        "role": "Centrality and component structure: PageRank, WCC, SCC, k-core, degree.",
        "load": load_epinions,
    },
    "dblp": {
        "project": "dblp",
        "title": "DBLP co-authorship network with ground-truth communities",
        "source": "https://snap.stanford.edu/data/com-DBLP.html",
        "licence": "Public research dataset (SNAP), cite Yang and Leskovec 2012",
        "role": "Community detection at scale, compared against published communities.",
        "load": load_dblp,
    },
    "citations": {
        "project": "citations",
        "title": "arXiv hep-th citation network with abstracts",
        "source": "https://snap.stanford.edu/data/cit-HepTh.html",
        "licence": "Public research dataset (SNAP), cite Leskovec et al. 2005",
        "role": "Text and document work: text indexes, string functions, dates, documents.",
        "load": load_citations,
    },
    "social": {
        "project": "social",
        "title": "Facebook combined ego networks",
        "source": "https://snap.stanford.edu/data/ego-Facebook.html",
        "licence": "Public research dataset (SNAP), cite Leskovec and Mcauley 2012",
        "role": "Dense undirected neighbourhoods: triangle counting and clustering coefficient.",
        "load": load_social,
    },
    "email": {
        "project": "email",
        "title": "European research institution email network",
        "source": "https://snap.stanford.edu/data/email-Eu-core.html",
        "licence": "Public research dataset (SNAP), cite Yin et al. 2017",
        "role": "Small labelled graph with 42 known departments for checking community output.",
        "load": load_email,
    },
    "library": {
        "project": "library",
        "title": "Eight arXiv papers with embedded abstracts",
        "source": "https://snap.stanford.edu/data/cit-HepTh.html",
        "licence": "Public research dataset (SNAP), cite Leskovec et al. 2005",
        "role": "Vector and text search: the largest corpus for which the vector index publishes.",
        "load": load_library,
    },
    "overflow": {
        "project": "overflow",
        "title": "MathOverflow temporal interaction network",
        "source": "https://snap.stanford.edu/data/sx-mathoverflow.html",
        "licence": "Public research dataset (SNAP), cite Paranjape et al. 2017",
        "role": "Large timestamped interaction stream for windowed time analysis.",
        "load": load_overflow,
    },
}


def main() -> int:
    global ENDPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated dataset names")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--list", action="store_true", help="print the roster and exit")
    arguments = parser.parse_args()
    ENDPOINT = arguments.endpoint

    if arguments.list:
        for name, dataset in DATASETS.items():
            print(f"{name:<10} {dataset['title']}")
            print(f"{'':<10} {dataset['role']}")
        return 0

    selected = arguments.only.split(",") if arguments.only else list(DATASETS)
    unknown = [name for name in selected if name not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    manifest: dict[str, Any] = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())

    present = existing_projects()

    for name in selected:
        dataset = DATASETS[name]
        print(f"\n{name} — {dataset['title']}", flush=True)
        if name in present:
            log(f"already present as project {name}; skipped")
            continue
        started = time.time()
        try:
            summary = dataset["load"]()
        except Exception as error:  # noqa: BLE001 - a failed dataset must not stop the rest
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)
            continue
        summary["load_seconds"] = round(time.time() - started, 1)
        manifest[name] = {
            key: dataset[key] for key in ("project", "title", "source", "licence", "role")
        } | summary
        log(f"done: {summary}")
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nmanifest written to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
