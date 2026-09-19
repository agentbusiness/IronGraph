# Reference datasets

Data-backed examples in this reference run against bundled datasets. Each is loaded into its own
IronGraph project named after it, so an example that begins `USE trust` runs against the Bitcoin OTC
rating network and nothing else. Nine datasets preserve their cited open sources; `fraud` is a small,
deterministic walkthrough fixture for learning multi-hop investigation queries.

The open datasets were chosen so that each capability has somewhere real to be demonstrated: a
weighted network to route across, a timestamped one to window, a labelled one to check a community
algorithm against, and a text-bearing one to search.

## fraud

Deterministic transfer-path walkthrough

| | |
| --- | --- |
| Project | `fraud` |
| Nodes | 7 |
| Relationships | 8 |
| Source | Bundled illustrative fixture |

This compact fixture makes the introductory multi-hop query immediately runnable. It contains three
transfer routes from `acct-100` to a high-risk destination, with amounts and UTC event times.

**Graph model**

`(:Account {id, holder, risk_score})`
`(:Account)-[:TRANSFERRED_TO {amount, occurred_at}]->(:Account)`

## Importing a dataset

Open **Training**, choose an investigation, and select **Import dataset**. IronGraph ships the
reference data with the application and imports it in the database process. You do not need Python,
a separate download, or a command-line loader.

Each dataset is imported into a project named after its slug. IronGraph checks the project catalog
first. If the project already exists, the import is skipped, so a repeat press never duplicates rows
or replaces an existing graph.

To replace a dataset deliberately, drop its project yourself, then use **Import dataset** again.
`DROP PROJECT ... CASCADE` is destructive and is never an implicit part of sample-data import.

## flights

OpenFlights airports, airlines and routes

| | |
| --- | --- |
| Project | `flights` |
| Nodes | 13,859 |
| Relationships | 66,770 |
| Source | https://github.com/jpatokal/openflights |
| Licence | Open Database License (ODbL) 1.0 |

Weighted routing: shortest paths, Dijkstra, traversal, numeric and string functions.

**Graph model**

`(:Airport {airport_id, iata, icao, name, city, country, latitude, longitude, altitude_ft, timezone})`
`(:Airline {airline_id, name, iata, icao, country, active})`
`(:Airport)-[:ROUTE {airline, stops, equipment, km}]->(:Airport)`

`km` is the great-circle distance between the two airports, computed at load time so there is a genuine numeric weight to route on.

## trust

Bitcoin OTC signed trust ratings

| | |
| --- | --- |
| Project | `trust` |
| Nodes | 5,881 |
| Relationships | 35,592 |
| Source | https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html |
| Licence | Public research dataset (SNAP), cite Kumar et al. 2016 |

The temporal reference dataset: real 2010-2016 event times, declared temporal properties, HISTORY, AT TIME, WINDOW, rollups and every aggregate.

**Event time span** — `2010-11-08T18:45:11Z` to `2016-01-25T01:12:03Z`.

**Temporal history** — 35,592 samples on a declared temporal property, each stamped with its own event time.

**Graph model**

`(:Account {account_id, reputation})`
`(:Account)-[:RATED {rating, at, at_epoch}]->(:Account)`

`rating` is an integer from -10 to +10. `at` is a datetime and `at_epoch` the same instant in seconds. `Account.reputation` is a **declared temporal property**: each rating wrote one history sample stamped with that rating's own event time, so `HISTORY` and `AT TIME` read the real 2010-2016 timeline rather than the time the data was loaded.

## epinions

Epinions directed trust network

| | |
| --- | --- |
| Project | `epinions` |
| Nodes | 75,879 |
| Relationships | 508,837 |
| Source | https://snap.stanford.edu/data/soc-Epinions1.html |
| Licence | Public research dataset (SNAP), cite Richardson et al. 2003 |

Centrality and component structure: PageRank, WCC, SCC, k-core, degree.

**Graph model**

`(:User {user_id})`
`(:User)-[:TRUSTS]->(:User)`

## dblp

DBLP co-authorship network with ground-truth communities

| | |
| --- | --- |
| Project | `dblp` |
| Nodes | 322,080 |
| Relationships | 1,162,094 |
| Source | https://snap.stanford.edu/data/com-DBLP.html |
| Licence | Public research dataset (SNAP), cite Yang and Leskovec 2012 |

Community detection at scale, compared against published communities.

**Graph model**

`(:Author {author_id})`
`(:Community {community_id, size})`
`(:Author)-[:COAUTHORED]->(:Author)`
`(:Author)-[:MEMBER_OF]->(:Community)`

The communities are the published ground truth, so an algorithm's output can be checked against groups it never saw.

## citations

arXiv hep-th citation network with abstracts

| | |
| --- | --- |
| Project | `citations` |
| Nodes | 58,572 |
| Relationships | 352,807 |
| Source | https://snap.stanford.edu/data/cit-HepTh.html |
| Licence | Public research dataset (SNAP), cite Leskovec et al. 2005 |

Text and document work: text indexes, string functions, dates, documents.

**Text** — 29,555 papers carry a full abstract.

**Graph model**

`(:Paper {paper_id, arxiv_id, title, authors, journal_ref, submitted, abstract})`
`(:Paper)-[:CITES]->(:Paper)`

29,555 papers carry a real title, author list and abstract, which is what the text functions and text indexes work on.

## library

Eight arXiv papers with embedded abstracts

| | |
| --- | --- |
| Project | `library` |
| Nodes | 8 |
| Relationships | 0 |
| Source | https://snap.stanford.edu/data/cit-HepTh.html |
| Licence | Public research dataset (SNAP), cite Leskovec et al. 2005 |

Vector and text search: a compact set of papers for exploring retrieval by meaning.

**Graph model**

`(:Paper {paper_id, arxiv_id, title, authors, submitted, abstract, embedding})`

A subset of `citations` containing eight papers chosen for subject spread. This dataset size is an
example choice, not a semantic-search limit. The complete abstracts remain on their paper nodes.
Automatic semantic search includes their meaningful content, and the dataset also provides a
field-specific index for abstract searches. Generated vectors are derived index data; applications
do not need to write placeholder vectors before declaring an embedding index.

## social

Facebook combined ego networks

| | |
| --- | --- |
| Project | `social` |
| Nodes | 4,039 |
| Relationships | 88,234 |
| Source | https://snap.stanford.edu/data/ego-Facebook.html |
| Licence | Public research dataset (SNAP), cite Leskovec and Mcauley 2012 |

Dense undirected neighbourhoods: triangle counting and clustering coefficient.

**Graph model**

`(:Person {person_id})`
`(:Person)-[:FRIEND]->(:Person)`

## email

European research institution email network

| | |
| --- | --- |
| Project | `email` |
| Nodes | 1,005 |
| Relationships | 24,929 |
| Source | https://snap.stanford.edu/data/email-Eu-core.html |
| Licence | Public research dataset (SNAP), cite Yin et al. 2017 |

Small labelled graph with 42 known departments for checking community output.

**Graph model**

`(:Member {member_id, department})`
`(:Member)-[:EMAILED]->(:Member)`

Every member's real department is recorded, giving 42 known groups to measure a community algorithm against.

## overflow

MathOverflow temporal interaction network

| | |
| --- | --- |
| Project | `overflow` |
| Nodes | 24,818 |
| Relationships | 506,550 |
| Source | https://snap.stanford.edu/data/sx-mathoverflow.html |
| Licence | Public research dataset (SNAP), cite Paranjape et al. 2017 |

Large timestamped interaction stream for windowed time analysis.

**Event time span** — `2009-09-29T02:56:28Z` to `2016-03-06T11:05:55Z`.

**Graph model**

`(:User {user_id})`
`(:User)-[:INTERACTED {at, at_epoch}]->(:User)`

Half a million interactions carrying real timestamps from 2009 to 2016.

## Licences and attribution

Each dataset keeps the licence and citation of its source, listed above. They are downloaded from
their original publishers rather than redistributed here, and the download step records exactly
where each file came from.
