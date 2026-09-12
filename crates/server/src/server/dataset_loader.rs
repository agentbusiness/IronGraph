//! Bundled reference datasets, imported through the ordinary Cypher surface.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    io::{BufRead, BufReader, Read},
};

use chrono::{DateTime, SecondsFormat, Utc};
use flate2::read::GzDecoder;
use serde_json::{Value, json};
use tar::Archive;
use uuid::Uuid;

use super::Database;
use crate::{
    Bookmark, Error, ErrorCode, Result,
    protocol::{
        BatchColumn, QueryColumn, QueryExecutor, QueryRequest, QueryStatistics, QueryStreamEvent,
        TypedValue,
    },
};

const BATCH: usize = 20_000;

const AIRPORTS: &[u8] = include_bytes!("../../../../datasets/raw/airports.dat");
const AIRLINES: &[u8] = include_bytes!("../../../../datasets/raw/airlines.dat");
const ROUTES: &[u8] = include_bytes!("../../../../datasets/raw/routes.dat");
const TRUST: &[u8] = include_bytes!("../../../../datasets/raw/soc-sign-bitcoinotc.csv.gz");
const EPINIONS: &[u8] = include_bytes!("../../../../datasets/raw/soc-Epinions1.txt.gz");
const SOCIAL: &[u8] = include_bytes!("../../../../datasets/raw/facebook_combined.txt.gz");
const EMAIL_EDGES: &[u8] = include_bytes!("../../../../datasets/raw/email-Eu-core.txt.gz");
const EMAIL_DEPARTMENTS: &[u8] =
    include_bytes!("../../../../datasets/raw/email-Eu-core-department-labels.txt.gz");
const DBLP: &[u8] = include_bytes!("../../../../datasets/raw/com-dblp.ungraph.txt.gz");
const DBLP_COMMUNITIES: &[u8] =
    include_bytes!("../../../../datasets/raw/com-dblp.top5000.cmty.txt.gz");
const OVERFLOW: &[u8] = include_bytes!("../../../../datasets/raw/sx-mathoverflow.txt.gz");
const CITATION_EDGES: &[u8] = include_bytes!("../../../../datasets/raw/cit-HepTh.txt.gz");
const CITATION_DATES: &[u8] = include_bytes!("../../../../datasets/raw/cit-HepTh-dates.txt.gz");
const CITATION_ABSTRACTS: &[u8] =
    include_bytes!("../../../../datasets/raw/cit-HepTh-abstracts.tar.gz");

#[derive(Clone, Copy)]
struct DatasetSpec {
    name: &'static str,
    nodes: u64,
    relationships: u64,
}

const DATASETS: &[DatasetSpec] = &[
    DatasetSpec {
        name: "fraud",
        nodes: 7,
        relationships: 8,
    },
    DatasetSpec {
        name: "flights",
        nodes: 13_859,
        relationships: 66_770,
    },
    DatasetSpec {
        name: "trust",
        nodes: 5_881,
        relationships: 35_592,
    },
    DatasetSpec {
        name: "epinions",
        nodes: 75_879,
        relationships: 508_837,
    },
    DatasetSpec {
        name: "email",
        nodes: 1_005,
        relationships: 24_929,
    },
    DatasetSpec {
        name: "social",
        nodes: 4_039,
        relationships: 88_234,
    },
    DatasetSpec {
        name: "library",
        nodes: 8,
        relationships: 0,
    },
    DatasetSpec {
        name: "dblp",
        nodes: 327_080,
        relationships: 1_794_425,
    },
    DatasetSpec {
        name: "citations",
        nodes: 58_572,
        relationships: 352_807,
    },
    DatasetSpec {
        name: "overflow",
        nodes: 24_818,
        relationships: 506_550,
    },
];

impl Database {
    pub(super) fn import_example_dataset(
        &self,
        request: &QueryRequest,
        name: &str,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let normalized = name.to_ascii_lowercase();
        let spec = DATASETS
            .iter()
            .find(|entry| entry.name == normalized)
            .ok_or_else(|| {
                Error::invalid_data(format!(
                    "unknown bundled dataset `{name}`; choose {}",
                    DATASETS
                        .iter()
                        .map(|entry| entry.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;
        let mut runner = DatasetRunner::new(self, request);
        let status = if self.resolve_project(spec.name).is_ok() {
            runner.run("SHOW PROJECTS", BTreeMap::new())?;
            "already_present"
        } else {
            runner.run(&format!("CREATE PROJECT {}", spec.name), BTreeMap::new())?;
            let loaded = match spec.name {
                "fraud" => runner.load_fraud(),
                "flights" => runner.load_flights(),
                "trust" => runner.load_trust(),
                "epinions" => runner
                    .load_simple_edges("epinions", "User", "user_id", "TRUSTS", EPINIONS, false),
                "email" => runner.load_email(),
                "social" => runner.load_simple_edges(
                    "social",
                    "Person",
                    "person_id",
                    "FRIEND",
                    SOCIAL,
                    false,
                ),
                "library" => runner.load_library(),
                "dblp" => runner.load_dblp(),
                "citations" => runner.load_citations(),
                "overflow" => runner.load_overflow(),
                _ => unreachable!(),
            };
            if let Err(error) = loaded {
                let _ = runner.run(
                    &format!("DROP PROJECT {} CASCADE", spec.name),
                    BTreeMap::new(),
                );
                return Err(Error::new(
                    error.code,
                    format!("dataset `{}` import stopped: {}", spec.name, error.message),
                ));
            }
            "imported"
        };
        emit_import_result(request.request_id, runner.bookmark, spec, status, emit)
    }
}

struct DatasetRunner<'a> {
    database: &'a Database,
    template: &'a QueryRequest,
    bookmark: Bookmark,
}

impl<'a> DatasetRunner<'a> {
    fn new(database: &'a Database, template: &'a QueryRequest) -> Self {
        Self {
            database,
            template,
            bookmark: Bookmark::default(),
        }
    }

    fn run(&mut self, query: &str, parameters: BTreeMap<String, Value>) -> Result<()> {
        if self.template.cancellation.is_cancelled() {
            return Err(Error::new(
                ErrorCode::Cancelled,
                "dataset import was cancelled",
            ));
        }
        let mut request = self.template.clone();
        request.request_id = Uuid::new_v4();
        request.project_id = None;
        request.query = query.to_owned();
        request.parameters = parameters;
        request.bookmark = None;
        self.database.execute_autocommit(request, &mut |event| {
            if let QueryStreamEvent::Summary { bookmark, .. } = event {
                self.bookmark = bookmark;
            }
            Ok(())
        })
    }

    fn rows(&mut self, query: &str, rows: Vec<Value>) -> Result<()> {
        let mut chunk = Vec::new();
        let mut bytes = 0usize;
        for row in rows {
            let row_bytes = serde_json::to_vec(&row).map_err(data_error)?.len();
            if !chunk.is_empty()
                && (chunk.len() >= BATCH || bytes.saturating_add(row_bytes) > 8 * 1024 * 1024)
            {
                let mut parameters = BTreeMap::new();
                parameters.insert("rows".to_owned(), Value::Array(std::mem::take(&mut chunk)));
                self.run(query, parameters)?;
                bytes = 0;
            }
            bytes = bytes.saturating_add(row_bytes);
            chunk.push(row);
        }
        if !chunk.is_empty() {
            let mut parameters = BTreeMap::new();
            parameters.insert("rows".to_owned(), Value::Array(chunk));
            self.run(query, parameters)?;
        }
        Ok(())
    }

    fn load_simple_edges(
        &mut self,
        project: &str,
        label: &str,
        id_property: &str,
        relationship: &str,
        bytes: &[u8],
        directed_timestamp: bool,
    ) -> Result<()> {
        let records = gzip_numbers(bytes)?;
        let nodes = records
            .iter()
            .flat_map(|row| row.iter().take(2).copied())
            .collect::<BTreeSet<_>>();
        self.rows(
            &format!(
                "USE {project} UNWIND $rows AS row CREATE (:{label} {{{id_property}: row.id}})"
            ),
            nodes.into_iter().map(|id| json!({ "id": id })).collect(),
        )?;
        self.run(
            &format!("USE {project} CREATE INDEX {project}_{id_property} FOR (n:{label}) ON (n.{id_property})"),
            BTreeMap::new(),
        )?;
        let query = if directed_timestamp {
            format!(
                "USE {project} UNWIND $rows AS row MATCH (source:{label} {{{id_property}: row.source}}), (target:{label} {{{id_property}: row.target}}) CREATE (source)-[:{relationship} {{at: datetime(row.at), at_epoch: row.at_epoch}}]->(target)"
            )
        } else {
            format!(
                "USE {project} UNWIND $rows AS row MATCH (source:{label} {{{id_property}: row.source}}), (target:{label} {{{id_property}: row.target}}) CREATE (source)-[:{relationship}]->(target)"
            )
        };
        let rows = records.into_iter().filter(|row| row.len() >= 2).map(|row| {
            if directed_timestamp && row.len() >= 3 {
                json!({ "source": row[0], "target": row[1], "at": iso(row[2]), "at_epoch": row[2] })
            } else {
                json!({ "source": row[0], "target": row[1] })
            }
        }).collect();
        self.rows(&query, rows)
    }

    fn load_fraud(&mut self) -> Result<()> {
        self.rows(
            "USE fraud UNWIND $rows AS row CREATE (:Account {id: row.id, holder: row.holder, risk_score: row.risk_score})",
            vec![
                json!({ "id": "acct-100", "holder": "Northwind Imports", "risk_score": 0.91 }),
                json!({ "id": "acct-200", "holder": "Blue Mesa Trading", "risk_score": 0.78 }),
                json!({ "id": "acct-300", "holder": "Kestrel Services", "risk_score": 0.83 }),
                json!({ "id": "acct-400", "holder": "Orchid Holdings", "risk_score": 0.96 }),
                json!({ "id": "acct-500", "holder": "Cedar Retail", "risk_score": 0.34 }),
                json!({ "id": "acct-600", "holder": "Harbor Supply", "risk_score": 0.42 }),
                json!({ "id": "acct-700", "holder": "Summit Works", "risk_score": 0.67 }),
            ],
        )?;
        self.run(
            "USE fraud CREATE INDEX fraud_account_by_id FOR (a:Account) ON (a.id)",
            BTreeMap::new(),
        )?;
        self.rows(
            "USE fraud UNWIND $rows AS row MATCH (source:Account {id: row.source}), (target:Account {id: row.target}) CREATE (source)-[:TRANSFERRED_TO {amount: row.amount, occurred_at: datetime(row.occurred_at)}]->(target)",
            vec![
                json!({ "source": "acct-100", "target": "acct-200", "amount": 48_000, "occurred_at": "2026-08-14T09:12:00Z" }),
                json!({ "source": "acct-200", "target": "acct-300", "amount": 47_500, "occurred_at": "2026-08-14T09:18:00Z" }),
                json!({ "source": "acct-300", "target": "acct-400", "amount": 46_900, "occurred_at": "2026-08-14T09:24:00Z" }),
                json!({ "source": "acct-100", "target": "acct-500", "amount": 12_250, "occurred_at": "2026-08-15T13:40:00Z" }),
                json!({ "source": "acct-500", "target": "acct-600", "amount": 11_975, "occurred_at": "2026-08-15T14:03:00Z" }),
                json!({ "source": "acct-600", "target": "acct-400", "amount": 11_800, "occurred_at": "2026-08-15T14:21:00Z" }),
                json!({ "source": "acct-200", "target": "acct-700", "amount": 8_300, "occurred_at": "2026-08-16T10:05:00Z" }),
                json!({ "source": "acct-700", "target": "acct-400", "amount": 8_050, "occurred_at": "2026-08-16T10:19:00Z" }),
            ],
        )
    }

    fn load_flights(&mut self) -> Result<()> {
        let mut airports = HashMap::<i64, (f64, f64)>::new();
        let mut airport_rows = Vec::new();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(AIRPORTS);
        for record in reader.records() {
            let record = record.map_err(data_error)?;
            if record.len() < 12 {
                continue;
            }
            let Ok(id) = record[0].parse::<i64>() else {
                continue;
            };
            let (Ok(latitude), Ok(longitude)) =
                (record[6].parse::<f64>(), record[7].parse::<f64>())
            else {
                continue;
            };
            airports.insert(id, (latitude, longitude));
            airport_rows.push(json!({
                "airport_id": id, "name": record[1], "city": record[2], "country": record[3],
                "iata": nullable(&record[4]), "icao": nullable(&record[5]), "latitude": latitude,
                "longitude": longitude, "altitude_ft": record[8].parse::<i64>().unwrap_or(0),
                "timezone": nullable(&record[11]),
            }));
        }
        self.rows("USE flights UNWIND $rows AS row CREATE (:Airport {airport_id: row.airport_id, name: row.name, city: row.city, country: row.country, iata: row.iata, icao: row.icao, latitude: row.latitude, longitude: row.longitude, altitude_ft: row.altitude_ft, timezone: row.timezone})", airport_rows)?;

        let mut airline_rows = Vec::new();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(AIRLINES);
        for record in reader.records() {
            let record = record.map_err(data_error)?;
            if record.len() < 8 {
                continue;
            }
            let Ok(id) = record[0].parse::<i64>() else {
                continue;
            };
            if id < 0 {
                continue;
            }
            airline_rows.push(json!({ "airline_id": id, "name": record[1], "iata": nullable(&record[3]), "icao": nullable(&record[4]), "country": nullable(&record[6]), "active": &record[7] == "Y" }));
        }
        self.rows("USE flights UNWIND $rows AS row CREATE (:Airline {airline_id: row.airline_id, name: row.name, iata: row.iata, icao: row.icao, country: row.country, active: row.active})", airline_rows)?;
        for query in [
            "USE flights CREATE INDEX airport_by_id FOR (a:Airport) ON (a.airport_id)",
            "USE flights CREATE INDEX airport_by_iata FOR (a:Airport) ON (a.iata)",
            "USE flights CREATE INDEX airline_by_id FOR (a:Airline) ON (a.airline_id)",
            "USE flights CREATE RANGE INDEX airport_by_latitude FOR (a:Airport) ON (a.latitude)",
        ] {
            self.run(query, BTreeMap::new())?;
        }

        let mut seen = HashSet::new();
        let mut route_rows = Vec::new();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(ROUTES);
        for record in reader.records() {
            let record = record.map_err(data_error)?;
            if record.len() < 9 {
                continue;
            }
            let (Ok(source), Ok(target)) = (record[3].parse::<i64>(), record[5].parse::<i64>())
            else {
                continue;
            };
            let (Some(&(lat1, lon1)), Some(&(lat2, lon2))) =
                (airports.get(&source), airports.get(&target))
            else {
                continue;
            };
            if source == target || !seen.insert((source, target, record[0].to_owned())) {
                continue;
            }
            route_rows.push(json!({ "source": source, "target": target, "airline": record[0], "stops": record[7].parse::<i64>().unwrap_or(0), "equipment": record[8].split(' ').next(), "km": haversine(lat1, lon1, lat2, lon2) }));
        }
        self.rows("USE flights UNWIND $rows AS row MATCH (source:Airport {airport_id: row.source}), (target:Airport {airport_id: row.target}) CREATE (source)-[:ROUTE {airline: row.airline, stops: row.stops, equipment: row.equipment, km: row.km}]->(target)", route_rows)
    }

    fn load_trust(&mut self) -> Result<()> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(GzDecoder::new(TRUST));
        let mut ratings = Vec::<(i64, i64, i64, i64)>::new();
        for record in reader.records() {
            let record = record.map_err(data_error)?;
            if record.len() < 4 {
                continue;
            }
            ratings.push((
                parse(&record[0])?,
                parse(&record[1])?,
                parse(&record[2])?,
                record[3].parse::<f64>().map_err(data_error)? as i64,
            ));
        }
        ratings.sort_by_key(|row| row.3);
        let accounts = ratings
            .iter()
            .flat_map(|row| [row.0, row.1])
            .collect::<BTreeSet<_>>();
        self.rows(
            "USE trust UNWIND $rows AS row CREATE (:Account {account_id: row.id, reputation: 0.0})",
            accounts.into_iter().map(|id| json!({ "id": id })).collect(),
        )?;
        self.run(
            "USE trust CREATE INDEX account_by_id FOR (a:Account) ON (a.account_id)",
            BTreeMap::new(),
        )?;
        self.rows("USE trust UNWIND $rows AS row MATCH (source:Account {account_id: row.source}), (target:Account {account_id: row.target}) CREATE (source)-[:RATED {rating: row.rating, at: datetime(row.at), at_epoch: row.at_epoch}]->(target)", ratings.iter().map(|&(source, target, rating, at)| json!({ "source": source, "target": target, "rating": rating, "at": iso(at), "at_epoch": at })).collect())?;
        self.run("USE trust ALTER NODE PROPERTY Account.reputation SET TEMPORAL FLOAT RETENTION duration('P7300D')", BTreeMap::new())?;
        let mut totals = HashMap::<i64, (i64, i64)>::new();
        let samples = ratings.into_iter().map(|(_, target, rating, at)| {
            let entry = totals.entry(target).or_default(); entry.0 += rating; entry.1 += 1;
            json!({ "account_id": target, "value": (entry.0 as f64 / entry.1 as f64 * 10_000.0).round() / 10_000.0, "at": iso(at) })
        }).collect();
        self.rows("USE trust UNWIND $rows AS row MATCH (a:Account {account_id: row.account_id}) SET a.reputation = row.value AT TIME datetime(row.at)", samples)?;
        self.run("USE trust CREATE ROLLUP reputation_monthly FOR (a:Account) ON a.reputation WINDOW TUMBLING duration('P30D') AGGREGATE avg, min, max, count", BTreeMap::new())
    }

    fn load_email(&mut self) -> Result<()> {
        let departments = gzip_numbers(EMAIL_DEPARTMENTS)?
            .into_iter()
            .filter(|row| row.len() >= 2)
            .map(|row| (row[0], row[1]))
            .collect::<HashMap<_, _>>();
        let edges = gzip_numbers(EMAIL_EDGES)?;
        let members = departments
            .keys()
            .copied()
            .chain(edges.iter().flat_map(|row| row.iter().take(2).copied()))
            .collect::<BTreeSet<_>>();
        self.rows("USE email UNWIND $rows AS row CREATE (:Member {member_id: row.id, department: row.department})", members.into_iter().map(|id| json!({ "id": id, "department": departments.get(&id).copied().unwrap_or(-1) })).collect())?;
        self.run(
            "USE email CREATE INDEX member_by_id FOR (m:Member) ON (m.member_id)",
            BTreeMap::new(),
        )?;
        self.rows("USE email UNWIND $rows AS row MATCH (source:Member {member_id: row.source}), (target:Member {member_id: row.target}) CREATE (source)-[:EMAILED]->(target)", edges.into_iter().filter(|row| row.len() >= 2 && row[0] != row[1]).map(|row| json!({ "source": row[0], "target": row[1] })).collect())
    }

    fn load_overflow(&mut self) -> Result<()> {
        self.load_simple_edges("overflow", "User", "user_id", "INTERACTED", OVERFLOW, true)
    }

    fn load_dblp(&mut self) -> Result<()> {
        self.load_simple_edges("dblp", "Author", "author_id", "COAUTHORED", DBLP, false)?;
        let communities = gzip_numbers(DBLP_COMMUNITIES)?;
        self.rows("USE dblp UNWIND $rows AS row CREATE (:Community {community_id: row.id, size: row.size})", communities.iter().enumerate().map(|(id, members)| json!({ "id": id, "size": members.len() })).collect())?;
        self.run(
            "USE dblp CREATE INDEX community_by_id FOR (c:Community) ON (c.community_id)",
            BTreeMap::new(),
        )?;
        self.rows("USE dblp UNWIND $rows AS row MATCH (a:Author {author_id: row.author_id}), (c:Community {community_id: row.community_id}) CREATE (a)-[:MEMBER_OF]->(c)", communities.into_iter().enumerate().flat_map(|(community, members)| members.into_iter().map(move |author| json!({ "author_id": author, "community_id": community }))).collect())
    }

    fn load_citations(&mut self) -> Result<()> {
        let edges = gzip_numbers(CITATION_EDGES)?;
        let dates = gzip_text(CITATION_DATES)?
            .lines()
            .filter_map(|line| {
                let mut parts = line.split_whitespace();
                Some((parts.next()?.parse::<i64>().ok()?, parts.next()?.to_owned()))
            })
            .collect::<HashMap<_, _>>();
        let metadata = citation_metadata(None)?;
        let papers = edges
            .iter()
            .flat_map(|row| row.iter().take(2).copied())
            .chain(metadata.keys().copied())
            .chain(dates.keys().copied())
            .collect::<BTreeSet<_>>();
        self.rows("USE citations UNWIND $rows AS row CREATE (:Paper {paper_id: row.paper_id, arxiv_id: row.arxiv_id, title: row.title, authors: row.authors, journal_ref: row.journal_ref, submitted: row.submitted, abstract: row.abstract})", papers.into_iter().map(|id| {
            let item = metadata.get(&id); json!({ "paper_id": id, "arxiv_id": format!("hep-th/{id:07}"), "title": item.and_then(|m| m.get("title")).cloned(), "authors": item.and_then(|m| m.get("authors")).cloned(), "journal_ref": item.and_then(|m| m.get("journal_ref")).cloned(), "submitted": dates.get(&id), "abstract": item.and_then(|m| m.get("abstract")).cloned() })
        }).collect())?;
        for query in [
            "USE citations CREATE INDEX paper_by_id FOR (p:Paper) ON (p.paper_id)",
            "USE citations CREATE TEXT INDEX paper_title_text FOR (p:Paper) ON (p.title)",
            "USE citations CREATE TEXT INDEX paper_abstract_text FOR (p:Paper) ON (p.abstract)",
        ] {
            self.run(query, BTreeMap::new())?;
        }
        self.rows("USE citations UNWIND $rows AS row MATCH (source:Paper {paper_id: row.source}), (target:Paper {paper_id: row.target}) CREATE (source)-[:CITES]->(target)", edges.into_iter().filter(|row| row.len() >= 2).map(|row| json!({ "source": row[0], "target": row[1] })).collect())
    }

    fn load_library(&mut self) -> Result<()> {
        const IDS: &[i64] = &[1008, 1023, 1027, 1090, 1112, 1124, 2209, 3064];
        let wanted = IDS.iter().copied().collect::<HashSet<_>>();
        let metadata = citation_metadata(Some(&wanted))?;
        let dates = gzip_text(CITATION_DATES)?
            .lines()
            .filter_map(|line| {
                let mut parts = line.split_whitespace();
                let id = parts.next()?.parse::<i64>().ok()?;
                wanted
                    .contains(&id)
                    .then(|| (id, parts.next().unwrap_or_default().to_owned()))
            })
            .collect::<HashMap<_, _>>();
        let rows = IDS.iter().map(|&id| {
            let item = metadata.get(&id); json!({ "paper_id": id, "arxiv_id": format!("hep-th/{id:07}"), "title": item.and_then(|m| m.get("title")).cloned(), "authors": item.and_then(|m| m.get("authors")).cloned(), "submitted": dates.get(&id), "abstract": item.and_then(|m| m.get("abstract")).cloned(), "embedding": [0.0] })
        }).collect();
        self.rows("USE library UNWIND $rows AS row CREATE (:Paper {paper_id: row.paper_id, arxiv_id: row.arxiv_id, title: row.title, authors: row.authors, submitted: row.submitted, abstract: row.abstract, embedding: row.embedding})", rows)?;
        self.run(
            "USE library CREATE INDEX library_paper_by_id FOR (p:Paper) ON (p.paper_id)",
            BTreeMap::new(),
        )?;
        self.run(
            "USE library CREATE TEXT INDEX library_title_text FOR (p:Paper) ON (p.title)",
            BTreeMap::new(),
        )?;
        self.run("USE library CREATE EMBEDDING INDEX abstract_semantic FOR (p:Paper) FROM p.abstract INTO p.embedding USING MODEL default SIMILARITY COSINE", BTreeMap::new())
    }
}

fn emit_import_result(
    request_id: Uuid,
    bookmark: Bookmark,
    spec: &DatasetSpec,
    status: &str,
    emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
) -> Result<()> {
    let definitions = [
        ("dataset", "STRING"),
        ("project", "STRING"),
        ("status", "STRING"),
        ("nodes", "INTEGER"),
        ("relationships", "INTEGER"),
    ];
    emit(QueryStreamEvent::Schema {
        request_id,
        columns: definitions
            .iter()
            .map(|(name, kind)| QueryColumn {
                name: (*name).to_owned(),
                value_type: (*kind).to_owned(),
                nullable: false,
            })
            .collect(),
    })?;
    emit(QueryStreamEvent::Batch {
        request_id,
        sequence: 0,
        row_count: 1,
        columns: vec![
            BatchColumn {
                name: "dataset".into(),
                value_type: "STRING".into(),
                values: vec![TypedValue::String(spec.name.into())],
            },
            BatchColumn {
                name: "project".into(),
                value_type: "STRING".into(),
                values: vec![TypedValue::String(spec.name.into())],
            },
            BatchColumn {
                name: "status".into(),
                value_type: "STRING".into(),
                values: vec![TypedValue::String(status.into())],
            },
            BatchColumn {
                name: "nodes".into(),
                value_type: "INTEGER".into(),
                values: vec![TypedValue::Integer(spec.nodes.to_string())],
            },
            BatchColumn {
                name: "relationships".into(),
                value_type: "INTEGER".into(),
                values: vec![TypedValue::Integer(spec.relationships.to_string())],
            },
        ],
    })?;
    emit(QueryStreamEvent::Summary {
        request_id,
        bookmark,
        statistics: QueryStatistics {
            rows: 1,
            updates: u64::from(status == "imported"),
            ..QueryStatistics::default()
        },
        truncated: false,
        truncation_reason: None,
    })
}

fn gzip_text(bytes: &[u8]) -> Result<String> {
    let mut text = String::new();
    GzDecoder::new(bytes)
        .read_to_string(&mut text)
        .map_err(data_error)?;
    Ok(text)
}

fn gzip_numbers(bytes: &[u8]) -> Result<Vec<Vec<i64>>> {
    let reader = BufReader::new(GzDecoder::new(bytes));
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(data_error)?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values = line
            .split(|ch: char| ch.is_ascii_whitespace() || ch == ',')
            .filter(|part| !part.is_empty())
            .map(parse)
            .collect::<Result<Vec<_>>>()?;
        rows.push(values);
    }
    Ok(rows)
}

fn citation_metadata(
    wanted: Option<&HashSet<i64>>,
) -> Result<HashMap<i64, HashMap<String, Value>>> {
    let mut result = HashMap::new();
    let mut archive = Archive::new(GzDecoder::new(CITATION_ABSTRACTS));
    for entry in archive.entries().map_err(data_error)? {
        let mut entry = entry.map_err(data_error)?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path().map_err(data_error)?;
        let Some(stem) = path.file_stem().and_then(|part| part.to_str()) else {
            continue;
        };
        let Ok(id) = stem.parse::<i64>() else {
            continue;
        };
        if wanted.is_some_and(|ids| !ids.contains(&id)) {
            continue;
        }
        let mut text = String::new();
        entry.read_to_string(&mut text).map_err(data_error)?;
        let mut sections = text.splitn(3, "\\\\");
        sections.next();
        let fields = sections.next().unwrap_or_default();
        let abstract_text = sections
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let mut item = HashMap::new();
        for (prefix, key) in [
            ("Title:", "title"),
            ("Authors:", "authors"),
            ("Journal-ref:", "journal_ref"),
        ] {
            if let Some(value) = fields.lines().find_map(|line| line.strip_prefix(prefix)) {
                item.insert(key.to_owned(), Value::String(value.trim().to_owned()));
            }
        }
        item.insert(
            "abstract".into(),
            Value::String(abstract_text.chars().take(4_000).collect()),
        );
        result.insert(id, item);
    }
    Ok(result)
}

fn nullable(value: &str) -> Value {
    if value.is_empty() || value == "\\N" || value == "-" {
        Value::Null
    } else {
        Value::String(value.to_owned())
    }
}

fn parse(value: &str) -> Result<i64> {
    value.parse::<i64>().map_err(data_error)
}

fn iso(epoch: i64) -> String {
    DateTime::<Utc>::from_timestamp(epoch, 0)
        .unwrap_or(DateTime::UNIX_EPOCH)
        .to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let a = ((p2 - p1) / 2.0).sin().powi(2)
        + p1.cos() * p2.cos() * ((lon2 - lon1).to_radians() / 2.0).sin().powi(2);
    (2.0 * 6_371.008_8 * a.sqrt().asin() * 1_000.0).round() / 1_000.0
}

fn data_error(error: impl std::fmt::Display) -> Error {
    Error::invalid_data(format!("bundled dataset is invalid: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bundled_edge_file_decodes() -> Result<()> {
        assert_eq!(gzip_numbers(EMAIL_EDGES)?.len(), 25_571);
        assert_eq!(gzip_numbers(SOCIAL)?.len(), 88_234);
        assert_eq!(gzip_numbers(EPINIONS)?.len(), 508_837);
        Ok(())
    }

    #[test]
    fn library_metadata_is_packaged_in_the_binary() -> Result<()> {
        let wanted = [1008, 1023, 1027, 1090, 1112, 1124, 2209, 3064]
            .into_iter()
            .collect();
        let metadata = citation_metadata(Some(&wanted))?;
        assert_eq!(metadata.len(), 8);
        assert!(metadata.values().all(|item| item.get("abstract").is_some()));
        Ok(())
    }

    #[test]
    fn import_statement_has_one_known_dataset_argument() {
        let parsed = irongraph_cypher::parse("IMPORT DATASET flights").unwrap();
        assert_eq!(
            parsed.statement,
            irongraph_cypher::Statement::ImportDataset {
                name: "flights".into()
            }
        );
    }
}
