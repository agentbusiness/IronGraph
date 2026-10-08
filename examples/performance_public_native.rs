//! Complete-query baseline through the native Rust library, without the packaged SDK ABI.
use irongraph::{
    client::{Query, QueryResult, RemoteClient},
    embedded::{EmbeddedDatabase, EmbeddedOptions, ExecutionDevice},
    protocol::TypedValue,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    error::Error,
    io::{self, Write},
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
enum Surface {
    Embedded(Box<EmbeddedDatabase>),
    Remote(RemoteClient),
}
impl Surface {
    fn query(&self, query: Query) -> Result<QueryResult> {
        Ok(match self {
            Self::Embedded(database) => database.query(query)?,
            Self::Remote(client) => client.query(query)?,
        })
    }
}
fn event(value: &Value) -> Result<()> {
    println!("{value}");
    io::stdout().flush()?;
    Ok(())
}
fn checkpoint(stage: &str) -> Result<()> {
    event(&json!({"event":"checkpoint","stage":stage,"pid":std::process::id()}))?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    if line.trim() != "continue" {
        return Err("memory observer disconnected".into());
    }
    Ok(())
}
fn complete(result: &QueryResult, rows: usize, expected: Option<f64>) -> Result<()> {
    if result.rows.len() != rows
        || result.summary.truncated
        || result
            .rows
            .iter()
            .any(|row| row.len() != result.columns.len())
    {
        return Err("incomplete SDK result".into());
    }
    if let Some(expected) = expected {
        let actual = match &result.rows[0][0] {
            TypedValue::Integer(value) => value.parse().ok(),
            TypedValue::Float(value) => Some(*value),
            _ => None,
        };
        if actual != Some(expected) {
            return Err(format!("unexpected scalar {actual:?}, expected {expected}").into());
        }
    }
    Ok(())
}
fn measured(
    operation: &str,
    nodes: usize,
    durations: &[f64],
    rows: usize,
    overlap: Option<bool>,
) -> Result<()> {
    let mut sorted = durations.to_vec();
    sorted.sort_by(f64::total_cmp);
    event(
        &json!({"event":"measurement","operation":operation,"nodes":nodes,"status":"ok",
      "p50":sorted[sorted.len()/2],"durations_us":durations,"result_rows":rows,"read_overlapped_write":overlap}),
    )
}
// Keep the complete fixture sequence and timing boundaries visible in one driver.
#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let option = |name: &str, fallback: &str| {
        args.iter()
            .position(|value| value == name)
            .map_or_else(|| fallback.to_owned(), |index| args[index + 1].clone())
    };
    let selected_transport = option("--transport", "internal");
    let transport = if selected_transport == "internal" {
        "embedded".to_owned()
    } else {
        selected_transport
    };
    let manifest: Value = serde_json::from_slice(&std::fs::read(option("--manifest", ""))?)?;
    let nodes = usize::try_from(manifest["nodes"].as_u64().ok_or("missing graph size")?)?;
    let expected_nodes = f64::from(u32::try_from(nodes)?);
    let samples: usize = option("--samples", "5").parse()?;
    let warmups: usize = option("--warmups", "2").parse()?;
    let surface = match transport.as_str() {
        "embedded" => Surface::Embedded(Box::new(EmbeddedDatabase::open(
            EmbeddedOptions::new(option("--data-dir", ""))
                .with_execution_device(ExecutionDevice::Cpu),
        )?)),
        "api" => Surface::Remote(RemoteClient::api(&option("--endpoint", ""))?),
        "bolt" => Surface::Remote(RemoteClient::bolt(&option("--endpoint", ""))?),
        _ => return Err("unsupported SDK transport".into()),
    };
    let admin = if transport == "bolt" {
        Some(RemoteClient::api(
            manifest["http_endpoint"]
                .as_str()
                .ok_or("missing HTTP endpoint")?,
        )?)
    } else {
        None
    };
    let administration = |query| -> Result<QueryResult> {
        match &admin {
            Some(client) => Ok(client.query(query)?),
            None => surface.query(query),
        }
    };
    checkpoint("empty_graph_encoder_ready")?;
    administration(Query::new("CREATE PROJECT public_bench"))?;
    let mut projects = BTreeMap::new();
    {
        let name = "public_bench";
        let selected = administration(Query::new(format!("USE {name} RETURN 1")))?;
        let project = selected
            .catalog
            .ok_or("missing catalog")?
            .project_id
            .ok_or("missing project")?;
        projects.insert(name, project);
    }
    let query = |statement: &str, parameters: Value, project: &str| -> Result<QueryResult> {
        let mut query = Query::new(format!("USE {project} {statement}"));
        if transport == "bolt" {
            query = query.with_project(projects[project]);
        }
        if let Value::Object(parameters) = parameters {
            for (name, value) in parameters {
                query = query.with_parameter(name, value);
            }
        }
        surface.query(query)
    };
    for (name, statement, multiplier) in [
        (
            "initial_node_ingest",
            "UNWIND range($start,$end) AS row CREATE (:Node {value:row % 1000,bucket:row % 64,body:CASE WHEN row % $stride=0 THEN toString(row)+\":\"+$body ELSE null END}) RETURN count(*)",
            1,
        ),
        (
            "initial_edge_ingest",
            "UNWIND range($start,$end) AS row UNWIND range(1,4) AS step MATCH (source:Node) WHERE id(source)=row+1 MATCH (target:Node) WHERE id(target)=((row+step*7919)%$nodes)+1 CREATE (source)-[:R]->(target) RETURN count(*)",
            4,
        ),
    ] {
        checkpoint(&format!("{name}:before"))?;
        let start = Instant::now();
        for offset in (0..nodes).step_by(8192) {
            let end = (nodes - 1).min(offset + 8191);
            let result = query(
                statement,
                json!({"start":offset,"end":end,"stride":(nodes/100).max(1),"body":"x".repeat(2048),"nodes":nodes}),
                "public_bench",
            )?;
            complete(
                &result,
                1,
                Some(f64::from(u32::try_from((end - offset + 1) * multiplier)?)),
            )?;
        }
        let duration = start.elapsed().as_secs_f64() * 1e6;
        checkpoint(&format!("{name}:after"))?;
        measured(name, nodes, &[duration], 1, None)?;
    }
    let workloads = manifest["workloads"]
        .as_array()
        .ok_or("missing workloads")?
        .clone();
    for workload in workloads {
        let name = workload[0].as_str().ok_or("missing workload name")?;
        let statement = workload[1].as_str().ok_or("missing query")?;
        let rows = usize::try_from(workload[2].as_u64().ok_or("missing expected rows")?)?;
        let expected = workload.get(3).and_then(Value::as_f64);
        let project = if name.starts_with("flights_") {
            "flights"
        } else {
            "public_bench"
        };
        checkpoint(&format!("{name}:before"))?;
        let mut durations = Vec::new();
        for iteration in 0..warmups + samples {
            let start = Instant::now();
            let result = query(statement, json!({}), project)?;
            let duration = start.elapsed().as_secs_f64() * 1e6;
            complete(&result, rows, expected)?;
            if iteration >= warmups {
                durations.push(duration);
            }
            checkpoint(&format!("{name}:iteration:{iteration}"))?;
        }
        checkpoint(&format!("{name}:after"))?;
        measured(name, nodes, &durations, rows, None)?;
    }
    for name in [
        "parallel_full_count_queries",
        "canonical_batch_insert",
        "reader_during_real_write",
    ] {
        checkpoint(&format!("{name}:before"))?;
        let mut durations = Vec::new();
        let mut overlap = None;
        for iteration in 0..warmups + samples {
            let finished = std::sync::atomic::AtomicBool::new(false);
            let start = Instant::now();
            let duration = std::thread::scope(|scope| -> Result<f64> {
                if name == "parallel_full_count_queries" {
                    let mut readers = Vec::new();
                    for _ in 0..8 {
                        readers.push(scope.spawn(|| -> Result<()> {
                            for _ in 0..100 {
                                complete(
                                    &query(
                                        "MATCH (n:Node) RETURN count(n)",
                                        json!({}),
                                        "public_bench",
                                    )?,
                                    1,
                                    Some(expected_nodes),
                                )?;
                            }
                            Ok(())
                        }));
                    }
                    for reader in readers {
                        reader.join().map_err(|_| "reader panicked")??;
                    }
                    Ok(start.elapsed().as_secs_f64() * 1e6)
                } else {
                    let count = if name == "canonical_batch_insert" {
                        256
                    } else {
                        4096
                    };
                    let query = &query;
                    let finished = &finished;
                    let writer = scope.spawn(move || -> Result<()> {
                        let statement = concat!(
                            "UNWIND $items AS value ",
                            "CREATE (:BenchWrite {value:value}) RETURN count(*)"
                        );
                        let result = query(
                            statement,
                            json!({"items":(0..count).collect::<Vec<_>>()}),
                            "public_bench",
                        )?;
                        complete(&result, 1, Some(f64::from(count)))?;
                        finished.store(true, std::sync::atomic::Ordering::Release);
                        Ok(())
                    });
                    if name == "reader_during_real_write" {
                        complete(
                            &query("MATCH (n:Node) RETURN count(n)", json!({}), "public_bench")?,
                            1,
                            Some(expected_nodes),
                        )?;
                        overlap = Some(!finished.load(std::sync::atomic::Ordering::Acquire));
                        let duration = start.elapsed().as_secs_f64() * 1e6;
                        writer.join().map_err(|_| "writer panicked")??;
                        Ok(duration)
                    } else {
                        writer.join().map_err(|_| "writer panicked")??;
                        Ok(start.elapsed().as_secs_f64() * 1e6)
                    }
                }
            })?;
            if name != "parallel_full_count_queries" {
                query("MATCH (n:BenchWrite) DELETE n", json!({}), "public_bench")?;
            }
            if iteration >= warmups {
                durations.push(duration);
            }
            checkpoint(&format!("{name}:iteration:{iteration}"))?;
        }
        checkpoint(&format!("{name}:after"))?;
        measured(
            name,
            nodes,
            &durations,
            if name == "parallel_full_count_queries" {
                800
            } else {
                1
            },
            overlap,
        )?;
    }
    if manifest["flights"] == true {
        administration(Query::new("IMPORT DATASET flights"))?;
        let selected = administration(Query::new("USE flights RETURN 1"))?;
        let project = selected
            .catalog
            .ok_or("missing flights catalog")?
            .project_id
            .ok_or("missing flights project")?;
        for workload in manifest["flight_workloads"]
            .as_array()
            .ok_or("missing flights workloads")?
        {
            let name = workload[0].as_str().ok_or("missing workload name")?;
            let statement = workload[1].as_str().ok_or("missing query")?;
            let rows = usize::try_from(workload[2].as_u64().ok_or("missing expected rows")?)?;
            checkpoint(&format!("{name}:before"))?;
            let mut durations = Vec::new();
            for iteration in 0..warmups + samples {
                let start = Instant::now();
                let mut request = Query::new(format!("USE flights {statement}"));
                if transport == "bolt" {
                    request = request.with_project(project);
                }
                let result = surface.query(request)?;
                let duration = start.elapsed().as_secs_f64() * 1e6;
                complete(&result, rows, None)?;
                if iteration >= warmups {
                    durations.push(duration);
                }
                checkpoint(&format!("{name}:iteration:{iteration}"))?;
            }
            checkpoint(&format!("{name}:after"))?;
            measured(name, nodes, &durations, rows, None)?;
        }
    }
    checkpoint("before_close_and_durable_snapshot")?;
    match surface {
        Surface::Embedded(database) => database.close()?,
        Surface::Remote(client) => drop(client),
    }
    event(&json!({"event":"complete"}))
}
