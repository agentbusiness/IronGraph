//! Explicit, reproducible qualification; synthetic judgments are not customer demand evidence.
use irongraph_client::Query;
use irongraph_embedded::{EmbeddedDatabase, EmbeddedOptions, EmbeddingDevice};
use irongraph_server::protocol::TypedValue;
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Deserialize)]
struct Fixture {
    documents: Vec<Document>,
    questions: Vec<Question>,
}
#[derive(Deserialize)]
struct Document {
    id: i64,
    body: String,
}
#[derive(Deserialize)]
struct Question {
    text: String,
    relevant: Vec<i64>,
    split: String,
}

fn fixture() -> Result<Fixture> {
    Ok(serde_json::from_str(include_str!(
        "../../../tools/qualification/relevance.json"
    ))?)
}

fn open(path: &std::path::Path) -> Result<EmbeddedDatabase> {
    let device = if std::env::var("IRONGRAPH_QUALIFY_DEVICE").as_deref() == Ok("metal") {
        EmbeddingDevice::Metal(0)
    } else {
        EmbeddingDevice::Cpu
    };
    Ok(EmbeddedDatabase::open(
        EmbeddedOptions::new(path).with_embedding_device(device),
    )?)
}

fn wait_for_rows(db: &EmbeddedDatabase, count: usize) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        let rows = db.query(Query::new("USE scorecard SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR TEXT 'reference' USING EXACT LIMIT 1000) SCORE AS score RETURN n.id"))?;
        if rows.rows.len() == count {
            return Ok(());
        }
        if Instant::now() > deadline {
            return Err(format!(
                "embedding catch-up incomplete: {} of {count}",
                rows.rows.len()
            )
            .into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn ids(rows: &[Vec<TypedValue>]) -> Result<Vec<i64>> {
    rows.iter()
        .map(|row| match row.first() {
            Some(TypedValue::Integer(value)) => Ok(value.parse()?),
            value => Err(format!("unexpected decoded identity: {value:?}").into()),
        })
        .collect()
}

fn ndcg(returned: &[i64], relevant: &[i64]) -> f64 {
    let gain = |rank: usize| 1.0 / ((rank + 2) as f64).log2();
    let ideal: f64 = (0..relevant.len().min(10)).map(gain).sum();
    returned
        .iter()
        .take(10)
        .enumerate()
        .filter(|(_, id)| relevant.contains(id))
        .map(|(rank, _)| gain(rank))
        .sum::<f64>()
        / ideal.max(f64::EPSILON)
}

#[test]
#[ignore = "release qualification with the unchanged pinned local encoder"]
fn retrieval_relevance_scorecard() -> Result<()> {
    let data = fixture()?;
    assert_eq!(data.questions.len(), 100);
    assert_eq!(
        data.questions
            .iter()
            .filter(|q| q.split == "heldout")
            .count(),
        30
    );
    let directory = tempfile::tempdir()?;
    let startup = Instant::now();
    let db = open(directory.path())?;
    let startup_ms = startup.elapsed().as_secs_f64() * 1000.0;
    db.query(Query::new("CREATE PROJECT scorecard"))?;
    let ingest = Instant::now();
    for document in &data.documents {
        db.query(
            Query::new("USE scorecard CREATE (:Document {id:$id,body:$body})")
                .with_parameter("id", json!(document.id))
                .with_parameter("body", json!(document.body)),
        )?;
    }
    wait_for_rows(&db, data.documents.len())?;
    db.flush()?;
    println!(
        "{}",
        json!({"event":"fixture_ready","pid":std::process::id(),"documents":data.documents.len(),"queries":100,"heldout":30,"startup_ms":startup_ms,"ingest_and_embedding_ms":ingest.elapsed().as_secs_f64()*1000.0,"fixture_sha_policy":"frozen source fixture; no tuning on heldout"})
    );
    // Warm the same query embeddings for every strategy. Report this cost separately rather
    // than charging cold inference only to whichever strategy happens to execute first.
    let warm = Instant::now();
    for question in &data.questions {
        db.query(Query::new("USE scorecard SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR TEXT $text USING EXACT LIMIT 10) SCORE AS score RETURN n.id, score")
            .with_parameter("text",json!(question.text)))?;
    }
    println!(
        "{}",
        json!({"event":"query_embedding_warmup","queries":data.questions.len(),"ms":warm.elapsed().as_secs_f64()*1000.0})
    );
    for hybrid in [false, true] {
        for method in ["EXACT", "PREFIX", "ADAPTIVE", "QUANTIZED"] {
            let input = if hybrid { "HYBRID TEXT" } else { "TEXT" };
            let query = format!(
                "USE scorecard SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR {input} $text USING {method} LIMIT 10) SCORE AS score RETURN n.id, score"
            );
            let cold = Instant::now();
            db.query(Query::new(&query).with_parameter("text", json!(data.questions[0].text)))?;
            let cold_ms = cold.elapsed().as_secs_f64() * 1000.0;
            for split in ["tuning", "heldout"] {
                let mut times = Vec::new();
                let mut gains = Vec::new();
                let mut recalled = 0;
                let mut possible = 0;
                for question in data.questions.iter().filter(|q| q.split == split) {
                    let started = Instant::now();
                    let output =
                        db.query(Query::new(&query).with_parameter("text", json!(question.text)))?;
                    let returned = ids(&output.rows)?;
                    assert!(output.rows.iter().all(|row| matches!(row.get(1),Some(TypedValue::Float(score)) if score.is_finite())));
                    // Include actual query, protocol conversion and complete identity decoding.
                    times.push(started.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(
                        returned
                            .iter()
                            .collect::<std::collections::BTreeSet<_>>()
                            .len(),
                        returned.len()
                    );
                    assert!(
                        returned
                            .iter()
                            .all(|id| data.documents.iter().any(|d| d.id == *id))
                    );
                    gains.push(ndcg(&returned, &question.relevant));
                    recalled += returned
                        .iter()
                        .filter(|id| question.relevant.contains(id))
                        .count();
                    possible += question.relevant.len();
                }
                times.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    json!({"event":"relevance","hybrid":hybrid,"method":method,"split":split,"queries":times.len(),"ndcg10":gains.iter().sum::<f64>()/gains.len() as f64,"recall10":recalled as f64/possible as f64,"p50_ms":times[times.len()/2],"p95_ms":times[(times.len()*95).div_ceil(100)-1],"first_query_ms":cold_ms,"timing":"complete embedded query and decoded identities; query embeddings prewarmed equally, inference warmup reported separately; p95 nearest rank"})
                );
            }
        }
    }
    db.close()?;
    println!("RETRIEVAL_SCORECARD_OK");
    Ok(())
}

#[test]
#[ignore = "context experiment only: does not alter production windows or encoder"]
fn contextual_span_scorecard() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let db = open(directory.path())?;
    db.query(Query::new("CREATE PROJECT scorecard"))?;
    // Repeated filler makes answer and title occupy different encoder windows.
    // Context addition is explicit experiment data, never a product preprocessing change.
    let mut baseline = Vec::new();
    let mut contextual = Vec::new();
    let topics = [
        "guitar",
        "astronomy",
        "gardening",
        "aviation",
        "photography",
    ];
    for (topic, name) in topics.iter().enumerate() {
        let filler =
            "General office notes discuss calendars, routine administration and stationery. "
                .repeat(100);
        for context in [false, true] {
            let id = (topic * 2 + usize::from(context)) as i64;
            let tail = if context {
                format!(
                    "For the {name} workshop, its registration code is Z{}. ",
                    topic + 101
                )
            } else {
                format!("Its registration code is Z{}. ", topic + 101)
            };
            let body = format!("The {name} workshop overview. {filler}{tail}");
            db.query(
                Query::new("USE scorecard CREATE (:Document {id:$id,body:$body})")
                    .with_parameter("id", json!(id))
                    .with_parameter("body", json!(body)),
            )?;
        }
    }
    wait_for_rows(&db, 10)?;
    for (topic, name) in topics.iter().enumerate() {
        let rows = db.query(Query::new("USE scorecard SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR TEXT $text USING EXACT LIMIT 10) SCORE AS score RETURN n.id, score")
            .with_parameter("text",json!(format!("What is the registration code for {name}?"))))?;
        let returned = ids(&rows.rows)?;
        baseline.push(
            returned
                .iter()
                .position(|id| *id == (topic * 2) as i64)
                .ok_or("baseline owner absent")?,
        );
        contextual.push(
            returned
                .iter()
                .position(|id| *id == (topic * 2 + 1) as i64)
                .ok_or("context owner absent")?,
        );
    }
    println!(
        "{}",
        json!({"event":"context_experiment","baseline_owner_ranks_zero_based":baseline,"contextual_owner_ranks_zero_based":contextual,"production_encoder_changed":false,"qualification":"small synthetic owner-rank experiment; not supporting-span accuracy or customer relevance"})
    );
    db.close()?;
    println!("CONTEXTUAL_SPAN_SCORECARD_OK");
    Ok(())
}
