use irongraph_client::{ApiClient, BoltClient, Query};
use irongraph_embedded::{EmbeddedDatabase, EmbeddedOptions, EmbeddingDevice, ExecutionDevice};
use irongraph_server::protocol::TypedValue;

#[test]
#[ignore = "requires an isolated running database supplied through IRONGRAPH_QUALIFY_API and IRONGRAPH_QUALIFY_BOLT"]
fn retrieval_remote_transports_preserve_methods_and_evidence()
-> Result<(), Box<dyn std::error::Error>> {
    let api = ApiClient::new(&std::env::var("IRONGRAPH_QUALIFY_API")?)?;
    let bolt = BoltClient::new(&std::env::var("IRONGRAPH_QUALIFY_BOLT")?)?;
    let project = format!("retrieval_wire_{}", std::process::id());
    api.query(Query::new(format!("CREATE PROJECT {project}")))?;
    let projects = api.query(Query::new("SHOW PROJECTS"))?;
    let id_column = projects
        .columns
        .iter()
        .position(|column| matches!(column.name.as_str(), "id" | "project_id"))
        .ok_or("project catalog has no identity column")?;
    let project_row = projects
        .rows
        .iter()
        .find(|row| {
            row.iter()
                .any(|value| matches!(value, TypedValue::String(name) if name == &project))
        })
        .ok_or("qualification project absent from catalog")?;
    let TypedValue::String(project_id) = &project_row[id_column] else {
        return Err("project identity is not a string".into());
    };
    let project_id = serde_json::from_value(serde_json::Value::String(project_id.clone()))?;
    api.query(Query::new(format!("USE {project} CREATE (d:Document {{body:'guitar music tuition E104'}}), (p:Person {{name:'Ada'}}), (d)-[:OWNED_BY {{description:'guitar music tuition E104'}}]->(p)")))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        let result = api.query(Query::new(format!("USE {project} SEARCH e IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar' USING EXACT LIMIT 10) SCORE AS score RETURN e, score")))?;
        if result.rows.len() == 3 {
            break;
        }
        if std::time::Instant::now() >= deadline {
            return Err("remote qualification owners were not embedded".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    for method in ["EXACT", "PREFIX", "ADAPTIVE", "QUANTIZED"] {
        let query = format!(
            "USE {project} SEARCH e IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar music tuition' USING {method} LIMIT 10) SCORE AS score EVIDENCE AS evidence RETURN e, evidence, score"
        );
        let http = api.query(Query::new(&query))?;
        let wire = bolt.query(Query::new(&query).with_project(project_id))?;
        assert_eq!(http.rows.len(), 3);
        assert_eq!(
            serde_json::to_value(&http.rows)?,
            serde_json::to_value(&wire.rows)?,
            "method {method}"
        );
        assert!(
            http.rows
                .iter()
                .all(|row| matches!(row[1], TypedValue::Path(_)))
        );
    }
    let query = format!(
        "USE {project} SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR HYBRID TEXT 'E104' USING EXACT LIMIT 10) SCORE AS score EVIDENCE AS evidence RETURN r, evidence, score"
    );
    let http = api.query(Query::new(&query))?;
    let wire = bolt.query(Query::new(&query).with_project(project_id))?;
    assert_eq!(http.rows.len(), 1);
    assert_eq!(
        serde_json::to_value(&http.rows)?,
        serde_json::to_value(&wire.rows)?
    );
    assert!(matches!(http.rows[0][0], TypedValue::Relationship(_)));
    Ok(())
}

#[test]
#[ignore = "release qualification: loads the pinned local embedding model"]
fn real_model_automatic_semantic_native_and_remote() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let device = match std::env::var("IRONGRAPH_QUALIFY_DEVICE").as_deref() {
        Ok("metal") => EmbeddingDevice::Metal(0),
        Ok("cuda") => EmbeddingDevice::Cuda(0),
        _ => EmbeddingDevice::Cpu,
    };
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_device(device),
    )?;
    database.query(Query::new("CREATE PROJECT qualification"))?;
    database.query(Query::new("USE qualification CREATE (p:Person {name:'Ada'}), (t:Task {title:'Arrange lessons'}), (t)-[:ASSIGNED_TO {description:'guitar music tuition'}]->(p)"))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let result = loop {
        let result = database.query(Query::new("USE qualification SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar music tuition' LIMIT 10) SCORE AS score RETURN entity, score"))?;
        if result.rows.len() == 3 {
            break result;
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "asynchronous embedding did not publish all owners: {} rows",
                result.rows.len()
            )
            .into());
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    };
    assert_eq!(result.rows.len(), 3);
    assert!(matches!(result.rows[0][0], TypedValue::Relationship(_)));
    assert!(result.rows.windows(2).all(|pair| matches!((&pair[0][1], &pair[1][1]), (TypedValue::Float(left), TypedValue::Float(right)) if left >= right)));
    for method in ["EXACT", "PREFIX", "ADAPTIVE", "QUANTIZED"] {
        let result = database.query(Query::new(format!(
            "USE qualification SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar music tuition' USING {method} LIMIT 10) SCORE AS score RETURN entity, score"
        )))?;
        assert_eq!(result.rows.len(), 3, "real encoder method {method}");
        assert!(
            matches!(result.rows[0][0], TypedValue::Relationship(_)),
            "relationship evidence must rank first for method {method}"
        );
    }
    let evidence = database.query(Query::new(
        "USE qualification SEARCH entity IN (EMBEDDING INDEX semantic_relationships FOR HYBRID TEXT 'guitar music tuition' USING EXACT LIMIT 10) SCORE AS score EVIDENCE AS evidence RETURN entity, evidence, score"
    ))?;
    assert_eq!(evidence.rows.len(), 1);
    assert!(matches!(evidence.rows[0][0], TypedValue::Relationship(_)));
    assert!(matches!(evidence.rows[0][1], TypedValue::Path(_)));
    let body = format!(
        "{} The pressure valve recovery code is E104. {}",
        "The orchard produces apples and pears every autumn. ".repeat(100),
        "The railway timetable changes in the spring. ".repeat(100)
    );
    database.query(
        Query::new("USE qualification CREATE (:Document {body:$body})")
            .with_parameter("body", serde_json::Value::String(body.clone())),
    )?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        let documents = database.query(Query::new(
            "USE qualification MATCH (d:Document) SEARCH d IN (EMBEDDING INDEX semantic_nodes FOR TEXT 'pressure valve E104' USING EXACT LIMIT 1) SCORE AS score RETURN d.body, score"
        ))?;
        if !documents.rows.is_empty() {
            assert!(matches!(&documents.rows[0][0], TypedValue::String(text) if text == &body));
            break;
        }
        if std::time::Instant::now() >= deadline {
            return Err("long-document owner spans were not published".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    database.close()?;
    if let Ok(url) = std::env::var("IRONGRAPH_QUALIFY_API") {
        let client = ApiClient::new(&url)?;
        let result = client.query(Query::new("USE acceptance SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract negotiation supplier agreements' LIMIT 8) SCORE AS score RETURN entity, score"))?;
        assert_eq!(result.rows.len(), 8);
        assert!(matches!(result.rows[0][0], TypedValue::Relationship(_)));
    }
    Ok(())
}
