use irongraph_client::{ApiClient, BoltClient, Query};
use irongraph_server::protocol::TypedValue;

#[test]
#[ignore = "requires an isolated running database with the local encoder ready"]
fn relationship_scope_is_shared_by_http_and_bolt() -> Result<(), Box<dyn std::error::Error>> {
    let api = ApiClient::new(&std::env::var("IRONGRAPH_QUALIFY_API")?)?;
    let bolt = BoltClient::new(&std::env::var("IRONGRAPH_QUALIFY_BOLT")?)?;
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let project = format!("scope_wire_{suffix}");
    api.query(Query::new(format!("CREATE PROJECT {project}")))?;
    let projects = api.query(Query::new("SHOW PROJECTS"))?;
    let id_column = projects
        .columns
        .iter()
        .position(|column| matches!(column.name.as_str(), "id" | "project_id"))
        .ok_or("project identity column absent")?;
    let row = projects
        .rows
        .iter()
        .find(|row| {
            row.iter()
                .any(|value| matches!(value, TypedValue::String(name) if name == &project))
        })
        .ok_or("project absent")?;
    let TypedValue::String(id) = &row[id_column] else {
        return Err("invalid project identity".into());
    };
    let project_id = serde_json::from_value(serde_json::Value::String(id.clone()))?;
    api.query(Query::new(format!("USE {project} CREATE (a:Document {{body:'contract'}}), (b:Person {{name:'Ada'}}), (a)-[:ABOUT {{body:'contract'}}]->(b)")))?;
    let search = format!(
        "USE {project} SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR TEXT 'contract' USING EXACT LIMIT 10) SCORE AS score RETURN r"
    );
    let wait = || -> Result<(), Box<dyn std::error::Error>> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(45);
        while api.query(Query::new(&search))?.rows.len() != 1 {
            if std::time::Instant::now() >= deadline {
                return Err("relationship catch-up timed out".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        Ok(())
    };
    wait()?;
    bolt.query(
        Query::new(format!(
            "USE {project} ALTER AUTOMATIC SEMANTIC SET RELATIONSHIPS NONE"
        ))
        .with_project(project_id),
    )?;
    assert!(api.query(Query::new(&search))?.rows.is_empty());
    assert!(
        bolt.query(Query::new(&search).with_project(project_id))?
            .rows
            .is_empty()
    );
    api.query(Query::new(format!(
        "USE {project} ALTER AUTOMATIC SEMANTIC SET RELATIONSHIPS ALL EXCEPT ABOUT"
    )))?;
    assert!(
        bolt.query(Query::new(&search).with_project(project_id))?
            .rows
            .is_empty()
    );
    api.query(Query::new(format!(
        "USE {project} ALTER AUTOMATIC SEMANTIC SET RELATIONSHIPS ALL"
    )))?;
    wait()?;
    assert_eq!(
        serde_json::to_value(api.query(Query::new(&search))?.rows)?,
        serde_json::to_value(
            bolt.query(Query::new(&search).with_project(project_id))?
                .rows
        )?
    );
    println!("RELATIONSHIP_SCOPE_HTTP_BOLT_OK");
    Ok(())
}
