use irongraph_client::Query;
use irongraph_embedded::{EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice};
use irongraph_server::protocol::TypedValue;

#[test]
fn public_quickstart_returns_the_documented_rows() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled),
    )?;

    database.query(Query::new("CREATE PROJECT recommendations"))?;
    database.query(Query::new(
        "USE recommendations CREATE \
         (ada:Developer {name: 'Ada', language: 'Rust'}), \
         (lin:Developer {name: 'Lin', language: 'Python'}), \
         (mira:Developer {name: 'Mira', language: 'TypeScript'}), \
         (graph:Technology {name: 'Graph databases'}), \
         (ada)-[:RECOMMENDS {score: 10}]->(graph), \
         (lin)-[:FOLLOWS]->(ada), \
         (mira)-[:FOLLOWS]->(lin)",
    ))?;
    let result = database.query(Query::new(
        "USE recommendations \
         MATCH (developer:Developer)-[:FOLLOWS*1..]->(:Developer)-[:RECOMMENDS]->(topic:Technology) \
         RETURN developer.name AS developer, topic.name AS recommendation \
         ORDER BY developer",
    ))?;

    assert!(matches!(
        result.rows.as_slice(),
        [first, second]
            if matches!(
                first.as_slice(),
                [TypedValue::String(developer), TypedValue::String(recommendation)]
                    if developer == "Lin" && recommendation == "Graph databases"
            ) && matches!(
                second.as_slice(),
                [TypedValue::String(developer), TypedValue::String(recommendation)]
                    if developer == "Mira" && recommendation == "Graph databases"
            )
    ));
    database.close()?;
    Ok(())
}
