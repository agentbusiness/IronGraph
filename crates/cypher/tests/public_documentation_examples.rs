use irongraph_cypher::parse;

#[test]
fn public_documentation_cypher_examples_parse() {
    let examples = [
        "CREATE PROJECT recommendations",
        "SHOW PROJECTS",
        "USE recommendations CREATE (ada:Developer {name: 'Ada', language: 'Rust'}), (lin:Developer {name: 'Lin', language: 'Python'}), (mira:Developer {name: 'Mira', language: 'TypeScript'}), (graph:Technology {name: 'Graph databases'}), (ada)-[:RECOMMENDS {score: 10}]->(graph), (lin)-[:FOLLOWS]->(ada), (mira)-[:FOLLOWS]->(lin)",
        "USE recommendations MATCH (developer:Developer)-[:FOLLOWS*1..]->(:Developer)-[:RECOMMENDS]->(topic:Technology) RETURN developer.name AS developer, topic.name AS recommendation ORDER BY developer",
        "CREATE PROJECT IF NOT EXISTS recommendations",
        "USE recommendations MERGE (ada:Developer {name: $name}) SET ada.language = $language",
        "USE recommendations MATCH (developer:Developer) RETURN developer.name AS name ORDER BY name",
        "USE recommendations MATCH (source)-[relationship]->(target) RETURN source, relationship, target",
        "USE recommendations USE LAYER OBSERVED, KNOWLEDGE, WORKSPACE MATCH (n) RETURN n",
        "USE recommendations USE LAYER WORKSPACE WRITE LAYER WORKSPACE CREATE (:Draft {name: 'candidate'})",
        "USE recommendations SHOW TOPICS",
        "USE recommendations SHOW QUEUES",
        "CREATE PROJECT IF NOT EXISTS installation_check",
        "USE installation_check RETURN 'IronGraph is ready' AS status",
        "CREATE PROJECT IF NOT EXISTS catalog",
        "USE catalog MERGE (product:Product {sku: $sku}) SET product.name = $name",
        "USE catalog MATCH (product:Product {sku: $sku}) RETURN product.name AS name",
        "USE catalog MATCH (product:Product) RETURN product.name AS name",
        "MATCH (product:Product {sku: $sku}) RETURN product.name AS name",
        "IMPORT DATASET fraud",
        "USE fraud MATCH path = (account:Account)-[:TRANSFERRED_TO*1..4]->(destination:Account) WHERE account.id = 'acct-100' RETURN path",
        "USE catalog SHOW INDEXES",
        "USE knowledge CREATE (:Document {title: 'Device handbook', body: 'Complete source text remains on this node.'})",
        "USE events CREATE TOPIC activity PARTITIONS 12",
        "USE jobs CREATE EXCHANGE routing TYPE TOPIC",
        "USE jobs CREATE QUEUE image_processing STREAM",
        "USE jobs BIND QUEUE image_processing TO EXCHANGE routing KEY images",
    ];

    for example in examples {
        assert!(
            parse(example).is_ok(),
            "documentation example did not parse: {example}"
        );
    }
}
