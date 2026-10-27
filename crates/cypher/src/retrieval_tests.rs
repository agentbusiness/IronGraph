use super::*;
use crate::graph::{EmbeddingDType, EmbeddingProfile, Similarity};

struct Encoder(EmbeddingProfile);
impl TextEmbedding for Encoder {
    fn profile(&self) -> &EmbeddingProfile {
        &self.0
    }
    fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![1.0, 0.0])
    }
}

fn encoder() -> Result<Encoder> {
    Ok(Encoder(EmbeddingProfile::new(
        [1; 32],
        [2; 32],
        2,
        EmbeddingDType::F16,
        true,
        Similarity::Cosine,
    )?))
}

fn context(graph: &GraphStore) -> ExecutionContext<'_> {
    ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::from([("vector".into(), ResultValue::Vector(vec![1.0, 0.0]))]),
        bookmark: Bookmark { term: 0, index: 0 },
        mutation_revision: 1,
        resolved_time_nanos: 0,
        transaction_time_nanos: None,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: usize::MAX,
        max_batch_rows: 1024,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
        resolved_query_at_time_nanos: None,
    }
}

fn add_node(graph: &GraphStore, id: u64, text: &str, layer: Layer) -> Result<()> {
    let property = graph.catalog().intern_property("body")?;
    graph.insert_node(crate::graph::NodeInput {
        id: NodeId(id),
        layer,
        revision: 1,
        labels: Vec::new(),
        properties: vec![(property, ScalarValue::String(text.into()))],
    })?;
    Ok(())
}

fn column(output: &ExecutionOutput, at: usize) -> Vec<ResultValue> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| batch.columns[at].values.clone())
        .collect()
}

#[test]
fn relationship_scope_filters_before_ranking_across_search_routes() -> Result<()> {
    use crate::graph::{IndexCatalog, IvfPqConfig, IvfPqIndex, RelationshipSemanticScope};
    let graph = GraphStore::default();
    add_node(&graph, 1, "source", Layer::Observed)?;
    add_node(&graph, 2, "target", Layer::Observed)?;
    let excluded = graph.catalog().intern_relationship_type("CALLS")?;
    let included = graph.catalog().intern_relationship_type("EXPLAINS")?;
    let vectors = VectorIndex::new(2, Similarity::Cosine)?;
    for id in 1..=80 {
        graph.insert_edge(crate::graph::EdgeInput {
            id: crate::EdgeId(id),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: if id <= 40 { excluded } else { included },
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
        vectors.upsert(id, &[1.0, if id <= 40 { 0.0 } else { id as f32 / 80.0 }], 1)?;
    }
    let ann = IvfPqIndex::build(
        &vectors,
        IvfPqConfig {
            coarse_centroids: 4,
            subquantizers: 1,
            bits_per_code: 2,
            probes: 4,
            candidate_budget: 16,
            iterations: 2,
            ..IvfPqConfig::default()
        },
    )?;
    let indexes = IndexCatalog::default();
    let mut context = context(&graph);
    context.scalar_indexes = Some(&indexes);
    context.vector_indexes.insert(
        crate::graph::SEMANTIC_RELATIONSHIP_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY,
            exact: &vectors,
            approximate: Some(&ann),
            profile_hash: [0; 32],
        },
    );
    let source = "SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR VECTOR $vector LIMIT 2) SCORE AS score RETURN r";
    let all = QueryEngine.execute(source, &mut context)?;
    assert_eq!(
        all.vector_searches[0].access,
        crate::execution::ResidentVectorAccess::IvfPq
    );
    assert_eq!(column(&all, 0).len(), 2);
    indexes.set_relationship_semantic_scope(RelationshipSemanticScope::AllExcept(
        BTreeSet::from(["CALLS".into()]),
    ))?;
    for method in ["", "USING EXACT", "USING ADAPTIVE", "USING QUANTIZED"] {
        for prefix in ["", "MATCH ()-[r]->() "] {
            let query = format!(
                "{prefix}SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR VECTOR $vector {method} LIMIT 2) SCORE AS score RETURN r"
            );
            let output = QueryEngine.execute(&query, &mut context)?;
            let relationships = column(&output, 0);
            assert_eq!(relationships.len(), 2, "{query}");
            assert!(
                relationships.iter().all(
                    |value| matches!(value, ResultValue::Relationship(edge) if edge.id.0 > 40)
                ),
                "{query}: {relationships:?}"
            );
        }
    }
    indexes.set_relationship_semantic_scope(RelationshipSemanticScope::None)?;
    assert!(column(&QueryEngine.execute(source, &mut context)?, 0).is_empty());
    indexes.set_relationship_semantic_scope(RelationshipSemanticScope::All)?;
    assert_eq!(
        column(&QueryEngine.execute(source, &mut context)?, 0).len(),
        2
    );
    Ok(())
}

#[test]
fn relationship_scope_preserves_trained_prefix_search() -> Result<()> {
    use crate::graph::{IndexCatalog, RelationshipSemanticScope};
    struct PrefixEncoder(EmbeddingProfile);
    impl TextEmbedding for PrefixEncoder {
        fn profile(&self) -> &EmbeddingProfile {
            &self.0
        }
        fn supports_prefix_dimensions(&self, dimensions: usize) -> bool {
            dimensions == 128
        }
        fn embed(&self, _: &str) -> Result<Vec<f32>> {
            let mut vector = vec![0.0; 128];
            vector[0] = 1.0;
            Ok(vector)
        }
    }
    let graph = GraphStore::default();
    add_node(&graph, 1, "source", Layer::Observed)?;
    add_node(&graph, 2, "target", Layer::Observed)?;
    let vectors = VectorIndex::new(128, Similarity::Cosine)?;
    for (id, name) in [(10, "CALLS"), (11, "EXPLAINS")] {
        graph.insert_edge(crate::graph::EdgeInput {
            id: crate::EdgeId(id),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: graph.catalog().intern_relationship_type(name)?,
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
        let mut vector = vec![0.0; 128];
        vector[0] = 1.0;
        vector[1] = if id == 10 { 0.0 } else { 0.1 };
        vectors.upsert(id, &vector, 1)?;
    }
    let indexes = IndexCatalog::default();
    indexes.set_relationship_semantic_scope(RelationshipSemanticScope::AllExcept(
        BTreeSet::from(["CALLS".into()]),
    ))?;
    let encoder = PrefixEncoder(EmbeddingProfile::new(
        [1; 32],
        [2; 32],
        128,
        EmbeddingDType::F16,
        true,
        Similarity::Cosine,
    )?);
    let mut context = context(&graph);
    context.scalar_indexes = Some(&indexes);
    context.text_embedding = Some(&encoder);
    context.vector_indexes.insert(
        crate::graph::SEMANTIC_RELATIONSHIP_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile().profile_hash,
        },
    );
    for input in ["TEXT 'needle'", "HYBRID TEXT 'needle'"] {
        let query = format!(
            "SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR {input} USING PREFIX LIMIT 1) SCORE AS score RETURN r"
        );
        let output = QueryEngine.execute(&query, &mut context)?;
        assert!(
            matches!(column(&output, 0).as_slice(), [ResultValue::Relationship(edge)] if edge.id == crate::EdgeId(11))
        );
    }
    indexes.set_relationship_semantic_scope(RelationshipSemanticScope::None)?;
    let output = QueryEngine.execute("SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR TEXT 'needle' USING PREFIX LIMIT 1) SCORE AS score RETURN r", &mut context)?;
    assert!(column(&output, 0).is_empty());
    Ok(())
}

#[test]
fn relationship_scope_applies_to_combined_and_hybrid_lexical_search() -> Result<()> {
    use crate::graph::{IndexCatalog, RelationshipSemanticScope};
    let graph = GraphStore::default();
    add_node(&graph, 1, "source", Layer::Observed)?;
    add_node(&graph, 2, "target", Layer::Observed)?;
    let body = graph.catalog().intern_property("body")?;
    let vectors = VectorIndex::new(2, Similarity::Cosine)?;
    let nodes = VectorIndex::new(2, Similarity::Cosine)?;
    nodes.upsert(1, &[0.0, 1.0], 1)?;
    for (id, kind, value) in [(10, "CALLS", [1.0, 0.0]), (11, "EXPLAINS", [0.8, 0.2])] {
        graph.insert_edge(crate::graph::EdgeInput {
            id: crate::EdgeId(id),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: graph.catalog().intern_relationship_type(kind)?,
            layer: Layer::Observed,
            revision: 1,
            properties: vec![(body, ScalarValue::String("needle evidence".into()))],
        })?;
        vectors.upsert(id, &value, 1)?;
    }
    let indexes = IndexCatalog::default();
    indexes.set_relationship_semantic_scope(RelationshipSemanticScope::AllExcept(
        BTreeSet::from(["CALLS".into()]),
    ))?;
    let encoder = encoder()?;
    let mut context = context(&graph);
    context.scalar_indexes = Some(&indexes);
    context.text_embedding = Some(&encoder);
    for (name, property, exact) in [
        (
            crate::graph::SEMANTIC_NODE_INDEX,
            crate::graph::SEMANTIC_NODE_PROPERTY,
            &nodes,
        ),
        (
            crate::graph::SEMANTIC_RELATIONSHIP_INDEX,
            crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY,
            &vectors,
        ),
    ] {
        context.vector_indexes.insert(
            name.into(),
            VectorSearchSource {
                property,
                exact,
                approximate: None,
                profile_hash: [0; 32],
            },
        );
    }
    for index in ["semantic_relationships", "graph_semantic"] {
        for input in ["VECTOR $vector", "HYBRID TEXT 'needle'"] {
            let query = format!(
                "SEARCH r IN (EMBEDDING INDEX {index} FOR {input} USING EXACT LIMIT 1) SCORE AS score RETURN r"
            );
            let output = QueryEngine.execute(&query, &mut context)?;
            assert!(
                matches!(column(&output, 0).as_slice(), [ResultValue::Relationship(edge)] if edge.id == crate::EdgeId(11)),
                "{query}"
            );
        }
    }
    indexes.set_relationship_semantic_scope(RelationshipSemanticScope::None)?;
    let combined = QueryEngine.execute("SEARCH r IN (EMBEDDING INDEX graph_semantic FOR VECTOR $vector USING EXACT LIMIT 10) SCORE AS score RETURN r", &mut context)?;
    assert!(
        matches!(column(&combined, 0).as_slice(), [ResultValue::Node(node)] if node.id == NodeId(1))
    );
    let relationships = QueryEngine.execute("SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR HYBRID TEXT 'needle' USING EXACT LIMIT 10) SCORE AS score RETURN r", &mut context)?;
    assert!(column(&relationships, 0).is_empty());
    Ok(())
}

#[test]
fn retrieval_candidate_policy_preserves_recall_on_a_larger_corpus() -> Result<()> {
    let graph = GraphStore::default();
    let vectors = VectorIndex::new(64, Similarity::Cosine)?;
    let mut seed = 71_u64;
    for owner in 1..=2048 {
        add_node(&graph, owner, "retrieval qualification", Layer::Observed)?;
        let vector: Vec<f32> = (0..64)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 32) % 2001) as f32 / 1000.0 - 1.0
            })
            .collect();
        vectors.upsert(owner, &vector, 1)?;
    }
    let mut context = context(&graph);
    context.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: [0; 32],
        },
    );
    let mut matches = 0;
    let mut total = 0;
    for _ in 0..20 {
        let query: Vec<f32> = (0..64)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 32) % 2001) as f32 / 1000.0 - 1.0
            })
            .collect();
        context
            .parameters
            .insert("vector".into(), ResultValue::Vector(query));
        let exact = QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR VECTOR $vector USING EXACT LIMIT 10) SCORE AS score RETURN n, score", &mut context)?;
        let approximate = QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR VECTOR $vector USING QUANTIZED LIMIT 10) SCORE AS score RETURN n, score", &mut context)?;
        let expected = column(&exact, 0);
        let actual = column(&approximate, 0);
        assert_eq!(actual.len(), 10);
        assert_eq!(approximate.vector_searches[0].candidate_budget, 512);
        matches += actual.iter().filter(|node| expected.contains(node)).count();
        total += expected.len();
    }
    assert!(
        matches as f64 / total as f64 >= 0.95,
        "candidate policy recall {matches}/{total}"
    );
    Ok(())
}

#[test]
fn retrieval_hybrid_lexical_only_owner_and_stable_ties() -> Result<()> {
    let graph = GraphStore::default();
    add_node(&graph, 1, "general payment help", Layer::Observed)?;
    add_node(&graph, 2, "E104 payment cancellation", Layer::Observed)?;
    add_node(&graph, 3, "E104 manual reconciliation", Layer::Observed)?;
    add_node(&graph, 4, "E104 secret workspace", Layer::Workspace)?;
    let vectors = VectorIndex::new(2, Similarity::Cosine)?;
    vectors.upsert(1, &[1.0, 0.0], 1)?;
    vectors.upsert(2, &[0.9, 0.1], 1)?;
    vectors.upsert(4, &[1.0, 0.0], 1)?;
    let encoder = encoder()?;
    let mut context = context(&graph);
    context.text_embedding = Some(&encoder);
    context.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile().profile_hash,
        },
    );
    let query = "SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR HYBRID TEXT 'E104' USING EXACT LIMIT 3) SCORE AS score RETURN n, score";
    let first = QueryEngine.execute(query, &mut context)?;
    let nodes = column(&first, 0);
    assert!(matches!(&nodes[0], ResultValue::Node(node) if node.id == NodeId(2)));
    assert!(
        nodes
            .iter()
            .any(|value| matches!(value, ResultValue::Node(node) if node.id == NodeId(3)))
    );
    assert!(
        !nodes
            .iter()
            .any(|value| matches!(value, ResultValue::Node(node) if node.id == NodeId(4)))
    );
    for _ in 0..4 {
        let repeated = QueryEngine.execute(query, &mut context)?;
        assert_eq!(column(&repeated, 0), nodes);
        assert_eq!(column(&repeated, 1), column(&first, 1));
    }
    let filtered = QueryEngine.execute("MATCH (n) WHERE n.body = 'general payment help' SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR HYBRID TEXT 'E104' USING ADAPTIVE LIMIT 3) SCORE AS score RETURN n", &mut context)?;
    assert_eq!(column(&filtered, 0).len(), 1);
    assert!(matches!(&column(&filtered, 0)[0], ResultValue::Node(node) if node.id == NodeId(1)));
    context.cancellation.cancel();
    assert_eq!(
        QueryEngine
            .execute(query, &mut context)
            .err()
            .map(|error| error.code),
        Some(ErrorCode::Cancelled)
    );
    Ok(())
}

#[test]
fn retrieval_evidence_real_relationship_path_and_layer_visibility() -> Result<()> {
    let graph = GraphStore::default();
    add_node(&graph, 1, "supplier", Layer::Observed)?;
    add_node(&graph, 2, "shipment", Layer::Observed)?;
    add_node(&graph, 3, "hidden", Layer::Workspace)?;
    let relation = graph.catalog().intern_relationship_type("DELAYED")?;
    for (id, target) in [(10, 2), (11, 3)] {
        graph.insert_edge(crate::graph::EdgeInput {
            id: crate::EdgeId(id),
            source: NodeId(1),
            target: NodeId(target),
            relationship_type: relation,
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
    }
    let vectors = VectorIndex::new(2, Similarity::Cosine)?;
    vectors.upsert(10, &[1.0, 0.0], 1)?;
    vectors.upsert(11, &[1.0, 0.0], 1)?;
    let mut context = context(&graph);
    context.vector_indexes.insert(
        crate::graph::SEMANTIC_RELATIONSHIP_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: [0; 32],
        },
    );
    let output = QueryEngine.execute("SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR VECTOR $vector USING EXACT LIMIT 10) SCORE AS score EVIDENCE AS evidence RETURN r, evidence, startNode(r), endNode(r)", &mut context)?;
    assert_eq!(column(&output, 0).len(), 1);
    assert!(
        matches!(&column(&output, 1)[0], ResultValue::Path { nodes, relationships }
        if nodes.iter().map(|node|node.id).collect::<Vec<_>>() == vec![NodeId(1), NodeId(2)]
        && relationships.len() == 1 && relationships[0].id == crate::EdgeId(10))
    );
    assert!(matches!(&column(&output, 2)[0], ResultValue::Node(node) if node.id == NodeId(1)));
    assert!(matches!(&column(&output, 3)[0], ResultValue::Node(node) if node.id == NodeId(2)));
    let traversed = QueryEngine.execute("SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR VECTOR $vector USING EXACT LIMIT 10) SCORE AS score EVIDENCE AS evidence MATCH (supplier)-[r]->(shipment) RETURN supplier, shipment", &mut context)?;
    assert_eq!(column(&traversed, 0).len(), 1);
    assert!(matches!(&column(&traversed, 0)[0], ResultValue::Node(node) if node.id == NodeId(1)));
    assert!(matches!(&column(&traversed, 1)[0], ResultValue::Node(node) if node.id == NodeId(2)));
    let alias_error = QueryEngine.execute("SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR VECTOR $vector LIMIT 1) SCORE AS score EVIDENCE AS r RETURN r", &mut context);
    assert_eq!(
        alias_error.err().map(|error| error.code),
        Some(ErrorCode::QuerySyntax)
    );
    Ok(())
}

#[test]
fn retrieval_diagnostics_never_cap_results_or_correlated_queries() -> Result<()> {
    let graph = GraphStore::default();
    let vectors = VectorIndex::new(2, Similarity::Cosine)?;
    for id in 1..=4200 {
        add_node(&graph, id, "owner", Layer::Observed)?;
        vectors.upsert(id, &[1.0, 0.0], 1)?;
    }
    let mut context = context(&graph);
    context.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: [0; 32],
        },
    );
    let large = QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR VECTOR $vector USING EXACT LIMIT 4200) SCORE AS score RETURN count(n)", &mut context)?;
    assert_eq!(
        column(&large, 0),
        vec![ResultValue::Scalar(ScalarValue::Integer(4200))]
    );
    assert_eq!(large.vector_searches[0].hits.len(), 4096);
    let correlated = QueryEngine.execute("UNWIND range(1, 129) AS q SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR VECTOR $vector USING EXACT LIMIT 1) SCORE AS score RETURN count(n)", &mut context)?;
    assert_eq!(
        column(&correlated, 0),
        vec![ResultValue::Scalar(ScalarValue::Integer(129))]
    );
    assert_eq!(correlated.vector_searches.len(), 128);
    Ok(())
}

#[test]
fn retrieval_prefix_rejects_untrained_and_external_vectors() -> Result<()> {
    let graph = GraphStore::default();
    add_node(&graph, 1, "owner", Layer::Observed)?;
    let vectors = VectorIndex::new(2, Similarity::Cosine)?;
    vectors.upsert(1, &[1.0, 0.0], 1)?;
    let encoder = encoder()?;
    let mut context = context(&graph);
    context.text_embedding = Some(&encoder);
    context.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile().profile_hash,
        },
    );
    for query in [
        "SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR TEXT 'owner' USING PREFIX LIMIT 1) SCORE AS score RETURN n",
        "SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR VECTOR $vector USING PREFIX LIMIT 1) SCORE AS score RETURN n",
    ] {
        assert_eq!(
            QueryEngine
                .execute(query, &mut context)
                .err()
                .map(|error| error.code),
            Some(ErrorCode::EmbeddingProfileMismatch)
        );
    }
    Ok(())
}

#[test]
fn retrieval_syntax_methods_and_evidence() -> Result<()> {
    for method in ["EXACT", "PREFIX", "ADAPTIVE", "QUANTIZED"] {
        let source = format!(
            "SEARCH r IN (EMBEDDING INDEX graph_semantic FOR HYBRID TEXT $q USING {method} LIMIT 10) SCORE AS score EVIDENCE AS path RETURN path"
        );
        let query = super::super::parse(&source)?;
        let super::super::Statement::Query(body) = query.statement else {
            return Err(Error::internal("expected query"));
        };
        assert!(
            matches!(&body.clauses[0], super::super::Clause::Search(search)
            if matches!(search.input, SearchInput::HybridText(_)) && search.evidence_variable.as_deref() == Some("path"))
        );
    }
    assert!(super::super::parse("SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR TEXT 'x' USING UNKNOWN LIMIT 1) SCORE AS score RETURN n").is_err());
    assert!(super::super::parse("SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR TEXT 'x' USING DYNAMIC LIMIT 1) SCORE AS score RETURN n").is_err());
    Ok(())
}

#[test]
fn retrieval_prefix_encodes_once_and_reranks_full_canonical_vector() -> Result<()> {
    struct PrefixEncoder {
        profile: EmbeddingProfile,
        calls: std::sync::atomic::AtomicUsize,
        query: Vec<f32>,
    }
    impl TextEmbedding for PrefixEncoder {
        fn profile(&self) -> &EmbeddingProfile {
            &self.profile
        }
        fn supports_prefix_dimensions(&self, dimensions: usize) -> bool {
            dimensions == 128
        }
        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(self.query.clone())
        }
    }
    let graph = GraphStore::default();
    add_node(&graph, 1, "prefix tie wrong tail", Layer::Observed)?;
    add_node(&graph, 2, "prefix tie correct tail", Layer::Observed)?;
    let mut query = vec![0.0; 384];
    query[0] = 1.0;
    query[383] = 1.0;
    let mut wrong = query.clone();
    wrong[383] = -1.0;
    let vectors = VectorIndex::new(384, Similarity::Cosine)?;
    vectors.upsert(1, &wrong, 1)?;
    vectors.upsert(2, &query, 1)?;
    let encoder = PrefixEncoder {
        profile: EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            384,
            EmbeddingDType::F16,
            true,
            Similarity::Cosine,
        )?,
        calls: std::sync::atomic::AtomicUsize::new(0),
        query,
    };
    let mut context = context(&graph);
    context.text_embedding = Some(&encoder);
    context.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile.profile_hash,
        },
    );
    let output = QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR TEXT 'tail' USING PREFIX LIMIT 1) SCORE AS score EVIDENCE AS evidence RETURN n, score, evidence", &mut context)?;
    assert!(matches!(&column(&output, 0)[0], ResultValue::Node(node) if node.id == NodeId(2)));
    assert!(
        matches!(&column(&output, 2)[0], ResultValue::Path { nodes, relationships } if nodes.len() == 1 && nodes[0].id == NodeId(2) && relationships.is_empty())
    );
    assert_eq!(encoder.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        output.vector_searches[0].retrieval_method,
        super::super::SearchMethod::Prefix
    );
    context.vector_indexes.insert(
        "manual".into(),
        VectorSearchSource {
            property: PropertyId(42),
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile.profile_hash,
        },
    );
    assert_eq!(QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX manual FOR TEXT 'tail' USING PREFIX LIMIT 1) SCORE AS score RETURN n", &mut context).err().map(|error|error.code), Some(ErrorCode::EmbeddingProfileMismatch));
    Ok(())
}

#[test]
fn retrieval_hybrid_null_and_empty_result_scope() -> Result<()> {
    let graph = GraphStore::default();
    let vectors = VectorIndex::new(2, Similarity::Cosine)?;
    let encoder = encoder()?;
    let mut context = context(&graph);
    context.text_embedding = Some(&encoder);
    context.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile().profile_hash,
        },
    );
    let null = QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR HYBRID TEXT null USING EXACT LIMIT 1) SCORE AS score RETURN n", &mut context);
    assert_eq!(
        null.err().map(|error| error.code),
        Some(ErrorCode::QueryType)
    );
    let empty = QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR HYBRID TEXT 'missing' USING EXACT LIMIT 1) SCORE AS score EVIDENCE AS evidence RETURN *", &mut context)?;
    assert_eq!(
        empty
            .result
            .batches
            .iter()
            .map(|batch| batch.row_count)
            .sum::<usize>(),
        0
    );
    assert_eq!(empty.result.schema.len(), 3);
    Ok(())
}

#[test]
fn retrieval_hybrid_reads_prior_and_current_transaction_changes() -> Result<()> {
    let graph = GraphStore::default();
    add_node(&graph, 1, "oldtoken", Layer::Observed)?;
    let body = graph
        .catalog()
        .property("body")
        .ok_or_else(|| Error::internal("body missing"))?;
    let changes = [GraphMutation::SetNodeProperty {
        node: NodeId(1),
        property: body,
        value: ScalarValue::String("newtoken".into()),
        revision: 2,
    }];
    let vectors = VectorIndex::new(2, Similarity::Cosine)?;
    let encoder = encoder()?;
    let mut prior = context(&graph);
    prior.prior_graph_mutations = &changes;
    prior.text_embedding = Some(&encoder);
    prior.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile().profile_hash,
        },
    );
    let output = QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR HYBRID TEXT 'newtoken' USING EXACT LIMIT 1) SCORE AS score RETURN n.body", &mut prior)?;
    assert_eq!(
        column(&output, 0),
        vec![ResultValue::Scalar(ScalarValue::String("newtoken".into()))]
    );
    assert_eq!(
        graph.node(NodeId(1)).and_then(|node| node.property(body)),
        Some(ScalarValue::String("oldtoken".into()))
    );
    let empty = GraphStore::default();
    let mut current = context(&empty);
    current.capabilities.write = true;
    current.text_embedding = Some(&encoder);
    current.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile().profile_hash,
        },
    );
    let created = QueryEngine.execute("CREATE (n {body:'newtoken'}) WITH n SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR HYBRID TEXT 'newtoken' USING EXACT LIMIT 1) SCORE AS score RETURN n.body", &mut current)?;
    assert_eq!(
        column(&created, 0),
        vec![ResultValue::Scalar(ScalarValue::String("newtoken".into()))]
    );
    assert_eq!(empty.node_count(), 0);
    Ok(())
}

#[test]
fn retrieval_span_default_search_preserves_tail_matches_after_ann_cold_build() -> Result<()> {
    let graph = GraphStore::default();
    let vectors = VectorIndex::new(2, Similarity::Cosine)?;
    for id in 1..=80 {
        add_node(&graph, id, "document", Layer::Observed)?;
        vectors.upsert(id, if id == 80 { &[1.0, 0.0] } else { &[0.2, 1.0] }, 1)?;
    }
    let config = crate::graph::IvfPqConfig {
        coarse_centroids: 2,
        subquantizers: 1,
        bits_per_code: 1,
        probes: 1,
        candidate_budget: 16,
        iterations: 4,
        ..crate::graph::IvfPqConfig::default()
    };
    let initial = IvfPqIndex::build(&vectors, config)?;
    let mut query_context = context(&graph);
    query_context.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: Some(&initial),
            profile_hash: [0; 32],
        },
    );
    let source = "SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR VECTOR $vector LIMIT 1) SCORE AS score RETURN n";
    let single = QueryEngine.execute(source, &mut query_context)?;
    assert_eq!(
        single.vector_searches[0].access,
        ResidentVectorAccess::IvfPq,
        "positive control must activate default ANN on this corpus size"
    );
    assert!(matches!(&column(&single, 0)[0], ResultValue::Node(node) if node.id == NodeId(80)));
    vectors.upsert_spans(
        80,
        &[
            vectors.quantize(&[1.0, 0.0])?,
            vectors.quantize(&[0.0, 1.0])?,
        ],
        2,
    )?;
    let rebuilt = IvfPqIndex::build(&vectors, config)?;
    assert_ne!(
        rebuilt.search(&vectors, &[0.0, 1.0], 1)?[0].entity_id,
        80,
        "negative control must show first-window ANN loses the tail-only winner"
    );
    query_context
        .parameters
        .insert("vector".into(), ResultValue::Vector(vec![0.0, 1.0]));
    query_context.vector_indexes.insert(
        crate::graph::SEMANTIC_NODE_INDEX.into(),
        VectorSearchSource {
            property: crate::graph::SEMANTIC_NODE_PROPERTY,
            exact: &vectors,
            approximate: Some(&rebuilt),
            profile_hash: [0; 32],
        },
    );
    let spans = QueryEngine.execute(source, &mut query_context)?;
    assert_eq!(spans.vector_searches[0].access, ResidentVectorAccess::Exact);
    assert!(matches!(&column(&spans, 0)[0], ResultValue::Node(node) if node.id == NodeId(80)));
    Ok(())
}

#[test]
fn retrieval_hybrid_declared_field_scope_respects_overlay_and_rejects_manual_vectors() -> Result<()>
{
    let graph = GraphStore::default();
    let label = graph.catalog().intern_label("Document")?;
    let body = graph.catalog().intern_property("body")?;
    let title = graph.catalog().intern_property("title")?;
    let metadata = graph.catalog().intern_property("_metadata")?;
    let target = graph.catalog().intern_property("embedding")?;
    for (id, body_text, title_text) in [(1, "unrelated", "needle"), (2, "needle", "unrelated")] {
        graph.insert_node(crate::graph::NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![
                (body, ScalarValue::String(body_text.into())),
                (title, ScalarValue::String(title_text.into())),
                (metadata, ScalarValue::String("private-marker".into())),
            ],
        })?;
    }
    let encoder = encoder()?;
    let catalog = crate::graph::IndexCatalog::default();
    let quantizer = VectorIndex::new(2, Similarity::Cosine)?;
    catalog.create_embedding_deferred(
        &graph,
        crate::graph::EmbeddingIndexDefinition {
            name: "documents".into(),
            label,
            source_property: body,
            target_property: target,
            model: "default".into(),
        },
        encoder.profile().clone(),
        vec![
            (1, quantizer.quantize(&[0.0, 1.0])?, 1),
            (2, quantizer.quantize(&[0.0, 1.0])?, 1),
        ],
    )?;
    let (vectors, _) = catalog
        .vector_search_source("documents")
        .ok_or_else(|| Error::internal("declared source missing"))?;
    let mut context = context(&graph);
    context.text_embedding = Some(&encoder);
    context.scalar_indexes = Some(&catalog);
    context.vector_indexes.insert(
        "documents".into(),
        VectorSearchSource {
            property: target,
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile().profile_hash,
        },
    );
    let query = "SEARCH n IN (EMBEDDING INDEX documents FOR HYBRID TEXT 'needle' USING EXACT LIMIT 2) SCORE AS score RETURN n";
    let selected = QueryEngine.execute(query, &mut context)?;
    assert!(
        matches!(&column(&selected, 0)[0], ResultValue::Node(node) if node.id == NodeId(2)),
        "a title-only match must not compete with the declared body text"
    );
    // This term appears only in unrelated operational metadata, so semantic rank stays intact.
    let private = QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX documents FOR HYBRID TEXT 'private-marker' USING EXACT LIMIT 2) SCORE AS score RETURN n, score", &mut context)?;
    assert!(matches!(&column(&private, 0)[0], ResultValue::Node(node) if node.id == NodeId(1)));
    assert!(
        matches!(&column(&private, 1)[0], ResultValue::Scalar(ScalarValue::Float(score))
        if (score.0 - 1.0 / 61.0).abs() < 1e-6),
        "excluded metadata must not contribute a lexical score"
    );
    let changes = [
        GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property: body,
            value: ScalarValue::String("needle".into()),
            revision: 2,
        },
        GraphMutation::SetNodeProperty {
            node: NodeId(2),
            property: body,
            value: ScalarValue::String("unrelated".into()),
            revision: 2,
        },
    ];
    context.prior_graph_mutations = &changes;
    let prior = QueryEngine.execute(query, &mut context)?;
    assert!(matches!(&column(&prior, 0)[0], ResultValue::Node(node) if node.id == NodeId(1)));
    assert_eq!(
        graph.node(NodeId(1)).and_then(|node| node.property(body)),
        Some(ScalarValue::String("unrelated".into()))
    );
    context.vector_indexes.insert(
        "manual".into(),
        VectorSearchSource {
            property: PropertyId(42),
            exact: &vectors,
            approximate: None,
            profile_hash: encoder.profile().profile_hash,
        },
    );
    let error = QueryEngine.execute("SEARCH n IN (EMBEDDING INDEX manual FOR HYBRID TEXT 'needle' USING EXACT LIMIT 2) SCORE AS score RETURN n", &mut context)
        .err().ok_or_else(||Error::internal("manual vector index invented a lexical scope"))?;
    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("declared text embedding index"));
    Ok(())
}
