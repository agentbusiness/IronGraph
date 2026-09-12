// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Exact compiler and CPU-oracle contract for the three remaining Set1 list-property scenarios.
//!
//! All three need one canonical document/list value lane.  Scenario [5] starts from an immutable
//! matched node and needs a post-write list-comprehension result.  Scenarios [6]/[7] start from a
//! statement-local CREATE row and need ordered list concatenation plus a post-write property
//! result.  Keeping those two composition seams explicit avoids three one-off query profiles.

use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BinaryOperator, BindCapabilities, ExecutionContext, Expression, PhysicalOperator,
        QueryEngine, ResultValue, SetItem, StatementStats, bind, parse, plan,
    },
    graph::{GraphMutation, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x5345_5431_5f4c_4953_545f_5245_4d41_494e,
));

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Seam {
    MatchedListContinuation,
    CreatedListOverlay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AddShape {
    None,
    PropertyThenList,
    ListThenProperty,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    scenario: u8,
    name: &'static str,
    query: &'static str,
    output: &'static str,
    seam: Seam,
    add_shape: AddShape,
    expected_stats: StatementStats,
    expected_output: &'static [f64],
}

const CASES: [Case; 3] = [
    Case {
        scenario: 5,
        name: "[5] Adding a list property",
        query: concat!(
            "MATCH (n:A) SET n.numbers = [1, 2, 3] ",
            "RETURN [i IN n.numbers | i / 2.0] AS x"
        ),
        output: "x",
        seam: Seam::MatchedListContinuation,
        add_shape: AddShape::None,
        expected_stats: StatementStats {
            nodes_created: 0,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 1,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_output: &[0.5, 1.0, 1.5],
    },
    Case {
        scenario: 6,
        name: "[6] Concatenate elements onto a list property",
        query: concat!(
            "CREATE (a {numbers: [1, 2, 3]}) ",
            "SET a.numbers = a.numbers + [4, 5] RETURN a.numbers"
        ),
        output: "a.numbers",
        seam: Seam::CreatedListOverlay,
        add_shape: AddShape::PropertyThenList,
        expected_stats: StatementStats {
            nodes_created: 1,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 1,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_output: &[1.0, 2.0, 3.0, 4.0, 5.0],
    },
    Case {
        scenario: 7,
        name: "[7] Concatenate elements in reverse onto a list property",
        query: concat!(
            "CREATE (a {numbers: [3, 4, 5]}) ",
            "SET a.numbers = [1, 2] + a.numbers RETURN a.numbers"
        ),
        output: "a.numbers",
        seam: Seam::CreatedListOverlay,
        add_shape: AddShape::ListThenProperty,
        expected_stats: StatementStats {
            nodes_created: 1,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 1,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_output: &[1.0, 2.0, 3.0, 4.0, 5.0],
    },
];

fn fixture(case: Case) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    if case.scenario == 5 {
        let a = graph.catalog_mut().intern_label("A")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![a],
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

fn context(graph: &GraphStore) -> ExecutionContext<'_> {
    ExecutionContext {
        project_id: PROJECT,
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark {
            term: 72,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: graph
            .nodes()
            .map(|node| node.id().0.saturating_add(1))
            .max()
            .unwrap_or(1),
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 64,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn semantic_operators(case: Case, graph: &GraphStore) -> Result<Vec<PhysicalOperator>> {
    let query = parse(case.query)?;
    let bound = bind(
        query,
        graph.catalog(),
        BindCapabilities {
            write: true,
            ..BindCapabilities::default()
        },
    )?;
    Ok(plan(bound)?
        .operators
        .into_iter()
        .filter(|operator| {
            !matches!(
                operator,
                PhysicalOperator::CardinalityCheckpoint { .. } | PhysicalOperator::Finish
            )
        })
        .collect())
}

fn apply_mutations(graph: &mut GraphStore, mutations: &[GraphMutation]) -> Result<()> {
    for mutation in mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(())
}

fn numeric_list(value: &ResultValue) -> Result<Vec<f64>> {
    let ResultValue::List(values) = value else {
        return Err(Error::internal(format!(
            "expected list value, got {value:?}"
        )));
    };
    values
        .iter()
        .map(|value| match value {
            ResultValue::Scalar(ScalarValue::Integer(value)) => Ok(*value as f64),
            ResultValue::Scalar(ScalarValue::Float(value)) => Ok(value.into_inner()),
            _ => Err(Error::internal(format!(
                "expected numeric list item, got {value:?}"
            ))),
        })
        .collect()
}

fn output_list(output: &irongraph::cypher::ExecutionOutput, column: &str) -> Result<Vec<f64>> {
    let value = output
        .result
        .batches
        .first()
        .and_then(|batch| {
            batch
                .columns
                .iter()
                .find(|candidate| candidate.name == column)
        })
        .and_then(|column| column.values.first())
        .ok_or_else(|| Error::internal(format!("missing `{column}` list output")))?;
    numeric_list(value)
}

fn stored_list(graph: &GraphStore) -> Result<Vec<f64>> {
    let property = graph
        .catalog()
        .property("numbers")
        .ok_or_else(|| Error::internal("numbers property token disappeared"))?;
    let value = graph
        .nodes()
        .next()
        .and_then(|node| node.property(property))
        .ok_or_else(|| Error::internal("numbers property value disappeared"))?;
    numeric_list(&ResultValue::from_property(value)?)
}

fn is_numbers_property(expression: &Expression, variable: &str) -> bool {
    matches!(
        expression,
        Expression::Property(source, property)
            if property == "numbers"
                && matches!(source.as_ref(), Expression::Variable(actual) if actual == variable)
    )
}

#[test]
fn exact_three_case_partition_has_two_composition_seams() {
    assert_eq!(
        CASES
            .iter()
            .map(|case| case.scenario)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([5, 6, 7])
    );
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.seam == Seam::MatchedListContinuation)
            .count(),
        1
    );
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.seam == Seam::CreatedListOverlay)
            .count(),
        2
    );
}

#[test]
fn exact_source_plans_pin_list_value_and_composition_shapes() -> Result<()> {
    for case in CASES {
        let graph = fixture(case)?;
        let operators = semantic_operators(case, &graph)?;
        match case.seam {
            Seam::MatchedListContinuation => assert!(matches!(
                operators.as_slice(),
                [
                    PhysicalOperator::ScanPattern {
                        optional: false,
                        ..
                    },
                    PhysicalOperator::Set(_),
                    PhysicalOperator::Project { .. }
                ]
            )),
            Seam::CreatedListOverlay => assert!(matches!(
                operators.as_slice(),
                [
                    PhysicalOperator::CreatePattern(_),
                    PhysicalOperator::Set(_),
                    PhysicalOperator::Project { .. }
                ]
            )),
        }

        let set = operators
            .iter()
            .find_map(|operator| match operator {
                PhysicalOperator::Set(items) => Some(items.as_slice()),
                _ => None,
            })
            .ok_or_else(|| Error::internal(format!("{} omitted SET", case.name)))?;
        let [
            SetItem::Property {
                target,
                value,
                event_time: None,
            },
        ] = set
        else {
            return Err(Error::internal(format!(
                "{} changed its single-property SET shape",
                case.name
            )));
        };
        assert_eq!(target.property, "numbers", "{}", case.name);

        match case.add_shape {
            AddShape::None => {
                assert_eq!(target.variable, "n", "{}", case.name);
                assert!(matches!(value, Expression::List(values) if values.len() == 3));
            }
            AddShape::PropertyThenList | AddShape::ListThenProperty => {
                assert_eq!(target.variable, "a", "{}", case.name);
                let Expression::Binary {
                    left,
                    operation: BinaryOperator::Add,
                    right,
                } = value
                else {
                    return Err(Error::internal(format!(
                        "{} lost its list Add expression",
                        case.name
                    )));
                };
                let property_then_list =
                    is_numbers_property(left, "a") && matches!(right.as_ref(), Expression::List(_));
                let list_then_property =
                    matches!(left.as_ref(), Expression::List(_)) && is_numbers_property(right, "a");
                assert_eq!(
                    (property_then_list, list_then_property),
                    (
                        case.add_shape == AddShape::PropertyThenList,
                        case.add_shape == AddShape::ListThenProperty
                    ),
                    "{}",
                    case.name
                );
            }
        }
    }
    Ok(())
}

#[test]
fn generic_cpu_oracle_pins_all_three_list_results_and_properties() -> Result<()> {
    for case in CASES {
        let graph = fixture(case)?;
        let output = QueryEngine.execute(case.query, &mut context(&graph))?;
        assert_eq!(
            output.result.statistics, case.expected_stats,
            "{}",
            case.name
        );
        assert_eq!(
            output_list(&output, case.output)?,
            case.expected_output,
            "{}",
            case.name
        );
        assert!(!output.result.truncated, "{}", case.name);
        assert!(output.temporal_mutations.is_empty(), "{}", case.name);

        let mut committed = graph.clone();
        apply_mutations(&mut committed, &output.graph_mutations)?;
        assert_eq!(committed.node_count(), 1, "{}", case.name);
        assert_eq!(
            stored_list(&committed)?,
            [1.0, 2.0, 3.0, 4.0, 5.0]
                .get(..case.expected_output.len())
                .ok_or_else(|| Error::internal("invalid expected Set1 list length"))?,
            "{}",
            case.name
        );
    }
    Ok(())
}
