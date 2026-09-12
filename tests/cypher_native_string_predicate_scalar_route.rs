// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    fs,
    time::{Duration, Instant},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::{Arc, Mutex};

use irongraph::{
    Bookmark, Error, ProjectId, Result, ScalarValue,
    cypher::{
        BinaryOperator, BindCapabilities, ColumnType, ExecutionContext, Expression,
        PhysicalOperator, PhysicalPlan, QueryEngine, ResultValue, StatementStats, bind, parse,
        plan,
    },
    gpu::{BackendKind, CpuBackend, ExecutionBackend},
    graph::{GraphStore, NameCatalog},
};
use sha2::{Digest as _, Sha256};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::{MetalBackend, ResidentProjectImage};

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MAX_RESULT_ROWS: usize = 64;
const CERTIFIED_TCK_REPORT: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";
const FRESH_TCK_REPORT: &str =
    "/tmp/irongraph-tck-full-20260721-after-merge6-set-typeconversion-return5-r1.json";
const FRESH_TCK_REPORT_SHA256: &str =
    "f47070718ad35a3fade3335ec4d3e81d4ac4a77bc3e143942f92041098ea4e77";

fn assert_certified_report_identities<'a>(
    identities: impl IntoIterator<Item = (usize, &'a str, &'a str)>,
) {
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(CERTIFIED_TCK_REPORT).expect("certified TCK report is readable"),
    )
    .expect("certified TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_184));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("certified report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);
    let mut selected = std::collections::BTreeSet::new();
    for (stored_id, feature, expanded_name) in identities {
        assert!(
            selected.insert((feature, expanded_name)),
            "duplicate local TCK identity ({feature}, {expanded_name})"
        );
        let matches = scenarios
            .iter()
            .enumerate()
            .filter_map(|(index, scenario)| {
                let path = scenario.get("path")?.as_str()?;
                let name = scenario.get("name")?.as_str()?;
                (path.ends_with(feature) && name == expanded_name).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "({feature}, {expanded_name}) resolved to {matches:?}"
        );
        assert_eq!(
            stored_id, matches[0],
            "wrong report index for {expanded_name}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct CrossProductCase {
    tck_id: u16,
    feature: &'static str,
    scenario: &'static str,
    operation: BinaryOperator,
    query: &'static str,
}

const CROSS_PRODUCT_CASES: &[CrossProductCase] = &[
    CrossProductCase {
        tck_id: 2792,
        feature: "expressions/string/String10.feature",
        scenario: "[8] Handling non-string operands for CONTAINS",
        operation: BinaryOperator::Contains,
        query: "WITH [1, 3.14, true, [], {}, null] AS operands \
                UNWIND operands AS op1 \
                UNWIND operands AS op2 \
                WITH op1 CONTAINS op2 AS v \
                RETURN v, count(*)",
    },
    CrossProductCase {
        tck_id: 2805,
        feature: "expressions/string/String8.feature",
        scenario: "[8] Handling non-string operands for STARTS WITH",
        operation: BinaryOperator::StartsWith,
        query: "WITH [1, 3.14, true, [], {}, null] AS operands \
                UNWIND operands AS op1 \
                UNWIND operands AS op2 \
                WITH op1 STARTS WITH op2 AS v \
                RETURN v, count(*)",
    },
    CrossProductCase {
        tck_id: 2814,
        feature: "expressions/string/String9.feature",
        scenario: "[8] Handling non-string operands for ENDS WITH",
        operation: BinaryOperator::EndsWith,
        query: "WITH [1, 3.14, true, [], {}, null] AS operands \
                UNWIND operands AS op1 \
                UNWIND operands AS op2 \
                WITH op1 ENDS WITH op2 AS v \
                RETURN v, count(*)",
    },
];

const PRECEDENCE_TCK_ID: u16 = 2179;
const PRECEDENCE_FEATURE: &str = "expressions/precedence/Precedence4.feature";
const PRECEDENCE_SCENARIO: &str =
    "[4] String predicate takes precedence over binary boolean operator";
const PRECEDENCE_QUERY: &str = "RETURN ('abc' STARTS WITH null OR true) = (('abc' STARTS WITH null) OR true) AS a, \
            ('abc' STARTS WITH null OR true) <> ('abc' STARTS WITH (null OR true)) AS b, \
            (true OR null STARTS WITH 'abc') = (true OR (null STARTS WITH 'abc')) AS c, \
            (true OR null STARTS WITH 'abc') <> ((true OR null) STARTS WITH 'abc') AS d";

#[derive(Clone, Debug, PartialEq)]
struct ObservedResult {
    schema: Vec<(String, ColumnType)>,
    rows: Vec<Vec<ResultValue>>,
}

fn physical_plan(query: &str) -> Result<PhysicalPlan> {
    let catalog = NameCatalog::default();
    plan(bind(parse(query)?, &catalog, BindCapabilities::default())?)
}

fn assert_heterogeneous_operands(expression: &Expression) {
    let Expression::List(values) = expression else {
        panic!("first WITH did not preserve the literal operand list: {expression:#?}");
    };
    assert_eq!(values.len(), 6, "literal operand list changed width");
    assert!(matches!(
        values[0],
        Expression::Literal(ScalarValue::Integer(1))
    ));
    assert!(matches!(
        &values[1],
        Expression::Literal(ScalarValue::Float(value)) if value.into_inner() == 3.14
    ));
    assert!(matches!(
        values[2],
        Expression::Literal(ScalarValue::Boolean(true))
    ));
    assert!(matches!(&values[3], Expression::List(values) if values.is_empty()));
    assert!(matches!(&values[4], Expression::Map(values) if values.is_empty()));
    assert!(matches!(values[5], Expression::Literal(ScalarValue::Null)));
}

fn assert_cross_product_plan(case: CrossProductCase) -> Result<()> {
    let planned = physical_plan(case.query)?;
    assert!(
        planned.read_only,
        "TCK {} unexpectedly planned a write",
        case.tck_id
    );
    assert!(
        planned.unions.is_empty(),
        "TCK {} unexpectedly planned a UNION",
        case.tck_id
    );

    let [
        PhysicalOperator::Project {
            keep_scope: false,
            projection: operands,
        },
        PhysicalOperator::Unwind {
            expression: left_source,
            variable: left_variable,
        },
        PhysicalOperator::Unwind {
            expression: right_source,
            variable: right_variable,
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: predicate,
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: grouped_count,
        },
    ] = planned.operators.as_slice()
    else {
        panic!(
            "TCK {} did not retain Project -> UNWIND -> UNWIND -> typed predicate -> grouped count: {:#?}",
            case.tck_id, planned.operators,
        );
    };

    let [operands] = operands.items.as_slice() else {
        panic!("TCK {} first WITH is not one binding", case.tck_id);
    };
    assert_eq!(operands.alias.as_deref(), Some("operands"));
    assert_heterogeneous_operands(&operands.expression);

    assert!(matches!(left_source, Expression::Variable(name) if name == "operands"));
    assert_eq!(left_variable, "op1");
    assert!(matches!(right_source, Expression::Variable(name) if name == "operands"));
    assert_eq!(right_variable, "op2");

    let [predicate] = predicate.items.as_slice() else {
        panic!("TCK {} predicate WITH is not one binding", case.tck_id);
    };
    assert_eq!(predicate.alias.as_deref(), Some("v"));
    assert!(matches!(
        &predicate.expression,
        Expression::Binary { left, operation, right }
            if *operation == case.operation
                && matches!(left.as_ref(), Expression::Variable(name) if name == "op1")
                && matches!(right.as_ref(), Expression::Variable(name) if name == "op2")
    ));

    assert!(!grouped_count.distinct);
    let [group_key, count] = grouped_count.items.as_slice() else {
        panic!(
            "TCK {} final projection lost group-key/count shape",
            case.tck_id
        );
    };
    assert!(matches!(&group_key.expression, Expression::Variable(name) if name == "v"));
    assert_eq!(group_key.column_name(0), "v");
    assert!(matches!(
        &count.expression,
        Expression::Function { name, distinct: false, arguments }
            if name.len() == 1
                && name[0].eq_ignore_ascii_case("count")
                && matches!(arguments.as_slice(), [Expression::Star])
    ));
    assert_eq!(count.column_name(1), "count(*)");
    Ok(())
}

fn count_binary_operator(expression: &Expression, target: BinaryOperator) -> usize {
    match expression {
        Expression::Binary {
            left,
            operation,
            right,
        } => {
            usize::from(*operation == target)
                + count_binary_operator(left, target)
                + count_binary_operator(right, target)
        }
        Expression::Unary { operand, .. }
        | Expression::IsNull {
            expression: operand,
            ..
        } => count_binary_operator(operand, target),
        _ => 0,
    }
}

fn assert_precedence_plan() -> Result<()> {
    let planned = physical_plan(PRECEDENCE_QUERY)?;
    let [
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
    ] = planned.operators.as_slice()
    else {
        panic!(
            "TCK {PRECEDENCE_TCK_ID} is not the expected one-row scalar projection: {:#?}",
            planned.operators,
        );
    };
    let [a, b, c, d] = projection.items.as_slice() else {
        panic!("TCK {PRECEDENCE_TCK_ID} did not retain four scalar outputs");
    };
    for (item, alias) in [(a, "a"), (b, "b"), (c, "c"), (d, "d")] {
        assert_eq!(item.alias.as_deref(), Some(alias));
        assert_eq!(
            count_binary_operator(&item.expression, BinaryOperator::StartsWith),
            2
        );
        assert_eq!(
            count_binary_operator(&item.expression, BinaryOperator::Or),
            2
        );
    }
    for (item, operation) in [
        (a, BinaryOperator::Equal),
        (b, BinaryOperator::NotEqual),
        (c, BinaryOperator::Equal),
        (d, BinaryOperator::NotEqual),
    ] {
        assert!(matches!(
            &item.expression,
            Expression::Binary { operation: actual, .. } if *actual == operation
        ));
    }
    Ok(())
}

fn context<'a>(graph: &'a GraphStore, backend: &'a dyn ExecutionBackend) -> ExecutionContext<'a> {
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
        bookmark: Bookmark { term: 0, index: 0 },
        mutation_revision: 1,
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            // This is decisive for Metal: if no complete resident plan is admitted, execution
            // must stop before the generic host evaluator can run any part of the statement.
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: MAX_RESULT_ROWS,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute(backend: &dyn ExecutionBackend, query: &str) -> Result<ObservedResult> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph, backend))?;
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
    {
        return Err(Error::internal(
            "read-only scalar predicate produced side effects or truncation",
        ));
    }

    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if batch.columns.len() != output.result.schema.len()
            || batch
                .columns
                .iter()
                .any(|column| column.values.len() != batch.row_count)
        {
            return Err(Error::internal(
                "scalar-predicate result batch is not rectangular",
            ));
        }
        for row in 0..batch.row_count {
            rows.push(
                batch
                    .columns
                    .iter()
                    .map(|column| column.values[row].clone())
                    .collect(),
            );
        }
    }
    Ok(ObservedResult {
        schema: output.result.schema,
        rows,
    })
}

fn expected_cross_product() -> ObservedResult {
    ObservedResult {
        schema: vec![
            ("v".to_owned(), ColumnType::Null),
            ("count(*)".to_owned(), ColumnType::Integer),
        ],
        rows: vec![vec![
            ResultValue::Scalar(ScalarValue::Null),
            ResultValue::Scalar(ScalarValue::Integer(36)),
        ]],
    }
}

fn expected_precedence() -> ObservedResult {
    ObservedResult {
        schema: vec![
            ("a".to_owned(), ColumnType::Boolean),
            ("b".to_owned(), ColumnType::Null),
            ("c".to_owned(), ColumnType::Boolean),
            ("d".to_owned(), ColumnType::Null),
        ],
        rows: vec![vec![
            ResultValue::Scalar(ScalarValue::Boolean(true)),
            ResultValue::Scalar(ScalarValue::Null),
            ResultValue::Scalar(ScalarValue::Boolean(true)),
            ResultValue::Scalar(ScalarValue::Null),
        ]],
    }
}

#[test]
fn pinned_upstream_manifest_and_physical_shapes_are_exact() -> Result<()> {
    assert_eq!(
        CROSS_PRODUCT_CASES
            .iter()
            .map(|case| (case.tck_id, case.feature, case.scenario))
            .collect::<Vec<_>>(),
        vec![
            (
                2792,
                "expressions/string/String10.feature",
                "[8] Handling non-string operands for CONTAINS",
            ),
            (
                2805,
                "expressions/string/String8.feature",
                "[8] Handling non-string operands for STARTS WITH",
            ),
            (
                2814,
                "expressions/string/String9.feature",
                "[8] Handling non-string operands for ENDS WITH",
            ),
        ],
    );
    assert_eq!(PRECEDENCE_TCK_ID, 2179);
    assert_eq!(
        PRECEDENCE_FEATURE,
        "expressions/precedence/Precedence4.feature"
    );
    assert_eq!(
        PRECEDENCE_SCENARIO,
        "[4] String predicate takes precedence over binary boolean operator",
    );
    for case in CROSS_PRODUCT_CASES {
        assert_cross_product_plan(*case)?;
    }
    assert_precedence_plan()
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_all_four_scalar_string_selectors() {
    assert_certified_report_identities(
        CROSS_PRODUCT_CASES
            .iter()
            .map(|case| (usize::from(case.tck_id), case.feature, case.scenario))
            .chain(std::iter::once((
                usize::from(PRECEDENCE_TCK_ID),
                PRECEDENCE_FEATURE,
                PRECEDENCE_SCENARIO,
            ))),
    );
}

#[test]
#[ignore = "external assurance gate: requires the exact fresh 3,735-Metal full report"]
fn fresh_report_pins_the_three_remaining_non_string_predicate_failures() {
    let bytes = fs::read(FRESH_TCK_REPORT).expect("fresh TCK report is readable");
    assert_eq!(hex::encode(Sha256::digest(&bytes)), FRESH_TCK_REPORT_SHA256);
    let report: serde_json::Value =
        serde_json::from_slice(&bytes).expect("fresh TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_735));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("fresh report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);

    for case in CROSS_PRODUCT_CASES {
        let scenario = &scenarios[usize::from(case.tck_id)];
        assert!(
            scenario["path"]
                .as_str()
                .is_some_and(|path| path.ends_with(case.feature)),
            "TCK {} feature identity changed",
            case.tck_id
        );
        assert_eq!(scenario["name"].as_str(), Some(case.scenario));
        assert_eq!(scenario["cpu_passed"].as_bool(), Some(true));
        assert_eq!(scenario["metal_passed"].as_bool(), Some(false));
        assert_eq!(scenario["operation_count"].as_u64(), Some(1));
        let failures = scenario["metal_failures"]
            .as_array()
            .expect("fresh scenario has Metal failures");
        assert_eq!(failures.len(), 1);
        let failure = failures[0].as_str().expect("fresh Metal failure is text");
        assert!(failure.contains(case.query));
        assert!(failure.contains("GpuAdmissionFailure"));
    }
}

#[test]
fn cpu_reference_proves_non_string_null_grouping_and_precedence_results() -> Result<()> {
    let cpu = CpuBackend::new(64 * 1024 * 1024, 16 * 1024 * 1024);
    assert_eq!(cpu.kind(), BackendKind::Cpu);
    for case in CROSS_PRODUCT_CASES {
        assert_eq!(
            execute(&cpu, case.query)?,
            expected_cross_product(),
            "TCK {} {}",
            case.tck_id,
            case.scenario,
        );
    }
    assert_eq!(
        execute(&cpu, PRECEDENCE_QUERY)?,
        expected_precedence(),
        "TCK {PRECEDENCE_TCK_ID} {PRECEDENCE_SCENARIO}",
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_matches_cpu_for_all_four_scenarios_without_host_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let graph = GraphStore::default();
    let image = ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?));
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 16 * 1024 * 1024);
    cpu.replace_all_projects(vec![image.clone()])?;
    let mut metal = MetalBackend::new(0, 128 * 1024 * 1024, 16 * 1024 * 1024)?;
    metal.replace_all_projects(vec![image])?;
    assert_eq!(metal.kind(), BackendKind::Metal);

    let mut failures = Vec::new();
    for (id, name, query, expected) in CROSS_PRODUCT_CASES
        .iter()
        .map(|case| {
            (
                case.tck_id,
                case.scenario,
                case.query,
                expected_cross_product(),
            )
        })
        .chain(std::iter::once((
            PRECEDENCE_TCK_ID,
            PRECEDENCE_SCENARIO,
            PRECEDENCE_QUERY,
            expected_precedence(),
        )))
    {
        let cpu_result = execute(&cpu, query);
        let metal_result = execute(&metal, query);
        match (cpu_result, metal_result) {
            (Ok(cpu_result), Ok(metal_result)) => {
                if cpu_result != expected {
                    failures.push(format!(
                        "TCK {id} {name}: CPU returned {cpu_result:?}, expected {expected:?}",
                    ));
                }
                if metal_result != expected {
                    failures.push(format!(
                        "TCK {id} {name}: Metal returned {metal_result:?}, expected {expected:?}",
                    ));
                }
                if metal_result != cpu_result {
                    failures.push(format!(
                        "TCK {id} {name}: CPU/Metal mismatch: CPU={cpu_result:?}, Metal={metal_result:?}",
                    ));
                }
            }
            (cpu_result, metal_result) => failures.push(format!(
                "TCK {id} {name}: strict native parity failed; CPU={cpu_result:?}; Metal={metal_result:?}",
            )),
        }
    }

    assert!(
        failures.is_empty(),
        "typed scalar string predicates are not natively conformant:\n{}",
        failures.join("\n"),
    );
    Ok(())
}
