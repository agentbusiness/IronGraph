// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Exact high-level acceptance gate for 29 openCypher aggregation scenarios that require a
//! resident relation source, grouping/reduction, and post-aggregate work in one backend command.
//!
//! The generic CPU replay is only the semantic oracle. Native acceptance requires the complete
//! query to cross exactly one segmented-aggregation command boundary. The observer advertises no
//! alternative native route, rejects every legacy primitive it can observe, and enables strict
//! native execution so the generic row evaluator cannot certify this gate.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
        ResidentNodeGroupPipelineRequest, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentNullableRelationRequest, ResidentNullableRelationResult, ResidentProjectImage,
        ResidentRowProgramRequest, ResidentRowProgramResult, ResidentSegmentedAggregationOperation,
        ResidentSegmentedAggregationRequest, ResidentSegmentedAggregationResult,
        ResidentSegmentedAggregationSource, ResidentSortRequest, ResidentSortResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, LayerMask},
    types::{LabelId, PropertyId},
};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const REPORT_PATH: &str = "/tmp/irongraph-tck-full-quantifier64-float-comparison-fixed.json";
const REPORT_SHA256: &str = "00d3beae4b7a33c678817c47304ce00a479b77e33eb7ab6dc53cf4e2ed4f7e5e";
const EXPECTED_FEATURE_ROOT: &str = "/tmp/irongraph-opencypher.oREeH5/tck/features";
const OFFICIAL_CASE_COUNT: usize = 29;
const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 100_000;

const MISSING_FUSED_CONTRACT: &str = "strict fused aggregation acceptance requires one complete \
resident relation-source + grouping/reduction + post-stage segmented-aggregation command";

/// `zero-based report index|feature path relative to the pinned root|exact expanded name`.
const EXACT_MANIFEST: &str = r#"
705|clauses/return/Return2.feature|[10] Return count aggregation over an empty graph
723|clauses/return/Return4.feature|[7] Keeping used expression 4
734|clauses/return/Return6.feature|[2] Projecting an arithmetic expression with aggregation
736|clauses/return/Return6.feature|[4] Support multiple divisions in aggregate function
737|clauses/return/Return6.feature|[5] Aggregates inside normal functions
741|clauses/return/Return6.feature|[9] Aggregates with arithmetics
742|clauses/return/Return6.feature|[10] Multiple aggregates on same variable
744|clauses/return/Return6.feature|[12] Counting matches per group
749|clauses/return/Return6.feature|[17] Handle constants and parameters inside an expression which contains an aggregation expression
750|clauses/return/Return6.feature|[18] Handle returned variables inside an expression which contains an aggregation expression
751|clauses/return/Return6.feature|[19] Handle returned property accesses inside an expression which contains an aggregation expression
771|clauses/return-orderby/ReturnOrderBy2.feature|[3] Sort on aggregated function
774|clauses/return-orderby/ReturnOrderBy2.feature|[6] Count star should count everything in scope
775|clauses/return-orderby/ReturnOrderBy2.feature|[7] Ordering with aggregation
779|clauses/return-orderby/ReturnOrderBy2.feature|[11] Aggregates ordered by arithmetics
783|clauses/return-orderby/ReturnOrderBy3.feature|[1] Sort on aggregate function and normal property
784|clauses/return-orderby/ReturnOrderBy4.feature|[1] ORDER BY of a column introduced in RETURN should return salient results in ascending order
787|clauses/return-orderby/ReturnOrderBy6.feature|[1] Handle constants and parameters inside an order by item which contains an aggregation expression
788|clauses/return-orderby/ReturnOrderBy6.feature|[2] Handle returned aliases inside an order by item which contains an aggregation expression
789|clauses/return-orderby/ReturnOrderBy6.feature|[3] Handle returned property accesses inside an order by item which contains an aggregation expression
920|clauses/with/With6.feature|[1] Implicit grouping with single expression as grouping key and single aggregation
1247|clauses/with-where/WithWhere6.feature|[1] Filter a single aggregate
1251|expressions/aggregation/Aggregation1.feature|[1] Count only non-null values
1265|expressions/aggregation/Aggregation3.feature|[1] Sum only non-null values
1266|expressions/aggregation/Aggregation3.feature|[2] No overflow during summation
1267|expressions/aggregation/Aggregation5.feature|[1] `collect()` filtering nulls
1268|expressions/aggregation/Aggregation5.feature|[2] OPTIONAL MATCH and `collect()` on node property
1282|expressions/aggregation/Aggregation8.feature|[1] Distinct on unbound node
1283|expressions/aggregation/Aggregation8.feature|[2] Distinct on null
"#;

const SCALAR_AGGREGATE_TAIL_MANIFEST: &str = r#"
734|clauses/return/Return6.feature|[2] Projecting an arithmetic expression with aggregation
736|clauses/return/Return6.feature|[4] Support multiple divisions in aggregate function
738|clauses/return/Return6.feature|[6] Handle aggregates inside non-aggregate expressions
741|clauses/return/Return6.feature|[9] Aggregates with arithmetics
749|clauses/return/Return6.feature|[17] Handle constants and parameters inside an expression which contains an aggregation expression
751|clauses/return/Return6.feature|[19] Handle returned property accesses inside an expression which contains an aggregation expression
924|clauses/with/With6.feature|[5] Handle constants and parameters inside an expression which contains an aggregation expression
925|clauses/with/With6.feature|[6] Handle projected variables inside an expression which contains an aggregation expression
926|clauses/with/With6.feature|[7] Handle projected property accesses inside an expression which contains an aggregation expression
787|clauses/return-orderby/ReturnOrderBy6.feature|[1] Handle constants and parameters inside an order by item which contains an aggregation expression
788|clauses/return-orderby/ReturnOrderBy6.feature|[2] Handle returned aliases inside an order by item which contains an aggregation expression
789|clauses/return-orderby/ReturnOrderBy6.feature|[3] Handle returned property accesses inside an order by item which contains an aggregation expression
1218|clauses/with-orderBy/WithOrderBy4.feature|[16] Handle constants and parameters inside an order by item which contains an aggregation expression
1219|clauses/with-orderBy/WithOrderBy4.feature|[17] Handle projected variables inside an order by item which contains an aggregation expression
1220|clauses/with-orderBy/WithOrderBy4.feature|[18]  Handle projected property accesses inside an order by item which contains an aggregation expression
"#;

/// Exact SHA-256 identities for every pinned source file containing a selected scenario.
const FEATURE_DIGESTS: [(&str, &str); 13] = [
    (
        "clauses/return/Return2.feature",
        "f12101b23bdd1d0ca662f5587cce6efde14f76838c3fe30b8706780d10ab68f8",
    ),
    (
        "clauses/return/Return4.feature",
        "892f234f2bc86d886fe1a9f1e29a85885f8e0e6c3fcc421b26a04b2945a6a496",
    ),
    (
        "clauses/return/Return6.feature",
        "f5848e39db9a93a491ea07ef5fd9dfde27ed5877d033b75f143c234cea9e850e",
    ),
    (
        "clauses/return-orderby/ReturnOrderBy2.feature",
        "b7807dc2d1837d4fbe05e07e1c3726cb5be08694604f5e4f3712122db0534503",
    ),
    (
        "clauses/return-orderby/ReturnOrderBy3.feature",
        "a456d0a34f915947cf84df7fb5147985b2faa7cd7fa738fa3774b0566029d2e2",
    ),
    (
        "clauses/return-orderby/ReturnOrderBy4.feature",
        "8b0972536db7571c4eda6a08091cbea430dee79e6e318acb3ecfa26ecde6e626",
    ),
    (
        "clauses/return-orderby/ReturnOrderBy6.feature",
        "00f71858c607c878dcb8c3780151f30c127bc051702ade7e81ee61f9757b5e7f",
    ),
    (
        "clauses/with/With6.feature",
        "c7d238b4c4c4ebac96ca686f90be8f6c343971c56f04d1cf92d65904e6c5fbd5",
    ),
    (
        "clauses/with-where/WithWhere6.feature",
        "008deca5e8251cefb4e35e828c3b0d91bffde8236e84df723cae4c86edd2023e",
    ),
    (
        "expressions/aggregation/Aggregation1.feature",
        "c21f0a94b64f97f64f58d0441c9c824d9617cfe656b5f17ef1fc91718da59bce",
    ),
    (
        "expressions/aggregation/Aggregation3.feature",
        "e2ed0ae6ba9e8c5ca0f03065af4c6151bb2dba09bd18ece91a265a89141ab369",
    ),
    (
        "expressions/aggregation/Aggregation5.feature",
        "f928d6abfb1160381d77399be5f1586b66b365f36c598bab9760b128487cb78d",
    ),
    (
        "expressions/aggregation/Aggregation8.feature",
        "fe12f0bab25361cf0c8dac585a935c7593ee2bf34d6ab180ad1e3ed40fde06b9",
    ),
];

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ManifestCase {
    report_id: usize,
    feature: String,
    name: String,
}

#[derive(Clone, Debug, Deserialize)]
struct CertifiedReport {
    total: usize,
    cpu_passed: usize,
    metal_passed: usize,
    matched: usize,
    fully_conformant: usize,
    scenarios: Vec<CertifiedScenario>,
}

#[derive(Clone, Debug, Deserialize)]
struct CertifiedScenario {
    path: String,
    name: String,
    operation_count: usize,
    cpu_passed: bool,
    metal_passed: bool,
    fully_conformant: bool,
    shared_failures: Vec<String>,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
    divergences: Vec<String>,
}

#[derive(Clone, Debug)]
struct Step {
    value: String,
    docstring: Option<String>,
    table: Vec<Vec<String>>,
}

#[derive(Clone, Debug)]
struct ParsedScenario {
    name: String,
    outline: bool,
    steps: Vec<Step>,
    examples: Vec<Vec<Vec<String>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResultOrder {
    Any,
    Exact,
    IgnoreListElementOrder,
}

#[derive(Clone, Debug)]
struct SourceCase {
    setup_queries: Vec<String>,
    parameter_literals: BTreeMap<String, String>,
    query: String,
    expected_headers: Vec<String>,
    expected_rows: Vec<Vec<String>>,
    result_order: ResultOrder,
}

fn parse_manifest(manifest: &str) -> Result<Vec<ManifestCase>> {
    manifest
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.splitn(3, '|');
            let report_id = fields
                .next()
                .ok_or_else(|| Error::internal("aggregation manifest omitted report ID"))?
                .parse::<usize>()
                .map_err(|error| Error::internal(format!("invalid report ID: {error}")))?;
            let feature = fields
                .next()
                .ok_or_else(|| Error::internal("aggregation manifest omitted feature"))?;
            let name = fields
                .next()
                .ok_or_else(|| Error::internal("aggregation manifest omitted scenario name"))?;
            Ok(ManifestCase {
                report_id,
                feature: feature.to_owned(),
                name: name.to_owned(),
            })
        })
        .collect()
}

fn manifest() -> Result<Vec<ManifestCase>> {
    parse_manifest(EXACT_MANIFEST)
}

fn scalar_aggregate_tail_manifest() -> Result<Vec<ManifestCase>> {
    parse_manifest(SCALAR_AGGREGATE_TAIL_MANIFEST)
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn certified_report() -> Result<CertifiedReport> {
    let bytes = fs::read(REPORT_PATH)
        .map_err(|error| Error::internal(format!("cannot read {REPORT_PATH}: {error}")))?;
    if sha256(&bytes) != REPORT_SHA256 {
        return Err(Error::internal(format!(
            "certified report digest changed: expected {REPORT_SHA256}, got {}",
            sha256(&bytes)
        )));
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::internal(format!("cannot decode {REPORT_PATH}: {error}")))
}

fn feature_root() -> Result<PathBuf> {
    env::var_os("OPENCYPHER_TCK_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| Error::invalid_data("OPENCYPHER_TCK_DIR is required for this gate"))
}

fn table_row(line: &str) -> Option<Vec<String>> {
    let line = line.trim();
    if line.len() < 2 || !line.starts_with('|') || !line.ends_with('|') {
        return None;
    }
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut characters = line[1..line.len() - 1].chars();
    while let Some(character) = characters.next() {
        match character {
            '|' => {
                cells.push(cell.trim().to_owned());
                cell.clear();
            }
            '\\' => match characters.next() {
                Some('|') => cell.push('|'),
                Some('\\') => cell.push('\\'),
                Some('n') => cell.push('\n'),
                Some(other) => {
                    cell.push('\\');
                    cell.push(other);
                }
                None => cell.push('\\'),
            },
            other => cell.push(other),
        }
    }
    cells.push(cell.trim().to_owned());
    Some(cells)
}

fn parse_feature(path: &Path) -> Result<Vec<ParsedScenario>> {
    let source = fs::read_to_string(path)
        .map_err(|error| Error::internal(format!("cannot read {}: {error}", path.display())))?;
    let mut scenarios = Vec::new();
    let mut current: Option<ParsedScenario> = None;
    let mut current_step: Option<usize> = None;
    let mut in_docstring = false;
    let mut in_examples = false;
    let mut example_rows = Vec::new();

    let finish = |current: &mut Option<ParsedScenario>,
                  example_rows: &mut Vec<Vec<String>>,
                  scenarios: &mut Vec<ParsedScenario>| {
        if let Some(mut scenario) = current.take() {
            if !example_rows.is_empty() {
                scenario.examples.push(std::mem::take(example_rows));
            }
            scenarios.push(scenario);
        }
    };

    for raw_line in source.lines() {
        let line = raw_line.trim_end_matches('\r');
        let trimmed = line.trim();
        if in_docstring {
            if trimmed == "\"\"\"" {
                in_docstring = false;
            } else if let Some(step) = current_step.and_then(|index| {
                current
                    .as_mut()
                    .and_then(|scenario| scenario.steps.get_mut(index))
            }) {
                let docstring = step.docstring.get_or_insert_with(String::new);
                if !docstring.is_empty() {
                    docstring.push('\n');
                }
                docstring.push_str(line.trim_start());
            }
            continue;
        }
        if trimmed == "\"\"\"" {
            in_docstring = true;
            continue;
        }
        if let Some(name) = trimmed.strip_prefix("Scenario Outline:") {
            finish(&mut current, &mut example_rows, &mut scenarios);
            current = Some(ParsedScenario {
                name: name.trim().to_owned(),
                outline: true,
                steps: Vec::new(),
                examples: Vec::new(),
            });
            current_step = None;
            in_examples = false;
            continue;
        }
        if let Some(name) = trimmed.strip_prefix("Scenario:") {
            finish(&mut current, &mut example_rows, &mut scenarios);
            current = Some(ParsedScenario {
                name: name.trim().to_owned(),
                outline: false,
                steps: Vec::new(),
                examples: Vec::new(),
            });
            current_step = None;
            in_examples = false;
            continue;
        }
        if trimmed.starts_with("Examples:") {
            if !example_rows.is_empty()
                && let Some(scenario) = current.as_mut()
            {
                scenario.examples.push(std::mem::take(&mut example_rows));
            }
            in_examples = true;
            current_step = None;
            continue;
        }
        if in_examples {
            if let Some(row) = table_row(trimmed) {
                example_rows.push(row);
                continue;
            }
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                in_examples = false;
            }
        }
        let is_step = ["Given ", "When ", "Then ", "And ", "But "]
            .iter()
            .any(|prefix| trimmed.starts_with(prefix));
        if is_step {
            let value = trimmed
                .split_once(' ')
                .map_or(trimmed, |(_, value)| value)
                .trim()
                .to_owned();
            if let Some(scenario) = current.as_mut() {
                scenario.steps.push(Step {
                    value,
                    docstring: None,
                    table: Vec::new(),
                });
                current_step = Some(scenario.steps.len() - 1);
            }
            continue;
        }
        if let Some(row) = table_row(trimmed)
            && let Some(step) = current_step.and_then(|index| {
                current
                    .as_mut()
                    .and_then(|scenario| scenario.steps.get_mut(index))
            })
        {
            step.table.push(row);
        }
    }
    if in_docstring {
        return Err(Error::internal(format!(
            "{} contains an unterminated docstring",
            path.display()
        )));
    }
    finish(&mut current, &mut example_rows, &mut scenarios);
    Ok(scenarios)
}

fn substitute(value: &str, headers: &[String], row: &[String]) -> String {
    headers
        .iter()
        .zip(row)
        .fold(value.to_owned(), |value, (header, cell)| {
            value.replace(&format!("<{header}>"), cell)
        })
}

fn substitute_step(step: &Step, headers: &[String], row: &[String]) -> Step {
    Step {
        value: substitute(&step.value, headers, row),
        docstring: step
            .docstring
            .as_ref()
            .map(|value| substitute(value, headers, row)),
        table: step
            .table
            .iter()
            .map(|table_row| {
                table_row
                    .iter()
                    .map(|cell| substitute(cell, headers, row))
                    .collect()
            })
            .collect(),
    }
}

fn source_base_name(name: &str) -> &str {
    if name.ends_with(']')
        && let Some(start) = name.rfind(" [")
        && let Some(digits) = name[start + 2..].strip_suffix(']')
        && !digits.is_empty()
        && digits.chars().all(|character| character.is_ascii_digit())
    {
        &name[..start]
    } else {
        name
    }
}

fn source_case_from_steps(name: &str, steps: &[Step]) -> Result<SourceCase> {
    if steps
        .iter()
        .filter(|step| matches!(step.value.as_str(), "any graph" | "an empty graph"))
        .count()
        != 1
    {
        return Err(Error::internal(format!(
            "{name} is not tied to exactly one supported graph fixture"
        )));
    }
    if steps
        .iter()
        .filter(|step| step.value == "no side effects")
        .count()
        != 1
        || steps
            .iter()
            .any(|step| step.value.starts_with("the side effects should be"))
    {
        return Err(Error::internal(format!(
            "{name} no longer has the exact no-side-effects contract"
        )));
    }
    let operations = steps
        .iter()
        .filter(|step| step.value.starts_with("executing query:"))
        .collect::<Vec<_>>();
    let [operation] = operations.as_slice() else {
        return Err(Error::internal(format!(
            "{name} must contain exactly one primary query"
        )));
    };
    let query = operation
        .docstring
        .as_ref()
        .ok_or_else(|| Error::internal(format!("{name} query omitted its docstring")))?
        .trim()
        .to_owned();
    let setup_queries = steps
        .iter()
        .filter(|step| {
            matches!(
                step.value.as_str(),
                "having executed:" | "after having executed:"
            )
        })
        .map(|step| {
            step.docstring
                .as_ref()
                .map(|query| query.trim().to_owned())
                .ok_or_else(|| Error::internal(format!("{name} setup omitted its docstring")))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut parameter_literals = BTreeMap::new();
    for step in steps.iter().filter(|step| step.value == "parameters are:") {
        for row in &step.table {
            let [parameter, literal] = row.as_slice() else {
                return Err(Error::internal(format!(
                    "{name} has a malformed parameter row {row:?}"
                )));
            };
            if parameter_literals
                .insert(parameter.clone(), literal.clone())
                .is_some()
            {
                return Err(Error::internal(format!(
                    "{name} repeats parameter {parameter}"
                )));
            }
        }
    }
    let expectations = steps
        .iter()
        .filter(|step| step.value.starts_with("the result should be"))
        .collect::<Vec<_>>();
    let [expectation] = expectations.as_slice() else {
        let error_steps = steps
            .iter()
            .filter(|step| step.value.contains("should be raised"))
            .collect::<Vec<_>>();
        return Err(Error::internal(format!(
            "{name} must remain a result scenario, found error assertions {error_steps:?}"
        )));
    };
    let result_order = match expectation.value.as_str() {
        "the result should be, in any order:" => ResultOrder::Any,
        "the result should be, in order:" => ResultOrder::Exact,
        "the result should be (ignoring element order for lists):" => {
            ResultOrder::IgnoreListElementOrder
        }
        other => {
            return Err(Error::internal(format!(
                "{name} has unsupported result-order contract `{other}`"
            )));
        }
    };
    let (headers, rows) = expectation
        .table
        .split_first()
        .ok_or_else(|| Error::internal(format!("{name} omitted its expected result table")))?;
    if headers.is_empty() || rows.iter().any(|row| row.len() != headers.len()) {
        return Err(Error::internal(format!(
            "{name} has an empty or inconsistent expected result table"
        )));
    }
    Ok(SourceCase {
        setup_queries,
        parameter_literals,
        query,
        expected_headers: headers.clone(),
        expected_rows: rows.to_vec(),
        result_order,
    })
}

fn source_catalog(cases: &[ManifestCase]) -> Result<BTreeMap<(String, String), SourceCase>> {
    let feature_root = feature_root()?;
    let features = cases
        .iter()
        .map(|case| case.feature.clone())
        .collect::<BTreeSet<_>>();
    let mut catalog = BTreeMap::new();
    for feature in features {
        let path = feature_root.join(&feature);
        for scenario in parse_feature(&path)? {
            let mut matching = cases
                .iter()
                .filter(|case| {
                    case.feature == feature && source_base_name(&case.name) == scenario.name
                })
                .collect::<Vec<_>>();
            matching.sort_by_key(|case| case.report_id);
            if matching.is_empty() {
                continue;
            }
            let variants = if scenario.outline {
                let mut variants = Vec::new();
                for table in &scenario.examples {
                    let (headers, rows) = table.split_first().ok_or_else(|| {
                        Error::internal(format!("{} has an empty Examples table", scenario.name))
                    })?;
                    for row in rows {
                        if row.len() != headers.len() {
                            return Err(Error::internal(format!(
                                "{} has a malformed Examples row",
                                scenario.name
                            )));
                        }
                        variants.push(
                            scenario
                                .steps
                                .iter()
                                .map(|step| substitute_step(step, headers, row))
                                .collect::<Vec<_>>(),
                        );
                    }
                }
                variants
            } else {
                if !scenario.examples.is_empty() {
                    return Err(Error::internal(format!(
                        "non-outline {} unexpectedly has examples",
                        scenario.name
                    )));
                }
                vec![scenario.steps.clone()]
            };
            if variants.len() != matching.len() {
                return Err(Error::internal(format!(
                    "{} expands to {} source rows but {} manifest rows",
                    scenario.name,
                    variants.len(),
                    matching.len()
                )));
            }
            for (case, steps) in matching.into_iter().zip(variants) {
                let expected_name = if scenario.outline {
                    format!("{} [{}]", scenario.name, case.report_id + 1)
                } else {
                    scenario.name.clone()
                };
                if case.name != expected_name {
                    return Err(Error::internal(format!(
                        "report index {} source expansion should be `{expected_name}`, not `{}`",
                        case.report_id, case.name
                    )));
                }
                let source = source_case_from_steps(&case.name, &steps)?;
                if catalog
                    .insert((case.feature.clone(), case.name.clone()), source)
                    .is_some()
                {
                    return Err(Error::internal(format!(
                        "duplicate source identity {} / {}",
                        case.feature, case.name
                    )));
                }
            }
        }
    }
    if catalog.len() != cases.len() {
        return Err(Error::internal(format!(
            "pinned source resolved {} of {} fused-aggregation manifest cases",
            catalog.len(),
            cases.len()
        )));
    }
    Ok(catalog)
}

fn context<'a>(
    graph: &'a GraphStore,
    parameters: BTreeMap<String, ResultValue>,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
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
        parameters,
        bookmark: Bookmark {
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 1_024,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        resolved_query_at_time_nanos: None,
    }
}

fn context_with_max_result_rows<'a>(
    graph: &'a GraphStore,
    parameters: BTreeMap<String, ResultValue>,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
    max_result_rows: usize,
) -> ExecutionContext<'a> {
    let mut context = context(graph, parameters, backend, require_native_execution);
    context.max_result_rows = max_result_rows;
    context
}

fn fixture_graph(source: &SourceCase) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for setup in &source.setup_queries {
        let output =
            QueryEngine.execute(setup, &mut context(&graph, BTreeMap::new(), None, false))?;
        if !output.temporal_mutations.is_empty() {
            return Err(Error::internal(
                "fused aggregation fixture unexpectedly produced temporal mutations",
            ));
        }
        for mutation in output.graph_mutations {
            graph.apply(mutation)?;
        }
    }
    Ok(graph)
}

fn parameter_value(graph: &GraphStore, literal: &str) -> Result<ResultValue> {
    let output = QueryEngine.execute(
        &format!("RETURN {literal} AS value"),
        &mut context(graph, BTreeMap::new(), None, false),
    )?;
    if output.result.schema.len() != 1 || output.result.schema[0].0 != "value" {
        return Err(Error::internal(format!(
            "parameter literal `{literal}` produced an invalid schema"
        )));
    }
    let values = output
        .result
        .batches
        .iter()
        .flat_map(|batch| batch.columns[0].values.iter())
        .cloned()
        .collect::<Vec<_>>();
    let [value] = values.as_slice() else {
        return Err(Error::internal(format!(
            "parameter literal `{literal}` produced {} rows",
            values.len()
        )));
    };
    Ok(value.clone())
}

fn parameters(graph: &GraphStore, source: &SourceCase) -> Result<BTreeMap<String, ResultValue>> {
    source
        .parameter_literals
        .iter()
        .map(|(name, literal)| Ok((name.clone(), parameter_value(graph, literal)?)))
        .collect()
}

fn tck_string(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn tck_scalar(value: &ScalarValue) -> Result<String> {
    match value {
        ScalarValue::Null => Ok("null".to_owned()),
        ScalarValue::Boolean(value) => Ok(value.to_string()),
        ScalarValue::Integer(value) => Ok(value.to_string()),
        ScalarValue::Float(value) => Ok(value.0.to_string()),
        ScalarValue::String(value) => Ok(tck_string(value)),
        other => Err(Error::internal(format!(
            "fused aggregation oracle cannot canonicalize scalar {other:?}"
        ))),
    }
}

fn tck_properties(properties: &BTreeMap<String, ScalarValue>) -> Result<String> {
    if properties.is_empty() {
        return Ok(String::new());
    }
    let entries = properties
        .iter()
        .map(|(name, value)| Ok(format!("{name}: {}", tck_scalar(value)?)))
        .collect::<Result<Vec<_>>>()?;
    Ok(format!("{{{}}}", entries.join(", ")))
}

fn tck_value(value: &ResultValue) -> Result<String> {
    match value {
        ResultValue::Scalar(value) => tck_scalar(value),
        ResultValue::Node(node) => {
            let labels = node.labels.join(":");
            let properties = tck_properties(&node.properties)?;
            Ok(match (labels.is_empty(), properties.is_empty()) {
                (true, true) => "()".to_owned(),
                (true, false) => format!("({properties})"),
                (false, true) => format!("(:{labels})"),
                (false, false) => format!("(:{labels} {properties})"),
            })
        }
        ResultValue::Relationship(relationship) => {
            let properties = tck_properties(&relationship.properties)?;
            Ok(if properties.is_empty() {
                format!("[:{}]", relationship.relationship_type)
            } else {
                format!("[:{} {properties}]", relationship.relationship_type)
            })
        }
        ResultValue::List(values) => Ok(format!(
            "[{}]",
            values
                .iter()
                .map(tck_value)
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        )),
        ResultValue::Map(values) => Ok(format!(
            "{{{}}}",
            values
                .iter()
                .map(|(name, value)| Ok(format!("{name}: {}", tck_value(value)?)))
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        )),
        other => Err(Error::internal(format!(
            "fused aggregation oracle cannot canonicalize result {other:?}"
        ))),
    }
}

fn split_top_level(value: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut round = 0_i32;
    let mut square = 0_i32;
    let mut curly = 0_i32;
    let mut quoted = false;
    let mut escaped = false;
    for (index, character) in value.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '\'' {
                quoted = false;
            }
            continue;
        }
        match character {
            '\'' => quoted = true,
            '(' => round += 1,
            ')' => round -= 1,
            '[' => square += 1,
            ']' => square -= 1,
            '{' => curly += 1,
            '}' => curly -= 1,
            ',' if round == 0 && square == 0 && curly == 0 => {
                parts.push(value[start..index].trim());
                start = index + character.len_utf8();
            }
            _ => {}
        }
    }
    if start < value.len() {
        parts.push(value[start..].trim());
    } else if value.trim().is_empty() {
        return Vec::new();
    }
    parts
}

fn normalize_list_elements(value: &str) -> String {
    let value = value.trim();
    if value.starts_with('[') && value.ends_with(']') && !value.starts_with("[:") {
        let mut children = split_top_level(&value[1..value.len() - 1])
            .into_iter()
            .map(normalize_list_elements)
            .collect::<Vec<_>>();
        children.sort();
        return format!("[{}]", children.join(", "));
    }
    let mut normalized = String::with_capacity(value.len());
    let mut index = 0;
    while index < value.len() {
        let suffix = &value[index..];
        if suffix.starts_with('[') && !suffix.starts_with("[:") {
            let mut depth = 0_i32;
            let mut quoted = false;
            let mut escaped = false;
            let mut end = None;
            for (offset, character) in suffix.char_indices() {
                if quoted {
                    if escaped {
                        escaped = false;
                    } else if character == '\\' {
                        escaped = true;
                    } else if character == '\'' {
                        quoted = false;
                    }
                    continue;
                }
                match character {
                    '\'' => quoted = true,
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(offset + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if let Some(end) = end {
                normalized.push_str(&normalize_list_elements(&suffix[..end]));
                index += end;
                continue;
            }
        }
        let character = suffix.chars().next().expect("non-empty suffix");
        normalized.push(character);
        index += character.len_utf8();
    }
    normalized
}

fn assert_exact_output(
    case: &ManifestCase,
    source: &SourceCase,
    output: &ExecutionOutput,
) -> Result<()> {
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
    {
        return Err(Error::internal(format!(
            "report {} / {} violated its exact no-side-effects/non-truncated contract",
            case.report_id, case.name
        )));
    }
    let headers = output
        .result
        .schema
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    if headers != source.expected_headers {
        return Err(Error::internal(format!(
            "report {} / {} expected columns {:?}, got {headers:?}",
            case.report_id, case.name, source.expected_headers
        )));
    }
    let mut actual_rows = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != source.expected_headers.len() {
            return Err(Error::internal(format!(
                "report {} / {} returned an invalid batch",
                case.report_id, case.name
            )));
        }
        for row in 0..batch.row_count {
            actual_rows.push(
                batch
                    .columns
                    .iter()
                    .map(|column| tck_value(&column.values[row]))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
    }
    let mut expected_rows = source.expected_rows.clone();
    match source.result_order {
        ResultOrder::Any => {
            expected_rows.sort();
            actual_rows.sort();
        }
        ResultOrder::Exact => {}
        ResultOrder::IgnoreListElementOrder => {
            for row in &mut expected_rows {
                for value in row {
                    *value = normalize_list_elements(value);
                }
            }
            for row in &mut actual_rows {
                for value in row {
                    *value = normalize_list_elements(value);
                }
            }
            expected_rows.sort();
            actual_rows.sort();
        }
    }
    if actual_rows != expected_rows {
        return Err(Error::internal(format!(
            "report {} / {} expected rows {expected_rows:?}, got {actual_rows:?}",
            case.report_id, case.name
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CallSnapshot {
    segmented: usize,
    sealed_program: usize,
    range_source: usize,
    pre_aggregate_limit: usize,
    row: usize,
    nullable: usize,
    node_pipeline: usize,
    precomputed_source: usize,
    other: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObservedSegmentedStage {
    Project,
    PropertyKeys,
    PropertyValue,
    Aggregate,
    Order,
    Skip(u32),
    Limit(u32),
    Filter,
    Unwind,
}

#[derive(Default)]
struct NativeCalls {
    segmented: AtomicUsize,
    sealed_program: AtomicUsize,
    range_source: AtomicUsize,
    pre_aggregate_limit: AtomicUsize,
    row: AtomicUsize,
    nullable: AtomicUsize,
    node_pipeline: AtomicUsize,
    precomputed_source: AtomicUsize,
    other: AtomicUsize,
    unit_source: AtomicUsize,
    graph_relation_source: AtomicUsize,
    post_aggregate_order: AtomicUsize,
    graph_relation_stage_sequences: Mutex<Vec<Vec<ObservedSegmentedStage>>>,
    graph_relation_binding_counts: Mutex<Vec<usize>>,
    graph_relation_source_capacities: Mutex<Vec<usize>>,
}

impl NativeCalls {
    fn snapshot(&self) -> CallSnapshot {
        CallSnapshot {
            segmented: self.segmented.load(Ordering::SeqCst),
            sealed_program: self.sealed_program.load(Ordering::SeqCst),
            range_source: self.range_source.load(Ordering::SeqCst),
            pre_aggregate_limit: self.pre_aggregate_limit.load(Ordering::SeqCst),
            row: self.row.load(Ordering::SeqCst),
            nullable: self.nullable.load(Ordering::SeqCst),
            node_pipeline: self.node_pipeline.load(Ordering::SeqCst),
            precomputed_source: self.precomputed_source.load(Ordering::SeqCst),
            other: self.other.load(Ordering::SeqCst),
        }
    }

    fn graph_relation_stage_sequences(&self) -> Result<Vec<Vec<ObservedSegmentedStage>>> {
        self.graph_relation_stage_sequences
            .lock()
            .map(|sequences| sequences.clone())
            .map_err(|_| Error::internal("graph-relation stage observer lock was poisoned"))
    }

    fn graph_relation_binding_counts(&self) -> Result<Vec<usize>> {
        self.graph_relation_binding_counts
            .lock()
            .map(|counts| counts.clone())
            .map_err(|_| Error::internal("graph-relation binding observer lock was poisoned"))
    }

    fn graph_relation_source_capacities(&self) -> Result<Vec<usize>> {
        self.graph_relation_source_capacities
            .lock()
            .map(|capacities| capacities.clone())
            .map_err(|_| Error::internal("graph-relation capacity observer lock was poisoned"))
    }
}

struct ObservedBackend {
    inner: Box<dyn ExecutionBackend>,
    reported_kind: BackendKind,
    calls: Arc<NativeCalls>,
}

impl ObservedBackend {
    fn strict_cpu() -> Self {
        Self {
            inner: Box::new(CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES)),
            reported_kind: BackendKind::Cpu,
            calls: Arc::new(NativeCalls::default()),
        }
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal() -> Result<Self> {
        let inner = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "fused aggregation acceptance did not construct a real Metal backend",
            ));
        }
        Ok(Self {
            inner: Box::new(inner),
            reported_kind: BackendKind::Metal,
            calls: Arc::new(NativeCalls::default()),
        })
    }

    fn calls(&self) -> Arc<NativeCalls> {
        Arc::clone(&self.calls)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.calls.other.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict fused aggregation gate rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for ObservedBackend {
    fn kind(&self) -> BackendKind {
        self.reported_kind
    }

    fn available_query_scratch_bytes(&self) -> usize {
        self.inner.available_query_scratch_bytes()
    }

    fn reserve_query_scratch(&self, bytes: usize) -> Result<ScratchReservation> {
        self.inner.reserve_query_scratch(bytes)
    }

    fn resident_project_bytes(&self, project: ProjectId) -> Option<usize> {
        self.inner.resident_project_bytes(project)
    }

    fn pin_project(&self, project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            reported_kind: self.reported_kind,
            calls: Arc::clone(&self.calls),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        self.inner.admit_project(image)
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        self.inner.advance_bookmark(bookmark);
    }

    fn resident_revision(&self) -> Option<u64> {
        self.inner.resident_revision()
    }

    fn resident_graph_revision(&self, project: ProjectId) -> Option<u64> {
        self.inner.resident_graph_revision(project)
    }

    fn resident_bookmark(&self, project: ProjectId) -> Option<Bookmark> {
        self.inner.resident_bookmark(project)
    }

    fn scan_nodes(
        &self,
        _project: ProjectId,
        _label: Option<LabelId>,
        _layers: LayerMask,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.calls.node_pipeline.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "strict fused aggregation gate rejected node-pipeline fallback",
        ))
    }

    fn execute_node_group_pipeline(
        &self,
        request: &ResidentNodeGroupPipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.calls.segmented.fetch_add(1, Ordering::SeqCst);
        self.calls.sealed_program.fetch_add(1, Ordering::SeqCst);
        self.inner
            .execute_node_group_pipeline(request, cancellation)
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.calls.row.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "strict fused aggregation gate rejected row-program fallback",
        ))
    }

    fn supports_nullable_relation_predicates(&self) -> bool {
        false
    }

    fn execute_nullable_relation(
        &self,
        _request: &ResidentNullableRelationRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        self.calls.nullable.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "strict fused aggregation gate rejected nullable-relation fallback",
        ))
    }

    fn supports_native_segmented_aggregation(&self) -> bool {
        self.inner.supports_native_segmented_aggregation()
    }

    fn execute_segmented_aggregation(
        &self,
        request: &ResidentSegmentedAggregationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSegmentedAggregationResult> {
        self.calls.segmented.fetch_add(1, Ordering::SeqCst);
        let Some(program) = request.program.as_ref() else {
            self.calls.precomputed_source.fetch_add(1, Ordering::SeqCst);
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "strict fused aggregation gate rejected legacy host-relation aggregation",
            ));
        };
        if request.input.row_count != 0
            || request.input.column_count != 0
            || !request.input.cells.is_empty()
            || !request.input.arena.is_empty()
        {
            self.calls.precomputed_source.fetch_add(1, Ordering::SeqCst);
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "strict fused aggregation gate rejected a sealed program carrying host rows",
            ));
        }
        program.validate()?;
        self.calls.sealed_program.fetch_add(1, Ordering::SeqCst);
        if matches!(
            program.source,
            ResidentSegmentedAggregationSource::Range { .. }
        ) {
            self.calls.range_source.fetch_add(1, Ordering::SeqCst);
        }
        if matches!(
            program.source,
            ResidentSegmentedAggregationSource::Unit { .. }
        ) {
            self.calls.unit_source.fetch_add(1, Ordering::SeqCst);
        }
        if let ResidentSegmentedAggregationSource::GraphRelation {
            request, bindings, ..
        } = &program.source
        {
            self.calls
                .graph_relation_source
                .fetch_add(1, Ordering::SeqCst);
            self.calls
                .graph_relation_binding_counts
                .lock()
                .map_err(|_| Error::internal("graph-relation binding observer lock was poisoned"))?
                .push(bindings.len());
            self.calls
                .graph_relation_source_capacities
                .lock()
                .map_err(|_| Error::internal("graph-relation capacity observer lock was poisoned"))?
                .push(request.capacities.max_output_rows as usize);
            let stages = program
                .stages
                .iter()
                .map(|stage| match &stage.operation {
                    ResidentSegmentedAggregationOperation::Project { .. } => {
                        ObservedSegmentedStage::Project
                    }
                    ResidentSegmentedAggregationOperation::PropertyKeys { .. } => {
                        ObservedSegmentedStage::PropertyKeys
                    }
                    ResidentSegmentedAggregationOperation::PropertyValue { .. } => {
                        ObservedSegmentedStage::PropertyValue
                    }
                    ResidentSegmentedAggregationOperation::Aggregate { .. } => {
                        ObservedSegmentedStage::Aggregate
                    }
                    ResidentSegmentedAggregationOperation::Order { .. } => {
                        ObservedSegmentedStage::Order
                    }
                    ResidentSegmentedAggregationOperation::Skip { rows } => {
                        ObservedSegmentedStage::Skip(*rows)
                    }
                    ResidentSegmentedAggregationOperation::Limit { rows } => {
                        ObservedSegmentedStage::Limit(*rows)
                    }
                    ResidentSegmentedAggregationOperation::Filter { .. } => {
                        ObservedSegmentedStage::Filter
                    }
                    ResidentSegmentedAggregationOperation::Unwind { .. } => {
                        ObservedSegmentedStage::Unwind
                    }
                })
                .collect::<Vec<_>>();
            self.calls
                .graph_relation_stage_sequences
                .lock()
                .map_err(|_| Error::internal("graph-relation stage observer lock was poisoned"))?
                .push(stages);
        }
        let aggregate_index = program
            .stages
            .iter()
            .position(|stage| {
                matches!(
                    stage.operation,
                    ResidentSegmentedAggregationOperation::Aggregate { .. }
                )
            })
            .ok_or_else(|| Error::internal("validated segmented program lost its aggregate"))?;
        if program.stages[..aggregate_index].iter().any(|stage| {
            matches!(
                stage.operation,
                ResidentSegmentedAggregationOperation::Limit { .. }
            )
        }) {
            self.calls
                .pre_aggregate_limit
                .fetch_add(1, Ordering::SeqCst);
        }
        if program.stages[aggregate_index + 1..].iter().any(|stage| {
            matches!(
                stage.operation,
                ResidentSegmentedAggregationOperation::Order { .. }
            )
        }) {
            self.calls
                .post_aggregate_order
                .fetch_add(1, Ordering::SeqCst);
        }
        self.inner
            .execute_segmented_aggregation(request, cancellation)
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject("exact_l2")
    }
}

fn call_delta(before: CallSnapshot, after: CallSnapshot) -> CallSnapshot {
    CallSnapshot {
        segmented: after.segmented.saturating_sub(before.segmented),
        sealed_program: after.sealed_program.saturating_sub(before.sealed_program),
        range_source: after.range_source.saturating_sub(before.range_source),
        pre_aggregate_limit: after
            .pre_aggregate_limit
            .saturating_sub(before.pre_aggregate_limit),
        row: after.row.saturating_sub(before.row),
        nullable: after.nullable.saturating_sub(before.nullable),
        node_pipeline: after.node_pipeline.saturating_sub(before.node_pipeline),
        precomputed_source: after
            .precomputed_source
            .saturating_sub(before.precomputed_source),
        other: after.other.saturating_sub(before.other),
    }
}

fn execute_strict_case(
    case: &ManifestCase,
    source: &SourceCase,
    backend: &mut ObservedBackend,
) -> Result<()> {
    let graph = fixture_graph(source)?;
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();
    let before = calls.snapshot();
    let output = QueryEngine.execute(
        &source.query,
        &mut context(&graph, parameters(&graph, source)?, Some(&*backend), true),
    );
    let delta = call_delta(before, calls.snapshot());
    let output = output.map_err(|error| {
        Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "{MISSING_FUSED_CONTRACT}; report {} / {} failed with {:?}: {}; native calls={delta:?}",
                case.report_id, case.name, error.code, error.message
            ),
        )
    })?;
    assert_exact_output(case, source, &output)?;
    let expected = if case.report_id == 1266 {
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            range_source: 1,
            pre_aggregate_limit: 1,
            ..CallSnapshot::default()
        }
    } else {
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            ..CallSnapshot::default()
        }
    };
    if delta != expected {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "{MISSING_FUSED_CONTRACT}; report {} / {} crossed {delta:?} instead of exactly one segmented command and zero fallbacks",
                case.report_id, case.name
            ),
        ));
    }
    Ok(())
}

fn run_strict_suite(backend: &mut ObservedBackend) -> Result<()> {
    let cases = manifest()?;
    let sources = source_catalog(&cases)?;
    let mut failures = Vec::new();
    for case in &cases {
        let source = sources
            .get(&(case.feature.clone(), case.name.clone()))
            .ok_or_else(|| Error::internal(format!("source omitted {}", case.name)))?;
        if let Err(error) = execute_strict_case(case, source, backend) {
            failures.push(format!(
                "{} / {}: {}",
                case.report_id, case.name, error.message
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "strict fused aggregation gate failed {} of {} exact scenarios; failures: {}",
                failures.len(),
                cases.len(),
                failures.join(" | ")
            ),
        ))
    }
}

#[test]
fn fused_aggregation_manifest_has_exact_29_case_shape() -> Result<()> {
    let cases = manifest()?;
    assert_eq!(cases.len(), OFFICIAL_CASE_COUNT);
    assert_eq!(
        cases
            .iter()
            .map(|case| case.report_id)
            .collect::<BTreeSet<_>>()
            .len(),
        OFFICIAL_CASE_COUNT
    );
    assert_eq!(
        cases.iter().map(|case| case.report_id).collect::<Vec<_>>(),
        vec![
            705, 723, 734, 736, 737, 741, 742, 744, 749, 750, 751, 771, 774, 775, 779, 783, 784,
            787, 788, 789, 920, 1247, 1251, 1265, 1266, 1267, 1268, 1282, 1283,
        ]
    );
    Ok(())
}

#[test]
#[ignore = "external integrity guard: requires the exact certified report and pinned TCK checkout"]
fn manifest_resolves_uniquely_against_exact_report_and_source_bytes() -> Result<()> {
    let feature_root = feature_root()?;
    if feature_root != Path::new(EXPECTED_FEATURE_ROOT) {
        return Err(Error::internal(format!(
            "OPENCYPHER_TCK_DIR changed: expected {EXPECTED_FEATURE_ROOT}, got {}",
            feature_root.display()
        )));
    }
    let cases = manifest()?;
    let report = certified_report()?;
    if report.total != 3_897
        || report.cpu_passed != 3_897
        || report.metal_passed != 3_277
        || report.matched != 3_277
        || report.fully_conformant != 3_277
        || report.scenarios.len() != 3_897
    {
        return Err(Error::internal("certified report totals changed"));
    }
    let selected_features = cases
        .iter()
        .map(|case| case.feature.as_str())
        .collect::<BTreeSet<_>>();
    let digested_features = FEATURE_DIGESTS
        .iter()
        .map(|(feature, _)| *feature)
        .collect::<BTreeSet<_>>();
    if selected_features != digested_features {
        return Err(Error::internal(format!(
            "source digest manifest mismatch: selected={selected_features:?}, digested={digested_features:?}"
        )));
    }
    for (feature, expected_digest) in FEATURE_DIGESTS {
        let bytes = fs::read(feature_root.join(feature)).map_err(|error| {
            Error::internal(format!("cannot read pinned feature {feature}: {error}"))
        })?;
        let actual = sha256(&bytes);
        if actual != expected_digest {
            return Err(Error::internal(format!(
                "pinned source digest changed for {feature}: expected {expected_digest}, got {actual}"
            )));
        }
    }
    let sources = source_catalog(&cases)?;
    if sources.len() != OFFICIAL_CASE_COUNT {
        return Err(Error::internal(
            "source catalog did not resolve exactly 29 cases",
        ));
    }
    for case in &cases {
        let matches = report
            .scenarios
            .iter()
            .enumerate()
            .filter(|(_, scenario)| {
                scenario.name == case.name && Path::new(&scenario.path).ends_with(&case.feature)
            })
            .collect::<Vec<_>>();
        let [(actual_index, scenario)] = matches.as_slice() else {
            return Err(Error::internal(format!(
                "report identity {} / {} resolved to {:?}",
                case.feature,
                case.name,
                matches.iter().map(|(index, _)| *index).collect::<Vec<_>>()
            )));
        };
        if *actual_index != case.report_id
            || scenario.operation_count != 1
            || !scenario.cpu_passed
            || scenario.metal_passed
            || scenario.fully_conformant
            || !scenario.shared_failures.is_empty()
            || !scenario.cpu_failures.is_empty()
            || scenario.metal_failures.len() != 1
            || scenario.divergences.len() != 1
            || !scenario.metal_failures[0].contains(
                "GpuAdmissionFailure: active GPU execution class has no complete resident implementation for this query plan",
            )
        {
            return Err(Error::internal(format!(
                "certified status changed for report {} / {}",
                case.report_id, case.name
            )));
        }
    }
    Ok(())
}

/// Semantic oracle only: this does not certify any native execution boundary.
#[test]
#[ignore = "external CPU oracle: requires the pinned openCypher TCK checkout"]
fn generic_cpu_oracle_replays_exact_29_official_scenarios() -> Result<()> {
    let cases = manifest()?;
    let sources = source_catalog(&cases)?;
    for case in &cases {
        let source = sources
            .get(&(case.feature.clone(), case.name.clone()))
            .ok_or_else(|| Error::internal(format!("source omitted {}", case.name)))?;
        let graph = fixture_graph(source)?;
        let output = QueryEngine
            .execute(
                &source.query,
                &mut context(&graph, parameters(&graph, source)?, None, false),
            )
            .map_err(|error| {
                Error::internal(format!(
                    "generic CPU oracle failed report {} / {}: {:?}: {}",
                    case.report_id, case.name, error.code, error.message
                ))
            })?;
        assert_exact_output(case, source, &output)?;
    }
    Ok(())
}

#[test]
fn report_784_metal_source_contract_is_device_resident_and_fail_closed() {
    let host = include_str!("../crates/gpu/src/accelerator.rs");
    let executor = host
        .split_once("fn execute_metal_segmented_unit_program(")
        .expect("segmented Unit executor disappeared")
        .1
        .split_once("fn execute_metal_segmented_graph_relation_program(")
        .expect("segmented Unit executor boundary disappeared")
        .0;
    assert!(executor.contains("&prelude,"));
    assert!(!executor.contains("prelude.to_vec1"));
    assert!(!executor.contains("decode_metal_quantifier_program"));
    assert!(executor.contains("dependency_count != 0"));

    let metal = include_str!("../kernels/metal/operators.metal");
    for proof in [
        "nullable[word] != program[expected_header_base + word]",
        "nullable[25] != actual_roots",
        "nullable[record + 2ul] != program[descriptor]",
        "nullable[record + 5ul] != program[descriptor + 3ul]",
        "expected_rows != actual_source_rows",
    ] {
        assert!(metal.contains(proof), "missing report-784 proof `{proof}`");
    }
}

#[test]
#[ignore = "external focused gate: requires the pinned openCypher TCK checkout"]
fn strict_cpu_reference_executes_report_1247_filter_in_one_command() -> Result<()> {
    let cases = manifest()?;
    let case = cases
        .iter()
        .find(|case| case.report_id == 1247)
        .ok_or_else(|| Error::internal("exact aggregation manifest omitted report 1247"))?;
    let sources = source_catalog(&cases)?;
    let source = sources
        .get(&(case.feature.clone(), case.name.clone()))
        .ok_or_else(|| Error::internal("source catalog omitted report 1247"))?;
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let sequence_count = calls.graph_relation_stage_sequences()?.len();
    execute_strict_case(case, source, &mut backend)?;
    assert_eq!(
        &calls.graph_relation_stage_sequences()?[sequence_count..],
        &[vec![
            ObservedSegmentedStage::Aggregate,
            ObservedSegmentedStage::Project,
            ObservedSegmentedStage::Filter,
            ObservedSegmentedStage::Project,
            ObservedSegmentedStage::Project,
        ]]
    );
    Ok(())
}

#[test]
#[ignore = "external focused gate: requires the pinned openCypher TCK checkout"]
fn strict_cpu_reference_executes_report_784_static_index_unwinds_in_one_command() -> Result<()> {
    let cases = manifest()?;
    let case = cases
        .iter()
        .find(|case| case.report_id == 784)
        .ok_or_else(|| Error::internal("exact aggregation manifest omitted report 784"))?;
    let sources = source_catalog(&cases)?;
    let source = sources
        .get(&(case.feature.clone(), case.name.clone()))
        .ok_or_else(|| Error::internal("source catalog omitted report 784"))?;
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let unit_sources_before = calls.unit_source.load(Ordering::SeqCst);
    execute_strict_case(case, source, &mut backend)?;
    assert_eq!(
        calls
            .unit_source
            .load(Ordering::SeqCst)
            .saturating_sub(unit_sources_before),
        1,
        "report 784 must execute from one backend-owned Unit source"
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "real-Metal report-784 gate: root serializes all device runtime tests"]
fn real_metal_executes_report_784_static_index_unwinds_in_one_command() -> Result<()> {
    let cases = manifest()?;
    let case = cases
        .iter()
        .find(|case| case.report_id == 784)
        .ok_or_else(|| Error::internal("exact aggregation manifest omitted report 784"))?;
    let sources = source_catalog(&cases)?;
    let source = sources
        .get(&(case.feature.clone(), case.name.clone()))
        .ok_or_else(|| Error::internal("source catalog omitted report 784"))?;
    let mut backend = ObservedBackend::real_metal()?;
    let calls = backend.calls();
    let before = calls.snapshot();
    let unit_sources_before = calls.unit_source.load(Ordering::SeqCst);
    execute_strict_case(case, source, &mut backend)?;
    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            ..CallSnapshot::default()
        },
        "report 784 must stay in one sealed Metal Unit-source command"
    );
    assert_eq!(
        calls
            .unit_source
            .load(Ordering::SeqCst)
            .saturating_sub(unit_sources_before),
        1,
        "report 784 must own exactly one Unit source"
    );
    Ok(())
}

#[test]
#[ignore = "external focused gate: requires the pinned openCypher TCK checkout"]
fn strict_cpu_reference_executes_all_scalar_and_document_tails() -> Result<()> {
    let cases = scalar_aggregate_tail_manifest()?;
    let sources = source_catalog(&cases)?;
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let graph_sources_before = calls.graph_relation_source.load(Ordering::SeqCst);
    let mut failures = Vec::new();
    for case in &cases {
        let source = sources
            .get(&(case.feature.clone(), case.name.clone()))
            .ok_or_else(|| Error::internal(format!("source omitted {}", case.name)))?;
        if let Err(error) = execute_strict_case(case, source, &mut backend) {
            failures.push(format!(
                "{} / {}: {:?}: {}",
                case.report_id, case.name, error.code, error.message
            ));
        }
    }
    if !failures.is_empty() {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "strict scalar aggregate tails had {} failure(s):\n{}",
                failures.len(),
                failures.join("\n")
            ),
        ));
    }
    assert_eq!(
        calls
            .graph_relation_source
            .load(Ordering::SeqCst)
            .saturating_sub(graph_sources_before),
        cases.len(),
        "every typed scalar or document tail must own one GraphRelation source"
    );
    Ok(())
}

#[test]
#[ignore = "external focused gate: requires the pinned openCypher TCK checkout"]
fn strict_cpu_reference_executes_report_1266_range_limit_sum_in_one_command() -> Result<()> {
    let cases = manifest()?;
    let case = cases
        .iter()
        .find(|case| case.report_id == 1266)
        .ok_or_else(|| Error::internal("exact aggregation manifest omitted report 1266"))?;
    let sources = source_catalog(&cases)?;
    let source = sources
        .get(&(case.feature.clone(), case.name.clone()))
        .ok_or_else(|| Error::internal("source catalog omitted report 1266"))?;
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let before = calls.snapshot();
    execute_strict_case(case, source, &mut backend)?;
    let delta = call_delta(before, calls.snapshot());
    assert_eq!(
        delta,
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            range_source: 1,
            pre_aggregate_limit: 1,
            ..CallSnapshot::default()
        }
    );
    eprintln!("report 1266 native calls: {delta:?}");
    Ok(())
}

#[test]
fn strict_cpu_executes_grouped_graph_aggregation_and_order_in_one_command() -> Result<()> {
    let cases = [
        (
            ManifestCase {
                report_id: 771,
                feature: "clauses/return-orderby/ReturnOrderBy2.feature".to_owned(),
                name: "[3] Sort on aggregated function".to_owned(),
            },
            SourceCase {
                // Deliberately oppose encounter order to aggregate-key order. This makes an
                // omitted post-aggregate Order stage observable, unlike the official fixture.
                setup_queries: vec![
                    "CREATE (:Employee {division: 'C', age: 55})".to_owned(),
                    "CREATE (:Employee {division: 'B', age: 33})".to_owned(),
                    "CREATE (:Employee {division: 'A', age: 22})".to_owned(),
                    "CREATE (:Employee {division: 'B', age: 44})".to_owned(),
                ],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) RETURN n.division, max(n.age) ORDER BY max(n.age)".to_owned(),
                expected_headers: vec!["n.division".to_owned(), "max(n.age)".to_owned()],
                expected_rows: vec![
                    vec!["'A'".to_owned(), "22".to_owned()],
                    vec!["'B'".to_owned(), "44".to_owned()],
                    vec!["'C'".to_owned(), "55".to_owned()],
                ],
                result_order: ResultOrder::Exact,
            },
        ),
        (
            ManifestCase {
                report_id: 775,
                feature: "clauses/return-orderby/ReturnOrderBy2.feature".to_owned(),
                name: "[7] Ordering with aggregation".to_owned(),
            },
            SourceCase {
                setup_queries: vec![
                    "CREATE (:Person {name: 'zeta'})".to_owned(),
                    "CREATE (:Person {name: 'alpha'})".to_owned(),
                    "CREATE (:Person {name: 'zeta'})".to_owned(),
                ],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) RETURN n.name, count(*) AS foo ORDER BY n.name".to_owned(),
                expected_headers: vec!["n.name".to_owned(), "foo".to_owned()],
                expected_rows: vec![
                    vec!["'alpha'".to_owned(), "1".to_owned()],
                    vec!["'zeta'".to_owned(), "2".to_owned()],
                ],
                result_order: ResultOrder::Exact,
            },
        ),
    ];
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let graph_sources_before = calls.graph_relation_source.load(Ordering::SeqCst);
    let orders_before = calls.post_aggregate_order.load(Ordering::SeqCst);
    for (case, source) in &cases {
        execute_strict_case(case, source, &mut backend)?;
    }
    assert_eq!(
        calls
            .graph_relation_source
            .load(Ordering::SeqCst)
            .saturating_sub(graph_sources_before),
        cases.len()
    );
    assert_eq!(
        calls
            .post_aggregate_order
            .load(Ordering::SeqCst)
            .saturating_sub(orders_before),
        cases.len()
    );
    Ok(())
}

#[test]
fn strict_cpu_executes_exact_with_where6_01_filter_in_one_command() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (a {name: 'A'}), (b {name: 'B'}) \
             CREATE (a)-[:REL]->(), (a)-[:REL]->(), (a)-[:REL]->(), (b)-[:REL]->()"
                .to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: "MATCH (a)-->() \
                WITH a, count(*) AS relCount \
                WHERE relCount > 1 \
                RETURN a"
            .to_owned(),
        expected_headers: vec!["a".to_owned()],
        expected_rows: vec![vec!["({name: 'A'})".to_owned()]],
        result_order: ResultOrder::Any,
    };
    let case = ManifestCase {
        report_id: 1247,
        feature: "clauses/with-where/WithWhere6.feature".to_owned(),
        name: "[1] Filter a single aggregate".to_owned(),
    };
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let sequence_count = calls.graph_relation_stage_sequences()?.len();
    execute_strict_case(&case, &source, &mut backend)?;
    assert_eq!(
        &calls.graph_relation_stage_sequences()?[sequence_count..],
        &[vec![
            ObservedSegmentedStage::Aggregate,
            ObservedSegmentedStage::Project,
            ObservedSegmentedStage::Filter,
            ObservedSegmentedStage::Project,
            ObservedSegmentedStage::Project,
        ]]
    );
    Ok(())
}

#[test]
fn strict_cpu_executes_global_string_minimum_and_maximum_in_one_command() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (:N {name: 'zeta'}), (:N {name: 'alpha'}), (:N {name: 'middle'})".to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: "MATCH (n:N) RETURN min(n.name) AS first, max(n.name) AS last".to_owned(),
        expected_headers: vec!["first".to_owned(), "last".to_owned()],
        expected_rows: vec![vec!["'alpha'".to_owned(), "'zeta'".to_owned()]],
        result_order: ResultOrder::Exact,
    };
    execute_strict_case(
        &ManifestCase {
            report_id: usize::MAX,
            feature: "native/fused-string-min-max".to_owned(),
            name: "global string minimum and maximum".to_owned(),
        },
        &source,
        &mut ObservedBackend::strict_cpu(),
    )
}

#[test]
fn strict_cpu_carries_string_reduction_widths_through_alias_renames() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (:N {name: 'zeta'}), (:N {name: 'alpha'}), (:N {name: 'middle'})".to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: "MATCH (n:N) \
                WITH min(n.name) AS smallest, max(n.name) AS largest \
                RETURN smallest AS first, smallest AS first_copy, largest AS last"
            .to_owned(),
        expected_headers: vec![
            "first".to_owned(),
            "first_copy".to_owned(),
            "last".to_owned(),
        ],
        expected_rows: vec![vec![
            "'alpha'".to_owned(),
            "'alpha'".to_owned(),
            "'zeta'".to_owned(),
        ]],
        result_order: ResultOrder::Exact,
    };
    execute_strict_case(
        &ManifestCase {
            report_id: usize::MAX,
            feature: "native/fused-string-min-max".to_owned(),
            name: "string reduction aliases retain independent output capacity".to_owned(),
        },
        &source,
        &mut ObservedBackend::strict_cpu(),
    )
}

#[test]
fn strict_cpu_executes_grouped_string_keys_and_string_minimum_maximum() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (:N {team: 'red', name: 'zeta'}), \
                    (:N {team: 'red', name: 'alpha'}), \
                    (:N {team: 'blue', name: 'omega'}), \
                    (:N {team: 'blue', name: 'beta'})"
                .to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: "MATCH (n:N) \
                RETURN n.team AS team, min(n.name) AS first, max(n.name) AS last \
                ORDER BY team"
            .to_owned(),
        expected_headers: vec!["team".to_owned(), "first".to_owned(), "last".to_owned()],
        expected_rows: vec![
            vec![
                "'blue'".to_owned(),
                "'beta'".to_owned(),
                "'omega'".to_owned(),
            ],
            vec![
                "'red'".to_owned(),
                "'alpha'".to_owned(),
                "'zeta'".to_owned(),
            ],
        ],
        result_order: ResultOrder::Exact,
    };
    execute_strict_case(
        &ManifestCase {
            report_id: usize::MAX,
            feature: "native/fused-string-min-max".to_owned(),
            name: "grouped string key with string minimum and maximum".to_owned(),
        },
        &source,
        &mut ObservedBackend::strict_cpu(),
    )
}

#[test]
fn strict_cpu_returns_null_string_minimum_and_maximum_for_empty_global_input() -> Result<()> {
    let source = SourceCase {
        setup_queries: Vec::new(),
        parameter_literals: BTreeMap::new(),
        query: "MATCH (n:N) RETURN min(n.name) AS first, max(n.name) AS last".to_owned(),
        expected_headers: vec!["first".to_owned(), "last".to_owned()],
        expected_rows: vec![vec!["null".to_owned(), "null".to_owned()]],
        result_order: ResultOrder::Exact,
    };
    execute_strict_case(
        &ManifestCase {
            report_id: usize::MAX,
            feature: "native/fused-string-min-max".to_owned(),
            name: "empty global string minimum and maximum".to_owned(),
        },
        &source,
        &mut ObservedBackend::strict_cpu(),
    )
}

#[test]
fn graph_fused_string_reductions_reject_unbounded_or_unsupported_shapes() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (:N {name: 'alpha', suffix: '!', score: 1}), \
                    (:N {name: 'beta', suffix: '?', score: 2})"
                .to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: String::new(),
        expected_headers: Vec::new(),
        expected_rows: Vec::new(),
        result_order: ResultOrder::Any,
    };
    let graph = fixture_graph(&source)?;
    let mut backend = ObservedBackend::strict_cpu();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();

    for query in [
        "MATCH (n:N) RETURN sum(n.name)",
        "MATCH (n:N) RETURN avg(n.name)",
        "MATCH (n:N) RETURN min(n.name + n.suffix)",
        "MATCH (n:N) RETURN max(n.name + n.suffix)",
        "MATCH (n:N) RETURN min(DISTINCT n.name)",
        "MATCH (n:N) RETURN max(DISTINCT n.name)",
    ] {
        let before = calls.snapshot();
        let error = QueryEngine
            .execute(
                query,
                &mut context(&graph, BTreeMap::new(), Some(&backend), true),
            )
            .expect_err("unsupported string reduction must fail closed");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "query={query}");
        assert_eq!(
            call_delta(before, calls.snapshot()).segmented,
            0,
            "unsupported string reduction reached segmented execution: {query}"
        );
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_arithmetic_graph_groups_and_reduction_inputs_in_one_command() -> Result<()> {
    let setup = "CREATE (:A {num: 1, num2: 4}), (:A {num: 5, num2: 2}), \
                 (:A {num: 9, num2: 0}), (:A {num: 3, num2: 3}), \
                 (:A {num: 7, num2: 1})";
    let cases = [
        (
            ManifestCase {
                report_id: 1213,
                feature: "clauses/with-orderBy/WithOrderBy4.feature".to_owned(),
                name: "[11] Sort by an aggregate projection".to_owned(),
            },
            "MATCH (a:A) \
             WITH a.num2 % 3 AS mod, sum(a.num + a.num2) AS sum \
             ORDER BY sum(a.num + a.num2) LIMIT 2 \
             RETURN mod, sum",
        ),
        (
            ManifestCase {
                report_id: 1214,
                feature: "clauses/with-orderBy/WithOrderBy4.feature".to_owned(),
                name: "[12] Sort by an aliased aggregate projection".to_owned(),
            },
            "MATCH (a:A) \
             WITH a.num2 % 3 AS mod, sum(a.num + a.num2) AS sum \
             ORDER BY sum LIMIT 2 \
             RETURN mod, sum",
        ),
    ];
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let sequence_count = calls.graph_relation_stage_sequences()?.len();
    let binding_count = calls.graph_relation_binding_counts()?.len();

    for (case, query) in &cases {
        execute_strict_case(
            case,
            &SourceCase {
                setup_queries: vec![setup.to_owned()],
                parameter_literals: BTreeMap::new(),
                query: (*query).to_owned(),
                expected_headers: vec!["mod".to_owned(), "sum".to_owned()],
                expected_rows: vec![
                    vec!["2".to_owned(), "7".to_owned()],
                    vec!["1".to_owned(), "13".to_owned()],
                ],
                result_order: ResultOrder::Any,
            },
            &mut backend,
        )?;
    }

    assert_eq!(
        &calls.graph_relation_stage_sequences()?[sequence_count..],
        &[
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Limit(2),
                ObservedSegmentedStage::Project,
            ],
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Limit(2),
                ObservedSegmentedStage::Project,
            ],
        ]
    );
    assert_eq!(
        &calls.graph_relation_binding_counts()?[binding_count..],
        &[2, 2],
        "the sealed graph source must export only the deduplicated num/num2 property leaves"
    );
    Ok(())
}

#[test]
fn strict_cpu_erases_exact_grouped_relationship_rebound_and_keeps_both_orders() -> Result<()> {
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let sequence_count = calls.graph_relation_stage_sequences()?.len();
    let binding_count = calls.graph_relation_binding_counts()?.len();
    execute_strict_case(
        &ManifestCase {
            report_id: 1217,
            feature: "clauses/with-orderBy/WithOrderBy4.feature".to_owned(),
            name: "[15] Sort by an aliased aggregate projection does allow subsequent matching"
                .to_owned(),
        },
        &SourceCase {
            setup_queries: vec![
                "CREATE ()-[:T1 {id: 0}]->(:X), \
                        ()-[:T2 {id: 1}]->(:X), \
                        ()-[:T2 {id: 2}]->()"
                    .to_owned(),
            ],
            parameter_literals: BTreeMap::new(),
            query: "MATCH (a)-[r]->(b:X) \
                    WITH a, r, b, count(*) AS c ORDER BY c \
                    MATCH (a)-[r]->(b) \
                    RETURN r AS rel ORDER BY rel.id"
                .to_owned(),
            expected_headers: vec!["rel".to_owned()],
            expected_rows: vec![
                vec!["[:T1 {id: 0}]".to_owned()],
                vec!["[:T2 {id: 1}]".to_owned()],
            ],
            result_order: ResultOrder::Exact,
        },
        &mut backend,
    )?;

    assert_eq!(
        &calls.graph_relation_stage_sequences()?[sequence_count..],
        &[vec![
            ObservedSegmentedStage::Aggregate,
            ObservedSegmentedStage::Project,
            ObservedSegmentedStage::Order,
            ObservedSegmentedStage::Project,
            ObservedSegmentedStage::Order,
            ObservedSegmentedStage::Project,
        ]],
        "the rebound route must preserve both physical ORDER stages in one sealed command"
    );
    assert_eq!(
        &calls.graph_relation_binding_counts()?[binding_count..],
        &[4],
        "the graph source must export a, r, b, and the functionally dependent r.id key"
    );
    Ok(())
}

#[test]
fn grouped_relationship_rebound_rewrite_rejects_every_structural_near_miss() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec![
            "CREATE ()-[:T1 {id: 0}]->(:X), \
                    ()-[:T2 {id: 1}]->(:X), \
                    ()-[:T2 {id: 2}]->()"
                .to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: String::new(),
        expected_headers: Vec::new(),
        expected_rows: Vec::new(),
        result_order: ResultOrder::Any,
    };
    let graph = fixture_graph(&source)?;
    let mut backend = ObservedBackend::strict_cpu();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();

    for (near_miss, query) in [
        (
            "reversed rebound tuple",
            "MATCH (a)-[r]->(b:X) \
             WITH a, r, b, count(*) AS c ORDER BY c \
             MATCH (b)-[r]->(a) \
             RETURN r AS rel ORDER BY rel.id",
        ),
        (
            "typed rebound relationship",
            "MATCH (a)-[r]->(b:X) \
             WITH a, r, b, count(*) AS c ORDER BY c \
             MATCH (a)-[r:T1]->(b) \
             RETURN r AS rel ORDER BY rel.id",
        ),
        (
            "constrained rebound endpoint",
            "MATCH (a)-[r]->(b:X) \
             WITH a, r, b, count(*) AS c ORDER BY c \
             MATCH (a)-[r]->(b:X) \
             RETURN r AS rel ORDER BY rel.id",
        ),
        (
            "endpoint not retained as a group identity",
            "MATCH (a)-[r]->(b:X) \
             WITH a, r, count(*) AS c ORDER BY c \
             MATCH (a)-[r]->(b) \
             RETURN r AS rel ORDER BY rel.id",
        ),
        (
            "different hidden relationship property",
            "MATCH (a)-[r]->(b:X) \
             WITH a, r, b, count(*) AS c ORDER BY c \
             MATCH (a)-[r]->(b) \
             RETURN r AS rel ORDER BY rel.other",
        ),
        (
            "different first ordering direction",
            "MATCH (a)-[r]->(b:X) \
             WITH a, r, b, count(*) AS c ORDER BY c DESC \
             MATCH (a)-[r]->(b) \
             RETURN r AS rel ORDER BY rel.id",
        ),
    ] {
        let before = calls.snapshot();
        let error = QueryEngine
            .execute(
                query,
                &mut context(&graph, BTreeMap::new(), Some(&backend), true),
            )
            .expect_err("an unproved relationship rebound must fail closed");
        assert_eq!(
            error.code,
            ErrorCode::GpuAdmissionFailure,
            "near_miss={near_miss}; query={query}"
        );
        let delta = call_delta(before, calls.snapshot());
        assert_eq!(
            (delta.segmented, delta.sealed_program),
            (0, 0),
            "near miss reached segmented execution: {near_miss}; query={query}"
        );
    }
    Ok(())
}

#[test]
fn graph_fused_arithmetic_sources_fail_closed_when_not_proven_numeric() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (:N {name: 'alpha', suffix: '!', score: 1}), \
                    (:N {name: 'beta', suffix: '?', score: 2})"
                .to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: String::new(),
        expected_headers: Vec::new(),
        expected_rows: Vec::new(),
        result_order: ResultOrder::Any,
    };
    let graph = fixture_graph(&source)?;
    let mut backend = ObservedBackend::strict_cpu();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();

    for query in [
        "MATCH (n:N) RETURN n.name + n.suffix AS bucket, count(*) AS c",
        "MATCH (n:N) RETURN min(n.name + n.suffix) AS first",
        "MATCH (n:N) RETURN sum(size(n.name)) AS total",
    ] {
        let before = calls.snapshot();
        QueryEngine
            .execute(
                query,
                &mut context(&graph, BTreeMap::new(), Some(&backend), true),
            )
            .expect_err("unsupported or non-numeric graph expression must fail closed");
        assert_eq!(
            call_delta(before, calls.snapshot()).segmented,
            0,
            "unsupported graph expression reached segmented execution: {query}"
        );
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_post_aggregate_order_skip_and_limit_in_physical_order() -> Result<()> {
    let cases = [
        (
            ManifestCase {
                report_id: 810,
                feature: "clauses/return-skip-limit/ReturnSkipLimit2.feature".to_owned(),
                name: "[8] Limit to more rows than actual results 2".to_owned(),
            },
            SourceCase {
                setup_queries: vec![
                    "CREATE (a:A), (n1 {num: 1}), (n2 {num: 2}), (m1), (m2) \
                     CREATE (a)-[:T]->(n1), (n1)-[:T]->(m1), (a)-[:T]->(n2), \
                     (n2)-[:T]->(m2)"
                        .to_owned(),
                ],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a:A)-->(n)-->(m) RETURN n.num, count(*) \
                        ORDER BY n.num LIMIT 1000"
                    .to_owned(),
                expected_headers: vec!["n.num".to_owned(), "count(*)".to_owned()],
                expected_rows: vec![
                    vec!["1".to_owned(), "1".to_owned()],
                    vec!["2".to_owned(), "1".to_owned()],
                ],
                result_order: ResultOrder::Exact,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Limit(1000),
            ],
        ),
        (
            ManifestCase {
                report_id: 1224,
                feature: "clauses/with-skip-limit/WithSkipLimit1.feature".to_owned(),
                name: "[2] Ordering and skipping on aggregate".to_owned(),
            },
            SourceCase {
                setup_queries: vec![
                    "CREATE ()-[:T1 {num: 3}]->(x:X), ()-[:T2 {num: 2}]->(x), \
                     ()-[:T3 {num: 1}]->(:Y)"
                        .to_owned(),
                ],
                parameter_literals: BTreeMap::new(),
                query: "MATCH ()-[r1]->(x) WITH x, sum(r1.num) AS c \
                        ORDER BY c SKIP 1 RETURN x, c"
                    .to_owned(),
                expected_headers: vec!["x".to_owned(), "c".to_owned()],
                expected_rows: vec![vec!["(:X)".to_owned(), "5".to_owned()]],
                result_order: ResultOrder::Any,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Skip(1),
                ObservedSegmentedStage::Project,
            ],
        ),
        (
            ManifestCase {
                report_id: 1228,
                feature: "clauses/with-skip-limit/WithSkipLimit2.feature".to_owned(),
                name: "[4] Ordering and limiting on aggregate".to_owned(),
            },
            SourceCase {
                setup_queries: vec![
                    "CREATE ()-[:T1 {num: 3}]->(x:X), ()-[:T2 {num: 2}]->(x), \
                     ()-[:T3 {num: 1}]->(:Y)"
                        .to_owned(),
                ],
                parameter_literals: BTreeMap::new(),
                query: "MATCH ()-[r1]->(x) WITH x, sum(r1.num) AS c \
                        ORDER BY c LIMIT 1 RETURN x, c"
                    .to_owned(),
                expected_headers: vec!["x".to_owned(), "c".to_owned()],
                expected_rows: vec![vec!["(:Y)".to_owned(), "1".to_owned()]],
                result_order: ResultOrder::Any,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Limit(1),
                ObservedSegmentedStage::Project,
            ],
        ),
    ];

    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    for (case, source, expected_stages) in &cases {
        let sequence_count = calls.graph_relation_stage_sequences()?.len();
        execute_strict_case(case, source, &mut backend)?;
        let sequences = calls.graph_relation_stage_sequences()?;
        assert_eq!(
            &sequences[sequence_count..],
            std::slice::from_ref(expected_stages),
            "report {} / {} lowered an unexpected segmented stage tail",
            case.report_id,
            case.name
        );
    }
    Ok(())
}

#[test]
fn strict_cpu_applies_skip_before_limit_after_ordering_all_aggregate_groups() -> Result<()> {
    let source = SourceCase {
        // Six source rows collapse into three groups. Encounter order opposes aggregate order,
        // and SKIP/LIMIT select only the middle group, so applying either window before Aggregate
        // or swapping their physical order produces an observably different result.
        setup_queries: vec![
            "CREATE ()-[:R {num: 9}]->(:C), ()-[:R {num: 2}]->(b:B), \
             ()-[:R {num: 2}]->(b), ()-[:R {num: 1}]->(a:A), \
             ()-[:R {num: 1}]->(a), ()-[:R {num: 1}]->(a)"
                .to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: "MATCH ()-[r]->(x) WITH x, sum(r.num) AS c \
                ORDER BY c SKIP 1 LIMIT 1 RETURN x, c"
            .to_owned(),
        expected_headers: vec!["x".to_owned(), "c".to_owned()],
        expected_rows: vec![vec!["(:B)".to_owned(), "4".to_owned()]],
        result_order: ResultOrder::Exact,
    };
    let graph = fixture_graph(&source)?;
    let mut backend = ObservedBackend::strict_cpu();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();
    let before = calls.snapshot();
    let output = QueryEngine.execute(
        &source.query,
        &mut context_with_max_result_rows(&graph, BTreeMap::new(), Some(&backend), true, 3),
    )?;
    assert_exact_output(
        &ManifestCase {
            report_id: 0,
            feature: "adversarial/post-aggregate-window".to_owned(),
            name: "Aggregate, Order, Skip, Limit retain physical order".to_owned(),
        },
        &source,
        &output,
    )?;
    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            ..CallSnapshot::default()
        }
    );
    assert_eq!(
        calls.graph_relation_stage_sequences()?,
        [vec![
            ObservedSegmentedStage::Aggregate,
            ObservedSegmentedStage::Project,
            ObservedSegmentedStage::Order,
            ObservedSegmentedStage::Skip(1),
            ObservedSegmentedStage::Limit(1),
            ObservedSegmentedStage::Project,
        ]]
    );
    Ok(())
}

#[test]
fn graph_fused_post_aggregate_windows_fail_closed_for_invalid_or_oversized_rows() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec!["CREATE (:N {value: 1}), (:N {value: 2})".to_owned()],
        parameter_literals: BTreeMap::new(),
        query: String::new(),
        expected_headers: Vec::new(),
        expected_rows: Vec::new(),
        result_order: ResultOrder::Any,
    };
    let graph = fixture_graph(&source)?;
    let mut backend = ObservedBackend::strict_cpu();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();

    for query in [
        "MATCH (n:N) RETURN n.value, count(*) ORDER BY n.value LIMIT n.value",
        "MATCH (n:N) RETURN n.value, count(*) ORDER BY n.value LIMIT -1",
        "MATCH (n:N) RETURN n.value, count(*) ORDER BY n.value SKIP -1",
    ] {
        let before = calls.snapshot();
        QueryEngine
            .execute(
                query,
                &mut context(&graph, BTreeMap::new(), Some(&backend), true),
            )
            .expect_err("invalid post-aggregate window must fail closed");
        assert_eq!(
            call_delta(before, calls.snapshot()).segmented,
            0,
            "invalid query reached the segmented backend: {query}"
        );
    }

    for (query, expected_fragment) in [
        (
            "MATCH (n:N) RETURN n.value, count(*) ORDER BY n.value LIMIT 4294967296",
            "TopK limit exceeds u32",
        ),
        (
            "MATCH (n:N) RETURN n.value, count(*) SKIP 4294967296",
            "SKIP exceeds u32",
        ),
    ] {
        let before = calls.snapshot();
        let error = QueryEngine
            .execute(
                query,
                &mut context(&graph, BTreeMap::new(), Some(&backend), true),
            )
            .expect_err("oversized post-aggregate window must fail before backend execution");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded, "query={query}");
        assert!(
            error.message.contains(expected_fragment),
            "query={query}; unexpected error: {}",
            error.message
        );
        assert_eq!(
            call_delta(before, calls.snapshot()).segmented,
            0,
            "oversized query reached the segmented backend: {query}"
        );
    }
    Ok(())
}

#[test]
fn strict_cpu_global_count_fences_the_complete_graph_without_enumerating_entities() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (:N {value: 1})".to_owned(),
            "CREATE (:N {value: 2})".to_owned(),
            "CREATE (:N {value: 3})".to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: "MATCH (n) RETURN count(*) AS total".to_owned(),
        expected_headers: vec!["total".to_owned()],
        expected_rows: vec![vec!["3".to_owned()]],
        result_order: ResultOrder::Exact,
    };
    let graph = fixture_graph(&source)?;
    let mut backend = ObservedBackend::strict_cpu();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();
    let before = calls.snapshot();
    let output = QueryEngine.execute(
        &source.query,
        &mut context_with_max_result_rows(&graph, BTreeMap::new(), Some(&backend), true, 1),
    )?;
    assert_exact_output(
        &ManifestCase {
            report_id: 741,
            feature: "clauses/return/Return6.feature".to_owned(),
            name: "complete graph source capacity".to_owned(),
        },
        &source,
        &output,
    )?;
    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            ..CallSnapshot::default()
        }
    );
    assert!(output.dependencies.entities.is_empty());
    assert!(!output.dependencies.predicates.is_empty());
    Ok(())
}

#[test]
fn strict_cpu_executes_compatible_return6_post_aggregate_expressions() -> Result<()> {
    let cases = [
        (
            ManifestCase {
                report_id: 734,
                feature: "clauses/return/Return6.feature".to_owned(),
                name: "[2] Projecting an arithmetic expression with aggregation".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ({id: 42})".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a) RETURN a, count(a) + 3".to_owned(),
                expected_headers: vec!["a".to_owned(), "count(a) + 3".to_owned()],
                expected_rows: vec![vec!["({id: 42})".to_owned(), "4".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 736,
                feature: "clauses/return/Return6.feature".to_owned(),
                name: "[4] Support multiple divisions in aggregate function".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE (:N)".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) RETURN count(n) / 60 / 60 AS count".to_owned(),
                expected_headers: vec!["count".to_owned()],
                expected_rows: vec![vec!["0".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 741,
                feature: "clauses/return/Return6.feature".to_owned(),
                name: "[9] Aggregates with arithmetics".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE (:N)".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH () RETURN count(*) * 10 AS c".to_owned(),
                expected_headers: vec!["c".to_owned()],
                expected_rows: vec![vec!["10".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 749,
                feature: "clauses/return/Return6.feature".to_owned(),
                name: "[17] Handle constants and parameters inside an expression which contains an aggregation expression".to_owned(),
            },
            SourceCase {
                setup_queries: Vec::new(),
                parameter_literals: BTreeMap::from([("age".to_owned(), "38".to_owned())]),
                query: "MATCH (person) RETURN $age + avg(person.age) - 1000".to_owned(),
                expected_headers: vec!["$age + avg(person.age) - 1000".to_owned()],
                expected_rows: vec![vec!["null".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
    ];
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let graph_sources_before = calls.graph_relation_source.load(Ordering::SeqCst);
    for (case, source) in &cases {
        execute_strict_case(case, source, &mut backend)?;
    }
    assert_eq!(
        calls
            .graph_relation_source
            .load(Ordering::SeqCst)
            .saturating_sub(graph_sources_before),
        cases.len()
    );
    Ok(())
}

#[test]
fn strict_cpu_executes_return6_18_preaggregate_property_alias_in_one_command() -> Result<()> {
    let cases = [
        (
            ManifestCase {
                report_id: 750,
                feature: "clauses/return/Return6.feature".to_owned(),
                name: "[18] Handle returned variables inside an expression which contains an aggregation expression"
                    .to_owned(),
            },
            SourceCase {
                setup_queries: Vec::new(),
                parameter_literals: BTreeMap::new(),
                query: "MATCH (me: Person)--(you: Person) \
                        WITH me.age AS age, you \
                        RETURN age, age + count(you.age)"
                    .to_owned(),
                expected_headers: vec!["age".to_owned(), "age + count(you.age)".to_owned()],
                expected_rows: Vec::new(),
                result_order: ResultOrder::Any,
            },
        ),
        // The official scenario has an empty graph. This non-empty semantic twin prevents that
        // cardinality from concealing an incorrect alias substitution or grouping source.
        (
            ManifestCase {
                report_id: 750,
                feature: "clauses/return/Return6.feature".to_owned(),
                name: "Return6 [18] non-empty alias-substitution twin".to_owned(),
            },
            SourceCase {
                setup_queries: vec![
                    "CREATE (:Person {age: 10})-[:R]->(:Person {age: 4}), \
                            (:Person {age: 10})-[:R]->(:Person {age: 5})"
                        .to_owned(),
                ],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (me: Person)-->(you: Person) \
                        WITH me.age AS age, you \
                        RETURN age, age + count(you.age)"
                    .to_owned(),
                expected_headers: vec!["age".to_owned(), "age + count(you.age)".to_owned()],
                expected_rows: vec![vec!["10".to_owned(), "12".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
    ];
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let graph_sources_before = calls.graph_relation_source.load(Ordering::SeqCst);
    for (case, source) in &cases {
        execute_strict_case(case, source, &mut backend)?;
    }
    assert_eq!(
        calls
            .graph_relation_source
            .load(Ordering::SeqCst)
            .saturating_sub(graph_sources_before),
        cases.len(),
        "Return6 [18] did not use the sealed graph-relation source"
    );
    Ok(())
}

#[test]
fn strict_cpu_executes_exact_comparison_named_path_multiscan_and_distinct_tranche() -> Result<()> {
    let cases = [
        (
            ManifestCase {
                report_id: 705,
                feature: "clauses/return/Return2.feature".to_owned(),
                name: "[10] Return count aggregation over an empty graph".to_owned(),
            },
            SourceCase {
                setup_queries: Vec::new(),
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a) RETURN count(a) > 0".to_owned(),
                expected_headers: vec!["count(a) > 0".to_owned()],
                expected_rows: vec![vec!["false".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 723,
                feature: "clauses/return/Return4.feature".to_owned(),
                name: "[7] Keeping used expression 4".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ()".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH p = (n)-->(b) RETURN aVg(    n.aGe     )".to_owned(),
                expected_headers: vec!["aVg(    n.aGe     )".to_owned()],
                expected_rows: vec![vec!["null".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 779,
                feature: "clauses/return-orderby/ReturnOrderBy2.feature".to_owned(),
                name: "[11] Aggregates ordered by arithmetics".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE (:A), (:X), (:X)".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a:A), (b:X) \
                        RETURN count(a) * 10 + count(b) * 5 AS x \
                        ORDER BY x"
                    .to_owned(),
                expected_headers: vec!["x".to_owned()],
                expected_rows: vec![vec!["30".to_owned()]],
                result_order: ResultOrder::Exact,
            },
        ),
        (
            ManifestCase {
                report_id: 1282,
                feature: "expressions/aggregation/Aggregation8.feature".to_owned(),
                name: "[1] Distinct on unbound node".to_owned(),
            },
            SourceCase {
                setup_queries: Vec::new(),
                parameter_literals: BTreeMap::new(),
                query: "OPTIONAL MATCH (a) RETURN count(DISTINCT a)".to_owned(),
                expected_headers: vec!["count(DISTINCT a)".to_owned()],
                expected_rows: vec![vec!["0".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 1283,
                feature: "expressions/aggregation/Aggregation8.feature".to_owned(),
                name: "[2] Distinct on null".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ()".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a) RETURN count(DISTINCT a.name)".to_owned(),
                expected_headers: vec!["count(DISTINCT a.name)".to_owned()],
                expected_rows: vec![vec!["0".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
    ];
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let graph_sources_before = calls.graph_relation_source.load(Ordering::SeqCst);
    for (case, source) in &cases {
        execute_strict_case(case, source, &mut backend)?;
    }
    assert_eq!(
        calls
            .graph_relation_source
            .load(Ordering::SeqCst)
            .saturating_sub(graph_sources_before),
        cases.len()
    );
    Ok(())
}

#[test]
fn strict_cpu_executes_exact_collect_and_list_semantics_in_one_command() -> Result<()> {
    let cases = [
        (
            ManifestCase {
                report_id: 737,
                feature: "clauses/return/Return6.feature".to_owned(),
                name: "[5] Aggregates inside normal functions".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["UNWIND range(0, 10) AS i CREATE ()".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a) RETURN size(collect(a))".to_owned(),
                expected_headers: vec!["size(collect(a))".to_owned()],
                expected_rows: vec![vec!["11".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 742,
                feature: "clauses/return/Return6.feature".to_owned(),
                name: "[10] Multiple aggregates on same variable".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ()".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) RETURN count(n), collect(n)".to_owned(),
                expected_headers: vec!["count(n)".to_owned(), "collect(n)".to_owned()],
                expected_rows: vec![vec!["1".to_owned(), "[()]".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 1267,
                feature: "expressions/aggregation/Aggregation5.feature".to_owned(),
                name: "[1] `collect()` filtering nulls".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ()".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) RETURN n, collect(x)"
                    .to_owned(),
                expected_headers: vec!["n".to_owned(), "collect(x)".to_owned()],
                expected_rows: vec![vec!["()".to_owned(), "[]".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 1268,
                feature: "expressions/aggregation/Aggregation5.feature".to_owned(),
                name: "[2] OPTIONAL MATCH and `collect()` on node property".to_owned(),
            },
            SourceCase {
                setup_queries: vec![
                    "CREATE (:DoesExist {num: 42})".to_owned(),
                    "CREATE (:DoesExist {num: 43})".to_owned(),
                    "CREATE (:DoesExist {num: 44})".to_owned(),
                ],
                parameter_literals: BTreeMap::new(),
                query: "OPTIONAL MATCH (f:DoesExist) OPTIONAL MATCH (n:DoesNotExist) \
                        RETURN collect(DISTINCT n.num) AS a, collect(DISTINCT f.num) AS b"
                    .to_owned(),
                expected_headers: vec!["a".to_owned(), "b".to_owned()],
                expected_rows: vec![vec!["[]".to_owned(), "[42, 43, 44]".to_owned()]],
                result_order: ResultOrder::IgnoreListElementOrder,
            },
        ),
    ];
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let before = calls.snapshot();
    for (case, source) in &cases {
        execute_strict_case(case, source, &mut backend)?;
    }
    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot {
            segmented: cases.len(),
            sealed_program: cases.len(),
            ..CallSnapshot::default()
        }
    );
    assert_eq!(
        calls.graph_relation_source.load(Ordering::SeqCst),
        cases.len(),
        "collect cases did not remain on the sealed graph-relation source"
    );
    Ok(())
}

#[test]
fn graph_fused_compiler_declines_nested_documents_variable_paths_and_unsupported_shapes()
-> Result<()> {
    for (setup, query) in [
        (
            vec!["CREATE (:N)"],
            "MATCH (n) RETURN collect([n]) AS nested_documents",
        ),
        (
            vec!["CREATE (:N)"],
            "MATCH (n) RETURN reverse(collect(n)) AS reversed_documents",
        ),
        (
            vec!["CREATE ()-[:R]->()"],
            "MATCH p = (a)-[:R]->(b) RETURN count(p) AS paths",
        ),
        (Vec::new(), "OPTIONAL MATCH (a) RETURN count(a)"),
        (
            vec!["CREATE (:N {v: 1})"],
            "MATCH (n) RETURN sum(DISTINCT n.v)",
        ),
    ] {
        let source = SourceCase {
            setup_queries: setup.into_iter().map(str::to_owned).collect(),
            parameter_literals: BTreeMap::new(),
            query: query.to_owned(),
            expected_headers: Vec::new(),
            expected_rows: Vec::new(),
            result_order: ResultOrder::Any,
        };
        let graph = fixture_graph(&source)?;
        let mut backend = ObservedBackend::strict_cpu();
        backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
            graph.snapshot()?,
        ))])?;
        let calls = backend.calls();
        let before = calls.snapshot();
        let error = QueryEngine
            .execute(
                query,
                &mut context(&graph, BTreeMap::new(), Some(&backend), true),
            )
            .expect_err("unsupported fused graph shape must fail closed in strict native mode");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "query={query}");
        assert_eq!(
            call_delta(before, calls.snapshot()).segmented,
            0,
            "query={query}"
        );
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_count_then_sum_in_one_graph_command() -> Result<()> {
    let case = ManifestCase {
        report_id: 0,
        feature: "local/two-aggregate-boundaries".to_owned(),
        name: "count then sum".to_owned(),
    };
    let source = SourceCase {
        setup_queries: vec!["CREATE (:N), (:N)".to_owned()],
        parameter_literals: BTreeMap::new(),
        query: "MATCH (n) WITH count(*) AS c RETURN sum(c) AS total".to_owned(),
        expected_headers: vec!["total".to_owned()],
        expected_rows: vec![vec!["2".to_owned()]],
        result_order: ResultOrder::Any,
    };
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let before = calls.snapshot();
    let stages_before = calls.graph_relation_stage_sequences()?.len();

    execute_strict_case(&case, &source, &mut backend)?;

    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            ..CallSnapshot::default()
        },
        "two aggregate boundaries must remain one sealed graph command without fallback"
    );
    assert_eq!(
        &calls.graph_relation_stage_sequences()?[stages_before..],
        &[vec![
            ObservedSegmentedStage::Aggregate,
            ObservedSegmentedStage::Project,
            ObservedSegmentedStage::Aggregate,
            ObservedSegmentedStage::Project,
        ]]
    );
    Ok(())
}

#[test]
fn valid_limit_collect_declines_the_sealed_range_sum_compiler() -> Result<()> {
    const QUERY: &str = "UNWIND range(1, 4) AS i WITH i LIMIT 2 RETURN collect(i) AS values";
    let graph = GraphStore::default();
    let output = QueryEngine.execute(QUERY, &mut context(&graph, BTreeMap::new(), None, false))?;
    assert_eq!(output.result.schema[0].0, "values");
    let values = output
        .result
        .batches
        .iter()
        .flat_map(|batch| batch.columns[0].values.iter())
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        [ResultValue::List(vec![
            ResultValue::Scalar(ScalarValue::Integer(1)),
            ResultValue::Scalar(ScalarValue::Integer(2)),
        ])]
    );

    let mut backend = ObservedBackend::strict_cpu();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();
    let before = calls.snapshot();
    let native_context_output = QueryEngine.execute(
        QUERY,
        &mut context(&graph, BTreeMap::new(), Some(&backend), true),
    )?;
    let native_context_values = native_context_output
        .result
        .batches
        .iter()
        .flat_map(|batch| batch.columns[0].values.iter())
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(native_context_values, values);
    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot::default()
    );
    Ok(())
}

#[test]
fn sealed_range_sum_enforces_query_intermediate_budget_at_128_129() -> Result<()> {
    let graph = GraphStore::default();
    let mut backend = ObservedBackend::strict_cpu();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();
    let expected_calls = CallSnapshot {
        segmented: 1,
        sealed_program: 1,
        range_source: 1,
        pre_aggregate_limit: 1,
        ..CallSnapshot::default()
    };

    let before = calls.snapshot();
    let output = QueryEngine.execute(
        "UNWIND range(1, 200) AS i WITH i LIMIT 128 RETURN sum(i) AS total",
        &mut context_with_max_result_rows(&graph, BTreeMap::new(), Some(&backend), true, 128),
    )?;
    assert_eq!(call_delta(before, calls.snapshot()), expected_calls);
    let values = output
        .result
        .batches
        .iter()
        .flat_map(|batch| batch.columns[0].values.iter())
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(values, [ResultValue::Scalar(ScalarValue::Integer(8_256))]);

    let before = calls.snapshot();
    let error = QueryEngine
        .execute(
            "UNWIND range(1, 200) AS i WITH i LIMIT 129 RETURN sum(i) AS total",
            &mut context_with_max_result_rows(&graph, BTreeMap::new(), Some(&backend), true, 128),
        )
        .expect_err("LIMIT 129 must exceed max_result_rows 128 before aggregation");
    assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    assert_eq!(call_delta(before, calls.snapshot()), expected_calls);
    Ok(())
}

#[test]
fn strict_cpu_reduces_more_than_one_million_logical_range_rows_without_an_engine_ceiling()
-> Result<()> {
    const ROWS: usize = 1_048_577;
    const QUERY: &str = "UNWIND range(1, 1048577) AS i WITH i LIMIT 1048577 RETURN sum(i) AS total";
    let graph = GraphStore::default();
    let mut backend = ObservedBackend::strict_cpu();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();
    let before = calls.snapshot();
    let output = QueryEngine.execute(
        QUERY,
        &mut context_with_max_result_rows(&graph, BTreeMap::new(), Some(&backend), true, ROWS),
    )?;
    let values = output
        .result
        .batches
        .iter()
        .flat_map(|batch| batch.columns[0].values.iter())
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        [ResultValue::Scalar(ScalarValue::Integer(549_757_386_753))]
    );
    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            range_source: 1,
            pre_aggregate_limit: 1,
            ..CallSnapshot::default()
        }
    );
    Ok(())
}

fn execute_shared_group_reduction_padding_regression(backend: &mut ObservedBackend) -> Result<()> {
    let case = ManifestCase {
        report_id: usize::MAX,
        feature: "native/fused-graph-padding".to_owned(),
        name: "shared group and reduction input ignores capacity padding".to_owned(),
    };
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (:N {v: 7}), (:N {v: 7}), (:N), \
                    (:Noise {v: 90}), (:Noise {v: 91}), (:Noise {v: 92})"
                .to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: "MATCH (n:N) \
                RETURN n.v, count(*) AS all_rows, count(n.v) AS present, sum(n.v) AS total \
                ORDER BY n.v"
            .to_owned(),
        expected_headers: vec![
            "n.v".to_owned(),
            "all_rows".to_owned(),
            "present".to_owned(),
            "total".to_owned(),
        ],
        expected_rows: vec![
            vec![
                "7".to_owned(),
                "2".to_owned(),
                "2".to_owned(),
                "14".to_owned(),
            ],
            vec![
                "null".to_owned(),
                "1".to_owned(),
                "0".to_owned(),
                "0".to_owned(),
            ],
        ],
        result_order: ResultOrder::Exact,
    };
    let calls = backend.calls();
    let capacities_before = calls.graph_relation_source_capacities()?.len();
    let bindings_before = calls.graph_relation_binding_counts()?.len();
    execute_strict_case(&case, &source, backend)?;
    let capacities = calls.graph_relation_source_capacities()?;
    let bindings = calls.graph_relation_binding_counts()?;
    assert_eq!(&capacities[capacities_before..], &[6]);
    assert_eq!(
        &bindings[bindings_before..],
        &[1],
        "group key and both value reductions must share one exported graph binding"
    );
    Ok(())
}

#[test]
fn strict_cpu_fixture_proves_shared_group_reduction_padding_slack() -> Result<()> {
    execute_shared_group_reduction_padding_regression(&mut ObservedBackend::strict_cpu())
}

#[test]
fn strict_cpu_proves_return5_2_nullable_distinct_as_one_graph_grouping_command() -> Result<()> {
    let case = ManifestCase {
        report_id: 729,
        feature: "clauses/return/Return5.feature".to_owned(),
        name: "[2] DISTINCT on nullable values".to_owned(),
    };
    let source = SourceCase {
        setup_queries: vec!["CREATE ({name: 'Florescu'}), (), ()".to_owned()],
        parameter_literals: BTreeMap::new(),
        query: "MATCH (n) RETURN DISTINCT n.name".to_owned(),
        expected_headers: vec!["n.name".to_owned()],
        expected_rows: vec![vec!["'Florescu'".to_owned()], vec!["null".to_owned()]],
        result_order: ResultOrder::Any,
    };
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let graph_sources_before = calls.graph_relation_source.load(Ordering::SeqCst);
    let unit_sources_before = calls.unit_source.load(Ordering::SeqCst);
    let stage_count_before = calls.graph_relation_stage_sequences()?.len();
    let binding_count_before = calls.graph_relation_binding_counts()?.len();
    let capacity_count_before = calls.graph_relation_source_capacities()?.len();

    execute_strict_case(&case, &source, &mut backend)?;

    assert_eq!(
        calls.graph_relation_source.load(Ordering::SeqCst),
        graph_sources_before + 1
    );
    assert_eq!(
        calls.unit_source.load(Ordering::SeqCst),
        unit_sources_before
    );
    assert_eq!(
        &calls.graph_relation_stage_sequences()?[stage_count_before..],
        &[vec![ObservedSegmentedStage::Aggregate]]
    );
    assert_eq!(
        &calls.graph_relation_binding_counts()?[binding_count_before..],
        &[1]
    );
    assert_eq!(
        &calls.graph_relation_source_capacities()?[capacity_count_before..],
        &[3]
    );
    Ok(())
}

#[test]
fn strict_cpu_proves_direct_property_distinct_and_group_alias_order_tails() -> Result<()> {
    let shared_setup = vec![
        "CREATE ({name: 'A'}), ({name: 'A'}), ({name: 'B'}), ({name: 'C'}), ({name: 'C'})"
            .to_owned(),
    ];
    let cases = [
        (
            ManifestCase {
                report_id: 777,
                feature: "clauses/return-orderby/ReturnOrderBy2.feature".to_owned(),
                name: "[9] Using aliased DISTINCT expression in ORDER BY".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ({id: 1}), ({id: 10})".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) RETURN DISTINCT n.id AS id ORDER BY id DESC".to_owned(),
                expected_headers: vec!["id".to_owned()],
                expected_rows: vec![vec!["10".to_owned()], vec!["1".to_owned()]],
                result_order: ResultOrder::Exact,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Project,
            ],
        ),
        (
            ManifestCase {
                report_id: 1081,
                feature: "clauses/with-orderBy/WithOrderBy2.feature".to_owned(),
                name: "[23] grouped alias source expression ASC".to_owned(),
            },
            SourceCase {
                setup_queries: shared_setup.clone(),
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a) WITH a.name AS name, count(*) AS cnt \
                        ORDER BY a.name + 'C' ASC LIMIT 1 RETURN name, cnt"
                    .to_owned(),
                expected_headers: vec!["name".to_owned(), "cnt".to_owned()],
                expected_rows: vec![vec!["'A'".to_owned(), "2".to_owned()]],
                result_order: ResultOrder::Any,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Limit(1),
                ObservedSegmentedStage::Project,
            ],
        ),
        (
            ManifestCase {
                report_id: 1082,
                feature: "clauses/with-orderBy/WithOrderBy2.feature".to_owned(),
                name: "[23] grouped alias source expression DESC".to_owned(),
            },
            SourceCase {
                setup_queries: shared_setup.clone(),
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a) WITH a.name AS name, count(*) AS cnt \
                        ORDER BY a.name + 'C' DESC LIMIT 1 RETURN name, cnt"
                    .to_owned(),
                expected_headers: vec!["name".to_owned(), "cnt".to_owned()],
                expected_rows: vec![vec!["'C'".to_owned(), "2".to_owned()]],
                result_order: ResultOrder::Any,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Limit(1),
                ObservedSegmentedStage::Project,
            ],
        ),
        (
            ManifestCase {
                report_id: 1083,
                feature: "clauses/with-orderBy/WithOrderBy2.feature".to_owned(),
                name: "[24] direct property DISTINCT ASC".to_owned(),
            },
            SourceCase {
                setup_queries: shared_setup.clone(),
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a) WITH DISTINCT a.name AS name \
                        ORDER BY a.name ASC LIMIT 1 RETURN *"
                    .to_owned(),
                expected_headers: vec!["name".to_owned()],
                expected_rows: vec![vec!["'A'".to_owned()]],
                result_order: ResultOrder::Any,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Limit(1),
                ObservedSegmentedStage::Project,
            ],
        ),
        (
            ManifestCase {
                report_id: 1084,
                feature: "clauses/with-orderBy/WithOrderBy2.feature".to_owned(),
                name: "[24] direct property DISTINCT DESC".to_owned(),
            },
            SourceCase {
                setup_queries: shared_setup,
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a) WITH DISTINCT a.name AS name \
                        ORDER BY a.name DESC LIMIT 1 RETURN *"
                    .to_owned(),
                expected_headers: vec!["name".to_owned()],
                expected_rows: vec![vec!["'C'".to_owned()]],
                result_order: ResultOrder::Any,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
                ObservedSegmentedStage::Order,
                ObservedSegmentedStage::Limit(1),
                ObservedSegmentedStage::Project,
            ],
        ),
    ];
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let stages_before = calls.graph_relation_stage_sequences()?.len();
    for (case, source, _) in &cases {
        execute_strict_case(case, source, &mut backend)?;
    }
    assert_eq!(
        &calls.graph_relation_stage_sequences()?[stages_before..],
        &cases
            .iter()
            .map(|(_, _, stages)| stages.clone())
            .collect::<Vec<_>>()
    );
    Ok(())
}

fn execute_relationship_key_value_control(backend: &mut ObservedBackend) -> Result<()> {
    let cases = [
        (
            ManifestCase {
                report_id: 646,
                feature: "clauses/merge/Merge6.feature".to_owned(),
                name: "relationship key/value control with one property".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ()-[:TYPE {name: 'foo'}]->()".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH ()-[r:TYPE]->() \
                        RETURN [key IN keys(r) | key + '->' + r[key]] AS keyValue"
                    .to_owned(),
                expected_headers: vec!["keyValue".to_owned()],
                expected_rows: vec![vec!["['name->foo']".to_owned()]],
                result_order: ResultOrder::IgnoreListElementOrder,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
            ],
        ),
        (
            ManifestCase {
                report_id: 647,
                feature: "clauses/merge/Merge6.feature".to_owned(),
                name: "relationship key/value control with no properties".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ()-[:TYPE]->()".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH ()-[r:TYPE]->() \
                        RETURN [key IN keys(r) | key + '->' + r[key]] AS keyValue"
                    .to_owned(),
                expected_headers: vec!["keyValue".to_owned()],
                expected_rows: vec![vec!["[]".to_owned()]],
                result_order: ResultOrder::IgnoreListElementOrder,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
            ],
        ),
        (
            ManifestCase {
                report_id: 649,
                feature: "clauses/merge/Merge6.feature".to_owned(),
                name: "relationship key/value control with two properties".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ()-[:TYPE {name: 'bar', name2: 'baz'}]->()".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH ()-[r:TYPE]->() \
                        RETURN [key IN keys(r) | key + '->' + r[key]] AS keyValue"
                    .to_owned(),
                expected_headers: vec!["keyValue".to_owned()],
                expected_rows: vec![vec!["['name->bar', 'name2->baz']".to_owned()]],
                result_order: ResultOrder::IgnoreListElementOrder,
            },
            vec![
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
            ],
        ),
    ];
    let calls = backend.calls();
    let stages_before = calls.graph_relation_stage_sequences()?.len();
    let bindings_before = calls.graph_relation_binding_counts()?.len();
    for (case, source, _) in &cases {
        execute_strict_case(case, source, backend)?;
    }
    assert_eq!(
        &calls.graph_relation_stage_sequences()?[stages_before..],
        &cases
            .iter()
            .map(|(_, _, stages)| stages.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        &calls.graph_relation_binding_counts()?[bindings_before..],
        &[2, 1, 3],
        "the sealed source must export the relationship entity plus every fenced string property"
    );
    Ok(())
}

#[test]
fn strict_cpu_proves_relationship_keys_dynamic_property_control_as_one_command() -> Result<()> {
    execute_relationship_key_value_control(&mut ObservedBackend::strict_cpu())
}

#[test]
fn strict_cpu_proves_return5_and_with5_document_distinct_and_list_grouping() -> Result<()> {
    let shared_setup = vec!["CREATE ({list: ['A', 'B']}), ({list: ['A', 'B']})".to_owned()];
    let cases = vec![
        (
            ManifestCase {
                report_id: 728,
                feature: "clauses/return/Return5.feature".to_owned(),
                name: "[1] DISTINCT inside aggregation should work with lists in maps".to_owned(),
            },
            SourceCase {
                setup_queries: shared_setup.clone(),
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) RETURN count(DISTINCT {name: n.list}) AS count".to_owned(),
                expected_headers: vec!["count".to_owned()],
                expected_rows: vec![vec!["1".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 730,
                feature: "clauses/return/Return5.feature".to_owned(),
                name: "[3] DISTINCT inside aggregation should work with nested lists in maps"
                    .to_owned(),
            },
            SourceCase {
                setup_queries: shared_setup.clone(),
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) RETURN count(DISTINCT \
                        {name: [[n.list, n.list], [n.list, n.list]]}) AS count"
                    .to_owned(),
                expected_headers: vec!["count".to_owned()],
                expected_rows: vec![vec!["1".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 731,
                feature: "clauses/return/Return5.feature".to_owned(),
                name:
                    "[4] DISTINCT inside aggregation should work with nested lists of maps in maps"
                        .to_owned(),
            },
            SourceCase {
                setup_queries: shared_setup,
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) RETURN count(DISTINCT \
                        {name: [{name2: n.list}, {baz: {apa: n.list}}]}) AS count"
                    .to_owned(),
                expected_headers: vec!["count".to_owned()],
                expected_rows: vec![vec!["1".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 732,
                feature: "clauses/return/Return5.feature".to_owned(),
                name: "[5] Aggregate on list values".to_owned(),
            },
            SourceCase {
                setup_queries: vec![
                    "CREATE ({color: ['red']})".to_owned(),
                    "CREATE ({color: ['blue']})".to_owned(),
                    "CREATE ({color: ['red']})".to_owned(),
                ],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (a) RETURN DISTINCT a.color, count(*)".to_owned(),
                expected_headers: vec!["a.color".to_owned(), "count(*)".to_owned()],
                expected_rows: vec![
                    vec!["['red']".to_owned(), "2".to_owned()],
                    vec!["['blue']".to_owned(), "1".to_owned()],
                ],
                result_order: ResultOrder::Any,
            },
        ),
        (
            ManifestCase {
                report_id: 919,
                feature: "clauses/with/With5.feature".to_owned(),
                name: "[2] Handling DISTINCT with lists in maps".to_owned(),
            },
            SourceCase {
                setup_queries: vec!["CREATE ({list: ['A', 'B']}), ({list: ['A', 'B']})".to_owned()],
                parameter_literals: BTreeMap::new(),
                query: "MATCH (n) WITH DISTINCT {name: n.list} AS map RETURN count(*)".to_owned(),
                expected_headers: vec!["count(*)".to_owned()],
                expected_rows: vec![vec!["1".to_owned()]],
                result_order: ResultOrder::Any,
            },
        ),
    ];
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let stages_before = calls.graph_relation_stage_sequences()?.len();
    for (case, source) in &cases {
        execute_strict_case(case, source, &mut backend)?;
    }
    assert_eq!(
        &calls.graph_relation_stage_sequences()?[stages_before..],
        &vec![
            vec![
                ObservedSegmentedStage::PropertyValue,
                ObservedSegmentedStage::Aggregate,
                ObservedSegmentedStage::Project,
            ];
            5
        ],
        "every remaining Return5/With5 scenario must read its canonical document property and aggregate in one command"
    );
    Ok(())
}

#[test]
fn strict_cpu_proves_return6_6_empty_document_tail_in_one_command() -> Result<()> {
    let case = ManifestCase {
        report_id: 738,
        feature: "clauses/return/Return6.feature".to_owned(),
        name: "[6] Handle aggregates inside non-aggregate expressions".to_owned(),
    };
    let source = SourceCase {
        setup_queries: Vec::new(),
        parameter_literals: BTreeMap::new(),
        query: "MATCH (a {name: 'Andres'})<-[:FATHER]-(child) \
                RETURN a.name, {foo: a.name='Andres', kids: collect(child.name)}"
            .to_owned(),
        expected_headers: vec![
            "a.name".to_owned(),
            "{foo: a.name='Andres', kids: collect(child.name)}".to_owned(),
        ],
        expected_rows: Vec::new(),
        result_order: ResultOrder::Any,
    };
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let before = calls.snapshot();
    let stages_before = calls.graph_relation_stage_sequences()?.len();

    execute_strict_case(&case, &source, &mut backend)?;

    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            ..CallSnapshot::default()
        },
        "the empty document projection must still execute one sealed graph command"
    );
    assert_eq!(
        &calls.graph_relation_stage_sequences()?[stages_before..],
        &[vec![
            ObservedSegmentedStage::Aggregate,
            ObservedSegmentedStage::Project,
        ]]
    );
    Ok(())
}

#[test]
fn strict_cpu_proves_return6_16_two_aggregate_boundaries_in_one_graph_command() -> Result<()> {
    let case = ManifestCase {
        report_id: 748,
        feature: "clauses/return/Return6.feature".to_owned(),
        name: "[16] Aggregation on complex expressions".to_owned(),
    };
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (andres {name: 'Andres'}), \
                    (michael {name: 'Michael'}), \
                    (peter {name: 'Peter'}), \
                    (bread {type: 'Bread'}), \
                    (veggies {type: 'Veggies'}), \
                    (meat {type: 'Meat'})"
                .to_owned(),
            "MATCH (andres {name: 'Andres'}), \
                   (michael {name: 'Michael'}), \
                   (peter {name: 'Peter'}), \
                   (bread {type: 'Bread'}), \
                   (veggies {type: 'Veggies'}), \
                   (meat {type: 'Meat'}) \
             CREATE (andres)-[:ATE {times: 10}]->(bread), \
                    (andres)-[:ATE {times: 8}]->(veggies), \
                    (michael)-[:ATE {times: 4}]->(veggies), \
                    (michael)-[:ATE {times: 6}]->(bread), \
                    (michael)-[:ATE {times: 9}]->(meat), \
                    (peter)-[:ATE {times: 7}]->(veggies), \
                    (peter)-[:ATE {times: 7}]->(bread), \
                    (peter)-[:ATE {times: 4}]->(meat)"
                .to_owned(),
        ],
        parameter_literals: BTreeMap::new(),
        query: "MATCH (me)-[r1:ATE]->()<-[r2:ATE]-(you) \
                WHERE me.name = 'Michael' \
                WITH me, count(DISTINCT r1) AS H1, count(DISTINCT r2) AS H2, you \
                MATCH (me)-[r1:ATE]->()<-[r2:ATE]-(you) \
                RETURN me, you, \
                  sum((1 - abs(r1.times / H1 - r2.times / H2)) \
                    * (r1.times + r2.times) / (H1 + H2)) AS sum"
            .to_owned(),
        expected_headers: vec!["me".to_owned(), "you".to_owned(), "sum".to_owned()],
        expected_rows: vec![
            vec![
                "({name: 'Michael'})".to_owned(),
                "({name: 'Andres'})".to_owned(),
                "-7".to_owned(),
            ],
            vec![
                "({name: 'Michael'})".to_owned(),
                "({name: 'Peter'})".to_owned(),
                "0".to_owned(),
            ],
        ],
        result_order: ResultOrder::Any,
    };
    let mut backend = ObservedBackend::strict_cpu();
    let calls = backend.calls();
    let before = calls.snapshot();
    let stages_before = calls.graph_relation_stage_sequences()?.len();

    execute_strict_case(&case, &source, &mut backend)?;

    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            ..CallSnapshot::default()
        }
    );
    assert_eq!(
        &calls.graph_relation_stage_sequences()?[stages_before..],
        &[vec![
            ObservedSegmentedStage::Aggregate,
            ObservedSegmentedStage::Project,
            ObservedSegmentedStage::Unwind,
            ObservedSegmentedStage::Aggregate,
            ObservedSegmentedStage::Project,
        ]]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "real-Metal padding gate: root serializes all device runtime tests"]
fn real_metal_shared_group_and_reduction_inputs_ignore_capacity_padding() -> Result<()> {
    execute_shared_group_reduction_padding_regression(&mut ObservedBackend::real_metal()?)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "real-Metal relationship key/value gate: root serializes all device runtime tests"]
fn real_metal_proves_relationship_keys_dynamic_property_control_as_one_command() -> Result<()> {
    execute_relationship_key_value_control(&mut ObservedBackend::real_metal()?)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "real-Metal streaming gate: requires a Metal device"]
fn real_metal_reduces_more_than_one_million_logical_range_rows_without_an_engine_ceiling()
-> Result<()> {
    const ROWS: usize = 1_048_577;
    const QUERY: &str = "UNWIND range(1, 1048577) AS i WITH i LIMIT 1048577 RETURN sum(i) AS total";
    let graph = GraphStore::default();
    let mut backend = ObservedBackend::real_metal()?;
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();
    let before = calls.snapshot();
    let output = QueryEngine.execute(
        QUERY,
        &mut context_with_max_result_rows(&graph, BTreeMap::new(), Some(&backend), true, ROWS),
    )?;
    let values = output
        .result
        .batches
        .iter()
        .flat_map(|batch| batch.columns[0].values.iter())
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        [ResultValue::Scalar(ScalarValue::Integer(549_757_386_753))]
    );
    assert_eq!(
        call_delta(before, calls.snapshot()),
        CallSnapshot {
            segmented: 1,
            sealed_program: 1,
            range_source: 1,
            pre_aggregate_limit: 1,
            ..CallSnapshot::default()
        }
    );
    Ok(())
}

#[test]
#[ignore = "external native gate: requires the pinned openCypher TCK checkout"]
fn strict_cpu_reference_executes_exact_29_in_one_segmented_command() -> Result<()> {
    run_strict_suite(&mut ObservedBackend::strict_cpu())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "red real-Metal gate: requires fused resident source + aggregate + post-stage"]
fn real_metal_executes_exact_29_in_one_segmented_command_without_fallback() -> Result<()> {
    run_strict_suite(&mut ObservedBackend::real_metal()?)
}

#[cfg(not(all(feature = "accelerator", target_os = "macos")))]
#[test]
#[ignore = "real Metal fused aggregation acceptance must run on macOS with --features accelerator"]
fn real_metal_exact_29_gate_requires_metal() -> Result<()> {
    Err(Error::new(
        ErrorCode::GpuAdmissionFailure,
        "run the exact-29 fused aggregation gate on a real Metal device",
    ))
}
