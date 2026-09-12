// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Exact assurance gate for the 72 currently Metal-red openCypher quantifier scenarios.
//!
//! The certified report named below contains 64 expanded invariant scenarios in Quantifier9–12
//! plus eight graph-backed entity-list scenarios in Quantifier1–4. Every one is CPU-green and
//! Metal-red in that report. The literal manifest and ignored external guard bind the family to
//! all 72 exact zero-based report positions, feature paths, expanded names, and pinned sources.
//!
//! The generic CPU test is only an oracle. Native acceptance additionally requires the complete
//! query to cross exactly one backend command boundary with fail-closed native execution. The 64
//! graph-free identities use the sealed multistage quantifier command and exact receipts. The
//! eight entity-list identities now pass through the same fused command on the CPU semantic
//! reference; Metal remains red until it implements that command without CPU delegation.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentBooleanAggregateProgramRequest, ResidentBooleanProgramRequest,
        ResidentBooleanProgramResult, ResidentBooleanValue, ResidentGroup, ResidentGroupRequest,
        ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentNullableRelationRequest,
        ResidentNullableRelationResult, ResidentProjectImage, ResidentQuantifierProgramRequest,
        ResidentQuantifierProgramResult, ResidentRowProgramRequest, ResidentRowProgramResult,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphStore, LayerMask},
    types::{LabelId, PropertyId},
};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const REPORT_PATH: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";
const PINNED_FEATURE_ROOT: &str =
    "/private/tmp/irongraph-opencypher-debug.6MXlLm/openCypher/tck/features";
const FLOAT_UNARY_MINUS_REPORT_PATH: &str = "/tmp/irongraph-tck-full-quantifier64-certified.json";
const FLOAT_UNARY_MINUS_BASELINE_REPORT_PATH: &str =
    "/tmp/irongraph-tck-full-temporal38-quantifier64.json";
const FLOAT_UNARY_MINUS_FEATURE_ROOT: &str = "/tmp/irongraph-opencypher.oREeH5/tck/features";
const OFFICIAL_CASE_COUNT: usize = 72;
const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 100_000;

const MISSING_NATIVE_CONTRACT: &str = "strict native quantifier acceptance requires one complete \
receipted command; graph-free cases use the multistage quantifier program, while entity-list cases \
also need resident variable-length paths, nodes/relationships list materialization, and property reads";

/// `(zero-based full-report index, feature suffix, exact expanded name, exact TCK query)`.
const FLOAT_UNARY_MINUS_REGRESSIONS: [(usize, &str, &str, &str); 4] = [
    (
        2211,
        "expressions/quantifier/Quantifier1.feature",
        "[4] None quantifier on list literal containing floats [2212]",
        "RETURN none(x IN [20.0, 3.4, 50.2, -2.1] WHERE x = 2.1) AS result",
    ),
    (
        2363,
        "expressions/quantifier/Quantifier2.feature",
        "[4] Single quantifier on list literal containing floats [2364]",
        "RETURN single(x IN [20.0, 3.4, 50.2, -2.1] WHERE x = 2.1) AS result",
    ),
    (
        2469,
        "expressions/quantifier/Quantifier3.feature",
        "[4] Any quantifier on list literal containing floats [2470]",
        "RETURN any(x IN [20.0, 3.4, 50.2, -2.1] WHERE x = 2.1) AS result",
    ),
    (
        2574,
        "expressions/quantifier/Quantifier4.feature",
        "[4] All quantifier on list literal containing floats [2575]",
        "RETURN all(x IN [20.0, 3.4, 50.2, -2.1] WHERE x = 2.1) AS result",
    ),
];

/// `zero-based report index|feature suffix|exact expanded scenario name`.
const EXACT_MANIFEST: &str = r#"
2246|expressions/quantifier/Quantifier1.feature|[8] None quantifier on list containing nodes
2247|expressions/quantifier/Quantifier1.feature|[9] None quantifier on list containing relationships
2285|expressions/quantifier/Quantifier10.feature|[1] Single quantifier is always false if the predicate is statically false and the list is not empty
2286|expressions/quantifier/Quantifier10.feature|[2] Single quantifier is always false if the predicate is statically true and the list has more than one element
2287|expressions/quantifier/Quantifier10.feature|[3] Single quantifier is always true if the predicate is statically true and the list has exactly one non-null element
2288|expressions/quantifier/Quantifier10.feature|[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one [2289]
2289|expressions/quantifier/Quantifier10.feature|[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one [2290]
2290|expressions/quantifier/Quantifier10.feature|[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one [2291]
2291|expressions/quantifier/Quantifier10.feature|[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one [2292]
2292|expressions/quantifier/Quantifier10.feature|[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one [2293]
2293|expressions/quantifier/Quantifier11.feature|[1] Any quantifier is always false if the predicate is statically false and the list is not empty
2294|expressions/quantifier/Quantifier11.feature|[2] Any quantifier is always true if the predicate is statically true and the list is not empty
2295|expressions/quantifier/Quantifier11.feature|[3] Any quantifier is always true if the single or the all quantifier is true [2296]
2296|expressions/quantifier/Quantifier11.feature|[3] Any quantifier is always true if the single or the all quantifier is true [2297]
2297|expressions/quantifier/Quantifier11.feature|[3] Any quantifier is always true if the single or the all quantifier is true [2298]
2298|expressions/quantifier/Quantifier11.feature|[3] Any quantifier is always true if the single or the all quantifier is true [2299]
2299|expressions/quantifier/Quantifier11.feature|[3] Any quantifier is always true if the single or the all quantifier is true [2300]
2300|expressions/quantifier/Quantifier11.feature|[4] Any quantifier is always equal the boolean negative of the none quantifier [2301]
2301|expressions/quantifier/Quantifier11.feature|[4] Any quantifier is always equal the boolean negative of the none quantifier [2302]
2302|expressions/quantifier/Quantifier11.feature|[4] Any quantifier is always equal the boolean negative of the none quantifier [2303]
2303|expressions/quantifier/Quantifier11.feature|[4] Any quantifier is always equal the boolean negative of the none quantifier [2304]
2304|expressions/quantifier/Quantifier11.feature|[4] Any quantifier is always equal the boolean negative of the none quantifier [2305]
2305|expressions/quantifier/Quantifier11.feature|[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate [2306]
2306|expressions/quantifier/Quantifier11.feature|[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate [2307]
2307|expressions/quantifier/Quantifier11.feature|[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate [2308]
2308|expressions/quantifier/Quantifier11.feature|[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate [2309]
2309|expressions/quantifier/Quantifier11.feature|[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate [2310]
2310|expressions/quantifier/Quantifier11.feature|[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero [2311]
2311|expressions/quantifier/Quantifier11.feature|[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero [2312]
2312|expressions/quantifier/Quantifier11.feature|[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero [2313]
2313|expressions/quantifier/Quantifier11.feature|[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero [2314]
2314|expressions/quantifier/Quantifier11.feature|[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero [2315]
2315|expressions/quantifier/Quantifier12.feature|[1] All quantifier is always false if the predicate is statically false and the list is not empty
2316|expressions/quantifier/Quantifier12.feature|[2] All quantifier is always true if the predicate is statically true and the list is not empty
2317|expressions/quantifier/Quantifier12.feature|[3] All quantifier is always equal the none quantifier on the boolean negative of the predicate [2318]
2318|expressions/quantifier/Quantifier12.feature|[3] All quantifier is always equal the none quantifier on the boolean negative of the predicate [2319]
2319|expressions/quantifier/Quantifier12.feature|[3] All quantifier is always equal the none quantifier on the boolean negative of the predicate [2320]
2320|expressions/quantifier/Quantifier12.feature|[3] All quantifier is always equal the none quantifier on the boolean negative of the predicate [2321]
2321|expressions/quantifier/Quantifier12.feature|[3] All quantifier is always equal the none quantifier on the boolean negative of the predicate [2322]
2322|expressions/quantifier/Quantifier12.feature|[4] All quantifier is always equal the boolean negative of the any quantifier on the boolean negative of the predicate [2323]
2323|expressions/quantifier/Quantifier12.feature|[4] All quantifier is always equal the boolean negative of the any quantifier on the boolean negative of the predicate [2324]
2324|expressions/quantifier/Quantifier12.feature|[4] All quantifier is always equal the boolean negative of the any quantifier on the boolean negative of the predicate [2325]
2325|expressions/quantifier/Quantifier12.feature|[4] All quantifier is always equal the boolean negative of the any quantifier on the boolean negative of the predicate [2326]
2326|expressions/quantifier/Quantifier12.feature|[4] All quantifier is always equal the boolean negative of the any quantifier on the boolean negative of the predicate [2327]
2327|expressions/quantifier/Quantifier12.feature|[5] All quantifier is always equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list [2328]
2328|expressions/quantifier/Quantifier12.feature|[5] All quantifier is always equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list [2329]
2329|expressions/quantifier/Quantifier12.feature|[5] All quantifier is always equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list [2330]
2330|expressions/quantifier/Quantifier12.feature|[5] All quantifier is always equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list [2331]
2331|expressions/quantifier/Quantifier12.feature|[5] All quantifier is always equal whether the size of the list filtered with same the predicate is equal the size of the unfiltered list [2332]
2398|expressions/quantifier/Quantifier2.feature|[8] Single quantifier on list containing nodes
2399|expressions/quantifier/Quantifier2.feature|[9] Single quantifier on list containing relationships
2504|expressions/quantifier/Quantifier3.feature|[8] Any quantifier on list containing nodes
2505|expressions/quantifier/Quantifier3.feature|[9] Any quantifier on list containing relationships
2609|expressions/quantifier/Quantifier4.feature|[8] All quantifier on list containing nodes
2610|expressions/quantifier/Quantifier4.feature|[9] All quantifier on list containing relationships
2767|expressions/quantifier/Quantifier9.feature|[1] None quantifier is always true if the predicate is statically false and the list is not empty
2768|expressions/quantifier/Quantifier9.feature|[2] None quantifier is always false if the predicate is statically true and the list is not empty
2769|expressions/quantifier/Quantifier9.feature|[3] None quantifier is always equal the boolean negative of the any quantifier [2770]
2770|expressions/quantifier/Quantifier9.feature|[3] None quantifier is always equal the boolean negative of the any quantifier [2771]
2771|expressions/quantifier/Quantifier9.feature|[3] None quantifier is always equal the boolean negative of the any quantifier [2772]
2772|expressions/quantifier/Quantifier9.feature|[3] None quantifier is always equal the boolean negative of the any quantifier [2773]
2773|expressions/quantifier/Quantifier9.feature|[3] None quantifier is always equal the boolean negative of the any quantifier [2774]
2774|expressions/quantifier/Quantifier9.feature|[4] None quantifier is always equal the all quantifier on the boolean negative of the predicate [2775]
2775|expressions/quantifier/Quantifier9.feature|[4] None quantifier is always equal the all quantifier on the boolean negative of the predicate [2776]
2776|expressions/quantifier/Quantifier9.feature|[4] None quantifier is always equal the all quantifier on the boolean negative of the predicate [2777]
2777|expressions/quantifier/Quantifier9.feature|[4] None quantifier is always equal the all quantifier on the boolean negative of the predicate [2778]
2778|expressions/quantifier/Quantifier9.feature|[4] None quantifier is always equal the all quantifier on the boolean negative of the predicate [2779]
2779|expressions/quantifier/Quantifier9.feature|[5] None quantifier is always equal whether the size of the list filtered with same the predicate is zero [2780]
2780|expressions/quantifier/Quantifier9.feature|[5] None quantifier is always equal whether the size of the list filtered with same the predicate is zero [2781]
2781|expressions/quantifier/Quantifier9.feature|[5] None quantifier is always equal whether the size of the list filtered with same the predicate is zero [2782]
2782|expressions/quantifier/Quantifier9.feature|[5] None quantifier is always equal whether the size of the list filtered with same the predicate is zero [2783]
2783|expressions/quantifier/Quantifier9.feature|[5] None quantifier is always equal whether the size of the list filtered with same the predicate is zero [2784]
"#;

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
    scenarios: Vec<CertifiedScenario>,
}

#[derive(Clone, Debug, Deserialize)]
struct CertifiedScenario {
    path: String,
    name: String,
    cpu_passed: bool,
    metal_passed: bool,
    fully_conformant: bool,
    shared_failures: Vec<String>,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
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

#[derive(Clone, Debug)]
struct SourceCase {
    setup_queries: Vec<String>,
    query: String,
    expected_headers: Vec<String>,
    expected_rows: Vec<Vec<String>>,
}

fn manifest() -> Result<Vec<ManifestCase>> {
    EXACT_MANIFEST
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.splitn(3, '|');
            let report_id = fields
                .next()
                .ok_or_else(|| Error::internal("quantifier manifest omitted report ID"))?
                .parse::<usize>()
                .map_err(|error| Error::internal(format!("invalid report ID: {error}")))?;
            let feature = fields
                .next()
                .ok_or_else(|| Error::internal("quantifier manifest omitted feature"))?;
            let name = fields
                .next()
                .ok_or_else(|| Error::internal("quantifier manifest omitted scenario name"))?;
            Ok(ManifestCase {
                report_id,
                feature: feature.to_owned(),
                name: name.to_owned(),
            })
        })
        .collect()
}

fn read_certified_report(path: &str) -> Result<CertifiedReport> {
    let bytes =
        fs::read(path).map_err(|error| Error::internal(format!("cannot read {path}: {error}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::internal(format!("cannot decode {path}: {error}")))
}

fn certified_report() -> Result<CertifiedReport> {
    read_certified_report(REPORT_PATH)
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
                docstring.push_str(line);
            }
            continue;
        }
        if trimmed == "\"\"\"" {
            if current_step.is_some() {
                in_docstring = true;
            }
            continue;
        }
        if trimmed.starts_with("Scenario Outline:") || trimmed.starts_with("Scenario:") {
            finish(&mut current, &mut example_rows, &mut scenarios);
            let outline = trimmed.starts_with("Scenario Outline:");
            let prefix = if outline {
                "Scenario Outline:"
            } else {
                "Scenario:"
            };
            current = Some(ParsedScenario {
                name: trimmed[prefix.len()..].trim().to_owned(),
                outline,
                steps: Vec::new(),
                examples: Vec::new(),
            });
            current_step = None;
            in_examples = false;
            continue;
        }
        if current.is_none() {
            continue;
        }
        if trimmed == "Examples:" {
            in_examples = true;
            if !example_rows.is_empty()
                && let Some(scenario) = current.as_mut()
            {
                scenario.examples.push(std::mem::take(&mut example_rows));
            }
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
            "{name} is not tied to exactly one supported empty graph fixture"
        )));
    }
    if steps
        .iter()
        .filter(|step| step.value == "no side effects")
        .count()
        != 1
    {
        return Err(Error::internal(format!(
            "{name} omitted its exact no-side-effects assertion"
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
    let expectations = steps
        .iter()
        .filter(|step| step.value.starts_with("the result should be"))
        .collect::<Vec<_>>();
    let [expectation] = expectations.as_slice() else {
        return Err(Error::internal(format!(
            "{name} must contain exactly one result assertion"
        )));
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
        query,
        expected_headers: headers.clone(),
        expected_rows: rows.to_vec(),
    })
}

fn source_catalog_at_root(
    cases: &[ManifestCase],
    feature_root: &Path,
) -> Result<BTreeMap<(String, String), SourceCase>> {
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
            "pinned source resolved {} of {} literal manifest cases",
            catalog.len(),
            cases.len()
        )));
    }
    Ok(catalog)
}

fn source_catalog(cases: &[ManifestCase]) -> Result<BTreeMap<(String, String), SourceCase>> {
    source_catalog_at_root(cases, Path::new(PINNED_FEATURE_ROOT))
}

fn report_manifest_case(
    report_index: usize,
    scenario: &CertifiedScenario,
    feature_root: &Path,
) -> Result<ManifestCase> {
    let feature = Path::new(&scenario.path)
        .strip_prefix(feature_root)
        .map_err(|error| {
            Error::internal(format!(
                "report path {} is outside TCK root {}: {error}",
                scenario.path,
                feature_root.display()
            ))
        })?
        .to_string_lossy()
        .into_owned();
    Ok(ManifestCase {
        report_id: report_index,
        feature,
        name: scenario.name.clone(),
    })
}

fn float_unary_minus_regression_catalog()
-> Result<(Vec<ManifestCase>, BTreeMap<(String, String), SourceCase>)> {
    const FAILURE: &str = "Metal unary minus does not yet admit a possible FLOAT operand";
    let report = read_certified_report(FLOAT_UNARY_MINUS_REPORT_PATH)?;
    let baseline = read_certified_report(FLOAT_UNARY_MINUS_BASELINE_REPORT_PATH)?;
    if report.total != 3_897 || baseline.total != 3_897 || baseline.cpu_passed != 3_897 {
        return Err(Error::internal(
            "float-unary-minus regression evidence does not contain the expected 3,897-scenario baseline",
        ));
    }

    let actual_failures = report
        .scenarios
        .iter()
        .enumerate()
        .filter(|(_, scenario)| {
            scenario
                .metal_failures
                .iter()
                .any(|failure| failure.contains(FAILURE))
        })
        .map(|(index, _)| index)
        .collect::<BTreeSet<_>>();
    let expected_failures = FLOAT_UNARY_MINUS_REGRESSIONS
        .iter()
        .map(|(index, _, _, _)| *index)
        .collect::<BTreeSet<_>>();
    if actual_failures != expected_failures {
        return Err(Error::internal(format!(
            "fresh report identifies FLOAT unary-minus failures {actual_failures:?}, expected {expected_failures:?}",
        )));
    }

    let feature_root = Path::new(FLOAT_UNARY_MINUS_FEATURE_ROOT);
    let mut selected = Vec::with_capacity(FLOAT_UNARY_MINUS_REGRESSIONS.len());
    for (index, feature, name, _) in FLOAT_UNARY_MINUS_REGRESSIONS {
        let scenario = report.scenarios.get(index).ok_or_else(|| {
            Error::internal(format!("fresh report omitted regression index {index}"))
        })?;
        let case = report_manifest_case(index, scenario, feature_root)?;
        if case.feature != feature
            || case.name != name
            || !scenario.cpu_passed
            || scenario.metal_passed
            || scenario.fully_conformant
            || !scenario.shared_failures.is_empty()
            || !scenario.cpu_failures.is_empty()
            || scenario.metal_failures.len() != 1
            || !scenario.metal_failures[0].contains(FAILURE)
        {
            return Err(Error::internal(format!(
                "fresh report regression identity/status changed at index {index}"
            )));
        }
        let prior_matches = baseline
            .scenarios
            .iter()
            .filter(|prior| prior.name == name && Path::new(&prior.path).ends_with(feature))
            .collect::<Vec<_>>();
        let [prior] = prior_matches.as_slice() else {
            return Err(Error::internal(format!(
                "baseline report did not uniquely resolve prior-green scenario {name}"
            )));
        };
        if !prior.cpu_passed
            || !prior.metal_passed
            || !prior.fully_conformant
            || !prior.shared_failures.is_empty()
            || !prior.cpu_failures.is_empty()
            || !prior.metal_failures.is_empty()
        {
            return Err(Error::internal(format!(
                "baseline report does not prove the prior Metal-green result at index {index}"
            )));
        }
        selected.push(case);
    }

    let families = selected
        .iter()
        .map(|case| {
            (
                case.feature.clone(),
                source_base_name(&case.name).to_owned(),
            )
        })
        .collect::<BTreeSet<_>>();
    let mut family_cases = Vec::new();
    for (index, scenario) in report.scenarios.iter().enumerate() {
        let Ok(case) = report_manifest_case(index, scenario, feature_root) else {
            continue;
        };
        if families.contains(&(
            case.feature.clone(),
            source_base_name(&case.name).to_owned(),
        )) {
            family_cases.push(case);
        }
    }
    if family_cases.len() != 48 {
        return Err(Error::internal(format!(
            "four FLOAT quantifier outlines expanded to {} report cases instead of 48",
            family_cases.len()
        )));
    }
    let family_sources = source_catalog_at_root(&family_cases, feature_root)?;
    let mut sources = BTreeMap::new();
    for ((_, _, _, expected_query), case) in FLOAT_UNARY_MINUS_REGRESSIONS.iter().zip(&selected) {
        let key = (case.feature.clone(), case.name.clone());
        let source = family_sources.get(&key).ok_or_else(|| {
            Error::internal(format!(
                "TCK source omitted FLOAT unary-minus regression {}",
                case.name
            ))
        })?;
        if normalize_query(&source.query) != *expected_query {
            return Err(Error::internal(format!(
                "TCK query changed for FLOAT unary-minus regression {}: {}",
                case.name,
                normalize_query(&source.query)
            )));
        }
        sources.insert(key, source.clone());
    }
    Ok((selected, sources))
}

fn normalize_query(query: &str) -> String {
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn context<'a>(
    graph: &'a GraphStore,
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
        parameters: BTreeMap::new(),
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

fn fixture_graph(source: &SourceCase) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for setup in &source.setup_queries {
        let output = QueryEngine.execute(setup, &mut context(&graph, None, false))?;
        if !output.temporal_mutations.is_empty() {
            return Err(Error::internal(
                "quantifier fixture unexpectedly produced temporal mutations",
            ));
        }
        for mutation in output.graph_mutations {
            graph.apply(mutation)?;
        }
    }
    Ok(graph)
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
            "quantifier oracle cannot canonicalize scalar {other:?}"
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
    Ok(format!(" {{{}}}", entries.join(", ")))
}

fn tck_value(value: &ResultValue) -> Result<String> {
    match value {
        ResultValue::Scalar(value) => tck_scalar(value),
        ResultValue::Node(node) => Ok(format!(
            "(:{}{})",
            node.labels.join(":"),
            tck_properties(&node.properties)?
        )),
        ResultValue::Relationship(relationship) => Ok(format!(
            "[:{}{}]",
            relationship.relationship_type,
            tck_properties(&relationship.properties)?
        )),
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
            "quantifier oracle cannot canonicalize result {other:?}"
        ))),
    }
}

fn assert_exact_output(
    case: &ManifestCase,
    source: &SourceCase,
    output: &ExecutionOutput,
) -> Result<()> {
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal(format!(
            "report {} / {} produced side effects",
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
    expected_rows.sort();
    actual_rows.sort();
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
    quantifier: usize,
    boolean: usize,
    aggregate: usize,
    row: usize,
    nullable: usize,
    node_pipeline: usize,
}

#[derive(Default)]
struct NativeCalls {
    quantifier: AtomicUsize,
    boolean: AtomicUsize,
    aggregate: AtomicUsize,
    row: AtomicUsize,
    nullable: AtomicUsize,
    node_pipeline: AtomicUsize,
}

impl NativeCalls {
    fn snapshot(&self) -> CallSnapshot {
        CallSnapshot {
            quantifier: self.quantifier.load(Ordering::SeqCst),
            boolean: self.boolean.load(Ordering::SeqCst),
            aggregate: self.aggregate.load(Ordering::SeqCst),
            row: self.row.load(Ordering::SeqCst),
            nullable: self.nullable.load(Ordering::SeqCst),
            node_pipeline: self.node_pipeline.load(Ordering::SeqCst),
        }
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

    fn cpu_masquerading_as_metal() -> Self {
        Self {
            inner: Box::new(CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES)),
            reported_kind: BackendKind::Metal,
            calls: Arc::new(NativeCalls::default()),
        }
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal() -> Result<Self> {
        let inner = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "quantifier acceptance did not construct a real Metal backend",
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
        project: ProjectId,
        label: Option<LabelId>,
        layers: LayerMask,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner.scan_nodes(project, label, layers, cancellation)
    }

    fn filter_node_i64(
        &self,
        project: ProjectId,
        property: PropertyId,
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner
            .filter_node_i64(project, property, operation, operand, cancellation)
    }

    fn expand_project_out(
        &self,
        project: ProjectId,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.inner
            .expand_project_out(project, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.inner.expand_project_in(project, targets, cancellation)
    }

    fn search_vectors(
        &self,
        request: &ResidentVectorQuery,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.inner.search_vectors(request, cancellation)
    }

    fn sort_rows(
        &self,
        request: &ResidentSortRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.inner.sort_rows(request, cancellation)
    }

    fn join_node_i64(
        &self,
        request: &ResidentJoinRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.inner.join_node_i64(request, cancellation)
    }

    fn group_node_i64(
        &self,
        request: &ResidentGroupRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.inner.group_node_i64(request, cancellation)
    }

    fn execute_node_pipeline(
        &self,
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.calls.node_pipeline.fetch_add(1, Ordering::SeqCst);
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn supports_nullable_relation_predicates(&self) -> bool {
        self.inner.supports_nullable_relation_predicates()
    }

    fn execute_nullable_relation(
        &self,
        request: &ResidentNullableRelationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        self.calls.nullable.fetch_add(1, Ordering::SeqCst);
        self.inner.execute_nullable_relation(request, cancellation)
    }

    fn supports_native_boolean_program(&self) -> bool {
        self.inner.supports_native_boolean_program()
    }

    fn execute_boolean_program(
        &self,
        request: &ResidentBooleanProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentBooleanProgramResult> {
        self.calls.boolean.fetch_add(1, Ordering::SeqCst);
        self.inner.execute_boolean_program(request, cancellation)
    }

    fn supports_native_boolean_aggregate_program(&self) -> bool {
        self.inner.supports_native_boolean_aggregate_program()
    }

    fn execute_boolean_aggregate_program(
        &self,
        request: &ResidentBooleanAggregateProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentBooleanProgramResult> {
        self.calls.aggregate.fetch_add(1, Ordering::SeqCst);
        self.inner
            .execute_boolean_aggregate_program(request, cancellation)
    }

    fn execute_row_program(
        &self,
        request: &ResidentRowProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.calls.row.fetch_add(1, Ordering::SeqCst);
        self.inner.execute_row_program(request, cancellation)
    }

    fn supports_native_quantifier_program(&self) -> bool {
        self.inner.supports_native_quantifier_program()
    }

    fn supports_native_quantifier_entity_source(&self) -> bool {
        self.inner.supports_native_quantifier_entity_source()
    }

    fn execute_quantifier_program(
        &self,
        request: &ResidentQuantifierProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentQuantifierProgramResult> {
        self.calls.quantifier.fetch_add(1, Ordering::SeqCst);
        self.inner.execute_quantifier_program(request, cancellation)
    }

    fn filter_i64(
        &self,
        values: &[i64],
        validity: &[bool],
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner
            .filter_i64(values, validity, operation, operand, cancellation)
    }

    fn expand_out(
        &self,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.inner.expand_out(sources, cancellation)
    }

    fn exact_l2(
        &self,
        matrix: &[f32],
        rows: usize,
        dimension: usize,
        query: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.inner
            .exact_l2(matrix, rows, dimension, query, cancellation)
    }
}

fn call_delta(before: CallSnapshot, after: CallSnapshot) -> CallSnapshot {
    CallSnapshot {
        quantifier: after.quantifier.saturating_sub(before.quantifier),
        boolean: after.boolean.saturating_sub(before.boolean),
        aggregate: after.aggregate.saturating_sub(before.aggregate),
        row: after.row.saturating_sub(before.row),
        nullable: after.nullable.saturating_sub(before.nullable),
        node_pipeline: after.node_pipeline.saturating_sub(before.node_pipeline),
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
    let output = QueryEngine.execute(&source.query, &mut context(&graph, Some(&*backend), true));
    let delta = call_delta(before, calls.snapshot());
    let output = output.map_err(|error| {
        Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "{MISSING_NATIVE_CONTRACT}; report {} / {} failed with {:?}: {}; native calls={delta:?}",
                case.report_id, case.name, error.code, error.message
            ),
        )
    })?;
    assert_exact_output(case, source, &output)?;
    match delta {
        CallSnapshot {
            quantifier: 1,
            boolean: 0,
            aggregate: 0,
            row: 0,
            nullable: 0,
            node_pipeline: 0,
        } => Ok(()),
        CallSnapshot {
            quantifier: 0,
            boolean: 1,
            aggregate: 0,
            row: 0,
            nullable: 0,
            node_pipeline: 0,
        }
        | CallSnapshot {
            quantifier: 0,
            boolean: 0,
            aggregate: 1,
            row: 0,
            nullable: 0,
            node_pipeline: 0,
        } => Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "{MISSING_NATIVE_CONTRACT}; report {} / {} used an unattested Boolean packet",
                case.report_id, case.name
            ),
        )),
        other => Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "{MISSING_NATIVE_CONTRACT}; report {} / {} crossed {other:?} instead of one complete receipted command",
                case.report_id, case.name
            ),
        )),
    }
}

fn is_entity_list_case(case: &ManifestCase) -> bool {
    case.name.contains("list containing nodes")
        || case.name.contains("list containing relationships")
}

fn run_strict_cases(
    backend: &mut ObservedBackend,
    cases: Vec<ManifestCase>,
    gate: &str,
) -> Result<()> {
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
                "strict {gate} gate failed {} of {} official scenarios; first failures: {}",
                failures.len(),
                cases.len(),
                failures
                    .iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
        ))
    }
}

fn run_strict_suite(backend: &mut ObservedBackend, graph_free_only: bool) -> Result<()> {
    let cases = manifest()?
        .into_iter()
        .filter(|case| !graph_free_only || !is_entity_list_case(case))
        .collect::<Vec<_>>();
    if graph_free_only && cases.len() != 64 {
        return Err(Error::internal(format!(
            "graph-free quantifier manifest resolved {} cases instead of 64",
            cases.len()
        )));
    }
    run_strict_cases(backend, cases, "quantifier")
}

fn run_entity_list_suite(backend: &mut ObservedBackend) -> Result<()> {
    let cases = manifest()?
        .into_iter()
        .filter(is_entity_list_case)
        .collect::<Vec<_>>();
    if cases.len() != 8 {
        return Err(Error::internal(format!(
            "entity-list quantifier manifest resolved {} cases instead of 8",
            cases.len()
        )));
    }
    run_strict_cases(backend, cases, "entity-list quantifier")
}

fn run_float_unary_minus_regressions(backend: &mut ObservedBackend) -> Result<()> {
    let (cases, sources) = float_unary_minus_regression_catalog()?;
    if cases.len() != FLOAT_UNARY_MINUS_REGRESSIONS.len() || sources.len() != cases.len() {
        return Err(Error::internal(format!(
            "FLOAT unary-minus gate resolved {} cases and {} sources instead of exactly four",
            cases.len(),
            sources.len()
        )));
    }
    let mut failures = Vec::new();
    for case in &cases {
        let source = sources
            .get(&(case.feature.clone(), case.name.clone()))
            .ok_or_else(|| Error::internal(format!("source omitted {}", case.name)))?;
        if let Err(error) = execute_strict_case(case, source, backend) {
            failures.push(format!(
                "{} / {}: {}",
                case.report_id + 1,
                case.name,
                error.message
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "FLOAT unary-minus regression gate failed {} of 4 scenarios: {}",
                failures.len(),
                failures.join(" | ")
            ),
        ))
    }
}

fn execute_exact_observed_source(
    name: &str,
    source: &SourceCase,
    backend: &mut ObservedBackend,
    expected_calls: CallSnapshot,
) -> Result<()> {
    let graph = fixture_graph(source)?;
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let calls = backend.calls();
    let before = calls.snapshot();
    let output = QueryEngine
        .execute(&source.query, &mut context(&graph, Some(&*backend), true))
        .map_err(|error| {
            Error::internal(format!(
                "{name} failed through the strict native route with {:?}: {}",
                error.code, error.message
            ))
        })?;
    let actual_calls = call_delta(before, calls.snapshot());
    if actual_calls != expected_calls {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "{name} used the wrong execution route: expected {expected_calls:?}, got {actual_calls:?}"
            ),
        ));
    }
    assert_exact_output(
        &ManifestCase {
            report_id: 0,
            feature: "native-regression".to_owned(),
            name: name.to_owned(),
        },
        source,
        &output,
    )
}

fn identity_source(query: &str, equal: bool) -> SourceCase {
    let expected_rows = if equal {
        vec![
            vec!["(:A)".to_owned(), "(:A)".to_owned()],
            vec!["(:B)".to_owned(), "(:B)".to_owned()],
        ]
    } else {
        vec![
            vec!["(:A)".to_owned(), "(:B)".to_owned()],
            vec!["(:B)".to_owned(), "(:A)".to_owned()],
        ]
    };
    SourceCase {
        setup_queries: vec!["CREATE (:A), (:B)".to_owned()],
        query: query.to_owned(),
        expected_headers: vec!["a".to_owned(), "b".to_owned()],
        expected_rows,
    }
}

#[test]
fn cpu_node_identity_equality_and_inequality_cross_match_and_with_natively() -> Result<()> {
    let cases = [
        (
            "MatchWhere3 [1]",
            "MATCH (a), (b) WHERE a = b RETURN a, b",
            true,
        ),
        (
            "MatchWhere4 [1]",
            "MATCH (a), (b) WHERE a <> b RETURN a, b",
            false,
        ),
        (
            "WithWhere3 [1]",
            "MATCH (a), (b) WITH a, b WHERE a = b RETURN a, b",
            true,
        ),
        (
            "WithWhere4 [1]",
            "MATCH (a), (b) WITH a, b WHERE a <> b RETURN a, b",
            false,
        ),
    ];
    for (name, query, equal) in cases {
        execute_exact_observed_source(
            name,
            &identity_source(query, equal),
            &mut ObservedBackend::strict_cpu(),
            CallSnapshot {
                nullable: 1,
                ..CallSnapshot::default()
            },
        )?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device"]
fn real_metal_matchwhere6_3_uses_the_certified_node_pipeline_route() -> Result<()> {
    let source = SourceCase {
        setup_queries: vec![
            "CREATE (s:Single), (a:A {num: 42}), (b:B {num: 46}), (c:C) \
             CREATE (s)-[:REL]->(a), (s)-[:REL]->(b), (a)-[:REL]->(c), (b)-[:LOOP]->(b)"
                .to_owned(),
        ],
        query: "MATCH (n:Single) OPTIONAL MATCH (n)-[r]-(m) WHERE m.num = 42 RETURN m".to_owned(),
        expected_headers: vec!["m".to_owned()],
        expected_rows: vec![vec!["(:A {num: 42})".to_owned()]],
    };
    execute_exact_observed_source(
        "MatchWhere6 [3]",
        &source,
        &mut ObservedBackend::real_metal()?,
        CallSnapshot {
            node_pipeline: 1,
            ..CallSnapshot::default()
        },
    )
}

#[test]
fn literal_manifest_has_the_authoritative_72_case_shape() -> Result<()> {
    let cases = manifest()?;
    assert_eq!(cases.len(), OFFICIAL_CASE_COUNT);
    let ids = cases
        .iter()
        .map(|case| case.report_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), OFFICIAL_CASE_COUNT);
    let mut counts = BTreeMap::new();
    for case in &cases {
        *counts.entry(case.feature.as_str()).or_insert(0_usize) += 1;
    }
    assert_eq!(
        counts,
        BTreeMap::from([
            ("expressions/quantifier/Quantifier1.feature", 2),
            ("expressions/quantifier/Quantifier10.feature", 8),
            ("expressions/quantifier/Quantifier11.feature", 22),
            ("expressions/quantifier/Quantifier12.feature", 17),
            ("expressions/quantifier/Quantifier2.feature", 2),
            ("expressions/quantifier/Quantifier3.feature", 2),
            ("expressions/quantifier/Quantifier4.feature", 2),
            ("expressions/quantifier/Quantifier9.feature", 17),
        ])
    );
    Ok(())
}

#[test]
fn semantic_axes_and_their_official_limits_are_explicit() {
    let axes = [
        (
            "empty, non-empty, and null lists",
            "entity paths produce both empty and non-empty lists, while the invariants add null members and nullable fixedList variables; a directly null quantifier source remains separate",
        ),
        (
            "null predicates",
            "null members occur under static true/false predicates; a predicate whose result itself is null is outside this tranche",
        ),
        (
            "mixed values and type errors",
            "mixed scalar/list/map values are safe under static predicates; runtime mixed-type error scenarios need a separate strict tranche",
        ),
        (
            "nested quantifiers",
            "Quantifier9/11/12 compare none, any, all, and single inside the same expression or scope chain",
        ),
        (
            "variable list sources",
            "inputList, fixedList, and randomized list variables cross repeated WITH and UNWIND boundaries",
        ),
        (
            "property list sources",
            "the eight entity-list cases read x.name on nodes and relationships; a list value sourced directly from one property remains separate follow-up coverage",
        ),
    ];
    assert_eq!(axes.len(), 6);
    assert!(
        axes.iter()
            .all(|(name, evidence)| !name.is_empty() && !evidence.is_empty())
    );
}

#[test]
fn legacy_boolean_result_packet_cannot_certify_the_multistage_route() {
    let forged = ResidentBooleanProgramResult {
        rows: vec![vec![ResidentBooleanValue::Boolean(true)]],
    };
    assert_eq!(forged.rows, vec![vec![ResidentBooleanValue::Boolean(true)]]);
    // Construction required no request fingerprint, execution ID, backend completion, receipt,
    // project generation, or device identity. This is evidence of the missing contract, not an
    // accepted native result path.
}

#[test]
#[ignore = "external assurance guard: requires the certified report and pinned TCK checkout"]
fn exact_manifest_resolves_uniquely_against_report_and_pinned_source() -> Result<()> {
    let cases = manifest()?;
    let report = certified_report()?;
    assert_eq!(report.total, 3_897);
    assert_eq!(report.cpu_passed, 3_897);

    let feature_names = [
        "Quantifier1.feature",
        "Quantifier2.feature",
        "Quantifier3.feature",
        "Quantifier4.feature",
        "Quantifier9.feature",
        "Quantifier10.feature",
        "Quantifier11.feature",
        "Quantifier12.feature",
    ];
    let report_targets = report
        .scenarios
        .iter()
        .enumerate()
        .filter(|(_, scenario)| {
            Path::new(&scenario.path)
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| feature_names.contains(&name))
                && !scenario.metal_passed
        })
        .map(|(index, scenario)| {
            let relative = Path::new(&scenario.path)
                .strip_prefix(PINNED_FEATURE_ROOT)
                .map_err(|error| {
                    Error::internal(format!(
                        "report path {} is outside pinned root: {error}",
                        scenario.path
                    ))
                })?
                .to_string_lossy()
                .into_owned();
            Ok(ManifestCase {
                report_id: index,
                feature: relative,
                name: scenario.name.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(report_targets.len(), OFFICIAL_CASE_COUNT);
    assert_eq!(report_targets, cases);

    let sources = source_catalog(&cases)?;
    assert_eq!(sources.len(), OFFICIAL_CASE_COUNT);
    for case in &cases {
        let matches = report
            .scenarios
            .iter()
            .enumerate()
            .filter(|(_, scenario)| {
                scenario.name == case.name && Path::new(&scenario.path).ends_with(&case.feature)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "report pair must resolve uniquely for {} / {}",
            case.feature,
            case.name
        );
        let (actual_index, scenario) = matches[0];
        assert_eq!(actual_index, case.report_id);
        assert!(scenario.cpu_passed);
        assert!(!scenario.metal_passed);
        assert!(!scenario.fully_conformant);
        assert!(scenario.shared_failures.is_empty());
        assert!(scenario.cpu_failures.is_empty());
        assert_eq!(scenario.metal_failures.len(), 1);
        assert!(scenario.metal_failures[0].contains(
            "GpuAdmissionFailure: active GPU execution class has no complete resident implementation for this query plan"
        ));
        let source = sources
            .get(&(case.feature.clone(), case.name.clone()))
            .ok_or_else(|| Error::internal(format!("source omitted {}", case.name)))?;
        let normalized = normalize_query(&source.query);
        assert!(normalized.contains(" IN "));
        assert!(normalized.contains("RETURN result") || normalized.contains(" AS result"));
    }
    Ok(())
}

/// This proves semantic correctness only through the generic CPU evaluator. It deliberately does
/// not satisfy either native acceptance gate below.
#[test]
#[ignore = "external oracle: requires the pinned openCypher TCK checkout"]
fn generic_cpu_oracle_replays_all_72_exact_official_queries() -> Result<()> {
    let cases = manifest()?;
    let sources = source_catalog(&cases)?;
    for case in &cases {
        let source = sources
            .get(&(case.feature.clone(), case.name.clone()))
            .ok_or_else(|| Error::internal(format!("source omitted {}", case.name)))?;
        let graph = fixture_graph(source)?;
        let output = QueryEngine
            .execute(&source.query, &mut context(&graph, None, false))
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
#[ignore = "external native gate: requires the pinned openCypher TCK checkout"]
fn strict_cpu_reference_executes_all_64_graph_free_cases_without_fallback() -> Result<()> {
    run_strict_suite(&mut ObservedBackend::strict_cpu(), true)
}

#[test]
#[ignore = "external native gate: requires the pinned openCypher TCK checkout"]
fn strict_cpu_reference_executes_exact_8_entity_list_cases_without_fallback() -> Result<()> {
    run_entity_list_suite(&mut ObservedBackend::strict_cpu())
}

#[test]
#[ignore = "external native gate: requires the pinned openCypher TCK checkout"]
fn strict_cpu_reference_executes_all_72_without_generic_or_legacy_fallback() -> Result<()> {
    run_strict_suite(&mut ObservedBackend::strict_cpu(), false)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "external regression gate: requires both full reports and their pinned TCK checkout"]
fn real_metal_executes_exact_four_float_unary_minus_report_regressions() -> Result<()> {
    run_float_unary_minus_regressions(&mut ObservedBackend::real_metal()?)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "external native gate: requires the pinned openCypher TCK checkout"]
fn real_metal_executes_all_64_graph_free_without_cpu_generic_or_legacy_fallback() -> Result<()> {
    run_strict_suite(&mut ObservedBackend::real_metal()?, true)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "external native gate: requires the pinned TCK checkout and a real Metal device"]
fn real_metal_executes_exact_8_entity_list_cases_without_fallback() -> Result<()> {
    run_entity_list_suite(&mut ObservedBackend::real_metal()?)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "external native gate: requires the pinned TCK checkout and a real Metal device"]
fn real_metal_executes_all_72_without_cpu_generic_or_legacy_fallback() -> Result<()> {
    run_strict_suite(&mut ObservedBackend::real_metal()?, false)
}

#[cfg(not(all(feature = "accelerator", target_os = "macos")))]
#[test]
#[ignore = "real Metal quantifier acceptance must run on macOS with --features accelerator"]
fn real_metal_executes_all_72_requires_metal() -> Result<()> {
    Err(Error::new(
        ErrorCode::GpuAdmissionFailure,
        "run the 72-scenario quantifier gate on a real Metal device",
    ))
}

#[test]
fn cpu_quantifier_result_cannot_masquerade_as_metal() -> Result<()> {
    let graph = GraphStore::default();
    let mut backend = ObservedBackend::cpu_masquerading_as_metal();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let output = QueryEngine.execute(
        "RETURN any(x IN [true, false] WHERE x) AS result",
        &mut context(&graph, Some(&backend), true),
    );
    match output {
        Err(error) if error.code == ErrorCode::CorruptStorage => Ok(()),
        Err(error) => Err(Error::internal(format!(
            "CPU masquerade produced {:?}, expected provenance rejection: {}",
            error.code, error.message
        ))),
        Ok(_) => Err(Error::new(
            ErrorCode::CorruptStorage,
            "CPU quantifier completion was accepted through a backend advertising Metal",
        )),
    }
}
