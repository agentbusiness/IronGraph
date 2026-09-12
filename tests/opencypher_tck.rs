// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Full-corpus openCypher TCK inventory and parser gate.
//!
//! The upstream files are intentionally loaded with a small TCK-specific structural reader. The
//! corpus contains valid Cypher strings in table cells that generic Gherkin parsers reject as
//! malformed escaped text.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use chrono::Offset;
use irongraph::{
    Bookmark, ErrorCode, ProjectId, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ProcedureCatalog, ProcedureDefinition, ProcedureField,
        ProcedureValueType, QueryEngine, QueryResult, ResultValue, bind_with_parameters,
        bind_with_procedures, parse, plan,
    },
    gpu::{BackendKind, CpuBackend, ExecutionBackend, ResidentProjectImage},
    graph::{GraphMutation, GraphStore},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;
#[cfg(all(feature = "accelerator", target_os = "macos"))]
use serde::Serialize;

const EXPECTED_FEATURE_FILES: usize = 220;
const EXPECTED_EXPANDED_SCENARIOS: usize = 3_897;

#[derive(Clone, Debug)]
struct Step {
    value: String,
    docstring: Option<String>,
    table: Vec<Vec<String>>,
}

#[derive(Clone, Debug)]
struct Scenario {
    path: PathBuf,
    name: String,
    outline: bool,
    steps: Vec<Step>,
    examples: Vec<Vec<Vec<String>>>,
}

#[derive(Clone, Debug)]
struct ExpandedScenario {
    path: PathBuf,
    name: String,
    steps: Vec<Step>,
}

fn feature_files(root: &Path) -> irongraph::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_feature_files(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_feature_files(root: &Path, files: &mut Vec<PathBuf>) -> irongraph::Result<()> {
    for entry in fs::read_dir(root).map_err(|error| {
        irongraph::Error::internal(format!(
            "cannot read TCK directory {}: {error}",
            root.display()
        ))
    })? {
        let entry = entry.map_err(|error| irongraph::Error::internal(error.to_string()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_feature_files(&path, files)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "feature")
        {
            files.push(path);
        }
    }
    Ok(())
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
                    // Gherkin tables define only `\\|`, `\\\\`, and `\\n`. Preserve every
                    // other escape byte-for-byte because TCK cells themselves contain Cypher
                    // string literals such as `\\'` and `\\u01FF`.
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

#[test]
fn tck_table_rows_follow_gherkin_escape_rules_without_consuming_cypher_escapes()
-> irongraph::Result<()> {
    assert_eq!(
        table_row(r"| a\|b | c\\d | e\nf | q\'r |"),
        Some(vec![
            "a|b".to_owned(),
            r"c\d".to_owned(),
            "e\nf".to_owned(),
            r"q\'r".to_owned(),
        ])
    );

    let row = table_row(r#"| literal | 'a\\\\bcn5t\'"\\\\//\\\\"\'' |"#)
        .ok_or_else(|| irongraph::Error::internal("expected a TCK table row"))?;
    let parsed = TckValueParser::parse(&row[1])?;
    let TckValue::String(parsed) = parsed else {
        return Err(irongraph::Error::internal(
            "expected the table cell to parse as a TCK string",
        ));
    };
    assert_eq!(parsed, r#"a\bcn5t'"\//\"'"#);
    Ok(())
}

fn parse_feature(path: &Path) -> irongraph::Result<Vec<Scenario>> {
    let source = fs::read_to_string(path)
        .map_err(|error| irongraph::Error::internal(format!("{}: {error}", path.display())))?;
    let mut scenarios = Vec::new();
    let mut current: Option<Scenario> = None;
    let mut current_step: Option<usize> = None;
    let mut background_steps = Vec::new();
    let mut background_step: Option<usize> = None;
    let mut in_background = false;
    let mut in_docstring = false;
    let mut in_examples = false;
    let mut example_rows = Vec::new();

    let finish = |current: &mut Option<Scenario>,
                  example_rows: &mut Vec<Vec<String>>,
                  scenarios: &mut Vec<Scenario>| {
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
            } else {
                let step = if in_background {
                    background_step.and_then(|index| background_steps.get_mut(index))
                } else {
                    current_step.and_then(|index| {
                        current
                            .as_mut()
                            .and_then(|scenario| scenario.steps.get_mut(index))
                    })
                };
                if let Some(step) = step {
                    let doc = step.docstring.get_or_insert_with(String::new);
                    if !doc.is_empty() {
                        doc.push('\n');
                    }
                    doc.push_str(line);
                }
            }
            continue;
        }
        if trimmed == "\"\"\"" {
            if (in_background && background_step.is_some())
                || (!in_background && current_step.is_some())
            {
                in_docstring = true;
            }
            continue;
        }
        if trimmed == "Background:" {
            finish(&mut current, &mut example_rows, &mut scenarios);
            in_background = true;
            background_step = None;
            current_step = None;
            in_examples = false;
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
            current = Some(Scenario {
                path: path.to_owned(),
                name: trimmed[prefix.len()..].trim().to_owned(),
                outline,
                steps: background_steps.clone(),
                examples: Vec::new(),
            });
            current_step = None;
            in_background = false;
            background_step = None;
            in_examples = false;
            continue;
        }
        if current.is_none() && !in_background {
            continue;
        }
        if trimmed == "Examples:" {
            in_examples = true;
            if !example_rows.is_empty() {
                if let Some(scenario) = current.as_mut() {
                    scenario.examples.push(std::mem::take(&mut example_rows));
                }
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
            let step = Step {
                value,
                docstring: None,
                table: Vec::new(),
            };
            if in_background {
                background_steps.push(step);
                background_step = Some(background_steps.len() - 1);
            } else if let Some(scenario) = current.as_mut() {
                scenario.steps.push(step);
                current_step = Some(scenario.steps.len() - 1);
            }
            continue;
        }
        if let Some(row) = table_row(trimmed) {
            if in_examples {
                example_rows.push(row);
            } else if in_background {
                if let Some(step) =
                    background_step.and_then(|index| background_steps.get_mut(index))
                {
                    step.table.push(row);
                }
            } else if let Some(step) = current_step.and_then(|index| {
                current
                    .as_mut()
                    .and_then(|scenario| scenario.steps.get_mut(index))
            }) {
                step.table.push(row);
            }
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

fn expand(scenario: Scenario, ordinal: &mut usize) -> Vec<ExpandedScenario> {
    let mut expanded = Vec::new();
    for table in &scenario.examples {
        if let Some(headers) = table.first() {
            for row in table.iter().skip(1) {
                *ordinal += 1;
                expanded.push(ExpandedScenario {
                    path: scenario.path.clone(),
                    name: format!("{} [{}]", scenario.name, *ordinal),
                    steps: scenario
                        .steps
                        .iter()
                        .map(|step| Step {
                            value: substitute(&step.value, headers, row),
                            docstring: step
                                .docstring
                                .as_ref()
                                .map(|doc| substitute(doc, headers, row)),
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
                        })
                        .collect(),
                });
            }
        }
    }
    if expanded.is_empty() || !scenario.outline {
        *ordinal += 1;
        if expanded.is_empty() {
            expanded.push(ExpandedScenario {
                path: scenario.path,
                name: scenario.name,
                steps: scenario.steps,
            });
        }
    }
    expanded
}

fn load_scenarios(root: &Path) -> irongraph::Result<Vec<ExpandedScenario>> {
    let mut scenarios = Vec::new();
    let mut ordinal = 0;
    for path in feature_files(root)? {
        for scenario in parse_feature(&path)? {
            scenarios.extend(expand(scenario, &mut ordinal));
        }
    }
    Ok(scenarios)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedErrorPhase {
    Compile,
    Runtime,
    Any,
}

fn expected_error_phase(steps: &[Step], query_step: usize) -> Option<ExpectedErrorPhase> {
    steps.iter().skip(query_step + 1).find_map(|step| {
        if !step.value.starts_with("a ") && !step.value.starts_with("an ") {
            return None;
        }
        if !step.value.contains(" should be raised at ") {
            return None;
        }
        if step.value.contains("compile time") {
            Some(ExpectedErrorPhase::Compile)
        } else if step.value.contains("runtime") {
            Some(ExpectedErrorPhase::Runtime)
        } else if step.value.contains("any time") {
            Some(ExpectedErrorPhase::Any)
        } else {
            None
        }
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GraphFixture {
    Empty,
    Named(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ErrorExpectation {
    kind: String,
    phase: ExpectedErrorPhase,
    detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ResultExpectation {
    Empty,
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        ordered: bool,
        ignore_list_order: bool,
    },
    Error(ErrorExpectation),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SideEffects {
    nodes_added: i64,
    nodes_removed: i64,
    relationships_added: i64,
    relationships_removed: i64,
    properties_added: i64,
    properties_removed: i64,
    labels_added: i64,
    labels_removed: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TckOperation {
    query: String,
    control: bool,
    expectation: Option<ResultExpectation>,
    side_effects: Option<SideEffects>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureProcedureType {
    Boolean,
    Integer,
    Float,
    Number,
    String,
}

impl FixtureProcedureType {
    fn parse(value: &str) -> irongraph::Result<Self> {
        match value {
            "BOOLEAN" => Ok(Self::Boolean),
            "INTEGER" => Ok(Self::Integer),
            "FLOAT" => Ok(Self::Float),
            "NUMBER" => Ok(Self::Number),
            "STRING" => Ok(Self::String),
            other => Err(irongraph::Error::invalid_data(format!(
                "unsupported TCK fixture procedure type `{other}`"
            ))),
        }
    }

    fn accepts(self, value: &TckValue, nullable: bool) -> bool {
        match value {
            TckValue::Null => nullable,
            TckValue::Boolean(_) => self == Self::Boolean,
            TckValue::Integer(_) => matches!(self, Self::Integer | Self::Float | Self::Number),
            TckValue::Float(_) => matches!(self, Self::Float | Self::Number),
            TckValue::String(_) => self == Self::String,
            TckValue::List(_)
            | TckValue::Map(_)
            | TckValue::Node { .. }
            | TckValue::Relationship { .. }
            | TckValue::Path(_) => false,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Boolean => "BOOLEAN",
            Self::Integer => "INTEGER",
            Self::Float => "FLOAT",
            Self::Number => "NUMBER",
            Self::String => "STRING",
        }
    }

    const fn procedure_type(self) -> ProcedureValueType {
        match self {
            Self::Boolean => ProcedureValueType::Boolean,
            Self::Integer => ProcedureValueType::Integer,
            Self::Float => ProcedureValueType::Float,
            Self::Number => ProcedureValueType::Number,
            Self::String => ProcedureValueType::String,
        }
    }
}

fn fixture_procedure_catalog(case: &TckCase) -> irongraph::Result<ProcedureCatalog> {
    let mut catalog = ProcedureCatalog::default();
    for fixture in &case.procedures {
        let fields = |fields: &[FixtureProcedureField]| {
            fields
                .iter()
                .map(|field| {
                    ProcedureField::new(
                        field.name.clone(),
                        field.field_type.procedure_type(),
                        field.nullable,
                    )
                })
                .collect::<irongraph::Result<Vec<_>>>()
        };
        let rows = fixture
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| TckValueParser::parse(value)?.into_parameter())
                    .collect::<irongraph::Result<Vec<_>>>()
            })
            .collect::<irongraph::Result<Vec<_>>>()?;
        catalog.register(ProcedureDefinition::new(
            fixture.name.clone(),
            fields(&fixture.inputs)?,
            fields(&fixture.outputs)?,
            rows,
        )?)?;
    }
    Ok(catalog)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FixtureProcedureField {
    name: String,
    field_type: FixtureProcedureType,
    nullable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FixtureProcedure {
    name: String,
    inputs: Vec<FixtureProcedureField>,
    outputs: Vec<FixtureProcedureField>,
    /// Positional values in declared input-then-output order. The TCK table header is validated
    /// and removed here before the typed execution-scoped catalog is constructed.
    rows: Vec<Vec<String>>,
}

impl FixtureProcedure {
    fn describe(&self) -> String {
        let fields = |fields: &[FixtureProcedureField]| {
            fields
                .iter()
                .map(|field| {
                    format!(
                        "{} :: {}{}",
                        field.name,
                        field.field_type.name(),
                        if field.nullable { "?" } else { "" }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "{}({}) :: ({}) [{} fixture rows]",
            self.name,
            fields(&self.inputs),
            fields(&self.outputs),
            self.rows.len()
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TckCase {
    fixture: GraphFixture,
    setup_queries: Vec<String>,
    parameters: BTreeMap<String, String>,
    procedures: Vec<FixtureProcedure>,
    operations: Vec<TckOperation>,
}

fn is_fixture_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn fixture_procedure_fields(value: &str) -> irongraph::Result<Vec<FixtureProcedureField>> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|field| {
            let (name, field_type) = field.split_once("::").ok_or_else(|| {
                irongraph::Error::invalid_data(format!(
                    "invalid TCK fixture procedure field `{field}`"
                ))
            })?;
            let name = name.trim();
            if !is_fixture_identifier(name) {
                return Err(irongraph::Error::invalid_data(format!(
                    "invalid TCK fixture procedure field name `{name}`"
                )));
            }
            let field_type = field_type.trim();
            let (field_type, nullable) = field_type
                .strip_suffix('?')
                .map_or((field_type, false), |field_type| (field_type.trim(), true));
            Ok(FixtureProcedureField {
                name: name.to_owned(),
                field_type: FixtureProcedureType::parse(field_type)?,
                nullable,
            })
        })
        .collect()
}

fn parse_fixture_procedure(step: &Step) -> irongraph::Result<FixtureProcedure> {
    let declaration = step
        .value
        .strip_prefix("there exists a procedure ")
        .and_then(|value| value.strip_suffix(':'))
        .map(str::trim)
        .ok_or_else(|| {
            irongraph::Error::invalid_data(format!(
                "invalid TCK fixture procedure declaration `{}`",
                step.value
            ))
        })?;
    let signature_separator = declaration.find(") :: (").ok_or_else(|| {
        irongraph::Error::invalid_data(format!(
            "TCK fixture procedure declaration has no output signature `{declaration}`"
        ))
    })?;
    if !declaration.ends_with(')') {
        return Err(irongraph::Error::invalid_data(format!(
            "TCK fixture procedure declaration has an unterminated output signature `{declaration}`"
        )));
    }
    let callable = &declaration[..signature_separator];
    let output_start = signature_separator + ") :: (".len();
    let outputs = &declaration[output_start..declaration.len() - 1];
    let arguments_start = callable.find('(').ok_or_else(|| {
        irongraph::Error::invalid_data(format!(
            "TCK fixture procedure declaration has no input signature `{declaration}`"
        ))
    })?;
    let name = callable[..arguments_start].trim();
    if name.is_empty() || !name.split('.').all(is_fixture_identifier) {
        return Err(irongraph::Error::invalid_data(format!(
            "invalid TCK fixture procedure name `{name}`"
        )));
    }
    let inputs = fixture_procedure_fields(&callable[arguments_start + 1..])?;
    let outputs = fixture_procedure_fields(outputs)?;
    let fields = inputs.iter().chain(&outputs).collect::<Vec<_>>();
    let rows = if fields.is_empty() {
        if !step.table.is_empty() {
            return Err(irongraph::Error::invalid_data(format!(
                "output-free TCK fixture procedure `{name}` has a non-empty table"
            )));
        }
        Vec::new()
    } else {
        let (headers, rows) = step.table.split_first().ok_or_else(|| {
            irongraph::Error::invalid_data(format!(
                "TCK fixture procedure `{name}` has no table header"
            ))
        })?;
        let expected_headers = fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>();
        if headers.iter().map(String::as_str).collect::<Vec<_>>() != expected_headers {
            return Err(irongraph::Error::invalid_data(format!(
                "TCK fixture procedure `{name}` expected table columns {expected_headers:?}, got {headers:?}"
            )));
        }
        for (row_index, row) in rows.iter().enumerate() {
            if row.len() != fields.len() {
                return Err(irongraph::Error::invalid_data(format!(
                    "TCK fixture procedure `{name}` row {} has {} values, expected {}",
                    row_index + 1,
                    row.len(),
                    fields.len()
                )));
            }
            for (column_index, (value, field)) in row.iter().zip(&fields).enumerate() {
                let parsed = TckValueParser::parse(value)?;
                if !field.field_type.accepts(&parsed, field.nullable) {
                    return Err(irongraph::Error::invalid_data(format!(
                        "TCK fixture procedure `{name}` row {}, column {} (`{}`) does not match {}{}",
                        row_index + 1,
                        column_index + 1,
                        field.name,
                        field.field_type.name(),
                        if field.nullable { "?" } else { "" }
                    )));
                }
            }
        }
        rows.to_vec()
    };
    Ok(FixtureProcedure {
        name: name.to_owned(),
        inputs,
        outputs,
        rows,
    })
}

fn query_from_step(step: &Step, prefix: &str) -> irongraph::Result<String> {
    if let Some(query) = &step.docstring {
        return Ok(query.trim().to_owned());
    }
    let query = step
        .value
        .strip_prefix(prefix)
        .map(str::trim)
        .unwrap_or_default();
    if query.is_empty() {
        return Err(irongraph::Error::invalid_data(format!(
            "TCK step `{}` has no query text",
            step.value
        )));
    }
    Ok(query.to_owned())
}

fn parse_error_expectation(step: &Step) -> irongraph::Result<ErrorExpectation> {
    let value = step
        .value
        .strip_prefix("a ")
        .or_else(|| step.value.strip_prefix("an "))
        .ok_or_else(|| irongraph::Error::invalid_data("TCK error step has no article"))?;
    let (kind, rest) = value.split_once(" should be raised at ").ok_or_else(|| {
        irongraph::Error::invalid_data(format!("invalid TCK error step `{}`", step.value))
    })?;
    let (phase, detail) = rest.split_once(':').ok_or_else(|| {
        irongraph::Error::invalid_data(format!("invalid TCK error phase `{}`", step.value))
    })?;
    let phase = match phase.trim() {
        "compile time" => ExpectedErrorPhase::Compile,
        "runtime" => ExpectedErrorPhase::Runtime,
        "any time" => ExpectedErrorPhase::Any,
        other => {
            return Err(irongraph::Error::invalid_data(format!(
                "unknown TCK error phase `{other}`"
            )));
        }
    };
    Ok(ErrorExpectation {
        kind: kind.trim().to_owned(),
        phase,
        detail: detail.trim().to_owned(),
    })
}

fn result_expectation(step: &Step) -> irongraph::Result<ResultExpectation> {
    if step.value == "the result should be empty" {
        return Ok(ResultExpectation::Empty);
    }
    let ordered = step.value.contains("in order");
    let ignore_list_order = step.value.contains("ignoring element order for lists");
    if !step.value.starts_with("the result should be") {
        return Err(irongraph::Error::invalid_data(format!(
            "invalid TCK result step `{}`",
            step.value
        )));
    }
    let (headers, rows) = step.table.split_first().ok_or_else(|| {
        irongraph::Error::invalid_data("TCK result table is missing its header row")
    })?;
    if rows.iter().any(|row| row.len() != headers.len()) {
        return Err(irongraph::Error::invalid_data(
            "TCK result table has inconsistent column counts",
        ));
    }
    Ok(ResultExpectation::Table {
        headers: headers.clone(),
        rows: rows.to_vec(),
        ordered,
        ignore_list_order,
    })
}

fn side_effects(step: &Step) -> irongraph::Result<SideEffects> {
    let mut effects = SideEffects::default();
    for row in &step.table {
        if row.len() != 2 {
            return Err(irongraph::Error::invalid_data(
                "TCK side-effect table must have exactly two columns",
            ));
        }
        let value = row[1].parse::<i64>().map_err(|error| {
            irongraph::Error::invalid_data(format!(
                "TCK side-effect value `{}` is not an integer: {error}",
                row[1]
            ))
        })?;
        match row[0].trim() {
            "+nodes" => effects.nodes_added = value,
            "-nodes" => effects.nodes_removed = value,
            "+relationships" => effects.relationships_added = value,
            "-relationships" => effects.relationships_removed = value,
            "+properties" => effects.properties_added = value,
            "-properties" => effects.properties_removed = value,
            "+labels" => effects.labels_added = value,
            "-labels" => effects.labels_removed = value,
            other => {
                return Err(irongraph::Error::invalid_data(format!(
                    "unknown TCK side-effect `{other}`"
                )));
            }
        }
    }
    Ok(effects)
}

fn last_operation_mut(case: &mut TckCase) -> irongraph::Result<&mut TckOperation> {
    case.operations
        .last_mut()
        .ok_or_else(|| irongraph::Error::invalid_data("TCK assertion has no preceding query"))
}

fn parse_case(scenario: &ExpandedScenario) -> irongraph::Result<TckCase> {
    let mut case = TckCase {
        fixture: GraphFixture::Empty,
        setup_queries: Vec::new(),
        parameters: BTreeMap::new(),
        procedures: Vec::new(),
        operations: Vec::new(),
    };
    for step in &scenario.steps {
        match step.value.as_str() {
            "an empty graph" | "any graph" => case.fixture = GraphFixture::Empty,
            "having executed:" | "after having executed:" => {
                case.setup_queries
                    .push(query_from_step(step, "having executed:")?);
            }
            "parameters are:" | "parameter values are:" => {
                for row in &step.table {
                    if row.len() != 2 {
                        return Err(irongraph::Error::invalid_data(
                            "TCK parameter table must have exactly two columns",
                        ));
                    }
                    case.parameters.insert(row[0].clone(), row[1].clone());
                }
            }
            "the result should be empty" => {
                last_operation_mut(&mut case)?.expectation = Some(ResultExpectation::Empty);
            }
            "no side effects" => {
                last_operation_mut(&mut case)?.side_effects = Some(SideEffects::default());
            }
            "the side effects should be:" => {
                last_operation_mut(&mut case)?.side_effects = Some(side_effects(step)?);
            }
            _ if step.value.starts_with("the ") && step.value.ends_with(" graph") => {
                let name = step
                    .value
                    .strip_prefix("the ")
                    .and_then(|value| value.strip_suffix(" graph"))
                    .ok_or_else(|| irongraph::Error::invalid_data("invalid named graph step"))?;
                case.fixture = GraphFixture::Named(name.to_owned());
            }
            _ if step.value.starts_with("there exists a procedure ") => {
                case.procedures.push(parse_fixture_procedure(step)?);
            }
            _ if step.value.starts_with("executing control query:") => {
                case.operations.push(TckOperation {
                    query: query_from_step(step, "executing control query:")?,
                    control: true,
                    expectation: None,
                    side_effects: None,
                });
            }
            _ if step.value.starts_with("executing query:") => {
                case.operations.push(TckOperation {
                    query: query_from_step(step, "executing query:")?,
                    control: false,
                    expectation: None,
                    side_effects: None,
                });
            }
            _ if step.value.starts_with("the result should be") => {
                last_operation_mut(&mut case)?.expectation = Some(result_expectation(step)?);
            }
            _ if (step.value.starts_with("a ") || step.value.starts_with("an "))
                && step.value.contains(" should be raised at ") =>
            {
                last_operation_mut(&mut case)?.expectation =
                    Some(ResultExpectation::Error(parse_error_expectation(step)?));
            }
            other => {
                return Err(irongraph::Error::invalid_data(format!(
                    "unimplemented TCK step in {} / {}: `{other}`",
                    scenario.path.display(),
                    scenario.name
                )));
            }
        }
    }
    if case.operations.is_empty() {
        return Err(irongraph::Error::invalid_data(format!(
            "TCK scenario {} / {} has no query",
            scenario.path.display(),
            scenario.name
        )));
    }
    if case
        .operations
        .iter()
        .any(|operation| operation.expectation.is_none())
    {
        return Err(irongraph::Error::invalid_data(format!(
            "TCK scenario {} / {} has a query with no expectation",
            scenario.path.display(),
            scenario.name
        )));
    }
    Ok(case)
}

#[test]
fn fixture_procedure_declaration_preserves_typed_signature_and_rows() -> irongraph::Result<()> {
    let step = Step {
        value: "there exists a procedure test.my.proc(name :: STRING?, id :: INTEGER?) :: (city :: STRING?, country_code :: INTEGER?):".to_owned(),
        docstring: None,
        table: vec![
            vec!["name", "id", "city", "country_code"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            vec!["'Stefan'", "1", "'Berlin'", "49"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        ],
    };
    let procedure = parse_fixture_procedure(&step)?;
    assert_eq!(procedure.name, "test.my.proc");
    assert_eq!(
        procedure.inputs,
        vec![
            FixtureProcedureField {
                name: "name".to_owned(),
                field_type: FixtureProcedureType::String,
                nullable: true,
            },
            FixtureProcedureField {
                name: "id".to_owned(),
                field_type: FixtureProcedureType::Integer,
                nullable: true,
            },
        ]
    );
    assert_eq!(
        procedure.outputs,
        vec![
            FixtureProcedureField {
                name: "city".to_owned(),
                field_type: FixtureProcedureType::String,
                nullable: true,
            },
            FixtureProcedureField {
                name: "country_code".to_owned(),
                field_type: FixtureProcedureType::Integer,
                nullable: true,
            },
        ]
    );
    assert_eq!(
        procedure.rows,
        vec![vec![
            "'Stefan'".to_owned(),
            "1".to_owned(),
            "'Berlin'".to_owned(),
            "49".to_owned(),
        ]]
    );
    Ok(())
}

#[test]
fn fixture_procedure_declaration_accepts_empty_output_relation() -> irongraph::Result<()> {
    let procedure = parse_fixture_procedure(&Step {
        value: "there exists a procedure test.doNothing() :: ():".to_owned(),
        docstring: None,
        table: Vec::new(),
    })?;
    assert_eq!(procedure.name, "test.doNothing");
    assert!(procedure.inputs.is_empty());
    assert!(procedure.outputs.is_empty());
    assert!(procedure.rows.is_empty());
    Ok(())
}

#[test]
fn fixture_procedure_declaration_rejects_table_type_mismatch() -> irongraph::Result<()> {
    let error = parse_fixture_procedure(&Step {
        value: "there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: STRING?):"
            .to_owned(),
        docstring: None,
        table: vec![
            vec!["in".to_owned(), "out".to_owned()],
            vec!["true".to_owned(), "'invalid input'".to_owned()],
        ],
    })
    .err()
    .ok_or_else(|| {
        irongraph::Error::internal("BOOLEAN fixture value unexpectedly satisfied INTEGER")
    })?;
    assert!(error.message.contains("does not match INTEGER?"));
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TckPathDirection {
    Outgoing,
    Incoming,
    Undirected,
}

#[derive(Clone, Debug)]
struct TckPathNode {
    labels: BTreeSet<String>,
    properties: BTreeMap<String, TckValue>,
}

#[derive(Clone, Debug)]
struct TckPathRelationship {
    direction: TckPathDirection,
    relationship_type: Option<String>,
    properties: BTreeMap<String, TckValue>,
}

#[derive(Clone, Debug)]
struct TckPath {
    nodes: Vec<TckPathNode>,
    relationships: Vec<TckPathRelationship>,
}

#[derive(Clone, Debug)]
enum TckValue {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    List(Vec<TckValue>),
    Map(BTreeMap<String, TckValue>),
    Node {
        labels: BTreeSet<String>,
        properties: BTreeMap<String, TckValue>,
    },
    Relationship {
        relationship_type: Option<String>,
        properties: BTreeMap<String, TckValue>,
    },
    Path(TckPath),
}

impl TckValue {
    fn into_parameter(self) -> irongraph::Result<ResultValue> {
        match self {
            Self::Null => Ok(ResultValue::Scalar(ScalarValue::Null)),
            Self::Boolean(value) => Ok(ResultValue::Scalar(ScalarValue::Boolean(value))),
            Self::Integer(value) => Ok(ResultValue::Scalar(ScalarValue::Integer(value))),
            Self::Float(value) => Ok(ResultValue::Scalar(ScalarValue::Float(value.into()))),
            Self::String(value) => Ok(ResultValue::Scalar(ScalarValue::String(Arc::from(value)))),
            Self::List(values) => values
                .into_iter()
                .map(Self::into_parameter)
                .collect::<irongraph::Result<Vec<_>>>()
                .map(ResultValue::List),
            Self::Map(values) => values
                .into_iter()
                .map(|(name, value)| Ok((name, value.into_parameter()?)))
                .collect::<irongraph::Result<BTreeMap<_, _>>>()
                .map(ResultValue::Map),
            Self::Node { .. } | Self::Relationship { .. } | Self::Path(_) => Err(
                irongraph::Error::invalid_data("TCK parameters may not be graph elements or paths"),
            ),
        }
    }

    fn matches_actual(&self, actual: &ResultValue, ignore_list_order: bool) -> bool {
        match (self, actual) {
            (Self::Null, ResultValue::Scalar(ScalarValue::Null)) => true,
            (Self::Boolean(expected), ResultValue::Scalar(ScalarValue::Boolean(actual))) => {
                expected == actual
            }
            (Self::Integer(expected), ResultValue::Scalar(ScalarValue::Integer(actual))) => {
                expected == actual
            }
            (Self::Float(expected), ResultValue::Scalar(ScalarValue::Float(actual))) => {
                expected.is_nan() && actual.is_nan() || expected == &actual.0
            }
            (Self::String(expected), ResultValue::Scalar(ScalarValue::String(actual))) => {
                expected == actual.as_ref()
            }
            // The TCK table grammar has no temporal-literal notation.  It represents temporal
            // expected values as quoted ISO text, while a Cypher implementation must return a
            // typed temporal scalar.  Compare that scalar with the standard external literal;
            // do not stringify arbitrary query results or accept a host-side substitute.
            (Self::String(expected), ResultValue::Scalar(actual)) => {
                tck_temporal_literal(actual).is_some_and(|actual| expected == &actual)
            }
            (Self::List(expected), ResultValue::List(actual)) => {
                list_matches(expected, actual, ignore_list_order)
            }
            (Self::Map(expected), ResultValue::Map(actual)) => {
                expected.len() == actual.len()
                    && expected.iter().all(|(name, expected)| {
                        actual.get(name).is_some_and(|actual| {
                            expected.matches_actual(actual, ignore_list_order)
                        })
                    })
            }
            (Self::Node { labels, properties }, ResultValue::Node(actual)) => {
                labels.len() == actual.labels.len()
                    && labels
                        .iter()
                        .all(|label| actual.labels.iter().any(|actual| actual == label))
                    && properties.len() == actual.properties.len()
                    && properties.iter().all(|(name, expected)| {
                        actual.properties.get(name).is_some_and(|actual| {
                            ResultValue::from_property(actual.clone()).is_ok_and(|actual| {
                                expected.matches_actual(&actual, ignore_list_order)
                            })
                        })
                    })
            }
            (
                Self::Relationship {
                    relationship_type,
                    properties,
                },
                ResultValue::Relationship(actual),
            ) => {
                relationship_type
                    .as_ref()
                    .is_none_or(|expected| expected == &actual.relationship_type)
                    && properties.len() == actual.properties.len()
                    && properties.iter().all(|(name, expected)| {
                        actual.properties.get(name).is_some_and(|actual| {
                            ResultValue::from_property(actual.clone()).is_ok_and(|actual| {
                                expected.matches_actual(&actual, ignore_list_order)
                            })
                        })
                    })
            }
            (Self::Path(expected), actual @ ResultValue::Path { .. }) => {
                expected.matches_actual(actual, ignore_list_order)
            }
            _ => false,
        }
    }
}

impl TckPath {
    fn matches_actual(&self, actual: &ResultValue, ignore_list_order: bool) -> bool {
        let ResultValue::Path {
            nodes,
            relationships,
        } = actual
        else {
            return false;
        };
        if self.nodes.len() != nodes.len()
            || self.relationships.len() != relationships.len()
            || nodes.len() != relationships.len().saturating_add(1)
        {
            return false;
        }
        if !self.nodes.iter().zip(nodes).all(|(expected, actual)| {
            expected.labels.len() == actual.labels.len()
                && expected
                    .labels
                    .iter()
                    .all(|label| actual.labels.iter().any(|actual| actual == label))
                && expected.properties.len() == actual.properties.len()
                && expected.properties.iter().all(|(name, expected)| {
                    actual.properties.get(name).is_some_and(|actual| {
                        ResultValue::from_property(actual.clone())
                            .is_ok_and(|actual| expected.matches_actual(&actual, ignore_list_order))
                    })
                })
        }) {
            return false;
        }
        self.relationships
            .iter()
            .enumerate()
            .all(|(index, expected)| {
                let actual = &relationships[index];
                let source = nodes[index].id;
                let target = nodes[index + 1].id;
                let direction_matches = match expected.direction {
                    TckPathDirection::Outgoing => {
                        actual.source == source && actual.target == target
                    }
                    TckPathDirection::Incoming => {
                        actual.target == source && actual.source == target
                    }
                    TckPathDirection::Undirected => {
                        actual.source == source && actual.target == target
                            || actual.target == source && actual.source == target
                    }
                };
                direction_matches
                    && expected
                        .relationship_type
                        .as_ref()
                        .is_none_or(|relationship_type| {
                            relationship_type == &actual.relationship_type
                        })
                    && expected.properties.len() == actual.properties.len()
                    && expected.properties.iter().all(|(name, expected)| {
                        actual.properties.get(name).is_some_and(|actual| {
                            ResultValue::from_property(actual.clone()).is_ok_and(|actual| {
                                expected.matches_actual(&actual, ignore_list_order)
                            })
                        })
                    })
            })
    }
}

fn tck_temporal_literal(value: &ScalarValue) -> Option<String> {
    match value {
        ScalarValue::Date(days) => {
            let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?;
            epoch
                .checked_add_signed(chrono::Duration::days(i64::from(*days)))
                .map(|date| date.format("%Y-%m-%d").to_string())
        }
        ScalarValue::LocalTime(nanos) => tck_time_literal(*nanos),
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => Some(format!(
            "{}{}",
            tck_time_literal(*nanos)?,
            tck_offset_literal(*offset_seconds)?
        )),
        ScalarValue::LocalDateTime { seconds, nanos } => {
            let value = chrono::DateTime::<chrono::Utc>::from_timestamp(*seconds, *nanos)?;
            Some(tck_datetime_literal(value.date_naive(), value.time()))
        }
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            let value = chrono::DateTime::<chrono::Utc>::from_timestamp(*seconds, *nanos)?;
            if timezone.as_ref() == "UTC" {
                return Some(format!(
                    "{}Z",
                    tck_datetime_literal(value.date_naive(), value.time())
                ));
            }
            if let Ok(zone) = timezone.parse::<chrono_tz::Tz>() {
                let local = value.with_timezone(&zone);
                return Some(format!(
                    "{}{}[{}]",
                    tck_datetime_literal(local.date_naive(), local.time()),
                    tck_offset_literal(local.offset().fix().local_minus_utc())?,
                    timezone,
                ));
            }
            let offset = tck_fixed_offset_timezone(timezone)
                .or_else(|| timezone.parse::<chrono::FixedOffset>().ok())?;
            let local = value.with_timezone(&offset);
            Some(format!(
                "{}{}",
                tck_datetime_literal(local.date_naive(), local.time()),
                tck_offset_literal(offset.local_minus_utc())?,
            ))
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => Some(tck_duration_literal(*months, *days, *seconds, *nanos)),
        _ => None,
    }
}

fn tck_time_literal(nanos: i64) -> Option<String> {
    const NANOS_PER_SECOND: i64 = 1_000_000_000;
    const NANOS_PER_DAY: i64 = 86_400 * NANOS_PER_SECOND;
    if !(0..NANOS_PER_DAY).contains(&nanos) {
        return None;
    }
    let total_seconds = nanos / NANOS_PER_SECOND;
    let fraction = u32::try_from(nanos % NANOS_PER_SECOND).ok()?;
    let hour = total_seconds / 3_600;
    let minute = (total_seconds % 3_600) / 60;
    let second = total_seconds % 60;
    let mut result = format!("{hour:02}:{minute:02}");
    if second != 0 || fraction != 0 {
        result.push_str(&format!(":{second:02}"));
    }
    if fraction != 0 {
        let fraction = format!("{fraction:09}");
        result.push('.');
        result.push_str(fraction.trim_end_matches('0'));
    }
    Some(result)
}

fn tck_offset_literal(offset_seconds: i32) -> Option<String> {
    if offset_seconds == 0 {
        return Some("Z".to_owned());
    }
    let absolute = offset_seconds.unsigned_abs();
    if absolute >= 86_400 {
        return None;
    }
    let hour = absolute / 3_600;
    let minute = (absolute % 3_600) / 60;
    let second = absolute % 60;
    let sign = if offset_seconds < 0 { '-' } else { '+' };
    if second == 0 {
        Some(format!("{sign}{hour:02}:{minute:02}"))
    } else {
        Some(format!("{sign}{hour:02}:{minute:02}:{second:02}"))
    }
}

fn tck_datetime_literal(date: chrono::NaiveDate, time: chrono::NaiveTime) -> String {
    use chrono::Timelike;

    let nanos =
        i64::from(time.num_seconds_from_midnight()) * 1_000_000_000 + i64::from(time.nanosecond());
    format!(
        "{}T{}",
        date.format("%Y-%m-%d"),
        tck_time_literal(nanos).unwrap_or_default()
    )
}

/// Chrono's string parser accepts RFC 3339 minute offsets but not every valid Cypher offset
/// spelling with second precision. Decode the engine's fixed-offset label directly so TCK's
/// quoted temporal literal still compares against the typed returned value.
fn tck_fixed_offset_timezone(value: &str) -> Option<chrono::FixedOffset> {
    let bytes = value.as_bytes();
    let (&sign, digits) = bytes.split_first()?;
    let negative = match sign {
        b'+' => false,
        b'-' => true,
        _ => return None,
    };
    let decimal = |start: usize, width: usize| -> Option<u32> {
        let input = digits.get(start..start.checked_add(width)?)?;
        input.iter().try_fold(0_u32, |value, digit| match digit {
            b'0'..=b'9' => value.checked_mul(10)?.checked_add(u32::from(*digit - b'0')),
            _ => None,
        })
    };
    let (hours, minutes, seconds) = match digits.len() {
        2 => (decimal(0, 2)?, 0, 0),
        4 => (decimal(0, 2)?, decimal(2, 2)?, 0),
        5 if digits[2] == b':' => (decimal(0, 2)?, decimal(3, 2)?, 0),
        6 => (decimal(0, 2)?, decimal(2, 2)?, decimal(4, 2)?),
        8 if digits[2] == b':' && digits[5] == b':' => {
            (decimal(0, 2)?, decimal(3, 2)?, decimal(6, 2)?)
        }
        _ => return None,
    };
    if hours > 18 || minutes > 59 || seconds > 59 || (hours == 18 && (minutes != 0 || seconds != 0))
    {
        return None;
    }
    let absolute = hours
        .checked_mul(3_600)?
        .checked_add(minutes.checked_mul(60)?)?
        .checked_add(seconds)?;
    let signed = i32::try_from(absolute).ok()?;
    chrono::FixedOffset::east_opt(if negative { -signed } else { signed })
}

fn tck_duration_literal(months: i64, days: i64, seconds: i64, nanos: i32) -> String {
    const NANOS_PER_SECOND: i128 = 1_000_000_000;
    const NANOS_PER_MINUTE: i128 = 60 * NANOS_PER_SECOND;
    const NANOS_PER_HOUR: i128 = 60 * NANOS_PER_MINUTE;

    let mut result = String::from("P");
    let years = months / 12;
    let remaining_months = months % 12;
    if years != 0 {
        result.push_str(&format!("{years}Y"));
    }
    if remaining_months != 0 {
        result.push_str(&format!("{remaining_months}M"));
    }
    if days != 0 {
        result.push_str(&format!("{days}D"));
    }

    let total_nanos = i128::from(seconds) * NANOS_PER_SECOND + i128::from(nanos);
    let hours = total_nanos / NANOS_PER_HOUR;
    let after_hours = total_nanos % NANOS_PER_HOUR;
    let minutes = after_hours / NANOS_PER_MINUTE;
    let after_minutes = after_hours % NANOS_PER_MINUTE;
    let whole_seconds = after_minutes / NANOS_PER_SECOND;
    let fractional_nanos = after_minutes % NANOS_PER_SECOND;
    let has_time = hours != 0 || minutes != 0 || whole_seconds != 0 || fractional_nanos != 0;
    if has_time {
        result.push('T');
        if hours != 0 {
            result.push_str(&format!("{hours}H"));
        }
        if minutes != 0 {
            result.push_str(&format!("{minutes}M"));
        }
        if whole_seconds != 0 || fractional_nanos != 0 {
            if fractional_nanos == 0 {
                result.push_str(&format!("{whole_seconds}S"));
            } else {
                let sign = if whole_seconds < 0 || fractional_nanos < 0 {
                    "-"
                } else {
                    ""
                };
                let fraction = format!("{:09}", fractional_nanos.unsigned_abs())
                    .trim_end_matches('0')
                    .to_owned();
                result.push_str(&format!(
                    "{sign}{}.{}S",
                    whole_seconds.unsigned_abs(),
                    fraction
                ));
            }
        }
    } else if result == "P" {
        result.push_str("T0S");
    }
    result
}

#[test]
fn tck_quoted_temporal_expectations_require_matching_typed_values() -> irongraph::Result<()> {
    let expected_date = TckValue::String("2015-07-21".to_owned());
    assert!(expected_date.matches_actual(&ResultValue::Scalar(ScalarValue::Date(16_637)), false));
    assert!(!expected_date.matches_actual(&ResultValue::Scalar(ScalarValue::Date(16_638)), false));

    let expected_time = TckValue::String("21:40".to_owned());
    assert!(expected_time.matches_actual(
        &ResultValue::Scalar(ScalarValue::LocalTime(78_000_000_000_000)),
        false
    ));

    let parsed = chrono::DateTime::parse_from_rfc3339("2015-07-21T21:40:32.142+02:00")
        .map_err(|error| irongraph::Error::invalid_data(error.to_string()))?;
    let expected_datetime =
        TckValue::String("2015-07-21T21:40:32.142+02:00[Europe/Stockholm]".to_owned());
    assert!(expected_datetime.matches_actual(
        &ResultValue::Scalar(ScalarValue::ZonedDateTime {
            seconds: parsed.timestamp(),
            nanos: parsed.timestamp_subsec_nanos(),
            timezone: Arc::from("Europe/Stockholm"),
        }),
        false,
    ));

    let expected_duration = TckValue::String("P12Y5M14DT16H13M10S".to_owned());
    assert!(expected_duration.matches_actual(
        &ResultValue::Scalar(ScalarValue::Duration {
            months: 149,
            days: 14,
            seconds: 58_390,
            nanos: 0,
        }),
        false,
    ));
    let expected_negative_duration = TckValue::String("PT-23H-59M-59.9S".to_owned());
    assert!(expected_negative_duration.matches_actual(
        &ResultValue::Scalar(ScalarValue::Duration {
            months: 0,
            days: 0,
            seconds: -86_400,
            nanos: 100_000_000,
        }),
        false,
    ));

    // Ordinary Cypher strings remain ordinary strings; the temporal branch is not a wildcard.
    assert!(expected_date.matches_actual(
        &ResultValue::Scalar(ScalarValue::String(Arc::from("2015-07-21"))),
        false,
    ));
    assert!(
        !expected_date.matches_actual(&ResultValue::Scalar(ScalarValue::Integer(16_637)), false,)
    );
    Ok(())
}

fn list_matches(expected: &[TckValue], actual: &[ResultValue], ignore_order: bool) -> bool {
    if expected.len() != actual.len() {
        return false;
    }
    if !ignore_order {
        return expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| expected.matches_actual(actual, false));
    }
    let mut used = vec![false; actual.len()];
    expected.iter().all(|expected| {
        actual.iter().enumerate().any(|(index, actual)| {
            !used[index] && expected.matches_actual(actual, true) && {
                used[index] = true;
                true
            }
        })
    })
}

struct TckValueParser<'a> {
    source: &'a str,
    offset: usize,
}

impl<'a> TckValueParser<'a> {
    fn parse(source: &'a str) -> irongraph::Result<TckValue> {
        let mut parser = Self { source, offset: 0 };
        let value = parser.value()?;
        parser.whitespace();
        if parser.offset != parser.source.len() {
            return Err(irongraph::Error::invalid_data(format!(
                "unexpected TCK value suffix `{}`",
                &parser.source[parser.offset..]
            )));
        }
        Ok(value)
    }

    fn peek(&self) -> Option<char> {
        self.source[self.offset..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.offset += character.len_utf8();
        Some(character)
    }

    fn whitespace(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            let _ = self.advance();
        }
    }

    fn consume(&mut self, expected: char) -> irongraph::Result<()> {
        self.whitespace();
        match self.advance() {
            Some(actual) if actual == expected => Ok(()),
            actual => Err(irongraph::Error::invalid_data(format!(
                "expected `{expected}` in TCK value, found {actual:?}"
            ))),
        }
    }

    fn value(&mut self) -> irongraph::Result<TckValue> {
        self.whitespace();
        match self.peek() {
            Some('\'') | Some('\"') => self.string().map(TckValue::String),
            Some('[') => self.bracket_value(),
            Some('{') => self.map().map(TckValue::Map),
            Some('(') => self.node(),
            Some('<') => self.path(),
            Some(_) => self.atom(),
            None => Err(irongraph::Error::invalid_data(
                "unexpected end of TCK value",
            )),
        }
    }

    fn string(&mut self) -> irongraph::Result<String> {
        self.whitespace();
        let quote = self
            .advance()
            .ok_or_else(|| irongraph::Error::invalid_data("missing TCK string quote"))?;
        let mut output = String::new();
        loop {
            let character = self
                .advance()
                .ok_or_else(|| irongraph::Error::invalid_data("unterminated TCK string literal"))?;
            if character == quote {
                return Ok(output);
            }
            if character != '\\' {
                output.push(character);
                continue;
            }
            let escaped = self
                .advance()
                .ok_or_else(|| irongraph::Error::invalid_data("unterminated TCK string escape"))?;
            match escaped {
                '\\' => output.push('\\'),
                '\'' => output.push('\''),
                '\"' => output.push('\"'),
                'b' => output.push('\u{0008}'),
                'f' => output.push('\u{000C}'),
                'n' => output.push('\n'),
                'r' => output.push('\r'),
                't' => output.push('\t'),
                'u' => {
                    let digits = (0..4)
                        .map(|_| {
                            self.advance().ok_or_else(|| {
                                irongraph::Error::invalid_data("truncated TCK unicode escape")
                            })
                        })
                        .collect::<irongraph::Result<String>>()?;
                    let codepoint = u32::from_str_radix(&digits, 16).map_err(|error| {
                        irongraph::Error::invalid_data(format!(
                            "invalid TCK unicode escape `{digits}`: {error}"
                        ))
                    })?;
                    let character = char::from_u32(codepoint).ok_or_else(|| {
                        irongraph::Error::invalid_data("invalid TCK unicode code point")
                    })?;
                    output.push(character);
                }
                other => output.push(other),
            }
        }
    }

    fn identifier(&mut self) -> irongraph::Result<String> {
        self.whitespace();
        if self.peek() == Some('`') {
            let _ = self.advance();
            let start = self.offset;
            while self.peek() != Some('`') {
                if self.advance().is_none() {
                    return Err(irongraph::Error::invalid_data(
                        "unterminated escaped TCK identifier",
                    ));
                }
            }
            let output = self.source[start..self.offset].replace("``", "`");
            let _ = self.advance();
            return Ok(output);
        }
        let start = self.offset;
        while self.peek().is_some_and(|character| {
            character.is_alphanumeric() || character == '_' || character == '.'
        }) {
            let _ = self.advance();
        }
        if start == self.offset {
            return Err(irongraph::Error::invalid_data("expected TCK identifier"));
        }
        Ok(self.source[start..self.offset].to_owned())
    }

    fn map(&mut self) -> irongraph::Result<BTreeMap<String, TckValue>> {
        self.consume('{')?;
        let mut values = BTreeMap::new();
        self.whitespace();
        if self.peek() == Some('}') {
            let _ = self.advance();
            return Ok(values);
        }
        loop {
            let key = match self.peek() {
                Some('\'') | Some('\"') => self.string()?,
                _ => self.identifier()?,
            };
            self.consume(':')?;
            let value = self.value()?;
            if values.insert(key.clone(), value).is_some() {
                return Err(irongraph::Error::invalid_data(format!(
                    "duplicate TCK map key `{key}`"
                )));
            }
            self.whitespace();
            match self.advance() {
                Some('}') => return Ok(values),
                Some(',') => {}
                actual => {
                    return Err(irongraph::Error::invalid_data(format!(
                        "expected `,` or `}}` in TCK map, found {actual:?}"
                    )));
                }
            }
        }
    }

    fn bracket_value(&mut self) -> irongraph::Result<TckValue> {
        let bracket_start = self.offset;
        self.consume('[')?;
        self.whitespace();
        if self.peek() == Some(']') {
            let _ = self.advance();
            return Ok(TckValue::List(Vec::new()));
        }
        let saved = self.offset;
        let relationship = match self.peek() {
            Some(':') => true,
            Some(character)
                if character.is_alphabetic() || character == '_' || character == '`' =>
            {
                let _ = self.identifier()?;
                self.whitespace();
                self.peek() == Some(':')
            }
            _ => false,
        };
        self.offset = saved;
        if relationship {
            self.offset = bracket_start;
            return self.relationship();
        }
        let mut values = Vec::new();
        loop {
            values.push(self.value()?);
            self.whitespace();
            match self.advance() {
                Some(']') => return Ok(TckValue::List(values)),
                Some(',') => {}
                actual => {
                    return Err(irongraph::Error::invalid_data(format!(
                        "expected `,` or `]` in TCK list, found {actual:?}"
                    )));
                }
            }
        }
    }

    fn node(&mut self) -> irongraph::Result<TckValue> {
        self.consume('(')?;
        let mut labels = BTreeSet::new();
        let mut properties = BTreeMap::new();
        self.whitespace();
        if self
            .peek()
            .is_some_and(|character| character.is_alphabetic() || character == '_')
        {
            let _ = self.identifier()?;
        }
        loop {
            self.whitespace();
            if self.peek() != Some(':') {
                break;
            }
            let _ = self.advance();
            labels.insert(self.identifier()?);
        }
        self.whitespace();
        if self.peek() == Some('{') {
            properties = self.map()?;
        }
        self.consume(')')?;
        Ok(TckValue::Node { labels, properties })
    }

    fn relationship(&mut self) -> irongraph::Result<TckValue> {
        self.consume('[')?;
        self.whitespace();
        let relationship_type = if self.peek() == Some(']') {
            None
        } else if self.peek() == Some(':') {
            let _ = self.advance();
            Some(self.identifier()?)
        } else {
            let _ = self.identifier()?;
            self.whitespace();
            if self.peek() == Some(':') {
                let _ = self.advance();
                Some(self.identifier()?)
            } else {
                None
            }
        };
        self.whitespace();
        let properties = if self.peek() == Some('{') {
            self.map()?
        } else {
            BTreeMap::new()
        };
        self.consume(']')?;
        Ok(TckValue::Relationship {
            relationship_type,
            properties,
        })
    }

    fn path(&mut self) -> irongraph::Result<TckValue> {
        self.consume('<')?;
        let mut nodes = vec![self.path_node()?];
        let mut relationships = Vec::new();
        loop {
            self.whitespace();
            if self.peek() == Some('>') {
                let _ = self.advance();
                return Ok(TckValue::Path(TckPath {
                    nodes,
                    relationships,
                }));
            }
            let direction = if self.source[self.offset..].starts_with("<-") {
                self.offset += 2;
                TckPathDirection::Incoming
            } else {
                self.consume('-')?;
                TckPathDirection::Undirected
            };
            let relationship = self.relationship()?;
            let TckValue::Relationship {
                relationship_type,
                properties,
            } = relationship
            else {
                return Err(irongraph::Error::invalid_data(
                    "TCK path step does not contain a relationship",
                ));
            };
            self.consume('-')?;
            let direction = if direction == TckPathDirection::Undirected && self.peek() == Some('>')
            {
                let _ = self.advance();
                TckPathDirection::Outgoing
            } else {
                direction
            };
            relationships.push(TckPathRelationship {
                direction,
                relationship_type,
                properties,
            });
            nodes.push(self.path_node()?);
        }
    }

    fn path_node(&mut self) -> irongraph::Result<TckPathNode> {
        let node = self.node()?;
        let TckValue::Node { labels, properties } = node else {
            return Err(irongraph::Error::invalid_data(
                "TCK path step does not contain a node",
            ));
        };
        Ok(TckPathNode { labels, properties })
    }

    fn atom(&mut self) -> irongraph::Result<TckValue> {
        let start = self.offset;
        while self.peek().is_some_and(|character| {
            !character.is_whitespace() && !matches!(character, ',' | ']' | '}' | ')')
        }) {
            let _ = self.advance();
        }
        let token = &self.source[start..self.offset];
        match token {
            "null" => Ok(TckValue::Null),
            "true" => Ok(TckValue::Boolean(true)),
            "false" => Ok(TckValue::Boolean(false)),
            "NaN" => Ok(TckValue::Float(f64::NAN)),
            "Inf" => Ok(TckValue::Float(f64::INFINITY)),
            "-Inf" => Ok(TckValue::Float(f64::NEG_INFINITY)),
            _ if !token.contains(['.', 'e', 'E']) => token
                .parse::<i64>()
                .map(TckValue::Integer)
                .map_err(|error| {
                    irongraph::Error::invalid_data(format!(
                        "invalid TCK integer `{token}`: {error}"
                    ))
                }),
            _ => token.parse::<f64>().map(TckValue::Float).map_err(|error| {
                irongraph::Error::invalid_data(format!("invalid TCK value `{token}`: {error}"))
            }),
        }
    }
}

#[test]
fn tck_path_parser_consumes_arrowheads_and_preserves_directions() -> irongraph::Result<()> {
    let value =
        TckValueParser::parse("<(:Start {name: 's'})<-[:LEFT {weight: 1}]-()-[:RIGHT]->(:End)>")?;
    let TckValue::Path(path) = value else {
        return Err(irongraph::Error::invalid_data("expected a parsed TCK path"));
    };
    assert_eq!(path.nodes.len(), 3);
    assert_eq!(path.relationships.len(), 2);
    assert_eq!(path.relationships[0].direction, TckPathDirection::Incoming);
    assert_eq!(path.relationships[1].direction, TckPathDirection::Outgoing);
    assert_eq!(
        path.relationships[0].relationship_type.as_deref(),
        Some("LEFT")
    );
    assert_eq!(
        path.relationships[1].relationship_type.as_deref(),
        Some("RIGHT")
    );
    assert!(path.nodes[0].labels.contains("Start"));
    assert!(path.nodes[2].labels.contains("End"));

    let relationship = TckValueParser::parse("[r:REL {id: 1}]")?;
    let TckValue::Relationship {
        relationship_type,
        properties,
    } = relationship
    else {
        return Err(irongraph::Error::invalid_data(
            "expected a parsed standalone TCK relationship",
        ));
    };
    assert_eq!(relationship_type.as_deref(), Some("REL"));
    assert!(matches!(properties.get("id"), Some(TckValue::Integer(1))));
    Ok(())
}

const TCK_PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const TCK_MEMORY_LIMIT: usize = 512 * 1024 * 1024;
const TCK_RESERVED_MEMORY: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
struct ObservedSuccess {
    result: QueryResult,
    side_effects: SideEffects,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ObservedError {
    code: ErrorCode,
    message: String,
    phase: ExpectedErrorPhase,
}

#[derive(Clone, Debug, PartialEq)]
enum ObservedOutcome {
    Success(ObservedSuccess),
    Error(ObservedError),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GraphObservability {
    nodes: BTreeSet<u64>,
    relationships: BTreeSet<u64>,
    properties: BTreeSet<String>,
    labels: BTreeSet<String>,
}

impl GraphObservability {
    fn capture(graph: &GraphStore) -> Self {
        let catalog = graph.catalog();
        let mut observability = Self::default();
        for node in graph.nodes() {
            observability.nodes.insert(node.id().0);
            for label in node.labels() {
                if let Some(name) = catalog.label_name(*label) {
                    observability.labels.insert(name.to_owned());
                }
            }
            for (property, value) in node.properties() {
                if let Some(name) = catalog.property_name(property) {
                    observability
                        .properties
                        .insert(format!("n:{}:{name}:{value:?}", node.id().0));
                }
            }
        }
        for relationship in graph.edges() {
            observability.relationships.insert(relationship.id().0);
            for (property, value) in relationship.properties() {
                if let Some(name) = catalog.property_name(property) {
                    observability
                        .properties
                        .insert(format!("r:{}:{name}:{value:?}", relationship.id().0));
                }
            }
        }
        observability
    }

    fn side_effects_after(&self, after: &Self) -> SideEffects {
        SideEffects {
            nodes_added: set_difference_count(&after.nodes, &self.nodes),
            nodes_removed: set_difference_count(&self.nodes, &after.nodes),
            relationships_added: set_difference_count(&after.relationships, &self.relationships),
            relationships_removed: set_difference_count(&self.relationships, &after.relationships),
            properties_added: set_difference_count(&after.properties, &self.properties),
            properties_removed: set_difference_count(&self.properties, &after.properties),
            labels_added: set_difference_count(&after.labels, &self.labels),
            labels_removed: set_difference_count(&self.labels, &after.labels),
        }
    }
}

fn set_difference_count<T: Ord>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> i64 {
    i64::try_from(left.difference(right).count()).unwrap_or(i64::MAX)
}

fn next_ids(graph: &GraphStore) -> (u64, u64) {
    let next_node = graph
        .nodes()
        .map(|node| node.id().0.saturating_add(1))
        .max()
        .unwrap_or(1);
    let next_edge = graph
        .edges()
        .map(|relationship| relationship.id().0.saturating_add(1))
        .max()
        .unwrap_or(1);
    (next_node, next_edge)
}

fn tck_context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    parameters: BTreeMap<String, ResultValue>,
) -> ExecutionContext<'a> {
    let (next_node_id, next_edge_id) = next_ids(graph);
    ExecutionContext {
        project_id: TCK_PROJECT,
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
        next_node_id,
        next_edge_id,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            schema: false,
            knowledge_write: false,
            // The conformance corpus is ordinary openCypher over the user-visible layers. Writing
            // `Workspace` is an internal capability granted only to the trusted server path
            // internal database records, so the harness is denied it alongside `knowledge_write`.
            workspace_write: false,
            require_native_execution: backend
                .is_some_and(|backend| backend.kind() != BackendKind::Cpu),
        },
        max_result_rows: 100_000,
        max_batch_rows: 1_024,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        resolved_query_at_time_nanos: None,
    }
}

fn parameter_values(case: &TckCase) -> irongraph::Result<BTreeMap<String, ResultValue>> {
    case.parameters
        .iter()
        .map(|(name, value)| {
            Ok((
                name.clone(),
                TckValueParser::parse(value)?.into_parameter()?,
            ))
        })
        .collect()
}

fn apply_mutations(graph: &mut GraphStore, mutations: &[GraphMutation]) -> irongraph::Result<()> {
    for mutation in mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(())
}

fn execute_setup_query(graph: &mut GraphStore, query: &str) -> irongraph::Result<()> {
    let mut context = tck_context(graph, None, BTreeMap::new());
    let output = QueryEngine.execute(query, &mut context)?;
    apply_mutations(graph, &output.graph_mutations)
}

fn setup_graph(root: &Path, case: &TckCase) -> irongraph::Result<GraphStore> {
    let mut graph = GraphStore::default();
    if let GraphFixture::Named(name) = &case.fixture {
        // Callers historically pass either `.../tck` or `.../tck/features`. Named fixtures are
        // always stored in `.../tck/graphs`; resolve both accepted roots deterministically so a
        // harness invocation can never turn the 19 triadic scenarios into false engine failures.
        let direct_graphs = root.join("graphs");
        let graphs = if direct_graphs.is_dir() {
            direct_graphs
        } else if root.file_name().is_some_and(|name| name == "features") {
            root.parent()
                .map_or_else(|| root.join("graphs"), |parent| parent.join("graphs"))
        } else {
            direct_graphs
        };
        let graph_path = graphs.join(name).join(format!("{name}.cypher"));
        let source = fs::read_to_string(&graph_path).map_err(|error| {
            irongraph::Error::invalid_data(format!(
                "cannot read TCK named graph {}: {error}",
                graph_path.display()
            ))
        })?;
        execute_setup_query(&mut graph, &source)?;
    }
    for query in &case.setup_queries {
        execute_setup_query(&mut graph, query)?;
    }
    Ok(graph)
}

fn preflight_phase(
    query: &str,
    graph: &GraphStore,
    parameters: BTreeMap<String, ResultValue>,
    procedures: Option<&ProcedureCatalog>,
) -> ExpectedErrorPhase {
    let capabilities = tck_context(graph, None, parameters.clone()).capabilities;
    parse(query)
        .and_then(|query| match procedures {
            Some(procedures) => bind_with_procedures(
                query,
                graph.catalog(),
                capabilities,
                procedures,
                &parameters,
            ),
            None => bind_with_parameters(query, graph.catalog(), capabilities, &parameters),
        })
        .and_then(plan)
        .map_or(ExpectedErrorPhase::Compile, |_| ExpectedErrorPhase::Runtime)
}

fn execute_operation(
    graph: &mut GraphStore,
    backend: &mut dyn ExecutionBackend,
    operation: &TckOperation,
    parameters: &BTreeMap<String, ResultValue>,
    procedures: Option<&ProcedureCatalog>,
) -> irongraph::Result<ObservedOutcome> {
    // A reusable TCK backend must have exactly the current operation's graph-only project set.
    // Replacing the complete set is intentionally stronger than admitting one project: it clears
    // any prior resident project, delta overlay, temporal/index image, bookmark, and physical
    // layout even when two unrelated fixtures happen to have the same graph revision.
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let before = GraphObservability::capture(graph);
    let phase = preflight_phase(&operation.query, graph, parameters.clone(), procedures);
    let mut context = tck_context(graph, Some(backend), parameters.clone());
    let execution = match procedures {
        Some(procedures) => {
            let scoped = context.with_procedures(procedures);
            QueryEngine.execute_with_procedures(&operation.query, scoped)
        }
        None => QueryEngine.execute(&operation.query, &mut context),
    };
    match execution {
        Ok(output) => {
            apply_mutations(graph, &output.graph_mutations)?;
            let side_effects = before.side_effects_after(&GraphObservability::capture(graph));
            Ok(ObservedOutcome::Success(ObservedSuccess {
                result: output.result,
                side_effects,
            }))
        }
        Err(error) => Ok(ObservedOutcome::Error(ObservedError {
            code: error.code,
            message: error.message.into_owned(),
            phase,
        })),
    }
}

fn rows(result: &QueryResult) -> irongraph::Result<Vec<Vec<ResultValue>>> {
    let mut rows = Vec::new();
    for batch in &result.batches {
        if !batch.validate() || batch.columns.len() != result.schema.len() {
            return Err(irongraph::Error::internal(
                "query result contains an invalid columnar batch",
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
    Ok(rows)
}

fn row_matches(expected: &[TckValue], actual: &[ResultValue], ignore_list_order: bool) -> bool {
    expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| expected.matches_actual(actual, ignore_list_order))
}

fn result_matches(
    expectation: &ResultExpectation,
    outcome: &ObservedOutcome,
) -> irongraph::Result<()> {
    match (expectation, outcome) {
        (ResultExpectation::Error(expected), ObservedOutcome::Error(actual)) => {
            let expected_code = match expected.kind.as_str() {
                "SyntaxError" => ErrorCode::QuerySyntax,
                "SemanticError"
                | "ParameterMissing"
                | "ConstraintVerificationFailed"
                | "ConstraintValidationFailed"
                | "EntityNotFound"
                | "PropertyNotFound"
                | "LabelNotFound"
                | "TypeError"
                | "ArgumentError"
                | "ArithmeticError"
                | "ProcedureError" => ErrorCode::QueryType,
                other => {
                    return Err(irongraph::Error::invalid_data(format!(
                        "unsupported TCK error kind `{other}`"
                    )));
                }
            };
            if actual.code != expected_code {
                return Err(irongraph::Error::invalid_data(format!(
                    "expected {} ({:?}), got {:?}: {}",
                    expected.kind, expected_code, actual.code, actual.message
                )));
            }
            if expected.phase != ExpectedErrorPhase::Any && actual.phase != expected.phase {
                return Err(irongraph::Error::invalid_data(format!(
                    "expected {:?} error phase, got {:?}: {}",
                    expected.phase, actual.phase, actual.message
                )));
            }
            if expected.detail != "*" && !actual.message.contains(&expected.detail) {
                return Err(irongraph::Error::invalid_data(format!(
                    "expected TCK error detail `{}`, got `{}`",
                    expected.detail, actual.message
                )));
            }
            Ok(())
        }
        (ResultExpectation::Error(expected), ObservedOutcome::Success(actual)) => {
            Err(irongraph::Error::invalid_data(format!(
                "expected {} error but query succeeded with {:?}",
                expected.kind, actual.result
            )))
        }
        (_, ObservedOutcome::Error(actual)) => Err(irongraph::Error::invalid_data(format!(
            "query unexpectedly failed at {:?} with {:?}: {}",
            actual.phase, actual.code, actual.message
        ))),
        (ResultExpectation::Empty, ObservedOutcome::Success(actual)) => {
            if rows(&actual.result)?.is_empty() && actual.result.schema.is_empty() {
                Ok(())
            } else {
                Err(irongraph::Error::invalid_data(format!(
                    "expected empty result, got schema {:?} and rows {:?}",
                    actual.result.schema,
                    rows(&actual.result)?
                )))
            }
        }
        (
            ResultExpectation::Table {
                headers,
                rows: expected_rows,
                ordered,
                ignore_list_order,
            },
            ObservedOutcome::Success(actual),
        ) => {
            let actual_headers = actual
                .result
                .schema
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>();
            if &actual_headers != headers {
                return Err(irongraph::Error::invalid_data(format!(
                    "expected columns {headers:?}, got {actual_headers:?}"
                )));
            }
            let expected = expected_rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|value| TckValueParser::parse(value))
                        .collect::<irongraph::Result<Vec<_>>>()
                })
                .collect::<irongraph::Result<Vec<_>>>()?;
            let actual_rows = rows(&actual.result)?;
            if expected.len() != actual_rows.len() {
                return Err(irongraph::Error::invalid_data(format!(
                    "expected {} rows, got {}",
                    expected.len(),
                    actual_rows.len()
                )));
            }
            if *ordered {
                if expected
                    .iter()
                    .zip(&actual_rows)
                    .all(|(expected, actual)| row_matches(expected, actual, *ignore_list_order))
                {
                    return Ok(());
                }
                return Err(irongraph::Error::invalid_data(format!(
                    "ordered result differs: expected {expected_rows:?}, got {actual_rows:?}"
                )));
            }
            let mut used = vec![false; actual_rows.len()];
            if expected.iter().all(|expected| {
                actual_rows.iter().enumerate().any(|(index, actual)| {
                    !used[index] && row_matches(expected, actual, *ignore_list_order) && {
                        used[index] = true;
                        true
                    }
                })
            }) {
                Ok(())
            } else {
                Err(irongraph::Error::invalid_data(format!(
                    "unordered result differs: expected {expected_rows:?}, got {actual_rows:?}"
                )))
            }
        }
    }
}

fn side_effects_match(
    expected: &Option<SideEffects>,
    outcome: &ObservedOutcome,
) -> irongraph::Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let ObservedOutcome::Success(actual) = outcome else {
        return Ok(());
    };
    if expected == &actual.side_effects {
        Ok(())
    } else {
        Err(irongraph::Error::invalid_data(format!(
            "expected side effects {expected:?}, got {:?}",
            actual.side_effects
        )))
    }
}

fn compare_cpu_and_gpu(cpu: &ObservedOutcome, gpu: &ObservedOutcome) -> irongraph::Result<()> {
    if cpu == gpu {
        Ok(())
    } else {
        Err(irongraph::Error::invalid_data(format!(
            "CPU/GPU semantic divergence: CPU={cpu:?}, GPU={gpu:?}"
        )))
    }
}

fn new_tck_cpu_backend() -> CpuBackend {
    CpuBackend::new(TCK_MEMORY_LIMIT, TCK_RESERVED_MEMORY)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn new_tck_metal_backend() -> irongraph::Result<MetalBackend> {
    let governor = irongraph::gpu::DeviceMemoryGovernor::new(TCK_MEMORY_LIMIT, TCK_RESERVED_MEMORY);
    MetalBackend::with_governor(0, governor)
}

fn harness_operation(query: &str) -> TckOperation {
    TckOperation {
        query: query.to_owned(),
        control: false,
        expectation: None,
        side_effects: None,
    }
}

fn harness_graph(setup: &str) -> irongraph::Result<GraphStore> {
    let mut graph = GraphStore::default();
    if !setup.is_empty() {
        execute_setup_query(&mut graph, setup)?;
    }
    Ok(graph)
}

fn successful_rows(outcome: &ObservedOutcome) -> irongraph::Result<Vec<Vec<ResultValue>>> {
    match outcome {
        ObservedOutcome::Success(success) => rows(&success.result),
        ObservedOutcome::Error(error) => Err(irongraph::Error::internal(format!(
            "expected harness query success, got {error:?}"
        ))),
    }
}

fn assert_same_revision_graph_replacement(
    backend: &mut dyn ExecutionBackend,
) -> irongraph::Result<()> {
    let mut previous =
        harness_graph("CREATE (:OldFixture {marker: 1}), (:OldFixture {marker: 2})")?;
    let mut replacement = harness_graph("CREATE (:NewFixture {marker: 3})")?;
    replacement.compact()?;
    assert_eq!(previous.revision(), replacement.revision());
    assert_ne!(previous.layout_version(), replacement.layout_version());

    // Prove complete-set replacement, not merely replacement of the nil TCK project.
    let poison_project = ProjectId(uuid::Uuid::from_u128(1));
    let mut poison = ResidentProjectImage::graph_only(Arc::new(previous.snapshot()?));
    poison.project = poison_project;
    backend.admit_project(poison)?;
    assert!(backend.resident_project_bytes(poison_project).is_some());

    let prior = execute_operation(
        &mut previous,
        backend,
        &harness_operation("MATCH (n:OldFixture) RETURN n"),
        &BTreeMap::new(),
        None,
    )?;
    assert_eq!(successful_rows(&prior)?.len(), 2);
    assert!(backend.resident_project_bytes(poison_project).is_none());

    let next = execute_operation(
        &mut replacement,
        backend,
        &harness_operation("MATCH (n:NewFixture) RETURN n"),
        &BTreeMap::new(),
        None,
    )?;
    let next_rows = successful_rows(&next)?;
    assert_eq!(next_rows.len(), 1);
    let Some(ResultValue::Node(node)) = next_rows.first().and_then(|row| row.first()) else {
        return Err(irongraph::Error::internal(
            "replacement graph query did not return one node",
        ));
    };
    assert_eq!(
        node.properties.get("marker"),
        Some(&ScalarValue::Integer(3))
    );
    assert!(node.labels.iter().any(|label| label == "NewFixture"));
    assert!(!node.labels.iter().any(|label| label == "OldFixture"));
    assert_eq!(
        backend.resident_bookmark(TCK_PROJECT),
        Some(Bookmark {
            term: 0,
            index: replacement.revision(),
        })
    );
    assert_eq!(
        backend.resident_graph_revision(TCK_PROJECT),
        Some(replacement.revision())
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum HarnessOutcomeKind {
    Success { rows: usize },
    RuntimeError,
    Mutation,
    EmptyResult,
}

struct HarnessLifecycleCase {
    name: &'static str,
    graph: GraphStore,
    operation: TckOperation,
    parameters: BTreeMap<String, ResultValue>,
    expected: HarnessOutcomeKind,
}

fn harness_lifecycle_cases() -> irongraph::Result<Vec<HarnessLifecycleCase>> {
    Ok(vec![
        HarnessLifecycleCase {
            name: "success",
            graph: harness_graph("CREATE (:Visible {value: 1}), (:Visible {value: 2})")?,
            operation: harness_operation("MATCH (n:Visible) RETURN n"),
            parameters: BTreeMap::new(),
            expected: HarnessOutcomeKind::Success { rows: 2 },
        },
        HarnessLifecycleCase {
            name: "runtime error",
            graph: GraphStore::default(),
            operation: harness_operation("RETURN range(1, 3, 0) AS values"),
            parameters: BTreeMap::new(),
            expected: HarnessOutcomeKind::RuntimeError,
        },
        HarnessLifecycleCase {
            name: "mutation",
            graph: harness_graph(
                "CREATE (p:Person {count: 5}), (a:Target), (b:Target), \
                 (p)-[:KNOWS {weight: 1}]->(a), \
                 (p)-[:KNOWS {weight: 10}]->(b)",
            )?,
            operation: harness_operation(
                "MATCH (n:Person)-[r:KNOWS]->(:Target) \
                 WHERE n.count > 0 \
                 SET n.count = n.count + $step, \
                     r.weight = r.weight + 2, \
                     n.native = $flag",
            ),
            parameters: BTreeMap::from([
                (
                    "step".to_owned(),
                    ResultValue::Scalar(ScalarValue::Integer(1)),
                ),
                (
                    "flag".to_owned(),
                    ResultValue::Scalar(ScalarValue::Boolean(true)),
                ),
            ]),
            expected: HarnessOutcomeKind::Mutation,
        },
        HarnessLifecycleCase {
            name: "empty result",
            graph: harness_graph("CREATE (:Visible {value: 1})")?,
            operation: harness_operation("MATCH (n:Visible) WHERE n.value = 999 RETURN n"),
            parameters: BTreeMap::new(),
            expected: HarnessOutcomeKind::EmptyResult,
        },
    ])
}

fn assert_harness_outcome(
    name: &str,
    expected: HarnessOutcomeKind,
    outcome: &ObservedOutcome,
) -> irongraph::Result<()> {
    match (expected, outcome) {
        (HarnessOutcomeKind::Success { rows: expected }, ObservedOutcome::Success(success)) => {
            assert_eq!(rows(&success.result)?.len(), expected, "{name}");
        }
        (HarnessOutcomeKind::RuntimeError, ObservedOutcome::Error(error)) => {
            assert_eq!(error.phase, ExpectedErrorPhase::Runtime, "{name}");
            assert_eq!(error.code, ErrorCode::QueryType, "{name}");
            assert!(
                error.message.contains("range step must not be zero"),
                "{name}"
            );
        }
        (HarnessOutcomeKind::Mutation, ObservedOutcome::Success(success)) => {
            assert!(rows(&success.result)?.is_empty(), "{name}");
            assert!(success.side_effects.properties_added > 0, "{name}");
            assert!(success.side_effects.properties_removed > 0, "{name}");
        }
        (HarnessOutcomeKind::EmptyResult, ObservedOutcome::Success(success)) => {
            assert!(rows(&success.result)?.is_empty(), "{name}");
        }
        _ => {
            return Err(irongraph::Error::internal(format!(
                "unexpected {name} harness outcome: {outcome:?}"
            )));
        }
    }
    Ok(())
}

fn assert_reusable_matches_fresh<B, F>(
    reusable: &mut B,
    mut fresh_backend: F,
) -> irongraph::Result<()>
where
    B: ExecutionBackend,
    F: FnMut() -> irongraph::Result<B>,
{
    for case in harness_lifecycle_cases()? {
        let mut reusable_graph = case.graph.clone();
        let reusable_outcome = execute_operation(
            &mut reusable_graph,
            reusable,
            &case.operation,
            &case.parameters,
            None,
        )?;
        let reusable_resident_bytes = reusable.resident_project_bytes(TCK_PROJECT);
        let reusable_scratch = reusable.available_query_scratch_bytes();

        let mut fresh = fresh_backend()?;
        let mut fresh_graph = case.graph;
        let fresh_outcome = execute_operation(
            &mut fresh_graph,
            &mut fresh,
            &case.operation,
            &case.parameters,
            None,
        )?;

        assert_eq!(reusable_outcome, fresh_outcome, "{}", case.name);
        assert_eq!(
            GraphObservability::capture(&reusable_graph),
            GraphObservability::capture(&fresh_graph),
            "{}",
            case.name
        );
        assert_eq!(
            reusable_resident_bytes,
            fresh.resident_project_bytes(TCK_PROJECT),
            "{} resident bytes",
            case.name
        );
        assert_eq!(
            reusable_scratch,
            fresh.available_query_scratch_bytes(),
            "{} scratch accounting",
            case.name
        );
        assert_harness_outcome(case.name, case.expected, &reusable_outcome)?;
    }
    Ok(())
}

#[test]
fn reusable_cpu_tck_harness_replaces_state_and_matches_fresh_backends() -> irongraph::Result<()> {
    let mut reusable = new_tck_cpu_backend();
    assert_same_revision_graph_replacement(&mut reusable)?;
    assert_reusable_matches_fresh(&mut reusable, || Ok(new_tck_cpu_backend()))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn reusable_metal_tck_harness_replaces_state_and_matches_fresh_backends() -> irongraph::Result<()> {
    let mut reusable = new_tck_metal_backend()?;
    assert_same_revision_graph_replacement(&mut reusable)?;
    assert_reusable_matches_fresh(&mut reusable, new_tck_metal_backend)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[derive(Default)]
struct CaseReport {
    shared_failures: Vec<String>,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
    divergences: Vec<String>,
    compared_operations: usize,
    operation_count: usize,
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
struct ReusableTckBackends {
    cpu: CpuBackend,
    metal: MetalBackend,
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
impl ReusableTckBackends {
    fn new() -> irongraph::Result<Self> {
        let cpu = new_tck_cpu_backend();
        let metal = new_tck_metal_backend()?;
        assert_eq!(cpu.kind(), BackendKind::Cpu);
        assert_eq!(metal.kind(), BackendKind::Metal);
        Ok(Self { cpu, metal })
    }
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
impl CaseReport {
    fn passed(&self) -> bool {
        self.shared_failures.is_empty()
            && self.cpu_failures.is_empty()
            && self.metal_failures.is_empty()
            && self.divergences.is_empty()
    }

    fn cpu_passed(&self) -> bool {
        self.shared_failures.is_empty() && self.cpu_failures.is_empty()
    }

    fn metal_passed(&self) -> bool {
        self.shared_failures.is_empty() && self.metal_failures.is_empty()
    }

    fn matched(&self) -> bool {
        self.shared_failures.is_empty()
            && self.divergences.is_empty()
            && self.compared_operations == self.operation_count
    }

    fn describe(&self) -> String {
        self.shared_failures
            .iter()
            .map(|failure| format!("fixture: {failure}"))
            .chain(
                self.cpu_failures
                    .iter()
                    .map(|failure| format!("CPU: {failure}")),
            )
            .chain(
                self.metal_failures
                    .iter()
                    .map(|failure| format!("Metal: {failure}")),
            )
            .chain(
                self.divergences
                    .iter()
                    .map(|failure| format!("CPU/Metal: {failure}")),
            )
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Optional machine-readable result for a complete strict run. Keeping all four dimensions
/// makes the report useful for prioritizing semantic CPU defects separately from missing native
/// GPU execution, without treating either category as an exemption.
#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[derive(Serialize)]
struct TckScenarioReport {
    path: String,
    name: String,
    cpu_passed: bool,
    metal_passed: bool,
    cpu_metal_matched: bool,
    fully_conformant: bool,
    operation_count: usize,
    shared_failures: Vec<String>,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
    divergences: Vec<String>,
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[derive(Serialize)]
struct TckRunReport {
    total: usize,
    cpu_passed: usize,
    metal_passed: usize,
    matched: usize,
    fully_conformant: usize,
    scenarios: Vec<TckScenarioReport>,
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn operation_name(operation: &TckOperation, index: usize) -> String {
    let kind = if operation.control {
        "control"
    } else {
        "primary"
    };
    let query = operation
        .query
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let mut characters = query.chars();
    let excerpt = characters.by_ref().take(160).collect::<String>();
    let suffix = characters
        .next()
        .is_some()
        .then_some("…")
        .unwrap_or_default();
    format!("{kind} operation {} `{excerpt}{suffix}`", index + 1)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn validate_backend_outcome(operation: &TckOperation, outcome: &ObservedOutcome) -> Vec<String> {
    let mut failures = Vec::new();
    let Some(expectation) = operation.expectation.as_ref() else {
        failures.push("decoded TCK operation has no expectation".to_owned());
        return failures;
    };
    if let Err(error) = result_matches(expectation, outcome) {
        failures.push(error.message.into_owned());
    }
    if let Err(error) = side_effects_match(&operation.side_effects, outcome) {
        failures.push(error.message.into_owned());
    }
    failures
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn run_case(
    root: &Path,
    scenario: &ExpandedScenario,
    backends: &mut ReusableTckBackends,
) -> CaseReport {
    let mut report = CaseReport::default();
    let case = match parse_case(scenario) {
        Ok(case) => case,
        Err(error) => {
            report.shared_failures.push(error.message.into_owned());
            return report;
        }
    };
    let procedure_catalog = match fixture_procedure_catalog(&case) {
        Ok(catalog) => catalog,
        Err(error) => {
            let fixtures = case
                .procedures
                .iter()
                .map(FixtureProcedure::describe)
                .collect::<Vec<_>>()
                .join("; ");
            report.shared_failures.push(format!(
                "procedure fixture registration failed for {fixtures}: {}",
                error.message
            ));
            return report;
        }
    };
    let procedures = (!procedure_catalog.is_empty()).then_some(&procedure_catalog);
    let parameters = match parameter_values(&case) {
        Ok(parameters) => parameters,
        Err(error) => {
            report.shared_failures.push(error.message.into_owned());
            return report;
        }
    };
    let initial = match setup_graph(root, &case) {
        Ok(initial) => initial,
        Err(error) => {
            report.shared_failures.push(error.message.into_owned());
            return report;
        }
    };
    let mut cpu_graph = initial.clone();
    let mut gpu_graph = initial;
    report.operation_count = case.operations.len();

    for (index, operation) in case.operations.iter().enumerate() {
        let operation_name = operation_name(operation, index);
        let cpu_outcome = execute_operation(
            &mut cpu_graph,
            &mut backends.cpu,
            operation,
            &parameters,
            procedures,
        );
        let metal_outcome = execute_operation(
            &mut gpu_graph,
            &mut backends.metal,
            operation,
            &parameters,
            procedures,
        );

        match &cpu_outcome {
            Ok(outcome) => {
                for failure in validate_backend_outcome(operation, outcome) {
                    report
                        .cpu_failures
                        .push(format!("{operation_name}: {failure}"));
                }
            }
            Err(error) => report.cpu_failures.push(format!(
                "{operation_name}: execution harness failure: {}",
                error.message
            )),
        }
        match &metal_outcome {
            Ok(outcome) => {
                for failure in validate_backend_outcome(operation, outcome) {
                    report
                        .metal_failures
                        .push(format!("{operation_name}: {failure}"));
                }
            }
            Err(error) => report.metal_failures.push(format!(
                "{operation_name}: execution harness failure: {}",
                error.message
            )),
        }
        if let (Ok(cpu_outcome), Ok(metal_outcome)) = (&cpu_outcome, &metal_outcome) {
            report.compared_operations += 1;
            if let Err(error) = compare_cpu_and_gpu(cpu_outcome, metal_outcome) {
                report
                    .divergences
                    .push(format!("{operation_name}: {}", error.message));
            }
        }
    }
    report
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires a pinned upstream openCypher checkout and a real Metal device"]
fn full_opencypher_tck_gpu_conformance() -> irongraph::Result<()> {
    let root = env::var_os("OPENCYPHER_TCK_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| irongraph::Error::invalid_data("OPENCYPHER_TCK_DIR is required"))?;
    let scenarios = load_scenarios(&root)?;
    assert_eq!(scenarios.len(), EXPECTED_EXPANDED_SCENARIOS);
    let filter = env::var("OPENCYPHER_TCK_FILTER").ok();
    let progress = env::var_os("OPENCYPHER_TCK_PROGRESS").is_some();
    let report_path = env::var_os("OPENCYPHER_TCK_REPORT").map(PathBuf::from);
    let diagnostics_path = report_path.clone();
    let mut total = 0usize;
    let mut cpu_passed = 0usize;
    let mut metal_passed = 0usize;
    let mut matched = 0usize;
    let mut passed = 0usize;
    let mut failures = Vec::new();
    let mut scenario_reports = report_path.as_ref().map(|_| Vec::new());
    // The gate is deliberately sequential. One backend of each kind owns the whole run; every
    // operation atomically replaces its complete resident project set before execution.
    let mut backends = ReusableTckBackends::new()?;
    for scenario in &scenarios {
        let identity = format!("{} / {}", scenario.path.display(), scenario.name);
        if filter
            .as_ref()
            .is_some_and(|filter| !identity.contains(filter))
        {
            continue;
        }
        total += 1;
        if progress {
            eprintln!("TCK RUN {total}: {identity}");
        }
        let report = run_case(&root, scenario, &mut backends);
        let case_cpu_passed = report.cpu_passed();
        let case_metal_passed = report.metal_passed();
        let case_matched = report.matched();
        let case_passed = report.passed();
        if let Some(scenario_reports) = scenario_reports.as_mut() {
            scenario_reports.push(TckScenarioReport {
                path: scenario.path.display().to_string(),
                name: scenario.name.clone(),
                cpu_passed: case_cpu_passed,
                metal_passed: case_metal_passed,
                cpu_metal_matched: case_matched,
                fully_conformant: case_passed,
                operation_count: report.operation_count,
                shared_failures: report.shared_failures.clone(),
                cpu_failures: report.cpu_failures.clone(),
                metal_failures: report.metal_failures.clone(),
                divergences: report.divergences.clone(),
            });
        }
        if case_cpu_passed {
            cpu_passed += 1;
        }
        if case_metal_passed {
            metal_passed += 1;
        }
        if case_matched {
            matched += 1;
        }
        if case_passed {
            passed += 1;
        } else {
            failures.push(format!("{identity}: {}", report.describe()));
        }
    }
    if let Some(path) = report_path {
        let report = TckRunReport {
            total,
            cpu_passed,
            metal_passed,
            matched,
            fully_conformant: passed,
            scenarios: scenario_reports.unwrap_or_default(),
        };
        let contents = serde_json::to_vec_pretty(&report).map_err(|error| {
            irongraph::Error::internal(format!("cannot serialize TCK run report: {error}"))
        })?;
        fs::write(&path, contents).map_err(|error| {
            irongraph::Error::internal(format!(
                "cannot write TCK report {}: {error}",
                path.display()
            ))
        })?;
    }
    assert!(total > 0, "TCK filter matched no scenarios");
    let failure_details = diagnostics_path.as_ref().map_or_else(
        || failures.join("\n"),
        |path| {
            format!(
                "complete per-scenario diagnostics were written to {}",
                path.display()
            )
        },
    );
    assert!(
        failures.is_empty(),
        "{}/{} TCK scenarios failed; CPU expected-result passes: {cpu_passed}/{total}; Metal expected-result passes: {metal_passed}/{total}; CPU/Metal matching outcomes: {matched}/{total}; fully conformant: {passed}/{total}.\n{}",
        failures.len(),
        total,
        failure_details
    );
    Ok(())
}

#[cfg(not(all(feature = "accelerator", target_os = "macos")))]
#[test]
#[ignore = "the full TCK GPU gate must run on a real Metal runner"]
fn full_opencypher_tck_gpu_conformance_requires_metal() -> irongraph::Result<()> {
    Err(irongraph::Error::internal(
        "run the full openCypher TCK GPU gate on macOS with --features accelerator",
    ))
}

#[test]
#[ignore = "requires a pinned upstream openCypher checkout; run via tools/run-opencypher-tck.sh"]
fn full_opencypher_tck_step_contract_is_decoded() -> irongraph::Result<()> {
    let root = env::var_os("OPENCYPHER_TCK_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| irongraph::Error::invalid_data("OPENCYPHER_TCK_DIR is required"))?;
    let scenarios = load_scenarios(&root)?;
    assert_eq!(scenarios.len(), EXPECTED_EXPANDED_SCENARIOS);
    for scenario in &scenarios {
        parse_case(scenario).map_err(|error| {
            irongraph::Error::invalid_data(format!(
                "{} / {}: {}",
                scenario.path.display(),
                scenario.name,
                error.message
            ))
        })?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a pinned upstream openCypher checkout; run via tools/run-opencypher-tck.sh"]
fn full_opencypher_tck_fixture_procedure_catalog_is_decoded() -> irongraph::Result<()> {
    let root = env::var_os("OPENCYPHER_TCK_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| irongraph::Error::invalid_data("OPENCYPHER_TCK_DIR is required"))?;
    let scenarios = load_scenarios(&root)?;
    assert_eq!(scenarios.len(), EXPECTED_EXPANDED_SCENARIOS);
    let mut fixture_scenarios = 0usize;
    let mut names = BTreeSet::new();
    for scenario in &scenarios {
        let case = parse_case(scenario).map_err(|error| {
            irongraph::Error::invalid_data(format!(
                "{} / {}: {}",
                scenario.path.display(),
                scenario.name,
                error.message
            ))
        })?;
        if !case.procedures.is_empty() {
            fixture_scenarios += 1;
            names.extend(
                case.procedures
                    .iter()
                    .map(|procedure| procedure.name.clone()),
            );
        }
    }
    assert_eq!(fixture_scenarios, 50);
    assert_eq!(
        names,
        BTreeSet::from([
            "test.doNothing".to_owned(),
            "test.labels".to_owned(),
            "test.my.proc".to_owned(),
        ])
    );
    Ok(())
}

#[test]
#[ignore = "requires a pinned upstream openCypher checkout; run via tools/run-opencypher-tck.sh"]
fn full_opencypher_tck_corpus_parses_and_is_accounted_for() -> irongraph::Result<()> {
    let root = env::var_os("OPENCYPHER_TCK_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| irongraph::Error::invalid_data("OPENCYPHER_TCK_DIR is required"))?;
    let files = feature_files(&root)?;
    assert_eq!(
        files.len(),
        EXPECTED_FEATURE_FILES,
        "upstream TCK file count changed"
    );
    let scenarios = load_scenarios(&root)?;
    assert_eq!(
        scenarios.len(),
        EXPECTED_EXPANDED_SCENARIOS,
        "upstream TCK scenario count changed"
    );

    let mut parsed_queries = 0usize;
    let mut valid_queries = 0usize;
    let mut expected_compile_errors = 0usize;
    let mut expected_runtime_errors = 0usize;
    let mut unexpected_parser_rejections = Vec::new();
    for scenario in &scenarios {
        for (step_index, step) in scenario.steps.iter().enumerate() {
            if step.value != "executing query:" {
                continue;
            }
            let query = step
                .docstring
                .as_deref()
                .ok_or_else(|| irongraph::Error::internal("TCK query step has no docstring"))?;
            match (
                parse(query),
                expected_error_phase(&scenario.steps, step_index),
            ) {
                (Ok(_), None) => valid_queries += 1,
                (Ok(_), Some(ExpectedErrorPhase::Compile)) => {
                    unexpected_parser_rejections.push(format!(
                        "{} / {}: expected compile-time rejection, parser accepted query",
                        scenario.path.display(),
                        scenario.name
                    ));
                    expected_compile_errors += 1;
                }
                (Ok(_), Some(ExpectedErrorPhase::Runtime)) => expected_runtime_errors += 1,
                (Err(error), Some(ExpectedErrorPhase::Compile)) => {
                    let _ = error;
                    expected_compile_errors += 1;
                }
                (Err(error), Some(ExpectedErrorPhase::Runtime)) => {
                    unexpected_parser_rejections.push(format!(
                        "{} / {}: runtime error scenario rejected during parsing: {error}",
                        scenario.path.display(),
                        scenario.name
                    ));
                    expected_runtime_errors += 1;
                }
                (Ok(_), Some(ExpectedErrorPhase::Any))
                | (Err(_), Some(ExpectedErrorPhase::Any)) => {
                    expected_runtime_errors += 1;
                }
                (Err(error), None) => {
                    unexpected_parser_rejections.push(format!(
                        "{} / {}: valid scenario rejected during parsing: {error}",
                        scenario.path.display(),
                        scenario.name
                    ));
                    valid_queries += 1;
                }
            }
            parsed_queries += 1;
        }
    }
    assert!(parsed_queries > 0);
    assert!(
        unexpected_parser_rejections.is_empty(),
        "{} parser-phase mismatches (valid={valid_queries}, compile-errors={expected_compile_errors}, runtime-errors={expected_runtime_errors}):\n{}",
        unexpected_parser_rejections.len(),
        unexpected_parser_rejections.join("\n")
    );
    Ok(())
}
