//! Built-in graph procedure signatures and small execution-scoped typed procedure catalogs.

use std::collections::{BTreeMap, BTreeSet};

pub use crate::execution::{ProcedureDefinition, ProcedureField};
use crate::{Error, ErrorCode, Result};

use super::{
    CallArgumentMode, CallClause, CallYieldMode, Expression, ResultValue,
    expression::contains_aggregate,
};
/// Per-execution procedure definitions. This catalog is never global or persisted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProcedureCatalog {
    procedures: BTreeMap<String, ProcedureDefinition>,
}

impl ProcedureCatalog {
    pub fn register(&mut self, procedure: ProcedureDefinition) -> Result<()> {
        let canonical_name = procedure.name().to_ascii_lowercase();
        if signature(&canonical_name).is_some() {
            return Err(Error::invalid_data(format!(
                "execution-scoped procedure `{}` conflicts with a built-in procedure",
                procedure.name()
            )));
        }
        if self.procedures.contains_key(&canonical_name) {
            return Err(Error::invalid_data(format!(
                "procedure `{canonical_name}` is registered more than once"
            )));
        }
        self.procedures.insert(canonical_name, procedure);
        Ok(())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.procedures.is_empty()
    }

    fn get(&self, name: &str) -> Option<&ProcedureDefinition> {
        self.procedures.get(&name.to_ascii_lowercase())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcedureSignature {
    pub name: &'static str,
    pub accepted_arities: &'static [usize],
    pub outputs: &'static [&'static str],
}

const SIGNATURES: &[ProcedureSignature] = &[
    ProcedureSignature {
        name: "graph.bfs",
        accepted_arities: &[1],
        outputs: &["node", "distance"],
    },
    ProcedureSignature {
        name: "graph.clusteringcoefficient",
        accepted_arities: &[0],
        outputs: &["node", "coefficient"],
    },
    ProcedureSignature {
        name: "graph.degree",
        accepted_arities: &[0],
        outputs: &["node", "outDegree", "inDegree", "degree"],
    },
    ProcedureSignature {
        name: "graph.dfs",
        accepted_arities: &[1],
        outputs: &["node", "order"],
    },
    ProcedureSignature {
        name: "graph.dijkstra",
        accepted_arities: &[1, 2],
        outputs: &["node", "cost", "predecessor"],
    },
    ProcedureSignature {
        name: "graph.kcore",
        accepted_arities: &[0],
        outputs: &["node", "core"],
    },
    ProcedureSignature {
        name: "graph.louvain",
        accepted_arities: &[0],
        outputs: &["node", "community"],
    },
    ProcedureSignature {
        name: "graph.pagerank",
        accepted_arities: &[0, 3],
        outputs: &["node", "score"],
    },
    ProcedureSignature {
        name: "graph.scc",
        accepted_arities: &[0],
        outputs: &["node", "component"],
    },
    ProcedureSignature {
        name: "graph.shortestpath",
        accepted_arities: &[2],
        outputs: &["path", "cost"],
    },
    ProcedureSignature {
        name: "graph.trianglecount",
        accepted_arities: &[0],
        outputs: &["triangleCount"],
    },
    ProcedureSignature {
        name: "graph.wcc",
        accepted_arities: &[0],
        outputs: &["node", "component"],
    },
];

pub enum ResolvedProcedure<'a> {
    Builtin(&'static ProcedureSignature),
    ExecutionScoped(&'a ProcedureDefinition),
}

impl ResolvedProcedure<'_> {
    pub fn output_names(&self) -> Vec<&str> {
        match self {
            Self::Builtin(signature) => signature.outputs.to_vec(),
            Self::ExecutionScoped(procedure) => procedure
                .outputs()
                .iter()
                .map(|field| field.name.as_str())
                .collect(),
        }
    }
}

pub fn signature(name: &str) -> Option<&'static ProcedureSignature> {
    SIGNATURES.iter().find(|signature| signature.name == name)
}

pub fn validate_call<'a>(
    call: &CallClause,
    catalog: Option<&'a ProcedureCatalog>,
    parameters: Option<&BTreeMap<String, ResultValue>>,
) -> Result<ResolvedProcedure<'a>> {
    let name = call.name.join(".").to_ascii_lowercase();
    let resolved = if let Some(procedure) = catalog.and_then(|catalog| catalog.get(&name)) {
        ResolvedProcedure::ExecutionScoped(procedure)
    } else if let Some(signature) = signature(&name) {
        ResolvedProcedure::Builtin(signature)
    } else {
        return Err(Error::new(
            ErrorCode::QueryType,
            format!("ProcedureNotFound: procedure `{name}` is not registered"),
        ));
    };

    if call.arguments.iter().any(contains_aggregate) {
        return Err(Error::new(
            ErrorCode::QuerySyntax,
            "InvalidAggregation: procedure arguments may not contain aggregate functions",
        ));
    }

    match &resolved {
        ResolvedProcedure::Builtin(signature) => validate_builtin_arguments(call, signature)?,
        ResolvedProcedure::ExecutionScoped(procedure) => {
            validate_execution_scoped_arguments(call, procedure, parameters)?
        }
    }
    validate_yields(call, &resolved)?;
    Ok(resolved)
}

fn validate_builtin_arguments(call: &CallClause, signature: &ProcedureSignature) -> Result<()> {
    if call.argument_mode == CallArgumentMode::Implicit {
        if signature.accepted_arities.contains(&0) {
            return Ok(());
        }
        return Err(Error::new(
            ErrorCode::QuerySyntax,
            "InvalidArgumentPassingMode: built-in procedure inputs require explicit arguments",
        ));
    }
    if !signature.accepted_arities.contains(&call.arguments.len()) {
        let accepted = signature
            .accepted_arities
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" or ");
        return Err(Error::new(
            ErrorCode::QueryType,
            format!(
                "procedure {} expects {accepted} arguments, received {}",
                signature.name,
                call.arguments.len()
            ),
        ));
    }
    Ok(())
}

fn validate_execution_scoped_arguments(
    call: &CallClause,
    procedure: &ProcedureDefinition,
    parameters: Option<&BTreeMap<String, ResultValue>>,
) -> Result<()> {
    match call.argument_mode {
        CallArgumentMode::Explicit => {
            if call.arguments.len() != procedure.inputs().len() {
                return Err(invalid_arity(
                    &procedure.name(),
                    procedure.inputs().len().to_string(),
                    call.arguments.len(),
                ));
            }
            for (argument, field) in call.arguments.iter().zip(procedure.inputs()) {
                if let Some(value) = known_expression_value(argument, parameters)? {
                    validate_known_argument(&procedure.name(), field, &value)?;
                }
            }
        }
        CallArgumentMode::Implicit => {
            let Some(parameters) = parameters else {
                return Ok(());
            };
            for field in procedure.inputs() {
                let value = parameters.get(&field.name).ok_or_else(|| {
                    Error::new(
                        ErrorCode::QueryType,
                        format!(
                            "MissingParameter: implicit procedure argument `${}` is missing",
                            field.name
                        ),
                    )
                })?;
                validate_known_argument(&procedure.name(), field, value)?;
            }
        }
    }
    Ok(())
}

fn invalid_arity(name: &str, expected: String, actual: usize) -> Error {
    Error::new(
        ErrorCode::QuerySyntax,
        format!(
            "InvalidNumberOfArguments: procedure {name} expects {expected} arguments, received {actual}"
        ),
    )
}

fn known_expression_value(
    expression: &Expression,
    parameters: Option<&BTreeMap<String, ResultValue>>,
) -> Result<Option<ResultValue>> {
    match expression {
        Expression::Literal(value) => Ok(Some(ResultValue::Scalar(value.clone()))),
        Expression::Parameter(name) => match parameters {
            Some(parameters) => parameters.get(name).cloned().map(Some).ok_or_else(|| {
                Error::new(
                    ErrorCode::QueryType,
                    format!("MissingParameter: parameter `${name}` is missing"),
                )
            }),
            None => Ok(None),
        },
        Expression::List(_) => Ok(Some(ResultValue::List(Vec::new()))),
        Expression::Map(_) => Ok(Some(ResultValue::Map(BTreeMap::new()))),
        Expression::Unary { operand, .. } => known_expression_value(operand, parameters),
        _ => Ok(None),
    }
}

fn validate_known_argument(
    procedure: &str,
    field: &ProcedureField,
    value: &ResultValue,
) -> Result<()> {
    if field.value_type.accepts(value, field.nullable) {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::QuerySyntax,
            format!(
                "InvalidArgumentType: procedure {procedure} input `{}` expects {}{}",
                field.name,
                field.value_type.name(),
                if field.nullable { "?" } else { "" }
            ),
        ))
    }
}

fn validate_yields(call: &CallClause, procedure: &ResolvedProcedure<'_>) -> Result<()> {
    if call.yield_mode != CallYieldMode::Explicit {
        if call.yields.is_empty() {
            return Ok(());
        }
        return Err(Error::internal(
            "non-explicit procedure YIELD has named items",
        ));
    }
    let outputs = procedure.output_names();
    let mut aliases = BTreeSet::new();
    for item in &call.yields {
        let Expression::Variable(output) = &item.expression else {
            return Err(Error::internal("procedure YIELD item is not a variable"));
        };
        if !outputs.contains(&output.as_str()) {
            return Err(Error::new(
                ErrorCode::QueryType,
                format!("procedure has no output `{output}`"),
            ));
        }
        let alias = item.alias.as_deref().unwrap_or(output);
        if !aliases.insert(alias) {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                format!("VariableAlreadyBound: procedure YIELD alias `{alias}` is repeated"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{ScalarValue, cypher::ProjectionItem, execution::ProcedureValueType};
    use ordered_float::OrderedFloat;

    fn explicit_call(name: &[&str], arguments: Vec<Expression>) -> CallClause {
        CallClause {
            name: name.iter().map(|part| (*part).to_owned()).collect(),
            argument_mode: CallArgumentMode::Explicit,
            arguments,
            yield_mode: CallYieldMode::Omitted,
            yields: Vec::new(),
            standalone: true,
        }
    }

    #[test]
    fn rejects_unknown_yield_and_wrong_builtin_arity() {
        let call = explicit_call(&["graph", "bfs"], Vec::new());
        assert!(validate_call(&call, None, None).is_err());

        let mut call = explicit_call(&["graph", "wcc"], Vec::new());
        call.yield_mode = CallYieldMode::Explicit;
        call.yields.push(ProjectionItem {
            expression: Expression::Variable("score".to_owned()),
            alias: None,
            source_text: None,
        });
        assert!(validate_call(&call, None, None).is_err());
    }

    #[test]
    fn typed_relation_matches_inputs_and_projects_outputs() -> Result<()> {
        let definition = ProcedureDefinition::new(
            "test.city",
            vec![ProcedureField::new(
                "name",
                ProcedureValueType::String,
                true,
            )?],
            vec![ProcedureField::new(
                "city",
                ProcedureValueType::String,
                true,
            )?],
            vec![vec![
                ResultValue::Scalar(ScalarValue::String(Arc::from("Stefan"))),
                ResultValue::Scalar(ScalarValue::String(Arc::from("Berlin"))),
            ]],
        )?;
        assert_eq!(
            definition.execute(vec![ResultValue::Scalar(ScalarValue::String(Arc::from(
                "Stefan"
            )))])?,
            vec![vec![ResultValue::Scalar(ScalarValue::String(Arc::from(
                "Berlin"
            )))]]
        );
        assert!(
            definition
                .execute(vec![ResultValue::Scalar(ScalarValue::String(Arc::from(
                    "Petra"
                )))])?
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn float_inputs_accept_and_normalize_integers() -> Result<()> {
        let definition = ProcedureDefinition::new(
            "test.float",
            vec![ProcedureField::new(
                "input",
                ProcedureValueType::Float,
                true,
            )?],
            vec![ProcedureField::new(
                "output",
                ProcedureValueType::String,
                true,
            )?],
            vec![vec![
                ResultValue::Scalar(ScalarValue::Float(OrderedFloat(42.0))),
                ResultValue::Scalar(ScalarValue::String(Arc::from("matched"))),
            ]],
        )?;
        assert_eq!(
            definition
                .execute(vec![ResultValue::Scalar(ScalarValue::Integer(42))])?
                .len(),
            1
        );
        Ok(())
    }
}
