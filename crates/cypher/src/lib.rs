//! Cypher lexer, parser, binder, physical planner, and typed execution stream.

#![allow(
    clippy::approx_constant,
    clippy::cloned_ref_to_slice_refs,
    clippy::collapsible_if,
    clippy::double_ended_iterator_last,
    clippy::if_same_then_else,
    clippy::iter_skip_next,
    clippy::manual_contains,
    clippy::manual_ignore_case_cmp,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::manual_saturating_arithmetic,
    clippy::manual_slice_size_calculation,
    clippy::match_like_matches_macro,
    clippy::needless_borrow,
    clippy::needless_lifetimes,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::nonminimal_bool,
    clippy::obfuscated_if_else,
    clippy::only_used_in_recursion,
    clippy::question_mark,
    clippy::skip_while_next,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_filter_map,
    clippy::unnecessary_lazy_evaluations,
    clippy::useless_conversion,
    clippy::useless_vec
)]

mod ast;
mod binder;
mod equality_key;
mod executor;
mod expression;
mod lexer;
mod optimizer;
mod order_key;
mod parser;
mod plan_cache;
mod planner;
mod procedure;
mod resident;
mod resident_row;
mod resident_variable_path;
mod value;
mod view;

pub use crate::execution::TextEmbedding;
pub use crate::execution::{
    ColumnType, ExecutionStreamItem, QueryResult, ResultBatch, ResultColumn, ResultEdge,
    ResultNode, ResultValue, StatementStats,
};
pub use crate::execution::{ProcedureDefinition, ProcedureField, ProcedureValueType};
pub use ast::*;
pub use binder::{
    BindCapabilities, BoundQuery, DependencyKind, DependencyStamp, bind, bind_with_parameters,
    bind_with_procedures,
};
pub use executor::{
    EntityDependency, ExecutionContext, ExecutionOutput, PreparedTemporalMutation,
    ProcedureExecutionContext, QueryEngine, TransactionDependencies, VectorSearchSource,
    VectorSearchTrace,
};
pub use executor::{
    apply_duration_to_temporal_scalar, evaluate_temporal_accessor, format_temporal_scalar,
};
pub use lexer::{Lexer, Span, Symbol, Token, TokenKind, lex};
pub use optimizer::{OptimizationProfile, OptimizerInput, optimize};
#[cfg(test)]
pub use order_key::encode_total_order_key;
pub use order_key::{append_total_order_key, total_order_key_len};
pub use parser::parse;
pub use planner::{PhysicalOperator, PhysicalPlan, ScanAccessPath, VectorAccessPath, plan};
pub use procedure::ProcedureCatalog;
pub use view::sparse_node_states_after_mutations;

/// Canonical scalar comparison entrypoint shared by native temporal backends. The selected
/// backend still constructs both operands; this helper owns only Cypher's three-valued equality
/// and ordering rules so the CPU reference cannot drift from the ordinary evaluator.
pub fn compare_temporal_scalars(
    left: &crate::ScalarValue,
    right: &crate::ScalarValue,
    operation: crate::execution::CompareOp,
) -> crate::Result<crate::ScalarValue> {
    let left = ResultValue::Scalar(left.clone());
    let right = ResultValue::Scalar(right.clone());
    let truth = match operation {
        crate::execution::CompareOp::Eq => value::equal(&left, &right)?,
        crate::execution::CompareOp::NotEq => value::equal(&left, &right)?.not(),
        crate::execution::CompareOp::Less => {
            value::compare_predicate(&left, &right, BinaryOperator::Less)?
        }
        crate::execution::CompareOp::LessOrEqual => {
            value::compare_predicate(&left, &right, BinaryOperator::LessOrEqual)?
        }
        crate::execution::CompareOp::Greater => {
            value::compare_predicate(&left, &right, BinaryOperator::Greater)?
        }
        crate::execution::CompareOp::GreaterOrEqual => {
            value::compare_predicate(&left, &right, BinaryOperator::GreaterOrEqual)?
        }
    };
    let ResultValue::Scalar(value) = truth.into_value() else {
        return Err(crate::Error::internal(
            "temporal comparison returned a non-scalar truth value",
        ));
    };
    Ok(value)
}
pub use irongraph_execution as execution;
pub use irongraph_graph as graph;
pub use irongraph_types as types;
pub use irongraph_types::document;
pub use irongraph_types::{
    Bookmark, DocumentItem, DocumentList, DocumentMap, EdgeId, Error, ErrorCode, Layer, NodeId,
    ProjectId, Result, ScalarValue,
};

pub mod cypher {
    pub use crate::*;
}
