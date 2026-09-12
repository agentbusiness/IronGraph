//! Backend-neutral contracts shared by query planning, execution devices, and transports.

#![allow(
    clippy::collapsible_if,
    clippy::double_must_use,
    clippy::filter_map_bool_then,
    clippy::get_first,
    clippy::large_enum_variant,
    clippy::len_zero,
    clippy::manual_range_contains,
    clippy::match_like_matches_macro,
    clippy::needless_range_loop,
    clippy::nonminimal_bool,
    clippy::obfuscated_if_else,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_lazy_evaluations,
    clippy::useless_conversion
)]

pub mod embedding;
pub mod order_key;
pub mod ordering;
pub mod procedure;
pub mod result;
pub mod temporal;

// These aliases let the resident execution implementation keep its original, explicit ownership
// paths while it lives in this crate.
pub mod execution {
    pub use crate::*;
}
pub use irongraph_graph as graph;
pub use irongraph_types as types;

pub mod backend;

pub use backend::*;
pub use embedding::TextEmbedding;
pub use order_key::{
    append_total_order_key, compare_i64_f64, encode_total_order_key, max_total_order_key_len,
    normalized_zoned_time, total_order_key_len,
};
pub use ordering::total_compare;
pub use procedure::{
    ProcedureDefinition, ProcedureField, ProcedureValueType, ResidentProcedureColumn,
    ResidentProcedureTable, ResidentProcedureTableFingerprint,
};
pub use result::{
    ColumnType, ExecutionStreamItem, QueryResult, ResultBatch, ResultColumn, ResultEdge,
    ResultNode, ResultValue, StatementStats,
};
pub use temporal::{
    CompareOp, ResidentRowValueType, ResidentTemporalAccessor, apply_duration_to_temporal_scalar,
    compare_temporal_scalars, evaluate_temporal_accessor, format_temporal_scalar,
};
