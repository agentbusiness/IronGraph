//! Backend-neutral typed row programs over one complete resident row stream.
//!
//! The source is either one wrapped resident graph pipeline or one bounded set of typed literal
//! input columns (for example a scalar `UNWIND`). This layer then evaluates one bounded SSA
//! program, applies one stable sort/pagination boundary, and publishes selected typed registers
//! together with every aligned graph column. No generic Cypher evaluator is part of this contract.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    mem::size_of,
    sync::Arc,
};

use ordered_float::OrderedFloat;
use tokio_util::sync::CancellationToken;

use crate::execution::{ResidentRowValueType, ResidentTemporalAccessor};
use crate::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    graph::LayerMask,
    types::{LabelId, PropertyId, RelationshipTypeId},
};

use super::{
    BackendKind, CompareOp, RESIDENT_NULL_ROW, ResidentDeviceCompletion, ResidentDirection,
    ResidentEntityBinding, ResidentExecutionId, ResidentExecutionObligation,
    ResidentExecutionReceipt, ResidentGraphLayoutVersion, ResidentNodeBinding,
    ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentObligationKind,
    ResidentObligationScope, ResidentRangeProgramRequest, ResidentVariablePath,
    ResidentVariablePathRequest, ensure_not_cancelled,
};

/// Compact instruction-shape limits shared by semantic-reference and accelerator implementations.
/// Logical row cardinality is governed by checked byte admission and addressability instead.
pub const RESIDENT_ROW_PROGRAM_MAX_REGISTERS: usize = 256;
pub const RESIDENT_ROW_PROGRAM_MAX_CONSTANTS: usize = 256;
pub const RESIDENT_ROW_PROGRAM_MAX_SORT_KEYS: usize = 64;
pub const RESIDENT_ROW_PROGRAM_MAX_PROJECTIONS: usize = 256;

/// Hard bounds for one complete staged nullable relation command. These limits are shared by the
/// future CPU and accelerator implementations; exceeding one is an admission failure, never a
/// request to spill an intermediate relation to the host.
pub const RESIDENT_NULLABLE_RELATION_MAX_STAGES: usize = 256;
pub const RESIDENT_NULLABLE_RELATION_MAX_BINDINGS: usize = 256;
pub const RESIDENT_NULLABLE_RELATION_MAX_OUTPUTS: usize = 256;
pub const RESIDENT_NULLABLE_RELATION_MAX_FILTERS: usize = 256;
pub const RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_NODES: usize = 256;
pub const RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_DEPTH: usize = 32;
pub const RESIDENT_NULLABLE_RELATION_MAX_LITERAL_STRING_BYTES: usize = 64 * 1024;
/// Maximum UTF-8 width of Cypher's canonical decimal rendering for one signed 64-bit integer.
/// `i64::MIN` is the widest value: `-9223372036854775808` (20 bytes).
pub const RESIDENT_NULLABLE_RELATION_INTEGER_STRING_MAXIMUM_BYTES: u32 = 20;
pub const RESIDENT_NULLABLE_RELATION_MAX_PROPERTY_LANES: usize =
    RESIDENT_NULLABLE_RELATION_MAX_FILTERS * RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_NODES * 2
        + RESIDENT_NULLABLE_RELATION_MAX_OUTPUTS;

/// Canonical null encoding for node and relationship binding columns in the staged relation ABI.
/// A backend must never use this value as a resident dense address.
pub const RESIDENT_NULLABLE_RELATION_NULL_ROW: u32 = RESIDENT_NULL_ROW;
/// One row-local operation. Registers are the zero-based positions of prior instructions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentRowOperation {
    /// One immutable typed source column. All such columns in a program must have exactly the same
    /// cardinality and may not be mixed with graph-property loads.
    InputColumn(ResidentRowColumn),
    LoadBooleanProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
    },
    LoadIntegerProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
    },
    LoadFloatProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
    },
    LoadStringProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
        /// Exact maximum UTF-8 width in the immutable source column for this graph revision.
        maximum_bytes: u32,
    },
    LoadDateProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
    },
    LoadLocalTimeProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
    },
    LoadZonedTimeProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
    },
    LoadLocalDateTimeProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
    },
    LoadZonedDateTimeProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
        /// Exact maximum timezone UTF-8 width in the immutable source column for this graph
        /// revision. This is admission metadata, never a host-computed order key.
        maximum_timezone_bytes: u32,
    },
    /// Loads one canonically stored flat homogeneous INTEGER list. `maximum_elements` is the
    /// exact largest non-null list in the immutable graph revision and therefore participates in
    /// admission and the manifest fingerprint.
    LoadListProperty {
        binding: ResidentEntityBinding,
        property: PropertyId,
        maximum_elements: u32,
    },
    BooleanConstant(bool),
    IntegerConstant(i64),
    /// Exact IEEE-754 bits; no host normalization is permitted.
    FloatConstant(u64),
    StringConstant(String),
    BooleanNot {
        operand: u16,
    },
    BooleanAnd {
        left: u16,
        right: u16,
    },
    NumericAdd {
        left: u16,
        right: u16,
    },
    NumericSubtract {
        left: u16,
        right: u16,
    },
    NumericMultiply {
        left: u16,
        right: u16,
    },
    /// Exact INTEGER remainder. Nulls propagate, zero divisors fail, and `i64::MIN % -1` is zero.
    IntegerModulo {
        left: u16,
        right: u16,
    },
    NumericNegate {
        operand: u16,
    },
    StringConcat {
        left: u16,
        right: u16,
    },
    /// Returns the first non-null STRING operand for each row. The output owns its own bounded
    /// string-arena slot so CPU and device execution preserve ordered `coalesce` semantics,
    /// including a valid empty left string.
    StringCoalesce {
        left: u16,
        right: u16,
    },
    /// Converts one nullable BOOLEAN register to canonical Cypher STRING bytes. The output owns
    /// at most five UTF-8 bytes per row (`true` or `false`); NULL propagates without hidden bytes.
    BooleanToString {
        operand: u16,
    },
    /// Builds one row-local flat INTEGER list from prior INTEGER registers. Element validity is
    /// retained independently from list validity, so a null indexed value remains a null element.
    List {
        elements: Vec<u16>,
    },
    /// Indexes a flat INTEGER list with a prior INTEGER register. Negative offsets address from
    /// the end; null and out-of-range indexes produce a null INTEGER result.
    ListIndex {
        list: u16,
        index: u16,
    },
    /// Concatenates two flat INTEGER lists. Null propagates.
    ListConcat {
        left: u16,
        right: u16,
    },
    /// Adds one already-normalized constant duration to a prior temporal register. Calendar and
    /// timezone semantics belong to the shared Cypher helper; row backends preserve only the
    /// operand's exact physical temporal shape.
    TemporalAddDuration {
        temporal: u16,
        months: i64,
        days: i64,
        seconds: i64,
        nanos: i32,
    },
    /// Projects one scalar accessor from a prior temporal register. Named-zone rules are an
    /// immutable transition image; the selected backend resolves offsets and local civil fields.
    TemporalAccessor {
        temporal: u16,
        accessor: ResidentTemporalAccessor,
        /// Exact maximum UTF-8 width for STRING output, zero for INTEGER output.
        maximum_string_bytes: u32,
        /// Serialized IANA transition image shared (through `Arc`) by all accessors in a request.
        named_zone_table: Arc<[u8]>,
    },
    /// Projects one integer accessor directly from a canonical DURATION property. Duration is
    /// deliberately not a row value type: its four components remain a typed property gather.
    LoadDurationAccessor {
        binding: ResidentEntityBinding,
        property: PropertyId,
        accessor: ResidentTemporalAccessor,
    },
}

/// One explicitly typed SSA definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentRowInstruction {
    pub output_type: ResidentRowValueType,
    pub operation: ResidentRowOperation,
}

/// Immutable row-local program. Instruction position is the resulting register number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentRowProgram {
    pub instructions: Vec<ResidentRowInstruction>,
}

impl ResidentRowProgram {
    pub fn validate(&self) -> Result<()> {
        if self.instructions.is_empty()
            || self.instructions.len() > RESIDENT_ROW_PROGRAM_MAX_REGISTERS
            || self.instructions.len() > u16::MAX as usize
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row program register count is outside its bounded contract",
            ));
        }
        let constants = self
            .instructions
            .iter()
            .filter(|instruction| {
                matches!(
                    instruction.operation,
                    ResidentRowOperation::BooleanConstant(_)
                        | ResidentRowOperation::IntegerConstant(_)
                        | ResidentRowOperation::FloatConstant(_)
                        | ResidentRowOperation::StringConstant(_)
                )
            })
            .count();
        if constants > RESIDENT_ROW_PROGRAM_MAX_CONSTANTS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row program constant count exceeds its bounded contract",
            ));
        }

        for (index, instruction) in self.instructions.iter().enumerate() {
            let prior_type = |register: u16| -> Result<ResidentRowValueType> {
                if register as usize >= index {
                    return Err(Error::new(
                        ErrorCode::QueryType,
                        "resident row instruction reads a register before it is defined",
                    ));
                }
                Ok(self.instructions[register as usize].output_type)
            };
            let require = |actual: ResidentRowValueType,
                           expected: ResidentRowValueType,
                           detail: &'static str|
             -> Result<()> {
                if actual != expected {
                    return Err(Error::new(ErrorCode::QueryType, detail));
                }
                Ok(())
            };
            match &instruction.operation {
                ResidentRowOperation::InputColumn(column) => {
                    column.validate(instruction.output_type, column.len())?;
                }
                ResidentRowOperation::LoadBooleanProperty { .. }
                | ResidentRowOperation::BooleanConstant(_) => require(
                    instruction.output_type,
                    ResidentRowValueType::Boolean,
                    "resident Boolean instruction has a non-Boolean output register",
                )?,
                ResidentRowOperation::LoadIntegerProperty { .. }
                | ResidentRowOperation::IntegerConstant(_) => require(
                    instruction.output_type,
                    ResidentRowValueType::Integer,
                    "resident integer instruction has a non-integer output register",
                )?,
                ResidentRowOperation::LoadFloatProperty { .. }
                | ResidentRowOperation::FloatConstant(_) => require(
                    instruction.output_type,
                    ResidentRowValueType::Float,
                    "resident float instruction has a non-float output register",
                )?,
                ResidentRowOperation::LoadStringProperty { .. }
                | ResidentRowOperation::StringConstant(_) => require(
                    instruction.output_type,
                    ResidentRowValueType::String,
                    "resident string instruction has a non-string output register",
                )?,
                ResidentRowOperation::LoadDateProperty { .. } => require(
                    instruction.output_type,
                    ResidentRowValueType::Date,
                    "resident date instruction has a non-date output register",
                )?,
                ResidentRowOperation::LoadLocalTimeProperty { .. } => require(
                    instruction.output_type,
                    ResidentRowValueType::LocalTime,
                    "resident local-time instruction has a non-local-time output register",
                )?,
                ResidentRowOperation::LoadZonedTimeProperty { .. } => require(
                    instruction.output_type,
                    ResidentRowValueType::ZonedTime,
                    "resident zoned-time instruction has a non-zoned-time output register",
                )?,
                ResidentRowOperation::LoadLocalDateTimeProperty { .. } => require(
                    instruction.output_type,
                    ResidentRowValueType::LocalDateTime,
                    "resident local-datetime instruction has a non-local-datetime output register",
                )?,
                ResidentRowOperation::LoadZonedDateTimeProperty { .. } => require(
                    instruction.output_type,
                    ResidentRowValueType::ZonedDateTime,
                    "resident zoned-datetime instruction has a non-zoned-datetime output register",
                )?,
                ResidentRowOperation::LoadListProperty { .. } => require(
                    instruction.output_type,
                    ResidentRowValueType::List,
                    "resident list instruction has a non-list output register",
                )?,
                ResidentRowOperation::BooleanNot { operand } => {
                    require(
                        prior_type(*operand)?,
                        ResidentRowValueType::Boolean,
                        "resident Boolean NOT requires a Boolean input register",
                    )?;
                    require(
                        instruction.output_type,
                        ResidentRowValueType::Boolean,
                        "resident Boolean NOT has a non-Boolean output register",
                    )?;
                }
                ResidentRowOperation::BooleanAnd { left, right } => {
                    require(
                        prior_type(*left)?,
                        ResidentRowValueType::Boolean,
                        "resident Boolean AND requires Boolean input registers",
                    )?;
                    require(
                        prior_type(*right)?,
                        ResidentRowValueType::Boolean,
                        "resident Boolean AND requires Boolean input registers",
                    )?;
                    require(
                        instruction.output_type,
                        ResidentRowValueType::Boolean,
                        "resident Boolean AND has a non-Boolean output register",
                    )?;
                }
                ResidentRowOperation::NumericAdd { left, right }
                | ResidentRowOperation::NumericSubtract { left, right }
                | ResidentRowOperation::NumericMultiply { left, right } => {
                    let left = prior_type(*left)?;
                    let right = prior_type(*right)?;
                    if !left.is_numeric() || !right.is_numeric() {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident numeric instruction requires numeric input registers",
                        ));
                    }
                    let expected = if left == ResidentRowValueType::Float
                        || right == ResidentRowValueType::Float
                    {
                        ResidentRowValueType::Float
                    } else {
                        ResidentRowValueType::Integer
                    };
                    require(
                        instruction.output_type,
                        expected,
                        "resident numeric instruction output does not match INTEGER/FLOAT promotion",
                    )?;
                }
                ResidentRowOperation::IntegerModulo { left, right } => {
                    require(
                        prior_type(*left)?,
                        ResidentRowValueType::Integer,
                        "resident integer modulo requires INTEGER input registers",
                    )?;
                    require(
                        prior_type(*right)?,
                        ResidentRowValueType::Integer,
                        "resident integer modulo requires INTEGER input registers",
                    )?;
                    require(
                        instruction.output_type,
                        ResidentRowValueType::Integer,
                        "resident integer modulo has a non-INTEGER output register",
                    )?;
                }
                ResidentRowOperation::NumericNegate { operand } => {
                    let operand = prior_type(*operand)?;
                    if !operand.is_numeric() {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident numeric negation requires a numeric input register",
                        ));
                    }
                    require(
                        instruction.output_type,
                        operand,
                        "resident numeric negation must preserve its input register type",
                    )?;
                }
                ResidentRowOperation::StringConcat { left, right } => {
                    require(
                        prior_type(*left)?,
                        ResidentRowValueType::String,
                        "resident string concatenation requires string input registers",
                    )?;
                    require(
                        prior_type(*right)?,
                        ResidentRowValueType::String,
                        "resident string concatenation requires string input registers",
                    )?;
                    require(
                        instruction.output_type,
                        ResidentRowValueType::String,
                        "resident string concatenation has a non-string output register",
                    )?;
                }
                ResidentRowOperation::StringCoalesce { left, right } => {
                    require(
                        prior_type(*left)?,
                        ResidentRowValueType::String,
                        "resident string coalesce requires string input registers",
                    )?;
                    require(
                        prior_type(*right)?,
                        ResidentRowValueType::String,
                        "resident string coalesce requires string input registers",
                    )?;
                    require(
                        instruction.output_type,
                        ResidentRowValueType::String,
                        "resident string coalesce has a non-string output register",
                    )?;
                }
                ResidentRowOperation::BooleanToString { operand } => {
                    require(
                        prior_type(*operand)?,
                        ResidentRowValueType::Boolean,
                        "resident Boolean-to-string conversion requires a Boolean input register",
                    )?;
                    require(
                        instruction.output_type,
                        ResidentRowValueType::String,
                        "resident Boolean-to-string conversion has a non-string output register",
                    )?;
                }
                ResidentRowOperation::List { elements } => {
                    for element in elements {
                        require(
                            prior_type(*element)?,
                            ResidentRowValueType::Integer,
                            "resident list construction requires INTEGER element registers",
                        )?;
                    }
                    require(
                        instruction.output_type,
                        ResidentRowValueType::List,
                        "resident list construction has a non-list output register",
                    )?;
                }
                ResidentRowOperation::ListIndex { list, index } => {
                    require(
                        prior_type(*list)?,
                        ResidentRowValueType::List,
                        "resident list indexing requires a LIST input register",
                    )?;
                    require(
                        prior_type(*index)?,
                        ResidentRowValueType::Integer,
                        "resident list indexing requires an INTEGER index register",
                    )?;
                    require(
                        instruction.output_type,
                        ResidentRowValueType::Integer,
                        "resident list indexing has a non-INTEGER output register",
                    )?;
                }
                ResidentRowOperation::ListConcat { left, right } => {
                    require(
                        prior_type(*left)?,
                        ResidentRowValueType::List,
                        "resident list concatenation requires LIST input registers",
                    )?;
                    require(
                        prior_type(*right)?,
                        ResidentRowValueType::List,
                        "resident list concatenation requires LIST input registers",
                    )?;
                    require(
                        instruction.output_type,
                        ResidentRowValueType::List,
                        "resident list concatenation has a non-list output register",
                    )?;
                }
                ResidentRowOperation::TemporalAddDuration {
                    temporal, nanos, ..
                } => {
                    let temporal_type = prior_type(*temporal)?;
                    if !temporal_type.is_temporal() {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident temporal duration addition requires a temporal input register",
                        ));
                    }
                    require(
                        instruction.output_type,
                        temporal_type,
                        "resident temporal duration addition must preserve its input register type",
                    )?;
                    if !(0..1_000_000_000).contains(nanos) {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident temporal duration addition requires normalized nanoseconds",
                        ));
                    }
                }
                ResidentRowOperation::TemporalAccessor {
                    temporal,
                    accessor,
                    maximum_string_bytes,
                    named_zone_table,
                } => {
                    let temporal_type = prior_type(*temporal)?;
                    if !accessor.supports(temporal_type) {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident temporal accessor does not apply to its input register",
                        ));
                    }
                    require(
                        instruction.output_type,
                        accessor.output_type(),
                        "resident temporal accessor has the wrong output register type",
                    )?;
                    let expected_string_bytes = match accessor {
                        ResidentTemporalAccessor::Timezone
                            if temporal_type == ResidentRowValueType::ZonedDateTime =>
                        {
                            self.timezone_register_capacity(*temporal)?.ok_or_else(|| {
                                Error::internal(
                                    "validated temporal accessor timezone capacity disappeared",
                                )
                            })?
                        }
                        ResidentTemporalAccessor::Timezone | ResidentTemporalAccessor::Offset => 9,
                        _ => 0,
                    };
                    if usize::try_from(*maximum_string_bytes).ok() != Some(expected_string_bytes) {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident temporal accessor string capacity is not canonical",
                        ));
                    }
                    if temporal_type != ResidentRowValueType::ZonedDateTime
                        && !named_zone_table.is_empty()
                    {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident non-zoned-datetime accessor carries named-zone rules",
                        ));
                    }
                    u32::try_from(named_zone_table.len()).map_err(|_| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident named-zone rules exceed u32 addressing",
                        )
                    })?;
                }
                ResidentRowOperation::LoadDurationAccessor { accessor, .. } => {
                    if !accessor.is_duration() {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident duration property accessor is not a DURATION field",
                        ));
                    }
                    require(
                        instruction.output_type,
                        ResidentRowValueType::Integer,
                        "resident duration accessor has a non-INTEGER output register",
                    )?;
                }
            }
        }
        for register in 0..self.instructions.len() {
            let register = u16::try_from(register).map_err(|_| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident string register exceeds u16",
                )
            })?;
            self.string_register_capacity(register)?;
            self.timezone_register_capacity(register)?;
            self.list_register_capacity(register)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn register_type(&self, register: u16) -> Option<ResidentRowValueType> {
        self.instructions
            .get(register as usize)
            .map(|instruction| instruction.output_type)
    }

    /// Maximum bytes owned by one row of a STRING register. This is part of scratch admission,
    /// not a runtime estimate; every concatenation is checked before any device allocation.
    pub fn string_register_capacity(&self, register: u16) -> Result<Option<usize>> {
        let Some(instruction) = self.instructions.get(register as usize) else {
            return Ok(None);
        };
        let capacity = match &instruction.operation {
            ResidentRowOperation::InputColumn(ResidentRowColumn::String { offsets, .. }) => offsets
                .windows(2)
                .map(|pair| (pair[1] - pair[0]) as usize)
                .max()
                .unwrap_or(0),
            ResidentRowOperation::LoadStringProperty { maximum_bytes, .. } => {
                *maximum_bytes as usize
            }
            ResidentRowOperation::StringConstant(value) => value.len(),
            ResidentRowOperation::BooleanToString { .. } => 5,
            ResidentRowOperation::StringConcat { left, right } => self
                .string_register_capacity(*left)?
                .ok_or_else(|| Error::internal("validated string left capacity disappeared"))?
                .checked_add(self.string_register_capacity(*right)?.ok_or_else(|| {
                    Error::internal("validated string right capacity disappeared")
                })?)
                .ok_or_else(scratch_overflow)?,
            ResidentRowOperation::StringCoalesce { left, right } => self
                .string_register_capacity(*left)?
                .ok_or_else(|| Error::internal("validated string left capacity disappeared"))?
                .max(self.string_register_capacity(*right)?.ok_or_else(|| {
                    Error::internal("validated string right capacity disappeared")
                })?),
            ResidentRowOperation::TemporalAccessor {
                maximum_string_bytes,
                ..
            } if instruction.output_type == ResidentRowValueType::String => {
                *maximum_string_bytes as usize
            }
            _ => return Ok(None),
        };
        if capacity > u32::MAX as usize {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident string register capacity exceeds u32",
            ));
        }
        Ok(Some(capacity))
    }

    /// Maximum timezone bytes owned by one row of a ZONED DATETIME register. The fixed temporal
    /// tuple remains in the register frame; only its variable-width timezone lives in this arena.
    ///
    /// # Errors
    ///
    /// Returns an admission error when the register's immutable timezone width exceeds the
    /// addressable row contract.
    pub fn timezone_register_capacity(&self, register: u16) -> Result<Option<usize>> {
        let Some(instruction) = self.instructions.get(register as usize) else {
            return Ok(None);
        };
        let capacity = match &instruction.operation {
            ResidentRowOperation::InputColumn(ResidentRowColumn::ZonedDateTime {
                timezone_offsets,
                ..
            }) => timezone_offsets
                .windows(2)
                .map(|pair| (pair[1] - pair[0]) as usize)
                .max()
                .unwrap_or(0),
            ResidentRowOperation::LoadZonedDateTimeProperty {
                maximum_timezone_bytes,
                ..
            } => *maximum_timezone_bytes as usize,
            ResidentRowOperation::TemporalAddDuration { temporal, .. }
                if instruction.output_type == ResidentRowValueType::ZonedDateTime =>
            {
                self.timezone_register_capacity(*temporal)?.ok_or_else(|| {
                    Error::internal(
                        "validated zoned-datetime duration operand capacity disappeared",
                    )
                })?
            }
            _ => return Ok(None),
        };
        if capacity > u32::MAX as usize {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident zoned-datetime timezone capacity exceeds u32",
            ));
        }
        Ok(Some(capacity))
    }

    /// Maximum number of INTEGER elements owned by one row of a LIST register. Values and their
    /// element-validity bytes live in a separately admitted row-major arena.
    pub fn list_register_capacity(&self, register: u16) -> Result<Option<usize>> {
        let Some(instruction) = self.instructions.get(register as usize) else {
            return Ok(None);
        };
        let capacity = match &instruction.operation {
            ResidentRowOperation::InputColumn(ResidentRowColumn::List { offsets, .. }) => offsets
                .windows(2)
                .map(|pair| (pair[1] - pair[0]) as usize)
                .max()
                .unwrap_or(0),
            ResidentRowOperation::LoadListProperty {
                maximum_elements, ..
            } => *maximum_elements as usize,
            ResidentRowOperation::List { elements } => elements.len(),
            ResidentRowOperation::ListConcat { left, right } => {
                self.list_register_capacity(*left)?
                    .ok_or_else(|| Error::internal("validated list left capacity disappeared"))?
                    .checked_add(self.list_register_capacity(*right)?.ok_or_else(|| {
                        Error::internal("validated list right capacity disappeared")
                    })?)
                    .ok_or_else(scratch_overflow)?
            }
            _ => return Ok(None),
        };
        if capacity > u32::MAX as usize {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident list register capacity exceeds u32",
            ));
        }
        Ok(Some(capacity))
    }
}

/// One key in the single stable row-program sort boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentRowSortKey {
    pub register: u16,
    pub descending: bool,
    pub nulls_first: bool,
}

/// Deterministic semantic fingerprint. This is an integrity fence, not a cryptographic digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResidentRowManifestFingerprint(pub [u8; 32]);

/// Independently enumerable obligations for every instruction and the sort/pagination boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentRowProgramManifest {
    pub instruction_obligations: Vec<ResidentExecutionObligation>,
    pub sort_obligation: ResidentExecutionObligation,
    pub fingerprint: ResidentRowManifestFingerprint,
}

impl ResidentRowProgramManifest {
    /// Creates a manifest with consecutive, non-zero caller-owned obligation IDs.
    pub fn build(
        program: &ResidentRowProgram,
        sort_keys: &[ResidentRowSortKey],
        offset: usize,
        limit: usize,
        max_output_rows: usize,
        final_registers: &[u16],
        first_obligation_id: u64,
    ) -> Result<Self> {
        program.validate()?;
        if first_obligation_id == 0 {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row manifest requires non-zero obligation IDs",
            ));
        }
        let mut instruction_obligations = Vec::with_capacity(program.instructions.len());
        for index in 0..program.instructions.len() {
            let id = first_obligation_id
                .checked_add(index as u64)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident row manifest obligation ID overflow",
                    )
                })?;
            instruction_obligations.push(ResidentExecutionObligation {
                id,
                kind: ResidentObligationKind::Expression,
                scope: ResidentObligationScope::Expression(index as u16),
            });
        }
        let sort_index = u16::try_from(program.instructions.len()).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row sort scope exceeds u16",
            )
        })?;
        let sort_obligation = ResidentExecutionObligation {
            id: first_obligation_id
                .checked_add(program.instructions.len() as u64)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident row sort obligation ID overflow",
                    )
                })?,
            kind: ResidentObligationKind::Sort,
            // Existing Pattern and Mutation scopes remain unchanged. The row manifest binds this
            // otherwise-unused expression index specifically to its sort boundary.
            scope: ResidentObligationScope::Expression(sort_index),
        };
        let mut manifest = Self {
            instruction_obligations,
            sort_obligation,
            fingerprint: ResidentRowManifestFingerprint([0; 32]),
        };
        manifest.fingerprint = resident_row_manifest_fingerprint(
            program,
            sort_keys,
            offset,
            limit,
            max_output_rows,
            final_registers,
            &manifest,
        )?;
        Ok(manifest)
    }

    pub fn obligations(&self) -> impl Iterator<Item = ResidentExecutionObligation> + '_ {
        self.instruction_obligations
            .iter()
            .copied()
            .chain(std::iter::once(self.sort_obligation))
    }

    fn validate_shape(&self, instruction_count: usize) -> Result<()> {
        if self.instruction_obligations.len() != instruction_count {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row manifest does not cover every instruction",
            ));
        }
        let mut ids = BTreeSet::new();
        let mut scopes = BTreeSet::new();
        for (index, obligation) in self.instruction_obligations.iter().enumerate() {
            let expected_scope = ResidentObligationScope::Expression(index as u16);
            if obligation.id == 0
                || obligation.kind != ResidentObligationKind::Expression
                || obligation.scope != expected_scope
                || !ids.insert(obligation.id)
                || !scopes.insert(obligation.scope)
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident row manifest has an invalid or duplicate instruction obligation",
                ));
            }
        }
        let expected_sort_scope = ResidentObligationScope::Expression(
            u16::try_from(instruction_count).map_err(|_| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident row sort scope exceeds u16",
                )
            })?,
        );
        if self.sort_obligation.id == 0
            || self.sort_obligation.kind != ResidentObligationKind::Sort
            || self.sort_obligation.scope != expected_sort_scope
            || !ids.insert(self.sort_obligation.id)
            || !scopes.insert(self.sort_obligation.scope)
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row manifest has an invalid or duplicate sort obligation",
            ));
        }
        Ok(())
    }
}

/// One complete typed-row execution tied to an exact resident graph image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentRowProgramRequest {
    pub project: ProjectId,
    pub expected_bookmark: Bookmark,
    pub expected_graph_revision: u64,
    pub expected_layout_version: ResidentGraphLayoutVersion,
    /// Caller-owned and fresh for this semantic execution.
    pub execution: ResidentExecutionId,
    pub input: ResidentNodePipelineRequest,
    pub program: ResidentRowProgram,
    pub manifest: ResidentRowProgramManifest,
    pub sort_keys: Vec<ResidentRowSortKey>,
    pub offset: usize,
    pub limit: usize,
    pub max_output_rows: usize,
    pub final_registers: Vec<u16>,
}

impl ResidentRowProgramRequest {
    /// Returns the exact typed-literal source cardinality, or `None` for a graph-pipeline source.
    pub fn scalar_input_rows(&self) -> Result<Option<usize>> {
        let mut rows = None;
        for instruction in &self.program.instructions {
            if let ResidentRowOperation::InputColumn(column) = &instruction.operation {
                match rows {
                    Some(expected) if expected != column.len() => {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident typed input columns have different cardinalities",
                        ));
                    }
                    Some(_) => {}
                    None => rows = Some(column.len()),
                }
            }
        }
        Ok(rows)
    }

    #[must_use]
    pub fn has_graph_input(&self) -> bool {
        !self.program.instructions.iter().any(|instruction| {
            matches!(&instruction.operation, ResidentRowOperation::InputColumn(_))
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.execution.high == 0 && self.execution.low == 0 {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row program requires a non-zero execution ID",
            ));
        }
        if self.project != self.input.project {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row program and wrapped pipeline name different projects",
            ));
        }
        self.program.validate()?;
        let scalar_input_rows = self.scalar_input_rows()?;
        self.input.validate_mutation()?;
        if !self.input.orders.is_empty() || self.input.offset != 0 || self.input.limit != usize::MAX
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "wrapped resident pipeline must leave ordering, offset, and limit to the row program",
            ));
        }
        if !self.input.integer_projections.is_empty()
            || !self.input.property_null_projections.is_empty()
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "wrapped resident pipeline must leave typed projection to the row program",
            ));
        }
        if let Some(rows) = scalar_input_rows {
            let scalar_placeholder_is_empty = self.input.labels.is_empty()
                && !self.input.initial_optional
                && self.input.expansion.is_none()
                && self.input.continuations.is_empty()
                && self.input.correlated_optional.is_none()
                && self.input.relationship_null_filter.is_none()
                && self.input.predicates.is_empty()
                && self.input.property_filters.is_empty()
                && self.input.value_matrix.is_none()
                && self.input.mutation.is_none();
            if !scalar_placeholder_is_empty || self.input.max_output_rows != rows {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident typed literal input has a non-empty graph source descriptor",
                ));
            }
            if self.program.instructions.iter().any(|instruction| {
                matches!(
                    instruction.operation,
                    ResidentRowOperation::LoadBooleanProperty { .. }
                        | ResidentRowOperation::LoadIntegerProperty { .. }
                        | ResidentRowOperation::LoadFloatProperty { .. }
                        | ResidentRowOperation::LoadStringProperty { .. }
                        | ResidentRowOperation::LoadDateProperty { .. }
                        | ResidentRowOperation::LoadLocalTimeProperty { .. }
                        | ResidentRowOperation::LoadZonedTimeProperty { .. }
                        | ResidentRowOperation::LoadLocalDateTimeProperty { .. }
                        | ResidentRowOperation::LoadZonedDateTimeProperty { .. }
                        | ResidentRowOperation::LoadListProperty { .. }
                        | ResidentRowOperation::LoadDurationAccessor { .. }
                )
            }) {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident typed literal input cannot be mixed with graph-property loads",
                ));
            }
        }
        if self.sort_keys.len() > RESIDENT_ROW_PROGRAM_MAX_SORT_KEYS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row sort-key count exceeds its bounded contract",
            ));
        }
        if self.final_registers.len() > RESIDENT_ROW_PROGRAM_MAX_PROJECTIONS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row projection count exceeds its bounded contract",
            ));
        }
        for key in &self.sort_keys {
            if self.program.register_type(key.register).is_none() {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "resident row sort key names an absent register",
                ));
            }
        }
        for register in &self.final_registers {
            let Some(_value_type) = self.program.register_type(*register) else {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "resident row projection names an absent register",
                ));
            };
        }
        for instruction in &self.program.instructions {
            match instruction.operation {
                ResidentRowOperation::LoadBooleanProperty { binding, .. }
                | ResidentRowOperation::LoadIntegerProperty { binding, .. }
                | ResidentRowOperation::LoadFloatProperty { binding, .. }
                | ResidentRowOperation::LoadStringProperty { binding, .. }
                | ResidentRowOperation::LoadDateProperty { binding, .. }
                | ResidentRowOperation::LoadLocalTimeProperty { binding, .. }
                | ResidentRowOperation::LoadZonedTimeProperty { binding, .. }
                | ResidentRowOperation::LoadLocalDateTimeProperty { binding, .. }
                | ResidentRowOperation::LoadZonedDateTimeProperty { binding, .. }
                | ResidentRowOperation::LoadListProperty { binding, .. }
                | ResidentRowOperation::LoadDurationAccessor { binding, .. } => {
                    validate_binding(&self.input, binding)?;
                }
                _ => {}
            }
        }
        self.manifest
            .validate_shape(self.program.instructions.len())?;
        let fingerprint = resident_row_manifest_fingerprint(
            &self.program,
            &self.sort_keys,
            self.offset,
            self.limit,
            self.max_output_rows,
            &self.final_registers,
            &self.manifest,
        )?;
        if self.manifest.fingerprint != fingerprint {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row manifest fingerprint does not match its immutable program",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn obligations(&self) -> impl Iterator<Item = ResidentExecutionObligation> + '_ {
        self.manifest.obligations()
    }

    pub fn output_cardinality(&self, input_rows: usize) -> Result<usize> {
        if input_rows > self.input.max_output_rows {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "wrapped resident pipeline returned more rows than admitted",
            ));
        }
        let output = input_rows.saturating_sub(self.offset).min(self.limit);
        if output > self.max_output_rows {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident row program result exceeds its output budget",
            ));
        }
        Ok(output)
    }

    /// Exact logical bytes simultaneously live during row-program evaluation. Every term uses
    /// checked arithmetic; exceeding the addressable contract fails rather than truncating.
    pub fn scratch_bytes(&self, input_rows: usize) -> Result<usize> {
        self.validate()?;
        let output_rows = self.output_cardinality(input_rows)?;
        let aligned_u32_columns = if self.has_graph_input() {
            aligned_u32_column_count(&self.input)?
        } else {
            0
        };
        let register_width =
            self.program
                .instructions
                .iter()
                .try_fold(0_usize, |width, instruction| {
                    width
                        .checked_add(instruction.output_type.payload_bytes())
                        .and_then(|width| width.checked_add(size_of::<u8>()))
                        .ok_or_else(scratch_overflow)
                })?;
        let projected_width =
            self.final_registers
                .iter()
                .try_fold(0_usize, |width, register| {
                    let value_type = self.program.register_type(*register).ok_or_else(|| {
                        Error::new(
                            ErrorCode::QueryType,
                            "resident row projection names an absent register",
                        )
                    })?;
                    width
                        .checked_add(value_type.payload_bytes())
                        .and_then(|width| width.checked_add(size_of::<u8>()))
                        .ok_or_else(scratch_overflow)
                })?;
        let input_base = checked_product(
            input_rows,
            checked_product(aligned_u32_columns, size_of::<u32>())?,
        )?;
        let registers = checked_product(input_rows, register_width)?;
        let variable_arena_width = self.program.instructions.iter().enumerate().try_fold(
            0_usize,
            |width, (register, instruction)| {
                let register = u16::try_from(register).map_err(|_| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident variable-width register exceeds u16",
                    )
                })?;
                let capacity = match instruction.output_type {
                    ResidentRowValueType::String => self
                        .program
                        .string_register_capacity(register)?
                        .ok_or_else(|| Error::internal("resident string capacity disappeared"))?,
                    ResidentRowValueType::ZonedDateTime => self
                        .program
                        .timezone_register_capacity(register)?
                        .ok_or_else(|| Error::internal("resident timezone capacity disappeared"))?,
                    ResidentRowValueType::List => self
                        .program
                        .list_register_capacity(register)?
                        .ok_or_else(|| Error::internal("resident list capacity disappeared"))?
                        .checked_mul(size_of::<i64>() + size_of::<u8>())
                        .ok_or_else(scratch_overflow)?,
                    _ => return Ok(width),
                };
                width.checked_add(capacity).ok_or_else(scratch_overflow)
            },
        )?;
        let projected_variable_arena_width =
            self.final_registers
                .iter()
                .try_fold(0_usize, |width, register| {
                    let capacity = match self.program.register_type(*register).ok_or_else(|| {
                        Error::new(
                            ErrorCode::QueryType,
                            "resident row projection names an absent register",
                        )
                    })? {
                        ResidentRowValueType::String => self
                            .program
                            .string_register_capacity(*register)?
                            .ok_or_else(|| {
                                Error::internal("resident projected string capacity disappeared")
                            })?,
                        ResidentRowValueType::ZonedDateTime => self
                            .program
                            .timezone_register_capacity(*register)?
                            .ok_or_else(|| {
                                Error::internal("resident projected timezone capacity disappeared")
                            })?,
                        ResidentRowValueType::List => self
                            .program
                            .list_register_capacity(*register)?
                            .ok_or_else(|| {
                                Error::internal("resident projected list capacity disappeared")
                            })?
                            .checked_mul(size_of::<i64>() + size_of::<u8>())
                            .ok_or_else(scratch_overflow)?,
                        _ => return Ok(width),
                    };
                    width.checked_add(capacity).ok_or_else(scratch_overflow)
                })?;
        let variable_arena = checked_product(input_rows, variable_arena_width)?;
        let positions = checked_product(input_rows, size_of::<u64>())?;
        let output_base = checked_product(
            output_rows,
            checked_product(aligned_u32_columns, size_of::<u32>())?,
        )?;
        let projections = checked_product(output_rows, projected_width)?;
        let projected_variable_arena =
            checked_product(output_rows, projected_variable_arena_width)?;
        let receipts = checked_product(
            self.obligations().count(),
            size_of::<ResidentExecutionReceipt>(),
        )?;
        [
            input_base,
            registers,
            variable_arena,
            positions,
            output_base,
            projections,
            projected_variable_arena,
            receipts,
        ]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            total.checked_add(bytes).ok_or_else(scratch_overflow)
        })
    }
}

/// Hard bounds for one complete graph-free list/quantifier command. These are execution bounds,
/// not parser limits: a query outside them is rejected before a backend is selected rather than
/// being split across native and generic evaluators.
pub const RESIDENT_QUANTIFIER_MAX_STAGES: usize = 64;
pub const RESIDENT_QUANTIFIER_MAX_SLOTS: usize = 128;
pub const RESIDENT_QUANTIFIER_MAX_EXPRESSION_NODES: usize = 2_048;
pub const RESIDENT_QUANTIFIER_MAX_EXPRESSION_DEPTH: usize = 32;
pub const RESIDENT_QUANTIFIER_MAX_LITERAL_ITEMS: usize = 4_096;

/// Compiler-assigned value slot. Relation slots and expression-local iterator slots share one
/// immutable address space so native backends never resolve variable names at execution time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResidentQuantifierSlot(pub u16);

/// Dense resident identity carried only inside one generation-fenced quantifier command. The
/// handle is an opaque graph-row locator, not a Cypher integer value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResidentQuantifierNodeHandle(pub u32);

/// Dense resident relationship identity carried only inside one generation-fenced quantifier
/// command. Equality is identity equality, independently of relationship properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResidentQuantifierRelationshipHandle(pub u32);

/// One typed entity value in an ordered resident list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidentQuantifierEntityHandle {
    Node(ResidentQuantifierNodeHandle),
    Relationship(ResidentQuantifierRelationshipHandle),
}

/// Ordered identity-preserving output of `tail(nodes(p))` or `tail(relationships(p))`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentQuantifierEntityList {
    pub values: Vec<ResidentQuantifierEntityHandle>,
}

impl ResidentQuantifierEntityList {
    /// Deterministic structural hash used as the first grouping discriminator. Callers must still
    /// verify full ordered-list equality after a hash match.
    #[must_use]
    pub fn structural_hash(&self) -> [u8; 32] {
        let mut hasher =
            blake3::Hasher::new_derive_key("irongraph.resident-quantifier-entity-list.v1");
        hasher.update(&(self.values.len() as u64).to_le_bytes());
        for value in &self.values {
            match value {
                ResidentQuantifierEntityHandle::Node(handle) => {
                    hasher.update(&[1]);
                    hasher.update(&handle.0.to_le_bytes());
                }
                ResidentQuantifierEntityHandle::Relationship(handle) => {
                    hasher.update(&[2]);
                    hasher.update(&handle.0.to_le_bytes());
                }
            }
        }
        *hasher.finalize().as_bytes()
    }
}

/// Entity domain materialized by a fused variable-path quantifier source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ResidentQuantifierEntityKind {
    Node = 1,
    Relationship = 2,
}

/// Canonical scalar shape admitted for one entity-property lane. The shape is part of the
/// immutable command fingerprint, so an accelerator never guesses how to decode a resident
/// column. Temporal, byte, list, map, and document values remain unsupported in this command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidentQuantifierEntityPropertyShape {
    Absent,
    Boolean,
    Integer,
    Float,
    String { maximum_bytes: u32 },
    MixedScalar { maximum_string_bytes: u32 },
}

impl ResidentQuantifierEntityPropertyShape {
    pub fn from_values(values: impl IntoIterator<Item = ScalarValue>) -> Result<Option<Self>> {
        const BOOLEAN: u8 = 1 << 0;
        const INTEGER: u8 = 1 << 1;
        const FLOAT: u8 = 1 << 2;
        const STRING: u8 = 1 << 3;

        let mut kinds = 0_u8;
        let mut maximum_string_bytes = 0_usize;
        for value in values {
            match value {
                ScalarValue::Null => {}
                ScalarValue::Boolean(_) => kinds |= BOOLEAN,
                ScalarValue::Integer(_) => kinds |= INTEGER,
                ScalarValue::Float(_) => kinds |= FLOAT,
                ScalarValue::String(value) => {
                    kinds |= STRING;
                    maximum_string_bytes = maximum_string_bytes.max(value.len());
                }
                ScalarValue::Bytes(_)
                | ScalarValue::Date(_)
                | ScalarValue::LocalTime(_)
                | ScalarValue::ZonedTime { .. }
                | ScalarValue::LocalDateTime { .. }
                | ScalarValue::ZonedDateTime { .. }
                | ScalarValue::Duration { .. }
                | ScalarValue::List(_)
                | ScalarValue::Map(_) => return Ok(None),
            }
        }
        let maximum_string_bytes = u32::try_from(maximum_string_bytes).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident quantifier property string width exceeds u32",
            )
        })?;
        Ok(Some(match kinds {
            0 => Self::Absent,
            BOOLEAN => Self::Boolean,
            INTEGER => Self::Integer,
            FLOAT => Self::Float,
            STRING => Self::String {
                maximum_bytes: maximum_string_bytes,
            },
            _ => Self::MixedScalar {
                maximum_string_bytes,
            },
        }))
    }

    pub const fn maximum_string_bytes(self) -> u32 {
        match self {
            Self::String { maximum_bytes } => maximum_bytes,
            Self::MixedScalar {
                maximum_string_bytes,
            } => maximum_string_bytes,
            Self::Absent | Self::Boolean | Self::Integer | Self::Float => 0,
        }
    }
}

/// Immutable catalog resolution for one property that may be read from a source entity handle.
/// `None` records a statically absent property and therefore evaluates to Cypher null.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierEntityProperty {
    pub key: String,
    pub property: Option<PropertyId>,
    pub shape: ResidentQuantifierEntityPropertyShape,
}

/// Generation-fenced graph read performed while constructing/evaluating an entity-list source.
/// `property: None` records the entity identity/topology read; `Some` records a canonical property
/// column read. Entries are published sorted and deduplicated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentQuantifierEntityReadDependency {
    pub entity: ResidentQuantifierEntityHandle,
    pub property: Option<PropertyId>,
}

pub const RESIDENT_QUANTIFIER_MAX_ENTITY_READ_DEPENDENCIES: usize = 4_194_304;

/// Initial relation for one complete quantifier command. `VariablePathEntityList` executes the
/// complete resident path request inside the same backend call and seeds `output` with one ordered
/// tail entity list per accepted path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentQuantifierSource {
    Unit,
    /// Device-generated row source for `UNWIND range(...)`. The range is represented by its
    /// scalar descriptor and is never materialized as a list before row generation.
    Range {
        request: ResidentRangeProgramRequest,
        output: ResidentQuantifierSlot,
        obligation: ResidentExecutionObligation,
    },
    VariablePathEntityList {
        path: ResidentVariablePathRequest,
        output: ResidentQuantifierSlot,
        entity_kind: ResidentQuantifierEntityKind,
        /// The exact remaining TCK family uses `tail(...)`; admission therefore requires one.
        skip: u32,
        properties: Vec<ResidentQuantifierEntityProperty>,
        materialize_obligation: ResidentExecutionObligation,
    },
}

impl ResidentQuantifierSource {
    fn initial_slots(&self) -> BTreeSet<ResidentQuantifierSlot> {
        match self {
            Self::Unit => BTreeSet::new(),
            Self::Range { output, .. } | Self::VariablePathEntityList { output, .. } => {
                BTreeSet::from([*output])
            }
        }
    }

    fn maximum_entity_list_items(&self) -> Result<usize> {
        let Self::VariablePathEntityList {
            path,
            entity_kind,
            skip,
            ..
        } = self
        else {
            return Ok(0);
        };
        let maximum_hops = path
            .segments
            .iter()
            .try_fold(0_usize, |total, segment| {
                let segment_maximum = segment
                    .maximum_hops
                    .map_or(path.expected_edge_slots, |maximum| maximum as usize)
                    .min(path.expected_edge_slots);
                total
                    .checked_add(segment_maximum)
                    .ok_or_else(scratch_overflow)
            })?
            .min(path.expected_edge_slots);
        let full_items = match entity_kind {
            ResidentQuantifierEntityKind::Node => {
                maximum_hops.checked_add(1).ok_or_else(scratch_overflow)?
            }
            ResidentQuantifierEntityKind::Relationship => maximum_hops,
        };
        Ok(full_items.saturating_sub(*skip as usize))
    }

    fn validate(
        &self,
        generation: ResidentQuantifierGeneration,
        execution: ResidentExecutionId,
        slot_count: usize,
        max_rows: usize,
        max_list_items: usize,
    ) -> Result<()> {
        if let Self::Range {
            request,
            output,
            obligation,
        } = self
        {
            request.validate_streaming()?;
            if usize::from(output.0) >= slot_count || request.max_values != max_rows {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident quantifier range source has an invalid output slot or row bound",
                ));
            }
            if obligation.id == 0
                || obligation.kind != ResidentObligationKind::Expression
                || obligation.scope != ResidentObligationScope::Expression(u16::MAX - 2)
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident quantifier range source has an invalid obligation",
                ));
            }
            return Ok(());
        }
        let Self::VariablePathEntityList {
            path,
            output,
            skip,
            properties,
            materialize_obligation,
            ..
        } = self
        else {
            return Ok(());
        };
        path.validate()?;
        if path.project != generation.project
            || path.expected_bookmark != generation.bookmark
            || path.expected_graph_revision != generation.graph_revision
            || path.expected_layout_version != generation.layout_version
            || path.execution != execution
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier entity source targets a different graph generation or execution",
            ));
        }
        if usize::from(output.0) >= slot_count || *skip != 1 {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier entity source has an invalid output slot or tail skip",
            ));
        }
        if path.optional || path.bound_terminal_scan.is_some() {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier entity source requires mandatory unbound path rows",
            ));
        }
        if path.maximum_output_rows > max_rows || self.maximum_entity_list_items()? > max_list_items
        {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident quantifier entity source exceeds its row or list capacity",
            ));
        }
        if properties.len() > RESIDENT_ROW_PROGRAM_MAX_PROJECTIONS
            || properties.iter().any(|property| property.key.is_empty())
            || properties.iter().any(|property| {
                property.property.is_none()
                    && property.shape != ResidentQuantifierEntityPropertyShape::Absent
            })
            || properties.windows(2).any(|pair| pair[0].key >= pair[1].key)
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier entity properties must be bounded, sorted, and unique",
            ));
        }
        let mut property_ids = BTreeSet::new();
        if properties
            .iter()
            .filter_map(|property| property.property)
            .any(|property| !property_ids.insert(property))
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier entity properties contain a duplicate canonical property",
            ));
        }
        if materialize_obligation.id == 0
            || materialize_obligation.kind != ResidentObligationKind::Expression
            || materialize_obligation.scope != ResidentObligationScope::Expression(u16::MAX)
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier entity-list materialization has an invalid obligation",
            ));
        }
        let property_reads = properties
            .iter()
            .filter(|property| property.property.is_some())
            .count();
        let entity_property_domain = match self {
            Self::VariablePathEntityList { entity_kind, .. } => match entity_kind {
                ResidentQuantifierEntityKind::Node => path.expected_node_slots,
                ResidentQuantifierEntityKind::Relationship => path.expected_edge_slots,
            },
            Self::Unit | Self::Range { .. } => 0,
        };
        let required_dependencies = path
            .expected_node_slots
            .checked_add(path.expected_edge_slots)
            .and_then(|count| {
                entity_property_domain
                    .checked_mul(property_reads)
                    .and_then(|properties| count.checked_add(properties))
            })
            .ok_or_else(scratch_overflow)?;
        if required_dependencies > RESIDENT_QUANTIFIER_MAX_ENTITY_READ_DEPENDENCIES {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier entity read-dependency capacity exceeds its bounded ABI",
            ));
        }
        Ok(())
    }

    pub fn maximum_read_dependencies(&self) -> Result<usize> {
        let Self::VariablePathEntityList {
            path,
            entity_kind,
            properties,
            ..
        } = self
        else {
            return Ok(0);
        };
        let property_reads = properties
            .iter()
            .filter(|property| property.property.is_some())
            .count();
        let entity_property_domain = match entity_kind {
            ResidentQuantifierEntityKind::Node => path.expected_node_slots,
            ResidentQuantifierEntityKind::Relationship => path.expected_edge_slots,
        };
        path.expected_node_slots
            .checked_add(path.expected_edge_slots)
            .and_then(|count| {
                entity_property_domain
                    .checked_mul(property_reads)
                    .and_then(|properties| count.checked_add(properties))
            })
            .ok_or_else(scratch_overflow)
    }
}

const fn quantifier_entity_dependency_obligation() -> ResidentExecutionObligation {
    ResidentExecutionObligation {
        id: 0x5155_414e_454e_5444,
        kind: ResidentObligationKind::Expression,
        scope: ResidentObligationScope::Expression(u16::MAX - 1),
    }
}

/// Backend-neutral scalar/document value image used by the multistage quantifier command. FLOAT
/// values retain their exact IEEE-754 bits. Generation-fenced graph handles use the separate
/// `ResidentQuantifierEntityHandle` ABI so an opaque dense row can never be confused with Cypher
/// INTEGER content.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidentQuantifierValue {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(u64),
    String(String),
    List(Vec<Self>),
    Map(Vec<(String, Self)>),
}

impl ResidentQuantifierValue {
    pub fn structural_hash(&self) -> Result<[u8; 32]> {
        let mut hasher = blake3::Hasher::new_derive_key("irongraph.resident-quantifier-value.v1");
        hash_quantifier_value(&mut hasher, self)?;
        Ok(*hasher.finalize().as_bytes())
    }

    pub fn validate(&self, depth: usize, items: &mut usize) -> Result<()> {
        if depth > RESIDENT_QUANTIFIER_MAX_EXPRESSION_DEPTH {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier value exceeds the bounded nesting depth",
            ));
        }
        *items = items.checked_add(1).ok_or_else(scratch_overflow)?;
        if *items > RESIDENT_QUANTIFIER_MAX_LITERAL_ITEMS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier literal image exceeds its bounded item count",
            ));
        }
        match self {
            Self::List(values) => {
                for value in values {
                    value.validate(depth + 1, items)?;
                }
            }
            Self::Map(entries) => {
                if entries.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident quantifier map keys must be sorted and unique",
                    ));
                }
                for (_, value) in entries {
                    value.validate(depth + 1, items)?;
                }
            }
            Self::Null | Self::Boolean(_) | Self::Integer(_) | Self::Float(_) | Self::String(_) => {
            }
        }
        Ok(())
    }

    fn validate_segmented_list_capacity(&self, maximum_list_items: usize) -> Result<()> {
        match self {
            Self::List(values) => {
                if values.len() > maximum_list_items {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident segmented literal list exceeds its declared capacity",
                    ));
                }
                for value in values {
                    value.validate_segmented_list_capacity(maximum_list_items)?;
                }
            }
            Self::Map(entries) => {
                for (_, value) in entries {
                    value.validate_segmented_list_capacity(maximum_list_items)?;
                }
            }
            Self::Null | Self::Boolean(_) | Self::Integer(_) | Self::Float(_) | Self::String(_) => {
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentQuantifierKind {
    All = 1,
    Any = 2,
    None = 3,
    Single = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentQuantifierUnary {
    Not = 1,
    Positive = 2,
    Negative = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentQuantifierBinary {
    Or = 1,
    Xor = 2,
    And = 3,
    Equal = 4,
    NotEqual = 5,
    Less = 6,
    LessOrEqual = 7,
    Greater = 8,
    GreaterOrEqual = 9,
    Add = 10,
    Subtract = 11,
    Multiply = 12,
    Divide = 13,
    Modulo = 14,
    StartsWith = 15,
    EndsWith = 16,
    Contains = 17,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentQuantifierFunction {
    Rand = 1,
    Reverse = 2,
    Size = 3,
    Coalesce = 4,
    Split = 5,
    ToString = 6,
    Substring = 7,
    Head = 8,
    Tail = 9,
}

/// Lowered expression tree for one command. The compiler has already resolved names, parameters,
/// and aggregate placement; the selected backend owns every runtime branch, list traversal,
/// random comparison, and three-valued reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentQuantifierExpression {
    Slot(ResidentQuantifierSlot),
    Literal(ResidentQuantifierValue),
    Property {
        source: Box<Self>,
        key: String,
    },
    List(Vec<Self>),
    Map(Vec<(String, Self)>),
    Case {
        operand: Option<Box<Self>>,
        alternatives: Vec<(Self, Self)>,
        default: Option<Box<Self>>,
    },
    ListComprehension {
        variable: ResidentQuantifierSlot,
        list: Box<Self>,
        predicate: Option<Box<Self>>,
        projection: Option<Box<Self>>,
    },
    Predicate {
        kind: ResidentQuantifierKind,
        variable: ResidentQuantifierSlot,
        list: Box<Self>,
        predicate: Box<Self>,
    },
    Function {
        function: ResidentQuantifierFunction,
        arguments: Vec<Self>,
    },
    Unary {
        operation: ResidentQuantifierUnary,
        operand: Box<Self>,
    },
    Binary {
        left: Box<Self>,
        operation: ResidentQuantifierBinary,
        right: Box<Self>,
    },
    IsNull {
        expression: Box<Self>,
        negated: bool,
    },
}

impl ResidentQuantifierExpression {
    fn validate(
        &self,
        live: &BTreeSet<ResidentQuantifierSlot>,
        locals: &BTreeSet<ResidentQuantifierSlot>,
        slot_count: usize,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<()> {
        if depth > RESIDENT_QUANTIFIER_MAX_EXPRESSION_DEPTH {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier expression exceeds its bounded depth",
            ));
        }
        *nodes = nodes.checked_add(1).ok_or_else(scratch_overflow)?;
        if *nodes > RESIDENT_QUANTIFIER_MAX_EXPRESSION_NODES {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier expression program exceeds its node budget",
            ));
        }
        let recurse = |expression: &Self,
                       locals: &BTreeSet<ResidentQuantifierSlot>,
                       nodes: &mut usize|
         -> Result<()> {
            expression.validate(live, locals, slot_count, depth + 1, nodes)
        };
        match self {
            Self::Slot(slot) => {
                if usize::from(slot.0) >= slot_count
                    || (!live.contains(slot) && !locals.contains(slot))
                {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident quantifier expression reads an out-of-scope slot",
                    ));
                }
            }
            Self::Literal(value) => {
                let mut literal_items = 0;
                value.validate(0, &mut literal_items)?;
            }
            Self::Property { source, .. }
            | Self::Unary {
                operand: source, ..
            }
            | Self::IsNull {
                expression: source, ..
            } => recurse(source, locals, nodes)?,
            Self::List(values) => {
                if values.len() > RESIDENT_QUANTIFIER_MAX_LITERAL_ITEMS {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident quantifier list expression exceeds its item budget",
                    ));
                }
                for value in values {
                    recurse(value, locals, nodes)?;
                }
            }
            Self::Map(entries) => {
                if entries.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident quantifier map expression keys must be sorted and unique",
                    ));
                }
                for (_, value) in entries {
                    recurse(value, locals, nodes)?;
                }
            }
            Self::Case {
                operand,
                alternatives,
                default,
            } => {
                if let Some(operand) = operand {
                    recurse(operand, locals, nodes)?;
                }
                for (when, then) in alternatives {
                    recurse(when, locals, nodes)?;
                    recurse(then, locals, nodes)?;
                }
                if let Some(default) = default {
                    recurse(default, locals, nodes)?;
                }
            }
            Self::ListComprehension {
                variable,
                list,
                predicate,
                projection,
            } => {
                if usize::from(variable.0) >= slot_count || live.contains(variable) {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident quantifier iterator slot is invalid or shadows a relation slot",
                    ));
                }
                recurse(list, locals, nodes)?;
                let mut nested = locals.clone();
                nested.insert(*variable);
                if let Some(predicate) = predicate {
                    recurse(predicate, &nested, nodes)?;
                }
                if let Some(projection) = projection {
                    recurse(projection, &nested, nodes)?;
                }
            }
            Self::Predicate {
                variable,
                list,
                predicate,
                ..
            } => {
                if usize::from(variable.0) >= slot_count || live.contains(variable) {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident quantifier iterator slot is invalid or shadows a relation slot",
                    ));
                }
                recurse(list, locals, nodes)?;
                let mut nested = locals.clone();
                nested.insert(*variable);
                recurse(predicate, &nested, nodes)?;
            }
            Self::Function {
                function,
                arguments,
            } => {
                let valid_arity = match function {
                    ResidentQuantifierFunction::Rand => arguments.is_empty(),
                    ResidentQuantifierFunction::Reverse
                    | ResidentQuantifierFunction::Size
                    | ResidentQuantifierFunction::ToString
                    | ResidentQuantifierFunction::Head
                    | ResidentQuantifierFunction::Tail => arguments.len() == 1,
                    ResidentQuantifierFunction::Coalesce => !arguments.is_empty(),
                    ResidentQuantifierFunction::Split => arguments.len() == 2,
                    ResidentQuantifierFunction::Substring => (2..=3).contains(&arguments.len()),
                };
                if !valid_arity {
                    return Err(Error::new(
                        ErrorCode::QueryType,
                        "resident quantifier function has an invalid arity",
                    ));
                }
                for argument in arguments {
                    recurse(argument, locals, nodes)?;
                }
            }
            Self::Binary { left, right, .. } => {
                recurse(left, locals, nodes)?;
                recurse(right, locals, nodes)?;
            }
        }
        Ok(())
    }

    fn validate_segmented(
        &self,
        live: &BTreeSet<ResidentQuantifierSlot>,
        locals: &BTreeSet<ResidentQuantifierSlot>,
        slot_count: usize,
        depth: usize,
        nodes: &mut usize,
        maximum_list_items: usize,
    ) -> Result<()> {
        self.validate(live, locals, slot_count, depth, nodes)?;
        self.validate_segmented_list_capacity(maximum_list_items)
    }

    fn validate_segmented_list_capacity(&self, maximum_list_items: usize) -> Result<()> {
        let recurse =
            |expression: &Self| expression.validate_segmented_list_capacity(maximum_list_items);
        match self {
            Self::Slot(_) => {}
            Self::Literal(value) => {
                value.validate_segmented_list_capacity(maximum_list_items)?;
            }
            Self::Property { source, .. }
            | Self::Unary {
                operand: source, ..
            }
            | Self::IsNull {
                expression: source, ..
            } => recurse(source)?,
            Self::List(values) => {
                if values.len() > maximum_list_items {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident segmented list expression exceeds its declared capacity",
                    ));
                }
                for value in values {
                    recurse(value)?;
                }
            }
            Self::Map(entries) => {
                for (_, value) in entries {
                    recurse(value)?;
                }
            }
            Self::Case {
                operand,
                alternatives,
                default,
            } => {
                if let Some(operand) = operand {
                    recurse(operand)?;
                }
                for (when, then) in alternatives {
                    recurse(when)?;
                    recurse(then)?;
                }
                if let Some(default) = default {
                    recurse(default)?;
                }
            }
            Self::ListComprehension {
                list,
                predicate,
                projection,
                ..
            } => {
                recurse(list)?;
                if let Some(predicate) = predicate {
                    recurse(predicate)?;
                }
                if let Some(projection) = projection {
                    recurse(projection)?;
                }
            }
            Self::Predicate {
                list, predicate, ..
            } => {
                recurse(list)?;
                recurse(predicate)?;
            }
            Self::Function { arguments, .. } => {
                for argument in arguments {
                    recurse(argument)?;
                }
            }
            Self::Binary { left, right, .. } => {
                recurse(left)?;
                recurse(right)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierProjection {
    pub output: ResidentQuantifierSlot,
    pub expression: ResidentQuantifierExpression,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentQuantifierStage {
    Project {
        keep_scope: bool,
        bindings: Vec<ResidentQuantifierProjection>,
    },
    Unwind {
        expression: ResidentQuantifierExpression,
        output: ResidentQuantifierSlot,
    },
    Filter {
        predicate: ResidentQuantifierExpression,
    },
    GroupCount {
        groups: Vec<ResidentQuantifierProjection>,
        count_outputs: Vec<ResidentQuantifierSlot>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierOutput {
    pub name: String,
    pub source: ResidentQuantifierSlot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierProgram {
    pub slot_count: u16,
    pub stages: Vec<ResidentQuantifierStage>,
    pub outputs: Vec<ResidentQuantifierOutput>,
}

impl ResidentQuantifierProgram {
    pub fn validate(&self) -> Result<()> {
        self.validate_with_initial_slots(&BTreeSet::new())
    }

    fn validate_with_initial_slots(
        &self,
        initial_slots: &BTreeSet<ResidentQuantifierSlot>,
    ) -> Result<()> {
        let slot_count = usize::from(self.slot_count);
        if slot_count == 0 || slot_count > RESIDENT_QUANTIFIER_MAX_SLOTS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier slot count is outside its bounded contract",
            ));
        }
        if self.stages.is_empty() || self.stages.len() > RESIDENT_QUANTIFIER_MAX_STAGES {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier stage count is outside its bounded contract",
            ));
        }
        if self.outputs.is_empty() || self.outputs.len() > RESIDENT_ROW_PROGRAM_MAX_PROJECTIONS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier output count is outside its bounded contract",
            ));
        }

        if initial_slots
            .iter()
            .any(|slot| usize::from(slot.0) >= slot_count)
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier source seeds an out-of-range slot",
            ));
        }
        let mut live = initial_slots.clone();
        let locals = BTreeSet::new();
        let mut expression_nodes = 0_usize;
        for stage in &self.stages {
            match stage {
                ResidentQuantifierStage::Project {
                    keep_scope,
                    bindings,
                } => {
                    if bindings.is_empty() && *keep_scope {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident quantifier scope-preserving projection stage is empty",
                        ));
                    }
                    let mut outputs = BTreeSet::new();
                    for binding in bindings {
                        if usize::from(binding.output.0) >= slot_count
                            || !outputs.insert(binding.output)
                        {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident quantifier projection has an invalid duplicate output slot",
                            ));
                        }
                        binding.expression.validate(
                            &live,
                            &locals,
                            slot_count,
                            0,
                            &mut expression_nodes,
                        )?;
                    }
                    if !*keep_scope {
                        live.clear();
                    }
                    live.extend(outputs);
                }
                ResidentQuantifierStage::Unwind { expression, output } => {
                    expression.validate(&live, &locals, slot_count, 0, &mut expression_nodes)?;
                    if usize::from(output.0) >= slot_count || live.contains(output) {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident quantifier UNWIND output slot is invalid or already live",
                        ));
                    }
                    live.insert(*output);
                }
                ResidentQuantifierStage::Filter { predicate } => {
                    predicate.validate(&live, &locals, slot_count, 0, &mut expression_nodes)?
                }
                ResidentQuantifierStage::GroupCount {
                    groups,
                    count_outputs,
                } => {
                    if count_outputs.is_empty() {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident quantifier grouping stage omits count(*)",
                        ));
                    }
                    let mut outputs = BTreeSet::new();
                    for group in groups {
                        group.expression.validate(
                            &live,
                            &locals,
                            slot_count,
                            0,
                            &mut expression_nodes,
                        )?;
                        if usize::from(group.output.0) >= slot_count
                            || !outputs.insert(group.output)
                        {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident quantifier grouping has an invalid output slot",
                            ));
                        }
                    }
                    for output in count_outputs {
                        if usize::from(output.0) >= slot_count || !outputs.insert(*output) {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident quantifier count(*) has an invalid output slot",
                            ));
                        }
                    }
                    live = outputs;
                }
            }
        }
        let mut names = BTreeSet::new();
        for output in &self.outputs {
            if output.name.is_empty()
                || !names.insert(output.name.as_str())
                || !live.contains(&output.source)
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident quantifier final projection has an invalid name or source slot",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod quantifier_program_tests {
    use super::{
        ResidentQuantifierExpression, ResidentQuantifierFunction, ResidentQuantifierOutput,
        ResidentQuantifierProgram, ResidentQuantifierProjection, ResidentQuantifierSlot,
        ResidentQuantifierStage, ResidentQuantifierValue,
    };

    fn program_with_first_stage(first: ResidentQuantifierStage) -> ResidentQuantifierProgram {
        ResidentQuantifierProgram {
            slot_count: 1,
            stages: vec![
                first,
                ResidentQuantifierStage::Project {
                    keep_scope: false,
                    bindings: vec![ResidentQuantifierProjection {
                        output: ResidentQuantifierSlot(0),
                        expression: ResidentQuantifierExpression::Literal(
                            ResidentQuantifierValue::Integer(7),
                        ),
                    }],
                },
            ],
            outputs: vec![ResidentQuantifierOutput {
                name: "value".to_owned(),
                source: ResidentQuantifierSlot(0),
            }],
        }
    }

    #[test]
    fn empty_project_is_valid_only_when_it_clears_scope() {
        assert!(
            program_with_first_stage(ResidentQuantifierStage::Project {
                keep_scope: false,
                bindings: Vec::new(),
            })
            .validate()
            .is_ok()
        );
        assert!(
            program_with_first_stage(ResidentQuantifierStage::Project {
                keep_scope: true,
                bindings: Vec::new(),
            })
            .validate()
            .is_err()
        );
    }

    #[test]
    fn map_expression_keys_are_canonical_or_fail_request_validation() {
        let project = |entries| ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings: vec![ResidentQuantifierProjection {
                output: ResidentQuantifierSlot(0),
                expression: ResidentQuantifierExpression::Map(entries),
            }],
        };
        let value = || ResidentQuantifierExpression::Literal(ResidentQuantifierValue::Integer(1));

        assert!(
            program_with_first_stage(project(vec![
                ("a".to_owned(), value()),
                ("z".to_owned(), value()),
            ]))
            .validate()
            .is_ok()
        );
        for entries in [
            vec![("z".to_owned(), value()), ("a".to_owned(), value())],
            vec![("a".to_owned(), value()), ("a".to_owned(), value())],
        ] {
            assert!(
                program_with_first_stage(project(entries))
                    .validate()
                    .is_err()
            );
        }
    }

    fn to_string_program(
        arguments: Vec<ResidentQuantifierExpression>,
    ) -> ResidentQuantifierProgram {
        ResidentQuantifierProgram {
            slot_count: 1,
            stages: vec![ResidentQuantifierStage::Project {
                keep_scope: false,
                bindings: vec![ResidentQuantifierProjection {
                    output: ResidentQuantifierSlot(0),
                    expression: ResidentQuantifierExpression::Function {
                        function: ResidentQuantifierFunction::ToString,
                        arguments,
                    },
                }],
            }],
            outputs: vec![ResidentQuantifierOutput {
                name: "value".to_owned(),
                source: ResidentQuantifierSlot(0),
            }],
        }
    }

    #[test]
    fn to_string_requires_exactly_one_argument() {
        let literal = ResidentQuantifierExpression::Literal(ResidentQuantifierValue::String(
            "text".to_owned(),
        ));
        assert!(to_string_program(vec![literal.clone()]).validate().is_ok());

        for arguments in [Vec::new(), vec![literal.clone(), literal]] {
            let error = to_string_program(arguments)
                .validate()
                .expect_err("toString arity must be sealed before execution");
            assert_eq!(error.code, crate::ErrorCode::QueryType);
        }
    }
}

/// One reduction in the sealed segmented-aggregation program. Unlike the legacy column-oriented
/// descriptor, `input` is evaluated by the selected backend against the current staged relation.
/// `None` is valid only for `count(*)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentSegmentedAggregationReduction {
    pub output: ResidentQuantifierSlot,
    pub kind: super::ResidentSegmentedAggregateKind,
    pub input: Option<ResidentQuantifierExpression>,
    pub distinct: bool,
    /// Raw IEEE-754 bits for the immutable percentile argument. Present exactly for the two
    /// percentile kinds; preserving bits keeps signed zero and makes the sealed request `Eq`.
    pub percentile: Option<u64>,
    pub obligation: ResidentExecutionObligation,
}

/// One stable post-aggregate ordering key. The first tranche does not execute ordering yet, but
/// the key belongs to the same sealed stage language so later exact-TCK additions do not need a
/// second command boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentSegmentedAggregationOrderKey {
    pub expression: ResidentQuantifierExpression,
    pub descending: bool,
    pub nulls_first: bool,
}

/// One immutable project-local property name considered by a native `keys(entity)` transform.
/// Descriptors are sealed in stable property-ID order; the selected backend tests canonical row
/// presence and emits `name` only when that exact property is present on the source entity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentSegmentedPropertyKeyDescriptor {
    pub property: PropertyId,
    pub name: String,
}

/// Backend-neutral stages around exactly one aggregation boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentSegmentedAggregationOperation {
    Project {
        keep_scope: bool,
        bindings: Vec<ResidentQuantifierProjection>,
    },
    PropertyKeys {
        input: ResidentQuantifierSlot,
        kind: ResidentNullableRelationBindingKind,
        descriptors: Vec<ResidentSegmentedPropertyKeyDescriptor>,
        output: ResidentQuantifierSlot,
    },
    /// Reads one generation-fenced canonical property from an entity slot without routing the
    /// value through the scalar-only nullable-relation columns. The bounded document metrics are
    /// derived from every live value in the sealed graph generation and rechecked by the selected
    /// backend before the value may participate in grouping or DISTINCT.
    PropertyValue {
        input: ResidentQuantifierSlot,
        kind: ResidentNullableRelationBindingKind,
        property: Option<PropertyId>,
        output: ResidentQuantifierSlot,
        maximum_value_items: u32,
        maximum_list_items: u32,
        maximum_string_bytes: u32,
    },
    Unwind {
        expression: ResidentQuantifierExpression,
        output: ResidentQuantifierSlot,
    },
    Filter {
        predicate: ResidentQuantifierExpression,
    },
    Aggregate {
        groups: Vec<ResidentQuantifierProjection>,
        reductions: Vec<ResidentSegmentedAggregationReduction>,
    },
    Order {
        keys: Vec<ResidentSegmentedAggregationOrderKey>,
    },
    Skip {
        rows: u32,
    },
    Limit {
        rows: u32,
    },
}

/// One receipted stage. Keeping the obligation adjacent to the operation makes stage removal,
/// reordering, or host-side execution observable at result publication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentSegmentedAggregationStage {
    pub operation: ResidentSegmentedAggregationOperation,
    pub obligation: ResidentExecutionObligation,
}

/// One typed column exported by a complete fixed-pattern relation into the segmented stage slot
/// space. `relation_output` addresses the nested request's immutable final projection. The
/// nested request fingerprint seals the physical source type; this binding seals only the
/// composition edge and cannot reinterpret the column on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentSegmentedGraphRelationBinding {
    pub relation_output: u16,
    pub output: ResidentQuantifierSlot,
}

/// Device-owned source for the sealed segmented program. No host-row variant is intentionally
/// provided: graph-backed sources remain complete generation-fenced native commands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentSegmentedUnitPrelude {
    Empty,
    /// Executes one raw temporal SSA program on the selected backend and exposes its explicitly
    /// selected homogeneous outputs as one typed list in the Unit source row. The destination is
    /// an ordinary segmented slot so the first receipted stage can reproject it without a host
    /// materialization boundary.
    RawTemporalList {
        program: super::ResidentTemporalValueProgramRequest,
        output: ResidentQuantifierSlot,
    },
}

impl ResidentSegmentedUnitPrelude {
    fn validate(&self, slot_count: usize, maximum_list_items: usize) -> Result<()> {
        let Self::RawTemporalList { program, output } = self else {
            return Ok(());
        };
        program.validate()?;
        if usize::from(output.0) >= slot_count
            || program.output_registers.is_empty()
            || program.output_registers.len() > maximum_list_items
            || program
                .output_registers
                .windows(2)
                .any(|registers| registers[0] >= registers[1])
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident segmented temporal Unit prelude has invalid output slots or registers",
            ));
        }
        let mut family = None;
        for kind in program.output_kinds()? {
            let super::ResidentTemporalRegisterKind::Temporal { function, .. } = kind else {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident segmented temporal Unit prelude selects a non-temporal register",
                ));
            };
            if function == super::ResidentTemporalValueFunction::Duration
                || family.is_some_and(|expected| expected != function)
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident segmented temporal Unit prelude requires one ordered temporal family",
                ));
            }
            family = Some(function);
        }
        Ok(())
    }

    fn initial_slots(&self) -> BTreeSet<ResidentQuantifierSlot> {
        match self {
            Self::Empty => BTreeSet::new(),
            Self::RawTemporalList { output, .. } => BTreeSet::from([*output]),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentSegmentedAggregationSource {
    Unit {
        prelude: ResidentSegmentedUnitPrelude,
        obligation: ResidentExecutionObligation,
    },
    Range {
        request: super::ResidentRangeProgramRequest,
        output: ResidentQuantifierSlot,
        obligation: ResidentExecutionObligation,
    },
    /// Executes one complete nullable fixed-pattern relation and exposes one projected nullable
    /// relationship as a sealed expression slot. `relationship_output` addresses the immutable
    /// nullable command's final output packet; the dense relationship row itself never becomes a
    /// host-shaped segmented input.
    NullableRelationship {
        request: Box<ResidentNullableRelationRequest>,
        relationship_output: u16,
        output: ResidentQuantifierSlot,
        obligation: ResidentExecutionObligation,
    },
    /// Executes one complete generation-fenced fixed-pattern relation and exposes a bounded set
    /// of its typed final columns as segmented expression slots. Documents and relationship-type
    /// tokens are deliberately absent from this first fused graph-source tranche.
    GraphRelation {
        request: Box<ResidentNullableRelationRequest>,
        bindings: Vec<ResidentSegmentedGraphRelationBinding>,
        obligation: ResidentExecutionObligation,
    },
}

impl ResidentSegmentedAggregationSource {
    pub fn obligation(&self) -> ResidentExecutionObligation {
        match self {
            Self::Unit { obligation, .. }
            | Self::Range { obligation, .. }
            | Self::NullableRelationship { obligation, .. }
            | Self::GraphRelation { obligation, .. } => *obligation,
        }
    }

    fn initial_slots(&self) -> BTreeSet<ResidentQuantifierSlot> {
        match self {
            Self::Unit { prelude, .. } => prelude.initial_slots(),
            Self::Range { output, .. } | Self::NullableRelationship { output, .. } => {
                BTreeSet::from([*output])
            }
            Self::GraphRelation { bindings, .. } => {
                bindings.iter().map(|binding| binding.output).collect()
            }
        }
    }

    fn entity_kind(
        &self,
        slot: ResidentQuantifierSlot,
    ) -> Option<ResidentNullableRelationBindingKind> {
        match self {
            Self::NullableRelationship { output, .. } if *output == slot => {
                Some(ResidentNullableRelationBindingKind::Relationship)
            }
            Self::GraphRelation {
                request, bindings, ..
            } => {
                let relation_output = bindings
                    .iter()
                    .find(|binding| binding.output == slot)?
                    .relation_output;
                let ResidentNullableRelationStage::FinalProject { bindings: outputs } =
                    request.program.stages.last()?
                else {
                    return None;
                };
                match &outputs.get(usize::from(relation_output))?.source {
                    ResidentNullableRelationOutputSource::Entity { kind, .. } => Some(*kind),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Exact source cardinality when it is structurally known before execution. Graph-backed
    /// nullable relations author their exact cardinality on the selected backend instead.
    pub fn logical_row_count(&self) -> Result<Option<usize>> {
        match self {
            Self::Unit { .. } => Ok(Some(1)),
            Self::Range { request, .. } => request.value_count().map(Some),
            Self::NullableRelationship { .. } => Ok(None),
            Self::GraphRelation { .. } => Ok(None),
        }
    }

    pub fn maximum_row_count(&self) -> usize {
        match self {
            Self::Unit { .. } => 1,
            Self::Range { request, .. } => request.max_values,
            Self::NullableRelationship { request, .. } => {
                request.capacities.max_output_rows as usize
            }
            Self::GraphRelation { request, .. } => request.capacities.max_output_rows as usize,
        }
    }
}

/// A complete source → pre-stage → aggregate → post-stage recipe. It is carried alongside the
/// legacy relation request during the migration, but the two modes are mutually exclusive: a
/// request with this program must carry a canonical empty legacy relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentSegmentedAggregationProgram {
    pub catalog_generation: [u8; 32],
    pub slot_count: u16,
    pub source: ResidentSegmentedAggregationSource,
    pub stages: Vec<ResidentSegmentedAggregationStage>,
    pub outputs: Vec<ResidentQuantifierOutput>,
    /// Hard bound for the rows entering the aggregate after all pre-aggregate stages. This is
    /// the query's configured intermediate row budget, not merely the final one-row result size.
    pub maximum_intermediate_rows: u32,
    pub maximum_list_items: u32,
    pub random_seed: u64,
}

/// Borrowed, fully validated view of the first sealed segmented-program tranche. Keeping this
/// shape check beside the program definition prevents native backends from drifting on which
/// source/stage/reduction recipe the optimized range SUM path is allowed to execute.
pub struct ResidentSegmentedRangeSumProgram<'a> {
    pub range: &'a super::ResidentRangeProgramRequest,
    pub pre_aggregate_stages: &'a [ResidentSegmentedAggregationStage],
    pub aggregate_stage: &'a ResidentSegmentedAggregationStage,
    pub reduction: &'a ResidentSegmentedAggregationReduction,
}

/// Borrowed, fully validated view of
/// `NullableRelationship -> GROUP BY relationship IS NULL -> count(*)`.
pub struct ResidentSegmentedNullableIsNullCountProgram<'a> {
    pub request: &'a ResidentNullableRelationRequest,
    pub relationship_output: u16,
    // Retained for the encoded program's shape; read by the device program, not host code.
    #[allow(dead_code)]
    pub source_slot: ResidentQuantifierSlot,
    pub aggregate_stage: &'a ResidentSegmentedAggregationStage,
    pub group: &'a ResidentQuantifierProjection,
    pub reduction: &'a ResidentSegmentedAggregationReduction,
}

impl ResidentSegmentedAggregationProgram {
    pub fn validate(&self) -> Result<()> {
        let slot_count = usize::from(self.slot_count);
        if slot_count == 0 || slot_count > RESIDENT_QUANTIFIER_MAX_SLOTS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident segmented program slot count is outside its bounded contract",
            ));
        }
        if self.stages.is_empty() || self.stages.len() > RESIDENT_QUANTIFIER_MAX_STAGES {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident segmented program stage count is outside its bounded contract",
            ));
        }
        if self.outputs.is_empty() || self.outputs.len() > RESIDENT_ROW_PROGRAM_MAX_PROJECTIONS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident segmented program output count is outside its bounded contract",
            ));
        }
        if self.maximum_list_items as usize > RESIDENT_QUANTIFIER_MAX_LITERAL_ITEMS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident segmented program list capacity exceeds its bounded contract",
            ));
        }
        match &self.source {
            ResidentSegmentedAggregationSource::Unit { prelude, .. } => {
                prelude.validate(slot_count, self.maximum_list_items as usize)?;
            }
            ResidentSegmentedAggregationSource::Range {
                request, output, ..
            } => {
                request.validate_streaming()?;
                if usize::from(output.0) >= slot_count {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident segmented range source writes an out-of-bounds slot",
                    ));
                }
            }
            ResidentSegmentedAggregationSource::NullableRelationship {
                request,
                relationship_output,
                output,
                ..
            } => {
                request.validate()?;
                if usize::from(output.0) >= slot_count {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident segmented nullable source writes an out-of-bounds slot",
                    ));
                }
                let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
                    request.program.stages.last()
                else {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident segmented nullable source omits its final projection",
                    ));
                };
                let source = bindings
                    .get(usize::from(*relationship_output))
                    .map(|binding| &binding.source);
                if !matches!(
                    source,
                    Some(&ResidentNullableRelationOutputSource::Entity {
                        kind: ResidentNullableRelationBindingKind::Relationship,
                        ..
                    })
                ) {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident segmented nullable source must address one relationship entity output",
                    ));
                }
            }
            ResidentSegmentedAggregationSource::GraphRelation {
                request, bindings, ..
            } => {
                request.validate()?;
                if bindings.is_empty() || bindings.len() > RESIDENT_ROW_PROGRAM_MAX_PROJECTIONS {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident segmented graph source has an invalid output binding count",
                    ));
                }
                let Some(ResidentNullableRelationStage::FinalProject {
                    bindings: relation_outputs,
                }) = request.program.stages.last()
                else {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident segmented graph source omits its final projection",
                    ));
                };
                let mut relation_output_indices = BTreeSet::new();
                let mut output_slots = BTreeSet::new();
                for binding in bindings {
                    if usize::from(binding.output.0) >= slot_count
                        || !output_slots.insert(binding.output)
                        || !relation_output_indices.insert(binding.relation_output)
                    {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident segmented graph source has a duplicate or out-of-bounds binding",
                        ));
                    }
                    let source = relation_outputs
                        .get(usize::from(binding.relation_output))
                        .map(|output| &output.source);
                    if !matches!(
                        source,
                        Some(
                            &ResidentNullableRelationOutputSource::Entity { .. }
                                | ResidentNullableRelationOutputSource::IntegerProperty { .. }
                                | ResidentNullableRelationOutputSource::FloatProperty { .. }
                                | ResidentNullableRelationOutputSource::StringProperty { .. }
                                | ResidentNullableRelationOutputSource::NullProperty { .. }
                        )
                    ) {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident segmented graph source addresses an unsupported typed output",
                        ));
                    }
                }
            }
        }
        let source_obligation = self.source.obligation();
        if source_obligation.id == 0
            || source_obligation.kind != ResidentObligationKind::Expression
            || source_obligation.scope != ResidentObligationScope::Expression(u16::MAX - 2)
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident segmented source has an invalid execution obligation",
            ));
        }

        let mut live = self.source.initial_slots();
        let mut entity_kinds = live
            .iter()
            .filter_map(|slot| self.source.entity_kind(*slot).map(|kind| (*slot, kind)))
            .collect::<BTreeMap<_, _>>();
        let locals = BTreeSet::new();
        let mut expression_nodes = 0_usize;
        let mut aggregate_seen = false;
        let mut obligation_ids = BTreeSet::from([source_obligation.id]);
        for (stage_index, stage) in self.stages.iter().enumerate() {
            let stage_index = u16::try_from(stage_index).map_err(|_| scratch_overflow())?;
            if stage.obligation.id == 0 || !obligation_ids.insert(stage.obligation.id) {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident segmented program has a missing or duplicate stage obligation",
                ));
            }
            let expected_obligation = match &stage.operation {
                ResidentSegmentedAggregationOperation::Filter { .. } => (
                    ResidentObligationKind::Filter,
                    ResidentObligationScope::Filter(stage_index),
                ),
                ResidentSegmentedAggregationOperation::Aggregate { .. } => (
                    ResidentObligationKind::Aggregate,
                    ResidentObligationScope::PatternFinal,
                ),
                _ => (
                    ResidentObligationKind::Expression,
                    ResidentObligationScope::Expression(stage_index),
                ),
            };
            if (stage.obligation.kind, stage.obligation.scope) != expected_obligation {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident segmented stage obligation disagrees with its operation",
                ));
            }

            match &stage.operation {
                ResidentSegmentedAggregationOperation::Project {
                    keep_scope,
                    bindings,
                } => {
                    if bindings.is_empty() {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident segmented projection stage is empty",
                        ));
                    }
                    let mut projected = BTreeSet::new();
                    let mut projected_entity_kinds = Vec::with_capacity(bindings.len());
                    for binding in bindings {
                        if usize::from(binding.output.0) >= slot_count
                            || !projected.insert(binding.output)
                        {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident segmented projection has an invalid output slot",
                            ));
                        }
                        binding.expression.validate_segmented(
                            &live,
                            &locals,
                            slot_count,
                            0,
                            &mut expression_nodes,
                            self.maximum_list_items as usize,
                        )?;
                        projected_entity_kinds.push((
                            binding.output,
                            match &binding.expression {
                                ResidentQuantifierExpression::Slot(source) => {
                                    entity_kinds.get(source).copied()
                                }
                                _ => None,
                            },
                        ));
                    }
                    if !*keep_scope {
                        live.clear();
                        entity_kinds.clear();
                    }
                    for (output, kind) in projected_entity_kinds {
                        entity_kinds.remove(&output);
                        if let Some(kind) = kind {
                            entity_kinds.insert(output, kind);
                        }
                    }
                    live.extend(projected);
                }
                ResidentSegmentedAggregationOperation::PropertyKeys {
                    input,
                    kind,
                    descriptors,
                    output,
                } => {
                    if aggregate_seen
                        || !live.contains(input)
                        || entity_kinds.get(input).copied() != Some(*kind)
                        || usize::from(output.0) >= slot_count
                        || live.contains(output)
                        || descriptors.len() > self.maximum_list_items as usize
                        || descriptors.len() > RESIDENT_QUANTIFIER_MAX_LITERAL_ITEMS
                    {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident segmented property-keys transform has an invalid entity source or bounded output",
                        ));
                    }
                    let mut property_ids = BTreeSet::new();
                    let mut names = BTreeSet::new();
                    if descriptors.windows(2).any(|pair| {
                        (pair[0].property, pair[0].name.as_str())
                            >= (pair[1].property, pair[1].name.as_str())
                    }) || descriptors.iter().any(|descriptor| {
                        descriptor.name.is_empty()
                            || !property_ids.insert(descriptor.property)
                            || !names.insert(descriptor.name.as_str())
                    }) {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident segmented property-key descriptors are not sorted and unique",
                        ));
                    }
                    live.insert(*output);
                    entity_kinds.remove(output);
                }
                ResidentSegmentedAggregationOperation::PropertyValue {
                    input,
                    kind,
                    property,
                    output,
                    maximum_value_items,
                    maximum_list_items,
                    maximum_string_bytes: _,
                } => {
                    if !live.contains(input)
                        || entity_kinds.get(input).copied() != Some(*kind)
                        || usize::from(output.0) >= slot_count
                        || live.contains(output)
                        || *maximum_value_items == 0
                        || *maximum_value_items as usize > RESIDENT_QUANTIFIER_MAX_LITERAL_ITEMS
                        || *maximum_list_items > self.maximum_list_items
                        || (property.is_none()
                            && (*maximum_value_items != 1 || *maximum_list_items != 0))
                    {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident segmented property-value transform has an invalid entity source or bounded document shape",
                        ));
                    }
                    live.insert(*output);
                    entity_kinds.remove(output);
                }
                ResidentSegmentedAggregationOperation::Unwind { expression, output } => {
                    expression.validate_segmented(
                        &live,
                        &locals,
                        slot_count,
                        0,
                        &mut expression_nodes,
                        self.maximum_list_items as usize,
                    )?;
                    if usize::from(output.0) >= slot_count || live.contains(output) {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident segmented UNWIND has an invalid output slot",
                        ));
                    }
                    live.insert(*output);
                }
                ResidentSegmentedAggregationOperation::Filter { predicate } => {
                    predicate.validate_segmented(
                        &live,
                        &locals,
                        slot_count,
                        0,
                        &mut expression_nodes,
                        self.maximum_list_items as usize,
                    )?;
                }
                ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } => {
                    if reductions.is_empty() {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident segmented program requires non-empty aggregate stages",
                        ));
                    }
                    aggregate_seen = true;
                    let mut outputs = BTreeSet::new();
                    let mut output_entity_kinds = BTreeMap::new();
                    for group in groups {
                        group.expression.validate_segmented(
                            &live,
                            &locals,
                            slot_count,
                            0,
                            &mut expression_nodes,
                            self.maximum_list_items as usize,
                        )?;
                        if usize::from(group.output.0) >= slot_count
                            || !outputs.insert(group.output)
                        {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident segmented group has an invalid output slot",
                            ));
                        }
                        if let ResidentQuantifierExpression::Slot(source) = &group.expression
                            && let Some(kind) = entity_kinds.get(source).copied()
                        {
                            output_entity_kinds.insert(group.output, kind);
                        }
                    }
                    for (reduction_index, reduction) in reductions.iter().enumerate() {
                        let reduction_index =
                            u16::try_from(reduction_index).map_err(|_| scratch_overflow())?;
                        if usize::from(reduction.output.0) >= slot_count
                            || !outputs.insert(reduction.output)
                            || reduction.obligation.id == 0
                            || !obligation_ids.insert(reduction.obligation.id)
                            || reduction.obligation.kind != ResidentObligationKind::Aggregate
                            || reduction.obligation.scope
                                != ResidentObligationScope::Expression(reduction_index)
                        {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident segmented reduction has an invalid output or obligation",
                            ));
                        }
                        match (reduction.kind, reduction.input.as_ref(), reduction.distinct) {
                            (super::ResidentSegmentedAggregateKind::CountAll, None, false) => {}
                            (super::ResidentSegmentedAggregateKind::CountAll, _, _) => {
                                return Err(Error::new(
                                    ErrorCode::QueryType,
                                    "resident segmented count(*) cannot carry input or DISTINCT",
                                ));
                            }
                            (_, Some(input), _) => input.validate_segmented(
                                &live,
                                &locals,
                                slot_count,
                                0,
                                &mut expression_nodes,
                                self.maximum_list_items as usize,
                            )?,
                            _ => {
                                return Err(Error::new(
                                    ErrorCode::QueryType,
                                    "resident segmented value aggregate requires an input expression",
                                ));
                            }
                        }
                        super::resident_segmented_validate_percentile(
                            reduction.kind,
                            reduction.percentile,
                        )?;
                    }
                    live = outputs;
                    entity_kinds = output_entity_kinds;
                }
                ResidentSegmentedAggregationOperation::Order { keys } => {
                    if keys.is_empty() {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident segmented ordering requires a non-empty key list",
                        ));
                    }
                    for key in keys {
                        key.expression.validate_segmented(
                            &live,
                            &locals,
                            slot_count,
                            0,
                            &mut expression_nodes,
                            self.maximum_list_items as usize,
                        )?;
                    }
                }
                ResidentSegmentedAggregationOperation::Skip { .. }
                | ResidentSegmentedAggregationOperation::Limit { .. } => {}
            }
        }
        if !aggregate_seen {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident segmented program omits its aggregate stage",
            ));
        }
        let mut output_names = BTreeSet::new();
        for output in &self.outputs {
            if output.name.is_empty()
                || !output_names.insert(output.name.as_str())
                || !live.contains(&output.source)
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident segmented final output has an invalid name or source slot",
                ));
            }
        }
        Ok(())
    }

    pub fn aggregate(
        &self,
    ) -> Result<(
        &[ResidentQuantifierProjection],
        &[ResidentSegmentedAggregationReduction],
        ResidentExecutionObligation,
    )> {
        let mut aggregate = None;
        for stage in &self.stages {
            if let ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } =
                &stage.operation
            {
                // The transitional descriptor on the request mirrors only the final aggregate.
                // Earlier aggregate stages are fully sealed in this program and remain covered by
                // their own ordered stage/reduction receipts.
                aggregate = Some((groups.as_slice(), reductions.as_slice(), stage.obligation));
            }
        }
        aggregate.ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident segmented program omits its aggregate stage",
            )
        })
    }

    /// Validates and exposes exactly `Range -> LIMIT+ -> global SUM(range-slot)`. Both the CPU
    /// semantic reference and native accelerators use this owner before executing the specialized
    /// streaming reduction; all other sealed-program shapes remain explicit admission misses.
    pub fn range_limit_sum(&self) -> Result<ResidentSegmentedRangeSumProgram<'_>> {
        self.validate()?;
        let ResidentSegmentedAggregationSource::Range {
            request: range,
            output: source_slot,
            ..
        } = &self.source
        else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented range-SUM tranche requires a device-owned Range source",
            ));
        };
        let Some((aggregate_stage, pre_aggregate_stages)) = self.stages.split_last() else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented range-SUM tranche omitted its aggregate stage",
            ));
        };
        if pre_aggregate_stages.is_empty()
            || pre_aggregate_stages.iter().any(|stage| {
                !matches!(
                    stage.operation,
                    ResidentSegmentedAggregationOperation::Limit { .. }
                )
            })
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented range-SUM tranche admits only one or more pre-aggregate LIMIT stages",
            ));
        }
        let ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } =
            &aggregate_stage.operation
        else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented range-SUM tranche requires aggregation as its final stage",
            ));
        };
        let [reduction] = reductions.as_slice() else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented range-SUM tranche requires exactly one SUM reduction",
            ));
        };
        if !groups.is_empty()
            || reduction.kind != super::ResidentSegmentedAggregateKind::Sum
            || reduction.distinct
            || reduction.input != Some(ResidentQuantifierExpression::Slot(*source_slot))
            || self.outputs.len() != 1
            || self.outputs[0].source != reduction.output
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented range-SUM tranche admits only global SUM of the Range slot",
            ));
        }
        Ok(ResidentSegmentedRangeSumProgram {
            range,
            pre_aggregate_stages,
            aggregate_stage,
            reduction,
        })
    }

    /// Validates and exposes exactly
    /// `NullableRelationship -> GROUP BY relationship IS NULL -> count(*)`. The nullable request owns
    /// every graph stage and null-extension decision; this boundary only turns its projected
    /// relationship sentinel into a Boolean grouping key and counts rows in stable source order.
    pub fn nullable_relationship_is_null_count(
        &self,
    ) -> Result<ResidentSegmentedNullableIsNullCountProgram<'_>> {
        self.validate()?;
        let ResidentSegmentedAggregationSource::NullableRelationship {
            request,
            relationship_output,
            output: source_slot,
            ..
        } = &self.source
        else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented nullable-count tranche requires a nullable relationship source",
            ));
        };
        let [aggregate_stage] = self.stages.as_slice() else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented nullable-count tranche requires exactly one aggregate stage",
            ));
        };
        let ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } =
            &aggregate_stage.operation
        else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented nullable-count tranche requires aggregation as its only stage",
            ));
        };
        let ([group], [reduction]) = (groups.as_slice(), reductions.as_slice()) else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented nullable-count tranche requires one Boolean group and one reduction",
            ));
        };
        if group.expression
            != (ResidentQuantifierExpression::IsNull {
                expression: Box::new(ResidentQuantifierExpression::Slot(*source_slot)),
                negated: false,
            })
            || reduction.kind != super::ResidentSegmentedAggregateKind::CountAll
            || reduction.input.is_some()
            || reduction.distinct
            || self.maximum_list_items != 0
            || self.outputs.len() != 2
            || self
                .outputs
                .iter()
                .map(|output| output.source)
                .collect::<BTreeSet<_>>()
                != BTreeSet::from([group.output, reduction.output])
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "sealed segmented nullable-count tranche admits only Boolean IS NULL grouping and count(*)",
            ));
        }
        Ok(ResidentSegmentedNullableIsNullCountProgram {
            request,
            relationship_output: *relationship_output,
            source_slot: *source_slot,
            aggregate_stage,
            group,
            reduction,
        })
    }

    pub fn obligations(&self) -> Vec<ResidentExecutionObligation> {
        let mut obligations = Vec::with_capacity(self.stages.len().saturating_add(8));
        obligations.push(self.source.obligation());
        for stage in &self.stages {
            obligations.push(stage.obligation);
            if let ResidentSegmentedAggregationOperation::Aggregate { reductions, .. } =
                &stage.operation
            {
                obligations.extend(reductions.iter().map(|reduction| reduction.obligation));
            }
        }
        obligations
    }

    fn aggregate_capacity_bounds(&self) -> Result<(usize, usize)> {
        let mut rows = self.source.maximum_row_count();
        let mut final_aggregate_input_rows = None;
        let mut collect_items = 0_usize;
        for stage in &self.stages {
            match &stage.operation {
                ResidentSegmentedAggregationOperation::Project { .. }
                | ResidentSegmentedAggregationOperation::PropertyKeys { .. }
                | ResidentSegmentedAggregationOperation::PropertyValue { .. }
                | ResidentSegmentedAggregationOperation::Filter { .. }
                | ResidentSegmentedAggregationOperation::Order { .. } => {}
                ResidentSegmentedAggregationOperation::Skip { rows: skipped } => {
                    rows = rows.saturating_sub(*skipped as usize);
                }
                ResidentSegmentedAggregationOperation::Limit { rows: limit } => {
                    rows = rows.min(*limit as usize);
                }
                ResidentSegmentedAggregationOperation::Unwind { .. } => {
                    rows = rows
                        .checked_mul(self.maximum_list_items as usize)
                        .ok_or_else(scratch_overflow)?;
                }
                ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } => {
                    final_aggregate_input_rows = Some(rows);
                    let collect_count = reductions
                        .iter()
                        .filter(|reduction| {
                            reduction.kind == super::ResidentSegmentedAggregateKind::Collect
                        })
                        .count();
                    collect_items = rows
                        .checked_mul(collect_count)
                        .and_then(|items| collect_items.checked_add(items))
                        .ok_or_else(scratch_overflow)?;
                    rows = if groups.is_empty() { 1 } else { rows };
                }
            }
        }
        final_aggregate_input_rows
            .map(|rows| (rows, collect_items))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident segmented program omits its aggregate stage",
                )
            })
    }

    /// Maximum rows which need materialization before the final aggregate. A leading LIMIT may be
    /// fused into deterministic sources such as `range()` without changing its logical source
    /// cardinality receipt. Earlier aggregate boundaries update the bounded row stream and are not
    /// mistaken for the transitional final aggregate descriptor.
    pub fn maximum_aggregate_input_rows(&self) -> Result<usize> {
        self.validate()?;
        self.aggregate_capacity_bounds().map(|(rows, _)| rows)
    }

    /// Maximum number of non-null collect elements retained across every aggregate stage. This is
    /// a whole-program capacity: hidden lists consumed by a later UNWIND remain device-owned state
    /// even when the final aggregate has no COLLECT descriptor.
    pub fn maximum_collect_items(&self) -> Result<usize> {
        self.validate()?;
        self.aggregate_capacity_bounds().map(|(_, items)| items)
    }

    pub fn fingerprint(&self) -> Result<[u8; 32]> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new_derive_key("irongraph.resident-segmented-program.v4");
        hasher.update(&self.catalog_generation);
        hasher.update(&self.slot_count.to_le_bytes());
        match &self.source {
            ResidentSegmentedAggregationSource::Unit {
                prelude,
                obligation,
            } => {
                hasher.update(&[0]);
                match prelude {
                    ResidentSegmentedUnitPrelude::Empty => {
                        hasher.update(&[0]);
                    }
                    ResidentSegmentedUnitPrelude::RawTemporalList { program, output } => {
                        hasher.update(&[1]);
                        hash_quantifier_len(&mut hasher, program.invocations.len())?;
                        super::resident_hash_temporal_value_invocations(
                            &mut hasher,
                            &program.invocations,
                        )?;
                        hash_quantifier_len(&mut hasher, program.output_registers.len())?;
                        for register in &program.output_registers {
                            hasher.update(&register.to_le_bytes());
                        }
                        hasher.update(&output.0.to_le_bytes());
                    }
                }
                hash_quantifier_obligation(&mut hasher, *obligation);
            }
            ResidentSegmentedAggregationSource::Range {
                request,
                output,
                obligation,
            } => {
                hasher.update(&[1]);
                hasher.update(&request.start.to_le_bytes());
                hasher.update(&request.end.to_le_bytes());
                hasher.update(&request.step.to_le_bytes());
                hasher.update(&request.integer_operands.map(u8::from));
                hash_quantifier_len(&mut hasher, request.max_values)?;
                hasher.update(&output.0.to_le_bytes());
                hash_quantifier_obligation(&mut hasher, *obligation);
            }
            ResidentSegmentedAggregationSource::NullableRelationship {
                request,
                relationship_output,
                output,
                obligation,
            } => {
                hasher.update(&[2]);
                hasher.update(&request.manifest.fingerprint.0);
                hasher.update(&relationship_output.to_le_bytes());
                hasher.update(&output.0.to_le_bytes());
                hash_quantifier_obligation(&mut hasher, *obligation);
            }
            ResidentSegmentedAggregationSource::GraphRelation {
                request,
                bindings,
                obligation,
            } => {
                hasher.update(&[3]);
                hasher.update(&request.manifest.fingerprint.0);
                hash_quantifier_len(&mut hasher, bindings.len())?;
                for binding in bindings {
                    hasher.update(&binding.relation_output.to_le_bytes());
                    hasher.update(&binding.output.0.to_le_bytes());
                }
                hash_quantifier_obligation(&mut hasher, *obligation);
            }
        }
        hash_quantifier_len(&mut hasher, self.stages.len())?;
        for stage in &self.stages {
            hash_quantifier_obligation(&mut hasher, stage.obligation);
            match &stage.operation {
                ResidentSegmentedAggregationOperation::Project {
                    keep_scope,
                    bindings,
                } => {
                    hasher.update(&[0, u8::from(*keep_scope)]);
                    hash_quantifier_len(&mut hasher, bindings.len())?;
                    for binding in bindings {
                        hasher.update(&binding.output.0.to_le_bytes());
                        hash_quantifier_expression(&mut hasher, &binding.expression)?;
                    }
                }
                ResidentSegmentedAggregationOperation::PropertyKeys {
                    input,
                    kind,
                    descriptors,
                    output,
                } => {
                    hasher.update(&[7, *kind as u8]);
                    hasher.update(&input.0.to_le_bytes());
                    hasher.update(&output.0.to_le_bytes());
                    hash_quantifier_len(&mut hasher, descriptors.len())?;
                    for descriptor in descriptors {
                        hasher.update(&descriptor.property.0.to_le_bytes());
                        hash_quantifier_string(&mut hasher, &descriptor.name)?;
                    }
                }
                ResidentSegmentedAggregationOperation::PropertyValue {
                    input,
                    kind,
                    property,
                    output,
                    maximum_value_items,
                    maximum_list_items,
                    maximum_string_bytes,
                } => {
                    hasher.update(&[8, *kind as u8]);
                    hasher.update(&input.0.to_le_bytes());
                    hasher.update(&output.0.to_le_bytes());
                    match property {
                        Some(property) => {
                            hasher.update(&[1]);
                            hasher.update(&property.0.to_le_bytes());
                        }
                        None => {
                            hasher.update(&[0]);
                            hasher.update(&0_u64.to_le_bytes());
                        }
                    }
                    hasher.update(&maximum_value_items.to_le_bytes());
                    hasher.update(&maximum_list_items.to_le_bytes());
                    hasher.update(&maximum_string_bytes.to_le_bytes());
                }
                ResidentSegmentedAggregationOperation::Unwind { expression, output } => {
                    hasher.update(&[1]);
                    hasher.update(&output.0.to_le_bytes());
                    hash_quantifier_expression(&mut hasher, expression)?;
                }
                ResidentSegmentedAggregationOperation::Filter { predicate } => {
                    hasher.update(&[2]);
                    hash_quantifier_expression(&mut hasher, predicate)?;
                }
                ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } => {
                    hasher.update(&[3]);
                    hash_quantifier_len(&mut hasher, groups.len())?;
                    for group in groups {
                        hasher.update(&group.output.0.to_le_bytes());
                        hash_quantifier_expression(&mut hasher, &group.expression)?;
                    }
                    hash_quantifier_len(&mut hasher, reductions.len())?;
                    for reduction in reductions {
                        hasher.update(&[
                            reduction.kind as u8,
                            u8::from(reduction.distinct),
                            u8::from(reduction.input.is_some()),
                        ]);
                        hasher.update(&reduction.output.0.to_le_bytes());
                        if let Some(input) = &reduction.input {
                            hash_quantifier_expression(&mut hasher, input)?;
                        }
                        match reduction.percentile {
                            Some(percentile) => {
                                hasher.update(&[1]);
                                hasher.update(&percentile.to_le_bytes());
                            }
                            None => {
                                hasher.update(&[0]);
                                hasher.update(&0_u64.to_le_bytes());
                            }
                        }
                        hash_quantifier_obligation(&mut hasher, reduction.obligation);
                    }
                }
                ResidentSegmentedAggregationOperation::Order { keys } => {
                    hasher.update(&[4]);
                    hash_quantifier_len(&mut hasher, keys.len())?;
                    for key in keys {
                        hasher.update(&[u8::from(key.descending), u8::from(key.nulls_first)]);
                        hash_quantifier_expression(&mut hasher, &key.expression)?;
                    }
                }
                ResidentSegmentedAggregationOperation::Skip { rows } => {
                    hasher.update(&[5]);
                    hasher.update(&rows.to_le_bytes());
                }
                ResidentSegmentedAggregationOperation::Limit { rows } => {
                    hasher.update(&[6]);
                    hasher.update(&rows.to_le_bytes());
                }
            }
        }
        hash_quantifier_len(&mut hasher, self.outputs.len())?;
        for output in &self.outputs {
            hash_quantifier_string(&mut hasher, &output.name)?;
            hasher.update(&output.source.0.to_le_bytes());
        }
        hasher.update(&self.maximum_intermediate_rows.to_le_bytes());
        hasher.update(&self.maximum_list_items.to_le_bytes());
        hasher.update(&self.random_seed.to_le_bytes());
        Ok(*hasher.finalize().as_bytes())
    }
}

#[cfg(test)]
mod segmented_temporal_unit_prelude_tests {
    use std::collections::BTreeSet;

    use crate::{
        CompareOp, ResidentSegmentedAggregateKind, ResidentTemporalValueFunction,
        ResidentTemporalValueInput, ResidentTemporalValueInvocation,
        ResidentTemporalValueProgramRequest,
    };

    use super::*;

    fn obligation(
        id: u64,
        kind: ResidentObligationKind,
        scope: ResidentObligationScope,
    ) -> ResidentExecutionObligation {
        ResidentExecutionObligation { id, kind, scope }
    }

    fn temporal_request(
        functions: &[ResidentTemporalValueFunction],
        output_registers: Vec<u16>,
    ) -> ResidentTemporalValueProgramRequest {
        ResidentTemporalValueProgramRequest {
            invocations: functions
                .iter()
                .enumerate()
                .map(|(index, function)| ResidentTemporalValueInvocation {
                    function: *function,
                    input: ResidentTemporalValueInput::Map(index.to_le_bytes().to_vec()),
                })
                .collect(),
            output_registers,
        }
    }

    fn program_with_prelude(
        prelude: ResidentSegmentedUnitPrelude,
    ) -> ResidentSegmentedAggregationProgram {
        ResidentSegmentedAggregationProgram {
            catalog_generation: [0x5a; 32],
            slot_count: 3,
            source: ResidentSegmentedAggregationSource::Unit {
                prelude,
                obligation: obligation(
                    1,
                    ResidentObligationKind::Expression,
                    ResidentObligationScope::Expression(u16::MAX - 2),
                ),
            },
            stages: vec![ResidentSegmentedAggregationStage {
                operation: ResidentSegmentedAggregationOperation::Aggregate {
                    groups: Vec::new(),
                    reductions: vec![ResidentSegmentedAggregationReduction {
                        output: ResidentQuantifierSlot(2),
                        kind: ResidentSegmentedAggregateKind::CountAll,
                        input: None,
                        distinct: false,
                        percentile: None,
                        obligation: obligation(
                            3,
                            ResidentObligationKind::Aggregate,
                            ResidentObligationScope::Expression(0),
                        ),
                    }],
                },
                obligation: obligation(
                    2,
                    ResidentObligationKind::Aggregate,
                    ResidentObligationScope::PatternFinal,
                ),
            }],
            outputs: vec![ResidentQuantifierOutput {
                name: "count".to_owned(),
                source: ResidentQuantifierSlot(2),
            }],
            maximum_intermediate_rows: 1,
            maximum_list_items: 2,
            random_seed: 0,
        }
    }

    fn raw_prelude(
        functions: &[ResidentTemporalValueFunction],
        output_registers: Vec<u16>,
        output: u16,
    ) -> ResidentSegmentedUnitPrelude {
        ResidentSegmentedUnitPrelude::RawTemporalList {
            program: temporal_request(functions, output_registers),
            output: ResidentQuantifierSlot(output),
        }
    }

    #[test]
    fn raw_temporal_unit_prelude_requires_one_bounded_ordered_family() -> Result<()> {
        let functions = [
            ResidentTemporalValueFunction::Date,
            ResidentTemporalValueFunction::Date,
        ];
        let valid = program_with_prelude(raw_prelude(&functions, vec![0, 1], 0));
        valid.validate()?;
        assert_eq!(
            valid.source.initial_slots(),
            BTreeSet::from([ResidentQuantifierSlot(0)])
        );

        for prelude in [
            raw_prelude(&functions, Vec::new(), 0),
            raw_prelude(&functions, vec![0, 0], 0),
            raw_prelude(&functions, vec![1, 0], 0),
            raw_prelude(&functions, vec![0, 2], 0),
            raw_prelude(&functions, vec![0, 1], 3),
            raw_prelude(
                &[
                    ResidentTemporalValueFunction::Date,
                    ResidentTemporalValueFunction::LocalTime,
                ],
                vec![0, 1],
                0,
            ),
            raw_prelude(&[ResidentTemporalValueFunction::Duration], vec![0], 0),
        ] {
            assert!(program_with_prelude(prelude).validate().is_err());
        }

        let mut non_temporal = temporal_request(&functions, vec![2]);
        non_temporal
            .invocations
            .push(ResidentTemporalValueInvocation {
                function: ResidentTemporalValueFunction::Date,
                input: ResidentTemporalValueInput::Comparison {
                    operation: CompareOp::Less,
                    left: 0,
                    right: 1,
                },
            });
        assert!(
            program_with_prelude(ResidentSegmentedUnitPrelude::RawTemporalList {
                program: non_temporal,
                output: ResidentQuantifierSlot(0),
            })
            .validate()
            .is_err()
        );

        let mut over_capacity = valid;
        over_capacity.maximum_list_items = 1;
        assert!(over_capacity.validate().is_err());
        Ok(())
    }

    #[test]
    fn raw_temporal_unit_fingerprint_binds_program_selection_and_destination() -> Result<()> {
        let functions = [
            ResidentTemporalValueFunction::Date,
            ResidentTemporalValueFunction::Date,
            ResidentTemporalValueFunction::Date,
        ];
        let baseline = program_with_prelude(raw_prelude(&functions, vec![0, 1], 0));
        let mut raw_bytes = baseline.clone();
        let ResidentSegmentedAggregationSource::Unit {
            prelude: ResidentSegmentedUnitPrelude::RawTemporalList { program, .. },
            ..
        } = &mut raw_bytes.source
        else {
            unreachable!();
        };
        program.invocations[0].input = ResidentTemporalValueInput::Map(vec![0, 1]);

        let selected = program_with_prelude(raw_prelude(&functions, vec![0, 2], 0));
        let destination = program_with_prelude(raw_prelude(&functions, vec![0, 1], 1));
        let empty = program_with_prelude(ResidentSegmentedUnitPrelude::Empty);
        let fingerprints = [baseline, raw_bytes, selected, destination, empty]
            .into_iter()
            .map(|program| program.fingerprint())
            .collect::<Result<BTreeSet<_>>>()?;
        assert_eq!(fingerprints.len(), 5);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierGeneration {
    pub project: ProjectId,
    pub bookmark: Bookmark,
    pub graph_revision: u64,
    pub layout_version: ResidentGraphLayoutVersion,
    pub catalog_generation: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierFingerprint(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierProgramRequest {
    pub generation: ResidentQuantifierGeneration,
    pub execution: ResidentExecutionId,
    pub source: ResidentQuantifierSource,
    pub program: ResidentQuantifierProgram,
    pub max_rows: usize,
    pub max_list_items: usize,
    /// Seed for the command-owned deterministic PRNG used by each native `rand()` instruction.
    /// Supplying the seed is query input; random branch selection remains backend work.
    pub random_seed: u64,
    pub fingerprint: ResidentQuantifierFingerprint,
}

impl ResidentQuantifierProgramRequest {
    pub fn build(
        generation: ResidentQuantifierGeneration,
        execution: ResidentExecutionId,
        program: ResidentQuantifierProgram,
        max_rows: usize,
        max_list_items: usize,
        random_seed: u64,
    ) -> Result<Self> {
        Self::build_with_source(
            generation,
            execution,
            ResidentQuantifierSource::Unit,
            program,
            max_rows,
            max_list_items,
            random_seed,
        )
    }

    pub fn build_with_source(
        generation: ResidentQuantifierGeneration,
        execution: ResidentExecutionId,
        source: ResidentQuantifierSource,
        program: ResidentQuantifierProgram,
        max_rows: usize,
        max_list_items: usize,
        random_seed: u64,
    ) -> Result<Self> {
        let mut request = Self {
            generation,
            execution,
            source,
            program,
            max_rows,
            max_list_items,
            random_seed,
            fingerprint: ResidentQuantifierFingerprint([0; 32]),
        };
        request.fingerprint = request.compute_fingerprint()?;
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_structure()?;
        if self.compute_fingerprint()? != self.fingerprint {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier request fingerprint does not match its immutable command",
            ));
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<()> {
        if self.execution.high == 0 && self.execution.low == 0 {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier command requires a non-zero execution ID",
            ));
        }
        if self.max_rows == 0 {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier row capacity must be non-zero",
            ));
        }
        if self.max_list_items == 0 || self.max_list_items > RESIDENT_QUANTIFIER_MAX_LITERAL_ITEMS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier list capacity is outside its bounded contract",
            ));
        }
        self.source.validate(
            self.generation,
            self.execution,
            usize::from(self.program.slot_count),
            self.max_rows,
            self.max_list_items,
        )?;
        self.program
            .validate_with_initial_slots(&self.source.initial_slots())?;
        let obligations = self.obligations();
        let mut obligation_ids = BTreeSet::new();
        if obligations
            .iter()
            .any(|obligation| obligation.id == 0 || !obligation_ids.insert(obligation.id))
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier command has a zero or duplicate source/stage obligation",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn obligations(&self) -> Vec<ResidentExecutionObligation> {
        let mut obligations = match &self.source {
            ResidentQuantifierSource::Unit => Vec::new(),
            ResidentQuantifierSource::Range { obligation, .. } => vec![*obligation],
            ResidentQuantifierSource::VariablePathEntityList {
                path,
                materialize_obligation,
                ..
            } => {
                let mut obligations = path.obligations();
                obligations.push(*materialize_obligation);
                obligations.push(quantifier_entity_dependency_obligation());
                obligations
            }
        };
        obligations.extend(
            self.program
                .stages
                .iter()
                .enumerate()
                .map(|(index, stage)| {
                    let scope_index = u16::try_from(index).unwrap_or(u16::MAX - 1);
                    let (kind, scope) = match stage {
                        ResidentQuantifierStage::Filter { .. } => (
                            ResidentObligationKind::Filter,
                            ResidentObligationScope::Filter(scope_index),
                        ),
                        ResidentQuantifierStage::GroupCount { .. } => (
                            ResidentObligationKind::Aggregate,
                            ResidentObligationScope::Expression(scope_index),
                        ),
                        ResidentQuantifierStage::Project { .. }
                        | ResidentQuantifierStage::Unwind { .. } => (
                            ResidentObligationKind::Expression,
                            ResidentObligationScope::Expression(scope_index),
                        ),
                    };
                    ResidentExecutionObligation {
                        id: 0x5155_414e_0000_0001_u64 + index as u64,
                        kind,
                        scope,
                    }
                })
                .chain(std::iter::once(ResidentExecutionObligation {
                    id: 0x5155_414e_ffff_ffff,
                    kind: ResidentObligationKind::Expression,
                    scope: ResidentObligationScope::PatternFinal,
                })),
        );
        obligations
    }

    #[must_use]
    pub fn source_receipt_count(&self) -> usize {
        match &self.source {
            ResidentQuantifierSource::Unit => 0,
            ResidentQuantifierSource::Range { .. } => 1,
            ResidentQuantifierSource::VariablePathEntityList { path, .. } => {
                path.obligations().len() + 2
            }
        }
    }

    pub fn scratch_bytes(&self) -> Result<usize> {
        self.validate_structure()?;
        let row_bytes = self
            .max_rows
            .checked_mul(usize::from(self.program.slot_count))
            .and_then(|cells| cells.checked_mul(size_of::<ResidentQuantifierValue>()))
            .ok_or_else(scratch_overflow)?;
        let source_bytes = match &self.source {
            ResidentQuantifierSource::Unit | ResidentQuantifierSource::Range { .. } => 0,
            ResidentQuantifierSource::VariablePathEntityList {
                path,
                entity_kind,
                properties,
                ..
            } => {
                let handles = path
                    .maximum_output_rows
                    .checked_mul(self.source.maximum_entity_list_items()?)
                    .and_then(|items| {
                        items.checked_mul(size_of::<ResidentQuantifierEntityHandle>())
                    })
                    .ok_or_else(scratch_overflow)?;
                let lists = path
                    .maximum_output_rows
                    .checked_mul(size_of::<ResidentQuantifierEntityList>())
                    .ok_or_else(scratch_overflow)?;
                let dependencies = self
                    .source
                    .maximum_read_dependencies()?
                    .checked_mul(size_of::<ResidentQuantifierEntityReadDependency>())
                    .ok_or_else(scratch_overflow)?;
                let entity_rows = match entity_kind {
                    ResidentQuantifierEntityKind::Node => path.expected_node_slots,
                    ResidentQuantifierEntityKind::Relationship => path.expected_edge_slots,
                };
                let property_cells = entity_rows
                    .checked_mul(properties.len())
                    .and_then(|cells| cells.checked_mul(size_of::<ResidentQuantifierValue>()))
                    .ok_or_else(scratch_overflow)?;
                let property_string_stride =
                    properties.iter().try_fold(0_usize, |bytes, property| {
                        bytes
                            .checked_add(property.shape.maximum_string_bytes() as usize)
                            .ok_or_else(scratch_overflow)
                    })?;
                let property_strings = entity_rows
                    .checked_mul(property_string_stride)
                    .ok_or_else(scratch_overflow)?;
                path.scratch_bytes()?
                    .checked_add(handles)
                    .and_then(|bytes| bytes.checked_add(lists))
                    .and_then(|bytes| bytes.checked_add(dependencies))
                    .and_then(|bytes| bytes.checked_add(property_cells))
                    .and_then(|bytes| bytes.checked_add(property_strings))
                    .ok_or_else(scratch_overflow)?
            }
        };
        row_bytes
            .checked_add(source_bytes)
            .ok_or_else(scratch_overflow)
    }

    fn compute_fingerprint(&self) -> Result<ResidentQuantifierFingerprint> {
        self.validate_structure()?;
        let mut hasher = blake3::Hasher::new_derive_key("irongraph.resident-quantifier-program.v2");
        hasher.update(self.generation.project.0.as_bytes());
        for value in [
            self.generation.bookmark.term,
            self.generation.bookmark.index,
            self.generation.graph_revision,
            self.generation.layout_version,
            self.execution.high,
            self.execution.low,
            self.max_rows as u64,
            self.max_list_items as u64,
            self.random_seed,
            u64::from(self.program.slot_count),
        ] {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&self.generation.catalog_generation);
        match &self.source {
            ResidentQuantifierSource::Unit => {
                hasher.update(&[0]);
            }
            ResidentQuantifierSource::Range {
                request,
                output,
                obligation,
            } => {
                hasher.update(&[2]);
                hasher.update(&request.start.to_le_bytes());
                hasher.update(&request.end.to_le_bytes());
                hasher.update(&request.step.to_le_bytes());
                hasher.update(&request.integer_operands.map(u8::from));
                hasher.update(&(request.max_values as u64).to_le_bytes());
                hasher.update(&output.0.to_le_bytes());
                hash_quantifier_obligation(&mut hasher, *obligation);
            }
            ResidentQuantifierSource::VariablePathEntityList {
                path,
                output,
                entity_kind,
                skip,
                properties,
                materialize_obligation,
            } => {
                hasher.update(&[1, *entity_kind as u8]);
                hasher.update(path.fingerprint()?.as_bytes());
                hasher.update(&output.0.to_le_bytes());
                hasher.update(&skip.to_le_bytes());
                hash_quantifier_len(&mut hasher, properties.len())?;
                for property in properties {
                    hash_quantifier_string(&mut hasher, &property.key)?;
                    match property.property {
                        Some(property) => {
                            hasher.update(&[1]);
                            hasher.update(&property.0.to_le_bytes());
                        }
                        None => {
                            hasher.update(&[0]);
                        }
                    }
                    match property.shape {
                        ResidentQuantifierEntityPropertyShape::Absent => {
                            hasher.update(&[0]);
                        }
                        ResidentQuantifierEntityPropertyShape::Boolean => {
                            hasher.update(&[1]);
                        }
                        ResidentQuantifierEntityPropertyShape::Integer => {
                            hasher.update(&[2]);
                        }
                        ResidentQuantifierEntityPropertyShape::Float => {
                            hasher.update(&[3]);
                        }
                        ResidentQuantifierEntityPropertyShape::String { maximum_bytes } => {
                            hasher.update(&[4]);
                            hasher.update(&maximum_bytes.to_le_bytes());
                        }
                        ResidentQuantifierEntityPropertyShape::MixedScalar {
                            maximum_string_bytes,
                        } => {
                            hasher.update(&[5]);
                            hasher.update(&maximum_string_bytes.to_le_bytes());
                        }
                    }
                }
                hash_quantifier_obligation(&mut hasher, *materialize_obligation);
                hasher.update(&(self.source.maximum_read_dependencies()? as u64).to_le_bytes());
                hash_quantifier_obligation(&mut hasher, quantifier_entity_dependency_obligation());
            }
        }
        hash_quantifier_program(&mut hasher, &self.program)?;
        Ok(ResidentQuantifierFingerprint(*hasher.finalize().as_bytes()))
    }
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierProgramResultParts {
    pub generation: ResidentQuantifierGeneration,
    pub execution: ResidentExecutionId,
    pub fingerprint: ResidentQuantifierFingerprint,
    pub rows: Vec<Vec<ResidentQuantifierValue>>,
    pub receipts: Vec<ResidentExecutionReceipt>,
}

/// Typed entity-list output cell. `parts.rows[row][column]` is a null scalar placeholder so the
/// legacy graph-free scalar ABI cannot mistake a dense graph handle for Cypher INTEGER content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierEntityOutput {
    pub row: u32,
    pub column: u16,
    pub value: ResidentQuantifierEntityList,
}

/// Entity extension to a quantifier result. It repeats the immutable command identity so these
/// graph reads cannot be detached from their generation/fingerprint and attached to another
/// scalar result packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierEntityResultParts {
    pub generation: ResidentQuantifierGeneration,
    pub execution: ResidentExecutionId,
    pub fingerprint: ResidentQuantifierFingerprint,
    /// Internal accepted trails retained only to validate complete dependency publication. They
    /// are not a second path-backend result and are not exposed as query rows.
    pub source_paths: Vec<ResidentVariablePath>,
    pub outputs: Vec<ResidentQuantifierEntityOutput>,
    pub read_dependencies: Vec<ResidentQuantifierEntityReadDependency>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentQuantifierProgramResult {
    parts: ResidentQuantifierProgramResultParts,
    entity_parts: Option<ResidentQuantifierEntityResultParts>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedResidentQuantifierProgramResult {
    parts: ResidentQuantifierProgramResultParts,
    entity_parts: Option<ResidentQuantifierEntityResultParts>,
}

impl ResidentQuantifierProgramResult {
    #[allow(dead_code)] // Reserved for the CPU/accelerator implementations of this sealed ABI.
    pub fn completed(parts: ResidentQuantifierProgramResultParts) -> Self {
        Self {
            parts,
            entity_parts: None,
        }
    }

    pub fn completed_with_entity_parts(
        parts: ResidentQuantifierProgramResultParts,
        entity_parts: ResidentQuantifierEntityResultParts,
    ) -> Self {
        Self {
            parts,
            entity_parts: Some(entity_parts),
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn from_untrusted_parts(parts: ResidentQuantifierProgramResultParts) -> Self {
        Self {
            parts,
            entity_parts: None,
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn from_untrusted_entity_parts(
        parts: ResidentQuantifierProgramResultParts,
        entity_parts: ResidentQuantifierEntityResultParts,
    ) -> Self {
        Self {
            parts,
            entity_parts: Some(entity_parts),
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn into_untrusted_parts(self) -> ResidentQuantifierProgramResultParts {
        self.parts
    }

    #[doc(hidden)]
    #[must_use]
    pub fn into_untrusted_entity_parts(
        self,
    ) -> (
        ResidentQuantifierProgramResultParts,
        Option<ResidentQuantifierEntityResultParts>,
    ) {
        (self.parts, self.entity_parts)
    }

    pub fn validate(
        self,
        request: &ResidentQuantifierProgramRequest,
        backend: BackendKind,
    ) -> Result<ValidatedResidentQuantifierProgramResult> {
        request.validate()?;
        let parts = self.parts;
        let entity_parts = self.entity_parts;
        if parts.generation != request.generation
            || parts.execution != request.execution
            || parts.fingerprint != request.fingerprint
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident quantifier result belongs to a different command or graph generation",
            ));
        }
        if parts.rows.len() > request.max_rows
            || parts
                .rows
                .iter()
                .any(|row| row.len() != request.program.outputs.len())
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident quantifier result has an invalid row shape",
            ));
        }
        for row in &parts.rows {
            for value in row {
                let mut items = 0;
                value.validate(0, &mut items)?;
                if items > request.max_list_items {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident quantifier output exceeds its admitted value capacity",
                    ));
                }
            }
        }
        let completion = match backend {
            BackendKind::Cpu => ResidentDeviceCompletion::CpuReference,
            BackendKind::Metal => ResidentDeviceCompletion::Metal,
            BackendKind::Cuda => {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "selected backend has no quantifier completion provenance",
                ));
            }
        };
        let (dependency_count, source_path_count) =
            validate_quantifier_entity_parts(request, &parts.rows, entity_parts.as_ref())?;
        let obligations = request.obligations();
        if parts.receipts.len() != obligations.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident quantifier result omitted an ordered stage receipt",
            ));
        }
        for (receipt, obligation) in parts.receipts.iter().zip(&obligations) {
            if receipt.execution != request.execution
                || receipt.obligation != *obligation
                || receipt.completion != completion
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident quantifier receipt has invalid order or provenance",
                ));
            }
        }
        let mut receipt_index = 0_usize;
        let mut prior_rows = match &request.source {
            ResidentQuantifierSource::Unit => 1_u64,
            ResidentQuantifierSource::Range { .. } => {
                let source = parts.receipts[receipt_index];
                receipt_index += 1;
                if source.input_cardinality != 1
                    || source.output_cardinality > request.max_rows as u64
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident quantifier range receipt has invalid source cardinality",
                    ));
                }
                source.output_cardinality
            }
            ResidentQuantifierSource::VariablePathEntityList { path, .. } => {
                let path_receipt_count = path.obligations().len();
                let path_receipts = &parts.receipts[..path_receipt_count];
                let path_rows = path_receipts
                    .last()
                    .ok_or_else(|| Error::internal("quantifier path receipt disappeared"))?
                    .output_cardinality;
                path.validate_execution_receipts(path_receipts, completion, path_rows)?;
                receipt_index += path_receipt_count;
                let materialize = parts.receipts[receipt_index];
                receipt_index += 1;
                if materialize.input_cardinality != path_rows
                    || materialize.output_cardinality != path_rows
                    || source_path_count as u64 != path_rows
                    || path_rows > request.max_rows as u64
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident quantifier entity-list receipt has invalid source cardinality",
                    ));
                }
                let dependencies = parts.receipts[receipt_index];
                receipt_index += 1;
                if dependencies.input_cardinality != path_rows
                    || dependencies.output_cardinality != dependency_count as u64
                    || dependency_count > request.source.maximum_read_dependencies()?
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident quantifier dependency receipt has invalid cardinality",
                    ));
                }
                path_rows
            }
        };
        for stage in &request.program.stages {
            let receipt = parts.receipts[receipt_index];
            receipt_index += 1;
            if receipt.input_cardinality != prior_rows
                || receipt.output_cardinality > request.max_rows as u64
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident quantifier stage receipt has invalid cardinality",
                ));
            }
            match stage {
                ResidentQuantifierStage::Project { .. }
                    if receipt.output_cardinality != receipt.input_cardinality =>
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident quantifier projection changed row cardinality",
                    ));
                }
                ResidentQuantifierStage::Filter { .. }
                    if receipt.output_cardinality > receipt.input_cardinality =>
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident quantifier filter increased row cardinality",
                    ));
                }
                ResidentQuantifierStage::GroupCount { .. }
                    if receipt.output_cardinality > receipt.input_cardinality.max(1) =>
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident quantifier grouping returned an impossible group count",
                    ));
                }
                ResidentQuantifierStage::Unwind { .. }
                | ResidentQuantifierStage::Project { .. }
                | ResidentQuantifierStage::Filter { .. }
                | ResidentQuantifierStage::GroupCount { .. } => {}
            }
            prior_rows = receipt.output_cardinality;
        }
        let publication = parts.receipts[receipt_index];
        if publication.input_cardinality != prior_rows
            || publication.output_cardinality != parts.rows.len() as u64
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident quantifier publication receipt does not match its rows",
            ));
        }
        Ok(ValidatedResidentQuantifierProgramResult {
            parts,
            entity_parts,
        })
    }
}

impl ValidatedResidentQuantifierProgramResult {
    #[must_use]
    pub fn rows(&self) -> &[Vec<ResidentQuantifierValue>] {
        &self.parts.rows
    }

    #[must_use]
    pub fn receipts(&self) -> &[ResidentExecutionReceipt] {
        &self.parts.receipts
    }

    #[must_use]
    pub fn entity_outputs(&self) -> &[ResidentQuantifierEntityOutput] {
        self.entity_parts
            .as_ref()
            .map_or(&[], |parts| parts.outputs.as_slice())
    }

    #[must_use]
    pub fn read_dependencies(&self) -> &[ResidentQuantifierEntityReadDependency] {
        self.entity_parts
            .as_ref()
            .map_or(&[], |parts| parts.read_dependencies.as_slice())
    }

    #[must_use]
    pub fn entity_list(&self, row: usize, column: usize) -> Option<&ResidentQuantifierEntityList> {
        self.entity_outputs()
            .binary_search_by_key(&(row as u32, column as u16), |output| {
                (output.row, output.column)
            })
            .ok()
            .and_then(|index| self.entity_outputs().get(index))
            .map(|output| &output.value)
    }

    #[must_use]
    pub fn into_parts(self) -> ResidentQuantifierProgramResultParts {
        self.parts
    }

    #[must_use]
    pub fn into_entity_parts(
        self,
    ) -> (
        ResidentQuantifierProgramResultParts,
        Option<ResidentQuantifierEntityResultParts>,
    ) {
        (self.parts, self.entity_parts)
    }
}

fn validate_quantifier_entity_parts(
    request: &ResidentQuantifierProgramRequest,
    rows: &[Vec<ResidentQuantifierValue>],
    entity_parts: Option<&ResidentQuantifierEntityResultParts>,
) -> Result<(usize, usize)> {
    let ResidentQuantifierSource::VariablePathEntityList {
        path,
        entity_kind,
        properties,
        ..
    } = &request.source
    else {
        if entity_parts.is_some() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "graph-free quantifier result contains entity source data",
            ));
        }
        return Ok((0, 0));
    };
    let parts = entity_parts.ok_or_else(|| {
        Error::new(
            ErrorCode::CorruptStorage,
            "entity-source quantifier result omitted entity outputs and read dependencies",
        )
    })?;
    if parts.generation != request.generation
        || parts.execution != request.execution
        || parts.fingerprint != request.fingerprint
        || parts.source_paths.len() > path.maximum_output_rows
        || parts.read_dependencies.len() > request.source.maximum_read_dependencies()?
        || parts
            .read_dependencies
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || parts
            .outputs
            .windows(2)
            .any(|pair| (pair[0].row, pair[0].column) >= (pair[1].row, pair[1].column))
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "resident quantifier entity result is detached, unbounded, or non-canonical",
        ));
    }
    for source_path in &parts.source_paths {
        path.validate_path_trail(source_path)?;
    }
    let property_ids = properties
        .iter()
        .filter_map(|property| property.property)
        .collect::<BTreeSet<_>>();
    let valid_handle = |handle: ResidentQuantifierEntityHandle| match handle {
        ResidentQuantifierEntityHandle::Node(handle) => {
            (handle.0 as usize) < path.expected_node_slots
        }
        ResidentQuantifierEntityHandle::Relationship(handle) => {
            (handle.0 as usize) < path.expected_edge_slots
        }
    };
    let mut expected_dependencies = BTreeSet::new();
    for source_path in &parts.source_paths {
        for node in &source_path.nodes {
            expected_dependencies.insert(ResidentQuantifierEntityReadDependency {
                entity: ResidentQuantifierEntityHandle::Node(ResidentQuantifierNodeHandle(*node)),
                property: None,
            });
        }
        for relationship in &source_path.relationships {
            expected_dependencies.insert(ResidentQuantifierEntityReadDependency {
                entity: ResidentQuantifierEntityHandle::Relationship(
                    ResidentQuantifierRelationshipHandle(*relationship),
                ),
                property: None,
            });
        }
        let tail_entities = match entity_kind {
            ResidentQuantifierEntityKind::Node => source_path
                .nodes
                .iter()
                .skip(1)
                .map(|row| ResidentQuantifierEntityHandle::Node(ResidentQuantifierNodeHandle(*row)))
                .collect::<Vec<_>>(),
            ResidentQuantifierEntityKind::Relationship => source_path
                .relationships
                .iter()
                .skip(1)
                .map(|row| {
                    ResidentQuantifierEntityHandle::Relationship(
                        ResidentQuantifierRelationshipHandle(*row),
                    )
                })
                .collect::<Vec<_>>(),
        };
        for entity in tail_entities {
            for property in &property_ids {
                expected_dependencies.insert(ResidentQuantifierEntityReadDependency {
                    entity,
                    property: Some(*property),
                });
            }
        }
    }
    if expected_dependencies.into_iter().collect::<Vec<_>>() != parts.read_dependencies {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "resident quantifier result omitted or invented a source entity/property dependency",
        ));
    }
    for dependency in &parts.read_dependencies {
        if !valid_handle(dependency.entity)
            || dependency.property.is_some_and(|property| {
                !property_ids.contains(&property)
                    || !matches!(
                        (entity_kind, dependency.entity),
                        (
                            ResidentQuantifierEntityKind::Node,
                            ResidentQuantifierEntityHandle::Node(_)
                        ) | (
                            ResidentQuantifierEntityKind::Relationship,
                            ResidentQuantifierEntityHandle::Relationship(_)
                        )
                    )
            })
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident quantifier entity dependency has an invalid handle or property",
            ));
        }
    }
    for output in &parts.outputs {
        let row = usize::try_from(output.row).map_err(|_| scratch_overflow())?;
        let column = usize::from(output.column);
        if row >= rows.len()
            || column >= request.program.outputs.len()
            || rows[row][column] != ResidentQuantifierValue::Null
            || output.value.values.len() > request.max_list_items
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident quantifier entity output has an invalid result coordinate or capacity",
            ));
        }
        for handle in &output.value.values {
            let expected_kind = matches!(
                (entity_kind, handle),
                (
                    ResidentQuantifierEntityKind::Node,
                    ResidentQuantifierEntityHandle::Node(_)
                ) | (
                    ResidentQuantifierEntityKind::Relationship,
                    ResidentQuantifierEntityHandle::Relationship(_)
                )
            );
            let identity = ResidentQuantifierEntityReadDependency {
                entity: *handle,
                property: None,
            };
            if !expected_kind
                || !valid_handle(*handle)
                || parts.read_dependencies.binary_search(&identity).is_err()
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident quantifier entity output contains an invalid or unproven handle",
                ));
            }
        }
    }
    Ok((parts.read_dependencies.len(), parts.source_paths.len()))
}

fn hash_quantifier_obligation(
    hasher: &mut blake3::Hasher,
    obligation: ResidentExecutionObligation,
) {
    hasher.update(&obligation.id.to_le_bytes());
    hasher.update(&[obligation.kind as u8]);
    let (tag, index) = match obligation.scope {
        ResidentObligationScope::Selection => (0_u8, 0_u16),
        ResidentObligationScope::MutationCommand(index) => (1, index),
        ResidentObligationScope::Expression(index) => (2, index),
        ResidentObligationScope::Filter(index) => (3, index),
        ResidentObligationScope::PatternLeaf(index) => (4, index),
        ResidentObligationScope::PatternFinal => (5, 0),
        ResidentObligationScope::PatternScan => (6, 0),
        ResidentObligationScope::PatternScanN => (7, 0),
        ResidentObligationScope::PatternScanM => (8, 0),
        ResidentObligationScope::PatternCartesian => (9, 0),
    };
    hasher.update(&[tag]);
    hasher.update(&index.to_le_bytes());
}

fn hash_quantifier_len(hasher: &mut blake3::Hasher, length: usize) -> Result<()> {
    let length = u64::try_from(length).map_err(|_| scratch_overflow())?;
    hasher.update(&length.to_le_bytes());
    Ok(())
}

fn hash_quantifier_string(hasher: &mut blake3::Hasher, value: &str) -> Result<()> {
    hash_quantifier_len(hasher, value.len())?;
    hasher.update(value.as_bytes());
    Ok(())
}

fn hash_quantifier_value(
    hasher: &mut blake3::Hasher,
    value: &ResidentQuantifierValue,
) -> Result<()> {
    let tag = match value {
        ResidentQuantifierValue::Null => 0,
        ResidentQuantifierValue::Boolean(_) => 1,
        ResidentQuantifierValue::Integer(_) => 2,
        ResidentQuantifierValue::Float(_) => 3,
        ResidentQuantifierValue::String(_) => 4,
        ResidentQuantifierValue::List(_) => 5,
        ResidentQuantifierValue::Map(_) => 6,
    };
    hasher.update(&[tag]);
    match value {
        ResidentQuantifierValue::Null => {}
        ResidentQuantifierValue::Boolean(value) => {
            hasher.update(&[u8::from(*value)]);
        }
        ResidentQuantifierValue::Integer(value) => {
            hasher.update(&value.to_le_bytes());
        }
        ResidentQuantifierValue::Float(value) => {
            hasher.update(&value.to_le_bytes());
        }
        ResidentQuantifierValue::String(value) => hash_quantifier_string(hasher, value)?,
        ResidentQuantifierValue::List(values) => {
            hash_quantifier_len(hasher, values.len())?;
            for value in values {
                hash_quantifier_value(hasher, value)?;
            }
        }
        ResidentQuantifierValue::Map(entries) => {
            hash_quantifier_len(hasher, entries.len())?;
            for (key, value) in entries {
                hash_quantifier_string(hasher, key)?;
                hash_quantifier_value(hasher, value)?;
            }
        }
    }
    Ok(())
}

fn hash_quantifier_expression(
    hasher: &mut blake3::Hasher,
    expression: &ResidentQuantifierExpression,
) -> Result<()> {
    match expression {
        ResidentQuantifierExpression::Slot(slot) => {
            hasher.update(&[0]);
            hasher.update(&slot.0.to_le_bytes());
        }
        ResidentQuantifierExpression::Literal(value) => {
            hasher.update(&[1]);
            hash_quantifier_value(hasher, value)?;
        }
        ResidentQuantifierExpression::Property { source, key } => {
            hasher.update(&[2]);
            hash_quantifier_expression(hasher, source)?;
            hash_quantifier_string(hasher, key)?;
        }
        ResidentQuantifierExpression::List(values) => {
            hasher.update(&[3]);
            hash_quantifier_len(hasher, values.len())?;
            for value in values {
                hash_quantifier_expression(hasher, value)?;
            }
        }
        ResidentQuantifierExpression::Map(entries) => {
            hasher.update(&[4]);
            hash_quantifier_len(hasher, entries.len())?;
            for (key, value) in entries {
                hash_quantifier_string(hasher, key)?;
                hash_quantifier_expression(hasher, value)?;
            }
        }
        ResidentQuantifierExpression::Case {
            operand,
            alternatives,
            default,
        } => {
            hasher.update(&[5, u8::from(operand.is_some()), u8::from(default.is_some())]);
            if let Some(operand) = operand {
                hash_quantifier_expression(hasher, operand)?;
            }
            hash_quantifier_len(hasher, alternatives.len())?;
            for (when, then) in alternatives {
                hash_quantifier_expression(hasher, when)?;
                hash_quantifier_expression(hasher, then)?;
            }
            if let Some(default) = default {
                hash_quantifier_expression(hasher, default)?;
            }
        }
        ResidentQuantifierExpression::ListComprehension {
            variable,
            list,
            predicate,
            projection,
        } => {
            hasher.update(&[
                6,
                u8::from(predicate.is_some()),
                u8::from(projection.is_some()),
            ]);
            hasher.update(&variable.0.to_le_bytes());
            hash_quantifier_expression(hasher, list)?;
            if let Some(predicate) = predicate {
                hash_quantifier_expression(hasher, predicate)?;
            }
            if let Some(projection) = projection {
                hash_quantifier_expression(hasher, projection)?;
            }
        }
        ResidentQuantifierExpression::Predicate {
            kind,
            variable,
            list,
            predicate,
        } => {
            hasher.update(&[7, *kind as u8]);
            hasher.update(&variable.0.to_le_bytes());
            hash_quantifier_expression(hasher, list)?;
            hash_quantifier_expression(hasher, predicate)?;
        }
        ResidentQuantifierExpression::Function {
            function,
            arguments,
        } => {
            hasher.update(&[8, *function as u8]);
            hash_quantifier_len(hasher, arguments.len())?;
            for argument in arguments {
                hash_quantifier_expression(hasher, argument)?;
            }
        }
        ResidentQuantifierExpression::Unary { operation, operand } => {
            hasher.update(&[9, *operation as u8]);
            hash_quantifier_expression(hasher, operand)?;
        }
        ResidentQuantifierExpression::Binary {
            left,
            operation,
            right,
        } => {
            hasher.update(&[10, *operation as u8]);
            hash_quantifier_expression(hasher, left)?;
            hash_quantifier_expression(hasher, right)?;
        }
        ResidentQuantifierExpression::IsNull {
            expression,
            negated,
        } => {
            hasher.update(&[11, u8::from(*negated)]);
            hash_quantifier_expression(hasher, expression)?;
        }
    }
    Ok(())
}

fn hash_quantifier_program(
    hasher: &mut blake3::Hasher,
    program: &ResidentQuantifierProgram,
) -> Result<()> {
    hash_quantifier_len(hasher, program.stages.len())?;
    for stage in &program.stages {
        match stage {
            ResidentQuantifierStage::Project {
                keep_scope,
                bindings,
            } => {
                hasher.update(&[0, u8::from(*keep_scope)]);
                hash_quantifier_len(hasher, bindings.len())?;
                for binding in bindings {
                    hasher.update(&binding.output.0.to_le_bytes());
                    hash_quantifier_expression(hasher, &binding.expression)?;
                }
            }
            ResidentQuantifierStage::Unwind { expression, output } => {
                hasher.update(&[1]);
                hasher.update(&output.0.to_le_bytes());
                hash_quantifier_expression(hasher, expression)?;
            }
            ResidentQuantifierStage::Filter { predicate } => {
                hasher.update(&[2]);
                hash_quantifier_expression(hasher, predicate)?;
            }
            ResidentQuantifierStage::GroupCount {
                groups,
                count_outputs,
            } => {
                hasher.update(&[3]);
                hash_quantifier_len(hasher, groups.len())?;
                for group in groups {
                    hasher.update(&group.output.0.to_le_bytes());
                    hash_quantifier_expression(hasher, &group.expression)?;
                }
                hash_quantifier_len(hasher, count_outputs.len())?;
                for output in count_outputs {
                    hasher.update(&output.0.to_le_bytes());
                }
            }
        }
    }
    hash_quantifier_len(hasher, program.outputs.len())?;
    for output in &program.outputs {
        hash_quantifier_string(hasher, &output.name)?;
        hasher.update(&output.source.0.to_le_bytes());
    }
    Ok(())
}

/// Arbitrary compiler-assigned node or relationship binding slot in one staged relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResidentNullableRelationSlot(pub u16);

/// Physical graph-row domain owned by a staged relation slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ResidentNullableRelationBindingKind {
    Node = 1,
    Relationship = 2,
}

/// Mandatory MATCH drops a row whose bound input is null or has no candidate. OPTIONAL MATCH
/// preserves that row once and fills only newly introduced bindings with the null sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ResidentNullableRelationMatchMode {
    Mandatory = 1,
    Optional = 2,
}

/// Resolved node-label domain. `KnownEmpty` means that at least one required label is absent from
/// the exact catalog generation. It is a valid empty graph domain, not an unsupported query.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidentNullableNodeDomain {
    Any,
    Known(Vec<LabelId>),
    KnownEmpty,
}

impl ResidentNullableNodeDomain {
    #[must_use]
    pub const fn is_known_empty(&self) -> bool {
        matches!(self, Self::KnownEmpty)
    }

    fn validate(&self) -> Result<()> {
        if let Self::Known(labels) = self
            && (labels.is_empty() || labels.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable node domain must use sorted unique known labels or KnownEmpty",
            ));
        }
        Ok(())
    }
}

/// Resolved relationship-type domain. As with node labels, `KnownEmpty` is executable and
/// deterministically produces no graph candidates.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidentNullableRelationshipDomain {
    Any,
    Known(Vec<RelationshipTypeId>),
    KnownEmpty,
}

impl ResidentNullableRelationshipDomain {
    #[must_use]
    pub const fn is_known_empty(&self) -> bool {
        matches!(self, Self::KnownEmpty)
    }

    fn validate(&self) -> Result<()> {
        if let Self::Known(types) = self
            && (types.is_empty() || types.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relationship domain must use sorted unique known types or KnownEmpty",
            ));
        }
        Ok(())
    }
}

/// Canonical domain of one parser-produced `entity:Name[:Name...]` final expression.
///
/// Node names are a conjunctive label set. A relationship has exactly one type, so a satisfiable
/// relationship conjunction contains exactly one resolved type; repeated equal names collapse to
/// that type, while an absent name or two distinct names compile to `KnownEmpty`. `Any` is never a
/// valid colon expression because the grammar requires at least one name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidentNullableRelationEntityLabelDomain {
    Node(ResidentNullableNodeDomain),
    Relationship(ResidentNullableRelationshipDomain),
}

impl ResidentNullableRelationEntityLabelDomain {
    #[must_use]
    pub const fn kind(&self) -> ResidentNullableRelationBindingKind {
        match self {
            Self::Node(_) => ResidentNullableRelationBindingKind::Node,
            Self::Relationship(_) => ResidentNullableRelationBindingKind::Relationship,
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Node(domain) => {
                domain.validate()?;
                if matches!(domain, ResidentNullableNodeDomain::Any) {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable node label output requires at least one exact name",
                    ));
                }
            }
            Self::Relationship(domain) => {
                domain.validate()?;
                if matches!(domain, ResidentNullableRelationshipDomain::Any)
                    || matches!(domain, ResidentNullableRelationshipDomain::Known(types) if types.len() != 1)
                {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable relationship type-predicate output requires one exact type or KnownEmpty",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// An expansion endpoint is either an already-live node binding or one newly introduced column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidentNullableRelationTarget {
    Existing(ResidentNullableRelationSlot),
    Introduce(ResidentNullableRelationSlot),
}

/// One typed scalar source consumed only by a nullable-relation predicate. Entity bindings retain
/// their physical kind so validation and the backend cannot reinterpret a node row as a
/// relationship row. A missing property is compiled as `Null`; no host property evaluation is
/// permitted after native admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentNullableRelationPredicateValue {
    Null,
    Boolean(bool),
    Integer(i64),
    String(Arc<str>),
    Binding {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
    },
    IntegerProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
    },
    StringProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
    },
}

/// Canonical scalar tensor shape for one graph property consumed by nullable predicates. The
/// shape is sealed independently of the predicate tree so a backend cannot guess which resident
/// tensor family a property ID belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ResidentNullableRelationPropertyShape {
    Integer = 1,
    String = 2,
    Float = 3,
}

/// One physical graph-property tensor required by the complete predicate and final-output program. Binding slots
/// deliberately do not appear here: several staged-relation columns may gather through the same
/// immutable node or relationship tensor. Missing values remain ordinary Cypher nulls through the
/// tensor validity lane; a missing registry entry is an invalid native command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentNullableRelationPropertyLane {
    pub kind: ResidentNullableRelationBindingKind,
    pub property: PropertyId,
    pub shape: ResidentNullableRelationPropertyShape,
}

/// Deterministic, deduplicated registry of every graph-property tensor referenced by one nullable
/// predicate or final projection. Construction is owned by [`ResidentNullableRelationRequest`];
/// callers may inspect the sealed order but cannot provide an independently invented registry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResidentNullableRelationPropertyLaneRegistry {
    lanes: Vec<ResidentNullableRelationPropertyLane>,
}

impl ResidentNullableRelationPropertyLaneRegistry {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    #[must_use]
    pub fn lanes(&self) -> &[ResidentNullableRelationPropertyLane] {
        &self.lanes
    }
}

/// Endpoint of a live relationship used to seed an otherwise unbound OPTIONAL pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ResidentNullableRelationshipEndpoint {
    Source = 1,
    Target = 2,
    Either = 3,
}

/// Bounded, backend-neutral Cypher predicate tree. Backends evaluate this with exact three-valued
/// logic: only `True` survives a filter; `False` and `Null` are counted separately in the filter
/// receipt. The deliberately small value/type surface is fail-closed and can be extended without
/// allowing the generic expression evaluator into a native command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentNullableRelationPredicate {
    Constant(Option<bool>),
    IsNull {
        value: ResidentNullableRelationPredicateValue,
        negated: bool,
    },
    HasLabels {
        node: ResidentNullableRelationSlot,
        labels: ResidentNullableNodeDomain,
    },
    CompareInteger {
        left: ResidentNullableRelationPredicateValue,
        operation: CompareOp,
        right: ResidentNullableRelationPredicateValue,
    },
    CompareString {
        left: ResidentNullableRelationPredicateValue,
        operation: CompareOp,
        right: ResidentNullableRelationPredicateValue,
    },
    RelationshipEndpoint {
        relationship: ResidentNullableRelationSlot,
        node: ResidentNullableRelationSlot,
        endpoint: ResidentNullableRelationshipEndpoint,
    },
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

/// Exact semantic placement of a predicate relative to the immutable graph stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidentNullableRelationFilterPlacement {
    /// Filters a materialized relation after `stage`; it changes the input cardinality observed by
    /// the following graph or projection stage. This owns ordinary WHERE and post-WITH WHERE.
    RelationAfter { stage: u16 },
    /// Filters graph candidates produced by one OPTIONAL stage before that stage decides whether
    /// to null-extend its parent row.
    OptionalCandidates { stage: u16 },
    /// Filters complete paths for one multi-hop OPTIONAL group before atomic null extension.
    OptionalGroupCandidates { group: u16 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationFilterStage {
    pub placement: ResidentNullableRelationFilterPlacement,
    pub predicate: ResidentNullableRelationPredicate,
}

/// A contiguous multi-hop OPTIONAL pattern. Individual expansion stages may materialize partial
/// candidates internally, but this group boundary retains only complete paths. If no complete
/// candidate survives for a parent, it emits exactly one parent row with every group-introduced
/// binding null. Groups are non-overlapping and contain at least two expansion stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentNullableRelationOptionalGroup {
    pub first_stage: u16,
    pub last_stage: u16,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResidentNullableRelationPredicateProgram {
    pub filters: Vec<ResidentNullableRelationFilterStage>,
    pub optional_groups: Vec<ResidentNullableRelationOptionalGroup>,
}

impl ResidentNullableRelationPredicateProgram {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty() && self.optional_groups.is_empty()
    }

    #[must_use]
    pub fn requires_relationship_endpoint_seed(&self) -> bool {
        self.filters
            .iter()
            .any(|filter| filter.predicate.contains_relationship_endpoint_seed())
    }

    #[must_use]
    pub fn requires_string_property_equality(&self) -> bool {
        self.filters
            .iter()
            .any(|filter| filter.predicate.contains_string_property_equality())
    }
}

/// One entity-only WITH projection. `output` may preserve the same physical slot or name a fresh
/// slot; compilers normally allocate a fresh slot so alias/scope boundaries are explicit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationProjectionBinding {
    pub variable: String,
    pub source: ResidentNullableRelationSlot,
    pub output: ResidentNullableRelationSlot,
    /// Stable prefix retained by the WITH boundary before the following stage. `None` preserves
    /// every input row. Every binding in one scope projection must carry the same value.
    pub row_limit: Option<u64>,
}

/// Sealed source of one client-visible nullable-relation output column. Property sources retain
/// the dense entity row which owns the value; a backend must gather the scalar while executing the
/// native command and publish that source row beside the gathered payload. `NullProperty` is the
/// executable result of compiling a property absent from the exact catalog generation. It never
/// binds a property tensor and always publishes Cypher null, while retaining the source rows needed
/// to prove row alignment and null propagation. `RelationshipType` gathers the canonical
/// relationship-type token on the selected backend and retains the source relationship row so
/// publication can prove that the token belongs to that exact edge and catalog generation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidentNullableRelationOutputSource {
    Entity {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
    },
    IntegerProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
    },
    StringPropertyToInteger {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
    },
    FloatProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
    },
    IntegerPropertyToFloat {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
    },
    StringProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
        /// Exact maximum UTF-8 width in this immutable property tensor at the sealed generation.
        maximum_bytes: u32,
    },
    IntegerPropertyToString {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
    },
    NullProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
    },
    RelationshipType {
        slot: ResidentNullableRelationSlot,
    },
    /// Three-valued parser-native `entity:Name[:Name...]` output. The selected backend publishes
    /// Boolean bytes and validity while retaining the exact source entity row for provenance and
    /// null propagation.
    EntityLabelPredicate {
        slot: ResidentNullableRelationSlot,
        domain: ResidentNullableRelationEntityLabelDomain,
    },
}

impl ResidentNullableRelationOutputSource {
    #[must_use]
    pub const fn slot(&self) -> ResidentNullableRelationSlot {
        match self {
            Self::Entity { slot, .. }
            | Self::IntegerProperty { slot, .. }
            | Self::StringPropertyToInteger { slot, .. }
            | Self::FloatProperty { slot, .. }
            | Self::IntegerPropertyToFloat { slot, .. }
            | Self::StringProperty { slot, .. }
            | Self::IntegerPropertyToString { slot, .. }
            | Self::NullProperty { slot, .. }
            | Self::RelationshipType { slot }
            | Self::EntityLabelPredicate { slot, .. } => *slot,
        }
    }

    #[must_use]
    pub const fn kind(&self) -> ResidentNullableRelationBindingKind {
        match self {
            Self::Entity { kind, .. }
            | Self::IntegerProperty { kind, .. }
            | Self::StringPropertyToInteger { kind, .. }
            | Self::FloatProperty { kind, .. }
            | Self::IntegerPropertyToFloat { kind, .. }
            | Self::StringProperty { kind, .. }
            | Self::IntegerPropertyToString { kind, .. }
            | Self::NullProperty { kind, .. } => *kind,
            Self::RelationshipType { .. } => ResidentNullableRelationBindingKind::Relationship,
            Self::EntityLabelPredicate { domain, .. } => domain.kind(),
        }
    }

    pub const fn property_lane(&self) -> Option<ResidentNullableRelationPropertyLane> {
        match self {
            Self::IntegerProperty { kind, property, .. } => {
                Some(ResidentNullableRelationPropertyLane {
                    kind: *kind,
                    property: *property,
                    shape: ResidentNullableRelationPropertyShape::Integer,
                })
            }
            Self::StringPropertyToInteger { kind, property, .. } => {
                Some(ResidentNullableRelationPropertyLane {
                    kind: *kind,
                    property: *property,
                    shape: ResidentNullableRelationPropertyShape::String,
                })
            }
            Self::FloatProperty { kind, property, .. } => {
                Some(ResidentNullableRelationPropertyLane {
                    kind: *kind,
                    property: *property,
                    shape: ResidentNullableRelationPropertyShape::Float,
                })
            }
            Self::IntegerPropertyToFloat { kind, property, .. }
            | Self::IntegerPropertyToString { kind, property, .. } => {
                Some(ResidentNullableRelationPropertyLane {
                    kind: *kind,
                    property: *property,
                    shape: ResidentNullableRelationPropertyShape::Integer,
                })
            }
            Self::StringProperty { kind, property, .. } => {
                Some(ResidentNullableRelationPropertyLane {
                    kind: *kind,
                    property: *property,
                    shape: ResidentNullableRelationPropertyShape::String,
                })
            }
            Self::Entity { .. }
            | Self::NullProperty { .. }
            | Self::RelationshipType { .. }
            | Self::EntityLabelPredicate { .. } => None,
        }
    }

    fn validate_against(
        &self,
        live: &BTreeMap<ResidentNullableRelationSlot, ResidentNullableRelationBindingState>,
    ) -> Result<()> {
        let Some(state) = live.get(&self.slot()) else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable final projection names an absent source slot",
            ));
        };
        if state.kind != self.kind() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "resident nullable final projection source kind disagrees with the staged relation",
            ));
        }
        if let Self::EntityLabelPredicate { domain, .. } = self {
            domain.validate()?;
        }
        Ok(())
    }

    fn output_packet_bytes(&self) -> Result<(u64, u64)> {
        let source_row = size_of::<u32>() as u64;
        let (payload_per_row, fixed_bytes) = match self {
            Self::Entity { .. } | Self::NullProperty { .. } => (0, 0),
            Self::RelationshipType { .. } => (size_of::<u64>() as u64, 0),
            Self::EntityLabelPredicate { .. } => ((size_of::<u8>() + size_of::<u8>()) as u64, 0),
            Self::IntegerProperty { .. }
            | Self::StringPropertyToInteger { .. }
            | Self::FloatProperty { .. }
            | Self::IntegerPropertyToFloat { .. } => {
                ((size_of::<u64>() + size_of::<u8>()) as u64, 0)
            }
            Self::IntegerPropertyToString { .. } => (
                u64::from(RESIDENT_NULLABLE_RELATION_INTEGER_STRING_MAXIMUM_BYTES)
                    .checked_add((size_of::<u32>() + size_of::<u8>()) as u64)
                    .ok_or_else(nullable_relation_scratch_overflow)?,
                size_of::<u32>() as u64,
            ),
            Self::StringProperty { maximum_bytes, .. } => (
                u64::from(*maximum_bytes)
                    .checked_add((size_of::<u32>() + size_of::<u8>()) as u64)
                    .ok_or_else(nullable_relation_scratch_overflow)?,
                size_of::<u32>() as u64,
            ),
        };
        Ok((
            source_row
                .checked_add(payload_per_row)
                .ok_or_else(nullable_relation_scratch_overflow)?,
            fixed_bytes,
        ))
    }
}

/// One client-visible column in the final projection boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationOutputBinding {
    pub name: String,
    pub source: ResidentNullableRelationOutputSource,
}

/// One immutable relational stage. Every graph stage consumes the prior relation as a whole.
/// Node scans form a row-major product with their input relation. Expansions are correlated to
/// `source`; mandatory expansions never dereference a null source and optional expansions preserve
/// a null/no-candidate input once with introduced columns set to the null sentinel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentNullableRelationStage {
    NodeScan {
        mode: ResidentNullableRelationMatchMode,
        output: ResidentNullableRelationSlot,
        labels: ResidentNullableNodeDomain,
    },
    Expand {
        mode: ResidentNullableRelationMatchMode,
        uniqueness_group: u32,
        source: ResidentNullableRelationSlot,
        source_labels: ResidentNullableNodeDomain,
        relationship: Option<ResidentNullableRelationSlot>,
        different_from: Vec<ResidentNullableRelationSlot>,
        target: ResidentNullableRelationTarget,
        direction: ResidentDirection,
        relationship_types: ResidentNullableRelationshipDomain,
        target_labels: ResidentNullableNodeDomain,
    },
    ScopeProject {
        bindings: Vec<ResidentNullableRelationProjectionBinding>,
    },
    FinalProject {
        bindings: Vec<ResidentNullableRelationOutputBinding>,
    },
}

impl ResidentNullableRelationStage {
    #[must_use]
    pub const fn match_mode(&self) -> Option<ResidentNullableRelationMatchMode> {
        match self {
            Self::NodeScan { mode, .. } | Self::Expand { mode, .. } => Some(*mode),
            Self::ScopeProject { .. } | Self::FinalProject { .. } => None,
        }
    }

    #[must_use]
    pub const fn has_known_empty_candidate_domain(&self) -> bool {
        match self {
            Self::NodeScan { labels, .. } => labels.is_known_empty(),
            Self::Expand {
                source_labels,
                relationship_types,
                target_labels,
                ..
            } => {
                source_labels.is_known_empty()
                    || relationship_types.is_known_empty()
                    || target_labels.is_known_empty()
            }
            Self::ScopeProject { .. } | Self::FinalProject { .. } => false,
        }
    }
}

/// Complete immutable fixed-length nullable relation program. The unit relation (one row, zero
/// columns) is the implicit input to stage zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationProgram {
    pub layers: LayerMask,
    pub stages: Vec<ResidentNullableRelationStage>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentNullableRelationNullability {
    Never,
    Maybe,
    Always,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentNullableRelationBindingState {
    kind: ResidentNullableRelationBindingKind,
    nullability: ResidentNullableRelationNullability,
}

#[derive(Clone, Debug)]
struct ResidentNullableRelationAnalysis {
    binding_slot_count: usize,
    maximum_live_bindings: usize,
    requires_stable_scope_limit: bool,
    requires_existing_relationship: bool,
    stage_live_bindings: Vec<usize>,
    stage_states: Vec<BTreeMap<ResidentNullableRelationSlot, ResidentNullableRelationBindingState>>,
    final_bindings: Vec<ResidentNullableRelationOutputBinding>,
    final_states: BTreeMap<ResidentNullableRelationSlot, ResidentNullableRelationBindingState>,
}

impl ResidentNullableRelationProgram {
    pub fn validate(&self) -> Result<()> {
        self.analyze().map(|_| ())
    }

    /// Returns whether the immutable stage prefix proves that one live binding is always Cypher
    /// null. Compilers use this only for semantics-preserving elimination of graph work whose
    /// required endpoint cannot exist; it is derived by the same validator that later checks
    /// backend receipts, not by catalog or cardinality guesswork.
    pub fn binding_is_statically_always_null(
        &self,
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
    ) -> Result<bool> {
        let analysis = self.analyze()?;
        let state = analysis.final_states.get(&slot).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable static-null proof names a binding outside final scope",
            )
        })?;
        if state.kind != kind {
            return Err(Error::new(
                ErrorCode::QueryType,
                "resident nullable static-null proof uses the wrong binding kind",
            ));
        }
        Ok(state.nullability == ResidentNullableRelationNullability::Always)
    }

    /// Returns whether the immutable stage prefix proves that one live binding can never be
    /// Cypher null. This is deliberately the dual of `binding_is_statically_always_null`: native
    /// compilers use it only when an algebraic rewrite would otherwise change `collect`, which
    /// drops null values before a later `UNWIND`.
    pub fn binding_is_statically_never_null(
        &self,
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
    ) -> Result<bool> {
        let analysis = self.analyze()?;
        let state = analysis.final_states.get(&slot).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable static-non-null proof names a binding outside final scope",
            )
        })?;
        if state.kind != kind {
            return Err(Error::new(
                ErrorCode::QueryType,
                "resident nullable static-non-null proof uses the wrong binding kind",
            ));
        }
        Ok(state.nullability == ResidentNullableRelationNullability::Never)
    }

    fn analyze(&self) -> Result<ResidentNullableRelationAnalysis> {
        self.analyze_with_receipts(None)
    }

    pub fn requires_stable_scope_limit(&self) -> Result<bool> {
        Ok(self.analyze()?.requires_stable_scope_limit)
    }

    pub fn requires_existing_relationship(&self) -> Result<bool> {
        Ok(self.analyze()?.requires_existing_relationship)
    }

    fn analyze_with_receipts(
        &self,
        receipts: Option<&[ResidentNullableRelationReceipt]>,
    ) -> Result<ResidentNullableRelationAnalysis> {
        if self.layers.is_empty() || !LayerMask::ALL.contains(self.layers) {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation requires a valid non-empty layer mask",
            ));
        }
        if self.stages.is_empty() || self.stages.len() > RESIDENT_NULLABLE_RELATION_MAX_STAGES {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation stage count is outside its bounded contract",
            ));
        }
        if receipts.is_some_and(|receipts| receipts.len() != self.stages.len()) {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident nullable relation receipt analysis does not cover every stage",
            ));
        }

        let mut live =
            BTreeMap::<ResidentNullableRelationSlot, ResidentNullableRelationBindingState>::new();
        let mut allocated = BTreeSet::new();
        let mut maximum_live_bindings = 0_usize;
        let mut requires_stable_scope_limit = false;
        let mut requires_existing_relationship = false;
        let mut current_uniqueness_group = None;
        let mut group_relationships = BTreeSet::new();
        let mut group_expansion_count = 0_usize;
        let mut group_has_unretained_relationship = false;
        let mut stage_live_bindings = Vec::with_capacity(self.stages.len());
        let mut stage_states = Vec::with_capacity(self.stages.len());
        let mut final_bindings = None;

        let allocate = |slot: ResidentNullableRelationSlot,
                        kind: ResidentNullableRelationBindingKind,
                        nullability: ResidentNullableRelationNullability,
                        live: &mut BTreeMap<_, _>,
                        allocated: &mut BTreeSet<_>|
         -> Result<()> {
            if usize::from(slot.0) >= RESIDENT_NULLABLE_RELATION_MAX_BINDINGS
                || !allocated.insert(slot)
                || live
                    .insert(
                        slot,
                        ResidentNullableRelationBindingState { kind, nullability },
                    )
                    .is_some()
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident nullable relation reuses or exceeds a binding slot",
                ));
            }
            Ok(())
        };

        for (stage_index, stage) in self.stages.iter().enumerate() {
            if final_bindings.is_some() {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident nullable relation has work after its final projection",
                ));
            }
            match stage {
                ResidentNullableRelationStage::NodeScan {
                    mode,
                    output,
                    labels,
                } => {
                    labels.validate()?;
                    let candidate_cardinality = receipts
                        .and_then(|receipts| receipts.get(stage_index))
                        .map(|receipt| receipt.candidate_cardinality);
                    let nullability = match (mode, labels.is_known_empty(), candidate_cardinality) {
                        (ResidentNullableRelationMatchMode::Mandatory, _, _) => {
                            ResidentNullableRelationNullability::Never
                        }
                        (ResidentNullableRelationMatchMode::Optional, true, _)
                        | (ResidentNullableRelationMatchMode::Optional, false, Some(0)) => {
                            ResidentNullableRelationNullability::Always
                        }
                        (ResidentNullableRelationMatchMode::Optional, false, Some(_)) => {
                            // An uncorrelated node domain is shared by every input row. Once one
                            // candidate exists, every surviving output has a non-null scan slot.
                            ResidentNullableRelationNullability::Never
                        }
                        (ResidentNullableRelationMatchMode::Optional, false, None) => {
                            ResidentNullableRelationNullability::Maybe
                        }
                    };
                    allocate(
                        *output,
                        ResidentNullableRelationBindingKind::Node,
                        nullability,
                        &mut live,
                        &mut allocated,
                    )?;
                }
                ResidentNullableRelationStage::Expand {
                    mode,
                    uniqueness_group,
                    source,
                    source_labels,
                    relationship,
                    different_from,
                    target,
                    relationship_types,
                    target_labels,
                    ..
                } => {
                    source_labels.validate()?;
                    relationship_types.validate()?;
                    target_labels.validate()?;
                    match current_uniqueness_group {
                        Some(current) if *uniqueness_group < current => {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident nullable relationship uniqueness group reopens out of order",
                            ));
                        }
                        Some(current) if *uniqueness_group > current => {
                            current_uniqueness_group = Some(*uniqueness_group);
                            group_relationships.clear();
                            group_expansion_count = 0;
                            group_has_unretained_relationship = false;
                        }
                        None => current_uniqueness_group = Some(*uniqueness_group),
                        Some(_) => {}
                    }
                    let expected_different_from =
                        group_relationships.iter().copied().collect::<Vec<_>>();
                    if *different_from != expected_different_from {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable expansion omits or forges its exact DifferentRelationships exclusions",
                        ));
                    }
                    for excluded in different_from {
                        if live.get(excluded).is_none_or(|state| {
                            state.kind != ResidentNullableRelationBindingKind::Relationship
                        }) {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident nullable DifferentRelationships exclusion is not a live relationship slot",
                            ));
                        }
                    }
                    if group_expansion_count != 0
                        && (group_has_unretained_relationship || relationship.is_none())
                    {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable multi-expansion uniqueness group omits an anonymous relationship slot",
                        ));
                    }
                    let Some(source_state) = live.get(source).copied() else {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable expansion source is outside the live scope",
                        ));
                    };
                    if source_state.kind != ResidentNullableRelationBindingKind::Node {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident nullable expansion source is not a node binding",
                        ));
                    }
                    let empty = source_labels.is_known_empty()
                        || relationship_types.is_known_empty()
                        || target_labels.is_known_empty();
                    let stage_receipt = receipts.and_then(|receipts| receipts.get(stage_index));
                    let mut required_endpoint_is_always_null =
                        source_state.nullability == ResidentNullableRelationNullability::Always;
                    let introduced_nullability = match (mode, empty, stage_receipt) {
                        (ResidentNullableRelationMatchMode::Mandatory, _, _) => {
                            ResidentNullableRelationNullability::Never
                        }
                        (ResidentNullableRelationMatchMode::Optional, true, _)
                        | (
                            ResidentNullableRelationMatchMode::Optional,
                            false,
                            Some(ResidentNullableRelationReceipt {
                                candidate_cardinality: 0,
                                ..
                            }),
                        ) => ResidentNullableRelationNullability::Always,
                        (
                            ResidentNullableRelationMatchMode::Optional,
                            false,
                            Some(ResidentNullableRelationReceipt {
                                null_extension_cardinality: 0,
                                ..
                            }),
                        ) => ResidentNullableRelationNullability::Never,
                        (ResidentNullableRelationMatchMode::Optional, false, _) => {
                            ResidentNullableRelationNullability::Maybe
                        }
                    };
                    if let Some(relationship) = relationship {
                        if let Some(state) = live.get_mut(relationship) {
                            if state.kind != ResidentNullableRelationBindingKind::Relationship {
                                return Err(Error::new(
                                    ErrorCode::QueryType,
                                    "resident nullable expansion relationship slot is not a relationship binding",
                                ));
                            }
                            requires_existing_relationship = true;
                            required_endpoint_is_always_null |=
                                state.nullability == ResidentNullableRelationNullability::Always;
                            if *mode == ResidentNullableRelationMatchMode::Mandatory {
                                state.nullability = ResidentNullableRelationNullability::Never;
                            }
                        } else {
                            allocate(
                                *relationship,
                                ResidentNullableRelationBindingKind::Relationship,
                                introduced_nullability,
                                &mut live,
                                &mut allocated,
                            )?;
                        }
                        group_relationships.insert(*relationship);
                    } else {
                        group_has_unretained_relationship = true;
                    }
                    group_expansion_count = group_expansion_count.saturating_add(1);
                    match target {
                        ResidentNullableRelationTarget::Existing(target) => {
                            let Some(target_state) = live.get(target).copied() else {
                                return Err(Error::new(
                                    ErrorCode::GpuAdmissionFailure,
                                    "resident nullable expansion target is outside the live scope",
                                ));
                            };
                            if target_state.kind != ResidentNullableRelationBindingKind::Node {
                                return Err(Error::new(
                                    ErrorCode::QueryType,
                                    "resident nullable expansion target is not a node binding",
                                ));
                            }
                            required_endpoint_is_always_null |= target_state.nullability
                                == ResidentNullableRelationNullability::Always;
                            if *mode == ResidentNullableRelationMatchMode::Mandatory {
                                live.get_mut(target)
                                    .ok_or_else(|| {
                                        Error::internal(
                                            "validated live relation target disappeared",
                                        )
                                    })?
                                    .nullability = ResidentNullableRelationNullability::Never;
                            }
                        }
                        ResidentNullableRelationTarget::Introduce(target) => {
                            allocate(
                                *target,
                                ResidentNullableRelationBindingKind::Node,
                                introduced_nullability,
                                &mut live,
                                &mut allocated,
                            )?;
                        }
                    }
                    if required_endpoint_is_always_null
                        && stage_receipt.is_some_and(|receipt| {
                            receipt.candidate_cardinality != 0
                                || (*mode == ResidentNullableRelationMatchMode::Mandatory
                                    && receipt.output_cardinality != 0)
                        })
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "resident nullable expansion claimed a candidate from an always-null required endpoint",
                        ));
                    }
                    if *mode == ResidentNullableRelationMatchMode::Mandatory {
                        live.get_mut(source)
                            .ok_or_else(|| {
                                Error::internal("validated live relation source disappeared")
                            })?
                            .nullability = ResidentNullableRelationNullability::Never;
                    }
                }
                ResidentNullableRelationStage::ScopeProject { bindings } => {
                    if bindings.is_empty()
                        || bindings.len() > RESIDENT_NULLABLE_RELATION_MAX_BINDINGS
                    {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable scope projection is empty or too wide",
                        ));
                    }
                    let prior = live.clone();
                    let mut projected = BTreeMap::new();
                    let mut variables = BTreeSet::new();
                    let row_limit = bindings[0].row_limit;
                    for binding in bindings {
                        if binding.row_limit != row_limit {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident nullable scope projection has inconsistent stable limits",
                            ));
                        }
                        if binding.variable.is_empty() || !variables.insert(&binding.variable) {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident nullable scope projection has an empty or duplicate variable",
                            ));
                        }
                        let Some(state) = prior.get(&binding.source).copied() else {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident nullable scope projection names an absent source slot",
                            ));
                        };
                        if binding.output != binding.source && !allocated.insert(binding.output) {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident nullable scope projection reuses an allocated output slot",
                            ));
                        }
                        if usize::from(binding.output.0) >= RESIDENT_NULLABLE_RELATION_MAX_BINDINGS
                            || projected.insert(binding.output, state).is_some()
                        {
                            return Err(Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident nullable scope projection has an invalid duplicate output slot",
                            ));
                        }
                    }
                    requires_stable_scope_limit |= row_limit.is_some();
                    live = projected;
                }
                ResidentNullableRelationStage::FinalProject { bindings } => {
                    if bindings.is_empty()
                        || bindings.len() > RESIDENT_NULLABLE_RELATION_MAX_OUTPUTS
                    {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable final projection is empty or too wide",
                        ));
                    }
                    for binding in bindings {
                        binding.source.validate_against(&live)?;
                    }
                    final_bindings = Some(bindings.clone());
                }
            }
            maximum_live_bindings = maximum_live_bindings.max(live.len());
            stage_live_bindings.push(live.len());
            stage_states.push(live.clone());
            if stage_index >= u16::MAX as usize {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident nullable relation stage index exceeds u16",
                ));
            }
        }

        let Some(final_bindings) = final_bindings else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation omitted its final projection",
            ));
        };
        let binding_slot_count = allocated
            .iter()
            .next_back()
            .map_or(0_usize, |slot| usize::from(slot.0).saturating_add(1));
        Ok(ResidentNullableRelationAnalysis {
            binding_slot_count,
            maximum_live_bindings,
            requires_stable_scope_limit,
            requires_existing_relationship,
            stage_live_bindings,
            stage_states,
            final_bindings,
            final_states: live,
        })
    }
}

impl ResidentNullableRelationPredicateValue {
    pub fn property_lane(&self) -> Option<ResidentNullableRelationPropertyLane> {
        match self {
            Self::IntegerProperty { kind, property, .. } => {
                Some(ResidentNullableRelationPropertyLane {
                    kind: *kind,
                    property: *property,
                    shape: ResidentNullableRelationPropertyShape::Integer,
                })
            }
            Self::StringProperty { kind, property, .. } => {
                Some(ResidentNullableRelationPropertyLane {
                    kind: *kind,
                    property: *property,
                    shape: ResidentNullableRelationPropertyShape::String,
                })
            }
            Self::Null
            | Self::Boolean(_)
            | Self::Integer(_)
            | Self::String(_)
            | Self::Binding { .. } => None,
        }
    }

    fn validate_against(
        &self,
        live: &BTreeMap<ResidentNullableRelationSlot, ResidentNullableRelationBindingState>,
    ) -> Result<()> {
        let (slot, expected_kind) = match self {
            Self::Binding { slot, kind }
            | Self::IntegerProperty { slot, kind, .. }
            | Self::StringProperty { slot, kind, .. } => (*slot, *kind),
            Self::String(value) => {
                if value.len() > RESIDENT_NULLABLE_RELATION_MAX_LITERAL_STRING_BYTES {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable string literal exceeds its bounded command contract",
                    ));
                }
                return Ok(());
            }
            Self::Null | Self::Boolean(_) | Self::Integer(_) => return Ok(()),
        };
        let Some(state) = live.get(&slot) else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable predicate references a binding outside its stage scope",
            ));
        };
        if state.kind != expected_kind {
            return Err(Error::new(
                ErrorCode::QueryType,
                "resident nullable predicate binding kind disagrees with the staged relation",
            ));
        }
        Ok(())
    }

    fn is_integer_source(&self) -> bool {
        matches!(
            self,
            Self::Null | Self::Integer(_) | Self::IntegerProperty { .. }
        )
    }

    fn is_string_source(&self) -> bool {
        matches!(
            self,
            Self::Null | Self::String(_) | Self::StringProperty { .. }
        )
    }

    fn entity_kind(&self) -> Option<ResidentNullableRelationBindingKind> {
        match self {
            Self::Binding { kind, .. } => Some(*kind),
            Self::Null
            | Self::Boolean(_)
            | Self::Integer(_)
            | Self::String(_)
            | Self::IntegerProperty { .. }
            | Self::StringProperty { .. } => None,
        }
    }

    fn is_null_source(&self) -> bool {
        matches!(self, Self::Null)
    }
}

impl ResidentNullableRelationPredicate {
    fn contains_relationship_endpoint_seed(&self) -> bool {
        match self {
            Self::RelationshipEndpoint { .. } => true,
            Self::Not(operand) => operand.contains_relationship_endpoint_seed(),
            Self::And(left, right) | Self::Or(left, right) => {
                left.contains_relationship_endpoint_seed()
                    || right.contains_relationship_endpoint_seed()
            }
            Self::Constant(_)
            | Self::IsNull { .. }
            | Self::HasLabels { .. }
            | Self::CompareInteger { .. }
            | Self::CompareString { .. } => false,
        }
    }

    fn contains_string_property_equality(&self) -> bool {
        match self {
            Self::CompareString { .. } => true,
            Self::Not(operand) => operand.contains_string_property_equality(),
            Self::And(left, right) | Self::Or(left, right) => {
                left.contains_string_property_equality()
                    || right.contains_string_property_equality()
            }
            Self::Constant(_)
            | Self::IsNull { .. }
            | Self::HasLabels { .. }
            | Self::CompareInteger { .. }
            | Self::RelationshipEndpoint { .. } => false,
        }
    }

    fn validate_against(
        &self,
        live: &BTreeMap<ResidentNullableRelationSlot, ResidentNullableRelationBindingState>,
    ) -> Result<()> {
        let mut pending = vec![(self, 1_usize)];
        let mut nodes = 0_usize;
        while let Some((predicate, depth)) = pending.pop() {
            nodes = nodes
                .checked_add(1)
                .ok_or_else(nullable_relation_scratch_overflow)?;
            if nodes > RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_NODES
                || depth > RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_DEPTH
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident nullable predicate exceeds its bounded tree contract",
                ));
            }
            match predicate {
                Self::Constant(_) => {}
                Self::IsNull { value, .. } => value.validate_against(live)?,
                Self::HasLabels { node, labels } => {
                    labels.validate()?;
                    let Some(state) = live.get(node) else {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable label predicate references an absent binding",
                        ));
                    };
                    if state.kind != ResidentNullableRelationBindingKind::Node {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident nullable label predicate requires a node binding",
                        ));
                    }
                }
                Self::CompareInteger {
                    left,
                    operation,
                    right,
                } => {
                    left.validate_against(live)?;
                    right.validate_against(live)?;
                    let integer_comparison = left.is_integer_source() && right.is_integer_source();
                    let entity_identity_comparison =
                        matches!(operation, CompareOp::Eq | CompareOp::NotEq)
                            && match (left.entity_kind(), right.entity_kind()) {
                                (Some(left), Some(right)) => left == right,
                                (Some(_), None) => right.is_null_source(),
                                (None, Some(_)) => left.is_null_source(),
                                (None, None) => false,
                            };
                    if !integer_comparison && !entity_identity_comparison {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident nullable comparison has incompatible operand types",
                        ));
                    }
                }
                Self::CompareString {
                    left,
                    operation: _,
                    right,
                } => {
                    left.validate_against(live)?;
                    right.validate_against(live)?;
                    if !left.is_string_source() || !right.is_string_source() {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident nullable string comparison requires string-compatible operands",
                        ));
                    }
                }
                Self::RelationshipEndpoint {
                    relationship,
                    node,
                    endpoint: _,
                } => {
                    let Some(relationship_state) = live.get(relationship) else {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable endpoint seed references an absent relationship",
                        ));
                    };
                    let Some(node_state) = live.get(node) else {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable endpoint seed references an absent node",
                        ));
                    };
                    if relationship_state.kind != ResidentNullableRelationBindingKind::Relationship
                        || node_state.kind != ResidentNullableRelationBindingKind::Node
                    {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "resident nullable endpoint seed has incompatible binding kinds",
                        ));
                    }
                }
                Self::Not(operand) => pending.push((operand, depth.saturating_add(1))),
                Self::And(left, right) | Self::Or(left, right) => {
                    let next_depth = depth.saturating_add(1);
                    pending.push((right, next_depth));
                    pending.push((left, next_depth));
                }
            }
        }
        Ok(())
    }
}

impl ResidentNullableRelationPropertyLaneRegistry {
    fn build(
        program: &ResidentNullableRelationProgram,
        predicates: &ResidentNullableRelationPredicateProgram,
    ) -> Result<Self> {
        let mut lanes = BTreeSet::new();
        let mut shapes = BTreeMap::<
            (ResidentNullableRelationBindingKind, PropertyId),
            ResidentNullableRelationPropertyShape,
        >::new();
        let mut projected_string_widths =
            BTreeMap::<(ResidentNullableRelationBindingKind, PropertyId), u32>::new();
        let mut register = |value: &ResidentNullableRelationPredicateValue| -> Result<()> {
            let Some(lane) = value.property_lane() else {
                return Ok(());
            };
            let physical_lane = (lane.kind, lane.property);
            if let Some(shape) = shapes.get(&physical_lane)
                && *shape != lane.shape
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident nullable property lane is assigned incompatible scalar shapes",
                ));
            }
            shapes.insert(physical_lane, lane.shape);
            lanes.insert(lane);
            Ok(())
        };

        for filter in &predicates.filters {
            let mut pending = vec![&filter.predicate];
            while let Some(predicate) = pending.pop() {
                match predicate {
                    ResidentNullableRelationPredicate::IsNull { value, .. } => register(value)?,
                    ResidentNullableRelationPredicate::CompareInteger { left, right, .. }
                    | ResidentNullableRelationPredicate::CompareString { left, right, .. } => {
                        register(left)?;
                        register(right)?;
                    }
                    ResidentNullableRelationPredicate::Not(operand) => pending.push(operand),
                    ResidentNullableRelationPredicate::And(left, right)
                    | ResidentNullableRelationPredicate::Or(left, right) => {
                        pending.push(right);
                        pending.push(left);
                    }
                    ResidentNullableRelationPredicate::Constant(_)
                    | ResidentNullableRelationPredicate::HasLabels { .. }
                    | ResidentNullableRelationPredicate::RelationshipEndpoint { .. } => {}
                }
            }
        }
        let analysis = program.analyze()?;
        for output in &analysis.final_bindings {
            if let ResidentNullableRelationOutputSource::StringProperty {
                kind,
                property,
                maximum_bytes,
                ..
            } = &output.source
            {
                let physical_lane = (*kind, *property);
                if let Some(sealed) = projected_string_widths.get(&physical_lane)
                    && *sealed != *maximum_bytes
                {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable string property lane has inconsistent exact output widths",
                    ));
                }
                projected_string_widths.insert(physical_lane, *maximum_bytes);
            }
            if let Some(lane) = output.source.property_lane() {
                let physical_lane = (lane.kind, lane.property);
                if let Some(shape) = shapes.get(&physical_lane)
                    && *shape != lane.shape
                {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable property lane is assigned incompatible scalar shapes",
                    ));
                }
                shapes.insert(physical_lane, lane.shape);
                lanes.insert(lane);
            }
        }
        if lanes.len() > RESIDENT_NULLABLE_RELATION_MAX_PROPERTY_LANES {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable property lane registry exceeds its bounded command contract",
            ));
        }
        Ok(Self {
            lanes: lanes.into_iter().collect(),
        })
    }

    fn validate_for(
        &self,
        program: &ResidentNullableRelationProgram,
        predicates: &ResidentNullableRelationPredicateProgram,
    ) -> Result<()> {
        if self.lanes.len() > RESIDENT_NULLABLE_RELATION_MAX_PROPERTY_LANES
            || self.lanes.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable property lanes must be bounded, sorted, and unique",
            ));
        }
        let canonical = Self::build(program, predicates)?;
        if *self != canonical {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable property lane registry is missing, inventing, or mismatching a predicate or projection dependency",
            ));
        }
        Ok(())
    }

    fn index_of(&self, lane: ResidentNullableRelationPropertyLane) -> Result<u32> {
        let index = self.lanes.binary_search(&lane).map_err(|_| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident nullable property lane is absent from its sealed registry",
            )
        })?;
        u32::try_from(index).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable property lane index exceeds u32",
            )
        })
    }
}

impl ResidentNullableRelationPredicateProgram {
    fn validate_for(&self, program: &ResidentNullableRelationProgram) -> Result<()> {
        if self.filters.len() > RESIDENT_NULLABLE_RELATION_MAX_FILTERS
            || self.optional_groups.len() > RESIDENT_NULLABLE_RELATION_MAX_STAGES
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable predicate program exceeds its bounded stage contract",
            ));
        }
        let analysis = program.analyze()?;
        let mut previous_last = None;
        for group in &self.optional_groups {
            let first = usize::from(group.first_stage);
            let last = usize::from(group.last_stage);
            if first >= last
                || last >= program.stages.len()
                || previous_last.is_some_and(|previous| first <= previous)
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident nullable OPTIONAL groups must be ordered, non-overlapping, and multi-hop",
                ));
            }
            for (offset, stage) in program.stages[first..=last].iter().enumerate() {
                let valid = matches!(
                    stage,
                    ResidentNullableRelationStage::Expand {
                        mode: ResidentNullableRelationMatchMode::Optional,
                        ..
                    }
                ) || offset == 0
                    && matches!(
                        stage,
                        ResidentNullableRelationStage::NodeScan {
                            mode: ResidentNullableRelationMatchMode::Optional,
                            ..
                        }
                    );
                if !valid {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable OPTIONAL group has an invalid seed or expansion stage",
                    ));
                }
            }
            previous_last = Some(last);
        }

        for filter in &self.filters {
            let live = match filter.placement {
                ResidentNullableRelationFilterPlacement::RelationAfter { stage } => {
                    let stage = usize::from(stage);
                    if stage + 1 >= program.stages.len()
                        || self.optional_groups.iter().any(|group| {
                            usize::from(group.first_stage) <= stage
                                && stage < usize::from(group.last_stage)
                        })
                    {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable relation filter is outside a complete stage boundary",
                        ));
                    }
                    analysis.stage_states.get(stage)
                }
                ResidentNullableRelationFilterPlacement::OptionalCandidates { stage } => {
                    let stage = usize::from(stage);
                    let containing_group = self.optional_groups.iter().find(|group| {
                        usize::from(group.first_stage) <= stage
                            && stage <= usize::from(group.last_stage)
                    });
                    let optional_stage = matches!(
                        program.stages.get(stage),
                        Some(
                            ResidentNullableRelationStage::NodeScan {
                                mode: ResidentNullableRelationMatchMode::Optional,
                                ..
                            } | ResidentNullableRelationStage::Expand {
                                mode: ResidentNullableRelationMatchMode::Optional,
                                ..
                            }
                        )
                    );
                    let endpoint_seed_at_group_start = containing_group.is_some_and(|group| {
                        usize::from(group.first_stage) == stage
                            && matches!(
                                program.stages.get(stage),
                                Some(ResidentNullableRelationStage::NodeScan {
                                    mode: ResidentNullableRelationMatchMode::Optional,
                                    ..
                                })
                            )
                            && matches!(
                                &filter.predicate,
                                ResidentNullableRelationPredicate::RelationshipEndpoint { .. }
                            )
                    });
                    if !optional_stage
                        || containing_group.is_some() && !endpoint_seed_at_group_start
                    {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable candidate filter does not target one optional stage",
                        ));
                    }
                    analysis.stage_states.get(stage)
                }
                ResidentNullableRelationFilterPlacement::OptionalGroupCandidates { group } => {
                    let Some(group) = self.optional_groups.get(usize::from(group)) else {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable candidate filter references an absent OPTIONAL group",
                        ));
                    };
                    analysis.stage_states.get(usize::from(group.last_stage))
                }
            }
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident nullable predicate placement references an absent stage",
                )
            })?;
            filter.predicate.validate_against(live)?;
        }
        Ok(())
    }
}

/// Immutable graph/catalog generation against which a nullable relation program was compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationGeneration {
    pub project: ProjectId,
    pub bookmark: Bookmark,
    pub graph_revision: u64,
    pub layout_version: ResidentGraphLayoutVersion,
    pub catalog_generation: [u8; 32],
}

/// Every intermediate capacity admitted before the first semantic stage. `stage_*_rows` are in
/// exact program order. They are independent of `max_output_rows`, which bounds only the final
/// client-visible packet and must never truncate a graph scan or expansion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationCapacities {
    pub node_slot_count: u32,
    pub relationship_slot_count: u32,
    pub visible_node_rows: u32,
    pub visible_relationship_rows: u32,
    pub binding_slot_count: u16,
    pub maximum_live_bindings: u16,
    pub stage_live_bindings: Vec<u16>,
    pub stage_input_rows: Vec<u64>,
    pub stage_candidate_rows: Vec<u64>,
    pub stage_output_rows: Vec<u64>,
    pub max_output_rows: u32,
    pub final_output_columns: u16,
    /// Exact logical packet bytes contributed by each admitted output row across every final
    /// column, including retained dense source rows, payloads, validity, and UTF-8 offsets.
    pub final_output_bytes_per_row: u64,
    /// Row-independent packet bytes. Each string output owns one terminal UTF-8 offset.
    pub final_output_fixed_bytes: u64,
}

/// Compositional upper bound for a staged relation. A node factor means that the relation still
/// contains one independent full-domain node scan. `base_rows` bounds the rows for each assignment
/// of every retained factor. Expanding from (or to) such a binding consumes that factor: across the
/// complete node domain, one directed relationship can contribute at most one candidate instead of
/// every input row independently seeing every relationship.
#[derive(Clone, Debug)]
struct ResidentNullableReachableRows {
    base_rows: u64,
    node_factors: BTreeMap<ResidentNullableRelationSlot, BTreeSet<ResidentNullableRelationSlot>>,
}

impl ResidentNullableReachableRows {
    fn unit() -> Self {
        Self {
            base_rows: 1,
            node_factors: BTreeMap::new(),
        }
    }

    fn zero() -> Self {
        Self {
            base_rows: 0,
            node_factors: BTreeMap::new(),
        }
    }

    fn opaque(rows: u64) -> Self {
        Self {
            base_rows: rows,
            node_factors: BTreeMap::new(),
        }
    }

    fn rows(&self, visible_node_rows: u32) -> Result<u64> {
        let mut rows = self.base_rows;
        for _ in &self.node_factors {
            rows = nullable_relation_product(rows, u64::from(visible_node_rows))?;
        }
        Ok(rows)
    }

    fn introduce_node_factor(&mut self, slot: ResidentNullableRelationSlot) {
        self.node_factors.insert(slot, BTreeSet::from([slot]));
    }

    fn factor_for_slot(
        &self,
        slot: ResidentNullableRelationSlot,
    ) -> Option<ResidentNullableRelationSlot> {
        self.node_factors
            .iter()
            .find_map(|(factor, aliases)| aliases.contains(&slot).then_some(*factor))
    }

    fn consume_expansion_endpoints(
        &mut self,
        source: ResidentNullableRelationSlot,
        target: ResidentNullableRelationTarget,
    ) -> bool {
        let source_factor = self.factor_for_slot(source);
        let target_factor = match target {
            ResidentNullableRelationTarget::Existing(target) => self.factor_for_slot(target),
            ResidentNullableRelationTarget::Introduce(_) => None,
        };
        let consumed_source = source_factor.is_some();
        if let Some(factor) = source_factor {
            self.node_factors.remove(&factor);
        }
        if let Some(factor) = target_factor {
            self.node_factors.remove(&factor);
        }
        consumed_source
    }

    fn project(&mut self, bindings: &[ResidentNullableRelationProjectionBinding]) {
        for aliases in self.node_factors.values_mut() {
            let prior = std::mem::take(aliases);
            *aliases = bindings
                .iter()
                .filter_map(|binding| prior.contains(&binding.source).then_some(binding.output))
                .collect();
        }
    }

    fn optional_union(self, other: Self, visible_node_rows: u32) -> Result<Self> {
        if self.base_rows == 0 {
            return Ok(other);
        }
        if other.base_rows == 0 {
            return Ok(self);
        }

        let common_factors = self
            .node_factors
            .keys()
            .filter(|factor| other.node_factors.contains_key(factor))
            .copied()
            .collect::<BTreeSet<_>>();
        let scaled_base = |bound: &Self| -> Result<u64> {
            let mut base = bound.base_rows;
            for factor in bound.node_factors.keys() {
                if !common_factors.contains(factor) {
                    base = nullable_relation_product(base, u64::from(visible_node_rows))?;
                }
            }
            Ok(base)
        };
        let base_rows = scaled_base(&self)?
            .checked_add(scaled_base(&other)?)
            .ok_or_else(nullable_relation_scratch_overflow)?;
        let node_factors = common_factors
            .into_iter()
            .map(|factor| {
                let aliases = self.node_factors[&factor]
                    .intersection(&other.node_factors[&factor])
                    .copied()
                    .collect();
                (factor, aliases)
            })
            .collect();
        Ok(Self {
            base_rows,
            node_factors,
        })
    }
}

impl ResidentNullableRelationCapacities {
    #[allow(clippy::too_many_arguments)]
    pub fn derive(
        program: &ResidentNullableRelationProgram,
        node_slot_count: usize,
        relationship_slot_count: usize,
        visible_node_rows: usize,
        visible_relationship_rows: usize,
        max_output_rows: usize,
    ) -> Result<Self> {
        let analysis = program.analyze()?;
        let node_slot_count = nullable_relation_u32(node_slot_count, "node-slot count")?;
        let relationship_slot_count =
            nullable_relation_u32(relationship_slot_count, "relationship-slot count")?;
        let visible_node_rows = nullable_relation_u32(visible_node_rows, "visible node-row count")?;
        let visible_relationship_rows =
            nullable_relation_u32(visible_relationship_rows, "visible relationship-row count")?;
        let max_output_rows = nullable_relation_u32(max_output_rows, "final output-row budget")?;
        let binding_slot_count = u16::try_from(analysis.binding_slot_count).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation binding-slot count exceeds u16",
            )
        })?;
        let maximum_live_bindings =
            u16::try_from(analysis.maximum_live_bindings).map_err(|_| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident nullable relation live binding count exceeds u16",
                )
            })?;
        let stage_live_bindings = analysis
            .stage_live_bindings
            .iter()
            .map(|bindings| {
                u16::try_from(*bindings).map_err(|_| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable relation stage width exceeds u16",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut reachable_rows = ResidentNullableReachableRows::unit();
        let mut stage_input_rows = Vec::with_capacity(program.stages.len());
        let mut stage_candidate_rows = Vec::with_capacity(program.stages.len());
        let mut stage_output_rows = Vec::with_capacity(program.stages.len());
        for (stage_index, stage) in program.stages.iter().enumerate() {
            let input_rows = reachable_rows.rows(visible_node_rows)?;
            stage_input_rows.push(input_rows);
            let (candidate_rows, output_rows) = match stage {
                ResidentNullableRelationStage::NodeScan {
                    mode,
                    output,
                    labels,
                } => {
                    if labels.is_known_empty() || visible_node_rows == 0 {
                        reachable_rows = match mode {
                            ResidentNullableRelationMatchMode::Mandatory => {
                                ResidentNullableReachableRows::zero()
                            }
                            ResidentNullableRelationMatchMode::Optional => reachable_rows,
                        };
                        (0, reachable_rows.rows(visible_node_rows)?)
                    } else {
                        reachable_rows.introduce_node_factor(*output);
                        let candidates = reachable_rows.rows(visible_node_rows)?;
                        (candidates, candidates)
                    }
                }
                ResidentNullableRelationStage::Expand {
                    mode,
                    source,
                    source_labels,
                    relationship,
                    target,
                    direction,
                    relationship_types,
                    target_labels,
                    ..
                } => {
                    let input_bound = reachable_rows.clone();
                    let mut candidate_bound = reachable_rows.clone();
                    let consumed_source =
                        candidate_bound.consume_expansion_endpoints(*source, *target);
                    let relationship_is_existing = relationship.is_some_and(|relationship| {
                        stage_index > 0
                            && analysis.stage_states[stage_index - 1]
                                .get(&relationship)
                                .is_some_and(|state| {
                                    state.kind == ResidentNullableRelationBindingKind::Relationship
                                })
                    });
                    if source_labels.is_known_empty()
                        || relationship_types.is_known_empty()
                        || target_labels.is_known_empty()
                        || visible_relationship_rows == 0
                        || input_rows == 0
                    {
                        candidate_bound = ResidentNullableReachableRows::zero();
                    } else if !relationship_is_existing {
                        let orientation_multiplier = if *direction == ResidentDirection::Undirected
                            && consumed_source
                            && visible_node_rows > 1
                        {
                            2
                        } else {
                            1
                        };
                        let relationship_candidates = nullable_relation_product(
                            u64::from(visible_relationship_rows),
                            orientation_multiplier,
                        )?;
                        candidate_bound.base_rows = nullable_relation_product(
                            candidate_bound.base_rows,
                            relationship_candidates,
                        )?;
                    }
                    let candidates = candidate_bound.rows(visible_node_rows)?;
                    reachable_rows = match mode {
                        ResidentNullableRelationMatchMode::Mandatory => candidate_bound,
                        // The candidate and unmatched-parent bounds are a conservative union. Any
                        // node factor consumed by the join is deliberately no longer considered
                        // independently enumerable after the union.
                        ResidentNullableRelationMatchMode::Optional => {
                            candidate_bound.optional_union(input_bound, visible_node_rows)?
                        }
                    };
                    (candidates, reachable_rows.rows(visible_node_rows)?)
                }
                ResidentNullableRelationStage::ScopeProject { bindings } => {
                    reachable_rows.project(bindings);
                    let output_rows = bindings[0]
                        .row_limit
                        .map_or(input_rows, |limit| input_rows.min(limit));
                    if output_rows < input_rows {
                        reachable_rows = ResidentNullableReachableRows::opaque(output_rows);
                    }
                    (output_rows, output_rows)
                }
                ResidentNullableRelationStage::FinalProject { .. } => (input_rows, input_rows),
            };
            stage_candidate_rows.push(candidate_rows);
            stage_output_rows.push(output_rows);
        }
        let final_output_columns = u16::try_from(analysis.final_bindings.len()).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation output width exceeds u16",
            )
        })?;
        let (final_output_bytes_per_row, final_output_fixed_bytes) = analysis
            .final_bindings
            .iter()
            .try_fold((0_u64, 0_u64), |(row_bytes, fixed_bytes), binding| {
                let (binding_row_bytes, binding_fixed_bytes) =
                    binding.source.output_packet_bytes()?;
                Ok::<_, Error>((
                    row_bytes
                        .checked_add(binding_row_bytes)
                        .ok_or_else(nullable_relation_scratch_overflow)?,
                    fixed_bytes
                        .checked_add(binding_fixed_bytes)
                        .ok_or_else(nullable_relation_scratch_overflow)?,
                ))
            })?;
        Ok(Self {
            node_slot_count,
            relationship_slot_count,
            visible_node_rows,
            visible_relationship_rows,
            binding_slot_count,
            maximum_live_bindings,
            stage_live_bindings,
            stage_input_rows,
            stage_candidate_rows,
            stage_output_rows,
            max_output_rows,
            final_output_columns,
            final_output_bytes_per_row,
            final_output_fixed_bytes,
        })
    }

    fn validate_for(&self, program: &ResidentNullableRelationProgram) -> Result<()> {
        let expected = Self::derive(
            program,
            self.node_slot_count as usize,
            self.relationship_slot_count as usize,
            self.visible_node_rows as usize,
            self.visible_relationship_rows as usize,
            self.max_output_rows as usize,
        )?;
        if *self != expected {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation capacities do not match the immutable program",
            ));
        }
        Ok(())
    }
}

/// Stage-specific semantic work expected from a single completed native command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ResidentNullableRelationObligationKind {
    MandatoryNodeScan = 1,
    OptionalNodeScan = 2,
    MandatoryExpand = 3,
    OptionalExpand = 4,
    ScopeProjection = 5,
    FinalProjection = 6,
    RelationFilter = 7,
    OptionalCandidateFilter = 8,
    AtomicOptionalGroup = 9,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentNullableRelationProofAction {
    Stage(usize),
    Filter(usize),
    OptionalGroup(usize),
}

fn nullable_relation_proof_actions(
    program: &ResidentNullableRelationProgram,
    predicates: &ResidentNullableRelationPredicateProgram,
) -> Result<Vec<ResidentNullableRelationProofAction>> {
    predicates.validate_for(program)?;
    let mut actions = Vec::with_capacity(
        program
            .stages
            .len()
            .saturating_add(predicates.filters.len())
            .saturating_add(predicates.optional_groups.len()),
    );
    for stage in 0..program.stages.len() {
        for (filter, specification) in predicates.filters.iter().enumerate() {
            if specification.placement
                == (ResidentNullableRelationFilterPlacement::OptionalCandidates {
                    stage: u16::try_from(stage).map_err(|_| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable proof stage exceeds u16",
                        )
                    })?,
                })
            {
                actions.push(ResidentNullableRelationProofAction::Filter(filter));
            }
        }
        actions.push(ResidentNullableRelationProofAction::Stage(stage));
        for (group, specification) in predicates.optional_groups.iter().enumerate() {
            if usize::from(specification.last_stage) == stage {
                for (filter, candidate) in predicates.filters.iter().enumerate() {
                    if candidate.placement
                        == (ResidentNullableRelationFilterPlacement::OptionalGroupCandidates {
                            group: u16::try_from(group).map_err(|_| {
                                Error::new(
                                    ErrorCode::GpuAdmissionFailure,
                                    "resident nullable OPTIONAL-group index exceeds u16",
                                )
                            })?,
                        })
                    {
                        actions.push(ResidentNullableRelationProofAction::Filter(filter));
                    }
                }
                actions.push(ResidentNullableRelationProofAction::OptionalGroup(group));
            }
        }
        for (filter, specification) in predicates.filters.iter().enumerate() {
            if specification.placement
                == (ResidentNullableRelationFilterPlacement::RelationAfter {
                    stage: u16::try_from(stage).map_err(|_| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable proof stage exceeds u16",
                        )
                    })?,
                })
            {
                actions.push(ResidentNullableRelationProofAction::Filter(filter));
            }
        }
    }
    if actions.len()
        != program
            .stages
            .len()
            .saturating_add(predicates.filters.len())
            .saturating_add(predicates.optional_groups.len())
    {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "resident nullable proof schedule omitted a predicate or OPTIONAL group",
        ));
    }
    Ok(actions)
}

fn nullable_relation_action_kind(
    action: ResidentNullableRelationProofAction,
    program: &ResidentNullableRelationProgram,
    predicates: &ResidentNullableRelationPredicateProgram,
) -> Result<ResidentNullableRelationObligationKind> {
    Ok(match action {
        ResidentNullableRelationProofAction::Stage(stage) => nullable_relation_obligation_kind(
            program
                .stages
                .get(stage)
                .ok_or_else(|| Error::internal("resident nullable proof stage disappeared"))?,
        ),
        ResidentNullableRelationProofAction::Filter(filter) => {
            match predicates
                .filters
                .get(filter)
                .ok_or_else(|| Error::internal("resident nullable proof filter disappeared"))?
                .placement
            {
                ResidentNullableRelationFilterPlacement::RelationAfter { .. } => {
                    ResidentNullableRelationObligationKind::RelationFilter
                }
                ResidentNullableRelationFilterPlacement::OptionalCandidates { .. }
                | ResidentNullableRelationFilterPlacement::OptionalGroupCandidates { .. } => {
                    ResidentNullableRelationObligationKind::OptionalCandidateFilter
                }
            }
        }
        ResidentNullableRelationProofAction::OptionalGroup(_) => {
            ResidentNullableRelationObligationKind::AtomicOptionalGroup
        }
    })
}

/// One ordered stage obligation. IDs are caller-owned and stage numbers are exact program
/// positions; neither may be synthesized by the backend after execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentNullableRelationObligation {
    pub id: u64,
    pub stage: u16,
    pub kind: ResidentNullableRelationObligationKind,
}

/// Exact immutable staged-relation fingerprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResidentNullableRelationFingerprint(pub [u8; 32]);

/// Complete ordered proof manifest for one staged relation program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationManifest {
    pub obligations: Vec<ResidentNullableRelationObligation>,
    pub fingerprint: ResidentNullableRelationFingerprint,
}

impl ResidentNullableRelationManifest {
    fn build(
        generation: ResidentNullableRelationGeneration,
        execution: ResidentExecutionId,
        program: &ResidentNullableRelationProgram,
        predicates: &ResidentNullableRelationPredicateProgram,
        property_lanes: &ResidentNullableRelationPropertyLaneRegistry,
        capacities: &ResidentNullableRelationCapacities,
        first_obligation_id: u64,
    ) -> Result<Self> {
        if first_obligation_id == 0 {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation requires non-zero obligation IDs",
            ));
        }
        let actions = nullable_relation_proof_actions(program, predicates)?;
        let obligations = actions
            .iter()
            .enumerate()
            .map(|(index, action)| {
                let stage_index = u16::try_from(index).map_err(|_| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable relation stage index exceeds u16",
                    )
                })?;
                let id = first_obligation_id
                    .checked_add(index as u64)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable relation obligation ID overflow",
                        )
                    })?;
                Ok(ResidentNullableRelationObligation {
                    id,
                    stage: stage_index,
                    kind: nullable_relation_action_kind(*action, program, predicates)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut manifest = Self {
            obligations,
            fingerprint: ResidentNullableRelationFingerprint([0; 32]),
        };
        manifest.fingerprint = nullable_relation_fingerprint(
            generation,
            execution,
            program,
            predicates,
            property_lanes,
            capacities,
            &manifest,
        )?;
        Ok(manifest)
    }

    fn validate_shape(
        &self,
        program: &ResidentNullableRelationProgram,
        predicates: &ResidentNullableRelationPredicateProgram,
    ) -> Result<()> {
        let actions = nullable_relation_proof_actions(program, predicates)?;
        if self.obligations.len() != actions.len() {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation manifest does not cover every stage",
            ));
        }
        let mut ids = BTreeSet::new();
        for (index, (obligation, action)) in self.obligations.iter().zip(&actions).enumerate() {
            if obligation.id == 0
                || obligation.stage as usize != index
                || obligation.kind != nullable_relation_action_kind(*action, program, predicates)?
                || !ids.insert(obligation.id)
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "resident nullable relation manifest has a missing, duplicate, or wrong-stage obligation",
                ));
            }
        }
        Ok(())
    }
}

/// One complete staged nullable relation command. No prefix or intermediate relation is public.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationRequest {
    pub generation: ResidentNullableRelationGeneration,
    pub execution: ResidentExecutionId,
    pub program: ResidentNullableRelationProgram,
    pub predicate_program: ResidentNullableRelationPredicateProgram,
    property_lanes: ResidentNullableRelationPropertyLaneRegistry,
    pub capacities: ResidentNullableRelationCapacities,
    pub manifest: ResidentNullableRelationManifest,
}

impl ResidentNullableRelationRequest {
    pub fn build(
        generation: ResidentNullableRelationGeneration,
        execution: ResidentExecutionId,
        program: ResidentNullableRelationProgram,
        capacities: ResidentNullableRelationCapacities,
        first_obligation_id: u64,
    ) -> Result<Self> {
        Self::build_with_predicates(
            generation,
            execution,
            program,
            ResidentNullableRelationPredicateProgram::default(),
            capacities,
            first_obligation_id,
        )
    }

    pub fn build_with_predicates(
        generation: ResidentNullableRelationGeneration,
        execution: ResidentExecutionId,
        program: ResidentNullableRelationProgram,
        predicate_program: ResidentNullableRelationPredicateProgram,
        capacities: ResidentNullableRelationCapacities,
        first_obligation_id: u64,
    ) -> Result<Self> {
        program.validate()?;
        predicate_program.validate_for(&program)?;
        let property_lanes =
            ResidentNullableRelationPropertyLaneRegistry::build(&program, &predicate_program)?;
        capacities.validate_for(&program)?;
        let manifest = ResidentNullableRelationManifest::build(
            generation,
            execution,
            &program,
            &predicate_program,
            &property_lanes,
            &capacities,
            first_obligation_id,
        )?;
        let request = Self {
            generation,
            execution,
            program,
            predicate_program,
            property_lanes,
            capacities,
            manifest,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<()> {
        if self.execution.high == 0 && self.execution.low == 0 {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation requires a non-zero execution ID",
            ));
        }
        self.program.validate()?;
        self.predicate_program.validate_for(&self.program)?;
        self.property_lanes
            .validate_for(&self.program, &self.predicate_program)?;
        self.capacities.validate_for(&self.program)?;
        self.manifest
            .validate_shape(&self.program, &self.predicate_program)?;
        let fingerprint = nullable_relation_fingerprint(
            self.generation,
            self.execution,
            &self.program,
            &self.predicate_program,
            &self.property_lanes,
            &self.capacities,
            &self.manifest,
        )?;
        if fingerprint != self.manifest.fingerprint {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation fingerprint does not match its immutable request",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn obligations(&self) -> impl Iterator<Item = ResidentNullableRelationObligation> + '_ {
        self.manifest.obligations.iter().copied()
    }

    /// Canonical physical graph tensors required by every predicate and final projection in this
    /// command. Backends bind these lanes in the returned order and encode each property's
    /// relation slot separately.
    #[must_use]
    pub fn property_lanes(&self) -> &[ResidentNullableRelationPropertyLane] {
        self.property_lanes.lanes()
    }

    pub fn property_lane_index(
        &self,
        value: &ResidentNullableRelationPredicateValue,
    ) -> Result<Option<u32>> {
        value
            .property_lane()
            .map(|lane| self.property_lanes.index_of(lane))
            .transpose()
    }

    pub fn output_property_lane_index(
        &self,
        source: &ResidentNullableRelationOutputSource,
    ) -> Result<Option<u32>> {
        source
            .property_lane()
            .map(|lane| self.property_lanes.index_of(lane))
            .transpose()
    }

    pub fn proof_actions(&self) -> Result<Vec<ResidentNullableRelationProofAction>> {
        nullable_relation_proof_actions(&self.program, &self.predicate_program)
    }

    pub fn obligation_for(
        &self,
        action: ResidentNullableRelationProofAction,
    ) -> Result<ResidentNullableRelationObligation> {
        let actions = self.proof_actions()?;
        let position = actions
            .iter()
            .position(|candidate| *candidate == action)
            .ok_or_else(|| {
                Error::internal("resident nullable proof action disappeared from its manifest")
            })?;
        self.manifest
            .obligations
            .get(position)
            .copied()
            .ok_or_else(|| Error::internal("resident nullable proof obligation disappeared"))
    }

    pub fn optional_group_introduced_bindings(
        &self,
        group: usize,
    ) -> Result<
        Vec<(
            ResidentNullableRelationSlot,
            ResidentNullableRelationBindingKind,
        )>,
    > {
        let specification = self
            .predicate_program
            .optional_groups
            .get(group)
            .ok_or_else(|| Error::internal("resident nullable OPTIONAL group disappeared"))?;
        let analysis = self.program.analyze()?;
        let before = if specification.first_stage == 0 {
            BTreeMap::new()
        } else {
            analysis
                .stage_states
                .get(usize::from(specification.first_stage) - 1)
                .cloned()
                .ok_or_else(|| Error::internal("resident nullable group input scope disappeared"))?
        };
        let after = analysis
            .stage_states
            .get(usize::from(specification.last_stage))
            .ok_or_else(|| Error::internal("resident nullable group output scope disappeared"))?;
        Ok(after
            .iter()
            .filter_map(|(slot, state)| (!before.contains_key(slot)).then_some((*slot, state.kind)))
            .collect())
    }

    /// Exact logical bytes reserved before stage zero. Both relation images, candidate/compaction
    /// workspace, final packet, and all receipts are included with checked arithmetic.
    pub fn scratch_bytes(&self) -> Result<usize> {
        self.validate()?;
        let mut maximum_relation_cells = 0_usize;
        let mut maximum_candidate_rows = 0_usize;
        for ((rows, bindings), candidates) in self
            .capacities
            .stage_output_rows
            .iter()
            .zip(&self.capacities.stage_live_bindings)
            .zip(&self.capacities.stage_candidate_rows)
        {
            let rows = usize::try_from(*rows).map_err(|_| nullable_relation_scratch_overflow())?;
            let candidates =
                usize::try_from(*candidates).map_err(|_| nullable_relation_scratch_overflow())?;
            maximum_relation_cells =
                maximum_relation_cells.max(checked_product(rows, usize::from(*bindings))?);
            maximum_candidate_rows = maximum_candidate_rows.max(candidates);
        }
        let ping_pong_relations = checked_product(
            checked_product(maximum_relation_cells, size_of::<u32>())?,
            2,
        )?;
        // Candidate source/relationship/target lanes plus prefix/validity compaction workspace.
        let candidate_rows = checked_product(
            maximum_candidate_rows,
            3 * size_of::<u32>() + size_of::<u32>() + size_of::<u8>(),
        )?;
        let final_rows = usize::try_from(
            self.capacities
                .stage_output_rows
                .last()
                .copied()
                .unwrap_or(0)
                .min(u64::from(self.capacities.max_output_rows)),
        )
        .map_err(|_| nullable_relation_scratch_overflow())?;
        let final_bytes_per_row = usize::try_from(self.capacities.final_output_bytes_per_row)
            .map_err(|_| nullable_relation_scratch_overflow())?;
        let final_fixed_bytes = usize::try_from(self.capacities.final_output_fixed_bytes)
            .map_err(|_| nullable_relation_scratch_overflow())?;
        let final_packet = checked_product(final_rows, final_bytes_per_row)?
            .checked_add(final_fixed_bytes)
            .ok_or_else(nullable_relation_scratch_overflow)?;
        let receipts = checked_product(
            self.manifest.obligations.len(),
            size_of::<ResidentNullableRelationReceipt>(),
        )?;
        let predicate_workspace = if self.predicate_program.is_empty() {
            0
        } else {
            let maximum_rows = usize::try_from(
                self.capacities
                    .stage_output_rows
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
            )
            .map_err(|_| nullable_relation_scratch_overflow())?;
            // One truth byte, one parent-lineage lane, and one group-completion byte per row.
            checked_product(maximum_rows, 2 * size_of::<u8>() + size_of::<u32>())?
        };
        [
            ping_pong_relations,
            candidate_rows,
            final_packet,
            receipts,
            predicate_workspace,
        ]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(nullable_relation_scratch_overflow)
        })
    }
}

/// Device-authored proof for one stage. Graph/projection fields retain their ordinary cardinality
/// meanings. For a filter receipt, `candidate_cardinality`, `matched_input_cardinality`, and
/// `null_extension_cardinality` are respectively the exact True, False, and Null counts, and
/// output cardinality equals the True count. This makes three-valued WHERE behavior observable
/// without publishing intermediate rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationReceipt {
    pub execution: ResidentExecutionId,
    pub obligation: ResidentNullableRelationObligation,
    pub input_cardinality: u64,
    pub matched_input_cardinality: u64,
    pub candidate_cardinality: u64,
    pub null_extension_cardinality: u64,
    pub output_cardinality: u64,
    pub completion: ResidentDeviceCompletion,
}

/// One device-authored final column in projection order. Property columns retain their source
/// dense entity rows so result validation can prove alignment and binding-null propagation without
/// rereading graph properties on the host. Invalid integer and float payloads are canonical
/// zeroes; invalid string payloads own empty UTF-8 ranges. A valid empty string therefore remains
/// distinct from null through its validity byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentNullableRelationOutputColumn {
    Entity {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        rows: Vec<u32>,
    },
    IntegerProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
        source_rows: Vec<u32>,
        values: Vec<i64>,
        validity: Vec<u8>,
    },
    StringPropertyToInteger {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
        source_rows: Vec<u32>,
        values: Vec<i64>,
        validity: Vec<u8>,
    },
    FloatProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
        source_rows: Vec<u32>,
        bits: Vec<u64>,
        validity: Vec<u8>,
    },
    IntegerPropertyToFloat {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
        source_rows: Vec<u32>,
        bits: Vec<u64>,
        validity: Vec<u8>,
    },
    StringProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
        maximum_bytes: u32,
        source_rows: Vec<u32>,
        offsets: Vec<u32>,
        bytes: Vec<u8>,
        validity: Vec<u8>,
    },
    IntegerPropertyToString {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        property: PropertyId,
        source_rows: Vec<u32>,
        offsets: Vec<u32>,
        bytes: Vec<u8>,
        validity: Vec<u8>,
    },
    NullProperty {
        slot: ResidentNullableRelationSlot,
        kind: ResidentNullableRelationBindingKind,
        source_rows: Vec<u32>,
    },
    /// Device-authored canonical relationship-type tokens aligned with `source_rows`. A null
    /// source row must carry `RelationshipTypeId(0)` as its canonical hidden payload. Token zero
    /// remains a valid non-null type because nullability is determined solely by the source row.
    RelationshipType {
        slot: ResidentNullableRelationSlot,
        source_rows: Vec<u32>,
        relationship_types: Vec<RelationshipTypeId>,
    },
    /// Device-authored three-valued result of one entity label/type predicate. `values` and
    /// `validity` are canonical Boolean bytes. A null source row is the only invalid result and
    /// must carry a zero value; every non-null entity row produces a valid true or false value.
    EntityLabelPredicate {
        slot: ResidentNullableRelationSlot,
        domain: ResidentNullableRelationEntityLabelDomain,
        source_rows: Vec<u32>,
        values: Vec<u8>,
        validity: Vec<u8>,
    },
}

impl ResidentNullableRelationOutputColumn {
    #[must_use]
    pub fn source(&self) -> ResidentNullableRelationOutputSource {
        match self {
            Self::Entity { slot, kind, .. } => ResidentNullableRelationOutputSource::Entity {
                slot: *slot,
                kind: *kind,
            },
            Self::IntegerProperty {
                slot,
                kind,
                property,
                ..
            } => ResidentNullableRelationOutputSource::IntegerProperty {
                slot: *slot,
                kind: *kind,
                property: *property,
            },
            Self::StringPropertyToInteger {
                slot,
                kind,
                property,
                ..
            } => ResidentNullableRelationOutputSource::StringPropertyToInteger {
                slot: *slot,
                kind: *kind,
                property: *property,
            },
            Self::FloatProperty {
                slot,
                kind,
                property,
                ..
            } => ResidentNullableRelationOutputSource::FloatProperty {
                slot: *slot,
                kind: *kind,
                property: *property,
            },
            Self::IntegerPropertyToFloat {
                slot,
                kind,
                property,
                ..
            } => ResidentNullableRelationOutputSource::IntegerPropertyToFloat {
                slot: *slot,
                kind: *kind,
                property: *property,
            },
            Self::StringProperty {
                slot,
                kind,
                property,
                maximum_bytes,
                ..
            } => ResidentNullableRelationOutputSource::StringProperty {
                slot: *slot,
                kind: *kind,
                property: *property,
                maximum_bytes: *maximum_bytes,
            },
            Self::IntegerPropertyToString {
                slot,
                kind,
                property,
                ..
            } => ResidentNullableRelationOutputSource::IntegerPropertyToString {
                slot: *slot,
                kind: *kind,
                property: *property,
            },
            Self::NullProperty { slot, kind, .. } => {
                ResidentNullableRelationOutputSource::NullProperty {
                    slot: *slot,
                    kind: *kind,
                }
            }
            Self::RelationshipType { slot, .. } => {
                ResidentNullableRelationOutputSource::RelationshipType { slot: *slot }
            }
            Self::EntityLabelPredicate { slot, domain, .. } => {
                ResidentNullableRelationOutputSource::EntityLabelPredicate {
                    slot: *slot,
                    domain: domain.clone(),
                }
            }
        }
    }

    #[must_use]
    pub fn source_rows(&self) -> &[u32] {
        match self {
            Self::Entity { rows, .. } => rows,
            Self::IntegerProperty { source_rows, .. }
            | Self::StringPropertyToInteger { source_rows, .. }
            | Self::FloatProperty { source_rows, .. }
            | Self::IntegerPropertyToFloat { source_rows, .. }
            | Self::StringProperty { source_rows, .. }
            | Self::IntegerPropertyToString { source_rows, .. }
            | Self::NullProperty { source_rows, .. }
            | Self::RelationshipType { source_rows, .. }
            | Self::EntityLabelPredicate { source_rows, .. } => source_rows,
        }
    }
}

/// Publicly constructible untrusted result parts for backend bring-up and corruption tests. The
/// contained graph rows remain private to query execution until full validation succeeds.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationResultParts {
    pub generation: ResidentNullableRelationGeneration,
    pub execution: ResidentExecutionId,
    pub fingerprint: ResidentNullableRelationFingerprint,
    pub row_count: usize,
    pub columns: Vec<ResidentNullableRelationOutputColumn>,
    pub receipts: Vec<ResidentNullableRelationReceipt>,
    pub scratch_bytes: usize,
}

/// Raw staged-relation output. Values have no public access before validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentNullableRelationResult {
    parts: ResidentNullableRelationResultParts,
}

/// Generation-, fingerprint-, capacity-, receipt-, and dense-row-validated result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedResidentNullableRelationResult {
    parts: ResidentNullableRelationResultParts,
}

impl ResidentNullableRelationResult {
    #[allow(dead_code)] // Reserved for the unwired CPU/Metal command constructors.
    pub fn completed(parts: ResidentNullableRelationResultParts) -> Self {
        Self { parts }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn from_untrusted_parts(parts: ResidentNullableRelationResultParts) -> Self {
        Self { parts }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn into_untrusted_parts(self) -> ResidentNullableRelationResultParts {
        self.parts
    }

    pub fn validate(
        self,
        request: &ResidentNullableRelationRequest,
        completion: ResidentDeviceCompletion,
    ) -> Result<ValidatedResidentNullableRelationResult> {
        request.validate()?;
        let parts = self.parts;
        if parts.generation != request.generation
            || parts.execution != request.execution
            || parts.fingerprint != request.manifest.fingerprint
            || parts.scratch_bytes != request.scratch_bytes()?
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident nullable relation result belongs to a different generation, request, or scratch admission",
            ));
        }
        validate_nullable_relation_receipts(request, &parts.receipts, completion)?;
        let expected_rows = parts.receipts.last().map_or(Ok(0_usize), |receipt| {
            usize::try_from(receipt.output_cardinality).map_err(|_| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "resident nullable relation output cardinality exceeds usize",
                )
            })
        })?;
        if parts.row_count != expected_rows
            || parts.row_count > request.capacities.max_output_rows as usize
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident nullable relation result row count violates its final receipt or output budget",
            ));
        }

        let actions = request.proof_actions()?;
        let mut stage_receipts = actions
            .iter()
            .zip(&parts.receipts)
            .filter_map(|(action, receipt)| {
                matches!(action, ResidentNullableRelationProofAction::Stage(_)).then_some(*receipt)
            })
            .collect::<Vec<_>>();
        for (group_index, group) in request.predicate_program.optional_groups.iter().enumerate() {
            let group_receipt = actions
                .iter()
                .zip(&parts.receipts)
                .find_map(|(action, receipt)| {
                    (*action == ResidentNullableRelationProofAction::OptionalGroup(group_index))
                        .then_some(*receipt)
                })
                .ok_or_else(|| {
                    Error::internal("resident atomic OPTIONAL result receipt disappeared")
                })?;
            for stage in usize::from(group.first_stage)..=usize::from(group.last_stage) {
                let receipt = stage_receipts.get_mut(stage).ok_or_else(|| {
                    Error::internal("resident atomic OPTIONAL stage receipt disappeared")
                })?;
                // Grouped hops are candidate-only internally. Nullability becomes observable only
                // at the atomic boundary, which nulls every binding introduced by the group.
                receipt.candidate_cardinality = group_receipt.candidate_cardinality;
                receipt.null_extension_cardinality = group_receipt.null_extension_cardinality;
                receipt.output_cardinality = group_receipt.output_cardinality;
            }
        }
        let analysis = request
            .program
            .analyze_with_receipts(Some(&stage_receipts))?;
        if parts.columns.len() != analysis.final_bindings.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident nullable relation result omitted or added final columns",
            ));
        }
        for (column, output) in parts.columns.iter().zip(&analysis.final_bindings) {
            let state = analysis
                .final_states
                .get(&output.source.slot())
                .ok_or_else(|| {
                    Error::internal("resident nullable final binding state disappeared")
                })?;
            if column.source() != output.source.clone()
                || column.source_rows().len() != parts.row_count
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident nullable relation result column disagrees with its final projection",
                ));
            }
            let dense_capacity = match output.source.kind() {
                ResidentNullableRelationBindingKind::Node => request.capacities.node_slot_count,
                ResidentNullableRelationBindingKind::Relationship => {
                    request.capacities.relationship_slot_count
                }
            };
            for row in column.source_rows() {
                let is_null = *row == RESIDENT_NULLABLE_RELATION_NULL_ROW;
                if (!is_null && *row >= dense_capacity)
                    || (state.nullability == ResidentNullableRelationNullability::Never && is_null)
                    || (state.nullability == ResidentNullableRelationNullability::Always
                        && !is_null)
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident nullable relation result contains an invalid dense row or null sentinel",
                    ));
                }
            }
            match column {
                ResidentNullableRelationOutputColumn::Entity { .. }
                | ResidentNullableRelationOutputColumn::NullProperty { .. } => {}
                ResidentNullableRelationOutputColumn::IntegerProperty {
                    source_rows,
                    values,
                    validity,
                    ..
                }
                | ResidentNullableRelationOutputColumn::StringPropertyToInteger {
                    source_rows,
                    values,
                    validity,
                    ..
                } => {
                    if values.len() != parts.row_count
                        || validity.len() != parts.row_count
                        || validity.iter().any(|valid| *valid > 1)
                        || values
                            .iter()
                            .zip(validity)
                            .any(|(value, valid)| *valid == 0 && *value != 0)
                        || source_rows.iter().zip(validity).any(|(row, valid)| {
                            *row == RESIDENT_NULLABLE_RELATION_NULL_ROW && *valid != 0
                        })
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "resident nullable integer output has invalid payload, validity, or source-null propagation",
                        ));
                    }
                }
                ResidentNullableRelationOutputColumn::FloatProperty {
                    source_rows,
                    bits,
                    validity,
                    ..
                }
                | ResidentNullableRelationOutputColumn::IntegerPropertyToFloat {
                    source_rows,
                    bits,
                    validity,
                    ..
                } => {
                    if bits.len() != parts.row_count
                        || validity.len() != parts.row_count
                        || validity.iter().any(|valid| *valid > 1)
                        || bits.iter().zip(validity).any(|(bits, valid)| {
                            (*valid == 0 && *bits != 0)
                                || (*valid == 1 && !f64::from_bits(*bits).is_finite())
                        })
                        || source_rows.iter().zip(validity).any(|(row, valid)| {
                            *row == RESIDENT_NULLABLE_RELATION_NULL_ROW && *valid != 0
                        })
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "resident nullable float output has invalid payload, validity, or source-null propagation",
                        ));
                    }
                }
                ResidentNullableRelationOutputColumn::StringProperty {
                    maximum_bytes,
                    source_rows,
                    offsets,
                    bytes,
                    validity,
                    ..
                } => {
                    let bounded_ranges = offsets.windows(2).all(|pair| {
                        pair[1]
                            .checked_sub(pair[0])
                            .is_some_and(|width| width <= *maximum_bytes)
                    });
                    if validity.len() != parts.row_count
                        || validity.iter().any(|valid| *valid > 1)
                        || !canonical_utf8_ranges(offsets, bytes, validity, parts.row_count)
                        || !bounded_ranges
                        || source_rows.iter().zip(validity).any(|(row, valid)| {
                            *row == RESIDENT_NULLABLE_RELATION_NULL_ROW && *valid != 0
                        })
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "resident nullable string output has invalid UTF-8, width, validity, or source-null propagation",
                        ));
                    }
                }
                ResidentNullableRelationOutputColumn::IntegerPropertyToString {
                    source_rows,
                    offsets,
                    bytes,
                    validity,
                    ..
                } => {
                    let bounded_ranges = offsets.windows(2).all(|pair| {
                        pair[1].checked_sub(pair[0]).is_some_and(|width| {
                            width <= RESIDENT_NULLABLE_RELATION_INTEGER_STRING_MAXIMUM_BYTES
                        })
                    });
                    if validity.len() != parts.row_count
                        || validity.iter().any(|valid| *valid > 1)
                        || !canonical_utf8_ranges(offsets, bytes, validity, parts.row_count)
                        || !bounded_ranges
                        || source_rows.iter().zip(validity).any(|(row, valid)| {
                            *row == RESIDENT_NULLABLE_RELATION_NULL_ROW && *valid != 0
                        })
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "resident nullable integer-to-string output has invalid UTF-8, width, validity, or source-null propagation",
                        ));
                    }
                }
                ResidentNullableRelationOutputColumn::RelationshipType {
                    source_rows,
                    relationship_types,
                    ..
                } => {
                    if relationship_types.len() != parts.row_count
                        || source_rows
                            .iter()
                            .zip(relationship_types)
                            .any(|(row, token)| {
                                *row == RESIDENT_NULLABLE_RELATION_NULL_ROW && token.0 != 0
                            })
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "resident nullable relationship-type output has an invalid token count or non-canonical null payload",
                        ));
                    }
                }
                ResidentNullableRelationOutputColumn::EntityLabelPredicate {
                    source_rows,
                    values,
                    validity,
                    ..
                } => {
                    if values.len() != parts.row_count
                        || validity.len() != parts.row_count
                        || values.iter().any(|value| *value > 1)
                        || validity.iter().any(|valid| *valid > 1)
                        || source_rows.iter().zip(values).zip(validity).any(
                            |((source_row, value), valid)| {
                                (*valid == 0 && *value != 0)
                                    || (*source_row == RESIDENT_NULLABLE_RELATION_NULL_ROW)
                                        != (*valid == 0)
                            },
                        )
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "resident nullable entity-label output has invalid Boolean bytes or null propagation",
                        ));
                    }
                }
            }
        }
        Ok(ValidatedResidentNullableRelationResult { parts })
    }
}

impl ValidatedResidentNullableRelationResult {
    #[must_use]
    pub fn columns(&self) -> &[ResidentNullableRelationOutputColumn] {
        &self.parts.columns
    }

    #[must_use]
    pub fn receipts(&self) -> &[ResidentNullableRelationReceipt] {
        &self.parts.receipts
    }

    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.parts.row_count
    }

    #[must_use]
    pub fn into_parts(self) -> ResidentNullableRelationResultParts {
        self.parts
    }
}

fn nullable_relation_obligation_kind(
    stage: &ResidentNullableRelationStage,
) -> ResidentNullableRelationObligationKind {
    match stage {
        ResidentNullableRelationStage::NodeScan {
            mode: ResidentNullableRelationMatchMode::Mandatory,
            ..
        } => ResidentNullableRelationObligationKind::MandatoryNodeScan,
        ResidentNullableRelationStage::NodeScan {
            mode: ResidentNullableRelationMatchMode::Optional,
            ..
        } => ResidentNullableRelationObligationKind::OptionalNodeScan,
        ResidentNullableRelationStage::Expand {
            mode: ResidentNullableRelationMatchMode::Mandatory,
            ..
        } => ResidentNullableRelationObligationKind::MandatoryExpand,
        ResidentNullableRelationStage::Expand {
            mode: ResidentNullableRelationMatchMode::Optional,
            ..
        } => ResidentNullableRelationObligationKind::OptionalExpand,
        ResidentNullableRelationStage::ScopeProject { .. } => {
            ResidentNullableRelationObligationKind::ScopeProjection
        }
        ResidentNullableRelationStage::FinalProject { .. } => {
            ResidentNullableRelationObligationKind::FinalProjection
        }
    }
}

fn validate_nullable_relation_receipts(
    request: &ResidentNullableRelationRequest,
    receipts: &[ResidentNullableRelationReceipt],
    completion: ResidentDeviceCompletion,
) -> Result<()> {
    let actions = request.proof_actions()?;
    if receipts.len() != request.manifest.obligations.len() || receipts.len() != actions.len() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "resident nullable relation result has a missing or extra stage receipt",
        ));
    }
    let mut expected_input = 1_u64;
    let mut candidate_filter_outputs = BTreeMap::<(u8, usize), u64>::new();
    let mut optional_group_parent_inputs = BTreeMap::<usize, u64>::new();
    for ((receipt, obligation), action) in receipts
        .iter()
        .zip(&request.manifest.obligations)
        .zip(&actions)
    {
        if receipt.execution != request.execution
            || receipt.obligation != *obligation
            || receipt.completion != completion
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident nullable relation result has a forged or wrong-backend receipt",
            ));
        }
        match *action {
            ResidentNullableRelationProofAction::Filter(filter_index) => {
                let filter = request
                    .predicate_program
                    .filters
                    .get(filter_index)
                    .ok_or_else(|| {
                        Error::internal("resident nullable receipt filter disappeared")
                    })?;
                let classified = receipt
                    .candidate_cardinality
                    .checked_add(receipt.matched_input_cardinality)
                    .and_then(|value| value.checked_add(receipt.null_extension_cardinality))
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::CorruptStorage,
                            "resident nullable filter receipt cardinality overflow",
                        )
                    })?;
                if classified != receipt.input_cardinality
                    || receipt.output_cardinality != receipt.candidate_cardinality
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident nullable filter receipt violates exact three-valued semantics",
                    ));
                }
                match filter.placement {
                    ResidentNullableRelationFilterPlacement::RelationAfter { stage } => {
                        if receipt.input_cardinality != expected_input
                            || receipt.output_cardinality
                                > u64::from(
                                    request.capacities.stage_output_rows[usize::from(stage)],
                                )
                        {
                            return Err(Error::new(
                                ErrorCode::CorruptStorage,
                                "resident nullable relation-filter receipt has a forged input or capacity",
                            ));
                        }
                        expected_input = receipt.output_cardinality;
                    }
                    ResidentNullableRelationFilterPlacement::OptionalCandidates { stage } => {
                        let key = (1_u8, usize::from(stage));
                        if let Some(previous) = candidate_filter_outputs.get(&key)
                            && receipt.input_cardinality != *previous
                        {
                            return Err(Error::new(
                                ErrorCode::CorruptStorage,
                                "resident nullable candidate-filter chain changed cardinality",
                            ));
                        }
                        if receipt.input_cardinality
                            > u64::from(request.capacities.stage_candidate_rows[usize::from(stage)])
                        {
                            return Err(Error::new(
                                ErrorCode::CorruptStorage,
                                "resident nullable candidate-filter receipt exceeds capacity",
                            ));
                        }
                        candidate_filter_outputs.insert(key, receipt.output_cardinality);
                    }
                    ResidentNullableRelationFilterPlacement::OptionalGroupCandidates { group } => {
                        let key = (2_u8, usize::from(group));
                        if let Some(previous) = candidate_filter_outputs.get(&key) {
                            if receipt.input_cardinality != *previous {
                                return Err(Error::new(
                                    ErrorCode::CorruptStorage,
                                    "resident nullable OPTIONAL-group filter chain changed cardinality",
                                ));
                            }
                        } else if receipt.input_cardinality > expected_input {
                            return Err(Error::new(
                                ErrorCode::CorruptStorage,
                                "resident nullable OPTIONAL-group filter tested more rows than its internal relation",
                            ));
                        }
                        candidate_filter_outputs.insert(key, receipt.output_cardinality);
                    }
                }
            }
            ResidentNullableRelationProofAction::Stage(stage_index) => {
                let stage = &request.program.stages[stage_index];
                for (group_index, _group) in request
                    .predicate_program
                    .optional_groups
                    .iter()
                    .enumerate()
                    .filter(|(_, group)| usize::from(group.first_stage) == stage_index)
                {
                    optional_group_parent_inputs.insert(group_index, receipt.input_cardinality);
                }
                let optional_group =
                    request
                        .predicate_program
                        .optional_groups
                        .iter()
                        .position(|group| {
                            usize::from(group.first_stage) <= stage_index
                                && stage_index <= usize::from(group.last_stage)
                        });
                let candidate_capacity =
                    u64::from(request.capacities.stage_candidate_rows[stage_index]);
                let output_capacity = u64::from(request.capacities.stage_output_rows[stage_index]);
                if receipt.input_cardinality != expected_input
                    || receipt.candidate_cardinality > candidate_capacity
                    || receipt.output_cardinality > output_capacity
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident nullable graph-stage receipt has a forged input or capacity",
                    ));
                }
                let filtered_candidates =
                    candidate_filter_outputs.get(&(1_u8, stage_index)).copied();
                if filtered_candidates
                    .is_some_and(|candidates| candidates != receipt.candidate_cardinality)
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident nullable graph-stage receipt disagrees with its candidate filter",
                    ));
                }
                let matched_inputs = receipt.matched_input_cardinality;
                let candidates = receipt.candidate_cardinality;
                if matched_inputs > receipt.input_cardinality
                    || (matched_inputs == 0) != (candidates == 0)
                    || candidates < matched_inputs
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident nullable graph-stage receipt has an impossible matched-input cardinality",
                    ));
                }
                match stage {
                    ResidentNullableRelationStage::NodeScan { mode, labels, .. } => {
                        let exact_cartesian_shape = filtered_candidates.is_some()
                            || if receipt.input_cardinality == 0 {
                                receipt.candidate_cardinality == 0
                            } else if receipt.candidate_cardinality == 0 {
                                true
                            } else {
                                receipt.candidate_cardinality % receipt.input_cardinality == 0
                            };
                        if filtered_candidates.is_none()
                            && receipt.candidate_cardinality != 0
                            && receipt.matched_input_cardinality != receipt.input_cardinality
                        {
                            return Err(Error::new(
                                ErrorCode::CorruptStorage,
                                "resident unfiltered node-scan receipt omitted a matched input",
                            ));
                        }
                        if optional_group.is_some() {
                            if *mode != ResidentNullableRelationMatchMode::Optional
                                || receipt.null_extension_cardinality != 0
                                || receipt.output_cardinality != receipt.candidate_cardinality
                                || !exact_cartesian_shape
                                || (labels.is_known_empty() && receipt.candidate_cardinality != 0)
                            {
                                return Err(Error::new(
                                    ErrorCode::CorruptStorage,
                                    "resident atomic OPTIONAL node seed performed independent null extension",
                                ));
                            }
                        } else {
                            match mode {
                                ResidentNullableRelationMatchMode::Mandatory => {
                                    if receipt.null_extension_cardinality != 0
                                        || receipt.output_cardinality
                                            != receipt.candidate_cardinality
                                        || !exact_cartesian_shape
                                        || (labels.is_known_empty()
                                            && receipt.candidate_cardinality != 0)
                                    {
                                        return Err(Error::new(
                                            ErrorCode::CorruptStorage,
                                            "resident mandatory node-scan receipt violates candidate/drop semantics",
                                        ));
                                    }
                                }
                                ResidentNullableRelationMatchMode::Optional => {
                                    let exact_nulls = receipt
                                    .input_cardinality
                                    .checked_sub(receipt.matched_input_cardinality)
                                    .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "resident optional node-scan matched-input count exceeds input"))?;
                                    if receipt.null_extension_cardinality != exact_nulls
                                    || receipt.output_cardinality
                                        != receipt
                                            .candidate_cardinality
                                            .checked_add(receipt.null_extension_cardinality)
                                            .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "resident optional node-scan output overflow"))?
                                    || receipt.output_cardinality < receipt.input_cardinality
                                    || !exact_cartesian_shape
                                    || (labels.is_known_empty()
                                        && receipt.candidate_cardinality != 0)
                                {
                                    return Err(Error::new(
                                        ErrorCode::CorruptStorage,
                                        "resident optional node-scan receipt violates null-extension semantics",
                                    ));
                                }
                                }
                            }
                        }
                    }
                    ResidentNullableRelationStage::Expand {
                        mode,
                        source_labels,
                        relationship_types,
                        target_labels,
                        ..
                    } => {
                        let known_empty = source_labels.is_known_empty()
                            || relationship_types.is_known_empty()
                            || target_labels.is_known_empty();
                        if optional_group.is_some() {
                            if *mode != ResidentNullableRelationMatchMode::Optional
                                || receipt.null_extension_cardinality != 0
                                || receipt.output_cardinality != receipt.candidate_cardinality
                                || (known_empty && receipt.candidate_cardinality != 0)
                            {
                                return Err(Error::new(
                                    ErrorCode::CorruptStorage,
                                    "resident atomic OPTIONAL hop performed independent null extension",
                                ));
                            }
                        } else {
                            match mode {
                                ResidentNullableRelationMatchMode::Mandatory => {
                                    if receipt.null_extension_cardinality != 0
                                        || receipt.output_cardinality
                                            != receipt.candidate_cardinality
                                        || (known_empty && receipt.candidate_cardinality != 0)
                                    {
                                        return Err(Error::new(
                                            ErrorCode::CorruptStorage,
                                            "resident mandatory expansion receipt violates candidate/drop semantics",
                                        ));
                                    }
                                }
                                ResidentNullableRelationMatchMode::Optional => {
                                    let exact_nulls = receipt
                                    .input_cardinality
                                    .checked_sub(receipt.matched_input_cardinality)
                                    .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "resident optional expansion matched-input count exceeds input"))?;
                                    if receipt.null_extension_cardinality != exact_nulls
                                    || receipt.output_cardinality
                                        != receipt
                                            .candidate_cardinality
                                            .checked_add(receipt.null_extension_cardinality)
                                            .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "resident optional expansion output overflow"))?
                                    || receipt.output_cardinality < receipt.input_cardinality
                                    || (known_empty && receipt.candidate_cardinality != 0)
                                {
                                    return Err(Error::new(
                                        ErrorCode::CorruptStorage,
                                        "resident optional expansion receipt violates null-extension semantics",
                                    ));
                                }
                                }
                            }
                        }
                    }
                    ResidentNullableRelationStage::ScopeProject { bindings } => {
                        let expected_output = bindings[0]
                            .row_limit
                            .map_or(receipt.input_cardinality, |limit| {
                                receipt.input_cardinality.min(u64::from(limit))
                            });
                        if receipt.null_extension_cardinality != 0
                            || receipt.matched_input_cardinality != expected_output
                            || receipt.candidate_cardinality != expected_output
                            || receipt.output_cardinality != expected_output
                        {
                            return Err(Error::new(
                                ErrorCode::CorruptStorage,
                                "resident nullable scope projection receipt violates its stable limit",
                            ));
                        }
                    }
                    ResidentNullableRelationStage::FinalProject { .. } => {
                        if receipt.null_extension_cardinality != 0
                            || receipt.matched_input_cardinality != receipt.input_cardinality
                            || receipt.candidate_cardinality != receipt.input_cardinality
                            || receipt.output_cardinality != receipt.input_cardinality
                        {
                            return Err(Error::new(
                                ErrorCode::CorruptStorage,
                                "resident nullable final projection receipt changed relation cardinality",
                            ));
                        }
                    }
                }
                expected_input = receipt.output_cardinality;
            }
            ResidentNullableRelationProofAction::OptionalGroup(group_index) => {
                let group = request
                    .predicate_program
                    .optional_groups
                    .get(group_index)
                    .ok_or_else(|| {
                        Error::internal("resident nullable receipt group disappeared")
                    })?;
                let parent_input = optional_group_parent_inputs
                    .remove(&group_index)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::CorruptStorage,
                            "resident atomic OPTIONAL omitted its parent input proof",
                        )
                    })?;
                let complete_candidates = candidate_filter_outputs
                    .get(&(2_u8, group_index))
                    .copied()
                    .unwrap_or(expected_input);
                let output_capacity =
                    u64::from(request.capacities.stage_output_rows[usize::from(group.last_stage)]);
                if receipt.input_cardinality != parent_input
                    || receipt.output_cardinality > output_capacity
                    || receipt
                        .matched_input_cardinality
                        .checked_add(receipt.null_extension_cardinality)
                        != Some(receipt.input_cardinality)
                    || receipt.candidate_cardinality != complete_candidates
                    || receipt.candidate_cardinality < receipt.matched_input_cardinality
                    || receipt.output_cardinality
                        != receipt
                            .candidate_cardinality
                            .checked_add(receipt.null_extension_cardinality)
                            .ok_or_else(|| {
                                Error::new(
                                    ErrorCode::CorruptStorage,
                                    "resident atomic OPTIONAL output overflow",
                                )
                            })?
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident atomic OPTIONAL receipt violates complete-path null-extension semantics",
                    ));
                }
                expected_input = receipt.output_cardinality;
            }
        }
    }
    if !optional_group_parent_inputs.is_empty() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "resident nullable relation left an atomic OPTIONAL parent proof unfinished",
        ));
    }
    Ok(())
}

fn nullable_relation_fingerprint(
    generation: ResidentNullableRelationGeneration,
    execution: ResidentExecutionId,
    program: &ResidentNullableRelationProgram,
    predicates: &ResidentNullableRelationPredicateProgram,
    property_lanes: &ResidentNullableRelationPropertyLaneRegistry,
    capacities: &ResidentNullableRelationCapacities,
    manifest: &ResidentNullableRelationManifest,
) -> Result<ResidentNullableRelationFingerprint> {
    let mut hash = blake3::Hasher::new_derive_key("irongraph.resident-nullable-relation.v7");
    hash.update(generation.project.0.as_bytes());
    nullable_hash_u64(&mut hash, generation.bookmark.term);
    nullable_hash_u64(&mut hash, generation.bookmark.index);
    nullable_hash_u64(&mut hash, generation.graph_revision);
    nullable_hash_u64(&mut hash, generation.layout_version);
    hash.update(&generation.catalog_generation);
    nullable_hash_u64(&mut hash, execution.high);
    nullable_hash_u64(&mut hash, execution.low);
    hash.update(&[program.layers.bits()]);
    nullable_hash_len(&mut hash, program.stages.len())?;
    for stage in &program.stages {
        match stage {
            ResidentNullableRelationStage::NodeScan {
                mode,
                output,
                labels,
            } => {
                hash.update(&[1, *mode as u8]);
                nullable_hash_slot(&mut hash, *output);
                nullable_hash_node_domain(&mut hash, labels)?;
            }
            ResidentNullableRelationStage::Expand {
                mode,
                uniqueness_group,
                source,
                source_labels,
                relationship,
                different_from,
                target,
                direction,
                relationship_types,
                target_labels,
            } => {
                hash.update(&[2, *mode as u8]);
                nullable_hash_u64(&mut hash, u64::from(*uniqueness_group));
                nullable_hash_slot(&mut hash, *source);
                nullable_hash_node_domain(&mut hash, source_labels)?;
                match relationship {
                    Some(slot) => {
                        hash.update(&[1]);
                        nullable_hash_slot(&mut hash, *slot);
                    }
                    None => {
                        hash.update(&[0]);
                    }
                }
                nullable_hash_len(&mut hash, different_from.len())?;
                for slot in different_from {
                    nullable_hash_slot(&mut hash, *slot);
                }
                match target {
                    ResidentNullableRelationTarget::Existing(slot) => {
                        hash.update(&[1]);
                        nullable_hash_slot(&mut hash, *slot);
                    }
                    ResidentNullableRelationTarget::Introduce(slot) => {
                        hash.update(&[2]);
                        nullable_hash_slot(&mut hash, *slot);
                    }
                }
                hash.update(&[match direction {
                    ResidentDirection::Outgoing => 1,
                    ResidentDirection::Incoming => 2,
                    ResidentDirection::Undirected => 3,
                }]);
                nullable_hash_relationship_domain(&mut hash, relationship_types)?;
                nullable_hash_node_domain(&mut hash, target_labels)?;
            }
            ResidentNullableRelationStage::ScopeProject { bindings } => {
                hash.update(&[3]);
                nullable_hash_len(&mut hash, bindings.len())?;
                for binding in bindings {
                    nullable_hash_bytes(&mut hash, binding.variable.as_bytes())?;
                    nullable_hash_slot(&mut hash, binding.source);
                    nullable_hash_slot(&mut hash, binding.output);
                    match binding.row_limit {
                        Some(limit) => {
                            hash.update(&[1]);
                            nullable_hash_u64(&mut hash, u64::from(limit));
                        }
                        None => {
                            hash.update(&[0]);
                        }
                    }
                }
            }
            ResidentNullableRelationStage::FinalProject { bindings } => {
                hash.update(&[4]);
                nullable_hash_len(&mut hash, bindings.len())?;
                for binding in bindings {
                    nullable_hash_bytes(&mut hash, binding.name.as_bytes())?;
                    match &binding.source {
                        ResidentNullableRelationOutputSource::Entity { slot, kind } => {
                            hash.update(&[1, *kind as u8]);
                            nullable_hash_slot(&mut hash, *slot);
                        }
                        ResidentNullableRelationOutputSource::IntegerProperty {
                            slot,
                            kind,
                            property,
                        } => {
                            hash.update(&[2, *kind as u8]);
                            nullable_hash_slot(&mut hash, *slot);
                            nullable_hash_u64(&mut hash, property.0);
                        }
                        ResidentNullableRelationOutputSource::StringPropertyToInteger {
                            slot,
                            kind,
                            property,
                        } => {
                            hash.update(&[8, *kind as u8]);
                            nullable_hash_slot(&mut hash, *slot);
                            nullable_hash_u64(&mut hash, property.0);
                        }
                        ResidentNullableRelationOutputSource::FloatProperty {
                            slot,
                            kind,
                            property,
                        } => {
                            hash.update(&[6, *kind as u8]);
                            nullable_hash_slot(&mut hash, *slot);
                            nullable_hash_u64(&mut hash, property.0);
                        }
                        ResidentNullableRelationOutputSource::IntegerPropertyToFloat {
                            slot,
                            kind,
                            property,
                        } => {
                            hash.update(&[9, *kind as u8]);
                            nullable_hash_slot(&mut hash, *slot);
                            nullable_hash_u64(&mut hash, property.0);
                        }
                        ResidentNullableRelationOutputSource::StringProperty {
                            slot,
                            kind,
                            property,
                            maximum_bytes,
                        } => {
                            hash.update(&[3, *kind as u8]);
                            nullable_hash_slot(&mut hash, *slot);
                            nullable_hash_u64(&mut hash, property.0);
                            hash.update(&maximum_bytes.to_le_bytes());
                        }
                        ResidentNullableRelationOutputSource::IntegerPropertyToString {
                            slot,
                            kind,
                            property,
                        } => {
                            hash.update(&[10, *kind as u8]);
                            nullable_hash_slot(&mut hash, *slot);
                            nullable_hash_u64(&mut hash, property.0);
                        }
                        ResidentNullableRelationOutputSource::NullProperty { slot, kind } => {
                            hash.update(&[4, *kind as u8]);
                            nullable_hash_slot(&mut hash, *slot);
                        }
                        ResidentNullableRelationOutputSource::RelationshipType { slot } => {
                            hash.update(&[
                                5,
                                ResidentNullableRelationBindingKind::Relationship as u8,
                            ]);
                            nullable_hash_slot(&mut hash, *slot);
                        }
                        ResidentNullableRelationOutputSource::EntityLabelPredicate {
                            slot,
                            domain,
                        } => {
                            hash.update(&[7, domain.kind() as u8]);
                            nullable_hash_slot(&mut hash, *slot);
                            match domain {
                                ResidentNullableRelationEntityLabelDomain::Node(domain) => {
                                    hash.update(&[1]);
                                    nullable_hash_node_domain(&mut hash, domain)?;
                                }
                                ResidentNullableRelationEntityLabelDomain::Relationship(domain) => {
                                    hash.update(&[2]);
                                    nullable_hash_relationship_domain(&mut hash, domain)?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    nullable_hash_len(&mut hash, predicates.filters.len())?;
    for filter in &predicates.filters {
        match filter.placement {
            ResidentNullableRelationFilterPlacement::RelationAfter { stage } => {
                hash.update(&[1]);
                hash.update(&stage.to_le_bytes());
            }
            ResidentNullableRelationFilterPlacement::OptionalCandidates { stage } => {
                hash.update(&[2]);
                hash.update(&stage.to_le_bytes());
            }
            ResidentNullableRelationFilterPlacement::OptionalGroupCandidates { group } => {
                hash.update(&[3]);
                hash.update(&group.to_le_bytes());
            }
        }
        nullable_hash_predicate(&mut hash, &filter.predicate)?;
    }
    nullable_hash_len(&mut hash, predicates.optional_groups.len())?;
    for group in &predicates.optional_groups {
        hash.update(&group.first_stage.to_le_bytes());
        hash.update(&group.last_stage.to_le_bytes());
    }
    nullable_hash_len(&mut hash, property_lanes.lanes.len())?;
    for lane in &property_lanes.lanes {
        hash.update(&[lane.kind as u8]);
        nullable_hash_u64(&mut hash, lane.property.0);
        hash.update(&[lane.shape as u8]);
    }
    for value in [
        capacities.node_slot_count,
        capacities.relationship_slot_count,
        capacities.visible_node_rows,
        capacities.visible_relationship_rows,
        u32::from(capacities.binding_slot_count),
        u32::from(capacities.maximum_live_bindings),
        capacities.max_output_rows,
        u32::from(capacities.final_output_columns),
    ] {
        hash.update(&value.to_le_bytes());
    }
    nullable_hash_u64(&mut hash, capacities.final_output_bytes_per_row);
    nullable_hash_u64(&mut hash, capacities.final_output_fixed_bytes);
    nullable_hash_u16_slice(&mut hash, &capacities.stage_live_bindings)?;
    nullable_hash_u64_slice(&mut hash, &capacities.stage_input_rows)?;
    nullable_hash_u64_slice(&mut hash, &capacities.stage_candidate_rows)?;
    nullable_hash_u64_slice(&mut hash, &capacities.stage_output_rows)?;
    nullable_hash_len(&mut hash, manifest.obligations.len())?;
    for obligation in &manifest.obligations {
        nullable_hash_u64(&mut hash, obligation.id);
        hash.update(&obligation.stage.to_le_bytes());
        hash.update(&[obligation.kind as u8]);
    }
    Ok(ResidentNullableRelationFingerprint(
        *hash.finalize().as_bytes(),
    ))
}

fn nullable_hash_predicate(
    hash: &mut blake3::Hasher,
    predicate: &ResidentNullableRelationPredicate,
) -> Result<()> {
    match predicate {
        ResidentNullableRelationPredicate::Constant(value) => {
            hash.update(&[1, value.map_or(0, |value| if value { 2 } else { 1 })]);
        }
        ResidentNullableRelationPredicate::IsNull { value, negated } => {
            hash.update(&[2, u8::from(*negated)]);
            nullable_hash_predicate_value(hash, value)?;
        }
        ResidentNullableRelationPredicate::HasLabels { node, labels } => {
            hash.update(&[3]);
            nullable_hash_slot(hash, *node);
            nullable_hash_node_domain(hash, labels)?;
        }
        ResidentNullableRelationPredicate::CompareInteger {
            left,
            operation,
            right,
        } => {
            hash.update(&[4, *operation as u8]);
            nullable_hash_predicate_value(hash, left)?;
            nullable_hash_predicate_value(hash, right)?;
        }
        ResidentNullableRelationPredicate::CompareString {
            left,
            operation,
            right,
        } => {
            hash.update(&[9, *operation as u8]);
            nullable_hash_predicate_value(hash, left)?;
            nullable_hash_predicate_value(hash, right)?;
        }
        ResidentNullableRelationPredicate::RelationshipEndpoint {
            relationship,
            node,
            endpoint,
        } => {
            hash.update(&[8, *endpoint as u8]);
            nullable_hash_slot(hash, *relationship);
            nullable_hash_slot(hash, *node);
        }
        ResidentNullableRelationPredicate::Not(operand) => {
            hash.update(&[5]);
            nullable_hash_predicate(hash, operand)?;
        }
        ResidentNullableRelationPredicate::And(left, right) => {
            hash.update(&[6]);
            nullable_hash_predicate(hash, left)?;
            nullable_hash_predicate(hash, right)?;
        }
        ResidentNullableRelationPredicate::Or(left, right) => {
            hash.update(&[7]);
            nullable_hash_predicate(hash, left)?;
            nullable_hash_predicate(hash, right)?;
        }
    }
    Ok(())
}

fn nullable_hash_predicate_value(
    hash: &mut blake3::Hasher,
    value: &ResidentNullableRelationPredicateValue,
) -> Result<()> {
    match value {
        ResidentNullableRelationPredicateValue::Null => {
            hash.update(&[1]);
        }
        ResidentNullableRelationPredicateValue::Boolean(value) => {
            hash.update(&[2, u8::from(*value)]);
        }
        ResidentNullableRelationPredicateValue::Integer(value) => {
            hash.update(&[3]);
            hash.update(&value.to_le_bytes());
        }
        ResidentNullableRelationPredicateValue::String(value) => {
            hash.update(&[6]);
            nullable_hash_len(hash, value.len())?;
            hash.update(value.as_bytes());
        }
        ResidentNullableRelationPredicateValue::Binding { slot, kind } => {
            hash.update(&[4, *kind as u8]);
            nullable_hash_slot(hash, *slot);
        }
        ResidentNullableRelationPredicateValue::IntegerProperty {
            slot,
            kind,
            property,
        } => {
            hash.update(&[5, *kind as u8]);
            nullable_hash_slot(hash, *slot);
            nullable_hash_u64(hash, property.0);
        }
        ResidentNullableRelationPredicateValue::StringProperty {
            slot,
            kind,
            property,
        } => {
            hash.update(&[7, *kind as u8]);
            nullable_hash_slot(hash, *slot);
            nullable_hash_u64(hash, property.0);
        }
    }
    Ok(())
}

fn nullable_hash_node_domain(
    hash: &mut blake3::Hasher,
    domain: &ResidentNullableNodeDomain,
) -> Result<()> {
    match domain {
        ResidentNullableNodeDomain::Any => {
            hash.update(&[1]);
        }
        ResidentNullableNodeDomain::Known(labels) => {
            hash.update(&[2]);
            nullable_hash_len(hash, labels.len())?;
            for label in labels {
                nullable_hash_u64(hash, label.0);
            }
        }
        ResidentNullableNodeDomain::KnownEmpty => {
            hash.update(&[3]);
        }
    }
    Ok(())
}

fn nullable_hash_relationship_domain(
    hash: &mut blake3::Hasher,
    domain: &ResidentNullableRelationshipDomain,
) -> Result<()> {
    match domain {
        ResidentNullableRelationshipDomain::Any => {
            hash.update(&[1]);
        }
        ResidentNullableRelationshipDomain::Known(types) => {
            hash.update(&[2]);
            nullable_hash_len(hash, types.len())?;
            for relationship_type in types {
                nullable_hash_u64(hash, relationship_type.0);
            }
        }
        ResidentNullableRelationshipDomain::KnownEmpty => {
            hash.update(&[3]);
        }
    }
    Ok(())
}

fn nullable_hash_u16_slice(hash: &mut blake3::Hasher, values: &[u16]) -> Result<()> {
    nullable_hash_len(hash, values.len())?;
    for value in values {
        hash.update(&value.to_le_bytes());
    }
    Ok(())
}

fn nullable_hash_u64_slice(hash: &mut blake3::Hasher, values: &[u64]) -> Result<()> {
    nullable_hash_len(hash, values.len())?;
    for value in values {
        hash.update(&value.to_le_bytes());
    }
    Ok(())
}

fn nullable_hash_slot(hash: &mut blake3::Hasher, slot: ResidentNullableRelationSlot) {
    hash.update(&slot.0.to_le_bytes());
}

fn nullable_hash_bytes(hash: &mut blake3::Hasher, bytes: &[u8]) -> Result<()> {
    nullable_hash_len(hash, bytes.len())?;
    hash.update(bytes);
    Ok(())
}

fn nullable_hash_len(hash: &mut blake3::Hasher, value: usize) -> Result<()> {
    nullable_hash_u64(
        hash,
        u64::try_from(value).map_err(|_| nullable_relation_scratch_overflow())?,
    );
    Ok(())
}

fn nullable_hash_u64(hash: &mut blake3::Hasher, value: u64) {
    hash.update(&value.to_le_bytes());
}

fn nullable_relation_product(left: u64, right: u64) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(nullable_relation_scratch_overflow)
}

fn nullable_relation_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            format!("resident nullable relation {name} exceeds u32"),
        )
    })
}

fn nullable_relation_scratch_overflow() -> Error {
    Error::new(
        ErrorCode::ResultBudgetExceeded,
        "resident nullable relation capacity or scratch accounting overflow",
    )
}

/// One nullable typed payload column. Float payloads retain their exact IEEE-754 bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentRowColumn {
    Boolean {
        values: Vec<u8>,
        validity: Vec<u8>,
    },
    Integer {
        values: Vec<i64>,
        validity: Vec<u8>,
    },
    Float {
        bits: Vec<u64>,
        validity: Vec<u8>,
    },
    /// Canonical row-major UTF-8. Offsets contain one range per row; NULL rows own an empty range
    /// so hidden payload bytes can never survive validity changes.
    String {
        offsets: Vec<u32>,
        bytes: Vec<u8>,
        validity: Vec<u8>,
    },
    /// Canonical row-major flat INTEGER lists. Offsets contain one range per row; NULL lists own
    /// an empty range. Element validity is independent so computed list literals may retain null
    /// elements, whose hidden integer payload is canonically zero.
    List {
        offsets: Vec<u32>,
        values: Vec<i64>,
        element_validity: Vec<u8>,
        validity: Vec<u8>,
    },
    Date {
        days: Vec<i64>,
        validity: Vec<u8>,
    },
    LocalTime {
        nanos: Vec<i64>,
        validity: Vec<u8>,
    },
    ZonedTime {
        nanos: Vec<i64>,
        offset_seconds: Vec<i32>,
        validity: Vec<u8>,
    },
    LocalDateTime {
        seconds: Vec<i64>,
        nanos: Vec<u32>,
        validity: Vec<u8>,
    },
    /// Canonical `(seconds, nanos, timezone UTF-8)` tuples. Timezone offsets contain one range per
    /// row and NULL rows own an empty range, exactly like STRING columns.
    ZonedDateTime {
        seconds: Vec<i64>,
        nanos: Vec<u32>,
        timezone_offsets: Vec<u32>,
        timezone_bytes: Vec<u8>,
        validity: Vec<u8>,
    },
}

impl ResidentRowColumn {
    #[must_use]
    pub const fn value_type(&self) -> ResidentRowValueType {
        match self {
            Self::Boolean { .. } => ResidentRowValueType::Boolean,
            Self::Integer { .. } => ResidentRowValueType::Integer,
            Self::Float { .. } => ResidentRowValueType::Float,
            Self::String { .. } => ResidentRowValueType::String,
            Self::List { .. } => ResidentRowValueType::List,
            Self::Date { .. } => ResidentRowValueType::Date,
            Self::LocalTime { .. } => ResidentRowValueType::LocalTime,
            Self::ZonedTime { .. } => ResidentRowValueType::ZonedTime,
            Self::LocalDateTime { .. } => ResidentRowValueType::LocalDateTime,
            Self::ZonedDateTime { .. } => ResidentRowValueType::ZonedDateTime,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Boolean { values, .. } => values.len(),
            Self::Integer { values, .. } => values.len(),
            Self::Float { bits, .. } => bits.len(),
            Self::String { validity, .. } => validity.len(),
            Self::List { validity, .. } => validity.len(),
            Self::Date { days, .. } => days.len(),
            Self::LocalTime { nanos, .. } | Self::ZonedTime { nanos, .. } => nanos.len(),
            Self::LocalDateTime { seconds, .. } | Self::ZonedDateTime { seconds, .. } => {
                seconds.len()
            }
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn validity(&self) -> &[u8] {
        match self {
            Self::Boolean { validity, .. }
            | Self::Integer { validity, .. }
            | Self::Float { validity, .. }
            | Self::String { validity, .. }
            | Self::List { validity, .. }
            | Self::Date { validity, .. }
            | Self::LocalTime { validity, .. }
            | Self::ZonedTime { validity, .. }
            | Self::LocalDateTime { validity, .. }
            | Self::ZonedDateTime { validity, .. } => validity,
        }
    }

    fn validate(&self, expected_type: ResidentRowValueType, rows: usize) -> Result<()> {
        if self.value_type() != expected_type
            || self.len() != rows
            || self.validity().len() != rows
            || self.validity().iter().any(|valid| *valid > 1)
            || matches!(self, Self::Boolean { values, .. } if values.iter().any(|value| *value > 1))
            || matches!(self, Self::ZonedTime { offset_seconds, .. } if offset_seconds.len() != rows)
            || matches!(self, Self::LocalDateTime { nanos, .. } if nanos.len() != rows)
            || matches!(self, Self::ZonedDateTime { nanos, .. } if nanos.len() != rows)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident row typed column has an invalid type, payload, or validity shape",
            ));
        }
        match self {
            Self::String {
                offsets,
                bytes,
                validity,
            } => {
                if !canonical_utf8_ranges(offsets, bytes, validity, rows) {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident row string column has an invalid canonical UTF-8 shape",
                    ));
                }
            }
            Self::ZonedDateTime {
                timezone_offsets,
                timezone_bytes,
                validity,
                ..
            } => {
                if !canonical_utf8_ranges(timezone_offsets, timezone_bytes, validity, rows) {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident row zoned-datetime column has an invalid canonical timezone shape",
                    ));
                }
            }
            Self::List {
                offsets,
                values,
                element_validity,
                validity,
            } if !canonical_list_ranges(offsets, values, element_validity, validity, rows) => {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row list column has an invalid canonical shape",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn select(&self, positions: &[u64]) -> Result<Self> {
        let select_validity = |validity: &[u8]| -> Result<Vec<u8>> {
            positions
                .iter()
                .map(|position| {
                    let position = usize::try_from(*position).map_err(|_| {
                        Error::new(
                            ErrorCode::CorruptStorage,
                            "resident row projection position exceeds host addressability",
                        )
                    })?;
                    validity.get(position).copied().ok_or_else(|| {
                        Error::new(
                            ErrorCode::CorruptStorage,
                            "resident row projection position is out of bounds",
                        )
                    })
                })
                .collect()
        };
        match self {
            Self::Boolean { values, validity } => Ok(Self::Boolean {
                values: select_values(values, positions)?,
                validity: select_validity(validity)?,
            }),
            Self::Integer { values, validity } => Ok(Self::Integer {
                values: select_values(values, positions)?,
                validity: select_validity(validity)?,
            }),
            Self::Float { bits, validity } => Ok(Self::Float {
                bits: select_values(bits, positions)?,
                validity: select_validity(validity)?,
            }),
            Self::Date { days, validity } => Ok(Self::Date {
                days: select_values(days, positions)?,
                validity: select_validity(validity)?,
            }),
            Self::LocalTime { nanos, validity } => Ok(Self::LocalTime {
                nanos: select_values(nanos, positions)?,
                validity: select_validity(validity)?,
            }),
            Self::ZonedTime {
                nanos,
                offset_seconds,
                validity,
            } => Ok(Self::ZonedTime {
                nanos: select_values(nanos, positions)?,
                offset_seconds: select_values(offset_seconds, positions)?,
                validity: select_validity(validity)?,
            }),
            Self::LocalDateTime {
                seconds,
                nanos,
                validity,
            } => Ok(Self::LocalDateTime {
                seconds: select_values(seconds, positions)?,
                nanos: select_values(nanos, positions)?,
                validity: select_validity(validity)?,
            }),
            Self::String {
                offsets,
                bytes,
                validity,
            } => {
                let (selected_offsets, selected_bytes, selected_validity) =
                    select_utf8_ranges(offsets, bytes, validity, positions)?;
                Ok(Self::String {
                    offsets: selected_offsets,
                    bytes: selected_bytes,
                    validity: selected_validity,
                })
            }
            Self::List {
                offsets,
                values,
                element_validity,
                validity,
            } => {
                let (selected_offsets, selected_values, selected_element_validity, validity) =
                    select_list_ranges(offsets, values, element_validity, validity, positions)?;
                Ok(Self::List {
                    offsets: selected_offsets,
                    values: selected_values,
                    element_validity: selected_element_validity,
                    validity,
                })
            }
            Self::ZonedDateTime {
                seconds,
                nanos,
                timezone_offsets,
                timezone_bytes,
                validity,
            } => {
                let (selected_offsets, selected_bytes, selected_validity) =
                    select_utf8_ranges(timezone_offsets, timezone_bytes, validity, positions)?;
                Ok(Self::ZonedDateTime {
                    seconds: select_values(seconds, positions)?,
                    nanos: select_values(nanos, positions)?,
                    timezone_offsets: selected_offsets,
                    timezone_bytes: selected_bytes,
                    validity: selected_validity,
                })
            }
        }
    }
}

fn canonical_list_ranges(
    offsets: &[u32],
    values: &[i64],
    element_validity: &[u8],
    validity: &[u8],
    rows: usize,
) -> bool {
    offsets.len() == rows.saturating_add(1)
        && offsets.first().copied() == Some(0)
        && offsets.last().copied().map(|offset| offset as usize) == Some(values.len())
        && element_validity.len() == values.len()
        && element_validity.iter().all(|valid| *valid <= 1)
        && values
            .iter()
            .zip(element_validity)
            .all(|(value, valid)| *valid != 0 || *value == 0)
        && offsets.windows(2).all(|pair| pair[0] <= pair[1])
        && offsets
            .windows(2)
            .zip(validity)
            .all(|(pair, valid)| *valid != 0 || pair[0] == pair[1])
}

fn select_list_ranges(
    offsets: &[u32],
    values: &[i64],
    element_validity: &[u8],
    validity: &[u8],
    positions: &[u64],
) -> Result<(Vec<u32>, Vec<i64>, Vec<u8>, Vec<u8>)> {
    let mut selected_offsets = Vec::with_capacity(positions.len().saturating_add(1));
    let mut selected_values = Vec::new();
    let mut selected_element_validity = Vec::new();
    let mut selected_validity = Vec::with_capacity(positions.len());
    selected_offsets.push(0);
    for position in positions {
        let row = usize::try_from(*position).map_err(|_| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row list projection position exceeds host addressability",
            )
        })?;
        let start = *offsets.get(row).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row list projection position is out of bounds",
            )
        })? as usize;
        let end = *offsets.get(row.saturating_add(1)).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row list projection position is out of bounds",
            )
        })? as usize;
        selected_values.extend_from_slice(values.get(start..end).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row list projection range is out of bounds",
            )
        })?);
        selected_element_validity.extend_from_slice(element_validity.get(start..end).ok_or_else(
            || {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row list element validity range is out of bounds",
                )
            },
        )?);
        selected_offsets.push(u32::try_from(selected_values.len()).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident row selected list elements exceed u32",
            )
        })?);
        selected_validity.push(*validity.get(row).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row list projection validity is out of bounds",
            )
        })?);
    }
    Ok((
        selected_offsets,
        selected_values,
        selected_element_validity,
        selected_validity,
    ))
}

fn canonical_utf8_ranges(offsets: &[u32], bytes: &[u8], validity: &[u8], rows: usize) -> bool {
    offsets.len() == rows.saturating_add(1)
        && offsets.first().copied() == Some(0)
        && offsets.last().copied().map(|offset| offset as usize) == Some(bytes.len())
        && offsets.windows(2).all(|pair| pair[0] <= pair[1])
        && offsets.windows(2).zip(validity).all(|(pair, valid)| {
            if *valid == 0 && pair[0] != pair[1] {
                return false;
            }
            let start = pair[0] as usize;
            let end = pair[1] as usize;
            bytes
                .get(start..end)
                .is_some_and(|value| std::str::from_utf8(value).is_ok())
        })
}

fn select_utf8_ranges(
    offsets: &[u32],
    bytes: &[u8],
    validity: &[u8],
    positions: &[u64],
) -> Result<(Vec<u32>, Vec<u8>, Vec<u8>)> {
    let mut selected_offsets = Vec::with_capacity(positions.len().saturating_add(1));
    let mut selected_bytes = Vec::new();
    let mut selected_validity = Vec::with_capacity(positions.len());
    selected_offsets.push(0);
    for position in positions {
        let row = usize::try_from(*position).map_err(|_| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row UTF-8 projection position exceeds host addressability",
            )
        })?;
        let start = *offsets.get(row).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row UTF-8 projection position is out of bounds",
            )
        })? as usize;
        let end = *offsets.get(row.saturating_add(1)).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row UTF-8 projection position is out of bounds",
            )
        })? as usize;
        selected_bytes.extend_from_slice(bytes.get(start..end).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row UTF-8 projection range is out of bounds",
            )
        })?);
        selected_offsets.push(u32::try_from(selected_bytes.len()).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident row selected UTF-8 bytes exceed u32",
            )
        })?);
        selected_validity.push(*validity.get(row).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident row UTF-8 projection validity is out of bounds",
            )
        })?);
    }
    Ok((selected_offsets, selected_bytes, selected_validity))
}

/// One selected final register, retaining projection order and duplicate selections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentRowProjectedColumn {
    pub register: u16,
    pub column: ResidentRowColumn,
}

/// Publicly constructible untrusted parts support backend bring-up and corruption tests. Values
/// remain unusable as validated query rows until [`ResidentRowProgramResult::validate`] succeeds.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentRowProgramResultParts {
    pub project: ProjectId,
    pub execution: ResidentExecutionId,
    pub bookmark: Bookmark,
    pub graph_revision: u64,
    pub layout_version: ResidentGraphLayoutVersion,
    pub manifest_fingerprint: ResidentRowManifestFingerprint,
    pub input_cardinality: usize,
    pub source_positions: Vec<u64>,
    pub rows: ResidentNodePipelineResult,
    pub projected_columns: Vec<ResidentRowProjectedColumn>,
    pub receipts: Vec<ResidentExecutionReceipt>,
    pub scratch_bytes: usize,
}

/// Raw backend output. Access to row values requires provenance and shape validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentRowProgramResult {
    parts: ResidentRowProgramResultParts,
}

/// Receipt-validated row-program output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedResidentRowProgramResult {
    parts: ResidentRowProgramResultParts,
}

impl ResidentRowProgramResult {
    pub fn completed(parts: ResidentRowProgramResultParts) -> Self {
        Self { parts }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn from_untrusted_parts(parts: ResidentRowProgramResultParts) -> Self {
        Self { parts }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn into_untrusted_parts(self) -> ResidentRowProgramResultParts {
        self.parts
    }

    pub fn validate(
        self,
        request: &ResidentRowProgramRequest,
        backend: BackendKind,
    ) -> Result<ValidatedResidentRowProgramResult> {
        request.validate()?;
        let mut parts = self.parts;
        if (
            parts.project,
            parts.execution,
            parts.bookmark,
            parts.graph_revision,
            parts.layout_version,
            parts.manifest_fingerprint,
        ) != (
            request.project,
            request.execution,
            request.expected_bookmark,
            request.expected_graph_revision,
            request.expected_layout_version,
            request.manifest.fingerprint,
        ) {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident row result belongs to a different execution, manifest, or graph image",
            ));
        }
        let output_rows = request.output_cardinality(parts.input_cardinality)?;
        if parts.source_positions.len() != output_rows {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident row result has an invalid source-position cardinality",
            ));
        }
        let mut seen_positions = BTreeSet::new();
        for position in &parts.source_positions {
            if usize::try_from(*position)
                .map_or(true, |position| position >= parts.input_cardinality)
                || !seen_positions.insert(*position)
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row result has an invalid or duplicate source position",
                ));
            }
        }
        if request.sort_keys.is_empty() {
            let expected = (request.offset..request.offset.saturating_add(output_rows))
                .map(|position| u64::try_from(position).map_err(|_| scratch_overflow()))
                .collect::<Result<Vec<_>>>()?;
            if parts.source_positions != expected {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "unsorted resident row result does not preserve stable input order",
                ));
            }
        }
        validate_base_rows(request, &parts.rows, output_rows)?;
        if parts.projected_columns.len() != request.final_registers.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident row result omitted a selected projection register",
            ));
        }
        for (projected, expected_register) in
            parts.projected_columns.iter().zip(&request.final_registers)
        {
            let expected_type = request
                .program
                .register_type(*expected_register)
                .ok_or_else(|| Error::internal("validated row projection register disappeared"))?;
            if projected.register != *expected_register {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row projections are not in the selected register order",
                ));
            }
            projected.column.validate(expected_type, output_rows)?;
            if expected_type == ResidentRowValueType::ZonedDateTime {
                let declared = request
                    .program
                    .timezone_register_capacity(*expected_register)?
                    .ok_or_else(|| {
                        Error::internal("validated projected timezone capacity disappeared")
                    })?;
                let ResidentRowColumn::ZonedDateTime {
                    timezone_offsets, ..
                } = &projected.column
                else {
                    return Err(Error::internal(
                        "validated zoned-datetime projection changed physical type",
                    ));
                };
                if timezone_offsets
                    .windows(2)
                    .any(|pair| (pair[1] - pair[0]) as usize > declared)
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident row zoned-datetime projection exceeds its admitted timezone width",
                    ));
                }
            }
            if expected_type == ResidentRowValueType::List {
                let declared = request
                    .program
                    .list_register_capacity(*expected_register)?
                    .ok_or_else(|| {
                        Error::internal("validated projected list capacity disappeared")
                    })?;
                let ResidentRowColumn::List { offsets, .. } = &projected.column else {
                    return Err(Error::internal(
                        "validated list projection changed physical type",
                    ));
                };
                if offsets
                    .windows(2)
                    .any(|pair| (pair[1] - pair[0]) as usize > declared)
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "resident row list projection exceeds its admitted element capacity",
                    ));
                }
            }
        }
        let expected_scratch = request.scratch_bytes(parts.input_cardinality)?;
        if parts.scratch_bytes != expected_scratch {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident row result reports an invalid scratch shape",
            ));
        }

        let completion = match backend {
            BackendKind::Cpu => ResidentDeviceCompletion::CpuReference,
            BackendKind::Metal => ResidentDeviceCompletion::Metal,
            BackendKind::Cuda => {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "selected backend has no validated resident row-program completion kind",
                ));
            }
        };
        let expected = request.obligations().collect::<Vec<_>>();
        if parts.receipts.len() != expected.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident row result omitted an execution obligation receipt",
            ));
        }
        let expected_positions = expected
            .iter()
            .copied()
            .enumerate()
            .map(|(index, obligation)| (obligation, index))
            .collect::<BTreeMap<_, _>>();
        let mut ordered_receipts = vec![None; expected.len()];
        for receipt in parts.receipts {
            if receipt.execution != request.execution || receipt.completion != completion {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row receipt has invalid execution or completion provenance",
                ));
            }
            let Some(index) = expected_positions.get(&receipt.obligation).copied() else {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row receipt set contains an unexpected obligation",
                ));
            };
            if ordered_receipts[index].replace(receipt).is_some() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row receipt set contains a duplicate obligation",
                ));
            }
        }
        let mut receipts = Vec::with_capacity(expected.len());
        for (index, receipt) in ordered_receipts.into_iter().enumerate() {
            let receipt = receipt.ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row receipt set is incomplete",
                )
            })?;
            let is_sort = index == request.program.instructions.len();
            let expected_output = if is_sort {
                output_rows
            } else {
                parts.input_cardinality
            };
            if receipt.input_cardinality != parts.input_cardinality as u64
                || receipt.output_cardinality != expected_output as u64
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row receipt has an impossible instruction or sort cardinality",
                ));
            }
            receipts.push(receipt);
        }
        parts.receipts = receipts;
        Ok(ValidatedResidentRowProgramResult { parts })
    }
}

impl ValidatedResidentRowProgramResult {
    #[must_use]
    pub const fn project(&self) -> ProjectId {
        self.parts.project
    }

    #[must_use]
    pub const fn execution(&self) -> ResidentExecutionId {
        self.parts.execution
    }

    #[must_use]
    pub const fn bookmark(&self) -> Bookmark {
        self.parts.bookmark
    }

    #[must_use]
    pub const fn graph_revision(&self) -> u64 {
        self.parts.graph_revision
    }

    #[must_use]
    pub const fn layout_version(&self) -> ResidentGraphLayoutVersion {
        self.parts.layout_version
    }

    #[must_use]
    pub const fn manifest_fingerprint(&self) -> ResidentRowManifestFingerprint {
        self.parts.manifest_fingerprint
    }

    #[must_use]
    pub const fn input_cardinality(&self) -> usize {
        self.parts.input_cardinality
    }

    #[must_use]
    pub fn source_positions(&self) -> &[u64] {
        &self.parts.source_positions
    }

    #[must_use]
    pub fn rows(&self) -> &ResidentNodePipelineResult {
        &self.parts.rows
    }

    #[must_use]
    pub fn projected_columns(&self) -> &[ResidentRowProjectedColumn] {
        &self.parts.projected_columns
    }

    #[must_use]
    pub fn receipts(&self) -> &[ResidentExecutionReceipt] {
        &self.parts.receipts
    }

    #[must_use]
    pub const fn scratch_bytes(&self) -> usize {
        self.parts.scratch_bytes
    }

    #[must_use]
    pub fn into_parts(self) -> ResidentRowProgramResultParts {
        self.parts
    }
}

/// Property loader used by the backend-neutral CPU semantic-reference evaluator.
pub type ResidentRowPropertyLoader<'a> = dyn FnMut(
        ResidentEntityBinding,
        PropertyId,
        ResidentRowValueType,
        Option<u32>,
        Option<ResidentTemporalAccessor>,
        &[u32],
        &CancellationToken,
    ) -> Result<ResidentRowColumn>
    + 'a;

pub fn execute_reference_row_program(
    request: &ResidentRowProgramRequest,
    mut input: ResidentNodePipelineResult,
    scratch_bytes: usize,
    loader: &mut ResidentRowPropertyLoader<'_>,
    cancellation: &CancellationToken,
) -> Result<ResidentRowProgramResult> {
    ensure_not_cancelled(cancellation)?;
    request.validate()?;
    let input_rows = request
        .scalar_input_rows()?
        .unwrap_or_else(|| input.start_rows.len());
    validate_base_rows(request, &input, input_rows)?;
    let expected_scratch = request.scratch_bytes(input_rows)?;
    if scratch_bytes != expected_scratch {
        return Err(Error::internal(
            "CPU resident row scratch reservation differs from the validated shape",
        ));
    }

    let mut registers = Vec::with_capacity(request.program.instructions.len());
    for (index, instruction) in request.program.instructions.iter().enumerate() {
        if index & 31 == 0 {
            ensure_not_cancelled(cancellation)?;
        }
        let column = match &instruction.operation {
            ResidentRowOperation::InputColumn(column) => column.clone(),
            ResidentRowOperation::LoadBooleanProperty { binding, property } => loader(
                *binding,
                *property,
                ResidentRowValueType::Boolean,
                None,
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::LoadListProperty {
                binding,
                property,
                maximum_elements,
            } => loader(
                *binding,
                *property,
                ResidentRowValueType::List,
                Some(*maximum_elements),
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::LoadIntegerProperty { binding, property } => loader(
                *binding,
                *property,
                ResidentRowValueType::Integer,
                None,
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::LoadFloatProperty { binding, property } => loader(
                *binding,
                *property,
                ResidentRowValueType::Float,
                None,
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::LoadStringProperty {
                binding,
                property,
                maximum_bytes,
            } => loader(
                *binding,
                *property,
                ResidentRowValueType::String,
                Some(*maximum_bytes),
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::LoadDateProperty { binding, property } => loader(
                *binding,
                *property,
                ResidentRowValueType::Date,
                None,
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::LoadLocalTimeProperty { binding, property } => loader(
                *binding,
                *property,
                ResidentRowValueType::LocalTime,
                None,
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::LoadZonedTimeProperty { binding, property } => loader(
                *binding,
                *property,
                ResidentRowValueType::ZonedTime,
                None,
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::LoadLocalDateTimeProperty { binding, property } => loader(
                *binding,
                *property,
                ResidentRowValueType::LocalDateTime,
                None,
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::LoadZonedDateTimeProperty {
                binding,
                property,
                maximum_timezone_bytes,
            } => loader(
                *binding,
                *property,
                ResidentRowValueType::ZonedDateTime,
                Some(*maximum_timezone_bytes),
                None,
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
            ResidentRowOperation::BooleanConstant(value) => ResidentRowColumn::Boolean {
                values: vec![u8::from(*value); input_rows],
                validity: vec![1; input_rows],
            },
            ResidentRowOperation::IntegerConstant(value) => ResidentRowColumn::Integer {
                values: vec![*value; input_rows],
                validity: vec![1; input_rows],
            },
            ResidentRowOperation::FloatConstant(bits) => ResidentRowColumn::Float {
                bits: vec![*bits; input_rows],
                validity: vec![1; input_rows],
            },
            ResidentRowOperation::StringConstant(value) => {
                let byte_count = value
                    .len()
                    .checked_mul(input_rows)
                    .ok_or_else(scratch_overflow)?;
                let mut offsets = Vec::with_capacity(input_rows.saturating_add(1));
                let mut bytes = Vec::with_capacity(byte_count);
                offsets.push(0);
                for _ in 0..input_rows {
                    bytes.extend_from_slice(value.as_bytes());
                    offsets.push(u32::try_from(bytes.len()).map_err(|_| scratch_overflow())?);
                }
                ResidentRowColumn::String {
                    offsets,
                    bytes,
                    validity: vec![1; input_rows],
                }
            }
            ResidentRowOperation::BooleanNot { operand } => {
                boolean_not(register(&registers, *operand)?)?
            }
            ResidentRowOperation::BooleanAnd { left, right } => {
                boolean_and(register(&registers, *left)?, register(&registers, *right)?)?
            }
            ResidentRowOperation::NumericAdd { left, right } => numeric_binary(
                register(&registers, *left)?,
                register(&registers, *right)?,
                instruction.output_type,
                NumericOperation::Add,
            )?,
            ResidentRowOperation::NumericSubtract { left, right } => numeric_binary(
                register(&registers, *left)?,
                register(&registers, *right)?,
                instruction.output_type,
                NumericOperation::Subtract,
            )?,
            ResidentRowOperation::NumericMultiply { left, right } => numeric_binary(
                register(&registers, *left)?,
                register(&registers, *right)?,
                instruction.output_type,
                NumericOperation::Multiply,
            )?,
            ResidentRowOperation::IntegerModulo { left, right } => numeric_binary(
                register(&registers, *left)?,
                register(&registers, *right)?,
                instruction.output_type,
                NumericOperation::Modulo,
            )?,
            ResidentRowOperation::NumericNegate { operand } => {
                numeric_negate(register(&registers, *operand)?)?
            }
            ResidentRowOperation::StringConcat { left, right } => {
                string_concat(register(&registers, *left)?, register(&registers, *right)?)?
            }
            ResidentRowOperation::StringCoalesce { left, right } => {
                string_coalesce(register(&registers, *left)?, register(&registers, *right)?)?
            }
            ResidentRowOperation::BooleanToString { operand } => {
                boolean_to_string(register(&registers, *operand)?)?
            }
            ResidentRowOperation::List { elements } => {
                list_build(&registers, elements, input_rows)?
            }
            ResidentRowOperation::ListIndex { list, index } => {
                list_index(register(&registers, *list)?, register(&registers, *index)?)?
            }
            ResidentRowOperation::ListConcat { left, right } => {
                list_concat(register(&registers, *left)?, register(&registers, *right)?)?
            }
            ResidentRowOperation::TemporalAddDuration {
                temporal,
                months,
                days,
                seconds,
                nanos,
            } => temporal_add_duration(
                register(&registers, *temporal)?,
                *months,
                *days,
                *seconds,
                *nanos,
                cancellation,
            )?,
            ResidentRowOperation::TemporalAccessor {
                temporal, accessor, ..
            } => temporal_accessor(register(&registers, *temporal)?, *accessor, cancellation)?,
            ResidentRowOperation::LoadDurationAccessor {
                binding,
                property,
                accessor,
            } => loader(
                *binding,
                *property,
                ResidentRowValueType::Integer,
                None,
                Some(*accessor),
                binding_rows(&input, *binding)?,
                cancellation,
            )?,
        };
        column.validate(instruction.output_type, input_rows)?;
        registers.push(column);
    }

    let mut positions = (0..input_rows)
        .map(|position| u64::try_from(position).map_err(|_| scratch_overflow()))
        .collect::<Result<Vec<_>>>()?;
    if !request.sort_keys.is_empty() {
        // Explicit source-position tie breaking gives stable semantics while permitting the
        // allocation-free unstable sorter required by exact scratch accounting.
        positions.sort_unstable_by(|left, right| {
            compare_sort_keys(&registers, &request.sort_keys, *left, *right)
                .then_with(|| left.cmp(right))
        });
    }
    let output_rows = request.output_cardinality(input_rows)?;
    let start = request.offset.min(input_rows);
    let end = start
        .checked_add(output_rows)
        .ok_or_else(scratch_overflow)?;
    positions.truncate(end);
    if start != 0 {
        positions.drain(..start);
    }
    let source_positions = positions;
    if request.has_graph_input() {
        select_base_rows(&mut input, &source_positions)?;
    }
    let projected_columns = request
        .final_registers
        .iter()
        .map(|register_index| {
            Ok(ResidentRowProjectedColumn {
                register: *register_index,
                column: register(&registers, *register_index)?.select(&source_positions)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure_not_cancelled(cancellation)?;

    let input_cardinality = u64::try_from(input_rows).map_err(|_| scratch_overflow())?;
    let output_cardinality = u64::try_from(output_rows).map_err(|_| scratch_overflow())?;
    let mut receipts = Vec::with_capacity(request.obligations().count());
    receipts.extend(
        request
            .manifest
            .instruction_obligations
            .iter()
            .copied()
            .map(|obligation| ResidentExecutionReceipt {
                execution: request.execution,
                obligation,
                input_cardinality,
                output_cardinality: input_cardinality,
                completion: ResidentDeviceCompletion::CpuReference,
            }),
    );
    receipts.push(ResidentExecutionReceipt {
        execution: request.execution,
        obligation: request.manifest.sort_obligation,
        input_cardinality,
        output_cardinality,
        completion: ResidentDeviceCompletion::CpuReference,
    });
    Ok(ResidentRowProgramResult::completed(
        ResidentRowProgramResultParts {
            project: request.project,
            execution: request.execution,
            bookmark: request.expected_bookmark,
            graph_revision: request.expected_graph_revision,
            layout_version: request.expected_layout_version,
            manifest_fingerprint: request.manifest.fingerprint,
            input_cardinality: input_rows,
            source_positions,
            rows: input,
            projected_columns,
            receipts,
            scratch_bytes,
        },
    ))
}

fn validate_binding(
    input: &ResidentNodePipelineRequest,
    binding: ResidentEntityBinding,
) -> Result<()> {
    match binding {
        ResidentEntityBinding::Node(ResidentNodeBinding::Start) => Ok(()),
        ResidentEntityBinding::Node(ResidentNodeBinding::End) => {
            if input.expansion_steps().next().is_none() {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "resident row program names an absent end-node binding",
                ));
            }
            Ok(())
        }
        ResidentEntityBinding::Node(ResidentNodeBinding::Intermediate(index)) => {
            let intermediate_count = input.expansion_steps().count().saturating_sub(1);
            if index as usize >= intermediate_count {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "resident row program names an absent intermediate-node binding",
                ));
            }
            Ok(())
        }
        ResidentEntityBinding::Relationship(index) => {
            if index as usize >= input.relationship_column_count() {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "resident row program names an absent relationship binding",
                ));
            }
            Ok(())
        }
    }
}

fn binding_rows(
    rows: &ResidentNodePipelineResult,
    binding: ResidentEntityBinding,
) -> Result<&[u32]> {
    let column = match binding {
        ResidentEntityBinding::Node(ResidentNodeBinding::Start) => Some(&rows.start_rows),
        ResidentEntityBinding::Node(ResidentNodeBinding::End) => Some(&rows.end_rows),
        ResidentEntityBinding::Node(ResidentNodeBinding::Intermediate(index)) => {
            rows.intermediate_node_rows.get(index as usize)
        }
        ResidentEntityBinding::Relationship(index) => {
            let index = index as usize;
            if index < rows.intermediate_edge_rows.len() {
                rows.intermediate_edge_rows.get(index)
            } else if index == rows.intermediate_edge_rows.len() {
                Some(&rows.edge_rows)
            } else {
                None
            }
        }
    };
    column.map(Vec::as_slice).ok_or_else(|| {
        Error::new(
            ErrorCode::CorruptStorage,
            "resident row input omitted a referenced graph binding column",
        )
    })
}

fn aligned_u32_column_count(input: &ResidentNodePipelineRequest) -> Result<usize> {
    1_usize
        .checked_add(input.expansion_steps().count())
        .and_then(|count| count.checked_add(input.relationship_column_count()))
        .and_then(|count| count.checked_add(if input.value_matrix.is_some() { 2 } else { 0 }))
        .ok_or_else(scratch_overflow)
}

fn validate_base_rows(
    request: &ResidentRowProgramRequest,
    rows: &ResidentNodePipelineResult,
    row_count: usize,
) -> Result<()> {
    if !request.has_graph_input() {
        if !rows.start_rows.is_empty()
            || !rows.intermediate_node_rows.is_empty()
            || !rows.intermediate_edge_rows.is_empty()
            || !rows.edge_rows.is_empty()
            || !rows.end_rows.is_empty()
            || !rows.integer_columns.is_empty()
            || !rows.boolean_columns.is_empty()
            || !rows.value_left_indices.is_empty()
            || !rows.value_right_indices.is_empty()
            || rows.mutation.is_some()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident scalar row source published an unexpected graph column",
            ));
        }
        return Ok(());
    }
    let graph = &request.input;
    let expansion_count = graph.expansion_steps().count();
    let relationship_count = graph.relationship_column_count();
    let aligned = |column: &[u32], present: bool| {
        if (present && column.len() != row_count) || (!present && !column.is_empty()) {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident row graph column is absent or misaligned",
            ));
        }
        Ok(())
    };
    aligned(&rows.start_rows, true)?;
    if rows.intermediate_node_rows.len() != expansion_count.saturating_sub(1)
        || rows
            .intermediate_node_rows
            .iter()
            .any(|column| column.len() != row_count)
        || rows.intermediate_edge_rows.len() != relationship_count.saturating_sub(1)
        || rows
            .intermediate_edge_rows
            .iter()
            .any(|column| column.len() != row_count)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "resident row intermediate graph columns are misaligned",
        ));
    }
    aligned(&rows.end_rows, expansion_count != 0)?;
    aligned(&rows.edge_rows, relationship_count != 0)?;
    if !rows.integer_columns.is_empty() || !rows.boolean_columns.is_empty() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "wrapped resident row input crossed an obsolete projection boundary",
        ));
    }
    aligned(&rows.value_left_indices, graph.value_matrix.is_some())?;
    aligned(&rows.value_right_indices, graph.value_matrix.is_some())?;
    if rows.mutation.is_some() != graph.mutation.is_some() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "wrapped resident row input changed mutation-result presence",
        ));
    }
    Ok(())
}

fn select_base_rows(rows: &mut ResidentNodePipelineResult, positions: &[u64]) -> Result<()> {
    rows.start_rows = select_values(&rows.start_rows, positions)?;
    for column in &mut rows.intermediate_node_rows {
        *column = select_values(column, positions)?;
    }
    for column in &mut rows.intermediate_edge_rows {
        *column = select_values(column, positions)?;
    }
    if !rows.edge_rows.is_empty() {
        rows.edge_rows = select_values(&rows.edge_rows, positions)?;
    }
    if !rows.end_rows.is_empty() {
        rows.end_rows = select_values(&rows.end_rows, positions)?;
    }
    if !rows.value_left_indices.is_empty() {
        rows.value_left_indices = select_values(&rows.value_left_indices, positions)?;
        rows.value_right_indices = select_values(&rows.value_right_indices, positions)?;
    }
    Ok(())
}

fn select_values<T: Copy>(values: &[T], positions: &[u64]) -> Result<Vec<T>> {
    positions
        .iter()
        .map(|position| {
            let position = usize::try_from(*position).map_err(|_| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row source position exceeds host addressability",
                )
            })?;
            values.get(position).copied().ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "resident row source position is out of bounds",
                )
            })
        })
        .collect()
}

fn register(registers: &[ResidentRowColumn], index: u16) -> Result<&ResidentRowColumn> {
    registers.get(index as usize).ok_or_else(|| {
        Error::internal("validated resident row register disappeared during execution")
    })
}

fn boolean_not(input: &ResidentRowColumn) -> Result<ResidentRowColumn> {
    let ResidentRowColumn::Boolean { values, validity } = input else {
        return Err(Error::internal(
            "validated resident Boolean NOT register changed type",
        ));
    };
    Ok(ResidentRowColumn::Boolean {
        values: values.iter().map(|value| 1 - *value).collect(),
        validity: validity.clone(),
    })
}

fn boolean_and(left: &ResidentRowColumn, right: &ResidentRowColumn) -> Result<ResidentRowColumn> {
    let ResidentRowColumn::Boolean {
        values: left_values,
        validity: left_validity,
    } = left
    else {
        return Err(Error::internal(
            "validated resident Boolean AND left register changed type",
        ));
    };
    let ResidentRowColumn::Boolean {
        values: right_values,
        validity: right_validity,
    } = right
    else {
        return Err(Error::internal(
            "validated resident Boolean AND right register changed type",
        ));
    };
    let mut values = Vec::with_capacity(left_values.len());
    let mut validity = Vec::with_capacity(left_values.len());
    for (((left, left_valid), right), right_valid) in left_values
        .iter()
        .zip(left_validity)
        .zip(right_values)
        .zip(right_validity)
    {
        let known_false = (*left_valid != 0 && *left == 0) || (*right_valid != 0 && *right == 0);
        let valid = (*left_valid != 0 && *right_valid != 0) || known_false;
        values.push(u8::from(valid && !known_false));
        validity.push(u8::from(valid));
    }
    Ok(ResidentRowColumn::Boolean { values, validity })
}

#[derive(Clone, Copy)]
enum NumericOperation {
    Add,
    Subtract,
    Multiply,
    Modulo,
}

fn numeric_binary(
    left: &ResidentRowColumn,
    right: &ResidentRowColumn,
    output_type: ResidentRowValueType,
    operation: NumericOperation,
) -> Result<ResidentRowColumn> {
    let rows = left.len();
    if right.len() != rows {
        return Err(Error::internal(
            "validated resident numeric registers became misaligned",
        ));
    }
    let validity = left
        .validity()
        .iter()
        .zip(right.validity())
        .map(|(left, right)| u8::from(*left != 0 && *right != 0))
        .collect::<Vec<_>>();
    if output_type == ResidentRowValueType::Integer {
        let (
            ResidentRowColumn::Integer { values: left, .. },
            ResidentRowColumn::Integer { values: right, .. },
        ) = (left, right)
        else {
            return Err(Error::internal(
                "validated resident integer arithmetic register changed type",
            ));
        };
        let mut values = Vec::with_capacity(rows);
        for ((left, right), valid) in left.iter().zip(right).zip(&validity) {
            if *valid == 0 {
                values.push(0);
                continue;
            }
            let value = match operation {
                NumericOperation::Add => left.checked_add(*right),
                NumericOperation::Subtract => left.checked_sub(*right),
                NumericOperation::Multiply => left.checked_mul(*right),
                NumericOperation::Modulo => {
                    if *right == 0 {
                        return Err(Error::new(ErrorCode::QueryType, "modulo by zero"));
                    }
                    Some(left.wrapping_rem(*right))
                }
            }
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "integer arithmetic overflow"))?;
            values.push(value);
        }
        return Ok(ResidentRowColumn::Integer { values, validity });
    }
    let mut bits = Vec::with_capacity(rows);
    for row in 0..rows {
        if validity[row] == 0 {
            bits.push(0);
            continue;
        }
        let left = numeric_as_f64(left, row)?;
        let right = numeric_as_f64(right, row)?;
        bits.push(match operation {
            NumericOperation::Add => (left + right).to_bits(),
            NumericOperation::Subtract => (left - right).to_bits(),
            NumericOperation::Multiply => (left * right).to_bits(),
            NumericOperation::Modulo => {
                return Err(Error::internal(
                    "validated resident INTEGER modulo was promoted to FLOAT",
                ));
            }
        });
    }
    Ok(ResidentRowColumn::Float { bits, validity })
}

fn numeric_as_f64(column: &ResidentRowColumn, row: usize) -> Result<f64> {
    match column {
        ResidentRowColumn::Integer { values, .. } => values
            .get(row)
            .copied()
            .map(|value| value as f64)
            .ok_or_else(|| Error::internal("resident integer register row disappeared")),
        ResidentRowColumn::Float { bits, .. } => bits
            .get(row)
            .copied()
            .map(f64::from_bits)
            .ok_or_else(|| Error::internal("resident float register row disappeared")),
        ResidentRowColumn::Boolean { .. }
        | ResidentRowColumn::String { .. }
        | ResidentRowColumn::List { .. }
        | ResidentRowColumn::Date { .. }
        | ResidentRowColumn::LocalTime { .. }
        | ResidentRowColumn::ZonedTime { .. }
        | ResidentRowColumn::LocalDateTime { .. }
        | ResidentRowColumn::ZonedDateTime { .. } => Err(Error::internal(
            "validated resident numeric register changed to a non-numeric type",
        )),
    }
}

fn numeric_negate(input: &ResidentRowColumn) -> Result<ResidentRowColumn> {
    match input {
        ResidentRowColumn::Integer { values, validity } => {
            let mut output = Vec::with_capacity(values.len());
            for (value, valid) in values.iter().zip(validity) {
                if *valid == 0 {
                    output.push(0);
                } else {
                    output.push(value.checked_neg().ok_or_else(|| {
                        Error::new(ErrorCode::QueryType, "integer negation overflow")
                    })?);
                }
            }
            Ok(ResidentRowColumn::Integer {
                values: output,
                validity: validity.clone(),
            })
        }
        ResidentRowColumn::Float { bits, validity } => Ok(ResidentRowColumn::Float {
            bits: bits.iter().map(|bits| bits ^ (1_u64 << 63)).collect(),
            validity: validity.clone(),
        }),
        ResidentRowColumn::Boolean { .. }
        | ResidentRowColumn::String { .. }
        | ResidentRowColumn::List { .. }
        | ResidentRowColumn::Date { .. }
        | ResidentRowColumn::LocalTime { .. }
        | ResidentRowColumn::ZonedTime { .. }
        | ResidentRowColumn::LocalDateTime { .. }
        | ResidentRowColumn::ZonedDateTime { .. } => Err(Error::internal(
            "validated resident numeric negation register changed type",
        )),
    }
}

fn string_concat(left: &ResidentRowColumn, right: &ResidentRowColumn) -> Result<ResidentRowColumn> {
    let (
        ResidentRowColumn::String {
            offsets: left_offsets,
            bytes: left_bytes,
            validity: left_validity,
        },
        ResidentRowColumn::String {
            offsets: right_offsets,
            bytes: right_bytes,
            validity: right_validity,
        },
    ) = (left, right)
    else {
        return Err(Error::internal(
            "validated string concatenation register changed type",
        ));
    };
    if left_validity.len() != right_validity.len() {
        return Err(Error::internal(
            "validated string concatenation registers became misaligned",
        ));
    }
    let mut offsets = Vec::with_capacity(left_validity.len().saturating_add(1));
    let mut bytes = Vec::new();
    let mut validity = Vec::with_capacity(left_validity.len());
    offsets.push(0);
    for row in 0..left_validity.len() {
        let valid = left_validity[row] != 0 && right_validity[row] != 0;
        validity.push(u8::from(valid));
        if valid {
            let left_start = left_offsets[row] as usize;
            let left_end = left_offsets[row + 1] as usize;
            let right_start = right_offsets[row] as usize;
            let right_end = right_offsets[row + 1] as usize;
            bytes.extend_from_slice(&left_bytes[left_start..left_end]);
            bytes.extend_from_slice(&right_bytes[right_start..right_end]);
        }
        offsets.push(u32::try_from(bytes.len()).map_err(|_| scratch_overflow())?);
    }
    Ok(ResidentRowColumn::String {
        offsets,
        bytes,
        validity,
    })
}

fn string_coalesce(
    left: &ResidentRowColumn,
    right: &ResidentRowColumn,
) -> Result<ResidentRowColumn> {
    let (
        ResidentRowColumn::String {
            offsets: left_offsets,
            bytes: left_bytes,
            validity: left_validity,
        },
        ResidentRowColumn::String {
            offsets: right_offsets,
            bytes: right_bytes,
            validity: right_validity,
        },
    ) = (left, right)
    else {
        return Err(Error::internal(
            "validated string coalesce register changed type",
        ));
    };
    if left_validity.len() != right_validity.len() {
        return Err(Error::internal(
            "validated string coalesce registers became misaligned",
        ));
    }
    let mut offsets = Vec::with_capacity(left_validity.len().saturating_add(1));
    let mut bytes = Vec::new();
    let mut validity = Vec::with_capacity(left_validity.len());
    offsets.push(0);
    for row in 0..left_validity.len() {
        let selected = if left_validity[row] != 0 {
            Some((&left_offsets, &left_bytes))
        } else if right_validity[row] != 0 {
            Some((&right_offsets, &right_bytes))
        } else {
            None
        };
        validity.push(u8::from(selected.is_some()));
        if let Some((selected_offsets, selected_bytes)) = selected {
            let start = selected_offsets[row] as usize;
            let end = selected_offsets[row + 1] as usize;
            bytes.extend_from_slice(&selected_bytes[start..end]);
        }
        offsets.push(u32::try_from(bytes.len()).map_err(|_| scratch_overflow())?);
    }
    Ok(ResidentRowColumn::String {
        offsets,
        bytes,
        validity,
    })
}

fn boolean_to_string(input: &ResidentRowColumn) -> Result<ResidentRowColumn> {
    let ResidentRowColumn::Boolean { values, validity } = input else {
        return Err(Error::internal(
            "validated resident Boolean-to-string register changed type",
        ));
    };
    let maximum_bytes = values.len().checked_mul(5).ok_or_else(scratch_overflow)?;
    let mut offsets = Vec::with_capacity(values.len().saturating_add(1));
    let mut bytes = Vec::with_capacity(maximum_bytes);
    offsets.push(0);
    for (value, valid) in values.iter().zip(validity) {
        if (*valid == 0 && *value != 0) || *value > 1 {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident Boolean-to-string operand has a non-canonical payload",
            ));
        }
        if *valid != 0 {
            bytes.extend_from_slice(if *value == 0 { b"false" } else { b"true" });
        }
        offsets.push(u32::try_from(bytes.len()).map_err(|_| scratch_overflow())?);
    }
    Ok(ResidentRowColumn::String {
        offsets,
        bytes,
        validity: validity.clone(),
    })
}

fn list_build(
    registers: &[ResidentRowColumn],
    elements: &[u16],
    rows: usize,
) -> Result<ResidentRowColumn> {
    let element_columns = elements
        .iter()
        .map(|element| {
            let ResidentRowColumn::Integer { values, validity } = register(registers, *element)?
            else {
                return Err(Error::internal(
                    "validated resident list element register changed type",
                ));
            };
            if values.len() != rows || validity.len() != rows {
                return Err(Error::internal(
                    "validated resident list element register became misaligned",
                ));
            }
            Ok((values.as_slice(), validity.as_slice()))
        })
        .collect::<Result<Vec<_>>>()?;
    let capacity = rows
        .checked_mul(element_columns.len())
        .ok_or_else(scratch_overflow)?;
    let mut offsets = Vec::with_capacity(rows.saturating_add(1));
    let mut values = Vec::with_capacity(capacity);
    let mut element_validity = Vec::with_capacity(capacity);
    offsets.push(0);
    for row in 0..rows {
        for (column, validity) in &element_columns {
            let valid = validity[row];
            values.push(if valid == 0 { 0 } else { column[row] });
            element_validity.push(valid);
        }
        offsets.push(u32::try_from(values.len()).map_err(|_| scratch_overflow())?);
    }
    Ok(ResidentRowColumn::List {
        offsets,
        values,
        element_validity,
        validity: vec![1; rows],
    })
}

fn list_index(list: &ResidentRowColumn, index: &ResidentRowColumn) -> Result<ResidentRowColumn> {
    let ResidentRowColumn::List {
        offsets,
        values,
        element_validity,
        validity: list_validity,
    } = list
    else {
        return Err(Error::internal(
            "validated resident list-index source changed type",
        ));
    };
    let ResidentRowColumn::Integer {
        values: indices,
        validity: index_validity,
    } = index
    else {
        return Err(Error::internal(
            "validated resident list-index index changed type",
        ));
    };
    if list_validity.len() != indices.len() || indices.len() != index_validity.len() {
        return Err(Error::internal(
            "validated resident list-index registers became misaligned",
        ));
    }
    let mut output = Vec::with_capacity(indices.len());
    let mut validity = Vec::with_capacity(indices.len());
    for row in 0..indices.len() {
        if list_validity[row] == 0 || index_validity[row] == 0 {
            output.push(0);
            validity.push(0);
            continue;
        }
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        let length = end - start;
        let relative = if indices[row] < 0 {
            usize::try_from(indices[row].unsigned_abs())
                .ok()
                .and_then(|offset| length.checked_sub(offset))
        } else {
            usize::try_from(indices[row]).ok()
        };
        let Some(relative) = relative else {
            output.push(0);
            validity.push(0);
            continue;
        };
        let Some(position) = start
            .checked_add(relative)
            .filter(|position| *position < end)
        else {
            output.push(0);
            validity.push(0);
            continue;
        };
        let valid = element_validity[position];
        output.push(if valid == 0 { 0 } else { values[position] });
        validity.push(valid);
    }
    Ok(ResidentRowColumn::Integer {
        values: output,
        validity,
    })
}

fn list_concat(left: &ResidentRowColumn, right: &ResidentRowColumn) -> Result<ResidentRowColumn> {
    let (
        ResidentRowColumn::List {
            offsets: left_offsets,
            values: left_values,
            element_validity: left_element_validity,
            validity: left_validity,
        },
        ResidentRowColumn::List {
            offsets: right_offsets,
            values: right_values,
            element_validity: right_element_validity,
            validity: right_validity,
        },
    ) = (left, right)
    else {
        return Err(Error::internal(
            "validated resident list-concatenation register changed type",
        ));
    };
    if left_validity.len() != right_validity.len() {
        return Err(Error::internal(
            "validated resident list-concatenation registers became misaligned",
        ));
    }
    let mut offsets = Vec::with_capacity(left_validity.len().saturating_add(1));
    let mut values = Vec::new();
    let mut element_validity = Vec::new();
    let mut validity = Vec::with_capacity(left_validity.len());
    offsets.push(0);
    for row in 0..left_validity.len() {
        let valid = left_validity[row] != 0 && right_validity[row] != 0;
        validity.push(u8::from(valid));
        if valid {
            let left_start = left_offsets[row] as usize;
            let left_end = left_offsets[row + 1] as usize;
            let right_start = right_offsets[row] as usize;
            let right_end = right_offsets[row + 1] as usize;
            values.extend_from_slice(&left_values[left_start..left_end]);
            element_validity.extend_from_slice(&left_element_validity[left_start..left_end]);
            values.extend_from_slice(&right_values[right_start..right_end]);
            element_validity.extend_from_slice(&right_element_validity[right_start..right_end]);
        }
        offsets.push(u32::try_from(values.len()).map_err(|_| scratch_overflow())?);
    }
    Ok(ResidentRowColumn::List {
        offsets,
        values,
        element_validity,
        validity,
    })
}

fn temporal_add_duration(
    input: &ResidentRowColumn,
    months: i64,
    days: i64,
    seconds: i64,
    nanos: i32,
    cancellation: &CancellationToken,
) -> Result<ResidentRowColumn> {
    let duration = ResidentRowDuration {
        months,
        days,
        seconds,
        nanos,
    };
    match input {
        ResidentRowColumn::Date {
            days: values,
            validity,
        } => temporal_add_date(values, validity, duration, cancellation),
        ResidentRowColumn::LocalTime {
            nanos: values,
            validity,
        } => temporal_add_local_time(values, validity, duration, cancellation),
        ResidentRowColumn::ZonedTime {
            nanos: values,
            offset_seconds,
            validity,
        } => temporal_add_zoned_time(values, offset_seconds, validity, duration, cancellation),
        ResidentRowColumn::LocalDateTime {
            seconds: values,
            nanos: subsecond,
            validity,
        } => temporal_add_local_datetime(values, subsecond, validity, duration, cancellation),
        ResidentRowColumn::ZonedDateTime {
            seconds: values,
            nanos: subsecond,
            timezone_offsets,
            timezone_bytes,
            validity,
        } => temporal_add_zoned_datetime(
            values,
            subsecond,
            timezone_offsets,
            timezone_bytes,
            validity,
            duration,
            cancellation,
        ),
        ResidentRowColumn::Boolean { .. }
        | ResidentRowColumn::Integer { .. }
        | ResidentRowColumn::Float { .. }
        | ResidentRowColumn::String { .. }
        | ResidentRowColumn::List { .. } => Err(Error::internal(
            "validated temporal duration operand changed to a non-temporal type",
        )),
    }
}

#[derive(Clone, Copy)]
struct ResidentRowDuration {
    months: i64,
    days: i64,
    seconds: i64,
    nanos: i32,
}

impl ResidentRowDuration {
    fn apply(self, value: ScalarValue) -> Result<ScalarValue> {
        crate::execution::apply_duration_to_temporal_scalar(
            &value,
            self.months,
            self.days,
            self.seconds,
            self.nanos,
        )
    }
}

fn ensure_temporal_row_not_cancelled(row: usize, cancellation: &CancellationToken) -> Result<()> {
    if row.trailing_zeros() >= 12 {
        ensure_not_cancelled(cancellation)?;
    }
    Ok(())
}

fn temporal_duration_wrong_type() -> Error {
    Error::internal("shared temporal duration arithmetic returned the wrong scalar type")
}

fn temporal_add_date(
    values: &[i64],
    validity: &[u8],
    duration: ResidentRowDuration,
    cancellation: &CancellationToken,
) -> Result<ResidentRowColumn> {
    let mut output = Vec::with_capacity(values.len());
    for (row, (value, valid)) in values.iter().zip(validity).enumerate() {
        ensure_temporal_row_not_cancelled(row, cancellation)?;
        if *valid == 0 {
            output.push(0);
            continue;
        }
        let ScalarValue::Date(value) = duration.apply(ScalarValue::Date(*value))? else {
            return Err(temporal_duration_wrong_type());
        };
        output.push(value);
    }
    Ok(ResidentRowColumn::Date {
        days: output,
        validity: validity.to_vec(),
    })
}

fn temporal_add_local_time(
    values: &[i64],
    validity: &[u8],
    duration: ResidentRowDuration,
    cancellation: &CancellationToken,
) -> Result<ResidentRowColumn> {
    let mut output = Vec::with_capacity(values.len());
    for (row, (value, valid)) in values.iter().zip(validity).enumerate() {
        ensure_temporal_row_not_cancelled(row, cancellation)?;
        if *valid == 0 {
            output.push(0);
            continue;
        }
        let ScalarValue::LocalTime(value) = duration.apply(ScalarValue::LocalTime(*value))? else {
            return Err(temporal_duration_wrong_type());
        };
        output.push(value);
    }
    Ok(ResidentRowColumn::LocalTime {
        nanos: output,
        validity: validity.to_vec(),
    })
}

fn temporal_add_zoned_time(
    values: &[i64],
    offsets: &[i32],
    validity: &[u8],
    duration: ResidentRowDuration,
    cancellation: &CancellationToken,
) -> Result<ResidentRowColumn> {
    let mut output_nanos = Vec::with_capacity(values.len());
    let mut output_offsets = Vec::with_capacity(values.len());
    for (row, ((value, offset), valid)) in values.iter().zip(offsets).zip(validity).enumerate() {
        ensure_temporal_row_not_cancelled(row, cancellation)?;
        if *valid == 0 {
            output_nanos.push(0);
            output_offsets.push(0);
            continue;
        }
        let ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } = duration.apply(ScalarValue::ZonedTime {
            nanos: *value,
            offset_seconds: *offset,
        })?
        else {
            return Err(temporal_duration_wrong_type());
        };
        output_nanos.push(nanos);
        output_offsets.push(offset_seconds);
    }
    Ok(ResidentRowColumn::ZonedTime {
        nanos: output_nanos,
        offset_seconds: output_offsets,
        validity: validity.to_vec(),
    })
}

fn temporal_add_local_datetime(
    seconds: &[i64],
    nanos: &[u32],
    validity: &[u8],
    duration: ResidentRowDuration,
    cancellation: &CancellationToken,
) -> Result<ResidentRowColumn> {
    let mut output_seconds = Vec::with_capacity(seconds.len());
    let mut output_nanos = Vec::with_capacity(seconds.len());
    for (row, ((seconds, nanos), valid)) in seconds.iter().zip(nanos).zip(validity).enumerate() {
        ensure_temporal_row_not_cancelled(row, cancellation)?;
        if *valid == 0 {
            output_seconds.push(0);
            output_nanos.push(0);
            continue;
        }
        let ScalarValue::LocalDateTime { seconds, nanos } =
            duration.apply(ScalarValue::LocalDateTime {
                seconds: *seconds,
                nanos: *nanos,
            })?
        else {
            return Err(temporal_duration_wrong_type());
        };
        output_seconds.push(seconds);
        output_nanos.push(nanos);
    }
    Ok(ResidentRowColumn::LocalDateTime {
        seconds: output_seconds,
        nanos: output_nanos,
        validity: validity.to_vec(),
    })
}

fn temporal_add_zoned_datetime(
    seconds: &[i64],
    nanos: &[u32],
    timezone_offsets: &[u32],
    timezone_bytes: &[u8],
    validity: &[u8],
    duration: ResidentRowDuration,
    cancellation: &CancellationToken,
) -> Result<ResidentRowColumn> {
    let maximum_timezone_bytes = timezone_offsets
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as usize)
        .max()
        .unwrap_or(0);
    let mut output_seconds = Vec::with_capacity(seconds.len());
    let mut output_nanos = Vec::with_capacity(seconds.len());
    let mut output_timezone_offsets = Vec::with_capacity(seconds.len().saturating_add(1));
    let mut output_timezone_bytes = Vec::new();
    output_timezone_offsets.push(0);
    for (row, ((seconds, nanos), valid)) in seconds.iter().zip(nanos).zip(validity).enumerate() {
        ensure_temporal_row_not_cancelled(row, cancellation)?;
        if *valid == 0 {
            output_seconds.push(0);
            output_nanos.push(0);
        } else {
            let start = timezone_offsets[row] as usize;
            let end = timezone_offsets[row + 1] as usize;
            let timezone = std::str::from_utf8(&timezone_bytes[start..end]).map_err(|_| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "resident zoned-datetime duration operand has invalid timezone UTF-8",
                )
            })?;
            let ScalarValue::ZonedDateTime {
                seconds,
                nanos,
                timezone,
            } = duration.apply(ScalarValue::ZonedDateTime {
                seconds: *seconds,
                nanos: *nanos,
                timezone: timezone.into(),
            })?
            else {
                return Err(temporal_duration_wrong_type());
            };
            if timezone.len() > maximum_timezone_bytes {
                return Err(Error::internal(
                    "shared temporal duration arithmetic expanded the timezone shape",
                ));
            }
            output_seconds.push(seconds);
            output_nanos.push(nanos);
            output_timezone_bytes.extend_from_slice(timezone.as_bytes());
        }
        output_timezone_offsets
            .push(u32::try_from(output_timezone_bytes.len()).map_err(|_| scratch_overflow())?);
    }
    Ok(ResidentRowColumn::ZonedDateTime {
        seconds: output_seconds,
        nanos: output_nanos,
        timezone_offsets: output_timezone_offsets,
        timezone_bytes: output_timezone_bytes,
        validity: validity.to_vec(),
    })
}

fn temporal_accessor(
    column: &ResidentRowColumn,
    accessor: ResidentTemporalAccessor,
    cancellation: &CancellationToken,
) -> Result<ResidentRowColumn> {
    let rows = column.len();
    let validity = column.validity().to_vec();
    let value_at = |row: usize| -> Result<ScalarValue> {
        Ok(match column {
            ResidentRowColumn::Date { days, .. } => ScalarValue::Date(days[row]),
            ResidentRowColumn::LocalTime { nanos, .. } => ScalarValue::LocalTime(nanos[row]),
            ResidentRowColumn::ZonedTime {
                nanos,
                offset_seconds,
                ..
            } => ScalarValue::ZonedTime {
                nanos: nanos[row],
                offset_seconds: offset_seconds[row],
            },
            ResidentRowColumn::LocalDateTime { seconds, nanos, .. } => ScalarValue::LocalDateTime {
                seconds: seconds[row],
                nanos: nanos[row],
            },
            ResidentRowColumn::ZonedDateTime {
                seconds,
                nanos,
                timezone_offsets,
                timezone_bytes,
                ..
            } => {
                let start = timezone_offsets[row] as usize;
                let end = timezone_offsets[row + 1] as usize;
                let timezone = std::str::from_utf8(&timezone_bytes[start..end]).map_err(|_| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "resident temporal accessor timezone is not UTF-8",
                    )
                })?;
                ScalarValue::ZonedDateTime {
                    seconds: seconds[row],
                    nanos: nanos[row],
                    timezone: Arc::from(timezone),
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "resident temporal accessor received a non-temporal column",
                ));
            }
        })
    };
    match accessor.output_type() {
        ResidentRowValueType::Integer => {
            let mut values = Vec::with_capacity(rows);
            for row in 0..rows {
                ensure_temporal_row_not_cancelled(row, cancellation)?;
                if validity[row] == 0 {
                    values.push(0);
                    continue;
                }
                let ScalarValue::Integer(value) =
                    crate::execution::evaluate_temporal_accessor(&value_at(row)?, accessor)?
                else {
                    return Err(Error::internal(
                        "validated integer temporal accessor returned a string",
                    ));
                };
                values.push(value);
            }
            Ok(ResidentRowColumn::Integer { values, validity })
        }
        ResidentRowValueType::String => {
            let mut offsets = Vec::with_capacity(rows.saturating_add(1));
            let mut bytes = Vec::new();
            offsets.push(0);
            for row in 0..rows {
                ensure_temporal_row_not_cancelled(row, cancellation)?;
                if validity[row] != 0 {
                    let ScalarValue::String(value) =
                        crate::execution::evaluate_temporal_accessor(&value_at(row)?, accessor)?
                    else {
                        return Err(Error::internal(
                            "validated string temporal accessor returned an integer",
                        ));
                    };
                    bytes.extend_from_slice(value.as_bytes());
                }
                offsets.push(u32::try_from(bytes.len()).map_err(|_| scratch_overflow())?);
            }
            Ok(ResidentRowColumn::String {
                offsets,
                bytes,
                validity,
            })
        }
        _ => Err(Error::internal(
            "temporal accessor declared a non-scalar output type",
        )),
    }
}

fn compare_sort_keys(
    registers: &[ResidentRowColumn],
    keys: &[ResidentRowSortKey],
    left: u64,
    right: u64,
) -> Ordering {
    keys.iter()
        .map(|key| compare_sort_key(&registers[key.register as usize], *key, left, right))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

fn compare_sort_key(
    column: &ResidentRowColumn,
    key: ResidentRowSortKey,
    left: u64,
    right: u64,
) -> Ordering {
    let left = left as usize;
    let right = right as usize;
    let left_valid = column.validity()[left] != 0;
    let right_valid = column.validity()[right] != 0;
    match (left_valid, right_valid) {
        (false, false) => Ordering::Equal,
        (false, true) => {
            if key.nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (true, false) => {
            if key.nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (true, true) => {
            let ordering = match column {
                ResidentRowColumn::Boolean { values, .. } => values[left].cmp(&values[right]),
                ResidentRowColumn::Integer { values, .. } => values[left].cmp(&values[right]),
                ResidentRowColumn::Float { bits, .. } => OrderedFloat(f64::from_bits(bits[left]))
                    .cmp(&OrderedFloat(f64::from_bits(bits[right]))),
                ResidentRowColumn::String { offsets, bytes, .. } => {
                    let left_start = offsets[left] as usize;
                    let left_end = offsets[left + 1] as usize;
                    let right_start = offsets[right] as usize;
                    let right_end = offsets[right + 1] as usize;
                    bytes[left_start..left_end].cmp(&bytes[right_start..right_end])
                }
                ResidentRowColumn::List {
                    offsets,
                    values,
                    element_validity,
                    ..
                } => compare_integer_list_rows(offsets, values, element_validity, left, right),
                ResidentRowColumn::Date { days, .. } => days[left].cmp(&days[right]),
                ResidentRowColumn::LocalTime { nanos, .. } => nanos[left].cmp(&nanos[right]),
                ResidentRowColumn::ZonedTime {
                    nanos,
                    offset_seconds,
                    ..
                } => (
                    normalized_resident_zoned_time(nanos[left], offset_seconds[left]),
                    offset_seconds[left],
                )
                    .cmp(&(
                        normalized_resident_zoned_time(nanos[right], offset_seconds[right]),
                        offset_seconds[right],
                    )),
                ResidentRowColumn::LocalDateTime { seconds, nanos, .. } => {
                    (seconds[left], nanos[left]).cmp(&(seconds[right], nanos[right]))
                }
                ResidentRowColumn::ZonedDateTime {
                    seconds,
                    nanos,
                    timezone_offsets,
                    timezone_bytes,
                    ..
                } => {
                    let left_timezone = &timezone_bytes
                        [timezone_offsets[left] as usize..timezone_offsets[left + 1] as usize];
                    let right_timezone = &timezone_bytes
                        [timezone_offsets[right] as usize..timezone_offsets[right + 1] as usize];
                    (seconds[left], nanos[left], left_timezone).cmp(&(
                        seconds[right],
                        nanos[right],
                        right_timezone,
                    ))
                }
            };
            if key.descending {
                ordering.reverse()
            } else {
                ordering
            }
        }
    }
}

fn compare_integer_list_rows(
    offsets: &[u32],
    values: &[i64],
    element_validity: &[u8],
    left: usize,
    right: usize,
) -> Ordering {
    let left_range = offsets[left] as usize..offsets[left + 1] as usize;
    let right_range = offsets[right] as usize..offsets[right + 1] as usize;
    let left_values = values[left_range.clone()]
        .iter()
        .zip(&element_validity[left_range.clone()]);
    let right_values = values[right_range.clone()]
        .iter()
        .zip(&element_validity[right_range.clone()]);
    for ((left_value, left_valid), (right_value, right_valid)) in left_values.zip(right_values) {
        let ordering = match (*left_valid != 0, *right_valid != 0) {
            (false, false) => Ordering::Equal,
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (true, true) => left_value.cmp(right_value),
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left_range.len().cmp(&right_range.len())
}

/// Mirrors the canonical tuple component in `cypher::order_key`: UTC-equivalent nanos first,
/// followed by the original offset as a deterministic tie-breaker. Saturation keeps this total
/// for an unvalidated in-memory value while canonical graph values remain in the non-overflowing
/// temporal range.
fn normalized_resident_zoned_time(nanos: i64, offset_seconds: i32) -> i64 {
    nanos.saturating_sub(i64::from(offset_seconds).saturating_mul(1_000_000_000))
}

fn resident_row_manifest_fingerprint(
    program: &ResidentRowProgram,
    sort_keys: &[ResidentRowSortKey],
    offset: usize,
    limit: usize,
    max_output_rows: usize,
    final_registers: &[u16],
    manifest: &ResidentRowProgramManifest,
) -> Result<ResidentRowManifestFingerprint> {
    let mut hash = StableManifestHash::new();
    hash.bytes(b"irongraph-resident-row-v8");
    hash.usize(program.instructions.len())?;
    for instruction in &program.instructions {
        hash.byte(instruction.output_type as u8);
        match &instruction.operation {
            ResidentRowOperation::InputColumn(column) => {
                hash.byte(12);
                hash.byte(column.value_type() as u8);
                hash.usize(column.len())?;
                for valid in column.validity() {
                    hash.byte(*valid);
                }
                match column {
                    ResidentRowColumn::Boolean { values, .. } => {
                        for value in values {
                            hash.byte(*value);
                        }
                    }
                    ResidentRowColumn::Integer { values, .. } => {
                        for value in values {
                            hash.u64(*value as u64);
                        }
                    }
                    ResidentRowColumn::Float { bits, .. } => {
                        for bits in bits {
                            hash.u64(*bits);
                        }
                    }
                    ResidentRowColumn::String { offsets, bytes, .. } => {
                        hash.usize(offsets.len())?;
                        for offset in offsets {
                            hash.u64(u64::from(*offset));
                        }
                        hash.usize(bytes.len())?;
                        hash.bytes(bytes);
                    }
                    ResidentRowColumn::List {
                        offsets,
                        values,
                        element_validity,
                        ..
                    } => {
                        hash.usize(offsets.len())?;
                        for offset in offsets {
                            hash.u64(u64::from(*offset));
                        }
                        hash.usize(values.len())?;
                        for value in values {
                            hash.u64(value.cast_unsigned());
                        }
                        hash.bytes(element_validity);
                    }
                    ResidentRowColumn::Date { days, .. } => {
                        for day in days {
                            hash.u64(day.cast_unsigned());
                        }
                    }
                    ResidentRowColumn::LocalTime { nanos, .. } => {
                        for nanos in nanos {
                            hash.u64(nanos.cast_unsigned());
                        }
                    }
                    ResidentRowColumn::ZonedTime {
                        nanos,
                        offset_seconds,
                        ..
                    } => {
                        for nanos in nanos {
                            hash.u64(nanos.cast_unsigned());
                        }
                        for offset in offset_seconds {
                            hash.u64(u64::from(offset.cast_unsigned()));
                        }
                    }
                    ResidentRowColumn::LocalDateTime { seconds, nanos, .. } => {
                        for seconds in seconds {
                            hash.u64(seconds.cast_unsigned());
                        }
                        for nanos in nanos {
                            hash.u64(u64::from(*nanos));
                        }
                    }
                    ResidentRowColumn::ZonedDateTime {
                        seconds,
                        nanos,
                        timezone_offsets,
                        timezone_bytes,
                        ..
                    } => {
                        for seconds in seconds {
                            hash.u64(seconds.cast_unsigned());
                        }
                        for nanos in nanos {
                            hash.u64(u64::from(*nanos));
                        }
                        hash.usize(timezone_offsets.len())?;
                        for offset in timezone_offsets {
                            hash.u64(u64::from(*offset));
                        }
                        hash.usize(timezone_bytes.len())?;
                        hash.bytes(timezone_bytes);
                    }
                }
            }
            ResidentRowOperation::LoadBooleanProperty { binding, property } => {
                hash.byte(1);
                hash.binding(*binding);
                hash.u64(property.0);
            }
            ResidentRowOperation::LoadIntegerProperty { binding, property } => {
                hash.byte(2);
                hash.binding(*binding);
                hash.u64(property.0);
            }
            ResidentRowOperation::LoadFloatProperty { binding, property } => {
                hash.byte(3);
                hash.binding(*binding);
                hash.u64(property.0);
            }
            ResidentRowOperation::LoadStringProperty {
                binding,
                property,
                maximum_bytes,
            } => {
                hash.byte(15);
                hash.binding(*binding);
                hash.u64(property.0);
                hash.u64(u64::from(*maximum_bytes));
            }
            ResidentRowOperation::LoadDateProperty { binding, property } => {
                hash.byte(18);
                hash.binding(*binding);
                hash.u64(property.0);
            }
            ResidentRowOperation::LoadLocalTimeProperty { binding, property } => {
                hash.byte(19);
                hash.binding(*binding);
                hash.u64(property.0);
            }
            ResidentRowOperation::LoadZonedTimeProperty { binding, property } => {
                hash.byte(20);
                hash.binding(*binding);
                hash.u64(property.0);
            }
            ResidentRowOperation::LoadLocalDateTimeProperty { binding, property } => {
                hash.byte(21);
                hash.binding(*binding);
                hash.u64(property.0);
            }
            ResidentRowOperation::LoadZonedDateTimeProperty {
                binding,
                property,
                maximum_timezone_bytes,
            } => {
                hash.byte(22);
                hash.binding(*binding);
                hash.u64(property.0);
                hash.u64(u64::from(*maximum_timezone_bytes));
            }
            ResidentRowOperation::LoadListProperty {
                binding,
                property,
                maximum_elements,
            } => {
                hash.byte(24);
                hash.binding(*binding);
                hash.u64(property.0);
                hash.u64(u64::from(*maximum_elements));
            }
            ResidentRowOperation::BooleanConstant(value) => {
                hash.byte(4);
                hash.byte(u8::from(*value));
            }
            ResidentRowOperation::IntegerConstant(value) => {
                hash.byte(5);
                hash.u64(*value as u64);
            }
            ResidentRowOperation::FloatConstant(bits) => {
                hash.byte(6);
                hash.u64(*bits);
            }
            ResidentRowOperation::StringConstant(value) => {
                hash.byte(16);
                hash.usize(value.len())?;
                hash.bytes(value.as_bytes());
            }
            ResidentRowOperation::BooleanNot { operand } => {
                hash.byte(7);
                hash.u16(*operand);
            }
            ResidentRowOperation::BooleanAnd { left, right } => {
                hash.byte(8);
                hash.u16(*left);
                hash.u16(*right);
            }
            ResidentRowOperation::NumericAdd { left, right } => {
                hash.byte(9);
                hash.u16(*left);
                hash.u16(*right);
            }
            ResidentRowOperation::NumericSubtract { left, right } => {
                hash.byte(13);
                hash.u16(*left);
                hash.u16(*right);
            }
            ResidentRowOperation::NumericMultiply { left, right } => {
                hash.byte(10);
                hash.u16(*left);
                hash.u16(*right);
            }
            ResidentRowOperation::IntegerModulo { left, right } => {
                hash.byte(14);
                hash.u16(*left);
                hash.u16(*right);
            }
            ResidentRowOperation::NumericNegate { operand } => {
                hash.byte(11);
                hash.u16(*operand);
            }
            ResidentRowOperation::StringConcat { left, right } => {
                hash.byte(17);
                hash.u16(*left);
                hash.u16(*right);
            }
            ResidentRowOperation::StringCoalesce { left, right } => {
                hash.byte(31);
                hash.u16(*left);
                hash.u16(*right);
            }
            ResidentRowOperation::BooleanToString { operand } => {
                hash.byte(30);
                hash.u16(*operand);
            }
            ResidentRowOperation::List { elements } => {
                hash.byte(25);
                hash.usize(elements.len())?;
                for element in elements {
                    hash.u16(*element);
                }
            }
            ResidentRowOperation::ListIndex { list, index } => {
                hash.byte(26);
                hash.u16(*list);
                hash.u16(*index);
            }
            ResidentRowOperation::ListConcat { left, right } => {
                hash.byte(27);
                hash.u16(*left);
                hash.u16(*right);
            }
            ResidentRowOperation::TemporalAddDuration {
                temporal,
                months,
                days,
                seconds,
                nanos,
            } => {
                hash.byte(23);
                hash.u16(*temporal);
                hash.u64(months.cast_unsigned());
                hash.u64(days.cast_unsigned());
                hash.u64(seconds.cast_unsigned());
                hash.u64(u64::from(nanos.cast_unsigned()));
            }
            ResidentRowOperation::TemporalAccessor {
                temporal,
                accessor,
                maximum_string_bytes,
                named_zone_table,
            } => {
                hash.byte(28);
                hash.u16(*temporal);
                hash.byte(*accessor as u8);
                hash.u64(u64::from(*maximum_string_bytes));
                hash.usize(named_zone_table.len())?;
                hash.bytes(named_zone_table);
            }
            ResidentRowOperation::LoadDurationAccessor {
                binding,
                property,
                accessor,
            } => {
                hash.byte(29);
                hash.binding(*binding);
                hash.u64(property.0);
                hash.byte(*accessor as u8);
            }
        }
    }
    hash.usize(sort_keys.len())?;
    for key in sort_keys {
        hash.u16(key.register);
        hash.byte(u8::from(key.descending));
        hash.byte(u8::from(key.nulls_first));
    }
    hash.usize(offset)?;
    hash.usize(limit)?;
    hash.usize(max_output_rows)?;
    hash.usize(final_registers.len())?;
    for register in final_registers {
        hash.u16(*register);
    }
    hash.usize(manifest.instruction_obligations.len())?;
    for obligation in manifest.obligations() {
        hash.u64(obligation.id);
        hash.byte(obligation.kind as u8);
        let ResidentObligationScope::Expression(index) = obligation.scope else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident row manifest uses a non-expression generic scope",
            ));
        };
        hash.u16(index);
    }
    Ok(ResidentRowManifestFingerprint(hash.finish()))
}

struct StableManifestHash {
    lanes: [u64; 4],
}

impl StableManifestHash {
    const fn new() -> Self {
        Self {
            lanes: [
                0xcbf2_9ce4_8422_2325,
                0x8422_2325_cbf2_9ce4,
                0x9e37_79b9_7f4a_7c15,
                0xd6e8_feb8_6659_fd93,
            ],
        }
    }

    fn byte(&mut self, byte: u8) {
        const PRIMES: [u64; 4] = [
            0x0000_0100_0000_01b3,
            0x9e37_79b1_85eb_ca87,
            0xc2b2_ae3d_27d4_eb4f,
            0x1656_67b1_9e37_79f9,
        ];
        for (lane, prime) in self.lanes.iter_mut().zip(PRIMES) {
            *lane ^= u64::from(byte);
            *lane = lane.wrapping_mul(prime).rotate_left(11);
        }
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.byte(*byte);
        }
    }

    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) -> Result<()> {
        self.u64(u64::try_from(value).map_err(|_| scratch_overflow())?);
        Ok(())
    }

    fn binding(&mut self, binding: ResidentEntityBinding) {
        match binding {
            ResidentEntityBinding::Node(ResidentNodeBinding::Start) => self.byte(1),
            ResidentEntityBinding::Node(ResidentNodeBinding::Intermediate(index)) => {
                self.byte(2);
                self.u16(index);
            }
            ResidentEntityBinding::Node(ResidentNodeBinding::End) => self.byte(3),
            ResidentEntityBinding::Relationship(index) => {
                self.byte(4);
                self.u16(index);
            }
        }
    }

    fn finish(self) -> [u8; 32] {
        let mut bytes = [0_u8; 32];
        for (index, lane) in self.lanes.into_iter().enumerate() {
            bytes[index * 8..index * 8 + 8].copy_from_slice(&lane.to_le_bytes());
        }
        bytes
    }
}

fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(scratch_overflow)
}

fn scratch_overflow() -> Error {
    Error::new(
        ErrorCode::ResultBudgetExceeded,
        "resident row program scratch shape overflow",
    )
}

const RESIDENT_DURATION_NANOS_PER_SECOND: i128 = 1_000_000_000;
const RESIDENT_DURATION_SECONDS_PER_DAY: f64 = 86_400.0;
const RESIDENT_DURATION_AVERAGE_SECONDS_PER_MONTH: f64 = 2_629_746.0;

#[derive(Clone, Copy)]
struct ResidentArithmeticDuration {
    months: i64,
    days: i64,
    seconds: i64,
    nanos: i32,
}

#[derive(Clone, Copy)]
enum ResidentArithmeticNumber {
    Integer(i64),
    Float(f64),
}

impl ResidentArithmeticNumber {
    fn as_f64(self) -> f64 {
        match self {
            Self::Integer(value) => value as f64,
            Self::Float(value) => value,
        }
    }
}

fn resident_arithmetic_duration(value: &ScalarValue) -> Option<ResidentArithmeticDuration> {
    let ScalarValue::Duration {
        months,
        days,
        seconds,
        nanos,
    } = value
    else {
        return None;
    };
    Some(ResidentArithmeticDuration {
        months: *months,
        days: *days,
        seconds: *seconds,
        nanos: *nanos,
    })
}

fn resident_arithmetic_number(value: &ScalarValue) -> Option<ResidentArithmeticNumber> {
    match value {
        ScalarValue::Integer(value) => Some(ResidentArithmeticNumber::Integer(*value)),
        ScalarValue::Float(value) => Some(ResidentArithmeticNumber::Float(value.into_inner())),
        _ => None,
    }
}

fn resident_normalized_duration(
    months: i128,
    days: i128,
    seconds: i128,
    nanos: i128,
) -> Result<ScalarValue> {
    let seconds = seconds
        .checked_add(nanos.div_euclid(RESIDENT_DURATION_NANOS_PER_SECOND))
        .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "duration seconds overflow"))?;
    let nanos = nanos.rem_euclid(RESIDENT_DURATION_NANOS_PER_SECOND);
    Ok(ScalarValue::Duration {
        months: i64::try_from(months)
            .map_err(|_| Error::new(ErrorCode::TemporalRange, "duration months overflow"))?,
        days: i64::try_from(days)
            .map_err(|_| Error::new(ErrorCode::TemporalRange, "duration days overflow"))?,
        seconds: i64::try_from(seconds)
            .map_err(|_| Error::new(ErrorCode::TemporalRange, "duration seconds overflow"))?,
        nanos: i32::try_from(nanos)
            .map_err(|_| Error::new(ErrorCode::TemporalRange, "duration nanoseconds overflow"))?,
    })
}

fn resident_combine_duration(
    left: ResidentArithmeticDuration,
    right: ResidentArithmeticDuration,
    sign: i128,
) -> Result<ScalarValue> {
    let combine = |left: i64, right: i64, component: &'static str| {
        let right = i128::from(right)
            .checked_mul(sign)
            .ok_or_else(|| Error::new(ErrorCode::TemporalRange, component))?;
        i128::from(left)
            .checked_add(right)
            .ok_or_else(|| Error::new(ErrorCode::TemporalRange, component))
    };
    resident_normalized_duration(
        combine(left.months, right.months, "duration months overflow")?,
        combine(left.days, right.days, "duration days overflow")?,
        combine(left.seconds, right.seconds, "duration seconds overflow")?,
        i128::from(left.nanos)
            .checked_add(i128::from(right.nanos).checked_mul(sign).ok_or_else(|| {
                Error::new(ErrorCode::TemporalRange, "duration nanoseconds overflow")
            })?)
            .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "duration nanoseconds overflow"))?,
    )
}

fn resident_duration_f64_to_i64(value: f64, component: &'static str) -> Result<i64> {
    const LOWER: f64 = i64::MIN as f64;
    const UPPER_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
    if value.is_finite() && value >= LOWER && value < UPPER_EXCLUSIVE {
        Ok(value.trunc() as i64)
    } else {
        Err(Error::new(
            ErrorCode::TemporalRange,
            format!("duration {component} component is outside the supported range"),
        ))
    }
}

fn resident_approximate_duration(
    months: f64,
    mut days: f64,
    mut seconds: f64,
    mut nanos: f64,
) -> Result<ScalarValue> {
    let months_whole = resident_duration_f64_to_i64(months, "months")?;
    days += RESIDENT_DURATION_AVERAGE_SECONDS_PER_MONTH * (months - months_whole as f64)
        / RESIDENT_DURATION_SECONDS_PER_DAY;
    let days_whole = resident_duration_f64_to_i64(days, "days")?;
    seconds += RESIDENT_DURATION_SECONDS_PER_DAY * (days - days_whole as f64);
    let seconds_whole = resident_duration_f64_to_i64(seconds, "seconds")?;
    nanos += RESIDENT_DURATION_NANOS_PER_SECOND as f64 * (seconds - seconds_whole as f64);
    let nanos_whole = resident_duration_f64_to_i64(nanos, "nanoseconds")?;
    resident_normalized_duration(
        i128::from(months_whole),
        i128::from(days_whole),
        i128::from(seconds_whole),
        i128::from(nanos_whole),
    )
}

fn resident_scale_duration(
    duration: ResidentArithmeticDuration,
    factor: ResidentArithmeticNumber,
) -> Result<ScalarValue> {
    match factor {
        ResidentArithmeticNumber::Integer(factor) => {
            let factor = i128::from(factor);
            resident_normalized_duration(
                i128::from(duration.months)
                    .checked_mul(factor)
                    .ok_or_else(|| {
                        Error::new(ErrorCode::TemporalRange, "duration months overflow")
                    })?,
                i128::from(duration.days)
                    .checked_mul(factor)
                    .ok_or_else(|| {
                        Error::new(ErrorCode::TemporalRange, "duration days overflow")
                    })?,
                i128::from(duration.seconds)
                    .checked_mul(factor)
                    .ok_or_else(|| {
                        Error::new(ErrorCode::TemporalRange, "duration seconds overflow")
                    })?,
                i128::from(duration.nanos)
                    .checked_mul(factor)
                    .ok_or_else(|| {
                        Error::new(ErrorCode::TemporalRange, "duration nanoseconds overflow")
                    })?,
            )
        }
        ResidentArithmeticNumber::Float(factor) => resident_approximate_duration(
            duration.months as f64 * factor,
            duration.days as f64 * factor,
            duration.seconds as f64 * factor,
            f64::from(duration.nanos) * factor,
        ),
    }
}

fn resident_negated_duration(
    duration: ResidentArithmeticDuration,
) -> Result<ResidentArithmeticDuration> {
    let ScalarValue::Duration {
        months,
        days,
        seconds,
        nanos,
    } = resident_normalized_duration(
        -i128::from(duration.months),
        -i128::from(duration.days),
        -i128::from(duration.seconds),
        -i128::from(duration.nanos),
    )?
    else {
        return Err(Error::internal(
            "duration negation returned the wrong scalar type",
        ));
    };
    Ok(ResidentArithmeticDuration {
        months,
        days,
        seconds,
        nanos,
    })
}

const fn resident_is_temporal_scalar(value: &ScalarValue) -> bool {
    matches!(
        value,
        ScalarValue::Date(_)
            | ScalarValue::LocalTime(_)
            | ScalarValue::ZonedTime { .. }
            | ScalarValue::LocalDateTime { .. }
            | ScalarValue::ZonedDateTime { .. }
    )
}

/// Backend-neutral native duration arithmetic. `None` means incompatible non-NULL operand
/// types; semantic range and zero-division failures retain their canonical Cypher error codes.
pub fn evaluate_resident_temporal_arithmetic(
    operation: super::ResidentTemporalArithmeticOperation,
    left: &ScalarValue,
    right: &ScalarValue,
) -> Result<Option<ScalarValue>> {
    if matches!(left, ScalarValue::Null) || matches!(right, ScalarValue::Null) {
        return Ok(Some(ScalarValue::Null));
    }
    let left_duration = resident_arithmetic_duration(left);
    let right_duration = resident_arithmetic_duration(right);
    let result = match operation {
        super::ResidentTemporalArithmeticOperation::Add => match (left_duration, right_duration) {
            (Some(left), Some(right)) => resident_combine_duration(left, right, 1).map(Some),
            (Some(duration), None) if resident_is_temporal_scalar(right) => {
                crate::execution::apply_duration_to_temporal_scalar(
                    right,
                    duration.months,
                    duration.days,
                    duration.seconds,
                    duration.nanos,
                )
                .map(Some)
            }
            (None, Some(duration)) if resident_is_temporal_scalar(left) => {
                crate::execution::apply_duration_to_temporal_scalar(
                    left,
                    duration.months,
                    duration.days,
                    duration.seconds,
                    duration.nanos,
                )
                .map(Some)
            }
            _ => Ok(None),
        },
        super::ResidentTemporalArithmeticOperation::Subtract => {
            match (left_duration, right_duration) {
                (Some(left), Some(right)) => resident_combine_duration(left, right, -1).map(Some),
                (None, Some(duration)) if resident_is_temporal_scalar(left) => {
                    let duration = resident_negated_duration(duration)?;
                    crate::execution::apply_duration_to_temporal_scalar(
                        left,
                        duration.months,
                        duration.days,
                        duration.seconds,
                        duration.nanos,
                    )
                    .map(Some)
                }
                _ => Ok(None),
            }
        }
        super::ResidentTemporalArithmeticOperation::Multiply => {
            if let (Some(duration), Some(factor)) =
                (left_duration, resident_arithmetic_number(right))
            {
                resident_scale_duration(duration, factor).map(Some)
            } else if let (Some(factor), Some(duration)) =
                (resident_arithmetic_number(left), right_duration)
            {
                resident_scale_duration(duration, factor).map(Some)
            } else {
                Ok(None)
            }
        }
        super::ResidentTemporalArithmeticOperation::Divide => {
            let (Some(duration), Some(divisor)) =
                (left_duration, resident_arithmetic_number(right))
            else {
                return Ok(None);
            };
            let divisor = divisor.as_f64();
            if divisor == 0.0 {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "cannot divide a duration by zero",
                ));
            }
            resident_approximate_duration(
                duration.months as f64 / divisor,
                duration.days as f64 / divisor,
                duration.seconds as f64 / divisor,
                f64::from(duration.nanos) / divisor,
            )
            .map(Some)
        }
    }?;
    Ok(result)
}

#[cfg(test)]
mod nullable_relation_tests {
    use crate::{
        Bookmark, ErrorCode, ProjectId, Result,
        graph::LayerMask,
        types::{LabelId, PropertyId, RelationshipTypeId},
    };

    use super::{
        RESIDENT_NULLABLE_RELATION_NULL_ROW, ResidentDeviceCompletion, ResidentExecutionId,
        ResidentNullableNodeDomain, ResidentNullableRelationBindingKind,
        ResidentNullableRelationCapacities, ResidentNullableRelationFilterPlacement,
        ResidentNullableRelationFilterStage, ResidentNullableRelationGeneration,
        ResidentNullableRelationMatchMode, ResidentNullableRelationObligationKind,
        ResidentNullableRelationOptionalGroup, ResidentNullableRelationOutputBinding,
        ResidentNullableRelationOutputColumn, ResidentNullableRelationOutputSource,
        ResidentNullableRelationPredicate, ResidentNullableRelationPredicateProgram,
        ResidentNullableRelationPredicateValue, ResidentNullableRelationProgram,
        ResidentNullableRelationProjectionBinding, ResidentNullableRelationPropertyLane,
        ResidentNullableRelationPropertyShape, ResidentNullableRelationReceipt,
        ResidentNullableRelationRequest, ResidentNullableRelationResult,
        ResidentNullableRelationResultParts, ResidentNullableRelationSlot,
        ResidentNullableRelationStage, ResidentNullableRelationTarget,
        ResidentNullableRelationshipDomain,
    };

    fn generation() -> ResidentNullableRelationGeneration {
        ResidentNullableRelationGeneration {
            project: ProjectId(uuid::Uuid::from_u128(7)),
            bookmark: Bookmark { term: 3, index: 9 },
            graph_revision: 9,
            layout_version: 2,
            catalog_generation: [0x5a; 32],
        }
    }

    fn string_predicate_request(
        predicate: ResidentNullableRelationPredicate,
        execution_low: u64,
    ) -> Result<ResidentNullableRelationRequest> {
        let node = ResidentNullableRelationSlot(0);
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: node,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "i".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: node,
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 4, 0, 4, 0, 4)?;
        ResidentNullableRelationRequest::build_with_predicates(
            generation(),
            ResidentExecutionId {
                high: 89,
                low: execution_low,
            },
            program,
            ResidentNullableRelationPredicateProgram {
                filters: vec![ResidentNullableRelationFilterStage {
                    placement: ResidentNullableRelationFilterPlacement::RelationAfter { stage: 0 },
                    predicate,
                }],
                optional_groups: Vec::new(),
            },
            capacities,
            907,
        )
    }

    fn string_var_value() -> ResidentNullableRelationPredicateValue {
        ResidentNullableRelationPredicateValue::StringProperty {
            slot: ResidentNullableRelationSlot(0),
            kind: ResidentNullableRelationBindingKind::Node,
            property: PropertyId(17),
        }
    }

    fn string_var_greater_than_te() -> ResidentNullableRelationPredicate {
        ResidentNullableRelationPredicate::CompareString {
            left: string_var_value(),
            operation: super::CompareOp::Greater,
            right: ResidentNullableRelationPredicateValue::String("te".into()),
        }
    }

    #[test]
    fn nullable_string_comparison_seals_all_six_operations_without_weakening_types() -> Result<()> {
        for (index, operation) in [
            super::CompareOp::Eq,
            super::CompareOp::NotEq,
            super::CompareOp::Less,
            super::CompareOp::LessOrEqual,
            super::CompareOp::Greater,
            super::CompareOp::GreaterOrEqual,
        ]
        .into_iter()
        .enumerate()
        {
            let request = string_predicate_request(
                ResidentNullableRelationPredicate::CompareString {
                    left: string_var_value(),
                    operation,
                    right: ResidentNullableRelationPredicateValue::String("te".into()),
                },
                101 + index as u64,
            )?;
            request.validate()?;
            assert!(
                request
                    .predicate_program
                    .requires_string_property_equality()
            );
            assert_eq!(
                request.property_lanes(),
                &[ResidentNullableRelationPropertyLane {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(17),
                    shape: ResidentNullableRelationPropertyShape::String,
                }]
            );
        }

        let error = string_predicate_request(
            ResidentNullableRelationPredicate::CompareString {
                left: ResidentNullableRelationPredicateValue::Integer(1),
                operation: super::CompareOp::Greater,
                right: ResidentNullableRelationPredicateValue::String("te".into()),
            },
            113,
        )
        .expect_err("an ordered string opcode must not admit an integer operand");
        assert_eq!(error.code, ErrorCode::QueryType);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable string comparison requires string-compatible operands"
        );
        Ok(())
    }

    #[test]
    fn nullable_string_greater_seals_match_where5_three_valued_shapes() -> Result<()> {
        let predicates = [
            string_var_greater_than_te(),
            ResidentNullableRelationPredicate::And(
                Box::new(string_var_greater_than_te()),
                Box::new(ResidentNullableRelationPredicate::HasLabels {
                    node: ResidentNullableRelationSlot(0),
                    labels: ResidentNullableNodeDomain::Known(vec![LabelId(29)]),
                }),
            ),
            ResidentNullableRelationPredicate::And(
                Box::new(string_var_greater_than_te()),
                Box::new(ResidentNullableRelationPredicate::IsNull {
                    value: string_var_value(),
                    negated: true,
                }),
            ),
            ResidentNullableRelationPredicate::Or(
                Box::new(ResidentNullableRelationPredicate::Constant(None)),
                Box::new(ResidentNullableRelationPredicate::IsNull {
                    value: string_var_value(),
                    negated: true,
                }),
            ),
        ];

        for (index, predicate) in predicates.into_iter().enumerate() {
            let request = string_predicate_request(predicate, 127 + index as u64)?;
            request.validate()?;
            assert_eq!(request.predicate_program.filters.len(), 1);
            assert_eq!(
                request.property_lanes(),
                &[ResidentNullableRelationPropertyLane {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(17),
                    shape: ResidentNullableRelationPropertyShape::String,
                }]
            );
        }
        Ok(())
    }

    fn match3_three_bound_nodes_program() -> ResidentNullableRelationProgram {
        let a = ResidentNullableRelationSlot(0);
        let b = ResidentNullableRelationSlot(1);
        let c = ResidentNullableRelationSlot(2);
        let first_relationship = ResidentNullableRelationSlot(3);
        let x = ResidentNullableRelationSlot(4);
        let second_relationship = ResidentNullableRelationSlot(5);
        let third_relationship = ResidentNullableRelationSlot(6);
        ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: a,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: b,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: c,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 1,
                    source: a,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(first_relationship),
                    different_from: vec![],
                    target: ResidentNullableRelationTarget::Introduce(x),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 1,
                    source: b,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(second_relationship),
                    different_from: vec![first_relationship],
                    target: ResidentNullableRelationTarget::Existing(x),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 1,
                    source: c,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(third_relationship),
                    different_from: vec![first_relationship, second_relationship],
                    target: ResidentNullableRelationTarget::Existing(x),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "x".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: x,
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        }
    }

    fn optional_known_empty_request() -> Result<ResidentNullableRelationRequest> {
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    output: ResidentNullableRelationSlot(7),
                    labels: ResidentNullableNodeDomain::KnownEmpty,
                },
                ResidentNullableRelationStage::ScopeProject {
                    bindings: vec![ResidentNullableRelationProjectionBinding {
                        variable: "alias".to_owned(),
                        source: ResidentNullableRelationSlot(7),
                        output: ResidentNullableRelationSlot(42),
                        row_limit: None,
                    }],
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "alias".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: ResidentNullableRelationSlot(42),
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 3, 2, 3, 2, 8)?;
        ResidentNullableRelationRequest::build(
            generation(),
            ResidentExecutionId { high: 4, low: 5 },
            program,
            capacities,
            11,
        )
    }

    #[allow(dead_code)]
    fn atomic_predicate_request() -> Result<ResidentNullableRelationRequest> {
        let root = ResidentNullableRelationSlot(0);
        let one_hop_relationship = ResidentNullableRelationSlot(1);
        let one_hop_target = ResidentNullableRelationSlot(2);
        let first_relationship = ResidentNullableRelationSlot(3);
        let middle = ResidentNullableRelationSlot(4);
        let second_relationship = ResidentNullableRelationSlot(5);
        let target = ResidentNullableRelationSlot(6);
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: root,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    uniqueness_group: 0,
                    source: root,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(one_hop_relationship),
                    different_from: Vec::new(),
                    target: ResidentNullableRelationTarget::Introduce(one_hop_target),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    uniqueness_group: 1,
                    source: root,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(first_relationship),
                    different_from: Vec::new(),
                    target: ResidentNullableRelationTarget::Introduce(middle),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    uniqueness_group: 1,
                    source: middle,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(second_relationship),
                    different_from: vec![first_relationship],
                    target: ResidentNullableRelationTarget::Introduce(target),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "target".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: target,
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        };
        let predicate_program = ResidentNullableRelationPredicateProgram {
            // Deliberately store these out of execution order. The proof schedule owns ordering.
            filters: vec![
                ResidentNullableRelationFilterStage {
                    placement: ResidentNullableRelationFilterPlacement::OptionalCandidates {
                        stage: 1,
                    },
                    predicate: ResidentNullableRelationPredicate::CompareInteger {
                        left: ResidentNullableRelationPredicateValue::IntegerProperty {
                            slot: one_hop_target,
                            kind: ResidentNullableRelationBindingKind::Node,
                            property: crate::types::PropertyId(1),
                        },
                        operation: super::CompareOp::Greater,
                        right: ResidentNullableRelationPredicateValue::Integer(0),
                    },
                },
                ResidentNullableRelationFilterStage {
                    placement: ResidentNullableRelationFilterPlacement::OptionalGroupCandidates {
                        group: 0,
                    },
                    predicate: ResidentNullableRelationPredicate::CompareInteger {
                        left: ResidentNullableRelationPredicateValue::IntegerProperty {
                            slot: target,
                            kind: ResidentNullableRelationBindingKind::Node,
                            property: crate::types::PropertyId(2),
                        },
                        operation: super::CompareOp::Less,
                        right: ResidentNullableRelationPredicateValue::Integer(10),
                    },
                },
                ResidentNullableRelationFilterStage {
                    placement: ResidentNullableRelationFilterPlacement::RelationAfter { stage: 0 },
                    predicate: ResidentNullableRelationPredicate::HasLabels {
                        node: root,
                        labels: ResidentNullableNodeDomain::Any,
                    },
                },
                ResidentNullableRelationFilterStage {
                    placement: ResidentNullableRelationFilterPlacement::RelationAfter { stage: 3 },
                    predicate: ResidentNullableRelationPredicate::IsNull {
                        value: ResidentNullableRelationPredicateValue::Binding {
                            slot: second_relationship,
                            kind: ResidentNullableRelationBindingKind::Relationship,
                        },
                        negated: false,
                    },
                },
            ],
            optional_groups: vec![ResidentNullableRelationOptionalGroup {
                first_stage: 2,
                last_stage: 3,
            }],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 8, 3, 2, 3, 32)?;
        ResidentNullableRelationRequest::build_with_predicates(
            generation(),
            ResidentExecutionId { high: 17, low: 19 },
            program,
            predicate_program,
            capacities,
            101,
        )
    }

    fn property_lane_request() -> Result<ResidentNullableRelationRequest> {
        let root = ResidentNullableRelationSlot(0);
        let relationship = ResidentNullableRelationSlot(1);
        let target = ResidentNullableRelationSlot(2);
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: root,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 0,
                    source: root,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(relationship),
                    different_from: Vec::new(),
                    target: ResidentNullableRelationTarget::Introduce(target),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![
                        ResidentNullableRelationOutputBinding {
                            name: "target".to_owned(),
                            source: ResidentNullableRelationOutputSource::Entity {
                                slot: target,
                                kind: ResidentNullableRelationBindingKind::Node,
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "root_count".to_owned(),
                            source: ResidentNullableRelationOutputSource::IntegerProperty {
                                slot: root,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(11),
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "target_name".to_owned(),
                            source: ResidentNullableRelationOutputSource::StringProperty {
                                slot: target,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(12),
                                maximum_bytes: 7,
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "edge_count".to_owned(),
                            source: ResidentNullableRelationOutputSource::IntegerProperty {
                                slot: relationship,
                                kind: ResidentNullableRelationBindingKind::Relationship,
                                property: PropertyId(13),
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "edge_name".to_owned(),
                            source: ResidentNullableRelationOutputSource::StringProperty {
                                slot: relationship,
                                kind: ResidentNullableRelationBindingKind::Relationship,
                                property: PropertyId(14),
                                maximum_bytes: 9,
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "missing".to_owned(),
                            source: ResidentNullableRelationOutputSource::NullProperty {
                                slot: target,
                                kind: ResidentNullableRelationBindingKind::Node,
                            },
                        },
                    ],
                },
            ],
        };
        let predicate_program = ResidentNullableRelationPredicateProgram {
            filters: vec![ResidentNullableRelationFilterStage {
                placement: ResidentNullableRelationFilterPlacement::RelationAfter { stage: 1 },
                predicate: ResidentNullableRelationPredicate::And(
                    Box::new(ResidentNullableRelationPredicate::And(
                        Box::new(ResidentNullableRelationPredicate::CompareInteger {
                            left: ResidentNullableRelationPredicateValue::IntegerProperty {
                                slot: root,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(11),
                            },
                            operation: super::CompareOp::Eq,
                            // A second slot gathering the same physical property must not invent
                            // a duplicate registry lane.
                            right: ResidentNullableRelationPredicateValue::IntegerProperty {
                                slot: target,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(11),
                            },
                        }),
                        Box::new(ResidentNullableRelationPredicate::CompareString {
                            left: ResidentNullableRelationPredicateValue::StringProperty {
                                slot: target,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(12),
                            },
                            operation: super::CompareOp::Eq,
                            right: ResidentNullableRelationPredicateValue::String("node".into()),
                        }),
                    )),
                    Box::new(ResidentNullableRelationPredicate::And(
                        Box::new(ResidentNullableRelationPredicate::CompareInteger {
                            left: ResidentNullableRelationPredicateValue::IntegerProperty {
                                slot: relationship,
                                kind: ResidentNullableRelationBindingKind::Relationship,
                                property: PropertyId(13),
                            },
                            operation: super::CompareOp::Greater,
                            right: ResidentNullableRelationPredicateValue::Integer(0),
                        }),
                        Box::new(ResidentNullableRelationPredicate::CompareString {
                            left: ResidentNullableRelationPredicateValue::StringProperty {
                                slot: relationship,
                                kind: ResidentNullableRelationBindingKind::Relationship,
                                property: PropertyId(14),
                            },
                            operation: super::CompareOp::NotEq,
                            right: ResidentNullableRelationPredicateValue::String("edge".into()),
                        }),
                    )),
                ),
            }],
            optional_groups: Vec::new(),
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 4, 4, 2, 2, 16)?;
        ResidentNullableRelationRequest::build_with_predicates(
            generation(),
            ResidentExecutionId { high: 37, low: 41 },
            program,
            predicate_program,
            capacities,
            307,
        )
    }

    fn different_relationships_program() -> ResidentNullableRelationProgram {
        let root = ResidentNullableRelationSlot(0);
        let first_relationship = ResidentNullableRelationSlot(1);
        let first_target = ResidentNullableRelationSlot(2);
        let second_relationship = ResidentNullableRelationSlot(3);
        let second_target = ResidentNullableRelationSlot(4);
        let third_relationship = ResidentNullableRelationSlot(5);
        let third_target = ResidentNullableRelationSlot(6);
        ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: root,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 0,
                    source: root,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(first_relationship),
                    different_from: Vec::new(),
                    target: ResidentNullableRelationTarget::Introduce(first_target),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 0,
                    source: first_target,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(second_relationship),
                    different_from: vec![first_relationship],
                    target: ResidentNullableRelationTarget::Introduce(second_target),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 0,
                    source: second_target,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(third_relationship),
                    different_from: vec![first_relationship, second_relationship],
                    target: ResidentNullableRelationTarget::Introduce(third_target),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "target".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: third_target,
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        }
    }

    fn different_relationships_request() -> Result<ResidentNullableRelationRequest> {
        let program = different_relationships_program();
        let capacities = ResidentNullableRelationCapacities::derive(&program, 7, 3, 1, 1, 8)?;
        ResidentNullableRelationRequest::build(
            generation(),
            ResidentExecutionId { high: 29, low: 31 },
            program,
            capacities,
            211,
        )
    }

    fn expansion_uniqueness_mut(
        program: &mut ResidentNullableRelationProgram,
        stage: usize,
    ) -> (&mut u32, &mut Vec<ResidentNullableRelationSlot>) {
        let ResidentNullableRelationStage::Expand {
            uniqueness_group,
            different_from,
            ..
        } = &mut program.stages[stage]
        else {
            panic!("DifferentRelationships fixture stage {stage} is not an expansion")
        };
        (uniqueness_group, different_from)
    }

    fn assert_different_relationships_rejected(
        program: &ResidentNullableRelationProgram,
        expected_message: &str,
    ) {
        let error = program
            .validate()
            .expect_err("corrupt DifferentRelationships program must be rejected");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(error.message.as_ref(), expected_message);
    }

    fn valid_receipts(
        request: &ResidentNullableRelationRequest,
    ) -> Vec<ResidentNullableRelationReceipt> {
        let cardinalities = [(1, 0, 0, 1, 1), (1, 1, 1, 0, 1), (1, 1, 1, 0, 1)];
        request
            .obligations()
            .zip(cardinalities)
            .map(
                |(obligation, (input, matched_inputs, candidates, null_extensions, output))| {
                    ResidentNullableRelationReceipt {
                        execution: request.execution,
                        obligation,
                        input_cardinality: input,
                        matched_input_cardinality: matched_inputs,
                        candidate_cardinality: candidates,
                        null_extension_cardinality: null_extensions,
                        output_cardinality: output,
                        completion: ResidentDeviceCompletion::CpuReference,
                    }
                },
            )
            .collect()
    }

    fn valid_result_parts(
        request: &ResidentNullableRelationRequest,
    ) -> Result<ResidentNullableRelationResultParts> {
        Ok(ResidentNullableRelationResultParts {
            generation: request.generation,
            execution: request.execution,
            fingerprint: request.manifest.fingerprint,
            row_count: 1,
            columns: vec![ResidentNullableRelationOutputColumn::Entity {
                slot: ResidentNullableRelationSlot(42),
                kind: ResidentNullableRelationBindingKind::Node,
                rows: vec![RESIDENT_NULLABLE_RELATION_NULL_ROW],
            }],
            receipts: valid_receipts(request),
            scratch_bytes: request.scratch_bytes()?,
        })
    }

    fn typed_output_request() -> Result<ResidentNullableRelationRequest> {
        let node = ResidentNullableRelationSlot(0);
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: node,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![
                        ResidentNullableRelationOutputBinding {
                            name: "node".to_owned(),
                            source: ResidentNullableRelationOutputSource::Entity {
                                slot: node,
                                kind: ResidentNullableRelationBindingKind::Node,
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "count".to_owned(),
                            source: ResidentNullableRelationOutputSource::IntegerProperty {
                                slot: node,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(21),
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "name".to_owned(),
                            source: ResidentNullableRelationOutputSource::StringProperty {
                                slot: node,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(22),
                                maximum_bytes: 2,
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "absent".to_owned(),
                            source: ResidentNullableRelationOutputSource::NullProperty {
                                slot: node,
                                kind: ResidentNullableRelationBindingKind::Node,
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "price".to_owned(),
                            source: ResidentNullableRelationOutputSource::FloatProperty {
                                slot: node,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(23),
                            },
                        },
                    ],
                },
            ],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 3, 0, 3, 0, 3)?;
        ResidentNullableRelationRequest::build(
            generation(),
            ResidentExecutionId { high: 53, low: 59 },
            program,
            capacities,
            503,
        )
    }

    fn typed_output_result_parts(
        request: &ResidentNullableRelationRequest,
    ) -> Result<ResidentNullableRelationResultParts> {
        let cardinalities = [(1, 1, 3, 0, 3), (3, 3, 3, 0, 3)];
        let receipts = request
            .obligations()
            .zip(cardinalities)
            .map(
                |(obligation, (input, matched_inputs, candidates, null_extensions, output))| {
                    ResidentNullableRelationReceipt {
                        execution: request.execution,
                        obligation,
                        input_cardinality: input,
                        matched_input_cardinality: matched_inputs,
                        candidate_cardinality: candidates,
                        null_extension_cardinality: null_extensions,
                        output_cardinality: output,
                        completion: ResidentDeviceCompletion::CpuReference,
                    }
                },
            )
            .collect();
        Ok(ResidentNullableRelationResultParts {
            generation: request.generation,
            execution: request.execution,
            fingerprint: request.manifest.fingerprint,
            row_count: 3,
            columns: vec![
                ResidentNullableRelationOutputColumn::Entity {
                    slot: ResidentNullableRelationSlot(0),
                    kind: ResidentNullableRelationBindingKind::Node,
                    rows: vec![0, 1, 2],
                },
                ResidentNullableRelationOutputColumn::IntegerProperty {
                    slot: ResidentNullableRelationSlot(0),
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(21),
                    source_rows: vec![0, 1, 2],
                    values: vec![7, 0, 9],
                    validity: vec![1, 0, 1],
                },
                ResidentNullableRelationOutputColumn::StringProperty {
                    slot: ResidentNullableRelationSlot(0),
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(22),
                    maximum_bytes: 2,
                    source_rows: vec![0, 1, 2],
                    offsets: vec![0, 0, 0, 2],
                    bytes: b"ok".to_vec(),
                    validity: vec![1, 0, 1],
                },
                ResidentNullableRelationOutputColumn::NullProperty {
                    slot: ResidentNullableRelationSlot(0),
                    kind: ResidentNullableRelationBindingKind::Node,
                    source_rows: vec![0, 1, 2],
                },
                ResidentNullableRelationOutputColumn::FloatProperty {
                    slot: ResidentNullableRelationSlot(0),
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(23),
                    source_rows: vec![0, 1, 2],
                    bits: vec![1.5_f64.to_bits(), 0, (-0.0_f64).to_bits()],
                    validity: vec![1, 0, 1],
                },
            ],
            receipts,
            scratch_bytes: request.scratch_bytes()?,
        })
    }

    fn relationship_type_output_request() -> Result<ResidentNullableRelationRequest> {
        let source = ResidentNullableRelationSlot(0);
        let relationship = ResidentNullableRelationSlot(1);
        let target = ResidentNullableRelationSlot(2);
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: source,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    uniqueness_group: 0,
                    source,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(relationship),
                    different_from: Vec::new(),
                    target: ResidentNullableRelationTarget::Introduce(target),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "type(r)".to_owned(),
                        source: ResidentNullableRelationOutputSource::RelationshipType {
                            slot: relationship,
                        },
                    }],
                },
            ],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 2, 1, 2, 1, 2)?;
        ResidentNullableRelationRequest::build(
            generation(),
            ResidentExecutionId { high: 71, low: 73 },
            program,
            capacities,
            701,
        )
    }

    fn relationship_type_output_parts(
        request: &ResidentNullableRelationRequest,
    ) -> Result<ResidentNullableRelationResultParts> {
        let cardinalities = [(1, 1, 2, 0, 2), (2, 1, 1, 1, 2), (2, 2, 2, 0, 2)];
        let receipts = request
            .obligations()
            .zip(cardinalities)
            .map(
                |(obligation, (input, matched_inputs, candidates, null_extensions, output))| {
                    ResidentNullableRelationReceipt {
                        execution: request.execution,
                        obligation,
                        input_cardinality: input,
                        matched_input_cardinality: matched_inputs,
                        candidate_cardinality: candidates,
                        null_extension_cardinality: null_extensions,
                        output_cardinality: output,
                        completion: ResidentDeviceCompletion::CpuReference,
                    }
                },
            )
            .collect();
        Ok(ResidentNullableRelationResultParts {
            generation: request.generation,
            execution: request.execution,
            fingerprint: request.manifest.fingerprint,
            row_count: 2,
            columns: vec![ResidentNullableRelationOutputColumn::RelationshipType {
                slot: ResidentNullableRelationSlot(1),
                source_rows: vec![0, RESIDENT_NULLABLE_RELATION_NULL_ROW],
                relationship_types: vec![RelationshipTypeId(7), RelationshipTypeId(0)],
            }],
            receipts,
            scratch_bytes: request.scratch_bytes()?,
        })
    }

    #[test]
    fn nullable_relation_seals_arbitrary_slots_capacities_and_ordered_obligations() -> Result<()> {
        let request = optional_known_empty_request()?;
        request.validate()?;
        assert_eq!(request.capacities.binding_slot_count, 43);
        assert_eq!(request.capacities.maximum_live_bindings, 1);
        assert_eq!(request.capacities.stage_input_rows, [1, 1, 1]);
        assert_eq!(request.capacities.stage_candidate_rows, [0, 1, 1]);
        assert_eq!(request.capacities.stage_output_rows, [1, 1, 1]);
        assert_eq!(
            request
                .obligations()
                .map(|obligation| (obligation.id, obligation.stage, obligation.kind))
                .collect::<Vec<_>>(),
            [
                (
                    11,
                    0,
                    ResidentNullableRelationObligationKind::OptionalNodeScan
                ),
                (
                    12,
                    1,
                    ResidentNullableRelationObligationKind::ScopeProjection
                ),
                (
                    13,
                    2,
                    ResidentNullableRelationObligationKind::FinalProjection
                ),
            ]
        );
        assert_ne!(request.manifest.fingerprint.0, [0; 32]);
        assert_eq!(
            request.scratch_bytes()?,
            2 * std::mem::size_of::<u32>()
                + (4 * std::mem::size_of::<u32>() + std::mem::size_of::<u8>())
                + std::mem::size_of::<u32>()
                + 3 * std::mem::size_of::<ResidentNullableRelationReceipt>()
        );
        Ok(())
    }

    #[test]
    fn nullable_relation_typed_outputs_share_property_lanes_and_account_exact_packet_bytes()
    -> Result<()> {
        let request = typed_output_request()?;
        assert_eq!(
            request.property_lanes(),
            [
                ResidentNullableRelationPropertyLane {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(21),
                    shape: ResidentNullableRelationPropertyShape::Integer,
                },
                ResidentNullableRelationPropertyLane {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(22),
                    shape: ResidentNullableRelationPropertyShape::String,
                },
                ResidentNullableRelationPropertyLane {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(23),
                    shape: ResidentNullableRelationPropertyShape::Float,
                },
            ]
        );
        assert_eq!(request.capacities.final_output_columns, 5);
        // Entity=4; integer=4+8+1; two-byte string=4+4+1+2; null=4; float=4+8+1.
        assert_eq!(request.capacities.final_output_bytes_per_row, 45);
        assert_eq!(request.capacities.final_output_fixed_bytes, 4);
        assert_eq!(
            request.scratch_bytes()?,
            3 * std::mem::size_of::<u32>() * 2
                + 3 * (4 * std::mem::size_of::<u32>() + std::mem::size_of::<u8>())
                + 3 * 45
                + 4
                + 2 * std::mem::size_of::<ResidentNullableRelationReceipt>()
        );

        let validated =
            ResidentNullableRelationResult::completed(typed_output_result_parts(&request)?)
                .validate(&request, ResidentDeviceCompletion::CpuReference)?;
        assert_eq!(validated.row_count(), 3);
        let ResidentNullableRelationOutputColumn::StringProperty {
            offsets, validity, ..
        } = &validated.columns()[2]
        else {
            panic!("typed fixture string output changed shape")
        };
        assert_eq!(offsets, &[0, 0, 0, 2]);
        assert_eq!(validity, &[1, 0, 1]);
        let ResidentNullableRelationOutputColumn::FloatProperty { bits, validity, .. } =
            &validated.columns()[4]
        else {
            panic!("typed fixture float output changed shape")
        };
        assert_eq!(bits, &[1.5_f64.to_bits(), 0, (-0.0_f64).to_bits()]);
        assert_eq!(validity, &[1, 0, 1]);
        Ok(())
    }

    #[test]
    fn nullable_relation_typed_output_corruption_fails_closed() -> Result<()> {
        let request = typed_output_request()?;

        let mut wrong_source = typed_output_result_parts(&request)?;
        let ResidentNullableRelationOutputColumn::IntegerProperty { property, .. } =
            &mut wrong_source.columns[1]
        else {
            panic!("typed fixture integer output changed shape")
        };
        *property = PropertyId(99);
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(wrong_source)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        let mut hidden_integer = typed_output_result_parts(&request)?;
        let ResidentNullableRelationOutputColumn::IntegerProperty { values, .. } =
            &mut hidden_integer.columns[1]
        else {
            panic!("typed fixture integer output changed shape")
        };
        values[1] = 44;
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(hidden_integer)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        let mut over_width = typed_output_result_parts(&request)?;
        let ResidentNullableRelationOutputColumn::StringProperty {
            offsets,
            bytes,
            validity,
            ..
        } = &mut over_width.columns[2]
        else {
            panic!("typed fixture string output changed shape")
        };
        *offsets = vec![0, 3, 3, 3];
        *bytes = b"bad".to_vec();
        *validity = vec![1, 0, 1];
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(over_width)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        let mut hidden_string = typed_output_result_parts(&request)?;
        let ResidentNullableRelationOutputColumn::StringProperty { offsets, bytes, .. } =
            &mut hidden_string.columns[2]
        else {
            panic!("typed fixture string output changed shape")
        };
        *offsets = vec![0, 0, 1, 3];
        *bytes = b"xok".to_vec();
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(hidden_string)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        let mut hidden_float = typed_output_result_parts(&request)?;
        let ResidentNullableRelationOutputColumn::FloatProperty { bits, .. } =
            &mut hidden_float.columns[4]
        else {
            panic!("typed fixture float output changed shape")
        };
        bits[1] = 1.0_f64.to_bits();
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(hidden_float)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        let mut non_finite_float = typed_output_result_parts(&request)?;
        let ResidentNullableRelationOutputColumn::FloatProperty { bits, .. } =
            &mut non_finite_float.columns[4]
        else {
            panic!("typed fixture float output changed shape")
        };
        bits[0] = f64::NAN.to_bits();
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(non_finite_float)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn nullable_relationship_type_output_seals_kind_bytes_fingerprint_and_null_payload()
    -> Result<()> {
        let request = relationship_type_output_request()?;
        request.validate()?;
        assert!(request.property_lanes().is_empty());
        assert_eq!(request.capacities.final_output_columns, 1);
        assert_eq!(
            request.capacities.final_output_bytes_per_row,
            (std::mem::size_of::<u32>() + std::mem::size_of::<u64>()) as u64
        );
        assert_eq!(request.capacities.final_output_fixed_bytes, 0);
        let validated =
            ResidentNullableRelationResult::completed(relationship_type_output_parts(&request)?)
                .validate(&request, ResidentDeviceCompletion::CpuReference)?;
        assert_eq!(validated.row_count(), 2);

        let mut missing_token = relationship_type_output_parts(&request)?;
        let ResidentNullableRelationOutputColumn::RelationshipType {
            relationship_types, ..
        } = &mut missing_token.columns[0]
        else {
            panic!("relationship-type fixture changed output shape")
        };
        relationship_types.pop();
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(missing_token)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        let mut hidden_null_token = relationship_type_output_parts(&request)?;
        let ResidentNullableRelationOutputColumn::RelationshipType {
            relationship_types, ..
        } = &mut hidden_null_token.columns[0]
        else {
            panic!("relationship-type fixture changed output shape")
        };
        relationship_types[1] = RelationshipTypeId(9);
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(hidden_null_token)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        let mut changed_program = request.program.clone();
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            changed_program.stages.last_mut()
        else {
            panic!("relationship-type fixture final projection disappeared")
        };
        bindings[0].source = ResidentNullableRelationOutputSource::Entity {
            slot: ResidentNullableRelationSlot(1),
            kind: ResidentNullableRelationBindingKind::Relationship,
        };
        let changed_fingerprint = super::nullable_relation_fingerprint(
            request.generation,
            request.execution,
            &changed_program,
            &request.predicate_program,
            &request.property_lanes,
            &request.capacities,
            &request.manifest,
        )?;
        assert_ne!(changed_fingerprint, request.manifest.fingerprint);

        let mut wrong_kind = request.program;
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            wrong_kind.stages.last_mut()
        else {
            panic!("relationship-type fixture final projection disappeared")
        };
        bindings[0].source = ResidentNullableRelationOutputSource::RelationshipType {
            slot: ResidentNullableRelationSlot(0),
        };
        let error = wrong_kind
            .validate()
            .expect_err("type() output must not reinterpret a node binding");
        assert_eq!(error.code, ErrorCode::QueryType);
        Ok(())
    }

    #[test]
    fn nullable_relation_typed_output_sources_are_checked_against_final_scope() -> Result<()> {
        let request = typed_output_request()?;
        let mut wrong_kind = request.program.clone();
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            wrong_kind.stages.last_mut()
        else {
            panic!("typed-output fixture final projection disappeared")
        };
        bindings[0].source = ResidentNullableRelationOutputSource::Entity {
            slot: ResidentNullableRelationSlot(0),
            kind: ResidentNullableRelationBindingKind::Relationship,
        };
        let error = wrong_kind
            .validate()
            .expect_err("an output may not reinterpret a node row as a relationship row");
        assert_eq!(error.code, ErrorCode::QueryType);

        let mut absent = request.program;
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            absent.stages.last_mut()
        else {
            panic!("typed-output fixture final projection disappeared")
        };
        bindings[1].source = ResidentNullableRelationOutputSource::IntegerProperty {
            slot: ResidentNullableRelationSlot(7),
            kind: ResidentNullableRelationBindingKind::Node,
            property: PropertyId(21),
        };
        let error = absent
            .validate()
            .expect_err("an output may not gather through a slot outside final scope");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        Ok(())
    }

    #[test]
    fn nullable_relation_property_outputs_propagate_a_null_binding_without_host_lookup()
    -> Result<()> {
        let node = ResidentNullableRelationSlot(0);
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    output: node,
                    labels: ResidentNullableNodeDomain::KnownEmpty,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![
                        ResidentNullableRelationOutputBinding {
                            name: "count".to_owned(),
                            source: ResidentNullableRelationOutputSource::IntegerProperty {
                                slot: node,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(31),
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "name".to_owned(),
                            source: ResidentNullableRelationOutputSource::StringProperty {
                                slot: node,
                                kind: ResidentNullableRelationBindingKind::Node,
                                property: PropertyId(32),
                                maximum_bytes: 8,
                            },
                        },
                        ResidentNullableRelationOutputBinding {
                            name: "absent".to_owned(),
                            source: ResidentNullableRelationOutputSource::NullProperty {
                                slot: node,
                                kind: ResidentNullableRelationBindingKind::Node,
                            },
                        },
                    ],
                },
            ],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 3, 0, 3, 0, 1)?;
        let request = ResidentNullableRelationRequest::build(
            generation(),
            ResidentExecutionId { high: 61, low: 67 },
            program,
            capacities,
            601,
        )?;
        let cardinalities = [(1, 0, 0, 1, 1), (1, 1, 1, 0, 1)];
        let receipts = request
            .obligations()
            .zip(cardinalities)
            .map(
                |(obligation, (input, matched_inputs, candidates, null_extensions, output))| {
                    ResidentNullableRelationReceipt {
                        execution: request.execution,
                        obligation,
                        input_cardinality: input,
                        matched_input_cardinality: matched_inputs,
                        candidate_cardinality: candidates,
                        null_extension_cardinality: null_extensions,
                        output_cardinality: output,
                        completion: ResidentDeviceCompletion::CpuReference,
                    }
                },
            )
            .collect();
        let parts = ResidentNullableRelationResultParts {
            generation: request.generation,
            execution: request.execution,
            fingerprint: request.manifest.fingerprint,
            row_count: 1,
            columns: vec![
                ResidentNullableRelationOutputColumn::IntegerProperty {
                    slot: node,
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(31),
                    source_rows: vec![RESIDENT_NULLABLE_RELATION_NULL_ROW],
                    values: vec![0],
                    validity: vec![0],
                },
                ResidentNullableRelationOutputColumn::StringProperty {
                    slot: node,
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(32),
                    maximum_bytes: 8,
                    source_rows: vec![RESIDENT_NULLABLE_RELATION_NULL_ROW],
                    offsets: vec![0, 0],
                    bytes: Vec::new(),
                    validity: vec![0],
                },
                ResidentNullableRelationOutputColumn::NullProperty {
                    slot: node,
                    kind: ResidentNullableRelationBindingKind::Node,
                    source_rows: vec![RESIDENT_NULLABLE_RELATION_NULL_ROW],
                },
            ],
            receipts,
            scratch_bytes: request.scratch_bytes()?,
        };
        ResidentNullableRelationResult::completed(parts.clone())
            .validate(&request, ResidentDeviceCompletion::CpuReference)?;

        let mut forged = parts;
        let ResidentNullableRelationOutputColumn::StringProperty { validity, .. } =
            &mut forged.columns[1]
        else {
            panic!("null-propagation fixture string output changed shape")
        };
        validity[0] = 1;
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(forged)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn nullable_relation_capacity_consumes_three_independent_bound_node_factors() -> Result<()> {
        let capacities = ResidentNullableRelationCapacities::derive(
            &match3_three_bound_nodes_program(),
            11,
            15,
            11,
            15,
            32,
        )?;

        assert_eq!(
            capacities.stage_input_rows,
            [1, 11, 121, 1_331, 1_815, 2_475, 3_375]
        );
        assert_eq!(
            capacities.stage_candidate_rows,
            [11, 121, 1_331, 1_815, 2_475, 3_375, 3_375]
        );
        assert_eq!(
            capacities.stage_output_rows,
            [11, 121, 1_331, 1_815, 2_475, 3_375, 3_375]
        );
        Ok(())
    }

    #[test]
    fn nullable_relation_capacity_preserves_dropped_scan_multiplicity() -> Result<()> {
        let a = ResidentNullableRelationSlot(0);
        let dropped = ResidentNullableRelationSlot(1);
        let relationship = ResidentNullableRelationSlot(2);
        let target = ResidentNullableRelationSlot(3);
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: a,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: dropped,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::ScopeProject {
                    bindings: vec![ResidentNullableRelationProjectionBinding {
                        variable: "a".to_owned(),
                        source: a,
                        output: a,
                        row_limit: None,
                    }],
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 0,
                    source: a,
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: Some(relationship),
                    different_from: vec![],
                    target: ResidentNullableRelationTarget::Introduce(target),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "target".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: target,
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 11, 15, 11, 15, 32)?;

        assert_eq!(capacities.stage_input_rows, [1, 11, 121, 121, 165]);
        assert_eq!(capacities.stage_candidate_rows, [11, 121, 121, 165, 165]);
        assert_eq!(capacities.stage_output_rows, [11, 121, 121, 165, 165]);
        Ok(())
    }

    #[test]
    fn nullable_relation_fingerprint_rejects_generation_stage_and_capacity_changes() -> Result<()> {
        let request = optional_known_empty_request()?;

        let mut wrong_generation = request.clone();
        wrong_generation.generation.catalog_generation[0] ^= 1;
        assert!(wrong_generation.validate().is_err());

        let mut wrong_stage = request.clone();
        let ResidentNullableRelationStage::NodeScan { mode, .. } =
            &mut wrong_stage.program.stages[0]
        else {
            unreachable!("fixture stage changed")
        };
        *mode = ResidentNullableRelationMatchMode::Mandatory;
        assert!(wrong_stage.validate().is_err());

        let mut wrong_capacity = request;
        wrong_capacity.capacities.stage_output_rows[0] = 2;
        assert!(wrong_capacity.validate().is_err());
        Ok(())
    }

    #[test]
    fn nullable_relation_property_lanes_are_canonical_for_node_and_relationship_tensors()
    -> Result<()> {
        let request = property_lane_request()?;
        assert_eq!(
            request.property_lanes(),
            [
                ResidentNullableRelationPropertyLane {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(11),
                    shape: ResidentNullableRelationPropertyShape::Integer,
                },
                ResidentNullableRelationPropertyLane {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(12),
                    shape: ResidentNullableRelationPropertyShape::String,
                },
                ResidentNullableRelationPropertyLane {
                    kind: ResidentNullableRelationBindingKind::Relationship,
                    property: PropertyId(13),
                    shape: ResidentNullableRelationPropertyShape::Integer,
                },
                ResidentNullableRelationPropertyLane {
                    kind: ResidentNullableRelationBindingKind::Relationship,
                    property: PropertyId(14),
                    shape: ResidentNullableRelationPropertyShape::String,
                },
            ]
        );

        let same_node_tensor_from_another_slot =
            ResidentNullableRelationPredicateValue::IntegerProperty {
                slot: ResidentNullableRelationSlot(2),
                kind: ResidentNullableRelationBindingKind::Node,
                property: PropertyId(11),
            };
        assert_eq!(
            request.property_lane_index(&same_node_tensor_from_another_slot)?,
            Some(0)
        );
        assert_eq!(
            request.property_lane_index(
                &ResidentNullableRelationPredicateValue::StringProperty {
                    slot: ResidentNullableRelationSlot(1),
                    kind: ResidentNullableRelationBindingKind::Relationship,
                    property: PropertyId(14),
                }
            )?,
            Some(3)
        );
        assert_eq!(
            request.property_lane_index(&ResidentNullableRelationPredicateValue::Integer(7))?,
            None
        );
        assert_eq!(
            request.output_property_lane_index(
                &ResidentNullableRelationOutputSource::StringProperty {
                    slot: ResidentNullableRelationSlot(2),
                    kind: ResidentNullableRelationBindingKind::Node,
                    property: PropertyId(12),
                    maximum_bytes: 7,
                }
            )?,
            Some(1)
        );
        assert_eq!(
            request.output_property_lane_index(
                &ResidentNullableRelationOutputSource::NullProperty {
                    slot: ResidentNullableRelationSlot(2),
                    kind: ResidentNullableRelationBindingKind::Node,
                }
            )?,
            None
        );

        let error = request
            .property_lane_index(&ResidentNullableRelationPredicateValue::StringProperty {
                slot: ResidentNullableRelationSlot(0),
                kind: ResidentNullableRelationBindingKind::Node,
                property: PropertyId(11),
            })
            .expect_err("a backend must not reinterpret an integer lane as a string lane");
        assert_eq!(error.code, ErrorCode::CorruptStorage);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable property lane is absent from its sealed registry"
        );
        Ok(())
    }

    #[test]
    fn nullable_relation_property_lanes_reject_missing_duplicate_and_mismatched_entries()
    -> Result<()> {
        let request = property_lane_request()?;

        let mut missing = request.clone();
        missing.property_lanes.lanes.remove(0);
        let error = missing
            .validate()
            .expect_err("a missing predicate or projection dependency must fail admission");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable property lane registry is missing, inventing, or mismatching a predicate or projection dependency"
        );

        let mut duplicate = request.clone();
        duplicate
            .property_lanes
            .lanes
            .insert(1, duplicate.property_lanes.lanes[0]);
        let error = duplicate
            .validate()
            .expect_err("a duplicate property lane must fail admission");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable property lanes must be bounded, sorted, and unique"
        );

        let mut mismatched = request;
        mismatched.property_lanes.lanes[0].shape = ResidentNullableRelationPropertyShape::String;
        let error = mismatched
            .validate()
            .expect_err("a wrong scalar tensor shape must fail admission");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable property lane registry is missing, inventing, or mismatching a predicate or projection dependency"
        );
        Ok(())
    }

    #[test]
    fn nullable_relation_property_lanes_reject_conflicting_physical_shapes() -> Result<()> {
        let request = property_lane_request()?;
        let mut predicates = request.predicate_program.clone();
        predicates
            .filters
            .push(ResidentNullableRelationFilterStage {
                placement: ResidentNullableRelationFilterPlacement::RelationAfter { stage: 1 },
                predicate: ResidentNullableRelationPredicate::CompareString {
                    left: ResidentNullableRelationPredicateValue::StringProperty {
                        slot: ResidentNullableRelationSlot(0),
                        kind: ResidentNullableRelationBindingKind::Node,
                        property: PropertyId(11),
                    },
                    operation: super::CompareOp::Eq,
                    right: ResidentNullableRelationPredicateValue::String("wrong".into()),
                },
            });
        let error = ResidentNullableRelationRequest::build_with_predicates(
            request.generation,
            ResidentExecutionId { high: 43, low: 47 },
            request.program.clone(),
            predicates,
            request.capacities.clone(),
            401,
        )
        .expect_err("one physical property cannot name integer and string tensors");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable property lane is assigned incompatible scalar shapes"
        );

        let mut conflicting_projection = request.program.clone();
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            conflicting_projection.stages.last_mut()
        else {
            panic!("property-lane fixture final projection disappeared")
        };
        bindings.push(ResidentNullableRelationOutputBinding {
            name: "wrong_shape".to_owned(),
            source: ResidentNullableRelationOutputSource::StringProperty {
                slot: ResidentNullableRelationSlot(0),
                kind: ResidentNullableRelationBindingKind::Node,
                property: PropertyId(11),
                maximum_bytes: 4,
            },
        });
        let conflicting_capacities = ResidentNullableRelationCapacities::derive(
            &conflicting_projection,
            request.capacities.node_slot_count as usize,
            request.capacities.relationship_slot_count as usize,
            request.capacities.visible_node_rows as usize,
            request.capacities.visible_relationship_rows as usize,
            request.capacities.max_output_rows as usize,
        )?;
        let error = ResidentNullableRelationRequest::build_with_predicates(
            request.generation,
            ResidentExecutionId { high: 71, low: 73 },
            conflicting_projection,
            request.predicate_program.clone(),
            conflicting_capacities,
            701,
        )
        .expect_err("predicate and projection must not reinterpret one physical property");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable property lane is assigned incompatible scalar shapes"
        );

        let mut inconsistent_width = request.program.clone();
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            inconsistent_width.stages.last_mut()
        else {
            panic!("property-lane fixture final projection disappeared")
        };
        bindings.push(ResidentNullableRelationOutputBinding {
            name: "same_name_wider".to_owned(),
            source: ResidentNullableRelationOutputSource::StringProperty {
                slot: ResidentNullableRelationSlot(2),
                kind: ResidentNullableRelationBindingKind::Node,
                property: PropertyId(12),
                maximum_bytes: 8,
            },
        });
        let inconsistent_capacities = ResidentNullableRelationCapacities::derive(
            &inconsistent_width,
            request.capacities.node_slot_count as usize,
            request.capacities.relationship_slot_count as usize,
            request.capacities.visible_node_rows as usize,
            request.capacities.visible_relationship_rows as usize,
            request.capacities.max_output_rows as usize,
        )?;
        let error = ResidentNullableRelationRequest::build_with_predicates(
            request.generation,
            ResidentExecutionId { high: 79, low: 83 },
            inconsistent_width,
            request.predicate_program.clone(),
            inconsistent_capacities,
            709,
        )
        .expect_err("one physical string property must seal one exact maximum width");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable string property lane has inconsistent exact output widths"
        );
        Ok(())
    }

    #[test]
    fn nullable_relation_property_lane_registry_is_fingerprint_sealed() -> Result<()> {
        let request = property_lane_request()?;
        let mut changed_lanes = request.property_lanes.clone();
        changed_lanes.lanes[0].shape = ResidentNullableRelationPropertyShape::String;
        let changed_fingerprint = super::nullable_relation_fingerprint(
            request.generation,
            request.execution,
            &request.program,
            &request.predicate_program,
            &changed_lanes,
            &request.capacities,
            &request.manifest,
        )?;
        assert_ne!(changed_fingerprint, request.manifest.fingerprint);
        Ok(())
    }

    #[test]
    fn nullable_relation_typed_output_source_and_capacity_are_fingerprint_sealed() -> Result<()> {
        let request = typed_output_request()?;
        let mut changed_program = request.program.clone();
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            changed_program.stages.last_mut()
        else {
            panic!("typed-output fixture final projection disappeared")
        };
        let ResidentNullableRelationOutputSource::StringProperty { maximum_bytes, .. } =
            &mut bindings[2].source
        else {
            panic!("typed-output fixture string source changed shape")
        };
        *maximum_bytes = 3;
        let changed_fingerprint = super::nullable_relation_fingerprint(
            request.generation,
            request.execution,
            &changed_program,
            &request.predicate_program,
            &request.property_lanes,
            &request.capacities,
            &request.manifest,
        )?;
        assert_ne!(changed_fingerprint, request.manifest.fingerprint);

        let mut forged_capacity = request;
        forged_capacity.capacities.final_output_bytes_per_row += 1;
        assert!(forged_capacity.validate().is_err());
        Ok(())
    }

    #[test]
    fn different_relationships_accepts_exact_sorted_exclusions() -> Result<()> {
        let request = different_relationships_request()?;
        request.validate()?;
        assert_eq!(
            request
                .program
                .stages
                .iter()
                .filter_map(|stage| match stage {
                    ResidentNullableRelationStage::Expand { different_from, .. } => {
                        Some(different_from.clone())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![
                vec![],
                vec![ResidentNullableRelationSlot(1)],
                vec![
                    ResidentNullableRelationSlot(1),
                    ResidentNullableRelationSlot(3)
                ],
            ]
        );
        Ok(())
    }

    #[test]
    fn different_relationships_rejects_omitted_exclusions() {
        let mut program = different_relationships_program();
        expansion_uniqueness_mut(&mut program, 2).1.clear();
        assert_different_relationships_rejected(
            &program,
            "resident nullable expansion omits or forges its exact DifferentRelationships exclusions",
        );
    }

    #[test]
    fn different_relationships_rejects_duplicate_and_unsorted_exclusions() {
        let mut duplicate = different_relationships_program();
        *expansion_uniqueness_mut(&mut duplicate, 3).1 = vec![
            ResidentNullableRelationSlot(1),
            ResidentNullableRelationSlot(1),
        ];
        assert_different_relationships_rejected(
            &duplicate,
            "resident nullable expansion omits or forges its exact DifferentRelationships exclusions",
        );

        let mut unsorted = different_relationships_program();
        expansion_uniqueness_mut(&mut unsorted, 3).1.reverse();
        assert_different_relationships_rejected(
            &unsorted,
            "resident nullable expansion omits or forges its exact DifferentRelationships exclusions",
        );
    }

    #[test]
    fn different_relationships_rejects_non_live_and_node_exclusion_slots() {
        let mut non_live = different_relationships_program();
        non_live.stages.insert(
            2,
            ResidentNullableRelationStage::ScopeProject {
                bindings: vec![ResidentNullableRelationProjectionBinding {
                    variable: "middle".to_owned(),
                    source: ResidentNullableRelationSlot(2),
                    output: ResidentNullableRelationSlot(2),
                    row_limit: None,
                }],
            },
        );
        assert_different_relationships_rejected(
            &non_live,
            "resident nullable DifferentRelationships exclusion is not a live relationship slot",
        );

        let mut node_kind = different_relationships_program();
        *expansion_uniqueness_mut(&mut node_kind, 2).1 = vec![ResidentNullableRelationSlot(0)];
        assert_different_relationships_rejected(
            &node_kind,
            "resident nullable expansion omits or forges its exact DifferentRelationships exclusions",
        );
    }

    #[test]
    fn different_relationships_rejects_uniqueness_group_regression_and_reopen() {
        let mut regression = different_relationships_program();
        *expansion_uniqueness_mut(&mut regression, 1).0 = 2;
        assert_different_relationships_rejected(
            &regression,
            "resident nullable relationship uniqueness group reopens out of order",
        );

        let mut reopen = different_relationships_program();
        let (second_group, second_exclusions) = expansion_uniqueness_mut(&mut reopen, 2);
        *second_group = 1;
        second_exclusions.clear();
        *expansion_uniqueness_mut(&mut reopen, 3).0 = 0;
        assert_different_relationships_rejected(
            &reopen,
            "resident nullable relationship uniqueness group reopens out of order",
        );
    }

    #[test]
    fn different_relationships_fingerprint_seals_groups_and_exclusions() -> Result<()> {
        let request = different_relationships_request()?;

        let mut changed_group_program = request.program.clone();
        for stage in 1..=3 {
            *expansion_uniqueness_mut(&mut changed_group_program, stage).0 = 7;
        }
        changed_group_program.validate()?;
        let changed_group_fingerprint = super::nullable_relation_fingerprint(
            request.generation,
            request.execution,
            &changed_group_program,
            &request.predicate_program,
            &request.property_lanes,
            &request.capacities,
            &request.manifest,
        )?;
        assert_ne!(changed_group_fingerprint, request.manifest.fingerprint);

        let mut changed_exclusions_program = request.program.clone();
        expansion_uniqueness_mut(&mut changed_exclusions_program, 3)
            .1
            .reverse();
        let changed_exclusions_fingerprint = super::nullable_relation_fingerprint(
            request.generation,
            request.execution,
            &changed_exclusions_program,
            &request.predicate_program,
            &request.property_lanes,
            &request.capacities,
            &request.manifest,
        )?;
        assert_ne!(changed_exclusions_fingerprint, request.manifest.fingerprint);

        let mut changed_source_labels_program = request.program.clone();
        let ResidentNullableRelationStage::Expand { source_labels, .. } =
            &mut changed_source_labels_program.stages[1]
        else {
            panic!("DifferentRelationships source-label fixture stage is not an expansion")
        };
        *source_labels = ResidentNullableNodeDomain::KnownEmpty;
        changed_source_labels_program.validate()?;
        let changed_source_labels_fingerprint = super::nullable_relation_fingerprint(
            request.generation,
            request.execution,
            &changed_source_labels_program,
            &request.predicate_program,
            &request.property_lanes,
            &request.capacities,
            &request.manifest,
        )?;
        assert_ne!(
            changed_source_labels_fingerprint,
            request.manifest.fingerprint
        );

        let mut post_seal_group_mutation = request.clone();
        post_seal_group_mutation.program = changed_group_program;
        let error = post_seal_group_mutation
            .validate()
            .expect_err("post-seal uniqueness-group mutation must be rejected");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable relation fingerprint does not match its immutable request"
        );

        let mut forged_fingerprint = request;
        forged_fingerprint.manifest.fingerprint.0[0] ^= 1;
        let error = forged_fingerprint
            .validate()
            .expect_err("forged DifferentRelationships fingerprint must be rejected");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(
            error.message.as_ref(),
            "resident nullable relation fingerprint does not match its immutable request"
        );
        Ok(())
    }

    #[test]
    fn known_empty_optional_result_requires_null_and_exact_receipts() -> Result<()> {
        let request = optional_known_empty_request()?;
        ResidentNullableRelationResult::completed(valid_result_parts(&request)?)
            .validate(&request, ResidentDeviceCompletion::CpuReference)?;

        let mut non_null = valid_result_parts(&request)?;
        let ResidentNullableRelationOutputColumn::Entity { rows, .. } = &mut non_null.columns[0]
        else {
            panic!("known-empty fixture must project one entity column")
        };
        rows[0] = 0;
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(non_null)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        let mut reordered = valid_result_parts(&request)?;
        reordered.receipts.swap(0, 1);
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(reordered)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        // The old max(input,candidates)..=input+candidates inequality accepted 1 here. The
        // backend-authored exact null-extension count proves that 0 + 0 cannot produce 1 row.
        let mut impossible_exact_count = valid_result_parts(&request)?;
        impossible_exact_count.receipts[0].null_extension_cardinality = 0;
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(impossible_exact_count)
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err()
        );

        let wrong_backend = valid_result_parts(&request)?;
        assert!(
            ResidentNullableRelationResult::from_untrusted_parts(wrong_backend)
                .validate(&request, ResidentDeviceCompletion::Metal)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn optional_node_scan_receipt_proves_exact_cartesian_or_all_null_extension() -> Result<()> {
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: ResidentNullableRelationSlot(0),
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    output: ResidentNullableRelationSlot(1),
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: ResidentNullableRelationSlot(2),
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "c".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: ResidentNullableRelationSlot(2),
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 2, 0, 2, 0, 16)?;
        let request = ResidentNullableRelationRequest::build(
            generation(),
            ResidentExecutionId { high: 31, low: 37 },
            program,
            capacities,
            31,
        )?;
        let receipt = |stage: usize, input, matched_inputs, candidates, null_extensions, output| {
            ResidentNullableRelationReceipt {
                execution: request.execution,
                obligation: request.manifest.obligations[stage],
                input_cardinality: input,
                matched_input_cardinality: matched_inputs,
                candidate_cardinality: candidates,
                null_extension_cardinality: null_extensions,
                output_cardinality: output,
                completion: ResidentDeviceCompletion::CpuReference,
            }
        };
        let valid = [
            receipt(0, 1, 1, 2, 0, 2),
            receipt(1, 2, 2, 4, 0, 4),
            receipt(2, 4, 4, 8, 0, 8),
            receipt(3, 8, 8, 8, 0, 8),
        ];
        super::validate_nullable_relation_receipts(
            &request,
            &valid,
            ResidentDeviceCompletion::CpuReference,
        )?;

        let mut matched_exceeds_input = valid;
        matched_exceeds_input[1].matched_input_cardinality = 3;
        assert!(
            super::validate_nullable_relation_receipts(
                &request,
                &matched_exceeds_input,
                ResidentDeviceCompletion::CpuReference,
            )
            .is_err()
        );

        let mut candidates_without_a_matched_input = valid;
        candidates_without_a_matched_input[1].matched_input_cardinality = 0;
        assert!(
            super::validate_nullable_relation_receipts(
                &request,
                &candidates_without_a_matched_input,
                ResidentDeviceCompletion::CpuReference,
            )
            .is_err()
        );

        let mut fewer_candidates_than_matched_inputs = valid;
        fewer_candidates_than_matched_inputs[1].candidate_cardinality = 1;
        assert!(
            super::validate_nullable_relation_receipts(
                &request,
                &fewer_candidates_than_matched_inputs,
                ResidentDeviceCompletion::CpuReference,
            )
            .is_err()
        );

        let mut partial_mandatory_scan_match = valid;
        partial_mandatory_scan_match[2].matched_input_cardinality = 3;
        assert!(
            super::validate_nullable_relation_receipts(
                &request,
                &partial_mandatory_scan_match,
                ResidentDeviceCompletion::CpuReference,
            )
            .is_err()
        );

        let mut partial_projection_match = valid;
        partial_projection_match[3].matched_input_cardinality = 7;
        assert!(
            super::validate_nullable_relation_receipts(
                &request,
                &partial_projection_match,
                ResidentDeviceCompletion::CpuReference,
            )
            .is_err()
        );

        let mut non_divisible = valid;
        non_divisible[1] = receipt(1, 2, 2, 3, 0, 3);
        assert!(
            super::validate_nullable_relation_receipts(
                &request,
                &non_divisible,
                ResidentDeviceCompletion::CpuReference,
            )
            .is_err()
        );

        // This also satisfied the old inequality: max(2,2) <= 3 <= 2+2. It is impossible once
        // the backend says exactly zero input rows were null-extended.
        let mut impossible_equation = valid;
        impossible_equation[1] = receipt(1, 2, 2, 2, 0, 3);
        assert!(
            super::validate_nullable_relation_receipts(
                &request,
                &impossible_equation,
                ResidentDeviceCompletion::CpuReference,
            )
            .is_err()
        );

        // Mandatory scan previously proved only output=candidates. Seven candidates cannot be a
        // Cartesian scan of one domain repeated for four input rows.
        let mut impossible_mandatory_cartesian = valid;
        impossible_mandatory_cartesian[2] = receipt(2, 4, 4, 7, 0, 7);
        impossible_mandatory_cartesian[3] = receipt(3, 7, 7, 7, 0, 7);
        assert!(
            super::validate_nullable_relation_receipts(
                &request,
                &impossible_mandatory_cartesian,
                ResidentDeviceCompletion::CpuReference,
            )
            .is_err()
        );

        let all_unmatched = [
            receipt(0, 1, 1, 2, 0, 2),
            receipt(1, 2, 0, 0, 2, 2),
            receipt(2, 2, 2, 4, 0, 4),
            receipt(3, 4, 4, 4, 0, 4),
        ];
        super::validate_nullable_relation_receipts(
            &request,
            &all_unmatched,
            ResidentDeviceCompletion::CpuReference,
        )?;
        Ok(())
    }

    #[test]
    fn mandatory_expansion_from_nullable_source_has_an_explicit_zero_row_drop_boundary()
    -> Result<()> {
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    output: ResidentNullableRelationSlot(7),
                    labels: ResidentNullableNodeDomain::KnownEmpty,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 0,
                    source: ResidentNullableRelationSlot(7),
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: None,
                    different_from: Vec::new(),
                    target: ResidentNullableRelationTarget::Introduce(
                        ResidentNullableRelationSlot(19),
                    ),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "target".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: ResidentNullableRelationSlot(19),
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 3, 2, 3, 2, 8)?;
        let request = ResidentNullableRelationRequest::build(
            generation(),
            ResidentExecutionId { high: 7, low: 8 },
            program,
            capacities,
            21,
        )?;
        let cardinalities = [(1, 0, 0, 1, 1), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)];
        let receipts = request
            .obligations()
            .zip(cardinalities)
            .map(
                |(obligation, (input, matched_inputs, candidates, null_extensions, output))| {
                    ResidentNullableRelationReceipt {
                        execution: request.execution,
                        obligation,
                        input_cardinality: input,
                        matched_input_cardinality: matched_inputs,
                        candidate_cardinality: candidates,
                        null_extension_cardinality: null_extensions,
                        output_cardinality: output,
                        completion: ResidentDeviceCompletion::CpuReference,
                    }
                },
            )
            .collect();
        ResidentNullableRelationResult::from_untrusted_parts(ResidentNullableRelationResultParts {
            generation: request.generation,
            execution: request.execution,
            fingerprint: request.manifest.fingerprint,
            row_count: 0,
            columns: vec![ResidentNullableRelationOutputColumn::Entity {
                slot: ResidentNullableRelationSlot(19),
                kind: ResidentNullableRelationBindingKind::Node,
                rows: Vec::new(),
            }],
            receipts,
            scratch_bytes: request.scratch_bytes()?,
        })
        .validate(&request, ResidentDeviceCompletion::CpuReference)?;

        let forged_cardinalities = [(1, 0, 0, 1, 1), (1, 1, 1, 0, 1), (1, 1, 1, 0, 1)];
        let forged_receipts = request
            .obligations()
            .zip(forged_cardinalities)
            .map(
                |(obligation, (input, matched_inputs, candidates, null_extensions, output))| {
                    ResidentNullableRelationReceipt {
                        execution: request.execution,
                        obligation,
                        input_cardinality: input,
                        matched_input_cardinality: matched_inputs,
                        candidate_cardinality: candidates,
                        null_extension_cardinality: null_extensions,
                        output_cardinality: output,
                        completion: ResidentDeviceCompletion::CpuReference,
                    }
                },
            )
            .collect();
        let forged = ResidentNullableRelationResult::from_untrusted_parts(
            ResidentNullableRelationResultParts {
                generation: request.generation,
                execution: request.execution,
                fingerprint: request.manifest.fingerprint,
                row_count: 1,
                columns: vec![ResidentNullableRelationOutputColumn::Entity {
                    slot: ResidentNullableRelationSlot(19),
                    kind: ResidentNullableRelationBindingKind::Node,
                    rows: vec![0],
                }],
                receipts: forged_receipts,
                scratch_bytes: request.scratch_bytes()?,
            },
        );
        assert!(
            forged
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err(),
            "an always-null source must not certify a mandatory expansion candidate"
        );
        Ok(())
    }

    #[test]
    fn mandatory_expansion_to_an_always_null_bound_target_rejects_candidates() -> Result<()> {
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: ResidentNullableRelationSlot(7),
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    output: ResidentNullableRelationSlot(11),
                    labels: ResidentNullableNodeDomain::KnownEmpty,
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    uniqueness_group: 0,
                    source: ResidentNullableRelationSlot(7),
                    source_labels: ResidentNullableNodeDomain::Any,
                    relationship: None,
                    different_from: Vec::new(),
                    target: ResidentNullableRelationTarget::Existing(ResidentNullableRelationSlot(
                        11,
                    )),
                    direction: super::ResidentDirection::Outgoing,
                    relationship_types: ResidentNullableRelationshipDomain::Any,
                    target_labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "target".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: ResidentNullableRelationSlot(11),
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        };
        let capacities = ResidentNullableRelationCapacities::derive(&program, 2, 1, 2, 1, 8)?;
        let request = ResidentNullableRelationRequest::build(
            generation(),
            ResidentExecutionId { high: 9, low: 10 },
            program,
            capacities,
            25,
        )?;
        let cardinalities = [
            (1, 1, 1, 0, 1),
            (1, 0, 0, 1, 1),
            (1, 1, 1, 0, 1),
            (1, 1, 1, 0, 1),
        ];
        let receipts = request
            .obligations()
            .zip(cardinalities)
            .map(
                |(obligation, (input, matched_inputs, candidates, null_extensions, output))| {
                    ResidentNullableRelationReceipt {
                        execution: request.execution,
                        obligation,
                        input_cardinality: input,
                        matched_input_cardinality: matched_inputs,
                        candidate_cardinality: candidates,
                        null_extension_cardinality: null_extensions,
                        output_cardinality: output,
                        completion: ResidentDeviceCompletion::CpuReference,
                    }
                },
            )
            .collect();
        let forged = ResidentNullableRelationResult::from_untrusted_parts(
            ResidentNullableRelationResultParts {
                generation: request.generation,
                execution: request.execution,
                fingerprint: request.manifest.fingerprint,
                row_count: 1,
                columns: vec![ResidentNullableRelationOutputColumn::Entity {
                    slot: ResidentNullableRelationSlot(11),
                    kind: ResidentNullableRelationBindingKind::Node,
                    rows: vec![0],
                }],
                receipts,
                scratch_bytes: request.scratch_bytes()?,
            },
        );
        assert!(
            forged
                .validate(&request, ResidentDeviceCompletion::CpuReference)
                .is_err(),
            "an always-null bound target must not certify a mandatory expansion candidate"
        );
        Ok(())
    }

    #[test]
    fn empty_known_vectors_are_not_a_known_empty_catalog_domain() {
        let invalid = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    output: ResidentNullableRelationSlot(0),
                    labels: ResidentNullableNodeDomain::Known(Vec::new()),
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "n".to_owned(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot: ResidentNullableRelationSlot(0),
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        };
        assert!(invalid.validate().is_err());
    }
}

#[cfg(test)]
mod cpu_temporal_tests {
    use tokio_util::sync::CancellationToken;

    use crate::{
        Bookmark, ErrorCode, ProjectId, Result,
        backend::{BackendKind, ResidentExecutionId, ResidentNodePipelineRequest},
        graph::LayerMask,
    };

    use super::{
        ResidentNodePipelineResult, ResidentRowColumn, ResidentRowInstruction,
        ResidentRowOperation, ResidentRowProgram, ResidentRowProgramManifest,
        ResidentRowProgramRequest, ResidentRowSortKey, ResidentRowValueType,
        ResidentTemporalAccessor, compare_sort_keys, execute_reference_row_program,
    };

    fn sorted_positions(
        column: ResidentRowColumn,
        descending: bool,
        nulls_first: bool,
    ) -> Vec<u64> {
        let rows = column.len();
        let registers = [column];
        let keys = [ResidentRowSortKey {
            register: 0,
            descending,
            nulls_first,
        }];
        let mut positions = (0..rows as u64).collect::<Vec<_>>();
        positions.sort_unstable_by(|left, right| {
            compare_sort_keys(&registers, &keys, *left, *right).then_with(|| left.cmp(right))
        });
        positions
    }

    #[test]
    fn all_temporal_columns_use_the_canonical_order_tuples() {
        assert_eq!(
            sorted_positions(
                ResidentRowColumn::Date {
                    days: vec![2, -1, 2, 0],
                    validity: vec![1, 1, 0, 1],
                },
                false,
                false,
            ),
            [1, 3, 0, 2]
        );
        assert_eq!(
            sorted_positions(
                ResidentRowColumn::LocalTime {
                    nanos: vec![9, 1, 5, 0],
                    validity: vec![1, 1, 1, 0],
                },
                true,
                false,
            ),
            [0, 2, 1, 3]
        );

        let hour = 3_600_i64 * 1_000_000_000;
        let zoned_time = ResidentRowColumn::ZonedTime {
            // UTC tuples are (10h,+02), (11h,+00), (10h,+01), then NULL.
            nanos: vec![12 * hour, 11 * hour, 11 * hour, 0],
            offset_seconds: vec![7_200, 0, 3_600, 0],
            validity: vec![1, 1, 1, 0],
        };
        assert_eq!(
            sorted_positions(zoned_time.clone(), false, false),
            [2, 0, 1, 3]
        );
        assert_eq!(sorted_positions(zoned_time, true, false), [1, 0, 2, 3]);

        assert_eq!(
            sorted_positions(
                ResidentRowColumn::LocalDateTime {
                    seconds: vec![10, 9, 10, 0],
                    nanos: vec![1, 999_999_999, 0, 0],
                    validity: vec![1, 1, 1, 0],
                },
                false,
                false,
            ),
            [1, 2, 0, 3]
        );

        let timezone_bytes = b"Europe/ParisEurope/LondonUTCUTC".to_vec();
        let zoned_datetime = ResidentRowColumn::ZonedDateTime {
            seconds: vec![10, 10, 9, 10, 0],
            nanos: vec![0, 0, 999_999_999, 1, 0],
            timezone_offsets: vec![0, 12, 25, 28, 31, 31],
            timezone_bytes,
            validity: vec![1, 1, 1, 1, 0],
        };
        assert_eq!(
            sorted_positions(zoned_datetime.clone(), false, false),
            [2, 1, 0, 3, 4]
        );
        assert_eq!(
            sorted_positions(zoned_datetime, true, true),
            [4, 3, 0, 1, 2]
        );
    }

    #[test]
    fn zoned_datetime_selection_rebuilds_canonical_timezone_ranges() -> Result<()> {
        let column = ResidentRowColumn::ZonedDateTime {
            seconds: vec![1, 2, 3],
            nanos: vec![4, 5, 6],
            timezone_offsets: vec![0, 3, 3, 5],
            timezone_bytes: b"UTC\xc3\x89".to_vec(),
            validity: vec![1, 0, 1],
        };
        column.validate(ResidentRowValueType::ZonedDateTime, 3)?;
        assert_eq!(
            column.select(&[2, 1, 0])?,
            ResidentRowColumn::ZonedDateTime {
                seconds: vec![3, 2, 1],
                nanos: vec![6, 5, 4],
                timezone_offsets: vec![0, 2, 2, 5],
                timezone_bytes: b"\xc3\x89UTC".to_vec(),
                validity: vec![1, 0, 1],
            }
        );

        let invalid_null_payload = ResidentRowColumn::ZonedDateTime {
            seconds: vec![0],
            nanos: vec![0],
            timezone_offsets: vec![0, 3],
            timezone_bytes: b"UTC".to_vec(),
            validity: vec![0],
        };
        assert!(
            invalid_null_payload
                .validate(ResidentRowValueType::ZonedDateTime, 1)
                .is_err()
        );
        let invalid_utf8 = ResidentRowColumn::ZonedDateTime {
            seconds: vec![0],
            nanos: vec![0],
            timezone_offsets: vec![0, 1],
            timezone_bytes: vec![0xff],
            validity: vec![1],
        };
        assert!(
            invalid_utf8
                .validate(ResidentRowValueType::ZonedDateTime, 1)
                .is_err()
        );
        Ok(())
    }

    fn scalar_request(column: ResidentRowColumn) -> Result<ResidentRowProgramRequest> {
        let rows = column.len();
        let value_type = column.value_type();
        let program = ResidentRowProgram {
            instructions: vec![ResidentRowInstruction {
                output_type: value_type,
                operation: ResidentRowOperation::InputColumn(column),
            }],
        };
        let sort_keys = vec![ResidentRowSortKey {
            register: 0,
            descending: false,
            nulls_first: false,
        }];
        let final_registers = vec![0];
        let manifest = ResidentRowProgramManifest::build(
            &program,
            &sort_keys,
            0,
            usize::MAX,
            rows,
            &final_registers,
            1,
        )?;
        Ok(ResidentRowProgramRequest {
            project: ProjectId(uuid::Uuid::nil()),
            expected_bookmark: Bookmark { term: 1, index: 1 },
            expected_graph_revision: 1,
            expected_layout_version: 1,
            execution: ResidentExecutionId { high: 1, low: 1 },
            input: ResidentNodePipelineRequest {
                project: ProjectId(uuid::Uuid::nil()),
                labels: Vec::new(),
                layers: LayerMask::OBSERVED,
                initial_optional: false,
                expansion: None,
                continuations: Vec::new(),
                correlated_optional: None,
                relationship_null_filter: None,
                predicates: Vec::new(),
                property_filters: Vec::new(),
                value_matrix: None,
                mutation: None,
                orders: Vec::new(),
                offset: 0,
                limit: usize::MAX,
                integer_projections: Vec::new(),
                property_null_projections: Vec::new(),
                max_output_rows: rows,
            },
            program,
            manifest,
            sort_keys,
            offset: 0,
            limit: usize::MAX,
            max_output_rows: rows,
            final_registers,
        })
    }

    fn scalar_duration_request(
        column: ResidentRowColumn,
        months: i64,
        days: i64,
        seconds: i64,
        nanos: i32,
    ) -> Result<ResidentRowProgramRequest> {
        let mut request = scalar_request(column)?;
        let output_type = request.program.instructions[0].output_type;
        request.program.instructions.push(ResidentRowInstruction {
            output_type,
            operation: ResidentRowOperation::TemporalAddDuration {
                temporal: 0,
                months,
                days,
                seconds,
                nanos,
            },
        });
        request.sort_keys[0].register = 1;
        request.final_registers[0] = 1;
        request.manifest = ResidentRowProgramManifest::build(
            &request.program,
            &request.sort_keys,
            request.offset,
            request.limit,
            request.max_output_rows,
            &request.final_registers,
            1,
        )?;
        Ok(request)
    }

    #[test]
    fn typed_row_admission_uses_checked_bytes_not_a_fixed_row_count() -> Result<()> {
        let mut request = scalar_request(ResidentRowColumn::Integer {
            values: vec![0],
            validity: vec![1],
        })?;
        request.program.instructions[0].operation = ResidentRowOperation::IntegerConstant(0);
        let above_former_count = (1_usize << 20) + 1;
        request.input.max_output_rows = above_former_count;
        request.max_output_rows = above_former_count;
        request.manifest = ResidentRowProgramManifest::build(
            &request.program,
            &request.sort_keys,
            request.offset,
            request.limit,
            request.max_output_rows,
            &request.final_registers,
            1,
        )?;
        request.validate()?;
        assert_eq!(
            request.output_cardinality(above_former_count)?,
            above_former_count
        );
        assert!(request.scratch_bytes(above_former_count)? > 0);

        request.input.max_output_rows = usize::MAX;
        request.max_output_rows = usize::MAX;
        request.manifest = ResidentRowProgramManifest::build(
            &request.program,
            &request.sort_keys,
            request.offset,
            request.limit,
            request.max_output_rows,
            &request.final_registers,
            1,
        )?;
        let error = request
            .scratch_bytes(usize::MAX)
            .expect_err("unaddressable row bytes must fail admission");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
        Ok(())
    }

    #[test]
    fn temporal_duration_validation_requires_prior_matching_temporal_type() -> Result<()> {
        let columns = [
            ResidentRowColumn::Date {
                days: vec![0],
                validity: vec![1],
            },
            ResidentRowColumn::LocalTime {
                nanos: vec![0],
                validity: vec![1],
            },
            ResidentRowColumn::ZonedTime {
                nanos: vec![0],
                offset_seconds: vec![3_600],
                validity: vec![1],
            },
            ResidentRowColumn::LocalDateTime {
                seconds: vec![0],
                nanos: vec![0],
                validity: vec![1],
            },
            ResidentRowColumn::ZonedDateTime {
                seconds: vec![0],
                nanos: vec![0],
                timezone_offsets: vec![0, 3],
                timezone_bytes: b"UTC".to_vec(),
                validity: vec![1],
            },
        ];
        for column in columns {
            scalar_duration_request(column, 1, 2, 3, 4)?.validate()?;
        }

        let program = |column: ResidentRowColumn,
                       output_type: ResidentRowValueType,
                       temporal: u16,
                       nanos: i32| ResidentRowProgram {
            instructions: vec![
                ResidentRowInstruction {
                    output_type: column.value_type(),
                    operation: ResidentRowOperation::InputColumn(column),
                },
                ResidentRowInstruction {
                    output_type,
                    operation: ResidentRowOperation::TemporalAddDuration {
                        temporal,
                        months: 0,
                        days: 0,
                        seconds: 0,
                        nanos,
                    },
                },
            ],
        };
        let integer = ResidentRowColumn::Integer {
            values: vec![1],
            validity: vec![1],
        };
        let error = program(integer, ResidentRowValueType::Integer, 0, 0)
            .validate()
            .expect_err("non-temporal duration operand must fail");
        assert_eq!(error.code, ErrorCode::QueryType);

        let date = || ResidentRowColumn::Date {
            days: vec![0],
            validity: vec![1],
        };
        let error = program(date(), ResidentRowValueType::LocalTime, 0, 0)
            .validate()
            .expect_err("duration result must preserve its temporal type");
        assert_eq!(error.code, ErrorCode::QueryType);
        let error = program(date(), ResidentRowValueType::Date, 1, 0)
            .validate()
            .expect_err("duration operand must be a prior register");
        assert_eq!(error.code, ErrorCode::QueryType);
        for nanos in [-1, 1_000_000_000] {
            let error = program(date(), ResidentRowValueType::Date, 0, nanos)
                .validate()
                .expect_err("duration nanoseconds must be normalized");
            assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        }
        Ok(())
    }

    #[test]
    fn temporal_duration_fingerprint_and_scratch_bind_exact_components() -> Result<()> {
        let date = || ResidentRowColumn::Date {
            days: vec![0, 0],
            validity: vec![1, 0],
        };
        let fingerprints = [
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]
        .into_iter()
        .map(|(months, days, seconds, nanos)| {
            scalar_duration_request(date(), months, days, seconds, nanos)
                .map(|request| request.manifest.fingerprint)
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
        assert_eq!(fingerprints.len(), 5);

        let request = scalar_duration_request(
            ResidentRowColumn::ZonedDateTime {
                seconds: vec![0, 0],
                nanos: vec![0, 0],
                timezone_offsets: vec![0, 13, 13],
                timezone_bytes: b"Europe/London".to_vec(),
                validity: vec![1, 0],
            },
            1,
            2,
            3,
            4,
        )?;
        assert_eq!(request.program.timezone_register_capacity(1)?, Some(13));
        let rows = 2;
        let register_width =
            ResidentRowValueType::ZonedDateTime.payload_bytes() + std::mem::size_of::<u8>();
        let expected = rows * register_width * 2
            + rows * 13 * 2
            + rows * std::mem::size_of::<u64>()
            + rows * register_width
            + rows * 13
            + request.obligations().count()
                * std::mem::size_of::<crate::ResidentExecutionReceipt>();
        assert_eq!(request.scratch_bytes(rows)?, expected);
        Ok(())
    }

    fn unexpected_property_loader(
        _: crate::ResidentEntityBinding,
        _: crate::types::PropertyId,
        _: ResidentRowValueType,
        _: Option<u32>,
        _: Option<ResidentTemporalAccessor>,
        _: &[u32],
        _: &CancellationToken,
    ) -> Result<ResidentRowColumn> {
        Err(crate::Error::internal(
            "typed scalar reference test unexpectedly requested a graph property",
        ))
    }

    #[test]
    fn reference_program_sorts_and_publishes_zoned_datetime_without_order_keys() -> Result<()> {
        let request = scalar_request(ResidentRowColumn::ZonedDateTime {
            seconds: vec![10, 9, 10, 0],
            nanos: vec![0, 999_999_999, 0, 0],
            timezone_offsets: vec![0, 12, 15, 28, 28],
            timezone_bytes: b"Europe/ParisUTCEurope/London".to_vec(),
            validity: vec![1, 1, 1, 0],
        })?;
        let scratch = request.scratch_bytes(4)?;
        let mut loader = unexpected_property_loader;
        let raw = execute_reference_row_program(
            &request,
            ResidentNodePipelineResult {
                start_rows: Vec::new(),
                intermediate_node_rows: Vec::new(),
                intermediate_edge_rows: Vec::new(),
                edge_rows: Vec::new(),
                end_rows: Vec::new(),
                integer_columns: Vec::new(),
                boolean_columns: Vec::new(),
                value_left_indices: Vec::new(),
                value_right_indices: Vec::new(),
                mutation: None,
            },
            scratch,
            &mut loader,
            &CancellationToken::new(),
        )?;
        let mut corrupted_parts = raw.clone().into_untrusted_parts();
        corrupted_parts.projected_columns[0].column = ResidentRowColumn::ZonedDateTime {
            seconds: vec![0; 4],
            nanos: vec![0; 4],
            timezone_offsets: vec![0, 14, 14, 14, 14],
            timezone_bytes: vec![b'x'; 14],
            validity: vec![1, 1, 1, 0],
        };
        let error = super::ResidentRowProgramResult::from_untrusted_parts(corrupted_parts)
            .validate(&request, BackendKind::Cpu)
            .expect_err("oversized timezone result must fail closed");
        assert_eq!(error.code, ErrorCode::CorruptStorage);

        let result = raw.validate(&request, BackendKind::Cpu)?;
        assert_eq!(result.source_positions(), [1, 2, 0, 3]);
        assert_eq!(
            result.projected_columns()[0].column,
            ResidentRowColumn::ZonedDateTime {
                seconds: vec![9, 10, 10, 0],
                nanos: vec![999_999_999, 0, 0, 0],
                timezone_offsets: vec![0, 3, 16, 28, 28],
                timezone_bytes: b"UTCEurope/LondonEurope/Paris".to_vec(),
                validity: vec![1, 1, 1, 0],
            }
        );
        Ok(())
    }

    #[test]
    fn integer_lists_use_lexicographic_stable_order_and_canonical_selection() -> Result<()> {
        let column = ResidentRowColumn::List {
            // [2], [1, 9], [1], NULL
            offsets: vec![0, 1, 3, 4, 4],
            values: vec![2, 1, 9, 1],
            element_validity: vec![1, 1, 1, 1],
            validity: vec![1, 1, 1, 0],
        };
        column.validate(ResidentRowValueType::List, 4)?;
        assert_eq!(sorted_positions(column.clone(), false, false), [2, 1, 0, 3]);
        assert_eq!(
            column.select(&[2, 1, 0, 3])?,
            ResidentRowColumn::List {
                offsets: vec![0, 1, 3, 4, 4],
                values: vec![1, 1, 9, 2],
                element_validity: vec![1, 1, 1, 1],
                validity: vec![1, 1, 1, 0],
            }
        );

        for malformed in [
            ResidentRowColumn::List {
                offsets: vec![0, 2],
                values: vec![1],
                element_validity: vec![1],
                validity: vec![1],
            },
            ResidentRowColumn::List {
                offsets: vec![0, 1],
                values: vec![9],
                element_validity: vec![0],
                validity: vec![1],
            },
            ResidentRowColumn::List {
                offsets: vec![0, 1],
                values: vec![0],
                element_validity: vec![1],
                validity: vec![0],
            },
        ] {
            assert!(malformed.validate(ResidentRowValueType::List, 1).is_err());
        }
        Ok(())
    }

    #[test]
    fn list_index_normalizes_negative_indices_and_nulls_invalid_accesses() -> Result<()> {
        let rows = 6;
        let program = ResidentRowProgram {
            instructions: vec![
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::List,
                    operation: ResidentRowOperation::InputColumn(ResidentRowColumn::List {
                        // [10,20] x4, [10,NULL,30], NULL
                        offsets: vec![0, 2, 4, 6, 8, 11, 11],
                        values: vec![10, 20, 10, 20, 10, 20, 10, 20, 10, 0, 30],
                        element_validity: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                        validity: vec![1, 1, 1, 1, 1, 0],
                    }),
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Integer,
                    operation: ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                        values: vec![-1, -2, -3, 2, 1, 0],
                        validity: vec![1; rows],
                    }),
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Integer,
                    operation: ResidentRowOperation::ListIndex { list: 0, index: 1 },
                },
            ],
        };
        let final_registers = vec![2];
        let manifest = ResidentRowProgramManifest::build(
            &program,
            &[],
            0,
            usize::MAX,
            rows,
            &final_registers,
            1,
        )?;
        let mut request = scalar_request(ResidentRowColumn::Integer {
            values: vec![0; rows],
            validity: vec![1; rows],
        })?;
        request.program = program;
        request.manifest = manifest;
        request.sort_keys.clear();
        request.final_registers = final_registers;
        request.validate()?;

        let scratch = request.scratch_bytes(rows)?;
        let mut loader = unexpected_property_loader;
        let result = execute_reference_row_program(
            &request,
            ResidentNodePipelineResult {
                start_rows: Vec::new(),
                intermediate_node_rows: Vec::new(),
                intermediate_edge_rows: Vec::new(),
                edge_rows: Vec::new(),
                end_rows: Vec::new(),
                integer_columns: Vec::new(),
                boolean_columns: Vec::new(),
                value_left_indices: Vec::new(),
                value_right_indices: Vec::new(),
                mutation: None,
            },
            scratch,
            &mut loader,
            &CancellationToken::new(),
        )?
        .validate(&request, BackendKind::Cpu)?;
        assert_eq!(result.source_positions(), [0, 1, 2, 3, 4, 5]);
        assert_eq!(
            result.projected_columns()[0].column,
            ResidentRowColumn::Integer {
                values: vec![20, 10, 0, 0, 0, 0],
                validity: vec![1, 1, 0, 0, 0, 0],
            }
        );
        Ok(())
    }

    #[test]
    fn list_build_and_concat_publish_one_canonical_list_column() -> Result<()> {
        let rows = 2;
        let program = ResidentRowProgram {
            instructions: vec![
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Integer,
                    operation: ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                        values: vec![7, 8],
                        validity: vec![1, 1],
                    }),
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::List,
                    operation: ResidentRowOperation::List { elements: vec![0] },
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::List,
                    operation: ResidentRowOperation::ListConcat { left: 1, right: 1 },
                },
            ],
        };
        let final_registers = vec![2];
        let manifest = ResidentRowProgramManifest::build(
            &program,
            &[],
            0,
            usize::MAX,
            rows,
            &final_registers,
            1,
        )?;
        let mut request = scalar_request(ResidentRowColumn::Integer {
            values: vec![0; rows],
            validity: vec![1; rows],
        })?;
        request.program = program;
        request.manifest = manifest;
        request.sort_keys.clear();
        request.final_registers = final_registers;
        request.validate()?;
        let scratch = request.scratch_bytes(rows)?;
        let mut loader = unexpected_property_loader;
        let result = execute_reference_row_program(
            &request,
            ResidentNodePipelineResult {
                start_rows: Vec::new(),
                intermediate_node_rows: Vec::new(),
                intermediate_edge_rows: Vec::new(),
                edge_rows: Vec::new(),
                end_rows: Vec::new(),
                integer_columns: Vec::new(),
                boolean_columns: Vec::new(),
                value_left_indices: Vec::new(),
                value_right_indices: Vec::new(),
                mutation: None,
            },
            scratch,
            &mut loader,
            &CancellationToken::new(),
        )?
        .validate(&request, BackendKind::Cpu)?;
        assert_eq!(
            result.projected_columns()[0].column,
            ResidentRowColumn::List {
                offsets: vec![0, 2, 4],
                values: vec![7, 7, 8, 8],
                element_validity: vec![1, 1, 1, 1],
                validity: vec![1, 1],
            }
        );
        Ok(())
    }
}
