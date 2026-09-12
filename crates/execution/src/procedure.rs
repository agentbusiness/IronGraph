//! Backend-neutral typed procedure relations used by execution backends.

use std::{collections::BTreeSet, sync::Arc};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use irongraph_types::{Error, ErrorCode, Result, ScalarValue};

use crate::ResultValue;

/// Scalar types accepted by an execution-scoped procedure signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProcedureValueType {
    Boolean,
    Integer,
    Float,
    Number,
    String,
}

impl ProcedureValueType {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Boolean => "BOOLEAN",
            Self::Integer => "INTEGER",
            Self::Float => "FLOAT",
            Self::Number => "NUMBER",
            Self::String => "STRING",
        }
    }

    pub fn accepts(self, value: &ResultValue, nullable: bool) -> bool {
        match value {
            ResultValue::Scalar(ScalarValue::Null) => nullable,
            ResultValue::Scalar(ScalarValue::Boolean(_)) => self == Self::Boolean,
            ResultValue::Scalar(ScalarValue::Integer(_)) => {
                matches!(self, Self::Integer | Self::Float | Self::Number)
            }
            ResultValue::Scalar(ScalarValue::Float(_)) => {
                matches!(self, Self::Float | Self::Number)
            }
            ResultValue::Scalar(ScalarValue::String(_)) => self == Self::String,
            ResultValue::Scalar(
                ScalarValue::Bytes(_)
                | ScalarValue::Date(_)
                | ScalarValue::LocalTime(_)
                | ScalarValue::ZonedTime { .. }
                | ScalarValue::LocalDateTime { .. }
                | ScalarValue::ZonedDateTime { .. }
                | ScalarValue::Duration { .. }
                | ScalarValue::List(_)
                | ScalarValue::Map(_),
            )
            | ResultValue::Node(_)
            | ResultValue::Relationship(_)
            | ResultValue::Path { .. }
            | ResultValue::Vector(_)
            | ResultValue::List(_)
            | ResultValue::Map(_) => false,
        }
    }

    pub fn normalize(
        self,
        value: ResultValue,
        nullable: bool,
        code: ErrorCode,
    ) -> Result<ResultValue> {
        if !self.accepts(&value, nullable) {
            return Err(Error::new(
                code,
                format!(
                    "InvalidArgumentType: expected {}{}, received {}",
                    self.name(),
                    if nullable { "?" } else { "" },
                    format!("{:?}", value.column_type()).to_ascii_uppercase()
                ),
            ));
        }
        match (self, value) {
            (Self::Float, ResultValue::Scalar(ScalarValue::Integer(value))) => Ok(
                ResultValue::Scalar(ScalarValue::Float(OrderedFloat(value as f64))),
            ),
            (_, value) => Ok(value),
        }
    }
}

/// One named input or output in a typed procedure signature.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProcedureField {
    pub name: String,
    pub value_type: ProcedureValueType,
    pub nullable: bool,
}

impl ProcedureField {
    pub fn new(
        name: impl Into<String>,
        value_type: ProcedureValueType,
        nullable: bool,
    ) -> Result<Self> {
        let name = name.into();
        if !valid_identifier(&name) {
            return Err(Error::invalid_data(format!(
                "invalid procedure field name `{name}`"
            )));
        }
        Ok(Self {
            name,
            value_type,
            nullable,
        })
    }
}

const RESIDENT_PROCEDURE_FINGERPRINT_DOMAIN: &[u8] = b"irongraph.resident-procedure-table.v1";

/// Deterministic identity of one normalized immutable procedure relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidentProcedureTableFingerprint([u8; 32]);

impl ResidentProcedureTableFingerprint {
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// One input or output field in a resident procedure relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentProcedureField {
    name: String,
    value_type: ProcedureValueType,
    nullable: bool,
    source_ordinal: u32,
}

impl ResidentProcedureField {
    fn from_field(field: &ProcedureField, source_ordinal: u32) -> Self {
        Self {
            name: field.name.clone(),
            value_type: field.value_type,
            nullable: field.nullable,
            source_ordinal,
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn value_type(&self) -> ProcedureValueType {
        self.value_type
    }

    #[must_use]
    pub const fn nullable(&self) -> bool {
        self.nullable
    }

    /// Ordinal in the source fixture row, where inputs precede outputs.
    #[must_use]
    pub const fn source_ordinal(&self) -> u32 {
        self.source_ordinal
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentProcedureBooleanColumn {
    validity: Vec<u8>,
    values: Vec<u8>,
}

impl ResidentProcedureBooleanColumn {
    #[must_use]
    pub fn validity(&self) -> &[u8] {
        &self.validity
    }

    #[must_use]
    pub fn values(&self) -> &[u8] {
        &self.values
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentProcedureIntegerColumn {
    validity: Vec<u8>,
    values: Vec<i64>,
}

impl ResidentProcedureIntegerColumn {
    #[must_use]
    pub fn validity(&self) -> &[u8] {
        &self.validity
    }

    #[must_use]
    pub fn values(&self) -> &[i64] {
        &self.values
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentProcedureFloatColumn {
    validity: Vec<u8>,
    value_bits: Vec<u64>,
}

impl ResidentProcedureFloatColumn {
    #[must_use]
    pub fn validity(&self) -> &[u8] {
        &self.validity
    }

    /// Canonical IEEE-754 binary64 payloads. Invalid rows contain zero.
    #[must_use]
    pub fn value_bits(&self) -> &[u64] {
        &self.value_bits
    }
}

/// Per-row payload kind for a NUMBER column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResidentProcedureNumberKind {
    Null = 0,
    Integer = 1,
    Float = 2,
}

impl ResidentProcedureNumberKind {
    #[must_use]
    pub const fn is_null(self) -> bool {
        matches!(self, Self::Null)
    }

    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::Integer)
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float)
    }

    const fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentProcedureNumberColumn {
    validity: Vec<u8>,
    kinds: Vec<ResidentProcedureNumberKind>,
    integer_values: Vec<i64>,
    float_value_bits: Vec<u64>,
}

impl ResidentProcedureNumberColumn {
    #[must_use]
    pub fn validity(&self) -> &[u8] {
        &self.validity
    }

    #[must_use]
    pub fn kinds(&self) -> &[ResidentProcedureNumberKind] {
        &self.kinds
    }

    /// Integer payload by row. Non-integer rows contain zero.
    #[must_use]
    pub fn integer_values(&self) -> &[i64] {
        &self.integer_values
    }

    /// Canonical IEEE-754 binary64 payload by row. Non-float rows contain zero.
    #[must_use]
    pub fn float_value_bits(&self) -> &[u64] {
        &self.float_value_bits
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentProcedureStringColumn {
    validity: Vec<u8>,
    offsets: Vec<u32>,
    utf8: Vec<u8>,
}

impl ResidentProcedureStringColumn {
    #[must_use]
    pub fn validity(&self) -> &[u8] {
        &self.validity
    }

    /// Row offsets into [`Self::utf8`]. There are always `row_count + 1` entries.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Complete UTF-8 bytes. Resident matching must compare these bytes, not a hash alone.
    #[must_use]
    pub fn utf8(&self) -> &[u8] {
        &self.utf8
    }
}

/// One normalized nullable column ready for backend-specific upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentProcedureColumn {
    Boolean(ResidentProcedureBooleanColumn),
    Integer(ResidentProcedureIntegerColumn),
    Float(ResidentProcedureFloatColumn),
    Number(ResidentProcedureNumberColumn),
    String(ResidentProcedureStringColumn),
}

impl ResidentProcedureColumn {
    #[must_use]
    pub const fn value_type(&self) -> ProcedureValueType {
        match self {
            Self::Boolean(_) => ProcedureValueType::Boolean,
            Self::Integer(_) => ProcedureValueType::Integer,
            Self::Float(_) => ProcedureValueType::Float,
            Self::Number(_) => ProcedureValueType::Number,
            Self::String(_) => ProcedureValueType::String,
        }
    }

    #[must_use]
    pub fn validity(&self) -> &[u8] {
        match self {
            Self::Boolean(column) => column.validity(),
            Self::Integer(column) => column.validity(),
            Self::Float(column) => column.validity(),
            Self::Number(column) => column.validity(),
            Self::String(column) => column.validity(),
        }
    }

    #[must_use]
    pub const fn as_boolean(&self) -> Option<&ResidentProcedureBooleanColumn> {
        match self {
            Self::Boolean(column) => Some(column),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_integer(&self) -> Option<&ResidentProcedureIntegerColumn> {
        match self {
            Self::Integer(column) => Some(column),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_float(&self) -> Option<&ResidentProcedureFloatColumn> {
        match self {
            Self::Float(column) => Some(column),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_number(&self) -> Option<&ResidentProcedureNumberColumn> {
        match self {
            Self::Number(column) => Some(column),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_string(&self) -> Option<&ResidentProcedureStringColumn> {
        match self {
            Self::String(column) => Some(column),
            _ => None,
        }
    }

    fn value_at(
        &self,
        procedure: &str,
        field: &ResidentProcedureField,
        row: usize,
    ) -> Result<ResultValue> {
        let validity = self.validity().get(row).copied().ok_or_else(|| {
            Error::invalid_data(format!(
                "resident procedure {procedure} column `{}` omits row {}",
                field.name,
                row + 1
            ))
        })?;
        if validity == 0 {
            return Ok(ResultValue::Scalar(ScalarValue::Null));
        }
        let scalar = match self {
            Self::Boolean(column) => ScalarValue::Boolean(
                *column.values.get(row).ok_or_else(|| {
                    Error::invalid_data(format!(
                        "resident procedure {procedure} BOOLEAN column `{}` omits row {}",
                        field.name,
                        row + 1
                    ))
                })? != 0,
            ),
            Self::Integer(column) => ScalarValue::Integer(*column.values.get(row).ok_or_else(|| {
                Error::invalid_data(format!(
                    "resident procedure {procedure} INTEGER column `{}` omits row {}",
                    field.name,
                    row + 1
                ))
            })?),
            Self::Float(column) => ScalarValue::Float(OrderedFloat(f64::from_bits(
                *column.value_bits.get(row).ok_or_else(|| {
                    Error::invalid_data(format!(
                        "resident procedure {procedure} FLOAT column `{}` omits row {}",
                        field.name,
                        row + 1
                    ))
                })?,
            ))),
            Self::Number(column) => match column.kinds.get(row).copied().ok_or_else(|| {
                Error::invalid_data(format!(
                    "resident procedure {procedure} NUMBER column `{}` omits row {}",
                    field.name,
                    row + 1
                ))
            })? {
                ResidentProcedureNumberKind::Integer => ScalarValue::Integer(
                    *column.integer_values.get(row).ok_or_else(|| {
                        Error::invalid_data(format!(
                            "resident procedure {procedure} NUMBER integer column `{}` omits row {}",
                            field.name,
                            row + 1
                        ))
                    })?,
                ),
                ResidentProcedureNumberKind::Float => ScalarValue::Float(OrderedFloat(
                    f64::from_bits(*column.float_value_bits.get(row).ok_or_else(|| {
                        Error::invalid_data(format!(
                            "resident procedure {procedure} NUMBER float column `{}` omits row {}",
                            field.name,
                            row + 1
                        ))
                    })?),
                )),
                ResidentProcedureNumberKind::Null => {
                    return Err(Error::invalid_data(format!(
                        "resident procedure {procedure} NUMBER column `{}` marks row {} valid but stores a NULL kind",
                        field.name,
                        row + 1
                    )));
                }
            },
            Self::String(column) => {
                let start = usize::try_from(*column.offsets.get(row).ok_or_else(|| {
                    Error::invalid_data(format!(
                        "resident procedure {procedure} STRING column `{}` omits row {} start offset",
                        field.name,
                        row + 1
                    ))
                })?)
                .map_err(|_| Error::invalid_data("resident procedure string offset does not fit usize"))?;
                let end = usize::try_from(*column.offsets.get(row + 1).ok_or_else(|| {
                    Error::invalid_data(format!(
                        "resident procedure {procedure} STRING column `{}` omits row {} end offset",
                        field.name,
                        row + 1
                    ))
                })?)
                .map_err(|_| Error::invalid_data("resident procedure string offset does not fit usize"))?;
                let value = std::str::from_utf8(column.utf8.get(start..end).ok_or_else(|| {
                    Error::invalid_data(format!(
                        "resident procedure {procedure} STRING column `{}` row {} escapes its UTF-8 arena",
                        field.name,
                        row + 1
                    ))
                })?)
                .map_err(|error| {
                    Error::invalid_data(format!(
                        "resident procedure {procedure} STRING column `{}` row {} is not UTF-8: {error}",
                        field.name,
                        row + 1
                    ))
                })?;
                ScalarValue::String(Arc::from(value))
            }
        };
        Ok(ResultValue::Scalar(scalar))
    }

    fn validate(
        &self,
        procedure: &str,
        field: &ResidentProcedureField,
        row_count: usize,
    ) -> Result<()> {
        if self.value_type() != field.value_type {
            return Err(Error::invalid_data(format!(
                "resident procedure {procedure} column `{}` has type {}, expected {}",
                field.name,
                self.value_type().name(),
                field.value_type.name()
            )));
        }
        if self.validity().len() != row_count {
            return Err(Error::invalid_data(format!(
                "resident procedure {procedure} column `{}` has {} validity rows, expected {row_count}",
                field.name,
                self.validity().len()
            )));
        }
        for (row, validity) in self.validity().iter().copied().enumerate() {
            if validity > 1 {
                return Err(Error::invalid_data(format!(
                    "resident procedure {procedure} column `{}` row {} has invalid validity {validity}",
                    field.name,
                    row + 1
                )));
            }
            if validity == 0 && !field.nullable {
                return Err(Error::invalid_data(format!(
                    "resident procedure {procedure} non-nullable column `{}` row {} is null",
                    field.name,
                    row + 1
                )));
            }
        }

        match self {
            Self::Boolean(column) => {
                validate_payload_len(procedure, field, "boolean", column.values.len(), row_count)?;
                for (row, (&validity, &value)) in
                    column.validity.iter().zip(&column.values).enumerate()
                {
                    if value > 1 || (validity == 0 && value != 0) {
                        return Err(Error::invalid_data(format!(
                            "resident procedure {procedure} BOOLEAN column `{}` row {} has a non-canonical payload",
                            field.name,
                            row + 1
                        )));
                    }
                }
            }
            Self::Integer(column) => {
                validate_payload_len(procedure, field, "integer", column.values.len(), row_count)?;
                validate_null_zeroes(
                    procedure,
                    field,
                    &column.validity,
                    column.values.iter().map(|value| *value == 0),
                )?;
            }
            Self::Float(column) => {
                validate_payload_len(
                    procedure,
                    field,
                    "float",
                    column.value_bits.len(),
                    row_count,
                )?;
                for (row, (&validity, &bits)) in
                    column.validity.iter().zip(&column.value_bits).enumerate()
                {
                    if (validity == 0 && bits != 0)
                        || canonical_float_bits(f64::from_bits(bits)) != bits
                    {
                        return Err(Error::invalid_data(format!(
                            "resident procedure {procedure} FLOAT column `{}` row {} has a non-canonical payload",
                            field.name,
                            row + 1
                        )));
                    }
                }
            }
            Self::Number(column) => {
                for (kind, len) in [
                    ("number kind", column.kinds.len()),
                    ("number integer", column.integer_values.len()),
                    ("number float", column.float_value_bits.len()),
                ] {
                    validate_payload_len(procedure, field, kind, len, row_count)?;
                }
                for row in 0..row_count {
                    let validity = column.validity[row];
                    let kind = column.kinds[row];
                    let integer = column.integer_values[row];
                    let float_bits = column.float_value_bits[row];
                    let canonical = match (validity, kind) {
                        (0, ResidentProcedureNumberKind::Null) => integer == 0 && float_bits == 0,
                        (1, ResidentProcedureNumberKind::Integer) => float_bits == 0,
                        (1, ResidentProcedureNumberKind::Float) => {
                            integer == 0
                                && canonical_float_bits(f64::from_bits(float_bits)) == float_bits
                        }
                        _ => false,
                    };
                    if !canonical {
                        return Err(Error::invalid_data(format!(
                            "resident procedure {procedure} NUMBER column `{}` row {} has a non-canonical payload",
                            field.name,
                            row + 1
                        )));
                    }
                }
            }
            Self::String(column) => {
                let expected_offsets = row_count.checked_add(1).ok_or_else(|| {
                    Error::invalid_data(format!(
                        "resident procedure {procedure} STRING column `{}` offset count overflows",
                        field.name
                    ))
                })?;
                validate_payload_len(
                    procedure,
                    field,
                    "string offset",
                    column.offsets.len(),
                    expected_offsets,
                )?;
                if column.offsets.first().copied() != Some(0) {
                    return Err(Error::invalid_data(format!(
                        "resident procedure {procedure} STRING column `{}` does not start at offset zero",
                        field.name
                    )));
                }
                for row in 0..row_count {
                    let start = usize::try_from(column.offsets[row]).map_err(|_| {
                        Error::invalid_data(format!(
                            "resident procedure {procedure} STRING column `{}` row {} start offset does not fit usize",
                            field.name,
                            row + 1
                        ))
                    })?;
                    let end = usize::try_from(column.offsets[row + 1]).map_err(|_| {
                        Error::invalid_data(format!(
                            "resident procedure {procedure} STRING column `{}` row {} end offset does not fit usize",
                            field.name,
                            row + 1
                        ))
                    })?;
                    if start > end || end > column.utf8.len() {
                        return Err(Error::invalid_data(format!(
                            "resident procedure {procedure} STRING column `{}` row {} has offsets {start}..{end} outside {} bytes",
                            field.name,
                            row + 1,
                            column.utf8.len()
                        )));
                    }
                    if column.validity[row] == 0 && start != end {
                        return Err(Error::invalid_data(format!(
                            "resident procedure {procedure} STRING column `{}` null row {} owns bytes",
                            field.name,
                            row + 1
                        )));
                    }
                    std::str::from_utf8(&column.utf8[start..end]).map_err(|error| {
                        Error::invalid_data(format!(
                            "resident procedure {procedure} STRING column `{}` row {} is not valid UTF-8: {error}",
                            field.name,
                            row + 1
                        ))
                    })?;
                }
                let final_offset = column.offsets.last().copied().unwrap_or_default();
                if usize::try_from(final_offset).ok() != Some(column.utf8.len()) {
                    return Err(Error::invalid_data(format!(
                        "resident procedure {procedure} STRING column `{}` final offset {final_offset} does not match {} bytes",
                        field.name,
                        column.utf8.len()
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Backend-neutral immutable procedure table used as the CPU/Metal CALL relation source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentProcedureTable {
    name: String,
    row_count: u32,
    inputs: Vec<ResidentProcedureField>,
    outputs: Vec<ResidentProcedureField>,
    columns: Vec<ResidentProcedureColumn>,
    fingerprint: ResidentProcedureTableFingerprint,
}

impl ResidentProcedureTable {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of relation rows, independent of the number of columns.
    #[must_use]
    pub const fn row_count(&self) -> u32 {
        self.row_count
    }

    #[must_use]
    pub fn inputs(&self) -> &[ResidentProcedureField] {
        &self.inputs
    }

    #[must_use]
    pub fn outputs(&self) -> &[ResidentProcedureField] {
        &self.outputs
    }

    /// Columns in stable source-ordinal order: all inputs, then all outputs.
    #[must_use]
    pub fn columns(&self) -> &[ResidentProcedureColumn] {
        &self.columns
    }

    #[must_use]
    pub fn column(&self, source_ordinal: u32) -> Option<&ResidentProcedureColumn> {
        usize::try_from(source_ordinal)
            .ok()
            .and_then(|ordinal| self.columns.get(ordinal))
    }

    #[must_use]
    pub const fn fingerprint(&self) -> ResidentProcedureTableFingerprint {
        self.fingerprint
    }

    /// Revalidates upload-facing shape and content integrity before backend admission.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() || !self.name.split('.').all(valid_identifier) {
            return Err(Error::invalid_data(format!(
                "invalid resident procedure name `{}`",
                self.name
            )));
        }
        validate_resident_fields(&self.name, "input", &self.inputs, 0)?;
        let output_base = u32::try_from(self.inputs.len()).map_err(|_| {
            Error::invalid_data(format!(
                "resident procedure {} has too many input columns",
                self.name
            ))
        })?;
        validate_resident_fields(&self.name, "output", &self.outputs, output_base)?;

        let fields = self.inputs.iter().chain(&self.outputs).collect::<Vec<_>>();
        if self.columns.len() != fields.len() {
            return Err(Error::invalid_data(format!(
                "resident procedure {} has {} columns, expected {}",
                self.name,
                self.columns.len(),
                fields.len()
            )));
        }
        let row_count = usize::try_from(self.row_count).map_err(|_| {
            Error::invalid_data(format!(
                "resident procedure {} row count does not fit usize",
                self.name
            ))
        })?;
        for (column, field) in self.columns.iter().zip(fields) {
            column.validate(&self.name, field, row_count)?;
        }

        let expected = resident_procedure_table_fingerprint(self);
        if self.fingerprint != expected {
            return Err(Error::invalid_data(format!(
                "resident procedure {} fingerprint does not match its immutable relation",
                self.name
            )));
        }
        Ok(())
    }

    /// Normalizes one invocation's immutable scalar arguments against this table's declared
    /// signature. This is compiler-side type admission only; relation matching remains backend
    /// work for GPU execution.
    pub fn normalize_arguments(&self, arguments: Vec<ResultValue>) -> Result<Vec<ResultValue>> {
        if arguments.len() != self.inputs.len() {
            return Err(invalid_arity(
                &self.name,
                self.inputs.len().to_string(),
                arguments.len(),
            ));
        }
        arguments
            .into_iter()
            .zip(&self.inputs)
            .map(|(value, field)| {
                field
                    .value_type
                    .normalize(value, field.nullable, ErrorCode::QueryType)
            })
            .collect()
    }

    /// CPU semantic reference for only the table-row selection stage. Accelerator backends must
    /// produce the same stable source-row IDs themselves; callers may materialize typed outputs
    /// only after the backend result and its table fingerprint have been validated.
    pub fn matching_rows_reference(&self, arguments: Vec<ResultValue>) -> Result<Vec<u32>> {
        self.validate()?;
        let arguments = self.normalize_arguments(arguments)?;
        let row_count = usize::try_from(self.row_count).map_err(|_| {
            Error::invalid_data(format!(
                "resident procedure {} row count does not fit usize",
                self.name
            ))
        })?;
        let mut output = Vec::new();
        for row in 0..row_count {
            let mut matches = true;
            for (field, argument) in self.inputs.iter().zip(&arguments) {
                let expected = self
                    .column(field.source_ordinal)
                    .ok_or_else(|| {
                        Error::invalid_data(format!(
                            "resident procedure {} omits input column `{}`",
                            self.name, field.name
                        ))
                    })?
                    .value_at(&self.name, field, row)?;
                if !procedure_value_equal(&expected, argument)? {
                    matches = false;
                    break;
                }
            }
            if !matches {
                continue;
            }
            output.push(u32::try_from(row).map_err(|_| {
                Error::invalid_data(format!(
                    "resident procedure {} row index exceeds u32",
                    self.name
                ))
            })?);
        }
        Ok(output)
    }

    /// Decodes output columns for backend-selected source rows. This routine never compares
    /// arguments or chooses rows. Strictly increasing row IDs make duplicate, reordered, or
    /// fabricated relation output detectable before query projection.
    pub fn materialize_output_rows(&self, rows: &[u32]) -> Result<Vec<Vec<ResultValue>>> {
        self.validate()?;
        if rows.windows(2).any(|pair| pair[0] >= pair[1])
            || rows.iter().any(|row| *row >= self.row_count)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident procedure backend returned duplicate, unordered, or out-of-bounds rows",
            ));
        }
        rows.iter()
            .copied()
            .map(|row| {
                let row = usize::try_from(row).map_err(|_| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "resident procedure backend row does not fit host addressing",
                    )
                })?;
                self.outputs
                    .iter()
                    .map(|field| {
                        self.column(field.source_ordinal)
                            .ok_or_else(|| {
                                Error::invalid_data(format!(
                                    "resident procedure {} omits output column `{}`",
                                    self.name, field.name
                                ))
                            })?
                            .value_at(&self.name, field, row)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect()
    }
}

/// A deterministic typed relation. Each row stores declared inputs followed by outputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcedureDefinition {
    name: String,
    inputs: Vec<ProcedureField>,
    outputs: Vec<ProcedureField>,
    row_count: usize,
    rows: Vec<Vec<ResultValue>>,
}

impl ProcedureDefinition {
    pub fn new(
        name: impl Into<String>,
        inputs: Vec<ProcedureField>,
        outputs: Vec<ProcedureField>,
        rows: Vec<Vec<ResultValue>>,
    ) -> Result<Self> {
        let row_count = if inputs.is_empty() && outputs.is_empty() && rows.is_empty() {
            // Preserve the established fixture contract deliberately: the compact fully-void
            // spelling used by test.doNothing is the unit relation, not the empty relation.
            1
        } else {
            rows.len()
        };
        Self::new_with_row_count(name, inputs, outputs, row_count, rows)
    }

    /// Constructs a relation with an explicit cardinality.
    ///
    /// For a zero-column relation, `rows` may be empty because `row_count` carries the complete
    /// cardinality. Non-zero-column relations must provide exactly one source row per relation row.
    pub fn new_with_row_count(
        name: impl Into<String>,
        inputs: Vec<ProcedureField>,
        outputs: Vec<ProcedureField>,
        row_count: usize,
        rows: Vec<Vec<ResultValue>>,
    ) -> Result<Self> {
        let name = name.into();
        if name.is_empty() || !name.split('.').all(valid_identifier) {
            return Err(Error::invalid_data(format!(
                "invalid procedure name `{name}`"
            )));
        }
        validate_unique_fields(&name, "input", &inputs)?;
        validate_unique_fields(&name, "output", &outputs)?;
        let fields = inputs.iter().chain(&outputs).collect::<Vec<_>>();
        let zero_column = fields.is_empty();
        if zero_column {
            if !rows.is_empty() && rows.len() != row_count {
                return Err(Error::invalid_data(format!(
                    "procedure {name} declares {row_count} rows but provides {} zero-column rows",
                    rows.len()
                )));
            }
        } else if rows.len() != row_count {
            return Err(Error::invalid_data(format!(
                "procedure {name} declares {row_count} rows but provides {} source rows",
                rows.len()
            )));
        }
        let mut normalized_rows = Vec::with_capacity(rows.len());
        for (row_index, row) in rows.into_iter().enumerate() {
            if row.len() != fields.len() {
                return Err(Error::invalid_data(format!(
                    "procedure {name} row {} has {} values, expected {}",
                    row_index + 1,
                    row.len(),
                    fields.len()
                )));
            }
            normalized_rows.push(
                row.into_iter()
                    .zip(&fields)
                    .map(|(value, field)| {
                        field
                            .value_type
                            .normalize(value, field.nullable, ErrorCode::InvalidData)
                            .map_err(|error| {
                                Error::invalid_data(format!(
                                    "procedure {name} row {}, field `{}`: {}",
                                    row_index + 1,
                                    field.name,
                                    error.message
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        Ok(Self {
            name,
            inputs,
            outputs,
            row_count,
            // Zero-column rows carry no values. Canonicalize away redundant empty tuples so the
            // explicit cardinality is their single source of truth.
            rows: if zero_column {
                Vec::new()
            } else {
                normalized_rows
            },
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn inputs(&self) -> &[ProcedureField] {
        &self.inputs
    }

    #[must_use]
    pub fn outputs(&self) -> &[ProcedureField] {
        &self.outputs
    }

    /// Number of rows in the immutable procedure relation, independent of its column count.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    /// Freezes this definition into a validated immutable columnar relation for resident execution.
    pub fn resident_table(&self) -> Result<ResidentProcedureTable> {
        if self.name.is_empty() || !self.name.split('.').all(valid_identifier) {
            return Err(Error::invalid_data(format!(
                "invalid procedure name `{}`",
                self.name
            )));
        }
        validate_unique_fields(&self.name, "input", &self.inputs)?;
        validate_unique_fields(&self.name, "output", &self.outputs)?;

        let fields = self.inputs.iter().chain(&self.outputs).collect::<Vec<_>>();
        if fields.is_empty() {
            if !self.rows.is_empty() {
                return Err(Error::invalid_data(format!(
                    "procedure {} stores source rows for a zero-column relation",
                    self.name
                )));
            }
        } else if self.rows.len() != self.row_count {
            return Err(Error::invalid_data(format!(
                "procedure {} declares {} rows but stores {} source rows",
                self.name,
                self.row_count,
                self.rows.len()
            )));
        }
        let row_count = u32::try_from(self.row_count).map_err(|_| {
            Error::invalid_data(format!(
                "procedure {} has too many rows for resident execution",
                self.name
            ))
        })?;

        for (row_index, row) in self.rows.iter().enumerate() {
            if row.len() != fields.len() {
                return Err(Error::invalid_data(format!(
                    "procedure {} row {} has {} values, expected {}",
                    self.name,
                    row_index + 1,
                    row.len(),
                    fields.len()
                )));
            }
        }

        let mut inputs = Vec::with_capacity(self.inputs.len());
        let mut outputs = Vec::with_capacity(self.outputs.len());
        for (ordinal, field) in self.inputs.iter().enumerate() {
            inputs.push(ResidentProcedureField::from_field(
                field,
                u32::try_from(ordinal).map_err(|_| {
                    Error::invalid_data(format!(
                        "procedure {} has too many input columns",
                        self.name
                    ))
                })?,
            ));
        }
        for (output_ordinal, field) in self.outputs.iter().enumerate() {
            let ordinal = self
                .inputs
                .len()
                .checked_add(output_ordinal)
                .and_then(|ordinal| u32::try_from(ordinal).ok())
                .ok_or_else(|| {
                    Error::invalid_data(format!("procedure {} has too many columns", self.name))
                })?;
            outputs.push(ResidentProcedureField::from_field(field, ordinal));
        }

        let columns = fields
            .into_iter()
            .enumerate()
            .map(|(ordinal, field)| {
                build_resident_procedure_column(&self.name, ordinal, field, &self.rows)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut table = ResidentProcedureTable {
            name: self.name.clone(),
            row_count,
            inputs,
            outputs,
            columns,
            fingerprint: ResidentProcedureTableFingerprint([0; 32]),
        };
        table.fingerprint = resident_procedure_table_fingerprint(&table);
        table.validate()?;
        Ok(table)
    }

    pub fn execute(&self, arguments: Vec<ResultValue>) -> Result<Vec<Vec<ResultValue>>> {
        if arguments.len() != self.inputs.len() {
            return Err(invalid_arity(
                &self.name,
                self.inputs.len().to_string(),
                arguments.len(),
            ));
        }
        let arguments = arguments
            .into_iter()
            .zip(&self.inputs)
            .map(|(value, field)| {
                field
                    .value_type
                    .normalize(value, field.nullable, ErrorCode::QueryType)
            })
            .collect::<Result<Vec<_>>>()?;

        if self.inputs.is_empty() && self.outputs.is_empty() {
            return Ok(vec![Vec::new(); self.row_count]);
        }

        let mut output = Vec::new();
        for row in &self.rows {
            let matches = row
                .iter()
                .take(self.inputs.len())
                .zip(&arguments)
                .try_fold(true, |matches, (expected, actual)| {
                    if !matches {
                        return Ok(false);
                    }
                    procedure_value_equal(expected, actual)
                })?;
            if matches {
                output.push(row[self.inputs.len()..].to_vec());
            }
        }
        Ok(output)
    }
}

fn build_resident_procedure_column(
    procedure: &str,
    ordinal: usize,
    field: &ProcedureField,
    rows: &[Vec<ResultValue>],
) -> Result<ResidentProcedureColumn> {
    let normalized = rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            let value = row.get(ordinal).ok_or_else(|| {
                Error::invalid_data(format!(
                    "procedure {procedure} row {} has no source ordinal {ordinal}",
                    row_index + 1
                ))
            })?;
            field
                .value_type
                .normalize(value.clone(), field.nullable, ErrorCode::InvalidData)
                .map_err(|error| {
                    Error::invalid_data(format!(
                        "procedure {procedure} row {}, field `{}`: {}",
                        row_index + 1,
                        field.name,
                        error.message
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;

    match field.value_type {
        ProcedureValueType::Boolean => {
            let mut validity = Vec::with_capacity(rows.len());
            let mut values = Vec::with_capacity(rows.len());
            for value in normalized {
                match value {
                    ResultValue::Scalar(ScalarValue::Null) => {
                        validity.push(0);
                        values.push(0);
                    }
                    ResultValue::Scalar(ScalarValue::Boolean(value)) => {
                        validity.push(1);
                        values.push(u8::from(value));
                    }
                    _ => return Err(resident_normalization_error(procedure, field)),
                }
            }
            Ok(ResidentProcedureColumn::Boolean(
                ResidentProcedureBooleanColumn { validity, values },
            ))
        }
        ProcedureValueType::Integer => {
            let mut validity = Vec::with_capacity(rows.len());
            let mut values = Vec::with_capacity(rows.len());
            for value in normalized {
                match value {
                    ResultValue::Scalar(ScalarValue::Null) => {
                        validity.push(0);
                        values.push(0);
                    }
                    ResultValue::Scalar(ScalarValue::Integer(value)) => {
                        validity.push(1);
                        values.push(value);
                    }
                    _ => return Err(resident_normalization_error(procedure, field)),
                }
            }
            Ok(ResidentProcedureColumn::Integer(
                ResidentProcedureIntegerColumn { validity, values },
            ))
        }
        ProcedureValueType::Float => {
            let mut validity = Vec::with_capacity(rows.len());
            let mut value_bits = Vec::with_capacity(rows.len());
            for value in normalized {
                match value {
                    ResultValue::Scalar(ScalarValue::Null) => {
                        validity.push(0);
                        value_bits.push(0);
                    }
                    ResultValue::Scalar(ScalarValue::Float(value)) => {
                        validity.push(1);
                        value_bits.push(canonical_float_bits(value.0));
                    }
                    _ => return Err(resident_normalization_error(procedure, field)),
                }
            }
            Ok(ResidentProcedureColumn::Float(
                ResidentProcedureFloatColumn {
                    validity,
                    value_bits,
                },
            ))
        }
        ProcedureValueType::Number => {
            let mut validity = Vec::with_capacity(rows.len());
            let mut kinds = Vec::with_capacity(rows.len());
            let mut integer_values = Vec::with_capacity(rows.len());
            let mut float_value_bits = Vec::with_capacity(rows.len());
            for value in normalized {
                match value {
                    ResultValue::Scalar(ScalarValue::Null) => {
                        validity.push(0);
                        kinds.push(ResidentProcedureNumberKind::Null);
                        integer_values.push(0);
                        float_value_bits.push(0);
                    }
                    ResultValue::Scalar(ScalarValue::Integer(value)) => {
                        validity.push(1);
                        kinds.push(ResidentProcedureNumberKind::Integer);
                        integer_values.push(value);
                        float_value_bits.push(0);
                    }
                    ResultValue::Scalar(ScalarValue::Float(value)) => {
                        validity.push(1);
                        kinds.push(ResidentProcedureNumberKind::Float);
                        integer_values.push(0);
                        float_value_bits.push(canonical_float_bits(value.0));
                    }
                    _ => return Err(resident_normalization_error(procedure, field)),
                }
            }
            Ok(ResidentProcedureColumn::Number(
                ResidentProcedureNumberColumn {
                    validity,
                    kinds,
                    integer_values,
                    float_value_bits,
                },
            ))
        }
        ProcedureValueType::String => {
            let mut validity = Vec::with_capacity(rows.len());
            let mut offsets = Vec::with_capacity(rows.len().saturating_add(1));
            let mut utf8 = Vec::new();
            offsets.push(0);
            for value in normalized {
                match value {
                    ResultValue::Scalar(ScalarValue::Null) => validity.push(0),
                    ResultValue::Scalar(ScalarValue::String(value)) => {
                        validity.push(1);
                        let next_len = utf8.len().checked_add(value.len()).ok_or_else(|| {
                            Error::invalid_data(format!(
                                "procedure {procedure} STRING field `{}` byte length overflows",
                                field.name
                            ))
                        })?;
                        u32::try_from(next_len).map_err(|_| {
                            Error::invalid_data(format!(
                                "procedure {procedure} STRING field `{}` exceeds the resident u32 offset range",
                                field.name
                            ))
                        })?;
                        utf8.extend_from_slice(value.as_bytes());
                    }
                    _ => return Err(resident_normalization_error(procedure, field)),
                }
                offsets.push(u32::try_from(utf8.len()).map_err(|_| {
                    Error::invalid_data(format!(
                        "procedure {procedure} STRING field `{}` exceeds the resident u32 offset range",
                        field.name
                    ))
                })?);
            }
            Ok(ResidentProcedureColumn::String(
                ResidentProcedureStringColumn {
                    validity,
                    offsets,
                    utf8,
                },
            ))
        }
    }
}

fn resident_normalization_error(procedure: &str, field: &ProcedureField) -> Error {
    Error::invalid_data(format!(
        "procedure {procedure} field `{}` did not normalize to {}",
        field.name,
        field.value_type.name()
    ))
}

fn validate_payload_len(
    procedure: &str,
    field: &ResidentProcedureField,
    payload: &str,
    actual: usize,
    expected: usize,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::invalid_data(format!(
            "resident procedure {procedure} column `{}` has {actual} {payload} entries, expected {expected}",
            field.name
        )))
    }
}

fn validate_null_zeroes(
    procedure: &str,
    field: &ResidentProcedureField,
    validity: &[u8],
    zeroes: impl IntoIterator<Item = bool>,
) -> Result<()> {
    for (row, (&validity, is_zero)) in validity.iter().zip(zeroes).enumerate() {
        if validity == 0 && !is_zero {
            return Err(Error::invalid_data(format!(
                "resident procedure {procedure} column `{}` null row {} has a non-zero payload",
                field.name,
                row + 1
            )));
        }
    }
    Ok(())
}

fn validate_resident_fields(
    procedure: &str,
    kind: &str,
    fields: &[ResidentProcedureField],
    ordinal_base: u32,
) -> Result<()> {
    let mut names = BTreeSet::new();
    for (index, field) in fields.iter().enumerate() {
        if !valid_identifier(&field.name) {
            return Err(Error::invalid_data(format!(
                "resident procedure {procedure} has invalid {kind} field `{}`",
                field.name
            )));
        }
        if !names.insert(field.name.as_str()) {
            return Err(Error::invalid_data(format!(
                "resident procedure {procedure} repeats {kind} field `{}`",
                field.name
            )));
        }
        let expected = ordinal_base
            .checked_add(u32::try_from(index).map_err(|_| {
                Error::invalid_data(format!(
                    "resident procedure {procedure} has too many {kind} fields"
                ))
            })?)
            .ok_or_else(|| {
                Error::invalid_data(format!(
                    "resident procedure {procedure} {kind} source ordinal overflows"
                ))
            })?;
        if field.source_ordinal != expected {
            return Err(Error::invalid_data(format!(
                "resident procedure {procedure} {kind} field `{}` has source ordinal {}, expected {expected}",
                field.name, field.source_ordinal
            )));
        }
    }
    Ok(())
}

fn resident_procedure_table_fingerprint(
    table: &ResidentProcedureTable,
) -> ResidentProcedureTableFingerprint {
    let mut hasher = blake3::Hasher::new();
    fingerprint_bytes(&mut hasher, RESIDENT_PROCEDURE_FINGERPRINT_DOMAIN);
    fingerprint_bytes(&mut hasher, table.name.as_bytes());
    hasher.update(&table.row_count.to_le_bytes());
    fingerprint_fields(&mut hasher, 0, &table.inputs);
    fingerprint_fields(&mut hasher, 1, &table.outputs);
    fingerprint_len(&mut hasher, table.columns.len());
    for column in &table.columns {
        hasher.update(&[procedure_value_type_code(column.value_type())]);
        fingerprint_bytes(&mut hasher, column.validity());
        match column {
            ResidentProcedureColumn::Boolean(column) => {
                fingerprint_bytes(&mut hasher, &column.values);
            }
            ResidentProcedureColumn::Integer(column) => {
                fingerprint_i64s(&mut hasher, &column.values);
            }
            ResidentProcedureColumn::Float(column) => {
                fingerprint_u64s(&mut hasher, &column.value_bits);
            }
            ResidentProcedureColumn::Number(column) => {
                fingerprint_len(&mut hasher, column.kinds.len());
                for kind in &column.kinds {
                    hasher.update(&[kind.code()]);
                }
                fingerprint_i64s(&mut hasher, &column.integer_values);
                fingerprint_u64s(&mut hasher, &column.float_value_bits);
            }
            ResidentProcedureColumn::String(column) => {
                fingerprint_len(&mut hasher, column.offsets.len());
                for offset in &column.offsets {
                    hasher.update(&offset.to_le_bytes());
                }
                fingerprint_bytes(&mut hasher, &column.utf8);
            }
        }
    }
    ResidentProcedureTableFingerprint(*hasher.finalize().as_bytes())
}

fn fingerprint_fields(
    hasher: &mut blake3::Hasher,
    direction: u8,
    fields: &[ResidentProcedureField],
) {
    hasher.update(&[direction]);
    fingerprint_len(hasher, fields.len());
    for field in fields {
        fingerprint_bytes(hasher, field.name.as_bytes());
        hasher.update(&[procedure_value_type_code(field.value_type)]);
        hasher.update(&[u8::from(field.nullable)]);
        hasher.update(&field.source_ordinal.to_le_bytes());
    }
}

fn fingerprint_i64s(hasher: &mut blake3::Hasher, values: &[i64]) {
    fingerprint_len(hasher, values.len());
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn fingerprint_u64s(hasher: &mut blake3::Hasher, values: &[u64]) {
    fingerprint_len(hasher, values.len());
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn fingerprint_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    fingerprint_len(hasher, bytes.len());
    hasher.update(bytes);
}

fn fingerprint_len(hasher: &mut blake3::Hasher, len: usize) {
    hasher.update(&(len as u64).to_le_bytes());
}

const fn procedure_value_type_code(value_type: ProcedureValueType) -> u8 {
    match value_type {
        ProcedureValueType::Boolean => 0,
        ProcedureValueType::Integer => 1,
        ProcedureValueType::Float => 2,
        ProcedureValueType::Number => 3,
        ProcedureValueType::String => 4,
    }
}

fn canonical_float_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else if value.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        value.to_bits()
    }
}

fn invalid_arity(name: &str, expected: String, actual: usize) -> Error {
    Error::new(
        ErrorCode::QuerySyntax,
        format!(
            "InvalidNumberOfArguments: procedure {name} expects {expected} arguments, received {actual}"
        ),
    )
}

fn procedure_value_equal(left: &ResultValue, right: &ResultValue) -> Result<bool> {
    Ok(match (left, right) {
        (ResultValue::Scalar(ScalarValue::Null), ResultValue::Scalar(ScalarValue::Null)) => true,
        (
            ResultValue::Scalar(ScalarValue::Integer(left)),
            ResultValue::Scalar(ScalarValue::Float(right)),
        ) => !right.0.is_nan() && crate::compare_i64_f64(*left, right.0).is_eq(),
        (
            ResultValue::Scalar(ScalarValue::Float(left)),
            ResultValue::Scalar(ScalarValue::Integer(right)),
        ) => !left.0.is_nan() && crate::compare_i64_f64(*right, left.0).is_eq(),
        (
            ResultValue::Scalar(ScalarValue::Float(left)),
            ResultValue::Scalar(ScalarValue::Float(right)),
        ) => !left.0.is_nan() && !right.0.is_nan() && left == right,
        _ => left == right,
    })
}

fn validate_unique_fields(procedure: &str, kind: &str, fields: &[ProcedureField]) -> Result<()> {
    let mut names = BTreeSet::new();
    for field in fields {
        if !valid_identifier(&field.name) {
            return Err(Error::invalid_data(format!(
                "procedure {procedure} has invalid {kind} field `{}`",
                field.name
            )));
        }
        if !names.insert(field.name.as_str()) {
            return Err(Error::invalid_data(format!(
                "procedure {procedure} repeats {kind} field `{}`",
                field.name
            )));
        }
    }
    Ok(())
}

fn valid_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resident_table_validation_rejects_corrupt_upload_shapes() -> Result<()> {
        let definition = ProcedureDefinition::new(
            "test.resident",
            vec![ProcedureField::new(
                "name",
                ProcedureValueType::String,
                true,
            )?],
            vec![ProcedureField::new(
                "number",
                ProcedureValueType::Number,
                true,
            )?],
            vec![vec![
                ResultValue::Scalar(ScalarValue::String(Arc::from("Malmö"))),
                ResultValue::Scalar(ScalarValue::Integer(7)),
            ]],
        )?;
        let table = definition.resident_table()?;

        let mut corrupt_offsets = table.clone();
        let ResidentProcedureColumn::String(column) = &mut corrupt_offsets.columns[0] else {
            panic!("resident STRING fixture changed type");
        };
        column.offsets[1] = u32::MAX;
        assert_eq!(
            corrupt_offsets
                .validate()
                .expect_err("out-of-bounds UTF-8 offsets must fail")
                .code,
            ErrorCode::InvalidData
        );

        let mut corrupt_validity = table.clone();
        let ResidentProcedureColumn::Number(column) = &mut corrupt_validity.columns[1] else {
            panic!("resident NUMBER fixture changed type");
        };
        column.validity[0] = 2;
        assert_eq!(
            corrupt_validity
                .validate()
                .expect_err("non-canonical validity must fail")
                .code,
            ErrorCode::InvalidData
        );

        let mut corrupt_fingerprint = table;
        corrupt_fingerprint.fingerprint.0[0] ^= 1;
        assert_eq!(
            corrupt_fingerprint
                .validate()
                .expect_err("content with the wrong fingerprint must fail")
                .code,
            ErrorCode::InvalidData
        );
        Ok(())
    }

    #[test]
    fn resident_table_validation_binds_row_count_to_every_column_kind() -> Result<()> {
        let cases = [
            (
                ProcedureValueType::Boolean,
                ResultValue::Scalar(ScalarValue::Boolean(true)),
            ),
            (
                ProcedureValueType::Integer,
                ResultValue::Scalar(ScalarValue::Integer(7)),
            ),
            (
                ProcedureValueType::Float,
                ResultValue::Scalar(ScalarValue::Float(OrderedFloat(7.5))),
            ),
            (
                ProcedureValueType::Number,
                ResultValue::Scalar(ScalarValue::Integer(7)),
            ),
            (
                ProcedureValueType::String,
                ResultValue::Scalar(ScalarValue::String(Arc::from("seven"))),
            ),
        ];

        for (index, (value_type, value)) in cases.into_iter().enumerate() {
            let definition = ProcedureDefinition::new(
                format!("test.rowCount{index}"),
                Vec::new(),
                vec![ProcedureField::new("value", value_type, false)?],
                vec![vec![value]],
            )?;
            let mut table = definition.resident_table()?;
            table.row_count = 2;
            let error = table
                .validate()
                .expect_err("each column kind must be checked against relation row_count");
            assert_eq!(error.code, ErrorCode::InvalidData);
            assert!(
                error.message.contains("validity rows, expected 2"),
                "unexpected row-count validation error: {}",
                error.message
            );
        }
        Ok(())
    }

    #[test]
    fn resident_fingerprint_covers_source_ordinal() -> Result<()> {
        let definition = ProcedureDefinition::new_with_row_count(
            "test.ordinals",
            vec![
                ProcedureField::new("first", ProcedureValueType::String, true)?,
                ProcedureField::new("second", ProcedureValueType::String, true)?,
            ],
            Vec::new(),
            0,
            Vec::new(),
        )?;
        let table = definition.resident_table()?;
        let mut changed = table.clone();
        changed.inputs[0].source_ordinal = 1;
        changed.fingerprint = resident_procedure_table_fingerprint(&changed);

        assert_ne!(table.fingerprint, changed.fingerprint);
        let error = changed
            .validate()
            .expect_err("an invalid source ordinal must not validate");
        assert_eq!(error.code, ErrorCode::InvalidData);
        assert!(error.message.contains("source ordinal 1, expected 0"));
        Ok(())
    }
}
