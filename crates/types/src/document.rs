//! Canonical checked encoding for non-temporal list and map properties.
//!
//! The durable/device form is one flat byte stream per property row. Recursive host values exist
//! only while accepting a mutation or materializing a result; canonical graph columns never own a
//! per-element object graph.

use std::{collections::BTreeMap, sync::Arc};

use ordered_float::OrderedFloat;

use crate::{DocumentList, DocumentMap, Error, ErrorCode, Result, ScalarValue};

const FORMAT_VERSION: u8 = 3;
const ENCODED_LENGTH_BYTES: usize = std::mem::size_of::<u64>();

const NULL: u8 = 0;
const FALSE: u8 = 1;
const TRUE: u8 = 2;
const INTEGER: u8 = 3;
const FLOAT: u8 = 4;
const STRING: u8 = 5;
const BYTES: u8 = 6;
const DATE: u8 = 7;
const LOCAL_TIME: u8 = 8;
const ZONED_TIME: u8 = 9;
const LOCAL_DATETIME: u8 = 10;
const ZONED_DATETIME: u8 = 11;
const DURATION: u8 = 12;
const LIST: u8 = 13;
const MAP: u8 = 14;

/// Root kind carried by the enclosing typed property column and checked during deserialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocumentRoot {
    List,
    Map,
}

impl DocumentRoot {
    const fn tag(self) -> u8 {
        match self {
            Self::List => LIST,
            Self::Map => MAP,
        }
    }
}

/// Transient semantic document tree used only at mutation/result boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DocumentItem {
    Scalar(ScalarValue),
    List(Vec<DocumentItem>),
    Map(BTreeMap<Arc<str>, DocumentItem>),
}

impl DocumentList {
    /// Encodes one list into its deterministic flat representation.
    pub fn new(values: Vec<DocumentItem>) -> Result<Self> {
        encode(DocumentItem::List(values), DocumentRoot::List).map(Self)
    }

    /// Decodes a result-boundary tree after the canonical bytes have passed strict validation.
    pub fn items(&self) -> Result<Vec<DocumentItem>> {
        match decode(&self.0, DocumentRoot::List)? {
            DocumentItem::List(values) => Ok(values),
            _ => Err(Error::new(
                ErrorCode::CorruptStorage,
                "list document has a non-list root",
            )),
        }
    }

    /// Canonical flat bytes uploaded directly with the property column.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Wraps bytes already validated by a canonical storage or device boundary.
    #[doc(hidden)]
    #[must_use]
    pub fn from_canonical(bytes: Arc<[u8]>) -> Self {
        Self(bytes)
    }
}

impl DocumentMap {
    /// Encodes one map in lexical key order into its deterministic flat representation.
    pub fn new(values: BTreeMap<Arc<str>, DocumentItem>) -> Result<Self> {
        encode(DocumentItem::Map(values), DocumentRoot::Map).map(Self)
    }

    /// Decodes a result-boundary tree after the canonical bytes have passed strict validation.
    pub fn entries(&self) -> Result<BTreeMap<Arc<str>, DocumentItem>> {
        match decode(&self.0, DocumentRoot::Map)? {
            DocumentItem::Map(values) => Ok(values),
            _ => Err(Error::new(
                ErrorCode::CorruptStorage,
                "map document has a non-map root",
            )),
        }
    }

    /// Canonical flat bytes uploaded directly with the property column.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Wraps bytes already validated by a canonical storage or device boundary.
    #[doc(hidden)]
    #[must_use]
    pub fn from_canonical(bytes: Arc<[u8]>) -> Self {
        Self(bytes)
    }
}

fn encode(value: DocumentItem, root: DocumentRoot) -> Result<Arc<[u8]>> {
    if !matches!(
        (&value, root),
        (DocumentItem::List(_), DocumentRoot::List) | (DocumentItem::Map(_), DocumentRoot::Map)
    ) {
        return Err(Error::new(
            ErrorCode::QueryType,
            "document root does not match its property type",
        ));
    }
    let mut bytes = Vec::new();
    push_byte(&mut bytes, FORMAT_VERSION)?;
    encode_item(&mut bytes, value)?;
    if bytes.get(1).copied() != Some(root.tag()) {
        return Err(Error::new(
            ErrorCode::QueryType,
            "document root does not match its property type",
        ));
    }
    Ok(bytes.into())
}

enum EncodeFrame {
    Item(DocumentItem),
    List(std::vec::IntoIter<DocumentItem>),
    Map(std::collections::btree_map::IntoIter<Arc<str>, DocumentItem>),
}

fn encode_item(output: &mut Vec<u8>, value: DocumentItem) -> Result<()> {
    let mut stack = Vec::new();
    push_frame(&mut stack, EncodeFrame::Item(value))?;
    while let Some(frame) = stack.pop() {
        match frame {
            EncodeFrame::Item(DocumentItem::Scalar(value)) => encode_scalar(output, &value)?,
            EncodeFrame::Item(DocumentItem::List(values)) => {
                push_byte(output, LIST)?;
                push_len(output, values.len())?;
                if !values.is_empty() {
                    push_frame(&mut stack, EncodeFrame::List(values.into_iter()))?;
                }
            }
            EncodeFrame::Item(DocumentItem::Map(values)) => {
                push_byte(output, MAP)?;
                push_len(output, values.len())?;
                if !values.is_empty() {
                    push_frame(&mut stack, EncodeFrame::Map(values.into_iter()))?;
                }
            }
            EncodeFrame::List(mut values) => {
                if let Some(value) = values.next() {
                    reserve_frames(&mut stack, 2)?;
                    stack.push(EncodeFrame::List(values));
                    stack.push(EncodeFrame::Item(value));
                }
            }
            EncodeFrame::Map(mut values) => {
                if let Some((key, value)) = values.next() {
                    push_bytes(output, key.as_bytes())?;
                    reserve_frames(&mut stack, 2)?;
                    stack.push(EncodeFrame::Map(values));
                    stack.push(EncodeFrame::Item(value));
                }
            }
        }
    }
    Ok(())
}

fn encode_scalar(output: &mut Vec<u8>, value: &ScalarValue) -> Result<()> {
    match value {
        ScalarValue::Null => push_byte(output, NULL)?,
        ScalarValue::Boolean(false) => push_byte(output, FALSE)?,
        ScalarValue::Boolean(true) => push_byte(output, TRUE)?,
        ScalarValue::Integer(value) => {
            push_byte(output, INTEGER)?;
            extend_output(output, &value.to_le_bytes())?;
        }
        ScalarValue::Float(value) => {
            push_byte(output, FLOAT)?;
            let value = value.into_inner();
            let bits = if value.is_nan() {
                f64::NAN.to_bits()
            } else if value == 0.0 {
                0.0_f64.to_bits()
            } else {
                value.to_bits()
            };
            extend_output(output, &bits.to_le_bytes())?;
        }
        ScalarValue::String(value) => {
            push_byte(output, STRING)?;
            push_bytes(output, value.as_bytes())?;
        }
        ScalarValue::Bytes(value) => {
            push_byte(output, BYTES)?;
            push_bytes(output, value)?;
        }
        ScalarValue::Date(value) => {
            push_byte(output, DATE)?;
            extend_output(output, &value.to_le_bytes())?;
        }
        ScalarValue::LocalTime(value) => {
            push_byte(output, LOCAL_TIME)?;
            extend_output(output, &value.to_le_bytes())?;
        }
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => {
            push_byte(output, ZONED_TIME)?;
            extend_output(output, &nanos.to_le_bytes())?;
            extend_output(output, &offset_seconds.to_le_bytes())?;
        }
        ScalarValue::LocalDateTime { seconds, nanos } => {
            push_byte(output, LOCAL_DATETIME)?;
            extend_output(output, &seconds.to_le_bytes())?;
            extend_output(output, &nanos.to_le_bytes())?;
        }
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            push_byte(output, ZONED_DATETIME)?;
            extend_output(output, &seconds.to_le_bytes())?;
            extend_output(output, &nanos.to_le_bytes())?;
            push_bytes(output, timezone.as_bytes())?;
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => {
            push_byte(output, DURATION)?;
            extend_output(output, &months.to_le_bytes())?;
            extend_output(output, &days.to_le_bytes())?;
            extend_output(output, &seconds.to_le_bytes())?;
            extend_output(output, &nanos.to_le_bytes())?;
        }
        ScalarValue::List(_) | ScalarValue::Map(_) => {
            return Err(Error::new(
                ErrorCode::QueryType,
                "nested document must use a list or map item, not a scalar wrapper",
            ));
        }
    }
    Ok(())
}

fn decode(bytes: &[u8], expected: DocumentRoot) -> Result<DocumentItem> {
    validate_size(bytes)?;
    let mut cursor = Cursor { bytes, position: 0 };
    if cursor.byte()? != FORMAT_VERSION {
        return Err(corrupt("document format version is unsupported"));
    }
    if cursor.bytes.get(cursor.position).copied() != Some(expected.tag()) {
        return Err(corrupt("document root kind is invalid"));
    }
    let value = cursor.item()?;
    if cursor.position != bytes.len() {
        return Err(corrupt("document contains trailing bytes"));
    }
    if !matches!(
        (&value, expected),
        (DocumentItem::List(_), DocumentRoot::List) | (DocumentItem::Map(_), DocumentRoot::Map)
    ) {
        return Err(corrupt("document root kind is invalid"));
    }
    Ok(value)
}

/// Validates untrusted checkpoint/log bytes before a document wrapper can enter canonical state.
/// Validates bytes crossing a canonical storage or device boundary.
#[doc(hidden)]
pub fn validate_canonical(bytes: &[u8], expected: DocumentRoot) -> Result<()> {
    validate_size(bytes)?;
    let mut cursor = Cursor { bytes, position: 0 };
    if cursor.byte()? != FORMAT_VERSION {
        return Err(corrupt("document format version is unsupported"));
    }
    if cursor.bytes.get(cursor.position).copied() != Some(expected.tag()) {
        return Err(corrupt("document root kind is invalid"));
    }
    cursor.skip_item()?;
    if cursor.position != bytes.len() {
        return Err(corrupt("document contains trailing bytes"));
    }
    Ok(())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

enum DecodeFrame {
    List {
        remaining: usize,
        values: Vec<DocumentItem>,
    },
    Map {
        remaining: usize,
        values: BTreeMap<Arc<str>, DocumentItem>,
        previous_key: (usize, usize),
        pending_key: Option<Arc<str>>,
    },
}

enum ValidationFrame {
    List {
        remaining: usize,
    },
    Map {
        remaining: usize,
        previous_key: (usize, usize),
    },
}

impl Cursor<'_> {
    fn item(&mut self) -> Result<DocumentItem> {
        let mut frames = Vec::new();
        let mut completed = None;

        loop {
            if completed.is_none() {
                let tag = self.byte()?;
                completed = match tag {
                    NULL => Some(DocumentItem::Scalar(ScalarValue::Null)),
                    FALSE => Some(DocumentItem::Scalar(ScalarValue::Boolean(false))),
                    TRUE => Some(DocumentItem::Scalar(ScalarValue::Boolean(true))),
                    INTEGER => Some(DocumentItem::Scalar(ScalarValue::Integer(self.i64()?))),
                    FLOAT => {
                        let bits = self.u64()?;
                        validate_float_bits(bits)?;
                        Some(DocumentItem::Scalar(ScalarValue::Float(OrderedFloat(
                            f64::from_bits(bits),
                        ))))
                    }
                    STRING => Some(DocumentItem::Scalar(ScalarValue::String(Arc::from(
                        self.string()?,
                    )))),
                    BYTES => Some(DocumentItem::Scalar(ScalarValue::Bytes(Arc::from(
                        self.bytes()?,
                    )))),
                    DATE => Some(DocumentItem::Scalar(ScalarValue::Date(self.i64()?))),
                    LOCAL_TIME => Some(DocumentItem::Scalar(ScalarValue::LocalTime(self.i64()?))),
                    ZONED_TIME => Some(DocumentItem::Scalar(ScalarValue::ZonedTime {
                        nanos: self.i64()?,
                        offset_seconds: self.i32()?,
                    })),
                    LOCAL_DATETIME => Some(DocumentItem::Scalar(ScalarValue::LocalDateTime {
                        seconds: self.i64()?,
                        nanos: self.u32()?,
                    })),
                    ZONED_DATETIME => Some(DocumentItem::Scalar(ScalarValue::ZonedDateTime {
                        seconds: self.i64()?,
                        nanos: self.u32()?,
                        timezone: Arc::from(self.string()?),
                    })),
                    DURATION => Some(DocumentItem::Scalar(ScalarValue::Duration {
                        months: self.i64()?,
                        days: self.i64()?,
                        seconds: self.i64()?,
                        nanos: self.i32()?,
                    })),
                    LIST => {
                        let length = self.collection_length(1)?;
                        if length == 0 {
                            Some(DocumentItem::List(Vec::new()))
                        } else {
                            let mut values = Vec::new();
                            values
                                .try_reserve_exact(length)
                                .map_err(|_| document_allocation_failed())?;
                            push_decode_frame(
                                &mut frames,
                                DecodeFrame::List {
                                    remaining: length,
                                    values,
                                },
                            )?;
                            None
                        }
                    }
                    MAP => {
                        let length = self.collection_length(ENCODED_LENGTH_BYTES + 1)?;
                        if length == 0 {
                            Some(DocumentItem::Map(BTreeMap::new()))
                        } else {
                            let (previous_key, pending_key) = self.map_key(None)?;
                            push_decode_frame(
                                &mut frames,
                                DecodeFrame::Map {
                                    remaining: length,
                                    values: BTreeMap::new(),
                                    previous_key,
                                    pending_key: Some(pending_key),
                                },
                            )?;
                            None
                        }
                    }
                    _ => return Err(corrupt("document value tag is invalid")),
                };
                if completed.is_none() {
                    continue;
                }
            }

            let mut item = completed
                .take()
                .ok_or_else(|| corrupt("document decoder lost a completed value"))?;
            loop {
                let Some(frame) = frames.last_mut() else {
                    return Ok(item);
                };
                let frame_complete = match frame {
                    DecodeFrame::List { remaining, values } => {
                        values.push(item);
                        *remaining = remaining
                            .checked_sub(1)
                            .ok_or_else(|| corrupt("document list accounting underflow"))?;
                        *remaining == 0
                    }
                    DecodeFrame::Map {
                        remaining,
                        values,
                        previous_key,
                        pending_key,
                    } => {
                        let key = pending_key
                            .take()
                            .ok_or_else(|| corrupt("document map key is missing"))?;
                        if values.insert(key, item).is_some() {
                            return Err(corrupt("document map keys are not canonical"));
                        }
                        *remaining = remaining
                            .checked_sub(1)
                            .ok_or_else(|| corrupt("document map accounting underflow"))?;
                        if *remaining != 0 {
                            let (next_range, next_key) = self.map_key(Some(*previous_key))?;
                            *previous_key = next_range;
                            *pending_key = Some(next_key);
                        }
                        *remaining == 0
                    }
                };
                if !frame_complete {
                    break;
                }
                item = match frames
                    .pop()
                    .ok_or_else(|| corrupt("document decoder frame disappeared"))?
                {
                    DecodeFrame::List { values, .. } => DocumentItem::List(values),
                    DecodeFrame::Map { values, .. } => DocumentItem::Map(values),
                };
            }
        }
    }

    fn skip_item(&mut self) -> Result<()> {
        let mut frames = Vec::new();
        loop {
            let tag = self.byte()?;
            let mut item_complete = match tag {
                NULL | FALSE | TRUE => true,
                INTEGER | DATE | LOCAL_TIME => {
                    self.take(8)?;
                    true
                }
                FLOAT => {
                    validate_float_bits(self.u64()?)?;
                    true
                }
                STRING => {
                    self.string()?;
                    true
                }
                BYTES => {
                    self.bytes()?;
                    true
                }
                ZONED_TIME | LOCAL_DATETIME => {
                    self.take(12)?;
                    true
                }
                ZONED_DATETIME => {
                    self.take(12)?;
                    self.string()?;
                    true
                }
                DURATION => {
                    self.take(28)?;
                    true
                }
                LIST => {
                    let length = self.collection_length(1)?;
                    if length == 0 {
                        true
                    } else {
                        push_validation_frame(
                            &mut frames,
                            ValidationFrame::List { remaining: length },
                        )?;
                        false
                    }
                }
                MAP => {
                    let length = self.collection_length(ENCODED_LENGTH_BYTES + 1)?;
                    if length == 0 {
                        true
                    } else {
                        let previous_key = self.map_key_range(None)?;
                        push_validation_frame(
                            &mut frames,
                            ValidationFrame::Map {
                                remaining: length,
                                previous_key,
                            },
                        )?;
                        false
                    }
                }
                _ => return Err(corrupt("document value tag is invalid")),
            };

            while item_complete {
                let Some(frame) = frames.last_mut() else {
                    return Ok(());
                };
                item_complete = match frame {
                    ValidationFrame::List { remaining } => {
                        *remaining = remaining
                            .checked_sub(1)
                            .ok_or_else(|| corrupt("document list accounting underflow"))?;
                        *remaining == 0
                    }
                    ValidationFrame::Map {
                        remaining,
                        previous_key,
                    } => {
                        *remaining = remaining
                            .checked_sub(1)
                            .ok_or_else(|| corrupt("document map accounting underflow"))?;
                        if *remaining != 0 {
                            *previous_key = self.map_key_range(Some(*previous_key))?;
                        }
                        *remaining == 0
                    }
                };
                if item_complete {
                    frames.pop();
                }
            }
        }
    }

    fn take(&mut self, length: usize) -> Result<&[u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| corrupt("document offset overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| corrupt("document is truncated"))?;
        self.position = end;
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| corrupt("document integer is truncated"))?,
        ))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| corrupt("document integer is truncated"))?,
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| corrupt("document integer is truncated"))?,
        ))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| corrupt("document integer is truncated"))?,
        ))
    }

    fn encoded_length(&mut self) -> Result<u64> {
        self.u64()
    }

    fn byte_length(&mut self) -> Result<usize> {
        let length = self.encoded_length()?;
        if length > self.remaining_as_u64() {
            return Err(corrupt("document byte sequence is truncated"));
        }
        usize::try_from(length).map_err(|_| corrupt("document length exceeds host address space"))
    }

    fn collection_length(&mut self, minimum_entry_bytes: usize) -> Result<usize> {
        let length = self.encoded_length()?;
        let maximum_from_input = self.remaining() / minimum_entry_bytes;
        if length > usize_as_u64(maximum_from_input) {
            return Err(corrupt(
                "document collection length exceeds the remaining canonical bytes",
            ));
        }
        usize::try_from(length)
            .map_err(|_| corrupt("document collection length exceeds host address space"))
    }

    fn bytes(&mut self) -> Result<&[u8]> {
        let length = self.byte_length()?;
        self.take(length)
    }

    fn string(&mut self) -> Result<&str> {
        std::str::from_utf8(self.bytes()?)
            .map_err(|_| corrupt("document string is not valid UTF-8"))
    }

    fn string_range(&mut self) -> Result<(usize, usize)> {
        let length = self.byte_length()?;
        let start = self.position;
        let value = self.take(length)?;
        std::str::from_utf8(value).map_err(|_| corrupt("document string is not valid UTF-8"))?;
        Ok((start, self.position))
    }

    fn map_key(&mut self, previous: Option<(usize, usize)>) -> Result<((usize, usize), Arc<str>)> {
        let current = self.map_key_range(previous)?;
        let key = std::str::from_utf8(&self.bytes[current.0..current.1])
            .map_err(|_| corrupt("document string is not valid UTF-8"))?;
        Ok((current, Arc::from(key)))
    }

    fn map_key_range(&mut self, previous: Option<(usize, usize)>) -> Result<(usize, usize)> {
        let current = self.string_range()?;
        if previous.is_some_and(|previous| {
            self.bytes[previous.0..previous.1] >= self.bytes[current.0..current.1]
        }) {
            return Err(corrupt("document map keys are not canonical"));
        }
        Ok(current)
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn remaining_as_u64(&self) -> u64 {
        usize_as_u64(self.remaining())
    }
}

fn push_len(output: &mut Vec<u8>, length: usize) -> Result<()> {
    let length = u64::try_from(length).map_err(|_| document_address_space_exhausted())?;
    extend_output(output, &length.to_le_bytes())
}

fn push_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    push_len(output, bytes.len())?;
    extend_output(output, bytes)
}

fn validate_size(bytes: &[u8]) -> Result<()> {
    if bytes.len() < 2 {
        return Err(corrupt("document is truncated before its root tag"));
    }
    Ok(())
}

fn push_byte(output: &mut Vec<u8>, value: u8) -> Result<()> {
    output
        .try_reserve(1)
        .map_err(|_| document_allocation_failed())?;
    output.push(value);
    Ok(())
}

fn extend_output(output: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| document_allocation_failed())?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn reserve_frames<T>(frames: &mut Vec<T>, additional: usize) -> Result<()> {
    frames
        .try_reserve(additional)
        .map_err(|_| document_allocation_failed())
}

fn push_frame(frames: &mut Vec<EncodeFrame>, frame: EncodeFrame) -> Result<()> {
    reserve_frames(frames, 1)?;
    frames.push(frame);
    Ok(())
}

fn push_decode_frame(frames: &mut Vec<DecodeFrame>, frame: DecodeFrame) -> Result<()> {
    reserve_frames(frames, 1)?;
    frames.push(frame);
    Ok(())
}

fn push_validation_frame(frames: &mut Vec<ValidationFrame>, frame: ValidationFrame) -> Result<()> {
    reserve_frames(frames, 1)?;
    frames.push(frame);
    Ok(())
}

fn validate_float_bits(bits: u64) -> Result<()> {
    let value = f64::from_bits(bits);
    if (value.is_nan() && bits != f64::NAN.to_bits()) || (value == 0.0 && bits != 0.0_f64.to_bits())
    {
        return Err(corrupt("document float encoding is not canonical"));
    }
    Ok(())
}

fn usize_as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn document_address_space_exhausted() -> Error {
    Error::new(
        ErrorCode::ResultBudgetExceeded,
        "document length exceeds the canonical u64 address space",
    )
}

fn document_allocation_failed() -> Error {
    Error::new(
        ErrorCode::ResultBudgetExceeded,
        "document bytes or traversal state cannot be allocated in the host address space",
    )
}

fn corrupt(message: &'static str) -> Error {
    Error::new(ErrorCode::CorruptStorage, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_map_round_trip_is_key_ordered() -> Result<()> {
        let mut nested = BTreeMap::new();
        nested.insert(
            Arc::from("z"),
            DocumentItem::Scalar(ScalarValue::Integer(7)),
        );
        nested.insert(
            Arc::from("a"),
            DocumentItem::List(vec![
                DocumentItem::Scalar(ScalarValue::Boolean(true)),
                DocumentItem::Scalar(ScalarValue::Null),
            ]),
        );
        let value = DocumentMap::new(nested.clone())?;
        assert_eq!(value.entries()?, nested);
        let second = DocumentMap::new(nested)?;
        assert_eq!(value.as_bytes(), second.as_bytes());
        Ok(())
    }

    #[test]
    fn canonical_document_round_trips_wide_dates_without_narrowing() -> Result<()> {
        let wide_days = 365_242_499_634_i64;
        let entries = BTreeMap::from([(
            Arc::from("wide"),
            DocumentItem::Scalar(ScalarValue::Date(wide_days)),
        )]);
        let value = DocumentMap::new(entries.clone())?;
        assert_eq!(value.as_bytes().first().copied(), Some(FORMAT_VERSION));
        validate_canonical(value.as_bytes(), DocumentRoot::Map)?;
        assert_eq!(value.entries()?, entries);
        Ok(())
    }

    #[test]
    fn canonical_lengths_use_the_full_u64_field() -> Result<()> {
        let value = DocumentMap::new(BTreeMap::from([(
            Arc::from("key"),
            DocumentItem::List(vec![DocumentItem::Scalar(ScalarValue::Null)]),
        )]))?;
        let bytes = value.as_bytes();
        assert_eq!(bytes[0], FORMAT_VERSION);
        assert_eq!(bytes[1], MAP);
        assert_eq!(&bytes[2..10], &1_u64.to_le_bytes());
        assert_eq!(&bytes[10..18], &3_u64.to_le_bytes());
        assert_eq!(&bytes[22..30], &1_u64.to_le_bytes());
        validate_canonical(bytes, DocumentRoot::Map)?;

        let mut impossible = vec![FORMAT_VERSION, LIST];
        impossible.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(validate_canonical(&impossible, DocumentRoot::List).is_err());
        Ok(())
    }

    #[test]
    fn canonical_document_crosses_the_former_byte_boundary() -> Result<()> {
        let former_boundary = 16 * 1024 * 1024;
        let value = DocumentList::new(vec![DocumentItem::Scalar(ScalarValue::Bytes(Arc::from(
            vec![0xa5; former_boundary + 1],
        )))])?;
        assert!(value.as_bytes().len() > former_boundary);
        validate_canonical(value.as_bytes(), DocumentRoot::List)?;

        let mut items = value.items()?;
        let Some(DocumentItem::Scalar(ScalarValue::Bytes(payload))) = items.pop() else {
            return Err(Error::internal("large canonical byte value changed shape"));
        };
        assert_eq!(payload.len(), former_boundary + 1);
        assert_eq!(payload.first().copied(), Some(0xa5));
        assert_eq!(payload.last().copied(), Some(0xa5));
        Ok(())
    }

    #[test]
    fn canonical_document_crosses_the_former_depth_boundary_iteratively() -> Result<()> {
        const NESTING: usize = 4_096;
        let mut item = DocumentItem::Scalar(ScalarValue::Integer(7));
        for _ in 0..NESTING {
            item = DocumentItem::List(vec![item]);
        }
        let value = DocumentList::new(vec![item])?;
        validate_canonical(value.as_bytes(), DocumentRoot::List)?;

        let mut root = value.items()?;
        let mut item = root
            .pop()
            .ok_or_else(|| Error::internal("deep canonical list is empty"))?;
        for _ in 0..NESTING {
            let DocumentItem::List(mut values) = item else {
                return Err(Error::internal("deep canonical list changed shape"));
            };
            if values.len() != 1 {
                return Err(Error::internal("deep canonical list changed width"));
            }
            item = values
                .pop()
                .ok_or_else(|| Error::internal("deep canonical list child is absent"))?;
        }
        assert_eq!(item, DocumentItem::Scalar(ScalarValue::Integer(7)));
        Ok(())
    }

    #[test]
    fn canonical_document_crosses_the_former_value_count_boundary() -> Result<()> {
        let value_count = 1_000_001;
        let values = std::iter::repeat_with(|| DocumentItem::Scalar(ScalarValue::Null))
            .take(value_count)
            .collect();
        let value = DocumentList::new(values)?;
        validate_canonical(value.as_bytes(), DocumentRoot::List)?;
        let items = value.items()?;
        assert_eq!(items.len(), value_count);
        assert!(matches!(
            items.first(),
            Some(DocumentItem::Scalar(ScalarValue::Null))
        ));
        assert!(matches!(
            items.last(),
            Some(DocumentItem::Scalar(ScalarValue::Null))
        ));
        Ok(())
    }

    #[test]
    fn deserialization_rejects_wrong_root_and_trailing_bytes() -> Result<()> {
        let list = DocumentList::new(Vec::new())?;
        assert!(validate_canonical(list.as_bytes(), DocumentRoot::Map).is_err());
        let mut corrupt = list.as_bytes().to_vec();
        corrupt.push(0);
        assert!(validate_canonical(&corrupt, DocumentRoot::List).is_err());
        Ok(())
    }
}
