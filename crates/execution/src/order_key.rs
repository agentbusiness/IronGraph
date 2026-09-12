//! Prefix-free radix keys for Cypher's deterministic total value order.
//!
//! Zero is reserved for variable-length and sequence terminators. Every payload octet is split
//! into two non-zero radix digits, so bytewise comparison is directly usable by GPU radix kernels
//! without losing embedded NUL bytes or prefix ordering.

use std::cmp::Ordering;

use irongraph_types::{Error, ErrorCode, Result, ScalarValue};

use super::ResultValue;

const MAP: u8 = 1;
const NODE: u8 = 2;
const RELATIONSHIP: u8 = 3;
const LIST: u8 = 4;
const PATH: u8 = 5;
const VECTOR: u8 = 6;
const ZONED_DATETIME: u8 = 7;
const LOCAL_DATETIME: u8 = 8;
const DATE: u8 = 9;
const ZONED_TIME: u8 = 10;
const LOCAL_TIME: u8 = 11;
const DURATION: u8 = 12;
const STRING_OR_BYTES: u8 = 13;
const BOOLEAN: u8 = 14;
const NUMBER: u8 = 15;
const NULL: u8 = 16;

const STRING: u8 = 1;
const BYTES: u8 = 2;

const NEGATIVE_INFINITY: u8 = 1;
const NEGATIVE_FINITE: u8 = 2;
const ZERO: u8 = 3;
const POSITIVE_FINITE: u8 = 4;
const POSITIVE_INFINITY: u8 = 5;
const NAN: u8 = 6;

const MIN_BINARY_EXPONENT: i32 = -1_074;

/// Encodes one value into a prefix-free sequence of GPU-radix digits.
pub fn encode_total_order_key(value: &ResultValue) -> Result<Vec<u8>> {
    let capacity = total_order_key_len(value)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| key_allocation_failed())?;
    append_value(&mut output, value)?;
    debug_assert_eq!(output.len(), capacity);
    Ok(output)
}

/// Appends one encoded key directly to an admitted aggregate column without allocating an
/// intermediate key buffer. The caller can inspect [`total_order_key_len`] before this call to
/// enforce its query budget; allocation remains fallible in case the process allocator refuses
/// the already-admitted growth.
pub fn append_total_order_key(output: &mut Vec<u8>, value: &ResultValue) -> Result<usize> {
    let length = total_order_key_len(value)?;
    output
        .try_reserve_exact(length)
        .map_err(|_| key_allocation_failed())?;
    let start = output.len();
    append_value(output, value)?;
    debug_assert_eq!(output.len() - start, length);
    Ok(length)
}

/// Returns the largest encoded key length in a materialized value column.
pub fn max_total_order_key_len(values: &[ResultValue]) -> Result<usize> {
    values.iter().try_fold(0_usize, |maximum, value| {
        Ok(maximum.max(total_order_key_len(value)?))
    })
}

fn append_value(output: &mut Vec<u8>, value: &ResultValue) -> Result<()> {
    if let ResultValue::Scalar(value) = value
        && value.is_document()
    {
        return append_value(output, &ResultValue::from_property(value.clone())?);
    }
    match value {
        ResultValue::Map(values) => {
            output.push(MAP);
            append_u64(output, length_u64(values.len())?);
            for name in values.keys() {
                append_bytes(output, name.as_bytes());
            }
            for value in values.values() {
                append_value(output, value)?;
            }
        }
        ResultValue::Node(node) => {
            output.push(NODE);
            append_u64(output, node.id.0);
        }
        ResultValue::Relationship(relationship) => {
            output.push(RELATIONSHIP);
            append_u64(output, relationship.id.0);
        }
        ResultValue::List(values) => {
            output.push(LIST);
            for value in values {
                append_value(output, value)?;
            }
            output.push(0);
        }
        ResultValue::Path {
            nodes,
            relationships,
        } => {
            validate_path(nodes.len(), relationships.len())?;
            output.push(PATH);
            for index in 0..nodes.len().saturating_add(relationships.len()) {
                if index % 2 == 0 {
                    append_u64(output, nodes[index / 2].id.0);
                } else {
                    append_u64(output, relationships[index / 2].id.0);
                }
            }
            output.push(0);
        }
        ResultValue::Vector(values) => {
            output.push(VECTOR);
            for value in values {
                append_u32(output, sortable_f32_bits(*value));
            }
            output.push(0);
        }
        ResultValue::Scalar(value) => append_scalar(output, value)?,
    }
    Ok(())
}

fn append_scalar(output: &mut Vec<u8>, value: &ScalarValue) -> Result<()> {
    match value {
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            output.push(ZONED_DATETIME);
            append_i64(output, *seconds);
            append_u32(output, *nanos);
            append_bytes(output, timezone.as_bytes());
        }
        ScalarValue::LocalDateTime { seconds, nanos } => {
            output.push(LOCAL_DATETIME);
            append_i64(output, *seconds);
            append_u32(output, *nanos);
        }
        ScalarValue::Date(value) => {
            output.push(DATE);
            append_i64(output, *value);
        }
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => {
            output.push(ZONED_TIME);
            append_i64(output, normalized_zoned_time(*nanos, *offset_seconds));
            append_i32(output, *offset_seconds);
        }
        ScalarValue::LocalTime(value) => {
            output.push(LOCAL_TIME);
            append_i64(output, *value);
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => {
            output.push(DURATION);
            append_i128(
                output,
                normalized_duration(*months, *days, *seconds, *nanos),
            );
        }
        ScalarValue::String(value) => {
            output.extend_from_slice(&[STRING_OR_BYTES, STRING]);
            append_bytes(output, value.as_bytes());
        }
        ScalarValue::Bytes(value) => {
            output.extend_from_slice(&[STRING_OR_BYTES, BYTES]);
            append_bytes(output, value);
        }
        ScalarValue::Boolean(value) => {
            output.extend_from_slice(&[BOOLEAN, u8::from(*value) + 1]);
        }
        ScalarValue::Integer(value) => {
            output.push(NUMBER);
            append_number(output, ordered_integer(*value));
        }
        ScalarValue::Float(value) => {
            output.push(NUMBER);
            append_number(output, ordered_float(value.into_inner()));
        }
        ScalarValue::Null => output.push(NULL),
        ScalarValue::List(_) | ScalarValue::Map(_) => {
            return Err(unmaterialized_document());
        }
    }
    Ok(())
}

pub fn total_order_key_len(value: &ResultValue) -> Result<usize> {
    if let ResultValue::Scalar(value) = value
        && value.is_document()
    {
        return total_order_key_len(&ResultValue::from_property(value.clone())?);
    }
    match value {
        ResultValue::Map(values) => {
            let keys = values.keys().try_fold(0_usize, |length, name| {
                checked_add(length, encoded_bytes_len(name.len())?)
            })?;
            let nested = values.values().try_fold(0_usize, |length, value| {
                checked_add(length, total_order_key_len(value)?)
            })?;
            checked_add(17, checked_add(keys, nested)?)
        }
        ResultValue::Node(_) | ResultValue::Relationship(_) => Ok(17),
        ResultValue::List(values) => values.iter().try_fold(2_usize, |length, value| {
            checked_add(length, total_order_key_len(value)?)
        }),
        ResultValue::Path {
            nodes,
            relationships,
        } => {
            validate_path(nodes.len(), relationships.len())?;
            checked_add(
                2,
                nodes
                    .len()
                    .checked_add(relationships.len())
                    .and_then(|length| length.checked_mul(16))
                    .ok_or_else(key_too_large)?,
            )
        }
        ResultValue::Vector(values) => {
            checked_add(2, values.len().checked_mul(8).ok_or_else(key_too_large)?)
        }
        ResultValue::Scalar(value) => scalar_key_len(value),
    }
}

fn scalar_key_len(value: &ScalarValue) -> Result<usize> {
    match value {
        ScalarValue::ZonedDateTime { timezone, .. } => {
            checked_add(25, encoded_bytes_len(timezone.len())?)
        }
        ScalarValue::LocalDateTime { .. } | ScalarValue::ZonedTime { .. } => Ok(25),
        ScalarValue::Date(_) => Ok(17),
        ScalarValue::LocalTime(_) => Ok(17),
        ScalarValue::Duration { .. } => Ok(33),
        ScalarValue::String(value) => checked_add(2, encoded_bytes_len(value.len())?),
        ScalarValue::Bytes(value) => checked_add(2, encoded_bytes_len(value.len())?),
        ScalarValue::Boolean(_) => Ok(2),
        ScalarValue::Integer(_) | ScalarValue::Float(_) => Ok(22),
        ScalarValue::Null => Ok(1),
        ScalarValue::List(_) | ScalarValue::Map(_) => Err(unmaterialized_document()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OrderedNumber {
    category: u8,
    exponent: u16,
    significand: u64,
}

pub fn compare_i64_f64(integer: i64, float: f64) -> Ordering {
    ordered_integer(integer).cmp(&ordered_float(float))
}

/// Cypher orders zoned times by their UTC-equivalent time-of-day before using the offset as a
/// deterministic tie-breaker. The accepted temporal range cannot overflow this subtraction;
/// saturation also keeps ordering total for a corrupt/unvalidated in-memory value.
pub fn normalized_zoned_time(nanos: i64, offset_seconds: i32) -> i64 {
    nanos.saturating_sub(i64::from(offset_seconds).saturating_mul(1_000_000_000))
}

fn ordered_integer(value: i64) -> OrderedNumber {
    if value == 0 {
        return OrderedNumber::zero();
    }
    let magnitude = value.unsigned_abs();
    let width = 64_u32 - magnitude.leading_zeros();
    ordered_finite(
        value.is_negative(),
        width as i32 - 1,
        magnitude << (64 - width),
    )
}

fn ordered_float(value: f64) -> OrderedNumber {
    if value.is_nan() {
        return OrderedNumber::category(NAN);
    }
    if value == f64::NEG_INFINITY {
        return OrderedNumber::category(NEGATIVE_INFINITY);
    }
    if value == f64::INFINITY {
        return OrderedNumber::category(POSITIVE_INFINITY);
    }
    if value == 0.0 {
        return OrderedNumber::zero();
    }

    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let stored_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (exponent, significand) = if stored_exponent == 0 {
        let width = 64_u32 - fraction.leading_zeros();
        (
            MIN_BINARY_EXPONENT + width as i32 - 1,
            fraction << (64 - width),
        )
    } else {
        (stored_exponent - 1_023, ((1_u64 << 52) | fraction) << 11)
    };
    ordered_finite(negative, exponent, significand)
}

fn ordered_finite(negative: bool, exponent: i32, significand: u64) -> OrderedNumber {
    debug_assert!((MIN_BINARY_EXPONENT..=1_023).contains(&exponent));
    let biased = (exponent - MIN_BINARY_EXPONENT) as u16;
    if negative {
        OrderedNumber {
            category: NEGATIVE_FINITE,
            exponent: !biased,
            significand: !significand,
        }
    } else {
        OrderedNumber {
            category: POSITIVE_FINITE,
            exponent: biased,
            significand,
        }
    }
}

impl OrderedNumber {
    const fn category(category: u8) -> Self {
        Self {
            category,
            exponent: 0,
            significand: 0,
        }
    }

    const fn zero() -> Self {
        Self::category(ZERO)
    }
}

fn append_number(output: &mut Vec<u8>, value: OrderedNumber) {
    output.push(value.category);
    append_u16(output, value.exponent);
    append_u64(output, value.significand);
}

pub(super) fn normalized_duration(months: i64, days: i64, seconds: i64, nanos: i32) -> i128 {
    const MONTH_NANOS: i128 = 2_629_746_000_000_000;
    i128::from(months)
        .saturating_mul(MONTH_NANOS)
        .saturating_add(i128::from(days).saturating_mul(86_400_000_000_000))
        .saturating_add(i128::from(seconds).saturating_mul(1_000_000_000))
        .saturating_add(i128::from(nanos))
}

fn sortable_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    }
}

fn append_bytes(output: &mut Vec<u8>, bytes: &[u8]) {
    for byte in bytes {
        append_octet(output, *byte);
    }
    output.push(0);
}

fn append_i128(output: &mut Vec<u8>, value: i128) {
    append_octets(output, &((value as u128) ^ (1_u128 << 127)).to_be_bytes());
}

fn append_i64(output: &mut Vec<u8>, value: i64) {
    append_u64(output, (value as u64) ^ (1_u64 << 63));
}

fn append_i32(output: &mut Vec<u8>, value: i32) {
    append_u32(output, (value as u32) ^ (1_u32 << 31));
}

fn append_u64(output: &mut Vec<u8>, value: u64) {
    append_octets(output, &value.to_be_bytes());
}

fn append_u32(output: &mut Vec<u8>, value: u32) {
    append_octets(output, &value.to_be_bytes());
}

fn append_u16(output: &mut Vec<u8>, value: u16) {
    append_octets(output, &value.to_be_bytes());
}

fn append_octets(output: &mut Vec<u8>, bytes: &[u8]) {
    for byte in bytes {
        append_octet(output, *byte);
    }
}

fn append_octet(output: &mut Vec<u8>, byte: u8) {
    output.extend_from_slice(&[(byte >> 4) + 1, (byte & 0x0f) + 1]);
}

fn encoded_bytes_len(length: usize) -> Result<usize> {
    length
        .checked_mul(2)
        .and_then(|length| length.checked_add(1))
        .ok_or_else(key_too_large)
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or_else(key_too_large)
}

fn length_u64(length: usize) -> Result<u64> {
    u64::try_from(length).map_err(|_| key_too_large())
}

fn validate_path(nodes: usize, relationships: usize) -> Result<()> {
    if nodes != relationships.saturating_add(1) {
        return Err(Error::new(
            ErrorCode::QueryType,
            "path result must contain exactly one more node than relationship",
        ));
    }
    Ok(())
}

fn key_too_large() -> Error {
    Error::new(
        ErrorCode::ResultBudgetExceeded,
        "total-order key length exceeds the addressable result budget",
    )
}

fn key_allocation_failed() -> Error {
    Error::new(
        ErrorCode::ResultBudgetExceeded,
        "total-order key bytes cannot be admitted within the result memory budget",
    )
}

fn unmaterialized_document() -> Error {
    Error::new(
        ErrorCode::CorruptStorage,
        "document scalar reached total-order encoding without materialization",
    )
}

#[cfg(test)]
mod tests {
    use std::{cmp::Ordering, collections::BTreeMap, sync::Arc};

    use ordered_float::OrderedFloat;
    use proptest::{collection, prelude::*};

    use irongraph_types::{EdgeId, Layer, NodeId, ScalarValue};

    use super::{encode_total_order_key, max_total_order_key_len};
    use crate::{ResultEdge, ResultNode, ResultValue, total_compare};

    fn node(id: u64) -> ResultNode {
        ResultNode {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 0,
            labels: Vec::new(),
            properties: BTreeMap::new(),
        }
    }

    fn edge(id: u64) -> ResultEdge {
        ResultEdge {
            id: EdgeId(id),
            source: NodeId(0),
            target: NodeId(0),
            relationship_type: String::new(),
            layer: Layer::Observed,
            revision: 0,
            properties: BTreeMap::new(),
        }
    }

    fn leaf_value() -> impl Strategy<Value = ResultValue> {
        prop_oneof![
            Just(ResultValue::Scalar(ScalarValue::Null)),
            any::<bool>().prop_map(|value| ResultValue::Scalar(ScalarValue::Boolean(value))),
            any::<i64>().prop_map(|value| ResultValue::Scalar(ScalarValue::Integer(value))),
            any::<u64>().prop_map(|bits| ResultValue::Scalar(ScalarValue::Float(OrderedFloat(
                f64::from_bits(bits)
            )))),
            any::<String>()
                .prop_map(|value| ResultValue::Scalar(ScalarValue::String(value.into()))),
            collection::vec(any::<u8>(), 0..24)
                .prop_map(|value| ResultValue::Scalar(ScalarValue::Bytes(value.into()))),
            any::<i64>().prop_map(|value| ResultValue::Scalar(ScalarValue::Date(value))),
            any::<i64>().prop_map(|value| ResultValue::Scalar(ScalarValue::LocalTime(value))),
            (any::<i64>(), any::<i32>()).prop_map(|(nanos, offset_seconds)| {
                ResultValue::Scalar(ScalarValue::ZonedTime {
                    nanos,
                    offset_seconds,
                })
            }),
            (any::<i64>(), any::<u32>()).prop_map(|(seconds, nanos)| {
                ResultValue::Scalar(ScalarValue::LocalDateTime { seconds, nanos })
            }),
            (any::<i64>(), any::<u32>(), any::<String>()).prop_map(|(seconds, nanos, timezone)| {
                ResultValue::Scalar(ScalarValue::ZonedDateTime {
                    seconds,
                    nanos,
                    timezone: timezone.into(),
                })
            },),
            (any::<i64>(), any::<i64>(), any::<i64>(), any::<i32>()).prop_map(
                |(months, days, seconds, nanos)| {
                    ResultValue::Scalar(ScalarValue::Duration {
                        months,
                        days,
                        seconds,
                        nanos,
                    })
                },
            ),
            any::<u64>().prop_map(|id| ResultValue::Node(node(id))),
            any::<u64>().prop_map(|id| ResultValue::Relationship(edge(id))),
            collection::vec(any::<u32>(), 0..12).prop_map(|bits| ResultValue::Vector(
                bits.into_iter().map(f32::from_bits).collect()
            )),
            (
                collection::vec(any::<u64>(), 1..8),
                collection::vec(any::<u64>(), 0..7)
            )
                .prop_filter_map("well-formed path", |(nodes, relationships)| {
                    (nodes.len() == relationships.len() + 1).then(|| ResultValue::Path {
                        nodes: nodes.into_iter().map(node).collect(),
                        relationships: relationships.into_iter().map(edge).collect(),
                    })
                }),
        ]
    }

    fn any_value() -> impl Strategy<Value = ResultValue> {
        leaf_value().prop_recursive(3, 48, 6, |inner| {
            prop_oneof![
                collection::vec(inner.clone(), 0..8).prop_map(ResultValue::List),
                collection::btree_map(any::<String>(), inner, 0..8).prop_map(ResultValue::Map),
            ]
        })
    }

    fn ordering_sign(ordering: Ordering) -> i8 {
        match ordering {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4_096))]

        #[test]
        fn encoded_key_order_matches_total_compare(left in any_value(), right in any_value()) {
            let left_key = encode_total_order_key(&left)
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            let right_key = encode_total_order_key(&right)
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            prop_assert_eq!(
                ordering_sign(left_key.cmp(&right_key)),
                ordering_sign(total_compare(&left, &right)),
            );
            prop_assert!(left_key.iter().all(|digit| *digit <= 16));
            prop_assert!(right_key.iter().all(|digit| *digit <= 16));
            if total_compare(&left, &right) != Ordering::Equal {
                prop_assert!(!left_key.starts_with(&right_key));
                prop_assert!(!right_key.starts_with(&left_key));
            }
        }

        #[test]
        fn mixed_integer_float_keys_are_exact(integer in any::<i64>(), bits in any::<u64>()) {
            let integer = ResultValue::Scalar(ScalarValue::Integer(integer));
            let float = ResultValue::Scalar(ScalarValue::Float(OrderedFloat(f64::from_bits(bits))));
            let integer_key = encode_total_order_key(&integer)
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            let float_key = encode_total_order_key(&float)
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            prop_assert_eq!(
                ordering_sign(integer_key.cmp(&float_key)),
                ordering_sign(total_compare(&integer, &float)),
            );
        }

        #[test]
        fn float_keys_match_ordered_float_for_every_bit_pattern(
            left_bits in any::<u64>(),
            right_bits in any::<u64>(),
        ) {
            let left = ResultValue::Scalar(ScalarValue::Float(OrderedFloat(
                f64::from_bits(left_bits),
            )));
            let right = ResultValue::Scalar(ScalarValue::Float(OrderedFloat(
                f64::from_bits(right_bits),
            )));
            let left_key = encode_total_order_key(&left)
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            let right_key = encode_total_order_key(&right)
                .map_err(|error| TestCaseError::fail(error.to_string()))?;
            prop_assert_eq!(
                ordering_sign(left_key.cmp(&right_key)),
                ordering_sign(total_compare(&left, &right)),
            );
        }
    }

    #[test]
    fn mixed_numeric_boundaries_and_nonfinite_values_are_exact() -> irongraph_types::Result<()> {
        let values = [
            f64::NEG_INFINITY,
            -9_223_372_036_854_775_808.0,
            -9_007_199_254_740_992.0,
            -0.0,
            0.0,
            9_007_199_254_740_992.0,
            9_223_372_036_854_775_808.0,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0xfff0_0000_0000_0001),
        ];
        let mut mixed = values
            .into_iter()
            .map(|value| ResultValue::Scalar(ScalarValue::Float(OrderedFloat(value))))
            .chain(
                [
                    i64::MIN,
                    -9_007_199_254_740_993,
                    -1,
                    0,
                    1,
                    9_007_199_254_740_993,
                    i64::MAX,
                ]
                .into_iter()
                .map(|value| ResultValue::Scalar(ScalarValue::Integer(value))),
            )
            .collect::<Vec<_>>();
        mixed.sort_by(total_compare);
        for pair in mixed.windows(2) {
            assert!(encode_total_order_key(&pair[0])? <= encode_total_order_key(&pair[1])?);
        }
        let zero = ResultValue::Scalar(ScalarValue::Integer(0));
        for value in [-0.0, 0.0] {
            assert_eq!(
                encode_total_order_key(&zero)?,
                encode_total_order_key(&ResultValue::Scalar(ScalarValue::Float(OrderedFloat(
                    value
                ))))?
            );
        }
        let integer = ResultValue::Scalar(ScalarValue::Integer(9_007_199_254_740_993));
        let rounded =
            ResultValue::Scalar(ScalarValue::Float(OrderedFloat(9_007_199_254_740_992.0)));
        assert!(encode_total_order_key(&integer)? > encode_total_order_key(&rounded)?);
        Ok(())
    }

    #[test]
    fn variable_bytes_are_prefix_safe_and_preserve_nul_and_ff() -> irongraph_types::Result<()> {
        let values = [
            Vec::new(),
            vec![0],
            vec![0, 0],
            vec![0, 0xff],
            vec![1],
            vec![0xff],
            vec![0xff, 0],
        ];
        for pair in values.windows(2) {
            let left = ResultValue::Scalar(ScalarValue::Bytes(Arc::from(pair[0].as_slice())));
            let right = ResultValue::Scalar(ScalarValue::Bytes(Arc::from(pair[1].as_slice())));
            assert_eq!(
                encode_total_order_key(&left)?.cmp(&encode_total_order_key(&right)?),
                pair[0].cmp(&pair[1]),
            );
        }
        let nul = ResultValue::Scalar(ScalarValue::String("a\0b".into()));
        let longer = ResultValue::Scalar(ScalarValue::String("a\0b\0".into()));
        assert!(encode_total_order_key(&nul)? < encode_total_order_key(&longer)?);
        Ok(())
    }

    #[test]
    fn string_and_bytes_have_an_explicit_transitive_suborder() -> irongraph_types::Result<()> {
        let string_a = ResultValue::Scalar(ScalarValue::String("a".into()));
        let string_z = ResultValue::Scalar(ScalarValue::String("z".into()));
        let bytes = ResultValue::Scalar(ScalarValue::Bytes(Arc::from([0_u8, 0xff])));
        assert_eq!(total_compare(&string_a, &string_z), Ordering::Less);
        assert_eq!(total_compare(&string_z, &bytes), Ordering::Less);
        assert_eq!(total_compare(&string_a, &bytes), Ordering::Less);
        assert!(encode_total_order_key(&string_z)? < encode_total_order_key(&bytes)?);
        Ok(())
    }

    #[test]
    fn date_key_length_covers_the_full_i64_payload() -> irongraph_types::Result<()> {
        for days in [i64::MIN, -1, 0, 1, i64::MAX] {
            let key = encode_total_order_key(&ResultValue::Scalar(ScalarValue::Date(days)))?;
            assert_eq!(key.len(), 17);
        }
        Ok(())
    }

    #[test]
    fn recursive_length_semantics_and_maximum_are_exact() -> irongraph_types::Result<()> {
        let prefix = ResultValue::List(vec![ResultValue::Scalar(ScalarValue::Integer(1))]);
        let longer = ResultValue::List(vec![
            ResultValue::Scalar(ScalarValue::Integer(1)),
            ResultValue::Scalar(ScalarValue::Null),
        ]);
        assert!(encode_total_order_key(&prefix)? < encode_total_order_key(&longer)?);

        let mut short_map = BTreeMap::new();
        short_map.insert("z".to_owned(), ResultValue::Scalar(ScalarValue::Integer(1)));
        let mut long_map = short_map.clone();
        long_map.insert("a".to_owned(), ResultValue::Scalar(ScalarValue::Integer(0)));
        let short_map = ResultValue::Map(short_map);
        let long_map = ResultValue::Map(long_map);
        assert!(encode_total_order_key(&short_map)? < encode_total_order_key(&long_map)?);

        let values = vec![prefix, longer, short_map, long_map];
        assert_eq!(
            max_total_order_key_len(&values)?,
            values
                .iter()
                .map(encode_total_order_key)
                .collect::<irongraph_types::Result<Vec<_>>>()?
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0),
        );
        Ok(())
    }
}
