//! Backend-neutral total ordering for typed execution values.

use std::{cmp::Ordering, collections::BTreeMap};

use irongraph_types::{Result, ScalarValue};

use crate::{ResultEdge, ResultNode, ResultValue};

const NEGATIVE_INFINITY: u8 = 1;
const NEGATIVE_FINITE: u8 = 2;
const ZERO: u8 = 3;
const POSITIVE_FINITE: u8 = 4;
const POSITIVE_INFINITY: u8 = 5;
const NAN: u8 = 6;
const MIN_BINARY_EXPONENT: i32 = -1_074;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OrderedNumber {
    category: u8,
    exponent: u16,
    significand: u64,
}

pub(super) fn compare_i64_f64(integer: i64, float: f64) -> Ordering {
    ordered_integer(integer).cmp(&ordered_float(float))
}

/// Cypher orders zoned times by their UTC-equivalent time-of-day before using the offset as a
/// deterministic tie-breaker. The accepted temporal range cannot overflow this subtraction;
/// saturation also keeps ordering total for a corrupt/unvalidated in-memory value.
pub(super) fn normalized_zoned_time(nanos: i64, offset_seconds: i32) -> i64 {
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
pub(super) fn normalized_duration(months: i64, days: i64, seconds: i64, nanos: i32) -> i128 {
    const MONTH_NANOS: i128 = 2_629_746_000_000_000;
    i128::from(months)
        .saturating_mul(MONTH_NANOS)
        .saturating_add(i128::from(days).saturating_mul(86_400_000_000_000))
        .saturating_add(i128::from(seconds).saturating_mul(1_000_000_000))
        .saturating_add(i128::from(nanos))
}
fn compare_same_scalar(left: &ScalarValue, right: &ScalarValue) -> Result<Option<Ordering>> {
    let result = match (left, right) {
        (ScalarValue::Boolean(left), ScalarValue::Boolean(right)) => Some(left.cmp(right)),
        (ScalarValue::Integer(left), ScalarValue::Integer(right)) => Some(left.cmp(right)),
        (ScalarValue::Float(left), ScalarValue::Float(right)) => Some(left.cmp(right)),
        (ScalarValue::String(left), ScalarValue::String(right)) => Some(left.cmp(right)),
        (ScalarValue::Date(left), ScalarValue::Date(right)) => Some(left.cmp(right)),
        (ScalarValue::LocalTime(left), ScalarValue::LocalTime(right)) => Some(left.cmp(right)),
        (
            ScalarValue::ZonedTime {
                nanos: left_nanos,
                offset_seconds: left_offset,
            },
            ScalarValue::ZonedTime {
                nanos: right_nanos,
                offset_seconds: right_offset,
            },
        ) => Some(
            (
                normalized_zoned_time(*left_nanos, *left_offset),
                *left_offset,
            )
                .cmp(&(
                    normalized_zoned_time(*right_nanos, *right_offset),
                    *right_offset,
                )),
        ),
        (
            ScalarValue::LocalDateTime {
                seconds: left_seconds,
                nanos: left_nanos,
            },
            ScalarValue::LocalDateTime {
                seconds: right_seconds,
                nanos: right_nanos,
            },
        ) => Some((*left_seconds, *left_nanos).cmp(&(*right_seconds, *right_nanos))),
        (
            ScalarValue::ZonedDateTime {
                seconds: left_seconds,
                nanos: left_nanos,
                timezone: left_zone,
            },
            ScalarValue::ZonedDateTime {
                seconds: right_seconds,
                nanos: right_nanos,
                timezone: right_zone,
            },
        ) => Some((*left_seconds, *left_nanos, left_zone).cmp(&(
            *right_seconds,
            *right_nanos,
            right_zone,
        ))),
        (ScalarValue::Duration { .. }, ScalarValue::Duration { .. }) => None,
        _ => return Ok(None),
    };
    Ok(result)
}

/// Total ordering used only by `ORDER BY`, grouping, min/max, and deterministic tie-breaking.
pub fn total_compare(left: &ResultValue, right: &ResultValue) -> Ordering {
    if let ResultValue::Scalar(value) = left
        && value.is_document()
        && let Ok(materialized) = ResultValue::from_property(value.clone())
    {
        return total_compare(&materialized, right);
    }
    if let ResultValue::Scalar(value) = right
        && value.is_document()
        && let Ok(materialized) = ResultValue::from_property(value.clone())
    {
        return total_compare(left, &materialized);
    }
    let left_rank = value_rank(left);
    let right_rank = value_rank(right);
    if left_rank != right_rank {
        return left_rank.cmp(&right_rank);
    }
    match (left, right) {
        (ResultValue::Map(left), ResultValue::Map(right)) => compare_maps(left, right),
        (ResultValue::Node(left), ResultValue::Node(right)) => left.id.cmp(&right.id),
        (ResultValue::Relationship(left), ResultValue::Relationship(right)) => {
            left.id.cmp(&right.id)
        }
        (ResultValue::List(left), ResultValue::List(right)) => compare_sequences(left, right),
        (
            ResultValue::Path {
                nodes: left_nodes,
                relationships: left_relationships,
            },
            ResultValue::Path {
                nodes: right_nodes,
                relationships: right_relationships,
            },
        ) => compare_path_ids(
            left_nodes,
            left_relationships,
            right_nodes,
            right_relationships,
        ),
        (ResultValue::Vector(left), ResultValue::Vector(right)) => left
            .iter()
            .zip(right)
            .map(|(left, right)| left.total_cmp(right))
            .find(|order| *order != Ordering::Equal)
            .unwrap_or_else(|| left.len().cmp(&right.len())),
        (ResultValue::Scalar(left), ResultValue::Scalar(right)) => {
            total_compare_scalar(left, right)
        }
        _ => Ordering::Equal,
    }
}

fn compare_sequences(left: &[ResultValue], right: &[ResultValue]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| total_compare(left, right))
        .find(|order| *order != Ordering::Equal)
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}

fn compare_maps(
    left: &BTreeMap<String, ResultValue>,
    right: &BTreeMap<String, ResultValue>,
) -> Ordering {
    left.len()
        .cmp(&right.len())
        .then_with(|| left.keys().cmp(right.keys()))
        .then_with(|| {
            left.values()
                .zip(right.values())
                .map(|(left, right)| total_compare(left, right))
                .find(|order| *order != Ordering::Equal)
                .unwrap_or(Ordering::Equal)
        })
}

fn compare_path_ids(
    left_nodes: &[ResultNode],
    left_relationships: &[ResultEdge],
    right_nodes: &[ResultNode],
    right_relationships: &[ResultEdge],
) -> Ordering {
    let left_len = left_nodes.len().saturating_add(left_relationships.len());
    let right_len = right_nodes.len().saturating_add(right_relationships.len());
    for index in 0..left_len.min(right_len) {
        let order = if index % 2 == 0 {
            left_nodes[index / 2].id.cmp(&right_nodes[index / 2].id)
        } else {
            left_relationships[index / 2]
                .id
                .cmp(&right_relationships[index / 2].id)
        };
        if order != Ordering::Equal {
            return order;
        }
    }
    left_len.cmp(&right_len)
}

fn total_compare_scalar(left: &ScalarValue, right: &ScalarValue) -> Ordering {
    if let (ScalarValue::Integer(left), ScalarValue::Float(right)) = (left, right) {
        return compare_i64_f64(*left, right.into_inner());
    }
    if let (ScalarValue::Float(left), ScalarValue::Integer(right)) = (left, right) {
        return compare_i64_f64(*right, left.into_inner()).reverse();
    }
    match (left, right) {
        (ScalarValue::String(_), ScalarValue::Bytes(_)) => return Ordering::Less,
        (ScalarValue::Bytes(_), ScalarValue::String(_)) => return Ordering::Greater,
        (ScalarValue::Bytes(left), ScalarValue::Bytes(right)) => return left.cmp(right),
        _ => {}
    }
    let left_rank = scalar_rank(left);
    let right_rank = scalar_rank(right);
    if left_rank != right_rank {
        return left_rank.cmp(&right_rank);
    }
    match compare_same_scalar(left, right) {
        Ok(Some(ordering)) => ordering,
        Ok(None) => compare_duration(left, right),
        Err(_) => Ordering::Equal,
    }
}

fn compare_duration(left: &ScalarValue, right: &ScalarValue) -> Ordering {
    let ScalarValue::Duration {
        months: left_months,
        days: left_days,
        seconds: left_seconds,
        nanos: left_nanos,
    } = left
    else {
        return Ordering::Equal;
    };
    let ScalarValue::Duration {
        months: right_months,
        days: right_days,
        seconds: right_seconds,
        nanos: right_nanos,
    } = right
    else {
        return Ordering::Equal;
    };
    // Cypher's total sort order treats one month as 30.436875 days. Integer nanoseconds avoid
    // platform-dependent floating-point ordering for the supported i64 fields.
    normalized_duration(*left_months, *left_days, *left_seconds, *left_nanos).cmp(
        &normalized_duration(*right_months, *right_days, *right_seconds, *right_nanos),
    )
}

const fn value_rank(value: &ResultValue) -> u8 {
    match value {
        ResultValue::Map(_) => 0,
        ResultValue::Node(_) => 1,
        ResultValue::Relationship(_) => 2,
        ResultValue::List(_) => 3,
        ResultValue::Path { .. } => 4,
        ResultValue::Vector(_) => 5,
        ResultValue::Scalar(value) => 6_u8.saturating_add(scalar_rank(value)),
    }
}

// Temporal values precede strings, booleans, numbers; null is last.
const fn scalar_rank(value: &ScalarValue) -> u8 {
    match value {
        ScalarValue::ZonedDateTime { .. } => 0,
        ScalarValue::LocalDateTime { .. } => 1,
        ScalarValue::Date(_) => 2,
        ScalarValue::ZonedTime { .. } => 3,
        ScalarValue::LocalTime(_) => 4,
        ScalarValue::Duration { .. } => 5,
        ScalarValue::String(_) | ScalarValue::Bytes(_) => 6,
        ScalarValue::Boolean(_) => 7,
        ScalarValue::Integer(_) | ScalarValue::Float(_) => 8,
        ScalarValue::Null => 9,
        ScalarValue::List(_) => 10,
        ScalarValue::Map(_) => 11,
    }
}
