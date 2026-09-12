//! Cypher value equality, three-valued predicates, and total result ordering.

use std::{cmp::Ordering, collections::BTreeMap};

pub use crate::execution::total_compare;
use crate::{Error, ErrorCode, Result, ScalarValue};

use super::{
    BinaryOperator, ResultEdge, ResultNode, ResultValue,
    order_key::{compare_i64_f64, normalized_zoned_time},
};

fn vector_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

const fn bool_truth(value: bool) -> Truth {
    if value { Truth::True } else { Truth::False }
}

const fn scalar_family(value: &ScalarValue) -> u8 {
    match value {
        ScalarValue::Integer(_) | ScalarValue::Float(_) => 0,
        ScalarValue::Boolean(_) => 1,
        ScalarValue::String(_) => 2,
        ScalarValue::Bytes(_) => 3,
        ScalarValue::Date(_) => 4,
        ScalarValue::LocalTime(_) => 5,
        ScalarValue::ZonedTime { .. } => 6,
        ScalarValue::LocalDateTime { .. } => 7,
        ScalarValue::ZonedDateTime { .. } => 8,
        ScalarValue::Duration { .. } => 9,
        ScalarValue::Null => 10,
        ScalarValue::List(_) => 11,
        ScalarValue::Map(_) => 12,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Truth {
    False,
    True,
    Unknown,
}

impl Truth {
    pub const fn into_value(self) -> ResultValue {
        match self {
            Self::False => ResultValue::Scalar(ScalarValue::Boolean(false)),
            Self::True => ResultValue::Scalar(ScalarValue::Boolean(true)),
            Self::Unknown => ResultValue::Scalar(ScalarValue::Null),
        }
    }

    pub const fn not(self) -> Self {
        match self {
            Self::False => Self::True,
            Self::True => Self::False,
            Self::Unknown => Self::Unknown,
        }
    }

    pub const fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::False, _) | (_, Self::False) => Self::False,
            (Self::True, Self::True) => Self::True,
            _ => Self::Unknown,
        }
    }

    pub const fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, _) | (_, Self::True) => Self::True,
            (Self::False, Self::False) => Self::False,
            _ => Self::Unknown,
        }
    }

    pub const fn xor(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::False, Self::False) | (Self::True, Self::True) => Self::False,
            _ => Self::True,
        }
    }
}

pub fn truth(value: &ResultValue) -> Result<Truth> {
    match value {
        ResultValue::Scalar(ScalarValue::Boolean(false)) => Ok(Truth::False),
        ResultValue::Scalar(ScalarValue::Boolean(true)) => Ok(Truth::True),
        ResultValue::Scalar(ScalarValue::Null) => Ok(Truth::Unknown),
        _ => Err(Error::new(
            ErrorCode::QueryType,
            "predicate expression must be BOOLEAN or NULL",
        )),
    }
}

pub const fn is_null(value: &ResultValue) -> bool {
    matches!(value, ResultValue::Scalar(ScalarValue::Null))
}

/// Cypher equality. `Unknown` is returned when equality depends on a null value.
pub fn equal(left: &ResultValue, right: &ResultValue) -> Result<Truth> {
    if let ResultValue::Scalar(value) = left
        && value.is_document()
    {
        let materialized = ResultValue::from_property(value.clone())?;
        return equal(&materialized, right);
    }
    if let ResultValue::Scalar(value) = right
        && value.is_document()
    {
        let materialized = ResultValue::from_property(value.clone())?;
        return equal(left, &materialized);
    }
    if is_null(left) || is_null(right) {
        return Ok(Truth::Unknown);
    }
    match (left, right) {
        (
            ResultValue::Scalar(ScalarValue::Integer(left)),
            ResultValue::Scalar(ScalarValue::Float(right)),
        ) => Ok(bool_truth(
            !right.is_nan() && compare_i64_f64(*left, right.into_inner()).is_eq(),
        )),
        (
            ResultValue::Scalar(ScalarValue::Float(left)),
            ResultValue::Scalar(ScalarValue::Integer(right)),
        ) => Ok(bool_truth(
            !left.is_nan() && compare_i64_f64(*right, left.into_inner()).is_eq(),
        )),
        (ResultValue::Scalar(left), ResultValue::Scalar(right)) => {
            if scalar_family(left) != scalar_family(right) {
                return Ok(Truth::False);
            }
            Ok(if scalar_is_nan(left) || scalar_is_nan(right) {
                Truth::False
            } else if left == right {
                Truth::True
            } else {
                Truth::False
            })
        }
        (ResultValue::Node(left), ResultValue::Node(right)) => Ok(bool_truth(left.id == right.id)),
        (ResultValue::Relationship(left), ResultValue::Relationship(right)) => {
            Ok(bool_truth(left.id == right.id))
        }
        (
            ResultValue::Path {
                nodes: left_nodes,
                relationships: left_relationships,
            },
            ResultValue::Path {
                nodes: right_nodes,
                relationships: right_relationships,
            },
        ) => Ok(bool_truth(
            left_nodes == right_nodes && left_relationships == right_relationships,
        )),
        (ResultValue::Vector(left), ResultValue::Vector(right)) => {
            Ok(bool_truth(vector_equal(left, right)))
        }
        (ResultValue::List(left), ResultValue::List(right)) => sequence_equal(left, right),
        (ResultValue::Map(left), ResultValue::Map(right)) => map_equal(left, right),
        (
            ResultValue::Path {
                nodes,
                relationships,
            },
            ResultValue::List(values),
        )
        | (
            ResultValue::List(values),
            ResultValue::Path {
                nodes,
                relationships,
            },
        ) => path_list_equal(nodes, relationships, values),
        _ => Ok(Truth::False),
    }
}

const fn scalar_is_nan(value: &ScalarValue) -> bool {
    matches!(value, ScalarValue::Float(value) if value.0.is_nan())
}

fn sequence_equal(left: &[ResultValue], right: &[ResultValue]) -> Result<Truth> {
    if left.len() != right.len() {
        return Ok(Truth::False);
    }
    let mut result = Truth::True;
    for (left, right) in left.iter().zip(right) {
        result = result.and(equal(left, right)?);
        if result == Truth::False {
            break;
        }
    }
    Ok(result)
}

fn map_equal(
    left: &BTreeMap<String, ResultValue>,
    right: &BTreeMap<String, ResultValue>,
) -> Result<Truth> {
    if left.len() != right.len() || !left.keys().eq(right.keys()) {
        return Ok(Truth::False);
    }
    let mut result = Truth::True;
    for (left, right) in left.values().zip(right.values()) {
        result = result.and(equal(left, right)?);
        if result == Truth::False {
            break;
        }
    }
    Ok(result)
}

fn path_list_equal(
    nodes: &[ResultNode],
    relationships: &[ResultEdge],
    values: &[ResultValue],
) -> Result<Truth> {
    if values.len() != nodes.len().saturating_add(relationships.len())
        || nodes.len() != relationships.len().saturating_add(1)
    {
        return Ok(Truth::False);
    }
    for (index, value) in values.iter().enumerate() {
        if index % 2 == 0 {
            let Some(node) = nodes.get(index / 2) else {
                return Ok(Truth::False);
            };
            if !matches!(value, ResultValue::Node(value) if value.id == node.id) {
                return Ok(Truth::False);
            }
        } else {
            let Some(relationship) = relationships.get(index / 2) else {
                return Ok(Truth::False);
            };
            if !matches!(value, ResultValue::Relationship(value) if value.id == relationship.id) {
                return Ok(Truth::False);
            }
        }
    }
    Ok(Truth::True)
}

pub fn contains(list: &[ResultValue], needle: &ResultValue) -> Result<Truth> {
    if list.is_empty() {
        return Ok(Truth::False);
    }
    let mut unknown = false;
    for value in list {
        match equal(needle, value)? {
            Truth::True => return Ok(Truth::True),
            Truth::Unknown => unknown = true,
            Truth::False => {}
        }
    }
    Ok(if unknown {
        Truth::Unknown
    } else {
        Truth::False
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartialComparison {
    Ordered(Ordering),
    /// Cypher's three-valued result for nulls, incomparable value families, and durations.
    Unknown,
    /// IEEE NaN is comparable to no number, but its four ordering predicates are all `false`
    /// rather than `null`.
    Unordered,
}

/// Evaluates one of Cypher's four ordering predicates with its exact three-valued semantics.
/// Values from different families yield `null` (numbers are the sole cross-family exception),
/// lists compare lexicographically, and a numeric comparison involving NaN is always false.
pub fn compare_predicate(
    left: &ResultValue,
    right: &ResultValue,
    operation: BinaryOperator,
) -> Result<Truth> {
    let ordering = match partial_compare(left, right)? {
        PartialComparison::Ordered(ordering) => ordering,
        PartialComparison::Unknown => return Ok(Truth::Unknown),
        PartialComparison::Unordered => return Ok(Truth::False),
    };
    let value = match operation {
        BinaryOperator::Less => ordering.is_lt(),
        BinaryOperator::LessOrEqual => !ordering.is_gt(),
        BinaryOperator::Greater => ordering.is_gt(),
        BinaryOperator::GreaterOrEqual => !ordering.is_lt(),
        _ => {
            return Err(Error::internal(
                "comparison predicate requires an ordering operator",
            ));
        }
    };
    Ok(bool_truth(value))
}

fn partial_compare(left: &ResultValue, right: &ResultValue) -> Result<PartialComparison> {
    if let ResultValue::Scalar(value) = left
        && value.is_document()
    {
        let materialized = ResultValue::from_property(value.clone())?;
        return partial_compare(&materialized, right);
    }
    if let ResultValue::Scalar(value) = right
        && value.is_document()
    {
        let materialized = ResultValue::from_property(value.clone())?;
        return partial_compare(left, &materialized);
    }
    if is_null(left) || is_null(right) {
        return Ok(PartialComparison::Unknown);
    }
    match (left, right) {
        (
            ResultValue::Scalar(ScalarValue::Integer(left)),
            ResultValue::Scalar(ScalarValue::Float(right)),
        ) => Ok(if right.is_nan() {
            PartialComparison::Unordered
        } else {
            PartialComparison::Ordered(compare_i64_f64(*left, right.into_inner()))
        }),
        (
            ResultValue::Scalar(ScalarValue::Float(left)),
            ResultValue::Scalar(ScalarValue::Integer(right)),
        ) => Ok(if left.is_nan() {
            PartialComparison::Unordered
        } else {
            PartialComparison::Ordered(compare_i64_f64(*right, left.into_inner()).reverse())
        }),
        (ResultValue::Scalar(left), ResultValue::Scalar(right)) => {
            if scalar_family(left) != scalar_family(right) {
                return Ok(PartialComparison::Unknown);
            }
            if scalar_is_nan(left) || scalar_is_nan(right) {
                return Ok(PartialComparison::Unordered);
            }
            Ok(match compare_same_scalar(left, right) {
                Ok(Some(ordering)) => PartialComparison::Ordered(ordering),
                Ok(None) | Err(_) => PartialComparison::Unknown,
            })
        }
        (ResultValue::List(left), ResultValue::List(right)) => compare_lists(left, right),
        _ => Ok(PartialComparison::Unknown),
    }
}

fn compare_lists(left: &[ResultValue], right: &[ResultValue]) -> Result<PartialComparison> {
    for (left, right) in left.iter().zip(right) {
        match partial_compare(left, right)? {
            PartialComparison::Ordered(Ordering::Equal) => {}
            result => return Ok(result),
        }
    }
    Ok(PartialComparison::Ordered(left.len().cmp(&right.len())))
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
        _ => {
            return Err(incomparable(
                "ordering",
                left.type_name(),
                right.type_name(),
            ));
        }
    };
    Ok(result)
}

fn incomparable(operation: &str, left: &str, right: &str) -> Error {
    Error::new(
        ErrorCode::QueryType,
        format!("{operation} is not defined between {left} and {right}"),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ordered_float::OrderedFloat;

    use crate::ScalarValue;

    use super::{Truth, compare_predicate, contains, equal, total_compare};
    use crate::cypher::{BinaryOperator, ResultValue};

    fn integer(value: i64) -> ResultValue {
        ResultValue::Scalar(ScalarValue::Integer(value))
    }

    fn float(value: f64) -> ResultValue {
        ResultValue::Scalar(ScalarValue::Float(OrderedFloat(value)))
    }

    #[test]
    fn equality_and_in_follow_three_valued_logic() {
        let null = ResultValue::Scalar(ScalarValue::Null);
        assert_eq!(equal(&null, &null).ok(), Some(Truth::Unknown));
        assert_eq!(equal(&integer(2), &float(2.0)).ok(), Some(Truth::True));
        assert_eq!(
            contains(&[integer(1), null.clone(), integer(3)], &integer(2)).ok(),
            Some(Truth::Unknown)
        );
        assert_eq!(contains(&[], &null).ok(), Some(Truth::False));
    }

    #[test]
    fn nan_equality_is_false_and_inequality_is_true_for_every_operand_family() {
        let nan = float(f64::NAN);
        let operands = [
            integer(1),
            float(1.0),
            nan.clone(),
            ResultValue::Scalar(ScalarValue::String("a".into())),
        ];

        for operand in operands {
            for (left, right) in [(&nan, &operand), (&operand, &nan)] {
                let equality = equal(left, right).ok();
                assert_eq!(equality, Some(Truth::False));
                assert_eq!(equality.map(Truth::not), Some(Truth::True));
            }
        }
    }

    #[test]
    fn mixed_integer_float_comparison_does_not_lose_large_integer_bits() {
        let integer = integer(9_007_199_254_740_993);
        let rounded_float = float(9_007_199_254_740_992.0);
        assert_eq!(equal(&integer, &rounded_float).ok(), Some(Truth::False));
        assert_eq!(
            compare_predicate(&integer, &rounded_float, BinaryOperator::Greater).ok(),
            Some(Truth::True)
        );
    }

    #[test]
    fn ordering_predicates_follow_null_list_and_nan_semantics() {
        let string = ResultValue::Scalar(ScalarValue::String("1".into()));
        assert_eq!(
            compare_predicate(&string, &integer(1), BinaryOperator::Less).ok(),
            Some(Truth::Unknown)
        );

        let longer = ResultValue::List(vec![integer(1), ResultValue::Scalar(ScalarValue::Null)]);
        let prefix = ResultValue::List(vec![integer(1)]);
        assert_eq!(
            compare_predicate(&longer, &prefix, BinaryOperator::GreaterOrEqual).ok(),
            Some(Truth::True)
        );
        let null_tail = ResultValue::List(vec![integer(1), ResultValue::Scalar(ScalarValue::Null)]);
        assert_eq!(
            compare_predicate(
                &ResultValue::List(vec![integer(1), integer(2)]),
                &null_tail,
                BinaryOperator::GreaterOrEqual,
            )
            .ok(),
            Some(Truth::Unknown)
        );

        let nan = float(f64::NAN);
        for operation in [
            BinaryOperator::Less,
            BinaryOperator::LessOrEqual,
            BinaryOperator::Greater,
            BinaryOperator::GreaterOrEqual,
        ] {
            assert_eq!(
                compare_predicate(&nan, &integer(1), operation).ok(),
                Some(Truth::False)
            );
        }
    }

    #[test]
    fn total_order_uses_map_list_temporal_string_boolean_number_null_hierarchy() {
        let map = ResultValue::Map(BTreeMap::new());
        let list = ResultValue::List(Vec::new());
        let date = ResultValue::Scalar(ScalarValue::Date(0));
        let string = ResultValue::Scalar(ScalarValue::String("a".into()));
        let boolean = ResultValue::Scalar(ScalarValue::Boolean(false));
        let null = ResultValue::Scalar(ScalarValue::Null);
        let values = [map, list, date, string, boolean, integer(1), null];
        assert!(
            values
                .windows(2)
                .all(|pair| { total_compare(&pair[0], &pair[1]) == std::cmp::Ordering::Less })
        );
    }

    #[test]
    fn zoned_time_order_uses_the_utc_equivalent_time() {
        let early_utc = ResultValue::Scalar(ScalarValue::ZonedTime {
            nanos: (12 * 60 * 60 + 35 * 60 + 15) * 1_000_000_000,
            offset_seconds: 5 * 60 * 60,
        });
        let late_utc = ResultValue::Scalar(ScalarValue::ZonedTime {
            nanos: (10 * 60 * 60 + 35 * 60) * 1_000_000_000,
            offset_seconds: -8 * 60 * 60,
        });
        assert_eq!(
            total_compare(&early_utc, &late_utc),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_predicate(&early_utc, &late_utc, BinaryOperator::Less).ok(),
            Some(Truth::True)
        );
    }
}
