//! Canonical keys for Cypher grouping and duplicate elimination.
//!
//! Wire serialization is deliberately not used here: Cypher considers an integer and an exactly
//! equal floating-point value equivalent, while their serialized representations differ. These
//! keys encode that semantic equivalence and remain independent of checkpoint format versions.

use crate::{DocumentItem, ScalarValue};

use super::ResultValue;

pub fn value_key(value: &ResultValue) -> Vec<u8> {
    let mut output = Vec::new();
    append_value(&mut output, value);
    output
}

pub fn append_value(output: &mut Vec<u8>, value: &ResultValue) {
    match value {
        ResultValue::Scalar(value) => append_scalar(output, value),
        ResultValue::Node(node) => {
            output.push(16);
            output.extend_from_slice(&node.id.0.to_be_bytes());
        }
        ResultValue::Relationship(relationship) => {
            output.push(17);
            output.extend_from_slice(&relationship.id.0.to_be_bytes());
        }
        ResultValue::Path {
            nodes,
            relationships,
        } => {
            // A path is equality-compatible with the corresponding alternating list.
            output.push(18);
            push_len(output, nodes.len().saturating_add(relationships.len()));
            for index in 0..nodes.len().saturating_add(relationships.len()) {
                if index % 2 == 0 {
                    output.push(16);
                    output.extend_from_slice(&nodes[index / 2].id.0.to_be_bytes());
                } else {
                    output.push(17);
                    output.extend_from_slice(&relationships[index / 2].id.0.to_be_bytes());
                }
            }
        }
        ResultValue::Vector(vector) => {
            output.push(19);
            push_len(output, vector.len());
            for coordinate in vector {
                // Vector equality is intentionally bitwise, unlike scalar numeric equality.
                output.extend_from_slice(&coordinate.to_bits().to_be_bytes());
            }
        }
        ResultValue::List(values) => {
            output.push(18);
            push_len(output, values.len());
            for value in values {
                append_value(output, value);
            }
        }
        ResultValue::Map(values) => {
            output.push(20);
            push_len(output, values.len());
            for (name, value) in values {
                push_bytes(output, name.as_bytes());
                append_value(output, value);
            }
        }
    }
}

fn append_scalar(output: &mut Vec<u8>, value: &ScalarValue) {
    match value {
        ScalarValue::Null => output.push(0),
        ScalarValue::Boolean(value) => {
            output.extend_from_slice(&[1, u8::from(*value)]);
        }
        ScalarValue::Integer(value) => append_integer(output, *value),
        ScalarValue::Float(value) => {
            let value = value.into_inner();
            if let Some(integer) = exactly_equivalent_integer(value) {
                append_integer(output, integer);
            } else {
                output.push(3);
                let bits = if value.is_nan() {
                    f64::NAN.to_bits()
                } else if value == 0.0 {
                    0.0_f64.to_bits()
                } else {
                    value.to_bits()
                };
                output.extend_from_slice(&bits.to_be_bytes());
            }
        }
        ScalarValue::String(value) => {
            output.push(4);
            push_bytes(output, value.as_bytes());
        }
        ScalarValue::Bytes(value) => {
            output.push(5);
            push_bytes(output, value);
        }
        ScalarValue::Date(value) => {
            output.push(6);
            output.extend_from_slice(&value.to_be_bytes());
        }
        ScalarValue::LocalTime(value) => {
            output.push(7);
            output.extend_from_slice(&value.to_be_bytes());
        }
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => {
            output.push(8);
            output.extend_from_slice(&nanos.to_be_bytes());
            output.extend_from_slice(&offset_seconds.to_be_bytes());
        }
        ScalarValue::LocalDateTime { seconds, nanos } => {
            output.push(9);
            output.extend_from_slice(&seconds.to_be_bytes());
            output.extend_from_slice(&nanos.to_be_bytes());
        }
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            output.push(10);
            output.extend_from_slice(&seconds.to_be_bytes());
            output.extend_from_slice(&nanos.to_be_bytes());
            push_bytes(output, timezone.as_bytes());
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => {
            output.push(11);
            output.extend_from_slice(&months.to_be_bytes());
            output.extend_from_slice(&days.to_be_bytes());
            output.extend_from_slice(&seconds.to_be_bytes());
            output.extend_from_slice(&nanos.to_be_bytes());
        }
        ScalarValue::List(value) => {
            output.push(18);
            if let Ok(values) = value.items() {
                push_len(output, values.len());
                for value in &values {
                    append_document_item(output, value);
                }
            } else {
                push_bytes(output, value.as_bytes());
            }
        }
        ScalarValue::Map(value) => {
            output.push(20);
            if let Ok(values) = value.entries() {
                push_len(output, values.len());
                for (name, value) in &values {
                    push_bytes(output, name.as_bytes());
                    append_document_item(output, value);
                }
            } else {
                push_bytes(output, value.as_bytes());
            }
        }
    }
}

fn append_document_item(output: &mut Vec<u8>, value: &DocumentItem) {
    match value {
        DocumentItem::Scalar(value) => append_scalar(output, value),
        DocumentItem::List(values) => {
            output.push(18);
            push_len(output, values.len());
            for value in values {
                append_document_item(output, value);
            }
        }
        DocumentItem::Map(values) => {
            output.push(20);
            push_len(output, values.len());
            for (name, value) in values {
                push_bytes(output, name.as_bytes());
                append_document_item(output, value);
            }
        }
    }
}

fn append_integer(output: &mut Vec<u8>, value: i64) {
    output.push(2);
    output.extend_from_slice(&value.to_be_bytes());
}

fn exactly_equivalent_integer(value: f64) -> Option<i64> {
    const TWO_TO_63: f64 = 9_223_372_036_854_775_808.0;
    if value.is_finite() && value.fract() == 0.0 && (-TWO_TO_63..TWO_TO_63).contains(&value) {
        Some(value as i64)
    } else {
        None
    }
}

fn push_len(output: &mut Vec<u8>, value: usize) {
    output.extend_from_slice(&u64::try_from(value).unwrap_or(u64::MAX).to_be_bytes());
}

fn push_bytes(output: &mut Vec<u8>, value: &[u8]) {
    push_len(output, value.len());
    output.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use crate::ScalarValue;

    use super::{ResultValue, value_key};

    #[test]
    fn numeric_keys_follow_cross_type_equality() {
        let integer = ResultValue::Scalar(ScalarValue::Integer(42));
        let float = ResultValue::Scalar(ScalarValue::Float(OrderedFloat(42.0)));
        let fraction = ResultValue::Scalar(ScalarValue::Float(OrderedFloat(42.5)));
        assert_eq!(value_key(&integer), value_key(&float));
        assert_ne!(value_key(&integer), value_key(&fraction));
    }

    #[test]
    fn path_and_alternating_list_share_a_key() {
        use crate::execution::{ResultEdge, ResultNode};
        use crate::{EdgeId, Layer, NodeId};
        use std::collections::BTreeMap;

        let node = |id| ResultNode {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: BTreeMap::new(),
        };
        let edge = ResultEdge {
            id: EdgeId(2),
            source: NodeId(1),
            target: NodeId(3),
            relationship_type: "R".into(),
            layer: Layer::Observed,
            revision: 1,
            properties: BTreeMap::new(),
        };
        let path = ResultValue::Path {
            nodes: vec![node(1), node(3)],
            relationships: vec![edge.clone()],
        };
        let list = ResultValue::List(vec![
            ResultValue::Node(node(1)),
            ResultValue::Relationship(edge),
            ResultValue::Node(node(3)),
        ]);
        assert_eq!(value_key(&path), value_key(&list));
    }
}
