//! The locked Knowledge-layer contract.
//!
//! The `Knowledge` layer is standardized around a small, fixed set of reserved property names that
//! carry the model-visible skeleton of a fact — and nothing else is reserved (no metadata). These
//! names are deliberately *not* `__`-prefixed so they reach the model through attention-layer
//! injection rather than being hidden from it.
//!
//! - [`NAME`] (node, required): the canonical human identifier used for retrieval and shown to the
//!   model.
//! - [`TEXT`] (edge, optional): free-text context/description — e.g. a paragraph lifted from a PDF.
//! - [`AT`] (node or edge, optional): a temporal scalar marking when the fact holds.
//!
//! The contract is enforced at the single write funnel (`GraphStore::insert_node` /
//! `insert_edge`) for `Layer::Knowledge` writes.
//! The `Observed` layer is deliberately left unconstrained — that is the user's business data.

use super::{GraphMutation, GraphStore, NameCatalog};
use crate::types::{PropertyId, ScalarValue};
use crate::{Error, Layer, Result};

/// Reserved node identifier property (required on a Knowledge node).
pub const NAME: &str = "name";
/// Reserved edge description property (optional free text, e.g. a PDF paragraph).
pub const TEXT: &str = "text";
/// Reserved point-in-time property (optional temporal scalar).
pub const AT: &str = "at";

/// Render priority for the reserved fields, so the memory renderer surfaces them first and the
/// property cap can never drop them. Lower sorts earlier; non-reserved names fall through.
#[must_use]
pub fn render_priority(name: &str) -> usize {
    match () {
        () if name.eq_ignore_ascii_case(NAME) => 0,
        () if name.eq_ignore_ascii_case(TEXT) => 1,
        () if name.eq_ignore_ascii_case(AT) => 2,
        () => usize::MAX,
    }
}

/// True when `value` is a temporal scalar acceptable for the reserved [`AT`] field.
#[must_use]
pub fn is_temporal_scalar(value: &ScalarValue) -> bool {
    matches!(
        value,
        ScalarValue::Date(_)
            | ScalarValue::LocalTime(_)
            | ScalarValue::ZonedTime { .. }
            | ScalarValue::LocalDateTime { .. }
            | ScalarValue::ZonedDateTime { .. }
            | ScalarValue::Duration { .. }
    )
}

fn reserved_value<'a>(
    catalog: &NameCatalog,
    properties: &'a [(PropertyId, ScalarValue)],
    reserved: &str,
) -> Option<&'a ScalarValue> {
    let id = catalog.property(reserved)?;
    properties
        .iter()
        .find(|(property, _)| *property == id)
        .map(|(_, value)| value)
}

/// Enforces the locked contract for a Knowledge-layer node: a non-blank string [`NAME`], plus a
/// well-typed [`AT`] when present.
pub fn validate_node(
    catalog: &NameCatalog,
    properties: &[(PropertyId, ScalarValue)],
) -> Result<()> {
    match reserved_value(catalog, properties, NAME) {
        Some(ScalarValue::String(value)) if !value.trim().is_empty() => {}
        Some(ScalarValue::String(_)) => {
            return Err(Error::invalid_data(
                "KNOWLEDGE node `name` must not be blank",
            ));
        }
        Some(_) => {
            return Err(Error::invalid_data(
                "KNOWLEDGE node `name` must be a STRING",
            ));
        }
        None => {
            return Err(Error::invalid_data(
                "KNOWLEDGE node requires a `name` property",
            ));
        }
    }
    validate_at(catalog, properties)
}

/// Enforces the locked contract for a Knowledge-layer edge: [`TEXT`] is optional but must be a
/// string, and [`AT`] must be a temporal scalar when present.
pub fn validate_edge(
    catalog: &NameCatalog,
    properties: &[(PropertyId, ScalarValue)],
) -> Result<()> {
    if let Some(text) = reserved_value(catalog, properties, TEXT) {
        if !matches!(text, ScalarValue::String(_)) {
            return Err(Error::invalid_data(
                "KNOWLEDGE edge `text` must be a STRING",
            ));
        }
    }
    validate_at(catalog, properties)
}

fn validate_at(catalog: &NameCatalog, properties: &[(PropertyId, ScalarValue)]) -> Result<()> {
    if let Some(at) = reserved_value(catalog, properties, AT) {
        if !is_temporal_scalar(at) {
            return Err(Error::invalid_data(
                "KNOWLEDGE `at` must be a temporal value (date/time/datetime/duration)",
            ));
        }
    }
    Ok(())
}

/// The single enforcement point for the locked Knowledge contract on the durable write path. Only
/// `Knowledge`-layer creates and reserved-field updates are constrained; `Observed` writes and the
/// low-level storage primitive stay generic.
pub fn validate_mutation(graph: &GraphStore, mutation: &GraphMutation) -> Result<()> {
    match mutation {
        GraphMutation::InsertNode(input) if input.layer == Layer::Knowledge => {
            validate_node(graph.catalog(), &input.properties)
        }
        GraphMutation::InsertEdge(input) if input.layer == Layer::Knowledge => {
            validate_edge(graph.catalog(), &input.properties)
        }
        GraphMutation::SetNodeProperty {
            node,
            property,
            value,
            ..
        } if graph
            .node(*node)
            .is_some_and(|node| node.layer() == Layer::Knowledge) =>
        {
            validate_reserved_update(graph.catalog(), *property, value, ReservedTarget::Node)
        }
        GraphMutation::SetEdgeProperty {
            edge,
            property,
            value,
            ..
        } if graph
            .edge(*edge)
            .is_some_and(|edge| edge.layer() == Layer::Knowledge) =>
        {
            validate_reserved_update(graph.catalog(), *property, value, ReservedTarget::Edge)
        }
        _ => Ok(()),
    }
}

enum ReservedTarget {
    Node,
    Edge,
}

/// Validates a bounded mutation journal without publishing its schema or data.
pub fn validate_batch(graph: &GraphStore, mutations: &[GraphMutation]) -> Result<()> {
    let mut names = std::collections::BTreeMap::<PropertyId, &str>::new();
    let mut node_layers = std::collections::BTreeMap::new();
    let mut edge_layers = std::collections::BTreeMap::new();
    for mutation in mutations {
        if let GraphMutation::DeclareProperty { name, id } = mutation {
            names.insert(*id, name.as_str());
            continue;
        }
        let property_name = |property: PropertyId| -> Option<std::sync::Arc<str>> {
            names
                .get(&property)
                .map(|name| std::sync::Arc::from(*name))
                .or_else(|| graph.catalog().property_name(property))
        };
        let validate = |property: PropertyId, value: &ScalarValue, node: bool| -> Result<()> {
            let Some(name) = property_name(property) else {
                return Ok(());
            };
            if node
                && name.eq_ignore_ascii_case(NAME)
                && !matches!(value, ScalarValue::String(value) if !value.trim().is_empty())
            {
                return Err(Error::invalid_data(
                    "KNOWLEDGE node `name` must be a non-blank STRING",
                ));
            }
            if !node && name.eq_ignore_ascii_case(TEXT) && !matches!(value, ScalarValue::String(_))
            {
                return Err(Error::invalid_data(
                    "KNOWLEDGE edge `text` must be a STRING",
                ));
            }
            if name.eq_ignore_ascii_case(AT) && !is_temporal_scalar(value) {
                return Err(Error::invalid_data(
                    "KNOWLEDGE `at` must be a temporal value",
                ));
            }
            Ok(())
        };
        match mutation {
            GraphMutation::InsertNode(input) => {
                node_layers.insert(input.id, input.layer);
                if input.layer == Layer::Knowledge {
                    if !input.properties.iter().any(|(property, _)| {
                        property_name(*property).is_some_and(|name| name.eq_ignore_ascii_case(NAME))
                    }) {
                        return Err(Error::invalid_data(
                            "KNOWLEDGE node requires a `name` property",
                        ));
                    }
                    for (property, value) in &input.properties {
                        validate(*property, value, true)?;
                    }
                }
            }
            GraphMutation::InsertEdge(input) => {
                edge_layers.insert(input.id, input.layer);
                if input.layer == Layer::Knowledge {
                    for (property, value) in &input.properties {
                        validate(*property, value, false)?;
                    }
                }
            }
            GraphMutation::SetNodeProperty {
                node,
                property,
                value,
                ..
            } if node_layers
                .get(node)
                .copied()
                .or_else(|| graph.node(*node).map(|row| row.layer()))
                == Some(Layer::Knowledge) =>
            {
                validate(*property, value, true)?
            }
            GraphMutation::SetEdgeProperty {
                edge,
                property,
                value,
                ..
            } if edge_layers
                .get(edge)
                .copied()
                .or_else(|| graph.edge(*edge).map(|row| row.layer()))
                == Some(Layer::Knowledge) =>
            {
                validate(*property, value, false)?
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_reserved_update(
    catalog: &NameCatalog,
    property: PropertyId,
    value: &ScalarValue,
    target: ReservedTarget,
) -> Result<()> {
    let Some(name) = catalog.property_name(property) else {
        return Ok(());
    };
    if matches!(target, ReservedTarget::Node) && name.eq_ignore_ascii_case(NAME) {
        match value {
            ScalarValue::String(value) if !value.trim().is_empty() => {}
            _ => {
                return Err(Error::invalid_data(
                    "KNOWLEDGE node `name` must be a non-blank STRING",
                ));
            }
        }
    } else if matches!(target, ReservedTarget::Edge)
        && name.eq_ignore_ascii_case(TEXT)
        && !matches!(value, ScalarValue::String(_))
    {
        return Err(Error::invalid_data(
            "KNOWLEDGE edge `text` must be a STRING",
        ));
    } else if name.eq_ignore_ascii_case(AT) && !is_temporal_scalar(value) {
        return Err(Error::invalid_data(
            "KNOWLEDGE `at` must be a temporal value (date/time/datetime/duration)",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct Schema {
        catalog: NameCatalog,
    }

    impl Schema {
        fn new() -> Self {
            let catalog = NameCatalog::default();
            for reserved in [NAME, TEXT, AT, "city"] {
                catalog.intern_property(reserved).expect("intern reserved");
            }
            Self { catalog }
        }

        fn id(&self, name: &str) -> PropertyId {
            self.catalog.property(name).expect("interned")
        }

        fn prop(&self, name: &str, value: ScalarValue) -> (PropertyId, ScalarValue) {
            (self.id(name), value)
        }
    }

    fn text(value: &str) -> ScalarValue {
        ScalarValue::String(Arc::from(value))
    }

    #[test]
    fn knowledge_node_requires_a_non_blank_string_name() {
        let schema = Schema::new();
        assert!(validate_node(&schema.catalog, &[]).is_err());
        assert!(
            validate_node(&schema.catalog, &[schema.prop(NAME, text("  "))]).is_err(),
            "blank name rejected"
        );
        assert!(
            validate_node(
                &schema.catalog,
                &[schema.prop(NAME, ScalarValue::Integer(7))]
            )
            .is_err(),
            "non-string name rejected"
        );
        validate_node(&schema.catalog, &[schema.prop(NAME, text("Acme Corp"))])
            .expect("named knowledge node accepted");
    }

    #[test]
    fn reserved_at_must_be_temporal_everywhere() {
        let schema = Schema::new();
        let named = schema.prop(NAME, text("Acme Corp"));
        assert!(
            validate_node(
                &schema.catalog,
                &[named.clone(), schema.prop(AT, text("yesterday"))]
            )
            .is_err(),
            "string `at` rejected"
        );
        validate_node(
            &schema.catalog,
            &[named, schema.prop(AT, ScalarValue::Date(19_000))],
        )
        .expect("date `at` accepted");
        assert!(
            validate_edge(&schema.catalog, &[schema.prop(AT, ScalarValue::Integer(1))]).is_err(),
            "integer `at` on an edge rejected"
        );
    }

    #[test]
    fn knowledge_edge_text_is_optional_but_must_be_a_string() {
        let schema = Schema::new();
        validate_edge(&schema.catalog, &[]).expect("edge without text accepted");
        validate_edge(
            &schema.catalog,
            &[schema.prop(TEXT, text("Extracted from the 2024 filing."))],
        )
        .expect("string text accepted");
        assert!(
            validate_edge(
                &schema.catalog,
                &[schema.prop(TEXT, ScalarValue::Integer(1))]
            )
            .is_err(),
            "non-string text rejected"
        );
    }

    #[test]
    fn reserved_fields_sort_ahead_of_business_properties() {
        assert!(render_priority(NAME) < render_priority("city"));
        assert!(render_priority(TEXT) < render_priority("city"));
        assert!(render_priority(AT) < render_priority("city"));
        assert_eq!(render_priority("CITY"), usize::MAX);
    }
}
