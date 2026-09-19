//! Rebuildable semantic descriptions of canonical graph owners.
//!
//! Owner text is complete; only the identifying name copied into a relationship is bounded.
//! The mutation overlay reads changed owners and their adjacency, never the unrelated corpus.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    DocumentItem, EdgeId, GraphMutation, GraphStore, LabelId, NodeId, PropertyId,
    RelationshipTypeId, Result, ScalarValue,
};

/// One derived description, or a tombstone removing a deleted owner's vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticText {
    pub entity_id: u64,
    pub text: Option<String>,
}

/// Node and relationship identities inhabit separate namespaces.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SemanticTextBatch {
    pub nodes: Vec<SemanticText>,
    pub relationships: Vec<SemanticText>,
}

/// Cold build for all live canonical owners, including owners with only a label or type.
pub fn semantic_texts(graph: &GraphStore) -> Result<SemanticTextBatch> {
    let overlay = Overlay::new(graph, &[], &[]);
    overlay.render(
        graph.nodes().map(|node| node.id()).collect(),
        graph.edges().map(|edge| edge.id()).collect(),
    )
}

/// Describes current mutations against the state after earlier mutations in a transaction.
/// Changes to endpoint names/labels also refresh incident relationship descriptions. Deleted
/// owners produce tombstones, including relationships removed by a detached node deletion.
pub fn semantic_text_delta(
    graph: &GraphStore,
    prior: &[GraphMutation],
    current: &[GraphMutation],
) -> Result<SemanticTextBatch> {
    let overlay = Overlay::new(graph, prior, current);
    let mut nodes = BTreeSet::new();
    let mut edges = BTreeSet::new();
    let mut endpoints = BTreeSet::new();
    for mutation in current {
        match mutation {
            GraphMutation::InsertNode(node) => {
                nodes.insert(node.id);
                endpoints.insert(node.id);
            }
            GraphMutation::SetNodeProperty { node, property, .. } => {
                if overlay
                    .property_name(*property)
                    .is_some_and(meaningful_field)
                {
                    nodes.insert(*node);
                }
                if overlay.property_name(*property).is_some_and(identity_field) {
                    endpoints.insert(*node);
                }
            }
            GraphMutation::AddNodeLabels { node, .. }
            | GraphMutation::RemoveNodeLabels { node, .. }
            | GraphMutation::DeleteNode { node, .. } => {
                nodes.insert(*node);
                endpoints.insert(*node);
            }
            GraphMutation::InsertEdge(edge) => {
                edges.insert(edge.id);
            }
            GraphMutation::SetEdgeProperty { edge, property, .. } => {
                if overlay
                    .property_name(*property)
                    .is_some_and(meaningful_field)
                {
                    edges.insert(*edge);
                }
            }
            GraphMutation::DeleteEdge { edge, .. } => {
                edges.insert(*edge);
            }
            _ => {}
        }
    }
    for node in &endpoints {
        if graph.node(*node).is_some() {
            edges.extend(graph.incident_edge_ids(*node)?);
        }
    }
    // Inserted relationships have no canonical adjacency until commit.
    for mutation in prior.iter().chain(current) {
        if let GraphMutation::InsertEdge(edge) = mutation {
            if endpoints.contains(&edge.source) || endpoints.contains(&edge.target) {
                edges.insert(edge.id);
            }
        }
    }
    overlay.render(nodes, edges)
}

struct Overlay<'a> {
    graph: &'a GraphStore,
    nodes: BTreeMap<NodeId, Vec<&'a GraphMutation>>,
    edges: BTreeMap<EdgeId, Vec<&'a GraphMutation>>,
    labels: BTreeMap<LabelId, &'a str>,
    properties: BTreeMap<PropertyId, &'a str>,
    types: BTreeMap<RelationshipTypeId, &'a str>,
}

struct NodeText {
    labels: Vec<LabelId>,
    properties: BTreeMap<PropertyId, ScalarValue>,
}

impl<'a> Overlay<'a> {
    fn new(
        graph: &'a GraphStore,
        prior: &'a [GraphMutation],
        current: &'a [GraphMutation],
    ) -> Self {
        let mut overlay = Self {
            graph,
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            labels: BTreeMap::new(),
            properties: BTreeMap::new(),
            types: BTreeMap::new(),
        };
        for mutation in prior.iter().chain(current) {
            match mutation {
                GraphMutation::DeclareLabel { name, id } => {
                    overlay.labels.insert(*id, name);
                }
                GraphMutation::DeclareProperty { name, id } => {
                    overlay.properties.insert(*id, name);
                }
                GraphMutation::DeclareRelationshipType { name, id } => {
                    overlay.types.insert(*id, name);
                }
                GraphMutation::InsertNode(node) => {
                    overlay.nodes.entry(node.id).or_default().push(mutation)
                }
                GraphMutation::SetNodeProperty { node, .. }
                | GraphMutation::AddNodeLabels { node, .. }
                | GraphMutation::RemoveNodeLabels { node, .. }
                | GraphMutation::DeleteNode { node, .. } => {
                    overlay.nodes.entry(*node).or_default().push(mutation)
                }
                GraphMutation::InsertEdge(edge) => {
                    overlay.edges.entry(edge.id).or_default().push(mutation)
                }
                GraphMutation::SetEdgeProperty { edge, .. }
                | GraphMutation::DeleteEdge { edge, .. } => {
                    overlay.edges.entry(*edge).or_default().push(mutation)
                }
            }
        }
        overlay
    }

    fn property_name(&self, id: PropertyId) -> Option<&str> {
        self.properties
            .get(&id)
            .copied()
            .or_else(|| self.graph.catalog().property_name(id))
    }

    fn property_allowed(&self, id: PropertyId, identity: bool) -> bool {
        self.property_name(id)
            .is_some_and(|name| meaningful_field(name) && (!identity || identity_field(name)))
    }

    fn node(&self, id: NodeId, identity: bool) -> Option<NodeText> {
        let mut state = self.graph.node(id).map(|node| NodeText {
            labels: node.labels().to_vec(),
            properties: self
                .graph
                .catalog()
                .properties()
                .filter(|(property, _)| self.property_allowed(*property, identity))
                .filter_map(|(property, _)| node.property(property).map(|value| (property, value)))
                .collect(),
        });
        for mutation in self.nodes.get(&id).into_iter().flatten() {
            match mutation {
                GraphMutation::InsertNode(node) => {
                    state = Some(NodeText {
                        labels: node.labels.clone(),
                        properties: node
                            .properties
                            .iter()
                            .filter(|(property, _)| self.property_allowed(*property, identity))
                            .cloned()
                            .collect(),
                    });
                }
                GraphMutation::DeleteNode { .. } => state = None,
                GraphMutation::SetNodeProperty {
                    property, value, ..
                } => {
                    if self.property_allowed(*property, identity) {
                        if let Some(node) = &mut state {
                            node.properties.insert(*property, value.clone());
                        }
                    }
                }
                GraphMutation::AddNodeLabels { labels, .. } => {
                    if let Some(node) = &mut state {
                        node.labels.extend(labels);
                    }
                }
                GraphMutation::RemoveNodeLabels { labels, .. } => {
                    if let Some(node) = &mut state {
                        node.labels.retain(|label| !labels.contains(label));
                    }
                }
                _ => {}
            }
        }
        state
    }

    fn labels(&self, node: &NodeText) -> String {
        let labels: BTreeSet<_> = node
            .labels
            .iter()
            .filter_map(|label| {
                self.labels
                    .get(label)
                    .copied()
                    .or_else(|| self.graph.catalog().label_name(*label))
            })
            .collect();
        if labels.is_empty() {
            "Node".to_owned()
        } else {
            labels
                .into_iter()
                .map(readable_name)
                .collect::<Vec<_>>()
                .join(", ")
        }
    }

    fn node_text(&self, id: NodeId) -> Result<Option<String>> {
        let Some(node) = self.node(id, false) else {
            return Ok(None);
        };
        let mut text = self.labels(&node);
        self.append_properties(&mut text, node.properties)?;
        Ok(Some(text))
    }

    fn endpoint_text(&self, id: NodeId) -> Option<String> {
        let node = self.node(id, true)?;
        let mut text = self.labels(&node);
        for (property, value) in node.properties {
            if let ScalarValue::String(value) = value {
                if meaningful_string(&value) {
                    text.push_str("; ");
                    text.push_str(&readable_name(
                        self.property_name(property).unwrap_or("name"),
                    ));
                    text.push_str(": ");
                    // A relationship borrows an identifying excerpt, not an owner's full content.
                    text.extend(value.trim().chars().take(256));
                }
            }
        }
        Some(text)
    }

    fn edge_text(&self, id: EdgeId) -> Result<Option<String>> {
        let mut state = self.graph.edge(id).map(|edge| {
            (
                edge.source(),
                edge.target(),
                edge.relationship_type(),
                self.graph
                    .catalog()
                    .properties()
                    .filter(|(property, _)| self.property_allowed(*property, false))
                    .filter_map(|(property, _)| {
                        edge.property(property).map(|value| (property, value))
                    })
                    .collect::<BTreeMap<_, _>>(),
            )
        });
        for mutation in self.edges.get(&id).into_iter().flatten() {
            match mutation {
                GraphMutation::InsertEdge(edge) => {
                    state = Some((
                        edge.source,
                        edge.target,
                        edge.relationship_type,
                        edge.properties
                            .iter()
                            .filter(|(property, _)| self.property_allowed(*property, false))
                            .cloned()
                            .collect(),
                    ))
                }
                GraphMutation::SetEdgeProperty {
                    property, value, ..
                } => {
                    if self.property_allowed(*property, false) {
                        if let Some((_, _, _, properties)) = &mut state {
                            properties.insert(*property, value.clone());
                        }
                    }
                }
                GraphMutation::DeleteEdge { .. } => state = None,
                _ => {}
            }
        }
        let Some((source, target, kind, properties)) = state else {
            return Ok(None);
        };
        let (Some(source), Some(target)) = (self.endpoint_text(source), self.endpoint_text(target))
        else {
            return Ok(None);
        };
        let kind = self
            .types
            .get(&kind)
            .copied()
            .or_else(|| self.graph.catalog().relationship_type_name(kind))
            .unwrap_or("Relationship");
        let mut text = format!("({source}) — {} → ({target})", readable_name(kind));
        self.append_properties(&mut text, properties)?;
        Ok(Some(text))
    }

    fn append_properties(
        &self,
        text: &mut String,
        properties: BTreeMap<PropertyId, ScalarValue>,
    ) -> Result<()> {
        // Sort by name, so equivalent data has the same description regardless of schema IDs.
        let properties: BTreeMap<_, _> = properties
            .into_iter()
            .filter_map(|(id, value)| self.property_name(id).map(|name| (name, value)))
            .collect();
        for (name, value) in properties {
            append_value(text, name, value)?;
        }
        Ok(())
    }

    fn render(
        &self,
        nodes: BTreeSet<NodeId>,
        edges: BTreeSet<EdgeId>,
    ) -> Result<SemanticTextBatch> {
        Ok(SemanticTextBatch {
            nodes: nodes
                .into_iter()
                .map(|id| {
                    Ok(SemanticText {
                        entity_id: id.0,
                        text: self.node_text(id)?,
                    })
                })
                .collect::<Result<_>>()?,
            relationships: edges
                .into_iter()
                .map(|id| {
                    Ok(SemanticText {
                        entity_id: id.0,
                        text: self.edge_text(id)?,
                    })
                })
                .collect::<Result<_>>()?,
        })
    }
}

fn readable_name(name: &str) -> String {
    let mut text = String::new();
    let mut previous_lowercase = false;
    for character in name.chars() {
        if character == '_' || character == '-' || character == '.' {
            text.push(' ');
        } else {
            if previous_lowercase && character.is_uppercase() {
                text.push(' ');
            }
            text.push(character);
        }
        previous_lowercase = character.is_lowercase();
    }
    text
}

fn meaningful_field(name: &str) -> bool {
    let normalized = readable_name(name).to_ascii_lowercase();
    if name.starts_with('_') {
        return false;
    }
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    !words.iter().any(|word| {
        matches!(
            *word,
            "id" | "ids"
                | "uuid"
                | "guid"
                | "metadata"
                | "meta"
                | "timestamp"
                | "timestamps"
                | "created"
                | "updated"
                | "modified"
                | "deleted"
                | "ingested"
                | "synced"
                | "sync"
                | "revision"
                | "version"
                | "etag"
                | "hash"
                | "checksum"
                | "url"
                | "urls"
                | "uri"
                | "href"
                | "path"
                | "filename"
                | "mime"
                | "mimetype"
                | "encoding"
                | "credential"
                | "credentials"
                | "password"
                | "passwd"
                | "secret"
                | "secrets"
                | "token"
                | "tokens"
                | "authorization"
                | "auth"
                | "oauth"
                | "apikey"
                | "accesskey"
                | "privatekey"
                | "dsn"
                | "cookie"
                | "session"
                | "cache"
                | "embedding"
                | "embeddings"
                | "vector"
                | "vectors"
                | "score"
                | "rank"
                | "confidence"
                | "latency"
                | "offset"
                | "cursor"
                | "internal"
                | "system"
        )
    }) && !matches!(
        normalized.as_str(),
        "api key" | "access key" | "private key" | "public key" | "layer" | "source" | "provenance"
    )
}

fn identity_field(name: &str) -> bool {
    matches!(
        readable_name(name).to_ascii_lowercase().as_str(),
        "name"
            | "title"
            | "subject"
            | "display name"
            | "full name"
            | "first name"
            | "last name"
            | "label"
    )
}

fn meaningful_string(value: &str) -> bool {
    let value = value.trim();
    if value.is_empty()
        || (!value.contains(char::is_whitespace)
            && (value.starts_with("http://")
                || value.starts_with("https://")
                || value.starts_with("file://")))
    {
        return false;
    }
    // Do not turn opaque identifiers into semantic prose, even under an unfamiliar property name.
    !(value.len() >= 24
        && value
            .chars()
            .all(|character| character.is_ascii_hexdigit() || character == '-'))
}

fn append_value(text: &mut String, name: &str, value: ScalarValue) -> Result<()> {
    let mut pending = vec![(readable_name(name), DocumentItem::Scalar(value))];
    while let Some((name, item)) = pending.pop() {
        match item {
            DocumentItem::Scalar(ScalarValue::List(value)) => {
                pending.push((name, DocumentItem::List(value.items()?)))
            }
            DocumentItem::Scalar(ScalarValue::Map(value)) => {
                pending.push((name, DocumentItem::Map(value.entries()?)))
            }
            DocumentItem::List(values) => {
                pending.extend(values.into_iter().rev().map(|value| (name.clone(), value)));
            }
            DocumentItem::Map(values) => {
                pending.extend(
                    values
                        .into_iter()
                        .rev()
                        .filter(|(key, _)| meaningful_field(key))
                        .map(|(key, value)| (format!("{name}.{}", readable_name(&key)), value)),
                );
            }
            DocumentItem::Scalar(value) => {
                let value = match value {
                    ScalarValue::String(value) if meaningful_string(&value) => {
                        value.trim().to_owned()
                    }
                    ScalarValue::Integer(value) => value.to_string(),
                    ScalarValue::Float(value) if value.is_finite() => value.to_string(),
                    ScalarValue::Boolean(value) => value.to_string(),
                    value => match temporal_text(&value) {
                        Some(value) => value,
                        None => continue,
                    },
                };
                text.push('\n');
                text.push_str(&name);
                text.push_str(": ");
                text.push_str(&value);
            }
        }
    }
    Ok(())
}

fn temporal_text(value: &ScalarValue) -> Option<String> {
    use chrono::{DateTime, FixedOffset, NaiveTime, Utc};
    Some(match value {
        ScalarValue::Date(days) => days
            .checked_mul(86_400)
            .and_then(|seconds| DateTime::<Utc>::from_timestamp(seconds, 0))
            .map(|date| date.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| format!("{days} days relative to 1970-01-01")),
        ScalarValue::LocalTime(nanos) | ScalarValue::ZonedTime { nanos, .. } => {
            let clock = u32::try_from(nanos.div_euclid(1_000_000_000))
                .ok()
                .and_then(|seconds| {
                    NaiveTime::from_num_seconds_from_midnight_opt(
                        seconds,
                        nanos.rem_euclid(1_000_000_000) as u32,
                    )
                })
                .map(|time| time.format("%H:%M:%S%.f").to_string())
                .unwrap_or_else(|| format!("{nanos} nanoseconds after midnight"));
            if let ScalarValue::ZonedTime { offset_seconds, .. } = value {
                let offset = FixedOffset::east_opt(*offset_seconds)
                    .map(|offset| offset.to_string())
                    .unwrap_or_else(|| format!(" UTC offset {offset_seconds} seconds"));
                format!("{clock}{offset}")
            } else {
                clock
            }
        }
        ScalarValue::LocalDateTime { seconds, nanos } => {
            DateTime::<Utc>::from_timestamp(*seconds, *nanos)
                .map(|date| date.naive_utc().format("%Y-%m-%d %H:%M:%S%.f").to_string())
                .unwrap_or_else(|| {
                    format!("{seconds} seconds and {nanos} nanoseconds relative to 1970-01-01")
                })
        }
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            let date = DateTime::<Utc>::from_timestamp(*seconds, *nanos);
            if let (Some(date), Ok(zone)) = (date, timezone.parse::<chrono_tz::Tz>()) {
                format!("{} [{timezone}]", date.with_timezone(&zone).to_rfc3339())
            } else if let (Some(date), Ok(offset)) = (date, timezone.parse::<FixedOffset>()) {
                date.with_timezone(&offset).to_rfc3339()
            } else {
                format!(
                    "{} [{timezone}]",
                    date.map(|date| date.to_rfc3339())
                        .unwrap_or_else(|| format!(
                            "{seconds} seconds and {nanos} nanoseconds since 1970-01-01 UTC"
                        ))
                )
            }
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => format!("{months} months, {days} days, {seconds} seconds, {nanos} nanoseconds"),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DocumentList, DocumentMap, EdgeInput, Layer, NodeInput};
    use std::sync::Arc;

    fn string(value: &str) -> ScalarValue {
        ScalarValue::String(value.into())
    }

    fn insert_node(
        graph: &mut GraphStore,
        id: u64,
        label: &str,
        properties: Vec<(&str, ScalarValue)>,
    ) -> Result<()> {
        let label = graph.catalog_mut().intern_label(label)?;
        let properties = properties
            .into_iter()
            .map(|(name, value)| Ok((graph.catalog_mut().intern_property(name)?, value)))
            .collect::<Result<_>>()?;
        graph.apply(GraphMutation::InsertNode(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties,
        }))
    }

    fn insert_edge(graph: &mut GraphStore, id: u64, source: u64, target: u64) -> Result<()> {
        let relationship_type = graph.catalog_mut().intern_relationship_type("WORKS_ON")?;
        let reason = graph.catalog_mut().intern_property("reason")?;
        graph.apply(GraphMutation::InsertEdge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: 1,
            properties: vec![(reason, string("Responsible for the launch"))],
        }))
    }

    #[test]
    fn semantic_content_covers_owners_and_filters_metadata_at_every_depth() -> Result<()> {
        let mut graph = GraphStore::default();
        let payload = DocumentMap::new(BTreeMap::from([
            (
                Arc::from("summary"),
                DocumentItem::Scalar(string("Quarterly revenue")),
            ),
            (
                Arc::from("rows"),
                DocumentItem::List(vec![DocumentItem::Map(BTreeMap::from([
                    (Arc::from("product"), DocumentItem::Scalar(string("Coffee"))),
                    (
                        Arc::from("amount"),
                        DocumentItem::Scalar(ScalarValue::Integer(42)),
                    ),
                    (
                        Arc::from("approved"),
                        DocumentItem::Scalar(ScalarValue::Boolean(true)),
                    ),
                    (
                        Arc::from("externalId"),
                        DocumentItem::Scalar(string("hidden-row-id")),
                    ),
                ]))]),
            ),
            (
                Arc::from("metadata"),
                DocumentItem::Map(BTreeMap::from([(
                    Arc::from("body"),
                    DocumentItem::Scalar(string("hidden-nested-metadata")),
                )])),
            ),
        ]))?;
        insert_node(
            &mut graph,
            1,
            "Table",
            vec![
                ("content", ScalarValue::Map(payload)),
                ("api_key", string("hidden-credential")),
                ("createdAt", string("hidden-timestamp")),
                ("source_url", string("https://hidden.example")),
                ("source", string("hidden-source-file.pdf")),
                ("provenance", string("hidden-import-system")),
            ],
        )?;
        insert_node(
            &mut graph,
            2,
            "Email",
            vec![
                ("subject", string("Launch plan")),
                ("body", string("Meet with Alice about the launch.")),
                (
                    "labels",
                    ScalarValue::List(DocumentList::new(vec![
                        DocumentItem::Scalar(string("Important")),
                        DocumentItem::Scalar(string("Work")),
                    ])?),
                ),
            ],
        )?;
        insert_node(
            &mut graph,
            3,
            "Person",
            vec![
                ("name", string("Alice")),
                ("remoteID", string("hidden-remote-id")),
            ],
        )?;
        insert_node(&mut graph, 4, "Task", vec![])?;
        insert_node(
            &mut graph,
            5,
            "CalendarPlan",
            vec![("title", string("Launch meeting"))],
        )?;
        insert_edge(&mut graph, 1, 3, 4)?;
        let batch = semantic_texts(&graph)?;
        assert_eq!(batch.nodes.len(), 5);
        assert_eq!(batch.relationships.len(), 1);
        let text = batch
            .nodes
            .iter()
            .filter_map(|item| item.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n");
        for expected in [
            "Quarterly revenue",
            "Coffee",
            "amount: 42",
            "approved: true",
            "Launch plan",
            "Meet with Alice",
            "Important",
            "Alice",
            "Task",
            "Launch meeting",
        ] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
        assert!(!text.contains("hidden"));
        let edge = batch.relationships[0].text.as_deref().unwrap_or_default();
        assert!(edge.contains("Alice"));
        assert!(edge.contains("WORKS ON"));
        assert!(edge.contains("Responsible for the launch"));
        Ok(())
    }

    #[test]
    fn semantic_sparse_delta_preserves_complete_owner_prose_and_bounds_endpoint_content()
    -> Result<()> {
        let mut graph = GraphStore::default();
        let long_body = format!(
            "{} FINAL UNIQUE SENTENCE",
            "Long meaningful document prose. ".repeat(20_000)
        );
        for id in 1..=100 {
            insert_node(
                &mut graph,
                id,
                "Document",
                vec![
                    ("title", string(&format!("Document {id}"))),
                    ("body", string(&long_body)),
                    ("metadata", string(&"opaque".repeat(20_000))),
                ],
            )?;
        }
        insert_edge(&mut graph, 1, 1, 2)?;
        insert_edge(&mut graph, 2, 50, 51)?;
        let title = graph.catalog_mut().intern_property("title")?;
        let body = graph.catalog_mut().intern_property("body")?;
        let current = [GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property: title,
            value: string("Changed title"),
            revision: 2,
        }];
        let batch = semantic_text_delta(&graph, &[], &current)?;
        assert_eq!(
            batch
                .nodes
                .iter()
                .map(|item| item.entity_id)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(
            batch
                .relationships
                .iter()
                .map(|item| item.entity_id)
                .collect::<Vec<_>>(),
            vec![1]
        );
        let node = batch.nodes[0].text.as_deref().unwrap_or_default();
        assert!(node.contains(&long_body));
        let edge = batch.relationships[0].text.as_deref().unwrap_or_default();
        assert!(edge.contains("Changed title"));
        assert!(!edge.contains("FINAL UNIQUE SENTENCE"));
        assert!(edge.len() < 400);
        let body_only = [GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property: body,
            value: string("Changed prose"),
            revision: 2,
        }];
        let batch = semantic_text_delta(&graph, &[], &body_only)?;
        assert_eq!(batch.nodes.len(), 1);
        assert!(batch.relationships.is_empty());
        Ok(())
    }

    #[test]
    fn semantic_transaction_overlay_handles_new_schema_edges_renames_and_detach() -> Result<()> {
        let mut graph = GraphStore::default();
        insert_node(&mut graph, 1, "Person", vec![("name", string("Alice"))])?;
        insert_node(&mut graph, 2, "Task", vec![("title", string("Launch"))])?;
        insert_edge(&mut graph, 1, 1, 2)?;
        let name = graph.catalog_mut().intern_property("name")?;
        let kind = graph.catalog_mut().intern_relationship_type("OWNS")?;
        let prior = [GraphMutation::InsertEdge(EdgeInput {
            id: EdgeId(2),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: kind,
            layer: Layer::Observed,
            revision: 2,
            properties: vec![],
        })];
        let rename = [GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property: name,
            value: string("Alicia"),
            revision: 2,
        }];
        let batch = semantic_text_delta(&graph, &prior, &rename)?;
        assert_eq!(batch.relationships.len(), 2);
        assert!(batch.relationships.iter().all(|item| {
            item.text
                .as_deref()
                .is_some_and(|text| text.contains("Alicia") && !text.contains("Alice"))
        }));
        let detached = [GraphMutation::DeleteNode {
            node: NodeId(1),
            detach: true,
            revision: 2,
        }];
        let batch = semantic_text_delta(&graph, &prior, &detached)?;
        assert_eq!(
            batch.nodes,
            vec![SemanticText {
                entity_id: 1,
                text: None
            }]
        );
        assert_eq!(
            batch.relationships,
            vec![
                SemanticText {
                    entity_id: 1,
                    text: None
                },
                SemanticText {
                    entity_id: 2,
                    text: None
                }
            ]
        );
        let label = LabelId(99);
        let property = PropertyId(99);
        let new_node = [
            GraphMutation::DeclareLabel {
                name: "CalendarPlan".into(),
                id: label,
            },
            GraphMutation::DeclareProperty {
                name: "agenda".into(),
                id: property,
            },
            GraphMutation::InsertNode(NodeInput {
                id: NodeId(3),
                layer: Layer::Observed,
                revision: 2,
                labels: vec![label],
                properties: vec![(property, string("Discuss launch"))],
            }),
        ];
        let batch = semantic_text_delta(&graph, &[], &new_node)?;
        assert_eq!(
            batch.nodes[0].text.as_deref(),
            Some("Calendar Plan\nagenda: Discuss launch")
        );
        Ok(())
    }

    #[test]
    fn semantic_metadata_only_changes_do_not_reembed() -> Result<()> {
        let mut graph = GraphStore::default();
        insert_node(&mut graph, 1, "Person", vec![("name", string("Alice"))])?;
        let metadata = graph.catalog_mut().intern_property("syncRevision")?;
        let mutations = [GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property: metadata,
            value: ScalarValue::Integer(19),
            revision: 2,
        }];
        assert_eq!(
            semantic_text_delta(&graph, &[], &mutations)?,
            SemanticTextBatch::default()
        );
        Ok(())
    }

    #[test]
    fn semantic_calendar_dates_and_email_addresses_are_content() -> Result<()> {
        let mut graph = GraphStore::default();
        insert_node(
            &mut graph,
            1,
            "CalendarPlan",
            vec![
                ("title", string("Planning session")),
                (
                    "start",
                    ScalarValue::ZonedDateTime {
                        seconds: 0,
                        nanos: 0,
                        timezone: "Europe/Paris".into(),
                    },
                ),
                ("due_date", ScalarValue::Date(1)),
                (
                    "duration",
                    ScalarValue::Duration {
                        months: 0,
                        days: 0,
                        seconds: 3600,
                        nanos: 0,
                    },
                ),
                ("created_at", ScalarValue::Date(10)),
                ("email", string("alice@example.com")),
            ],
        )?;
        let batch = semantic_texts(&graph)?;
        let text = batch.nodes[0].text.as_deref().unwrap_or_default();
        for expected in [
            "1970-01-01T01:00:00+01:00",
            "Europe/Paris",
            "1970-01-02",
            "3600 seconds",
            "alice@example.com",
        ] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
        assert!(!text.contains("1970-01-11"));
        assert!(!text.contains("created"));
        Ok(())
    }

    #[test]
    fn semantic_sparse_delta_does_not_decode_unrelated_or_endpoint_body_values() -> Result<()> {
        let mut graph = GraphStore::default();
        insert_node(&mut graph, 1, "Person", vec![("name", string("Alice"))])?;
        insert_node(
            &mut graph,
            2,
            "Document",
            vec![
                ("title", string("Research")),
                (
                    "body",
                    ScalarValue::Map(DocumentMap::from_canonical(Arc::from([255_u8]))),
                ),
            ],
        )?;
        insert_node(
            &mut graph,
            3,
            "Document",
            vec![
                ("title", string("Unrelated")),
                (
                    "body",
                    ScalarValue::Map(DocumentMap::from_canonical(Arc::from([255_u8]))),
                ),
            ],
        )?;
        insert_edge(&mut graph, 1, 1, 2)?;
        let name = graph.catalog_mut().intern_property("name")?;
        let current = [GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property: name,
            value: string("Alicia"),
            revision: 2,
        }];
        // Positive control: these deliberately poisoned values fail if full content is decoded.
        assert!(semantic_texts(&graph).is_err());
        // The local change must not decode either the unrelated owner or its neighbor's body.
        let batch = semantic_text_delta(&graph, &[], &current)?;
        assert_eq!(batch.nodes.len(), 1);
        assert_eq!(batch.relationships.len(), 1);
        assert!(
            batch.relationships[0]
                .text
                .as_deref()
                .is_some_and(|text| text.contains("Alicia") && text.contains("Research"))
        );
        Ok(())
    }

    #[test]
    fn semantic_edge_updates_deletes_and_endpoint_labels_match_cold_state() -> Result<()> {
        let mut graph = GraphStore::default();
        insert_node(&mut graph, 1, "Person", vec![("name", string("Alice"))])?;
        insert_node(&mut graph, 2, "Task", vec![("title", string("Launch"))])?;
        insert_edge(&mut graph, 1, 1, 2)?;
        let role = graph.catalog_mut().intern_property("role")?;
        let participant = graph.catalog_mut().intern_label("Participant")?;
        let mutations = [
            GraphMutation::SetEdgeProperty {
                edge: EdgeId(1),
                property: role,
                value: string("Project lead"),
                revision: 2,
            },
            GraphMutation::AddNodeLabels {
                node: NodeId(1),
                labels: vec![participant],
                revision: 2,
            },
        ];
        let incremental = semantic_text_delta(&graph, &[], &mutations)?;
        for mutation in mutations {
            graph.apply(mutation)?;
        }
        let cold = semantic_texts(&graph)?;
        assert_eq!(incremental.relationships, cold.relationships);
        assert_eq!(incremental.nodes, vec![cold.nodes[0].clone()]);
        let deleted = semantic_text_delta(
            &graph,
            &[],
            &[GraphMutation::DeleteEdge {
                edge: EdgeId(1),
                revision: 3,
            }],
        )?;
        assert_eq!(
            deleted.relationships,
            vec![SemanticText {
                entity_id: 1,
                text: None
            }]
        );
        assert!(deleted.nodes.is_empty());
        Ok(())
    }
}
