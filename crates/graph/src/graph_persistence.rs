use super::*;
use crate::persistence::{PersistenceOp, corrupt, decode, ids, key, put};

#[derive(Default)]
struct Selected {
    prior_row: Option<u32>,
    replace: bool,
    properties: BTreeSet<PropertyId>,
}

/// Changed record identities captured before ordered publication; never contains graph values.
#[derive(Default)]
pub struct GraphPersistenceSelection {
    nodes: BTreeMap<NodeId, Selected>,
    edges: BTreeMap<EdgeId, Selected>,
    schema: BTreeMap<(u8, u64), String>,
}

impl GraphStore {
    /// Capture affected IDs before applying a mutation batch, including detach-delete edges.
    pub fn prepare_persistence(
        &self,
        mutations: &[GraphMutation],
    ) -> Result<GraphPersistenceSelection> {
        let mut result = GraphPersistenceSelection::default();
        for mutation in mutations {
            match mutation {
                GraphMutation::DeclareLabel { name, id } => {
                    result.schema.insert((0, id.0), name.clone());
                }
                GraphMutation::DeclareProperty { name, id } => {
                    result.schema.insert((1, id.0), name.clone());
                }
                GraphMutation::DeclareRelationshipType { name, id } => {
                    result.schema.insert((2, id.0), name.clone());
                }
                GraphMutation::InsertNode(input) => {
                    let item = result.nodes.entry(input.id).or_default();
                    item.replace = true;
                    item.properties
                        .extend(input.properties.iter().map(|(p, _)| *p));
                }
                GraphMutation::InsertEdge(input) => {
                    let item = result.edges.entry(input.id).or_default();
                    item.replace = true;
                    item.properties
                        .extend(input.properties.iter().map(|(p, _)| *p));
                }
                GraphMutation::SetNodeProperty { node, property, .. } => {
                    result
                        .nodes
                        .entry(*node)
                        .or_default()
                        .properties
                        .insert(*property);
                }
                GraphMutation::SetEdgeProperty { edge, property, .. } => {
                    result
                        .edges
                        .entry(*edge)
                        .or_default()
                        .properties
                        .insert(*property);
                }
                GraphMutation::AddNodeLabels { node, .. }
                | GraphMutation::RemoveNodeLabels { node, .. } => {
                    result.nodes.entry(*node).or_default();
                }
                GraphMutation::DeleteNode { node, detach, .. } => {
                    result.nodes.entry(*node).or_default().replace = true;
                    if *detach && self.node(*node).is_some() {
                        for edge in self.incident_edge_ids(*node)? {
                            result.edges.entry(edge).or_default().replace = true;
                        }
                    }
                }
                GraphMutation::DeleteEdge { edge, .. } => {
                    result.edges.entry(*edge).or_default().replace = true;
                }
            }
        }
        for (id, item) in &mut result.nodes {
            item.prior_row = self.0.node_lookup.pin().get(&id.0).copied();
        }
        for (id, item) in &mut result.edges {
            item.prior_row = self.0.edge_lookup.pin().get(&id.0).copied();
        }
        Ok(result)
    }

    /// Encode only selected current rows and properties after ordered application.
    pub fn persistence_changes(
        &self,
        selection: &GraphPersistenceSelection,
    ) -> Result<Vec<PersistenceOp>> {
        let mut output = Vec::new();
        for ((kind, id), name) in &selection.schema {
            output.push(put(key(&[b'g', b'/', b'0', *kind], &[*id]), name)?);
        }
        let mut nodes = BTreeMap::<u32, (bool, BTreeSet<PropertyId>)>::new();
        let mut edges = BTreeMap::<u32, (bool, BTreeSet<PropertyId>)>::new();
        for (id, item) in &selection.nodes {
            for row in item
                .prior_row
                .into_iter()
                .chain(self.0.node_lookup.pin().get(&id.0).copied())
            {
                let selected = nodes.entry(row).or_default();
                selected.0 |= item.replace;
                selected.1.extend(&item.properties);
            }
        }
        for (id, item) in &selection.edges {
            for row in item
                .prior_row
                .into_iter()
                .chain(self.0.edge_lookup.pin().get(&id.0).copied())
            {
                let selected = edges.entry(row).or_default();
                selected.0 |= item.replace;
                selected.1.extend(&item.properties);
            }
        }
        for (row, (replace, properties)) in nodes {
            output.push(put(
                key(b"g/1", &[u64::from(row)]),
                &NodeWire {
                    input: StoredNodeInput {
                        id: NodeId(load(&self.0.node_ids, row)),
                        layer: layer(load(&self.0.node_layers, row)),
                        revision: load(&self.0.node_revisions, row),
                        labels: self.labels(row).to_vec(),
                        properties: Vec::new(),
                    },
                    active: self.node_active(row),
                },
            )?);
            let prefix = key(b"g/3", &[u64::from(row)]);
            if replace {
                output.push(PersistenceOp::DeletePrefix {
                    prefix: prefix.clone(),
                });
            }
            let values = if replace {
                self.0.node_properties.values(row)
            } else {
                properties
                    .iter()
                    .filter_map(|property| {
                        self.0
                            .node_properties
                            .get(row, *property)
                            .map(|value| (*property, value))
                    })
                    .collect()
            };
            for (property, value) in values {
                output.push(PersistenceOp::Put {
                    key: key(&prefix, &[property.0]),
                    value: super::super::columns::mixed_payload(&value)?.unwrap_or_default(),
                });
            }
            if !replace {
                for property in properties {
                    if self.0.node_properties.get(row, property).is_none() {
                        output.push(PersistenceOp::Delete {
                            key: key(&prefix, &[property.0]),
                        });
                    }
                }
            }
        }
        for (row, (replace, properties)) in edges {
            let source = load(&self.0.edge_sources, row) as u32;
            let target = load(&self.0.edge_targets, row) as u32;
            output.push(put(
                key(b"g/2", &[u64::from(row)]),
                &EdgeWire {
                    input: StoredEdgeInput {
                        id: EdgeId(load(&self.0.edge_ids, row)),
                        source: NodeId(load(&self.0.node_ids, source)),
                        target: NodeId(load(&self.0.node_ids, target)),
                        relationship_type: RelationshipTypeId(load(&self.0.edge_types, row)),
                        layer: layer(load(&self.0.edge_layers, row)),
                        revision: load(&self.0.edge_revisions, row),
                        properties: Vec::new(),
                    },
                    source,
                    target,
                    active: self.edge_active(row),
                },
            )?);
            let prefix = key(b"g/4", &[u64::from(row)]);
            if replace {
                output.push(PersistenceOp::DeletePrefix {
                    prefix: prefix.clone(),
                });
            }
            let values = if replace {
                self.0.edge_properties.values(row)
            } else {
                properties
                    .iter()
                    .filter_map(|property| {
                        self.0
                            .edge_properties
                            .get(row, *property)
                            .map(|value| (*property, value))
                    })
                    .collect()
            };
            for (property, value) in values {
                output.push(PersistenceOp::Put {
                    key: key(&prefix, &[property.0]),
                    value: super::super::columns::mixed_payload(&value)?.unwrap_or_default(),
                });
            }
            if !replace {
                for property in properties {
                    if self.0.edge_properties.get(row, property).is_none() {
                        output.push(PersistenceOp::Delete {
                            key: key(&prefix, &[property.0]),
                        });
                    }
                }
            }
        }
        if !selection.nodes.is_empty()
            || !selection.edges.is_empty()
            || !selection.schema.is_empty()
        {
            output.push(put(
                b"g/9".to_vec(),
                &(self.revision(), self.layout_version()),
            )?);
        }
        Ok(output)
    }

    /// Restore one ordered current-state record into this canonical graph allocation.
    pub fn restore_persistence_record(&self, record_key: &[u8], value: &[u8]) -> Result<bool> {
        if !record_key.starts_with(b"g/") {
            return Ok(false);
        }
        match record_key.get(2).copied() {
            Some(b'0') => {
                let id = ids(
                    record_key
                        .get(4..)
                        .ok_or_else(|| corrupt("truncated schema key"))?,
                    1,
                )?[0];
                let name: String = decode(value)?;
                match record_key.get(3) {
                    Some(0) => self.catalog().declare_label(name, LabelId(id))?,
                    Some(1) => self.catalog().declare_property(name, PropertyId(id))?,
                    Some(2) => self
                        .catalog()
                        .declare_relationship_type(name, RelationshipTypeId(id))?,
                    _ => return Err(corrupt("invalid schema record kind")),
                }
            }
            Some(b'1') => {
                let row = ids(&record_key[3..], 1)?[0];
                if row != self.node_slot_count() as u64 {
                    return Err(corrupt("node records are not contiguous"));
                }
                let wire: NodeWire = decode(value)?;
                if !wire.input.properties.is_empty() {
                    return Err(corrupt("node header contains property payload"));
                }
                if self.contains_node_id(wire.input.id) {
                    return Err(corrupt("duplicate persisted node identity"));
                }
                self.append_node(wire.input.into(), wire.active)?;
            }
            Some(b'2') => {
                let row = ids(&record_key[3..], 1)?[0];
                if row != self.edge_slot_count() as u64 {
                    return Err(corrupt("relationship records are not contiguous"));
                }
                let wire: EdgeWire = decode(value)?;
                if !wire.input.properties.is_empty() {
                    return Err(corrupt("relationship header contains property payload"));
                }
                if self.contains_edge_id(wire.input.id) {
                    return Err(corrupt("duplicate persisted relationship identity"));
                }
                self.append_edge(wire.input.into(), wire.source, wire.target, wire.active)?;
            }
            Some(b'3') | Some(b'4') => {
                let id = ids(&record_key[3..], 2)?;
                let row = u32::try_from(id[0]).map_err(|_| corrupt("invalid property row"))?;
                let property = PropertyId(id[1]);
                if self.catalog().property_name(property).is_none() {
                    return Err(corrupt("persisted property is undeclared"));
                }
                let scalar = if value.is_empty() {
                    ScalarValue::Null
                } else {
                    super::super::columns::mixed_value(value)?
                };
                if record_key[2] == b'3' {
                    if !self.node_active(row) {
                        return Err(corrupt("property belongs to inactive node"));
                    }
                    self.0.node_properties.set(row, property, scalar)?;
                } else {
                    if !self.edge_active(row) {
                        return Err(corrupt("property belongs to inactive relationship"));
                    }
                    self.0.edge_properties.set(row, property, scalar)?;
                }
            }
            Some(b'9') if record_key == b"g/9" => {
                let (revision, layout): (u64, u64) = decode(value)?;
                if revision < self.revision() {
                    return Err(corrupt("graph revision precedes persisted row"));
                }
                self.0.revision.store(revision, Ordering::Release);
                self.0.layout.store(layout, Ordering::Release);
            }
            _ => return Err(corrupt("unknown graph persistence record")),
        }
        Ok(true)
    }

    /// Validate restored structure and make imported deleted slots available for reuse.
    pub fn finish_persistence_restore(&self) -> Result<()> {
        self.recycle_imported_tombstones();
        self.validate_structure()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::apply_records;

    fn publish(
        graph: &GraphStore,
        records: &mut BTreeMap<Vec<u8>, Vec<u8>>,
        mutation: GraphMutation,
    ) -> Result<Vec<PersistenceOp>> {
        let selection = graph.prepare_persistence(std::slice::from_ref(&mutation))?;
        graph.apply(mutation)?;
        let operations = graph.persistence_changes(&selection)?;
        apply_records(records, operations.clone());
        Ok(operations)
    }
    fn restore(records: &BTreeMap<Vec<u8>, Vec<u8>>) -> Result<GraphStore> {
        let graph = GraphStore::default();
        for (key, value) in records {
            assert!(graph.restore_persistence_record(key, value)?);
        }
        graph.finish_persistence_restore()?;
        Ok(graph)
    }

    #[test]
    fn dirty_property_does_not_encode_large_untouched_values() -> Result<()> {
        let graph = GraphStore::default();
        let mut records = BTreeMap::new();
        for (id, name) in [(1, "body"), (2, "rank")] {
            publish(
                &graph,
                &mut records,
                GraphMutation::DeclareProperty {
                    name: name.into(),
                    id: PropertyId(id),
                },
            )?;
        }
        for id in 1..=1000 {
            publish(
                &graph,
                &mut records,
                GraphMutation::InsertNode(NodeInput {
                    id: NodeId(id),
                    layer: Layer::Observed,
                    revision: 1,
                    labels: Vec::new(),
                    properties: if id == 1 {
                        vec![(
                            PropertyId(1),
                            ScalarValue::String("x".repeat(4 * 1024 * 1024).into()),
                        )]
                    } else {
                        Vec::new()
                    },
                }),
            )?;
        }
        let changes = publish(
            &graph,
            &mut records,
            GraphMutation::SetNodeProperty {
                node: NodeId(1),
                property: PropertyId(2),
                value: ScalarValue::Integer(7),
                revision: 2,
            },
        )?;
        let bytes: usize = changes
            .iter()
            .map(|op| match op {
                PersistenceOp::Put { key, value } => key.len() + value.len(),
                PersistenceOp::Delete { key } => key.len(),
                PersistenceOp::DeletePrefix { prefix } => prefix.len(),
            })
            .sum();
        assert!(bytes < 1024, "dirty property encoded {bytes} bytes");
        assert_eq!(changes.len(), 3);
        let loaded = restore(&records)?;
        assert_eq!(loaded.node_count(), 1000);
        assert_eq!(
            loaded
                .node(NodeId(1))
                .and_then(|node| node.property(PropertyId(2))),
            Some(ScalarValue::Integer(7))
        );
        assert_eq!(
            loaded
                .node(NodeId(1))
                .and_then(|node| node.property(PropertyId(1)))
                .and_then(|value| match value {
                    ScalarValue::String(s) => Some(s.len()),
                    _ => None,
                }),
            Some(4 * 1024 * 1024)
        );
        Ok(())
    }

    #[test]
    fn detach_delete_and_recycled_slots_restore_exact_ordinals() -> Result<()> {
        let graph = GraphStore::default();
        let mut records = BTreeMap::new();
        publish(
            &graph,
            &mut records,
            GraphMutation::DeclareProperty {
                name: "value".into(),
                id: PropertyId(1),
            },
        )?;
        publish(
            &graph,
            &mut records,
            GraphMutation::DeclareRelationshipType {
                name: "LINK".into(),
                id: RelationshipTypeId(1),
            },
        )?;
        for id in 1..=2 {
            publish(
                &graph,
                &mut records,
                GraphMutation::InsertNode(NodeInput {
                    id: NodeId(id),
                    layer: Layer::Observed,
                    revision: 1,
                    labels: Vec::new(),
                    properties: vec![(PropertyId(1), ScalarValue::Integer(id as i64))],
                }),
            )?;
        }
        publish(
            &graph,
            &mut records,
            GraphMutation::InsertEdge(EdgeInput {
                id: EdgeId(1),
                source: NodeId(1),
                target: NodeId(2),
                relationship_type: RelationshipTypeId(1),
                layer: Layer::Observed,
                revision: 2,
                properties: vec![(PropertyId(1), ScalarValue::Integer(55))],
            }),
        )?;
        publish(
            &graph,
            &mut records,
            GraphMutation::DeleteNode {
                node: NodeId(1),
                detach: true,
                revision: 3,
            },
        )?;
        let deleted = restore(&records)?;
        assert!(deleted.node(NodeId(1)).is_none());
        assert!(deleted.edge(EdgeId(1)).is_none());
        assert_eq!(deleted.node_slot_count(), 2);
        assert_eq!(deleted.edge_slot_count(), 1);
        publish(
            &graph,
            &mut records,
            GraphMutation::InsertNode(NodeInput {
                id: NodeId(3),
                layer: Layer::Knowledge,
                revision: 4,
                labels: Vec::new(),
                properties: Vec::new(),
            }),
        )?;
        let loaded = restore(&records)?;
        assert_eq!(
            loaded.node(NodeId(3)).map(|node| node.dense()),
            graph.node(NodeId(3)).map(|node| node.dense())
        );
        assert!(
            loaded
                .node(NodeId(3))
                .and_then(|node| node.property(PropertyId(1)))
                .is_none()
        );
        assert_eq!(
            loaded
                .node(NodeId(2))
                .and_then(|node| node.property(PropertyId(1))),
            Some(ScalarValue::Integer(2))
        );
        Ok(())
    }

    #[test]
    fn malformed_record_keys_fail_without_panicking() -> Result<()> {
        let graph = GraphStore::default();
        for key in [b"g/0".as_slice(), b"g/1", b"g/3", b"g/x"] {
            assert!(graph.restore_persistence_record(key, &[]).is_err());
        }
        Ok(())
    }
}
