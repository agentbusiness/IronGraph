use super::*;
use crate::persistence::{PersistenceOp, corrupt, decode, ids, key, put};

/// Identities needed to persist changed physical index memberships and canonical vectors.
#[derive(Default)]
pub struct IndexPersistenceSelection {
    nodes: BTreeMap<crate::NodeId, Option<u32>>,
    owners: BTreeSet<u64>,
    vector_rows: BTreeMap<(PropertyId, u64), Option<usize>>,
    runtimes: BTreeMap<String, usize>,
    descriptors: BTreeMap<String, (GraphIndexDefinition, DerivedIndexState, Option<String>)>,
    scalar_properties: BTreeMap<crate::NodeId, BTreeSet<PropertyId>>,
    scalar_all: BTreeSet<crate::NodeId>,
    vector_properties: BTreeSet<PropertyId>,
}

#[derive(Serialize, Deserialize)]
struct EntryWire {
    definition: GraphIndexDefinition,
    state: DerivedIndexState,
    diagnostic: Option<String>,
    present: bool,
    approximate: Option<Arc<IvfPqIndex>>,
}
#[derive(Serialize, Deserialize)]
enum Membership {
    Key(IndexKey),
    Text(Vec<String>),
}
#[derive(Serialize, Deserialize)]
struct VectorMetadata {
    dimension: usize,
    similarity: Similarity,
    dtype: EmbeddingDType,
    sequence: u64,
    built_slots: usize,
}
#[derive(Serialize, Deserialize)]
struct VectorRow {
    entity: u64,
    deleted_revision: u64,
    payload: Option<Arc<VectorPayload>>,
    dirty: Option<u64>,
}

fn named(prefix: &[u8], name: &str) -> Vec<u8> {
    let mut bytes = key(prefix, &[name.len() as u64]);
    bytes.extend_from_slice(name.as_bytes());
    bytes
}
fn name_and_tail(bytes: &[u8]) -> Result<(String, &[u8])> {
    if bytes.len() < 8 {
        return Err(corrupt("truncated index name key"));
    }
    let n = usize::try_from(ids(&bytes[..8], 1)?[0])
        .map_err(|_| corrupt("index name length overflow"))?;
    let end = 8usize
        .checked_add(n)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| corrupt("truncated index name"))?;
    let name = std::str::from_utf8(&bytes[8..end])
        .map_err(|_| corrupt("invalid index name"))?
        .to_owned();
    Ok((name, &bytes[end..]))
}

impl IndexCatalog {
    /// Include an indirectly changed owner, such as relationship freshness advanced by an endpoint.
    /// Only existing vector-column row identities are captured; no coordinates are copied.
    pub fn include_persistence_owner(&self, selected: &mut IndexPersistenceSelection, owner: u64) {
        selected.owners.insert(owner);
        for (property, matrix) in self.vector_columns.iter() {
            selected
                .vector_rows
                .insert((property, owner), matrix.store.rows.get(&owner));
        }
    }

    /// Capture row identities before applying graph/vector mutations; no payloads are cloned.
    pub fn prepare_persistence(
        &self,
        graph: &GraphStore,
        mutations: &[GraphMutation],
        vectors: &[ResolvedVectorMutation],
    ) -> Result<IndexPersistenceSelection> {
        let mut selected = IndexPersistenceSelection::default();
        for mutation in mutations {
            match mutation {
                GraphMutation::InsertNode(node) => {
                    selected.nodes.entry(node.id).or_insert(None);
                    selected.owners.insert(node.id.0);
                    selected.scalar_all.insert(node.id);
                }
                GraphMutation::SetNodeProperty { node, property, .. } => {
                    selected
                        .nodes
                        .insert(*node, graph.node(*node).map(|node| node.dense()));
                    selected.owners.insert(node.0);
                    selected
                        .scalar_properties
                        .entry(*node)
                        .or_default()
                        .insert(*property);
                }
                GraphMutation::AddNodeLabels { node, .. }
                | GraphMutation::RemoveNodeLabels { node, .. }
                | GraphMutation::DeleteNode { node, .. } => {
                    selected
                        .nodes
                        .insert(*node, graph.node(*node).map(|node| node.dense()));
                    selected.owners.insert(node.0);
                    selected.scalar_all.insert(*node);
                    if matches!(mutation, GraphMutation::DeleteNode { detach: true, .. })
                        && graph.node(*node).is_some()
                    {
                        selected
                            .owners
                            .extend(graph.incident_edge_ids(*node)?.into_iter().map(|id| id.0));
                    }
                }
                GraphMutation::InsertEdge(edge) => {
                    selected.owners.insert(edge.id.0);
                }
                GraphMutation::SetEdgeProperty { edge, .. }
                | GraphMutation::DeleteEdge { edge, .. } => {
                    selected.owners.insert(edge.0);
                }
                _ => {}
            }
        }
        for mutation in vectors {
            match mutation {
                ResolvedVectorMutation::Upsert {
                    property,
                    entity_id,
                    ..
                }
                | ResolvedVectorMutation::UpsertSpans {
                    property,
                    entity_id,
                    ..
                }
                | ResolvedVectorMutation::Remove {
                    property,
                    entity_id,
                    ..
                } => {
                    selected.owners.insert(*entity_id);
                    selected.vector_rows.insert(
                        (*property, *entity_id),
                        self.vector_columns
                            .get(property)
                            .and_then(|matrix| matrix.store.rows.get(entity_id)),
                    );
                }
            }
        }
        for (property, matrix) in self.vector_columns.iter() {
            selected.vector_properties.insert(property);
            for owner in &selected.owners {
                selected
                    .vector_rows
                    .insert((property, *owner), matrix.store.rows.get(owner));
            }
        }
        for (name, entry) in self.entries.iter() {
            selected.runtimes.insert(
                name.clone(),
                entry
                    .runtime
                    .as_ref()
                    .map_or(0, |runtime| Arc::as_ptr(runtime) as usize),
            );
            selected.descriptors.insert(
                name,
                (
                    entry.definition.clone(),
                    entry.state,
                    entry.diagnostic.clone(),
                ),
            );
        }
        Ok(selected)
    }

    /// Encode selected memberships and vector rows. A changed built runtime exports only its own base.
    pub fn persistence_changes(
        &self,
        graph: &GraphStore,
        selected: &IndexPersistenceSelection,
        catalog_changed: bool,
    ) -> Result<Vec<PersistenceOp>> {
        let mut output = Vec::new();
        let mut cold_vectors = BTreeSet::new();
        let changed_runtime = self.entries.iter().any(|(name, entry)| {
            selected.runtimes.get(&name).is_none_or(|old| {
                *old != entry
                    .runtime
                    .as_ref()
                    .map_or(0, |runtime| Arc::as_ptr(runtime) as usize)
            })
        });
        if catalog_changed || changed_runtime {
            output.push(put(
                b"i/5".to_vec(),
                self.relationship_semantic.get().as_ref(),
            )?);
            output.push(put(
                b"i/0".to_vec(),
                &(
                    self.profile.get(),
                    self.embeddings.iter().collect::<Vec<_>>(),
                ),
            )?);
        }
        for name in selected.runtimes.keys() {
            if !self.entries.contains_key(name) {
                output.push(PersistenceOp::Delete {
                    key: named(b"i/1", name),
                });
                output.push(PersistenceOp::DeletePrefix {
                    prefix: named(b"i/2", name),
                });
            }
        }
        for (name, entry) in self.entries.iter() {
            let pointer = entry
                .runtime
                .as_ref()
                .map_or(0, |runtime| Arc::as_ptr(runtime) as usize);
            let cold = selected
                .runtimes
                .get(&name)
                .is_none_or(|old| *old != pointer);
            let descriptor_changed =
                selected
                    .descriptors
                    .get(&name)
                    .is_none_or(|(definition, state, diagnostic)| {
                        *definition != entry.definition
                            || *state != entry.state
                            || *diagnostic != entry.diagnostic
                    });
            if cold || descriptor_changed {
                let approximate = match entry.runtime.as_deref() {
                    Some(RuntimeIndex::Vector { approximate, .. }) => approximate.clone(),
                    _ => None,
                };
                output.push(put(
                    named(b"i/1", &name),
                    &EntryWire {
                        definition: entry.definition.clone(),
                        state: entry.state,
                        diagnostic: entry.diagnostic.clone(),
                        present: entry.runtime.is_some(),
                        approximate,
                    },
                )?);
            }
            match entry.runtime.as_deref() {
                Some(RuntimeIndex::Vector { property, .. }) => {
                    if cold {
                        cold_vectors.insert(*property);
                    }
                }
                Some(_) => {
                    let prefix = named(b"i/2", &name);
                    let mut rows = BTreeSet::new();
                    if cold {
                        output.push(PersistenceOp::DeletePrefix {
                            prefix: prefix.clone(),
                        });
                        rows.extend(graph.nodes().map(|node| node.dense()));
                    }
                    for (id, old) in &selected.nodes {
                        if selected.scalar_all.contains(id)
                            || selected
                                .scalar_properties
                                .get(id)
                                .is_some_and(|properties| {
                                    entry
                                        .definition
                                        .properties
                                        .iter()
                                        .any(|p| properties.contains(p))
                                })
                        {
                            rows.extend(*old);
                            rows.extend(graph.node(*id).map(|node| node.dense()));
                        }
                    }
                    for row in rows {
                        let record = key(&prefix, &[u64::from(row)]);
                        let membership = graph
                            .node_dense(row)
                            .filter(|node| node.labels().contains(&entry.definition.label))
                            .map(|node| -> Result<Option<Membership>> {
                                Ok(match entry.runtime.as_deref() {
                                    Some(RuntimeIndex::Equality(_))
                                    | Some(RuntimeIndex::Range(_)) => {
                                        composite_key(node, &entry.definition.properties)?
                                            .map(Membership::Key)
                                    }
                                    Some(RuntimeIndex::Text(index)) => {
                                        index.rows.get(&row).map(|terms| {
                                            Membership::Text(terms.iter().cloned().collect())
                                        })
                                    }
                                    _ => None,
                                })
                            })
                            .transpose()?
                            .flatten();
                        if let Some(value) = membership {
                            output.push(put(record, &value)?);
                        } else {
                            output.push(PersistenceOp::Delete { key: record });
                        }
                    }
                }
                None => {}
            }
        }
        for property in &selected.vector_properties {
            if !self.vector_columns.contains_key(property) {
                output.push(PersistenceOp::Delete {
                    key: key(b"i/3", &[property.0]),
                });
                output.push(PersistenceOp::DeletePrefix {
                    prefix: key(b"i/4", &[property.0]),
                });
            }
        }
        for (property, matrix) in self.vector_columns.iter() {
            let cold =
                cold_vectors.contains(&property) || !selected.vector_properties.contains(&property);
            let mut rows = BTreeSet::new();
            if cold {
                rows.extend(0..matrix.row_count());
                output.push(PersistenceOp::DeletePrefix {
                    prefix: key(b"i/4", &[property.0]),
                });
            }
            for owner in &selected.owners {
                rows.extend(
                    selected
                        .vector_rows
                        .get(&(property, *owner))
                        .copied()
                        .flatten(),
                );
                rows.extend(matrix.store.rows.get(owner));
            }
            if cold || !rows.is_empty() || catalog_changed {
                output.push(put(
                    key(b"i/3", &[property.0]),
                    &VectorMetadata {
                        dimension: matrix.dimension,
                        similarity: matrix.similarity,
                        dtype: matrix.dtype,
                        sequence: matrix
                            .store
                            .sequence
                            .load(std::sync::atomic::Ordering::Acquire),
                        built_slots: self.semantic_built_slots.get(&property).unwrap_or(0),
                    },
                )?);
            }
            for row in rows {
                let slot = matrix
                    .store
                    .slots
                    .get(row)
                    .ok_or_else(|| corrupt("selected vector row missing"))?;
                output.push(put(
                    key(b"i/4", &[property.0, row as u64]),
                    &VectorRow {
                        entity: slot.entity_id.load(std::sync::atomic::Ordering::Acquire),
                        deleted_revision: slot
                            .deleted_revision
                            .load(std::sync::atomic::Ordering::Acquire),
                        payload: slot.payload.load_full(),
                        dirty: matrix.store.dirty.get(&row),
                    },
                )?);
            }
        }
        Ok(output)
    }

    /// Restore one ordered record into existing index types. Call finish after all graph records.
    pub fn restore_persistence_record(&self, record_key: &[u8], value: &[u8]) -> Result<bool> {
        if !record_key.starts_with(b"i/") {
            return Ok(false);
        }
        match record_key.get(2).copied() {
            Some(b'5') if record_key == b"i/5" => {
                let state: RelationshipSemanticState = decode(value)?;
                if state.reconciled_revision > state.revision {
                    return Err(corrupt(
                        "relationship scope completion exceeds its revision",
                    ));
                }
                self.relationship_semantic.restore(state);
            }
            Some(b'0') if record_key == b"i/0" => {
                let (profile, definitions): (
                    Option<EmbeddingProfile>,
                    Vec<(String, EmbeddingIndexDefinition)>,
                ) = decode(value)?;
                if let Some(profile) = profile {
                    self.profile.set(profile);
                }
                for (name, definition) in definitions {
                    if name != definition.name {
                        return Err(corrupt("embedding definition name differs"));
                    }
                    self.embeddings.insert(name, definition);
                }
            }
            Some(b'1') => {
                let (name, tail) = name_and_tail(&record_key[3..])?;
                if !tail.is_empty() {
                    return Err(corrupt("index descriptor key has suffix"));
                }
                let wire: EntryWire = decode(value)?;
                if name != wire.definition.name {
                    return Err(corrupt("index definition name differs"));
                }
                let runtime = if wire.present {
                    Some(Arc::new(match wire.definition.kind {
                        GraphIndexKind::Equality => {
                            RuntimeIndex::Equality(EqualityIndex::default())
                        }
                        GraphIndexKind::Range => RuntimeIndex::Range(RangeIndex::default()),
                        GraphIndexKind::Text => RuntimeIndex::Text(TextIndex::default()),
                        GraphIndexKind::Vector => RuntimeIndex::Vector {
                            property: *wire
                                .definition
                                .properties
                                .first()
                                .ok_or_else(|| corrupt("vector index property missing"))?,
                            approximate: wire.approximate,
                        },
                    }))
                } else {
                    None
                };
                self.entries.insert(
                    name,
                    Arc::new(IndexEntry {
                        definition: wire.definition,
                        state: wire.state,
                        diagnostic: wire.diagnostic,
                        runtime,
                    }),
                );
            }
            Some(b'2') => {
                let (name, tail) = name_and_tail(&record_key[3..])?;
                let row = u32::try_from(ids(tail, 1)?[0])
                    .map_err(|_| corrupt("index row exceeds ordinal width"))?;
                let membership: Membership = decode(value)?;
                let entry = self
                    .entries
                    .get(&name)
                    .ok_or_else(|| corrupt("index membership has no definition"))?;
                match (entry.runtime.as_deref(), membership) {
                    (Some(RuntimeIndex::Equality(index)), Membership::Key(key)) => {
                        index.insert_key(key, row)
                    }
                    (Some(RuntimeIndex::Range(index)), Membership::Key(key)) => {
                        index.insert_key(key, row)
                    }
                    (Some(RuntimeIndex::Text(index)), Membership::Text(terms)) => {
                        let terms: BTreeSet<_> = terms.into_iter().collect();
                        for term in &terms {
                            index.postings.insert(term.clone(), row);
                        }
                        index.rows.insert(row, Arc::new(terms));
                    }
                    _ => return Err(corrupt("index membership differs from runtime kind")),
                }
            }
            Some(b'3') => {
                let property = PropertyId(ids(&record_key[3..], 1)?[0]);
                let wire: VectorMetadata = decode(value)?;
                let matrix =
                    VectorIndex::new_with_dtype(wire.dimension, wire.similarity, wire.dtype)?;
                matrix
                    .store
                    .sequence
                    .store(wire.sequence, std::sync::atomic::Ordering::Release);
                self.vector_columns.insert(property, Arc::new(matrix));
                self.semantic_built_slots.insert(property, wire.built_slots);
            }
            Some(b'4') => {
                let id = ids(&record_key[3..], 2)?;
                let property = PropertyId(id[0]);
                let row = usize::try_from(id[1])
                    .map_err(|_| corrupt("vector row exceeds platform width"))?;
                let wire: VectorRow = decode(value)?;
                let matrix = self
                    .vector_columns
                    .get(&property)
                    .ok_or_else(|| corrupt("vector row has no metadata"))?;
                if row != matrix.row_count() {
                    return Err(corrupt("vector rows are not contiguous"));
                }
                if let Some(payload) = &wire.payload {
                    if payload.entity_id != wire.entity {
                        return Err(corrupt("vector payload owner differs"));
                    }
                    match &payload.coordinates {
                        VectorCoordinates::Quantized(values) => {
                            matrix.validate_coordinates(values)?
                        }
                        VectorCoordinates::Spans(spans) => {
                            matrix.validate_spans(spans.iter().map(AsRef::as_ref))?
                        }
                        VectorCoordinates::PropertyOwner { .. } => {}
                    }
                    if matrix.store.rows.contains_key(&wire.entity) {
                        return Err(corrupt("duplicate canonical vector owner"));
                    }
                    matrix.store.multi_span_owners.fetch_add(
                        usize::from(payload.is_multi_span()),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    matrix.store.payload_bytes.fetch_add(
                        payload.resident_bytes(),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    matrix.store.rows.insert(wire.entity, row);
                    matrix
                        .store
                        .live
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                matrix.store.free.ensure(
                    u32::try_from(row).map_err(|_| corrupt("vector row ordinal overflow"))?,
                )?;
                let active = wire.payload.is_some();
                matrix.store.slots.push(VectorSlot {
                    entity_id: std::sync::atomic::AtomicU64::new(wire.entity),
                    deleted_revision: std::sync::atomic::AtomicU64::new(wire.deleted_revision),
                    payload: arc_swap::ArcSwapOption::from(wire.payload),
                })?;
                if !active {
                    matrix.store.free.release(row as u32);
                }
                if let Some(dirty) = wire.dirty {
                    matrix.store.dirty.insert(row, dirty);
                }
            }
            _ => return Err(corrupt("unknown index persistence record")),
        }
        Ok(true)
    }

    /// Bind restored owner references and rebuild only ephemeral retrieval helpers.
    pub fn finish_persistence_restore(&self, graph: &GraphStore) -> Result<()> {
        self.bind_catalog_graph(graph);
        for matrix in self.vector_columns.values() {
            matrix.rebuild_retrieval()?;
        }
        for entry in self.entries.values() {
            if let Some(RuntimeIndex::Vector {
                property,
                approximate: Some(ann),
            }) = entry.runtime.as_deref()
            {
                let matrix = self
                    .vector_columns
                    .get(property)
                    .ok_or_else(|| corrupt("ANN base has no vectors"))?;
                if ann.rows.len() != ann.built_versions.len()
                    || ann
                        .rows
                        .iter()
                        .any(|row| *row as usize >= matrix.row_count())
                    || ann.dimension != matrix.dimension
                    || ann.similarity != matrix.similarity
                {
                    return Err(corrupt("ANN base differs from canonical vector matrix"));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::apply_records;

    fn restore(graph: &GraphStore, records: &BTreeMap<Vec<u8>, Vec<u8>>) -> Result<IndexCatalog> {
        let catalog = IndexCatalog::default();
        for (key, value) in records {
            assert!(catalog.restore_persistence_record(key, value)?);
        }
        catalog.finish_persistence_restore(graph)?;
        Ok(catalog)
    }

    #[test]
    fn relationship_semantic_scope_persists_without_an_encoder_and_fences_completion() -> Result<()>
    {
        let graph = GraphStore::default();
        let catalog = IndexCatalog::default();
        assert_eq!(
            catalog.relationship_semantic_state(),
            (RelationshipSemanticScope::All, 0)
        );
        assert!(!catalog.relationship_semantic_reconciliation_pending());
        assert!(catalog.is_empty());
        let all_generation = catalog.optimizer_generation();
        assert!(!catalog.set_relationship_semantic_scope(RelationshipSemanticScope::All)?);
        assert!(catalog.set_relationship_semantic_scope(RelationshipSemanticScope::None)?);
        assert!(!catalog.is_empty());
        assert_ne!(catalog.optimizer_generation(), all_generation);
        let excluded = RelationshipSemanticScope::AllExcept(BTreeSet::from([
            "CALLS".to_owned(),
            "Future Type".to_owned(),
        ]));
        assert!(catalog.set_relationship_semantic_scope(excluded.clone())?);
        assert!(!catalog.complete_relationship_semantic_reconciliation(1));
        assert!(catalog.relationship_semantic_reconciliation_pending());
        let selected = catalog.prepare_persistence(&graph, &[], &[])?;
        let mut records = BTreeMap::new();
        apply_records(
            &mut records,
            catalog.persistence_changes(&graph, &selected, true)?,
        );
        let loaded = restore(&graph, &records)?;
        assert_eq!(loaded.relationship_semantic_state(), (excluded, 2));
        assert!(loaded.profile().is_none());
        assert!(!loaded.relationship_semantic_included("CALLS"));
        assert!(!loaded.relationship_semantic_included("Future Type"));
        assert!(loaded.relationship_semantic_included("calls"));
        assert!(loaded.relationship_semantic_reconciliation_pending());
        assert!(loaded.complete_relationship_semantic_reconciliation(2));
        assert!(loaded.complete_relationship_semantic_reconciliation(2));
        assert!(!loaded.relationship_semantic_reconciliation_pending());
        let selected = loaded.prepare_persistence(&graph, &[], &[])?;
        apply_records(
            &mut records,
            loaded.persistence_changes(&graph, &selected, true)?,
        );
        let loaded = restore(&graph, &records)?;
        assert!(!loaded.relationship_semantic_reconciliation_pending());
        assert!(loaded.set_relationship_semantic_scope(RelationshipSemanticScope::All)?);
        assert_eq!(loaded.relationship_semantic_revision(), 3);
        assert!(!loaded.complete_relationship_semantic_reconciliation(2));
        assert!(loaded.relationship_semantic_reconciliation_pending());
        Ok(())
    }

    #[test]
    fn relationship_scope_publication_keeps_scope_and_revision_together() -> Result<()> {
        let catalog = IndexCatalog::default();
        std::thread::scope(|threads| {
            let writer = threads.spawn(|| -> Result<()> {
                for revision in 1..=1_000 {
                    let scope = if revision % 2 == 0 {
                        RelationshipSemanticScope::All
                    } else {
                        RelationshipSemanticScope::None
                    };
                    assert!(catalog.set_relationship_semantic_scope(scope)?);
                    if revision % 3 == 0 {
                        catalog.complete_relationship_semantic_reconciliation(revision);
                    }
                }
                Ok(())
            });
            for _ in 0..10_000 {
                let (scope, revision) = catalog.relationship_semantic_state();
                assert_eq!(
                    scope,
                    if revision % 2 == 0 {
                        RelationshipSemanticScope::All
                    } else {
                        RelationshipSemanticScope::None
                    }
                );
            }
            writer
                .join()
                .map_err(|_| corrupt("scope publication writer failed"))?
        })?;
        assert_eq!(catalog.relationship_semantic_revision(), 1_000);
        assert!(!catalog.complete_relationship_semantic_reconciliation(999));
        assert!(catalog.complete_relationship_semantic_reconciliation(1_000));
        Ok(())
    }

    #[test]
    fn absent_relationship_scope_record_restores_all_and_clones_share_policy() -> Result<()> {
        let graph = GraphStore::default();
        let loaded = restore(&graph, &BTreeMap::new())?;
        let shared = loaded.clone();
        assert_eq!(
            loaded.relationship_semantic_state(),
            (RelationshipSemanticScope::All, 0)
        );
        shared.set_relationship_semantic_scope(RelationshipSemanticScope::None)?;
        assert_eq!(
            loaded.relationship_semantic_state(),
            (RelationshipSemanticScope::None, 1)
        );
        assert!(loaded.complete_relationship_semantic_reconciliation(1));
        assert!(!shared.relationship_semantic_reconciliation_pending());
        assert!(
            shared.set_relationship_semantic_scope(RelationshipSemanticScope::AllExcept(
                BTreeSet::new()
            ))?
        );
        assert_eq!(
            loaded.relationship_semantic_state(),
            (RelationshipSemanticScope::All, 2)
        );
        Ok(())
    }

    #[test]
    fn scalar_memberships_restore_and_dirty_updates_do_not_copy_postings() -> Result<()> {
        let graph = GraphStore::default();
        let label = graph.catalog().intern_label("Item")?;
        let property = graph.catalog().intern_property("rank")?;
        let body = graph.catalog().intern_property("body")?;
        for id in 1..=1000 {
            graph.insert_node(crate::NodeInput {
                id: crate::NodeId(id),
                layer: crate::Layer::Observed,
                revision: 1,
                labels: vec![label],
                properties: vec![
                    (property, ScalarValue::Integer(7)),
                    (body, ScalarValue::String("alpha beta".into())),
                ],
            })?;
        }
        let catalog = IndexCatalog::default();
        let selection = catalog.prepare_persistence(&graph, &[], &[])?;
        for (name, kind, property) in [
            ("equality", GraphIndexKind::Equality, property),
            ("range", GraphIndexKind::Range, property),
            ("text", GraphIndexKind::Text, body),
        ] {
            catalog.create(
                &graph,
                GraphIndexDefinition {
                    name: name.into(),
                    kind,
                    label,
                    properties: vec![property],
                    unique: false,
                },
            )?;
        }
        let mut records = BTreeMap::new();
        apply_records(
            &mut records,
            catalog.persistence_changes(&graph, &selection, true)?,
        );
        let mutation = GraphMutation::SetNodeProperty {
            node: crate::NodeId(1),
            property,
            value: ScalarValue::Integer(9),
            revision: 2,
        };
        let selection =
            catalog.prepare_persistence(&graph, std::slice::from_ref(&mutation), &[])?;
        catalog.before_graph_apply(&graph, &mutation)?;
        graph.apply(mutation.clone())?;
        catalog.after_graph_apply(&graph, &mutation)?;
        let changes = catalog.persistence_changes(&graph, &selection, false)?;
        let bytes: usize = changes
            .iter()
            .map(|op| match op {
                PersistenceOp::Put { value, .. } => value.len(),
                _ => 0,
            })
            .sum();
        assert!(
            bytes < 1024,
            "single membership update encoded {bytes} bytes"
        );
        apply_records(&mut records, changes);
        let loaded = restore(&graph, &records)?;
        let values = BTreeMap::from([(property, ScalarValue::Integer(9))]);
        assert_eq!(
            loaded.equality_candidates_named("equality", label, &values)?,
            Some(vec![0])
        );
        assert_eq!(
            loaded.equality_candidates_named("range", label, &values)?,
            Some(vec![0])
        );
        assert_eq!(loaded.bounded_text_candidates("alpha", 4)?.len(), 4);
        Ok(())
    }

    #[test]
    fn ann_base_spans_and_exact_delta_restore_without_retraining() -> Result<()> {
        let graph = GraphStore::default();
        for id in 1..=8 {
            graph.insert_node(crate::NodeInput {
                id: crate::NodeId(id),
                layer: crate::Layer::Observed,
                revision: 1,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        let catalog = IndexCatalog::default();
        let initial = catalog.prepare_persistence(&graph, &[], &[])?;
        let profile = EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            4,
            EmbeddingDType::F16,
            true,
            Similarity::Cosine,
        )?;
        catalog.initialize_semantic(profile)?;
        let matrix = catalog
            .vector_columns
            .get(&SEMANTIC_NODE_PROPERTY)
            .ok_or_else(|| corrupt("semantic matrix missing"))?;
        for id in 1..=8 {
            let vector = matrix.quantize(&[id as f32, 1.0, 0.5, 0.25])?;
            matrix.upsert_quantized(id, &vector, 1)?;
        }
        catalog.rebuild(&graph, SEMANTIC_NODE_INDEX)?;
        let base = catalog
            .vector_search_source(SEMANTIC_NODE_INDEX)
            .and_then(|(_, ann)| ann)
            .ok_or_else(|| corrupt("ANN base missing"))?;
        let mut records = BTreeMap::new();
        apply_records(
            &mut records,
            catalog.persistence_changes(&graph, &initial, true)?,
        );
        let mutation = ResolvedVectorMutation::UpsertSpans {
            property: SEMANTIC_NODE_PROPERTY,
            entity_id: 1,
            coordinates: vec![
                matrix.quantize(&[0.0, 1.0, 0.0, 0.0])?,
                matrix.quantize(&[0.0, 0.0, 1.0, 0.0])?,
            ],
            revision: 2,
        };
        let selection =
            catalog.prepare_persistence(&graph, &[], std::slice::from_ref(&mutation))?;
        catalog.apply_vector_mutation_from_graph(&graph, &mutation)?;
        let changes = catalog.persistence_changes(&graph, &selection, false)?;
        assert!(
            changes.iter().all(|op| match op {
                PersistenceOp::Put { key, .. } => !key.starts_with(b"i/1"),
                _ => true,
            }),
            "ordinary vector update rewrote ANN base"
        );
        apply_records(&mut records, changes);
        let loaded = restore(&graph, &records)?;
        let (source, ann) = loaded
            .vector_search_source(SEMANTIC_NODE_INDEX)
            .ok_or_else(|| corrupt("restored semantic source missing"))?;
        let ann = ann.ok_or_else(|| corrupt("restored ANN missing"))?;
        assert_eq!(ann.build_generation(), base.build_generation());
        assert_eq!(source.row_count(), 8);
        assert_eq!(source.store.dirty.get(&0), matrix.store.dirty.get(&0));
        assert_eq!(
            crate::persistence::encode(source.as_ref())?,
            crate::persistence::encode(matrix.as_ref())?
        );
        Ok(())
    }

    #[test]
    fn indirect_relationship_freshness_persists_without_graph_edge_mutation() -> Result<()> {
        let graph = GraphStore::default();
        let property = graph.catalog().intern_property("metadata")?;
        let relationship_type = graph.catalog().intern_relationship_type("LINK")?;
        for id in 1..=2 {
            graph.insert_node(crate::NodeInput {
                id: crate::NodeId(id),
                layer: crate::Layer::Observed,
                revision: 1,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        graph.insert_edge(crate::EdgeInput {
            id: crate::EdgeId(99),
            source: crate::NodeId(1),
            target: crate::NodeId(2),
            relationship_type,
            layer: crate::Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
        let catalog = IndexCatalog::default();
        let initial = catalog.prepare_persistence(&graph, &[], &[])?;
        catalog.initialize_semantic(EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            4,
            EmbeddingDType::F16,
            true,
            Similarity::Cosine,
        )?)?;
        let matrix = catalog
            .vector_columns
            .get(&SEMANTIC_RELATIONSHIP_PROPERTY)
            .ok_or_else(|| corrupt("relationship vectors missing"))?;
        matrix.upsert_quantized(99, &matrix.quantize(&[1.0, 0.0, 0.0, 0.0])?, 1)?;
        let mut records = BTreeMap::new();
        apply_records(
            &mut records,
            catalog.persistence_changes(&graph, &initial, true)?,
        );
        let mutation = GraphMutation::SetNodeProperty {
            node: crate::NodeId(1),
            property,
            value: ScalarValue::Integer(7),
            revision: 2,
        };
        let mut selection =
            catalog.prepare_persistence(&graph, std::slice::from_ref(&mutation), &[])?;
        graph.apply(mutation)?;
        matrix.advance_row_revision(99, 1, 2);
        catalog.include_persistence_owner(&mut selection, 99);
        apply_records(
            &mut records,
            catalog.persistence_changes(&graph, &selection, false)?,
        );
        let loaded = restore(&graph, &records)?;
        let (source, _) = loaded
            .vector_search_source(SEMANTIC_RELATIONSHIP_INDEX)
            .ok_or_else(|| corrupt("restored relationship vectors missing"))?;
        assert_eq!(source.row_revision(99), Some(2));
        Ok(())
    }
}
