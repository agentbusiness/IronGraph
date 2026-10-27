use super::*;
use crate::persistence::{PersistenceOp, corrupt, decode, ids, key, put};

/// Temporal row range and expiry identities captured before ordered mutation.
#[derive(Default)]
pub struct TemporalPersistenceSelection {
    columns: BTreeMap<Key, (u64, Vec<u64>)>,
}

fn column_key(prefix: &[u8], column: Key) -> Vec<u8> {
    key(prefix, &[u64::from(column.0), column.1, column.2.0])
}
fn parse_column(bytes: &[u8]) -> Result<Key> {
    let id = ids(bytes, 3)?;
    let kind = u8::try_from(id[0]).map_err(|_| corrupt("temporal entity kind is invalid"))?;
    if kind != EntityKind::Node as u8 && kind != EntityKind::Relationship as u8 {
        return Err(corrupt("temporal entity kind is invalid"));
    }
    Ok((kind, id[1], PropertyId(id[2])))
}

impl TemporalStore {
    /// Capture retention removals using the incoming declaration's horizon before a schema change.
    pub fn prepare_declaration_persistence(
        &self,
        declaration: &TemporalDeclaration,
        now: i64,
    ) -> Result<TemporalPersistenceSelection> {
        let key = Self::key(
            declaration.entity_kind,
            declaration.target,
            declaration.property,
        );
        let mut selection = self.prepare_persistence(
            &[(
                declaration.entity_kind,
                declaration.target,
                declaration.property,
            )],
            now,
        )?;
        if let Some(column) = self.column(key) {
            let oldest = now.saturating_sub(declaration.retention_nanos);
            let expired = column
                .expiry
                .lock()
                .map_err(|_| Error::internal("temporal expiry poisoned"))?
                .range(..(oldest, 0))
                .map(|((_, row), ())| *row)
                .collect();
            selection
                .columns
                .insert(key, (column.next_row.load(Ordering::Acquire), expired));
        }
        Ok(selection)
    }

    /// Capture only new-row cursors and retention-expired row IDs in the specified columns.
    pub fn prepare_persistence(
        &self,
        changed: &[(EntityKind, u64, PropertyId)],
        now: i64,
    ) -> Result<TemporalPersistenceSelection> {
        let mut selection = TemporalPersistenceSelection::default();
        for (kind, target, property) in changed {
            let key = Self::key(*kind, *target, *property);
            let (start, expired) = if let Some(column) = self.column(key) {
                let oldest = now.saturating_sub(column.declaration.load().retention_nanos);
                let expired = column
                    .expiry
                    .lock()
                    .map_err(|_| Error::internal("temporal expiry poisoned"))?
                    .range(..(oldest, 0))
                    .map(|((_, row), ())| *row)
                    .collect();
                (column.next_row.load(Ordering::Acquire), expired)
            } else {
                (0, Vec::new())
            };
            selection.columns.insert(key, (start, expired));
        }
        Ok(selection)
    }

    /// Encode only appended samples and expired deletions; catalog records are small definitions.
    pub fn persistence_changes(
        &self,
        selection: &TemporalPersistenceSelection,
        catalog_changed: bool,
    ) -> Result<Vec<PersistenceOp>> {
        let mut output = Vec::new();
        if catalog_changed {
            for (column_key_value, column) in self.columns() {
                output.push(put(
                    column_key(b"t/0", column_key_value),
                    &**column.declaration.load(),
                )?);
            }
            output.push(PersistenceOp::DeletePrefix {
                prefix: b"t/3".to_vec(),
            });
            for definition in self.rollup_definitions() {
                let mut record = b"t/3".to_vec();
                record.extend_from_slice(definition.name.as_bytes());
                output.push(put(record, &definition)?);
            }
        }
        for (column_key_value, (start, expired)) in &selection.columns {
            let Some(column) = self.column(*column_key_value) else {
                output.push(PersistenceOp::Delete {
                    key: column_key(b"t/0", *column_key_value),
                });
                output.push(PersistenceOp::DeletePrefix {
                    prefix: column_key(b"t/1", *column_key_value),
                });
                output.push(PersistenceOp::Delete {
                    key: column_key(b"t/2", *column_key_value),
                });
                continue;
            };
            let end = column.next_row.load(Ordering::Acquire);
            if end < *start {
                return Err(corrupt("temporal row cursor moved backwards"));
            }
            output.push(put(
                column_key(b"t/0", *column_key_value),
                &**column.declaration.load(),
            )?);
            let prefix = column_key(b"t/1", *column_key_value);
            for row in expired {
                if column.sample(*row).is_none() {
                    output.push(PersistenceOp::Delete {
                        key: key(&prefix, &[*row]),
                    });
                }
            }
            for row in *start..end {
                if let Some(sample) = column.sample(row) {
                    output.push(put(key(&prefix, &[row]), &sample)?);
                }
            }
            output.push(put(column_key(b"t/2", *column_key_value), &end)?);
        }
        Ok(output)
    }

    /// Restore ordered temporal records into the existing canonical temporal store.
    pub fn restore_persistence_record(&self, record: &[u8], value: &[u8]) -> Result<bool> {
        if !record.starts_with(b"t/") {
            return Ok(false);
        }
        match record.get(2).copied() {
            Some(b'0') => {
                let key = parse_column(&record[3..])?;
                let declaration: TemporalDeclaration = decode(value)?;
                if key
                    != Self::key(
                        declaration.entity_kind,
                        declaration.target,
                        declaration.property,
                    )
                    || declaration.retention_nanos <= 0
                {
                    return Err(corrupt("invalid temporal declaration record"));
                }
                if self.column(key).is_some() {
                    return Err(corrupt("duplicate temporal declaration"));
                }
                self.state
                    .columns
                    .pin()
                    .insert(key, Arc::new(Column::new(declaration)));
            }
            Some(b'1') => {
                if record.len() != 35 {
                    return Err(corrupt("invalid temporal sample key"));
                }
                let key = parse_column(&record[3..27])?;
                let row = ids(&record[27..], 1)?[0];
                let sample: TemporalSample = decode(value)?;
                let column = self
                    .column(key)
                    .ok_or_else(|| corrupt("temporal sample has no declaration"))?;
                if sample.property != key.2
                    || !column.declaration.load().value_type.accepts(&sample.value)
                {
                    return Err(corrupt("temporal sample differs from declaration"));
                }
                let next = column.next_row.load(Ordering::Acquire);
                if row < next {
                    return Err(corrupt("temporal sample rows are unordered"));
                }
                column.next_row.store(row, Ordering::Release);
                column.insert(sample)?;
            }
            Some(b'2') => {
                let key = parse_column(&record[3..])?;
                let end: u64 = decode(value)?;
                let column = self
                    .column(key)
                    .ok_or_else(|| corrupt("temporal row cursor has no declaration"))?;
                if end < column.next_row.load(Ordering::Acquire) {
                    return Err(corrupt("temporal row cursor precedes samples"));
                }
                column.next_row.store(end, Ordering::Release);
            }
            Some(b'3') => {
                let definition: TemporalRollupDefinition = decode(value)?;
                if record[3..] != *definition.name.as_bytes() {
                    return Err(corrupt("temporal rollup name differs"));
                }
                self.create_rollup(definition)?;
            }
            _ => return Err(corrupt("unknown temporal persistence record")),
        }
        Ok(true)
    }

    /// Current pointers and rollup aggregates are restored by the existing insertion paths.
    pub fn finish_persistence_restore(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::apply_records;

    #[test]
    fn temporal_append_and_retention_update_only_changed_rows() -> Result<()> {
        let store = TemporalStore::default();
        let property = PropertyId(2);
        let changed = [(EntityKind::Node, 7, property)];
        let initial = store.prepare_persistence(&changed, 0)?;
        store.declare(
            TemporalDeclaration {
                entity_kind: EntityKind::Node,
                target: 7,
                property,
                value_type: TemporalType::Integer,
                retention_nanos: 1000,
            },
            0,
        )?;
        let mut records = BTreeMap::new();
        apply_records(&mut records, store.persistence_changes(&initial, true)?);
        for time in 0..1000 {
            let selection = store.prepare_persistence(&changed, time)?;
            store.append(
                EntityKind::Node,
                7,
                TemporalSample {
                    entity_id: 9,
                    property,
                    event_time_nanos: time,
                    sequence_index: time as u64,
                    value: ScalarValue::Integer(time),
                },
                time,
            )?;
            apply_records(&mut records, store.persistence_changes(&selection, false)?);
        }
        store.create_rollup(TemporalRollupDefinition {
            name: "rollup".into(),
            entity_kind: EntityKind::Node,
            target: 7,
            property,
            window: WindowSpec::tumbling(100),
            aggregates: AggregateSet::all(),
        })?;
        apply_records(
            &mut records,
            store.persistence_changes(&TemporalPersistenceSelection::default(), true)?,
        );
        let selection = store.prepare_persistence(&changed, 1001)?;
        store.append(
            EntityKind::Node,
            7,
            TemporalSample {
                entity_id: 9,
                property,
                event_time_nanos: 1001,
                sequence_index: 1001,
                value: ScalarValue::Integer(1001),
            },
            1001,
        )?;
        let changes = store.persistence_changes(&selection, false)?;
        assert_eq!(
            changes.len(),
            4,
            "expected declaration, one expired deletion, one appended sample, and cursor"
        );
        apply_records(&mut records, changes);
        let loaded = TemporalStore::default();
        for (key, value) in records {
            assert!(loaded.restore_persistence_record(&key, &value)?);
        }
        loaded.finish_persistence_restore()?;
        assert_eq!(
            loaded.history(EntityKind::Node, 7, 9, property, 0, 2000, u64::MAX)?,
            store.history(EntityKind::Node, 7, 9, property, 0, 2000, u64::MAX)?
        );
        assert_eq!(
            loaded.rollup_buckets("rollup", 9, 0, 2000)?,
            store.rollup_buckets("rollup", 9, 0, 2000)?
        );
        Ok(())
    }

    #[test]
    fn shortening_retention_deletes_the_newly_expired_record_range() -> Result<()> {
        let store = TemporalStore::default();
        let property = PropertyId(2);
        let mut declaration = TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: 7,
            property,
            value_type: TemporalType::Integer,
            retention_nanos: 1000,
        };
        let initial = store.prepare_declaration_persistence(&declaration, 0)?;
        store.declare(declaration.clone(), 0)?;
        let mut records = BTreeMap::new();
        apply_records(&mut records, store.persistence_changes(&initial, true)?);
        for time in [10, 100, 190] {
            let selection = store.prepare_persistence(&[(EntityKind::Node, 7, property)], time)?;
            store.append(
                EntityKind::Node,
                7,
                TemporalSample {
                    entity_id: 9,
                    property,
                    event_time_nanos: time,
                    sequence_index: time as u64,
                    value: ScalarValue::Integer(time),
                },
                time,
            )?;
            apply_records(&mut records, store.persistence_changes(&selection, false)?);
        }
        declaration.retention_nanos = 20;
        let selection = store.prepare_declaration_persistence(&declaration, 200)?;
        store.declare(declaration, 200)?;
        apply_records(&mut records, store.persistence_changes(&selection, true)?);
        let loaded = TemporalStore::default();
        for (key, value) in records {
            loaded.restore_persistence_record(&key, &value)?;
        }
        assert_eq!(
            loaded
                .history(EntityKind::Node, 7, 9, property, 0, 300, u64::MAX)?
                .len(),
            1
        );
        Ok(())
    }
}
