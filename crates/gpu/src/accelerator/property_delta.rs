//! Row-addressed publication for every canonical property representation.

use super::*;

pub(crate) struct Plan {
    backing: GraphSharedBacking,
    nodes: BTreeMap<PropertyId, Vec<usize>>,
    edges: BTreeMap<PropertyId, Vec<usize>>,
    node_dictionary: Option<DictionaryAppend>,
    edge_dictionary: Option<DictionaryAppend>,
    structure: structure_delta::Plan,
    pub(super) bytes: usize,
}

fn widths(column: &TypedColumn) -> &'static [usize] {
    match column {
        TypedColumn::Boolean { .. } => &[1, 1],
        TypedColumn::Integer { .. } | TypedColumn::Date { .. } | TypedColumn::LocalTime { .. } => {
            &[8, 1]
        }
        TypedColumn::Float { .. } | TypedColumn::ZonedTime { .. } => &[8, 8, 1],
        TypedColumn::String { .. } => &[4, 1],
        TypedColumn::LocalDateTime { .. } => &[8, 4, 1],
        TypedColumn::ZonedDateTime { .. } => &[8, 4, 4, 1],
        TypedColumn::Duration { .. } => &[8, 8, 8, 8, 1],
        TypedColumn::Bytes { .. } | TypedColumn::List { .. } | TypedColumn::Map { .. } => {
            &[8, 1, 1]
        }
    }
}

pub(super) fn changed_rows(
    before: &PropertyColumns,
    after: &PropertyColumns,
    rows: impl Iterator<Item = usize>,
) -> Result<(BTreeMap<PropertyId, Vec<usize>>, usize)> {
    let properties = before
        .property_ids()
        .chain(after.property_ids())
        .collect::<BTreeSet<_>>();
    let mut changed = BTreeMap::<_, Vec<_>>::new();
    for row in rows {
        for property in &properties {
            if matches!(
                (before.physical_column(*property), after.physical_column(*property)),
                (Some(before_column), Some(after_column))
                    if std::ptr::eq(before_column, after_column)
                        && before.is_mixed(*property) == after.is_mixed(*property)
            ) {
                continue;
            }
            let unchanged = match (
                before.get(row as u32, *property),
                after.get(row as u32, *property),
            ) {
                (Some(ScalarValue::Float(before)), Some(ScalarValue::Float(after))) => {
                    before.0.to_bits() == after.0.to_bits()
                }
                (before, after) => before == after,
            };
            if !unchanged {
                changed.entry(*property).or_default().push(row);
            }
        }
    }
    if before.rows() != after.rows() {
        for property in &properties {
            changed.entry(*property).or_default();
        }
    }
    let mut bytes = 0_usize;
    for (property, rows) in &mut changed {
        rows.sort_unstable();
        rows.dedup();
        let physical = after
            .physical_column(*property)
            .ok_or_else(|| Error::internal("property delta removed its physical column"))?;
        for width in widths(physical) {
            let pages = rows
                .iter()
                .map(|row| row * width / metal_pages::page_bytes())
                .collect::<BTreeSet<_>>();
            bytes =
                bytes.saturating_add((pages.len() + 2).saturating_mul(metal_pages::page_bytes()));
        }
        // Temporary row bytes, native writes and allocator metadata scale only with changed
        // payloads. No flat view or dictionary-wide gather is needed to plan a generation.
        for row in rows.iter().copied() {
            let payload = match physical {
                TypedColumn::Bytes { values, .. } => values.get(row).map_or(0, |value| value.len()),
                TypedColumn::List { values, .. } | TypedColumn::Map { values, .. } => {
                    values.get(row as u32).map_or(0, <[u8]>::len)
                }
                _ => 0,
            };
            bytes = bytes.saturating_add(payload.saturating_mul(40));
        }
    }
    Ok((changed, bytes))
}

fn fixed_staging_bytes(before: &GraphSharedBacking, delta: &ResidentProjectDelta) -> usize {
    let mut pages = BTreeSet::new();
    let page_bytes = metal_pages::page_bytes();
    for row in &delta.graph.nodes {
        let dense = row.dense as usize;
        for (lane, changed, width) in [
            (0, before.node_ids.get(dense) != Some(&row.id), 8),
            (1, before.node_ids.get(dense) != Some(&row.id), 8),
            (
                2,
                before.node_revisions.get(dense) != Some(&row.revision),
                8,
            ),
            (3, before.node_layers.get(dense) != Some(&row.layer), 1),
            (
                4,
                before.node_active.get(dense).copied() != Some(u8::from(row.active)),
                1,
            ),
        ] {
            if changed {
                pages.insert((lane, dense * width / page_bytes));
            }
        }
        let old = before.node_labels.get(row.dense).unwrap_or(&[]);
        for label in old.iter().chain(&row.labels) {
            if old.contains(label) != row.labels.contains(label) {
                pages.insert((16 + label.0 as usize, dense / page_bytes));
            }
        }
    }
    for row in &delta.graph.edges {
        let dense = row.dense as usize;
        for (lane, changed, width) in [
            (5, before.edge_ids.get(dense) != Some(&row.id), 8),
            (6, before.edge_sources.get(dense) != Some(&row.source), 4),
            (7, before.edge_targets.get(dense) != Some(&row.target), 4),
            (
                8,
                before.edge_types.get(dense) != Some(&row.relationship_type),
                8,
            ),
            (9, before.edge_layers.get(dense) != Some(&row.layer), 1),
            (
                10,
                before.edge_active.get(dense).copied() != Some(u8::from(row.active)),
                1,
            ),
        ] {
            if changed {
                pages.insert((lane, dense * width / page_bytes));
            }
        }
    }
    pages.len().saturating_mul(page_bytes)
}

pub(super) fn plan(current: &CandleResident, delta: &ResidentProjectDelta) -> Result<Option<Plan>> {
    let Some(before) = current.shared_graph.as_ref() else {
        return Ok(None);
    };
    let mut backing = before.clone();
    backing.apply_device_delta_deferred_adjacency(&delta.graph)?;
    let (nodes, node_bytes) = changed_rows(
        &before.node_properties,
        &backing.node_properties,
        delta.graph.nodes.iter().map(|row| row.dense as usize),
    )?;
    let (edges, edge_bytes) = changed_rows(
        &before.edge_properties,
        &backing.edge_properties,
        delta.graph.edges.iter().map(|row| row.dense as usize),
    )?;
    let node_dictionary =
        DictionaryAppend::plan(&current.node_string_dictionary, &backing.node_properties)?;
    let edge_dictionary =
        DictionaryAppend::plan(&current.edge_string_dictionary, &backing.edge_properties)?;
    let dictionary_bytes = node_dictionary
        .as_ref()
        .map_or(0, |plan| plan.staging_bytes)
        .saturating_add(
            edge_dictionary
                .as_ref()
                .map_or(0, |plan| plan.staging_bytes),
        );
    let structure = structure_delta::plan(current, delta)?;
    let structure_bytes = structure.bytes;
    Ok(Some(Plan {
        backing,
        nodes,
        edges,
        node_dictionary,
        edge_dictionary,
        structure,
        bytes: node_bytes
            .saturating_add(edge_bytes)
            .saturating_add(dictionary_bytes)
            .saturating_add(fixed_staging_bytes(before, delta))
            .saturating_add(structure_bytes),
    }))
}

fn patch<T: Copy + WithDType>(
    tensor: &mut Option<Tensor>,
    row_count: usize,
    rows: &[usize],
    value: impl Fn(usize) -> T,
    device: &Device,
) -> Result<()> {
    if rows.is_empty()
        && tensor
            .as_ref()
            .is_some_and(|tensor| tensor.elem_count() == row_count)
    {
        return Ok(());
    }
    let previous = match tensor.as_ref() {
        Some(tensor) => tensor.clone(),
        None => metal_pages::zeros(T::DTYPE, row_count, device)?,
    };
    let updates = rows
        .iter()
        .map(|row| (*row, value(*row)))
        .collect::<Vec<_>>();
    *tensor = Some(metal_pages::extend(&previous, &updates, row_count, device)?);
    Ok(())
}

fn tensor_bytes(tensor: &Option<Tensor>) -> usize {
    tensor.as_ref().map_or(0, metal_pages::buffer_bytes)
}

fn document_bytes(column: &DocumentColumn) -> usize {
    metal_pages::buffer_bytes(&column.offsets)
        + tensor_bytes(&column.bytes)
        + tensor_bytes(&column.validity)
}

fn base_bytes(base: &MixedBaseColumn) -> usize {
    match base {
        MixedBaseColumn::Boolean(column) => {
            tensor_bytes(&column.values) + tensor_bytes(&column.validity)
        }
        MixedBaseColumn::Integer(column)
        | MixedBaseColumn::Date(column)
        | MixedBaseColumn::LocalTime(column) => {
            tensor_bytes(&column.values) + tensor_bytes(&column.validity)
        }
        MixedBaseColumn::Float(column) => {
            tensor_bytes(&column.values)
                + tensor_bytes(&column.order_keys)
                + tensor_bytes(&column.validity)
        }
        MixedBaseColumn::String(column) => {
            tensor_bytes(&column.values) + tensor_bytes(&column.validity)
        }
        MixedBaseColumn::ZonedTime(column) => {
            tensor_bytes(&column.nanos)
                + tensor_bytes(&column.offsets)
                + tensor_bytes(&column.validity)
        }
        MixedBaseColumn::LocalDateTime(column) => {
            tensor_bytes(&column.seconds)
                + tensor_bytes(&column.nanos)
                + tensor_bytes(&column.validity)
        }
        MixedBaseColumn::ZonedDateTime(column) => {
            tensor_bytes(&column.seconds)
                + tensor_bytes(&column.nanos)
                + tensor_bytes(&column.timezones)
                + tensor_bytes(&column.validity)
        }
        MixedBaseColumn::Duration(column) => {
            tensor_bytes(&column.months)
                + tensor_bytes(&column.days)
                + tensor_bytes(&column.seconds)
                + tensor_bytes(&column.nanos)
                + tensor_bytes(&column.validity)
        }
        MixedBaseColumn::Bytes(column)
        | MixedBaseColumn::List(column)
        | MixedBaseColumn::Map(column) => document_bytes(column),
    }
}

fn base_validity(base: &MixedBaseColumn) -> Option<Tensor> {
    match base {
        MixedBaseColumn::Boolean(column) => column.validity.clone(),
        MixedBaseColumn::Integer(column)
        | MixedBaseColumn::Date(column)
        | MixedBaseColumn::LocalTime(column) => column.validity.clone(),
        MixedBaseColumn::Float(column) => column.validity.clone(),
        MixedBaseColumn::String(column) => column.validity.clone(),
        MixedBaseColumn::ZonedTime(column) => column.validity.clone(),
        MixedBaseColumn::LocalDateTime(column) => column.validity.clone(),
        MixedBaseColumn::ZonedDateTime(column) => column.validity.clone(),
        MixedBaseColumn::Duration(column) => column.validity.clone(),
        MixedBaseColumn::Bytes(column)
        | MixedBaseColumn::List(column)
        | MixedBaseColumn::Map(column) => column.validity.clone(),
    }
}

fn take_base(
    resident: &mut CandleResident,
    nodes: bool,
    property: PropertyId,
) -> Option<Box<MixedBaseColumn>> {
    macro_rules! take {
        ($node:ident, $edge:ident, $kind:ident) => {
            if let Some(column) = if nodes {
                resident.$node.remove(&property)
            } else {
                resident.$edge.remove(&property)
            } {
                return Some(Box::new(MixedBaseColumn::$kind(column)));
            }
        };
    }
    take!(boolean_nodes, boolean_edges, Boolean);
    take!(integer_nodes, integer_edges, Integer);
    take!(float_nodes, float_edges, Float);
    take!(string_nodes, string_edges, String);
    take!(date_nodes, date_edges, Date);
    take!(local_time_nodes, local_time_edges, LocalTime);
    take!(zoned_time_nodes, zoned_time_edges, ZonedTime);
    take!(local_datetime_nodes, local_datetime_edges, LocalDateTime);
    take!(zoned_datetime_nodes, zoned_datetime_edges, ZonedDateTime);
    take!(duration_nodes, duration_edges, Duration);
    take!(byte_nodes, byte_edges, Bytes);
    take!(list_nodes, list_edges, List);
    take!(map_nodes, map_edges, Map);
    None
}

fn property_bytes(resident: &CandleResident, nodes: bool, property: PropertyId) -> usize {
    let mut bytes = 0_usize;
    macro_rules! lanes {
        ($node:ident, $edge:ident, $($lane:ident),+) => {
            if let Some(column) = if nodes { resident.$node.get(&property) } else { resident.$edge.get(&property) } {
                $(bytes = bytes.saturating_add(tensor_bytes(&column.$lane));)+
            }
        };
    }
    lanes!(boolean_nodes, boolean_edges, values, validity);
    lanes!(integer_nodes, integer_edges, values, validity);
    lanes!(float_nodes, float_edges, values, order_keys, validity);
    lanes!(string_nodes, string_edges, values, validity);
    lanes!(date_nodes, date_edges, values, validity);
    lanes!(local_time_nodes, local_time_edges, values, validity);
    lanes!(zoned_time_nodes, zoned_time_edges, nanos, offsets, validity);
    lanes!(
        local_datetime_nodes,
        local_datetime_edges,
        seconds,
        nanos,
        validity
    );
    lanes!(
        zoned_datetime_nodes,
        zoned_datetime_edges,
        seconds,
        nanos,
        timezones,
        validity
    );
    lanes!(
        duration_nodes,
        duration_edges,
        months,
        days,
        seconds,
        nanos,
        validity
    );
    for column in [
        if nodes {
            resident.byte_nodes.get(&property)
        } else {
            resident.byte_edges.get(&property)
        },
        if nodes {
            resident.list_nodes.get(&property)
        } else {
            resident.list_edges.get(&property)
        },
        if nodes {
            resident.map_nodes.get(&property)
        } else {
            resident.map_edges.get(&property)
        },
    ]
    .into_iter()
    .flatten()
    {
        bytes = bytes.saturating_add(document_bytes(column));
    }
    if let Some(column) = if nodes {
        resident.mixed_nodes.get(&property)
    } else {
        resident.mixed_edges.get(&property)
    } {
        bytes = bytes
            .saturating_add(metal_pages::buffer_bytes(&column.offsets))
            .saturating_add(tensor_bytes(&column.bytes))
            .saturating_add(tensor_bytes(&column.validity))
            .saturating_add(tensor_bytes(&column.overrides))
            .saturating_add(column.base.as_deref().map_or(0, base_bytes));
    }
    bytes
}

fn fixed_bytes(resident: &CandleResident) -> usize {
    [
        &resident.node_entity_ids,
        &resident.node_id_order_keys,
        &resident.node_revisions,
        &resident.node_layers,
        &resident.node_active,
        &resident.edge_entity_ids,
        &resident.edge_sources,
        &resident.edge_targets,
        &resident.edge_types,
        &resident.edge_layers,
        &resident.edge_active,
    ]
    .into_iter()
    .map(tensor_bytes)
    .sum::<usize>()
    .saturating_add(
        resident
            .node_labels
            .values()
            .map(metal_pages::buffer_bytes)
            .sum::<usize>(),
    )
    .saturating_add(metal_pages::buffer_bytes(&resident.outgoing_offsets))
    .saturating_add(metal_pages::buffer_bytes(&resident.incoming_offsets))
}

pub(super) fn patch_property_columns(
    resident: &mut CandleResident,
    nodes: bool,
    columns: &PropertyColumns,
    changes: &BTreeMap<PropertyId, Vec<usize>>,
    device: &Device,
) -> Result<()> {
    let row_count = columns.rows();
    macro_rules! column {
        ($node:ident, $edge:ident, $property:expr, $initial:expr) => {
            if nodes {
                &mut resident.$node
            } else {
                &mut resident.$edge
            }
            .entry(*$property)
            .or_insert_with(|| $initial)
        };
    }
    for (property, rows) in changes {
        let before_bytes = property_bytes(resident, nodes, *property);
        let physical = columns
            .physical_column(*property)
            .ok_or_else(|| Error::internal("property patch lost its canonical column"))?;
        if columns.is_mixed(*property) {
            let exists = if nodes {
                resident.mixed_nodes.contains_key(property)
            } else {
                resident.mixed_edges.contains_key(property)
            };
            if !exists {
                let base = take_base(resident, nodes, *property);
                let validity = base.as_deref().and_then(base_validity);
                let maximum_string_bytes = base.as_deref().map_or(0, |base| match base {
                    MixedBaseColumn::String(column) => column.maximum_bytes,
                    _ => 0,
                });
                let overrides = base
                    .as_ref()
                    .map(|_| metal_pages::zeros(DType::U8, row_count, device))
                    .transpose()?;
                let column = MixedColumn {
                    offsets: metal_pages::zeros(
                        DType::U32,
                        row_count
                            .checked_mul(2)
                            .ok_or_else(|| Error::internal("mixed descriptor size overflow"))?,
                        device,
                    )?,
                    bytes: None,
                    validity,
                    maximum_string_bytes,
                    base,
                    overrides,
                    allocator: variable_payload::Allocator::default(),
                };
                if nodes {
                    resident.mixed_nodes.insert(*property, column);
                    resident.unsupported_node_properties.remove(property);
                    resident.opaque_node_validity.remove(property);
                } else {
                    resident.mixed_edges.insert(*property, column);
                    resident.unsupported_edge_properties.remove(property);
                    resident.opaque_edge_validity.remove(property);
                }
            }
            let column = if nodes {
                resident.mixed_nodes.get_mut(property)
            } else {
                resident.mixed_edges.get_mut(property)
            }
            .ok_or_else(|| Error::internal("mixed patch column missing"))?;
            let TypedColumn::Bytes { values, .. } = physical else {
                return Err(Error::internal("mixed physical column is not bytes"));
            };
            let replacements = rows
                .iter()
                .map(|row| {
                    let payload = values
                        .get(*row)
                        .ok_or_else(|| Error::internal("mixed patch row missing"))?
                        .into_owned();
                    if payload.first() == Some(&MIXED_STRING_TAG) {
                        column.maximum_string_bytes = column
                            .maximum_string_bytes
                            .max(payload.len().saturating_sub(1));
                    }
                    Ok((*row, payload))
                })
                .collect::<Result<Vec<_>>>()?;
            let (allocator, offsets, bytes) = column.allocator.patch(
                &column.offsets,
                &column.bytes,
                row_count,
                &replacements,
                device,
            )?;
            column.allocator = allocator;
            column.offsets = offsets;
            column.bytes = bytes;
            patch(
                &mut column.validity,
                row_count,
                rows,
                |row| u8::from(physical.validity().is_present(row)),
                device,
            )?;
            if column.base.is_some() {
                patch(&mut column.overrides, row_count, rows, |_| 1_u8, device)?;
            }
        } else if matches!(
            physical,
            TypedColumn::Bytes { .. } | TypedColumn::List { .. } | TypedColumn::Map { .. }
        ) {
            let target = match physical {
                TypedColumn::Bytes { .. } => {
                    if nodes {
                        &mut resident.byte_nodes
                    } else {
                        &mut resident.byte_edges
                    }
                }
                TypedColumn::List { .. } => {
                    if nodes {
                        &mut resident.list_nodes
                    } else {
                        &mut resident.list_edges
                    }
                }
                _ => {
                    if nodes {
                        &mut resident.map_nodes
                    } else {
                        &mut resident.map_edges
                    }
                }
            };
            if let std::collections::btree_map::Entry::Vacant(entry) = target.entry(*property) {
                entry.insert(DocumentColumn {
                    offsets: metal_pages::zeros(
                        DType::U32,
                        row_count
                            .checked_mul(2)
                            .ok_or_else(|| Error::internal("property descriptor size overflow"))?,
                        device,
                    )?,
                    bytes: None,
                    validity: None,
                    rows: row_count,
                    maximum_bytes: 0,
                    allocator: variable_payload::Allocator::default(),
                });
            }
            let column = target
                .get_mut(property)
                .ok_or_else(|| Error::internal("variable property column missing"))?;
            let replacements = rows
                .iter()
                .map(|row| {
                    let payload = if physical.validity().is_present(*row) {
                        match physical {
                            TypedColumn::Bytes { values, .. } => {
                                values.get(*row).map(|value| value.into_owned())
                            }
                            TypedColumn::List { values, .. } | TypedColumn::Map { values, .. } => {
                                values.get(*row as u32).map(<[u8]>::to_vec)
                            }
                            _ => None,
                        }
                        .ok_or_else(|| Error::internal("variable patch row missing"))?
                    } else {
                        Vec::new()
                    };
                    column.maximum_bytes = column.maximum_bytes.max(payload.len());
                    Ok((*row, payload))
                })
                .collect::<Result<Vec<_>>>()?;
            let (allocator, offsets, bytes) = column.allocator.patch(
                &column.offsets,
                &column.bytes,
                row_count,
                &replacements,
                device,
            )?;
            column.allocator = allocator;
            column.offsets = offsets;
            column.bytes = bytes;
            column.rows = row_count;
            patch(
                &mut column.validity,
                row_count,
                rows,
                |row| u8::from(physical.validity().is_present(row)),
                device,
            )?;
            let validity = column.validity.clone();
            if nodes {
                resident.unsupported_node_properties.insert(*property);
                if matches!(physical, TypedColumn::Bytes { .. }) {
                    resident.opaque_node_validity.insert(*property, validity);
                }
            } else {
                resident.unsupported_edge_properties.insert(*property);
                if matches!(physical, TypedColumn::Bytes { .. }) {
                    resident.opaque_edge_validity.insert(*property, validity);
                }
            }
        } else {
            let validity = match physical {
                TypedColumn::Boolean { values, .. } => {
                    let column = column!(
                        boolean_nodes,
                        boolean_edges,
                        property,
                        BooleanColumn {
                            values: None,
                            validity: None,
                            rows: row_count
                        }
                    );
                    patch(
                        &mut column.values,
                        row_count,
                        rows,
                        |row| u8::from(values[row]),
                        device,
                    )?;
                    column.rows = row_count;
                    &mut column.validity
                }
                TypedColumn::Integer { values, .. }
                | TypedColumn::Date { values, .. }
                | TypedColumn::LocalTime { values, .. } => {
                    let initial = IntegerColumn {
                        values: None,
                        validity: None,
                        rows: row_count,
                    };
                    let column = match physical {
                        TypedColumn::Integer { .. } => {
                            column!(integer_nodes, integer_edges, property, initial)
                        }
                        TypedColumn::Date { .. } => {
                            column!(date_nodes, date_edges, property, initial)
                        }
                        _ => column!(local_time_nodes, local_time_edges, property, initial),
                    };
                    patch(
                        &mut column.values,
                        row_count,
                        rows,
                        |row| values[row],
                        device,
                    )?;
                    column.rows = row_count;
                    &mut column.validity
                }
                TypedColumn::Float { values, .. } => {
                    let column = column!(
                        float_nodes,
                        float_edges,
                        property,
                        FloatColumn {
                            values: None,
                            order_keys: None,
                            validity: None,
                            rows: row_count
                        }
                    );
                    patch(
                        &mut column.values,
                        row_count,
                        rows,
                        |row| values[row].0.to_bits() as i64,
                        device,
                    )?;
                    patch(
                        &mut column.order_keys,
                        row_count,
                        rows,
                        |row| ordered_float_sort_key(values[row].0),
                        device,
                    )?;
                    column.rows = row_count;
                    &mut column.validity
                }
                TypedColumn::String { values, .. } => {
                    let column = column!(
                        string_nodes,
                        string_edges,
                        property,
                        StringColumn {
                            values: None,
                            validity: None,
                            rows: row_count,
                            maximum_bytes: 0
                        }
                    );
                    patch(
                        &mut column.values,
                        row_count,
                        rows,
                        |row| values[row],
                        device,
                    )?;
                    for row in rows {
                        if physical.validity().is_present(*row) {
                            column.maximum_bytes = column.maximum_bytes.max(
                                columns
                                    .string_dictionary()
                                    .resolve(values[*row])
                                    .map_or(0, str::len),
                            );
                        }
                    }
                    column.rows = row_count;
                    &mut column.validity
                }
                TypedColumn::ZonedTime { nanos, offsets, .. } => {
                    let column = column!(
                        zoned_time_nodes,
                        zoned_time_edges,
                        property,
                        ZonedTimeColumn {
                            nanos: None,
                            offsets: None,
                            validity: None,
                            rows: row_count
                        }
                    );
                    patch(&mut column.nanos, row_count, rows, |row| nanos[row], device)?;
                    patch(
                        &mut column.offsets,
                        row_count,
                        rows,
                        |row| i64::from(offsets[row]),
                        device,
                    )?;
                    column.rows = row_count;
                    &mut column.validity
                }
                TypedColumn::LocalDateTime { seconds, nanos, .. } => {
                    let column = column!(
                        local_datetime_nodes,
                        local_datetime_edges,
                        property,
                        LocalDateTimeColumn {
                            seconds: None,
                            nanos: None,
                            validity: None,
                            rows: row_count
                        }
                    );
                    patch(
                        &mut column.seconds,
                        row_count,
                        rows,
                        |row| seconds[row],
                        device,
                    )?;
                    patch(&mut column.nanos, row_count, rows, |row| nanos[row], device)?;
                    column.rows = row_count;
                    &mut column.validity
                }
                TypedColumn::ZonedDateTime {
                    seconds,
                    nanos,
                    timezones,
                    ..
                } => {
                    let column = column!(
                        zoned_datetime_nodes,
                        zoned_datetime_edges,
                        property,
                        ZonedDateTimeColumn {
                            seconds: None,
                            nanos: None,
                            timezones: None,
                            validity: None,
                            rows: row_count,
                            maximum_timezone_bytes: 0
                        }
                    );
                    patch(
                        &mut column.seconds,
                        row_count,
                        rows,
                        |row| seconds[row],
                        device,
                    )?;
                    patch(&mut column.nanos, row_count, rows, |row| nanos[row], device)?;
                    patch(
                        &mut column.timezones,
                        row_count,
                        rows,
                        |row| timezones[row],
                        device,
                    )?;
                    for row in rows {
                        if physical.validity().is_present(*row) {
                            column.maximum_timezone_bytes = column.maximum_timezone_bytes.max(
                                columns
                                    .string_dictionary()
                                    .resolve(timezones[*row])
                                    .map_or(0, str::len),
                            );
                        }
                    }
                    column.rows = row_count;
                    &mut column.validity
                }
                TypedColumn::Duration {
                    months,
                    days,
                    seconds,
                    nanos,
                    ..
                } => {
                    let column = column!(
                        duration_nodes,
                        duration_edges,
                        property,
                        DurationColumn {
                            months: None,
                            days: None,
                            seconds: None,
                            nanos: None,
                            validity: None,
                            rows: row_count
                        }
                    );
                    patch(
                        &mut column.months,
                        row_count,
                        rows,
                        |row| months[row],
                        device,
                    )?;
                    patch(&mut column.days, row_count, rows, |row| days[row], device)?;
                    patch(
                        &mut column.seconds,
                        row_count,
                        rows,
                        |row| seconds[row],
                        device,
                    )?;
                    patch(
                        &mut column.nanos,
                        row_count,
                        rows,
                        |row| i64::from(nanos[row]),
                        device,
                    )?;
                    column.rows = row_count;
                    &mut column.validity
                }
                _ => return Err(Error::internal("unhandled property representation")),
            };
            patch(
                validity,
                row_count,
                rows,
                |row| u8::from(physical.validity().is_present(row)),
                device,
            )?;
            if matches!(physical, TypedColumn::Duration { .. }) {
                if nodes {
                    resident.unsupported_node_properties.insert(*property);
                } else {
                    resident.unsupported_edge_properties.insert(*property);
                }
            }
        }
        let after_bytes = property_bytes(resident, nodes, *property);
        resident.allocated_bytes = resident
            .allocated_bytes
            .saturating_sub(before_bytes)
            .saturating_add(after_bytes);
    }
    Ok(())
}

impl Plan {
    pub(super) fn apply(
        mut self,
        resident: &mut CandleResident,
        delta: &ResidentProjectDelta,
        device: &Device,
    ) -> Result<()> {
        let old_fixed_bytes = fixed_bytes(resident);
        structure_delta::patch_fixed_and_labels(resident, delta, device)?;
        for (plan, dictionary, columns) in [
            (
                self.node_dictionary.take(),
                &mut resident.node_string_dictionary,
                &mut self.backing.node_properties,
            ),
            (
                self.edge_dictionary.take(),
                &mut resident.edge_string_dictionary,
                &mut self.backing.edge_properties,
            ),
        ] {
            if let Some(plan) = plan {
                let previous_bytes = dictionary_bytes(dictionary);
                plan.apply(dictionary, columns, device)?;
                resident.allocated_bytes = resident
                    .allocated_bytes
                    .saturating_sub(previous_bytes)
                    .saturating_add(dictionary_bytes(dictionary));
            }
        }
        patch_property_columns(
            resident,
            true,
            &self.backing.node_properties,
            &self.nodes,
            device,
        )?;
        patch_property_columns(
            resident,
            false,
            &self.backing.edge_properties,
            &self.edges,
            device,
        )?;
        structure_delta::patch_adjacency(
            resident,
            &mut self.backing,
            delta,
            device,
            self.structure,
        )?;
        resident.allocated_bytes = resident
            .allocated_bytes
            .saturating_sub(old_fixed_bytes)
            .saturating_add(fixed_bytes(resident));
        resident.node_count = delta.graph.node_capacity;
        resident.edge_count = delta.graph.edge_capacity;
        resident.shared_graph = Some(self.backing);
        Ok(())
    }
}

fn dictionary_bytes(dictionary: &StringDictionary) -> usize {
    metal_pages::buffer_bytes(&dictionary.offsets)
        + tensor_bytes(&dictionary.bytes)
        + dictionary
            .ranks
            .as_ref()
            .map_or(0, metal_pages::buffer_bytes)
        + dictionary
            .order
            .as_ref()
            .map_or(0, dictionary_order::LexicalOrder::bytes)
}

pub(super) struct DictionaryAppend {
    offsets: Vec<(usize, u32)>,
    bytes: Vec<(usize, u8)>,
    offset_len: usize,
    byte_len: usize,
    maximum_bytes: usize,
    pub(super) staging_bytes: usize,
    order: dictionary_order::OrderAppend,
}

impl DictionaryAppend {
    pub(super) fn plan(
        current: &StringDictionary,
        columns: &PropertyColumns,
    ) -> Result<Option<Self>> {
        let before = current.offsets.elem_count().saturating_sub(1);
        let after = columns.string_dictionary().len();
        if before == after {
            return Ok(None);
        }
        if before > after {
            return Err(Error::internal(
                "property dictionary shrank during a row delta",
            ));
        }
        let mut offsets = Vec::new();
        let mut bytes = Vec::new();
        let mut byte_len = current.bytes.as_ref().map_or(0, Tensor::elem_count);
        let mut maximum_bytes = current.maximum_bytes;
        for id in before..after {
            let value = columns
                .string_dictionary()
                .resolve(id as u32)
                .ok_or_else(|| Error::internal("property dictionary append lost an entry"))?;
            maximum_bytes = maximum_bytes.max(value.len());
            bytes.extend(
                value
                    .bytes()
                    .enumerate()
                    .map(|(offset, value)| (byte_len + offset, value)),
            );
            byte_len = byte_len
                .checked_add(value.len())
                .ok_or_else(|| Error::internal("dictionary byte length overflow"))?;
            offsets.push((id + 1, checked_u32(byte_len, "dictionary bytes")?));
        }
        let order = current
            .order
            .as_ref()
            .ok_or_else(|| Error::internal("dictionary append lost its order index"))?
            .prepare_append(before, columns)?;
        let staging_bytes = bytes
            .len()
            .saturating_add(offsets.len().saturating_mul(4))
            .div_ceil(metal_pages::page_bytes())
            .saturating_add(6)
            .saturating_mul(metal_pages::page_bytes())
            .saturating_add(bytes.len().saturating_mul(size_of::<(usize, u8)>()))
            .saturating_add(offsets.len().saturating_mul(size_of::<(usize, u32)>()))
            .saturating_add(order.staging_bytes);
        Ok(Some(Self {
            offsets,
            bytes,
            offset_len: after + 1,
            byte_len,
            maximum_bytes,
            staging_bytes,
            order,
        }))
    }

    pub(super) fn apply(
        self,
        current: &mut StringDictionary,
        _columns: &mut PropertyColumns,
        device: &Device,
    ) -> Result<()> {
        current.offsets =
            metal_pages::extend(&current.offsets, &self.offsets, self.offset_len, device)?;
        if self.byte_len != 0 {
            current.bytes = Some(match current.bytes.as_ref() {
                Some(tensor) => metal_pages::extend(tensor, &self.bytes, self.byte_len, device)?,
                None => {
                    let values = self
                        .bytes
                        .iter()
                        .map(|(_, value)| *value)
                        .collect::<Vec<_>>();
                    let mut upload = TensorUpload::new(device);
                    upload.immutable_properties = true;
                    upload
                        .optional(&values)?
                        .ok_or_else(|| Error::internal("dictionary append omitted its bytes"))?
                }
            });
        }
        current.order = Some(self.order.apply(device)?);
        current.ranks = None;
        current.host = None;
        current.maximum_bytes = self.maximum_bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeId;

    #[test]
    fn property_change_detection_skips_shared_payloads_and_preserves_float_bits() -> Result<()> {
        let body = PropertyId(1);
        let number = PropertyId(2);
        let mut before = PropertyColumns::default();
        before.push_row(&[
            (body, ScalarValue::Bytes(vec![0x5a; 256 * 1_024].into())),
            (number, ScalarValue::Float(ordered_float::OrderedFloat(0.0))),
        ])?;
        let mut after = before.clone();
        after.set(
            0,
            number,
            &ScalarValue::Float(ordered_float::OrderedFloat(-0.0)),
        )?;
        let (changed, _) = changed_rows(&before, &after, [0].into_iter())?;
        assert_eq!(changed, BTreeMap::from([(number, vec![0])]));
        assert!(std::ptr::eq(
            before.physical_column(body).unwrap(),
            after.physical_column(body).unwrap()
        ));
        let old_nan = ScalarValue::Float(ordered_float::OrderedFloat(f64::from_bits(
            0x7ff8000000000001,
        )));
        let new_nan = ScalarValue::Float(ordered_float::OrderedFloat(f64::from_bits(
            0x7ff8000000000002,
        )));
        before.set(0, number, &old_nan)?;
        after = before.clone();
        after.set(0, number, &new_nan)?;
        assert_eq!(
            changed_rows(&before, &after, [0].into_iter())?.0,
            BTreeMap::from([(number, vec![0])])
        );
        after = before.clone();
        after.push_row(&[])?;
        assert!(
            changed_rows(&before, &after, [1].into_iter())?
                .0
                .contains_key(&body)
        );
        Ok(())
    }

    fn nullable_base_rows(
        resident: &CandleResident,
        property: PropertyId,
        device: &Device,
    ) -> Result<Vec<u32>> {
        nullable_comparison_rows(resident, property, CompareOp::Eq, device)
    }

    fn nullable_comparison_rows(
        resident: &CandleResident,
        property: PropertyId,
        operation: CompareOp,
        device: &Device,
    ) -> Result<Vec<u32>> {
        use crate::*;
        let slot = ResidentNullableRelationSlot(0);
        let program = ResidentNullableRelationProgram {
            layers: LayerMask::OBSERVED,
            stages: vec![
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Mandatory,
                    output: slot,
                    labels: ResidentNullableNodeDomain::Any,
                },
                ResidentNullableRelationStage::FinalProject {
                    bindings: vec![ResidentNullableRelationOutputBinding {
                        name: "node".into(),
                        source: ResidentNullableRelationOutputSource::Entity {
                            slot,
                            kind: ResidentNullableRelationBindingKind::Node,
                        },
                    }],
                },
            ],
        };
        let predicates = ResidentNullableRelationPredicateProgram {
            filters: vec![ResidentNullableRelationFilterStage {
                placement: ResidentNullableRelationFilterPlacement::RelationAfter { stage: 0 },
                predicate: ResidentNullableRelationPredicate::CompareString {
                    left: ResidentNullableRelationPredicateValue::StringProperty {
                        slot,
                        kind: ResidentNullableRelationBindingKind::Node,
                        property,
                    },
                    operation,
                    right: ResidentNullableRelationPredicateValue::String("base".into()),
                },
            }],
            optional_groups: Vec::new(),
        };
        let capacities = ResidentNullableRelationCapacities::derive(
            &program,
            resident.node_count,
            resident.edge_count,
            resident.node_count,
            resident.edge_count,
            resident.node_count,
        )?;
        let request = ResidentNullableRelationRequest::build_with_predicates(
            ResidentNullableRelationGeneration {
                project: resident.project,
                bookmark: resident.bookmark,
                graph_revision: resident.revision,
                layout_version: resident.layout_version,
                catalog_generation: [0; 32],
            },
            ResidentExecutionId { high: 41, low: 71 },
            program,
            predicates,
            capacities,
            0x5200,
        )?;
        let result = resident
            .execute_nullable_relation(device, &request, &CancellationToken::new())?
            .validate(&request, ResidentDeviceCompletion::Metal)?;
        match &result.columns()[0] {
            ResidentNullableRelationOutputColumn::Entity { rows, .. } => Ok(rows.clone()),
            _ => Err(Error::internal("nullable test returned a non-entity lane")),
        }
    }

    fn quantifier_base_rows(
        resident: &CandleResident,
        property: PropertyId,
        literal: crate::ResidentQuantifierValue,
        shape: crate::ResidentQuantifierEntityPropertyShape,
        device: &Device,
    ) -> Result<Vec<Vec<crate::ResidentQuantifierValue>>> {
        use crate::ResidentQuantifierExpression as Expr;
        use crate::*;
        let execution = ResidentExecutionId { high: 43, low: 73 };
        let source = ResidentQuantifierSource::VariablePathEntityList {
            path: ResidentVariablePathRequest {
                project: resident.project,
                expected_bookmark: resident.bookmark,
                expected_graph_revision: resident.revision,
                expected_layout_version: resident.layout_version,
                expected_node_slots: resident.node_count,
                expected_edge_slots: resident.edge_count,
                layers: LayerMask::OBSERVED,
                multiplicity_scans: Vec::new(),
                bound_terminal_scan: None,
                cartesian_obligation: None,
                input: ResidentVariablePathInput::Rows(vec![0]),
                execution,
                segments: vec![ResidentVariablePathSegment {
                    direction: ResidentDirection::Outgoing,
                    relationship_types: Vec::new(),
                    relationship_types_known_empty: false,
                    relationship_integer_predicates: Vec::new(),
                    target_labels: Vec::new(),
                    target_labels_known_empty: false,
                    target_predicates: Vec::new(),
                    target_equals_path_start: false,
                    minimum_hops: 1,
                    maximum_hops: Some(1),
                    obligation: ResidentExecutionObligation {
                        id: 11,
                        kind: ResidentObligationKind::PatternTraversal,
                        scope: ResidentObligationScope::PatternLeaf(0),
                    },
                }],
                optional: false,
                output_limit: None,
                final_obligation: ResidentExecutionObligation {
                    id: 12,
                    kind: ResidentObligationKind::PatternFilter,
                    scope: ResidentObligationScope::PatternFinal,
                },
                final_projection: ResidentVariablePathFinalProjection::Publications,
                maximum_frontier_paths: 8,
                maximum_output_rows: 8,
                distinct_endpoints: false,
            },
            output: ResidentQuantifierSlot(0),
            entity_kind: ResidentQuantifierEntityKind::Node,
            skip: 1,
            properties: vec![ResidentQuantifierEntityProperty {
                key: "p".into(),
                property: Some(property),
                shape,
            }],
            materialize_obligation: ResidentExecutionObligation {
                id: 13,
                kind: ResidentObligationKind::Expression,
                scope: ResidentObligationScope::Expression(u16::MAX),
            },
        };
        let program = ResidentQuantifierProgram {
            slot_count: 3,
            stages: vec![ResidentQuantifierStage::Project {
                keep_scope: false,
                bindings: vec![ResidentQuantifierProjection {
                    output: ResidentQuantifierSlot(2),
                    expression: Expr::Predicate {
                        kind: ResidentQuantifierKind::Any,
                        variable: ResidentQuantifierSlot(1),
                        list: Box::new(Expr::Slot(ResidentQuantifierSlot(0))),
                        predicate: Box::new(Expr::Binary {
                            left: Box::new(Expr::Property {
                                source: Box::new(Expr::Slot(ResidentQuantifierSlot(1))),
                                key: "p".into(),
                            }),
                            operation: ResidentQuantifierBinary::Equal,
                            right: Box::new(Expr::Literal(literal)),
                        }),
                    },
                }],
            }],
            outputs: vec![ResidentQuantifierOutput {
                name: "matches".into(),
                source: ResidentQuantifierSlot(2),
            }],
        };
        let request = ResidentQuantifierProgramRequest::build_with_source(
            ResidentQuantifierGeneration {
                project: resident.project,
                bookmark: resident.bookmark,
                graph_revision: resident.revision,
                layout_version: resident.layout_version,
                catalog_generation: [0; 32],
            },
            execution,
            source,
            program,
            8,
            8,
            19,
        )?;
        let result = execute_metal_quantifier_program(
            device,
            resident,
            &request,
            &CancellationToken::new(),
        )?;
        let rows = result.clone().into_untrusted_parts().rows;
        result.validate(&request, BackendKind::Metal)?;
        Ok(rows)
    }

    #[test]
    fn surgical_promoted_bases_execute_nullable_and_quantifier_programs() -> Result<()> {
        use crate::{
            ResidentQuantifierEntityPropertyShape as Shape, ResidentQuantifierValue as Value,
        };
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let mut graph = GraphStore::default();
        let relationship = graph.catalog_mut().intern_relationship_type("LINK")?;
        let originals = [
            (
                ScalarValue::Boolean(true),
                Value::Boolean(true),
                Shape::Boolean,
            ),
            (ScalarValue::Integer(7), Value::Integer(7), Shape::Integer),
            (
                ScalarValue::Float(OrderedFloat(3.5)),
                Value::Float(3.5_f64.to_bits()),
                Shape::Float,
            ),
            (
                ScalarValue::String("base".into()),
                Value::String("base".into()),
                Shape::String { maximum_bytes: 4 },
            ),
        ];
        let properties = (0..originals.len())
            .map(|index| graph.catalog_mut().intern_property(&format!("p{index}")))
            .collect::<Result<Vec<_>>>()?;
        for id in 1..=4 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels: Vec::new(),
                properties: properties
                    .iter()
                    .zip(&originals)
                    .map(|(property, (value, _, _))| (*property, value.clone()))
                    .collect(),
            })?;
        }
        for target in 2..=4 {
            graph.insert_edge(crate::graph::EdgeInput {
                id: crate::EdgeId(target - 1),
                source: NodeId(1),
                target: NodeId(target),
                relationship_type: relationship,
                layer: Layer::Observed,
                revision: 1,
                properties: Vec::new(),
            })?;
        }
        let original = CandleResident::upload(
            ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
            &device,
        )?;
        let mut pinned = Vec::new();
        for (property, (_, literal, shape)) in properties.iter().zip(&originals) {
            pinned.push(quantifier_base_rows(
                &original,
                *property,
                literal.clone(),
                *shape,
                &device,
            )?);
        }
        eprintln!("native promoted readers: checking original nullable generation");
        assert_eq!(
            nullable_base_rows(&original, properties[3], &device)?,
            vec![0, 1, 2, 3]
        );
        graph.rebind_shared(
            original
                .shared_graph
                .as_ref()
                .expect("original backing")
                .clone(),
        )?;
        for (index, property) in properties.iter().enumerate() {
            let conflict = if index == 3 {
                ScalarValue::Integer(42)
            } else {
                ScalarValue::String("different".into())
            };
            graph.set_node_property(NodeId(3), *property, conflict, 2)?;
            graph.set_node_property(NodeId(4), *property, ScalarValue::Null, 2)?;
        }
        graph.insert_node(NodeInput {
            id: NodeId(5),
            layer: Layer::Observed,
            revision: 2,
            labels: Vec::new(),
            properties: properties
                .iter()
                .zip(&originals)
                .map(|(property, (value, _, _))| (*property, value.clone()))
                .collect(),
        })?;
        graph.insert_edge(crate::graph::EdgeInput {
            id: crate::EdgeId(4),
            source: NodeId(1),
            target: NodeId(5),
            relationship_type: relationship,
            layer: Layer::Observed,
            revision: 2,
            properties: Vec::new(),
        })?;
        let staged = original.stage_delta(
            &ResidentProjectDelta {
                project: original.project,
                bookmark: Bookmark { term: 1, index: 2 },
                graph: graph.device_delta(2)?,
                temporal: Vec::new(),
                vectors: Vec::new(),
                invalidate_derived: false,
            },
            &device,
        )?;
        let cold = CandleResident::upload(
            ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
            &device,
        )?;
        for (index, (property, (_, literal, shape))) in
            properties.iter().zip(&originals).enumerate()
        {
            eprintln!("native promoted readers: checking quantifier base {index}");
            let mixed = Shape::MixedScalar {
                maximum_string_bytes: if index == 3 { 4 } else { 9 },
            };
            let actual = quantifier_base_rows(&staged, *property, literal.clone(), mixed, &device)?;
            let expected = quantifier_base_rows(&cold, *property, literal.clone(), mixed, &device)?;
            assert_eq!(actual, expected, "native quantifier base {index}");
            assert_eq!(
                actual,
                vec![
                    vec![Value::Boolean(true)],
                    vec![Value::Boolean(false)],
                    vec![Value::Null],
                    vec![Value::Boolean(true)]
                ]
            );
            assert_eq!(
                quantifier_base_rows(&original, *property, literal.clone(), *shape, &device)?,
                pinned[index]
            );
        }
        eprintln!("native promoted readers: checking staged nullable generation");
        for (operation, expected) in [
            (CompareOp::Eq, vec![0, 1, 4]),
            (CompareOp::NotEq, vec![2]),
            (CompareOp::Less, vec![]),
            (CompareOp::LessOrEqual, vec![0, 1, 4]),
            (CompareOp::Greater, vec![]),
            (CompareOp::GreaterOrEqual, vec![0, 1, 4]),
        ] {
            let actual = nullable_comparison_rows(&staged, properties[3], operation, &device)?;
            assert_eq!(actual, expected, "nullable comparison {operation:?}");
            assert_eq!(
                actual,
                nullable_comparison_rows(&cold, properties[3], operation, &device)?
            );
        }
        assert_eq!(
            nullable_base_rows(&original, properties[3], &device)?,
            vec![0, 1, 2, 3]
        );
        Ok(())
    }

    fn payload(offsets: &Tensor, bytes: &Option<Tensor>, row: usize) -> Result<Vec<u8>> {
        let pair = offsets
            .narrow(0, row * 2, 2)
            .and_then(|value| value.to_vec1::<u32>())
            .map_err(candle_error)?;
        if pair[0] == pair[1] {
            return Ok(Vec::new());
        }
        bytes
            .as_ref()
            .ok_or_else(|| Error::internal("variable bytes missing"))?
            .narrow(0, pair[0] as usize, (pair[1] - pair[0]) as usize)
            .and_then(|value| value.to_vec1::<u8>())
            .map_err(candle_error)
    }

    fn mixed_string_payload(
        resident: &CandleResident,
        property: PropertyId,
        row: usize,
    ) -> Result<Vec<u8>> {
        let column = resident
            .mixed_nodes
            .get(&property)
            .ok_or_else(|| Error::internal("mixed test column missing"))?;
        let overridden = column
            .overrides
            .as_ref()
            .map(|mask| {
                mask.narrow(0, row, 1)
                    .and_then(|value| value.to_vec1::<u8>())
                    .map_err(candle_error)
            })
            .transpose()?
            .is_none_or(|mask| mask[0] != 0);
        if overridden {
            return payload(&column.offsets, &column.bytes, row);
        }
        let Some(MixedBaseColumn::String(base)) = column.base.as_deref() else {
            return Err(Error::internal("mixed string base missing"));
        };
        let id = base
            .values
            .as_ref()
            .ok_or_else(|| Error::internal("mixed base IDs missing"))?
            .narrow(0, row, 1)
            .and_then(|value| value.to_vec1::<u32>())
            .map_err(candle_error)?[0];
        let pair = resident
            .node_string_dictionary
            .offsets
            .narrow(0, id as usize, 2)
            .and_then(|value| value.to_vec1::<u32>())
            .map_err(candle_error)?;
        let mut bytes = vec![MIXED_STRING_TAG];
        if pair[0] != pair[1] {
            bytes.extend(
                resident
                    .node_string_dictionary
                    .bytes
                    .as_ref()
                    .ok_or_else(|| Error::internal("mixed base dictionary missing"))?
                    .narrow(0, pair[0] as usize, (pair[1] - pair[0]) as usize)
                    .and_then(|value| value.to_vec1::<u8>())
                    .map_err(candle_error)?,
            );
        }
        Ok(bytes)
    }

    #[test]
    fn surgical_property_all_variable_kinds_new_lanes_promotion_and_growth_match_cold() -> Result<()>
    {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let mut costs = Vec::new();
        for count in [4_096_u64, 32_768] {
            let mut graph = GraphStore::default();
            let label = graph.catalog_mut().intern_label("Record")?;
            let integer = graph.catalog_mut().intern_property("integer")?;
            let text = graph.catalog_mut().intern_property("text")?;
            let bytes = graph.catalog_mut().intern_property("bytes")?;
            let list = graph.catalog_mut().intern_property("list")?;
            let map = graph.catalog_mut().intern_property("map")?;
            let fresh = graph.catalog_mut().intern_property("fresh")?;
            let original_list = ScalarValue::List(DocumentList::new(vec![DocumentItem::Scalar(
                ScalarValue::String("dirty list λ".repeat(23).into()),
            )])?);
            let original_map = ScalarValue::Map(DocumentMap::new(BTreeMap::from([(
                Arc::from("body"),
                DocumentItem::Scalar(ScalarValue::String("dirty map λ".repeat(29).into())),
            )]))?);
            for id in 1..=count {
                graph.insert_node(NodeInput {
                    id: NodeId(id),
                    layer: Layer::Observed,
                    revision: 1,
                    labels: vec![label],
                    properties: vec![
                        (
                            integer,
                            ScalarValue::Integer((id as i64).wrapping_mul(7_919)),
                        ),
                        (
                            text,
                            ScalarValue::String(format!("original-{id:08}-λ").into()),
                        ),
                        (
                            bytes,
                            ScalarValue::Bytes(
                                (0..257)
                                    .map(|offset| ((id as usize * 31 + offset * 17) % 251) as u8)
                                    .collect::<Vec<_>>()
                                    .into(),
                            ),
                        ),
                        (list, original_list.clone()),
                        (map, original_map.clone()),
                    ],
                })?;
            }
            let original = CandleResident::upload(
                ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
                &device,
            )?;
            let old_text_id = original.string_nodes[&text].values.as_ref().map(Tensor::id);
            let old_bytes = payload(
                &original.byte_nodes[&bytes].offsets,
                &original.byte_nodes[&bytes].bytes,
                2_048,
            )?;
            graph.rebind_shared(
                original
                    .shared_graph
                    .as_ref()
                    .ok_or_else(|| Error::internal("original backing missing"))?
                    .clone(),
            )?;
            graph.set_node_property(NodeId(2_049), integer, ScalarValue::Integer(i64::MIN), 2)?;
            graph.set_node_property(NodeId(2_049), text, ScalarValue::Integer(42), 2)?;
            graph.set_node_property(NodeId(2_049), fresh, ScalarValue::Boolean(true), 2)?;
            graph.set_node_property(
                NodeId(2_049),
                bytes,
                ScalarValue::Bytes(vec![237; 65_539].into()),
                2,
            )?;
            graph.set_node_property(
                NodeId(2_049),
                list,
                ScalarValue::List(DocumentList::new(vec![DocumentItem::Scalar(
                    ScalarValue::String("changed λ".repeat(501).into()),
                )])?),
                2,
            )?;
            graph.set_node_property(
                NodeId(2_049),
                map,
                ScalarValue::Map(DocumentMap::new(BTreeMap::from([(
                    Arc::from("body"),
                    DocumentItem::Scalar(ScalarValue::String("changed map λ".repeat(4_097).into())),
                )]))?),
                2,
            )?;
            let delta = ResidentProjectDelta {
                project: original.project,
                bookmark: Bookmark { term: 1, index: 2 },
                graph: graph.device_delta(2)?,
                temporal: Vec::new(),
                vectors: Vec::new(),
                invalidate_derived: false,
            };
            let (cost, plan) = original.prepare_delta_staging(&delta)?;
            costs.push(cost);
            assert!(
                cost < 16 * 1024 * 1024,
                "private staging unexpectedly grew to {cost}"
            );
            let staged = original.stage_delta_prepared(&delta, &device, Some(plan))?;
            let cold = CandleResident::upload(
                ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
                &device,
            )?;
            for property in [bytes, list, map] {
                let (first, second) = if property == bytes {
                    (&staged.byte_nodes[&property], &cold.byte_nodes[&property])
                } else if property == list {
                    (&staged.list_nodes[&property], &cold.list_nodes[&property])
                } else {
                    (&staged.map_nodes[&property], &cold.map_nodes[&property])
                };
                for row in [7, 2_048, count as usize - 1] {
                    assert_eq!(
                        payload(&first.offsets, &first.bytes, row)?,
                        payload(&second.offsets, &second.bytes, row)?
                    );
                }
            }
            for row in [7, 2_048, count as usize - 1] {
                assert_eq!(
                    mixed_string_payload(&staged, text, row)?,
                    mixed_string_payload(&cold, text, row)?
                );
            }
            let selected_rows = Tensor::from_vec(
                vec![
                    7_u32,
                    2_048,
                    count as u32 - 1,
                    super::super::super::RESIDENT_NULL_ROW,
                ],
                4,
                &device,
            )
            .map_err(candle_error)?;
            for operation in [
                CompareOp::Eq,
                CompareOp::NotEq,
                CompareOp::Less,
                CompareOp::LessOrEqual,
                CompareOp::Greater,
                CompareOp::GreaterOrEqual,
            ] {
                let query = |resident: &CandleResident| {
                    resident
                        .selected_string_comparison(
                            text,
                            &selected_rows,
                            "original-00000008-λ".as_bytes(),
                            operation,
                            &device,
                            &CancellationToken::new(),
                        )?
                        .to_vec1::<u8>()
                        .map_err(candle_error)
                };
                let actual = query(&staged)?;
                assert_eq!(actual, query(&cold)?, "promoted comparison {operation:?}");
                assert_eq!(
                    actual[1],
                    match operation {
                        CompareOp::Eq => 1,
                        CompareOp::NotEq => 2,
                        _ => 0,
                    },
                    "non-string override versus STRING literal {operation:?}"
                );
                assert_eq!(actual[3], 0, "NULL binding stays NULL");
            }
            for operation in [
                super::super::super::ResidentStringPredicateOperation::StartsWith,
                super::super::super::ResidentStringPredicateOperation::EndsWith,
                super::super::super::ResidentStringPredicateOperation::Contains,
            ] {
                let query = |resident: &CandleResident| {
                    resident
                        .selected_string_match(
                            text,
                            &selected_rows,
                            "λ".as_bytes(),
                            operation,
                            &device,
                            &CancellationToken::new(),
                        )?
                        .to_vec1::<u8>()
                        .map_err(candle_error)
                };
                assert_eq!(
                    query(&staged)?,
                    query(&cold)?,
                    "promoted string match {operation:?}"
                );
            }
            let Some(MixedBaseColumn::String(base)) = staged.mixed_nodes[&text].base.as_deref()
            else {
                return Err(Error::internal("promotion rebuilt its base"));
            };
            assert_eq!(base.values.as_ref().map(Tensor::id), old_text_id);
            assert_eq!(
                payload(
                    &original.byte_nodes[&bytes].offsets,
                    &original.byte_nodes[&bytes].bytes,
                    2_048
                )?,
                old_bytes
            );
            assert_eq!(
                staged.boolean_nodes[&fresh]
                    .values
                    .as_ref()
                    .ok_or_else(|| Error::internal("new boolean missing"))?
                    .narrow(0, 2_048, 1)
                    .and_then(|value| value.to_vec1::<u8>())
                    .map_err(candle_error)?,
                vec![1]
            );
            let mut rejected = delta.clone();
            rejected.project = ProjectId(uuid::Uuid::new_v4());
            assert!(original.stage_delta(&rejected, &device).is_err());
            assert_eq!(
                payload(
                    &original.byte_nodes[&bytes].offsets,
                    &original.byte_nodes[&bytes].bytes,
                    2_048
                )?,
                old_bytes
            );
            graph.rebind_shared(
                staged
                    .shared_graph
                    .as_ref()
                    .ok_or_else(|| Error::internal("staged backing missing"))?
                    .clone(),
            )?;
            graph.insert_node(NodeInput {
                id: NodeId(count + 1),
                layer: Layer::Observed,
                revision: 3,
                labels: vec![label],
                properties: vec![
                    (integer, ScalarValue::Integer(777)),
                    (list, original_list.clone()),
                ],
            })?;
            let relationship_type = graph.catalog_mut().intern_relationship_type("LINK")?;
            graph.insert_edge(crate::graph::EdgeInput {
                id: crate::EdgeId(1),
                source: NodeId(1),
                target: NodeId(count + 1),
                relationship_type,
                layer: Layer::Observed,
                revision: 3,
                properties: vec![(list, original_list)],
            })?;
            let grown = staged.stage_delta(
                &ResidentProjectDelta {
                    project: original.project,
                    bookmark: Bookmark { term: 1, index: 3 },
                    graph: graph.device_delta(3)?,
                    temporal: Vec::new(),
                    vectors: Vec::new(),
                    invalidate_derived: false,
                },
                &device,
            )?;
            assert_eq!(grown.node_count, count as usize + 1);
            assert_eq!(grown.edge_count, 1);
            assert_eq!(
                payload(
                    &grown.list_nodes[&list].offsets,
                    &grown.list_nodes[&list].bytes,
                    count as usize
                )?,
                payload(
                    &grown.list_edges[&list].offsets,
                    &grown.list_edges[&list].bytes,
                    0
                )?
            );
            assert_eq!(
                payload(
                    &staged.byte_nodes[&bytes].offsets,
                    &staged.byte_nodes[&bytes].bytes,
                    2_048
                )?,
                vec![237; 65_539]
            );
            graph.rebind_shared(grown.shared_graph.as_ref().expect("grown backing").clone())?;
            for property in [bytes, list, map] {
                graph.set_node_property(NodeId(2_049), property, ScalarValue::Null, 4)?;
            }
            graph.set_node_property(NodeId(8), text, ScalarValue::Null, 4)?;
            let cleared = grown.stage_delta(
                &ResidentProjectDelta {
                    project: original.project,
                    bookmark: Bookmark { term: 1, index: 4 },
                    graph: graph.device_delta(4)?,
                    temporal: Vec::new(),
                    vectors: Vec::new(),
                    invalidate_derived: false,
                },
                &device,
            )?;
            for column in [
                &cleared.byte_nodes[&bytes],
                &cleared.list_nodes[&list],
                &cleared.map_nodes[&map],
            ] {
                assert!(payload(&column.offsets, &column.bytes, 2_048)?.is_empty());
                assert_eq!(
                    column
                        .validity
                        .as_ref()
                        .expect("variable validity")
                        .narrow(0, 2_048, 1)
                        .and_then(|value| value.to_vec1::<u8>())
                        .map_err(candle_error)?,
                    vec![0]
                );
            }
            let cleared_mixed = &cleared.mixed_nodes[&text];
            assert_eq!(
                cleared_mixed
                    .overrides
                    .as_ref()
                    .expect("override mask")
                    .narrow(0, 7, 1)
                    .and_then(|value| value.to_vec1::<u8>())
                    .map_err(candle_error)?,
                vec![1]
            );
            assert_eq!(
                cleared_mixed
                    .validity
                    .as_ref()
                    .expect("mixed validity")
                    .narrow(0, 7, 1)
                    .and_then(|value| value.to_vec1::<u8>())
                    .map_err(candle_error)?,
                vec![0]
            );
            assert_eq!(
                mixed_string_payload(&grown, text, 7)?,
                mixed_string_payload(&staged, text, 7)?
            );
            eprintln!("surgical complete property publication: rows={count}, staging_bytes={cost}");
        }
        assert_eq!(costs[0], costs[1]);
        Ok(())
    }

    fn scalar_lanes(resident: &CandleResident) -> Result<Vec<Vec<i64>>> {
        let mut result = Vec::new();
        macro_rules! lanes {
            ($map:ident, $($field:ident),+) => {
                for column in resident.$map.values() {
                    $(if let Some(tensor) = &column.$field {
                        result.push(tensor.to_dtype(DType::I64).and_then(|tensor| tensor.to_vec1::<i64>()).map_err(candle_error)?);
                    })+
                }
            }
        }
        lanes!(boolean_nodes, values, validity);
        lanes!(boolean_edges, values, validity);
        lanes!(integer_nodes, values, validity);
        lanes!(integer_edges, values, validity);
        lanes!(float_nodes, values, order_keys, validity);
        lanes!(float_edges, values, order_keys, validity);
        lanes!(string_nodes, values, validity);
        lanes!(string_edges, values, validity);
        lanes!(date_nodes, values, validity);
        lanes!(date_edges, values, validity);
        lanes!(local_time_nodes, values, validity);
        lanes!(local_time_edges, values, validity);
        lanes!(zoned_time_nodes, nanos, offsets, validity);
        lanes!(zoned_time_edges, nanos, offsets, validity);
        lanes!(local_datetime_nodes, seconds, nanos, validity);
        lanes!(local_datetime_edges, seconds, nanos, validity);
        lanes!(zoned_datetime_nodes, seconds, nanos, timezones, validity);
        lanes!(zoned_datetime_edges, seconds, nanos, timezones, validity);
        lanes!(duration_nodes, months, days, seconds, nanos, validity);
        lanes!(duration_edges, months, days, seconds, nanos, validity);
        for dictionary in [
            &resident.node_string_dictionary,
            &resident.edge_string_dictionary,
        ] {
            for tensor in [Some(&dictionary.offsets), dictionary.bytes.as_ref()]
                .into_iter()
                .flatten()
            {
                result.push(
                    tensor
                        .to_dtype(DType::I64)
                        .and_then(|tensor| tensor.to_vec1::<i64>())
                        .map_err(candle_error)?,
                );
            }
            let count = dictionary.offsets.elem_count().saturating_sub(1);
            if count != 0 {
                let ids = Tensor::arange(0_u32, count as u32, dictionary.offsets.device())
                    .map_err(candle_error)?;
                result.push(
                    dictionary
                        .ranks_for(&ids, dictionary.offsets.device())?
                        .to_dtype(DType::I64)
                        .and_then(|tensor| tensor.to_vec1::<i64>())
                        .map_err(candle_error)?,
                );
            }
        }
        Ok(result)
    }

    #[test]
    fn surgical_property_scalar_types_new_strings_nulls_and_failed_publish_match_cold_build()
    -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let mut measured = Vec::new();
        for count in [4_096_u64, 32_768] {
            let mut graph = GraphStore::default();
            let label = graph.catalog_mut().intern_label("Record")?;
            let relationship_type = graph.catalog_mut().intern_relationship_type("LINK")?;
            let pairs = [
                (ScalarValue::Boolean(false), ScalarValue::Boolean(true)),
                (
                    ScalarValue::Integer(i64::MAX),
                    ScalarValue::Integer(i64::MIN),
                ),
                (
                    ScalarValue::Float(OrderedFloat(-0.0)),
                    ScalarValue::Float(OrderedFloat(42.5)),
                ),
                (
                    ScalarValue::String("z".repeat(32_769).into()),
                    ScalarValue::String("é".repeat(5_000).into()),
                ),
                (ScalarValue::Date(2000), ScalarValue::Date(-2000)),
                (ScalarValue::LocalTime(2000), ScalarValue::LocalTime(3000)),
                (
                    ScalarValue::ZonedTime {
                        nanos: 2000,
                        offset_seconds: 3600,
                    },
                    ScalarValue::ZonedTime {
                        nanos: 4000,
                        offset_seconds: -3600,
                    },
                ),
                (
                    ScalarValue::LocalDateTime {
                        seconds: 0,
                        nanos: 2,
                    },
                    ScalarValue::LocalDateTime {
                        seconds: -2,
                        nanos: 4,
                    },
                ),
                (
                    ScalarValue::ZonedDateTime {
                        seconds: 0,
                        nanos: 2,
                        timezone: "UTC".into(),
                    },
                    ScalarValue::ZonedDateTime {
                        seconds: 2,
                        nanos: 4,
                        timezone: "Europe/Paris".into(),
                    },
                ),
                (
                    ScalarValue::Duration {
                        months: 1,
                        days: 2,
                        seconds: 3,
                        nanos: 4,
                    },
                    ScalarValue::Duration {
                        months: 5,
                        days: 6,
                        seconds: 7,
                        nanos: 8,
                    },
                ),
            ];
            let properties = pairs
                .iter()
                .enumerate()
                .map(|(index, (value, _))| {
                    Ok((
                        graph.catalog_mut().intern_property(&format!("p{index}"))?,
                        value.clone(),
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let fresh_properties = pairs
                .iter()
                .enumerate()
                .map(|(index, (_, value))| {
                    Ok((
                        graph
                            .catalog_mut()
                            .intern_property(&format!("fresh{index}"))?,
                        value.clone(),
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            for id in 1..=count {
                graph.insert_node(NodeInput {
                    id: NodeId(id),
                    layer: Layer::Observed,
                    revision: 1,
                    labels: vec![label],
                    properties: properties.clone(),
                })?;
            }
            graph.insert_edge(crate::graph::EdgeInput {
                id: crate::EdgeId(7),
                source: NodeId(1),
                target: NodeId(2),
                relationship_type,
                layer: Layer::Observed,
                revision: 1,
                properties: properties.clone(),
            })?;
            let original = CandleResident::upload(
                ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
                &device,
            )?;
            let before = scalar_lanes(&original)?;
            let mut current = original.clone();
            for revision in 2..=4 {
                for ((property, _), (_, value)) in properties
                    .iter()
                    .chain(&fresh_properties)
                    .zip(pairs.iter().cycle())
                {
                    let value = if revision == 3 {
                        ScalarValue::Null
                    } else {
                        value.clone()
                    };
                    graph.set_node_property(NodeId(1), *property, value.clone(), revision)?;
                    graph.set_edge_property(crate::EdgeId(7), *property, value, revision)?;
                }
                let delta = ResidentProjectDelta {
                    project: original.project,
                    bookmark: Bookmark {
                        term: 1,
                        index: revision,
                    },
                    graph: graph.device_delta(revision)?,
                    temporal: Vec::new(),
                    vectors: Vec::new(),
                    invalidate_derived: true,
                };
                let cost = plan(&current, &delta)?.expect("surgical scalar plan").bytes;
                measured.push(cost);
                let staged = current.stage_delta(&delta, &device)?;
                let cold = CandleResident::upload(
                    ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
                    &device,
                )?;
                assert_eq!(scalar_lanes(&staged)?, scalar_lanes(&cold)?);
                assert_eq!(scalar_lanes(&original)?, before);
                let mut rejected = delta.clone();
                rejected.project = ProjectId(uuid::Uuid::new_v4());
                assert!(current.stage_delta(&rejected, &device).is_err());
                assert_eq!(scalar_lanes(&original)?, before);
                current = staged;
            }
        }
        assert_eq!(&measured[..3], &measured[3..]);
        eprintln!("surgical all scalar publication costs: {measured:?}");
        Ok(())
    }
    use crate::DocumentItem;
    use crate::graph::{GraphStore, NodeInput};

    #[test]
    fn metal_property_point_writes_keep_pinned_generations_and_bounded_staging() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let device = Device::new_metal(0).map_err(candle_error)?;
        let mut measured = Vec::new();
        for count in [4_096_u64, 32_768] {
            let mut graph = GraphStore::default();
            let property = graph.catalog_mut().intern_property("value")?;
            let label = graph.catalog_mut().intern_label("Record")?;
            for id in 1..=count {
                graph.insert_node(NodeInput {
                    id: NodeId(id),
                    layer: Layer::Observed,
                    revision: 1,
                    labels: vec![label],
                    properties: vec![(
                        property,
                        ScalarValue::Integer((id as i64).wrapping_mul(7_919)),
                    )],
                })?;
            }
            let original = CandleResident::upload(
                ResidentProjectImage::graph_only(Arc::new(graph.snapshot()?)),
                &device,
            )?;
            let old_backing = original
                .shared_graph
                .as_ref()
                .ok_or_else(|| Error::internal("missing backing"))?
                .clone();
            graph.rebind_shared(old_backing.clone())?;
            let mut current = original.clone();
            let selected = Tensor::from_slice(&[2_048_u32, 7], 2, &device).map_err(candle_error)?;
            for revision in 2..=12 {
                graph.set_node_property(
                    NodeId(2_049),
                    property,
                    ScalarValue::Integer(-(revision as i64)),
                    revision,
                )?;
                let delta = ResidentProjectDelta {
                    project: original.project,
                    bookmark: Bookmark {
                        term: 1,
                        index: revision,
                    },
                    graph: graph.device_delta(revision)?,
                    temporal: Vec::new(),
                    vectors: Vec::new(),
                    invalidate_derived: false,
                };
                let (bytes, plan) = current.prepare_delta_staging(&delta)?;
                if revision == 2 {
                    measured.push(bytes);
                }
                assert!(
                    bytes < 128 * 1024,
                    "one-row staging grew to {bytes} bytes for {count} rows"
                );
                let staged = current.stage_delta_prepared(&delta, &device, Some(plan))?;
                let gather = |resident: &CandleResident| -> Result<Vec<i64>> {
                    resident.integer_nodes[&property]
                        .values
                        .as_ref()
                        .ok_or_else(|| Error::internal("missing values"))?
                        .index_select(&selected, 0)
                        .and_then(|tensor| tensor.to_vec1::<i64>())
                        .map_err(candle_error)
                };
                assert_eq!(gather(&staged)?, vec![-(revision as i64), 8 * 7_919]);
                assert_eq!(gather(&original)?, vec![2_049 * 7_919, 8 * 7_919]);
                assert_eq!(
                    old_backing.node_properties.get(2_048, property),
                    Some(ScalarValue::Integer(2_049 * 7_919))
                );
                assert_eq!(
                    current.integer_nodes[&property]
                        .validity
                        .as_ref()
                        .map(Tensor::elem_count),
                    Some(count as usize)
                );
                graph.rebind_shared(
                    staged
                        .shared_graph
                        .as_ref()
                        .ok_or_else(|| Error::internal("missing staged backing"))?
                        .clone(),
                )?;
                current = staged;
            }
            eprintln!(
                "Metal property update: rows={count}, staging_bytes={}",
                measured.last().copied().unwrap_or(0)
            );
            graph.insert_node(NodeInput {
                id: NodeId(count + 1),
                layer: Layer::Observed,
                revision: 13,
                labels: vec![label],
                properties: vec![(property, ScalarValue::Integer(777))],
            })?;
            let appended = current.stage_delta(
                &ResidentProjectDelta {
                    project: original.project,
                    bookmark: Bookmark { term: 1, index: 13 },
                    graph: graph.device_delta(13)?,
                    temporal: Vec::new(),
                    vectors: Vec::new(),
                    invalidate_derived: false,
                },
                &device,
            )?;
            let rows =
                Tensor::from_slice(&[2_048_u32, count as u32], 2, &device).map_err(candle_error)?;
            let values = appended.integer_nodes[&property]
                .values
                .as_ref()
                .ok_or_else(|| Error::internal("missing appended values"))?
                .index_select(&rows, 0)
                .and_then(|v| v.to_vec1::<i64>())
                .map_err(candle_error)?;
            assert_eq!(values, vec![-12, 777]);
        }
        assert_eq!(measured[0], measured[1]);
        Ok(())
    }
}
