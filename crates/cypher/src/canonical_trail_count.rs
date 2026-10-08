//! Fused canonical trail enumeration and scalar count. No graph image or path rows survive.
use super::*;

#[cfg(test)]
pub(super) fn reductions() -> usize {
    REDUCTIONS.with(std::cell::Cell::get)
}
#[cfg(test)]
pub(super) fn parallel_reductions() -> usize {
    PARALLEL_REDUCTIONS.with(std::cell::Cell::get)
}
#[cfg(test)]
thread_local! {
    static REDUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PARALLEL_REDUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn reduce(
    operators: &[PhysicalOperator],
    rows: &[Row],
    state: &mut ExecutionState<'_>,
    context: &ExecutionContext<'_>,
) -> Result<Option<(Vec<Row>, usize)>> {
    if rows.len() != 1
        || !state.scope_columns.is_empty()
        || context.backend.is_some()
        || context.capabilities.require_native_execution
        || context.resolved_query_at_time_nanos.is_some()
        || context
            .temporal
            .is_some_and(|temporal| !temporal.is_empty())
        || !context.prior_graph_mutations.is_empty()
        || !context.prior_temporal_mutations.is_empty()
        || !state.mutations.is_empty()
        || !state.temporal_mutations.is_empty()
    {
        return Ok(None);
    }
    let Some(PhysicalOperator::ScanPattern {
        optional: false,
        pattern,
        access,
        ..
    }) = operators.first()
    else {
        return Ok(None);
    };
    if matches!(
        access,
        ScanAccessPath::EqualityIndex { .. } | ScanAccessPath::StableId { .. }
    ) || pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.property_predicate_present
        || !pattern.start.properties.is_empty()
        || pattern.steps.is_empty()
    {
        return Ok(None);
    }
    let Some(start_variable) = pattern.start.variable.as_deref() else {
        return Ok(None);
    };
    let Some(end_variable) = pattern
        .steps
        .last()
        .and_then(|step| step.node.variable.as_deref())
    else {
        return Ok(None);
    };
    if start_variable == end_variable {
        return Ok(None);
    }
    let mut steps = Vec::with_capacity(pattern.steps.len());
    let mut total_hops = 0_u32;
    for (index, step) in pattern.steps.iter().enumerate() {
        let relationship = &step.relationship;
        if relationship.variable.is_some()
            || !relationship.properties.is_empty()
            || relationship.direction == Direction::Undirected
            || step.node.property_predicate_present
            || !step.node.properties.is_empty()
            || (index + 1 != pattern.steps.len() && step.node.variable.is_some())
        {
            return Ok(None);
        }
        let (min, max) = if relationship.variable_length {
            let Some(max) = relationship.max_hops else {
                return Ok(None);
            };
            (relationship.min_hops.unwrap_or(1), max)
        } else {
            (1, 1)
        };
        total_hops = total_hops.saturating_add(max);
        if min > max || total_hops > 64 {
            return Ok(None);
        }
        let types = relationship
            .types
            .iter()
            .filter_map(|name| state.graph.relationship_type(name))
            .collect::<Vec<_>>();
        let labels = step
            .node
            .labels
            .iter()
            .map(|name| state.graph.label(name))
            .collect::<Option<Vec<_>>>();
        steps.push(Step {
            min,
            max,
            outgoing: relationship.direction == Direction::Outgoing,
            typed: !relationship.types.is_empty(),
            types,
            labels,
        });
    }
    let mut cursor = 1;
    let mut predicates = Vec::new();
    while let Some(operator) = operators.get(cursor) {
        match operator {
            PhysicalOperator::CardinalityCheckpoint { .. } => {}
            PhysicalOperator::Filter(predicate)
                if safe_start_equality(predicate, Some(start_variable), context) =>
            {
                predicates.push(predicate)
            }
            _ => break,
        }
        cursor += 1;
    }
    if predicates.is_empty() {
        return Ok(None);
    }
    let Some(PhysicalOperator::Project {
        keep_scope: false,
        projection,
    }) = operators.get(cursor)
    else {
        return Ok(None);
    };
    if cursor + 1 != operators.len() || projection.distinct {
        return Ok(None);
    }
    let [item] = projection.items.as_slice() else {
        return Ok(None);
    };
    let Expression::Function {
        name,
        distinct,
        arguments,
    } = &item.expression
    else {
        return Ok(None);
    };
    if !matches!(name.as_slice(), [only] if only.eq_ignore_ascii_case("count"))
        || !matches!(arguments.as_slice(), [Expression::Variable(variable)] if variable == end_variable)
    {
        return Ok(None);
    }
    let Some(labels) = pattern
        .start
        .labels
        .iter()
        .map(|name| state.graph.label(name))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let Some(filters) = canonical_integer_seed_readers(&pattern.start, &predicates, state, context)
    else {
        return Ok(None);
    };
    #[cfg(test)]
    REDUCTIONS.with(|count| count.set(count.get() + 1));
    let budget = std::sync::atomic::AtomicUsize::new(0);
    let run = |start: usize, end: usize| {
        let mut walk = Walk {
            steps: &steps,
            predicates: &predicates,
            state,
            context,
            seed: Row::new(),
            edges: SmallVec::new(),
            count: 0,
            trails: 0,
            distinct: distinct.then(std::collections::HashSet::<NodeId>::new),
            ticks: 0,
            entities: BTreeMap::new(),
            budget: &budget,
        };
        let (reader, operand) = &filters[0];
        reader.visit_integer_candidates_range(
            start,
            end,
            *operand,
            || check_execution(context),
            |seed| {
                let Some(node) = context.graph.node_dense(seed) else {
                    return Ok(());
                };
                if !state.read_layers.contains_layer(node.layer())
                    || !node.has_labels(&labels)
                    || !filters.iter().all(|(reader, value)| {
                        canonical_seed_integer_may_match(reader.get_integer(node), *value)
                    })
                {
                    return Ok(());
                }
                walk.seed
                    .insert(start_variable.to_owned(), BindingValue::Node(seed));
                walk.record_node(node);
                if !walk.selected()? {
                    return Ok(());
                }
                walk.step(node, 0, 0)
            },
        )?;
        walk.add_budget(walk.trails % 64)?;
        Ok::<_, Error>((walk.count, walk.distinct, walk.entities))
    };
    let slots = context.graph.node_slot_count();
    let permit = (!distinct && slots >= 32768)
        .then(integer_scan_pool)
        .flatten()
        .and_then(IntegerScanPool::try_claim);
    let partials = if let Some(permit) = permit {
        #[cfg(test)]
        PARALLEL_REDUCTIONS.with(|count| count.set(count.get() + 1));
        let tasks = permit
            .0
            .workers
            .current_num_threads()
            .min(slots.div_ceil(4096));
        let width = slots.div_ceil(tasks).div_ceil(256) * 256;
        permit.0.workers.install(|| {
            (0..tasks)
                .into_par_iter()
                .with_max_len(1)
                .map(|task| run(task * width, ((task + 1) * width).min(slots)))
                .collect::<Result<Vec<_>>>()
        })?
    } else {
        vec![run(0, slots)?]
    };
    let mut count = 0_usize;
    if partials.len() == 1 {
        let (partial, ids, mut entities) = partials
            .into_iter()
            .next()
            .ok_or_else(|| Error::internal("trail reduction missing"))?;
        count = ids.as_ref().map_or(partial, |ids| ids.len());
        state.dependencies.entities.append(&mut entities);
    } else {
        let capacity = partials.iter().map(|(_, _, entries)| entries.len()).sum();
        let mut entries = Vec::with_capacity(capacity);
        for (partial, ids, entities) in partials {
            debug_assert!(ids.is_none());
            count = count
                .checked_add(partial)
                .ok_or_else(|| Error::internal("trail count overflow"))?;
            entries.extend(entities);
        }
        let mut entities = entries.into_iter().collect::<BTreeMap<_, _>>();
        state.dependencies.entities.append(&mut entities);
    }
    check_execution(context)?;
    let count = i64::try_from(count).map_err(|_| Error::internal("trail count exceeds INTEGER"))?;
    let name = item.column_name(0);
    let mut row = Row::new();
    row.insert(
        name.clone(),
        BindingValue::Value(ResultValue::Scalar(ScalarValue::Integer(count))),
    );
    state.final_columns = vec![name.clone()];
    state.scope_columns = BTreeSet::from([name]);
    Ok(Some((vec![row], operators.len())))
}

struct Step {
    min: u32,
    max: u32,
    outgoing: bool,
    typed: bool,
    types: Vec<RelationshipTypeId>,
    labels: Option<Vec<LabelId>>,
}

struct Walk<'a, 'g, 'c> {
    steps: &'a [Step],
    predicates: &'a [&'a Expression],
    state: &'a ExecutionState<'g>,
    context: &'a ExecutionContext<'c>,
    seed: Row,
    edges: SmallVec<[u32; 8]>,
    count: usize,
    trails: usize,
    distinct: Option<std::collections::HashSet<NodeId>>,
    ticks: usize,
    entities: BTreeMap<EntityDependency, u64>,
    budget: &'a std::sync::atomic::AtomicUsize,
}

impl Walk<'_, '_, '_> {
    fn selected(&self) -> Result<bool> {
        for predicate in self.predicates {
            if truth(&evaluate_state(
                predicate,
                &self.seed,
                self.state,
                self.context,
            )?)? != Truth::True
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn record_node(&mut self, node: crate::graph::NodeView<'_>) {
        self.entities
            .entry(EntityDependency::Node(node.id()))
            .or_insert(node.revision());
    }

    fn add_budget(&self, amount: usize) -> Result<()> {
        if amount == 0 {
            return Ok(());
        }
        let previous = self.budget.fetch_add(amount, AtomicOrdering::Relaxed);
        if previous
            .checked_add(amount)
            .is_none_or(|total| total > self.context.max_result_rows)
        {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "pattern match exceeded the result row budget",
            ));
        }
        Ok(())
    }

    fn step(&mut self, node: crate::graph::NodeView<'_>, index: usize, hops: u32) -> Result<()> {
        self.ticks += 1;
        if self.ticks.is_multiple_of(64) {
            check_execution(self.context)?;
        }
        if !self.state.read_layers.contains_layer(node.layer()) {
            return Ok(());
        }
        let step = &self.steps[index];
        if hops >= step.min
            && step
                .labels
                .as_ref()
                .is_some_and(|labels| node.has_labels(labels))
        {
            if index + 1 == self.steps.len() {
                self.trails = self
                    .trails
                    .checked_add(1)
                    .ok_or_else(|| Error::internal("trail budget overflow"))?;
                if self.trails > self.context.max_result_rows {
                    return Err(Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "pattern match exceeded the result row budget",
                    ));
                }
                if self.trails.is_multiple_of(64) {
                    self.add_budget(64)?;
                }
                if let Some(ids) = &mut self.distinct {
                    ids.insert(node.id());
                } else {
                    self.count += 1;
                }
            } else {
                self.step(node, index + 1, 0)?;
            }
        }
        if hops == step.max {
            return Ok(());
        }
        let graph = self.context.graph;
        graph.try_visit_neighbors(
            node,
            step.outgoing,
            self.state.read_layers,
            || check_execution(self.context),
            |edge, neighbor| {
                let edge_dense = edge.dense();
                if self.edges.contains(&edge_dense) {
                    return Ok(());
                }
                if step.typed && !step.types.contains(&edge.relationship_type()) {
                    return Ok(());
                }
                self.entities
                    .entry(EntityDependency::Relationship(edge.id()))
                    .or_insert(edge.revision());
                self.record_node(neighbor);
                self.edges.push(edge_dense);
                let result = self.step(neighbor, index, hops + 1);
                self.edges.pop();
                result
            },
        )?;
        Ok(())
    }
}
