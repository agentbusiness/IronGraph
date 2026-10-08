//! Deterministic CPU references for priority graph algorithms.

use std::{
    borrow::Cow,
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, VecDeque},
};

use ordered_float::OrderedFloat;

use crate::{Error, ErrorCode, Result};

use super::Csr;

/// A live topology accessor. Implementations may borrow canonical edges without an image.
pub trait AdjacencyRead {
    type Row<'a>: DoubleEndedIterator<Item = (u32, u32)> + ExactSizeIterator
    where
        Self: 'a;
    fn node_count(&self) -> usize;
    fn row(&self, node: u32) -> Option<Self::Row<'_>>;
    /// Appends one live row without clearing the caller's scratch buffer. The appended
    /// neighbors need not be sorted; false means that the node row is absent.
    fn append_row(&self, node: u32, output: &mut Vec<(u32, u32)>) -> bool {
        if let Some(row) = self.row(node) {
            output.extend(row);
            true
        } else {
            false
        }
    }
    fn degree(&self, node: u32) -> usize {
        self.row(node).map_or(0, |row| row.len())
    }
    fn edge_count(&self) -> usize {
        (0..self.node_count())
            .map(|row| self.degree(row as u32))
            .sum()
    }
}
impl AdjacencyRead for Csr {
    type Row<'a> = super::adjacency::CsrRow<'a>;
    fn node_count(&self) -> usize {
        self.offsets().len().saturating_sub(1)
    }
    fn row(&self, node: u32) -> Option<Self::Row<'_>> {
        Csr::row(self, node)
    }
    fn degree(&self, node: u32) -> usize {
        self.offsets()
            .get(node as usize)
            .zip(self.offsets().get(node as usize + 1))
            .map_or(0, |(start, end)| end.saturating_sub(*start) as usize)
    }
    fn edge_count(&self) -> usize {
        self.neighbors().len()
    }
}

/// Component assignment by dense node ordinal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Components {
    pub component: Vec<u32>,
    pub count: u32,
}

/// Distances and predecessor tree from deterministic Dijkstra execution.
#[derive(Clone, Debug, PartialEq)]
pub struct DijkstraResult {
    pub distance: Vec<Option<f64>>,
    pub predecessor: Vec<Option<u32>>,
}

/// Frozen PageRank convergence parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageRankConfig {
    pub damping: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            tolerance: 1.0e-9,
            max_iterations: 100,
        }
    }
}

/// Breadth-first hop distances from `source`.
pub fn bfs<A: AdjacencyRead>(adjacency: &A, source: u32) -> Result<Vec<Option<u32>>> {
    bfs_cancellable(adjacency, source, || Ok(()))
}

/// Breadth-first hop distances with bounded-latency cancellation checks.
///
/// Delegates to the CSR's multi-source frontier expansion (iterated boolean `Aᵀ·f`) with a single
/// seed; the level-synchronous formulation yields the same shortest-hop distances as a queue BFS.
pub fn bfs_cancellable<A: AdjacencyRead>(
    adjacency: &A,
    source: u32,
    check: impl FnMut() -> Result<()>,
) -> Result<Vec<Option<u32>>> {
    let node_count = adjacency.node_count();
    if source as usize >= node_count {
        return Err(Error::new(
            ErrorCode::QueryType,
            "BFS source is out of bounds",
        ));
    }
    let mut check = check;
    let mut distances = vec![None; node_count];
    let mut frontier = VecDeque::from([source]);
    distances[source as usize] = Some(0_u32);
    let mut scanned = 0_usize;
    let mut neighbors = Vec::new();
    while let Some(node) = frontier.pop_front() {
        check()?;
        let next = distances[node as usize].unwrap_or(0).saturating_add(1);
        neighbors.clear();
        if adjacency.append_row(node, &mut neighbors) {
            neighbors.sort_unstable();
            for &(neighbor, _) in &neighbors {
                scanned += 1;
                if scanned & 4095 == 0 {
                    check()?;
                }
                if distances[neighbor as usize].is_none() {
                    distances[neighbor as usize] = Some(next);
                    frontier.push_back(neighbor);
                }
            }
        }
    }
    Ok(distances)
}

/// Iterative deterministic depth-first preorder.
pub fn dfs<A: AdjacencyRead>(adjacency: &A, source: u32) -> Result<Vec<u32>> {
    dfs_cancellable(adjacency, source, || Ok(()))
}

/// Iterative deterministic depth-first preorder with cancellation checks.
pub fn dfs_cancellable<A: AdjacencyRead>(
    adjacency: &A,
    source: u32,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Vec<u32>> {
    let node_count = adjacency.node_count();
    if source as usize >= node_count {
        return Err(Error::new(
            ErrorCode::QueryType,
            "DFS source is out of bounds",
        ));
    }
    let mut visited = vec![false; node_count];
    let mut stack = vec![source];
    let mut order = Vec::new();
    let mut neighbors = Vec::new();
    while let Some(node) = stack.pop() {
        if order.len() & 1023 == 0 {
            check()?;
        }
        if visited[node as usize] {
            continue;
        }
        visited[node as usize] = true;
        order.push(node);
        neighbors.clear();
        if adjacency.append_row(node, &mut neighbors) {
            neighbors.sort_unstable();
            stack.extend(
                neighbors
                    .iter()
                    .rev()
                    .map(|&(neighbor, _)| neighbor)
                    .filter(|neighbor| !visited[*neighbor as usize]),
            );
        }
    }
    Ok(order)
}

/// Deterministic unweighted shortest path including both endpoints.
pub fn shortest_path<A: AdjacencyRead>(
    adjacency: &A,
    source: u32,
    target: u32,
) -> Result<Option<Vec<u32>>> {
    shortest_path_cancellable(adjacency, source, target, || Ok(()))
}

/// Deterministic unweighted shortest path with cancellation checks.
pub fn shortest_path_cancellable<A: AdjacencyRead>(
    adjacency: &A,
    source: u32,
    target: u32,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Option<Vec<u32>>> {
    let node_count = adjacency.node_count();
    if source as usize >= node_count || target as usize >= node_count {
        return Err(Error::new(
            ErrorCode::QueryType,
            "shortest-path endpoint is out of bounds",
        ));
    }
    check()?;
    let mut predecessor = vec![None; node_count];
    let mut visited = vec![false; node_count];
    let mut neighbors = Vec::new();
    visited[source as usize] = true;
    let mut queue = VecDeque::from([source]);
    while let Some(node) = queue.pop_front() {
        if node & 1023 == 0 {
            check()?;
        }
        if node == target {
            break;
        }
        neighbors.clear();
        adjacency.append_row(node, &mut neighbors);
        neighbors.sort_unstable();
        for &(neighbor, _) in &neighbors {
            if !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                predecessor[neighbor as usize] = Some(node);
                queue.push_back(neighbor);
            }
        }
    }
    check()?;
    if !visited[target as usize] {
        return Ok(None);
    }
    let mut path = vec![target];
    let mut cursor = target;
    while cursor != source {
        let Some(previous) = predecessor[cursor as usize] else {
            return Err(Error::internal(
                "shortest-path predecessor chain is incomplete",
            ));
        };
        path.push(previous);
        cursor = previous;
    }
    path.reverse();
    Ok(Some(path))
}

/// Dijkstra with non-negative finite edge weights supplied by edge ordinal.
pub fn dijkstra<A: AdjacencyRead, F>(
    adjacency: &A,
    source: u32,
    mut weight: F,
) -> Result<DijkstraResult>
where
    F: FnMut(u32) -> Result<f64>,
{
    dijkstra_cancellable(adjacency, source, &mut weight, || Ok(()))
}

/// Deterministic Dijkstra execution with cancellation checks.
pub fn dijkstra_cancellable<A: AdjacencyRead, F>(
    adjacency: &A,
    source: u32,
    mut weight: F,
    mut check: impl FnMut() -> Result<()>,
) -> Result<DijkstraResult>
where
    F: FnMut(u32) -> Result<f64>,
{
    let node_count = adjacency.node_count();
    if source as usize >= node_count {
        return Err(Error::new(
            ErrorCode::QueryType,
            "Dijkstra source is out of bounds",
        ));
    }
    let mut distance = vec![None; node_count];
    let mut predecessor = vec![None; node_count];
    distance[source as usize] = Some(0.0);
    let mut heap = BinaryHeap::from([(Reverse(OrderedFloat(0.0_f64)), Reverse(source))]);
    let mut visited_edges = 0_usize;
    let mut neighbors = Vec::new();
    while let Some((Reverse(OrderedFloat(candidate)), Reverse(node))) = heap.pop() {
        check()?;
        if distance[node as usize].is_some_and(|current| candidate > current) {
            continue;
        }
        neighbors.clear();
        if adjacency.append_row(node, &mut neighbors) {
            neighbors.sort_unstable();
            for &(neighbor, edge) in &neighbors {
                visited_edges = visited_edges.saturating_add(1);
                if visited_edges & 4095 == 0 {
                    check()?;
                }
                let edge_weight = weight(edge)?;
                if !edge_weight.is_finite() || edge_weight < 0.0 {
                    return Err(Error::new(
                        ErrorCode::QueryType,
                        "Dijkstra weight must be finite and non-negative",
                    ));
                }
                let next = candidate + edge_weight;
                // The source is the root of the shortest-path tree by definition. A zero-weight
                // cycle may return to it at equal distance, but must never manufacture a parent
                // for the root or enqueue it solely for a predecessor tie.
                if neighbor == source {
                    continue;
                }
                let improve = distance[neighbor as usize].is_none_or(|current| {
                    next < current
                        || (next == current
                            && predecessor[neighbor as usize].is_none_or(|old| node < old))
                });
                if improve {
                    distance[neighbor as usize] = Some(next);
                    predecessor[neighbor as usize] = Some(node);
                    heap.push((Reverse(OrderedFloat(next)), Reverse(neighbor)));
                }
            }
        }
    }
    Ok(DijkstraResult {
        distance,
        predecessor,
    })
}

/// Weak components over the union of outgoing and incoming edges.
pub fn weakly_connected_components<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
) -> Result<Components> {
    weakly_connected_components_cancellable(outgoing, incoming, || Ok(()))
}

/// Weak components with a bounded-latency cancellation/deadline callback.
pub fn weakly_connected_components_cancellable<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Components> {
    validate_pair(outgoing, incoming)?;
    let node_count = outgoing.node_count();
    let mut component = vec![u32::MAX; node_count];
    let mut count = 0_u32;
    let mut queue = VecDeque::new();
    let mut neighbors = Vec::new();
    for start in 0..node_count {
        if start & 1023 == 0 {
            check()?;
        }
        if component[start] != u32::MAX {
            continue;
        }
        component[start] = count;
        queue.push_back(start as u32);
        while let Some(node) = queue.pop_front() {
            if node & 1023 == 0 {
                check()?;
            }
            for adjacency in [outgoing, incoming] {
                neighbors.clear();
                if adjacency.append_row(node, &mut neighbors) {
                    neighbors.sort_unstable();
                    for &(neighbor, _) in &neighbors {
                        if component[neighbor as usize] == u32::MAX {
                            component[neighbor as usize] = count;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }
        count = count.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "component ID space exhausted",
            )
        })?;
    }
    Ok(Components { component, count })
}

/// Strong components via iterative Kosaraju traversal.
pub fn strongly_connected_components<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
) -> Result<Components> {
    strongly_connected_components_cancellable(outgoing, incoming, || Ok(()))
}

/// Strong components via iterative Kosaraju traversal with cancellation checks.
pub fn strongly_connected_components_cancellable<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Components> {
    validate_pair(outgoing, incoming)?;
    let node_count = outgoing.node_count();
    let mut visited = vec![false; node_count];
    let mut finish = Vec::with_capacity(node_count);
    let mut neighbors = Vec::new();
    for start in 0..node_count {
        if start & 1023 == 0 {
            check()?;
        }
        if visited[start] {
            continue;
        }
        let mut stack = vec![(start as u32, false)];
        while let Some((node, exiting)) = stack.pop() {
            if exiting {
                finish.push(node);
                continue;
            }
            if visited[node as usize] {
                continue;
            }
            visited[node as usize] = true;
            stack.push((node, true));
            neighbors.clear();
            if outgoing.append_row(node, &mut neighbors) {
                neighbors.sort_unstable();
                stack.extend(
                    neighbors
                        .iter()
                        .rev()
                        .map(|&(neighbor, _)| neighbor)
                        .filter(|neighbor| !visited[*neighbor as usize])
                        .map(|neighbor| (neighbor, false)),
                );
            }
        }
    }
    let mut component = vec![u32::MAX; node_count];
    let mut count = 0_u32;
    for start in finish.into_iter().rev() {
        if component[start as usize] != u32::MAX {
            continue;
        }
        let mut stack = vec![start];
        component[start as usize] = count;
        while let Some(node) = stack.pop() {
            if node & 1023 == 0 {
                check()?;
            }
            neighbors.clear();
            if incoming.append_row(node, &mut neighbors) {
                neighbors.sort_unstable();
                for &(neighbor, _) in &neighbors {
                    if component[neighbor as usize] == u32::MAX {
                        component[neighbor as usize] = count;
                        stack.push(neighbor);
                    }
                }
            }
        }
        count = count.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "component ID space exhausted",
            )
        })?;
    }
    // Kosaraju's discovery IDs depend on DFS finish order even though component IDs are nominal.
    // Canonicalize by each SCC's smallest dense node so CPU, Metal, and alternative parallel SCC
    // schedules expose one stable contract.  This leaves the partition unchanged.
    let mut minimum_by_old = vec![u32::MAX; count as usize];
    for (node, old) in component.iter().copied().enumerate() {
        let minimum = minimum_by_old
            .get_mut(old as usize)
            .ok_or_else(|| Error::internal("SCC produced an out-of-range component identifier"))?;
        *minimum = (*minimum).min(u32::try_from(node).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "SCC node ordinal exceeds u32",
            )
        })?);
    }
    let mut ordered = minimum_by_old
        .iter()
        .copied()
        .enumerate()
        .map(|(old, minimum)| (minimum, old))
        .collect::<Vec<_>>();
    ordered.sort_unstable();
    let mut canonical_by_old = vec![u32::MAX; count as usize];
    for (canonical, (_, old)) in ordered.into_iter().enumerate() {
        canonical_by_old[old] = u32::try_from(canonical).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "component ID space exhausted",
            )
        })?;
    }
    for value in &mut component {
        *value = *canonical_by_old
            .get(*value as usize)
            .ok_or_else(|| Error::internal("SCC canonicalization lost a component identifier"))?;
    }
    Ok(Components { component, count })
}

/// Deterministic power-iteration PageRank.
pub fn page_rank<A: AdjacencyRead>(adjacency: &A, config: PageRankConfig) -> Result<Vec<f64>> {
    page_rank_cancellable(adjacency, config, || Ok(()))
}

/// PageRank with cancellation/deadline checks inside every iteration and scan chunk.
pub fn page_rank_cancellable<A: AdjacencyRead>(
    adjacency: &A,
    config: PageRankConfig,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Vec<f64>> {
    if !(0.0..1.0).contains(&config.damping)
        || !config.tolerance.is_finite()
        || config.tolerance <= 0.0
        || config.max_iterations == 0
    {
        return Err(Error::new(
            ErrorCode::QueryType,
            "invalid PageRank configuration",
        ));
    }
    let node_count = adjacency.node_count();
    if node_count == 0 {
        return Ok(Vec::new());
    }
    let count = node_count as f64;
    let mut rank = vec![1.0 / count; node_count];
    let mut next = vec![0.0; node_count];
    let mut neighbors = Vec::new();
    for _ in 0..config.max_iterations {
        check()?;
        let dangling: f64 = rank
            .iter()
            .enumerate()
            .filter(|(row, _)| adjacency.degree(*row as u32) == 0)
            .map(|(_, value)| *value)
            .sum();
        let base = (1.0 - config.damping) / count + config.damping * dangling / count;
        next.fill(base);
        for source in 0..node_count {
            if source & 4095 == 0 {
                check()?;
            }
            let degree = adjacency.degree(source as u32);
            if degree == 0 {
                continue;
            }
            let contribution = config.damping * rank[source] / degree as f64;
            neighbors.clear();
            if adjacency.append_row(source as u32, &mut neighbors) {
                neighbors.sort_unstable();
                for &(target, _) in &neighbors {
                    next[target as usize] += contribution;
                }
            }
        }
        let change: f64 = rank
            .iter()
            .zip(&next)
            .map(|(left, right)| (left - right).abs())
            .sum();
        std::mem::swap(&mut rank, &mut next);
        if change <= config.tolerance {
            break;
        }
    }
    Ok(rank)
}

/// Undirected triangle count with each triangle counted exactly once.
pub fn triangle_count<A: AdjacencyRead>(outgoing: &A, incoming: &A) -> Result<u64> {
    triangle_count_cancellable(outgoing, incoming, || Ok(()))
}

/// Undirected triangle count with cancellation checks.
pub fn triangle_count_cancellable<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
    check: impl FnMut() -> Result<()>,
) -> Result<u64> {
    triangle_count_range_cancellable(outgoing, incoming, 0..outgoing.node_count(), check)
}

/// Counts triangles whose smallest vertex lies in the selected range.
/// Disjoint ranges share the live adjacency and use only two reusable row buffers.
pub fn triangle_count_range_cancellable<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
    range: std::ops::Range<usize>,
    mut check: impl FnMut() -> Result<()>,
) -> Result<u64> {
    validate_pair(outgoing, incoming)?;
    if range.start > range.end || range.end > outgoing.node_count() {
        return Err(Error::internal("triangle vertex range is invalid"));
    }
    check()?;
    let mut count = 0_u64;
    let mut first_neighbors = Vec::new();
    let mut second_neighbors = Vec::new();
    for first in range {
        if first & 255 == 0 {
            check()?;
        }
        undirected_row_into(outgoing, incoming, first as u32, &mut first_neighbors);
        let second_start =
            first_neighbors.partition_point(|(neighbor, _)| *neighbor <= first as u32);
        for (position, &(second, _)) in first_neighbors[second_start..].iter().enumerate() {
            if position != 0 && position & 255 == 0 {
                check()?;
            }
            let first_start = first_neighbors.partition_point(|(neighbor, _)| *neighbor <= second);
            if first_start == first_neighbors.len() {
                continue;
            }
            undirected_row_into(outgoing, incoming, second, &mut second_neighbors);
            let third_start = second_neighbors.partition_point(|(neighbor, _)| *neighbor <= second);
            let common = sorted_neighbor_intersection_count(
                &first_neighbors[first_start..],
                &second_neighbors[third_start..],
                &mut check,
            )?;
            count = count.checked_add(common).ok_or_else(|| {
                Error::new(ErrorCode::ResultBudgetExceeded, "triangle count overflow")
            })?;
        }
    }
    check()?;
    Ok(count)
}

/// Local undirected clustering coefficient for every node.
pub fn clustering_coefficients<A: AdjacencyRead>(outgoing: &A, incoming: &A) -> Result<Vec<f64>> {
    clustering_coefficients_cancellable(outgoing, incoming, || Ok(()))
}

/// Local undirected clustering coefficients with cancellation checks.
pub fn clustering_coefficients_cancellable<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
    check: impl FnMut() -> Result<()>,
) -> Result<Vec<f64>> {
    validate_pair(outgoing, incoming)?;
    let mut result = vec![0.0; outgoing.node_count()];
    clustering_coefficients_into_cancellable(outgoing, incoming, 0, &mut result, check)?;
    Ok(result)
}

/// Writes a contiguous vertex range into caller-owned coefficient output.
/// Disjoint output slices share live adjacency and retain only two reusable neighbor rows.
pub fn clustering_coefficients_into_cancellable<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
    start: usize,
    output: &mut [f64],
    mut check: impl FnMut() -> Result<()>,
) -> Result<()> {
    validate_pair(outgoing, incoming)?;
    if start
        .checked_add(output.len())
        .is_none_or(|end| end > outgoing.node_count())
    {
        return Err(Error::internal("clustering vertex range is invalid"));
    }
    check()?;
    let mut list = Vec::new();
    let mut left_neighbors = Vec::new();
    for (offset, coefficient) in output.iter_mut().enumerate() {
        let row = start + offset;
        if row & 255 == 0 {
            check()?;
        }
        undirected_row_into(outgoing, incoming, row as u32, &mut list);
        if list.len() < 2 {
            *coefficient = 0.0;
            continue;
        }
        let mut links = 0_u64;
        for left in 0..list.len().saturating_sub(1) {
            if left != 0 && left & 255 == 0 {
                check()?;
            }
            undirected_row_into(outgoing, incoming, list[left].0, &mut left_neighbors);
            links = links.saturating_add(sorted_neighbor_intersection_count(
                &list[left + 1..],
                &left_neighbors,
                &mut check,
            )?);
        }
        let possible = list.len().saturating_mul(list.len().saturating_sub(1)) / 2;
        *coefficient = links as f64 / possible as f64;
    }
    check()?;
    Ok(())
}

/// Undirected core number from deterministic degree peeling.
pub fn k_core<A: AdjacencyRead>(outgoing: &A, incoming: &A) -> Result<Vec<u32>> {
    k_core_cancellable(outgoing, incoming, || Ok(()))
}

/// Undirected core number with cancellation checks.
pub fn k_core_cancellable<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Vec<u32>> {
    validate_pair(outgoing, incoming)?;
    check()?;
    let node_count = outgoing.node_count();
    let mut neighbors = Vec::new();
    let mut degree = Vec::with_capacity(node_count);
    for node in 0..node_count {
        if node & 1023 == 0 {
            check()?;
        }
        undirected_row_into(outgoing, incoming, node as u32, &mut neighbors);
        degree.push(neighbors.len());
    }
    let mut removed = vec![false; node_count];
    let mut core = vec![0_u32; node_count];
    let mut heap = degree
        .iter()
        .copied()
        .enumerate()
        .map(|(node, degree)| Reverse((degree, node)))
        .collect::<BinaryHeap<_>>();
    let mut current = 0_usize;
    while let Some(Reverse((candidate_degree, node))) = heap.pop() {
        if node & 1023 == 0 {
            check()?;
        }
        if removed[node] || degree[node] != candidate_degree {
            continue;
        }
        removed[node] = true;
        current = current.max(candidate_degree);
        core[node] = u32::try_from(current)
            .map_err(|_| Error::new(ErrorCode::ResultBudgetExceeded, "core number overflow"))?;
        undirected_row_into(outgoing, incoming, node as u32, &mut neighbors);
        for (neighbor, _) in &neighbors {
            let neighbor = *neighbor as usize;
            // Degrees at the current core level cannot change the eventual core number.
            // Keeping them at that level avoids redundant heap entries during peeling.
            if !removed[neighbor] && degree[neighbor] > current {
                degree[neighbor] -= 1;
                heap.push(Reverse((degree[neighbor], neighbor)));
            }
        }
    }
    check()?;
    Ok(core)
}

/// Deterministic parallel-reference multilevel Louvain assignment over the undirected graph.
///
/// Input edges are unweighted. Parallel directed edges collapse to one undirected edge so the
/// result depends on graph topology rather than relationship insertion order. A local-moving pass
/// reads one immutable membership snapshot and proposes every strictly beneficial move. It then
/// applies a deterministic maximal set whose moves touch disjoint source/target communities. Those
/// modularity deltas are additive, so each parallel batch is non-decreasing without depending on
/// scheduler order. Gain comparisons use exact integer cross-products because every initial and
/// coarsened weight is an integer edge multiplicity. Candidate ties choose the smallest community;
/// matching conflicts use a stable hash of the dense node ordinal followed by that ordinal.
pub fn louvain_communities<A: AdjacencyRead>(outgoing: &A, incoming: &A) -> Result<Components> {
    louvain_communities_cancellable(outgoing, incoming, || Ok(()))
}

/// Deterministic multilevel Louvain with cancellation checks.
pub fn louvain_communities_cancellable<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Components> {
    validate_pair(outgoing, incoming)?;
    let original_count = outgoing.node_count();
    if original_count == 0 {
        return Ok(Components {
            component: Vec::new(),
            count: 0,
        });
    }
    let mut graph = LouvainTopology::Canonical(InitialWeighted { outgoing, incoming });
    let mut original_to_current = (0..original_count).collect::<Vec<_>>();
    // Zero-degree nodes can never propose a strictly beneficial modularity move. Keeping them in
    // the local-moving set made sparse real graphs pay every O(V) pass and allocate candidate maps
    // for millions of immutable singleton communities. They remain in `original_to_current`, so
    // canonical output still publishes one distinct community for every isolated real node; only
    // work that provably cannot affect the result is skipped.
    let mut active = (0..original_count)
        .map(|node| !graph.weights(node).is_empty())
        .collect::<Vec<_>>();
    let mut active_count = active.iter().filter(|active| **active).count();

    loop {
        check()?;
        let membership = louvain_local_move(&graph, &active, &mut check)?;
        let mut next_active = vec![false; original_count];
        for node in 0..original_count {
            if active[node] {
                next_active[membership[node]] = true;
            }
        }
        let community_count = next_active.iter().filter(|active| **active).count();
        for current in &mut original_to_current {
            *current = membership[*current];
        }
        if community_count == active_count || community_count <= 1 {
            break;
        }
        graph = LouvainTopology::Coarse(coarsen_communities(
            &graph,
            &membership,
            &active,
            &mut check,
        )?);
        active = next_active;
        active_count = community_count;
    }

    let mut canonical = BTreeMap::new();
    let mut next = 0_u32;
    let mut component = Vec::with_capacity(original_count);
    for community in original_to_current {
        let id = if let Some(id) = canonical.get(&community) {
            *id
        } else {
            let id = next;
            next = next.checked_add(1).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "community ID space exhausted",
                )
            })?;
            canonical.insert(community, id);
            id
        };
        component.push(id);
    }
    Ok(Components {
        component,
        count: next,
    })
}

fn louvain_local_move<G: WeightedAdjacency + ?Sized>(
    graph: &G,
    active_nodes: &[bool],
    check: &mut impl FnMut() -> Result<()>,
) -> Result<Vec<usize>> {
    louvain_local_move_traced(graph, active_nodes, check, &mut |_, _| Ok(()))
}

#[allow(clippy::too_many_lines)]
fn louvain_local_move_traced<G: WeightedAdjacency + ?Sized>(
    graph: &G,
    active_nodes: &[bool],
    check: &mut impl FnMut() -> Result<()>,
    observe: &mut impl FnMut(&[usize], usize) -> Result<()>,
) -> Result<Vec<usize>> {
    let node_count = graph.node_count();
    if active_nodes.len() != node_count {
        return Err(Error::internal("Louvain active-node shape mismatch"));
    }
    let degree = (0..graph.node_count())
        .map(|node| {
            graph
                .weights(node)
                .iter()
                .map(|(_, weight)| *weight)
                .sum::<u64>()
        })
        .collect::<Vec<_>>();
    let total_weight = degree.iter().copied().sum::<u64>();
    if total_weight == 0 {
        return Ok((0..node_count).collect());
    }
    let mut membership = active_nodes
        .iter()
        .enumerate()
        // Inactive zero-degree nodes do no local-moving work, but retain their original singleton
        // identity so expansion and final canonicalisation cannot collapse unrelated isolates.
        .map(|(node, _active)| node)
        .collect::<Vec<_>>();
    observe(&membership, 0)?;
    let mut community_weight = degree.clone();
    // There is deliberately no heuristic pass limit. Nodes move in stable dense-ordinal order and
    // each accepted move updates both community weights immediately. That is the CPU delta path:
    // one node and its incident communities change, never a cloned proposal/matching table over
    // the full graph. Every move is an exact strict modularity improvement, so finite membership
    // state remains the termination proof while hub-heavy graphs can absorb many leaves per pass.
    loop {
        // Reuse one flat candidate buffer across nodes. A BTreeMap here previously performed at
        // least one heap allocation per active node per pass. Sorting the neighboring community
        // IDs and reducing equal runs preserves the BTreeMap's ascending candidate/tie order while
        // making allocation track maximum degree rather than node count.
        let mut links = Vec::<(usize, u64)>::new();
        let mut accepted_count = 0_usize;
        for node in 0..node_count {
            if !active_nodes[node] {
                continue;
            }
            if node & 255 == 0 {
                check()?;
            }
            let current = membership[node];
            links.clear();
            links.push((current, 0));
            for (neighbor, weight) in graph.weights(node).iter() {
                // A coarse self-loop represents internal edges already absorbed into this node.
                // It remains internal whichever community the node joins, so it contributes to
                // degree but cancels from every move's neighboring-community link term.
                if *neighbor == node {
                    continue;
                }
                links.push((membership[*neighbor], *weight));
            }
            links.sort_unstable_by_key(|(community, _)| *community);
            let mut best = current;
            let mut best_link = 0_u64;
            let mut best_community_weight = community_weight[current].saturating_sub(degree[node]);
            let mut current_link = 0_u64;
            let mut candidate_index = 0_usize;
            while candidate_index < links.len() {
                let candidate = links[candidate_index].0;
                let mut link_weight = 0_u64;
                while candidate_index < links.len() && links[candidate_index].0 == candidate {
                    link_weight = link_weight
                        .checked_add(links[candidate_index].1)
                        .ok_or_else(|| {
                            Error::new(
                                ErrorCode::ResultBudgetExceeded,
                                "Louvain neighboring-community weight overflow",
                            )
                        })?;
                    candidate_index += 1;
                }
                if candidate == current {
                    current_link = link_weight;
                }
                let candidate_weight = if candidate == current {
                    community_weight[candidate].saturating_sub(degree[node])
                } else {
                    community_weight[candidate]
                };
                // Compare `link*T - degree*community` without signed subtraction. All operands
                // originate in a u32-addressed CSR, but u128 keeps the proof valid after every
                // coarsening level and exactly matches the Metal two-word arithmetic.
                let left = u128::from(link_weight) * u128::from(total_weight)
                    + u128::from(degree[node]) * u128::from(best_community_weight);
                let right = u128::from(best_link) * u128::from(total_weight)
                    + u128::from(degree[node]) * u128::from(candidate_weight);
                if left > right || (left == right && candidate < best) {
                    best = candidate;
                    best_link = link_weight;
                    best_community_weight = candidate_weight;
                }
            }
            let improvement_left = u128::from(best_link) * u128::from(total_weight)
                + u128::from(degree[node])
                    * u128::from(community_weight[current].saturating_sub(degree[node]));
            let improvement_right = u128::from(current_link) * u128::from(total_weight)
                + u128::from(degree[node]) * u128::from(best_community_weight);
            if best != current && improvement_left > improvement_right {
                community_weight[current] = community_weight[current]
                    .checked_sub(degree[node])
                    .ok_or_else(|| Error::internal("Louvain source community weight underflow"))?;
                community_weight[best] = community_weight[best]
                    .checked_add(degree[node])
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            "Louvain target community weight overflow",
                        )
                    })?;
                membership[node] = best;
                accepted_count = accepted_count.checked_add(1).ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "Louvain accepted-move count overflow",
                    )
                })?;
            }
        }
        if accepted_count == 0 {
            break;
        }
        observe(&membership, accepted_count)?;
    }
    Ok(membership)
}

fn coarsen_communities<G: WeightedAdjacency + ?Sized>(
    graph: &G,
    membership: &[usize],
    active_nodes: &[bool],
    check: &mut impl FnMut() -> Result<()>,
) -> Result<Vec<Vec<(usize, u64)>>> {
    let mut coarse = vec![BTreeMap::<usize, u64>::new(); graph.node_count()];
    for source in 0..graph.node_count() {
        let row = graph.weights(source);
        if !active_nodes[source] {
            continue;
        }
        if source & 255 == 0 {
            check()?;
        }
        let coarse_source = membership[source];
        for (target, weight) in row.iter() {
            let coarse_target = membership[*target];
            let coarse_weight = coarse[coarse_source].entry(coarse_target).or_insert(0);
            *coarse_weight = (*coarse_weight).checked_add(*weight).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Louvain coarse edge weight overflow",
                )
            })?;
        }
    }
    Ok(coarse
        .into_iter()
        .map(BTreeMap::into_iter)
        .map(Iterator::collect)
        .collect())
}

fn validate_pair<A: AdjacencyRead>(outgoing: &A, incoming: &A) -> Result<()> {
    if outgoing.node_count() != incoming.node_count() {
        return Err(Error::invalid_data("CSR and CSC node counts differ"));
    }
    Ok(())
}

fn undirected_row<A: AdjacencyRead>(outgoing: &A, incoming: &A, node: u32) -> Vec<u32> {
    let mut result = Vec::new();
    for adjacency in [outgoing, incoming] {
        if let Some(row) = adjacency.row(node) {
            result.extend(row.filter_map(|(neighbor, _)| (neighbor != node).then_some(neighbor)));
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}

fn undirected_row_into<A: AdjacencyRead>(
    outgoing: &A,
    incoming: &A,
    node: u32,
    output: &mut Vec<(u32, u32)>,
) {
    output.clear();
    outgoing.append_row(node, output);
    incoming.append_row(node, output);
    output.retain(|(neighbor, _)| *neighbor != node);
    output.sort_unstable();
    output.dedup_by_key(|(neighbor, _)| *neighbor);
}

fn sorted_neighbor_intersection_count(
    left: &[(u32, u32)],
    right: &[(u32, u32)],
    check: &mut impl FnMut() -> Result<()>,
) -> Result<u64> {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut steps = 0_usize;
    let mut count = 0_u64;
    while left_index < left.len() && right_index < right.len() {
        if steps != 0 && steps & 4095 == 0 {
            check()?;
        }
        steps += 1;
        match left[left_index].0.cmp(&right[right_index].0) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    Ok(count)
}

trait WeightedAdjacency {
    fn node_count(&self) -> usize;
    fn weights(&self, node: usize) -> Cow<'_, [(usize, u64)]>;
}
impl WeightedAdjacency for [Vec<(usize, u64)>] {
    fn node_count(&self) -> usize {
        self.len()
    }
    fn weights(&self, node: usize) -> Cow<'_, [(usize, u64)]> {
        Cow::Borrowed(&self[node])
    }
}
impl WeightedAdjacency for Vec<Vec<(usize, u64)>> {
    fn node_count(&self) -> usize {
        self.len()
    }
    fn weights(&self, node: usize) -> Cow<'_, [(usize, u64)]> {
        self.as_slice().weights(node)
    }
}
struct InitialWeighted<'a, A> {
    outgoing: &'a A,
    incoming: &'a A,
}
impl<A: AdjacencyRead> WeightedAdjacency for InitialWeighted<'_, A> {
    fn node_count(&self) -> usize {
        self.outgoing.node_count()
    }
    fn weights(&self, node: usize) -> Cow<'_, [(usize, u64)]> {
        Cow::Owned(
            undirected_row(self.outgoing, self.incoming, node as u32)
                .into_iter()
                .map(|neighbor| (neighbor as usize, 1))
                .collect(),
        )
    }
}
enum LouvainTopology<'a, A> {
    Canonical(InitialWeighted<'a, A>),
    Coarse(Vec<Vec<(usize, u64)>>),
}
impl<A: AdjacencyRead> WeightedAdjacency for LouvainTopology<'_, A> {
    fn node_count(&self) -> usize {
        match self {
            Self::Canonical(graph) => graph.node_count(),
            Self::Coarse(graph) => graph.node_count(),
        }
    }
    fn weights(&self, node: usize) -> Cow<'_, [(usize, u64)]> {
        match self {
            Self::Canonical(graph) => graph.weights(node),
            Self::Coarse(graph) => graph.weights(node),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(node_count: usize, edges: &[(u32, u32, u32)]) -> Result<(Csr, Csr)> {
        Ok((
            Csr::build(node_count, edges)?,
            Csr::build(
                node_count,
                &edges
                    .iter()
                    .map(|(source, target, edge)| (*target, *source, *edge))
                    .collect::<Vec<_>>(),
            )?,
        ))
    }

    fn exact_modularity_numerator(
        graph: &[Vec<(usize, u64)>],
        active: &[bool],
        membership: &[usize],
    ) -> i128 {
        let degree = graph
            .iter()
            .map(|row| row.iter().map(|(_, weight)| *weight).sum::<u64>())
            .collect::<Vec<_>>();
        let total_weight = degree.iter().copied().sum::<u64>();
        let mut internal_weight = 0_u64;
        let mut community_weight = BTreeMap::<usize, u64>::new();
        for node in 0..graph.len() {
            if !active[node] {
                continue;
            }
            *community_weight.entry(membership[node]).or_default() += degree[node];
            for (neighbor, weight) in &graph[node] {
                if active[*neighbor] && membership[*neighbor] == membership[node] {
                    internal_weight += *weight;
                }
            }
        }
        i128::from(internal_weight) * i128::from(total_weight)
            - community_weight
                .values()
                .map(|weight| i128::from(*weight).pow(2))
                .sum::<i128>()
    }

    #[test]
    fn all_reference_algorithms_have_deterministic_semantics() -> Result<()> {
        let edges = [
            (0, 1, 0),
            (1, 2, 1),
            (2, 0, 2),
            (2, 3, 3),
            (3, 4, 4),
            (4, 3, 5),
        ];
        let (outgoing, incoming) = pair(5, &edges)?;
        assert_eq!(
            bfs(&outgoing, 0)?,
            vec![Some(0), Some(1), Some(2), Some(3), Some(4)]
        );
        assert_eq!(dfs(&outgoing, 0)?, vec![0, 1, 2, 3, 4]);
        assert_eq!(shortest_path(&outgoing, 0, 4)?, Some(vec![0, 1, 2, 3, 4]));
        let distances = dijkstra(&outgoing, 0, |_| Ok(1.0))?;
        assert_eq!(distances.distance[4], Some(4.0));
        assert_eq!(weakly_connected_components(&outgoing, &incoming)?.count, 1);
        let strong = strongly_connected_components(&outgoing, &incoming)?;
        assert_eq!(strong.count, 2);
        assert_eq!(triangle_count(&outgoing, &incoming)?, 1);
        let coefficient = clustering_coefficients(&outgoing, &incoming)?;
        assert_eq!(coefficient[0], 1.0);
        assert_eq!(k_core(&outgoing, &incoming)?, vec![2, 2, 2, 1, 1]);
        Ok(())
    }

    struct AppendAdjacency {
        rows: Vec<Vec<(u32, u32)>>,
        row_reads: std::sync::atomic::AtomicUsize,
    }

    impl AdjacencyRead for AppendAdjacency {
        type Row<'a> = std::vec::IntoIter<(u32, u32)>;

        fn node_count(&self) -> usize {
            self.rows.len()
        }

        fn degree(&self, node: u32) -> usize {
            self.rows.get(node as usize).map_or(0, Vec::len)
        }

        fn row(&self, node: u32) -> Option<Self::Row<'_>> {
            self.row_reads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.rows.get(node as usize).cloned().map(Vec::into_iter)
        }

        fn append_row(&self, node: u32, output: &mut Vec<(u32, u32)>) -> bool {
            let Some(row) = self.rows.get(node as usize) else {
                return false;
            };
            output.extend(row.iter().copied());
            true
        }
    }

    #[test]
    fn live_unsorted_rows_preserve_traversal_results_without_owned_row_copies() -> Result<()> {
        let edges = [
            (0, 3, 0),
            (0, 1, 1),
            (0, 2, 2),
            (1, 2, 3),
            (2, 0, 4),
            (3, 4, 5),
            (4, 3, 6),
            (4, 4, 7),
            (6, 5, 8),
            (5, 6, 9),
        ];
        let (outgoing, incoming) = pair(8, &edges)?;
        let live = |reverse: bool| {
            let mut rows = vec![Vec::new(); 8];
            for &(source, target, edge) in &edges {
                let (source, target) = if reverse {
                    (target, source)
                } else {
                    (source, target)
                };
                rows[source as usize].push((target, edge));
            }
            AppendAdjacency {
                rows,
                row_reads: std::sync::atomic::AtomicUsize::new(0),
            }
        };
        let live_out = live(false);
        let live_in = live(true);
        assert_eq!(bfs(&live_out, 0)?, bfs(&outgoing, 0)?);
        assert_eq!(dfs(&live_out, 0)?, dfs(&outgoing, 0)?);
        assert_eq!(
            dijkstra(&live_out, 0, |_| Ok(1.0))?,
            dijkstra(&outgoing, 0, |_| Ok(1.0))?
        );
        assert_eq!(
            weakly_connected_components(&live_out, &live_in)?,
            weakly_connected_components(&outgoing, &incoming)?
        );
        assert_eq!(
            strongly_connected_components(&live_out, &live_in)?,
            strongly_connected_components(&outgoing, &incoming)?
        );
        assert_eq!(
            page_rank(&live_out, PageRankConfig::default())?,
            page_rank(&outgoing, PageRankConfig::default())?
        );
        for adjacency in [&live_out, &live_in] {
            assert_eq!(
                adjacency
                    .row_reads
                    .load(std::sync::atomic::Ordering::Relaxed),
                0
            );
        }
        Ok(())
    }

    #[test]
    fn triangle_ranges_and_large_intersections_remain_cancellable() -> Result<()> {
        let rows = (0..10000_u32).map(|node| (node, node)).collect::<Vec<_>>();
        let mut checks = 0;
        assert_eq!(
            sorted_neighbor_intersection_count(&rows[..8], &rows[..8], &mut || {
                checks += 1;
                Ok(())
            })?,
            8
        );
        assert_eq!(
            checks, 0,
            "short intersections use the outer loop checkpoint"
        );
        let cancelled = sorted_neighbor_intersection_count(&rows, &rows, &mut || {
            checks += 1;
            Err(Error::new(ErrorCode::Cancelled, "cancelled intersection"))
        });
        assert!(matches!(cancelled, Err(error) if error.code == ErrorCode::Cancelled));
        assert_eq!(checks, 1);
        let adjacency = AppendAdjacency {
            rows: vec![Vec::new(); 32],
            row_reads: std::sync::atomic::AtomicUsize::new(0),
        };
        let cancelled = triangle_count_range_cancellable(&adjacency, &adjacency, 17..18, || {
            Err(Error::new(ErrorCode::Cancelled, "cancelled range"))
        });
        assert!(matches!(cancelled, Err(error) if error.code == ErrorCode::Cancelled));
        assert_eq!(
            adjacency
                .row_reads
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let mut coefficient = [f64::NAN];
        let cancelled = clustering_coefficients_into_cancellable(
            &adjacency,
            &adjacency,
            17,
            &mut coefficient,
            || {
                Err(Error::new(
                    ErrorCode::Cancelled,
                    "cancelled clustering range",
                ))
            },
        );
        assert!(matches!(cancelled, Err(error) if error.code == ErrorCode::Cancelled));
        assert!(coefficient[0].is_nan());
        assert!(
            clustering_coefficients_into_cancellable(
                &adjacency,
                &adjacency,
                usize::MAX,
                &mut coefficient,
                || Ok(()),
            )
            .is_err()
        );
        assert_eq!(
            adjacency
                .row_reads
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        Ok(())
    }

    #[test]
    fn reusable_triangle_and_clustering_rows_match_independent_simple_graph_counts() -> Result<()> {
        for node_count in 0_u32..=12 {
            let mut outgoing = AppendAdjacency {
                rows: vec![Vec::new(); node_count as usize],
                row_reads: std::sync::atomic::AtomicUsize::new(0),
            };
            let mut incoming = AppendAdjacency {
                rows: vec![Vec::new(); node_count as usize],
                row_reads: std::sync::atomic::AtomicUsize::new(0),
            };
            let mut connected = vec![vec![false; node_count as usize]; node_count as usize];
            let mut edge = 0;
            for source in (0..node_count).rev() {
                for target in (0..node_count).rev() {
                    if source == target || (source * 7 + target * 3) % 5 < 2 {
                        // Unsorted reciprocal/parallel edges and self-loops must reduce to
                        // the same undirected simple graph as the independent matrix oracle.
                        for _ in 0..2 {
                            outgoing.rows[source as usize].push((target, edge));
                            incoming.rows[target as usize].push((source, edge));
                            edge += 1;
                        }
                        if source != target {
                            connected[source as usize][target as usize] = true;
                            connected[target as usize][source as usize] = true;
                        }
                    }
                }
            }
            let mut expected_triangles = 0;
            for first in 0..node_count as usize {
                for second in first + 1..node_count as usize {
                    for third in second + 1..node_count as usize {
                        expected_triangles += u64::from(
                            connected[first][second]
                                && connected[second][third]
                                && connected[first][third],
                        );
                    }
                }
            }
            assert_eq!(triangle_count(&outgoing, &incoming)?, expected_triangles);
            let mut expected_cores = vec![0; node_count as usize];
            for threshold in 1..node_count {
                let mut active = vec![true; node_count as usize];
                loop {
                    let remove = (0..node_count as usize)
                        .filter(|&node| {
                            active[node]
                                && connected[node]
                                    .iter()
                                    .zip(&active)
                                    .filter(|(edge, present)| **edge && **present)
                                    .count()
                                    < threshold as usize
                        })
                        .collect::<Vec<_>>();
                    if remove.is_empty() {
                        break;
                    }
                    for node in remove {
                        active[node] = false;
                    }
                }
                for (node, present) in active.into_iter().enumerate() {
                    if present {
                        expected_cores[node] = threshold;
                    }
                }
            }
            assert_eq!(k_core(&outgoing, &incoming)?, expected_cores);
            for source in 0..node_count {
                for target in 0..node_count {
                    let mut paths = vec![None; node_count as usize];
                    paths[source as usize] = Some(vec![source]);
                    let mut queue = VecDeque::from([source]);
                    while let Some(node) = queue.pop_front() {
                        for neighbor in 0..node_count {
                            if outgoing.rows[node as usize]
                                .iter()
                                .any(|&(next, _)| next == neighbor)
                                && paths[neighbor as usize].is_none()
                            {
                                let mut path = paths[node as usize].as_ref().unwrap().clone();
                                path.push(neighbor);
                                paths[neighbor as usize] = Some(path);
                                queue.push_back(neighbor);
                            }
                        }
                    }
                    assert_eq!(
                        shortest_path(&outgoing, source, target)?,
                        paths[target as usize]
                    );
                }
            }
            for width in 1..=4 {
                let partitioned = (0..node_count as usize)
                    .step_by(width)
                    .map(|start| {
                        triangle_count_range_cancellable(
                            &outgoing,
                            &incoming,
                            start..(start + width).min(node_count as usize),
                            || Ok(()),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .sum::<u64>();
                assert_eq!(partitioned, expected_triangles);
            }
            let coefficients = clustering_coefficients(&outgoing, &incoming)?;
            for width in 1..=4 {
                let mut partitioned = vec![f64::NAN; node_count as usize];
                for (task, output) in partitioned.chunks_mut(width).enumerate() {
                    clustering_coefficients_into_cancellable(
                        &outgoing,
                        &incoming,
                        task * width,
                        output,
                        || Ok(()),
                    )?;
                }
                assert_eq!(partitioned, coefficients);
            }
            for (node, coefficient) in coefficients.iter().enumerate() {
                let neighbors = (0..node_count as usize)
                    .filter(|neighbor| connected[node][*neighbor])
                    .collect::<Vec<_>>();
                let mut links = 0;
                for left in 0..neighbors.len() {
                    for right in left + 1..neighbors.len() {
                        links += usize::from(connected[neighbors[left]][neighbors[right]]);
                    }
                }
                let expected = if neighbors.len() < 2 {
                    0.0
                } else {
                    let possible = neighbors.len() * (neighbors.len() - 1) / 2;
                    links as f64 / possible as f64
                };
                assert_eq!(*coefficient, expected);
            }
            assert_eq!(
                outgoing
                    .row_reads
                    .load(std::sync::atomic::Ordering::Relaxed),
                0,
                "optimized algorithms must use the append API rather than allocate row copies"
            );
            assert_eq!(
                incoming
                    .row_reads
                    .load(std::sync::atomic::Ordering::Relaxed),
                0
            );
        }
        Ok(())
    }

    #[test]
    fn sorted_neighbor_intersections_cancel_inside_large_rows() {
        let neighbors = (0..20_000)
            .map(|neighbor| (neighbor, 0))
            .collect::<Vec<_>>();
        let mut checks = 0;
        let error = sorted_neighbor_intersection_count(&neighbors, &neighbors, &mut || {
            checks += 1;
            if checks == 2 {
                Err(Error::new(ErrorCode::Cancelled, "intersection cancelled"))
            } else {
                Ok(())
            }
        })
        .expect_err("intersection did not observe cancellation inside a large row");
        assert_eq!(error.code, ErrorCode::Cancelled);
        assert_eq!(checks, 2);
    }

    #[test]
    fn dijkstra_keeps_the_source_parentless_across_zero_weight_cycles_and_ties() -> Result<()> {
        let outgoing = Csr::build(
            4,
            &[
                (0, 1, 0),
                (1, 0, 1),
                (0, 2, 2),
                (1, 2, 3),
                (0, 2, 4),
                (2, 2, 5),
            ],
        )?;
        let result = dijkstra(&outgoing, 0, |_| Ok(0.0))?;
        assert_eq!(result.distance, vec![Some(0.0), Some(0.0), Some(0.0), None]);
        assert_eq!(result.predecessor, vec![None, Some(0), Some(0), None]);
        Ok(())
    }

    #[test]
    fn multilevel_louvain_separates_dense_communities() -> Result<()> {
        let mut edges = Vec::new();
        let mut edge = 0_u32;
        for community in [0_u32..4, 4_u32..8] {
            for source in community.clone() {
                for target in community.clone() {
                    if source < target {
                        edges.push((source, target, edge));
                        edge += 1;
                        edges.push((target, source, edge));
                        edge += 1;
                    }
                }
            }
        }
        edges.push((3, 4, edge));
        let (outgoing, incoming) = pair(8, &edges)?;
        let communities = louvain_communities(&outgoing, &incoming)?;
        assert_eq!(communities.count, 2);
        assert!(
            communities.component[..4]
                .iter()
                .all(|community| *community == communities.component[0])
        );
        assert!(
            communities.component[4..]
                .iter()
                .all(|community| *community == communities.component[4])
        );
        assert_ne!(communities.component[0], communities.component[4]);
        Ok(())
    }

    #[test]
    fn parallel_louvain_is_not_trapped_by_monotone_community_ids() -> Result<()> {
        // A lower-ID-only BSP implementation collapses this graph into one zero-modularity
        // community. Full proposal matching permits node 0's beneficial move to the higher-ID
        // community and deterministically finds the two positive-modularity pairs.
        let edges = [(0, 1, 0), (0, 2, 1), (1, 3, 2)];
        let (outgoing, incoming) = pair(4, &edges)?;
        let expected = Components {
            component: vec![0, 1, 0, 1],
            count: 2,
        };
        assert_eq!(louvain_communities(&outgoing, &incoming)?, expected);

        let reversed = edges
            .iter()
            .rev()
            .map(|(source, target, edge)| (*target, *source, *edge))
            .collect::<Vec<_>>();
        let (outgoing, incoming) = pair(4, &reversed)?;
        assert_eq!(louvain_communities(&outgoing, &incoming)?, expected);
        Ok(())
    }

    #[test]
    fn every_louvain_local_pass_strictly_increases_exact_modularity() -> Result<()> {
        // Exercise many weighted/coarsened-style symmetric graphs, including self-loops. The
        // observer is called only after a complete community-disjoint matching batch has been
        // applied. This is the executable progress invariant behind cap-free convergence.
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        let mut observed_improving_passes = 0_usize;
        for case in 0..96 {
            let node_count = 8 + case % 17;
            let mut graph = vec![BTreeMap::<usize, u64>::new(); node_count];
            for left in 0..node_count {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if state.is_multiple_of(5) {
                    // Coarse self-loop weights contain both directed halves.
                    graph[left].insert(left, 2 * (1 + (state >> 32) % 4));
                }
                for right in (left + 1)..node_count {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    if state % 7 < 3 {
                        let weight = 1 + (state >> 32) % 5;
                        graph[left].insert(right, weight);
                        graph[right].insert(left, weight);
                    }
                }
            }
            let graph = graph
                .into_iter()
                .map(BTreeMap::into_iter)
                .map(Iterator::collect)
                .collect::<Vec<Vec<_>>>();
            let active = vec![true; node_count];
            let mut prior = None;
            let mut trace = Vec::new();
            let first = louvain_local_move_traced(
                &graph,
                &active,
                &mut || Ok(()),
                &mut |membership, accepted| {
                    let score = exact_modularity_numerator(&graph, &active, membership);
                    if let Some(previous) = prior {
                        assert!(accepted > 0, "continuing pass accepted no move");
                        assert!(
                            score > previous,
                            "case {case}: modularity did not strictly rise: {previous} -> {score}"
                        );
                        observed_improving_passes += 1;
                    } else {
                        assert_eq!(accepted, 0, "initial observation accepted moves");
                    }
                    prior = Some(score);
                    trace.push(score);
                    Ok(())
                },
            )?;
            let second = louvain_local_move(&graph, &active, &mut || Ok(()))?;
            assert_eq!(
                first, second,
                "case {case}: local moving is not deterministic"
            );
            assert!(trace.windows(2).all(|scores| scores[1] > scores[0]));
        }
        assert!(
            observed_improving_passes > 100,
            "randomized invariant suite did not exercise enough improving passes"
        );
        Ok(())
    }

    #[test]
    fn louvain_local_moving_absorbs_a_hub_in_bounded_passes() -> Result<()> {
        // Hub-heavy real data used to accept exactly one leaf per full-graph pass because every
        // proposal conflicted at the hub. In-place community-weight deltas must let the same exact
        // strictly-improving moves complete together instead of restoring that O(V * leaves) path.
        let leaf_count = 128_usize;
        let mut graph = vec![Vec::<(usize, u64)>::new(); leaf_count + 1];
        for leaf in 1..=leaf_count {
            graph[0].push((leaf, 1));
            graph[leaf].push((0, 1));
        }
        let active = vec![true; graph.len()];
        let mut improving_passes = 0_usize;
        let membership =
            louvain_local_move_traced(&graph, &active, &mut || Ok(()), &mut |_, accepted| {
                if accepted != 0 {
                    improving_passes += 1;
                }
                Ok(())
            })?;
        assert!(
            improving_passes <= 2,
            "star required {improving_passes} full-graph passes"
        );
        assert!(
            membership
                .iter()
                .all(|community| *community == membership[0])
        );
        Ok(())
    }

    #[test]
    fn louvain_skips_isolated_local_work_without_merging_singletons() -> Result<()> {
        let node_count = 100_002_usize;
        let (outgoing, incoming) = pair(node_count, &[(100_000, 100_001, 0)])?;
        let result = louvain_communities(&outgoing, &incoming)?;
        assert_eq!(result.component.len(), node_count);
        assert_eq!(result.count, 100_001);
        assert_eq!(result.component[0], 0);
        assert_eq!(result.component[99_999], 99_999);
        assert_eq!(result.component[100_000], result.component[100_001]);
        Ok(())
    }

    #[test]
    fn strongly_connected_component_ids_follow_minimum_node_order() -> Result<()> {
        // Kosaraju visits these isolated SCCs in reverse finish order.  Public component IDs are
        // nevertheless canonical by minimum dense node so parallel device schedules agree.
        let (outgoing, incoming) = pair(4, &[])?;
        assert_eq!(
            strongly_connected_components(&outgoing, &incoming)?,
            Components {
                component: vec![0, 1, 2, 3],
                count: 4,
            }
        );

        let (outgoing, incoming) = pair(
            6,
            &[
                (0, 2, 0),
                (2, 0, 1),
                (1, 4, 2),
                (4, 1, 3),
                (2, 3, 4),
                (4, 5, 5),
            ],
        )?;
        assert_eq!(
            strongly_connected_components(&outgoing, &incoming)?.component,
            vec![0, 1, 0, 2, 1, 3]
        );
        Ok(())
    }

    #[test]
    fn cancellable_algorithm_propagates_failure() -> Result<()> {
        let edges = (0_u32..4_096)
            .map(|source| (source, source + 1, source))
            .collect::<Vec<_>>();
        let (outgoing, _) = pair(4_097, &edges)?;
        let mut checks = 0_usize;
        let error = bfs_cancellable(&outgoing, 0, || {
            checks += 1;
            if checks > 1 {
                Err(Error::new(ErrorCode::DeadlineExceeded, "cancelled"))
            } else {
                Ok(())
            }
        })
        .err()
        .ok_or_else(|| Error::internal("cancellable BFS did not stop"))?;
        assert_eq!(error.code, ErrorCode::DeadlineExceeded);
        Ok(())
    }
}
