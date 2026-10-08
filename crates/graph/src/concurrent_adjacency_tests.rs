use super::*;
use crate::{AdjacencyRead, clustering_coefficients_cancellable, triangle_count_cancellable};
use std::{sync::mpsc, thread, time::Duration};

struct Fixture {
    graph: GraphStore,
    kind: RelationshipTypeId,
    body: Arc<str>,
    edges: Vec<(u64, u64, u64, Layer, RelationshipTypeId)>,
}

fn fixture() -> Result<Fixture> {
    let graph = GraphStore::default();
    let property = graph.catalog().intern_property("body")?;
    let kind = graph.catalog().intern_relationship_type("LINK")?;
    let other_kind = graph.catalog().intern_relationship_type("OTHER")?;
    let body: Arc<str> = "complete owner body ".repeat(65536).into();
    for id in [10, 20, 30, 40, 50, 60] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: node_layer(id),
            revision: 1,
            labels: Vec::new(),
            properties: vec![(property, ScalarValue::String(Arc::clone(&body)))],
        })?;
    }
    let edges = vec![
        (100, 10, 20, Layer::Observed, kind),
        (101, 10, 20, Layer::Observed, kind),
        (102, 20, 10, Layer::Observed, kind),
        (103, 10, 10, Layer::Observed, kind),
        (104, 10, 30, Layer::Knowledge, kind),
        (105, 10, 50, Layer::Knowledge, kind),
        (106, 10, 60, Layer::Workspace, kind),
        (107, 10, 40, Layer::Observed, other_kind),
        (108, 20, 30, Layer::Observed, kind),
    ];
    for &(id, source, target, layer, relationship_type) in &edges {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer,
            revision: 1,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        kind,
        body,
        edges,
    })
}

fn node_layer(id: u64) -> Layer {
    match id {
        50 => Layer::Knowledge,
        60 => Layer::Workspace,
        _ => Layer::Observed,
    }
}

fn current_pairs(
    graph: &GraphStore,
    owner: NodeId,
    outgoing: bool,
    pairs: &[(u32, u32)],
) -> Result<Vec<(u64, u64)>> {
    let mut result = Vec::new();
    for &(neighbor, edge) in pairs {
        let other = graph
            .node_dense(neighbor)
            .ok_or_else(|| missing("neighbor"))?;
        let edge = graph.edge_dense(edge).ok_or_else(|| missing("edge"))?;
        assert_eq!(
            if outgoing {
                edge.source()
            } else {
                edge.target()
            },
            owner
        );
        assert_eq!(
            if outgoing {
                edge.target()
            } else {
                edge.source()
            },
            other.id()
        );
        result.push((edge.id().0, other.id().0));
    }
    result.sort_unstable();
    Ok(result)
}

#[test]
fn dense_adjacency_parallel_reverse_self_loop_and_layer_reference() -> Result<()> {
    let Fixture {
        graph,
        body,
        edges,
        kind,
    } = fixture()?;
    let body_references = Arc::strong_count(&body);
    for id in [10, 20, 30, 40, 50, 60] {
        let node = graph.node(NodeId(id)).ok_or_else(|| missing("owner"))?;
        for mask in [LayerMask::OBSERVED, LayerMask::AUTHORITY, LayerMask::ALL] {
            for outgoing in [true, false] {
                let mut pairs = Vec::new();
                assert!(
                    graph.visit_neighbor_denses(node, outgoing, mask, |neighbor, edge| {
                        assert_eq!(
                            Arc::strong_count(&body),
                            body_references,
                            "walker materialized an owner body"
                        );
                        pairs.push((neighbor, edge));
                    })
                );
                let mut expected = edges
                    .iter()
                    .filter_map(|&(edge, source, target, layer, _)| {
                        let (owner, other) = if outgoing {
                            (source, target)
                        } else {
                            (target, source)
                        };
                        (owner == id
                            && mask.contains_layer(node_layer(owner))
                            && mask.contains_layer(layer)
                            && mask.contains_layer(node_layer(other)))
                        .then_some((edge, other))
                    })
                    .collect::<Vec<_>>();
                expected.sort_unstable();
                assert_eq!(
                    current_pairs(&graph, NodeId(id), outgoing, &pairs)?,
                    expected
                );
                let expanded = if outgoing {
                    graph.expand_out(NodeId(id), None, mask)?
                } else {
                    graph.expand_in(NodeId(id), None, mask)?
                };
                let mut expanded = expanded
                    .into_iter()
                    .map(|(edge, other)| (edge.id().0, other.id().0))
                    .collect::<Vec<_>>();
                expanded.sort_unstable();
                assert_eq!(expanded, expected);
                let typed = if outgoing {
                    graph.expand_out(NodeId(id), Some(kind), mask)?
                } else {
                    graph.expand_in(NodeId(id), Some(kind), mask)?
                };
                assert!(
                    typed
                        .iter()
                        .all(|(edge, _)| edge.relationship_type() == kind)
                );
                assert_eq!(
                    typed.len(),
                    expected
                        .iter()
                        .filter(|(edge, _)| edges
                            .iter()
                            .any(|input| input.0 == *edge && input.4 == kind))
                        .count()
                );
            }
        }
    }
    println!("CANONICAL_DENSE_ADJACENCY_PASS");
    Ok(())
}

#[test]
fn dense_adjacency_roundtrip_rebuilds_heads_from_sparse_stable_ids() -> Result<()> {
    let Fixture { graph, .. } = fixture()?;
    let mut encoded = Vec::new();
    ciborium::ser::into_writer(&graph, &mut encoded)
        .map_err(|error| Error::internal(error.to_string()))?;
    let restored: GraphStore = ciborium::de::from_reader(encoded.as_slice())
        .map_err(|error| Error::internal(error.to_string()))?;
    restored.validate_structure()?;
    for id in [10, 20, 30, 40, 50, 60] {
        for outgoing in [true, false] {
            for mask in [LayerMask::OBSERVED, LayerMask::AUTHORITY, LayerMask::ALL] {
                let owner = restored
                    .node(NodeId(id))
                    .ok_or_else(|| missing("restored owner"))?;
                let mut pairs = Vec::new();
                restored.visit_neighbor_denses(owner, outgoing, mask, |neighbor, edge| {
                    pairs.push((neighbor, edge))
                });
                let before = if outgoing {
                    graph.expand_out(NodeId(id), None, mask)?
                } else {
                    graph.expand_in(NodeId(id), None, mask)?
                };
                let mut before = before
                    .into_iter()
                    .map(|(edge, other)| (edge.id().0, other.id().0))
                    .collect::<Vec<_>>();
                before.sort_unstable();
                assert_eq!(
                    current_pairs(&restored, NodeId(id), outgoing, &pairs)?,
                    before
                );
            }
        }
    }
    println!("CANONICAL_DENSE_ADJACENCY_PASS");
    Ok(())
}

#[test]
fn dense_adjacency_paused_captured_head_reuse_rejects_other_owner() -> Result<()> {
    for same_owner in [false, true] {
        let Fixture { graph, kind, .. } = fixture()?;
        let owner = graph.node(NodeId(10)).ok_or_else(|| missing("owner"))?;
        let head = load(&graph.0.outgoing, owner.dense());
        let old_edge = graph.edge(EdgeId(107)).ok_or_else(|| missing("old edge"))?;
        assert_eq!(head, u64::from(old_edge.dense()));
        thread::scope(|scope| -> Result<()> {
            let (captured_send, captured_receive) = mpsc::channel();
            let (resume_send, resume_receive) = mpsc::channel();
            let reader_graph = &graph;
            let reader = scope.spawn(move || -> Result<Vec<(u32, u32)>> {
                captured_send
                    .send(())
                    .map_err(|error| Error::internal(error.to_string()))?;
                resume_receive
                    .recv_timeout(Duration::from_secs(2))
                    .map_err(|error| Error::internal(error.to_string()))?;
                let mut result = Vec::new();
                reader_graph.walk_neighbor_denses_from(
                    owner,
                    true,
                    LayerMask::ALL,
                    head,
                    |neighbor, edge| result.push((neighbor, edge)),
                );
                Ok(result)
            });
            captured_receive
                .recv_timeout(Duration::from_secs(2))
                .map_err(|error| Error::internal(error.to_string()))?;
            graph.delete_edge(EdgeId(107), 2)?;
            let row = graph.insert_edge(EdgeInput {
                id: EdgeId(999),
                source: NodeId(if same_owner { 10 } else { 20 }),
                target: NodeId(30),
                relationship_type: kind,
                layer: Layer::Observed,
                revision: 3,
                properties: Vec::new(),
            })?;
            assert_eq!(row, old_edge.dense());
            resume_send
                .send(())
                .map_err(|error| Error::internal(error.to_string()))?;
            let pairs = reader
                .join()
                .map_err(|_| Error::internal("reader panicked"))??;
            assert!(!old_edge.current());
            if same_owner {
                let current = current_pairs(&graph, NodeId(10), true, &pairs)?;
                assert!(current.iter().any(|(edge, _)| *edge == 999));
                assert!(current.iter().all(|(edge, _)| *edge != 107));
            } else {
                assert!(
                    pairs.is_empty(),
                    "old head traversed another owner's recycled row"
                );
            }
            Ok(())
        })?;
    }
    println!("CANONICAL_DENSE_ADJACENCY_PASS");
    Ok(())
}

#[test]
fn dense_adjacency_reused_endpoints_keep_current_identity_and_layer() -> Result<()> {
    let Fixture { graph, kind, .. } = fixture()?;
    let old_node = graph.node(NodeId(20)).ok_or_else(|| missing("old node"))?;
    let source = graph.node(NodeId(10)).ok_or_else(|| missing("source"))?;
    let captured = graph.insert_edge(EdgeInput {
        id: EdgeId(800),
        source: NodeId(10),
        target: NodeId(20),
        relationship_type: kind,
        layer: Layer::Observed,
        revision: 2,
        properties: Vec::new(),
    })?;
    graph.delete_node(NodeId(20), true, 3)?;
    let reused = graph.insert_node(NodeInput {
        id: NodeId(200),
        layer: Layer::Workspace,
        revision: 4,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    assert_eq!(reused, old_node.dense());
    let edge = graph.insert_edge(EdgeInput {
        id: EdgeId(801),
        source: NodeId(10),
        target: NodeId(200),
        relationship_type: kind,
        layer: Layer::Workspace,
        revision: 5,
        properties: Vec::new(),
    })?;
    assert_eq!(edge, captured);
    assert!(!old_node.current());
    for mask in [LayerMask::OBSERVED, LayerMask::ALL] {
        let mut pairs = Vec::new();
        graph.walk_neighbor_denses_from(
            source,
            true,
            mask,
            u64::from(captured),
            |neighbor, edge| pairs.push((neighbor, edge)),
        );
        let current = current_pairs(&graph, NodeId(10), true, &pairs)?;
        assert!(current.iter().all(|(_, other)| *other != 20));
        assert_eq!(
            current
                .iter()
                .any(|(edge, other)| *edge == 801 && *other == 200),
            mask == LayerMask::ALL
        );
    }
    println!("CANONICAL_DENSE_ADJACENCY_PASS");
    Ok(())
}

#[test]
fn dense_adjacency_corrupt_cycles_and_truncated_tokens_are_bounded() -> Result<()> {
    let Fixture { graph, .. } = fixture()?;
    let owner = graph.node(NodeId(10)).ok_or_else(|| missing("owner"))?;
    let head = load(&graph.0.outgoing, owner.dense());
    let row = u32::try_from(head).map_err(|_| Error::internal("head ordinal"))?;
    let link = graph
        .0
        .next_out
        .get(row as usize)
        .ok_or_else(|| missing("head link"))?;
    let older = link.load(Ordering::Acquire);
    let older_row = u32::try_from(older).map_err(|_| Error::internal("older ordinal"))?;
    let older_link = graph
        .0
        .next_out
        .get(older_row as usize)
        .ok_or_else(|| missing("older link"))?;
    let mut pairs = Vec::new();
    link.store(head, Ordering::Release);
    graph.walk_neighbor_denses_from(owner, true, LayerMask::ALL, head, |neighbor, edge| {
        pairs.push((neighbor, edge))
    });
    assert_eq!(pairs.len(), 1);
    pairs.clear();
    link.store(older, Ordering::Release);
    older_link.store(head, Ordering::Release);
    graph.walk_neighbor_denses_from(owner, true, LayerMask::ALL, head, |neighbor, edge| {
        pairs.push((neighbor, edge))
    });
    assert_eq!(pairs.len(), 2);
    pairs.clear();
    graph.walk_neighbor_denses_from(
        owner,
        true,
        LayerMask::ALL,
        (1_u64 << 32) + head,
        |neighbor, edge| pairs.push((neighbor, edge)),
    );
    assert!(
        pairs.is_empty(),
        "bad 64-bit token was truncated to a live ordinal"
    );
    println!("CANONICAL_DENSE_ADJACENCY_PASS");
    Ok(())
}

#[test]
fn dense_adjacency_readers_finish_with_owner_writer_held() -> Result<()> {
    let Fixture { graph, .. } = fixture()?;
    let owner = graph.node(NodeId(10)).ok_or_else(|| missing("owner"))?;
    let permit = graph.claim_node(owner)?;
    thread::scope(|scope| -> Result<()> {
        let (send, receive) = mpsc::channel();
        let reader_graph = &graph;
        let reader = scope.spawn(move || {
            let mut pairs = Vec::new();
            let valid = reader_graph.visit_neighbor_denses(
                owner,
                true,
                LayerMask::ALL,
                |neighbor, edge| pairs.push((neighbor, edge)),
            );
            let _ = send.send((valid, pairs));
        });
        let read = receive.recv_timeout(Duration::from_secs(2));
        drop(permit);
        reader
            .join()
            .map_err(|_| Error::internal("reader panicked"))?;
        let (valid, pairs) =
            read.map_err(|error| Error::internal(format!("reader waited for writer: {error}")))?;
        assert!(valid);
        assert_eq!(pairs.len(), 7);
        current_pairs(&graph, NodeId(10), true, &pairs)?;
        Ok(())
    })?;
    println!("CANONICAL_DENSE_ADJACENCY_PASS");
    Ok(())
}

struct DirectAdjacency<'a> {
    graph: &'a GraphStore,
    outgoing: bool,
    visited: AtomicUsize,
}
impl AdjacencyRead for DirectAdjacency<'_> {
    type Row<'a>
        = std::vec::IntoIter<(u32, u32)>
    where
        Self: 'a;
    fn node_count(&self) -> usize {
        self.graph.node_slot_count()
    }
    fn row(&self, node: u32) -> Option<Self::Row<'_>> {
        let mut row = Vec::new();
        self.append_row(node, &mut row).then(|| row.into_iter())
    }
    fn append_row(&self, node: u32, output: &mut Vec<(u32, u32)>) -> bool {
        let Some(node) = self.graph.node_dense(node) else {
            return false;
        };
        self.graph
            .visit_neighbor_denses(node, self.outgoing, LayerMask::ALL, |neighbor, edge| {
                self.visited.fetch_add(1, Ordering::Relaxed);
                output.push((neighbor, edge));
            })
    }
}

#[test]
fn dense_adjacency_actual_triangle_clustering_reference_and_cancellation() -> Result<()> {
    let Fixture { graph, edges, .. } = fixture()?;
    let outgoing = DirectAdjacency {
        graph: &graph,
        outgoing: true,
        visited: AtomicUsize::new(0),
    };
    let incoming = DirectAdjacency {
        graph: &graph,
        outgoing: false,
        visited: AtomicUsize::new(0),
    };
    let mut matrix = [[false; 6]; 6];
    for (_, source, target, _, _) in edges {
        let left =
            usize::try_from(source / 10 - 1).map_err(|_| Error::internal("reference source"))?;
        let right =
            usize::try_from(target / 10 - 1).map_err(|_| Error::internal("reference target"))?;
        if left != right {
            matrix[left][right] = true;
            matrix[right][left] = true;
        }
    }
    let mut triangles = 0;
    for first in 0..6 {
        for second in first + 1..6 {
            for third in second + 1..6 {
                triangles += u64::from(
                    matrix[first][second] && matrix[first][third] && matrix[second][third],
                );
            }
        }
    }
    assert_eq!(
        triangle_count_cancellable(&outgoing, &incoming, || Ok(()))?,
        triangles
    );
    let coefficients = clustering_coefficients_cancellable(&outgoing, &incoming, || Ok(()))?;
    for (node, coefficient) in coefficients.into_iter().enumerate() {
        let neighbors = (0..6)
            .filter(|&other| matrix[node][other])
            .collect::<Vec<_>>();
        let mut links = 0;
        for (position, &left) in neighbors.iter().enumerate() {
            for &right in &neighbors[position + 1..] {
                links += usize::from(matrix[left][right]);
            }
        }
        let expected = if neighbors.len() < 2 {
            0.0
        } else {
            2.0 * links as f64 / (neighbors.len() * (neighbors.len() - 1)) as f64
        };
        assert_eq!(coefficient, expected);
    }
    outgoing.visited.store(0, Ordering::Relaxed);
    incoming.visited.store(0, Ordering::Relaxed);
    let mut checks = 0;
    let error = triangle_count_cancellable(&outgoing, &incoming, || {
        checks += 1;
        if checks == 3 {
            Err(Error::new(
                ErrorCode::Cancelled,
                "cancel after actual adjacency reads",
            ))
        } else {
            Ok(())
        }
    })
    .expect_err("actual adjacency algorithm ignored cancellation");
    assert_eq!(error.code, ErrorCode::Cancelled);
    assert!(
        outgoing.visited.load(Ordering::Relaxed) + incoming.visited.load(Ordering::Relaxed) > 0
    );
    println!("CANONICAL_DENSE_ADJACENCY_PASS");
    Ok(())
}
