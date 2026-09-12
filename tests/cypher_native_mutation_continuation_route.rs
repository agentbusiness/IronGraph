// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentDeviceCompletion, ResidentEntityBinding, ResidentExecutionReceipt, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentMutationOperation,
        ResidentMutationPostStage, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentObligationKind, ResidentObligationScope, ResidentProjectImage, ResidentSortRequest,
        ResidentSortResult, ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{
        EdgeInput, GraphMutation, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore,
    },
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 64;

const REMOVE_FEATURE: &str = "features/clauses/remove/Remove3.feature";
const SET_FEATURE: &str = "features/clauses/set/Set6.feature";
const REMOVE_FEATURE_FILTER: &str = "clauses/remove/Remove3.feature";
const SET_FEATURE_FILTER: &str = "clauses/set/Set6.feature";

/// The accepted complete-command boundary is `ResidentMutationProgram.continuation` inside one
/// pinned `execute_node_pipeline` call. It binds bookmark/revision/layout/fingerprint, ordered
/// stages, outputs, capacities, and obligations. Its raw completion and receipts correctly remain
/// private: module-private GPU tests own forged/missing/reordered receipt construction, while this
/// integration test injects faults only through public request and backend boundaries. No rows or
/// mutation intents may escape before the returned frame validates against the original request.
/// A mutation prefix followed by a host tail is not acceptable.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TailShape {
    LimitZero,
    SkipAll,
    PageTwo,
    PageAll,
    Filter,
    ReturnAggregate,
    WithAggregate,
}

impl TailShape {
    const ALL: [Self; 7] = [
        Self::LimitZero,
        Self::SkipAll,
        Self::PageTwo,
        Self::PageAll,
        Self::Filter,
        Self::ReturnAggregate,
        Self::WithAggregate,
    ];

    const fn index(self) -> usize {
        match self {
            Self::LimitZero => 0,
            Self::SkipAll => 1,
            Self::PageTwo => 2,
            Self::PageAll => 3,
            Self::Filter => 4,
            Self::ReturnAggregate => 5,
            Self::WithAggregate => 6,
        }
    }

    const fn selected_rows(self) -> usize {
        match self {
            Self::LimitZero | Self::SkipAll => 1,
            Self::PageTwo
            | Self::PageAll
            | Self::Filter
            | Self::ReturnAggregate
            | Self::WithAggregate => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MutationFamily {
    RemoveNodeProperty,
    RemoveNodeLabels,
    SetNodeProperty,
    SetNodeLabels,
    SetRelationshipProperty,
}

impl MutationFamily {
    const ALL: [Self; 5] = [
        Self::RemoveNodeProperty,
        Self::RemoveNodeLabels,
        Self::SetNodeProperty,
        Self::SetNodeLabels,
        Self::SetRelationshipProperty,
    ];

    const fn first_report_index(self) -> usize {
        match self {
            Self::RemoveNodeProperty => 673,
            Self::RemoveNodeLabels => 680,
            Self::SetNodeProperty => 855,
            Self::SetNodeLabels => 862,
            Self::SetRelationshipProperty => 869,
        }
    }

    const fn first_scenario_number(self) -> u8 {
        match self {
            Self::RemoveNodeProperty | Self::SetNodeProperty => 1,
            Self::RemoveNodeLabels | Self::SetNodeLabels => 8,
            Self::SetRelationshipProperty => 15,
        }
    }

    const fn feature(self) -> &'static str {
        match self {
            Self::RemoveNodeProperty | Self::RemoveNodeLabels => REMOVE_FEATURE,
            Self::SetNodeProperty | Self::SetNodeLabels | Self::SetRelationshipProperty => {
                SET_FEATURE
            }
        }
    }

    const fn feature_filter(self) -> &'static str {
        match self {
            Self::RemoveNodeProperty | Self::RemoveNodeLabels => REMOVE_FEATURE_FILTER,
            Self::SetNodeProperty | Self::SetNodeLabels | Self::SetRelationshipProperty => {
                SET_FEATURE_FILTER
            }
        }
    }

    const fn is_relationship(self) -> bool {
        matches!(self, Self::SetRelationshipProperty)
    }

    const fn writes_numeric_property(self) -> bool {
        matches!(self, Self::SetNodeProperty | Self::SetRelationshipProperty)
    }

    const fn names(self) -> &'static [&'static str; 7] {
        match self {
            Self::RemoveNodeProperty => &REMOVE_NODE_PROPERTY_NAMES,
            Self::RemoveNodeLabels => &REMOVE_NODE_LABEL_NAMES,
            Self::SetNodeProperty => &SET_NODE_PROPERTY_NAMES,
            Self::SetNodeLabels => &SET_NODE_LABEL_NAMES,
            Self::SetRelationshipProperty => &SET_RELATIONSHIP_PROPERTY_NAMES,
        }
    }

    const fn queries(self) -> &'static [&'static str; 7] {
        match self {
            Self::RemoveNodeProperty => &REMOVE_NODE_PROPERTY_QUERIES,
            Self::RemoveNodeLabels => &REMOVE_NODE_LABEL_QUERIES,
            Self::SetNodeProperty => &SET_NODE_PROPERTY_QUERIES,
            Self::SetNodeLabels => &SET_NODE_LABEL_QUERIES,
            Self::SetRelationshipProperty => &SET_RELATIONSHIP_PROPERTY_QUERIES,
        }
    }
}

const REMOVE_NODE_PROPERTY_NAMES: [&str; 7] = [
    "[1] Limiting to zero results after removing a property from nodes affects the result set but not the side effects",
    "[2] Skipping all results after removing a property from nodes affects the result set but not the side effects",
    "[3] Skipping and limiting to a few results after removing a property from nodes affects the result set but not the side effects",
    "[4] Skipping zero results and limiting to all results after removing a property from nodes does not affect the result set nor the side effects",
    "[5] Filtering after removing a property from nodes affects the result set but not the side effects",
    "[6] Aggregating in `RETURN` after removing a property from nodes affects the result set but not the side effects",
    "[7] Aggregating in `WITH` after removing a property from nodes affects the result set but not the side effects",
];

const REMOVE_NODE_LABEL_NAMES: [&str; 7] = [
    "[8] Limiting to zero results after removing a label from nodes affects the result set but not the side effects",
    "[9] Skipping all results after removing a label from nodes affects the result set but not the side effects",
    "[10] Skipping and limiting to a few results after removing a label from nodes affects the result set but not the side effects",
    "[11] Skipping zero result and limiting to all results after removing a label from nodes does not affect the result set nor the side effects",
    "[12] Filtering after removing a label from nodes affects the result set but not the side effects",
    "[13] Aggregating in `RETURN` after removing a label from nodes affects the result set but not the side effects",
    "[14] Aggregating in `WITH` after removing a label from nodes affects the result set but not the side effects",
];

const SET_NODE_PROPERTY_NAMES: [&str; 7] = [
    "[1] Limiting to zero results after setting a property on nodes affects the result set but not the side effects",
    "[2] Skipping all results after setting a property on nodes affects the result set but not the side effects",
    "[3] Skipping and limiting to a few results after setting a property on nodes affects the result set but not the side effects",
    "[4] Skipping zero results and limiting to all results after setting a property on nodes does not affect the result set nor the side effects",
    "[5] Filtering after setting a property on nodes affects the result set but not the side effects",
    "[6] Aggregating in `RETURN` after setting a property on nodes affects the result set but not the side effects",
    "[7] Aggregating in `WITH` after setting a property on nodes affects the result set but not the side effects",
];

const SET_NODE_LABEL_NAMES: [&str; 7] = [
    "[8] Limiting to zero results after adding a label on nodes affects the result set but not the side effects",
    "[9] Skipping all results after adding a label on nodes affects the result set but not the side effects",
    "[10] Skipping and limiting to a few results after adding a label on nodes affects the result set but not the side effects",
    "[11] Skipping zero result and limiting to all results after adding a label on nodes does not affect the result set nor the side effects",
    "[12] Filtering after adding a label on nodes affects the result set but not the side effects",
    "[13] Aggregating in `RETURN` after adding a label on nodes affects the result set but not the side effects",
    "[14] Aggregating in `WITH` after adding a label on nodes affects the result set but not the side effects",
];

const SET_RELATIONSHIP_PROPERTY_NAMES: [&str; 7] = [
    "[15] Limiting to zero results after setting a property on relationships affects the result set but not the side effects",
    "[16] Skipping all results after setting a property on relationships affects the result set but not the side effects",
    "[17] Skipping and limiting to a few results after setting a property on relationships affects the result set but not the side effects",
    "[18] Skipping zero result and limiting to all results after setting a property on relationships does not affect the result set nor the side effects",
    "[19] Filtering after setting a property on relationships affects the result set but not the side effects",
    "[20] Aggregating in `RETURN` after setting a property on relationships affects the result set but not the side effects",
    "[21] Aggregating in `WITH` after setting a property on relationships affects the result set but not the side effects",
];

const REMOVE_NODE_PROPERTY_QUERIES: [&str; 7] = [
    "MATCH (n:N) REMOVE n.num RETURN n LIMIT 0",
    "MATCH (n:N) REMOVE n.num RETURN n SKIP 1",
    "MATCH (n:N) REMOVE n.name RETURN n.num AS num SKIP 2 LIMIT 2",
    "MATCH (n:N) REMOVE n.name RETURN n.num AS num SKIP 0 LIMIT 5",
    "MATCH (n:N) REMOVE n.name WITH n WHERE n.num % 2 = 0 RETURN n.num AS num",
    "MATCH (n:N) REMOVE n.name RETURN sum(n.num) AS sum",
    "MATCH (n:N) REMOVE n.name WITH sum(n.num) AS sum RETURN sum",
];

const REMOVE_NODE_LABEL_QUERIES: [&str; 7] = [
    "MATCH (n:N) REMOVE n:N RETURN n LIMIT 0",
    "MATCH (n:N) REMOVE n:N RETURN n SKIP 1",
    "MATCH (n:N) REMOVE n:N RETURN n.num AS num SKIP 2 LIMIT 2",
    "MATCH (n:N) REMOVE n:N RETURN n.num AS num SKIP 0 LIMIT 5",
    "MATCH (n:N) REMOVE n:N WITH n WHERE n.num % 2 = 0 RETURN n.num AS num",
    "MATCH (n:N) REMOVE n:N RETURN sum(n.num) AS sum",
    "MATCH (n:N) REMOVE n:N WITH sum(n.num) AS sum RETURN sum",
];

const SET_NODE_PROPERTY_QUERIES: [&str; 7] = [
    "MATCH (n:N) SET n.num = 43 RETURN n LIMIT 0",
    "MATCH (n:N) SET n.num = 43 RETURN n SKIP 1",
    "MATCH (n:N) SET n.num = 42 RETURN n.num AS num SKIP 2 LIMIT 2",
    "MATCH (n:N) SET n.num = 42 RETURN n.num AS num SKIP 0 LIMIT 5",
    "MATCH (n:N) SET n.num = n.num + 1 WITH n WHERE n.num % 2 = 0 RETURN n.num AS num",
    "MATCH (n:N) SET n.num = n.num + 1 RETURN sum(n.num) AS sum",
    "MATCH (n:N) SET n.num = n.num + 1 WITH sum(n.num) AS sum RETURN sum",
];

const SET_NODE_LABEL_QUERIES: [&str; 7] = [
    "MATCH (n:N) SET n:Foo RETURN n LIMIT 0",
    "MATCH (n:N) SET n:Foo RETURN n SKIP 1",
    "MATCH (n:N) SET n:Foo RETURN n.num AS num SKIP 2 LIMIT 2",
    "MATCH (n:N) SET n:Foo RETURN n.num AS num SKIP 0 LIMIT 5",
    "MATCH (n:N) SET n:Foo WITH n WHERE n.num % 2 = 0 RETURN n.num AS num",
    "MATCH (n:N) SET n:Foo RETURN sum(n.num) AS sum",
    "MATCH (n:N) SET n:Foo WITH sum(n.num) AS sum RETURN sum",
];

const SET_RELATIONSHIP_PROPERTY_QUERIES: [&str; 7] = [
    "MATCH ()-[r:R]->() SET r.num = 43 RETURN r LIMIT 0",
    "MATCH ()-[r:R]->() SET r.num = 43 RETURN r SKIP 1",
    "MATCH ()-[r:R]->() SET r.num = 42 RETURN r.num AS num SKIP 2 LIMIT 2",
    "MATCH ()-[r:R]->() SET r.num = 42 RETURN r.num AS num SKIP 0 LIMIT 5",
    "MATCH ()-[r:R]->() SET r.num = r.num + 1 WITH r WHERE r.num % 2 = 0 RETURN r.num AS num",
    "MATCH ()-[r:R]->() SET r.num = r.num + 1 RETURN sum(r.num) AS sum",
    "MATCH ()-[r:R]->() SET r.num = r.num + 1 WITH sum(r.num) AS sum RETURN sum",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SetupShape {
    OneNode42,
    FiveNodes42,
    FiveNodesOneToFive,
    FiveNamedNodes42,
    FiveNamedNodesOneToFive,
    OneRelationship42,
    FiveRelationshipsOneToFive,
}

impl SetupShape {
    const fn cypher(self) -> &'static str {
        match self {
            Self::OneNode42 => "CREATE (:N {num: 42})",
            Self::FiveNodes42 => {
                "CREATE (:N {num: 42})\nCREATE (:N {num: 42})\nCREATE (:N {num: 42})\nCREATE (:N {num: 42})\nCREATE (:N {num: 42})"
            }
            Self::FiveNodesOneToFive => {
                "CREATE (:N {num: 1})\nCREATE (:N {num: 2})\nCREATE (:N {num: 3})\nCREATE (:N {num: 4})\nCREATE (:N {num: 5})"
            }
            Self::FiveNamedNodes42 => {
                "CREATE (:N {name: 'a', num: 42})\nCREATE (:N {name: 'a', num: 42})\nCREATE (:N {name: 'a', num: 42})\nCREATE (:N {name: 'a', num: 42})\nCREATE (:N {name: 'a', num: 42})"
            }
            Self::FiveNamedNodesOneToFive => {
                "CREATE (:N {name: 'a', num: 1})\nCREATE (:N {name: 'a', num: 2})\nCREATE (:N {name: 'a', num: 3})\nCREATE (:N {name: 'a', num: 4})\nCREATE (:N {name: 'a', num: 5})"
            }
            Self::OneRelationship42 => "CREATE ()-[r:R {num: 42}]->()",
            Self::FiveRelationshipsOneToFive => {
                "CREATE ()-[:R {num: 1}]->()\nCREATE ()-[:R {num: 2}]->()\nCREATE ()-[:R {num: 3}]->()\nCREATE ()-[:R {num: 4}]->()\nCREATE ()-[:R {num: 5}]->()"
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TckCase {
    /// Zero-based index in the certified JSON `scenarios` array.
    report_index: usize,
    /// Human-facing one-based report ID used in progress output.
    displayed_report_id: usize,
    feature_scenario: u8,
    feature: &'static str,
    feature_filter: &'static str,
    name: &'static str,
    family: MutationFamily,
    tail: TailShape,
    setup: SetupShape,
    query: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NoOpMutationKind {
    RemoveProperty,
    RemoveLabels,
    AddLabels,
}

#[derive(Clone, Copy, Debug)]
struct NoOpCase {
    name: &'static str,
    query: &'static str,
    kind: NoOpMutationKind,
}

const NO_OP_CASES: [NoOpCase; 5] = [
    NoOpCase {
        name: "REMOVE globally unknown node property",
        query: "MATCH (n:N) REMOVE n.globallyUnknown RETURN n.num AS num",
        kind: NoOpMutationKind::RemoveProperty,
    },
    NoOpCase {
        name: "REMOVE property token present only on Other",
        query: "MATCH (n:N) REMOVE n.onlyOnOther RETURN n.num AS num",
        kind: NoOpMutationKind::RemoveProperty,
    },
    NoOpCase {
        name: "REMOVE globally unknown label",
        query: "MATCH (n:N) REMOVE n:GloballyUnknown RETURN n.num AS num",
        kind: NoOpMutationKind::RemoveLabels,
    },
    NoOpCase {
        name: "REMOVE label token present only on Other",
        query: "MATCH (n:N) REMOVE n:OnlyOnOther RETURN n.num AS num",
        kind: NoOpMutationKind::RemoveLabels,
    },
    NoOpCase {
        name: "SET already-present label",
        query: "MATCH (n:N) SET n:N RETURN n.num AS num",
        kind: NoOpMutationKind::AddLabels,
    },
];

impl TckCase {
    fn label(self) -> String {
        format!(
            "report index {} / displayed ID {} / scenario [{}] {}",
            self.report_index, self.displayed_report_id, self.feature_scenario, self.name
        )
    }

    const fn expected_column(self) -> &'static str {
        match self.tail {
            TailShape::LimitZero | TailShape::SkipAll => {
                if self.family.is_relationship() {
                    "r"
                } else {
                    "n"
                }
            }
            TailShape::PageTwo | TailShape::PageAll | TailShape::Filter => "num",
            TailShape::ReturnAggregate | TailShape::WithAggregate => "sum",
        }
    }

    fn expected_values(self) -> Vec<i64> {
        match self.tail {
            TailShape::LimitZero | TailShape::SkipAll => Vec::new(),
            TailShape::PageTwo => vec![42, 42],
            TailShape::PageAll => vec![42; 5],
            TailShape::Filter if self.family.writes_numeric_property() => vec![2, 4, 6],
            TailShape::Filter => vec![2, 4],
            TailShape::ReturnAggregate | TailShape::WithAggregate
                if self.family.writes_numeric_property() =>
            {
                vec![20]
            }
            TailShape::ReturnAggregate | TailShape::WithAggregate => vec![15],
        }
    }

    fn expected_stats(self) -> StatementStats {
        let selected = self.tail.selected_rows() as u64;
        match self.family {
            MutationFamily::RemoveNodeProperty
            | MutationFamily::SetNodeProperty
            | MutationFamily::SetRelationshipProperty => StatementStats {
                properties_set: selected,
                ..StatementStats::default()
            },
            MutationFamily::RemoveNodeLabels => StatementStats {
                labels_removed: selected,
                ..StatementStats::default()
            },
            MutationFamily::SetNodeLabels => StatementStats {
                labels_added: selected,
                ..StatementStats::default()
            },
        }
    }
}

fn setup_for(family: MutationFamily, tail: TailShape) -> SetupShape {
    match (family, tail) {
        (MutationFamily::RemoveNodeProperty, TailShape::LimitZero | TailShape::SkipAll) => {
            SetupShape::OneNode42
        }
        (MutationFamily::RemoveNodeProperty, TailShape::PageTwo | TailShape::PageAll) => {
            SetupShape::FiveNamedNodes42
        }
        (MutationFamily::RemoveNodeProperty, _) => SetupShape::FiveNamedNodesOneToFive,
        (MutationFamily::RemoveNodeLabels, TailShape::LimitZero | TailShape::SkipAll) => {
            SetupShape::OneNode42
        }
        (MutationFamily::RemoveNodeLabels, TailShape::PageTwo | TailShape::PageAll) => {
            SetupShape::FiveNodes42
        }
        (MutationFamily::RemoveNodeLabels, _) => SetupShape::FiveNodesOneToFive,
        (MutationFamily::SetNodeProperty, TailShape::LimitZero | TailShape::SkipAll)
        | (MutationFamily::SetNodeLabels, TailShape::LimitZero | TailShape::SkipAll) => {
            SetupShape::OneNode42
        }
        (MutationFamily::SetNodeLabels, TailShape::PageTwo | TailShape::PageAll) => {
            SetupShape::FiveNodes42
        }
        (MutationFamily::SetNodeProperty | MutationFamily::SetNodeLabels, _) => {
            SetupShape::FiveNodesOneToFive
        }
        (MutationFamily::SetRelationshipProperty, TailShape::LimitZero | TailShape::SkipAll) => {
            SetupShape::OneRelationship42
        }
        (MutationFamily::SetRelationshipProperty, _) => SetupShape::FiveRelationshipsOneToFive,
    }
}

fn all_cases() -> impl Iterator<Item = TckCase> {
    MutationFamily::ALL.into_iter().flat_map(|family| {
        TailShape::ALL.into_iter().map(move |tail| {
            let offset = tail.index();
            let report_index = family.first_report_index() + offset;
            TckCase {
                report_index,
                displayed_report_id: report_index + 1,
                feature_scenario: family.first_scenario_number()
                    + u8::try_from(offset).expect("seven-case offset fits u8"),
                feature: family.feature(),
                feature_filter: family.feature_filter(),
                name: family.names()[offset],
                family,
                tail,
                setup: setup_for(family, tail),
                query: family.queries()[offset],
            }
        })
    })
}

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    num: PropertyId,
    name: PropertyId,
    n_label: LabelId,
}

impl Fixture {
    fn new(case: TckCase) -> Result<Self> {
        let mut graph = GraphStore::default();
        let n_label = graph.catalog_mut().intern_label("N")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let selected_rows = case.tail.selected_rows();

        if case.family.is_relationship() {
            for offset in 0..selected_rows {
                let source = NodeId(u64::try_from(offset * 2 + 1).expect("fixture ID fits u64"));
                let target = NodeId(u64::try_from(offset * 2 + 2).expect("fixture ID fits u64"));
                let revision = u64::try_from(offset * 3).expect("fixture revision fits u64");
                graph.insert_node(NodeInput {
                    id: source,
                    layer: Layer::Observed,
                    revision: revision + 1,
                    labels: Vec::new(),
                    properties: Vec::new(),
                })?;
                graph.insert_node(NodeInput {
                    id: target,
                    layer: Layer::Observed,
                    revision: revision + 2,
                    labels: Vec::new(),
                    properties: Vec::new(),
                })?;
                graph.insert_edge(EdgeInput {
                    id: EdgeId(100 + u64::try_from(offset).expect("fixture ID fits u64")),
                    source,
                    target,
                    relationship_type,
                    layer: Layer::Observed,
                    revision: revision + 3,
                    properties: vec![(num, ScalarValue::Integer(Self::initial_num(case, offset)))],
                })?;
            }
        } else {
            for offset in 0..selected_rows {
                let id = NodeId(u64::try_from(offset + 1).expect("fixture ID fits u64"));
                let mut properties =
                    vec![(num, ScalarValue::Integer(Self::initial_num(case, offset)))];
                if matches!(case.family, MutationFamily::RemoveNodeProperty)
                    && !matches!(case.tail, TailShape::LimitZero | TailShape::SkipAll)
                {
                    properties.push((name, ScalarValue::String(Arc::<str>::from("a"))));
                }
                graph.insert_node(NodeInput {
                    id,
                    layer: Layer::Observed,
                    revision: id.0,
                    labels: vec![n_label],
                    properties,
                })?;
            }
        }

        let bookmark = Bookmark {
            term: 37,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            num,
            name,
            n_label,
        })
    }

    fn initial_num(case: TckCase, offset: usize) -> i64 {
        match case.setup {
            SetupShape::OneNode42
            | SetupShape::FiveNodes42
            | SetupShape::FiveNamedNodes42
            | SetupShape::OneRelationship42 => 42,
            SetupShape::FiveNodesOneToFive
            | SetupShape::FiveNamedNodesOneToFive
            | SetupShape::FiveRelationshipsOneToFive => {
                i64::try_from(offset + 1).expect("five fixture rows fit i64")
            }
        }
    }

    fn duplicate_target() -> Result<Self> {
        let mut graph = GraphStore::default();
        let n_label = graph.catalog_mut().intern_label("N")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let name = graph.catalog_mut().intern_property("name")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![n_label],
            properties: vec![(num, ScalarValue::Integer(0))],
        })?;
        for target in [NodeId(2), NodeId(3)] {
            graph.insert_node(NodeInput {
                id: target,
                layer: Layer::Observed,
                revision: target.0,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        for (offset, target) in [NodeId(2), NodeId(3)].into_iter().enumerate() {
            graph.insert_edge(EdgeInput {
                id: EdgeId(10 + target.0),
                source: NodeId(1),
                target,
                relationship_type,
                layer: Layer::Observed,
                revision: 4 + u64::try_from(offset).expect("two edges fit u64"),
                properties: Vec::new(),
            })?;
        }
        let bookmark = Bookmark {
            term: 41,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            num,
            name,
            n_label,
        })
    }

    fn no_op_effects() -> Result<Self> {
        let mut graph = GraphStore::default();
        let n_label = graph.catalog_mut().intern_label("N")?;
        let other_label = graph.catalog_mut().intern_label("Other")?;
        let only_on_other_label = graph.catalog_mut().intern_label("OnlyOnOther")?;
        let _relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let name = graph.catalog_mut().intern_property("onlyOnOther")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![n_label],
            properties: vec![(num, ScalarValue::Integer(7))],
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![other_label, only_on_other_label],
            properties: vec![(name, ScalarValue::Integer(99))],
        })?;
        let bookmark = Bookmark {
            term: 43,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            num,
            name,
            n_label,
        })
    }

    fn image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }

    fn cpu(&self) -> Result<CpuBackend> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(self.image()?)?;
        Ok(cpu)
    }
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
        graph: &fixture.graph,
        binding_catalog: fixture.graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1_000,
        next_edge_id: 1_000,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 2,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn mutation_debug(output: &ExecutionOutput) -> Vec<String> {
    output
        .graph_mutations
        .iter()
        .map(|mutation| format!("{mutation:?}"))
        .collect()
}

fn entity_mutations(output: &ExecutionOutput) -> Vec<&GraphMutation> {
    output
        .graph_mutations
        .iter()
        .filter(|mutation| {
            matches!(
                mutation,
                GraphMutation::SetNodeProperty { .. }
                    | GraphMutation::SetEdgeProperty { .. }
                    | GraphMutation::AddNodeLabels { .. }
                    | GraphMutation::RemoveNodeLabels { .. }
            )
        })
        .collect()
}

fn node_property_integer_intents(
    output: &ExecutionOutput,
    node: NodeId,
    property: PropertyId,
) -> Vec<i64> {
    output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::SetNodeProperty {
                node: mutation_node,
                property: mutation_property,
                value: ScalarValue::Integer(value),
                ..
            } if *mutation_node == node && *mutation_property == property => Some(*value),
            _ => None,
        })
        .collect()
}

fn node_property_integer_intents_by_target(
    output: &ExecutionOutput,
    property: PropertyId,
) -> Vec<(NodeId, i64)> {
    output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::SetNodeProperty {
                node,
                property: mutation_property,
                value: ScalarValue::Integer(value),
                ..
            } if *mutation_property == property => Some((*node, *value)),
            _ => None,
        })
        .collect()
}

fn integer_values(
    output: &ExecutionOutput,
    column_name: &str,
) -> std::result::Result<Vec<i64>, String> {
    let mut values = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() {
            return Err("result batch has misaligned columns".to_owned());
        }
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == column_name)
            .ok_or_else(|| format!("result batch omitted expected `{column_name}` column"))?;
        for value in &column.values {
            match value {
                ResultValue::Scalar(ScalarValue::Integer(value)) => values.push(*value),
                _ => return Err(format!("expected integer result, got {value:?}")),
            }
        }
    }
    values.sort_unstable();
    Ok(values)
}

fn assert_official_output(
    fixture: &Fixture,
    case: TckCase,
    output: &ExecutionOutput,
) -> std::result::Result<(), String> {
    let expected_type = if matches!(case.tail, TailShape::LimitZero | TailShape::SkipAll) {
        ColumnType::Null
    } else {
        ColumnType::Integer
    };
    let expected_schema = vec![(case.expected_column().to_owned(), expected_type)];
    if output.result.schema != expected_schema {
        return Err(format!(
            "schema mismatch: expected {expected_schema:?}, got {:?}",
            output.result.schema
        ));
    }
    let actual_values = integer_values(output, case.expected_column())?;
    let mut expected_values = case.expected_values();
    expected_values.sort_unstable();
    if actual_values != expected_values {
        return Err(format!(
            "row mismatch: expected {expected_values:?}, got {actual_values:?}"
        ));
    }
    if output.result.statistics != case.expected_stats() {
        return Err(format!(
            "statistics mismatch: expected {:?}, got {:?}",
            case.expected_stats(),
            output.result.statistics
        ));
    }
    if output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
    {
        return Err("result changed its bookmark, truncated, or emitted temporal work".to_owned());
    }

    let mutations = entity_mutations(output);
    let selected_rows = case.tail.selected_rows();
    if mutations.len() != selected_rows {
        return Err(format!(
            "mutation intent cardinality mismatch: expected {selected_rows}, got {}: {:?}",
            mutations.len(),
            mutation_debug(output)
        ));
    }
    if output.dependencies.write_targets.len() != selected_rows {
        return Err(format!(
            "write-target cardinality mismatch: expected {selected_rows}, got {:?}",
            output.dependencies.write_targets
        ));
    }

    let exact_actions = match case.family {
        MutationFamily::RemoveNodeProperty => mutations.iter().all(|mutation| {
            matches!(
                mutation,
                GraphMutation::SetNodeProperty {
                    property,
                    value: ScalarValue::Null,
                    ..
                } if *property == if matches!(case.tail, TailShape::LimitZero | TailShape::SkipAll) {
                    fixture.num
                } else {
                    fixture.name
                }
            )
        }),
        MutationFamily::RemoveNodeLabels => mutations.iter().all(|mutation| {
            matches!(
                mutation,
                GraphMutation::RemoveNodeLabels { labels, .. }
                    if labels.as_slice() == [fixture.n_label]
            )
        }),
        MutationFamily::SetNodeProperty => mutations.iter().all(|mutation| {
            matches!(
                mutation,
                GraphMutation::SetNodeProperty {
                    property,
                    value: ScalarValue::Integer(_),
                    ..
                } if *property == fixture.num
            )
        }),
        MutationFamily::SetNodeLabels => {
            let foo = output.graph_mutations.iter().find_map(|mutation| match mutation {
                GraphMutation::DeclareLabel { name, id } if name == "Foo" => Some(*id),
                _ => None,
            });
            foo.is_some_and(|foo| {
                mutations.iter().all(|mutation| {
                    matches!(
                        mutation,
                        GraphMutation::AddNodeLabels { labels, .. } if labels.as_slice() == [foo]
                    )
                })
            })
        }
        MutationFamily::SetRelationshipProperty => mutations.iter().all(|mutation| {
            matches!(
                mutation,
                GraphMutation::SetEdgeProperty {
                    property,
                    value: ScalarValue::Integer(_),
                    ..
                } if *property == fixture.num
            )
        }),
    };
    if !exact_actions {
        return Err(format!(
            "mutation action mismatch: {:?}",
            mutation_debug(output)
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ResultFault {
    #[default]
    None,
    MissingMutationResult,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RequestFrameFault {
    #[default]
    None,
    WrongBookmark,
    WrongRevision,
    WrongLayout,
    DifferentFingerprint,
    ReorderedStages,
    DuplicateEffectObligationId,
    WrongEffectObligationKind,
    WrongEffectObligationScope,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum GenerationFault {
    #[default]
    None,
    WrongBookmark,
    WrongRevision,
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    complete_command_calls: AtomicUsize,
    unexpected_query_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentNodePipelineRequest>>,
    validated_mutations: Mutex<Vec<ValidatedMutationObservation>>,
}

#[derive(Clone, Debug)]
struct ValidatedMutationObservation {
    receipts: Vec<ResidentExecutionReceipt>,
    effectful_intents: usize,
}

/// Strict observer for the present public ABI. It allows only one pinned mutation-bearing
/// pipeline call and closes every generic scan/join/filter/aggregate route. The acceptance tests
/// treat that call as valid only if it returns the complete post-write result and side effects.
/// Once the dedicated hook documented above exists, this observer must forward that hook instead
/// and reject `execute_node_pipeline` as a legacy mutation-prefix route.
struct ObservedMutationContinuationBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    result_fault: ResultFault,
    request_frame_fault: RequestFrameFault,
    generation_fault: GenerationFault,
    observations: Arc<RouteObservations>,
}

impl ObservedMutationContinuationBackend {
    fn strict_cpu_reference(inner: CpuBackend) -> Result<Self> {
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
            ResultFault::None,
            RequestFrameFault::None,
            GenerationFault::None,
        )
    }

    fn wrong_receipt_provenance(inner: CpuBackend) -> Result<Self> {
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Cpu,
            ResultFault::None,
            RequestFrameFault::None,
            GenerationFault::None,
        )
    }

    fn result_fault(inner: CpuBackend, fault: ResultFault) -> Result<Self> {
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
            fault,
            RequestFrameFault::None,
            GenerationFault::None,
        )
    }

    fn request_frame_fault(inner: CpuBackend, fault: RequestFrameFault) -> Result<Self> {
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
            ResultFault::None,
            fault,
            GenerationFault::None,
        )
    }

    fn generation_fault(inner: CpuBackend, fault: GenerationFault) -> Result<Self> {
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
            ResultFault::None,
            RequestFrameFault::None,
            fault,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "mutation-continuation test did not construct a real Metal backend",
            ));
        }
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
            ResultFault::None,
            RequestFrameFault::None,
            GenerationFault::None,
        )
    }

    fn new<B: ExecutionBackend + 'static>(
        inner: B,
        advertised_kind: BackendKind,
        pinned_kind: BackendKind,
        actual_kind: BackendKind,
        result_fault: ResultFault,
        request_frame_fault: RequestFrameFault,
        generation_fault: GenerationFault,
    ) -> Result<Self> {
        let expected_bookmark = inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("mutation-continuation backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("mutation-continuation backend has no admitted graph revision")
        })?;
        Ok(Self {
            inner: Box::new(inner),
            advertised_kind,
            pinned_kind,
            actual_kind,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            result_fault,
            request_frame_fault,
            generation_fault,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn reject_query_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .unexpected_query_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict mutation-continuation test rejected `{route}` execution"),
        ))
    }

    fn refresh_expected_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement mutation-continuation project has no bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal("replacement mutation-continuation project has no revision")
            })?;
        Ok(())
    }
}

impl ExecutionBackend for ObservedMutationContinuationBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.pinned_kind
        } else {
            self.advertised_kind
        }
    }

    fn available_query_scratch_bytes(&self) -> usize {
        self.inner.available_query_scratch_bytes()
    }

    fn reserve_query_scratch(&self, bytes: usize) -> Result<ScratchReservation> {
        self.inner.reserve_query_scratch(bytes)
    }

    fn resident_project_bytes(&self, project: ProjectId) -> Option<usize> {
        self.inner.resident_project_bytes(project)
    }

    fn pin_project(&self, project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        if self.pinned {
            return self.reject_query_route("pin_project_twice");
        }
        if project != PROJECT {
            return self.reject_query_route("pin_wrong_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.actual_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "pinned mutation-continuation generation changed before dispatch",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            advertised_kind: self.advertised_kind,
            pinned_kind: self.pinned_kind,
            actual_kind: self.actual_kind,
            pinned: true,
            expected_bookmark: self.expected_bookmark,
            expected_graph_revision: self.expected_graph_revision,
            result_fault: self.result_fault,
            request_frame_fault: self.request_frame_fault,
            generation_fault: self.generation_fault,
            observations: Arc::clone(&self.observations),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        self.inner.admit_project(image)?;
        self.refresh_expected_fence()
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)?;
        self.refresh_expected_fence()
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        self.inner.advance_bookmark(bookmark);
    }

    fn resident_revision(&self) -> Option<u64> {
        self.inner.resident_revision()
    }

    fn resident_graph_revision(&self, project: ProjectId) -> Option<u64> {
        let revision = self.inner.resident_graph_revision(project)?;
        if self.pinned && self.generation_fault == GenerationFault::WrongRevision {
            Some(revision.saturating_add(1))
        } else {
            Some(revision)
        }
    }

    fn resident_bookmark(&self, project: ProjectId) -> Option<Bookmark> {
        let mut bookmark = self.inner.resident_bookmark(project)?;
        if self.pinned && self.generation_fault == GenerationFault::WrongBookmark {
            bookmark.index = bookmark.index.saturating_add(1);
        }
        Some(bookmark)
    }

    fn scan_nodes(
        &self,
        _project: ProjectId,
        _label: Option<LabelId>,
        _layers: LayerMask,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_query_route("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject_query_route("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_query_route("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_query_route("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        if !self.pinned {
            return self.reject_query_route("execute_node_pipeline_on_unpinned_generation");
        }
        if request.project != PROJECT
            || request.mutation.is_none()
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "mutation-continuation request escaped its pinned generation",
            ));
        }
        self.observations
            .complete_command_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        let mut dispatch = request.clone();
        if self.request_frame_fault != RequestFrameFault::None {
            let program = dispatch
                .mutation
                .as_mut()
                .ok_or_else(|| Error::internal("request-frame fault has no mutation program"))?;
            match self.request_frame_fault {
                RequestFrameFault::None => unreachable!(),
                RequestFrameFault::WrongBookmark => {
                    let continuation = program.continuation.as_mut().ok_or_else(|| {
                        Error::internal("request-frame fault has no mutation continuation")
                    })?;
                    continuation.expected_bookmark.index =
                        continuation.expected_bookmark.index.saturating_add(1);
                }
                RequestFrameFault::WrongRevision => {
                    let continuation = program.continuation.as_mut().ok_or_else(|| {
                        Error::internal("request-frame fault has no mutation continuation")
                    })?;
                    continuation.expected_graph_revision =
                        continuation.expected_graph_revision.saturating_add(1);
                }
                RequestFrameFault::WrongLayout => {
                    let continuation = program.continuation.as_mut().ok_or_else(|| {
                        Error::internal("request-frame fault has no mutation continuation")
                    })?;
                    continuation.expected_layout_version =
                        continuation.expected_layout_version.saturating_add(1);
                }
                RequestFrameFault::DifferentFingerprint => {
                    let continuation = program.continuation.as_mut().ok_or_else(|| {
                        Error::internal("request-frame fault has no mutation continuation")
                    })?;
                    continuation.maximum_output_arena_bytes =
                        continuation.maximum_output_arena_bytes.saturating_add(1);
                }
                RequestFrameFault::ReorderedStages => {
                    let continuation = program.continuation.as_mut().ok_or_else(|| {
                        Error::internal("request-frame fault has no mutation continuation")
                    })?;
                    if continuation.stages.len() < 3 || continuation.stage_obligations.len() < 3 {
                        return Err(Error::internal(
                            "reordered-stage fault requires PROJECT, SKIP, and LIMIT",
                        ));
                    }
                    continuation.stages.swap(1, 2);
                }
                RequestFrameFault::DuplicateEffectObligationId => {
                    let command = program.commands.first_mut().ok_or_else(|| {
                        Error::internal("effect fault requires one mutation command")
                    })?;
                    command.effect_obligation.id = command.rhs_obligation.id;
                }
                RequestFrameFault::WrongEffectObligationKind => {
                    program
                        .commands
                        .first_mut()
                        .ok_or_else(|| {
                            Error::internal("effect fault requires one mutation command")
                        })?
                        .effect_obligation
                        .kind = ResidentObligationKind::MutationRhs;
                }
                RequestFrameFault::WrongEffectObligationScope => {
                    program
                        .commands
                        .first_mut()
                        .ok_or_else(|| {
                            Error::internal("effect fault requires one mutation command")
                        })?
                        .effect_obligation
                        .scope = ResidentObligationScope::Selection;
                }
            }
            dispatch.seal_mutation_pipeline()?;
        }
        let mut result = self.inner.execute_node_pipeline(&dispatch, cancellation)?;
        match self.result_fault {
            ResultFault::None => {}
            ResultFault::MissingMutationResult => result.mutation = None,
        }
        if let Some(mutation) = result.mutation.clone() {
            let validated = mutation.validate_for_publication(&dispatch, self.actual_kind)?;
            self.observations
                .validated_mutations
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(ValidatedMutationObservation {
                    receipts: validated.receipts().to_vec(),
                    effectful_intents: validated.intents().len(),
                });
        }
        Ok(result)
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_query_route("exact_l2")
    }
}

fn assert_native_request(
    fixture: &Fixture,
    case: TckCase,
    request: &ResidentNodePipelineRequest,
) -> std::result::Result<(), String> {
    if request.project != PROJECT {
        return Err(format!("complete request targeted {:?}", request.project));
    }
    let program = request
        .mutation
        .as_ref()
        .ok_or_else(|| "complete request omitted its mutation program".to_owned())?;
    if program.commands.len() != 1 || program.max_intents < case.tail.selected_rows() {
        return Err(format!(
            "expected one command and capacity for at least {} intents, got {} commands / {} intents",
            case.tail.selected_rows(),
            program.commands.len(),
            program.max_intents
        ));
    }
    let operation_matches = match (case.family, &program.commands[0].operation) {
        (
            MutationFamily::RemoveNodeProperty,
            ResidentMutationOperation::RemoveProperty {
                target: ResidentEntityBinding::Node(_),
                ..
            },
        )
        | (MutationFamily::RemoveNodeLabels, ResidentMutationOperation::RemoveLabels { .. })
        | (
            MutationFamily::SetNodeProperty,
            ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Node(_),
                ..
            },
        )
        | (MutationFamily::SetNodeLabels, ResidentMutationOperation::AddLabels { .. })
        | (
            MutationFamily::SetRelationshipProperty,
            ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Relationship(_),
                ..
            },
        ) => true,
        _ => false,
    };
    if !operation_matches {
        return Err(format!(
            "wrong native mutation operation: {:?}",
            program.commands[0].operation
        ));
    }
    let continuation = program
        .continuation
        .as_ref()
        .ok_or_else(|| "result-producing mutation omitted its complete continuation".to_owned())?;
    if continuation.expected_bookmark != fixture.bookmark
        || continuation.expected_graph_revision != fixture.graph.revision()
        || continuation.expected_layout_version != fixture.graph.layout_version()
    {
        return Err(format!(
            "continuation generation fence differs from the fixture: {continuation:#?}"
        ));
    }
    let fingerprint = request
        .mutation_pipeline_fingerprint()
        .map_err(|error| format!("cannot recompute mutation fingerprint: {error}"))?;
    if continuation.fingerprint != fingerprint {
        return Err("continuation fingerprint does not seal the complete request".to_owned());
    }
    if continuation.stage_obligations.len() != continuation.stages.len()
        || continuation.outputs.len() != 1
        || continuation.outputs[0].name != case.expected_column()
        || continuation.maximum_output_cells
            != continuation
                .maximum_output_rows
                .saturating_mul(continuation.outputs.len())
        || continuation.maximum_output_arena_bytes != 0
    {
        return Err(format!(
            "continuation obligations/output capacities are incomplete: {continuation:#?}"
        ));
    }
    let exact_ordered_tail = match (case.tail, continuation.stages.as_slice()) {
        (
            TailShape::LimitZero,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Limit { rows: 0 },
            ],
        )
        | (
            TailShape::SkipAll,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Skip { rows: 1 },
            ],
        )
        | (
            TailShape::PageTwo,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Skip { rows: 2 },
                ResidentMutationPostStage::Limit { rows: 2 },
            ],
        )
        | (
            TailShape::PageAll,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Skip { rows: 0 },
                ResidentMutationPostStage::Limit { rows: 5 },
            ],
        )
        | (
            TailShape::Filter,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::FilterIntegerModuloEquals {
                    divisor: 2,
                    operand: 0,
                    ..
                },
                ResidentMutationPostStage::Project { .. },
            ],
        )
        | (TailShape::ReturnAggregate, [ResidentMutationPostStage::SumInteger { .. }])
        | (
            TailShape::WithAggregate,
            [
                ResidentMutationPostStage::SumInteger { .. },
                ResidentMutationPostStage::Project { .. },
            ],
        ) => true,
        _ => false,
    };
    if !exact_ordered_tail {
        return Err(format!(
            "continuation has the wrong ordered {:?} stages: {:?}",
            case.tail, continuation.stages
        ));
    }
    if request.offset != 0 || request.limit != usize::MAX {
        return Err(format!(
            "legacy request pagination was not neutralized: expected 0/{}, got {}/{}",
            usize::MAX,
            request.offset,
            request.limit
        ));
    }
    Ok(())
}

fn assert_same_publication(
    reference: &ExecutionOutput,
    native: &ExecutionOutput,
) -> std::result::Result<(), String> {
    if native.result != reference.result
        || mutation_debug(native) != mutation_debug(reference)
        || native.dependencies.entities != reference.dependencies.entities
        || native.dependencies.write_targets != reference.dependencies.write_targets
    {
        return Err(format!(
            "native publication differs from generic CPU oracle:\nCPU result: {:#?}\nnative result: {:#?}\nCPU mutations: {:?}\nnative mutations: {:?}",
            reference.result,
            native.result,
            mutation_debug(reference),
            mutation_debug(native)
        ));
    }
    Ok(())
}

fn assert_no_op_fixture(fixture: &Fixture) -> std::result::Result<(), String> {
    let catalog = fixture.graph.catalog();
    if catalog.property("globallyUnknown").is_some()
        || catalog.label("GloballyUnknown").is_some()
        || catalog.property("onlyOnOther") != Some(fixture.name)
    {
        return Err("no-op fixture does not distinguish unknown and existing tokens".to_owned());
    }
    let only_on_other_label = catalog
        .label("OnlyOnOther")
        .ok_or_else(|| "no-op fixture omitted the elsewhere-only label token".to_owned())?;
    let target = fixture
        .graph
        .node(NodeId(1))
        .ok_or_else(|| "no-op fixture omitted the matched N target".to_owned())?;
    let other = fixture
        .graph
        .node(NodeId(2))
        .ok_or_else(|| "no-op fixture omitted the Other node".to_owned())?;
    if target.labels() != [fixture.n_label]
        || target.property(fixture.num) != Some(ScalarValue::Integer(7))
        || target.property(fixture.name).is_some()
        || target.labels().contains(&only_on_other_label)
        || other.property(fixture.name) != Some(ScalarValue::Integer(99))
        || !other.labels().contains(&only_on_other_label)
    {
        return Err(
            "no-op fixture placed an elsewhere-only token on the matched target".to_owned(),
        );
    }
    Ok(())
}

fn assert_no_op_output(
    fixture: &Fixture,
    output: &ExecutionOutput,
) -> std::result::Result<(), String> {
    if output.result.schema != [("num".to_owned(), ColumnType::Integer)]
        || integer_values(output, "num")? != [7]
    {
        return Err(format!(
            "no-op result changed its row/schema: schema={:?}, rows={:?}",
            output.result.schema,
            integer_values(output, "num")?
        ));
    }
    if output.result.statistics != StatementStats::default()
        || !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || !output.dependencies.write_targets.is_empty()
    {
        return Err(format!(
            "no-op published an effect: statistics={:?}, mutations={:?}, temporal={}, targets={:?}",
            output.result.statistics,
            mutation_debug(output),
            output.temporal_mutations.len(),
            output.dependencies.write_targets
        ));
    }
    if output.result.bookmark != fixture.bookmark || output.result.truncated {
        return Err("no-op result changed its bookmark or truncation state".to_owned());
    }
    Ok(())
}

fn assert_no_op_effect_contract(
    request: &ResidentNodePipelineRequest,
    validated: &ValidatedMutationObservation,
    backend: BackendKind,
    attempted_targets_per_command: u64,
) -> std::result::Result<(), String> {
    let program = request
        .mutation
        .as_ref()
        .ok_or_else(|| "effect contract omitted its mutation program".to_owned())?;
    let expected_completion = match backend {
        BackendKind::Cpu => ResidentDeviceCompletion::CpuReference,
        BackendKind::Metal => ResidentDeviceCompletion::Metal,
        BackendKind::Cuda => {
            return Err(format!(
                "effect receipt completion is not exposed for {backend:?}"
            ));
        }
    };
    if validated.effectful_intents != 0 {
        return Err(format!(
            "validated no-op batch exposed {} effectful intent(s)",
            validated.effectful_intents
        ));
    }

    let mut obligations = vec![program.selection_obligation];
    obligations.extend(program.filter_obligations.iter().copied());
    obligations.extend(program.expression_obligations.iter().copied());
    for command in &program.commands {
        obligations.push(command.rhs_obligation);
        obligations.push(command.effect_obligation);
    }
    if let Some(continuation) = &program.continuation {
        obligations.push(continuation.overlay_obligation);
        obligations.extend(continuation.stage_obligations.iter().copied());
        obligations.push(continuation.final_relation_obligation);
    }

    let effect_receipt_count = validated
        .receipts
        .iter()
        .filter(|receipt| receipt.obligation.kind == ResidentObligationKind::MutationEffect)
        .count();
    if effect_receipt_count != program.commands.len() {
        return Err(format!(
            "validated result exposed {effect_receipt_count} effect receipt(s) for {} command(s)",
            program.commands.len()
        ));
    }
    for (index, command) in program.commands.iter().enumerate() {
        let command_index =
            u16::try_from(index).map_err(|_| "effect command index exceeds u16".to_owned())?;
        let expected_scope = ResidentObligationScope::MutationCommand(command_index);
        let effect = command.effect_obligation;
        if effect.kind != ResidentObligationKind::MutationEffect
            || effect.scope != expected_scope
            || effect.scope != command.rhs_obligation.scope
            || obligations
                .iter()
                .filter(|obligation| obligation.id == effect.id)
                .count()
                != 1
        {
            return Err(format!(
                "command {index} has an invalid or non-unique effect obligation: rhs={:?}, effect={effect:?}",
                command.rhs_obligation
            ));
        }
        let receipt_index = 1_usize
            .saturating_add(program.filter_obligations.len())
            .saturating_add(program.expression_obligations.len())
            .saturating_add(index.saturating_mul(2))
            .saturating_add(1);
        let receipt = validated.receipts.get(receipt_index).ok_or_else(|| {
            format!("command {index} omitted its ordered effect receipt at {receipt_index}")
        })?;
        if receipt.obligation != effect
            || receipt.execution != program.execution
            || receipt.input_cardinality != attempted_targets_per_command
            || receipt.output_cardinality != 0
            || receipt.completion != expected_completion
            || validated
                .receipts
                .iter()
                .filter(|candidate| candidate.obligation == effect)
                .count()
                != 1
        {
            return Err(format!(
                "command {index} effect receipt has wrong order/cardinality/provenance: expected obligation={effect:?}, execution={:?}, input={attempted_targets_per_command}, output=0, completion={expected_completion:?}; got {receipt:?}",
                program.execution
            ));
        }
    }
    Ok(())
}

fn execute_native_no_op_adversary(
    fixture: &Fixture,
    backend: &ObservedMutationContinuationBackend,
    case: NoOpCase,
) -> std::result::Result<(), String> {
    let mut failures = Vec::new();
    let reference = match QueryEngine.execute(case.query, &mut context(fixture, None, false)) {
        Ok(output) => {
            if let Err(error) = assert_no_op_output(fixture, &output) {
                failures.push(format!("generic CPU oracle mismatch: {error}"));
            }
            Some(output)
        }
        Err(error) => {
            failures.push(format!(
                "generic CPU oracle failed with {:?}: {error}",
                error.code
            ));
            None
        }
    };
    let reference_is_no_op = reference
        .as_ref()
        .is_some_and(|output| assert_no_op_output(fixture, output).is_ok());

    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.complete_command_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let validated_before = observations
        .validated_mutations
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let native = match QueryEngine.execute(case.query, &mut context(fixture, Some(backend), true)) {
        Ok(output) => Some(output),
        Err(error) => {
            failures.push(format!(
                "native execution failed with {:?}: {error}; pins={}, complete calls={}, rejected routes={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.complete_command_calls.load(Ordering::SeqCst) - calls_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before,
            ));
            None
        }
    };
    if native.is_some()
        && (observations.pins.load(Ordering::SeqCst) != pins_before + 1
            || observations.complete_command_calls.load(Ordering::SeqCst) != calls_before + 1
            || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before)
    {
        failures.push(
            "strict route was not exactly one pin/complete command and zero generic calls"
                .to_owned(),
        );
    }
    let request = native.as_ref().and_then(|_| {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            failures.push("observer did not retain exactly one request".to_owned());
            return None;
        }
        requests.last().cloned()
    });
    let validated = native.as_ref().and_then(|_| {
        let mutations = observations
            .validated_mutations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if mutations.len() != validated_before + 1 {
            failures
                .push("observer did not retain exactly one validated mutation result".to_owned());
            return None;
        }
        mutations.last().cloned()
    });
    if let Some(request) = &request {
        if let Err(error) = request.validate_mutation() {
            failures.push(format!("invalid complete request: {error}"));
        }
        let boundary_matches = request.mutation.as_ref().is_some_and(|program| {
            let operation_matches = program.commands.len() == 1
                && matches!(
                    (case.kind, &program.commands[0].operation),
                    (
                        NoOpMutationKind::RemoveProperty,
                        ResidentMutationOperation::RemoveProperty {
                            target: ResidentEntityBinding::Node(_),
                            ..
                        }
                    ) | (
                        NoOpMutationKind::RemoveLabels,
                        ResidentMutationOperation::RemoveLabels { .. }
                    ) | (
                        NoOpMutationKind::AddLabels,
                        ResidentMutationOperation::AddLabels { .. }
                    )
                );
            operation_matches
                && program.continuation.as_ref().is_some_and(|continuation| {
                    matches!(
                        continuation.stages.as_slice(),
                        [ResidentMutationPostStage::Project { .. }]
                    )
                })
        }) && request.offset == 0
            && request.limit == usize::MAX;
        if !boundary_matches {
            failures.push(format!(
                "no-op escaped the one-command continuation boundary: {request:#?}"
            ));
        }
    }
    if let (Some(request), Some(validated)) = (&request, &validated)
        && let Err(error) = assert_no_op_effect_contract(request, validated, backend.actual_kind, 1)
    {
        failures.push(format!("backend effect contract mismatch: {error}"));
    }
    if let Some(native) = &native {
        let native_is_no_op = match assert_no_op_output(fixture, native) {
            Ok(()) => true,
            Err(error) => {
                failures.push(format!("native output mismatch: {error}"));
                false
            }
        };
        if reference_is_no_op
            && native_is_no_op
            && let Some(reference) = &reference
            && let Err(error) = assert_same_publication(reference, native)
        {
            failures.push(error);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("{}: {}", case.name, failures.join("; ")))
    }
}

fn run_native_no_op_adversaries(
    fixture: &Fixture,
    backend: &ObservedMutationContinuationBackend,
) -> std::result::Result<(), Vec<String>> {
    let mut failures = Vec::new();
    for case in NO_OP_CASES {
        if let Err(error) = execute_native_no_op_adversary(fixture, backend, case) {
            failures.push(error);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn assert_overlay_adversary_output(
    fixture: &Fixture,
    output: &ExecutionOutput,
    expected_rows: &[i64],
) -> std::result::Result<(), String> {
    if integer_values(output, "num")? != expected_rows {
        return Err(format!(
            "overlay rows differ: expected {expected_rows:?}, got {:?}",
            integer_values(output, "num")?
        ));
    }
    if output.result.statistics
        != (StatementStats {
            properties_set: 2,
            ..StatementStats::default()
        })
    {
        return Err(format!(
            "overlay statistics differ: {:?}",
            output.result.statistics
        ));
    }
    if entity_mutations(output).len() != 2
        || node_property_integer_intents(output, NodeId(1), fixture.num) != [1, 2]
        || output.dependencies.write_targets.len() != 1
    {
        return Err(format!(
            "overlay intent/order/target mismatch: mutations={:?}, targets={:?}",
            mutation_debug(output),
            output.dependencies.write_targets
        ));
    }
    Ok(())
}

fn execute_native_overlay_adversary(
    fixture: &Fixture,
    backend: &ObservedMutationContinuationBackend,
    name: &str,
    query: &str,
    expected_rows: &[i64],
    expected_commands: usize,
) -> std::result::Result<(), String> {
    let reference = QueryEngine
        .execute(query, &mut context(fixture, None, false))
        .map_err(|error| {
            format!(
                "{name}: generic CPU oracle failed with {:?}: {error}",
                error.code
            )
        })?;
    assert_overlay_adversary_output(fixture, &reference, expected_rows)
        .map_err(|error| format!("{name}: generic CPU oracle mismatch: {error}"))?;

    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.complete_command_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let native = QueryEngine
        .execute(query, &mut context(fixture, Some(backend), true))
        .map_err(|error| {
            format!(
                "{name}: native execution failed with {:?}: {error}; pins={}, complete calls={}, rejected routes={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.complete_command_calls.load(Ordering::SeqCst) - calls_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before,
            )
        })?;
    if observations.pins.load(Ordering::SeqCst) != pins_before + 1
        || observations.complete_command_calls.load(Ordering::SeqCst) != calls_before + 1
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err(format!(
            "{name}: strict route was not exactly one pin/complete command and zero generic calls"
        ));
    }
    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            return Err(format!(
                "{name}: observer did not retain exactly one request"
            ));
        }
        requests
            .last()
            .cloned()
            .ok_or_else(|| format!("{name}: complete request disappeared"))?
    };
    let program = request
        .mutation
        .as_ref()
        .ok_or_else(|| format!("{name}: complete request omitted its mutation program"))?;
    if program.commands.len() != expected_commands || program.continuation.is_none() {
        return Err(format!(
            "{name}: expected {expected_commands} ordered command(s) plus continuation, got {} / continuation={}",
            program.commands.len(),
            program.continuation.is_some()
        ));
    }
    assert_overlay_adversary_output(fixture, &native, expected_rows)
        .map_err(|error| format!("{name}: native output mismatch: {error}"))?;
    assert_same_publication(&reference, &native).map_err(|error| format!("{name}: {error}"))
}

fn run_native_overlay_adversaries(
    fixture: &Fixture,
    backend: &ObservedMutationContinuationBackend,
) -> std::result::Result<(), Vec<String>> {
    let mut failures = Vec::new();
    for (name, query, expected_rows, expected_commands) in [
        (
            "duplicate target across source rows",
            "MATCH (n:N)-[:R]->() SET n.num = n.num + 1 RETURN n.num AS num",
            &[2, 2][..],
            1,
        ),
        (
            "two same-property commands in one row",
            "MATCH (n:N) SET n.num = n.num + 1, n.num = n.num + 1 RETURN n.num AS num",
            &[2][..],
            2,
        ),
    ] {
        if let Err(error) = execute_native_overlay_adversary(
            fixture,
            backend,
            name,
            query,
            expected_rows,
            expected_commands,
        ) {
            failures.push(error);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn assert_selection_budget_output(
    fixture: &Fixture,
    output: &ExecutionOutput,
    expected_rows: &[i64],
) -> std::result::Result<(), String> {
    let schema_matches = matches!(
        output.result.schema.as_slice(),
        [(name, ColumnType::Integer | ColumnType::Null)] if name == "num"
    ) && (expected_rows.is_empty()
        || output.result.schema[0].1 == ColumnType::Integer);
    if !schema_matches || integer_values(output, "num")? != expected_rows {
        return Err(format!(
            "selection-budget rows/schema differ: expected {expected_rows:?}, got schema={:?}, rows={:?}",
            output.result.schema,
            integer_values(output, "num")?
        ));
    }
    if output.result.statistics
        != (StatementStats {
            properties_set: 5,
            ..StatementStats::default()
        })
    {
        return Err(format!(
            "selection-budget statistics differ: {:?}",
            output.result.statistics
        ));
    }
    let expected_intents = (1_u64..=5)
        .map(|id| (NodeId(id), i64::try_from(id).expect("five IDs fit i64") + 1))
        .collect::<Vec<_>>();
    let actual_intents = node_property_integer_intents_by_target(output, fixture.num);
    if entity_mutations(output).len() != 5
        || actual_intents != expected_intents
        || output.dependencies.write_targets.len() != 5
    {
        return Err(format!(
            "selection budget capped mutation work: expected intents={expected_intents:?}, got intents={actual_intents:?}, targets={:?}",
            output.dependencies.write_targets
        ));
    }
    Ok(())
}

fn execute_native_selection_budget_adversary(
    fixture: &Fixture,
    backend: &ObservedMutationContinuationBackend,
    name: &str,
    query: &str,
    expected_rows: &[i64],
    expected_limit_then_skip: Option<(usize, usize)>,
) -> std::result::Result<(), String> {
    let reference = QueryEngine
        .execute(query, &mut context(fixture, None, false))
        .map_err(|error| {
            format!(
                "{name}: grammar/generic planner rejected the adversary with {:?}: {error}",
                error.code
            )
        })?;
    assert_selection_budget_output(fixture, &reference, expected_rows)
        .map_err(|error| format!("{name}: generic CPU oracle mismatch: {error}"))?;
    let reference_schema = reference.result.schema.clone();

    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.complete_command_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let native = QueryEngine
        .execute(query, &mut context(fixture, Some(backend), true))
        .map_err(|error| {
            format!(
                "{name}: native execution failed with {:?}: {error}; pins={}, complete calls={}, rejected routes={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.complete_command_calls.load(Ordering::SeqCst) - calls_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before,
            )
        })?;
    if observations.pins.load(Ordering::SeqCst) != pins_before + 1
        || observations.complete_command_calls.load(Ordering::SeqCst) != calls_before + 1
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err(format!(
            "{name}: strict route was not exactly one pin/complete command and zero generic calls"
        ));
    }
    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            return Err(format!(
                "{name}: observer did not retain exactly one request"
            ));
        }
        requests
            .last()
            .cloned()
            .ok_or_else(|| format!("{name}: complete request disappeared"))?
    };
    let program = request
        .mutation
        .as_ref()
        .ok_or_else(|| format!("{name}: complete request omitted its mutation program"))?;
    let continuation = program
        .continuation
        .as_ref()
        .ok_or_else(|| format!("{name}: complete request omitted its continuation"))?;
    if program.commands.len() != 1 || program.max_intents < 5 {
        return Err(format!(
            "{name}: selection budget admitted {} command(s) and only {} intent slots",
            program.commands.len(),
            program.max_intents
        ));
    }
    if request.offset != 0 || request.limit != usize::MAX {
        return Err(format!(
            "{name}: legacy request pagination was not neutralized: {}/{}",
            request.offset, request.limit
        ));
    }
    let exact_stages = match (expected_limit_then_skip, continuation.stages.as_slice()) {
        (
            None,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Limit { rows: 0 },
            ],
        ) => true,
        (
            Some((5, 3)),
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Limit { rows: 5 },
                ResidentMutationPostStage::Skip { rows: 3 },
            ],
        ) => true,
        _ => false,
    };
    if !exact_stages {
        return Err(format!(
            "{name}: continuation stages were reordered or omitted: {:?}",
            continuation.stages
        ));
    }
    if native.result.schema != reference_schema {
        return Err(format!(
            "{name}: native schema differs from the generic CPU oracle: expected {reference_schema:?}, got {:?}",
            native.result.schema
        ));
    }
    assert_selection_budget_output(fixture, &native, expected_rows)
        .map_err(|error| format!("{name}: native output mismatch: {error}"))?;
    assert_same_publication(&reference, &native).map_err(|error| format!("{name}: {error}"))
}

fn run_native_selection_budget_adversaries(
    fixture: &Fixture,
    backend: &ObservedMutationContinuationBackend,
) -> std::result::Result<(), Vec<String>> {
    let mut failures = Vec::new();
    for (name, query, expected_rows, limit_then_skip) in [
        (
            "five selected writes survive terminal LIMIT 0",
            "MATCH (n:N) SET n.num = n.num + 1 RETURN n.num AS num LIMIT 0",
            &[][..],
            None,
        ),
        (
            "ordered LIMIT then SKIP",
            "MATCH (n:N) SET n.num = n.num + 1 RETURN n.num AS num LIMIT 5 SKIP 3",
            &[5, 6][..],
            Some((5, 3)),
        ),
    ] {
        if let Err(error) = execute_native_selection_budget_adversary(
            fixture,
            backend,
            name,
            query,
            expected_rows,
            limit_then_skip,
        ) {
            failures.push(error);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn execute_native_case(
    fixture: &Fixture,
    backend: &ObservedMutationContinuationBackend,
    case: TckCase,
) -> std::result::Result<(), String> {
    let reference = QueryEngine
        .execute(case.query, &mut context(fixture, None, false))
        .map_err(|error| format!("generic CPU oracle failed with {:?}: {error}", error.code))?;
    assert_official_output(fixture, case, &reference)?;

    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.complete_command_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let native = QueryEngine
        .execute(case.query, &mut context(fixture, Some(backend), true))
        .map_err(|error| {
            format!(
                "native execution failed with {:?}: {error}; pins={}, complete calls={}, rejected routes={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.complete_command_calls.load(Ordering::SeqCst) - calls_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before,
            )
        })?;

    if observations.pins.load(Ordering::SeqCst) != pins_before + 1 {
        return Err("query did not pin exactly one immutable generation".to_owned());
    }
    if observations.complete_command_calls.load(Ordering::SeqCst) != calls_before + 1 {
        return Err(
            "query did not issue exactly one complete mutation-continuation command".to_owned(),
        );
    }
    if observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before {
        return Err("query entered a generic scan/join/filter/aggregate route".to_owned());
    }
    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            return Err("observer did not retain exactly one complete request".to_owned());
        }
        requests
            .last()
            .cloned()
            .ok_or_else(|| "complete request disappeared".to_owned())?
    };
    assert_native_request(fixture, case, &request)?;
    assert_official_output(fixture, case, &native)?;
    assert_same_publication(&reference, &native)
}

fn run_all_native_cases(
    backend: &mut ObservedMutationContinuationBackend,
) -> std::result::Result<(), Vec<String>> {
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = match Fixture::new(case) {
            Ok(fixture) => fixture,
            Err(error) => {
                failures.push(format!("{}: fixture failed: {error}", case.label()));
                continue;
            }
        };
        if let Err(error) = fixture
            .image()
            .and_then(|image| backend.replace_all_projects(vec![image]))
        {
            failures.push(format!(
                "{}: resident replacement failed: {error}",
                case.label()
            ));
            continue;
        }
        if let Err(error) = execute_native_case(&fixture, backend, case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn assert_no_failures(boundary: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{boundary} had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_is_exactly_remove3_1_through_14_and_set6_1_through_21() {
    let cases = all_cases().collect::<Vec<_>>();
    assert_eq!(cases.len(), 35);
    assert_eq!(
        cases
            .iter()
            .map(|case| case.report_index)
            .collect::<Vec<_>>(),
        (673_usize..=686).chain(855_usize..=875).collect::<Vec<_>>()
    );
    assert_eq!(
        cases
            .iter()
            .map(|case| case.displayed_report_id)
            .collect::<Vec<_>>(),
        (674_usize..=687).chain(856_usize..=876).collect::<Vec<_>>()
    );
    assert_eq!(
        cases[..14]
            .iter()
            .map(|case| case.feature_scenario)
            .collect::<Vec<_>>(),
        (1_u8..=14).collect::<Vec<_>>()
    );
    assert_eq!(
        cases[14..]
            .iter()
            .map(|case| case.feature_scenario)
            .collect::<Vec<_>>(),
        (1_u8..=21).collect::<Vec<_>>()
    );
    assert!(cases[..14].iter().all(|case| {
        case.feature == REMOVE_FEATURE && case.feature_filter == REMOVE_FEATURE_FILTER
    }));
    assert!(
        cases[14..].iter().all(|case| {
            case.feature == SET_FEATURE && case.feature_filter == SET_FEATURE_FILTER
        })
    );
    assert!(cases.iter().all(|case| {
        case.name
            .starts_with(&format!("[{}]", case.feature_scenario))
            && !case.query.is_empty()
            && !case.setup.cypher().is_empty()
    }));
    for family in MutationFamily::ALL {
        assert_eq!(
            cases.iter().filter(|case| case.family == family).count(),
            7,
            "{family:?} did not contribute exactly seven scenarios"
        );
    }
    assert!(!cases.iter().any(|case| case.query.contains("REMOVE r.")));
}

#[test]
fn generic_cpu_oracle_proves_rows_statistics_and_intents_for_exact_35() -> Result<()> {
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = Fixture::new(case)?;
        match QueryEngine.execute(case.query, &mut context(&fixture, None, false)) {
            Ok(output) => {
                if let Err(error) = assert_official_output(&fixture, case, &output) {
                    failures.push(format!("{}: {error}", case.label()));
                }
            }
            Err(error) => failures.push(format!(
                "{}: generic CPU oracle failed with {:?}: {error}",
                case.label(),
                error.code
            )),
        }
    }
    assert_no_failures("generic CPU oracle", failures);
    Ok(())
}

#[test]
fn generic_cpu_oracle_proves_backend_effect_no_ops_have_no_side_effects() -> Result<()> {
    let fixture = Fixture::no_op_effects()?;
    assert_no_op_fixture(&fixture).map_err(|error| Error::internal(error))?;
    let mut failures = Vec::new();
    for case in NO_OP_CASES {
        match QueryEngine.execute(case.query, &mut context(&fixture, None, false)) {
            Ok(output) => {
                if let Err(error) = assert_no_op_output(&fixture, &output) {
                    failures.push(format!("{}: {error}", case.name));
                }
            }
            Err(error) => failures.push(format!(
                "{}: generic CPU failed with {:?}: {error}",
                case.name, error.code
            )),
        }
    }
    assert_no_op_fixture(&fixture).map_err(|error| Error::internal(error))?;
    assert_no_failures("generic CPU backend-effect no-op oracle", failures);
    Ok(())
}

#[test]
fn current_native_route_is_complete_or_fails_closed_without_prefix_dispatch() -> Result<()> {
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = Fixture::new(case)?;
        let backend = ObservedMutationContinuationBackend::strict_cpu_reference(fixture.cpu()?)?;
        let observations = backend.observations();
        match QueryEngine.execute(case.query, &mut context(&fixture, Some(&backend), true)) {
            Ok(output) => {
                if observations.pins.load(Ordering::SeqCst) != 1
                    || observations.complete_command_calls.load(Ordering::SeqCst) != 1
                    || observations.unexpected_query_calls.load(Ordering::SeqCst) != 0
                {
                    failures.push(format!(
                        "{}: successful route did not use exactly one pinned complete command",
                        case.label()
                    ));
                } else if let Err(error) = assert_official_output(&fixture, case, &output) {
                    failures.push(format!(
                        "{}: partial successful publication: {error}",
                        case.label()
                    ));
                }
            }
            Err(error) if error.code == ErrorCode::GpuAdmissionFailure => {
                if observations.pins.load(Ordering::SeqCst) != 0
                    || observations.complete_command_calls.load(Ordering::SeqCst) != 0
                    || observations.unexpected_query_calls.load(Ordering::SeqCst) != 0
                {
                    failures.push(format!(
                        "{}: fail-closed query dispatched a mutation prefix or generic route",
                        case.label()
                    ));
                }
            }
            Err(error) => failures.push(format!(
                "{}: expected complete success or fail-closed admission, got {:?}: {error}",
                case.label(),
                error.code
            )),
        }
    }
    assert_no_failures("monotonic fail-closed boundary", failures);
    Ok(())
}

#[test]
fn adversarial_cpu_oracle_covers_duplicate_targets_read_your_writes_and_empty_tail() -> Result<()> {
    let duplicate = Fixture::duplicate_target()?;
    let duplicate_query =
        "MATCH (n:N)-[:R]->() SET n.num = n.num + 1 WITH n WHERE n.num = 2 RETURN n.num AS num";
    let duplicate_output =
        QueryEngine.execute(duplicate_query, &mut context(&duplicate, None, false))?;
    assert_eq!(integer_values(&duplicate_output, "num"), Ok(vec![2, 2]));
    assert_eq!(
        duplicate_output.result.statistics,
        StatementStats {
            properties_set: 2,
            ..StatementStats::default()
        }
    );
    assert_eq!(entity_mutations(&duplicate_output).len(), 2);
    assert_eq!(duplicate_output.dependencies.write_targets.len(), 1);
    assert_eq!(
        node_property_integer_intents(&duplicate_output, NodeId(1), duplicate.num),
        vec![1, 2],
        "row-major mutation evaluation must carry the first row's write into the second row"
    );

    let within_row = Fixture::duplicate_target()?;
    let within_row_query =
        "MATCH (n:N) SET n.num = n.num + 1, n.num = n.num + 1 RETURN n.num AS num";
    let within_row_output =
        QueryEngine.execute(within_row_query, &mut context(&within_row, None, false))?;
    assert_eq!(integer_values(&within_row_output, "num"), Ok(vec![2]));
    assert_eq!(within_row_output.result.statistics.properties_set, 2);
    assert_eq!(
        node_property_integer_intents(&within_row_output, NodeId(1), within_row.num),
        vec![1, 2],
        "later commands in one row must read writes from earlier commands"
    );
    assert_eq!(within_row_output.dependencies.write_targets.len(), 1);

    let rejecting_case = all_cases()
        .find(|case| {
            case.family == MutationFamily::SetNodeProperty && case.tail == TailShape::Filter
        })
        .ok_or_else(|| Error::internal("SET node-property filter case disappeared"))?;
    let rejecting = Fixture::new(rejecting_case)?;
    let rejecting_query =
        "MATCH (n:N) SET n.num = n.num + 1 WITH n WHERE n.num < 0 RETURN n.num AS num";
    let rejecting_output =
        QueryEngine.execute(rejecting_query, &mut context(&rejecting, None, false))?;
    assert_eq!(integer_values(&rejecting_output, "num"), Ok(Vec::new()));
    assert_eq!(rejecting_output.result.statistics.properties_set, 5);
    assert_eq!(entity_mutations(&rejecting_output).len(), 5);
    assert_eq!(rejecting_output.dependencies.write_targets.len(), 5);

    let cases = all_cases().collect::<Vec<_>>();
    assert!(cases.iter().any(|case| case.tail == TailShape::LimitZero));
    assert!(cases.iter().any(|case| case.tail == TailShape::SkipAll));
    assert!(
        cases
            .iter()
            .any(|case| case.tail == TailShape::ReturnAggregate)
    );
    assert!(cases.iter().any(|case| case.tail.selected_rows() > 1));
    Ok(())
}

#[test]
fn wrong_project_is_rejected_before_any_command() -> Result<()> {
    let case = all_cases()
        .next()
        .ok_or_else(|| Error::internal("mutation-continuation manifest is empty"))?;
    let fixture = Fixture::new(case)?;
    let backend = ObservedMutationContinuationBackend::strict_cpu_reference(fixture.cpu()?)?;
    let error = match backend.pin_project(ProjectId::random()) {
        Ok(_) => return Err(Error::internal("wrong project was pinned")),
        Err(error) => error,
    };
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(backend.observations.pins.load(Ordering::SeqCst), 0);
    assert_eq!(
        backend
            .observations
            .complete_command_calls
            .load(Ordering::SeqCst),
        0
    );
    Ok(())
}

#[test]
#[ignore = "red acceptance gate: all 35 cases require one complete native mutation-continuation command"]
fn strict_cpu_reference_runs_exact_35_through_one_pinned_complete_command() -> Result<()> {
    let first = all_cases()
        .next()
        .ok_or_else(|| Error::internal("mutation-continuation manifest is empty"))?;
    let fixture = Fixture::new(first)?;
    let mut backend = ObservedMutationContinuationBackend::strict_cpu_reference(fixture.cpu()?)?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    if let Err(failures) = run_all_native_cases(&mut backend) {
        assert_no_failures("strict CPU mutation-continuation", failures);
    }
    Ok(())
}

#[test]
fn complete_command_rejects_faults_before_rows_or_mutations() -> Result<()> {
    let case = all_cases()
        .find(|case| {
            case.family == MutationFamily::SetNodeProperty && case.tail == TailShape::PageAll
        })
        .ok_or_else(|| Error::internal("fault-injection case disappeared"))?;

    // Each wrapper gives the inner CPU backend a separately sealed clone. Generation-fence faults
    // belong to backend admission and must fail there; fingerprint/stage changes are valid for the
    // inner clone and must be rejected when its returned frame is checked against the untouched
    // original request.
    for (name, fault, expected_code) in [
        (
            "tampered dispatch bookmark",
            RequestFrameFault::WrongBookmark,
            ErrorCode::GpuAdmissionFailure,
        ),
        (
            "tampered dispatch revision",
            RequestFrameFault::WrongRevision,
            ErrorCode::GpuAdmissionFailure,
        ),
        (
            "tampered dispatch layout",
            RequestFrameFault::WrongLayout,
            ErrorCode::GpuAdmissionFailure,
        ),
        (
            "different returned fingerprint",
            RequestFrameFault::DifferentFingerprint,
            ErrorCode::CorruptStorage,
        ),
        (
            "reordered returned stages",
            RequestFrameFault::ReorderedStages,
            ErrorCode::CorruptStorage,
        ),
        (
            "duplicate mutation-effect obligation identity",
            RequestFrameFault::DuplicateEffectObligationId,
            ErrorCode::QueryType,
        ),
        (
            "wrong mutation-effect obligation kind",
            RequestFrameFault::WrongEffectObligationKind,
            ErrorCode::QueryType,
        ),
        (
            "wrong mutation-effect obligation scope",
            RequestFrameFault::WrongEffectObligationScope,
            ErrorCode::QueryType,
        ),
    ] {
        let fixture = Fixture::new(case)?;
        let backend =
            ObservedMutationContinuationBackend::request_frame_fault(fixture.cpu()?, fault)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(case.query, &mut context(&fixture, Some(&backend), true))
            .err()
            .ok_or_else(|| Error::internal(format!("{name} unexpectedly published")))?;
        assert_eq!(error.code, expected_code, "{name}: {error}");
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1, "{name}");
        assert_eq!(
            observations.complete_command_calls.load(Ordering::SeqCst),
            1,
            "{name}"
        );
        assert_eq!(
            observations.unexpected_query_calls.load(Ordering::SeqCst),
            0,
            "{name}"
        );
        assert_eq!(
            fixture
                .graph
                .node(NodeId(1))
                .and_then(|node| node.property(fixture.num)),
            Some(ScalarValue::Integer(1)),
            "{name} changed canonical state"
        );
    }

    // These are the result faults an integration wrapper can inject without exposing private
    // completion fields, intent effect flags, or raw receipts from the production result type.
    // Result-side effect count/ID/kind/scope/completion corruption remains module-private; the
    // public integration surface exposes only the already validated receipt slice used above.
    for (name, fault, wrong_backend) in [
        ("wrong reported backend", ResultFault::None, true),
        (
            "missing entire mutation result",
            ResultFault::MissingMutationResult,
            false,
        ),
    ] {
        let fixture = Fixture::new(case)?;
        let backend = if wrong_backend {
            ObservedMutationContinuationBackend::wrong_receipt_provenance(fixture.cpu()?)?
        } else {
            ObservedMutationContinuationBackend::result_fault(fixture.cpu()?, fault)?
        };
        let observations = backend.observations();
        let error = QueryEngine
            .execute(case.query, &mut context(&fixture, Some(&backend), true))
            .err()
            .ok_or_else(|| Error::internal(format!("{name} unexpectedly published")))?;
        assert_eq!(error.code, ErrorCode::CorruptStorage, "{name}: {error}");
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1, "{name}");
        assert_eq!(
            observations.complete_command_calls.load(Ordering::SeqCst),
            1,
            "{name}"
        );
        assert_eq!(
            observations.unexpected_query_calls.load(Ordering::SeqCst),
            0,
            "{name}"
        );
        assert_eq!(
            fixture
                .graph
                .node(NodeId(1))
                .and_then(|node| node.property(fixture.num)),
            Some(ScalarValue::Integer(1)),
            "{name} changed canonical state"
        );
    }

    // A stale pinned generation must fail before the complete command can be dispatched.
    for fault in [
        GenerationFault::WrongBookmark,
        GenerationFault::WrongRevision,
    ] {
        let fixture = Fixture::new(case)?;
        let backend = ObservedMutationContinuationBackend::generation_fault(fixture.cpu()?, fault)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(case.query, &mut context(&fixture, Some(&backend), true))
            .err()
            .ok_or_else(|| Error::internal(format!("{fault:?} unexpectedly published")))?;
        assert!(
            matches!(
                error.code,
                ErrorCode::CorruptStorage | ErrorCode::GpuAdmissionFailure
            ),
            "{fault:?}: {error}"
        );
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1, "{fault:?}");
        assert_eq!(
            observations.complete_command_calls.load(Ordering::SeqCst),
            0,
            "{fault:?} dispatched after a bad generation fence"
        );
        assert_eq!(
            observations.unexpected_query_calls.load(Ordering::SeqCst),
            0,
            "{fault:?}"
        );
        assert_eq!(
            fixture
                .graph
                .node(NodeId(1))
                .and_then(|node| node.property(fixture.num)),
            Some(ScalarValue::Integer(1)),
            "{fault:?} changed canonical state"
        );
    }

    // Missing, reordered, or forged private receipts and forged completion fields belong in the
    // module-private mutation receipt tests in `src/gpu/mod.rs`; this integration test deliberately
    // does not require an untrusted-parts constructor on the public result type.
    Ok(())
}

#[test]
fn strict_cpu_overlay_adversaries_use_one_complete_command_each() -> Result<()> {
    let fixture = Fixture::duplicate_target()?;
    let backend = ObservedMutationContinuationBackend::strict_cpu_reference(fixture.cpu()?)?;
    if let Err(failures) = run_native_overlay_adversaries(&fixture, &backend) {
        assert_no_failures("strict CPU mutation-overlay adversaries", failures);
    }
    Ok(())
}

#[test]
fn strict_cpu_selection_budget_and_tail_order_adversaries() -> Result<()> {
    let case = all_cases()
        .find(|case| {
            case.family == MutationFamily::SetNodeProperty && case.tail == TailShape::PageAll
        })
        .ok_or_else(|| Error::internal("five-node SET fixture disappeared"))?;
    let fixture = Fixture::new(case)?;
    let backend = ObservedMutationContinuationBackend::strict_cpu_reference(fixture.cpu()?)?;
    if let Err(failures) = run_native_selection_budget_adversaries(&fixture, &backend) {
        assert_no_failures("strict CPU selection-budget adversaries", failures);
    }
    Ok(())
}

#[test]
fn strict_cpu_backend_effect_no_op_adversaries() -> Result<()> {
    let fixture = Fixture::no_op_effects()?;
    assert_no_op_fixture(&fixture).map_err(|error| Error::internal(error))?;
    let backend = ObservedMutationContinuationBackend::strict_cpu_reference(fixture.cpu()?)?;
    if let Err(failures) = run_native_no_op_adversaries(&fixture, &backend) {
        assert_no_failures("strict CPU backend-effect no-op adversaries", failures);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "red acceptance gate: exact 35 plus overlay/budget/no-op adversaries must match CPU on real Metal"]
fn real_metal_matches_cpu_for_exact_35_and_adversaries_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let first = all_cases()
        .next()
        .ok_or_else(|| Error::internal("mutation-continuation manifest is empty"))?;
    let fixture = Fixture::new(first)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(fixture.image()?)?;
    let mut backend = ObservedMutationContinuationBackend::real_metal(metal)?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Metal);
    let mut failures = run_all_native_cases(&mut backend).err().unwrap_or_default();

    let duplicate = Fixture::duplicate_target()?;
    match duplicate
        .image()
        .and_then(|image| backend.replace_all_projects(vec![image]))
    {
        Ok(()) => {
            if let Err(mut overlay_failures) = run_native_overlay_adversaries(&duplicate, &backend)
            {
                failures.append(&mut overlay_failures);
            }
        }
        Err(error) => failures.push(format!(
            "overlay adversary resident replacement failed: {error}"
        )),
    }

    let budget_case = all_cases()
        .find(|case| {
            case.family == MutationFamily::SetNodeProperty && case.tail == TailShape::PageAll
        })
        .ok_or_else(|| Error::internal("five-node SET fixture disappeared"))?;
    let budget = Fixture::new(budget_case)?;
    match budget
        .image()
        .and_then(|image| backend.replace_all_projects(vec![image]))
    {
        Ok(()) => {
            if let Err(mut budget_failures) =
                run_native_selection_budget_adversaries(&budget, &backend)
            {
                failures.append(&mut budget_failures);
            }
        }
        Err(error) => failures.push(format!(
            "selection-budget adversary resident replacement failed: {error}"
        )),
    }

    let no_op = Fixture::no_op_effects()?;
    match no_op
        .image()
        .and_then(|image| backend.replace_all_projects(vec![image]))
    {
        Ok(()) => {
            if let Err(mut no_op_failures) = run_native_no_op_adversaries(&no_op, &backend) {
                failures.append(&mut no_op_failures);
            }
        }
        Err(error) => failures.push(format!(
            "backend-effect no-op adversary resident replacement failed: {error}"
        )),
    }
    assert_no_failures("real Metal mutation-continuation acceptance", failures);
    Ok(())
}
