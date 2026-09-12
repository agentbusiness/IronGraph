// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::BTreeSet;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::{Mutex, MutexGuard};

use irongraph::gpu::{
    BackendKind, CpuBackend, ExecutionBackend, ResidentDeviceCompletion, ResidentExecutionId,
    ResidentExecutionObligation, ResidentObligationKind, ResidentObligationScope,
    ResidentProjectImage, ResidentSegmentedAggregate, ResidentSegmentedAggregateKind,
    ResidentSegmentedAggregationRequest, ResidentSegmentedCell, ResidentSegmentedRelation,
    ResidentSegmentedValueTag, ValidatedResidentSegmentedAggregation,
};
use irongraph::graph::{GraphStore, IndexCatalog, TemporalStore};
use irongraph::{Bookmark, Error, ErrorCode, ProjectId, Result};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const BOOKMARK: Bookmark = Bookmark { term: 73, index: 0 };
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const CERTIFIED_TCK_REPORT: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";

fn assert_certified_report_identities(
    identities: impl IntoIterator<Item = (usize, String, String)>,
) {
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(CERTIFIED_TCK_REPORT).expect("certified TCK report is readable"),
    )
    .expect("certified TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_184));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("certified report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);
    let mut selected = BTreeSet::new();
    for (stored_id, feature, expanded_name) in identities {
        assert!(
            selected.insert((feature.clone(), expanded_name.clone())),
            "duplicate local TCK identity ({feature}, {expanded_name})"
        );
        let matches = scenarios
            .iter()
            .enumerate()
            .filter_map(|(index, scenario)| {
                let path = scenario.get("path")?.as_str()?;
                let name = scenario.get("name")?.as_str()?;
                (path.ends_with(&feature) && name == expanded_name).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "({feature}, {expanded_name}) resolved to {matches:?}"
        );
        assert_eq!(
            stored_id, matches[0],
            "wrong report index for {expanded_name}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct OfficialCase {
    report_id: u16,
    feature: &'static str,
    scenario: u8,
    title: &'static str,
    query: &'static str,
}

impl OfficialCase {
    fn label(self) -> String {
        format!(
            "TCK {} {} [{}] {}",
            self.report_id, self.feature, self.scenario, self.title
        )
    }
}

const OFFICIAL_CASES: [OfficialCase; 43] = [
    OfficialCase {
        report_id: 705,
        feature: "features/clauses/return/Return2.feature",
        scenario: 10,
        title: "Return count aggregation over an empty graph",
        query: "MATCH (a) RETURN count(a) > 0",
    },
    OfficialCase {
        report_id: 723,
        feature: "features/clauses/return/Return4.feature",
        scenario: 7,
        title: "Keeping used expression 4",
        query: "MATCH p = (n)-->(b) RETURN aVg(n.aGe)",
    },
    OfficialCase {
        report_id: 734,
        feature: "features/clauses/return/Return6.feature",
        scenario: 2,
        title: "Projecting an arithmetic expression with aggregation",
        query: "MATCH (a) RETURN a, count(a) + 3",
    },
    OfficialCase {
        report_id: 736,
        feature: "features/clauses/return/Return6.feature",
        scenario: 4,
        title: "Support multiple divisions in aggregate function",
        query: "MATCH (n) RETURN count(n) / 60 / 60 AS count",
    },
    OfficialCase {
        report_id: 737,
        feature: "features/clauses/return/Return6.feature",
        scenario: 5,
        title: "Aggregates inside normal functions",
        query: "MATCH (a) RETURN size(collect(a))",
    },
    OfficialCase {
        report_id: 741,
        feature: "features/clauses/return/Return6.feature",
        scenario: 9,
        title: "Aggregates with arithmetics",
        query: "MATCH () RETURN count(*) * 10 AS c",
    },
    OfficialCase {
        report_id: 742,
        feature: "features/clauses/return/Return6.feature",
        scenario: 10,
        title: "Multiple aggregates on same variable",
        query: "MATCH (n) RETURN count(n), collect(n)",
    },
    OfficialCase {
        report_id: 744,
        feature: "features/clauses/return/Return6.feature",
        scenario: 12,
        title: "Counting matches per group",
        query: "MATCH (a:L)-[rel]->(b) RETURN a, count(*)",
    },
    OfficialCase {
        report_id: 749,
        feature: "features/clauses/return/Return6.feature",
        scenario: 17,
        title: "Handle constants and parameters inside an expression which contains an aggregation expression",
        query: "MATCH (person) RETURN $age + avg(person.age) - 1000",
    },
    OfficialCase {
        report_id: 750,
        feature: "features/clauses/return/Return6.feature",
        scenario: 18,
        title: "Handle returned variables inside an expression which contains an aggregation expression",
        query: "MATCH (me:Person)--(you:Person) RETURN me.age, me.age + count(you.age)",
    },
    OfficialCase {
        report_id: 751,
        feature: "features/clauses/return/Return6.feature",
        scenario: 19,
        title: "Handle returned property accesses inside an expression which contains an aggregation expression",
        query: "MATCH (me:Person)--(you:Person) RETURN me.age, me.age + count(you.age)",
    },
    OfficialCase {
        report_id: 771,
        feature: "features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario: 3,
        title: "Sort on aggregated function",
        query: "MATCH (n) RETURN n.division, max(n.age) ORDER BY max(n.age)",
    },
    OfficialCase {
        report_id: 774,
        feature: "features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario: 6,
        title: "Count star should count everything in scope",
        query: "MATCH (a) RETURN a, count(*) ORDER BY count(*)",
    },
    OfficialCase {
        report_id: 775,
        feature: "features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario: 7,
        title: "Ordering with aggregation",
        query: "MATCH (n) RETURN n.name, count(*) AS foo ORDER BY n.name",
    },
    OfficialCase {
        report_id: 779,
        feature: "features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario: 11,
        title: "Aggregates ordered by arithmetics",
        query: "MATCH (a:A), (b:X) RETURN count(a) * 10 + count(b) * 5 AS x",
    },
    OfficialCase {
        report_id: 783,
        feature: "features/clauses/return-orderby/ReturnOrderBy3.feature",
        scenario: 1,
        title: "Sort on aggregate function and normal property",
        query: "MATCH (n) RETURN n.division, count(*) ORDER BY count(*) DESC, n.division",
    },
    OfficialCase {
        report_id: 784,
        feature: "features/clauses/return-orderby/ReturnOrderBy4.feature",
        scenario: 1,
        title: "ORDER BY of a column introduced in RETURN should return salient results in ascending order",
        query: "UNWIND prows AS p UNWIND qrows[p] AS q WITH p, count(q) AS rng RETURN p",
    },
    OfficialCase {
        report_id: 787,
        feature: "features/clauses/return-orderby/ReturnOrderBy6.feature",
        scenario: 1,
        title: "Handle constants and parameters inside an order by item which contains an aggregation expression",
        query: "MATCH (person) RETURN avg(person.age) AS avgAge ORDER BY $age + avg(person.age)",
    },
    OfficialCase {
        report_id: 788,
        feature: "features/clauses/return-orderby/ReturnOrderBy6.feature",
        scenario: 2,
        title: "Handle returned aliases inside an order by item which contains an aggregation expression",
        query: "MATCH (me:Person)--(you:Person) RETURN me.age AS age, count(you.age) AS cnt",
    },
    OfficialCase {
        report_id: 789,
        feature: "features/clauses/return-orderby/ReturnOrderBy6.feature",
        scenario: 3,
        title: "Handle returned property accesses inside an order by item which contains an aggregation expression",
        query: "MATCH (me:Person)--(you:Person) RETURN me.age AS age, count(you.age) AS cnt",
    },
    OfficialCase {
        report_id: 920,
        feature: "features/clauses/with/With6.feature",
        scenario: 1,
        title: "Implicit grouping with single expression as grouping key and single aggregation",
        query: "MATCH (a) WITH a.name AS name, count(*) AS relCount RETURN name, relCount",
    },
    OfficialCase {
        report_id: 1247,
        feature: "features/clauses/with-where/WithWhere6.feature",
        scenario: 1,
        title: "Filter a single aggregate",
        query: "MATCH (a)-->() WITH a, count(*) AS relCount WHERE relCount > 1 RETURN a",
    },
    OfficialCase {
        report_id: 1251,
        feature: "features/expressions/aggregation/Aggregation1.feature",
        scenario: 1,
        title: "Count only non-null values",
        query: "MATCH (n) RETURN n.name, count(n.num)",
    },
    OfficialCase {
        report_id: 1253,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 1,
        title: "`max()` over integers",
        query: "UNWIND [1, 2, 0, null, -1] AS x RETURN max(x)",
    },
    OfficialCase {
        report_id: 1254,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 2,
        title: "`min()` over integers",
        query: "UNWIND [1, 2, 0, null, -1] AS x RETURN min(x)",
    },
    OfficialCase {
        report_id: 1255,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 3,
        title: "`max()` over floats",
        query: "UNWIND [1.0, 2.0, 0.5, null] AS x RETURN max(x)",
    },
    OfficialCase {
        report_id: 1256,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 4,
        title: "`min()` over floats",
        query: "UNWIND [1.0, 2.0, 0.5, null] AS x RETURN min(x)",
    },
    OfficialCase {
        report_id: 1257,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 5,
        title: "`max()` over mixed numeric values",
        query: "UNWIND [1, 2.0, 5, null, 3.2, 0.1] AS x RETURN max(x)",
    },
    OfficialCase {
        report_id: 1258,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 6,
        title: "`min()` over mixed numeric values",
        query: "UNWIND [1, 2.0, 5, null, 3.2, 0.1] AS x RETURN min(x)",
    },
    OfficialCase {
        report_id: 1259,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 7,
        title: "`max()` over strings",
        query: "UNWIND ['a', 'b', 'B', null, 'abc', 'abc1'] AS i RETURN max(i)",
    },
    OfficialCase {
        report_id: 1260,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 8,
        title: "`min()` over strings",
        query: "UNWIND ['a', 'b', 'B', null, 'abc', 'abc1'] AS i RETURN min(i)",
    },
    OfficialCase {
        report_id: 1261,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 9,
        title: "`max()` over lists",
        query: "UNWIND [[1], [2], [2, 1]] AS x RETURN max(x)",
    },
    OfficialCase {
        report_id: 1262,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 10,
        title: "`min()` over lists",
        query: "UNWIND [[1], [2], [2, 1]] AS x RETURN min(x)",
    },
    OfficialCase {
        report_id: 1263,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 11,
        title: "`max()` over mixed values",
        query: "UNWIND [1, 'a', null, [1, 2], 0.2, 'b'] AS x RETURN max(x)",
    },
    OfficialCase {
        report_id: 1264,
        feature: "features/expressions/aggregation/Aggregation2.feature",
        scenario: 12,
        title: "`min()` over mixed values",
        query: "UNWIND [1, 'a', null, [1, 2], 0.2, 'b'] AS x RETURN min(x)",
    },
    OfficialCase {
        report_id: 1265,
        feature: "features/expressions/aggregation/Aggregation3.feature",
        scenario: 1,
        title: "Sum only non-null values",
        query: "MATCH (n) RETURN n.name, sum(n.num)",
    },
    OfficialCase {
        report_id: 1266,
        feature: "features/expressions/aggregation/Aggregation3.feature",
        scenario: 2,
        title: "No overflow during summation",
        query: "UNWIND range(1000000, 2000000) AS i WITH i LIMIT 3000 RETURN sum(i)",
    },
    OfficialCase {
        report_id: 1267,
        feature: "features/expressions/aggregation/Aggregation5.feature",
        scenario: 1,
        title: "`collect()` filtering nulls",
        query: "MATCH (n) OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) RETURN n, collect(x)",
    },
    OfficialCase {
        report_id: 1268,
        feature: "features/expressions/aggregation/Aggregation5.feature",
        scenario: 2,
        title: "OPTIONAL MATCH and `collect()` on node property",
        query: "OPTIONAL MATCH (f:DoesExist) OPTIONAL MATCH (n:DoesNotExist) RETURN collect(DISTINCT n.num), collect(DISTINCT f.num)",
    },
    OfficialCase {
        report_id: 1282,
        feature: "features/expressions/aggregation/Aggregation8.feature",
        scenario: 1,
        title: "Distinct on unbound node",
        query: "OPTIONAL MATCH (a) RETURN count(DISTINCT a)",
    },
    OfficialCase {
        report_id: 1283,
        feature: "features/expressions/aggregation/Aggregation8.feature",
        scenario: 2,
        title: "Distinct on null",
        query: "MATCH (a) RETURN count(DISTINCT a.name)",
    },
    OfficialCase {
        report_id: 1284,
        feature: "features/expressions/aggregation/Aggregation8.feature",
        scenario: 3,
        title: "Collect distinct nulls",
        query: "UNWIND [null, null] AS x RETURN collect(DISTINCT x)",
    },
    OfficialCase {
        report_id: 1285,
        feature: "features/expressions/aggregation/Aggregation8.feature",
        scenario: 4,
        title: "Collect distinct values mixed with nulls",
        query: "UNWIND [null, 1, null] AS x RETURN collect(DISTINCT x)",
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
enum OracleValue {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(u64),
    String(String),
    Bytes(Vec<u8>),
    Map(Vec<u8>),
    Node(u64),
    Relationship(u64),
    List(Vec<OracleValue>),
}

fn null() -> OracleValue {
    OracleValue::Null
}

fn integer(value: i64) -> OracleValue {
    OracleValue::Integer(value)
}

fn float(value: f64) -> OracleValue {
    assert!(value.is_finite());
    OracleValue::Float(value.to_bits())
}

fn string(value: &str) -> OracleValue {
    OracleValue::String(value.to_owned())
}

fn node(value: u64) -> OracleValue {
    OracleValue::Node(value)
}

fn list(values: Vec<OracleValue>) -> OracleValue {
    OracleValue::List(values)
}

#[derive(Clone, Debug)]
struct SemanticFixture {
    input_columns: u16,
    rows: Vec<Vec<OracleValue>>,
    grouping_columns: Vec<u16>,
    aggregates: Vec<ResidentSegmentedAggregate>,
    expected: Vec<Vec<OracleValue>>,
}

fn aggregate(
    kind: ResidentSegmentedAggregateKind,
    input_column: Option<u16>,
    distinct: bool,
) -> ResidentSegmentedAggregate {
    ResidentSegmentedAggregate {
        kind,
        input_column,
        distinct,
        percentile: None,
    }
}

fn count_all() -> ResidentSegmentedAggregate {
    aggregate(ResidentSegmentedAggregateKind::CountAll, None, false)
}

fn count(column: u16, distinct: bool) -> ResidentSegmentedAggregate {
    aggregate(
        ResidentSegmentedAggregateKind::CountValue,
        Some(column),
        distinct,
    )
}

fn value_aggregate(
    kind: ResidentSegmentedAggregateKind,
    column: u16,
) -> ResidentSegmentedAggregate {
    aggregate(kind, Some(column), false)
}

fn collect(column: u16, distinct: bool) -> ResidentSegmentedAggregate {
    aggregate(
        ResidentSegmentedAggregateKind::Collect,
        Some(column),
        distinct,
    )
}

fn semantic_fixture(report_id: u16) -> SemanticFixture {
    use ResidentSegmentedAggregateKind::{Average, Maximum, Minimum, Sum};
    match report_id {
        705 => SemanticFixture {
            input_columns: 1,
            rows: vec![],
            grouping_columns: vec![],
            aggregates: vec![count(0, false)],
            expected: vec![vec![integer(0)]],
        },
        723 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![integer(33)], vec![null()], vec![integer(42)]],
            grouping_columns: vec![],
            aggregates: vec![value_aggregate(Average, 0)],
            expected: vec![vec![float(37.5)]],
        },
        734 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![node(42)]],
            grouping_columns: vec![0],
            aggregates: vec![count(0, false)],
            expected: vec![vec![node(42), integer(1)]],
        },
        736 => {
            let rows = (0..=7_250).map(|id| vec![node(id)]).collect::<Vec<_>>();
            SemanticFixture {
                input_columns: 1,
                rows,
                grouping_columns: vec![],
                aggregates: vec![count(0, false)],
                expected: vec![vec![integer(7_251)]],
            }
        }
        737 => {
            let values = (0..=10).map(node).collect::<Vec<_>>();
            SemanticFixture {
                input_columns: 1,
                rows: values.iter().cloned().map(|value| vec![value]).collect(),
                grouping_columns: vec![],
                aggregates: vec![collect(0, false)],
                expected: vec![vec![list(values)]],
            }
        }
        741 => SemanticFixture {
            input_columns: 0,
            rows: vec![vec![]],
            grouping_columns: vec![],
            aggregates: vec![count_all()],
            expected: vec![vec![integer(1)]],
        },
        742 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![node(1)]],
            grouping_columns: vec![],
            aggregates: vec![count(0, false), collect(0, false)],
            expected: vec![vec![integer(1), list(vec![node(1)])]],
        },
        744 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![node(1)], vec![node(1)]],
            grouping_columns: vec![0],
            aggregates: vec![count_all()],
            expected: vec![vec![node(1), integer(2)]],
        },
        749 | 787 => SemanticFixture {
            input_columns: 1,
            rows: vec![],
            grouping_columns: vec![],
            aggregates: vec![value_aggregate(Average, 0)],
            expected: vec![vec![null()]],
        },
        750 | 751 | 788 | 789 => SemanticFixture {
            input_columns: 2,
            rows: vec![],
            grouping_columns: vec![0],
            aggregates: vec![count(1, false)],
            expected: vec![],
        },
        771 => SemanticFixture {
            input_columns: 2,
            rows: vec![
                vec![string("A"), integer(22)],
                vec![string("B"), integer(33)],
                vec![string("B"), integer(44)],
                vec![string("C"), integer(55)],
            ],
            grouping_columns: vec![0],
            aggregates: vec![value_aggregate(Maximum, 1)],
            expected: vec![
                vec![string("A"), integer(22)],
                vec![string("B"), integer(44)],
                vec![string("C"), integer(55)],
            ],
        },
        774 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![node(1)], vec![node(2)], vec![node(3)]],
            grouping_columns: vec![0],
            aggregates: vec![count_all()],
            expected: vec![
                vec![node(1), integer(1)],
                vec![node(2), integer(1)],
                vec![node(3), integer(1)],
            ],
        },
        775 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![string("nisse")]],
            grouping_columns: vec![0],
            aggregates: vec![count_all()],
            expected: vec![vec![string("nisse"), integer(1)]],
        },
        779 => SemanticFixture {
            input_columns: 2,
            rows: vec![vec![node(1), node(2)], vec![node(1), node(3)]],
            grouping_columns: vec![],
            aggregates: vec![count(0, false), count(1, false)],
            expected: vec![vec![integer(2), integer(2)]],
        },
        783 => SemanticFixture {
            input_columns: 1,
            rows: vec![
                vec![string("A")],
                vec![string("A")],
                vec![string("B")],
                vec![string("C")],
            ],
            grouping_columns: vec![0],
            aggregates: vec![count_all()],
            expected: vec![
                vec![string("A"), integer(2)],
                vec![string("B"), integer(1)],
                vec![string("C"), integer(1)],
            ],
        },
        784 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![integer(0)], vec![integer(1)], vec![integer(1)]],
            grouping_columns: vec![0],
            aggregates: vec![count_all()],
            expected: vec![vec![integer(0), integer(1)], vec![integer(1), integer(2)]],
        },
        920 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![string("A")], vec![string("A")], vec![string("B")]],
            grouping_columns: vec![0],
            aggregates: vec![count_all()],
            expected: vec![vec![string("A"), integer(2)], vec![string("B"), integer(1)]],
        },
        1247 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![node(1)], vec![node(1)], vec![node(2)]],
            grouping_columns: vec![0],
            aggregates: vec![count_all()],
            expected: vec![vec![node(1), integer(2)], vec![node(2), integer(1)]],
        },
        1251 => SemanticFixture {
            input_columns: 2,
            rows: vec![
                vec![string("a"), integer(33)],
                vec![string("a"), null()],
                vec![string("b"), integer(42)],
            ],
            grouping_columns: vec![0],
            aggregates: vec![count(1, false)],
            expected: vec![vec![string("a"), integer(1)], vec![string("b"), integer(1)]],
        },
        1253 | 1254 => SemanticFixture {
            input_columns: 1,
            rows: vec![
                vec![integer(1)],
                vec![integer(2)],
                vec![integer(0)],
                vec![null()],
                vec![integer(-1)],
            ],
            grouping_columns: vec![],
            aggregates: vec![value_aggregate(
                if report_id == 1253 { Maximum } else { Minimum },
                0,
            )],
            expected: vec![vec![integer(if report_id == 1253 { 2 } else { -1 })]],
        },
        1255 | 1256 => SemanticFixture {
            input_columns: 1,
            rows: vec![
                vec![float(1.0)],
                vec![float(2.0)],
                vec![float(0.5)],
                vec![null()],
            ],
            grouping_columns: vec![],
            aggregates: vec![value_aggregate(
                if report_id == 1255 { Maximum } else { Minimum },
                0,
            )],
            expected: vec![vec![float(if report_id == 1255 { 2.0 } else { 0.5 })]],
        },
        1257 | 1258 => SemanticFixture {
            input_columns: 1,
            rows: vec![
                vec![integer(1)],
                vec![float(2.0)],
                vec![integer(5)],
                vec![null()],
                vec![float(3.2)],
                vec![float(0.1)],
            ],
            grouping_columns: vec![],
            aggregates: vec![value_aggregate(
                if report_id == 1257 { Maximum } else { Minimum },
                0,
            )],
            expected: vec![vec![if report_id == 1257 {
                integer(5)
            } else {
                float(0.1)
            }]],
        },
        1259 | 1260 => SemanticFixture {
            input_columns: 1,
            rows: ["a", "b", "B", "abc", "abc1"]
                .into_iter()
                .map(|value| vec![string(value)])
                .chain(std::iter::once(vec![null()]))
                .collect(),
            grouping_columns: vec![],
            aggregates: vec![value_aggregate(
                if report_id == 1259 { Maximum } else { Minimum },
                0,
            )],
            expected: vec![vec![string(if report_id == 1259 { "b" } else { "B" })]],
        },
        1261 | 1262 => SemanticFixture {
            input_columns: 1,
            rows: vec![
                vec![list(vec![integer(1)])],
                vec![list(vec![integer(2)])],
                vec![list(vec![integer(2), integer(1)])],
            ],
            grouping_columns: vec![],
            aggregates: vec![value_aggregate(
                if report_id == 1261 { Maximum } else { Minimum },
                0,
            )],
            expected: vec![vec![if report_id == 1261 {
                list(vec![integer(2), integer(1)])
            } else {
                list(vec![integer(1)])
            }]],
        },
        1263 | 1264 => SemanticFixture {
            input_columns: 1,
            rows: vec![
                vec![integer(1)],
                vec![string("a")],
                vec![null()],
                vec![list(vec![integer(1), integer(2)])],
                vec![float(0.2)],
                vec![string("b")],
            ],
            grouping_columns: vec![],
            aggregates: vec![value_aggregate(
                if report_id == 1263 { Maximum } else { Minimum },
                0,
            )],
            expected: vec![vec![if report_id == 1263 {
                integer(1)
            } else {
                list(vec![integer(1), integer(2)])
            }]],
        },
        1265 => SemanticFixture {
            input_columns: 2,
            rows: vec![
                vec![string("a"), integer(33)],
                vec![string("a"), null()],
                vec![string("a"), integer(42)],
            ],
            grouping_columns: vec![0],
            aggregates: vec![value_aggregate(Sum, 1)],
            expected: vec![vec![string("a"), integer(75)]],
        },
        1266 => SemanticFixture {
            input_columns: 1,
            rows: (1_000_000..1_003_000)
                .map(|value| vec![integer(value)])
                .collect(),
            grouping_columns: vec![],
            aggregates: vec![value_aggregate(Sum, 0)],
            expected: vec![vec![integer(3_004_498_500)]],
        },
        1267 => SemanticFixture {
            input_columns: 2,
            rows: vec![vec![node(1), null()]],
            grouping_columns: vec![0],
            aggregates: vec![collect(1, false)],
            expected: vec![vec![node(1), list(vec![])]],
        },
        1268 => SemanticFixture {
            input_columns: 2,
            rows: vec![
                vec![null(), integer(42)],
                vec![null(), integer(43)],
                vec![null(), integer(44)],
            ],
            grouping_columns: vec![],
            aggregates: vec![collect(0, true), collect(1, true)],
            expected: vec![vec![
                list(vec![]),
                list(vec![integer(42), integer(43), integer(44)]),
            ]],
        },
        1282 | 1283 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![null()]],
            grouping_columns: vec![],
            aggregates: vec![count(0, true)],
            expected: vec![vec![integer(0)]],
        },
        1284 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![null()], vec![null()]],
            grouping_columns: vec![],
            aggregates: vec![collect(0, true)],
            expected: vec![vec![list(vec![])]],
        },
        1285 => SemanticFixture {
            input_columns: 1,
            rows: vec![vec![null()], vec![integer(1)], vec![null()]],
            grouping_columns: vec![],
            aggregates: vec![collect(0, true)],
            expected: vec![vec![list(vec![integer(1)])]],
        },
        _ => panic!("missing semantic fixture for official TCK report ID {report_id}"),
    }
}

fn encode_oracle_non_list(
    value: &OracleValue,
    arena: &mut Vec<u8>,
) -> Result<Option<ResidentSegmentedCell>> {
    Ok(Some(match value {
        OracleValue::Null => ResidentSegmentedCell::null(),
        OracleValue::Boolean(value) => ResidentSegmentedCell::boolean(*value),
        OracleValue::Integer(value) => ResidentSegmentedCell::integer(*value),
        OracleValue::Float(bits) => ResidentSegmentedCell::float(f64::from_bits(*bits))?,
        OracleValue::String(value) => {
            let start = arena.len();
            arena.extend_from_slice(value.as_bytes());
            ResidentSegmentedCell {
                tag: ResidentSegmentedValueTag::String,
                payload: start as u64,
                auxiliary: value.len() as u64,
            }
        }
        OracleValue::Bytes(value) => {
            let start = arena.len();
            arena.extend_from_slice(value);
            ResidentSegmentedCell {
                tag: ResidentSegmentedValueTag::Bytes,
                payload: start as u64,
                auxiliary: value.len() as u64,
            }
        }
        OracleValue::Map(value) => {
            let start = arena.len();
            arena.extend_from_slice(value);
            ResidentSegmentedCell {
                tag: ResidentSegmentedValueTag::Map,
                payload: start as u64,
                auxiliary: value.len() as u64,
            }
        }
        OracleValue::Node(id) => ResidentSegmentedCell::node(*id),
        OracleValue::Relationship(id) => ResidentSegmentedCell::relationship(*id),
        OracleValue::List(_) => return Ok(None),
    }))
}

fn encode_relation(fixture: &SemanticFixture) -> Result<ResidentSegmentedRelation> {
    let top_level_cells = fixture
        .rows
        .len()
        .checked_mul(fixture.input_columns as usize)
        .ok_or_else(|| Error::internal("independent oracle top-level cell count overflow"))?;
    let mut cells = Vec::with_capacity(top_level_cells);
    let mut arena = Vec::new();
    let mut suffixes = Vec::<(usize, Vec<ResidentSegmentedCell>)>::new();
    for row in &fixture.rows {
        if row.len() != fixture.input_columns as usize {
            return Err(Error::internal(
                "independent oracle fixture is not rectangular",
            ));
        }
        for value in row {
            if let OracleValue::List(values) = value {
                let mut suffix = Vec::with_capacity(values.len());
                for value in values {
                    let Some(child) = encode_oracle_non_list(value, &mut arena)? else {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "strict segmented flat-list fixture cannot contain a nested LIST",
                        ));
                    };
                    if !matches!(
                        child.tag,
                        ResidentSegmentedValueTag::Boolean
                            | ResidentSegmentedValueTag::Integer
                            | ResidentSegmentedValueTag::Float
                            | ResidentSegmentedValueTag::String
                            | ResidentSegmentedValueTag::Bytes
                    ) {
                        return Err(Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "strict segmented flat-list fixture requires non-null scalar children",
                        ));
                    }
                    suffix.push(child);
                }
                let index = cells.len();
                cells.push(ResidentSegmentedCell {
                    tag: ResidentSegmentedValueTag::List,
                    payload: 0,
                    auxiliary: suffix.len() as u64,
                });
                suffixes.push((index, suffix));
            } else {
                let cell = encode_oracle_non_list(value, &mut arena)?.ok_or_else(|| {
                    Error::internal("independent oracle lost a non-list input cell")
                })?;
                cells.push(cell);
            }
        }
    }
    if cells.len() != top_level_cells {
        return Err(Error::internal(
            "independent oracle changed the exact row-major top-level shape",
        ));
    }
    let has_lists = !suffixes.is_empty();
    for (index, suffix) in suffixes {
        let start = cells.len();
        cells[index].payload = start as u64;
        cells.extend(suffix);
    }
    let relation = ResidentSegmentedRelation {
        row_count: u32::try_from(fixture.rows.len())
            .map_err(|_| Error::internal("oracle row count exceeds u32"))?,
        column_count: fixture.input_columns,
        cells,
        arena,
    };
    if has_lists {
        relation.validate_output()?;
    } else {
        relation.validate_input()?;
    }
    Ok(relation)
}

fn obligation(id: u64, scope: ResidentObligationScope) -> ResidentExecutionObligation {
    ResidentExecutionObligation {
        id,
        kind: ResidentObligationKind::Aggregate,
        scope,
    }
}

fn request_for(
    report_id: u16,
    fixture: &Fixture,
    semantic: &SemanticFixture,
) -> Result<ResidentSegmentedAggregationRequest> {
    let input = encode_relation(semantic)?;
    let maximum_groups = if semantic.grouping_columns.is_empty() {
        1
    } else {
        semantic.rows.len().max(1)
    };
    let output_columns = semantic
        .grouping_columns
        .len()
        .checked_add(semantic.aggregates.len())
        .ok_or_else(|| Error::internal("oracle output column count overflow"))?;
    let collect_count = semantic
        .aggregates
        .iter()
        .filter(|aggregate| aggregate.kind == ResidentSegmentedAggregateKind::Collect)
        .count();
    let maximum_list_items = semantic
        .rows
        .iter()
        .flatten()
        .filter_map(|value| match value {
            OracleValue::List(values) => Some(values.len()),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let maximum_cells = maximum_groups
        .checked_mul(output_columns)
        .and_then(|cells| {
            semantic
                .rows
                .len()
                .checked_mul(collect_count)
                .and_then(|elements| cells.checked_add(elements))
        })
        .and_then(|cells| cells.checked_add(maximum_list_items))
        .ok_or_else(|| Error::internal("oracle output cell capacity overflow"))?;
    let obligation_base = u64::from(report_id) * 100;
    let request = ResidentSegmentedAggregationRequest {
        project: PROJECT,
        expected_bookmark: BOOKMARK,
        expected_graph_revision: fixture.graph.revision(),
        expected_layout_version: fixture.graph.layout_version(),
        execution: ResidentExecutionId {
            high: 0x5345_474d_454e_5400 | u64::from(report_id),
            low: 0x4147_4752_4547_0000 | u64::from(report_id),
        },
        segmentation_obligation: obligation(
            obligation_base + 1,
            ResidentObligationScope::PatternFinal,
        ),
        aggregate_obligations: semantic
            .aggregates
            .iter()
            .enumerate()
            .map(|(index, _)| {
                obligation(
                    obligation_base + 2 + index as u64,
                    ResidentObligationScope::Expression(index as u16),
                )
            })
            .collect(),
        input,
        program: None,
        grouping_columns: semantic.grouping_columns.clone(),
        aggregates: semantic.aggregates.clone(),
        maximum_output_groups: u32::try_from(maximum_groups)
            .map_err(|_| Error::internal("oracle group capacity exceeds u32"))?,
        maximum_output_cells: u32::try_from(maximum_cells)
            .map_err(|_| Error::internal("oracle cell capacity exceeds u32"))?,
        maximum_output_arena_bytes: u32::try_from(
            semantic
                .rows
                .iter()
                .flatten()
                .map(|value| match value {
                    OracleValue::String(value) => value.len(),
                    OracleValue::Bytes(value) => value.len(),
                    _ => 0,
                })
                .sum::<usize>()
                .saturating_mul(
                    semantic
                        .grouping_columns
                        .len()
                        .saturating_add(semantic.aggregates.len())
                        .saturating_add(1),
                )
                .saturating_add(64),
        )
        .map_err(|_| Error::internal("oracle byte capacity exceeds u32"))?,
    };
    request.validate()?;
    Ok(request)
}

fn decode_cell(
    relation: &ResidentSegmentedRelation,
    index: usize,
    allow_list: bool,
) -> Result<OracleValue> {
    let cell = relation
        .cells
        .get(index)
        .copied()
        .ok_or_else(|| Error::internal("segmented result cell disappeared"))?;
    Ok(match cell.tag {
        ResidentSegmentedValueTag::Null => OracleValue::Null,
        ResidentSegmentedValueTag::Boolean => OracleValue::Boolean(cell.payload != 0),
        ResidentSegmentedValueTag::Integer => OracleValue::Integer(cell.payload as i64),
        ResidentSegmentedValueTag::Float => OracleValue::Float(cell.payload),
        ResidentSegmentedValueTag::String
        | ResidentSegmentedValueTag::Bytes
        | ResidentSegmentedValueTag::Map => {
            let start = usize::try_from(cell.payload)
                .map_err(|_| Error::internal("segmented result byte offset exceeds usize"))?;
            let length = usize::try_from(cell.auxiliary)
                .map_err(|_| Error::internal("segmented result byte length exceeds usize"))?;
            let bytes = relation
                .arena
                .get(start..start.saturating_add(length))
                .ok_or_else(|| Error::internal("segmented result byte range disappeared"))?;
            if cell.tag == ResidentSegmentedValueTag::String {
                OracleValue::String(
                    std::str::from_utf8(bytes)
                        .map_err(|_| Error::internal("segmented result string is invalid UTF-8"))?
                        .to_owned(),
                )
            } else if cell.tag == ResidentSegmentedValueTag::Bytes {
                OracleValue::Bytes(bytes.to_vec())
            } else {
                OracleValue::Map(bytes.to_vec())
            }
        }
        ResidentSegmentedValueTag::Node => OracleValue::Node(cell.payload),
        ResidentSegmentedValueTag::Relationship => OracleValue::Relationship(cell.payload),
        ResidentSegmentedValueTag::List if allow_list => {
            let start = usize::try_from(cell.payload)
                .map_err(|_| Error::internal("segmented list offset exceeds usize"))?;
            let length = usize::try_from(cell.auxiliary)
                .map_err(|_| Error::internal("segmented list length exceeds usize"))?;
            OracleValue::List(
                (start..start.saturating_add(length))
                    .map(|child| decode_cell(relation, child, false))
                    .collect::<Result<Vec<_>>>()?,
            )
        }
        ResidentSegmentedValueTag::List => {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "segmented collect output contains a nested list",
            ));
        }
    })
}

fn decode_relation(relation: &ResidentSegmentedRelation) -> Result<Vec<Vec<OracleValue>>> {
    relation.validate_output()?;
    let columns = relation.column_count as usize;
    (0..relation.row_count as usize)
        .map(|row| {
            (0..columns)
                .map(|column| decode_cell(relation, row * columns + column, true))
                .collect::<Result<Vec<_>>>()
        })
        .collect()
}

struct Fixture {
    graph: GraphStore,
}

impl Fixture {
    fn new() -> Self {
        Self {
            graph: GraphStore::default(),
        }
    }

    fn image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            BOOKMARK,
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

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn metal(&self) -> Result<MetalBackend> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(self.image()?)?;
        Ok(metal)
    }
}

fn validate_native_result(
    request: &ResidentSegmentedAggregationRequest,
    validated: &ValidatedResidentSegmentedAggregation,
    backend: BackendKind,
    expected: &[Vec<OracleValue>],
    label: &str,
) -> Result<()> {
    if validated.project() != request.project
        || validated.bookmark() != request.expected_bookmark
        || validated.graph_revision() != request.expected_graph_revision
        || validated.layout_version() != request.expected_layout_version
        || validated.execution() != request.execution
        || validated.fingerprint() != request.fingerprint()?
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: immutable generation or fingerprint fence changed"),
        ));
    }
    let completion = match backend {
        BackendKind::Cpu => ResidentDeviceCompletion::CpuReference,
        BackendKind::Metal => ResidentDeviceCompletion::Metal,
        BackendKind::Cuda => {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "strict segmented test accepts only CPU reference or real Metal",
            ));
        }
    };
    let expected_obligations = std::iter::once(request.segmentation_obligation)
        .chain(request.aggregate_obligations.iter().copied())
        .collect::<Vec<_>>();
    if validated.receipts().len() != expected_obligations.len()
        || validated
            .receipts()
            .iter()
            .zip(expected_obligations)
            .any(|(receipt, obligation)| {
                receipt.execution != request.execution
                    || receipt.obligation != obligation
                    || receipt.input_cardinality != u64::from(request.input.row_count)
                    || receipt.output_cardinality != u64::from(validated.relation().row_count)
                    || receipt.completion != completion
            })
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: missing or forged exact native receipts"),
        ));
    }
    let actual = decode_relation(validated.relation())?;
    if actual != expected {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: independent oracle mismatch: expected {expected:?}, got {actual:?}"),
        ));
    }
    Ok(())
}

fn run_case(backend: &dyn ExecutionBackend, fixture: &Fixture, case: OfficialCase) -> Result<()> {
    if !backend.supports_native_segmented_aggregation() {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "{}: selected backend does not advertise the exact segmented ABI",
                case.label()
            ),
        ));
    }
    let semantic = semantic_fixture(case.report_id);
    let request = request_for(case.report_id, fixture, &semantic)?;
    let raw = backend.execute_segmented_aggregation(&request, &CancellationToken::new())?;
    let validated = raw.validate_for_publication(&request, backend.kind())?;
    validate_native_result(
        &request,
        &validated,
        backend.kind(),
        &semantic.expected,
        &case.label(),
    )
}

fn assert_no_failures(label: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{label} had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn exact_official_portfolio_is_43_unique_measured_failures() {
    let expected = BTreeSet::from([
        705, 723, 734, 736, 737, 741, 742, 744, 749, 750, 751, 771, 774, 775, 779, 783, 784, 787,
        788, 789, 920, 1247, 1251, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262,
        1263, 1264, 1265, 1266, 1267, 1268, 1282, 1283, 1284, 1285,
    ]);
    let actual = OFFICIAL_CASES
        .iter()
        .map(|case| case.report_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(OFFICIAL_CASES.len(), 43);
    assert_eq!(actual, expected);
    assert!(OFFICIAL_CASES.iter().all(|case| {
        !case.feature.is_empty()
            && !case.title.is_empty()
            && !case.query.is_empty()
            && semantic_fixture(case.report_id).aggregates.len() > 0
    }));
    assert!(
        [1261, 1262, 1263, 1264]
            .into_iter()
            .all(|report_id| actual.contains(&report_id))
    );
    assert!(!actual.contains(&1269), "percentiles are not in v1");
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_all_43_aggregation_selectors() {
    assert_certified_report_identities(OFFICIAL_CASES.iter().map(|case| {
        (
            usize::from(case.report_id),
            case.feature.to_owned(),
            format!("[{}] {}", case.scenario, case.title),
        )
    }));
}

#[test]
fn strict_cpu_executes_all_43_official_aggregation_stages_through_exact_abi() -> Result<()> {
    let fixture = Fixture::new();
    let cpu = fixture.cpu()?;
    assert_eq!(cpu.kind(), BackendKind::Cpu);
    assert!(cpu.supports_native_segmented_aggregation());
    let failures = OFFICIAL_CASES
        .iter()
        .copied()
        .filter_map(|case| {
            run_case(&cpu, &fixture, case)
                .err()
                .map(|error| error.to_string())
        })
        .collect();
    assert_no_failures("strict CPU segmented-aggregation portfolio", failures);
    Ok(())
}

fn adversarial_cases() -> Vec<(u16, &'static str, SemanticFixture)> {
    use ResidentSegmentedAggregateKind::{Average, Maximum, Minimum, Sum};
    vec![
        (
            60_001,
            "first-seen group order and stable collect order",
            SemanticFixture {
                input_columns: 2,
                rows: vec![
                    vec![string("b"), integer(1)],
                    vec![string("a"), integer(2)],
                    vec![string("b"), integer(3)],
                    vec![string("c"), integer(4)],
                    vec![string("a"), integer(5)],
                ],
                grouping_columns: vec![0],
                aggregates: vec![collect(1, false)],
                expected: vec![
                    vec![string("b"), list(vec![integer(1), integer(3)])],
                    vec![string("a"), list(vec![integer(2), integer(5)])],
                    vec![string("c"), list(vec![integer(4)])],
                ],
            },
        ),
        (
            60_002,
            "numeric DISTINCT canonicalizes integer/float equality and signed zero",
            SemanticFixture {
                input_columns: 1,
                rows: vec![
                    vec![integer(1)],
                    vec![float(1.0)],
                    vec![integer(0)],
                    vec![float(-0.0)],
                ],
                grouping_columns: vec![],
                aggregates: vec![count(0, true), collect(0, true)],
                expected: vec![vec![integer(2), list(vec![integer(1), integer(0)])]],
            },
        ),
        (
            60_003,
            "all empty global aggregate identities",
            SemanticFixture {
                input_columns: 1,
                rows: vec![],
                grouping_columns: vec![],
                aggregates: vec![
                    count_all(),
                    count(0, false),
                    value_aggregate(Sum, 0),
                    value_aggregate(Average, 0),
                    value_aggregate(Minimum, 0),
                    value_aggregate(Maximum, 0),
                    collect(0, false),
                ],
                expected: vec![vec![
                    integer(0),
                    integer(0),
                    integer(0),
                    null(),
                    null(),
                    null(),
                    list(vec![]),
                ]],
            },
        ),
        (
            60_004,
            "supported mixed-family min/max total order",
            SemanticFixture {
                input_columns: 1,
                rows: vec![
                    vec![node(7)],
                    vec![OracleValue::Relationship(3)],
                    vec![string("z")],
                    vec![OracleValue::Bytes(vec![0])],
                    vec![OracleValue::Boolean(false)],
                    vec![integer(-1)],
                ],
                grouping_columns: vec![],
                aggregates: vec![value_aggregate(Minimum, 0), value_aggregate(Maximum, 0)],
                expected: vec![vec![node(7), integer(-1)]],
            },
        ),
    ]
}

#[test]
fn cpu_matches_independent_adversarial_oracle_without_host_collect_materialization() -> Result<()> {
    let fixture = Fixture::new();
    let cpu = fixture.cpu()?;
    for (id, label, semantic) in adversarial_cases() {
        let request = request_for(id, &fixture, &semantic)?;
        let raw = cpu.execute_segmented_aggregation(&request, &CancellationToken::new())?;
        let validated = raw.validate_for_publication(&request, BackendKind::Cpu)?;
        validate_native_result(
            &request,
            &validated,
            BackendKind::Cpu,
            &semantic.expected,
            label,
        )?;
        if semantic
            .aggregates
            .iter()
            .any(|aggregate| aggregate.kind == ResidentSegmentedAggregateKind::Collect)
        {
            assert!(
                validated
                    .relation()
                    .cells
                    .iter()
                    .any(|cell| cell.tag == ResidentSegmentedValueTag::List),
                "{label}: collect did not stay in the typed cell arena"
            );
        }
    }
    Ok(())
}

#[test]
fn non_numeric_reduction_and_cardinality_overflow_fail_closed() -> Result<()> {
    let fixture = Fixture::new();
    let cpu = fixture.cpu()?;
    let non_numeric = SemanticFixture {
        input_columns: 1,
        rows: vec![vec![string("not a number")]],
        grouping_columns: vec![],
        aggregates: vec![value_aggregate(ResidentSegmentedAggregateKind::Sum, 0)],
        expected: vec![],
    };
    let request = request_for(60_010, &fixture, &non_numeric)?;
    assert_eq!(
        cpu.execute_segmented_aggregation(&request, &CancellationToken::new())
            .expect_err("STRING sum unexpectedly succeeded")
            .code,
        ErrorCode::QueryType
    );

    let too_many_groups = SemanticFixture {
        input_columns: 1,
        rows: vec![vec![string("a")], vec![string("b")]],
        grouping_columns: vec![0],
        aggregates: vec![count_all()],
        expected: vec![],
    };
    let mut request = request_for(60_011, &fixture, &too_many_groups)?;
    request.maximum_output_groups = 1;
    request.maximum_output_cells = 2;
    request.validate()?;
    assert_eq!(
        cpu.execute_segmented_aggregation(&request, &CancellationToken::new())
            .expect_err("group cardinality silently truncated")
            .code,
        ErrorCode::ResultBudgetExceeded
    );
    Ok(())
}

#[test]
fn fingerprint_generation_and_selected_backend_provenance_reject_replay_or_fallback() -> Result<()>
{
    let fixture = Fixture::new();
    let cpu = fixture.cpu()?;
    let semantic = semantic_fixture(1253);
    let request = request_for(1253, &fixture, &semantic)?;
    let raw = cpu.execute_segmented_aggregation(&request, &CancellationToken::new())?;
    raw.clone()
        .validate_for_publication(&request, BackendKind::Cpu)?;
    assert_eq!(
        raw.clone()
            .validate_for_publication(&request, BackendKind::Metal)
            .expect_err("CPU result masqueraded as selected Metal output")
            .code,
        ErrorCode::CorruptStorage
    );

    let mut changed = request.clone();
    changed.input.cells[0] = ResidentSegmentedCell::integer(999);
    assert_ne!(request.fingerprint()?, changed.fingerprint()?);
    assert_eq!(
        raw.validate_for_publication(&changed, BackendKind::Cpu)
            .expect_err("precomputed aggregate result replayed against changed input")
            .code,
        ErrorCode::CorruptStorage
    );

    let mut stale = request;
    stale.expected_graph_revision += 1;
    assert_eq!(
        cpu.execute_segmented_aggregation(&stale, &CancellationToken::new())
            .expect_err("stale generation executed segmented aggregation")
            .code,
        ErrorCode::GpuAdmissionFailure
    );
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
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_matches_cpu_for_all_43_with_exact_receipts_and_no_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new();
    let cpu = fixture.cpu()?;
    let metal = fixture.metal()?;
    assert_eq!(metal.kind(), BackendKind::Metal);
    assert!(
        metal.supports_native_segmented_aggregation(),
        "real Metal backend must advertise the native segmented-aggregation dispatch"
    );

    let mut failures = Vec::new();
    for case in OFFICIAL_CASES {
        let semantic = semantic_fixture(case.report_id);
        let request = request_for(case.report_id, &fixture, &semantic)?;
        let cpu_result = cpu
            .execute_segmented_aggregation(&request, &CancellationToken::new())
            .and_then(|raw| raw.validate_for_publication(&request, BackendKind::Cpu))
            .and_then(|validated| decode_relation(validated.relation()));
        let metal_result = metal
            .execute_segmented_aggregation(&request, &CancellationToken::new())
            .and_then(|raw| raw.validate_for_publication(&request, BackendKind::Metal))
            .and_then(|validated| {
                validate_native_result(
                    &request,
                    &validated,
                    BackendKind::Metal,
                    &semantic.expected,
                    &case.label(),
                )?;
                decode_relation(validated.relation())
            });
        match (cpu_result, metal_result) {
            (Ok(cpu_rows), Ok(metal_rows)) if cpu_rows == metal_rows => {}
            (Ok(cpu_rows), Ok(metal_rows)) => failures.push(format!(
                "{}: CPU/Metal mismatch: CPU={cpu_rows:?}, Metal={metal_rows:?}",
                case.label()
            )),
            (Err(cpu_error), Ok(_)) => {
                failures.push(format!("{}: strict CPU failed: {cpu_error}", case.label()))
            }
            (Ok(_), Err(metal_error)) => failures.push(format!(
                "{}: real Metal failed: {metal_error}",
                case.label()
            )),
            (Err(cpu_error), Err(metal_error)) => failures.push(format!(
                "{}: CPU failed: {cpu_error}; Metal failed: {metal_error}",
                case.label()
            )),
        }
    }
    assert_no_failures(
        "strict CPU versus real Metal segmented aggregation",
        failures,
    );
    Ok(())
}
