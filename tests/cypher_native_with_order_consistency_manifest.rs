// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Baseline manifest for the twelve remaining Metal-only `WithOrderBy1` gaps.
//!
//! This file pins report identity and exact expanded query text only. It deliberately contains no
//! execution backend or substitute semantics; native CPU/Metal coverage belongs to the eventual
//! implementation tranches.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Tranche {
    MixedGraphTotalOrder,
    ScalarConsistency,
    ListConsistency,
    TemporalConsistency,
}

#[derive(Clone, Copy, Debug)]
enum Query {
    MixedAscending,
    MixedDescending,
    #[allow(dead_code)]
    Consistency(&'static str),
}

impl Query {
    #[allow(dead_code)]
    fn text(self) -> String {
        match self {
            Self::MixedAscending => concat!(
                "MATCH p = (n:N)-[r:REL]->() ",
                "UNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, ",
                "{a: 'map'}] AS types WITH types ORDER BY types LIMIT 5 RETURN types"
            )
            .to_owned(),
            Self::MixedDescending => concat!(
                "MATCH p = (n:N)-[r:REL]->() ",
                "UNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, ",
                "{a: 'map'}] AS types WITH types ORDER BY types DESC LIMIT 5 RETURN types"
            )
            .to_owned(),
            Self::Consistency(values) => format!(
                "WITH {values} AS values \
                 WITH values, size(values) AS numOfValues \
                 UNWIND values AS value \
                 WITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues \
                 ORDER BY value \
                 WITH numOfValues, collect(x) AS orderedX \
                 RETURN orderedX = range(0, numOfValues-1) AS equal"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct Case {
    report_index: usize,
    scenario: u8,
    name: &'static str,
    query: Query,
    tranche: Tranche,
}

const CASES: [Case; 12] = [
    Case {
        report_index: 951,
        scenario: 21,
        name: "[21] Sort distinct types in ascending order",
        query: Query::MixedAscending,
        tranche: Tranche::MixedGraphTotalOrder,
    },
    Case {
        report_index: 952,
        scenario: 22,
        name: "[22] Sort distinct types in descending order",
        query: Query::MixedDescending,
        tranche: Tranche::MixedGraphTotalOrder,
    },
    Case {
        report_index: 1007,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1008]",
        query: Query::Consistency("[true, false]"),
        tranche: Tranche::ScalarConsistency,
    },
    Case {
        report_index: 1008,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1009]",
        query: Query::Consistency(
            "[351, -3974856, 93, -3, 123, 0, 3, -2, 20934587, 1, 20934585, 20934586, -10]",
        ),
        tranche: Tranche::ScalarConsistency,
    },
    Case {
        report_index: 1009,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1010]",
        query: Query::Consistency(
            "[351.5, -3974856.01, -3.203957, 123.0002, 123.0001, 123.00013, 123.00011, 0.0100000, 0.0999999, 0.00000001, 3.0, 209345.87, -10.654]",
        ),
        tranche: Tranche::ScalarConsistency,
    },
    Case {
        report_index: 1010,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1011]",
        query: Query::Consistency(
            "['Sort', 'order', ' ', 'should', 'be', '', 'consistent', 'with', 'comparisons', ', ', 'where', 'comparisons are', 'defined', '!']",
        ),
        tranche: Tranche::ScalarConsistency,
    },
    Case {
        report_index: 1011,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1012]",
        query: Query::Consistency(
            "[[2, 2], [2, -2], [1, 2], [], [1], [300, 0], [1, -20], [2, -2, 100]]",
        ),
        tranche: Tranche::ListConsistency,
    },
    Case {
        report_index: 1012,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1013]",
        query: Query::Consistency(
            "[date({year: 1910, month: 5, day: 6}), date({year: 1980, month: 12, day: 24}), date({year: 1984, month: 10, day: 12}), date({year: 1985, month: 5, day: 6}), date({year: 1980, month: 10, day: 24}), date({year: 1984, month: 10, day: 11})]",
        ),
        tranche: Tranche::TemporalConsistency,
    },
    Case {
        report_index: 1013,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1014]",
        query: Query::Consistency(
            "[localtime({hour: 10, minute: 35}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}), localtime({hour: 12, minute: 35, second: 13}), localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 15})]",
        ),
        tranche: Tranche::TemporalConsistency,
    },
    Case {
        report_index: 1014,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1015]",
        query: Query::Consistency(
            "[time({hour: 10, minute: 35, timezone: '-08:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}), time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}), time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'}), time({hour: 12, minute: 35, second: 15, timezone: '+01:00'})]",
        ),
        tranche: Tranche::TemporalConsistency,
    },
    Case {
        report_index: 1015,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1016]",
        query: Query::Consistency(
            "[localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}), localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}), localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}), localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})]",
        ),
        tranche: Tranche::TemporalConsistency,
    },
    Case {
        report_index: 1016,
        scenario: 45,
        name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1017]",
        query: Query::Consistency(
            "[datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}), datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}), datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}), datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}), datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})]",
        ),
        tranche: Tranche::TemporalConsistency,
    },
];

#[test]
fn collision_safe_tranche_partition_is_complete() {
    let expected = [
        (Tranche::MixedGraphTotalOrder, 2),
        (Tranche::ScalarConsistency, 4),
        (Tranche::ListConsistency, 1),
        (Tranche::TemporalConsistency, 5),
    ];
    assert_eq!(
        expected.iter().map(|(_, count)| count).sum::<usize>(),
        CASES.len()
    );
    for (tranche, count) in expected {
        assert_eq!(
            CASES.iter().filter(|case| case.tranche == tranche).count(),
            count,
            "wrong count for {tranche:?}"
        );
    }
}
