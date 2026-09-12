// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result,
    cypher::{Statement, parse},
};

fn parse_error(query: &str) -> Result<Error> {
    parse(query)
        .err()
        .ok_or_else(|| Error::internal("mixed UNION modes unexpectedly parsed successfully"))
}

#[test]
fn mixed_union_modes_are_invalid_clause_composition_syntax_errors() -> Result<()> {
    for query in [
        "RETURN 1 AS a UNION RETURN 2 AS a UNION ALL RETURN 3 AS a",
        "RETURN 1 AS a UNION ALL RETURN 2 AS a UNION RETURN 3 AS a",
    ] {
        let error = parse_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("InvalidClauseComposition"),
            "{query}: {}",
            error.message
        );
        assert!(
            !error.message.starts_with("UnexpectedSyntax"),
            "{query}: {}",
            error.message
        );
    }
    Ok(())
}

#[test]
fn homogeneous_multi_branch_union_modes_remain_legal() -> Result<()> {
    for (query, expected_all) in [
        (
            "RETURN 1 AS a UNION RETURN 2 AS a UNION RETURN 3 AS a",
            false,
        ),
        (
            "RETURN 1 AS a UNION ALL RETURN 2 AS a UNION ALL RETURN 3 AS a",
            true,
        ),
    ] {
        let parsed = parse(query)?;
        let Statement::Query(body) = parsed.statement else {
            return Err(Error::internal("expected a query statement"));
        };
        assert_eq!(body.unions.len(), 2, "{query}");
        assert!(
            body.unions.iter().all(|branch| branch.all == expected_all),
            "{query}: {body:?}"
        );
    }
    Ok(())
}
