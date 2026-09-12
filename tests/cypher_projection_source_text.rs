// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, Result,
    cypher::{Clause, Expression, Projection, ProjectionItem, Statement, parse},
};

fn projections(source: &str) -> Result<Vec<Projection>> {
    let query = parse(source)?;
    let Statement::Query(body) = query.statement else {
        return Err(Error::internal("expected a query body"));
    };
    Ok(body
        .clauses
        .into_iter()
        .filter_map(|clause| match clause {
            Clause::With(projection) | Clause::Return(projection) => Some(projection),
            _ => None,
        })
        .collect())
}

#[test]
fn parsed_projection_names_preserve_exact_expression_source() -> Result<()> {
    let parsed = projections(
        "WITH aVg(    n.aGe     ) AS mean\n\
         RETURN cOuNt( * ), {  first : n.aGe,   second: 2  }, \
         ( n.aGe   +  1 ), (n.aGe)",
    )?;
    assert_eq!(parsed.len(), 2);

    let with_item = &parsed[0].items[0];
    assert_eq!(
        with_item.source_text.as_deref(),
        Some("aVg(    n.aGe     )")
    );
    assert_eq!(with_item.column_name(0), "mean");

    let return_items = &parsed[1].items;
    let expected = [
        "cOuNt( * )",
        "{  first : n.aGe,   second: 2  }",
        "( n.aGe   +  1 )",
        "(n.aGe)",
    ];
    for (index, (item, expected)) in return_items.iter().zip(expected).enumerate() {
        assert_eq!(item.source_text.as_deref(), Some(expected));
        assert_eq!(item.column_name(index), expected);
    }
    Ok(())
}

#[test]
fn synthesized_projection_uses_canonical_fallback_without_source_text() {
    let item = ProjectionItem {
        expression: Expression::Function {
            name: vec!["count".to_owned()],
            distinct: false,
            arguments: vec![Expression::Star],
        },
        alias: None,
        source_text: None,
    };

    assert_eq!(item.column_name(0), "count(*)");
}
