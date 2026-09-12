// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result,
    cypher::{BindCapabilities, BoundQuery, bind, parse},
    graph::NameCatalog,
};

const TEMPORAL_FAMILIES: &[&str] = &["date", "localtime", "time", "localdatetime", "datetime"];
const CLOCKS: &[&str] = &["transaction", "statement", "realtime"];

fn bind_query(query: &str) -> Result<BoundQuery> {
    bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )
}

fn require_bind_error(query: &str, detail: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("invalid temporal function bound: {query}")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(error.message.contains(detail), "{query}: {error:?}");
    Ok(())
}

#[test]
fn all_temporal_clock_families_accept_null_timezone_and_no_argument() -> Result<()> {
    for family in TEMPORAL_FAMILIES {
        for clock in CLOCKS {
            for arguments in ["", "null", "'America/Los_Angeles'", "$timezone"] {
                let query = format!("RETURN {family}.{clock}({arguments}) AS current");
                bind_query(&query).map_err(|error| {
                    Error::internal(format!(
                        "legitimate temporal clock failed binding: {query}: {error}"
                    ))
                })?;
            }
        }
    }
    bind_query("RETURN DaTe.TrAnSaCtIoN(null) AS current")?;
    Ok(())
}

#[test]
fn temporal_clock_catalog_remains_exact() -> Result<()> {
    for query in [
        "RETURN date.transactional(null)",
        "RETURN date.monotonic(null)",
        "RETURN localtime.wallclock(null)",
        "RETURN datetime.transactiontime(null)",
        "RETURN duration.transaction(null)",
        "RETURN calendar.realtime(null)",
    ] {
        require_bind_error(query, "UnknownFunction")?;
    }
    Ok(())
}

#[test]
fn temporal_clock_validation_rejects_only_known_bad_arity_and_literal_types() -> Result<()> {
    for family in TEMPORAL_FAMILIES {
        for clock in CLOCKS {
            require_bind_error(
                &format!("RETURN {family}.{clock}('UTC', 'Europe/Stockholm')"),
                "InvalidArgumentCount",
            )?;
        }
    }
    for argument in ["1", "true", "[\"UTC\"]", "{timezone: 'UTC'}"] {
        require_bind_error(
            &format!("RETURN datetime.realtime({argument})"),
            "InvalidArgumentType",
        )?;
    }
    for query in [
        "WITH $timezone AS zone RETURN datetime.realtime(zone)",
        "RETURN datetime.realtime(CASE WHEN $useUtc THEN 'UTC' ELSE null END)",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "runtime-resolved timezone was rejected: {query}: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn temporal_clock_results_are_conservative_for_properties_and_quantifiers() -> Result<()> {
    for query in [
        "WITH date.transaction('UTC') AS current RETURN current.year",
        "RETURN localtime.statement(null).hour",
        "RETURN datetime.realtime($timezone).timezone",
        "RETURN any(value IN [date.transaction(null)] WHERE value % 2 = 0)",
        "RETURN any(value IN [date.transaction($timezone)] WHERE value % 2 = 0)",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "conservative temporal result was rejected: {query}: {error}"
            ))
        })?;
    }
    require_bind_error(
        "RETURN any(value IN [date.transaction('UTC')] WHERE value % 2 = 0)",
        "InvalidArgumentType",
    )?;
    Ok(())
}
