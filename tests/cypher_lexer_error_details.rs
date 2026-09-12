// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{Error, Result, cypher::lex};

#[test]
fn unicode_dash_is_not_silently_treated_as_ascii_subtraction() -> Result<()> {
    let error = lex("RETURN 42 — 41")
        .err()
        .ok_or_else(|| Error::internal("query was accepted"))?;
    let detail = error.to_string();
    if !detail.contains("InvalidUnicodeCharacter") {
        return Err(Error::internal(format!(
            "expected InvalidUnicodeCharacter detail, got `{detail}`"
        )));
    }
    Ok(())
}
