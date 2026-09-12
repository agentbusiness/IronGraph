// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{fs, path::Path};

const FORMER_LOGICAL_ROW_CEILING_SPELLINGS: [&str; 3] = ["1_048_576", "1,048,576", "1048576"];

fn visit_text_files(path: &Path, checked: &mut Vec<(String, String)>) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("ceiling-audit directory must be readable") {
            let entry = entry.expect("ceiling-audit entry must be readable");
            visit_text_files(&entry.path(), checked);
        }
        return;
    }

    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return;
    };
    if !matches!(extension, "rs" | "metal" | "md") {
        return;
    }
    checked.push((
        path.display().to_string(),
        fs::read_to_string(path).expect("ceiling-audit source must be UTF-8"),
    ));
}

#[test]
fn former_logical_row_ceiling_cannot_return_to_query_database_or_documentation_code() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut checked = Vec::new();
    for path in [
        root.join("crates/cypher/src"),
        root.join("crates/execution/src"),
        root.join("crates/gpu/src"),
        root.join("crates/graph/src"),
        root.join("crates/types/src/document.rs"),
        root.join("crates/server/src/protocol/query.rs"),
        root.join("crates/server/src/server/database.rs"),
        root.join("crates/server/src/server/http.rs"),
        root.join("crates/server/src/server/remote.rs"),
        root.join("kernels"),
        root.join("docs"),
    ] {
        visit_text_files(&path, &mut checked);
    }

    let violations = checked
        .iter()
        .flat_map(|(path, source)| {
            FORMER_LOGICAL_ROW_CEILING_SPELLINGS
                .iter()
                .filter(move |spelling| source.contains(**spelling))
                .map(move |spelling| format!("{path}: contains `{spelling}`"))
        })
        .collect::<Vec<_>>();
    assert!(
        violations.is_empty(),
        "the removed logical row ceiling was reintroduced:\n{}",
        violations.join("\n")
    );
}
