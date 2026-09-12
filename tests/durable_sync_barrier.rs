// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Guards the one durability barrier every WAL and snapshot write depends on.
//!
//! On Apple platforms `fsync(2)` returns before the drive flushes its own volatile write cache, so
//! a process can acknowledge an appended WAL entry and still lose it to sudden power loss. Only
//! `fcntl(F_FULLFSYNC)` provides the real barrier, and `std` offers no wrapper for it, so every
//! durability-critical flush routes through `storage::fs::sync_durable`.
//!
//! Calling `File::sync_all`/`File::sync_data` directly in these modules silently reintroduces the
//! weaker barrier on the platform the product contract names as primary, and no functional test
//! would notice: the write still lands, the data still reads back, and the loss only appears after
//! an unplanned power cut. A source-level invariant is the only thing that catches it.

use std::{fs, path::Path};

/// Files that legitimately call the platform primitives.
///
/// `fs.rs` implements the barrier itself. `snapshot_data.rs` exposes its own `sync_all` wrapper
/// whose body already delegates to `sync_durable` for each chunk it owns.
const BARRIER_IMPLEMENTORS: [&str; 2] = ["storage/src/fs.rs", "engine/snapshot_data.rs"];

fn collect_rust_sources(path: &Path, collected: &mut Vec<(String, String)>) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("durability-audit directory must be readable") {
            let entry = entry.expect("durability-audit entry must be readable");
            collect_rust_sources(&entry.path(), collected);
        }
        return;
    }
    if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
        return;
    }
    let display = path.display().to_string().replace('\\', "/");
    if BARRIER_IMPLEMENTORS
        .iter()
        .any(|allowed| display.ends_with(allowed))
    {
        return;
    }
    collected.push((
        display,
        fs::read_to_string(path).expect("durability-audit source must be UTF-8"),
    ));
}

#[test]
fn durability_critical_writes_use_the_full_barrier_not_plain_fsync() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    for path in [
        root.join("crates/storage/src"),
        root.join("crates/server/src/engine"),
        root.join("crates/server/src/server/database.rs"),
    ] {
        collect_rust_sources(&path, &mut sources);
    }
    assert!(
        !sources.is_empty(),
        "durability audit found no sources to check"
    );

    let violations = sources
        .iter()
        .flat_map(|(path, source)| {
            source
                .lines()
                .enumerate()
                .filter(|(_, line)| line.contains(".sync_all()") || line.contains(".sync_data()"))
                .map(move |(index, line)| {
                    format!("{path}:{}: {}", index.saturating_add(1), line.trim())
                })
        })
        .collect::<Vec<_>>();

    assert!(
        violations.is_empty(),
        "durability-critical code must flush through `storage::fs::sync_durable`, which issues \
         F_FULLFSYNC on Apple targets; these call the weaker platform primitive directly:\n{}",
        violations.join("\n")
    );
}
