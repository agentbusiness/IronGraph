//! Make cargo notice when the web bundle changes.
//!
//! `http.rs` embeds the built interface with `#[derive(RustEmbed)] #[folder = "../../web/dist"]`,
//! which reads that directory **at compile time**. Cargo knew nothing about it: with no build
//! script there was no `rerun-if-changed` on `web/dist`, so rebuilding the interface and then
//! rebuilding the binary produced a binary carrying the *previous* bundle, with no warning and no
//! visible difference. That is not hypothetical — it cost this repo an afternoon: a server was
//! serving `index-2hY_bseI.js` while `web/dist` held `index-tGi0pq7A.js`, and several rounds of
//! UI fixes were reported as landed while the running binary could not possibly have contained
//! them.
//!
//! Watching the directory alone is not enough. A build script that re-runs but emits the same
//! output leaves the crate's fingerprint unchanged, and cargo skips recompiling the library — so
//! the `RustEmbed` derive never re-reads the folder. The fix is to emit something that *changes
//! with the contents*: `IRONGRAPH_WEB_BUNDLE` carries a digest of the bundle, so a changed
//! interface changes the environment the crate is compiled with, which forces the recompile that
//! re-reads the folder.
//!
//! The same digest doubles as the build stamp the server can state about itself, so "which
//! interface is this binary carrying" has an answer that does not require unpacking the binary.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let dist = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/dist");

    // Re-run whenever anything under the bundle changes. Cargo walks a directory given here, so
    // this covers a new asset, a deleted one, and an edit to any of them.
    println!("cargo:rerun-if-changed={}", dist.display());
    println!("cargo:rerun-if-changed=build.rs");

    let (digest, entry) = fingerprint(&dist);
    // Changing this changes the crate's fingerprint, which is what actually forces `http.rs` to be
    // recompiled and the folder to be re-read. Without it the `rerun-if-changed` above would fire
    // and change nothing.
    println!("cargo:rustc-env=IRONGRAPH_WEB_BUNDLE={digest}");
    println!("cargo:rustc-env=IRONGRAPH_WEB_ENTRY={entry}");

    if !dist.exists() {
        // A build with no interface still has to succeed — the server is useful without it, and a
        // first checkout has not run the web build yet. Say so rather than failing.
        println!(
            "cargo:warning=web/dist does not exist; this binary will serve no interface. \
             Run `npm run build --prefix web` and rebuild."
        );
    }
}

/// A digest of the bundle, and the name of its entry chunk.
///
/// Content-addressed, not timestamped: Vite already puts a hash of the contents in every filename,
/// so the set of names *is* the state of the bundle, and a rebuild that changed nothing produces
/// the same digest and correctly recompiles nothing. Sizes are folded in as well, so an asset Vite
/// does not hash (`index.html`, `favicon.svg`) still moves the digest when it is edited.
fn fingerprint(dist: &Path) -> (String, String) {
    // A BTreeMap rather than a walk order: a directory listing is not ordered, and a digest that
    // depends on the order files happen to be returned in is a digest that changes at random.
    let mut seen: BTreeMap<String, u64> = BTreeMap::new();
    collect(dist, dist, &mut seen);

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (name, size) in &seen {
        for byte in name.as_bytes().iter().chain(&size.to_le_bytes()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    // The entry chunk is the one a stale binary is recognised by, so it is worth naming on its own.
    let entry = seen
        .keys()
        .find(|name| name.starts_with("assets/index-") && name.ends_with(".js"))
        .cloned()
        .unwrap_or_else(|| "none".to_owned());

    (format!("{hash:016x}"), entry)
}

fn collect(root: &Path, dir: &Path, out: &mut BTreeMap<String, u64>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            collect(root, &path, out);
        } else if let Ok(relative) = path.strip_prefix(root) {
            out.insert(relative.to_string_lossy().replace('\\', "/"), meta.len());
        }
    }
}
