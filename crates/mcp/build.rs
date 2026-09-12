use std::{env, fs, io, path::PathBuf};

fn main() -> io::Result<()> {
    let manifest = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or_else(|| io::Error::other("CARGO_MANIFEST_DIR is unavailable"))?,
    );
    let docs_root = manifest.join("../../docs/cypher");
    println!("cargo:rerun-if-changed={}", docs_root.display());

    let mut files = Vec::new();
    collect_markdown(&docs_root, &docs_root, &mut files)?;
    files.sort();

    let mut generated = String::from("pub const CYPHER_DOCS: &[CypherDoc] = &[\n");
    for relative in files {
        let path = relative.to_string_lossy().replace('\\', "/");
        generated.push_str(&format!(
            "    CypherDoc {{ path: {path:?}, markdown: include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/../../docs/cypher/{path}\")) }},\n"
        ));
    }
    generated.push_str("];\n");

    let output = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| io::Error::other("OUT_DIR is unavailable"))?,
    );
    fs::write(output.join("cypher_docs.rs"), generated)
}

fn collect_markdown(
    root: &std::path::Path,
    directory: &std::path::Path,
    files: &mut Vec<PathBuf>,
) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_markdown(root, &path, files)?;
        } else if file_type.is_file() && path.extension().is_some_and(|extension| extension == "md")
        {
            files.push(
                path.strip_prefix(root)
                    .map_err(io::Error::other)?
                    .to_owned(),
            );
        }
    }
    Ok(())
}
