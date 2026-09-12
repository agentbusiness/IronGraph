use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: String,
    targets: BTreeMap<String, Artifact>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    url: String,
    sha256: String,
}

fn main() {
    if let Err(error) = configure() {
        panic!("IronGraph native SDK setup failed: {error}");
    }
}

fn configure() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    // Documentation generation does not link or run the database, and docs.rs has no network.
    if env::var_os("DOCS_RS").is_some() {
        return Ok(());
    }
    println!("cargo:rerun-if-env-changed=IRONGRAPH_NATIVE_DIR");
    println!("cargo:rerun-if-changed=native-manifest.json");
    let target = env::var("TARGET")?;
    if !matches!(
        target.as_str(),
        "aarch64-apple-darwin" | "aarch64-unknown-linux-gnu" | "x86_64-unknown-linux-gnu"
    ) {
        return Err(format!("unsupported target {target}; supported targets are macOS ARM64 and glibc Linux ARM64/AMD64").into());
    }
    let native_dir = if let Some(directory) = env::var_os("IRONGRAPH_NATIVE_DIR") {
        let directory = fs::canonicalize(directory)?;
        if !directory.join("libirongraph_ffi.a").is_file() {
            return Err("IRONGRAPH_NATIVE_DIR must contain libirongraph_ffi.a for the selected target and package version".into());
        }
        println!(
            "cargo:rerun-if-changed={}",
            directory.join("libirongraph_ffi.a").display()
        );
        directory
    } else {
        let path =
            PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("missing CARGO_MANIFEST_DIR")?)
                .join("native-manifest.json");
        let bytes = fs::read(path).map_err(|error| format!("cannot read native-manifest.json: {error}. Install an official release package, or set IRONGRAPH_NATIVE_DIR to a directory containing the matching preinstalled libirongraph_ffi.a for offline use."))?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        if manifest.version != env::var("CARGO_PKG_VERSION")? {
            return Err("native manifest version does not match the Cargo package version".into());
        }
        let artifact = manifest
            .targets
            .get(&target)
            .ok_or("native manifest has no archive for the selected target")?;
        let url = reqwest::Url::parse(&artifact.url)?;
        if url.scheme() != "https"
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err("native archive URL must use HTTPS without credentials".into());
        }
        if artifact.sha256.len() != 64
            || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(
                "native manifest SHA256 must contain exactly 64 hexadecimal characters".into(),
            );
        }
        let directory = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing OUT_DIR")?)
            .join("irongraph-native");
        fs::create_dir_all(&directory)?;
        let archive = directory.join("libirongraph_ffi.a");
        let expected = artifact.sha256.to_ascii_lowercase();
        if !archive.is_file() || checksum(&archive)? != expected {
            let temporary = directory.join("libirongraph_ffi.a.download");
            let result = download(&artifact.url, &temporary, &expected);
            if let Err(error) = result {
                let _ = fs::remove_file(&temporary);
                return Err(error);
            }
            fs::rename(&temporary, &archive)?;
        }
        directory
    };
    println!("cargo:rustc-link-search=native={}", native_dir.display());
    println!("cargo:rustc-link-lib=static=irongraph_ffi");
    if target.ends_with("apple-darwin") {
        for framework in [
            "Security",
            "CoreFoundation",
            "CoreML",
            "ImageIO",
            "CoreGraphics",
            "CoreVideo",
            "Metal",
            "Foundation",
        ] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        for library in ["objc", "iconv", "System", "c", "m"] {
            println!("cargo:rustc-link-lib={library}");
        }
    } else {
        for library in ["gcc_s", "util", "rt", "pthread", "m", "dl", "c"] {
            println!("cargo:rustc-link-lib={library}");
        }
    }
    Ok(())
}

fn checksum(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut input = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn download(
    url: &str,
    destination: &Path,
    expected: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(1800))
        .build()?;
    let mut response = client.get(url).send()?.error_for_status()?;
    let mut output = fs::File::create(destination)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = response.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        output.write_all(&buffer[..count])?;
        digest.update(&buffer[..count]);
    }
    output.sync_all()?;
    if format!("{:x}", digest.finalize()) != expected {
        return Err("downloaded native archive SHA256 does not match the release manifest".into());
    }
    Ok(())
}
