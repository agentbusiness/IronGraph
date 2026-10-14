use std::path::{Component, Path, PathBuf};

use crate::{Error, ErrorCode, Result};

/// Application-selected embedding storage. Explicit paths override environment defaults.
#[derive(Clone, Debug, Default)]
pub struct EmbeddingStorage {
    pub model_dir: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
}

impl EmbeddingStorage {
    pub fn model_directory(&self) -> Result<PathBuf> {
        resolve_directory(
            self.model_dir.as_deref(),
            std::env::var_os("IRONGRAPH_MODEL_DIR").map(PathBuf::from),
            home_directory(),
            "models",
        )
    }

    pub fn cache_directory(&self) -> Result<PathBuf> {
        resolve_directory(
            self.cache_dir.as_deref(),
            std::env::var_os("IRONGRAPH_EMBEDDING_CACHE_DIR").map(PathBuf::from),
            home_directory(),
            "embeddings",
        )
    }
}

fn home_directory() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
}

fn resolve_directory(
    explicit: Option<&Path>,
    environment: Option<PathBuf>,
    home: Option<PathBuf>,
    default_name: &str,
) -> Result<PathBuf> {
    let directory = explicit
        .map(Path::to_path_buf)
        .or(environment)
        .or_else(|| home.map(|home| home.join(".irongraph").join(default_name)))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::EmbeddingUnavailable,
                "embedding storage needs an explicit directory or a home directory",
            )
        })?;
    if directory.as_os_str().is_empty() {
        return Err(Error::new(
            ErrorCode::EmbeddingUnavailable,
            "embedding storage directory is empty",
        ));
    }
    let absolute = if directory.is_absolute() {
        directory
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(directory))
            .map_err(Error::from)?
    };
    // Resolve lexical parents before the installer creates private directories. Walking
    // `app/../models` directly could otherwise change permissions on app's parent.
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            _ => normalized.push(component.as_os_str()),
        }
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_paths_resolve_explicit_environment_and_home_in_order() -> Result<()> {
        let root = tempfile::tempdir()?;
        let explicit = root.path().join("app-a/models");
        let environment = root.path().join("app-b/models");
        let home = root.path().join("home");
        assert_eq!(
            resolve_directory(
                Some(&explicit),
                Some(environment.clone()),
                Some(home.clone()),
                "models"
            )?,
            explicit
        );
        assert_eq!(
            resolve_directory(
                None,
                Some(environment.clone()),
                Some(home.clone()),
                "models"
            )?,
            environment
        );
        assert_eq!(
            resolve_directory(None, None, Some(home.clone()), "models")?,
            home.join(".irongraph/models")
        );
        assert_eq!(
            resolve_directory(None, None, Some(home), "embeddings")?,
            root.path().join("home/.irongraph/embeddings")
        );
        assert!(resolve_directory(Some(Path::new("")), None, None, "models").is_err());
        assert!(resolve_directory(None, None, None, "models").is_err());
        assert!(
            resolve_directory(Some(Path::new("relative/models")), None, None, "models")?
                .is_absolute()
        );
        assert_eq!(
            resolve_directory(
                Some(&root.path().join("app/../models")),
                None,
                None,
                "models"
            )?,
            root.path().join("models")
        );
        Ok(())
    }
}
