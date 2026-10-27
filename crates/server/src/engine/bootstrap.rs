use std::{
    fs::{File, OpenOptions, TryLockError},
    path::Path,
    sync::Arc,
    time::Duration,
};

use crate::{Error, ErrorCode, Result};

use super::{
    ExecutionClass, MutationStateBackend, NodeIdentity, NodeIdentityPublic, StandaloneNode,
    WriteRuntime,
};

const IDENTITY_FILE: &str = "node-identity.bin";
const DATABASE_DIRECTORY: &str = "database";
const PROCESS_LOCK_FILE: &str = ".node-process.lock";

/// Complete inputs for starting or recovering the first database node.
#[derive(Clone, Debug)]
pub struct SingleNodeBootstrapConfig {
    pub execution_class: ExecutionClass,
    pub startup_timeout: Duration,
}

/// Identity and live write runtime produced by an atomic standalone bootstrap.
pub struct BootstrappedNode<B> {
    identity: Arc<NodeIdentity>,
    runtime: Arc<WriteRuntime>,
    backend: Arc<B>,
    created_identity: bool,
    _process_lock: DataDirectoryLock,
}

impl<B> BootstrappedNode<B> {
    #[must_use]
    pub fn identity(&self) -> &Arc<NodeIdentity> {
        &self.identity
    }

    #[must_use]
    pub fn runtime(&self) -> &Arc<WriteRuntime> {
        &self.runtime
    }

    #[must_use]
    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    #[must_use]
    pub const fn created_identity(&self) -> bool {
        self.created_identity
    }
}

struct DataDirectoryLock {
    _file: File,
}

impl DataDirectoryLock {
    fn acquire(root: &Path) -> Result<Self> {
        let path = root.join(PROCESS_LOCK_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(Error::retryable(
                ErrorCode::Io,
                "another database process owns this data directory",
                None,
            )),
            Err(TryLockError::Error(error)) => Err(error.into()),
        }
    }
}

/// Standalone startup. Opening and recovery finish on a blocking worker before the ordered
/// application and persistence threads start.
pub async fn open_standalone<B, F>(
    root: impl AsRef<Path>,
    options: SingleNodeBootstrapConfig,
    open_backend: F,
) -> Result<BootstrappedNode<B>>
where
    B: MutationStateBackend,
    F: FnOnce(&Path, NodeIdentityPublic) -> Result<Arc<B>> + Send + 'static,
{
    let root = root.as_ref().to_owned();
    let (identity, created_identity, process_lock, backend) = tokio::task::spawn_blocking({
        let root = root.clone();
        move || -> Result<_> {
            std::fs::create_dir_all(&root)?;
            let process_lock = DataDirectoryLock::acquire(&root)?;
            for unsupported in ["write", "standalone-snapshots"] {
                if root.join(unsupported).exists() {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "data directory contains unsupported persistence files; choose an empty directory",
                    ));
                }
            }
            let (identity, created_identity) = load_or_generate_genesis_identity(&root)?;
            let backend = open_backend(&root.join(DATABASE_DIRECTORY), identity.public())?;
            Ok((identity, created_identity, process_lock, backend))
        }
    })
    .await
    .map_err(|error| Error::internal(format!("database bootstrap task failed: {error}")))??;
    start_write_runtime(options, identity, created_identity, process_lock, backend).await
}

async fn start_write_runtime<B>(
    options: SingleNodeBootstrapConfig,
    identity: NodeIdentity,
    created_identity: bool,
    process_lock: DataDirectoryLock,
    backend: Arc<B>,
) -> Result<BootstrappedNode<B>>
where
    B: MutationStateBackend,
{
    let identity = Arc::new(identity);
    let public = identity.public();
    let node = StandaloneNode {
        identity: public,
        execution_class: options.execution_class,
    };
    node.validate_for(public.node_id)?;
    let state_backend: Arc<dyn MutationStateBackend> = backend.clone();
    let runtime = Arc::new(WriteRuntime::start(node, state_backend).await?);
    Ok(BootstrappedNode {
        identity,
        runtime,
        backend,
        created_identity,
        _process_lock: process_lock,
    })
}

/// Loads or atomically creates the standalone genesis identity before transport provisioning.
/// Existing database state without that identity is always rejected.
pub fn load_or_generate_genesis_identity(root: impl AsRef<Path>) -> Result<(NodeIdentity, bool)> {
    let root = root.as_ref();
    std::fs::create_dir_all(root)?;
    let identity_path = root.join(IDENTITY_FILE);
    reject_missing_identity_over_existing_state(root, &identity_path)?;
    NodeIdentity::load_or_generate_genesis(identity_path)
}

pub fn load_existing_node_identity(root: impl AsRef<Path>) -> Result<Option<NodeIdentity>> {
    let root = root.as_ref();
    let identity_path = root.join(IDENTITY_FILE);
    match std::fs::symlink_metadata(&identity_path) {
        Ok(_) => NodeIdentity::load(identity_path).map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn reject_missing_identity_over_existing_state(root: &Path, identity: &Path) -> Result<()> {
    if identity.exists() {
        return Ok(());
    }
    {
        let path = root.join(DATABASE_DIRECTORY);
        match std::fs::read_dir(&path) {
            Ok(mut entries) => {
                if entries.next().is_some() {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "durable node state exists without its identity",
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use crate::{Bookmark, storage::MutationEntry};

    use super::*;
    use crate::engine::MutationApplyResult;

    struct EmptyBackend;

    #[async_trait]
    impl MutationStateBackend for EmptyBackend {
        async fn apply_mutation(&self, mutation: &MutationEntry) -> Result<MutationApplyResult> {
            Ok(MutationApplyResult {
                persistence: Some(crate::storage::RecordBatch {
                    first_index: mutation.index(),
                    bookmark: mutation.bookmark(),
                    operations: Vec::new(),
                }),
                ..MutationApplyResult::default()
            })
        }

        async fn applied_bookmark(&self) -> Bookmark {
            Bookmark { term: 1, index: 0 }
        }
    }

    fn test_options() -> SingleNodeBootstrapConfig {
        SingleNodeBootstrapConfig {
            execution_class: ExecutionClass::Cpu,
            startup_timeout: Duration::from_secs(5),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn database_open_does_not_block_the_async_worker() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let (release, wait_for_release) = std::sync::mpsc::sync_channel(0);
        let async_progress = tokio::spawn(async move {
            tokio::task::yield_now().await;
            release
                .send(())
                .map_err(|_| Error::internal("database opener stopped waiting"))
        });

        let boot = tokio::time::timeout(
            Duration::from_secs(2),
            open_standalone(directory.path(), test_options(), move |_, _| {
                wait_for_release
                    .recv_timeout(Duration::from_secs(1))
                    .map_err(|_| Error::internal("async worker was blocked by database open"))?;
                Ok(Arc::new(EmptyBackend))
            }),
        )
        .await
        .map_err(|_| Error::internal("standalone bootstrap timed out"))??;
        async_progress
            .await
            .map_err(|error| Error::internal(format!("progress task failed: {error}")))??;
        drop(boot);
        Ok(())
    }
}
