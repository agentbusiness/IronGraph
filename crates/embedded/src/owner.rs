use std::{
    ops::Deref,
    sync::atomic::{AtomicBool, Ordering},
    sync::{OnceLock, RwLock, mpsc},
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{EmbeddedDatabase, EmbeddedError, Result};

const MAX_ADAPTER_OWNERS: usize = 64;

struct Teardown {
    database: EmbeddedDatabase,
    _admission: AdapterOwnerAdmission,
}

struct Reaper {
    sender: mpsc::SyncSender<Teardown>,
    owners: std::sync::Arc<Semaphore>,
    failed: AtomicBool,
}

static REAPER: OnceLock<std::result::Result<Reaper, std::io::Error>> = OnceLock::new();

#[cfg(test)]
type TeardownGate = (mpsc::Sender<()>, mpsc::Receiver<()>);
#[cfg(test)]
static TEARDOWN_GATE: std::sync::Mutex<Option<TeardownGate>> = std::sync::Mutex::new(None);

fn reaper() -> Result<&'static Reaper> {
    let reaper = REAPER
        .get_or_init(|| {
            let (sender, receiver) = mpsc::sync_channel::<Teardown>(MAX_ADAPTER_OWNERS);
            std::thread::Builder::new()
                .name("irongraph-native-teardown".into())
                .spawn(move || {
                    for teardown in receiver {
                        let Teardown {
                            database,
                            _admission,
                        } = teardown;
                        // Native finalizers must never perform shutdown or graph destruction on the
                        // interpreter thread. A failed shutdown also remains confined to this worker.
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            #[cfg(test)]
                            if let Some((entered, release)) = TEARDOWN_GATE.lock().unwrap().take() {
                                entered.send(()).unwrap();
                                release.recv().unwrap();
                            }
                            if let Err(error) = database.close() {
                                tracing::error!(%error, "native adapter teardown failed");
                            }
                        }));
                        drop(_admission);
                    }
                })?;
            Ok(Reaper {
                sender,
                owners: std::sync::Arc::new(Semaphore::new(MAX_ADAPTER_OWNERS)),
                failed: AtomicBool::new(false),
            })
        })
        .as_ref()
        .map_err(|error| {
            EmbeddedError::Configuration(format!("native teardown worker failed to start: {error}"))
        })?;
    if reaper.failed.load(Ordering::Acquire) {
        return Err(EmbeddedError::Configuration(
            "native teardown worker is unavailable".into(),
        ));
    }
    Ok(reaper)
}

/// Reserves one of 64 native adapter lifecycles before opening a database.
/// Capacity remains occupied until the canonical owner has been fully destroyed.
pub struct AdapterOwnerAdmission {
    _permit: OwnedSemaphorePermit,
}

impl AdapterOwnerAdmission {
    pub fn reserve() -> Result<Self> {
        let owner = reaper()?.owners.clone().try_acquire_owned().map_err(|_| {
            irongraph_types::Error::new(
                irongraph_types::ErrorCode::Backpressure,
                "native adapter lifecycle budget exhausted before open",
            )
        })?;
        Ok(Self { _permit: owner })
    }
}

/// An identity-shared adapter owner whose final destruction runs on a bounded native worker.
pub struct AdapterDatabaseOwner {
    database: RwLock<Option<EmbeddedDatabase>>,
    admission: std::sync::Mutex<Option<AdapterOwnerAdmission>>,
}

impl AdapterDatabaseOwner {
    pub fn new(database: EmbeddedDatabase, admission: AdapterOwnerAdmission) -> Self {
        Self {
            database: RwLock::new(Some(database)),
            admission: std::sync::Mutex::new(Some(admission)),
        }
    }

    /// Runs explicit close on the caller's worker and releases lifecycle capacity afterwards.
    pub fn close(&self) -> Result<()> {
        let (database, admission) = {
            let mut owner = self.database.write().map_err(|_| {
                EmbeddedError::Configuration("embedded database owner is poisoned".into())
            })?;
            let Some(database) = owner.take() else {
                return Ok(());
            };
            let admission = self
                .admission
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            (database, admission)
        };
        let result = database.close();
        drop(admission);
        result
    }
}

impl Deref for AdapterDatabaseOwner {
    type Target = RwLock<Option<EmbeddedDatabase>>;

    fn deref(&self) -> &Self::Target {
        &self.database
    }
}

impl Drop for AdapterDatabaseOwner {
    fn drop(&mut self) {
        let Some(database) = self
            .database
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        else {
            return;
        };
        let admission = self
            .admission
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("live adapter owner holds admission");
        // Every queued or executing teardown retains a permit. Since the queue has the entire
        // admission capacity, this nonblocking send cannot encounter a legitimately full queue.
        let reaper = REAPER
            .get()
            .expect("adapter admission initialized teardown worker")
            .as_ref()
            .expect("admitted teardown worker exists");
        if let Err(error) = reaper.sender.try_send(Teardown {
            database,
            _admission: admission,
        }) {
            reaper.failed.store(true, Ordering::Release);
            // An impossible worker failure must not turn a finalizer into synchronous database
            // teardown. Retain this bounded owner and reject every subsequent lifecycle admission.
            let (mpsc::TrySendError::Full(teardown) | mpsc::TrySendError::Disconnected(teardown)) =
                error;
            std::mem::forget(teardown);
            tracing::error!("native teardown worker failed; further adapter opens are disabled");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalizer_returns_while_native_teardown_is_paused_and_retains_admission()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let _serial = crate::acceptance::TESTS.lock().unwrap();
        let directory = tempfile::tempdir()?;
        let database = EmbeddedDatabase::open(
            crate::EmbeddedOptions::new(directory.path())
                .with_embedding_policy(crate::EmbeddingPolicy::Disabled),
        )?;
        let owner = AdapterDatabaseOwner::new(database, AdapterOwnerAdmission::reserve()?);
        let (entered, observed) = mpsc::channel();
        let (release, blocked) = mpsc::channel();
        *TEARDOWN_GATE.lock().unwrap() = Some((entered, blocked));
        drop(owner);
        observed.recv_timeout(std::time::Duration::from_secs(5))?;
        assert_eq!(reaper()?.owners.available_permits(), MAX_ADAPTER_OWNERS - 1);
        let admissions = (0..MAX_ADAPTER_OWNERS - 1)
            .map(|_| AdapterOwnerAdmission::reserve())
            .collect::<Result<Vec<_>>>()?;
        assert!(AdapterOwnerAdmission::reserve().is_err());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        });
        drop(admissions);
        release.send(())?;
        runtime.block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                while reaper().unwrap().owners.available_permits() != MAX_ADAPTER_OWNERS {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            })
            .await
        })?;
        EmbeddedDatabase::open(
            crate::EmbeddedOptions::new(directory.path())
                .with_embedding_policy(crate::EmbeddingPolicy::Disabled),
        )?
        .close()?;
        Ok(())
    }
}
