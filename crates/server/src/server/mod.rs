mod database;
mod dataset_loader;
mod http;
mod remote;
mod tls;

pub use database::Database;

use crate::config::Config;

/// Opens durable state, starts every configured protocol listener, and waits for shutdown.
pub async fn run(config: Config) -> anyhow::Result<()> {
    http::run(config).await.map_err(anyhow::Error::from)
}
