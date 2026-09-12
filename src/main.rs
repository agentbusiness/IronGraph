use clap::Parser;
use irongraph::{config::Config, server};
use irongraph_mcp::{OnDemandApiDatabase, serve_http};
use std::{env, net::SocketAddr};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::parse();
    let query_url = format!("http://{}", config.http_addr);
    let mcp_address = env::var("IRONGRAPH_MCP_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:18488".to_owned())
        .parse::<SocketAddr>()?;
    let mcp = serve_http(OnDemandApiDatabase::new(query_url), mcp_address);
    tokio::select! {
        result = server::run(config) => result,
        result = mcp => result,
    }
}
