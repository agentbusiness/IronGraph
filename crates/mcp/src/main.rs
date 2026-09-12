use std::{net::SocketAddr, path::PathBuf};

use clap::{Args, Parser, Subcommand};
use irongraph_client::{ApiClient, MutualTls};
use irongraph_mcp::integrations::{IntegrationAction, IntegrationCommand};
use irongraph_mcp::{ApiDatabase, McpServer, OnDemandApiDatabase, serve_http};

#[derive(Debug, Parser)]
#[command(
    name = "irongraph-mcp",
    version,
    about = "IronGraph MCP server and host integration manager"
)]
struct Arguments {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    server: ServerArguments,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Install, inspect, or update IronGraph packages in supported AI hosts.
    Integrations(IntegrationArguments),
}

#[derive(Debug, Args)]
struct IntegrationArguments {
    #[command(subcommand)]
    action: IntegrationAction,

    /// Override the user profile root. Intended for packaging tests and managed deployments.
    #[arg(long, hide = true)]
    root: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ServerArguments {
    /// IronGraph local or remote Query API base URL.
    #[arg(
        long,
        env = "IRONGRAPH_MCP_URL",
        default_value = "http://127.0.0.1:18484"
    )]
    url: String,

    /// Client certificate for a remote mutual-TLS Query API.
    #[arg(long, env = "IRONGRAPH_MCP_TLS_CERT", requires_all = ["tls_key", "tls_ca"])]
    tls_cert: Option<PathBuf>,

    /// Client private key for a remote mutual-TLS Query API.
    #[arg(long, env = "IRONGRAPH_MCP_TLS_KEY", requires_all = ["tls_cert", "tls_ca"])]
    tls_key: Option<PathBuf>,

    /// Server certificate authority for a remote mutual-TLS Query API.
    #[arg(long, env = "IRONGRAPH_MCP_TLS_CA", requires_all = ["tls_cert", "tls_key"])]
    tls_ca: Option<PathBuf>,

    /// Serve Streamable HTTP MCP on this loopback address instead of stdio.
    #[arg(long)]
    http_addr: Option<SocketAddr>,
}

fn main() -> anyhow::Result<()> {
    let arguments = Arguments::parse();
    if let Some(Command::Integrations(integrations)) = arguments.command {
        return IntegrationCommand::new(integrations.root).run(integrations.action);
    }
    let server = arguments.server;
    let mutual_tls = match (server.tls_cert, server.tls_key, server.tls_ca) {
        (Some(certificate), Some(private_key), Some(certificate_authority)) => Some(
            MutualTls::new(certificate, private_key, certificate_authority),
        ),
        (None, None, None) => None,
        _ => return Err(anyhow::anyhow!("all three mutual-TLS files are required")),
    };
    if let Some(address) = server.http_addr {
        let database = mutual_tls.map_or_else(
            || OnDemandApiDatabase::new(server.url.clone()),
            |tls| OnDemandApiDatabase::with_mtls(server.url.clone(), tls),
        );
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(serve_http(database, address))
    } else {
        let client = match mutual_tls {
            Some(tls) => ApiClient::with_mtls(&server.url, &tls)?,
            None => ApiClient::new(&server.url)?,
        };
        McpServer::new(ApiDatabase::new(client)).run_stdio()
    }
}
