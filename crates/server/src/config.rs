//! Narrow runtime configuration with safe, binding defaults.

use std::{net::SocketAddr, path::PathBuf, time::Duration};

use clap::{Parser, ValueEnum, builder::TypedValueParser as _};

/// Device family selector. Graph execution accepts Auto/Cpu; inference selects independently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum BackendSelection {
    #[default]
    Auto,
    Cpu,
    Metal,
    Cuda,
}

/// Validated process configuration.
#[derive(Clone, Debug, Parser)]
#[command(name = "irongraph", version, about)]
pub struct Config {
    #[arg(
        long,
        env = "IRONGRAPH_DATA_DIR",
        default_value_os_t = default_data_dir()
    )]
    pub data_dir: PathBuf,

    #[arg(long, env = "IRONGRAPH_HTTP_ADDR", default_value = "127.0.0.1:18484")]
    pub http_addr: SocketAddr,

    #[arg(long, env = "IRONGRAPH_BOLT_ADDR", default_value = "127.0.0.1:18485")]
    pub bolt_addr: SocketAddr,

    #[arg(long, env = "IRONGRAPH_STREAM_ADDR", default_value = "127.0.0.1:18486")]
    pub stream_addr: SocketAddr,

    #[arg(long, env = "IRONGRAPH_QUEUE_ADDR", default_value = "127.0.0.1:18487")]
    pub queue_addr: SocketAddr,

    #[arg(long, env = "IRONGRAPH_REMOTE_QUERY_ADDR")]
    pub remote_query_addr: Option<SocketAddr>,

    #[arg(long, env = "IRONGRAPH_REMOTE_BOLT_ADDR")]
    pub remote_bolt_addr: Option<SocketAddr>,

    #[arg(long, env = "IRONGRAPH_REMOTE_STREAM_ADDR")]
    pub remote_stream_addr: Option<SocketAddr>,

    #[arg(long, env = "IRONGRAPH_REMOTE_QUEUE_ADDR")]
    pub remote_queue_addr: Option<SocketAddr>,

    #[arg(long = "remote-tls-cert", env = "IRONGRAPH_REMOTE_TLS_CERT")]
    pub remote_tls_cert_path: Option<PathBuf>,

    #[arg(long = "remote-tls-key", env = "IRONGRAPH_REMOTE_TLS_KEY")]
    pub remote_tls_key_path: Option<PathBuf>,

    #[arg(long = "remote-tls-trust", env = "IRONGRAPH_REMOTE_TLS_TRUST")]
    pub remote_tls_trust_path: Option<PathBuf>,

    #[arg(
        long,
        env = "IRONGRAPH_EXECUTION_BACKEND",
        value_parser = clap::builder::PossibleValuesParser::new(["auto", "cpu"])
            .map(|value| match value.as_str() {
                "cpu" => BackendSelection::Cpu,
                _ => BackendSelection::Auto,
            }),
        default_value = "auto"
    )]
    pub execution_backend: BackendSelection,

    #[arg(long, env = "IRONGRAPH_EXECUTION_DEVICE", default_value_t = 0)]
    pub execution_device: u32,

    /// Text inference device, selected independently of CPU graph execution.
    #[arg(
        long,
        env = "IRONGRAPH_EMBEDDING_BACKEND",
        value_enum,
        default_value = "auto"
    )]
    pub embedding_backend: BackendSelection,

    #[arg(
        long = "embedding-device",
        env = "IRONGRAPH_EMBEDDING_DEVICE",
        default_value_t = 0
    )]
    pub embedding_device_ordinal: u32,

    /// Immutable project selected by loopback Kafka/AMQP listeners. Remote listeners derive this
    /// from their authenticated service credential instead.
    #[arg(long, env = "IRONGRAPH_BROKER_PROJECT")]
    pub broker_project: Option<uuid::Uuid>,

    #[arg(
        long,
        env = "IRONGRAPH_BROKER_RETENTION_INTERVAL_SECS",
        default_value_t = 30
    )]
    broker_retention_interval_secs: u64,
}

/// The data directory this process runs against: the environment's, or the default.
///
/// The server takes its own from parsed configuration. This exists for the paths that need the same
/// directory without a [`Config`] in hand, so there stays exactly one definition of state.
#[must_use]
pub fn data_dir() -> PathBuf {
    std::env::var_os("IRONGRAPH_DATA_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .unwrap_or_else(default_data_dir)
}

fn default_data_dir() -> PathBuf {
    let base = std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    base.join(".irongraph").join("data")
}

impl Config {
    #[must_use]
    pub const fn request_timeout(&self) -> Duration {
        Duration::ZERO
    }

    #[must_use]
    pub const fn broker_retention_interval(&self) -> Duration {
        Duration::from_secs(self.broker_retention_interval_secs)
    }

    #[must_use]
    pub const fn startup_timeout(&self) -> Duration {
        Duration::ZERO
    }

    #[must_use]
    pub const fn embedding_device(&self) -> crate::embeddings::EmbeddingDevice {
        match self.embedding_backend {
            BackendSelection::Auto => crate::embeddings::EmbeddingDevice::Auto,
            BackendSelection::Cpu => crate::embeddings::EmbeddingDevice::Cpu,
            BackendSelection::Metal => {
                crate::embeddings::EmbeddingDevice::Metal(self.embedding_device_ordinal)
            }
            BackendSelection::Cuda => {
                crate::embeddings::EmbeddingDevice::Cuda(self.embedding_device_ordinal)
            }
        }
    }

    pub fn validate(&self) -> crate::Result<()> {
        if !matches!(
            self.execution_backend,
            BackendSelection::Auto | BackendSelection::Cpu
        ) {
            return Err(crate::Error::invalid_data(
                "graph execution is CPU-only; select GPU inference with --embedding-backend",
            ));
        }
        if self.execution_device != 0 {
            return Err(crate::Error::invalid_data(
                "CPU graph execution device ordinal must be zero",
            ));
        }
        if self.broker_retention_interval_secs == 0 {
            return Err(crate::Error::invalid_data(
                "broker retention interval must be positive",
            ));
        }
        let remote_enabled = self.remote_query_addr.is_some()
            || self.remote_bolt_addr.is_some()
            || self.remote_stream_addr.is_some()
            || self.remote_queue_addr.is_some();
        let supplied_remote_tls_files = [
            self.remote_tls_cert_path.is_some(),
            self.remote_tls_key_path.is_some(),
            self.remote_tls_trust_path.is_some(),
        ];
        if remote_enabled && !supplied_remote_tls_files.iter().all(|present| *present) {
            return Err(crate::Error::new(
                crate::ErrorCode::AuthenticationFailed,
                "remote listeners require a TLS certificate, private key, and client trust store",
            ));
        }
        if !remote_enabled && supplied_remote_tls_files.iter().any(|present| *present) {
            return Err(crate::Error::invalid_data(
                "remote TLS material was supplied without a remote listener",
            ));
        }
        if !self.http_addr.ip().is_loopback() {
            return Err(crate::Error::new(
                crate::ErrorCode::AuthenticationFailed,
                "the query web interface must bind to loopback",
            ));
        }
        if !self.bolt_addr.ip().is_loopback()
            || !self.stream_addr.ip().is_loopback()
            || !self.queue_addr.ip().is_loopback()
        {
            return Err(crate::Error::new(
                crate::ErrorCode::AuthenticationFailed,
                "plain Bolt, Kafka, and AMQP listeners must bind to loopback",
            ));
        }
        validate_listener_addresses(self)?;
        Ok(())
    }
}

fn validate_listener_addresses(config: &Config) -> crate::Result<()> {
    let mut listeners = vec![
        config.http_addr,
        config.bolt_addr,
        config.stream_addr,
        config.queue_addr,
    ];
    listeners.extend(
        [
            config.remote_query_addr,
            config.remote_bolt_addr,
            config.remote_stream_addr,
            config.remote_queue_addr,
        ]
        .into_iter()
        .flatten(),
    );
    for (index, listener) in listeners.iter().enumerate() {
        if listeners[index + 1..]
            .iter()
            .any(|other| socket_addresses_conflict(*listener, *other))
        {
            return Err(crate::Error::invalid_data(
                "configured listeners have colliding socket addresses",
            ));
        }
    }
    Ok(())
}

fn socket_addresses_conflict(left: SocketAddr, right: SocketAddr) -> bool {
    left.port() == right.port()
        && (left.ip() == right.ip() || left.ip().is_unspecified() || right.ip().is_unspecified())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::Parser as _;

    use super::{BackendSelection, Config};

    #[test]
    fn database_defaults_to_the_private_data_home() -> Result<(), Box<dyn std::error::Error>> {
        let config = Config::try_parse_from(["irongraph"])?;
        let home = std::env::var_os("HOME").ok_or("HOME is unavailable")?;
        assert_eq!(
            config.data_dir,
            PathBuf::from(home).join(".irongraph").join("data")
        );
        assert_eq!(config.http_addr.to_string(), "127.0.0.1:18484");
        assert_eq!(config.bolt_addr.to_string(), "127.0.0.1:18485");
        assert_eq!(config.stream_addr.to_string(), "127.0.0.1:18486");
        assert_eq!(config.queue_addr.to_string(), "127.0.0.1:18487");
        Ok(())
    }

    #[test]
    fn plaintext_protocols_are_loopback_only_and_listener_collisions_are_rejected()
    -> Result<(), Box<dyn std::error::Error>> {
        let exposed = Config::try_parse_from(["irongraph", "--bolt-addr", "0.0.0.0:18485"])?;
        assert_eq!(
            exposed.validate().err().map(|error| error.code),
            Some(crate::ErrorCode::AuthenticationFailed)
        );

        let collision = Config::try_parse_from([
            "irongraph",
            "--remote-query-addr",
            "127.0.0.1:18485",
            "--remote-tls-cert",
            "server.pem",
            "--remote-tls-key",
            "server-key.pem",
            "--remote-tls-trust",
            "clients.pem",
        ])?;
        assert!(collision.validate().is_err());
        Ok(())
    }

    #[test]
    fn database_role_uses_the_built_in_embedding_configuration()
    -> Result<(), Box<dyn std::error::Error>> {
        let config = Config::try_parse_from(["irongraph"])?;
        config.validate()?;
        Ok(())
    }

    #[test]
    fn embedding_selection_is_independent_of_cpu_graph_execution()
    -> Result<(), Box<dyn std::error::Error>> {
        let default = Config::try_parse_from(["irongraph", "--execution-backend", "cpu"])?;
        default.validate()?;
        assert_eq!(
            default.embedding_device(),
            crate::embeddings::EmbeddingDevice::Auto
        );
        let metal = Config::try_parse_from([
            "irongraph",
            "--execution-backend",
            "cpu",
            "--embedding-backend",
            "metal",
            "--embedding-device",
            "2",
        ])?;
        metal.validate()?;
        assert_eq!(
            metal.embedding_device(),
            crate::embeddings::EmbeddingDevice::Metal(2)
        );
        for backend in ["metal", "cuda"] {
            assert!(Config::try_parse_from(["irongraph", "--execution-backend", backend]).is_err());
        }
        let graph_gpu = Config {
            execution_backend: BackendSelection::Metal,
            ..default
        };
        assert!(graph_gpu.validate().is_err());
        Ok(())
    }
}
