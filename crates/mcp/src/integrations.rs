use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

pub const INTEGRATION_VERSION: &str = env!("CARGO_PKG_VERSION");

const PACKAGE_ASSETS: &[(&str, &str)] = &[
    (
        "plugin.json",
        include_str!("../../../integrations/agent-plugins/irongraph/plugin.json"),
    ),
    (
        "mcp.json",
        include_str!("../../../integrations/agent-plugins/irongraph/mcp.json"),
    ),
    (
        ".mcp.json",
        include_str!("../../../integrations/agent-plugins/irongraph/.mcp.json"),
    ),
    (
        ".codex-plugin/plugin.json",
        include_str!("../../../integrations/agent-plugins/irongraph/.codex-plugin/plugin.json"),
    ),
    (
        ".claude-plugin/plugin.json",
        include_str!("../../../integrations/agent-plugins/irongraph/.claude-plugin/plugin.json"),
    ),
    (
        "gemini-extension.json",
        include_str!("../../../integrations/agent-plugins/irongraph/gemini-extension.json"),
    ),
    (
        "GEMINI.md",
        include_str!("../../../integrations/agent-plugins/irongraph/GEMINI.md"),
    ),
    (
        "README.md",
        include_str!("../../../integrations/agent-plugins/irongraph/README.md"),
    ),
    (
        "skills/irongraph/SKILL.md",
        include_str!("../../../integrations/agent-plugins/irongraph/skills/irongraph/SKILL.md"),
    ),
    (
        "skills/irongraph/references/capabilities.md",
        include_str!(
            "../../../integrations/agent-plugins/irongraph/skills/irongraph/references/capabilities.md"
        ),
    ),
    (
        "skills/irongraph/references/cypher.md",
        include_str!(
            "../../../integrations/agent-plugins/irongraph/skills/irongraph/references/cypher.md"
        ),
    ),
];

const SKILL_ASSETS: &[(&str, &str)] = &[
    (
        "SKILL.md",
        include_str!("../../../integrations/agent-plugins/irongraph/skills/irongraph/SKILL.md"),
    ),
    (
        "references/capabilities.md",
        include_str!(
            "../../../integrations/agent-plugins/irongraph/skills/irongraph/references/capabilities.md"
        ),
    ),
    (
        "references/cypher.md",
        include_str!(
            "../../../integrations/agent-plugins/irongraph/skills/irongraph/references/cypher.md"
        ),
    ),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Host {
    Codex,
    Claude,
    Cursor,
    Copilot,
    Gemini,
    Kiro,
    Cline,
    Opencode,
    Roo,
    Continue,
    Windsurf,
    Zed,
    LmStudio,
    Warp,
    Goose,
    Hermes,
    Pi,
    Openclaw,
    Unsloth,
}

fn selected_hosts(target: &TargetArguments) -> Result<Vec<Host>> {
    if target.all {
        Ok(Host::ALL.to_vec())
    } else {
        target
            .host
            .map(|host| vec![host])
            .ok_or_else(|| anyhow!("select a host or pass --all"))
    }
}

fn print_statuses(statuses: &[HostStatus], json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(statuses)?);
        return Ok(());
    }
    for status in statuses {
        println!(
            "{:<19} {:<26} {:<20} {}",
            status.display_name,
            status.integration,
            status.state,
            if status.detected {
                "detected"
            } else {
                "not detected"
            }
        );
    }
    Ok(())
}

fn standard_mcp_entry() -> Value {
    json!({"command": "irongraph-mcp", "args": [], "env": {}})
}

fn verified_mcp_command(configured: Option<&Path>, current: Option<&Path>) -> Result<String> {
    let candidate = configured.or_else(|| {
        current.filter(|path| path.file_name().is_some_and(|name| name == "irongraph-mcp"))
    });
    let Some(path) = candidate else {
        // Library tests do not run inside the standalone MCP executable.
        return Ok("irongraph-mcp".to_owned());
    };
    if !path.is_absolute() || !path.is_file() {
        bail!("IRONGRAPH_MCP_BINARY must name an existing absolute executable path");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(path)?.permissions().mode() & 0o111 == 0 {
            bail!("IRONGRAPH_MCP_BINARY must name an executable file");
        }
    }
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("the MCP executable path must be UTF-8"))
}

fn loopback_query_url(configured: Option<&str>) -> Result<String> {
    let value = configured
        .unwrap_or("http://127.0.0.1:18484")
        .trim_end_matches('/');
    let authority = value
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("local integration IRONGRAPH_MCP_URL must use loopback HTTP"))?;
    let address = authority
        .parse::<std::net::SocketAddr>()
        .ok()
        .filter(|address| address.ip().is_loopback() && address.port() != 0);
    let localhost = authority
        .strip_prefix("localhost:")
        .and_then(|port| port.parse::<u16>().ok())
        .is_some_and(|port| port != 0);
    if address.is_none() && !localhost {
        bail!("local integration IRONGRAPH_MCP_URL must name a loopback host and nonzero port");
    }
    Ok(value.to_owned())
}

fn streamable_mcp_url(configured: Option<&str>) -> Result<String> {
    let address: std::net::SocketAddr = configured
        .unwrap_or("127.0.0.1:18488")
        .parse()
        .context("IRONGRAPH_MCP_ADDR must be a loopback socket address")?;
    if !address.ip().is_loopback() || address.port() == 0 {
        bail!("IRONGRAPH_MCP_ADDR must use loopback and a nonzero port");
    }
    Ok(format!("http://{address}/mcp"))
}

fn resolve_mcp_launch() -> Result<(String, String)> {
    let configured = env::var_os("IRONGRAPH_MCP_BINARY").map(PathBuf::from);
    let current = env::current_exe().ok();
    let command = verified_mcp_command(configured.as_deref(), current.as_deref())?;
    let configured_url = env::var("IRONGRAPH_MCP_URL").ok().or_else(|| {
        env::var("IRONGRAPH_HTTP_ADDR")
            .ok()
            .map(|address| format!("http://{address}"))
    });
    let query_url = loopback_query_url(configured_url.as_deref())?;
    Ok((command, query_url))
}

fn localize_mcp_config(value: &mut Value, command: &str, query_url: &str) {
    match value {
        Value::Object(object) => {
            let array_command =
                object
                    .get("command")
                    .and_then(Value::as_array)
                    .is_some_and(|arguments| {
                        arguments.first().and_then(Value::as_str) == Some("irongraph-mcp")
                    });
            if object.get("command").and_then(Value::as_str) == Some("irongraph-mcp")
                || array_command
            {
                if array_command {
                    object
                        .get_mut("command")
                        .and_then(Value::as_array_mut)
                        .expect("command array")[0] = json!(command);
                } else {
                    object.insert("command".to_owned(), json!(command));
                }
                let key = if array_command { "environment" } else { "env" };
                let environment = object.entry(key).or_insert_with(|| json!({}));
                if let Some(environment) = environment.as_object_mut() {
                    environment.insert("IRONGRAPH_MCP_URL".to_owned(), json!(query_url));
                }
            }
            for child in object.values_mut() {
                localize_mcp_config(child, command, query_url);
            }
        }
        Value::Array(values) => {
            for child in values {
                localize_mcp_config(child, command, query_url);
            }
        }
        _ => {}
    }
}

fn versioned_asset(relative: &str, contents: &str) -> Result<String> {
    let (command, query_url) = resolve_mcp_launch()?;
    versioned_asset_with_launch(relative, contents, &command, &query_url)
}

fn versioned_asset_with_launch(
    relative: &str,
    contents: &str,
    command: &str,
    query_url: &str,
) -> Result<String> {
    if !relative.ends_with(".json") {
        return Ok(contents.to_owned());
    }
    let mut value: Value =
        serde_json::from_str(contents).with_context(|| format!("parsing embedded {relative}"))?;
    if let Some(version) = value
        .as_object_mut()
        .and_then(|object| object.get_mut("version"))
    {
        *version = json!(INTEGRATION_VERSION);
    }
    localize_mcp_config(&mut value, command, query_url);
    Ok(serde_json::to_string_pretty(&value)?)
}

fn insert_nested_server(
    value: &mut Value,
    parent: &[&str],
    name: &str,
    entry: Value,
) -> Result<()> {
    let mut cursor = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("host configuration root must be a JSON object"))?;
    for segment in parent {
        let nested = cursor
            .entry((*segment).to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        cursor = nested
            .as_object_mut()
            .ok_or_else(|| anyhow!("host configuration field {segment} must be a JSON object"))?;
    }
    cursor.insert(name.to_owned(), entry);
    Ok(())
}

fn read_json_object(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(Value::Object(Map::new()));
    }
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let value: Value =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    if !value.is_object() {
        bail!("{} must contain a JSON object", path.display());
    }
    Ok(value)
}

fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if path.exists() {
        let backup = backup_path(path);
        if !backup.exists() {
            fs::copy(path, &backup).with_context(|| format!("backing up {}", path.display()))?;
        }
    }
    let temporary = temporary_path(path);
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("writing {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("replacing {}", path.display()))
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut file_name = path
        .file_name()
        .map(OsString::from)
        .unwrap_or_else(|| OsString::from("integration"));
    file_name.push(".irongraph.tmp");
    path.with_file_name(file_name)
}

fn backup_path(path: &Path) -> PathBuf {
    let mut file_name = path
        .file_name()
        .map(OsString::from)
        .unwrap_or_else(|| OsString::from("integration"));
    file_name.push(".irongraph-backup");
    path.with_file_name(file_name)
}

fn path_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn version_is_older(installed: &str, available: &str) -> bool {
    parse_version(installed) < parse_version(available)
}

fn parse_version(version: &str) -> (u64, u64, u64) {
    let mut parts = version
        .split_once('-')
        .map_or(version, |(stable, _)| stable)
        .split('.')
        .map(|part| part.parse::<u64>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

fn user_root() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn command_exists(command: &str) -> bool {
    env::var_os("PATH")
        .map(|paths| env::split_paths(&paths).any(|path| path.join(command).is_file()))
        .unwrap_or(false)
}

fn run_host_command(program: &str, arguments: &[&str]) -> Result<()> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .with_context(|| format!("starting {program}"))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        bail!("{program} exited with {}: {}", output.status, detail.trim());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        env::temp_dir().join(format!(
            "irongraph-mcp-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn integrations_install_package_and_record_current_version() {
        let root = temporary_root("install");
        let command = IntegrationCommand::new(Some(root.clone()));
        let status = command
            .install(Host::Hermes, false)
            .expect("install Hermes");

        assert_eq!(
            status.installed_version.as_deref(),
            Some(INTEGRATION_VERSION)
        );
        assert!(root.join(".hermes/plugins/irongraph/plugin.json").is_file());
        assert!(
            root.join(".hermes/plugins/irongraph/skills/irongraph/SKILL.md")
                .is_file()
        );
        fs::remove_dir_all(root).expect("remove temporary root");
    }

    #[test]
    fn integrations_preserve_unrelated_json_and_create_backup() {
        let root = temporary_root("merge");
        let config = root.join(".pi/agent/mcp.json");
        fs::create_dir_all(config.parent().expect("config parent")).expect("create config parent");
        fs::write(
            &config,
            r#"{"theme":"ink","mcpServers":{"existing":{"command":"other"}}}"#,
        )
        .expect("write existing config");

        IntegrationCommand::new(Some(root.clone()))
            .install(Host::Pi, false)
            .expect("install Pi");
        let value = read_json_object(&config).expect("read merged config");

        assert_eq!(value["theme"], "ink");
        assert_eq!(value["mcpServers"]["existing"]["command"], "other");
        assert_eq!(value["mcpServers"]["irongraph"]["command"], "irongraph-mcp");
        assert!(backup_path(&config).is_file());
        fs::remove_dir_all(root).expect("remove temporary root");
    }

    #[test]
    fn integrations_refuse_to_replace_malformed_configuration() {
        let root = temporary_root("malformed");
        let config = root.join(".lmstudio/mcp.json");
        fs::create_dir_all(config.parent().expect("config parent")).expect("create config parent");
        fs::write(&config, "not json").expect("write malformed config");

        let error = IntegrationCommand::new(Some(root.clone()))
            .install(Host::LmStudio, false)
            .expect_err("malformed config must fail");

        assert!(error.to_string().contains("parsing"));
        assert_eq!(
            fs::read_to_string(&config).expect("read original"),
            "not json"
        );
        fs::remove_dir_all(root).expect("remove temporary root");
    }

    #[test]
    fn integrations_report_and_repair_a_missing_managed_file() {
        let root = temporary_root("repair");
        let command = IntegrationCommand::new(Some(root.clone()));
        command
            .install(Host::Cursor, false)
            .expect("install Cursor");
        fs::remove_file(root.join(".cursor/plugins/local/irongraph/mcp.json"))
            .expect("remove managed MCP declaration");

        assert_eq!(
            command.status(Host::Cursor).expect("broken status").state,
            "repair-required"
        );
        command.install(Host::Cursor, false).expect("repair Cursor");
        assert_eq!(
            command.status(Host::Cursor).expect("repaired status").state,
            "current"
        );
        fs::remove_dir_all(root).expect("remove temporary root");
    }

    #[test]
    fn integrations_install_unsloth_streamable_http_import() {
        let root = temporary_root("unsloth");
        fs::create_dir_all(root.join(".unsloth/studio")).expect("create Unsloth marker");
        let command = IntegrationCommand::new(Some(root.clone()));
        assert!(command.status(Host::Unsloth).expect("status").detected);

        let status = command
            .install(Host::Unsloth, false)
            .expect("install Unsloth");
        let config =
            read_json_object(&root.join(".irongraph/integrations/unsloth/irongraph-mcp.json"))
                .expect("read Unsloth import");
        assert_eq!(config["mcpServers"]["irongraph"]["type"], "streamableHttp");
        assert_eq!(
            config["mcpServers"]["irongraph"]["url"],
            "http://127.0.0.1:18488/mcp"
        );
        assert_eq!(status.state, "activation-required");
        assert!(status.activation_instruction.is_some());
        fs::remove_dir_all(root).expect("remove temporary root");
    }

    #[test]
    fn integrations_expose_every_supported_host() {
        assert_eq!(Host::ALL.len(), 19);
        for host in Host::ALL {
            let status = IntegrationCommand::new(Some(temporary_root(host.slug())))
                .status(host)
                .expect("status");
            assert!(status.install_command.ends_with(host.slug()));
            assert!(status.update_command.ends_with(host.slug()));
        }
    }

    #[test]
    fn integrations_render_persistent_paths_and_custom_ports_in_every_json_asset() {
        let executable = "/private/application data/bin/v1-abcd/irongraph-mcp";
        let endpoint = "http://127.0.0.1:19584";
        let mut declarations = 0;
        for (relative, contents) in PACKAGE_ASSETS {
            let rendered = versioned_asset_with_launch(relative, contents, executable, endpoint)
                .expect("render package asset");
            if !relative.ends_with(".json") {
                assert_eq!(rendered, *contents);
                continue;
            }
            let value: Value = serde_json::from_str(&rendered).expect("rendered JSON");
            if let Some(entry) = value.pointer("/mcpServers/irongraph") {
                assert_eq!(entry["command"], executable);
                assert_eq!(entry["env"]["IRONGRAPH_MCP_URL"], endpoint);
                declarations += 1;
            }
            if value.get("version").is_some() {
                assert_eq!(value["version"], INTEGRATION_VERSION);
            }
        }
        assert_eq!(declarations, 3);
        for mut entry in [
            standard_mcp_entry(),
            json!({"type": "local", "command": ["irongraph-mcp"], "enabled": true}),
            json!({"command": "irongraph-mcp", "args": [], "lifecycle": "eager"}),
        ] {
            let array_command = entry["command"].is_array();
            localize_mcp_config(&mut entry, executable, endpoint);
            if array_command {
                assert_eq!(entry["command"][0], executable);
                assert_eq!(entry["environment"]["IRONGRAPH_MCP_URL"], endpoint);
            } else {
                assert_eq!(entry["command"], executable);
                assert_eq!(entry["env"]["IRONGRAPH_MCP_URL"], endpoint);
            }
        }
        let mut unrelated = json!({"command": "other", "env": {"existing": "retained"}});
        let original = unrelated.clone();
        localize_mcp_config(&mut unrelated, executable, endpoint);
        assert_eq!(unrelated, original);
    }

    #[test]
    fn integrations_validate_executables_without_modifying_process_environment() {
        let root = temporary_root("executable");
        fs::create_dir_all(&root).expect("temporary directory");
        let executable = root.join("irongraph-mcp");
        fs::write(&executable, "fixture").expect("fixture file");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert!(verified_mcp_command(Some(&executable), None).is_err());
            fs::set_permissions(&executable, fs::Permissions::from_mode(0o700))
                .expect("executable mode");
        }
        assert_eq!(
            verified_mcp_command(Some(&executable), None).expect("configured executable"),
            executable.to_str().expect("UTF-8")
        );
        assert_eq!(
            verified_mcp_command(None, Some(&executable)).expect("current executable"),
            executable.to_str().expect("UTF-8")
        );
        assert!(verified_mcp_command(Some(Path::new("irongraph-mcp")), Some(&executable)).is_err());
        assert!(verified_mcp_command(Some(&root.join("missing")), Some(&executable)).is_err());
        assert!(verified_mcp_command(Some(&root), None).is_err());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn integrations_validate_custom_loopback_endpoints() {
        assert_eq!(
            loopback_query_url(Some("http://127.0.0.1:19584/")).expect("custom URL"),
            "http://127.0.0.1:19584"
        );
        assert_eq!(
            loopback_query_url(Some("http://[::1]:19584")).expect("IPv6 URL"),
            "http://[::1]:19584"
        );
        assert_eq!(
            streamable_mcp_url(Some("127.0.0.1:19588")).expect("custom MCP"),
            "http://127.0.0.1:19588/mcp"
        );
        assert_eq!(
            streamable_mcp_url(Some("[::1]:19588")).expect("IPv6 MCP"),
            "http://[::1]:19588/mcp"
        );
        for bad in [
            "http://remote.example:19584",
            "http://user:secret@127.0.0.1:19584",
            "http://127.0.0.1:0",
            "http://127.0.0.1:19584/api/query",
        ] {
            assert!(loopback_query_url(Some(bad)).is_err());
        }
        for bad in ["0.0.0.0:19588", "127.0.0.1:0", "192.0.2.1:19588"] {
            assert!(streamable_mcp_url(Some(bad)).is_err());
        }
    }
}

impl Host {
    pub const ALL: [Self; 19] = [
        Self::Codex,
        Self::Claude,
        Self::Cursor,
        Self::Copilot,
        Self::Gemini,
        Self::Kiro,
        Self::Cline,
        Self::Opencode,
        Self::Roo,
        Self::Continue,
        Self::Windsurf,
        Self::Zed,
        Self::LmStudio,
        Self::Warp,
        Self::Goose,
        Self::Hermes,
        Self::Pi,
        Self::Openclaw,
        Self::Unsloth,
    ];

    pub const fn slug(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::Claude => "claude",
            Self::Cursor => "cursor",
            Self::Copilot => "copilot",
            Self::Gemini => "gemini",
            Self::Kiro => "kiro",
            Self::Cline => "cline",
            Self::Opencode => "opencode",
            Self::Roo => "roo",
            Self::Continue => "continue",
            Self::Windsurf => "windsurf",
            Self::Zed => "zed",
            Self::LmStudio => "lm-studio",
            Self::Warp => "warp",
            Self::Goose => "goose",
            Self::Hermes => "hermes",
            Self::Pi => "pi",
            Self::Openclaw => "openclaw",
            Self::Unsloth => "unsloth",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::Codex => "ChatGPT and Codex",
            Self::Claude => "Claude Code",
            Self::Cursor => "Cursor",
            Self::Copilot => "GitHub Copilot",
            Self::Gemini => "Gemini CLI",
            Self::Kiro => "Kiro",
            Self::Cline => "Cline",
            Self::Opencode => "OpenCode",
            Self::Roo => "Roo Code",
            Self::Continue => "Continue",
            Self::Windsurf => "Windsurf",
            Self::Zed => "Zed",
            Self::LmStudio => "LM Studio",
            Self::Warp => "Warp",
            Self::Goose => "Goose",
            Self::Hermes => "Hermes Agent",
            Self::Pi => "Pi",
            Self::Openclaw => "OpenClaw",
            Self::Unsloth => "Unsloth Studio",
        }
    }

    const fn integration(self) -> &'static str {
        match self {
            Self::Codex => "Codex plugin",
            Self::Claude => "Claude plugin",
            Self::Cursor | Self::Copilot => "Agent Plugin",
            Self::Gemini => "Gemini extension",
            Self::Kiro => "Kiro Power",
            Self::Cline | Self::Opencode | Self::Roo => "MCP + Agent Skill",
            Self::Continue | Self::Windsurf => "MCP + guidance",
            Self::Zed => "MCP context server",
            Self::LmStudio => "MCP integration",
            Self::Warp | Self::Goose => "staged MCP package",
            Self::Hermes | Self::Openclaw => "Agent Plugin",
            Self::Pi => "MCP extension + Agent Skill",
            Self::Unsloth => "Streamable HTTP MCP import",
        }
    }

    const fn requires_activation(self) -> bool {
        matches!(
            self,
            Self::Claude
                | Self::Warp
                | Self::Goose
                | Self::Hermes
                | Self::Pi
                | Self::Openclaw
                | Self::Unsloth
        )
    }

    const fn activation_instruction(self) -> Option<&'static str> {
        match self {
            Self::Unsloth => Some(
                "In Unsloth Studio choose MCP, Add custom MCP, then Import config and select ~/.irongraph/integrations/unsloth/irongraph-mcp.json.",
            ),
            _ => None,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum IntegrationAction {
    /// List supported hosts and the best available integration for each one.
    List(OutputArguments),
    /// Show installed and available package versions.
    Status(TargetArguments),
    /// Inject the current package into one host, or every supported host.
    Install(TargetArguments),
    /// Reinject hosts whose managed package is older than this IronGraph build.
    Update(TargetArguments),
}

#[derive(Debug, Args)]
pub struct OutputArguments {
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct TargetArguments {
    /// Host to inspect or change.
    #[arg(value_enum, required_unless_present = "all")]
    host: Option<Host>,
    /// Target every supported host.
    #[arg(long, conflicts_with = "host")]
    all: bool,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug)]
pub struct IntegrationCommand {
    root: PathBuf,
    activate_native_cli: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostStatus {
    host: Host,
    display_name: &'static str,
    integration: &'static str,
    detected: bool,
    state: &'static str,
    installed_version: Option<String>,
    available_version: &'static str,
    activation_required: bool,
    activation_instruction: Option<&'static str>,
    install_command: String,
    update_command: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ManagedState {
    host: String,
    version: String,
    managed_files: Vec<String>,
    activation_required: bool,
}

impl IntegrationCommand {
    pub fn new(root: Option<PathBuf>) -> Self {
        let activate_native_cli = root.is_none();
        Self {
            root: root.unwrap_or_else(user_root),
            activate_native_cli,
        }
    }

    pub fn run(&self, action: IntegrationAction) -> Result<()> {
        match action {
            IntegrationAction::List(output) => self.print_status(&Host::ALL, output.json),
            IntegrationAction::Status(target) => {
                self.print_status(&selected_hosts(&target)?, target.json)
            }
            IntegrationAction::Install(target) => {
                let statuses = selected_hosts(&target)?
                    .into_iter()
                    .map(|host| self.install(host, false))
                    .collect::<Result<Vec<_>>>()?;
                print_statuses(&statuses, target.json)
            }
            IntegrationAction::Update(target) => {
                let statuses = selected_hosts(&target)?
                    .into_iter()
                    .map(|host| self.install(host, true))
                    .collect::<Result<Vec<_>>>()?;
                print_statuses(&statuses, target.json)
            }
        }
    }

    fn print_status(&self, hosts: &[Host], json_output: bool) -> Result<()> {
        let statuses = hosts
            .iter()
            .copied()
            .map(|host| self.status(host))
            .collect::<Result<Vec<_>>>()?;
        print_statuses(&statuses, json_output)
    }

    fn status(&self, host: Host) -> Result<HostStatus> {
        let installed = self.read_state(host)?;
        let complete = installed
            .as_ref()
            .is_none_or(|state| self.managed_installation_complete(state));
        let state = match installed.as_ref().map(|state| state.version.as_str()) {
            None => "not-installed",
            Some(_) if !complete => "repair-required",
            Some(INTEGRATION_VERSION) if host.requires_activation() => "activation-required",
            Some(INTEGRATION_VERSION) => "current",
            Some(version) if version_is_older(version, INTEGRATION_VERSION) => "update-available",
            Some(_) => "newer-than-runtime",
        };
        Ok(HostStatus {
            host,
            display_name: host.display_name(),
            integration: host.integration(),
            detected: self.detected(host),
            state,
            installed_version: installed.map(|state| state.version),
            available_version: INTEGRATION_VERSION,
            activation_required: host.requires_activation(),
            activation_instruction: host.activation_instruction(),
            install_command: format!("irongraph-mcp integrations install {}", host.slug()),
            update_command: format!("irongraph-mcp integrations update {}", host.slug()),
        })
    }

    fn install(&self, host: Host, update_only: bool) -> Result<HostStatus> {
        let before = self.status(host)?;
        if update_only && !matches!(before.state, "update-available" | "repair-required") {
            return Ok(before);
        }

        let mut managed_files = Vec::new();
        match host {
            Host::Codex => self.install_codex(&mut managed_files)?,
            Host::Claude => self.install_claude(&mut managed_files)?,
            Host::Cursor => self.install_package(
                Path::new(".cursor/plugins/local/irongraph"),
                &mut managed_files,
            )?,
            Host::Copilot => self.install_package(
                Path::new(".copilot/installed-plugins/_direct/irongraph"),
                &mut managed_files,
            )?,
            Host::Gemini => self.install_package(
                Path::new(".gemini/extensions/irongraph"),
                &mut managed_files,
            )?,
            Host::Kiro => {
                self.install_package(Path::new(".kiro/powers/irongraph"), &mut managed_files)?
            }
            Host::Cline => self.install_skill_and_json(
                Path::new(".cline/skills/irongraph"),
                Path::new(".cline/mcp.json"),
                &["mcpServers"],
                standard_mcp_entry(),
                &mut managed_files,
            )?,
            Host::Opencode => self.install_skill_and_json(
                Path::new(".config/opencode/skills/irongraph"),
                Path::new(".config/opencode/opencode.json"),
                &["mcp"],
                json!({"type": "local", "command": ["irongraph-mcp"], "enabled": true}),
                &mut managed_files,
            )?,
            Host::Roo => self.install_skill_and_json(
                Path::new(".roo/skills/irongraph"),
                Path::new(".roo/mcp.json"),
                &["mcpServers"],
                standard_mcp_entry(),
                &mut managed_files,
            )?,
            Host::Continue => self.install_continue(&mut managed_files)?,
            Host::Windsurf => self.install_skill_and_json(
                Path::new(".codeium/windsurf/skills/irongraph"),
                Path::new(".codeium/windsurf/mcp_config.json"),
                &["mcpServers"],
                standard_mcp_entry(),
                &mut managed_files,
            )?,
            Host::Zed => self.install_json_only(
                Path::new(".config/zed/settings.json"),
                &["context_servers"],
                json!({"command": "irongraph-mcp", "args": [], "env": {}}),
                &mut managed_files,
            )?,
            Host::LmStudio => self.install_json_only(
                Path::new(".lmstudio/mcp.json"),
                &["mcpServers"],
                standard_mcp_entry(),
                &mut managed_files,
            )?,
            Host::Warp | Host::Goose => self.install_package(
                &PathBuf::from(".irongraph/integrations")
                    .join(host.slug())
                    .join("irongraph"),
                &mut managed_files,
            )?,
            Host::Hermes => {
                self.install_package(Path::new(".hermes/plugins/irongraph"), &mut managed_files)?
            }
            Host::Pi => self.install_skill_and_json(
                Path::new(".pi/agent/skills/irongraph"),
                Path::new(".pi/agent/mcp.json"),
                &["mcpServers"],
                json!({"command": "irongraph-mcp", "args": [], "lifecycle": "eager"}),
                &mut managed_files,
            )?,
            Host::Openclaw => self.install_package(
                Path::new(".irongraph/integrations/openclaw/irongraph"),
                &mut managed_files,
            )?,
            Host::Unsloth => self.install_unsloth(&mut managed_files)?,
        }
        self.write_state(host, managed_files)?;
        self.activate(host)?;
        self.status(host)
    }

    fn install_codex(&self, managed_files: &mut Vec<String>) -> Result<()> {
        self.install_package(
            Path::new(".agents/plugins/plugins/irongraph"),
            managed_files,
        )?;
        let marketplace = Path::new(".agents/plugins/marketplace.json");
        let entry = json!({
            "name": "irongraph",
            "source": {"source": "local", "path": "./plugins/irongraph"},
            "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
            "category": "Productivity"
        });
        self.merge_marketplace(marketplace, "personal", "Personal", entry)?;
        managed_files.push(path_text(marketplace));
        Ok(())
    }

    fn install_claude(&self, managed_files: &mut Vec<String>) -> Result<()> {
        let marketplace_root = Path::new(".irongraph/integrations/claude-marketplace");
        self.install_package(&marketplace_root.join("plugins/irongraph"), managed_files)?;
        let marketplace = marketplace_root.join(".claude-plugin/marketplace.json");
        let entry = json!({
            "name": "irongraph",
            "description": "Durable graph memory for Claude",
            "source": "./plugins/irongraph",
            "category": "development"
        });
        self.merge_marketplace(&marketplace, "irongraph-local", "IronGraph", entry)?;
        managed_files.push(path_text(&marketplace));
        Ok(())
    }

    fn install_continue(&self, managed_files: &mut Vec<String>) -> Result<()> {
        let mcp = Path::new(".continue/mcpServers/irongraph.json");
        self.write_managed(
            mcp,
            &versioned_asset(
                "irongraph.json",
                &serde_json::to_string_pretty(&json!({
                    "name": "IronGraph",
                    "version": INTEGRATION_VERSION,
                    "schema": "v1",
                    "mcpServers": {"irongraph": standard_mcp_entry()}
                }))?,
            )?,
        )?;
        managed_files.push(path_text(mcp));
        let rule = Path::new(".continue/rules/irongraph.md");
        self.write_managed(
            rule,
            include_str!("../../../integrations/agent-plugins/irongraph/GEMINI.md"),
        )?;
        managed_files.push(path_text(rule));
        Ok(())
    }

    fn install_unsloth(&self, managed_files: &mut Vec<String>) -> Result<()> {
        let config = Path::new(".irongraph/integrations/unsloth/irongraph-mcp.json");
        let url = streamable_mcp_url(env::var("IRONGRAPH_MCP_ADDR").ok().as_deref())?;
        self.write_managed(
            config,
            &serde_json::to_string_pretty(&json!({
                "irongraphPackageVersion": INTEGRATION_VERSION,
                "mcpServers": {
                    "irongraph": {
                        "type": "streamableHttp",
                        "url": url
                    }
                }
            }))?,
        )?;
        managed_files.push(path_text(config));
        Ok(())
    }

    fn install_package(&self, target: &Path, managed_files: &mut Vec<String>) -> Result<()> {
        for (relative, contents) in PACKAGE_ASSETS {
            let path = target.join(relative);
            self.write_managed(&path, &versioned_asset(relative, contents)?)?;
            managed_files.push(path_text(&path));
        }
        Ok(())
    }

    fn install_skill_and_json(
        &self,
        skill: &Path,
        config: &Path,
        parent: &[&str],
        entry: Value,
        managed_files: &mut Vec<String>,
    ) -> Result<()> {
        for (relative, contents) in SKILL_ASSETS {
            let path = skill.join(relative);
            self.write_managed(&path, contents)?;
            managed_files.push(path_text(&path));
        }
        self.install_json_only(config, parent, entry, managed_files)
    }

    fn install_json_only(
        &self,
        config: &Path,
        parent: &[&str],
        entry: Value,
        managed_files: &mut Vec<String>,
    ) -> Result<()> {
        let path = self.absolute(config);
        let mut value = read_json_object(&path)?;
        let mut entry = entry;
        let (command, query_url) = resolve_mcp_launch()?;
        localize_mcp_config(&mut entry, &command, &query_url);
        insert_nested_server(&mut value, parent, "irongraph", entry)?;
        write_json_atomic(&path, &value)?;
        managed_files.push(path_text(config));
        Ok(())
    }

    fn merge_marketplace(
        &self,
        relative: &Path,
        name: &str,
        display_name: &str,
        entry: Value,
    ) -> Result<()> {
        let path = self.absolute(relative);
        let mut value = if path.exists() {
            read_json_object(&path)?
        } else {
            json!({"name": name, "interface": {"displayName": display_name}, "plugins": []})
        };
        let plugins = value
            .as_object_mut()
            .and_then(|root| {
                root.entry("plugins")
                    .or_insert_with(|| json!([]))
                    .as_array_mut()
            })
            .ok_or_else(|| anyhow!("{} has a non-array plugins field", path.display()))?;
        plugins.retain(|candidate| candidate.get("name") != Some(&json!("irongraph")));
        plugins.push(entry);
        write_json_atomic(&path, &value)
    }

    fn write_managed(&self, relative: &Path, contents: &str) -> Result<()> {
        let path = self.absolute(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let temporary = temporary_path(&path);
        fs::write(&temporary, contents)
            .with_context(|| format!("writing {}", temporary.display()))?;
        fs::rename(&temporary, &path).with_context(|| format!("replacing {}", path.display()))
    }

    fn write_state(&self, host: Host, mut managed_files: Vec<String>) -> Result<()> {
        managed_files.sort();
        managed_files.dedup();
        let state = ManagedState {
            host: host.slug().to_owned(),
            version: INTEGRATION_VERSION.to_owned(),
            managed_files,
            activation_required: host.requires_activation(),
        };
        self.write_managed(
            &self.state_path(host),
            &serde_json::to_string_pretty(&state)?,
        )
    }

    fn read_state(&self, host: Host) -> Result<Option<ManagedState>> {
        let path = self.absolute(&self.state_path(host));
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))
            .map(Some)
    }

    fn managed_installation_complete(&self, state: &ManagedState) -> bool {
        !state.managed_files.is_empty()
            && state.managed_files.iter().all(|managed| {
                let path = Path::new(managed);
                path.components()
                    .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
                    && self.absolute(path).is_file()
            })
    }

    fn state_path(&self, host: Host) -> PathBuf {
        PathBuf::from(".irongraph/integrations/state").join(format!("{}.json", host.slug()))
    }

    fn absolute(&self, relative: &Path) -> PathBuf {
        self.root.join(relative)
    }

    fn detected(&self, host: Host) -> bool {
        let markers: &[&str] = match host {
            Host::Codex => &[".codex", ".agents/plugins"],
            Host::Claude => &[".claude"],
            Host::Cursor => &[".cursor"],
            Host::Copilot => &[".copilot"],
            Host::Gemini => &[".gemini"],
            Host::Kiro => &[".kiro"],
            Host::Cline => &[".cline"],
            Host::Opencode => &[".config/opencode"],
            Host::Roo => &[".roo"],
            Host::Continue => &[".continue"],
            Host::Windsurf => &[".codeium/windsurf"],
            Host::Zed => &[".config/zed"],
            Host::LmStudio => &[".lmstudio"],
            Host::Warp => &[".warp"],
            Host::Goose => &[".config/goose"],
            Host::Hermes => &[".hermes"],
            Host::Pi => &[".pi/agent"],
            Host::Openclaw => &[".openclaw"],
            Host::Unsloth => &[".unsloth"],
        };
        markers.iter().any(|marker| self.root.join(marker).exists())
    }

    fn activate(&self, host: Host) -> Result<()> {
        if !self.activate_native_cli {
            return Ok(());
        }
        match host {
            Host::Codex if command_exists("codex") => {
                run_host_command("codex", &["plugin", "add", "irongraph@personal"])?;
            }
            Host::Claude if command_exists("claude") => {
                let root = self.absolute(Path::new(".irongraph/integrations/claude-marketplace"));
                let root = root
                    .to_str()
                    .ok_or_else(|| anyhow!("Claude marketplace path is not UTF-8"))?;
                run_host_command("claude", &["plugin", "marketplace", "add", root])?;
                run_host_command(
                    "claude",
                    &["plugin", "install", "irongraph@irongraph-local"],
                )?;
            }
            Host::Hermes if command_exists("hermes") => {
                run_host_command("hermes", &["plugins", "enable", "irongraph"])?;
            }
            Host::Pi if command_exists("pi") => {
                run_host_command("pi", &["install", "npm:pi-mcp-extension"])?;
            }
            Host::Openclaw if command_exists("openclaw") => {
                let root = self.absolute(Path::new(".irongraph/integrations/openclaw/irongraph"));
                let root = root
                    .to_str()
                    .ok_or_else(|| anyhow!("OpenClaw bundle path is not UTF-8"))?;
                run_host_command("openclaw", &["plugins", "install", root])?;
            }
            _ => {}
        }
        Ok(())
    }
}
