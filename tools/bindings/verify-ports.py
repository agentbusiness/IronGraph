#!/usr/bin/env python3
"""Check default endpoint coherence without scanning stored dataset measurements."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
PORTS = {"HTTP": 18484, "BOLT": 18485, "STREAM": 18486, "QUEUE": 18487, "MCP": 18488}
OLD_ENDPOINT = re.compile(r":(?:8484|8485|8490|7687|7699|9092|5672|5173|28484|27687)(?!\d)")
SURFACES = (
    "src/main.rs", "start.sh", "crates/server/src/config.rs",
    "crates/server/src/protocol/bolt.rs", "crates/mcp/src/main.rs",
    "crates/mcp/src/integrations.rs", "crates/client/tests/transports.rs",
    "web/vite.config.ts", "bindings/javascript/src/react.test.tsx",
    "bindings/javascript/src/index.test.ts", "bindings/javascript/README.md",
    "bindings/node/README.md", "bindings/python/README.md", "bindings/rust/README.md",
    "docs/embedding.md", "docs/getting-started.md", "docs/mcp.md",
    "integrations/agent-plugins/irongraph/README.md", "scripts/verify-agent-integrations.mjs",
    "scripts/perf/verify-real-graph-routes.py", "scripts/perf/real-graph-matrix.mjs",
    "datasets/irongraph_client.py", "README.md", "docs/installation.md",
    "bindings/cli/cli.cjs", "bindings/cli/README.md",
)


def read(path):
    return (ROOT / path).read_text()


def require(pattern, text, description):
    if re.search(pattern, text) is None:
        raise AssertionError(description)


def main():
    # Positive and negative controls ensure obsolete-endpoint detection cannot pass vacuously.
    for port in (8484, 8485, 8490, 7687, 7699, 9092, 5672, 5173, 28484, 27687):
        assert OLD_ENDPOINT.search(f"http://127.0.0.1:{port}/web/")
        assert OLD_ENDPOINT.search(f"[::1]:{port}")
    assert not OLD_ENDPOINT.search("http://127.0.0.1:18484/web/")
    assert not OLD_ENDPOINT.search("measurement = 18484.8484")
    assert not OLD_ENDPOINT.search("127.0.0.1:84840")

    config, shell = read("crates/server/src/config.rs"), read("start.sh")
    launcher = read("bindings/cli/cli.cjs")
    for name, port in PORTS.items():
        require(rf'export IRONGRAPH_{name}_ADDR="\$\{{IRONGRAPH_{name}_ADDR:-127\.0\.0\.1:{port}\}}"',
                shell, f"developer launcher {name} default differs from {port}")
        require(rf"IRONGRAPH_{name}_ADDR: '127\.0\.0\.1:{port}'",
                launcher, f"npm launcher {name} default differs from {port}")
        if name != "MCP":
            require(rf'env = "IRONGRAPH_{name}_ADDR", default_value = "127\.0\.0\.1:{port}"',
                    config, f"native {name} default differs from {port}")
    require(r'IRONGRAPH_MCP_ADDR"\)\s*\.unwrap_or_else\(\|_\| "127\.0\.0\.1:18488"',
            read("src/main.rs"), "native MCP default differs")
    require(r'default_value = "http://127\.0\.0\.1:18484"',
            read("crates/mcp/src/main.rs"), "MCP client default differs")
    require(r"IRONGRAPH_SERVER \?\? 'http://127\.0\.0\.1:18484'",
            read("web/vite.config.ts"), "Vite proxy default differs")
    require(r"server:\s*\{\s*host: '127\.0\.0\.1',\s*port: 18489,\s*strictPort: true",
            read("web/vite.config.ts"), "Vite must bind loopback18489 without port fallback")
    assert "IRONGRAPH_NODE_ADDR" not in shell
    assert len(set(PORTS.values()) | {18489}) == 6
    assert all(1024 <= port < 32768 for port in (*PORTS.values(), 18489))

    for path in SURFACES:
        for number, line in enumerate(read(path).splitlines(), 1):
            if OLD_ENDPOINT.search(line):
                raise AssertionError(f"obsolete endpoint in {path}:{number}: {line.strip()}")
    print("default ports verified")


if __name__ == "__main__":
    main()
