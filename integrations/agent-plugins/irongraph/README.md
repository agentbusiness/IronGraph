# IronGraph agent integration package

This package connects a host agent to the local `irongraph-mcp` executable and teaches it to use IronGraph as durable graph memory.

The portable Agent Plugins files are `plugin.json`, `mcp.json`, and `skills/`. Native manifests add richer installation for Codex/ChatGPT, Claude Code, Cursor, and Gemini CLI without changing the MCP tools or behavior.

Prerequisites: install IronGraph and `irongraph-mcp`, start the database process, and keep the safe local listener at `http://127.0.0.1:18484`. For a remote instance, configure the MCP process with `IRONGRAPH_MCP_URL` and all three mutual-TLS file variables.
