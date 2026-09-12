import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const read = (path) => readFileSync(resolve(root, path), 'utf8');
const workspaceVersion = /^version = "([^"]+)"/m.exec(read('Cargo.toml'))?.[1];
if (!workspaceVersion) throw new Error('workspace package version is missing');

for (const path of [
  'integrations/agent-plugins/irongraph/plugin.json',
  'integrations/agent-plugins/irongraph/.codex-plugin/plugin.json',
  'integrations/agent-plugins/irongraph/.claude-plugin/plugin.json',
  'integrations/agent-plugins/irongraph/gemini-extension.json',
]) {
  const version = JSON.parse(read(path)).version;
  if (version !== workspaceVersion) throw new Error(`${path} is ${version}; expected ${workspaceVersion}`);
}

const skill = read('integrations/agent-plugins/irongraph/skills/irongraph/SKILL.md');
for (const capability of ['irongraph_run_cypher', 'irongraph_search', 'irongraph_save_document', 'irongraph_get_schema', 'irongraph_search_cypher_docs']) {
  if (!skill.includes(capability)) throw new Error(`skill does not route ${capability}`);
}
for (const phrase of ['any indexed node label', 'relationships and neighboring nodes', 'durable second brain']) {
  if (!skill.includes(phrase)) throw new Error(`skill is missing: ${phrase}`);
}

const installer = read('crates/mcp/src/integrations.rs');
const server = read('crates/server/src/server/http.rs');
const settings = read('web/src/components/settings/SettingsPage.tsx');
const app = read('web/src/App.tsx');
const mcpDocs = read('docs/mcp.md');
for (const host of ['hermes', 'pi', 'openclaw', 'unsloth']) {
  if (!installer.includes(`"${host}"`) || !server.includes(`"${host}"`)) throw new Error(`${host} is not wired through installer and local Settings API`);
}
if (settings.includes('Ollama')) throw new Error('Ollama explanation must not be baked into Settings');
if (!mcpDocs.includes('Unsloth Studio') || !mcpDocs.includes('Streamable HTTP')) throw new Error('Unsloth Studio MCP documentation is missing');
if (!installer.includes('http://127.0.0.1:18488/mcp')) throw new Error('Unsloth import does not target the local HTTP MCP listener');
for (const action of ["'install'", "'update'", "'repair'"]) {
  if (!settings.includes(action)) throw new Error(`Settings action ${action} is missing`);
}
if (!app.includes("settings: '/web/settings'")) throw new Error('top-level Settings route is missing');
if (!server.includes('"/system/local-ai-integrations"')) throw new Error('local Settings status endpoint is missing');

console.log(`agent integration package verification passed (${workspaceVersion})`);
