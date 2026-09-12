#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const mode = process.argv[2];
const version = readFileSync(join(root, "Cargo.toml"), "utf8").match(/^version\s*=\s*"([^"]+)"/m)?.[1];
const targets = ["aarch64-apple-darwin", "aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"];
const suffixes = ["darwin-arm64", "linux-arm64-gnu", "linux-x64-gnu"];

function fail(message) {
  console.error(message);
  process.exit(1);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: root, encoding: "utf8", ...options });
  if (result.status !== 0) fail((result.stderr || result.stdout || `${command} failed`).trim());
  return result.stdout;
}

function json(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function packageManifest(path) {
  const manifest = json(join(root, path));
  if (manifest.version !== version || manifest.license !== "Apache-2.0") fail(`${path}: version or license mismatch`);
  if (!manifest.description.toLowerCase().includes("embed")) fail(`${path}: embedding is not prominent in the description`);
  if (manifest.repository?.url !== "git+https://github.com/agentbusiness/IronGraph.git") fail(`${path}: repository URL mismatch`);
  return manifest;
}

function releaseStage() {
  const stage = join(root, "target", "releases", `v${version}`);
  const statePath = join(stage, "state.json");
  if (!existsSync(statePath)) fail(`Missing staged release: ${stage}`);
  const state = json(statePath);
  if (state.version !== version || !state.checksums) fail("The staged release has not completed build and package verification");
  return { stage, state, publication: join(stage, "publish") };
}

function digest(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function verifyChecksums(publication, state) {
  const names = readdirSync(publication).filter((name) => statSync(join(publication, name)).isFile()).sort();
  const expected = Object.keys(state.checksums).sort();
  if (JSON.stringify(names) !== JSON.stringify(expected)) fail("Publication inventory differs from the reviewed release state");
  for (const [name, expectedDigest] of Object.entries(state.checksums)) {
    if (digest(join(publication, name)) !== expectedDigest) fail(`${name}: checksum mismatch`);
  }
  return names;
}

function metadata() {
  run("python3", ["tools/release/check_boundary.py"]);
  for (const path of ["bindings/cli/package.json", "bindings/javascript/package.json", "bindings/node/package.json"]) packageManifest(path);
  const cli = json(join(root, "bindings/cli/package.json"));
  const node = json(join(root, "bindings/node/package.json"));
  if (Object.keys(cli.optionalDependencies).sort().join() !== suffixes.map((value) => `@irongraph/cli-${value}`).sort().join()) fail("CLI target packages mismatch");
  if (node.napi.targets.join() !== targets.join()) fail("Node target matrix mismatch");
  for (const path of ["bindings/python/pyproject.toml", "bindings/rust/Cargo.toml"]) {
    const text = readFileSync(join(root, path), "utf8");
    if (!text.includes(`version = "${version}"`) || !text.includes('license = "Apache-2.0"') || !text.toLowerCase().includes("embed") || !text.includes("https://github.com/agentbusiness/IronGraph")) fail(`${path}: release metadata mismatch`);
  }
  console.log("release metadata verification passed");
}

function archives() {
  const { publication, state } = releaseStage();
  const names = verifyChecksums(publication, state);
  const required = [
    `irongraph-${version}.tgz`, `irongraph-node-${version}.tgz`, `irongraph-client-${version}.tgz`,
    `irongraph-sdk-${version}.crate`, "native-manifest.json", "qualification.json", "SHA256SUMS",
    ...targets.flatMap((target) => [`libirongraph_ffi-${target}.a`, `irongraph-${version}-${target}.tar.gz`]),
    ...suffixes.flatMap((suffix) => [`irongraph-cli-${suffix}-${version}.tgz`, `irongraph-node-${suffix}-${version}.tgz`]),
  ];
  for (const name of required) if (!names.includes(name)) fail(`Missing release artifact: ${name}`);
  if (names.filter((name) => name.endsWith(".whl")).length !== 3) fail("Expected one Python wheel for each supported target");
  const audit = `import importlib.util,pathlib\np=pathlib.Path(r'${join(root, "tools/release/release.py")}')\ns=importlib.util.spec_from_file_location('release',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nd=pathlib.Path(r'${publication}')\nfor x in d.iterdir():\n k='wheel' if x.suffix=='.whl' else 'cargo' if x.suffix=='.crate' else 'standalone' if x.name.endswith('.tar.gz') else 'npm' if x.suffix=='.tgz' else None\n if k:m.audit_archive(x,k,'${version}')`;
  run("python3", ["-c", audit]);
  console.log("release archive verification passed");
}

function rustSdk() {
  const { publication, state } = releaseStage();
  verifyChecksums(publication, state);
  const manifest = json(join(publication, "native-manifest.json"));
  if (manifest.version !== version || Object.keys(manifest.targets).sort().join() !== targets.sort().join()) fail("Rust SDK native manifest target mismatch");
  const listing = run("tar", ["-tzf", join(publication, `irongraph-sdk-${version}.crate`)]);
  if (!listing.includes("native-manifest.json") || !listing.includes("src/lib.rs")) fail("Rust SDK archive inventory is incomplete");
  console.log("Rust SDK package verification passed");
}

function pythonPackage() {
  const { publication, state } = releaseStage();
  const names = verifyChecksums(publication, state);
  if (names.filter((name) => name.endsWith(".whl")).length !== 3) fail("Python wheel matrix is incomplete");
  const qualification = json(join(publication, "qualification.json"));
  for (const target of targets) if (qualification.targets?.[target]?.embedding_search !== true) fail(`${target}: Python embedding smoke qualification missing`);
  console.log("Python package verification passed");
}

function driver() {
  const plan = run("./release.sh", ["--dry-run"]);
  if (!plan.includes(`v${version}`) || !plan.includes("macOS ARM64") || !plan.includes("Linux ARM64/AMD64")) fail("Release plan does not cover the supported matrix");
  const source = readFileSync(join(root, "tools/release/release.py"), "utf8");
  if (!source.includes("parser.set_defaults(build_only=True)") || !source.includes('mode.add_argument("--publish"')) fail("Release driver does not default to build-only");
  console.log("release driver verification passed");
}

function regression() {
  run("python3", ["-m", "unittest", "discover", "-s", "tools/release/tests", "-v"], { stdio: "inherit" });
  run("python3", ["tools/release/check_boundary.py"], { stdio: "inherit" });
  run("node", ["--test", "bindings/cli/test/lifecycle.test.cjs"], { stdio: "inherit" });
  console.log("release regression verification passed");
}

const operations = { metadata, archives, "rust-sdk": rustSdk, python: pythonPackage, driver, regression };
if (!operations[mode]) fail("Usage: verify-release.mjs metadata|archives|rust-sdk|python|driver|regression");
operations[mode]();
