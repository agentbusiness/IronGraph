import { existsSync, readFileSync } from 'node:fs';

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

const accelerator = readFileSync('crates/gpu/src/accelerator.rs', 'utf8');
const metal = readFileSync('crates/gpu/src/metal.rs', 'utf8');
const cuda = readFileSync('crates/gpu/src/cuda.rs', 'utf8');
const exports = readFileSync('crates/gpu/src/lib.rs', 'utf8');
const louvain = readFileSync('crates/gpu/src/accelerator/graph_louvain.rs', 'utf8');

const stageStart = accelerator.indexOf('pub fn stage_delta(');
const stageEnd = accelerator.indexOf('\n    fn detach_mutated_tensors(', stageStart);
requireCondition(stageStart >= 0 && stageEnd > stageStart, 'cannot resolve Metal delta staging body');
requireCondition(
  !accelerator.slice(stageStart, stageEnd).includes('synchronize()'),
  'Metal delta staging still has a pre-publication full-device barrier',
);

const applyStart = accelerator.indexOf('fn apply_delta_in_place(');
const applyEnd = accelerator.indexOf('\n    pub fn scan_nodes(', applyStart);
requireCondition(applyStart >= 0 && applyEnd > applyStart, 'cannot resolve delta application body');
requireCondition(
  !accelerator.slice(applyStart, applyEnd).includes('synchronize()'),
  'delta application still has an internal full-device barrier',
);
requireCondition(
  metal.includes('drop(old);\n            self.device.synchronize()'),
  'Metal atomic owner publication lost its completion/reclamation barrier',
);
requireCondition(!cuda.includes('CudaKernelContract'), 'dead CUDA kernel contract is still compiled');
requireCondition(!exports.includes('CudaKernelContract'), 'dead CUDA kernel contract is still exported');
requireCondition(!existsSync('kernels/cuda/operators.cu'), 'non-launched CUDA source metadata remains');
requireCondition(
  (louvain.match(/wait_until_completed/g) ?? []).length === 9,
  'Metal Louvain must retain exactly one completion boundary per custom operation',
);

console.log('GPU dispatch verification passed');
