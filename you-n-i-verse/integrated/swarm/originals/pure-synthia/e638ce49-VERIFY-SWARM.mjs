import { spawnSync } from 'node:child_process';
const files = ['tests/swarm-body.node.mjs','tests/current-synthia-swarm.node.mjs','tests/full-wiring.node.mjs','tests/physiology-wiring.node.mjs','tests/world-port.node.mjs','tests/durable-restart.node.mjs'];
let failed = false;
for (const file of files) {
  const r = spawnSync(process.execPath, ['--test', file], { stdio: 'inherit' });
  if (r.status !== 0) failed = true;
}
if (failed) process.exit(1);
console.log('SYNTHIA SWARM VERIFY: PASS');
