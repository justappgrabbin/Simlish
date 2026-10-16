/** Dependency-aware, bounded-concurrency DAG executor with checkpoints. */

function validateGraph(nodes) {
  const ids = new Set(nodes.map((n) => n.id));
  if (ids.size !== nodes.length) throw new Error('DAG node ids must be unique');
  for (const node of nodes) {
    for (const dep of node.dependsOn ?? []) {
      if (!ids.has(dep)) throw new Error(`node ${node.id} depends on missing node ${dep}`);
    }
  }

  const visiting = new Set();
  const visited = new Set();
  const byId = new Map(nodes.map((n) => [n.id, n]));
  function visit(id) {
    if (visited.has(id)) return;
    if (visiting.has(id)) throw new Error(`cycle detected at ${id}`);
    visiting.add(id);
    for (const dep of byId.get(id).dependsOn ?? []) visit(dep);
    visiting.delete(id); visited.add(id);
  }
  for (const id of ids) visit(id);
}

export async function executeDAG(nodes, executor, options = {}) {
  validateGraph(nodes);
  const concurrency = Math.max(1, options.concurrency ?? 4);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const status = new Map(nodes.map((n) => [n.id, 'pending']));
  const results = new Map();
  const errors = new Map();
  const checkpoints = [];
  let running = 0;

  const checkpoint = () => {
    const cp = {
      timestamp: Date.now(),
      status: Object.fromEntries(status),
      results: Object.fromEntries(results),
      errors: Object.fromEntries([...errors].map(([k, e]) => [k, String(e?.message ?? e)])),
    };
    checkpoints.push(structuredClone(cp));
    options.onCheckpoint?.(structuredClone(cp));
  };

  return await new Promise((resolve, reject) => {
    const pump = () => {
      const pending = [...status].filter(([, s]) => s === 'pending').map(([id]) => id);
      const runnable = pending.filter((id) => {
        const node = byId.get(id);
        return (node.dependsOn ?? []).every((dep) => status.get(dep) === 'done');
      });

      while (running < concurrency && runnable.length) {
        const id = runnable.shift();
        const node = byId.get(id);
        status.set(id, 'running'); running++;
        Promise.resolve(executor(node, {
          results: Object.fromEntries(results),
          dependencies: Object.fromEntries((node.dependsOn ?? []).map((dep) => [dep, results.get(dep)])),
        }))
          .then((result) => {
            results.set(id, result); status.set(id, 'done');
          })
          .catch((error) => {
            errors.set(id, error); status.set(id, 'failed');
            for (const [otherId, otherNode] of byId) {
              if (status.get(otherId) === 'pending' && (otherNode.dependsOn ?? []).includes(id)) {
                status.set(otherId, 'blocked');
              }
            }
          })
          .finally(() => {
            running--; checkpoint(); pump();
          });
      }

      const terminal = [...status.values()].every((s) => ['done', 'failed', 'blocked'].includes(s));
      if (terminal && running === 0) {
        const output = {
          ok: errors.size === 0,
          status: Object.fromEntries(status),
          results: Object.fromEntries(results),
          errors: Object.fromEntries([...errors].map(([k, e]) => [k, String(e?.stack ?? e)])),
          checkpoints,
        };
        if (!output.ok && options.continueOnError === false) reject(Object.assign(new Error('DAG execution failed'), { output }));
        else resolve(output);
        return;
      }

      // No runnable work + nothing running means failed/blocked dependency state.
      if (running === 0 && runnable.length === 0 && pending.length > 0) {
        for (const id of pending) status.set(id, 'blocked');
        pump();
      }
    };

    checkpoint();
    pump();
  });
}
