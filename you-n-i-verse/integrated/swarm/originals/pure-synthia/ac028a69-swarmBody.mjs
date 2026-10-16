import { TaskAssignmentEngine } from '../orchestrator/taskAssignment.mjs';
import { ProcessAdapter } from './processAdapter.mjs';
import { MemoryCheckpointStore } from './checkpointStores.mjs';

const clone = (value) => {
  try { return structuredClone(value); }
  catch { try { return JSON.parse(JSON.stringify(value)); } catch { return null; } }
};
const nowIso = () => new Date().toISOString();

function entriesOfMesh(mesh) {
  if (mesh?.automatons instanceof Map) return [...mesh.automatons.entries()];
  if (mesh?.automata instanceof Map) return [...mesh.automata.entries()];
  return [];
}

export class SynthiaSwarmBody extends EventTarget {
  constructor({ identity = { id: 'synthia', name: 'Synthia' }, store = new MemoryCheckpointStore(), checkpointKey = 'synthia-swarm-root' } = {}) {
    super();
    this.identity = Object.freeze({ ...identity });
    this.store = store;
    this.checkpointKey = checkpointKey;
    this.workers = new Map();
    this.assigner = new TaskAssignmentEngine();
    this.pending = [];
    this.inflight = new Map();
    this.recovered = [];
    this.completed = [];
    this.events = [];
    this.cycle = 0;
    this.bootCount = 0;
    this.lastWakeAt = null;
    this.lastCheckpointAt = null;
  }

  #event(type, detail = {}) {
    const event = Object.freeze({ seq: this.events.length + 1, type, at: nowIso(), detail: clone(detail) });
    this.events.push(event); if (this.events.length > 256) this.events.splice(0, this.events.length - 256);
    this.dispatchEvent(new CustomEvent('swarm', { detail: event }));
    return event;
  }

  #uniqueId(id, target) {
    if (!this.workers.has(id)) return id;
    if (this.workers.get(id)?.target === target) return id;
    let n = 2; let candidate = `${id}#${n}`;
    while (this.workers.has(candidate)) candidate = `${id}#${++n}`;
    return candidate;
  }

  registerTarget(target, { id = target?.id, group = 'general', capabilities = null, address = undefined, maxConcurrency = 1, location = 'local', metadata = {} } = {}) {
    if (!id) throw new TypeError('registerTarget requires a target id');
    const workerId = this.#uniqueId(String(id), target);
    if (this.workers.has(workerId) && this.workers.get(workerId).target === target) return this.workers.get(workerId);
    const worker = new ProcessAdapter({ id: workerId, target, group, capabilities, address, maxConcurrency, location, metadata: { nativeId: String(id), ...metadata } });
    this.workers.set(worker.id, worker);
    this.#refreshAssignments();
    this.#event('worker:registered', worker.manifest());
    return worker;
  }

  registerExternal({ id, capabilities, execute, group = 'hands', address = null, maxConcurrency = 1, location = 'host', metadata = {} } = {}) {
    const workerId = this.#uniqueId(String(id), execute);
    const worker = new ProcessAdapter({ id: workerId, target: null, capabilities, group, address, maxConcurrency, location, metadata, execute });
    this.workers.set(worker.id, worker);
    this.#refreshAssignments();
    this.#event('worker:registered', worker.manifest());
    return worker;
  }

  registerMesh(mesh, { group = 'mesh', location = 'local', maxConcurrency = 1 } = {}) {
    const registered = [];
    for (const [id, target] of entriesOfMesh(mesh)) registered.push(this.registerTarget(target, { id, group, location, maxConcurrency, metadata: { source: 'mesh' } }));
    return registered;
  }

  registerMap(map, { group = 'map', location = 'local', maxConcurrency = 1 } = {}) {
    if (!(map instanceof Map)) return [];
    return [...map.entries()].map(([id, target]) => this.registerTarget(target, { id, group, location, maxConcurrency, metadata: { source: 'map' } }));
  }

  #refreshAssignments() {
    this.assigner.clearWorkers();
    this.assigner.registerWorkers([...this.workers.values()].map((worker) => ({
      id: worker.id,
      capabilities: worker.capabilities,
      status: 'available',
      location: worker.location,
      meta: { group: worker.group, maxConcurrency: worker.maxConcurrency },
    })));
  }

  find(capability) {
    return Object.freeze([...this.workers.values()].filter((worker) => worker.capabilities.includes(String(capability))).map((worker) => worker.manifest()));
  }

  async submit(tasks = [], { checkpoint = true } = {}) {
    const normalized = tasks.map((task, i) => ({
      id: String(task.id || `swarm-task-${this.cycle + 1}-${i + 1}`),
      capability: String(task.capability || ''),
      input: clone(task.input),
      resource: task.resource ?? null,
      target: task.target ?? null,
      priority: Number(task.priority || 0),
      meta: clone(task.meta || {}),
    }));
    if (normalized.some((t) => !t.capability)) throw new TypeError('Every swarm task requires capability');
    this.cycle += 1;
    const assignment = this.assigner.assign(normalized);
    for (const p of assignment.pending) this.pending.push({ ...clone(p.task), reason: p.reason, queuedAt: nowIso() });
    for (const item of assignment.assigned) {
      this.inflight.set(item.taskId, {
        task: clone(item.task), workerId: item.workerId, assignmentId: item.id,
        capability: item.capability, startedAt: nowIso(), replaySafe: item.task?.meta?.replaySafe === true,
      });
    }
    this.#event('batch:assigned', { cycle: this.cycle, assigned: assignment.assigned.length, pending: assignment.pending.length });
    // Write the dispatch journal before effects begin. A hard process death can
    // therefore recover the task instead of silently forgetting it.
    if (checkpoint && assignment.assigned.length) await this.checkpoint(`pre-execution-${this.cycle}`);

    const executions = await Promise.all(assignment.assigned.map(async (item) => {
      const worker = this.workers.get(item.workerId);
      if (!worker) return { assignmentId: item.id, workerId: item.workerId, status: 'failed', error: 'WORKER_DISAPPEARED' };
      try {
        const output = await worker.invoke(item.task.input, { task: item.task, cycle: this.cycle, swarm: this });
        this.assigner.complete(item.id, clone(output));
        this.inflight.delete(item.taskId);
        const result = Object.freeze({ assignmentId: item.id, taskId: item.taskId, workerId: item.workerId, capability: item.capability, status: 'complete', output });
        this.completed.push(clone(result));
        this.#event('task:complete', { taskId: item.taskId, workerId: item.workerId, capability: item.capability });
        return result;
      } catch (error) {
        this.assigner.release(item.id, error?.message || 'failed');
        this.inflight.delete(item.taskId);
        const result = Object.freeze({ assignmentId: item.id, taskId: item.taskId, workerId: item.workerId, capability: item.capability, status: 'failed', error: error?.message || String(error) });
        this.completed.push(clone(result));
        this.#event('task:failed', result);
        return result;
      }
    }));
    if (this.completed.length > 256) this.completed.splice(0, this.completed.length - 256);
    if (checkpoint) await this.checkpoint(`cycle-${this.cycle}`);
    return Object.freeze({ cycle: this.cycle, assignment, executions: Object.freeze(executions), pending: Object.freeze(clone(this.pending)) });
  }

  async checkpoint(reason = 'manual') {
    const root = {
      version: 'synthia.swarm-body.v0.4',
      identity: clone(this.identity),
      savedAt: nowIso(),
      reason,
      cycle: this.cycle,
      bootCount: this.bootCount,
      workers: [...this.workers.values()].map((worker) => worker.checkpoint()),
      pending: clone(this.pending),
      inflight: clone([...this.inflight.values()]),
      recovered: clone(this.recovered),
      completed: clone(this.completed.slice(-128)),
      events: clone(this.events.slice(-128)),
    };
    await this.store.put(this.checkpointKey, root);
    this.lastCheckpointAt = root.savedAt;
    this.#event('checkpoint', { reason, workers: root.workers.length });
    return Object.freeze(root);
  }

  async wake() {
    this.bootCount += 1;
    this.lastWakeAt = nowIso();
    const root = await this.store.get(this.checkpointKey);
    const restored = [];
    if (root?.workers) {
      this.cycle = Number(root.cycle || 0);
      this.pending = clone(root.pending || []);
      this.completed = clone(root.completed || []);
      this.recovered = clone(root.recovered || []);
      for (const interrupted of root.inflight || []) {
        if (!interrupted?.task?.id || this.recovered.some((r) => r.task?.id === interrupted.task.id)) continue;
        this.recovered.push({ ...clone(interrupted), recoveredAt: nowIso(), reason: 'RECOVERED_INFLIGHT' });
      }
      this.inflight.clear();
      for (const saved of root.workers) {
        const worker = this.workers.get(saved.id);
        if (!worker) continue;
        restored.push({ id: saved.id, ...worker.restore(saved) });
      }
    }
    this.#event('wake', { bootCount: this.bootCount, restored });
    return Object.freeze({ awake: true, bootCount: this.bootCount, restored: Object.freeze(restored), snapshot: this.snapshot() });
  }

  async sleep(reason = 'sleep') { return this.checkpoint(reason); }


  recoverableTasks() { return Object.freeze(clone(this.recovered)); }

  async resumeRecovered({ replaySafeOnly = true, checkpoint = true } = {}) {
    const eligible = this.recovered.filter((entry) => !replaySafeOnly || entry.replaySafe === true);
    if (!eligible.length) return Object.freeze({ resumed: 0, held: this.recovered.length, executions: [] });
    const ids = new Set(eligible.map((entry) => entry.task.id));
    this.recovered = this.recovered.filter((entry) => !ids.has(entry.task.id));
    const result = await this.submit(eligible.map((entry) => ({ ...clone(entry.task), meta: { ...(entry.task.meta || {}), recovered: true } })), { checkpoint });
    this.#event('recovery:resumed', { resumed: eligible.length, held: this.recovered.length });
    return Object.freeze({ resumed: eligible.length, held: this.recovered.length, executions: result.executions });
  }

  snapshot() {
    const groups = {};
    for (const worker of this.workers.values()) groups[worker.group] = (groups[worker.group] || 0) + 1;
    return Object.freeze({
      version: 'synthia.swarm-body.v0.4',
      identity: this.identity,
      apparentBodies: 1,
      processCount: this.workers.size,
      groups: Object.freeze(groups),
      workers: Object.freeze([...this.workers.values()].map((worker) => worker.manifest()).sort((a, b) => a.id.localeCompare(b.id))),
      pending: Object.freeze(clone(this.pending)),
      inflight: Object.freeze(clone([...this.inflight.values()])),
      recovered: Object.freeze(clone(this.recovered)),
      completedCount: this.completed.length,
      cycle: this.cycle,
      bootCount: this.bootCount,
      lastWakeAt: this.lastWakeAt,
      lastCheckpointAt: this.lastCheckpointAt,
    });
  }
}

export default SynthiaSwarmBody;
