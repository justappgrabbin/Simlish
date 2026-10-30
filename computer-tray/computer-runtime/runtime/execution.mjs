export class AutomataEngine {
  constructor({ bus, automata, tools, state } = {}) { Object.assign(this, { bus, automata, tools, state }); }
  register(d) { return this.automata.register(d.id, d); }
  async run(id, input, context = {}) {
    const a = this.automata.get(id);
    if (!a) throw new Error(`unknown automaton: ${id}`);
    const runId = `run-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    this.bus?.emit('automata:start', { runId, id, input });
    try {
      const output = await a.execute({ input, context, tools: this.tools, state: this.state, bus: this.bus });
      await this.state.set(`automata.runs.${runId}`, { id, input, output, status: 'complete', at: Date.now() }, { source: 'automata' });
      this.bus?.emit('automata:complete', { runId, id, output });
      return output;
    } catch (error) {
      await this.state.set(`automata.runs.${runId}`, { id, input, status: 'failed', error: String(error?.message ?? error), at: Date.now() }, { source: 'automata' });
      this.bus?.emit('automata:failed', { runId, id, error: String(error?.message ?? error) });
      throw error;
    }
  }
}

export class ProcessFabric {
  constructor({ bus, processes, state } = {}) { Object.assign(this, { bus, processes, state }); }
  register(d) {
    if (typeof d?.run !== 'function') throw new Error('process.run function required');
    return this.processes.register(d.id, d);
  }
  async execute(id, input, context = {}) {
    const p = this.processes.get(id);
    if (!p) throw new Error(`unknown process: ${id}`);
    this.bus?.emit('process:start', { id, input });
    try {
      const output = await p.run({ input, context, state: this.state, bus: this.bus });
      await this.state.set(`processes.last.${id}`, { input, output, at: Date.now() }, { source: 'process-fabric' });
      this.bus?.emit('process:complete', { id, output });
      return output;
    } catch (error) {
      this.bus?.emit('process:failed', { id, error: String(error?.message ?? error) });
      throw error;
    }
  }
}

export class BackendBroker {
  constructor({ bus, backends } = {}) { Object.assign(this, { bus, backends }); }
  register(id, adapter, { replace = false } = {}) {
    if (!adapter || typeof adapter.request !== 'function') throw new Error('backend adapter.request function required');
    return this.backends.register(id, { adapter, status: 'available' }, { replace });
  }
  async request(id, request) {
    const b = this.backends.get(id);
    if (!b) throw new Error(`backend not registered: ${id}`);
    this.bus?.emit('backend:request', { id, request });
    try {
      const result = await b.adapter.request(request);
      this.bus?.emit('backend:response', { id, result });
      return result;
    } catch (error) {
      this.bus?.emit('backend:failed', { id, error: String(error?.message ?? error) });
      throw error;
    }
  }
}
