import { activationSignature } from './activation.mjs';
import { canonicalAddress, normalizeAddress } from './address-space.mjs';
export const FUNCTIONAL_LEVELS = Object.freeze({
    movement: Object.freeze({ id: 'movement', operation: 'transition', accepts: ['state', 'event', 'timeline'] }),
    mind: Object.freeze({ id: 'mind', operation: 'encode', accepts: ['text', 'code', 'symbol', 'state'] }),
    design: Object.freeze({ id: 'design', operation: 'structure', accepts: ['graph', 'code', 'image', 'tool'] }),
    space: Object.freeze({ id: 'space', operation: 'integrate', accepts: ['graph', 'tool', 'artifact', 'context'] }),
    being: Object.freeze({ id: 'being', operation: 'instantiate', accepts: ['tool', 'artifact', 'game', 'runtime'] }),
});
export class AutomatonError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'AutomatonError'; this.code = code; this.details = details; }
}
const freezePort = (port, index) => Object.freeze({
    id: port.id || `port-${index}`,
    direction: port.direction || 'input',
    type: port.type || 'json',
    schemaVersion: port.schemaVersion || '1',
    schema: port.schema ? Object.freeze({ ...port.schema }) : null,
    requires: Object.freeze([...(port.requires || [])].sort()),
    guarantees: Object.freeze([...(port.guarantees || [])].sort()),
    provenance: port.provenance || null,
});
const portCompatibility = (output, input) => {
    const issues = [];
    if (output.type !== input.type)
        issues.push('TYPE_MISMATCH');
    if (output.schemaVersion !== input.schemaVersion)
        issues.push('SCHEMA_VERSION_MISMATCH');
    const missing = input.requires.filter((invariant) => !output.guarantees.includes(invariant));
    if (missing.length)
        issues.push('INVARIANT_MISMATCH');
    return Object.freeze({ compatible: issues.length === 0, issues: Object.freeze(issues), missing: Object.freeze(missing) });
};
export class Automaton {
    constructor({ id, address, structure, activeLevels, functionalLevel, ports = [], implementation, metadata = {}, manifestVersion = 'ato.automaton.v1', state = null } = {}) {
        if (typeof id !== 'string' || !id)
            throw new AutomatonError('MISSING_ID', 'Automaton requires an id');
        if (typeof implementation !== 'function')
            throw new AutomatonError('MISSING_IMPLEMENTATION', 'Automaton requires an executable implementation');
        const binding = FUNCTIONAL_LEVELS[functionalLevel];
        if (!binding)
            throw new AutomatonError('INVALID_FUNCTIONAL_LEVEL', `Unknown functional level: ${functionalLevel}`);
        this.id = id;
        this.address = normalizeAddress(address, { mode: address?.mode || 'macro' });
        this.addressKey = canonicalAddress(this.address);
        this.structure = activationSignature(structure, activeLevels);
        this.functionalLevel = functionalLevel;
        this.binding = binding;
        this.ports = Object.freeze(ports.map(freezePort));
        this.implementation = implementation;
        this.metadata = Object.freeze({ ...metadata });
        this.manifestVersion = manifestVersion;
        this.ownedState = state;
        this.lifecycle = 'ready';
        this.calls = 0;
        this.history = [];
    }
    async call(input, context = {}) {
        if (context.signal?.aborted)
            throw new AutomatonError('INTERRUPTED', `Automaton ${this.id} was interrupted before execution`);
        this.lifecycle = 'active';
        const call = Object.freeze({ sequence: ++this.calls, address: this.addressKey, functionalLevel: this.functionalLevel, structure: this.structure.mask, input });
        try {
            const output = await this.implementation(input, Object.freeze({ automaton: this, address: this.address, binding: this.binding, state: this.ownedState, context }));
            this.history.push(Object.freeze({ ...call, status: 'complete', output }));
            this.lifecycle = 'ready';
            return output;
        }
        catch (error) {
            this.history.push(Object.freeze({ ...call, status: 'failed', error: error.message }));
            this.lifecycle = 'error';
            throw error;
        }
    }
    exportState() { return this.ownedState === null ? null : structuredClone(this.ownedState); }
    manifest() { return Object.freeze({ manifestVersion: this.manifestVersion, id: this.id, address: this.address, addressKey: this.addressKey, structure: this.structure, functionalLevel: this.functionalLevel, binding: this.binding, ports: this.ports, metadata: this.metadata, lifecycle: this.lifecycle, calls: this.calls }); }
}
export class AutomataMesh {
    constructor({ manifestVersion = 'ato.mesh.v1' } = {}) { this.manifestVersion = manifestVersion; this.automatons = new Map(); this.edges = new Map(); this.runSequence = 0; }
    add(automaton) { if (!(automaton instanceof Automaton))
        throw new AutomatonError('INVALID_AUTOMATON', 'Mesh accepts Automaton instances'); if (this.automatons.has(automaton.id))
        throw new AutomatonError('DUPLICATE_AUTOMATON', `Duplicate Automaton: ${automaton.id}`); this.automatons.set(automaton.id, automaton); return automaton; }
    compatible(fromId, toId) { const a = this.automatons.get(fromId), b = this.automatons.get(toId); if (!a || !b)
        return []; const outputs = a.ports.filter(p => p.direction === 'output'); const inputs = b.ports.filter(p => p.direction === 'input'); return outputs.flatMap(out => inputs.map(input => Object.freeze({ from: out, to: input, type: out.type, ...portCompatibility(out, input) }))).filter(pair => pair.compatible); }
    #pathExists(start, target) { const seen = new Set(), queue = [start]; while (queue.length) {
        const id = queue.shift();
        if (id === target)
            return true;
        if (seen.has(id))
            continue;
        seen.add(id);
        for (const edge of this.edges.values())
            if (edge.fromId === id)
                queue.push(edge.toId);
    } return false; }
    connect(fromId, toId, { outputPort, inputPort, operator = 'transmit', recursive = false, termination = null, adapter = null, adapterId = null } = {}) {
        const a = this.automatons.get(fromId), b = this.automatons.get(toId);
        if (!a || !b)
            throw new AutomatonError('UNKNOWN_AUTOMATON', 'Both Automatons must exist');
        const output = a.ports.find(port => port.direction === 'output' && (!outputPort || port.id === outputPort));
        const input = b.ports.find(port => port.direction === 'input' && (!inputPort || port.id === inputPort));
        if (!output || !input)
            return Object.freeze({ status: 'not-reachable', reason: 'PORT_NOT_FOUND' });
        const direct = portCompatibility(output, input);
        if (!direct.compatible && typeof adapter !== 'function')
            return Object.freeze({ status: 'not-reachable', reason: 'INCOMPATIBLE_CONTRACT', issues: direct.issues, missing: direct.missing });
        if (typeof adapter === 'function' && (!adapterId || typeof adapterId !== 'string'))
            throw new AutomatonError('UNVERSIONED_ADAPTER', 'Cross-contract adapters require a stable adapterId');
        if (this.#pathExists(toId, fromId) && !recursive)
            return Object.freeze({ status: 'not-reachable', reason: 'CYCLE_REQUIRES_PERMISSION' });
        if (recursive && typeof termination !== 'function')
            throw new AutomatonError('MISSING_TERMINATION', 'Recursive edges require an explicit termination condition');
        const id = `${fromId}.${output.id}->${toId}.${input.id}`;
        const edge = Object.freeze({ id, fromId, toId, outputPort: output.id, inputPort: input.id, fromContract: Object.freeze({ type: output.type, schemaVersion: output.schemaVersion }), toContract: Object.freeze({ type: input.type, schemaVersion: input.schemaVersion }), operator, recursive, termination, adapter, adapterId });
        this.edges.set(id, edge);
        return Object.freeze({ status: 'connected', edge });
    }
    async run(startId, input, { maxHops = 32, maxVisitsPerAutomaton = 1, signal } = {}) {
        if (!this.automatons.has(startId))
            throw new AutomatonError('UNKNOWN_AUTOMATON', startId);
        const runId = ++this.runSequence, outputs = new Map(), visits = new Map(), log = [], queue = [{ id: startId, value: input, hops: 0, via: null }];
        while (queue.length) {
            if (signal?.aborted)
                throw new AutomatonError('INTERRUPTED', `Mesh run ${runId} was interrupted`);
            const current = queue.shift();
            if (current.hops > maxHops)
                throw new AutomatonError('HOP_LIMIT', 'Mesh hop limit exceeded');
            const count = (visits.get(current.id) || 0) + 1;
            visits.set(current.id, count);
            if (count > maxVisitsPerAutomaton)
                throw new AutomatonError('VISIT_LIMIT', `Automaton ${current.id} exceeded its visit limit`);
            const automaton = this.automatons.get(current.id);
            const result = await automaton.call(current.value, { mesh: this, signal, runId });
            outputs.set(current.id, result);
            log.push(Object.freeze({ sequence: log.length + 1, automatonId: current.id, address: automaton.addressKey, via: current.via, input: current.value, output: result }));
            const outgoing = [...this.edges.values()].filter(edge => edge.fromId === current.id).sort((left, right) => left.id.localeCompare(right.id));
            for (const edge of outgoing) {
                if (edge.recursive && edge.termination({ result, hops: current.hops, visits: Object.freeze(Object.fromEntries(visits)) }))
                    continue;
                const value = edge.adapter ? await edge.adapter(result, Object.freeze({ edge, runId })) : result;
                queue.push({ id: edge.toId, value, hops: current.hops + 1, via: edge.id });
            }
        }
        return Object.freeze({ runId, outputs: Object.freeze(Object.fromEntries(outputs)), visited: Object.freeze(log.map(event => event.automatonId)), log: Object.freeze(log) });
    }
    snapshot() { return Object.freeze({ manifestVersion: this.manifestVersion, automatons: Object.freeze([...this.automatons.values()].map(a => a.manifest()).sort((a, b) => a.id.localeCompare(b.id))), edges: Object.freeze([...this.edges.values()].map(({ adapter, termination, ...edge }) => edge).sort((a, b) => a.id.localeCompare(b.id))) }); }
}
