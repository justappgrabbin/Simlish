import { canonicalAddress, normalizeAddress } from './address-space.mjs';
export const LIFECYCLE = Object.freeze({
    intake: 'accept addressed input', processing: 'transform state', movement: 'transition address',
    communication: 'exchange through typed ports', repair: 'validate and replace', memory: 'record lineage and replay',
    growth: 'compose accepted tools', rest: 'quiescent resource state',
});
export class ToolOrganism {
    constructor({ id = 'anomata', engine, address, limits = {} } = {}) {
        if (!engine)
            throw new TypeError('ToolOrganism requires an Anomata engine');
        this.id = id;
        this.engine = engine;
        this.address = normalizeAddress(address, { mode: address?.mode || 'macro' });
        this.limits = Object.freeze({ maxTools: 64, maxConnections: 128, maxGeneration: 6, ...limits });
        this.tools = new Map();
        this.connections = new Map();
        this.candidates = new Map();
        this.generation = 0;
        this.state = 'rest';
    }
    ingest(artifact) {
        if (this.tools.size >= this.limits.maxTools)
            throw new Error('Tool budget exceeded');
        const mounted = this.engine.mount(artifact);
        this.tools.set(mounted.id, mounted);
        this.state = 'intake';
        return mounted;
    }
    connect({ fromTool, fromPort, toTool, toPort, operator = 'compose', address }) {
        if (this.connections.size >= this.limits.maxConnections)
            throw new Error('Connection budget exceeded');
        const source = this.#port(fromTool, fromPort, 'output');
        const target = this.#port(toTool, toPort, 'input');
        if (source.output !== target.input)
            return Object.freeze({ status: 'unresolved', reason: 'PORT_TYPE_MISMATCH', source: source.output, target: target.input });
        const edgeAddress = normalizeAddress(address || source.address, { mode: (address || source.address).mode || 'macro' });
        const id = `${fromTool}.${fromPort}->${toTool}.${toPort}`;
        const edge = Object.freeze({ id, fromTool, fromPort, toTool, toPort, operator, address: edgeAddress, addressKey: canonicalAddress(edgeAddress) });
        this.connections.set(id, edge);
        this.state = 'processing';
        return this.#candidateFrom(edge, source, target);
    }
    #port(toolId, portId, side) {
        const tool = this.tools.get(toolId);
        if (!tool)
            throw new Error(`Unknown tool: ${toolId}`);
        const port = tool.capabilities.find((item) => item.id === portId);
        if (!port)
            throw new Error(`Unknown port: ${portId}`);
        return port;
    }
    #candidateFrom(edge, source, target) {
        const id = `candidate:${this.id}:${this.generation + 1}:${edge.id}`;
        const candidate = Object.freeze({
            id, status: 'candidate', generation: this.generation + 1, address: edge.address, addressKey: edge.addressKey,
            lineage: Object.freeze([edge.fromTool, edge.toTool]), connections: Object.freeze([edge.id]),
            capability: Object.freeze({ id: `organ-${this.generation + 1}`, address: edge.address, input: source.input || source.output, output: target.output || target.input, effect: target.effect || source.effect }),
        });
        this.candidates.set(id, candidate);
        this.engine.record('growth-candidate', candidate);
        return candidate;
    }
    accept(candidateId, implementation) {
        const candidate = this.candidates.get(candidateId);
        if (!candidate)
            throw new Error(`Unknown growth candidate: ${candidateId}`);
        if (candidate.generation > this.limits.maxGeneration)
            throw new Error('Generation budget exceeded');
        const artifact = this.ingest({ id: candidate.id, address: candidate.address, capabilities: [{ ...candidate.capability, implementation }] });
        this.candidates.delete(candidateId);
        this.generation = Math.max(this.generation, candidate.generation);
        this.state = 'growth';
        this.engine.record('growth-accepted', { candidateId, artifactId: artifact.id, generation: this.generation, lineage: candidate.lineage });
        return artifact;
    }
    reject(candidateId, reason = 'rejected') {
        const candidate = this.candidates.get(candidateId);
        if (!candidate)
            return false;
        this.candidates.delete(candidateId);
        this.engine.record('growth-rejected', { candidateId, reason });
        return true;
    }
    rest() { this.state = 'rest'; this.engine.record('rest', { organism: this.id, generation: this.generation }); }
    snapshot() { return Object.freeze({ id: this.id, address: this.address, state: this.state, generation: this.generation, tools: Object.freeze([...this.tools.keys()]), connections: Object.freeze([...this.connections.values()]), candidates: Object.freeze([...this.candidates.values()]) }); }
}
