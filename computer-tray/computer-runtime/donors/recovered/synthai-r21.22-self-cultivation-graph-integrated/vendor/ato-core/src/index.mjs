export * from './boolean-ato.mjs';
export * from './address-space.mjs';
export * from './state-space-kernel.mjs';
export * from './anomata.mjs';
export * from './organism.mjs';
export * from './activation.mjs';
export * from './klein-iching.mjs';
export * from './automaton.mjs';
export * from './families.mjs';
export * from './workspace.mjs';
export * from './companion.mjs';
export * from './autonomy.mjs';
export * from './host.mjs';
export * from './emergence.mjs';
export * from './promotion.mjs';
export * from './success.mjs';
export * from './tray.mjs';
export * from './association.mjs';
export * from './expression.mjs';
export * from './generative-emergence.mjs';
export * from './klein-tools.mjs';
export * from './bootstrap.mjs';
export * from './quality-transfer.mjs';
export * from './interpretations.mjs';
export * from './browser-form.mjs';
export * from './research-browser.mjs';
export * from './media-renderers.mjs';
export * from './computational-grammar-coder.mjs';
export * as IChing from './iching.mjs';
export const STAGES = Object.freeze([
    'movement.create',
    'mind.encode',
    'design.structure',
    'space.integrate',
    'being.instantiate',
]);
export const ARCHITECTURES = Object.freeze({
    '63-4': { name: 'Logic', circuit: 'understanding', architecture: 'dff', behavior: 'feedforward' },
    '17-62': { name: 'Acceptance', circuit: 'understanding', architecture: 'rbm', behavior: 'bidirectional' },
    '18-58': { name: 'Judgment', circuit: 'understanding', architecture: 'hopfield', behavior: 'attractor' },
    '16-48': { name: 'Wavelength', circuit: 'understanding', architecture: 'sparse-autoencoder', behavior: 'compress' },
    '9-52': { name: 'Concentration', circuit: 'understanding', architecture: 'elm', behavior: 'focus' },
    '15-5': { name: 'Rhythm', circuit: 'understanding', architecture: 'kohonen', behavior: 'organize' },
    '31-7': { name: 'Alpha', circuit: 'understanding', architecture: 'dcn', behavior: 'scan' },
    '42-53': { name: 'Maturation', circuit: 'sensing', architecture: 'dbn', behavior: 'develop' },
});
export class ATOError extends Error {
    constructor(stage, code, message, details = {}) {
        super(message);
        this.name = 'ATOError';
        this.stage = stage;
        this.code = code;
        this.details = details;
    }
}
const intRange = (value, min, max) => Number.isInteger(value) && value >= min && value <= max;
export function normalizeMacroAddress(input) {
    const address = {
        gate: input?.gate,
        line: input?.line,
        color: input?.color,
        tone: input?.tone,
        base: input?.base,
    };
    const ranges = { gate: [1, 64], line: [1, 6], color: [1, 6], tone: [1, 6], base: [1, 5] };
    const issues = Object.entries(ranges)
        .filter(([key, [min, max]]) => !intRange(address[key], min, max))
        .map(([key, [min, max]]) => `${key} must be an integer from ${min} through ${max}`);
    if (issues.length)
        throw new ATOError('movement.create', 'INVALID_MACRO_ADDRESS', 'Macro address is invalid', { issues });
    return Object.freeze({ mode: 'macro', ...address });
}
export function normalizeMicroAddress(input) {
    const macro = normalizeMacroAddress(input);
    const address = {
        mode: 'micro',
        planetary: input?.planetary,
        dimension: input?.dimension,
        gate: macro.gate,
        line: macro.line,
        color: macro.color,
        tone: macro.tone,
        base: macro.base,
        degree: input?.degree,
        minute: input?.minute,
        second: input?.second,
        arc: input?.arc,
        zodiac: input?.zodiac,
        house: input?.house,
    };
    const issues = [];
    if (typeof address.planetary !== 'string' || !address.planetary.trim())
        issues.push('planetary is required');
    if (typeof address.dimension !== 'string' || !address.dimension.trim())
        issues.push('dimension is required');
    for (const [key, min, max] of [['degree', 0, 29], ['minute', 0, 59], ['second', 0, 59], ['arc', 0, 99], ['zodiac', 1, 12], ['house', 1, 12]]) {
        if (!intRange(address[key], min, max))
            issues.push(`${key} must be an integer from ${min} through ${max}`);
    }
    if (issues.length)
        throw new ATOError('movement.create', 'INVALID_MICRO_ADDRESS', 'Micro address is invalid', { issues });
    return Object.freeze(address);
}
export function addressKey(address) {
    const fields = address.mode === 'micro'
        ? ['planetary', 'dimension', 'gate', 'line', 'color', 'tone', 'base', 'degree', 'minute', 'second', 'arc', 'zodiac', 'house']
        : ['gate', 'line', 'color', 'tone', 'base'];
    return `${address.mode}:${fields.map((field) => `${field}=${address[field]}`).join('|')}`;
}
export function canonicalStringify(value) {
    if (value === null || typeof value !== 'object')
        return JSON.stringify(value);
    if (Array.isArray(value))
        return `[${value.map(canonicalStringify).join(',')}]`;
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalStringify(value[key])}`).join(',')}}`;
}
// Portable deterministic content identifier; deliberately avoids Node-only crypto.
export function stableId(value) {
    const text = canonicalStringify(value);
    let h1 = 0xdeadbeef ^ text.length;
    let h2 = 0x41c6ce57 ^ text.length;
    for (let index = 0; index < text.length; index += 1) {
        const code = text.charCodeAt(index);
        h1 = Math.imul(h1 ^ code, 2654435761);
        h2 = Math.imul(h2 ^ code, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return `${(h2 >>> 0).toString(16).padStart(8, '0')}${(h1 >>> 0).toString(16).padStart(8, '0')}`;
}
function assertPort(port, stage, edgeId, side) {
    if (!port || typeof port.type !== 'string' || !port.type) {
        throw new ATOError(stage, 'INVALID_PORT', `Edge ${edgeId} has an invalid ${side} port`);
    }
}
export class FunctionRegistry {
    constructor(entries = {}) { this.entries = new Map(Object.entries(entries)); }
    register(id, implementation) {
        if (typeof id !== 'string' || !id || typeof implementation !== 'function')
            throw new TypeError('Function registry requires an id and callable implementation');
        this.entries.set(id, implementation);
        return this;
    }
    resolve(id) { return this.entries.get(id); }
}
export class ATOEngine {
    constructor({ registry = new FunctionRegistry(), limits = {} } = {}) {
        this.registry = registry;
        this.limits = Object.freeze({ maxNodes: 64, maxEdges: 128, maxDepth: 12, maxFanIn: 6, maxFanOut: 6, ...limits });
    }
    compile(request) {
        const trace = [];
        const movement = this.#stage('movement.create', trace, () => this.#create(request));
        const mind = this.#stage('mind.encode', trace, () => this.#encode(movement));
        const design = this.#stage('design.structure', trace, () => this.#structure(mind));
        const space = this.#stage('space.integrate', trace, () => this.#integrate(design));
        const being = this.#stage('being.instantiate', trace, () => this.#instantiate(space));
        return Object.freeze({ ...being, trace: Object.freeze(trace) });
    }
    #stage(name, trace, operation) {
        try {
            const output = operation();
            trace.push(Object.freeze({ stage: name, status: 'passed', outputId: stableId(output) }));
            return output;
        }
        catch (error) {
            const wrapped = error instanceof ATOError ? error : new ATOError(name, 'UNEXPECTED', error.message, { cause: error.name });
            trace.push(Object.freeze({ stage: name, status: 'failed', code: wrapped.code, message: wrapped.message }));
            wrapped.trace = Object.freeze(trace);
            throw wrapped;
        }
    }
    #create(request) {
        if (!request || typeof request !== 'object')
            throw new ATOError('movement.create', 'MISSING_REQUEST', 'A build request is required');
        if (!request.address)
            throw new ATOError('movement.create', 'MISSING_ADDRESS', 'ATO builds only from a supplied address');
        const address = request.address.mode === 'micro' ? normalizeMicroAddress(request.address) : normalizeMacroAddress(request.address);
        if (!request.graph || !Array.isArray(request.graph.nodes) || !Array.isArray(request.graph.edges)) {
            throw new ATOError('movement.create', 'MISSING_GRAPH', 'A graph containing nodes and edges is required');
        }
        return Object.freeze({ requestId: request.id || stableId(request), address, purpose: request.purpose || null, graph: request.graph });
    }
    #encode(created) {
        const nodes = created.graph.nodes.map((node) => Object.freeze({ ...node }));
        const edges = created.graph.edges.map((edge) => Object.freeze({
            id: edge.id || `${edge.from}->${edge.to}`,
            from: edge.from,
            to: edge.to,
            fromPort: edge.fromPort || { type: 'json' },
            toPort: edge.toPort || { type: 'json' },
            functionId: edge.functionId,
            channel: edge.channel || null,
            regime: edge.regime || 'stable',
        }));
        return Object.freeze({ ...created, irVersion: 'ato.mir.v0.1', nodes: Object.freeze(nodes), edges: Object.freeze(edges) });
    }
    #structure(ir) {
        if (ir.nodes.length > this.limits.maxNodes)
            throw new ATOError('design.structure', 'NODE_BUDGET_EXCEEDED', 'Node budget exceeded');
        if (ir.edges.length > this.limits.maxEdges)
            throw new ATOError('design.structure', 'EDGE_BUDGET_EXCEEDED', 'Edge budget exceeded');
        const nodeIds = new Set();
        for (const node of ir.nodes) {
            if (!node.id || nodeIds.has(node.id))
                throw new ATOError('design.structure', 'INVALID_NODE_ID', `Node id is missing or duplicated: ${node.id}`);
            nodeIds.add(node.id);
        }
        const incoming = new Map(ir.nodes.map((node) => [node.id, 0]));
        const outgoing = new Map(ir.nodes.map((node) => [node.id, 0]));
        for (const edge of ir.edges) {
            if (!nodeIds.has(edge.from) || !nodeIds.has(edge.to))
                throw new ATOError('design.structure', 'DANGLING_EDGE', `Edge ${edge.id} references an unknown node`);
            if (edge.from === edge.to)
                throw new ATOError('design.structure', 'SELF_EDGE', `Edge ${edge.id} cannot call its own node`);
            assertPort(edge.fromPort, 'design.structure', edge.id, 'source');
            assertPort(edge.toPort, 'design.structure', edge.id, 'target');
            if (edge.fromPort.type !== edge.toPort.type)
                throw new ATOError('design.structure', 'PORT_TYPE_MISMATCH', `Edge ${edge.id} connects ${edge.fromPort.type} to ${edge.toPort.type}`);
            if (!edge.functionId || !this.registry.resolve(edge.functionId))
                throw new ATOError('design.structure', 'UNKNOWN_FUNCTION', `Edge ${edge.id} has no registered implementation: ${edge.functionId}`);
            incoming.set(edge.to, incoming.get(edge.to) + 1);
            outgoing.set(edge.from, outgoing.get(edge.from) + 1);
            if (incoming.get(edge.to) > this.limits.maxFanIn)
                throw new ATOError('design.structure', 'FAN_IN_EXCEEDED', `Node ${edge.to} exceeds fan-in budget`);
            if (outgoing.get(edge.from) > this.limits.maxFanOut)
                throw new ATOError('design.structure', 'FAN_OUT_EXCEEDED', `Node ${edge.from} exceeds fan-out budget`);
        }
        const order = this.#topologicalOrder(ir.nodes, ir.edges, incoming);
        if (order.length > this.limits.maxDepth)
            throw new ATOError('design.structure', 'DEPTH_BUDGET_EXCEEDED', 'Graph depth budget exceeded');
        return Object.freeze({ ...ir, order: Object.freeze(order), incoming, outgoing });
    }
    #topologicalOrder(nodes, edges, incoming) {
        const counts = new Map(incoming);
        const queue = nodes.filter((node) => counts.get(node.id) === 0).map((node) => node.id).sort();
        const order = [];
        while (queue.length) {
            const id = queue.shift();
            order.push(id);
            for (const edge of edges.filter((candidate) => candidate.from === id)) {
                counts.set(edge.to, counts.get(edge.to) - 1);
                if (counts.get(edge.to) === 0)
                    queue.push(edge.to);
            }
            queue.sort();
        }
        if (order.length !== nodes.length)
            throw new ATOError('design.structure', 'CYCLE_DETECTED', 'Executable tool graphs must be acyclic in the first vertical slice');
        return order;
    }
    #integrate(design) {
        const nodes = new Map(design.nodes.map((node) => [node.id, node]));
        const edgesBySource = new Map(design.nodes.map((node) => [node.id, []]));
        for (const edge of design.edges)
            edgesBySource.get(edge.from).push(edge);
        return Object.freeze({
            ...design,
            artifact: Object.freeze({
                id: `artifact-${stableId({ address: addressKey(design.address), nodes: design.nodes, edges: design.edges })}`,
                address: design.address,
                addressKey: addressKey(design.address),
                purpose: design.purpose,
                nodes,
                edgesBySource,
            }),
        });
    }
    #instantiate(integration) {
        const artifact = integration.artifact;
        const registry = this.registry;
        const execute = async (input, { signal } = {}) => {
            const values = new Map();
            const roots = integration.order.filter((id) => integration.incoming.get(id) === 0);
            for (const root of roots)
                values.set(root, input);
            for (const sourceId of integration.order) {
                if (signal?.aborted)
                    throw new ATOError('being.instantiate', 'INTERRUPTED', 'Execution was interrupted', { sourceId });
                const sourceValue = values.get(sourceId);
                for (const edge of artifact.edgesBySource.get(sourceId) || []) {
                    const fn = registry.resolve(edge.functionId);
                    const result = await fn(sourceValue, Object.freeze({ edge, address: artifact.address, purpose: artifact.purpose }));
                    if (values.has(edge.to)) {
                        const current = values.get(edge.to);
                        values.set(edge.to, Array.isArray(current) ? [...current, result] : [current, result]);
                    }
                    else
                        values.set(edge.to, result);
                }
            }
            const leaves = integration.order.filter((id) => integration.outgoing.get(id) === 0);
            return Object.freeze({ artifactId: artifact.id, outputs: Object.freeze(Object.fromEntries(leaves.map((id) => [id, values.get(id)]))), values });
        };
        return Object.freeze({ artifact, execute });
    }
}

export * from './klein-module-registry.mjs';
