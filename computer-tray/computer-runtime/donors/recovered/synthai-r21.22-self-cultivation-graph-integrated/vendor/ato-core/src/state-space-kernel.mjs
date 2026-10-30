import { completeAnalogy, hammingDistance, normalizeVector, operator, toBitString } from './boolean-ato.mjs';
import { addressed, canonicalAddress, gateState, normalizeAddress } from './address-space.mjs';
import { enumerateSpace } from './iching.mjs';
export const PROJECTIONS = Object.freeze(['position', 'transformation', 'expression', 'inference', 'code', 'image', 'timeline', 'game']);
function deterministicFeature(address, featureIndex) {
    const macro = normalizeAddress(address, { mode: address.mode || 'macro', allowUnresolved: true });
    // GLCTB coordinates modulate the six-line Gate substrate without replacing it.
    const seed = (macro.gate * 31 + (macro.line || 1) * 17 + (macro.color || 1) * 13 + (macro.tone || 1) * 7 + (macro.base || 1) * 5 + featureIndex * 11) >>> 0;
    return (seed ^ (seed >>> 7) ^ (seed >>> 13)) & 1;
}
export class StateSpaceKernel {
    constructor({ featureWidth = 24 } = {}) {
        if (!Number.isInteger(featureWidth) || featureWidth < 6)
            throw new RangeError('featureWidth must be at least 6');
        this.featureWidth = featureWidth;
        this.states = new Map();
        this.annotations = new Map();
        this.mount64();
    }
    mount64() {
        for (const state of enumerateSpace()) {
            const address = { mode: 'macro', gate: state.number, line: 1, color: 1, tone: 1, base: 1 };
            this.define(address, { primitive: true, hexagram: state.bits });
        }
        return this;
    }
    vector(address) {
        const normalized = normalizeAddress(address, { mode: address.mode || 'macro' });
        const base = [...gateState(normalized).bits].map(Number);
        while (base.length < this.featureWidth)
            base.push(deterministicFeature(normalized, base.length));
        return Object.freeze(base);
    }
    define(address, metadata = {}) {
        const normalized = normalizeAddress(address, { mode: address.mode || 'macro' });
        const key = canonicalAddress(normalized);
        const state = addressed(Object.freeze({ vector: this.vector(normalized), metadata: Object.freeze({ ...metadata }) }), normalized, { kind: 'state-space-node' });
        this.states.set(key, state);
        return state;
    }
    get(address) { return this.states.get(canonicalAddress(address)) || this.define(address); }
    annotate(address, projection, value, provenance = null) {
        if (!PROJECTIONS.includes(projection))
            throw new RangeError(`Unknown projection: ${projection}`);
        const state = this.get(address);
        const key = `${state.addressKey}::${projection}`;
        const annotation = Object.freeze({ state: state.addressKey, projection, value, provenance });
        this.annotations.set(key, annotation);
        return annotation;
    }
    project(address, projection) {
        const state = this.get(address);
        return this.annotations.get(`${state.addressKey}::${projection}`) || Object.freeze({ state: state.addressKey, projection, value: null, unresolved: true });
    }
    transform(aAddress, bAddress, mode = 'equivalence') {
        const a = this.get(aAddress);
        const b = this.get(bAddress);
        return Object.freeze({
            source: a.addressKey, target: b.addressKey, mode,
            vector: operator(a.value.vector, b.value.vector, mode),
        });
    }
    complete(aAddress, bAddress, cAddress, mode = 'equivalence') {
        const a = this.get(aAddress), b = this.get(bAddress), c = this.get(cAddress);
        const completion = completeAnalogy(a.value.vector, b.value.vector, c.value.vector, mode);
        const candidates = [...this.states.values()].map((state) => Object.freeze({ state, distance: hammingDistance(completion.result, state.value.vector) })).sort((x, y) => x.distance - y.distance || x.state.addressKey.localeCompare(y.state.addressKey));
        return Object.freeze({ a: a.addressKey, b: b.addressKey, c: c.addressKey, mode, relation: completion.relation, result: completion.result, candidates: Object.freeze(candidates.slice(0, 8)) });
    }
    describe(input) {
        // Descriptions become evidence attached to candidate states; they do not bypass the state space.
        const tokens = String(input).toLowerCase().match(/[a-z0-9_'-]+/g) || [];
        const cue = Array.from({ length: this.featureWidth }, (_, index) => tokens.reduce((bit, token) => bit ^ deterministicFeature({ mode: 'macro', gate: (token.charCodeAt(index % token.length) % 64) + 1, line: 1, color: 1, tone: 1, base: 1 }, index), 0));
        const candidates = [...this.states.values()].map((state) => ({ state, distance: hammingDistance(cue, state.value.vector) })).sort((a, b) => a.distance - b.distance || a.state.addressKey.localeCompare(b.state.addressKey));
        return Object.freeze({ input: String(input), cue: Object.freeze(cue), candidates: Object.freeze(candidates.slice(0, 8)) });
    }
    mountArtifact(manifest) {
        if (!manifest || typeof manifest !== 'object' || !manifest.address || !Array.isArray(manifest.capabilities))
            throw new TypeError('Artifact manifest requires an address and capabilities');
        const host = this.get(manifest.address);
        const capabilities = manifest.capabilities.map((capability, index) => Object.freeze({
            id: capability.id || `capability-${index}`,
            address: normalizeAddress(capability.address || manifest.address, { mode: (capability.address || manifest.address).mode || 'macro' }),
            input: capability.input || 'json', output: capability.output || 'json', effect: capability.effect || 'pure', implementation: capability.implementation || null,
        }));
        return Object.freeze({ id: manifest.id || `artifact:${host.addressKey}`, address: host.address, addressKey: host.addressKey, capabilities: Object.freeze(capabilities), mounted: true });
    }
}
