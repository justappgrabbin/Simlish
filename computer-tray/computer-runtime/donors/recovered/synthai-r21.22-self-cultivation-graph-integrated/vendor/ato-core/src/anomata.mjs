import { StateSpaceKernel } from './state-space-kernel.mjs';
import { completeAnalogy, toBitString } from './boolean-ato.mjs';
import * as IChing from './iching.mjs';
import { canonicalAddress, normalizeAddress } from './address-space.mjs';
export class Anomata {
    constructor({ kernel = new StateSpaceKernel(), maxEvents = 4096 } = {}) {
        this.kernel = kernel;
        this.maxEvents = maxEvents;
        this.events = [];
        this.artifacts = new Map();
        this.sequence = 0;
    }
    record(type, payload) {
        const event = Object.freeze({ sequence: ++this.sequence, type, payload, previous: this.events.at(-1)?.sequence || null });
        this.events.push(event);
        if (this.events.length > this.maxEvents)
            this.events.shift();
        return event;
    }
    inspect(address) {
        const state = this.kernel.get(address);
        const result = Object.freeze({ address: state.address, addressKey: state.addressKey, vector: state.value.vector, metadata: state.value.metadata, projections: Object.freeze(['expression', 'inference', 'code', 'image', 'timeline', 'game'].map((name) => this.kernel.project(address, name))) });
        this.record('inspect', { address: state.addressKey });
        return result;
    }
    converse(text) {
        const resolved = this.kernel.describe(text);
        const best = resolved.candidates[0]?.state;
        const response = Object.freeze({
            input: text,
            state: best?.address || null,
            addressKey: best?.addressKey || null,
            evidence: Object.freeze(resolved.candidates.map((entry) => ({ address: entry.state.addressKey, distance: entry.distance }))),
            reply: best ? `I resolved that description near ${best.addressKey}.` : 'The description remains unresolved.',
        });
        this.record('conversation', response);
        return response;
    }
    analogy(a, b, c, mode = 'equivalence') {
        const result = this.kernel.complete(a, b, c, mode);
        this.record('analogy', { a: canonicalAddress(a), b: canonicalAddress(b), c: canonicalAddress(c), mode, result: toBitString(result.result) });
        return result;
    }
    change(address, lines) {
        const source = this.kernel.get(address);
        const changed = IChing.changeLines(gateStateBits(source.address), lines);
        const targetAddress = Object.freeze({ ...source.address, gate: changed.result.number });
        const target = this.kernel.get(targetAddress);
        const result = Object.freeze({ source: source.address, target: target.address, changingLines: changed.changingLines, sourceBits: changed.source.bits, targetBits: changed.result.bits });
        this.record('state-transition', result);
        return result;
    }
    mount(manifest) {
        const artifact = this.kernel.mountArtifact(manifest);
        this.artifacts.set(artifact.id, artifact);
        this.record('artifact-mounted', { id: artifact.id, address: artifact.addressKey, capabilities: artifact.capabilities.map((c) => c.id) });
        return artifact;
    }
    async run(artifactId, capabilityId, input, context = {}) {
        const artifact = this.artifacts.get(artifactId);
        if (!artifact)
            throw new Error(`Artifact is not mounted: ${artifactId}`);
        const capability = artifact.capabilities.find((candidate) => candidate.id === capabilityId);
        if (!capability)
            throw new Error(`Capability is not mounted: ${capabilityId}`);
        if (typeof capability.implementation !== 'function')
            throw new Error(`Capability has no browser implementation: ${capabilityId}`);
        const result = await capability.implementation(input, Object.freeze({ address: capability.address, artifact: artifact.address, context }));
        this.record('capability-run', { artifactId, capabilityId, address: canonicalAddress(capability.address) });
        return Object.freeze({ artifactId, capabilityId, address: capability.address, result });
    }
    replay() { return Object.freeze([...this.events]); }
}
function gateStateBits(address) {
    const normalized = normalizeAddress(address, { mode: address.mode || 'macro' });
    return (normalized.gate - 1).toString(2).padStart(6, '0');
}
