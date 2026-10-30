import { Automaton } from './automaton.mjs';
import { canonicalAddress, normalizeAddress } from './address-space.mjs';
export class EmergenceError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'EmergenceError'; this.code = code; this.details = details; }
}
export class EmergenceRegistry {
    constructor({ arcRange = { min: 1, max: 99 }, arcMapper = null, neighborhoodRules = [] } = {}) { this.arcRange = Object.freeze({ ...arcRange }); this.arcMapper = arcMapper; this.neighborhoodRules = [...neighborhoodRules]; this.spots = new Map(); this.candidates = new Map(); this.channels = new Map(); }
    defineSpot({ address, layer = 'arc', status = 'potential', provenance = [] } = {}) { const normalized = normalizeAddress(address, { mode: address?.mode || 'micro', allowUnresolved: false }); if (layer === 'arc' && (!Number.isInteger(normalized.arc) || normalized.arc < this.arcRange.min || normalized.arc > this.arcRange.max))
        throw new EmergenceError('ARC_OUT_OF_RANGE', 'Arc emergence spot is outside configured range', { arc: normalized.arc, range: this.arcRange }); const key = `${layer}:${canonicalAddress(normalized)}`; const spot = Object.freeze({ key, address: normalized, addressKey: canonicalAddress(normalized), layer, status, provenance: Object.freeze([...provenance]) }); this.spots.set(key, spot); return spot; }
    deriveBigram(spot) { if (spot.layer !== 'arc')
        throw new EmergenceError('INVALID_BIGRAM_LAYER', 'Bigrams derive from Arc emergence spots'); if (typeof this.arcMapper !== 'function')
        return Object.freeze({ status: 'unresolved', reason: 'ARC_BIGRAM_MAPPING_REQUIRED', spot: spot.key }); const value = this.arcMapper(spot.address.arc, spot.address); if (!Array.isArray(value) || value.length !== 2 || value.some(bit => bit !== 0 && bit !== 1))
        throw new EmergenceError('INVALID_BIGRAM_MAPPING', 'Arc mapper must return two bits'); return Object.freeze({ status: 'resolved', spot: spot.key, bigram: Object.freeze(value), mask: value.join('') }); }
    neighborhood(a, b, context = {}) { const matches = this.neighborhoodRules.map(rule => rule(a, b, context)).filter(Boolean); return Object.freeze({ compatible: matches.length > 0, matches: Object.freeze(matches) }); }
    proposeChannel(left, right, { context = {}, functionalLevel = 'space', implementation = (input) => input } = {}) { const relation = this.neighborhood(left, right, context); if (!relation.compatible)
        return Object.freeze({ status: 'not-reachable', reason: 'NO_NEIGHBORHOOD_RELATION' }); const id = `channel:${left.id}:${right.id}:${relation.matches.map(match => match.id || match).join('+')}`; const candidate = Object.freeze({ id, status: 'candidate', left: left.id, right: right.id, leftAddress: left.addressKey, rightAddress: right.addressKey, relations: relation.matches, functionalLevel, implementation }); this.candidates.set(id, candidate); return candidate; }
    instantiate(candidateId, { address, ports = null } = {}) { const candidate = this.candidates.get(candidateId); if (!candidate)
        throw new EmergenceError('CANDIDATE_NOT_FOUND', candidateId); const leftPorts = ports || [{ id: 'in', direction: 'input', type: 'json' }, { id: 'out', direction: 'output', type: 'json' }]; const channel = new Automaton({ id: candidate.id, address, structure: 'bigram', activeLevels: [1], functionalLevel: candidate.functionalLevel, ports: leftPorts, implementation: candidate.implementation, metadata: { family: 'emergent-channel', members: [candidate.left, candidate.right], relations: candidate.relations, emergent: true } }); this.channels.set(channel.id, channel); this.candidates.delete(candidateId); return channel; }
}
export const neighborhoodRules = Object.freeze({
    sameArc: (a, b) => a.address?.arc !== undefined && a.address.arc === b.address?.arc ? Object.freeze({ id: 'same-arc', arc: a.address.arc }) : null,
    adjacentArc: (a, b) => Number.isInteger(a.address?.arc) && Number.isInteger(b.address?.arc) && Math.abs(a.address.arc - b.address.arc) === 1 ? Object.freeze({ id: 'adjacent-arc', distance: 1 }) : null,
    sameGate: (a, b) => a.address?.gate !== undefined && a.address.gate === b.address?.gate ? Object.freeze({ id: 'same-gate', gate: a.address.gate }) : null,
    compatiblePorts: (a, b) => { const outs = a.ports?.filter(p => p.direction === 'output') || [], ins = b.ports?.filter(p => p.direction === 'input') || []; const pair = outs.find(out => ins.some(input => input.type === out.type && input.schemaVersion === out.schemaVersion)); return pair ? Object.freeze({ id: 'compatible-ports', type: pair.type }) : null; },
});
