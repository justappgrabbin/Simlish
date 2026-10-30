import { toBitString } from './boolean-ato.mjs';
export const ADDRESS_FIELDS = Object.freeze([
    'planetary', 'dimension', 'gate', 'line', 'color', 'tone', 'base',
    'degree', 'minute', 'second', 'arc', 'zodiac', 'house', 'scope', 'originContext',
]);
export class AddressResolutionError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'AddressResolutionError'; this.code = code; this.details = details; }
}
const integer = (value, min, max) => Number.isInteger(value) && value >= min && value <= max;
export function normalizeAddress(input, { mode = input?.mode || 'macro', allowUnresolved = false } = {}) {
    const value = input || {};
    const required = mode === 'micro'
        ? ADDRESS_FIELDS
        : ['gate', 'line', 'color', 'tone', 'base'];
    const unresolved = required.filter((field) => value[field] === undefined || value[field] === null || value[field] === '');
    if (unresolved.length && !allowUnresolved)
        throw new AddressResolutionError('UNRESOLVED_ADDRESS', 'Required address coordinates are missing', { unresolved });
    const ranges = { gate: [1, 64], line: [1, 6], color: [1, 6], tone: [1, 6], base: [1, 5], degree: [0, 359], minute: [0, 59], second: [0, 59], zodiac: [1, 12], house: [1, 12] };
    const invalid = Object.entries(ranges).filter(([field, [min, max]]) => value[field] !== undefined && !integer(value[field], min, max)).map(([field]) => field);
    if (invalid.length)
        throw new AddressResolutionError('INVALID_ADDRESS', 'Address coordinates are outside their valid ranges', { invalid });
    return Object.freeze({ mode, ...Object.fromEntries(ADDRESS_FIELDS.filter((field) => value[field] !== undefined).map((field) => [field, value[field]])), unresolved: Object.freeze(unresolved) });
}
export function macroProjection(address) {
    const normalized = normalizeAddress(address, { mode: address?.mode || 'macro', allowUnresolved: true });
    return normalizeAddress(normalized, { mode: 'macro', allowUnresolved: true });
}
export function canonicalAddress(address) {
    const normalized = normalizeAddress(address, { mode: address?.mode || 'macro', allowUnresolved: true });
    return `${normalized.mode}:${ADDRESS_FIELDS.filter((field) => normalized[field] !== undefined).map((field) => `${field}=${encodeURIComponent(String(normalized[field]))}`).join('|')}`;
}
export function addressed(value, address, { kind = 'state', provenance = null } = {}) {
    const coordinate = normalizeAddress(address, { mode: address?.mode || 'macro' });
    return Object.freeze({ kind, address: coordinate, addressKey: canonicalAddress(coordinate), value, provenance });
}
export function transition(source, target, operator, { context = null, lineage = [] } = {}) {
    if (!source?.address || !target?.address)
        throw new AddressResolutionError('UNADDRESSED_TRANSITION', 'Transitions require addressed source and target states');
    return Object.freeze({
        source: source.addressKey || canonicalAddress(source.address),
        target: target.addressKey || canonicalAddress(target.address),
        operator,
        context,
        lineage: Object.freeze([...lineage]),
    });
}
// Projects the 64-node Gate coordinate into the six-bit state used by the I Ching/ATO substrate.
export function gateState(address) {
    const normalized = normalizeAddress(address, { mode: address?.mode || 'macro', allowUnresolved: true });
    if (!integer(normalized.gate, 1, 64))
        throw new AddressResolutionError('UNRESOLVED_GATE', 'A Gate coordinate is required to resolve the 64-state substrate');
    return Object.freeze({ gate: normalized.gate, index: normalized.gate - 1, bits: toBitString((normalized.gate - 1).toString(2).padStart(6, '0')) });
}
export class AddressSpace {
    constructor() { this.states = new Map(); this.edges = new Map(); }
    put(state) {
        if (!state?.address)
            throw new AddressResolutionError('UNADDRESSED_STATE', 'State must carry an address');
        const item = Object.freeze({ ...state, addressKey: state.addressKey || canonicalAddress(state.address) });
        this.states.set(item.addressKey, item);
        return item;
    }
    connect(edge) {
        if (!edge?.source || !edge?.target)
            throw new AddressResolutionError('UNADDRESSED_EDGE', 'Edge must carry source and target address keys');
        if (!this.states.has(edge.source) || !this.states.has(edge.target))
            throw new AddressResolutionError('NOT_REACHABLE', 'Both addressed states must exist before they can be connected', { source: edge.source, target: edge.target });
        const key = `${edge.source}->${edge.target}:${String(edge.operator)}`;
        this.edges.set(key, Object.freeze({ ...edge, key }));
        return this.edges.get(key);
    }
    resolve(address) { return this.states.get(canonicalAddress(address)) || null; }
    reachable(sourceAddress, targetAddress) {
        const source = canonicalAddress(sourceAddress);
        const target = canonicalAddress(targetAddress);
        return [...this.edges.values()].some((edge) => edge.source === source && edge.target === target);
    }
}
