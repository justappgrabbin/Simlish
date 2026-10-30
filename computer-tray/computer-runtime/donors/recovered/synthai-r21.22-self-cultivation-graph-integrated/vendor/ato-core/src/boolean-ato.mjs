/* Dependency-free implementation of Sheldon Klein's Boolean ATO algebra. */
export class VectorError extends TypeError {
}
export function normalizeVector(value, width) {
    let bits;
    if (typeof value === 'string')
        bits = [...value].map(Number);
    else if (Array.isArray(value) || ArrayBuffer.isView(value))
        bits = [...value].map(Number);
    else if (Number.isInteger(value) && Number.isInteger(width) && width > 0) {
        bits = value.toString(2).padStart(width, '0').split('').map(Number);
    }
    else
        throw new VectorError('Vector must be a bit string, bit array, typed array, or an integer with width');
    if (!bits.length || bits.some((bit) => bit !== 0 && bit !== 1))
        throw new VectorError('Vector contains a value other than 0 or 1');
    if (width !== undefined && bits.length !== width)
        throw new VectorError(`Expected ${width} bits, received ${bits.length}`);
    return Object.freeze(bits);
}
export const toBitString = (vector) => normalizeVector(vector).join('');
export function xor(a, b) {
    const left = normalizeVector(a);
    const right = normalizeVector(b, left.length);
    return Object.freeze(left.map((bit, index) => bit ^ right[index]));
}
// Strong equivalence is the symmetric counterpart used in Klein's examples.
export function equivalence(a, b) {
    const left = normalizeVector(a);
    const right = normalizeVector(b, left.length);
    return Object.freeze(left.map((bit, index) => Number(bit === right[index])));
}
export function operator(a, b, mode = 'equivalence') {
    if (mode === 'xor')
        return xor(a, b);
    if (mode === 'equivalence')
        return equivalence(a, b);
    throw new VectorError(`Unknown Boolean ATO mode: ${mode}`);
}
// A:B :: C:D, where D = *C*AB. Both supported operators are involutive.
export function completeAnalogy(a, b, c, mode = 'equivalence') {
    const relation = operator(a, b, mode);
    return Object.freeze({ relation, result: operator(c, relation, mode), mode });
}
export function verifyInvolution(a, b, mode = 'equivalence') {
    const relation = operator(a, b, mode);
    return toBitString(operator(a, relation, mode)) === toBitString(b)
        && toBitString(operator(b, relation, mode)) === toBitString(a);
}
export function hammingDistance(a, b) {
    const left = normalizeVector(a);
    const right = normalizeVector(b, left.length);
    return left.reduce((distance, bit, index) => distance + Number(bit !== right[index]), 0);
}
export class FeatureSpace {
    constructor(features = []) {
        if (!Array.isArray(features) || !features.length || new Set(features).size !== features.length) {
            throw new VectorError('FeatureSpace requires unique feature names');
        }
        this.features = Object.freeze([...features]);
        this.entries = new Map();
    }
    add(id, vector, metadata = {}) {
        if (typeof id !== 'string' || !id)
            throw new VectorError('Feature entry requires an id');
        this.entries.set(id, Object.freeze({ id, vector: normalizeVector(vector, this.features.length), metadata: Object.freeze({ ...metadata }) }));
        return this;
    }
    get(id) { return this.entries.get(id); }
    nearest(vector, { limit = 5 } = {}) {
        const target = normalizeVector(vector, this.features.length);
        return [...this.entries.values()]
            .map((entry) => Object.freeze({ ...entry, distance: hammingDistance(target, entry.vector) }))
            .sort((a, b) => a.distance - b.distance || a.id.localeCompare(b.id))
            .slice(0, limit);
    }
    analogy(aId, bId, cId, mode = 'equivalence') {
        const [a, b, c] = [aId, bId, cId].map((id) => {
            const entry = this.get(id);
            if (!entry)
                throw new VectorError(`Unknown feature entry: ${id}`);
            return entry;
        });
        const completed = completeAnalogy(a.vector, b.vector, c.vector, mode);
        return Object.freeze({ ...completed, a, b, c, candidates: Object.freeze(this.nearest(completed.result)) });
    }
}
function assertSameShape(a, b, c) {
    if (Array.isArray(a)) {
        if (!Array.isArray(b) || !Array.isArray(c) || a.length !== b.length || a.length !== c.length)
            throw new VectorError('Analogy trees must have the same shape');
        return;
    }
    if (!a || !b || !c || typeof a !== 'object' || typeof b !== 'object' || typeof c !== 'object')
        throw new VectorError('Analogy tree nodes must have matching shapes');
    const keys = Object.keys(a).sort().join('|');
    if (Object.keys(b).sort().join('|') !== keys || Object.keys(c).sort().join('|') !== keys)
        throw new VectorError('Analogy trees must have the same shape');
}
export function transformTree(a, b, c, mode = 'equivalence') {
    if (typeof a === 'string' || ArrayBuffer.isView(a))
        return completeAnalogy(a, b, c, mode).result;
    assertSameShape(a, b, c);
    if (Array.isArray(a))
        return Object.freeze(a.map((value, index) => transformTree(value, b[index], c[index], mode)));
    return Object.freeze(Object.fromEntries(Object.keys(a).map((key) => [key, transformTree(a[key], b[key], c[key], mode)])));
}
// Klein's plan transform: A→B→C→D can be retargeted to E by applying *DE to every state.
export function retargetPlan(states, target, mode = 'equivalence') {
    if (!Array.isArray(states) || states.length < 2)
        throw new VectorError('A plan requires at least two states');
    const width = normalizeVector(states[0]).length;
    const normalized = states.map((state) => normalizeVector(state, width));
    const desired = normalizeVector(target, width);
    const transform = operator(normalized.at(-1), desired, mode);
    const result = normalized.map((state) => operator(state, transform, mode));
    return Object.freeze({ source: Object.freeze(normalized), target: desired, transform, result: Object.freeze(result), mode });
}
export function hierarchy(states, mode = 'equivalence') {
    if (!Array.isArray(states) || states.length < 2)
        throw new VectorError('Hierarchy requires at least two states');
    const levels = [Object.freeze(states.map((state) => normalizeVector(state)))];
    while (levels.at(-1).length > 1) {
        const current = levels.at(-1);
        levels.push(Object.freeze(current.slice(0, -1).map((state, index) => operator(state, current[index + 1], mode))));
    }
    return Object.freeze(levels);
}
