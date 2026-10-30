export class InterpretationError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'InterpretationError'; this.code = code; this.details = details; }
}
const stable = value => { const text = JSON.stringify(value, Object.keys(value || {}).sort()); let h = 2166136261; for (const char of text)
    h = Math.imul(h ^ char.charCodeAt(0), 16777619); return (h >>> 0).toString(16).padStart(8, '0'); };
export class InterpretationRegistry {
    constructor() { this.sources = new Map(); this.derived = new Map(); }
    registerSource({ id, title, mechanism, provenance, version = 'historical.v1' } = {}) { if (!id || !mechanism)
        throw new InterpretationError('INVALID_SOURCE', 'Source needs id and mechanism'); if (this.sources.has(id))
        throw new InterpretationError('SOURCE_IMMUTABLE', 'A source record cannot be overwritten'); const record = Object.freeze({ id, title, mechanism: structuredClone(mechanism), provenance, version, immutable: true }); this.sources.set(id, record); return record; }
    derive(sourceId, { layer = 'modern', mappingVersion, changes = {}, profileId = null } = {}) { const source = this.sources.get(sourceId); if (!source)
        throw new InterpretationError('SOURCE_NOT_FOUND', sourceId); if (!['modern', 'personal', 'emergent'].includes(layer))
        throw new InterpretationError('INVALID_LAYER', layer); const id = `${sourceId}:${layer}:${mappingVersion || 'v1'}:${stable({ changes, profileId })}`; const record = Object.freeze({ id, sourceId, layer, mappingVersion: mappingVersion || 'v1', profileId, mechanism: Object.freeze({ ...structuredClone(source.mechanism), ...structuredClone(changes) }), lineage: Object.freeze([sourceId]), reversible: true }); this.derived.set(id, record); return record; }
    resolve(id) { return this.sources.get(id) || this.derived.get(id) || null; }
    compare(sourceId, derivedId) { const source = this.sources.get(sourceId), derived = this.derived.get(derivedId); if (!source || !derived)
        throw new InterpretationError('INTERPRETATION_NOT_FOUND', 'Both records required'); return Object.freeze({ source, derived, changed: Object.freeze(Object.keys(derived.mechanism).filter(key => JSON.stringify(source.mechanism[key]) !== JSON.stringify(derived.mechanism[key]))) }); }
}
