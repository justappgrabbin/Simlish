import { Automaton } from './automaton.mjs';
export const PROMOTION_ORDER = Object.freeze(['bigram', 'trigram', 'hexagram', 'decagram']);
export const STRUCTURAL_SIZE = Object.freeze({ bigram: 2, trigram: 3, hexagram: 6, decagram: 10 });
export class PromotionError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'PromotionError'; this.code = code; this.details = details; }
}
const kindOf = tool => tool.metadata?.structuralKind || tool.structure?.structure;
export class StructuralPromoter {
    constructor({ rules = {} } = {}) { this.rules = new Map(Object.entries(rules)); this.history = []; }
    register(from, to, rule) { if (!PROMOTION_ORDER.includes(from) || !PROMOTION_ORDER.includes(to) || typeof rule !== 'function')
        throw new PromotionError('INVALID_PROMOTION_RULE', 'Promotion rule is invalid'); this.rules.set(`${from}->${to}`, rule); return this; }
    next(kind) { return PROMOTION_ORDER[PROMOTION_ORDER.indexOf(kind) + 1] || null; }
    promote(tools, { targetKind = null, address, functionalLevel = 'space', implementation = null } = {}) {
        if (!Array.isArray(tools) || tools.length < 2)
            throw new PromotionError('INSUFFICIENT_COMPONENTS', 'Promotion requires at least two Tool-Automatons');
        const kinds = [...new Set(tools.map(kindOf))];
        if (kinds.length !== 1)
            throw new PromotionError('MIXED_STRUCTURAL_KINDS', 'Components must share a structural kind', { kinds });
        const sourceKind = kinds[0], target = targetKind || this.next(sourceKind);
        if (!target)
            throw new PromotionError('NO_HIGHER_STRUCTURE', sourceKind);
        const rule = this.rules.get(`${sourceKind}->${target}`);
        if (!rule)
            return Object.freeze({ status: 'unresolved', reason: 'VERSIONED_COMPOSITION_RULE_REQUIRED', sourceKind, targetKind: target });
        const resolution = rule(tools, { address, functionalLevel });
        if (!resolution?.valid)
            return Object.freeze({ status: 'not-reachable', reason: resolution?.reason || 'PROMOTION_RULE_FAILED', details: resolution });
        const id = resolution.id || `${target}:${tools.map(t => t.id).sort().join('+')}`;
        const size = STRUCTURAL_SIZE[target];
        const activeLevels = resolution.activeLevels || Array.from({ length: Math.max(1, size - 1) }, (_, i) => i + 1);
        const compositeImplementation = implementation || resolution.implementation || (async (input, context) => { let value = input; for (const tool of tools)
            value = await tool.call(value, { ...context, parent: id }); return value; });
        const promoted = new Automaton({ id, address, structure: { levelCount: size, minimumActive: activeLevels.length }, activeLevels, functionalLevel, ports: resolution.ports || [{ id: 'in', direction: 'input', type: 'json' }, { id: 'out', direction: 'output', type: 'json' }], implementation: compositeImplementation, metadata: { family: 'promoted-structure', structuralKind: target, components: tools.map(t => t.id), componentAddresses: tools.map(t => t.addressKey), operatorVersion: resolution.operatorVersion || 'unspecified', decomposable: true } });
        this.history.push(Object.freeze({ sourceKind, targetKind: target, components: Object.freeze(tools.map(t => t.id)), result: promoted.id, operatorVersion: promoted.metadata.operatorVersion }));
        return Object.freeze({ status: 'promoted', automaton: promoted });
    }
}
