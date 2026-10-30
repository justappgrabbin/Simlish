import { Automaton } from './automaton.mjs';
import { completeAnalogy, toBitString } from './boolean-ato.mjs';
import { TRIGRAMS } from './klein-iching.mjs';
export class KleinToolError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'KleinToolError'; this.code = code; this.details = details; }
}
const addr = (gate) => ({ mode: 'macro', gate, line: 1, color: 1, tone: 1, base: 1 });
const port = (id, direction, type, guarantees = []) => ({ id, direction, type, schemaVersion: '1', requires: [], guarantees });
const tokenize = (text) => String(text).toLowerCase().match(/[a-z0-9_'-]+/g) || [];
const seeded = (seed = 1) => { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
const normalize = (weights) => { const total = Object.values(weights).reduce((sum, n) => sum + n, 0) || 1; return Object.freeze(Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v / total]))); };
export class AutoLingMemory {
    constructor() { this.rules = new Map(); this.sequence = 0; }
    learn({ id, relation, pattern, template, arity = null, provenance = null } = {}) {
        const ruleId = id || `rule-${++this.sequence}`;
        if (!relation || !Array.isArray(pattern) || !pattern.length || typeof template !== 'string')
            throw new KleinToolError('INVALID_GRAMMAR_RULE', 'Rule requires relation, token pattern, and template');
        const variables = [...new Set(pattern.filter(token => token.startsWith('$')))];
        const rule = Object.freeze({ id: ruleId, relation, pattern: Object.freeze([...pattern]), template, arity: arity ?? variables.length, variables: Object.freeze(variables), provenance, revision: 1 });
        this.rules.set(ruleId, rule);
        return rule;
    }
    recognize(text) {
        const tokens = tokenize(text);
        const matches = [];
        for (const rule of this.rules.values()) {
            if (tokens.length !== rule.pattern.length)
                continue;
            const bindings = {};
            let valid = true;
            for (let i = 0; i < tokens.length; i++) {
                const expected = rule.pattern[i];
                if (expected.startsWith('$')) {
                    const prior = bindings[expected];
                    if (prior && prior !== tokens[i]) {
                        valid = false;
                        break;
                    }
                    bindings[expected] = tokens[i];
                }
                else if (expected.toLowerCase() !== tokens[i]) {
                    valid = false;
                    break;
                }
            }
            if (valid)
                matches.push(Object.freeze({ ruleId: rule.id, relation: rule.relation, bindings: Object.freeze({ ...bindings }), structure: Object.freeze(rule.variables.map(v => bindings[v])) }));
        }
        return Object.freeze({ ok: matches.length > 0, text: String(text), matches: Object.freeze(matches) });
    }
    generate(relation, args = []) { const rules = [...this.rules.values()].filter(rule => rule.relation === relation).sort((a, b) => a.id.localeCompare(b.id)); if (!rules.length)
        return Object.freeze({ ok: false, reason: 'RULE_NOT_FOUND', relation }); const rule = rules[0]; if (args.length !== rule.arity)
        return Object.freeze({ ok: false, reason: 'ARITY_MISMATCH', expected: rule.arity, received: args.length }); let surface = rule.template; rule.variables.forEach((variable, index) => { surface = surface.replaceAll(variable, args[index]); }); return Object.freeze({ ok: true, relation, ruleId: rule.id, surface }); }
    export() { return Object.freeze([...this.rules.values()]); }
}
export class DiseminerMemory {
    constructor({ windowSize = 2 } = {}) { this.windowSize = windowSize; this.vectors = new Map(); this.documents = []; this.sequence = 0; }
    ingest(text, { source = null, address = null } = {}) { const tokens = tokenize(text), document = Object.freeze({ id: `doc-${++this.sequence}`, source, address, tokens: Object.freeze(tokens) }); this.documents.push(document); for (let i = 0; i < tokens.length; i++) {
        const term = tokens[i];
        if (!this.vectors.has(term))
            this.vectors.set(term, new Map());
        const vector = this.vectors.get(term);
        for (let j = Math.max(0, i - this.windowSize); j <= Math.min(tokens.length - 1, i + this.windowSize); j++) {
            if (i === j)
                continue;
            const context = tokens[j];
            vector.set(context, (vector.get(context) || 0) + 1);
        }
    } return document; }
    similarity(a, b) { const x = this.vectors.get(String(a).toLowerCase()), y = this.vectors.get(String(b).toLowerCase()); if (!x || !y)
        return 0; const keys = new Set([...x.keys(), ...y.keys()]); let dot = 0, ax = 0, by = 0; for (const key of keys) {
        const xv = x.get(key) || 0, yv = y.get(key) || 0;
        dot += xv * yv;
        ax += xv * xv;
        by += yv * yv;
    } return ax && by ? dot / (Math.sqrt(ax) * Math.sqrt(by)) : 0; }
    neighbors(term, { limit = 5 } = {}) { const key = String(term).toLowerCase(); if (!this.vectors.has(key))
        return Object.freeze([]); return Object.freeze([...this.vectors.keys()].filter(other => other !== key).map(other => Object.freeze({ term: other, similarity: this.similarity(key, other) })).sort((a, b) => b.similarity - a.similarity || a.term.localeCompare(b.term)).slice(0, limit)); }
    infer(a, c, { minimum = 0 } = {}) { const candidates = this.neighbors(a, { limit: this.vectors.size }).filter(item => item.similarity >= minimum && this.similarity(item.term, c) > 0).map(item => Object.freeze({ via: item.term, score: (item.similarity + this.similarity(item.term, c)) / 2 })).sort((x, y) => y.score - x.score || x.via.localeCompare(y.via)); return Object.freeze({ source: a, target: c, candidates: Object.freeze(candidates) }); }
    export() { return Object.freeze({ windowSize: this.windowSize, documents: Object.freeze([...this.documents]), vectors: Object.freeze(Object.fromEntries([...this.vectors].map(([term, v]) => [term, Object.freeze(Object.fromEntries(v))]))) }); }
}
export class AutoNovelMemory {
    constructor() { this.domains = new Map(); this.examples = []; this.sequence = 0; }
    register(domain) { if (!domain?.id || !Array.isArray(domain.primitives) || !Array.isArray(domain.combinators))
        throw new KleinToolError('INVALID_DOMAIN', 'AutoNovel domain requires id, primitives, and combinators'); const frozen = Object.freeze({ ...domain, primitives: Object.freeze(domain.primitives.map(x => Object.freeze({ ...x }))), combinators: Object.freeze(domain.combinators.map(x => Object.freeze({ ...x }))) }); this.domains.set(domain.id, frozen); return frozen; }
    learn(structure) { this.examples.push(structuredClone(structure)); return this.examples.length; }
    generate({ domain: domainId, seeds = [], maxDepth = 8 } = {}) {
        const domain = this.domains.get(domainId);
        if (!domain)
            return Object.freeze({ ok: false, reason: 'DOMAIN_NOT_FOUND' });
        let nodes = seeds.map((seed, index) => Object.freeze({ id: `node-${index + 1}`, type: seed.type, value: seed.value ?? seed.type, features: Object.freeze({ ...seed.features }) }));
        const relations = [];
        const used = new Set();
        for (let depth = 0; depth < maxDepth; depth++) {
            const possible = [];
            for (const combinator of domain.combinators) {
                const left = nodes.find(n => n.type === combinator.inputs?.[0]), right = nodes.find(n => n.type === combinator.inputs?.[1] && n.id !== left?.id);
                if (left && right) {
                    const key = `${combinator.id}:${left.id}:${right.id}`;
                    if (!used.has(key))
                        possible.push({ combinator, left, right, key });
                }
            }
            if (!possible.length)
                break;
            possible.sort((a, b) => a.combinator.id.localeCompare(b.combinator.id) || a.key.localeCompare(b.key));
            const chosen = possible[0];
            used.add(chosen.key);
            const created = Object.freeze({ id: `node-${nodes.length + 1}`, type: chosen.combinator.output, value: chosen.combinator.apply ? chosen.combinator.apply(chosen.left.value, chosen.right.value) : [chosen.left.value, chosen.right.value], features: Object.freeze({ ...(chosen.left.features || {}), ...(chosen.right.features || {}) }) });
            nodes = [...nodes, created];
            relations.push(Object.freeze({ type: chosen.combinator.type || 'combine', source: Object.freeze([chosen.left.id, chosen.right.id]), target: created.id, combinator: chosen.combinator.id }));
        }
        return Object.freeze({ ok: true, id: `structure-${++this.sequence}`, domain: domainId, nodes: Object.freeze(nodes), relations: Object.freeze(relations), lineage: Object.freeze(relations.map(r => r.combinator)), generationDepth: relations.length });
    }
}
export class MessyMemory {
    simulate({ agents = [], rules = [], ticks = 5 } = {}) { let state = agents.map(a => ({ id: a.id, features: { ...a.features } })); const history = [Object.freeze({ tick: 0, state: structuredClone(state) })]; for (let tick = 1; tick <= ticks; tick++) {
        const previous = state;
        state = previous.map(agent => { const world = Object.freeze({ tick, agents: structuredClone(previous) }); const rule = rules.find(candidate => candidate.when(agent, world)); return rule ? rule.apply(structuredClone(agent), world) : agent; });
        history.push(Object.freeze({ tick, state: structuredClone(state) }));
    } return Object.freeze({ ok: true, ticks, finalState: Object.freeze(state), history: Object.freeze(history) }); }
}
const create = ({ id, gate, family, level = 'mind', state = null, implementation, input = 'json', output = 'json', metadata = {} }) => new Automaton({ id, address: addr(gate), structure: 'hexagram', activeLevels: [1, 2, 3, 4, 5], functionalLevel: level, ports: [port('in', 'input', input), port('out', 'output', output, ['addressed'])], state, implementation, metadata: { family, kleinTool: true, ...metadata } });
export function autoLingAutomaton({ memory = new AutoLingMemory() } = {}) { if (memory.rules.size === 0) {
    memory.learn({ id: 'has-property.v1', relation: 'HAS_PROPERTY', pattern: ['$object', 'is', '$property'], template: '$object is $property' });
    memory.learn({ id: 'part-of.v1', relation: 'PART_OF', pattern: ['$part', 'belongs', 'to', '$whole'], template: '$part belongs to $whole' });
} return create({ id: 'autoling', gate: 17, family: 'autoling', state: memory, implementation: (input, { state }) => input.operation === 'learn' ? state.learn(input.rule) : input.operation === 'recognize' ? state.recognize(input.text) : input.operation === 'generate' ? state.generate(input.relation, input.args) : input.operation === 'export' ? state.export() : Object.freeze({ ok: false, reason: 'UNKNOWN_OPERATION' }) }); }
export function diseminerAutomaton({ memory = new DiseminerMemory() } = {}) { return create({ id: 'diseminer', gate: 48, family: 'diseminer', state: memory, implementation: (input, { state }) => input.operation === 'ingest' ? state.ingest(input.text, input.context) : input.operation === 'neighbors' ? state.neighbors(input.term, input.options) : input.operation === 'infer' ? state.infer(input.source, input.target, input.options) : input.operation === 'export' ? state.export() : Object.freeze({ ok: false, reason: 'UNKNOWN_OPERATION' }) }); }
export function analogyAutomaton() { return create({ id: 'klein-analogy', gate: 4, family: 'analogy', implementation: (input) => { const vocab = Object.freeze([...new Set(input.vocab)]), bits = (items) => vocab.map(f => items.includes(f) ? 1 : 0), result = completeAnalogy(bits(input.A), bits(input.B), bits(input.C), 'xor'); return Object.freeze({ ok: true, relation: toBitString(result.relation), result: Object.freeze(vocab.filter((_, i) => result.result[i] === 1)) }); } }); }
export function iChingGrammarAutomaton() { return create({ id: 'iching-grammar', gate: 61, family: 'iching-grammar', implementation: (input) => { if (!Array.isArray(input.lines) || input.lines.length !== 6 || input.lines.some(x => x !== 0 && x !== 1))
        return Object.freeze({ ok: false, reason: 'SIX_BINARY_LINES_REQUIRED' }); const lower = input.lines.slice(0, 3).join(''), upper = input.lines.slice(3).join(''), find = (bits) => Object.entries(TRIGRAMS).find(([, v]) => v.bits === bits); return Object.freeze({ ok: true, lower: Object.freeze({ id: find(lower)?.[0] || null, bits: lower }), upper: Object.freeze({ id: find(upper)?.[0] || null, bits: upper }), binaryValue: Number.parseInt(input.lines.slice().reverse().join(''), 2) }); } }); }
export function languageContactAutomaton() { return create({ id: 'language-contact', gate: 12, family: 'language-contact', implementation: (input) => { const rng = seeded(input.seed ?? 1), rate = input.contactRate ?? 0.15; let grammar = { ...input.grammarA }; const trajectory = [Object.freeze({ generation: 0, grammar: Object.freeze({ ...grammar }) })]; for (let generation = 1; generation <= (input.generations ?? 10); generation++) {
        const next = { ...grammar };
        for (const rule of Object.keys(input.grammarB).sort())
            if (rng() < rate)
                next[rule] = input.grammarB[rule];
        grammar = next;
        trajectory.push(Object.freeze({ generation, grammar: Object.freeze({ ...grammar }) }));
    } return Object.freeze({ ok: true, finalGrammar: Object.freeze(grammar), trajectory: Object.freeze(trajectory) }); } }); }
export function historicalMonteCarloAutomaton() { return create({ id: 'historical-monte-carlo', gate: 32, family: 'historical-monte-carlo', implementation: (input) => { const rng = seeded(input.seed ?? 7), scale = input.mutationScale ?? 0.1; let weights = { ...input.variants }; const trajectory = [Object.freeze({ generation: 0, weights: normalize(weights) })]; for (let generation = 1; generation <= (input.generations ?? 20); generation++) {
        weights = Object.fromEntries(Object.entries(weights).map(([name, value]) => [name, Math.max(0, value * (1 + (rng() - .5) * 2 * scale))]));
        trajectory.push(Object.freeze({ generation, weights: normalize(weights) }));
    } const final = normalize(weights), dominant = Object.entries(final).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] || null; return Object.freeze({ ok: true, finalWeights: final, dominantVariant: dominant, trajectory: Object.freeze(trajectory) }); } }); }
export function autoNovelAutomaton({ memory = new AutoNovelMemory() } = {}) { return create({ id: 'autonovel', gate: 56, family: 'autonovel', state: memory, level: 'design', implementation: (input, { state }) => input.operation === 'register' ? state.register(input.domain) : input.operation === 'learn' ? state.learn(input.structure) : input.operation === 'generate' ? state.generate(input.spec) : Object.freeze({ ok: false, reason: 'UNKNOWN_OPERATION' }) }); }
export function messyAutomaton({ memory = new MessyMemory() } = {}) { return create({ id: 'messy', gate: 3, family: 'messy', state: memory, level: 'movement', implementation: (input, { state }) => state.simulate(input) }); }
export function bootstrapKleinTools(mesh, { include = [] } = {}) { const factories = [autoLingAutomaton, diseminerAutomaton, analogyAutomaton, iChingGrammarAutomaton, languageContactAutomaton, historicalMonteCarloAutomaton, autoNovelAutomaton, messyAutomaton]; const selected = include.length ? factories.filter(factory => include.includes(factory.name)) : factories; const tools = []; for (const factory of selected) {
    const tool = factory();
    if (!mesh.automatons.has(tool.id))
        mesh.add(tool);
    tools.push(tool);
} return Object.freeze(tools); }
