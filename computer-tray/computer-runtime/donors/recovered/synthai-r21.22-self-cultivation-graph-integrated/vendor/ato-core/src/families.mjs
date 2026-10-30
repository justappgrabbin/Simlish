import { Automaton } from './automaton.mjs';
import { FeatureSpace, completeAnalogy, toBitString } from './boolean-ato.mjs';
import { HOUSES, locateHexagram, transformHouse } from './klein-iching.mjs';
const defaultAddress = (gate) => ({ mode: 'macro', gate, line: 1, color: 1, tone: 1, base: 1 });
export function computationAutomaton({ id = 'computation', address = defaultAddress(1), functionalLevel = 'mind' } = {}) {
    return new Automaton({ id, address, structure: 'bigram', activeLevels: [1], functionalLevel, ports: [{ id: 'in', direction: 'input', type: 'number' }, { id: 'out', direction: 'output', type: 'number' }], metadata: { family: 'computation' }, implementation: (input, { context }) => {
            const operation = context.operation || 'identity';
            const args = Array.isArray(input) ? input : [input];
            const ops = { identity: v => v[0], sum: v => v.reduce((a, b) => Number(a) + Number(b), 0), product: v => v.reduce((a, b) => Number(a) * Number(b), 1), difference: v => v.slice(1).reduce((a, b) => Number(a) - Number(b), Number(v[0])), mean: v => v.reduce((a, b) => Number(a) + Number(b), 0) / v.length };
            if (!ops[operation])
                throw new Error(`Unknown computation: ${operation}`);
            return ops[operation](args);
        } });
}
export function semanticAutomaton({ id = 'semantic', address = defaultAddress(17), functionalLevel = 'mind', features = ['subject', 'relation', 'object', 'context', 'change', 'goal'] } = {}) {
    const space = new FeatureSpace(features);
    return new Automaton({ id, address, structure: 'trigram', activeLevels: [1, 2], functionalLevel, ports: [{ id: 'learn', direction: 'input', type: 'semantic' }, { id: 'meaning', direction: 'output', type: 'semantic' }], metadata: { family: 'semantic', features }, implementation: (input, { context }) => {
            if (input?.learn) {
                space.add(input.id, input.vector, input.metadata);
                return { learned: input.id, size: space.entries.size };
            }
            if (input?.analogy) {
                const { a, b, c, mode } = input.analogy;
                const result = space.analogy(a, b, c, mode);
                return { vector: toBitString(result.result), candidates: result.candidates.map(x => ({ id: x.id, distance: x.distance })) };
            }
            if (input?.vector)
                return space.nearest(input.vector, { limit: context.limit || 5 }).map(x => ({ id: x.id, distance: x.distance, metadata: x.metadata }));
            return { unresolved: true, reason: 'semantic input requires learn, analogy, or vector' };
        } });
}
export function codeAutomaton({ id = 'code', address = defaultAddress(31), functionalLevel = 'design' } = {}) {
    return new Automaton({ id, address, structure: 'hexagram', activeLevels: [1, 2, 3, 4, 5], functionalLevel, ports: [{ id: 'spec', direction: 'input', type: 'code' }, { id: 'source', direction: 'output', type: 'code' }], metadata: { family: 'code' }, implementation: (input) => {
            const spec = input || {};
            const name = String(spec.name || 'generatedTool').replace(/[^A-Za-z0-9_$]/g, '_');
            const params = (spec.params || ['input']).map(p => String(p).replace(/[^A-Za-z0-9_$]/g, '_'));
            const body = spec.body || 'return input;';
            return `export function ${name}(${params.join(', ')}) {\n  ${body}\n}\n`;
        } });
}
export function visualAutomaton({ id = 'visual', address = defaultAddress(44), functionalLevel = 'design' } = {}) {
    return new Automaton({ id, address, structure: 'trigram', activeLevels: [1, 3], functionalLevel, ports: [{ id: 'scene', direction: 'input', type: 'scene' }, { id: 'svg', direction: 'output', type: 'image' }], metadata: { family: 'visual' }, implementation: (input) => {
            const width = input?.width || 640, height = input?.height || 360, bg = input?.background || '#0b1320';
            const shapes = (input?.shapes || []).map(shape => shape.type === 'circle' ? `<circle cx="${shape.x}" cy="${shape.y}" r="${shape.r}" fill="${shape.fill || '#5de4d2'}"/>` : `<rect x="${shape.x}" y="${shape.y}" width="${shape.width}" height="${shape.height}" fill="${shape.fill || '#a58cff'}"/>`).join('');
            return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="${bg}"/>${shapes}</svg>`;
        } });
}
export function timelineAutomaton({ id = 'timeline', address = defaultAddress(53), functionalLevel = 'movement' } = {}) {
    return new Automaton({ id, address, structure: 'hexagram', activeLevels: [1, 2, 3, 4, 6], functionalLevel, ports: [{ id: 'events', direction: 'input', type: 'timeline' }, { id: 'frames', direction: 'output', type: 'timeline' }], metadata: { family: 'timeline' }, implementation: (input) => {
            const events = Array.isArray(input) ? input : input?.events || [];
            let time = 0;
            return events.map((event, index) => { time += Number(event.duration || 1000); return Object.freeze({ index, start: time - Number(event.duration || 1000), end: time, address: event.address || null, action: event.action || 'hold', value: event.value }); });
        } });
}
export function gameAutomaton({ id = 'game', address = defaultAddress(58), functionalLevel = 'being' } = {}) {
    const state = { tick: 0, entities: new Map() };
    return new Automaton({ id, address, structure: 'hexagram', activeLevels: [1, 2, 3, 5, 6], functionalLevel, ports: [{ id: 'command', direction: 'input', type: 'game' }, { id: 'state', direction: 'output', type: 'game' }], metadata: { family: 'game' }, implementation: (input) => {
            if (input?.spawn)
                state.entities.set(input.spawn.id, { ...input.spawn });
            if (input?.move && state.entities.has(input.move.id))
                Object.assign(state.entities.get(input.move.id), { x: input.move.x, y: input.move.y });
            if (input?.remove)
                state.entities.delete(input.remove);
            state.tick++;
            return Object.freeze({ tick: state.tick, entities: Object.freeze([...state.entities.values()].map(e => Object.freeze({ ...e }))) });
        } });
}
export function ichingAutomaton({ id = 'iching-state', address = defaultAddress(63), functionalLevel = 'movement' } = {}) {
    return new Automaton({ id, address, structure: 'hexagram', activeLevels: [1, 2, 3, 4, 5], functionalLevel, ports: [{ id: 'state', direction: 'input', type: 'state' }, { id: 'transform', direction: 'output', type: 'state' }], metadata: { family: 'state-space', houses: 8 }, implementation: (input) => {
            if (input?.bits)
                return locateHexagram(input.bits);
            if (input?.sourceHouse && input?.targetHouse)
                return transformHouse(input.sourceHouse, input.targetHouse);
            return { houses: HOUSES.map(h => ({ id: h.id, base: h.base })) };
        } });
}
export function conversationAutomaton({ id = 'conversation', address = defaultAddress(12), functionalLevel = 'space' } = {}) {
    return new Automaton({ id, address, structure: 'trigram', activeLevels: [2, 3], functionalLevel, ports: [{ id: 'context', direction: 'input', type: 'text' }, { id: 'utterance', direction: 'output', type: 'text' }], metadata: { family: 'conversation' }, implementation: (input, { context }) => {
            const contributions = context.contributions || [];
            const statements = contributions.filter(Boolean).map((item, index) => `[${index + 1}] ${typeof item === 'string' ? item : JSON.stringify(item)}`);
            return statements.length ? `${String(input).trim()}\n\n${statements.join('\n')}` : `${String(input).trim()} — context remains unresolved.`;
        } });
}
export function createPrimaryAutomata() { return Object.freeze([computationAutomaton(), semanticAutomaton(), codeAutomaton(), visualAutomaton(), timelineAutomaton(), gameAutomaton(), ichingAutomaton(), conversationAutomaton()]); }
