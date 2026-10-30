import { Automaton } from './automaton.mjs';
export class GrammarCoderError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'GrammarCoderError'; this.code = code; this.details = details; }
}
const ALL = Object.freeze(['ADJ', 'ADV', 'NOUN', 'VERB']);
const FUNCTION_WORDS = Object.freeze({ a: ['ART'], an: ['ART'], the: ['ART'], and: ['CONJC'], or: ['CONJC'], nor: ['CONJC'], but: ['CONJR'], is: ['VERB_IS'], was: ['VERB_IS'], be: ['VERB_IS'], are: ['VERB_IS'], am: ['VERB_IS'], to: ['PREP_1'], of: ['PREP_OF'], i: ['PN_S'], we: ['PN_S'], he: ['PN_S'], she: ['PN_S'], they: ['PN_S'], me: ['PN_O'], us: ['PN_O'], him: ['PN_O'], them: ['PN_O'], my: ['PN_POS'], our: ['PN_POS'], his: ['PN_POS'], their: ['PN_POS'], this: ['PN_DEM'], those: ['PN_DEM'], that: ['PN_REL'], who: ['PN_RCN'], which: ['PN_RCN'], where: ['PN_IND'], anywhere: ['PN_IND'], somewhere: ['PN_IND'] });
const SUFFIX_RULES = Object.freeze([{ suffix: 'ing', codes: ['/ING/'] }, { suffix: 'ed', codes: ['/ED/'] }, { suffix: 'ity', codes: ['NOUN'] }, { suffix: 'ness', codes: ['NOUN'] }, { suffix: 'ment', codes: ['NOUN'] }, { suffix: 'tion', codes: ['NOUN'] }, { suffix: 'ous', codes: ['ADJ'] }, { suffix: 'ful', codes: ['ADJ'] }, { suffix: 'ly', codes: ['ADV'] }, { suffix: 'ize', codes: ['VERB'] }, { suffix: 'ise', codes: ['VERB'] }, { suffix: 's', codes: ['NOUN', 'VERB'] }]);
const intersect = (sets) => sets.reduce((a, b) => a.filter(x => b.includes(x)));
export class ComputationalGrammarCoder {
    constructor({ dictionary = {}, contextFrames = [] } = {}) { this.dictionary = new Map(Object.entries({ ...FUNCTION_WORDS, ...dictionary }).map(([k, v]) => [k.toLowerCase(), Object.freeze([...v])])); this.contextFrames = contextFrames.map(frame => Object.freeze([...frame])); }
    candidates(word, { sentenceInitial = false } = {}) { const raw = String(word), lower = raw.toLowerCase(), tests = []; if (/^[.,;:!?]$/.test(raw))
        tests.push([raw === '.' ? '.TYPE' : ',TYPE']); if (/^\d+$/.test(raw))
        tests.push(['ADJ']); if (!sentenceInitial && /^[A-Z]/.test(raw))
        tests.push(['NOUN', 'ADJ']); const dictionary = this.dictionary.get(lower); if (dictionary)
        tests.push([...dictionary]); const suffix = SUFFIX_RULES.find(rule => lower.endsWith(rule.suffix)); if (suffix)
        tests.push([...suffix.codes]); if (!tests.length)
        tests.push([...ALL]); const common = intersect(tests); return Object.freeze({ word: raw, tests: Object.freeze(tests.map(x => Object.freeze(x))), codes: Object.freeze(common.length ? common : [...new Set(tests.flat())]), ambiguous: (common.length ? common : new Set(tests.flat())).length > 1 }); }
    code(sentence) { const words = String(sentence).match(/[A-Za-z0-9'-]+|[.,;:!?]/g) || [], items = words.map((word, index) => this.candidates(word, { sentenceInitial: index === 0 })); this.#applyFrames(items); return Object.freeze({ sentence: String(sentence), words: Object.freeze(items.map(item => Object.freeze({ ...item }))), ambiguous: Object.freeze(items.filter(item => item.codes.length > 1).map(item => item.word)) }); }
    #applyFrames(items) { for (let left = 0; left < items.length; left++) {
        if (items[left].codes.length !== 1)
            continue;
        for (let right = left + 1; right < items.length; right++) {
            if (items[right].codes.length !== 1)
                continue;
            const middle = items.slice(left + 1, right);
            if (!middle.length || middle.length > 3 || middle.every(x => x.codes.length === 1))
                continue;
            const frames = this.contextFrames.filter(frame => frame.length === items.length ? true : frame.length === middle.length + 2 && frame[0] === items[left].codes[0] && frame.at(-1) === items[right].codes[0]);
            for (let i = 0; i < middle.length; i++) {
                const allowed = [...new Set(frames.flatMap(frame => frame[i + 1] || []))];
                if (allowed.length) {
                    const reduced = middle[i].codes.filter(code => allowed.includes(code));
                    if (reduced.length)
                        middle[i].codes = reduced;
                }
            }
            break;
        }
    } }
}
export function computationalGrammarCoderAutomaton({ coder = new ComputationalGrammarCoder() } = {}) { return new Automaton({ id: 'computational-grammar-coder', address: { mode: 'macro', gate: 62, line: 1, color: 1, tone: 1, base: 1 }, structure: 'hexagram', activeLevels: [1, 2, 3, 4, 5], functionalLevel: 'mind', ports: [{ id: 'text', direction: 'input', type: 'text', schemaVersion: '1' }, { id: 'codes', direction: 'output', type: 'grammar-codes', schemaVersion: '1', guarantees: ['source-trace'] }], state: coder, metadata: { family: 'computational-grammar-coder', source: 'Klein & Simmons 1963', interpretation: 'modern-executable.v1' }, implementation: (input, { state }) => state.code(typeof input === 'string' ? input : input.text) }); }
