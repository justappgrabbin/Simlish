// Qualities of every particle glyph, each computed by a function from her sources (registry: primitive-sources.json
// `functions`). ATTRIBUTION below is the single table the field, the tests and ATTRIBUTION.md are built from.
import { encodeBytes } from '../bantu.mjs';
import { macroToGate, bitsOf, lineOf } from '../../state-space.mjs';
import { OntologicalAddress } from './originals/ontological-address/OntologicalAddress.mjs';
import { bantuSoundFreq } from './bantu-v10.mjs';

// Names of colors/tones/bases as printed in her color book (primitive-functions.json color.N / tone.N / base.N).
export const COLOR_NAMES = ['FEAR', 'HOPE', 'DESIRE', 'NEED', 'GUILT', 'INNOCENCE'];
export const TONE_NAMES = ['SECURITY', 'UNCERTENTY', 'ACTION', 'MEDITION', 'JUDGEMENT', 'ACCEPTENCE'];
export const TONE_SENSE = ['Smell', 'Taste', 'Outer Vision', 'Inner Vision', 'Feeling', 'Touch']; // primitive-attributes.json tone.N.sense

export const ATTRIBUTION = Object.freeze([
  { quality: 'glyph (character)', rendered: true, fn: [], source: 'the particle text emitted by her four primitives on her Drive docs (four/RUN-EVIDENCE.json)', status: 'HERS' },
  { quality: 'six-bit code', rendered: true, fn: ['encodeBytes'], source: 'her Bantu bridge.js lines 5-9 (integrated/bantu.mjs, verbatim)', status: 'HERS', filled: 'for a character >= 63 (her escape 63,hi,lo) the LOW token is used' },
  { quality: 'gate', rendered: true, fn: ['macroToGate'], source: 'state-space.mjs:35 Fu Xi decimal -> King Wen (vendor/kingwen.mjs from her zips); her Bantu README: a 6-bit token is six primitive positions', status: 'HERS', filled: 'reading the six-bit code as the Fu Xi decimal of a hexagram' },
  { quality: 'line / color / tone / base', rendered: true, fn: ['OntologicalAddress'], source: 'tool-factory and ATO particles carry her factory address G.L.C.T.B (integrated-tool-factory.mjs); OntologicalAddress defaults color=tone=base=1 (OntologicalAddress.js:175)', status: 'HERS', filled: 'line = 1 when the particle has no address (neural, ModelMaker); her constructor has no line default' },
  { quality: 'shape', rendered: true, fn: ['bitsOf', 'lineOf'], source: 'registry shape.hexagram (fn bitsOf, lineOf; state-space.mjs:22,25): the six lines of the glyph\'s hexagram, yang solid / yin broken', status: 'HERS', filled: 'bar thickness and gap width in pixels' },
  { quality: 'texture', rendered: true, fn: ['bitsOf', 'lineOf'], source: 'same hexagram bars, drawn as the cell ground of the Movement layer (opaque: this is what gives 100% coverage)', status: 'HERS', filled: 'using the hexagram bars as texture' },
  { quality: 'color (value)', rendered: true, fn: ['OntologicalAddress', 'DEG_PER_COLOR'], source: 'her color book: color.1-6 FEAR..INNOCENCE (primitive-functions.json, ctx color-book-lines); OntologicalAddress.js:81', status: 'HERS' },
  { quality: 'hue', rendered: true, fn: ['OntologicalAddress'], source: 'her spec §1.5 "color is computed from arc position"; arc position = OntologicalAddress eclipticDeg (OntologicalAddress.js:188-196)', status: 'HERS', filled: 'hue in degrees = ecliptic degree (identity); saturation 70% / ground 40%' },
  { quality: 'brightness', rendered: true, fn: ['computeHierarchy'], source: 'her generative level (Base/Tone/Color/line) from computeHierarchy -> computeLevel (OntologicalAddress.js:122-160)', status: 'HERS', filled: 'lightness = 38 + 8 x level (%)' },
  { quality: 'tone (value) + sense', rendered: false, fn: ['OntologicalAddress', 'DEG_PER_TONE'], source: 'her color book tone.1-6 SECURITY..ACCEPTENCE; sense from primitive-attributes.json tone.N.sense', status: 'HERS', note: 'carried on every glyph, not drawn' },
  { quality: 'sound', rendered: true, fn: ['bantuSoundFreq', 'OntologicalAddress'], source: 'her SWARM-OS v10 BANTU sound_freq = 20*1000^x (swarm_os_bantu.v10.py.txt:101,317)', status: 'HERS', filled: 'x = eclipticDeg/360 (her file uses random()); Web Audio sine voices, MUTED by default' },
  { quality: 'motion', rendered: true, fn: [], source: 'Alexis 2026-10-20: the B-H-A-N-T-U-G-H diagonal is in Movement ONLY; other dimensions run straight', status: 'HERS (instruction)', filled: 'speed = layer number + 1 cells per frame; 4 frames/s' },
  { quality: 'interference (layer stacking)', rendered: true, fn: [], source: 'Alexis 2026-10-19: five layers stack and interfere', status: 'HERS (instruction)', filled: "'lighter' compositing at alpha 0.55" },
  { quality: 'BANTU layer colors', rendered: false, fn: ['BANTU_LAYER_COLORS'], source: 'swarm_os_bantu.v10.py.txt:78', status: 'INVESTIGATING', note: 'ported; no relational place for them in the dimension field in her sources' },
  { quality: 'BANTU ASCII shapes', rendered: false, fn: ['BANTU_ASCII_SHAPES'], source: 'swarm_os_bantu.v10.py.txt:63', status: 'INVESTIGATING', note: 'ported; her file picks them at random, no relational index' },
  { quality: 'old hue/pitch (letterColor, letterHz)', rendered: false, fn: ['letterColor', 'letterHz'], source: 'dimension-visual core: builder PROVISIONAL formulas, not hers', status: 'REMOVED from the field', note: 'were used in 056bcd5' },
]);

const parseAddr = (s) => { const m = /G(\d+)\.L(\d)\.C(\d)\.T(\d)\.B(\d)/.exec(s || ''); return m ? { line: +m[2], color: +m[3], tone: +m[4], base: +m[5], from: 'particle' } : { line: 1, color: 1, tone: 1, base: 1, from: 'default' }; };
const cache = new Map();
export function glyphQualities(ch, address) {
  const key = ch + '|' + (address || ''); const hit = cache.get(key); if (hit) return hit;
  const toks = encodeBytes(new TextEncoder().encode(ch)); const six = toks[toks.length - 1] & 63;
  const gate = macroToGate(six), a = parseAddr(address);
  const oa = new OntologicalAddress({ gate, line: a.line, color: a.color, tone: a.tone, base: a.base });
  const q = Object.freeze({
    ch, six, gate, address: { ...a, string: oa.toString() },
    shape: Object.freeze([1, 2, 3, 4, 5, 6].map((l) => lineOf(six, l))), bits: bitsOf(six),
    color: { value: a.color, name: COLOR_NAMES[a.color - 1] }, tone: { value: a.tone, name: TONE_NAMES[a.tone - 1], sense: TONE_SENSE[a.tone - 1] },
    hue: Math.round(oa.eclipticDeg * 1000) / 1000, level: oa.hierarchy.level, lightness: 38 + 8 * oa.hierarchy.level,
    hz: Math.round(bantuSoundFreq(oa.eclipticDeg / 360) * 100) / 100,
  });
  cache.set(key, q); return q;
}
