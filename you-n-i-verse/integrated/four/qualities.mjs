// Qualities of every particle glyph, each computed by a function from her sources (registry: primitive-sources.json
// `functions`). ATTRIBUTION below is the single table the field, the tests and ATTRIBUTION.md are built from.
import { encodeBytes } from '../bantu.mjs';
import { macroToGate, bitsOf, lineOf } from '../../state-space.mjs';
import { OntologicalAddress } from './originals/ontological-address/OntologicalAddress.mjs';
import { bantuSoundFreq } from './bantu-v10.mjs';
import { trigramOfValue, trigramShape, trigramElement, ELEMENT_RGB_FILLED, FIVE_TONES_P51, hexagramElement, elementColour } from '../sources/govinda.mjs';

// Names of colors/tones/bases as printed in her color book (primitive-functions.json color.N / tone.N / base.N).
export const COLOR_NAMES = ['FEAR', 'HOPE', 'DESIRE', 'NEED', 'GUILT', 'INNOCENCE'];
export const TONE_NAMES = ['SECURITY', 'UNCERTENTY', 'ACTION', 'MEDITION', 'JUDGEMENT', 'ACCEPTENCE'];
export const TONE_SENSE = ['Smell', 'Taste', 'Outer Vision', 'Inner Vision', 'Feeling', 'Touch']; // primitive-attributes.json tone.N.sense

export const ATTRIBUTION = Object.freeze([
  { quality: 'glyph (character)', rendered: true, fn: [], source: 'the particle text emitted by her four primitives on her Drive docs (four/RUN-EVIDENCE.json)', status: 'HERS' },
  { quality: 'six-bit code', rendered: true, fn: ['encodeBytes'], source: 'her Bantu bridge.js lines 5-9 (integrated/bantu.mjs, verbatim)', status: 'HERS', filled: 'for a character >= 63 (her escape 63,hi,lo) the LOW token is used' },
  { quality: 'gate', rendered: true, fn: ['macroToGate'], source: 'state-space.mjs:35 Fu Xi decimal -> King Wen (vendor/kingwen.mjs from her zips); her Bantu README: a 6-bit token is six primitive positions', status: 'HERS', filled: 'reading the six-bit code as the Fu Xi decimal of a hexagram' },
  { quality: 'line / color / tone / base', rendered: true, fn: ['OntologicalAddress'], source: 'tool-factory and ATO particles carry her factory address G.L.C.T.B (integrated-tool-factory.mjs); OntologicalAddress defaults color=tone=base=1 (OntologicalAddress.js:175)', status: 'HERS', filled: 'line = 1 when the particle has no address (neural, ModelMaker); her constructor has no line default' },
  { quality: 'hexagram lines', rendered: false, fn: ['bitsOf', 'lineOf'], source: 'registry shape.hexagram (fn bitsOf, lineOf; state-space.mjs:22,25): the six lines of the glyph\'s hexagram, yang solid / yin broken', status: 'HERS', note: 'carried on every glyph; since this pass the ground draws the two trigram signs (texture row) instead of the six bars' },
  { quality: 'shape', rendered: true, fn: ['trigramShape', 'trigramOfValue'], source: 'Govinda p. ix ABBREV signs of the glyph\'s upper and lower trigrams (replaces the random BANTU v10 shapes; v10 constants kept as reference)', status: 'SOURCED', filled: 'Unicode stand-ins for the printed signs (⌅ Wind, ⊻ Lake are readings); 6px size' },
  { quality: 'texture', rendered: true, fn: ['trigramShape'], source: 'the two trigram signs, upper over lower, on the opaque Movement ground (replaces the hexagram-bar texture of b430902; opaque ground gives 100% coverage)', status: 'SOURCED', filled: 'layout: upper sign at top of cell, lower sign below' },
  { quality: 'trigram colour', rendered: true, fn: ['trigramElement'], source: 'Govinda pp.46-47 trigram directions + p.57 element places and colours (p.53 print agrees)', status: 'SOURCED (DERIVED for the four cardinal trigrams)', filled: 'RGB rendering of the colour names; the four intercardinal trigrams have no stated element -> INVESTIGATING, drawn in the glyph hue instead' },
  { quality: 'five tones', rendered: false, fn: ['FIVE_TONES_P51'], source: 'Govinda p.51 mentions five tones; no names or element mapping are printed', status: 'INVESTIGATING', note: 'sound stays BANTU v10' },
  { quality: 'color (value)', rendered: true, fn: ['OntologicalAddress', 'DEG_PER_COLOR'], source: 'her color book: color.1-6 FEAR..INNOCENCE (primitive-functions.json, ctx color-book-lines); OntologicalAddress.js:81', status: 'HERS' },
  { quality: 'drawn colour (glyph and trigram sign)', rendered: true, fn: ['hexagramElement', 'trigramElement'], source: 'INHERITED colour rule (Alexis 2026-10-20; Govinda pp.57-60): one colour per element group, every member the same; glyph inherits when both its trigrams share an element', status: 'SOURCED', filled: 'RGB rendering of the colour names; neutral grey = unattributed' },
  { quality: 'hue', rendered: false, fn: ['OntologicalAddress'], source: 'her spec §1.5 "color is computed from arc position"; arc position = OntologicalAddress eclipticDeg (OntologicalAddress.js:188-196)', status: 'HERS', filled: 'hue in degrees = ecliptic degree (identity)', note: 'carried, NO LONGER DRAWN: a computed hue broke the inherited element colour' },
  { quality: 'brightness', rendered: false, fn: ['computeHierarchy'], source: 'her generative level (Base/Tone/Color/line) from computeHierarchy -> computeLevel (OntologicalAddress.js:122-160)', status: 'HERS', filled: 'lightness = 38 + 8 x level (%)', note: 'carried, not drawn: drawn colour is the inherited element colour only' },
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
    trigrams: Object.freeze(Object.fromEntries([['upper', (six >> 3) & 7], ['lower', six & 7]].map(([k, v]) => { const t = trigramOfValue(v), e = trigramElement(t); return [k, Object.freeze({ trigram: t, shape: trigramShape(t), element: e.element, elementStatus: e.status, rgb: elementColour(e.element) })]; }))),
    element: hexagramElement(six), rgb: elementColour(hexagramElement(six)), colourStatus: hexagramElement(six) ? 'INHERITED' : 'unattributed',
    hz: Math.round(bantuSoundFreq(oa.eclipticDeg / 360) * 100) / 100,
  });
  cache.set(key, q); return q;
}
