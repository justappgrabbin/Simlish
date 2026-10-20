import test from 'node:test'; import assert from 'node:assert/strict'; import fs from 'node:fs'; import crypto from 'node:crypto';
import { docText, semanticUnits } from './doc-text.mjs';
import * as P from './primitives.mjs';
import { LAYERS, encodeLayer, decodeLayer, encodeParticle } from './bhantugh.mjs';
import { drawField, streams, coverage, CELL, cellIndex, motionOf, FIELD_DIMS, soundVoices, createSound } from './field.mjs';
import { ATTRIBUTION, glyphQualities } from './qualities.mjs';
import { attributionMarkdown } from './attribution.build.mjs';
import { bantuSoundFreq } from './bantu-v10.mjs';
import { FRAME_WHEEL_MAPPINGS, frameWheel } from '../play/adaptive.mjs';
const here = new URL('.', import.meta.url).pathname;
const doc = (f) => fs.readFileSync(here + 'docs/' + f);
const DOCX = 'Synthia_Semantic_Swarm_Master_Spec_v3.docx', MD = 'REASSEMBLY_MASTER_BUILD.md', PDF = 'Cynthia_core.pdf.pdftotext.txt';
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');

test('originals are byte-identical to her sources (PROVENANCE.json)', () => {
  const prov = JSON.parse(fs.readFileSync(here + 'PROVENANCE.json', 'utf8'));
  for (const o of prov.originals) assert.equal(sha(fs.readFileSync(here + o.path)), o.sha256, o.path);
});
test('her docx is read in pure JS and splits into semantic units', async () => {
  const t = await docText(DOCX, doc(DOCX)); assert.match(t, /SEMANTIC SWARM & FOUNDRY/); assert.ok(semanticUnits(t).length > 200);
});
for (const [name, f] of [['docx', DOCX], ['md', MD], ['pdf text', PDF]]) {
  test(`#1 tool factory produces particles from her ${name}`, async () => { const t = await docText(f.replace('.pdftotext.txt', '.txt'), doc(f)); const r = await P.runToolFactory(name, t, { limit: 400 }); assert.ok(r.particles.length > 0); assert.ok(r.particles.every((p) => p.primitive === 'tool-factory' && p.text && P.DIMS.includes(p.dimension))); });
  test(`#2 neural net builder produces particles from her ${name}`, async () => { const t = await docText(f.replace('.pdftotext.txt', '.txt'), doc(f)); const r = P.runNeuralBuilder(name, t, { limit: 400 }); assert.ok(r.particles.length > 0); assert.ok(r.composition.nodes >= 2); });
  test(`#3 ModelMaker's learned tokens land as particles in her ${name}`, async () => { const t = await docText(f.replace('.pdftotext.txt', '.txt'), doc(f)); const mm = JSON.parse(fs.readFileSync(here + 'modelmaker/modelmaker-run.json', 'utf8')); assert.equal(mm.vocabSize, 1024); const r = P.runModelMakerTokens(name, t, mm); assert.ok(r.particles.length > 0); });
  test(`#4 ATO engine + Foundry produces particles from her ${name}`, async () => { const t = await docText(f.replace('.pdftotext.txt', '.txt'), doc(f)); const r = await P.runAtoFoundry(name, t); assert.ok(r.particles.length > 0); assert.ok(r.particles.every((p) => ['WHO', 'WHAT', 'WHERE', 'WHEN', 'WHY'].includes(p.source.question))); });
}
test('B-H-A-N-T-U-G-H: eight layers, both H kept, invertible layers round-trip on real particles', () => {
  assert.deepEqual(LAYERS.map((l) => l.key).join(''), 'BHANTUGH'); assert.deepEqual(LAYERS.filter((l) => l.key === 'H').map((l) => l.name), ['html', 'hex']);
  const ps = JSON.parse(fs.readFileSync(here + 'particles.json', 'utf8')).particles;
  for (const p of [...ps.slice(0, 200), { text: 'Être ✓ 𝄞 — 42' }]) for (const l of LAYERS.filter((x) => x.invertible)) assert.equal(decodeLayer(l.name, encodeLayer(l.name, p.text)), p.text, l.name);
});
test('diagonal exists ONLY in Movement (encoding order and field motion)', () => {
  const ps = JSON.parse(fs.readFileSync(here + 'particles.json', 'utf8')).particles; const straight = [0, 1, 2, 3, 4, 5, 6, 7];
  for (const [i, p] of ps.entries()) { const e = encodeParticle(p, i); if (p.dimension === 'Movement') { assert.deepEqual([...e.diagonal].sort((a, b) => a - b), straight); assert.equal(e.diagonal[0], i % 8); } else assert.deepEqual(e.diagonal, straight, p.dimension); assert.deepEqual(p.diagonal, e.diagonal); }
  assert.ok(ps.some((p) => p.dimension === 'Movement' && p.diagonal[0] !== 0));
  for (const d of FIELD_DIMS) { const cols = 50, len = 997; const shift = cellIndex(d, 1, 0, cols, 0, len) - cellIndex(d, 0, 0, cols, 0, len); assert.equal(motionOf(d), d === 'Movement' ? 'diagonal' : 'straight'); assert.equal(shift, d === 'Movement' ? cols + 1 : cols, d); }
  assert.ok(LAYERS.every((l) => l.placement === 'INVESTIGATING' && !('quality' in l)));
});
test('each layer spells a function that exists in her primitive-functions registry (or is listed as a gap)', () => {
  const reg = JSON.parse(fs.readFileSync(here + '../primitive-functions.json', 'utf8')).primitives; const ids = new Set((Array.isArray(reg) ? reg : Object.values(reg)).map((x) => x.id));
  for (const l of LAYERS) if (l.registry) assert.ok(ids.has(l.registry), l.registry); else assert.match(l.note, /INVESTIGATING/);
});
test('field: five layers from real particles cover 100% of the screen (no background pixels)', () => {
  const ps = JSON.parse(fs.readFileSync(here + 'particles.json', 'utf8')).particles; const S = streams(ps); assert.equal(S.length, 5);
  for (const [W, H] of [[412, 915], [360, 640], [1080, 2400 / 2]]) {
    const a = new Uint8ClampedArray(W * H * 4); // software ctx: fillRect writes opaque alpha, fillText adds nothing to coverage
    const ctx = { set fillStyle(v) {}, set font(v) {}, set textBaseline(v) {}, set globalAlpha(v) {}, set globalCompositeOperation(v) {}, fillText() {}, fillRect(x, y, w, h) { for (let j = Math.max(0, y); j < Math.min(H, y + h); j++) for (let i = Math.max(0, x); i < Math.min(W, x + w); i++) a[(j * W + i) * 4 + 3] = 255; } };
    const info = drawField(ctx, W, H, S, 3); assert.equal(info.layers, 5); assert.equal(coverage(a), 1, W + 'x' + H);
  }
  const words = new Set(ps.flatMap((p) => p.words)); assert.ok(S.every((s) => s.map((q) => q.ch).join('').split(' ').filter(Boolean).every((w) => words.has(w) || ps.some((p) => p.text === w))));
});
test('frames x wheels: both hypotheses selectable, default unchanged', () => {
  assert.equal(frameWheel('sidereal'), 'fu-xi'); assert.equal(frameWheel('sidereal', 'hypothesis-A'), 'mawangdui'); assert.equal(frameWheel('draconic', 'hypothesis-A'), 'king-wen'); assert.equal(frameWheel('draconic', 'hypothesis-B'), 'fu-xi'); assert.ok(FRAME_WHEEL_MAPPINGS.current.default);
});

const FUNCS = JSON.parse(fs.readFileSync(here + '../primitive-sources.json', 'utf8')).functions;
test('every rendered quality maps to a registry function (or is labeled filled/instruction)', () => {
  for (const a of ATTRIBUTION) { for (const f of a.fn) assert.ok(FUNCS[f], f + ' missing from primitive-sources.json'); if (a.rendered && !a.fn.length) assert.ok(/HERS/.test(a.status), a.quality); }
  for (const need of ['shape', 'sound', 'hue', 'brightness', 'texture', 'motion', 'color (value)']) assert.ok(ATTRIBUTION.find((a) => a.quality === need && a.rendered), need);
  const q = glyphQualities('a', 'Space:G20.L2.C2.T2.B2'); assert.equal(q.gate, 27); assert.equal(q.color.name, 'HOPE'); assert.equal(q.shape.length, 6); assert.equal(q.hz, Math.round(bantuSoundFreq(q.hue / 360) * 100) / 100);
  assert.ok(!/letterColor|letterHz/.test(fs.readFileSync(here + 'field.mjs', 'utf8')));
});
test('ATTRIBUTION.md matches the code', () => { assert.equal(fs.readFileSync(here + 'ATTRIBUTION.md', 'utf8'), attributionMarkdown(FUNCS)); });
test('sound: real voices from glyph qualities, muted by default', () => {
  const ps = JSON.parse(fs.readFileSync(here + 'particles.json', 'utf8')).particles; const S = streams(ps); const v = soundVoices(S, 0, 50);
  assert.equal(v.length, 5); assert.ok(v.every((h) => h >= 20 && h <= 20000)); const snd = createSound(null); assert.equal(snd.muted, true); assert.equal(snd.unmute(), false);
});
test('texture is drawn: the Movement ground paints the two Govinda p. ix trigram signs of every cell', () => {
  const S = streams(JSON.parse(fs.readFileSync(here + 'particles.json', 'utf8')).particles); let rects = 0; const signs = new Set(['○', '□', '∪', '⌅', '△', '▽', '⊻', '∩']); let drawn = 0;
  const ctx = { set fillStyle(v) {}, set font(v) {}, set textBaseline(v) {}, set globalAlpha(v) {}, set globalCompositeOperation(v) {}, fillText(t) { if (signs.has(t)) drawn++; }, fillRect() { rects++; } };
  const info = drawField(ctx, 70, 24, S, 0); assert.equal(rects, info.cols * info.rows); assert.equal(drawn, info.cols * info.rows * 2);
});
