import test from 'node:test'; import assert from 'node:assert/strict'; import fs from 'node:fs'; import crypto from 'node:crypto';
import { docText, semanticUnits } from './doc-text.mjs';
import * as P from './primitives.mjs';
import { LAYERS, encodeLayer, decodeLayer, encodeParticle } from './bhantugh.mjs';
import { drawField, streams, coverage, CELL } from './field.mjs';
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
  const e = encodeParticle(ps[5], 5); assert.deepEqual([...e.diagonal].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7]); assert.equal(e.diagonal[0], 5);
});
test('each layer spells a function that exists in her primitive-functions registry (or is listed as a gap)', () => {
  const reg = JSON.parse(fs.readFileSync(here + '../primitive-functions.json', 'utf8')).primitives; const ids = new Set((Array.isArray(reg) ? reg : Object.values(reg)).map((x) => x.id));
  for (const l of LAYERS) if (l.registry) assert.ok(ids.has(l.registry), l.registry); else assert.match(l.note, /INVESTIGATING/);
  assert.deepEqual([...new Set(LAYERS.map((l) => l.quality))], ['Movement', 'Design', 'Space']);
});
test('field: five layers from real particles cover 100% of the screen (no background pixels)', () => {
  const ps = JSON.parse(fs.readFileSync(here + 'particles.json', 'utf8')).particles; const S = streams(ps); assert.equal(S.length, 5);
  for (const [W, H] of [[412, 915], [360, 640], [1080, 2400 / 2]]) {
    const a = new Uint8ClampedArray(W * H * 4); // software ctx: fillRect writes opaque alpha, fillText adds nothing to coverage
    const ctx = { set fillStyle(v) {}, set font(v) {}, set textBaseline(v) {}, set globalAlpha(v) {}, set globalCompositeOperation(v) {}, fillText() {}, fillRect(x, y, w, h) { for (let j = Math.max(0, y); j < Math.min(H, y + h); j++) for (let i = Math.max(0, x); i < Math.min(W, x + w); i++) a[(j * W + i) * 4 + 3] = 255; } };
    const info = drawField(ctx, W, H, S, 3); assert.equal(info.layers, 5); assert.equal(coverage(a), 1, W + 'x' + H);
  }
  const words = new Set(ps.flatMap((p) => p.words)); assert.ok(S.every((s) => s.join('').split(' ').filter(Boolean).every((w) => words.has(w) || ps.some((p) => p.text === w))));
});
test('frames x wheels: both hypotheses selectable, default unchanged', () => {
  assert.equal(frameWheel('sidereal'), 'fu-xi'); assert.equal(frameWheel('sidereal', 'hypothesis-A'), 'mawangdui'); assert.equal(frameWheel('draconic', 'hypothesis-A'), 'king-wen'); assert.equal(frameWheel('draconic', 'hypothesis-B'), 'fu-xi'); assert.ok(FRAME_WHEEL_MAPPINGS.current.default);
});
