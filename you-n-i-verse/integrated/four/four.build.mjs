// Box-side build (Node): runs the four primitives on her Drive docs and writes
//   particles.json     — what the app ships (capped, see CAP)  
//   RUN-EVIDENCE.json  — full counts per primitive per doc, doc sha256s.
// ModelMaker (Python+torch) must have been run first: modelmaker/run_modelmaker.py -> modelmaker/modelmaker-run.json.
import fs from 'node:fs'; import crypto from 'node:crypto';
import { docText } from './doc-text.mjs';
import * as P from './primitives.mjs';
import { encodeParticle } from './bhantugh.mjs';
const here = new URL('.', import.meta.url).pathname;
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const DOCS = [
  { id: 'spec-docx', file: 'Synthia_Semantic_Swarm_Master_Spec_v3.docx', drive: '1qxNrFwuV9sHBM-yQapy37ESYFQ2sO3NL' },
  { id: 'reassembly-md', file: 'REASSEMBLY_MASTER_BUILD.md', drive: '1l4x0gpaZ9OwbF0M8M_lY0q2jocxf8lSW' },
  { id: 'cynthia-core-pdf', file: 'Cynthia_core.pdf.pdftotext.txt', drive: '1HJ4C3DeMGsWrJrvGMUmPb_Uqxf8TVQJu', pdfSha256: '6dc6cf95a98386ca18fa167735e8165070312650c61b7b86da2924ee78e35589', note: 'PDF not committed (2 MB); text by poppler pdftotext -layout on the box' },
];
const CAP = 120; // FILLED: particles shipped per primitive per doc (evidence counts are uncapped)
const mm = JSON.parse(fs.readFileSync(here + 'modelmaker/modelmaker-run.json', 'utf8'));
const evidence = { generated: 'four.build.mjs', docs: [], primitives: {} }; const ship = [];
for (const d of DOCS) {
  const bytes = fs.readFileSync(here + 'docs/' + d.file); const text = await docText(d.file.replace(/\.pdftotext\.txt$/, '.txt'), bytes);
  evidence.docs.push({ ...d, sha256: sha(bytes), chars: text.length });
  const runs = { 'tool-factory': await P.runToolFactory(d.id, text), 'neural-builder': P.runNeuralBuilder(d.id, text), modelmaker: P.runModelMakerTokens(d.id, text, mm), 'ato-foundry': await P.runAtoFoundry(d.id, text) };
  for (const [k, r] of Object.entries(runs)) {
    const e = (evidence.primitives[k] ??= {}); e[d.id] = { particles: r.particles.length, ...(r.unrouted ? { unrouted: r.unrouted.length } : {}), ...(r.parseErrors ? { parseErrors: r.parseErrors } : {}), ...(r.composition ? { composition: r.composition } : {}), sample: r.particles[0] ?? null };
    ship.push(...r.particles.slice(0, CAP));
  }
}
const encoded = ship.map((p, i) => { const e = encodeParticle(p, i); return { id: p.id, primitive: p.primitive, doc: p.doc, dimension: p.dimension, text: p.text.slice(0, 160), words: p.words.slice(0, 24), diagonal: e.diagonal, hue: e.note.hue, hz: e.sound.hz }; });
fs.writeFileSync(here + 'particles.json', JSON.stringify({ note: 'particles from the four primitives run on her Drive docs (four.build.mjs); hue/hz = her dimension-visual letterColor/letterHz on the first character (PROTOTYPE formulas, not from her books)', count: encoded.length, particles: encoded }));
fs.writeFileSync(here + 'RUN-EVIDENCE.json', JSON.stringify(evidence, null, 1));
console.log(JSON.stringify(Object.fromEntries(Object.entries(evidence.primitives).map(([k, v]) => [k, Object.fromEntries(Object.entries(v).map(([d, x]) => [d, x.particles]))]))));
