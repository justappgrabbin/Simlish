// The four swarm primitives, run on her documents. WRAPPERS ONLY: every primitive below is her code,
// imported unmodified from ./originals (sha256 in four/PROVENANCE.json). What is FILLED IN by the builder
// is marked FILLED. Each primitive returns particles = semantic units: {id, primitive, doc, unit, text, dimension, words, source}.
import { IntegratedToolFactory, ATONativeBridge } from './originals/tool-factory/src/integrated-tool-factory.mjs';
import { Automaton, AutomataMesh } from './originals/ato-core/src/index.mjs';
import FoundryEngine from './originals/foundry/foundry-engine.mjs';
import { generateEmergentComposition, runComposition, ARCHITECTURE_FAMILY_IDS } from './originals/neural/architectureModulation.mjs';
import { semanticUnits } from './doc-text.mjs';

export const DIMS = ['Movement', 'Evolution', 'Being', 'Design', 'Space'];
// FILLED: FNV-1a 32-bit, used only for particle ids and seeds (deterministic; no Math.random).
export const fnv = (s) => { let h = 0x811c9dc5; for (const c of String(s)) { h ^= c.codePointAt(0); h = Math.imul(h, 0x01000193) >>> 0; } return h >>> 0; };
const pid = (p, doc, i, t) => p + '-' + fnv(doc + '|' + i + '|' + t).toString(16).padStart(8, '0');
const wordsOf = (t) => (t.toLowerCase().match(/[\p{L}\p{N}_-]+/gu) || []);

// #1 Tool factory: her IntegratedToolFactory routes each unit with its own DimensionRouter, generates a tool, executes it on the unit.
export async function runToolFactory(doc, text, { limit = Infinity } = {}) {
  const f = new IntegratedToolFactory(); const out = []; const unrouted = [];
  for (const [i, u] of semanticUnits(text).slice(0, limit).entries()) {
    const g = f.generate({ input: u });
    if (!g.tool) { unrouted.push({ unit: i, status: g.status }); continue; }
    let r; try { r = await g.tool.execute(u); } catch (e) { unrouted.push({ unit: i, error: e.code || e.message }); continue; }
    const m = g.tool.manifest();
    out.push({ id: pid('tf', doc, i, u), primitive: 'tool-factory', doc, unit: i, text: u, dimension: m.dimension || g.tool.plan?.dimension, words: Array.isArray(r?.nodes) ? r.nodes : wordsOf(u), source: { tool: m.name ?? g.tool.name, address: r?.address ?? m.addressKey ?? null } });
  }
  return { particles: out, unrouted };
}

// #2 Neural net builder: her generateEmergentComposition (seed FILLED = fnv(doc) mod 9973 + 1) built once per doc,
// runComposition on a 4-vector per unit. FILLED: the 4-vector (letters, vowels, digits, punctuation fractions)
// and dimension = DIMS[sum of node classIndex mod 5]. Both are builder choices, INVESTIGATING.
export function runNeuralBuilder(doc, text, { limit = Infinity } = {}) {
  const comp = generateEmergentComposition({ seed: fnv(doc) % 9973 + 1, nodeCount: 5 }); const out = [];
  for (const [i, u] of semanticUnits(text).slice(0, limit).entries()) {
    const n = u.length || 1, c = (re) => (u.match(re) || []).length / n;
    const run = runComposition(comp, [c(/\p{L}/gu), c(/[aeiou]/gi), c(/\d/g), c(/[^\p{L}\d\s]/gu)]);
    const cls = Object.values(run.predictions || {}).map((p) => p.classIndex ?? 0);
    out.push({ id: pid('nn', doc, i, u), primitive: 'neural-builder', doc, unit: i, text: u, dimension: DIMS[cls.reduce((a, b) => a + b, 0) % 5], words: wordsOf(u), source: { composition: comp.id, families: comp.nodes.map((x) => x.architectureId), classIndex: cls } });
  }
  return { particles: out, composition: { id: comp.id, nodes: comp.nodes.length, edges: comp.edges.length, familiesAvailable: ARCHITECTURE_FAMILY_IDS.length } };
}

// GPT-2 byte-level alphabet inverse (the standard bytes_to_unicode table used by ModelMaker's ByteLevel BPE).
const B2U = (() => { const bs = []; for (let b = 33; b <= 126; b++) bs.push(b); for (let b = 161; b <= 172; b++) bs.push(b); for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs]; let n = 0; for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); } const m = new Map(); bs.forEach((b, i) => m.set(String.fromCodePoint(cs[i]), b)); return m; })();
export function byteLevelDecode(tok) { try { return new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from(Array.from(tok), (c) => B2U.get(c) ?? 63)); } catch { return ''; } }
// #3 ModelMaker: her Python train_model() ran on the box (four/modelmaker/run_modelmaker.py); its learned BPE tokens
// are the particles. Here we place each learned token at the doc units that contain it. FILLED: dimension = DIMS[token id mod 5] (INVESTIGATING).
export function runModelMakerTokens(doc, text, run, { limit = Infinity } = {}) {
  const units = semanticUnits(text).slice(0, limit); const out = [];
  for (const { id, token } of run.learnedTokens) {
    const t = byteLevelDecode(token).trim(); if ((t.match(/\p{L}/gu) || []).length < 3) continue;
    const i = units.findIndex((u) => u.includes(t)); if (i < 0) continue;
    out.push({ id: pid('mm', doc, id, t), primitive: 'modelmaker', doc, unit: i, text: t, dimension: DIMS[id % 5], words: [t.toLowerCase()], source: { tokenId: id } });
  }
  return { particles: out };
}

// #4 ATO engine with Foundry: her ATONativeBridge mounts a factory tool as a genuine ATO Automaton in an AutomataMesh and runs it;
// her FoundryEngine.compile parses each unit into WHO/WHAT/WHERE/WHEN/WHY/HOW clauses. A particle is emitted for each unit
// whose Foundry parse is valid (her single-question rule); the ATO run supplies the words. Foundry's own ids are uuid/Math.random
// (her code) so particle ids are re-derived with fnv.
export async function runAtoFoundry(doc, text, { limit = Infinity } = {}) {
  const mesh = new AutomataMesh(); const bridge = new ATONativeBridge({ Automaton, mesh, factory: new IntegratedToolFactory() });
  const { automaton } = bridge.generateAndMount({ purpose: 'orchestrate a workflow system', dimension: 'Space' });
  const fe = new FoundryEngine(); const out = []; const parseErrors = {};
  for (const [i, u] of semanticUnits(text).slice(0, limit).entries()) {
    const st = fe.compile(u);
    if (!st.parse?.valid) { const k = st.error?.code || 'invalid'; parseErrors[k] = (parseErrors[k] || 0) + 1; continue; }
    const ato = await bridge.run(automaton.id, u); const o = Object.values(ato.outputs)[0];
    for (const cl of st.parse.clauses) out.push({ id: pid('ato', doc, i, cl.text), primitive: 'ato-foundry', doc, unit: i, text: cl.text, dimension: DIMS[(cl.base - 1 + 5) % 5], words: o?.nodes || wordsOf(u), source: { question: cl.question, base: cl.base, atoRun: ato.runId, atoAddress: o?.address ?? null } });
  }
  return { particles: out, parseErrors, automaton: automaton.id };
}
