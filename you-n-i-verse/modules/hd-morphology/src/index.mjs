// HD morphology science + body layer — pure data/model API. No DOM, no network, no LLM.
import { SOURCES, SOURCE_BY_ID } from '../data/sources.mjs';
import { CODONS, AMINO_ACIDS, STOP, translateCodon, codonsFor } from '../data/genetic-code.mjs';
import { ORGANS, SCIENCE_FACTS, HORMONE_PRECURSORS } from '../data/science.mjs';
import { HD_FACTS, HD_GATES, HD_CENTERS, RAVE_CODON_CLAIMS, PROJECT_RULE_NOTES, LOCKED_RULES, SOURCE_TABLES, CENTER_IDS } from '../data/hd.mjs';
import { UNSOURCED_FACTS, CENTER_REGION, LAYOUT } from '../data/unsourced.mjs';
import { computeMismatches } from './mismatches.mjs';

export const SCALES = Object.freeze(['body', 'organ', 'cell', 'dna', 'codon', 'amino-acid']);
export const LAYERS = Object.freeze({
  science: { label: 'ESTABLISHED SCIENCE', sourceKinds: ['science'] },
  hd: { label: 'HUMAN DESIGN CORRESPONDENCE', sourceKinds: ['hd', 'her-project'] },
  unsourced: { label: 'UNSOURCED / FILLED-IN', sourceKinds: [] },
});
export const VARIANTS = Object.freeze([
  { key: 'hdjs', id: 'synthia-hdjs', label: 'human-design.js (donor port)' },
  { key: 'codex', id: 'drive-codex', label: 'GateCodon Codex CSV (Drive)' },
  { key: 'dnahd', id: 'dna-hd-compiler', label: 'dna-hd-compiler gates.ts' },
  { key: 'synthai', id: 'synthai-digram', label: 'SynthAi digram rule v0.5.3' },
].map(Object.freeze));

export const FACTS = Object.freeze([...SCIENCE_FACTS, ...HD_FACTS, ...UNSOURCED_FACTS]);
const FACT_BY_ID = new Map(FACTS.map((f) => [f.id, f]));
export const factById = (id) => {
  const f = FACT_BY_ID.get(id);
  if (!f) throw new RangeError(`unknown fact ${id}`);
  return f;
};
export const factsFor = (subject, { scale, layer } = {}) =>
  FACTS.filter((f) => f.subject === subject && (!scale || f.scale === scale) && (!layer || f.layer === layer));

export const centers = () => HD_CENTERS;
export const center = (id) => HD_CENTERS.find((c) => c.id === id) ?? null;
export const gates = () => HD_GATES;
export const gate = (n) => HD_GATES.find((g) => g.gate === Number(n)) ?? null;
export const organ = (id) => (ORGANS[id] ? { id, ...ORGANS[id] } : null);

/** Everything known about one gate, grouped by layer and scale. */
export function describeGate(n) {
  const g = gate(n);
  if (!g) throw new RangeError('gate must be 1..64');
  const c = center(g.center);
  const aa = g.aminoAcid === '*' ? STOP : AMINO_ACIDS[g.aminoAcid];
  return Object.freeze({
    ...g, centerLabel: c.label, organs: c.organs.map(organ),
    aminoAcidInfo: aa ?? null,
    scienceCodonsForClaimedAminoAcid: g.aminoAcid ? codonsFor(g.aminoAcid).map((x) => x.rna) : [],
    hd: factsFor(`gate:${n}`, { layer: 'hd' }).filter((f) => f.status === 'claim'),
    variants: factsFor(`gate:${n}`, { layer: 'hd' }).filter((f) => f.status === 'variant'),
    mismatches: mismatches().filter((m) => m.subject === `gate:${n}` || m.basis.some((b) => b.startsWith(`hd.gate.${n}.`))),
  });
}

/** One center zoomed through every scale: organs → cells → DNA → codons → amino acids. */
export function describeCenter(id) {
  const c = center(id);
  if (!c) throw new RangeError(`unknown center ${id}`);
  const organSubjects = c.organs.map((o) => `organ:${o}`);
  const byScale = Object.fromEntries(SCALES.map((s) => [s, Object.freeze([
    ...factsFor(`center:${id}`, { scale: s }),
    ...organSubjects.flatMap((o) => factsFor(o, { scale: s })),
    ...(['cell', 'dna', 'codon', 'amino-acid'].includes(s) ? factsFor(`scale:${s}`, { scale: s }) : []),
  ])]));
  const aas = [...new Set(c.gates.map((g) => gate(g).aminoAcid))];
  return Object.freeze({ ...c, region: CENTER_REGION[id], organs: c.organs.map(organ), byScale,
    gateAminoAcids: aas.map((a) => ({ one: a, name: a === '*' ? 'Stop' : AMINO_ACIDS[a].name, codons: codonsFor(a).map((x) => x.rna),
      gates: c.gates.filter((g) => gate(g).aminoAcid === a) })) });
}

let MM;
export function mismatches() { return (MM ||= buildMismatches()).mismatches; }
export function consistencyChecks() { return (MM ||= buildMismatches()).checks; }
export function dataset() {
  return { codons: CODONS, aminoAcids: AMINO_ACIDS, organs: ORGANS, hdCenters: HD_CENTERS, hdGates: HD_GATES, centerRegion: CENTER_REGION,
    raveClaims: RAVE_CODON_CLAIMS, sourceTables: SOURCE_TABLES, projectNotes: PROJECT_RULE_NOTES, lockedRules: LOCKED_RULES,
    hormonePrecursors: HORMONE_PRECURSORS, variants: VARIANTS, factById };
}
function buildMismatches() { return computeMismatches(dataset()); }

export { SOURCES, SOURCE_BY_ID, CODONS, AMINO_ACIDS, STOP, translateCodon, codonsFor, ORGANS, LAYOUT, CENTER_IDS, LOCKED_RULES, computeMismatches };
