// LAYER 2 — HUMAN DESIGN CORRESPONDENCE. Claims as HD (or HD-derived) literature and Alexis's
// own project tables state them. These are NOT presented as proven; the UI badges every one.
// status: 'claim'   = HD literature (Black Book 1992, Rave I'Ching 2001, Gene Keys)
//         'variant' = one of Alexis's own project tables/rules; kept side by side, never merged
// Built deterministically from data/her-sources.mjs (generated read-only from her files).
import HER from './her-sources.mjs';
import { aminoAcidFromPrinted } from './genetic-code.mjs';

export const LOCKED_RULES = Object.freeze({
  primitives: Object.freeze(['Movement', 'Evolution', 'Being', 'Design']),
  spaceIsPrimitive: false,
  note: "Alexis's locked rule: four primitives; Space is not a primitive. This module assigns no primitive to any body part.",
});

export const CENTER_IDS = Object.freeze(['Head', 'Ajna', 'Throat', 'G', 'Heart', 'SolarPlexus', 'Spleen', 'Sacral', 'Root']);
export const CENTER_LABEL = Object.freeze({ Head: 'Head', Ajna: 'Ajna', Throat: 'Throat', G: 'G (Identity)', Heart: 'Heart (Ego)',
  SolarPlexus: 'Solar Plexus', Spleen: 'Spleen', Sacral: 'Sacral', Root: 'Root (Base of the Spine)' });

// Black Book center biology string -> organ ids in data/science.mjs ORGANS
const BIO_TO_ORGANS = [[/liver/i, 'liver'], [/ovar/i, 'ovary'], [/test/i, 'testis'], [/pituitar/i, 'pituitary'], [/kidney/i, 'kidney'],
  [/pancreas/i, 'pancreas'], [/parathyroid/i, 'parathyroid'], [/(^|[^a-z])thyroid/i, 'thyroid'], [/spleen/i, 'spleen'], [/lymph/i, 'lymphatic'],
  [/adrenal/i, 'adrenal'], [/thymus/i, 'thymus'], [/pineal/i, 'pineal']];
export function organsFromPrinted(text) {
  if (!text) return [];
  return [...new Set(BIO_TO_ORGANS.filter(([re]) => re.test(text)).map(([, id]) => id))];
}

const h = (id, subject, scale, text, cites, locator, status = 'claim', data = undefined) =>
  Object.freeze({ id, layer: 'hd', subject, scale, text, cites: Object.freeze(cites), locator, status, ...(data ? { data: Object.freeze(data) } : {}) });

const facts = [];
const gates = [];
for (const g of HER.gates) {
  const n = g.gate, s = `gate:${n}`;
  const bb = g.blackBook, gk = g.geneKeys;
  facts.push(h(`hd.gate.${n}.keynote`, s, 'body', g.raveKeynote, ['hd-rave-iching'], 'gate page'));
  facts.push(h(`hd.gate.${n}.center`, s, 'body', `Belongs to the ${CENTER_LABEL[bb.center]} center (as printed, OCR: "${bb.centerAsPrinted}").`, ['hd-blackbook'], `PDF p.${bb.pdfPage}`, 'claim', { center: bb.center, from: bb.centerFrom }));
  if (bb.biologyAsPrinted) facts.push(h(`hd.gate.${n}.biology`, s, 'organ', `Biology: ${bb.biologyAsPrinted}.`, ['hd-blackbook'], `PDF p.${bb.pdfPage}`, 'claim', { organs: organsFromPrinted(bb.biologyAsPrinted) }));
  const aa = aminoAcidFromPrinted(gk.aminoAcidAsPrinted);
  facts.push(h(`hd.gate.${n}.physiology`, s, 'organ', `Physiology: ${gk.physiologyAsPrinted}.`, ['hd-gene-keys'], `text line ${gk.textLine}`, 'claim', { organs: organsFromPrinted(gk.physiologyAsPrinted) }));
  facts.push(h(`hd.gate.${n}.aminoAcid`, s, 'amino-acid', `${gk.codonRing}; amino acid: ${aa ? aa.name : 'UNRESOLVED (' + gk.aminoAcidAsPrinted + ')'}.`, ['hd-gene-keys'], `text line ${gk.textLine}`, 'claim',
    { ring: gk.codonRing, ringMembers: gk.ringMembersAsPrinted, aminoAcid: aa ? aa.one : null }));
  facts.push(h(`hd.gate.${n}.codonTriplet`, s, 'codon', 'Which single codon triplet this gate is: not stated in the HD literature found (UNRESOLVED). See project variants.', ['hd-gene-keys', 'hd-rave-iching'], 'searched', 'claim', { triplet: null, unresolved: true }));
  facts.push(h(`hd.gate.${n}.hexagram`, s, 'dna', `Hexagram ${n}, ${g.kingWen.name}; lines bottom-first ${g.kingWen.linesBottomFirst}. Zhouyi header: "${g.zhouyi.headerAsPrinted}".`, ['synthia-kingwen', 'zhouyi-rutt'], `kingwen.js table; Rutt printed p.${g.zhouyi.printedPage}`, 'claim', { lines: g.kingWen.linesBottomFirst }));
  // project variants (Alexis's own tables) — never merged into the claim above
  const v = [
    ['hdjs', 'synthia-hdjs', aminoAcidFromPrinted(g.donorHdJs.aminoAcid), null, g.donorHdJs.organ, 'GATES[n].aminoAcid / organ'],
    ['codex', 'drive-codex', aminoAcidFromPrinted(g.driveCodex.codonFamily), null, null, 'Codon Family column'],
    ['dnahd', 'dna-hd-compiler', aminoAcidFromPrinted(g.dnaHdCompiler.aminoAcid), g.dnaHdCompiler.codon, g.dnaHdCompiler.organ, 'GATES[n].codon'],
    ['synthai', 'synthai-digram', aminoAcidFromPrinted(g.synthaiRule.aminoAcidAsComputed === '*' ? 'NONE' : g.synthaiRule.aminoAcidAsComputed), g.synthaiRule.rnaCodon, null, 'translateGate(n)'],
  ];
  for (const [k, src, vaa, codon, organ, loc] of v) {
    facts.push(h(`hd.gate.${n}.variant.${k}`, s, codon ? 'codon' : 'amino-acid',
      `${codon ? 'Codon ' + codon + ' → ' : ''}${vaa ? vaa.name : 'no amino acid given'}${organ ? '; organ: ' + organ : ''}.`, [src], loc, 'variant',
      { aminoAcid: vaa ? vaa.one : null, codon, organ }));
  }
  gates.push(Object.freeze({ gate: n, name: g.kingWen.name, keynote: g.raveKeynote, center: bb.center, lines: g.kingWen.linesBottomFirst,
    ring: gk.codonRing, aminoAcid: aa ? aa.one : null, zhouyiHeader: g.zhouyi.headerAsPrinted }));
}

const centers = CENTER_IDS.map((id) => {
  const bio = HER.centerBiologyBlackBook[id];
  const printed = HER.gates.find((g) => g.blackBook.center === id && g.blackBook.centerFrom === 'printed')?.blackBook.centerAsPrinted;
  facts.push(h(`hd.center.${id}.name`, `center:${id}`, 'body', `Printed as "${printed}".`, ['hd-blackbook'], 'gate pages'));
  facts.push(h(`hd.center.${id}.biology`, `center:${id}`, 'organ', `Biology: ${bio.biology} (stated on ${Object.values(bio.votes).reduce((a, b) => a + b, 0)} gate pages).`, ['hd-blackbook'], 'per-gate "Biology:" fields', 'claim', { organs: organsFromPrinted(bio.biology) }));
  return Object.freeze({ id, label: CENTER_LABEL[id], printed, organs: Object.freeze(organsFromPrinted(bio.biology)),
    gates: Object.freeze(gates.filter((g) => g.center === id).map((g) => g.gate)) });
});
const centersDoc = HER.centersFromDoc.filter((r) => r.Biology);
for (const r of centersDoc) facts.push(h(`hd.centersdoc.${r.Center}`, `center:${r.Center}`, 'organ', `Her centers table: ${r.Center} → ${r.Biology}.`, ['drive-centers-doc'], 'row', 'variant', { organs: organsFromPrinted(r.Biology) }));

// Rave I'Ching codon statements (verbatim, short)
export const RAVE_CODON_CLAIMS = Object.freeze([
  { id: 'hd.rave.41.start', gate: 41, claim: 'only-start', quote: "the 41st gate is the only 'Initiator (start) Codon'", cites: ['hd-rave-iching'], locator: 'text line ' + HER.raveCodonStatements[0].line },
  { id: 'hd.rave.56.stop', gate: 56, claim: 'stop', quote: "the 56th gate in its genetic relationship, is a 'stop Codon'", cites: ['hd-rave-iching'], locator: 'text line ' + HER.raveCodonStatements[1].line },
].map(Object.freeze));
for (const c of RAVE_CODON_CLAIMS) facts.push(h(c.id, `gate:${c.gate}`, 'codon', `Rave I'Ching: "${c.quote}".`, c.cites, c.locator));

export const PROJECT_RULE_NOTES = Object.freeze([
  { id: 'note.synthai.dimensions', source: 'synthai-digram', dimensionsListed: ['Design', 'Movement', 'Being', 'Evolution', 'Space'],
    text: 'SynthAi v0.5.3 docs (DNA-RNA-PROTEIN-TRANSLATION) map five dimensions to DNA/RNA/Ribosome/Amino acid/Protein, with Space → Protein.' },
]);

export const HD_GATES = Object.freeze(gates);
export const HD_CENTERS = Object.freeze(centers);
export const HD_FACTS = Object.freeze(facts);
export const SOURCE_TABLES = Object.freeze({ mergedCenters: HER.mergedCenters, channels: HER.channels,
  donorCenters: Object.fromEntries(HER.gates.map((g) => [g.gate, g.donorHdJs.center])) });
