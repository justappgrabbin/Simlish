/**
 * The 5 Orderings of the 64 Hexagrams — these become the 5 State Space layers.
 *
 * 1. Fu Xi       — binary order (natural / Heaven's sequence)
 * 2. Mawangdui   — upper-trigram grouped (silk manuscript archaeology)
 * 3. King Wen    — received/traditional sequence (the living I Ching)
 * 4. Reverse     — reverse Fu Xi (mirror / descending)
 * 5. Complement  — bitwise NOT of Fu Xi (shadow / opposite)
 *
 * Each sequence is an array of 64 fuxi values in the order of that sequence.
 * Position in the sequence = the node's rank within that layer.
 */

import { HEXAGRAMS, BY_FUXI } from './hexagrams.js';

// ── Mawangdui upper-trigram ordering ──────────────────────────────────────────
// Mawangdui groups hexagrams by upper trigram. The order of upper trigrams in
// the manuscript (per Shaughnessy 1997 reconstruction) is:
//   Qian, Kun, Zhen, Gen, Kan, Li, Dui, Xun
// Within each group, lower trigrams follow the same order.
const MAWANGDUI_TRIGRAM_ORDER = [7, 0, 1, 4, 2, 5, 3, 6]; // binary values

function mawangduiRank(fuxi) {
  const upperBits = (fuxi >> 3) & 0b111; // bits 3-5
  const lowerBits = fuxi & 0b111;        // bits 0-2
  const upperRank = MAWANGDUI_TRIGRAM_ORDER.indexOf(upperBits);
  const lowerRank = MAWANGDUI_TRIGRAM_ORDER.indexOf(lowerBits);
  return upperRank * 8 + lowerRank;
}

// ── Build all 5 sequences ─────────────────────────────────────────────────────
function buildSequences() {
  const allFuxi = HEXAGRAMS.map(h => h.fuxi); // 0-63 in King Wen order

  // 1. Fu Xi: sort by binary value ascending (Earth=0 first, Heaven=63 last)
  const fuxiSeq = [...Array(64).keys()]; // [0,1,2,...,63]

  // 2. Mawangdui: sort by mawangdui rank
  const mawangduiSeq = [...Array(64).keys()].sort(
    (a, b) => mawangduiRank(a) - mawangduiRank(b)
  );

  // 3. King Wen: the received sequence — fuxi values in KW order
  const kingWenSeq = HEXAGRAMS
    .slice()
    .sort((a, b) => a.kw - b.kw)
    .map(h => h.fuxi);

  // 4. Reverse Fu Xi: descending binary (Heaven=63 first, Earth=0 last)
  const reverseFuxiSeq = [...fuxiSeq].reverse(); // [63,62,...,0]

  // 5. Complement Fu Xi: XOR each fuxi with 63 (flip all 6 lines), then sort by result
  // This maps each hexagram to its line-complement, maintaining binary order of complements
  const complementSeq = fuxiSeq.map(f => f ^ 63); // position i → complement of hexagram i

  return {
    fuxi:       { id: 'fuxi',       name: 'Fu Xi',      description: 'Binary natural sequence — Heaven to Earth or Earth to Heaven. The innate / Xiantian order.', sequence: fuxiSeq },
    mawangdui:  { id: 'mawangdui',  name: 'Mawangdui',  description: 'Archaeological silk-manuscript sequence — grouped by upper trigram. Recovered from Mawangdui tomb, 168 BCE.', sequence: mawangduiSeq },
    kingwen:    { id: 'kingwen',    name: 'King Wen',   description: 'Traditional received sequence — the living I Ching. Relational, narrative, paired by inversion/complement.', sequence: kingWenSeq },
    reverse:    { id: 'reverse',    name: 'Reverse Fu Xi', description: 'Descending binary — temporal inversion. Reads time backward; used for retrospective causal tracing.', sequence: reverseFuxiSeq },
    complement: { id: 'complement', name: 'Complement Fu Xi', description: 'Bitwise NOT of Fu Xi — the shadow sequence. Every node is the line-opposite of its Fu Xi counterpart.', sequence: complementSeq },
  };
}

export const SEQUENCES = buildSequences();

// Given a sequence id and a fuxi value, return the node's rank (0-63) in that sequence
export function rankInSequence(seqId, fuxi) {
  return SEQUENCES[seqId].sequence.indexOf(fuxi);
}

// Given a sequence id and a rank (0-63), return the fuxi value at that position
export function fuxiAtRank(seqId, rank) {
  return SEQUENCES[seqId].sequence[rank];
}

export const SEQUENCE_IDS = ['fuxi', 'mawangdui', 'kingwen', 'reverse', 'complement'];
