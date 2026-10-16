// Pure Synthia Automata — state-space: 5-dimension x 64-node mesh StateSpace
//
// Ported from COHERENT donor substrate/mesh072/core/state_space_core.mjs
// (Synthia-OS-v2.0.0-COHERENT, ranked PORT #1 in docs/corpus/unique-pieces-survey-2.md).
// Fix-then-integrate changes vs the donor:
//   1. The donor takes the 64-gate table as a constructor argument and embeds
//      none ("pass the real table in"). Here we also export buildGateTable(),
//      which derives that table from OUR verified canonical source:
//      addressing.js KING_WEN_TO_FUXI_DECIMAL + gateBits() (line 1 = LSB,
//      yang = 1) and constants.js TRIGRAMS (Fu Xi values). No second
//      competing table is typed from memory — the exact defect class the
//      donor header warns about.
//   2. Epistemic status is carried as claim-status.js claims
//      (DIMENSION_CLAIMS) instead of the donor's loose `confirmed` flags.
//      The donor's own header flags are preserved verbatim in the claims'
//      evidence, so nothing goes behind curtains.
//
// Donor header semantics (preserved):
//   Dimension -> mesh-layer role -> I-Ching sequence -> chart-method,
//   per direct correction (Aug 12 2026):
//     Movement  = knowledge  layer, sequence 'reverse'    (UNCONFIRMED)
//     Evolution = causal     layer, Mawangdui sequence, Sidereal
//     Being     = state      layer, Fuxi sequence, Tropical (chart carried
//                 over from an earlier triad — NOT reconfirmed; partial)
//     Design    = temporal   layer, King Wen sequence,  Draconic
//     Space     = dependency layer, sequence 'complement' (UNCONFIRMED)
//   NO Crystals / Magnetic Monopole framing — explicitly rejected for the
//   state space by direct instruction (kept: that framing stays out).
//
// Pure JS ESM, zero deps, browser file://-safe, deterministic (no wall-clock,
// no randomness).

import { CLAIM_STATUS, claim } from './claim-status.js';
import { gateBits, KING_WEN_TO_FUXI_DECIMAL } from './addressing.js';
import { TRIGRAMS } from './constants.js';

/* Dimension -> role/sequence/chart spec. The donor's `confirmed` flag is
 * mapped to the claim-status vocabulary in DIMENSION_CLAIMS below; this bare
 * table stays for structural use (ordering code reads it directly). */
export const DIMENSIONS = Object.freeze({
  Movement:  Object.freeze({ layerRole: 'knowledge',  sequence: 'reverse',    chart: null }),
  Evolution: Object.freeze({ layerRole: 'causal',     sequence: 'mawangdui',  chart: 'sidereal' }),
  Being:     Object.freeze({ layerRole: 'state',      sequence: 'fuxi',       chart: 'tropical' }),
  Design:    Object.freeze({ layerRole: 'temporal',   sequence: 'kingwen',    chart: 'draconic' }),
  Space:     Object.freeze({ layerRole: 'dependency', sequence: 'complement', chart: null }),
});

/* Epistemic status per dimension assignment (claim-status idiom).
 * - Evolution/Mawangdui and Design/King Wen are source-anchored (see sources).
 * - Movement='reverse' and Space='complement' are the donor's own unconfirmed
 *   default assignments ("the two remaining named transforms ... assigned
 *   here as the most reasonable default ... Flag/override if wrong") — kept
 *   as PROJECT_HYPOTHESIS, never asserted.
 * - Being chart 'tropical' is 'partial' in the donor — hypothesis. */
export const DIMENSION_CLAIMS = Object.freeze({
  Movement: claim(DIMENSIONS.Movement, {
    status: CLAIM_STATUS.PROJECT_HYPOTHESIS,
    hypothesisId: 'H-MESHSS-MOVEMENT-REVERSE',
    evidence: { donorFlag: 'confirmed: false', note: 'donor default: gateReverse (line-order reversal), unconfirmed by user' },
  }),
  Evolution: claim(DIMENSIONS.Evolution, {
    status: CLAIM_STATUS.SOURCE_STATEMENT,
    source: 'Mawangdui silk manuscript octet grouping by upper trigram (family order Qian, Gen, Kan, Zhen, Kun, Dui, Li, Xun); donor substrate/mesh072/core/state_space_core.mjs header, confirmed per direct correction Aug 12 2026',
  }),
  Being: claim(DIMENSIONS.Being, {
    status: CLAIM_STATUS.PROJECT_HYPOTHESIS,
    hypothesisId: 'H-MESHSS-BEING-TROPICAL',
    evidence: { donorFlag: "confirmed: 'partial'", note: 'chart carried over from earlier Mind/Body/Heart triad, not reconfirmed' },
  }),
  Design: claim(DIMENSIONS.Design, {
    status: CLAIM_STATUS.SOURCE_STATEMENT,
    source: 'Standard King Wen received order (gate numbers ARE King Wen numbers under the Human Design / standard convention used throughout this project); donor header confirmed per direct correction Aug 12 2026',
  }),
  Space: claim(DIMENSIONS.Space, {
    status: CLAIM_STATUS.PROJECT_HYPOTHESIS,
    hypothesisId: 'H-MESHSS-SPACE-COMPLEMENT',
    evidence: { donorFlag: 'confirmed: false', note: 'donor default: complement = fuxi ^ 63, unconfirmed by user' },
  }),
});

/* Real Mawangdui upper-trigram family order (sourced, not guessed):
 * the manuscript groups all 64 hexagrams into 8 octets by upper trigram,
 * in this family order: Qian, Gen, Kan, Zhen, Kun, Dui, Li, Xun
 * (Legge-style spellings as recorded in the donor: Kian/Jen/Sun).
 * Within each octet this implementation sorts by lower trigram in the same
 * family order — the documented organizing PRINCIPLE, not a verbatim
 * byte-for-byte transcription of the manuscript's own internal per-octet
 * shift pattern (that finer detail is real but not yet sourced — flagged,
 * not faked; see MAWANGDUI_INTRA_OCTET_CLAIM). */
export const MAWANGDUI_TRIGRAM_FAMILY = Object.freeze(['Kian', 'Gen', 'Kan', 'Jen', 'Kun', 'Dui', 'Li', 'Sun']);

export const MAWANGDUI_INTRA_OCTET_CLAIM = claim('lower-trigram family-order sort within each upper-trigram octet', {
  status: CLAIM_STATUS.PROJECT_HYPOTHESIS,
  hypothesisId: 'H-MAWANGDUI-INTRA-OCTET',
  source: null,
  evidence: { note: 'organizing principle documented in donor; manuscript per-octet internal shift pattern not yet sourced' },
});

/* Our trigram ids (constants.js TRIGRAMS) -> Legge-style family names used by
 * MAWANGDUI_TRIGRAM_FAMILY. RENDERER_CONVENTION: a spelling bridge only. */
const TRIGRAM_ID_TO_FAMILY_NAME = Object.freeze({
  qian: 'Kian', gen: 'Gen', kan: 'Kan', zhen: 'Jen',
  kun: 'Kun', dui: 'Dui', li: 'Li', xun: 'Sun',
});

const TRIGRAM_BY_VALUE = Object.freeze(
  Object.fromEntries(TRIGRAMS.map((t) => [t.value, t])),
);

function trigramOf(bits3) {
  const value = bits3.reduce((acc, bit, i) => acc | (bit << i), 0);
  const t = TRIGRAM_BY_VALUE[value];
  return TRIGRAM_ID_TO_FAMILY_NAME[t.id];
}

/**
 * Derive the real 64-gate table from our canonical addressing layer:
 *   { gate, binary: [6 bits, line 1 (bottom) first, yang=1],
 *     trigrams: { upper, lower } }   // Legge-style names matching
 *                                    // MAWANGDUI_TRIGRAM_FAMILY
 * Binary comes from addressing.js gateBits (anchored: gate 1 = 111111,
 * gate 2 = 000000, gate 3 = bits [1,0,0,0,1,0]); trigram decomposition is
 * lower = lines 1-3, upper = lines 4-6 — structural math, not a new table.
 */
export function buildGateTable() {
  const table = [];
  for (let gate = 1; gate <= 64; gate++) {
    const binary = gateBits(gate);
    table.push({
      gate,
      binary,
      trigrams: {
        lower: trigramOf(binary.slice(0, 3)),
        upper: trigramOf(binary.slice(3, 6)),
      },
    });
  }
  return Object.freeze(table);
}

function fuxiValue(gate) {
  // Real binary-order value: read bottom-to-top, yin=0/yang=1, line1=LSB.
  // Consistency guard: equals KING_WEN_TO_FUXI_DECIMAL[gate.gate] by
  // construction when the gate came from buildGateTable().
  return gate.binary.reduce((acc, bit, i) => acc | (bit << i), 0);
}

function fuxiOrder(gates) {
  return [...gates].sort((a, b) => fuxiValue(a) - fuxiValue(b));
}

function kingWenOrder(gates) {
  // Gate numbers ARE King Wen numbers under the convention already used
  // throughout this project — identity sort, not a separate table.
  return [...gates].sort((a, b) => a.gate - b.gate);
}

function mawangduiOrder(gates) {
  const idx = (t) => MAWANGDUI_TRIGRAM_FAMILY.indexOf(t);
  return [...gates].sort((a, b) => {
    const byUpper = idx(a.trigrams.upper) - idx(b.trigrams.upper);
    if (byUpper !== 0) return byUpper;
    return idx(a.trigrams.lower) - idx(b.trigrams.lower);
  });
}

function reverseOf(gate) {
  // Line-order reversal, a genuine involution (reverse(reverse(x)) === x).
  return { ...gate, binary: [...gate.binary].reverse() };
}

function complementOf(gate) {
  // fuxi ^ 63, i.e. flip every line.
  return { ...gate, binary: gate.binary.map((b) => (b ^ 1)) };
}

export class DimensionLayer {
  constructor(name, spec, gateTable) {
    this.name = name;
    this.layerRole = spec.layerRole;
    this.sequenceName = spec.sequence;
    this.chart = spec.chart;
    this.claim = DIMENSION_CLAIMS[name] ?? null;
    // Real per-layer word registry, keyed by word id — this is what makes
    // cross-word dependency checkable: a word can depend on any other word
    // already registered in THIS dimension, not just ones in the same node.
    this._wordRegistry = new Map();
    this.nodes = this._order(gateTable).map((g) => ({
      gate: g.gate,
      binary: g.binary,
      trigrams: g.trigrams,
      // Real, honest, empty hook — NOT fabricated book content. Stays empty
      // until addWord() is called with real content from the source books.
      content: { words: [], rules: [] },
    }));
  }

  _order(gateTable) {
    switch (this.sequenceName) {
      case 'fuxi': return fuxiOrder(gateTable);
      case 'kingwen': return kingWenOrder(gateTable);
      case 'mawangdui': return mawangduiOrder(gateTable);
      case 'reverse': return kingWenOrder(gateTable).map(reverseOf);
      case 'complement': return kingWenOrder(gateTable).map(complementOf);
      default: throw new Error(`Unknown sequence: ${this.sequenceName}`);
    }
  }

  /* Real word-level content with real dependency edges. dependsOn must
   * already exist in this layer's registry (any gate, not just the same one)
   * — the dependency graph can never silently point at nothing. */
  addWord(gateNum, { id, text, dependsOn = [], relation = null }) {
    const node = this.nodes.find((n) => n.gate === gateNum);
    if (!node) throw new Error(`Gate ${gateNum} not found in ${this.name} layer`);
    if (this._wordRegistry.has(id)) {
      throw new Error(`Word id "${id}" already exists in ${this.name} layer`);
    }
    for (const depId of dependsOn) {
      if (!this._wordRegistry.has(depId)) {
        throw new Error(
          `"${id}" depends on "${depId}", which doesn't exist yet in ${this.name} — add dependencies before dependents`,
        );
      }
    }
    const entry = { id, text, gate: gateNum, dependsOn: [...dependsOn], relation };
    node.content.words.push(entry);
    this._wordRegistry.set(id, entry);
    return entry;
  }

  /* Reverse lookup: which words (anywhere in this dimension) depend on the
   * given one — dependency answerable in both directions. */
  dependentsOf(wordId) {
    const out = [];
    for (const entry of this._wordRegistry.values()) {
      if (entry.dependsOn.includes(wordId)) out.push(entry);
    }
    return out;
  }

  wordCount(gateNum) {
    const node = this.nodes.find((n) => n.gate === gateNum);
    return node ? node.content.words.length : 0;
  }

  letterCount(gateNum) {
    const node = this.nodes.find((n) => n.gate === gateNum);
    if (!node) return 0;
    return node.content.words.reduce(
      (sum, w) => sum + w.text.replace(/[^a-zA-Z]/g, '').length,
      0,
    );
  }

  totalWordCount() {
    return this.nodes.reduce((sum, n) => sum + n.content.words.length, 0);
  }

  totalLetterCount() {
    return this.nodes.reduce((sum, n) => sum + this.letterCount(n.gate), 0);
  }

  loadBookContent(gateNum, patch) {
    // Kept separate from addWord() on purpose — this is for coarser tags
    // (e.g. AutoLing-induced rules), not the real word/dependency graph.
    const node = this.nodes.find((n) => n.gate === gateNum);
    if (!node) throw new Error(`Gate ${gateNum} not found in ${this.name} layer`);
    if (patch.rules) node.content.rules.push(...patch.rules);
    return node;
  }
}

export class StateSpace {
  /* gateTable optional: defaults to buildGateTable() derived from our
   * canonical addressing.js — the donor required it to be passed in; here
   * the canonical table IS the one already verified in this repo. */
  constructor(gateTable = buildGateTable()) {
    if (!Array.isArray(gateTable) || gateTable.length !== 64) {
      throw new Error(
        'StateSpace requires the real 64-gate table (each {gate, binary:[6 bits], trigrams:{upper,lower}}). '
        + 'Use buildGateTable() or pass an equivalent verified table.',
      );
    }
    for (const g of gateTable) {
      if (!Array.isArray(g.binary) || g.binary.length !== 6) {
        throw new Error(`Gate ${g.gate}: binary must be a 6-element array of 0/1`);
      }
      if (!g.trigrams || !g.trigrams.upper || !g.trigrams.lower) {
        throw new Error(`Gate ${g.gate}: trigrams.upper/lower required`);
      }
      if (g.binary.every((b) => b === 0 || b === 1) === false) {
        throw new Error(`Gate ${g.gate}: binary bits must be 0/1`);
      }
      // Cross-check against the canonical map when the table carries real
      // King Wen gate numbers: a transposed bit pattern (the historical
      // gates-7/8 class of bug) fails loudly here instead of corrupting
      // every derived order.
      if (Number.isInteger(g.gate) && g.gate >= 1 && g.gate <= 64
        && KING_WEN_TO_FUXI_DECIMAL[g.gate] !== undefined
        && fuxiValue(g) !== KING_WEN_TO_FUXI_DECIMAL[g.gate]) {
        throw new Error(
          `Gate ${g.gate}: binary disagrees with canonical KING_WEN_TO_FUXI_DECIMAL `
          + `(got fuxi ${fuxiValue(g)}, want ${KING_WEN_TO_FUXI_DECIMAL[g.gate]})`,
        );
      }
    }
    this.dimensions = {};
    for (const [name, spec] of Object.entries(DIMENSIONS)) {
      this.dimensions[name] = new DimensionLayer(name, spec, gateTable);
    }
  }

  node(dimensionName, gateNum) {
    const layer = this.dimensions[dimensionName];
    if (!layer) throw new Error(`Unknown dimension: ${dimensionName}`);
    return layer.nodes.find((n) => n.gate === gateNum) || null;
  }

  /* Cross-dimension read for a single gate — every gate has a real
   * position/content slot in all 5 dimensions at once. */
  address(gateNum) {
    const out = {};
    for (const name of Object.keys(this.dimensions)) {
      out[name] = this.node(name, gateNum);
    }
    return out;
  }

  /* Real word/letter counts per dimension, side by side — the "some will be
   * longer than others" sufficiency signal, computed from real content. */
  contentSummary() {
    const out = {};
    for (const [name, layer] of Object.entries(this.dimensions)) {
      out[name] = { words: layer.totalWordCount(), letters: layer.totalLetterCount() };
    }
    return out;
  }
}

export default StateSpace;
