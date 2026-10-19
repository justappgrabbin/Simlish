// Hexagram mapper — dominant element -> upper trigram, weak element -> lower
// trigram -> primary hexagram (本卦); changing lines from birth signature +
// current time -> resulting hexagram (之卦). Reuses the 8-trigram table from
// iching/ichingResolver.js (TRIGRAM_BITS logic mirrored here structurally).

import { HEXAGRAMS, TRIGRAMS, getHexagram } from '../iching/ichingResolver.js';
import { shortHash } from '../util/hash.js';

// Five elements -> trigram mapping (classical associations). Some elements
// have a yin/yang pair of trigrams; the birth signature picks between them.
export const ELEMENT_TRIGRAMS = {
  wood:  ['zhen', 'xun'],  // thunder / wind
  fire:  ['li', 'li'],     // fire
  earth: ['kun', 'gen'],   // earth / mountain
  metal: ['qian', 'dui'],  // heaven / lake
  water: ['kan', 'kan'],   // water
};

const TRIGRAM_BITS = { qian: [1,1,1], dui: [1,1,0], li: [1,0,1], zhen: [1,0,0], xun: [0,1,1], kan: [0,1,0], gen: [0,0,1], kun: [0,0,0] };
const BITS_TO_TRIGRAM = Object.fromEntries(Object.entries(TRIGRAM_BITS).map(([k, v]) => [v.join(''), k]));

export function elementToTrigram(element, birthSignature = '') {
  const pair = ELEMENT_TRIGRAMS[element];
  if (!pair) throw new Error(`hexagramMapper: unknown element "${element}"`);
  if (pair[0] === pair[1]) return pair[0];
  const pick = parseInt(shortHash('trigram:' + element + ':' + birthSignature, 2), 16) % 2;
  return pair[pick];
}

export function hexagramByTrigrams(upper, lower) {
  const found = HEXAGRAMS.find((h) => h && h.upper === upper && h.lower === lower);
  if (!found) throw new Error(`hexagramMapper: no hexagram for ${upper}>${lower}`);
  return found;
}

/** Structural multi-line flip -> resulting hexagram number. */
export function flipLines(hexagramNumber, lineNumbers) {
  const hex = getHexagram(hexagramNumber);
  const lines = [...TRIGRAM_BITS[hex.lower], ...TRIGRAM_BITS[hex.upper]];
  for (const ln of lineNumbers) {
    if (ln < 1 || ln > 6) throw new Error(`hexagramMapper: line ${ln} out of range`);
    lines[ln - 1] = lines[ln - 1] ? 0 : 1;
  }
  const lower = BITS_TO_TRIGRAM[lines.slice(0, 3).join('')];
  const upper = BITS_TO_TRIGRAM[lines.slice(3, 6).join('')];
  return hexagramByTrigrams(upper, lower);
}

/**
 * Changing lines derived from birth signature + current time.
 * Deterministic: same (signature, timestamp bucket) -> same lines.
 * @returns {number[]} 1-3 distinct line numbers (1-6)
 */
export function changingLines(birthSignature, now = new Date()) {
  const bucket = Math.floor(now.getTime() / 60000); // minute-bucketed
  const h = shortHash(`changing:${birthSignature}:${bucket}`, 8);
  const n = (parseInt(h.slice(0, 2), 16) % 3) + 1; // 1-3 changing lines
  const lines = new Set();
  let i = 2;
  while (lines.size < n) {
    lines.add((parseInt(h.slice(i, i + 2), 16) % 6) + 1);
    i = (i + 2) % 8 === 2 ? 2 : i + 2;
    if (i > 7) i = 2;
    if (lines.size < n && i === 2) { // fallback: sequential fill
      for (let l = 1; l <= 6 && lines.size < n; l++) lines.add(l);
    }
  }
  return [...lines].sort((a, b) => a - b);
}

/**
 * Full mapping: BaZi element analysis -> primary + resulting hexagrams.
 * @param {object} elementAnalysis - output of analyzeElements()
 * @param {string} birthSignature - stable string (e.g. birthDataId + pillars)
 * @param {Date} [now]
 */
export function mapToHexagrams(elementAnalysis, birthSignature, now = new Date()) {
  if (!elementAnalysis?.dominant || !elementAnalysis?.weakest) {
    throw new Error('hexagramMapper: elementAnalysis with dominant/weakest required');
  }
  const upperTrigram = elementToTrigram(elementAnalysis.dominant, birthSignature);
  const lowerTrigram = elementToTrigram(elementAnalysis.weakest, birthSignature + ':weak');
  const primary = hexagramByTrigrams(upperTrigram, lowerTrigram);
  const lines = changingLines(birthSignature, now);
  const resulting = flipLines(primary.number, lines);
  return {
    primary: { number: primary.number, name: primary.name, upper: upperTrigram, lower: lowerTrigram,
      upperSymbol: TRIGRAMS[upperTrigram].symbol, lowerSymbol: TRIGRAMS[lowerTrigram].symbol },
    changingLines: lines,
    resulting: { number: resulting.number, name: resulting.name, upper: resulting.upper, lower: resulting.lower },
  };
}

export { TRIGRAM_BITS, BITS_TO_TRIGRAM };
