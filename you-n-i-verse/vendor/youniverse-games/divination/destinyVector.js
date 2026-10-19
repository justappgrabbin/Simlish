// Destiny vector — 512-dim deterministic vector composed of five segments:
//   pillars_embedding   [0..127]
//   elemental_profile   [128..191]
//   trigram_signature   [192..255]
//   temporal_resonance  [256..383]
//   accumulated_history [384..511]
// Hash-derived deterministic initialization; the history segment actually
// updates with each interaction (blend + renormalize).

import { deriveFloats } from '../util/hash.js';
import { FIVE_ELEMENTS } from './bazi.js';
import { ELEMENT_SIGNATURES } from '../resonance/embeddingSpace.js';

export const DESTINY_DIM = 512;
export const SEGMENTS = {
  pillars: { start: 0, len: 128 },
  elemental: { start: 128, len: 64 },
  trigram: { start: 192, len: 64 },
  temporal: { start: 256, len: 128 },
  history: { start: 384, len: 128 },
};

export function l2norm(v) {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

export function l2normalize(v) {
  const n = l2norm(v);
  if (n === 0) throw new Error('destinyVector: cannot normalize zero vector');
  return v.map((x) => x / n);
}

export function cosine(a, b) {
  if (a.length !== b.length) throw new Error('destinyVector.cosine: dimension mismatch');
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function fillSegment(vec, seg, values) {
  for (let i = 0; i < seg.len; i++) vec[seg.start + i] = values[i % values.length];
}

/**
 * Build the initial destiny vector.
 * @param {object} input - {playerId, pillars, elements, hexagrams:{primary,changingLines,resulting}}
 */
export function buildDestinyVector(input) {
  if (!input?.pillars || !input?.elements || !input?.hexagrams) {
    throw new Error('destinyVector.buildDestinyVector: pillars, elements, hexagrams required');
  }
  const { playerId = 'anon', pillars, elements, hexagrams } = input;
  const v = new Array(DESTINY_DIM).fill(0);
  const zc = (key, n) => deriveFloats(key, n).map((x) => (x - 0.5) * 2);
  const tile = (sig, len) => Array.from({ length: len }, (_, i) => sig[i % sig.length]);
  const mixSigs = (weighted) => {
    const mix = new Array(16).fill(0);
    for (const [el, w] of weighted) for (let i = 0; i < 16; i++) mix[i] += w * ELEMENT_SIGNATURES[el][i];
    return l2normalize(mix);
  };

  // pillars_embedding: zero-centered hash floats per pillar text, interleaved.
  const pillarFloats = ['year', 'month', 'day', 'hour']
    .flatMap((k) => zc(`pillar:${k}:${pillars[k].text}`, 32));
  fillSegment(v, SEGMENTS.pillars, pillarFloats);

  // elemental_profile: the chart's element mix baked in as tiled element
  // signatures weighted by ZERO-CENTERED count (above/below mean), so the
  // player's strong elements attract and starved elements repel in cosine.
  const mean = FIVE_ELEMENTS.reduce((s, e) => s + (elements.counts[e] ?? 0), 0) / FIVE_ELEMENTS.length;
  const elMix = mixSigs(FIVE_ELEMENTS.map((e) => [e, (elements.counts[e] ?? 0) - mean]));
  fillSegment(v, SEGMENTS.elemental,
    tile(elMix, SEGMENTS.elemental.len).map((x, i) => 2.0 * x + 0.1 * zc(`elemental:${playerId}`, SEGMENTS.elemental.len)[i]));

  // trigram_signature: deterministic hash of primary hexagram + changing lines.
  fillSegment(v, SEGMENTS.trigram,
    zc(`trigram:${hexagrams.primary.number}:${hexagrams.primary.upper}>${hexagrams.primary.lower}:${hexagrams.changingLines.join(',')}`, SEGMENTS.trigram.len)
      .map((x) => 0.6 * x));

  // temporal_resonance: deterministic from the birth day/hour pillar pair.
  fillSegment(v, SEGMENTS.temporal, zc(`temporal:${pillars.day.text}:${pillars.hour.text}`, SEGMENTS.temporal.len));

  // accumulated_history: zero-ish start (small deterministic noise floor).
  fillSegment(v, SEGMENTS.history, zc(`history-init:${playerId}`, SEGMENTS.history.len).map((x) => x * 0.05));

  return l2normalize(v);
}

/**
 * Update the history segment with a new interaction embedding.
 * blend: history <- normalize( (1-rate)*history + rate*interaction )
 * The full vector is renormalized after the update.
 * @returns {number[]} new 512-dim destiny vector
 */
export function updateHistory(destinyVec, interactionEmbedding, rate = 0.3) {
  if (destinyVec.length !== DESTINY_DIM) throw new Error('destinyVector.updateHistory: bad destiny dim');
  if (interactionEmbedding.length !== DESTINY_DIM) throw new Error('destinyVector.updateHistory: bad interaction dim');
  const seg = SEGMENTS.history;
  const out = destinyVec.slice();
  for (let i = 0; i < seg.len; i++) {
    const h = destinyVec[seg.start + i];
    const inter = interactionEmbedding[seg.start + i];
    out[seg.start + i] = (1 - rate) * h + rate * inter;
  }
  return l2normalize(out);
}

/** Cosine similarity restricted to a segment. */
export function segmentCosine(a, b, seg) {
  return cosine(a.slice(seg.start, seg.start + seg.len), b.slice(seg.start, seg.start + seg.len));
}
