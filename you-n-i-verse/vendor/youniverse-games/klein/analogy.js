// klein/analogy.js — proportional analogy over Boolean feature vectors.
// Klein's Boolean-group insight: semantic markers form a Boolean group under
// XOR, so A : B :: C : ? is solved by difference masking in the group:
//     D = C ⊕ (A ⊕ B)
// (A⊕B is the "transformation mask"; applying it to C transfers the relation.)
// The raw result is resolved to the nearest registered unit by Hamming
// distance, so imperfect masks still land on real lexicon entries.

import { hamming } from './units.js';

/**
 * Solve A : B :: C : ?
 * @param {Unit} a, b, c - units (or feature vectors)
 * @param {UnitRegistry} registry - for nearest-unit resolution
 * @returns {{unit, distance, rawFeatures, mask}} - winning unit + the
 *   difference mask A⊕B (inspectable, deterministic)
 */
export function solveAnalogy(a, b, c, registry) {
  if (!registry?.nearest) throw new Error('klein.solveAnalogy: registry with nearest() required');
  const av = a.features !== undefined ? a.features : a;
  const bv = b.features !== undefined ? b.features : b;
  const cv = c.features !== undefined ? c.features : c;
  if (typeof av !== 'bigint' || typeof bv !== 'bigint' || typeof cv !== 'bigint') {
    throw new Error('klein.solveAnalogy: units or BigInt feature vectors required');
  }
  const mask = av ^ bv;
  const raw = cv ^ mask;
  const { unit, distance } = registry.nearest(raw, { exclude: [a.name, b.name, c.name].filter(Boolean) });
  return { unit, distance, rawFeatures: raw, mask };
}

/**
 * Chain of analogies (for report/demo): solve several at once.
 */
export function solveAnalogies(triples, registry) {
  return triples.map(([a, b, c]) => ({
    query: `${a.name}:${b.name} :: ${c.name}:?`,
    ...solveAnalogy(a, b, c, registry),
  }));
}

export { hamming };
