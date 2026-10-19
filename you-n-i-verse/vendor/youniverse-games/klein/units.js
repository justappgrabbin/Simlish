// klein/units.js — semantic units per Sheldon Klein's relational-calculus
// semantics: OBJECTS (atoms or classes, may contain relational structures)
// and RELATIONS (which may act as logical operators). Every unit carries a
// fixed-width 64-bit Boolean feature vector (Klein's "semantic markers" as a
// Boolean group). Pure + deterministic; units serialize to plain JSON.
//
// FEATURE REGISTRY (bit index -> meaning):
//   0  ANIMATE        living/animate entity
//   1  AGENT          capable of initiating action
//   2  SENTIENT       perceiving/knowing
//   3  MOBILE         can change location under its own power
//   4-8   ELEMENT (one-hot): wood, fire, earth, metal, water
//   9-16  TRIGRAM membership (multi: upper+lower for hexagrams): qian, dui,
//         li, zhen, xun, kan, gen, kun
//   17-18 VALENCE: positive, negative
//   19-21 TEMPORALITY (one-hot): past, present, future
//   22-29 FORM-CLASS (one-hot): object, relation, scene, action, symbol,
//         emotion, embodiment, hexagram
//   30-36 SPECIES (one-hot): human, feline, avian, aquatic, terrane,
//         synthetic, hybrid
//   37-41 ROLE (multi-hot): warrior, scholar, sage, apprentice, guardian
//   42 PROVEN         has passed the ordeal
//   43 BLOCKED        currently barred by a guardian operator
//   44-49 HEXAGRAM LINE PATTERN: six Boolean lines, bottom line first
//         (native hexagram storage — the 64 hexagrams are literally 6-bit
//         Boolean vectors in this segment)
//   50-52 LIFE-STAGE (one-hot): youth, adult, elder
//   53 ARTIFACT       made rather than born
//   54-63 IDENTITY SEGMENT (10 bits): reserved; identity anchors from
//         identity/visualDNA.js map here (first 10 bits of anchorsHash), so
//         morph continuity can be read as feature overlap on this segment.

export const FEATURE_REGISTRY = {
  ANIMATE: 0, AGENT: 1, SENTIENT: 2, MOBILE: 3,
  EL_WOOD: 4, EL_FIRE: 5, EL_EARTH: 6, EL_METAL: 7, EL_WATER: 8,
  TRI_QIAN: 9, TRI_DUI: 10, TRI_LI: 11, TRI_ZHEN: 12,
  TRI_XUN: 13, TRI_KAN: 14, TRI_GEN: 15, TRI_KUN: 16,
  VAL_POSITIVE: 17, VAL_NEGATIVE: 18,
  T_PAST: 19, T_PRESENT: 20, T_FUTURE: 21,
  FC_OBJECT: 22, FC_RELATION: 23, FC_SCENE: 24, FC_ACTION: 25,
  FC_SYMBOL: 26, FC_EMOTION: 27, FC_EMBODIMENT: 28, FC_HEXAGRAM: 29,
  SP_HUMAN: 30, SP_FELINE: 31, SP_AVIAN: 32, SP_AQUATIC: 33,
  SP_TERRANE: 34, SP_SYNTHETIC: 35, SP_HYBRID: 36,
  ROLE_WARRIOR: 37, ROLE_SCHOLAR: 38, ROLE_SAGE: 39, ROLE_APPRENTICE: 40, ROLE_GUARDIAN: 41,
  PROVEN: 42, BLOCKED: 43,
  ST_YOUTH: 50, ST_ADULT: 51, ST_ELDER: 52,
  ARTIFACT: 53,
};
export const FEATURE_WIDTH = 64;
export const HEX_LINE_BASE = 44;      // bits 44..49 = hexagram lines (bottom-up)
export const IDENTITY_BASE = 54;      // bits 54..63 = reserved identity segment
export const IDENTITY_WIDTH = 10;

// One-hot groups: within a group a coherent unit sets at most one bit.
// Used by inheritance to detect conflicts (never silently overwritten).
export const FEATURE_GROUPS = {
  element: ['EL_WOOD', 'EL_FIRE', 'EL_EARTH', 'EL_METAL', 'EL_WATER'],
  formClass: ['FC_OBJECT', 'FC_RELATION', 'FC_SCENE', 'FC_ACTION', 'FC_SYMBOL', 'FC_EMOTION', 'FC_EMBODIMENT', 'FC_HEXAGRAM'],
  species: ['SP_HUMAN', 'SP_FELINE', 'SP_AVIAN', 'SP_AQUATIC', 'SP_TERRANE', 'SP_SYNTHETIC', 'SP_HYBRID'],
  temporality: ['T_PAST', 'T_PRESENT', 'T_FUTURE'],
  lifeStage: ['ST_YOUTH', 'ST_ADULT', 'ST_ELDER'],
};

const BIT_TO_NAME = Object.fromEntries(Object.entries(FEATURE_REGISTRY).map(([k, v]) => [v, k]));

/** Build a BigInt feature vector from feature names (or pass a BigInt through). */
export function featuresFrom(features) {
  if (typeof features === 'bigint') return features;
  if (!Array.isArray(features)) throw new Error('klein.featuresFrom: array of feature names or BigInt required');
  let v = 0n;
  for (const f of features) {
    const b = FEATURE_REGISTRY[f];
    if (b === undefined) throw new Error(`klein.featuresFrom: unknown feature "${f}"`);
    v |= (1n << BigInt(b));
  }
  return v;
}

/**
 * Create a semantic unit.
 * @param {object} def - {name, kind: 'object'|'relation', features: [names]|BigInt,
 *   structure? (relational contents, plain JSON), aliases?}
 */
export function makeUnit({ name, kind = 'object', features = [], structure = null, aliases = [] }) {
  if (!name) throw new Error('klein.makeUnit: name required');
  if (kind !== 'object' && kind !== 'relation') throw new Error(`klein.makeUnit: kind must be object|relation, got "${kind}"`);
  return { name, kind, features: featuresFrom(features), structure, aliases: [...aliases] };
}

export function hasFeature(unit, name) {
  const b = FEATURE_REGISTRY[name];
  if (b === undefined) throw new Error(`klein.hasFeature: unknown feature "${name}"`);
  return ((unit.features >> BigInt(b)) & 1n) === 1n;
}

export function featureNames(unit) {
  const out = [];
  for (const [name, b] of Object.entries(FEATURE_REGISTRY)) {
    if ((unit.features >> BigInt(b)) & 1n) out.push(name);
  }
  return out;
}

export function unionFeatures(a, b) { return (a.features ?? a) | (b.features ?? b); }
export function intersectFeatures(a, b) { return (a.features ?? a) & (b.features ?? b); }

/** A subsumes B when every feature of B is present in A (Klein's marker inclusion). */
export function subsumes(a, b) {
  const av = a.features ?? a, bv = b.features ?? b;
  return (av & bv) === bv;
}

/** Hamming distance between two feature vectors. */
export function hamming(a, b) {
  let x = (a.features ?? a) ^ (b.features ?? b);
  let n = 0;
  while (x) { n += Number(x & 1n); x >>= 1n; }
  return n;
}

/** Jaccard-style Boolean feature overlap, 0..1. */
export function featureOverlap(a, b) {
  const av = a.features ?? a, bv = b.features ?? b;
  const inter = popcount(av & bv), uni = popcount(av | bv);
  return uni === 0 ? 0 : inter / uni;
}

export function popcount(v) {
  let x = typeof v === 'bigint' ? v : v.features;
  let n = 0;
  while (x) { n += Number(x & 1n); x >>= 1n; }
  return n;
}

/** Map the first IDENTITY_WIDTH bits of an anchorsHash into the reserved segment. */
export function identityBitsFromHash(anchorsHash) {
  if (!anchorsHash || typeof anchorsHash !== 'string') throw new Error('klein.identityBitsFromHash: anchorsHash string required');
  const bits = BigInt(`0x${anchorsHash.slice(0, 4)}`) & ((1n << BigInt(IDENTITY_WIDTH)) - 1n);
  return bits << BigInt(IDENTITY_BASE);
}

// Canonical trigram Boolean patterns (yang=1), bottom line first. This is the
// single source of truth; iching/ichingResolver.js imports it from here.
export const TRIGRAM_BITS = {
  qian: [1, 1, 1], dui: [1, 1, 0], li: [1, 0, 1], zhen: [1, 0, 0],
  xun: [0, 1, 1], kan: [0, 1, 0], gen: [0, 0, 1], kun: [0, 0, 0],
};
export const BITS_TO_TRIGRAM = Object.fromEntries(Object.entries(TRIGRAM_BITS).map(([k, v]) => [v.join(''), k]));
const TRI_FEATURE = { qian: 'TRI_QIAN', dui: 'TRI_DUI', li: 'TRI_LI', zhen: 'TRI_ZHEN', xun: 'TRI_XUN', kan: 'TRI_KAN', gen: 'TRI_GEN', kun: 'TRI_KUN' };

/** Extract the 6-bit hexagram line pattern (bottom-up) from a feature vector. */
export function hexLinesOf(features) {
  const v = typeof features === 'bigint' ? features : features.features;
  const lines = [];
  for (let i = 0; i < 6; i++) lines.push(Number((v >> BigInt(HEX_LINE_BASE + i)) & 1n));
  return lines;
}

/** Feature vector for a 6-line pattern in the hexagram segment. */
export function hexLineBits(lines) {
  if (!Array.isArray(lines) || lines.length !== 6) throw new Error('klein.hexLineBits: 6 lines required');
  let v = 0n;
  for (let i = 0; i < 6; i++) if (lines[i]) v |= (1n << BigInt(HEX_LINE_BASE + i));
  return v;
}

/**
 * A hexagram as a native klein unit: 6-bit Boolean vector (bits 44-49) +
 * trigram membership relations (upper/lower trigram bits + relational
 * structure naming the trigram pair).
 */
export function makeHexagramUnit(number, name, upper, lower) {
  const lines = [...TRIGRAM_BITS[lower], ...TRIGRAM_BITS[upper]];
  if (!TRIGRAM_BITS[upper] || !TRIGRAM_BITS[lower]) throw new Error(`klein.makeHexagramUnit: unknown trigram(s) ${upper}/${lower}`);
  return makeUnit({
    name: `hexagram-${number}`,
    kind: 'object',
    features: featuresFrom(['FC_HEXAGRAM', TRI_FEATURE[upper], TRI_FEATURE[lower]]) | hexLineBits(lines),
    structure: { relation: 'upper-over-lower', number, hexName: name, upper, lower },
  });
}

/** Flip one hexagram line inside a feature vector (Boolean NOT on one bit). */
export function flipHexLine(features, movingLine) {
  if (movingLine < 1 || movingLine > 6) throw new Error('klein.flipHexLine: movingLine must be 1-6');
  const v = typeof features === 'bigint' ? features : features.features;
  return v ^ (1n << BigInt(HEX_LINE_BASE + movingLine - 1));
}

/** Serialize a unit to plain JSON (BigInt -> hex string). */
export function serializeUnit(unit) {
  return {
    name: unit.name, kind: unit.kind,
    features: `0x${unit.features.toString(16).padStart(16, '0')}`,
    structure: unit.structure, aliases: unit.aliases,
  };
}

export function deserializeUnit(data) {
  if (!data?.name || typeof data.features !== 'string') throw new Error('klein.deserializeUnit: {name, features hex string} required');
  return makeUnit({ name: data.name, kind: data.kind, features: BigInt(data.features), structure: data.structure ?? null, aliases: data.aliases ?? [] });
}

/**
 * Unit registry: the lexicon. Supports nearest-unit resolution by Hamming
 * distance (used by the analogy engine and DISEMINER).
 */
export class UnitRegistry {
  constructor() { this.units = new Map(); }
  register(unit) {
    if (!unit?.name) throw new Error('klein.UnitRegistry.register: unit with name required');
    this.units.set(unit.name, unit);
    for (const alias of unit.aliases ?? []) this.units.set(alias, unit);
    return unit;
  }
  get(name) {
    const u = this.units.get(name);
    if (!u) throw new Error(`klein.UnitRegistry: unknown unit "${name}"`);
    return u;
  }
  has(name) { return this.units.has(name); }
  /** Canonical (non-alias) entries. */
  all() { return [...new Set(this.units.values())]; }
  get size() { return this.all().length; }
  /**
   * Nearest registered unit to a feature vector by Hamming distance.
   * Deterministic: ties broken by registration order.
   */
  nearest(features, { exclude = [] } = {}) {
    const v = typeof features === 'bigint' ? features : features.features;
    let best = null;
    for (const u of this.all()) {
      if (exclude.includes(u.name)) continue;
      const d = hamming(v, u.features);
      if (!best || d < best.distance) best = { unit: u, distance: d };
    }
    if (!best) throw new Error('klein.UnitRegistry.nearest: registry empty');
    return best;
  }
}
