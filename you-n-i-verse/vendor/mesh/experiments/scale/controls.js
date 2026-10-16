// Pure Synthia Automata — experiments/scale: control conditions + ablation helpers (seeded, mapping-generic)

/**
 * PORT of pure-synthia-phase2-REPAIRED/src/experiments/controls.js
 * (ControlSystem + AblationStudy — identical in pass3-REPAIRED; diff
 * confirmed), generalized per the primitive->scale experiment seed:
 * every control generator accepts ANY mapping + a seeded rng.
 *
 * FROZEN ASSUMPTIONS:
 *   - (source, Appendix E / H.7) The control battery is: flat lookup
 *     (storage-heavy baseline), frequency baseline (no recursion), random
 *     grammar (same size, randomized relationships), shuffled structure
 *     (primitives preserved, positional/relational info permuted), and
 *     ablations S^{-p_i} / S^{-o_i} / S^{-A} / S^{-D} / S^{-G_k}.
 *   - (source defect, repaired by this port) The source's randomGrammar and
 *     shuffledStructure used Math.random() — non-deterministic, violating
 *     the determinism contract. Here every randomized control takes an
 *     explicit rng (default: mulberry32 with the frozen DEFAULT_CONTROL_SEED),
 *     so every condition is reproducible. This is the same defect class as
 *     phase-1 repair-log #2 (non-determinism in experiment records).
 *   - (this port) The engine-coupled AblationStudy (engine.clone()) is NOT
 *     ported — our engine has no clone(); ablations are expressed as pure
 *     registry/mapping transforms (withoutPrimitive, withoutOperator,
 *     ablateMapping) which is how the sealed D1/D2/D3 benchmarks implement
 *     them (registry minus one entry, operator set to null).
 *
 * PROVENANCE:
 *   - flatLookup / frequencyBaseline: SOURCE_STATEMENT (controls.js, verbatim;
 *     frequencyBaseline's sort is JS-stable so ties keep input order —
 *     deterministic given the same input).
 *   - randomGrammar / shuffledStructure / shuffleWith / controlMappings:
 *     behavior SOURCE_STATEMENT, rng parameterization IMPLEMENTATION_CHOICE.
 *   - modulo mapping control: IMPLEMENTATION_CHOICE (this port) — a
 *     systematic permutation (value shifted by one position) as the
 *     deterministic middle point between source and random.
 *   - withoutPrimitive / withoutOperator / ablateMapping: SOURCE_STATEMENT
 *     semantics (AblationStudy.withoutPrimitive/withoutOperator), expressed
 *     as pure functions over Map/object registries.
 */

import { mulberry32 } from '../../state-space/constants.js';

// Frozen default seed for all control generation (deterministic everywhere).
export const DEFAULT_CONTROL_SEED = 0xC047E01;

/* ------------------------------------------- source ControlSystem (pure) */

// Flat Lookup: storage-heavy baseline. storedResults: Map keyed by
// JSON.stringify(input).
export function flatLookup(input, storedResults) {
  const key = JSON.stringify(input);
  return storedResults.get(key) || null;
}

// Frequency Baseline: predicts using empirical frequencies, no recursion.
// components: iterable of ids; frequencies: Map id -> count.
export function frequencyBaseline(components, frequencies) {
  return components
    .map((c) => ({ component: c, score: frequencies.get(c) || 0 }))
    .sort((a, b) => b.score - a.score);
}

/* ------------------------------------------- seeded shuffling (general) */

/** Deterministic Fisher–Yates: returns a shuffled COPY of `values`. */
export function shuffleWith(values, rng = mulberry32(DEFAULT_CONTROL_SEED)) {
  const arr = [...values];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Random Grammar: same grammar size, randomized rule relationships.
// (Source used Math.random; here seeded — see header.)
export function randomGrammar(grammar, rng = mulberry32(DEFAULT_CONTROL_SEED)) {
  const rules = Object.entries({ ...grammar });
  for (let i = rules.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [rules[i], rules[j]] = [rules[j], rules[i]];
  }
  return Object.fromEntries(rules);
}

// Shuffled Structure: preserves primitives, permutes positional/relational
// info (source: members permuted). Seeded rng; the composite is deep-copied.
export function shuffledStructure(composite, rng = mulberry32(DEFAULT_CONTROL_SEED)) {
  const shuffled = JSON.parse(JSON.stringify(composite));
  if (shuffled.members) {
    shuffled.members = shuffleWith(shuffled.members, rng);
  }
  return shuffled;
}

/* --------------------------- generic mapping controls (primitive->scale) */

/**
 * The seed of the primitive->scale experiment: given ANY mapping
 * (plain object or Map: key -> value — e.g. letter->phoneme, word->address),
 * produce the control conditions as new mappings over the same keys:
 *
 *   source   — the mapping itself, untouched (the candidate);
 *   modulo   — deterministic systematic permutation: key_i -> value_{(i+1) mod n};
 *   shuffled — values permuted across keys by seeded Fisher–Yates;
 *   random   — each key assigned a uniformly seeded-random value from the
 *              codomain (with replacement);
 *   ablated  — source minus one key (S^{-p_i} on a mapping); pass `ablateKey`,
 *              defaults to the last sorted key.
 *
 * Keys are processed in sorted order so the result is independent of object
 * insertion order. Returns plain frozen objects plus lookup functions.
 */
export function controlMappings(mapping, { seed = DEFAULT_CONTROL_SEED, ablateKey = null } = {}) {
  const entries = mapping instanceof Map ? [...mapping.entries()] : Object.entries(mapping);
  const keys = entries.map(([k]) => k).sort();
  const base = new Map(entries);
  const values = keys.map((k) => base.get(k));
  const n = keys.length;
  if (n === 0) throw new RangeError('controlMappings requires a non-empty mapping');

  const rng = mulberry32(seed);

  const moduloValues = keys.map((_, i) => values[(i + 1) % n]);
  const shuffledValues = shuffleWith(values, rng);
  const randomValues = keys.map(() => values[Math.floor(rng() * n)]);

  const toObject = (vals, dropKey = null) => Object.freeze(Object.fromEntries(
    keys
      .map((k, i) => [k, vals[i]])
      .filter(([k]) => k !== dropKey),
  ));

  const abKey = ablateKey ?? keys[n - 1];

  const source = toObject(values);
  const modulo = toObject(moduloValues);
  const shuffled = toObject(shuffledValues);
  const random = toObject(randomValues);
  const ablated = toObject(values, abKey);

  return Object.freeze({
    seed,
    keys: Object.freeze(keys),
    ablatedKey: abKey,
    conditions: Object.freeze({ source, modulo, shuffled, random, ablated }),
    // Lookup functions (key -> value; ablated returns undefined off-support).
    lookup: Object.freeze({
      source: (k) => source[k],
      modulo: (k) => modulo[k],
      shuffled: (k) => shuffled[k],
      random: (k) => random[k],
      ablated: (k) => ablated[k],
    }),
  });
}

/* ------------------------------------------- ablation helpers (pure) */

/** S^{-p_i}: registry (Map) minus one primitive. Returns a NEW Map. */
export function withoutPrimitive(primitiveRegistry, primitiveId) {
  const modified = new Map(primitiveRegistry);
  modified.delete(primitiveId);
  return modified;
}

/** S^{-o_i}: operator registry (Map) minus one operator. Returns a NEW Map. */
export function withoutOperator(operatorRegistry, operatorId) {
  const modified = new Map(operatorRegistry);
  modified.delete(operatorId);
  return modified;
}

/** S^{-p_i} on a plain-object mapping: copy minus one key. */
export function ablateMapping(mapping, key) {
  const copy = { ...mapping };
  delete copy[key];
  return copy;
}

/**
 * Run one ablation: metricFn over baseline vs modified configuration.
 * (Ported from AblationStudy.runAblation; pure — caller supplies both
 * configurations and the test+metric functions.)
 */
export function runAblation(componentName, { baselineConfig, modifiedConfig, testFn, metricFn }) {
  const baseline = metricFn(testFn(baselineConfig));
  const ablated = metricFn(testFn(modifiedConfig));
  const impact = baseline - ablated;
  return Object.freeze({ componentName, baseline, ablated, impact });
}

export const CONTROL_PROVENANCE = Object.freeze({
  flatLookup: Object.freeze({ status: 'SOURCE_STATEMENT', source: 'pure-synthia-phase2-REPAIRED/src/experiments/controls.js' }),
  frequencyBaseline: Object.freeze({ status: 'SOURCE_STATEMENT', source: 'pure-synthia-phase2-REPAIRED/src/experiments/controls.js' }),
  randomGrammar: Object.freeze({ status: 'SOURCE_STATEMENT (behavior) / IMPLEMENTATION_CHOICE (seeded rng replaces Math.random)', source: 'pure-synthia-phase2-REPAIRED/src/experiments/controls.js' }),
  shuffledStructure: Object.freeze({ status: 'SOURCE_STATEMENT (behavior) / IMPLEMENTATION_CHOICE (seeded rng replaces Math.random)', source: 'pure-synthia-phase2-REPAIRED/src/experiments/controls.js' }),
  controlMappings: Object.freeze({ status: 'IMPLEMENTATION_CHOICE', source: 'this port — generalization of the source controls to any mapping + seeded rng (primitive->scale experiment seed)' }),
  ablations: Object.freeze({ status: 'SOURCE_STATEMENT (semantics) / IMPLEMENTATION_CHOICE (pure registry transforms instead of engine.clone())', source: 'pure-synthia-phase2-REPAIRED/src/experiments/controls.js AblationStudy' }),
});
