// Pure Synthia Automata — derivation Δ: deterministic, hashable execution record (spec §12, App. D)

/**
 * Derivation records and hashing.
 *
 * Every execution returns Δ = (input, decomposition(parse), primitives,
 * operators, relations, context, transforms, output, evaluation) plus the
 * merged complexity ledger (spec §12). Deterministic replay requires:
 * same (state, input, context, grammarVersion, operatorVersion) → identical
 * derivation hash.
 *
 * Hashing: `stableStringify` produces a canonical JSON serialization (object
 * keys sorted recursively), hashed with 32-bit FNV-1a over UTF-8 bytes.
 * The `hash` getter is lazy and cached. A `timestamp` field, when present,
 * is excluded from the hash payload (engine rule: no wall-clock inside the
 * hash) but included in toJSON().
 */

/**
 * Canonical serialization: object keys sorted recursively, arrays keep order,
 * non-finite numbers / undefined / functions / symbols serialize as null in
 * arrays and are omitted from objects (JSON semantics). Throws on cycles.
 */
export function stableStringify(value) {
  const seen = new Set();

  const encode = (v) => {
    if (v === null) return 'null';
    const t = typeof v;
    if (t === 'string') return JSON.stringify(v);
    if (t === 'boolean') return v ? 'true' : 'false';
    if (t === 'number') return Number.isFinite(v) ? JSON.stringify(v) : 'null';
    if (t === 'undefined' || t === 'function' || t === 'symbol' || t === 'bigint') {
      return undefined;
    }
    if (Array.isArray(v)) {
      if (seen.has(v)) throw new TypeError('stableStringify: circular reference');
      seen.add(v);
      const out = '[' + v.map((item) => {
        const s = encode(item);
        return s === undefined ? 'null' : s;
      }).join(',') + ']';
      seen.delete(v);
      return out;
    }
    if (t === 'object') {
      if (seen.has(v)) throw new TypeError('stableStringify: circular reference');
      seen.add(v);
      const keys = Object.keys(v).sort();
      const parts = [];
      for (const key of keys) {
        const s = encode(v[key]);
        if (s !== undefined) parts.push(JSON.stringify(key) + ':' + s);
      }
      seen.delete(v);
      return '{' + parts.join(',') + '}';
    }
    return undefined;
  };

  const result = encode(value);
  return result === undefined ? 'null' : result;
}

/** 32-bit FNV-1a hash of a string (UTF-8 encoded) → 8-char lowercase hex. */
export function fnv1a32(str) {
  const bytes = new TextEncoder().encode(String(str));
  let hash = 0x811c9dc5;
  for (let k = 0; k < bytes.length; k++) {
    hash ^= bytes[k];
    // 32-bit multiply via Math.imul (plain `*` loses precision above 2^53).
    hash = Math.imul(hash, 0x01000193) >>> 0; // FNV prime 16777619
  }
  return hash.toString(16).padStart(8, '0');
}

/** Deterministic hash of any JSON-like value. */
export function hashObject(x) {
  return fnv1a32(stableStringify(x));
}

/** Fields excluded from the derivation hash payload. */
const HASH_EXCLUDED = new Set(['timestamp']);

export class Derivation {
  constructor({
    id = null,
    input = null,
    parse = null,
    primitives = [],
    operators = [],
    relations = [],
    context = {},
    transforms = [],
    output = null,
    evaluation = {},
    ledger = null,
    engineVersion = '1.0.0',
    grammarVersion = 'automata.1.0',
    ...extra
  } = {}) {
    this.id = id;
    this.input = input;
    this.parse = parse;
    this.primitives = primitives;
    this.operators = operators;
    this.relations = relations;
    this.context = context;
    this.transforms = transforms;
    this.output = output;
    this.evaluation = evaluation;
    this.ledger = ledger;
    this.engineVersion = engineVersion;
    this.grammarVersion = grammarVersion;
    // Extra fields (e.g. chainResults, timestamp) are preserved and, except
    // for `timestamp`, participate in the hash.
    Object.assign(this, extra);
  }

  /** Lazy deterministic hash over all enumerable fields except `timestamp`. */
  get hash() {
    if (this._hashCache === undefined) {
      const payload = {};
      for (const key of Object.keys(this)) {
        if (!HASH_EXCLUDED.has(key)) payload[key] = this[key];
      }
      Object.defineProperty(this, '_hashCache', {
        value: hashObject(payload),
        enumerable: false,
        writable: true,
        configurable: true,
      });
    }
    return this._hashCache;
  }

  toJSON() {
    return { ...this, hash: this.hash };
  }
}
