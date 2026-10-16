import { sha256Hex } from '../browser/sha256.mjs';

/**
 * Browser-native ontological ingestion for Synthia.
 *
 * Identity is the ontological/combinatorial address. SHA-256 is used only as a
 * private integrity check inside persistence/reconstruction; it is deliberately
 * not returned as the object's identity.
 *
 * The dimensional formulas below are computational candidates, not claims that
 * replace Synthia's semantic canon. A caller can inject dimensionProjector to
 * supply a stricter/canonical projection without changing the ingestion spine.
 */

export const ACTIVE_DIMENSIONS = Object.freeze(['Movement', 'Being', 'Evolution', 'Design']);
// Package-compatible address ordering. ACTIVE_DIMENSIONS above is intentionally
// ordered by the four-field interaction model and is not an address-radix order.
export const FIVE_DIMENSIONS = Object.freeze(['Movement', 'Evolution', 'Being', 'Design', 'Space']);

export const ONTOLOGICAL_ADDRESS_STRUCTURE = Object.freeze({
  planetary: 13,
  dimension: 5,
  gate: 64,
  line: 6,
  color: 6,
  tone: 6,
  base: 5,
  degree: Object.freeze({ min: 0, max: 31 }),
  minute: Object.freeze({ min: 0, max: 59 }),
  second: Object.freeze({ min: 0, max: 59 }),
  arc: Object.freeze({ min: 0, max: 99 }),
  zodiac: 12,
  house: 'configurable',
});

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: false });
const MASK64 = (1n << 64n) - 1n;
const MIX = 0x9e3779b97f4a7c15n;

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));

function popcount8(value) {
  let v = value & 0xff;
  v -= (v >>> 1) & 0x55;
  v = (v & 0x33) + ((v >>> 2) & 0x33);
  return (v + (v >>> 4)) & 0x0f;
}

function bitTransitions(byte) {
  let count = 0;
  for (let bit = 7; bit > 0; bit--) {
    if (((byte >>> bit) & 1) !== ((byte >>> (bit - 1)) & 1)) count++;
  }
  return count;
}

function bitArray(byte) {
  return Object.freeze(Array.from({ length: 8 }, (_, bit) => (byte >>> (7 - bit)) & 1));
}

function rotateLeft64(value, shift) {
  const n = BigInt(shift % 64);
  return ((value << n) | (value >> (64n - n))) & MASK64;
}

/**
 * Deterministic state-space mixer. It is not exposed as Synthia's identity and
 * is not a cryptographic-address substitute; it only supplies reproducible
 * coordinate digits from the reduced primitive state.
 */
function stateWords(bytes, dimensionalState) {
  const words = [
    0x243f6a8885a308d3n,
    0x13198a2e03707344n,
    0xa4093822299f31d0n,
    0x082efa98ec4e6c89n,
  ];
  for (let index = 0; index < bytes.length; index++) {
    const b = BigInt(bytes[index] + 1);
    const i = index & 3;
    words[i] ^= (b + BigInt(index + 1) * MIX) & MASK64;
    words[i] = (rotateLeft64(words[i], 13 + i * 7) * (MIX | 1n)) & MASK64;
    words[(i + 1) & 3] ^= words[i] >> BigInt(11 + i);
  }
  for (let i = 0; i < FIVE_DIMENSIONS.length; i++) {
    const dimension = FIVE_DIMENSIONS[i];
    const quantized = BigInt(Math.round(clamp01(dimensionalState[dimension]?.score) * 1_000_000));
    words[i & 3] ^= (quantized + BigInt(i + 1) * 0x100000001b3n) & MASK64;
    words[i & 3] = rotateLeft64(words[i & 3], 17 + i);
  }
  return words.map((word) => word & MASK64);
}

function nextStateDigit(state, radix) {
  // xoshiro-like state evolution used only for deterministic coordinate digits.
  const result = (rotateLeft64((state[1] * 5n) & MASK64, 7) * 9n) & MASK64;
  const t = (state[1] << 17n) & MASK64;
  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];
  state[2] ^= t;
  state[3] = rotateLeft64(state[3], 45);
  return Number(result % BigInt(radix));
}

function dimensionIndex(name) {
  const index = FIVE_DIMENSIONS.indexOf(name);
  return index < 0 ? 0 : index;
}

function deriveSpace(active) {
  const values = ACTIVE_DIMENSIONS.map((name) => clamp01(active[name]?.score));
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let pairwise = 0;
  let pairs = 0;
  for (let i = 0; i < values.length; i++) {
    for (let j = i + 1; j < values.length; j++) {
      pairwise += Math.sqrt(values[i] * values[j]);
      pairs++;
    }
  }
  const relational = pairs ? pairwise / pairs : 0;
  const score = clamp01((mean + relational) / 2);
  return Object.freeze({
    score,
    derived: true,
    basis: 'resultant-of-Movement-Being-Evolution-Design',
    components: Object.freeze(Object.fromEntries(
      ACTIVE_DIMENSIONS.map((name) => [name, active[name]?.score ?? 0]),
    )),
  });
}

function defaultDimensionProjector({ byte, previousByte, previousPreviousByte }) {
  const prior = previousByte ?? byte;
  const priorPrior = previousPreviousByte ?? prior;
  const delta = byte - prior;
  const priorDelta = prior - priorPrior;
  const crossByteTransitions = popcount8(byte ^ prior);
  const withinByteTransitions = bitTransitions(byte);
  const density = popcount8(byte) / 8;

  const movement = Object.freeze({
    score: clamp01((Math.abs(delta) / 255 + crossByteTransitions / 8) / 2),
    direction: delta > 0 ? 'up' : delta < 0 ? 'down' : 'stationary',
    signed: delta / 255,
    basis: 'directional-change',
  });
  const being = Object.freeze({
    score: clamp01(1 - crossByteTransitions / 8),
    direction: byte < 128 ? 'left' : byte > 128 ? 'right' : 'center',
    signed: (byte - 127.5) / 127.5,
    basis: 'state-persistence',
  });
  const evolution = Object.freeze({
    score: clamp01((Math.abs(delta) / 255 + Math.abs(delta - priorDelta) / 510) / 2),
    direction: delta > priorDelta ? 'developing' : delta < priorDelta ? 'contracting' : 'holding',
    basis: 'change-of-change',
  });
  const design = Object.freeze({
    score: clamp01((withinByteTransitions / 7 + (1 - Math.abs(density - 0.5) * 2)) / 2),
    transitions: withinByteTransitions,
    density,
    basis: 'internal-organization',
  });

  const active = { Movement: movement, Being: being, Evolution: evolution, Design: design };
  return Object.freeze({ ...active, Space: deriveSpace(active) });
}

function aggregateDimensions(samples) {
  const out = {};
  for (const name of ACTIVE_DIMENSIONS) {
    const score = samples.length
      ? samples.reduce((sum, sample) => sum + (sample.dimensions[name]?.score ?? 0), 0) / samples.length
      : 0;
    out[name] = Object.freeze({ score: clamp01(score), derived: false });
  }
  out.Space = deriveSpace(out);
  return Object.freeze(out);
}

function dominantDimension(dimensions) {
  return FIVE_DIMENSIONS
    .map((name, order) => ({ name, score: dimensions[name]?.score ?? 0, order }))
    .sort((a, b) => b.score - a.score || a.order - b.order)[0]?.name ?? 'Space';
}

function deriveOntologicalAddress(bytes, dimensions, { houseCount = 12 } = {}) {
  if (!Number.isInteger(houseCount) || houseCount < 1) throw new RangeError('houseCount must be a positive integer');
  const state = stateWords(bytes, dimensions);
  const strongest = dominantDimension(dimensions);

  const address = Object.freeze({
    planetary: nextStateDigit(state, 13) + 1,
    dimension: strongest,
    gate: nextStateDigit(state, 64) + 1,
    line: nextStateDigit(state, 6) + 1,
    color: nextStateDigit(state, 6) + 1,
    tone: nextStateDigit(state, 6) + 1,
    base: nextStateDigit(state, 5) + 1,
    degree: nextStateDigit(state, 32),
    minute: nextStateDigit(state, 60),
    second: nextStateDigit(state, 60),
    arc: nextStateDigit(state, 100),
    zodiac: nextStateDigit(state, 12) + 1,
    house: nextStateDigit(state, houseCount) + 1,
  });
  validateOntologicalAddress(address, { houseCount });
  return address;
}

export function validateOntologicalAddress(address, { houseCount = 12 } = {}) {
  if (!address || typeof address !== 'object') throw new TypeError('ontological address must be an object');
  const integer = (name, min, max) => {
    const value = address[name];
    if (!Number.isInteger(value) || value < min || value > max) {
      throw new RangeError(`${name} out of range: expected ${min}..${max}, got ${value}`);
    }
  };
  integer('planetary', 1, 13);
  if (!FIVE_DIMENSIONS.includes(address.dimension)) throw new RangeError(`unknown dimension: ${address.dimension}`);
  integer('gate', 1, 64);
  integer('line', 1, 6);
  integer('color', 1, 6);
  integer('tone', 1, 6);
  integer('base', 1, 5);
  integer('degree', 0, 31);
  integer('minute', 0, 59);
  integer('second', 0, 59);
  integer('arc', 0, 99);
  integer('zodiac', 1, 12);
  integer('house', 1, houseCount);
  return true;
}

export function ontologicalAddressKey(address) {
  const fields = [
    'planetary', 'dimension', 'gate', 'line', 'color', 'tone', 'base',
    'degree', 'minute', 'second', 'arc', 'zodiac', 'house',
  ];
  return fields.map((field) => `${field}=${address[field]}`).join('|');
}

function extensionOf(name = '') {
  const match = String(name).toLowerCase().match(/\.([a-z0-9][a-z0-9.+_-]*)$/);
  return match?.[1] ?? '';
}

const TEXT_EXTENSIONS = new Set([
  'txt', 'md', 'markdown', 'json', 'jsonl', 'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx',
  'html', 'htm', 'css', 'scss', 'xml', 'svg', 'csv', 'tsv', 'yaml', 'yml', 'toml',
  'py', 'rb', 'rs', 'go', 'java', 'c', 'h', 'cpp', 'hpp', 'sh', 'sql', 'graphql',
]);

function looksTextual(bytes, type = '', name = '') {
  if (String(type).startsWith('text/')) return true;
  if (/json|javascript|xml|svg|yaml|toml|csv|sql/i.test(String(type))) return true;
  if (TEXT_EXTENSIONS.has(extensionOf(name))) return true;
  const limit = Math.min(bytes.length, 4096);
  if (!limit) return true;
  let acceptable = 0;
  for (let i = 0; i < limit; i++) {
    const b = bytes[i];
    if (b === 9 || b === 10 || b === 13 || (b >= 32 && b !== 127)) acceptable++;
  }
  return acceptable / limit >= 0.88;
}

function inferKind(name, type, textual) {
  const ext = extensionOf(name);
  if (['js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx'].includes(ext) || /javascript|typescript/.test(type)) return 'code';
  if (['json', 'jsonl'].includes(ext) || /json/.test(type)) return 'structured-data';
  if (['html', 'htm', 'css', 'svg', 'xml'].includes(ext)) return 'document-structure';
  if (textual) return 'text';
  if (/image\//.test(type)) return 'image';
  if (/audio\//.test(type)) return 'audio';
  if (/video\//.test(type)) return 'video';
  return 'binary';
}

async function normalizeArtifact(input, metadata = {}) {
  const meta = {
    name: String(metadata.name ?? input?.name ?? 'artifact'),
    type: String(metadata.type ?? input?.type ?? 'application/octet-stream'),
    lastModified: Number(metadata.lastModified ?? input?.lastModified ?? 0) || 0,
  };

  let bytes;
  if (input instanceof Uint8Array) {
    bytes = new Uint8Array(input);
  } else if (input instanceof ArrayBuffer) {
    bytes = new Uint8Array(input.slice(0));
  } else if (ArrayBuffer.isView(input)) {
    bytes = new Uint8Array(input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength));
  } else if (typeof input === 'string') {
    bytes = encoder.encode(input);
    if (meta.type === 'application/octet-stream') meta.type = 'text/plain';
  } else if (input && typeof input.arrayBuffer === 'function') {
    bytes = new Uint8Array(await input.arrayBuffer());
  } else if (input && input.bytes != null) {
    return normalizeArtifact(input.bytes, { ...input, ...metadata });
  } else if (input && input.content != null) {
    return normalizeArtifact(input.content, { ...input, ...metadata });
  } else if (input && typeof input === 'object') {
    bytes = encoder.encode(JSON.stringify(input));
    if (meta.type === 'application/octet-stream') meta.type = 'application/json';
  } else {
    throw new TypeError('ingest requires File, Blob, ArrayBuffer, typed bytes, string, or an artifact object');
  }

  const textual = looksTextual(bytes, meta.type, meta.name);
  const text = textual ? decoder.decode(bytes) : null;
  return Object.freeze({
    ...meta,
    bytes,
    textual,
    text,
    kind: inferKind(meta.name, meta.type, textual),
  });
}

function buildPrimitiveSamples(bytes, projector, maxMaterializedBytes) {
  const samples = [];
  const limit = Math.min(bytes.length, maxMaterializedBytes);
  for (let index = 0; index < limit; index++) {
    const byte = bytes[index];
    const dimensions = projector({
      byte,
      index,
      previousByte: index > 0 ? bytes[index - 1] : null,
      previousPreviousByte: index > 1 ? bytes[index - 2] : null,
      bytes,
    });
    samples.push(Object.freeze({
      index,
      byte,
      bits: bitArray(byte),
      primitiveScale: 'bit',
      dimensions,
    }));
  }
  return Object.freeze(samples);
}

function summarizeAllBytes(bytes, projector) {
  const sums = Object.fromEntries(FIVE_DIMENSIONS.map((name) => [name, 0]));
  let movementUp = 0;
  let movementDown = 0;
  let beingLeft = 0;
  let beingRight = 0;

  for (let index = 0; index < bytes.length; index++) {
    const dimensions = projector({
      byte: bytes[index],
      index,
      previousByte: index > 0 ? bytes[index - 1] : null,
      previousPreviousByte: index > 1 ? bytes[index - 2] : null,
      bytes,
    });
    for (const name of FIVE_DIMENSIONS) sums[name] += dimensions[name]?.score ?? 0;
    if (dimensions.Movement?.direction === 'up') movementUp++;
    if (dimensions.Movement?.direction === 'down') movementDown++;
    if (dimensions.Being?.direction === 'left') beingLeft++;
    if (dimensions.Being?.direction === 'right') beingRight++;
  }

  const active = {};
  for (const name of ACTIVE_DIMENSIONS) {
    active[name] = Object.freeze({ score: bytes.length ? clamp01(sums[name] / bytes.length) : 0, derived: false });
  }
  const dimensions = Object.freeze({ ...active, Space: deriveSpace(active) });
  return Object.freeze({
    dimensions,
    directionalBalance: Object.freeze({
      Movement: Object.freeze({ up: movementUp, down: movementDown }),
      Being: Object.freeze({ left: beingLeft, right: beingRight }),
    }),
  });
}

function publicRecord(record) {
  return Object.freeze({
    id: record.id,
    address: record.address,
    combinatorialAddress: record.combinatorialAddress,
    metadata: record.metadata,
    decomposition: record.decomposition,
    semantic: record.semantic,
    reconstruction: record.reconstruction,
  });
}

function bytesEqual(a, b) {
  if (!(a instanceof Uint8Array) || !(b instanceof Uint8Array) || a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export class MemoryOntologicalStore {
  constructor() {
    this.records = new Map();
  }

  async put(key, record, { supersede = false } = {}) {
    const prior = this.records.get(key);
    if (prior && !supersede && !bytesEqual(prior.originalBytes, record.originalBytes)) {
      const error = new Error('ONTOLOGICAL_ADDRESS_CONFLICT');
      error.code = 'ONTOLOGICAL_ADDRESS_CONFLICT';
      throw error;
    }
    this.records.set(key, record);
    return record;
  }

  async get(key) {
    return this.records.get(key) ?? null;
  }

  async has(key) {
    return this.records.has(key);
  }

  async delete(key) {
    return this.records.delete(key);
  }
}

export class IndexedDBOntologicalStore {
  constructor({ database = 'synthia-ontological-state', version = 1, objectStore = 'artifacts' } = {}) {
    this.database = database;
    this.version = version;
    this.objectStore = objectStore;
    this.dbPromise = null;
  }

  #open() {
    if (typeof indexedDB === 'undefined') {
      throw new Error('IndexedDB is unavailable in this runtime');
    }
    if (!this.dbPromise) {
      this.dbPromise = new Promise((resolve, reject) => {
        const request = indexedDB.open(this.database, this.version);
        request.onupgradeneeded = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains(this.objectStore)) db.createObjectStore(this.objectStore);
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
    }
    return this.dbPromise;
  }

  async get(key) {
    const db = await this.#open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.objectStore, 'readonly');
      const request = tx.objectStore(this.objectStore).get(key);
      request.onsuccess = () => resolve(request.result ?? null);
      request.onerror = () => reject(request.error);
    });
  }

  async has(key) {
    return Boolean(await this.get(key));
  }

  async put(key, record, { supersede = false } = {}) {
    const prior = await this.get(key);
    if (prior && !supersede && !bytesEqual(prior.originalBytes, record.originalBytes)) {
      const error = new Error('ONTOLOGICAL_ADDRESS_CONFLICT');
      error.code = 'ONTOLOGICAL_ADDRESS_CONFLICT';
      throw error;
    }
    const db = await this.#open();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(this.objectStore, 'readwrite');
      tx.objectStore(this.objectStore).put(record, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
    return record;
  }

  async delete(key) {
    const db = await this.#open();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(this.objectStore, 'readwrite');
      tx.objectStore(this.objectStore).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    return true;
  }
}

export class OntologicalFileIngestor {
  constructor({
    houseCount = 12,
    maxMaterializedBytes = 65536,
    dimensionProjector = defaultDimensionProjector,
    store = new MemoryOntologicalStore(),
    autoling = null,
    klein = null,
    publish = null,
  } = {}) {
    if (!Number.isInteger(houseCount) || houseCount < 1) throw new RangeError('houseCount must be a positive integer');
    if (!Number.isInteger(maxMaterializedBytes) || maxMaterializedBytes < 0) {
      throw new RangeError('maxMaterializedBytes must be a non-negative integer');
    }
    if (typeof dimensionProjector !== 'function') throw new TypeError('dimensionProjector must be a function');
    if (!store?.put || !store?.get) throw new TypeError('store must implement async put/get');

    this.houseCount = houseCount;
    this.maxMaterializedBytes = maxMaterializedBytes;
    this.dimensionProjector = dimensionProjector;
    this.store = store;
    this.autoling = autoling;
    this.klein = klein;
    this.publish = typeof publish === 'function' ? publish : null;
    this.reconstructors = new Map();
    this.sequence = 0;
  }

  registerReconstructor({ id, test = () => true, reconstruct, score = () => 1 } = {}) {
    if (!id || typeof reconstruct !== 'function') throw new TypeError('reconstructor requires id and reconstruct()');
    this.reconstructors.set(id, Object.freeze({ id, test, reconstruct, score }));
    return this;
  }

  async #semanticLayer(artifact, address, key) {
    if (!artifact.textual || artifact.text == null) {
      return Object.freeze({ available: false, reason: 'non-textual-artifact' });
    }

    let autoling = null;
    let klein = null;
    if (this.autoling?.runPipeline) {
      autoling = this.autoling.runPipeline(artifact.text);
    }
    if (this.klein?.observe) {
      const observed = this.klein.observe(key, artifact.text);
      klein = Object.freeze({ observed, vocabularySize: Number(this.klein.size ?? 0) });
    }

    return Object.freeze({
      available: true,
      inputClass: artifact.kind,
      address,
      autoling,
      klein,
    });
  }

  async ingest(input, options = {}) {
    const artifact = await normalizeArtifact(input, options.metadata ?? {});
    const summary = summarizeAllBytes(artifact.bytes, this.dimensionProjector);
    const address = deriveOntologicalAddress(artifact.bytes, summary.dimensions, { houseCount: this.houseCount });
    const combinatorialAddress = ontologicalAddressKey(address);
    const samples = buildPrimitiveSamples(artifact.bytes, this.dimensionProjector, this.maxMaterializedBytes);
    const semantic = await this.#semanticLayer(artifact, address, combinatorialAddress);
    const integrity = sha256Hex(artifact.bytes);

    const stored = {
      id: `ontological-artifact-${String(++this.sequence).padStart(6, '0')}`,
      address,
      combinatorialAddress,
      metadata: Object.freeze({
        name: artifact.name,
        type: artifact.type,
        kind: artifact.kind,
        textual: artifact.textual,
        byteLength: artifact.bytes.byteLength,
      }),
      decomposition: Object.freeze({
        lowestIdentifiablePrimitive: 'bit',
        primitiveCount: artifact.bytes.byteLength * 8,
        materializedByteCount: samples.length,
        materializedPrimitiveCount: samples.length * 8,
        completeMathematicalReduction: true,
        boundedObjectMaterialization: samples.length !== artifact.bytes.byteLength,
        dimensions: summary.dimensions,
        directionalBalance: summary.directionalBalance,
        samples,
        spaceRule: 'Space is derived from Movement, Being, Evolution, and Design; it is not an independent input bucket.',
        formulaStatus: 'computational-candidate-replaceable-by-dimensionProjector',
      }),
      semantic,
      reconstruction: Object.freeze({
        exactAvailable: true,
        adaptiveAvailable: true,
        exactMeaning: 'byte-for-byte original',
        adaptiveMeaning: 'best registered reconstruction for supplied micro/macro/future-use state; exact bytes are the fallback',
      }),
      originalBytes: new Uint8Array(artifact.bytes),
      internalIntegrity: integrity,
    };

    await this.store.put(combinatorialAddress, stored, { supersede: options.supersede === true });

    if (this.publish) {
      await this.publish(Object.freeze({
        type: 'ontological-ingestion',
        address,
        combinatorialAddress,
        metadata: stored.metadata,
        dimensions: summary.dimensions,
        semantic,
      }));
    }

    return publicRecord(stored);
  }

  async recall(addressOrKey) {
    const key = typeof addressOrKey === 'string' ? addressOrKey : ontologicalAddressKey(addressOrKey);
    const stored = await this.store.get(key);
    return stored ? publicRecord(stored) : null;
  }

  async reconstruct(addressOrRecord, {
    mode = 'adaptive',
    state = {},
    as = 'bytes',
  } = {}) {
    const key = typeof addressOrRecord === 'string'
      ? addressOrRecord
      : addressOrRecord?.combinatorialAddress ?? ontologicalAddressKey(addressOrRecord?.address ?? addressOrRecord);
    const stored = await this.store.get(key);
    if (!stored) {
      const error = new Error('ONTOLOGICAL_ADDRESS_NOT_FOUND');
      error.code = 'ONTOLOGICAL_ADDRESS_NOT_FOUND';
      throw error;
    }

    if (sha256Hex(stored.originalBytes) !== stored.internalIntegrity) {
      const error = new Error('INTEGRITY_MISMATCH');
      error.code = 'INTEGRITY_MISMATCH';
      throw error;
    }

    let selected = {
      id: 'exact-original',
      exact: true,
      bytes: new Uint8Array(stored.originalBytes),
      score: mode === 'exact' ? Number.POSITIVE_INFINITY : 1,
      reason: 'lossless-fallback',
    };

    if (mode !== 'exact') {
      for (const adapter of this.reconstructors.values()) {
        if (!(await adapter.test(publicRecord(stored), state))) continue;
        const candidate = await adapter.reconstruct(publicRecord(stored), {
          state,
          originalBytes: new Uint8Array(stored.originalBytes),
        });
        if (candidate == null) continue;
        const normalized = await normalizeArtifact(candidate?.value ?? candidate, {
          name: candidate?.name ?? stored.metadata.name,
          type: candidate?.type ?? stored.metadata.type,
        });
        const score = Number(candidate?.score ?? await adapter.score(publicRecord(stored), state, normalized));
        if (Number.isFinite(score) && score > selected.score) {
          selected = {
            id: adapter.id,
            exact: bytesEqual(normalized.bytes, stored.originalBytes),
            bytes: normalized.bytes,
            score,
            reason: candidate?.reason ?? 'registered-state-fit',
          };
        }
      }
    }

    const bytes = new Uint8Array(selected.bytes);
    if (as === 'bytes') {
      return Object.freeze({
        mode,
        selected: selected.id,
        exact: selected.exact,
        score: selected.score,
        reason: selected.reason,
        address: stored.address,
        combinatorialAddress: stored.combinatorialAddress,
        bytes,
      });
    }
    if (as === 'text') {
      return Object.freeze({
        mode,
        selected: selected.id,
        exact: selected.exact,
        address: stored.address,
        combinatorialAddress: stored.combinatorialAddress,
        text: decoder.decode(bytes),
      });
    }
    if (as === 'blob') {
      if (typeof Blob === 'undefined') throw new Error('Blob is unavailable in this runtime');
      return new Blob([bytes], { type: stored.metadata.type });
    }
    throw new RangeError(`unknown reconstruction output: ${as}`);
  }

  async learnLanguage(samples = []) {
    const results = [];
    for (const sample of samples) {
      const record = await this.ingest(sample.value ?? sample.text ?? sample, {
        metadata: {
          name: sample.name ?? 'language-sample.txt',
          type: sample.type ?? 'text/plain',
        },
      });
      results.push(record);
    }
    return Object.freeze(results);
  }
}

export const DEFAULT_DIMENSION_PROJECTOR = defaultDimensionProjector;
export default OntologicalFileIngestor;
