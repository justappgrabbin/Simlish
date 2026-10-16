import { validateCanonicalAddress, canonicalAddressKey, DIMENSIONS, AXES } from './canonical-address.mjs';
import { stableStringify, fnv1a32 } from './pure-synthia/engine/derivation.js';

const encoder = new TextEncoder();

function artifactBytes(artifact = {}) {
  if (artifact?.bytes instanceof Uint8Array) return artifact.bytes;
  if (artifact?.content instanceof Uint8Array) return artifact.content;
  if (typeof artifact?.content === 'string') return encoder.encode(artifact.content);
  if (typeof artifact?.originalContent === 'string') return encoder.encode(artifact.originalContent);
  return encoder.encode(stableStringify(artifact ?? null));
}

function popcount8(v) {
  v &= 0xff;
  v = v - ((v >>> 1) & 0x55);
  v = (v & 0x33) + ((v >>> 2) & 0x33);
  return (((v + (v >>> 4)) & 0x0f) * 0x01);
}

function safeResult(value) {
  try { return JSON.parse(stableStringify(value)); }
  catch { return String(value); }
}

function hash32(value) { return BigInt(`0x${fnv1a32(stableStringify(value))}`); }
function hash128(value) {
  // Four independently salted deterministic 32-bit projections. This is an
  // addressing projection, not a cryptographic digest.
  return (hash32(['A', value]) << 96n)
    | (hash32(['B', value]) << 64n)
    | (hash32(['C', value]) << 32n)
    | hash32(['D', value]);
}

function take(n, radix) {
  const r = BigInt(radix);
  return { digit: Number(n % r), rest: n / r };
}

/**
 * Full execution-address resolver.
 *
 * Important boundary: this resolver establishes a deterministic computational
 * coordinate from actual artifact + execution evidence. It does NOT claim the
 * resulting coordinate is an astronomical, Human Design, or metaphysical fact.
 */
export class ExecutionAddressResolver {
  constructor({ degreeBands = 5, degreeSubdivisions = 29 } = {}) {
    this.degreeBands = degreeBands;
    this.degreeSubdivisions = degreeSubdivisions;
  }

  /** Descend the artifact to bit/state primitives BEFORE execution. */
  descend(artifact = {}) {
    const bytes = artifactBytes(artifact);
    let ones = 0;
    let xor = 0;
    let sum = 0;
    const sixBitStates = [];
    let buffer = 0;
    let buffered = 0;
    for (const byte of bytes) {
      ones += popcount8(byte);
      xor ^= byte;
      sum = (sum + byte) >>> 0;
      buffer = (buffer << 8) | byte;
      buffered += 8;
      while (buffered >= 6) {
        const shift = buffered - 6;
        sixBitStates.push((buffer >>> shift) & 0x3f);
        buffered -= 6;
        buffer &= buffered ? ((1 << buffered) - 1) : 0;
      }
    }
    if (buffered) sixBitStates.push((buffer << (6 - buffered)) & 0x3f);
    const bits = bytes.length * 8;
    return Object.freeze({
      byteLength: bytes.length,
      bitLength: bits,
      ones,
      zeros: bits - ones,
      parity: ones & 1,
      byteXor: xor >>> 0,
      byteSum: sum >>> 0,
      sixBitStateCount: sixBitStates.length,
      sixBitStates: Object.freeze(sixBitStates.slice(0, 4096)),
      primitiveHash: fnv1a32(bytes.length ? Array.from(bytes).join(',') : ''),
    });
  }

  /**
   * Resolve the complete address AFTER the first execution attempt. The
   * execution path, success/failure, engine, errors, stdout/return shape and
   * low-level artifact primitives all participate in placement.
   */
  resolve({ artifact = {}, execution = {}, decomposition = null } = {}) {
    const primitives = decomposition ?? this.descend(artifact);
    const observed = {
      artifact: {
        name: artifact?.name ?? artifact?.originalName ?? null,
        type: artifact?.type ?? null,
        primitives,
      },
      execution: {
        ok: Boolean(execution?.ok),
        path: execution?.path ?? null,
        kind: execution?.kind ?? artifact?.type ?? null,
        runtimeAdapter: execution?.runtimeAdapter ?? null,
        engine: execution?.result?.engine ?? execution?.result?.result?.engine ?? null,
        stdout: safeResult(execution?.result?.stdout ?? execution?.result?.result?.stdout ?? null),
        returnValue: safeResult(execution?.result?.returnValue ?? execution?.result?.result?.returnValue ?? null),
        error: execution?.error ?? execution?.result?.error ?? null,
        originalArtifactExecuted: execution?.evidence?.originalArtifactExecuted ?? null,
      },
    };

    let n = hash128(observed);
    let x;
    x = take(n, 12); const house = x.digit + 1; n = x.rest;
    x = take(n, 12); const zodiac = x.digit + 1; n = x.rest;
    x = take(n, 99); const arcUnit = x.digit + 1; n = x.rest;
    x = take(n, 60); const second = x.digit; n = x.rest;
    x = take(n, 60); const minute = x.digit; n = x.rest;
    x = take(n, this.degreeSubdivisions); const degreeSubdivision = x.digit + 1; n = x.rest;
    x = take(n, this.degreeBands); const degreeBand = x.digit + 1; n = x.rest;
    x = take(n, 5); const base = x.digit + 1; n = x.rest;
    x = take(n, 6); const tone = x.digit + 1; n = x.rest;
    x = take(n, 6); const color = x.digit + 1; n = x.rest;
    x = take(n, 6); const line = x.digit + 1; n = x.rest;
    x = take(n, 64); const gate = x.digit + 1; n = x.rest;
    x = take(n, 5); const dimensionIndex = x.digit + 1; n = x.rest;
    x = take(n, 13); const planetary = x.digit + 1;

    const address = {
      planetary,
      dimension: DIMENSIONS[dimensionIndex - 1],
      gate,
      line,
      color,
      tone,
      base,
      degree: {
        band: degreeBand,
        subdivision: degreeSubdivision,
        structure: `${this.degreeBands}-of-${this.degreeSubdivisions}`,
      },
      minute,
      second,
      arcAxis: {
        arcUnit,
        axis: AXES[(arcUnit - 1) % AXES.length],
      },
      zodiac,
      house,
    };
    validateCanonicalAddress(address);

    const derivation = Object.freeze({
      basis: 'first-execution-derived',
      status: 'computational-coordinate',
      intrinsicSemanticClaim: false,
      primitiveHash: primitives.primitiveHash,
      observationHash: hash128(observed).toString(16).padStart(32, '0'),
      decomposition: primitives,
      observedExecution: observed.execution,
    });

    return Object.freeze({
      address: Object.freeze(address),
      key: canonicalAddressKey(address),
      derivation,
    });
  }
}

export default ExecutionAddressResolver;
