// Shared deterministic hashing + seeded RNG utilities (no dependencies).
import { createHash } from 'node:crypto';

export function sha256Hex(input) {
  const data = typeof input === 'string' ? input : JSON.stringify(input);
  return createHash('sha256').update(data).digest('hex');
}

export function shortHash(input, len = 16) {
  return sha256Hex(input).slice(0, len);
}

// Mulberry32 seeded PRNG, deterministic from a string seed.
export function seededRng(seedString) {
  let a = 0;
  const hex = sha256Hex(seedString).slice(0, 8);
  a = parseInt(hex, 16) >>> 0;
  return function rng() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function pickSeeded(arr, rng) {
  if (!Array.isArray(arr) || arr.length === 0) throw new Error('pickSeeded: empty array');
  return arr[Math.floor(rng() * arr.length)];
}

// Derive n floats in [0,1) deterministically from a seed string.
export function deriveFloats(seedString, n) {
  const rng = seededRng(seedString);
  const out = [];
  for (let i = 0; i < n; i++) out.push(rng());
  return out;
}
