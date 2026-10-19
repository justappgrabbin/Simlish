// state/ontologicalAddress.js — UPGRADE_PLAN §3.
// An OntologicalAddress is a persistent state-space position for every entity
// (player, NPC, world object):
//   position = f(identity × world conditions × purpose × lineage × history)
// Fully deterministic and reproducible from recorded state — no clocks, no
// ambient randomness. The address is RE-DERIVABLE: position depends only on
// the recorded fields, so any fresh instance reconstructs the same position.
//
// Storage: player address lives on the PlayerSeed (`ontologicalAddress` — a
// non-canonical, updatable field); NPC/object addresses live in game state.

import { sha256Hex, deriveFloats } from '../util/hash.js';

export const ADDRESS_DIMENSIONS = 8;
export const ADDRESS_VERSION = 1;

const r4 = (v) => +v.toFixed(4);

/**
 * Create an OntologicalAddress.
 * @param {object} input - {entityId, identityRef, worldId, purpose, lineageMarkers?}
 *   entityId    — stable id ('pl-...', 'npc:ferryman', 'obj:bronze-mirror')
 *   identityRef — hash/reference of the canonical identity (anchorsHash,
 *                 visualIdentity fingerprint, or synth-anchor hash for NPCs)
 *   worldId     — world name ('iching-story', ...)
 *   purpose     — short string; the entity's role/intent in this world
 *   lineageMarkers — inherited marks (persist across morphs)
 */
export function createAddress({ entityId, identityRef, worldId, purpose, lineageMarkers = [] }) {
  if (!entityId || !identityRef || !worldId || !purpose) {
    throw new Error('ontologicalAddress.createAddress: entityId, identityRef, worldId and purpose are all required');
  }
  const address = {
    version: ADDRESS_VERSION,
    entityId,
    identityRef,
    worldId,
    purpose,
    lineageMarkers: structuredClone(lineageMarkers),
    history: [],           // [{tick, event, detail}] — appended by recordEvent
    position: null,        // filled below
    addressHash: null,
  };
  return finalize(address);
}

/**
 * Append a history event and re-derive the position. The address "moves" as
 * history accrues, deterministically. Returns the same (mutated) address.
 */
export function recordEvent(address, { tick, event, detail = null }) {
  if (!address?.entityId) throw new Error('ontologicalAddress.recordEvent: an OntologicalAddress is required');
  if (!event) throw new Error('ontologicalAddress.recordEvent: event string required');
  address.history.push({ tick: tick ?? address.history.length, event, detail });
  return finalize(address);
}

/**
 * Re-derive position + hash from recorded fields. Exposed so a fresh process
 * can verify a loaded address is self-consistent (reproducibility check).
 */
export function finalize(address) {
  address.position = derivePosition(address);
  address.addressHash = sha256Hex({
    v: address.version, entityId: address.entityId, identityRef: address.identityRef,
    worldId: address.worldId, purpose: address.purpose,
    lineage: address.lineageMarkers.map((m) => m.symbol ?? m.gate ?? m.type ?? m),
    history: address.history,
  }).slice(0, 24);
  return address;
}

// position vector: deterministic from identity × world × purpose × lineage × history.
function derivePosition(address) {
  const basis = {
    identity: address.identityRef,
    world: address.worldId,
    purpose: address.purpose,
    lineage: address.lineageMarkers.map((m) => m.symbol ?? m.gate ?? m.type ?? m),
    history: address.history.map((h) => `${h.tick}:${h.event}`),
  };
  return deriveFloats(sha256Hex(basis), ADDRESS_DIMENSIONS).map(r4);
}

/** Euclidean distance between two addresses' positions (0 = same locus). */
export function addressDistance(a, b) {
  if (!a?.position || !b?.position) throw new Error('ontologicalAddress.addressDistance: two finalized addresses required');
  let s = 0;
  for (let i = 0; i < ADDRESS_DIMENSIONS; i++) s += (a.position[i] - b.position[i]) ** 2;
  return +Math.sqrt(s).toFixed(4);
}

/** Verify a loaded address re-derives to the same position/hash. */
export function verifyAddress(address) {
  const clone = structuredClone(address);
  const re = finalize(clone);
  return re.addressHash === address.addressHash
    && JSON.stringify(re.position) === JSON.stringify(address.position);
}
