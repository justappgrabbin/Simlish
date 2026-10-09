// gate-address.mjs
//
// Bridges three things that don't share a numbering scheme on their own:
//   - iching.mjs's hexagram(): binary value 0-63 -> "number" 1-64 (pure
//     binary order, e.g. number 1 = 000000, number 64 = 111111)
//   - king-wen.mjs's GATE_TO_LINES: classical King Wen gate 1-64 -> the
//     6-line yin/yang pattern for that gate (Gate 1 = 111111, "The
//     Creative")
//   - gate-data.mjs's GATE_DATA: King Wen gate number -> name/circuit/
//     keynote/archetype content
//
// These are NOT the same ordering. iching.mjs's "number" field is binary
// sequence order; GATE_DATA and GATE_TO_LINES are both King Wen order.
// Feeding a King Wen gate number straight into fromNumber() silently
// returns the wrong hexagram. This module is the only place that
// conversion should happen.

import { hexagram } from './iching.mjs';
import { GATE_TO_LINES, gateToLines } from './king-wen.mjs';
import { gateContent } from './gate-data.mjs';
import { addressed, normalizeAddress } from './address-space.mjs';

// Binary-value (0-63) -> King Wen gate number, built once from the
// classical table so the reverse lookup doesn't need its own hand-kept copy.
const BINARY_VALUE_TO_KING_WEN = new Map();
for (const [gate, lines] of Object.entries(GATE_TO_LINES)) {
  const bits = [...lines].reverse().join(''); // lines are bottom-to-top; iching.mjs bits are top-down
  BINARY_VALUE_TO_KING_WEN.set(parseInt(bits, 2), Number(gate));
}

/**
 * Resolve a classical King Wen gate number (1-64, the numbers Human
 * Design / gate-data.mjs / king-wen.mjs all use) into:
 *   - its ATO-Core hexagram (binary value + iching.mjs's own "number")
 *   - its content (name, circuit, keynote, archetype, etc.)
 *   - a canonical address usable with address-space.mjs
 */
export function resolveGate(kingWenGate) {
  const lines = gateToLines(kingWenGate); // throws on invalid gate
  const hex = hexagram([...lines].reverse()); // hexagram() expects top-down
  const content = gateContent(kingWenGate);
  if (!content) throw new Error(`resolveGate: no GATE_DATA content for gate ${kingWenGate}`);

  const address = addressed(content, {
    mode: 'macro',
    gate: kingWenGate,
    line: content.line,
    color: content.color,
    tone: content.tone,
    base: content.base,
  }, { kind: 'gate', provenance: 'GATE_DATA+king-wen.mjs' });

  return Object.freeze({
    kingWenGate,
    binaryValue: hex.index,
    atoCoreNumber: hex.number, // iching.mjs's own numbering, NOT King Wen
    bits: hex.bits,
    content,
    address,
  });
}

/** Reverse: given an ATO-Core binary hexagram value (0-63), find its King Wen gate, if known. */
export function kingWenGateForBinaryValue(binaryValue) {
  return BINARY_VALUE_TO_KING_WEN.get(binaryValue) ?? null;
}
