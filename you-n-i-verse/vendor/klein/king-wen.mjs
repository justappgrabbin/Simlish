// kingWen.js
//
// Gate number -> 6 binary lines (bottom-to-top, 1=yang/solid, 0=yin/broken),
// per the classical King Wen sequence. Human Design gate numbers ARE King
// Wen hexagram numbers -- this is not a new mapping, it's the fixed
// classical structure every hexagram has always had.
//
// This is the missing 200-byte piece that connects two existing, separate
// entry points into the same 64-gate substrate:
//   - resolve_address() (resonance_engine.js) enters via zodiac DEGREE
//   - IChingGrammar.run() enters via 6 binary LINES
// Neither tool changes. This just lets one feed the other.
//
// Spot-checked against gates already confirmed from primary source tonight:
//   Gate 56 = "Lu / The Wanderer"      (Black Book: 56.3 = Alienation)  match
//   Gate 55 = "Feng / Abundance"        (Black Book: 55.3 = Innocence)   match
//   Gate 28 = "Da Guo / Great Exceeding" (Black Book: 28.1 = Preparation) match
//   Gate 6  = "Song / Conflict"         (Black Book: 6.4 = Triumph)      match
//   Gate 59 = "Huan / Dispersion"       (session: "dispersion is sexuality") match
// Recommend spot-checking the rest against your Book of Letters at your
// convenience -- this is standard classical material, not proprietary or
// invented, but worth your own verification pass before it's load-bearing.

export const GATE_TO_LINES = {
  1: [1, 1, 1, 1, 1, 1],  2: [0, 0, 0, 0, 0, 0],  3: [1, 0, 0, 0, 1, 0],
  4: [0, 1, 0, 0, 0, 1],  5: [1, 1, 1, 0, 1, 0],  6: [0, 1, 0, 1, 1, 1],
  7: [0, 1, 0, 0, 0, 0],  8: [0, 0, 0, 0, 1, 0],  9: [1, 1, 1, 0, 1, 1],
  10: [1, 1, 0, 1, 1, 1], 11: [1, 1, 1, 0, 0, 0], 12: [0, 0, 0, 1, 1, 1],
  13: [1, 0, 1, 1, 1, 1], 14: [1, 1, 1, 1, 0, 1], 15: [0, 0, 1, 0, 0, 0],
  16: [0, 0, 0, 1, 0, 0], 17: [1, 0, 0, 1, 1, 0], 18: [0, 1, 1, 0, 0, 1],
  19: [1, 1, 0, 0, 0, 0], 20: [0, 0, 0, 0, 1, 1], 21: [1, 0, 0, 1, 0, 1],
  22: [1, 0, 1, 0, 0, 1], 23: [0, 0, 0, 0, 0, 1], 24: [1, 0, 0, 0, 0, 0],
  25: [1, 0, 0, 1, 1, 1], 26: [1, 1, 1, 0, 0, 1], 27: [1, 0, 0, 0, 0, 1],
  28: [0, 1, 1, 1, 1, 0], 29: [0, 1, 0, 0, 1, 0], 30: [1, 0, 1, 1, 0, 1],
  31: [0, 0, 1, 1, 1, 0], 32: [0, 1, 1, 1, 0, 0], 33: [0, 0, 1, 1, 1, 1],
  34: [1, 1, 1, 1, 0, 0], 35: [0, 0, 0, 1, 0, 1], 36: [1, 0, 1, 0, 0, 0],
  37: [1, 0, 1, 0, 1, 1], 38: [1, 1, 0, 1, 0, 1], 39: [0, 1, 0, 1, 0, 0],
  40: [0, 0, 1, 0, 1, 0], 41: [1, 1, 0, 0, 0, 1], 42: [1, 0, 0, 0, 1, 1],
  43: [1, 1, 1, 1, 1, 0], 44: [0, 1, 1, 1, 1, 1], 45: [0, 0, 0, 1, 1, 0],
  46: [0, 1, 1, 0, 0, 0], 47: [0, 1, 0, 1, 1, 0], 48: [0, 1, 1, 0, 1, 0],
  49: [1, 0, 1, 1, 1, 0], 50: [0, 1, 1, 1, 0, 1], 51: [1, 0, 0, 1, 0, 0],
  52: [0, 0, 1, 0, 0, 1], 53: [0, 0, 1, 0, 1, 1], 54: [1, 1, 0, 1, 0, 0],
  55: [1, 0, 1, 1, 0, 0], 56: [0, 0, 1, 1, 0, 1], 57: [0, 1, 1, 0, 1, 1],
  58: [1, 1, 0, 1, 1, 0], 59: [0, 1, 0, 0, 1, 1], 60: [1, 1, 0, 0, 1, 0],
  61: [1, 1, 0, 0, 1, 1], 62: [0, 0, 1, 1, 0, 0], 63: [1, 0, 1, 0, 1, 0],
  64: [0, 1, 0, 1, 0, 1],
};

/**
 * Given a gate number (1-64), return the 6-line array IChingGrammar.run()
 * expects. Throws on invalid gate rather than silently returning garbage.
 */
export function gateToLines(gate) {
  const lines = GATE_TO_LINES[gate];
  if (!lines) throw new Error(`gateToLines: no King Wen entry for gate ${gate}`);
  return [...lines];
}

/**
 * Convenience bridge: takes a gate number, runs it through IChingGrammar,
 * and returns the feature set -- closing the loop from
 * resolve_address()'s degree-based gate output straight into the
 * hexagram feature substrate, with zero changes to either original tool.
 */
export function gateToFeatures(gate, iChingGrammarTool) {
  const lines = gateToLines(gate);
  return iChingGrammarTool.run({ lines });
}
