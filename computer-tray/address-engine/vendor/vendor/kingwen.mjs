// Pure Synthia Automata — merged from synthia-recursive-media-field-v1.9/src/srmf-structural-growth-v16.mjs (KING_WEN_DECIMAL, ported there from the user's FuxiEncoder.ts)

/**
 * King Wen gate number (1..64, received sequence) <-> Fu Xi binary decimal (0..63).
 * Fu Xi decimal convention: line 1 (bottom) = bit 0, so gate 1 (all yang) = 63
 * and gate 2 (all yin) = 0. Verified against the source table:
 * gate 3 (Difficulty at the Beginning, thunder below / water above) = 0b010001 = 17.
 */
const TABLE = Object.freeze([
  63, 0, 17, 34, 23, 58, 2, 16,
  55, 59, 7, 56, 61, 47, 4, 8,
  25, 38, 3, 48, 41, 37, 32, 1,
  57, 39, 33, 30, 18, 45, 28, 14,
  60, 15, 40, 5, 53, 43, 20, 10,
  35, 49, 31, 62, 24, 6, 26, 22,
  29, 46, 9, 36, 52, 11, 13, 44,
  54, 27, 50, 19, 51, 12, 21, 42,
]);

/** King Wen gate# (1..64) -> Fu Xi decimal (0..63), full 64-entry table. */
export const KING_WEN_TO_FUXI_DECIMAL = Object.freeze(
  Object.fromEntries(TABLE.map((decimal, index) => [index + 1, decimal])),
);

/** Fu Xi decimal (0..63) -> King Wen gate# (1..64). */
export const FUXI_DECIMAL_TO_KING_WEN = Object.freeze(
  Object.fromEntries(TABLE.map((decimal, index) => [decimal, index + 1])),
);

/** King Wen hexagram names (standard Wilhelm/Baynes English titles), index = gate - 1. */
export const KING_WEN_HEXAGRAM_NAMES = Object.freeze([
  'The Creative', 'The Receptive', 'Difficulty at the Beginning', 'Youthful Folly',
  'Waiting (Nourishment)', 'Conflict', 'The Army', 'Holding Together (Union)',
  'The Taming Power of the Small', 'Treading (Conduct)', 'Peace', 'Standstill (Stagnation)',
  'Fellowship with Men', 'Possession in Great Measure', 'Modesty', 'Enthusiasm',
  'Following', 'Work on What Has Been Spoiled (Decay)', 'Approach', 'Contemplation (View)',
  'Biting Through', 'Grace', 'Splitting Apart', 'Return (The Turning Point)',
  'Innocence (The Unexpected)', 'The Taming Power of the Great', 'The Corners of the Mouth (Providing Nourishment)',
  'Preponderance of the Great', 'The Abysmal (Water)', 'The Clinging (Fire)',
  'Influence (Wooing)', 'Duration', 'Retreat', 'The Power of the Great',
  'Progress', 'Darkening of the Light', 'The Family (The Clan)', 'Opposition',
  'Obstruction', 'Deliverance', 'Decrease', 'Increase',
  'Break-through (Resoluteness)', 'Coming to Meet', 'Gathering Together (Massing)', 'Pushing Upward',
  'Oppression (Exhaustion)', 'The Well', 'Revolution (Molting)', 'The Caldron',
  'The Arousing (Shock, Thunder)', 'Keeping Still (Mountain)', 'Development (Gradual Progress)', 'The Marrying Maiden',
  'Abundance (Fullness)', 'The Wanderer', 'The Gentle (The Penetrating, Wind)', 'The Joyous (Lake)',
  'Dispersion (Dissolution)', 'Limitation', 'Inner Truth', 'Preponderance of the Small',
  'After Completion', 'Before Completion',
]);

/** King Wen gate# (1..64) -> Fu Xi decimal (0..63). */
export function gateToFuXiDecimal(gate) {
  if (!Number.isInteger(gate) || gate < 1 || gate > 64) {
    throw new RangeError('gate must be 1..64.');
  }
  return TABLE[gate - 1];
}

/** Fu Xi decimal (0..63) -> King Wen gate# (1..64). */
export function fuXiDecimalToGate(decimal) {
  const gate = FUXI_DECIMAL_TO_KING_WEN[decimal];
  if (!gate) {
    throw new RangeError(`No King Wen gate for Fu-Xi decimal ${decimal}.`);
  }
  return gate;
}

/** English King Wen name for a gate number (1..64). */
export function hexagramName(gate) {
  if (!Number.isInteger(gate) || gate < 1 || gate > 64) {
    throw new RangeError('gate must be 1..64.');
  }
  return KING_WEN_HEXAGRAM_NAMES[gate - 1];
}

/** 6-bit pattern for a King Wen gate, line 1 (bottom) first. */
export function gatePattern(gate) {
  const decimal = gateToFuXiDecimal(gate);
  return Object.freeze(Array.from({ length: 6 }, (_, i) => (decimal >> i) & 1));
}

export default Object.freeze({
  KING_WEN_TO_FUXI_DECIMAL,
  FUXI_DECIMAL_TO_KING_WEN,
  KING_WEN_HEXAGRAM_NAMES,
  gateToFuXiDecimal,
  fuXiDecimalToGate,
  hexagramName,
  gatePattern,
});
