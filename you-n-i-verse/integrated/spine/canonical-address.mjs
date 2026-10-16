/**
 * Canonical Synthia address schema.
 *
 * IMPORTANT: this is the full address spine. It is not reduced to the
 * fine-index subset used by fine-index.mjs.
 */
export const ADDRESS_FIELDS = Object.freeze([
  'planetary',
  'dimension',
  'gate',
  'line',
  'color',
  'tone',
  'base',
  'degree',
  'minute',
  'second',
  'arcAxis',
  'zodiac',
  'house',
]);

export const DIMENSIONS = Object.freeze([
  'Movement', 'Evolution', 'Being', 'Design', 'Space',
]);

export const AXES = Object.freeze(['Vertical', 'Horizontal', 'Diagonal']);

/**
 * Known canonical cardinalities / structures.
 * Degree is deliberately represented as the gate-span DMS structure instead
 * of being silently replaced with an unrelated 360-value simplification.
 */
export const ADDRESS_STRUCTURE = Object.freeze({
  planetary: 13,
  dimension: 5,
  gate: 64,
  line: 6,
  color: 6,
  tone: 6,
  base: 5,
  degree: Object.freeze({
    kind: '5-of-29',
    bands: 5,
    subdivisions: 29,
    // Historical gate-span descriptor is preserved as metadata; it is not
    // used to truncate minute/second/arc fields from the canonical spine.
    gateSpan: Object.freeze({ degrees: 5, minutes: 37, seconds: 30 }),
  }),
  minute: 60,
  second: 60,
  arcUnit: 99,
  zodiac: 12,
  house: 12,
});

function assertIntRange(name, value, min, max) {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} out of range: expected integer ${min}..${max}, got ${value}`);
  }
}

/**
 * Validate a canonical address without inventing missing semantics.
 *
 * `degree` is preserved as supplied and may be validated by a domain-specific
 * degree validator passed via options.degreeValidator. This keeps the canonical
 * 5°37′30″ gate-span structure intact instead of imposing a guessed encoding.
 */
export function validateCanonicalAddress(address, options = {}) {
  if (!address || typeof address !== 'object') throw new TypeError('address must be an object');

  for (const field of ADDRESS_FIELDS) {
    if (!(field in address)) throw new TypeError(`missing canonical address field: ${field}`);
  }

  assertIntRange('planetary', address.planetary, 1, 13);

  if (typeof address.dimension === 'number') {
    assertIntRange('dimension', address.dimension, 1, 5);
  } else if (!DIMENSIONS.includes(address.dimension)) {
    throw new RangeError(`dimension out of range: ${address.dimension}`);
  }

  assertIntRange('gate', address.gate, 1, 64);
  assertIntRange('line', address.line, 1, 6);
  assertIntRange('color', address.color, 1, 6);
  assertIntRange('tone', address.tone, 1, 6);
  assertIntRange('base', address.base, 1, 5);
  assertIntRange('minute', address.minute, 0, 59);
  assertIntRange('second', address.second, 0, 59);
  assertIntRange('zodiac', address.zodiac, 1, 12);
  assertIntRange('house', address.house, 1, 12);

  if (typeof options.degreeValidator === 'function') {
    if (!options.degreeValidator(address.degree, address)) {
      throw new RangeError(`degree rejected by degreeValidator: ${JSON.stringify(address.degree)}`);
    }
  } else if (address.degree === undefined || address.degree === null) {
    throw new TypeError('degree must be present');
  }

  const aa = address.arcAxis;
  if (!aa || typeof aa !== 'object') throw new TypeError('arcAxis must be an object');
  assertIntRange('arcAxis.arcUnit', aa.arcUnit, 1, 99);
  if (aa.axis !== undefined && !AXES.includes(aa.axis)) {
    throw new RangeError(`arcAxis.axis must be one of ${AXES.join(', ')}`);
  }

  return true;
}

export function canonicalAddressKey(address) {
  validateCanonicalAddress(address);
  return ADDRESS_FIELDS.map((field) => {
    const value = address[field];
    return `${field}=${typeof value === 'object' ? JSON.stringify(value) : String(value)}`;
  }).join('|');
}
