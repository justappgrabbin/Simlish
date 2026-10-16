/**
 * FIELD CORE — arc-position engine
 * JS-first (motor layer). Python non-motor port comes second, same math.
 *
 * Everything here follows one rule: nothing is stored, everything is derived
 * from precise arc position (degree.minute.second). Color, sound, and gate
 * identity are all readouts of the SAME position — never separate lookup tables.
 */

// ── LATTICE CONSTANTS ─────────────────────────────────────────────────────
const GATES = 64;
const LINES_PER_GATE = 6;
const COLORS_PER_LINE = 6;
const TONES_PER_COLOR = 6;
const BASES_PER_TONE = 5;
// 64 * 6 * 6 * 6 * 5 = 69,120 — confirmed against 360° / 64 = 5.625° per gate
const ARC_PER_GATE = 360 / GATES; // 5.625° = 5°37'30"

/**
 * Convert a raw ecliptic longitude (0–360°, decimal degrees) into the full
 * lattice address. This is the ONLY place gate/line/color/tone/base get
 * assigned — no other function should hardcode these.
 */
function positionToAddress(longitudeDeg) {
  const pos = ((longitudeDeg % 360) + 360) % 360; // normalize
  const gateFloat = pos / ARC_PER_GATE;
  const gate = Math.floor(gateFloat) + 1; // 1-indexed, 1..64

  const withinGate = (gateFloat - Math.floor(gateFloat)) * LINES_PER_GATE;
  const line = Math.floor(withinGate) + 1; // 1..6

  const withinLine = (withinGate - Math.floor(withinGate)) * COLORS_PER_LINE;
  const color = Math.floor(withinLine) + 1; // 1..6

  const withinColor = (withinLine - Math.floor(withinLine)) * TONES_PER_COLOR;
  const tone = Math.floor(withinColor) + 1; // 1..6

  const withinTone = (withinColor - Math.floor(withinColor)) * BASES_PER_TONE;
  const base = Math.floor(withinTone) + 1; // 1..5

  // Degree/minute/second WITHIN the gate's 5.625° span, kept as an exact
  // INTEGER arcsecond (0..20249) — not a float — so the fine-level
  // emergence check has real precision instead of float noise.
  const degWithinGate = pos - (gate - 1) * ARC_PER_GATE;
  const arcsecond = Math.round(degWithinGate * 3600);
  const deg = Math.floor(degWithinGate);
  const minFloat = (degWithinGate - deg) * 60;
  const min = Math.floor(minFloat);
  const sec = (minFloat - min) * 60;

  return {
    longitude: pos,
    gate, line, color, tone, base,
    arc: { deg, min, sec }, // display-only, kept for readability
    arcsecond // integer, used for the fine-level emergence check
  };
}

/**
 * Encode an address as a fixed-length bit vector so two nodes can be
 * compared with Hamming distance. 6-bit gate (0-63), 3-bit line (0-5),
 * 3-bit color (0-5), 3-bit tone (0-5), 3-bit base (0-4) = 18 bits total.
 */
function addressToBits(gate, line, color, tone, base) {
  const g = (gate - 1) & 0b111111;      // 6 bits
  const l = (line - 1) & 0b111;         // 3 bits
  const c = (color - 1) & 0b111;        // 3 bits
  const t = (tone - 1) & 0b111;         // 3 bits
  const b = (base - 1) & 0b111;         // 3 bits (5 values fit in 3 bits)
  return (g << 12) | (l << 9) | (c << 6) | (t << 3) | b; // 18-bit int
}

/**
 * Split bits by structural LEVEL so emergence can be checked separately at
 * each level, rather than as one flat comparison across the whole address.
 * Gate/Line = top level. Color/Tone/Base = middle level.
 */
function levelBits(gate, line, color, tone, base) {
  return {
    gateLine: ((gate - 1) & 0b111111) << 3 | ((line - 1) & 0b111),         // 9 bits
    colorToneBase: ((color - 1) & 0b111) << 6 | ((tone - 1) & 0b111) << 3 | ((base - 1) & 0b111) // 9 bits
  };
}

/**
 * Arcsecond (fine level) split into DMS-within-gate bits for its own
 * independent emergence check — degree(0-5, 3 bits) / minute(0-59, 6 bits)
 * / second(0-59, 6 bits).
 */
function arcsecLevelBits(arcsecWithinGate) {
  const deg = Math.floor(arcsecWithinGate / 3600) & 0b111;
  const remAfterDeg = arcsecWithinGate - deg * 3600;
  const min = Math.floor(remAfterDeg / 60) & 0b111111;
  const sec = Math.floor(remAfterDeg - min * 60) & 0b111111;
  return (deg << 12) | (min << 6) | sec; // 15 bits
}

function hammingDistance(bitsA, bitsB) {
  let x = bitsA ^ bitsB;
  let count = 0;
  while (x) { count += x & 1; x >>= 1; }
  return count;
}

/**
 * The Gate→Line relationship repeats structurally at every level of the
 * hierarchy — Gate/Line is one emergence site, Color/Tone/Base is another,
 * Degree/Minute/Second is another. A channel can arise independently at
 * ANY of these levels — a fine-arc channel doesn't require a Gate-level
 * channel to also be present. Returns which level(s), if any, emerged.
 */
function channelEmerges(addressA, addressB) {
  const levelsA = levelBits(addressA.gate, addressA.line, addressA.color, addressA.tone, addressA.base);
  const levelsB = levelBits(addressB.gate, addressB.line, addressB.color, addressB.tone, addressB.base);

  const gateLine = hammingDistance(levelsA.gateLine, levelsB.gateLine) === 1;
  const colorToneBase = hammingDistance(levelsA.colorToneBase, levelsB.colorToneBase) === 1;

  let arcsecond = false;
  if (addressA.arcsecond !== undefined && addressB.arcsecond !== undefined) {
    const arcA = arcsecLevelBits(addressA.arcsecond);
    const arcB = arcsecLevelBits(addressB.arcsecond);
    arcsecond = hammingDistance(arcA, arcB) === 1;
  }

  return {
    any: gateLine || colorToneBase || arcsecond,
    gateLine, colorToneBase, arcsecond
  };
}

/**
 * Standard astrological DMS format — degree WITHIN THE CURRENT SIGN (0-29),
 * not within the gate. This is the continuous-circle display format real
 * chart software uses (confirmed against reference screenshot: "14° 29' 50.56"").
 * Seconds keep decimal precision for display; this does NOT replace the
 * integer arcsecond-within-gate math used internally for lattice/channel
 * calculations — that stays gate-relative because it needs to be, this is
 * purely the human-facing readout.
 */
function positionToStandardDMS(longitudeDeg) {
  const pos = ((longitudeDeg % 360) + 360) % 360;
  const signIndex = Math.floor(pos / 30);
  const withinSign = pos - signIndex * 30; // 0..29.999...

  const degree = Math.floor(withinSign);
  const minFloat = (withinSign - degree) * 60;
  const minute = Math.floor(minFloat);
  const second = (minFloat - minute) * 60;

  return {
    sign: ZODIAC_SIGNS[signIndex],
    degree, minute,
    second: Math.round(second * 100) / 100, // 2 decimal places, matches reference
    formatted: `${degree}° ${minute}' ${second.toFixed(2)}"`
  };
}


const ARCSEC_PER_GATE = Math.round(ARC_PER_GATE * 3600); // 5.625° × 3600 = 20,250

const ZODIAC_SIGNS = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo',
  'Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces'];
const SEASONS = ['Spring','Summer','Autumn','Winter'];
// 13 planetary bodies — matches your established EPHEMERIS_13D set
const PLANETS = ['Sun','Earth','Moon','Mercury','Venus','Mars','Jupiter',
  'Saturn','Uranus','Neptune','Pluto','Chiron','Node'];
// 5 dimensions — Base layer axis (Movement/Evolution/Being/Design/Space)
const DIMENSIONS = ['Movement','Evolution','Being','Design','Space'];

/**
 * Resolve the FULL address for a given longitude. Gate/Line/Color/Tone/Base
 * plus the exact integer arcsecond position within the gate (0..20249,
 * never a float) — this is the fix for the depth-3 float collapse from
 * before. Zodiac is derived directly from longitude. Planet and Dimension
 * are inputs (which of the 13 bodies / 5 dimensions this reading is for),
 * not derivable from longitude alone. Season needs a calendar date. House
 * needs full natal data (birth time + location + house system) — flagged
 * as NOT YET IMPLEMENTED rather than faked.
 */
function resolveFullAddress(longitudeDeg, { planetIndex = 0, dimensionIndex = 0, dateForSeason = null } = {}) {
  const pos = ((longitudeDeg % 360) + 360) % 360;
  const gateFloat = pos / ARC_PER_GATE;
  const gate = Math.floor(gateFloat) + 1;

  // Exact integer arcsecond position within this gate — no float remainder.
  const gateOriginDeg = (gate - 1) * ARC_PER_GATE;
  const arcsecWithinGate = Math.round((pos - gateOriginDeg) * 3600); // 0..20249

  // Line/Color/Tone/Base derived from INTEGER arcsecond, not float division.
  const arcsecPerLine = Math.floor(ARCSEC_PER_GATE / LINES_PER_GATE);
  const line = Math.min(LINES_PER_GATE, Math.floor(arcsecWithinGate / arcsecPerLine) + 1);
  const remAfterLine = arcsecWithinGate - (line - 1) * arcsecPerLine;

  const arcsecPerColor = Math.floor(arcsecPerLine / COLORS_PER_LINE);
  const color = Math.min(COLORS_PER_LINE, Math.floor(remAfterLine / arcsecPerColor) + 1);
  const remAfterColor = remAfterLine - (color - 1) * arcsecPerColor;

  const arcsecPerTone = Math.floor(arcsecPerColor / TONES_PER_COLOR);
  const tone = Math.min(TONES_PER_COLOR, Math.floor(remAfterColor / arcsecPerTone) + 1);
  const remAfterTone = remAfterColor - (tone - 1) * arcsecPerTone;

  const arcsecPerBase = Math.floor(arcsecPerTone / BASES_PER_TONE);
  const base = Math.min(BASES_PER_TONE, Math.floor(remAfterTone / arcsecPerBase) + 1);

  const zodiacIndex = Math.floor(pos / 30); // 12 signs × 30°
  const season = dateForSeason ? SEASONS[Math.floor((dateForSeason.getMonth()) / 3)] : null;

  return {
    longitude: pos,
    gate, line, color, tone, base,
    arcsecond: arcsecWithinGate, // integer, 0..20249 — the exact "arc number"
    zodiac: ZODIAC_SIGNS[zodiacIndex],
    house: null, // NOT YET IMPLEMENTED — requires birth time + location + house system
    season,
    planet: PLANETS[planetIndex],
    dimension: DIMENSIONS[dimensionIndex],
    bits: addressToBits(gate, line, color, tone, base)
  };
}

const TOTAL_ADDRESS_SPACE =
  GATES * LINES_PER_GATE * COLORS_PER_LINE * TONES_PER_COLOR * BASES_PER_TONE *
  ARCSEC_PER_GATE * ZODIAC_SIGNS.length * 8 /* houses */ * SEASONS.length *
  PLANETS.length * DIMENSIONS.length;
// = 34,936,012,800,000 (~34.9 trillion)


// ── COLOR & TONE MEANINGS (interpretive layer, not existence logic) ────────
// These attach meaning to the 1-6 slot values already derived from arc
// position — they don't determine WHICH color/tone a node has (that's
// still computed from longitude), only what that computed value MEANS.
const COLOR_MEANINGS = {
  1: { response: 'Fear',      mode: ['Communist', 'Separatist'], binary: 'Splenic' },
  2: { response: 'Hope',      mode: ['Theist', 'Anti-theist'],   binary: 'Splenic' },
  3: { response: 'Desire',    mode: ['Leader', 'Follower'],      binary: 'Ajna' },
  4: { response: 'Need',      mode: ['Master', 'Novice'],        binary: 'Ajna' },
  5: { response: 'Guilt',     mode: ['Conditioner', 'Conditioned'], binary: 'Solar Plexus' },
  6: { response: 'Innocence', mode: ['Observer', 'Observed'],    binary: 'Solar Plexus' }
};

const TONE_MEANINGS = {
  1: { theme: 'Security',    department: 'Smell',        binary: 'Splenic' },
  2: { theme: 'Uncertainty', department: 'Taste',         binary: 'Splenic' },
  3: { theme: 'Action',      department: 'Outer Vision',  binary: 'Ajna' },
  4: { theme: 'Meditation',  department: 'Inner Vision',  binary: 'Ajna' },
  5: { theme: 'Judgement',   department: 'Feeling',       binary: 'Solar Plexus' },
  6: { theme: 'Acceptance',  department: 'Touch',         binary: 'Solar Plexus' }
};

function meaningOf(colorValue, toneValue) {
  return {
    color: COLOR_MEANINGS[colorValue] || null,
    tone: TONE_MEANINGS[toneValue] || null
  };
}

/**
 * Hue is a direct circular readout of ecliptic longitude — the same 0-360°
 * circle IS the color wheel. No separate color table; the position itself
 * is the color, expressed in a different unit.
 */
function positionToColor(longitudeDeg) {
  const hue = ((longitudeDeg % 360) + 360) % 360;
  return { hue, css: `hsl(${hue.toFixed(2)}, 70%, 55%)` };
}

// ── SOUND READOUT (same filter, different unit) ────────────────────────────
/**
 * Frequency is the SAME circular position, transposed into an audible
 * octave range (110Hz–220Hz, one octave, A2–A3) rather than a fixed
 * per-channel lookup table. Extend to Cousto-style multi-octave transposition
 * later if the single-octave range proves too narrow in practice.
 */
function positionToFrequency(longitudeDeg, baseFreq = 110, octaveSpan = 220 - 110) {
  const pos = ((longitudeDeg % 360) + 360) % 360;
  const fraction = pos / 360;
  return baseFreq + fraction * octaveSpan;
}

// ── CENTER SELECTION (mesh-level reference, emergent readout) ──────────────
/**
 * The 9 HD centers are NOT a gate-keyed lookup table. They are 9 reference
 * faces on the mesh, each carrying a baseline frequency. A node's center is
 * SELECTED at read time by matching its own derived frequency (positionToFrequency)
 * against these references — nearest match wins. No gate is ever permanently
 * "assigned" a center; the center is recomputed from field state every time.
 *
 * These 9 values are placeholder reference points spread across the same
 * audible octave used for node frequency (110-220Hz) — spacing/tuning is
 * provisional until you confirm the actual mesh face frequencies.
 */
const CENTER_FACE_FREQUENCIES = {
  Head:        110 + (220 - 110) * (0 / 8),
  Ajna:        110 + (220 - 110) * (1 / 8),
  Throat:      110 + (220 - 110) * (2 / 8),
  GCenter:     110 + (220 - 110) * (3 / 8),
  Heart:       110 + (220 - 110) * (4 / 8),
  SolarPlexus: 110 + (220 - 110) * (5 / 8),
  Sacral:      110 + (220 - 110) * (6 / 8),
  Spleen:      110 + (220 - 110) * (7 / 8),
  Root:        110 + (220 - 110) * (8 / 8)
};

function selectCenter(nodeFrequency) {
  let closest = null;
  let closestDist = Infinity;
  for (const [center, refFreq] of Object.entries(CENTER_FACE_FREQUENCIES)) {
    const dist = Math.abs(nodeFrequency - refFreq);
    if (dist < closestDist) {
      closestDist = dist;
      closest = center;
    }
  }
  return { center: closest, distance: closestDist };
}

/**
 * Single entry point: given a raw longitude, return the full node
 * readout — address, color, sound, SELECTED center, and Color/Tone
 * meanings, all derived from one filter, none of it stored.
 */
function readNode(longitudeDeg) {
  const address = positionToAddress(longitudeDeg);
  const frequency = positionToFrequency(longitudeDeg);
  return {
    address,
    color: positionToColor(longitudeDeg),
    frequency,
    center: selectCenter(frequency),
    meaning: meaningOf(address.color, address.tone)
  };
}

module.exports = {
  positionToAddress,
  resolveFullAddress,
  positionToStandardDMS,
  TOTAL_ADDRESS_SPACE,
  addressToBits,
  hammingDistance,
  channelEmerges,
  positionToColor,
  positionToFrequency,
  meaningOf,
  COLOR_MEANINGS,
  TONE_MEANINGS,
  selectCenter,
  CENTER_FACE_FREQUENCIES,
  readNode,
  ARC_PER_GATE,
  ARCSEC_PER_GATE
};

// ── Quick self-test (run directly: node field_core.js) ─────────────────────
if (require.main === module) {
  console.log('Total address space:', TOTAL_ADDRESS_SPACE.toLocaleString(), '(~34.9 trillion)');

  const full = resolveFullAddress(302.1, { planetIndex: 3, dimensionIndex: 1, dateForSeason: new Date('2026-07-03') });
  console.log('\nFull resolved address for longitude 302.1°:');
  console.log(full);

  const dms = positionToStandardDMS(302.1);
  console.log('\nStandard display format (matches reference chart style):');
  console.log(`  ${full.gate}.${full.line}  ${full.color} ${full.tone} ${full.base}   ${dms.formatted} ${dms.sign}`);

  const nodeA = readNode(302.1);
  const nodeB = readNode(302.1 + ARC_PER_GATE);
  console.log('\nFlat readout still works for color/sound/center:');
  console.log('Node A:', nodeA);
  console.log('Channel emergence (A,B), per level:', channelEmerges(nodeA.address, nodeB.address));
}
