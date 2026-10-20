/**
 * OntologicalAddress — precise coordinate system for every point in the
 * 5-layer × 64-node × 6-line phase space.
 *
 * Address format (as specified):
 *   [p].[DIM].[G{gate}].[L{line}].[C{color}].[T{tone}].[B{base}].[DEG{d}].[MIN{m}].[SEC{s}].[ARC{a}].[ZOD{sign}].[H{house}]
 *
 * Capitalization rules (user-specified):
 *   p    = planet abbreviation (mixed, e.g. "Sun", "Moon")
 *   DIM  = all caps
 *   G    = uppercase prefix, number follows
 *   L    = uppercase prefix, number follows
 *   C    = uppercase C only (color), number follows
 *   T    = uppercase T only (tone), number follows
 *   B    = uppercase B only (base), number follows
 *   DEG  = all caps
 *   MIN  = all caps
 *   SEC  = all caps
 *   ARC  = all caps
 *   ZOD  = all caps (+ sign abbreviation all caps)
 *   H    = uppercase H, number follows
 *
 * Generative hierarchy:
 *   Level 0 — Base (B1-B5):         monogram fragments (5-state)
 *   Level 1 — Tone (T1-T6):         monogram (yin/yang classification)
 *   Level 2 — Color (C1-C6):        monogram (one I Ching line)
 *   Level 3 — Bigram (C+T):         2-line pattern from ARC × ZOD
 *   Level 4 — Trigram (C+T+B):      3-line pattern → lower or upper trigram
 *   Level 5 — Hexagram (Gate):      two trigrams → 64 hexagrams
 *   Level 6 — Channel:              two Gates via changing line
 *   Level 7 — Circuit/Triple:       three channels → AutoLing-scale construct
 *
 * Each level's address IS its unique identifier — there are no collisions.
 */

// ── Planet table ─────────────────────────────────────────────────────────────
export const PLANETS = {
  Sun:     { symbol: '☉', abbr: 'Sun',  DIM: 1 },
  Earth:   { symbol: '⊕', abbr: 'Erth', DIM: 2 },
  Moon:    { symbol: '☽', abbr: 'Moon', DIM: 3 },
  NNode:   { symbol: '☊', abbr: 'NNod', DIM: 4 },
  SNode:   { symbol: '☋', abbr: 'SNod', DIM: 5 },
  Mercury: { symbol: '☿', abbr: 'Merc', DIM: 6 },
  Venus:   { symbol: '♀', abbr: 'Ven',  DIM: 7 },
  Mars:    { symbol: '♂', abbr: 'Mars', DIM: 8 },
  Jupiter: { symbol: '♃', abbr: 'Jup',  DIM: 9 },
  Saturn:  { symbol: '♄', abbr: 'Sat',  DIM: 10 },
  Uranus:  { symbol: '♅', abbr: 'Ura',  DIM: 11 },
  Neptune: { symbol: '♆', abbr: 'Nep',  DIM: 12 },
  Pluto:   { symbol: '♇', abbr: 'Plu',  DIM: 13 },
  Chiron:  { symbol: '⚷', abbr: 'Chi',  DIM: 14 },
};

// ── Zodiac table ──────────────────────────────────────────────────────────────
export const ZODIAC = [
  { name: 'Aries',       abbr: 'ARI', start:   0, end:  30, symbol: '♈', bigram: 0b01 },
  { name: 'Taurus',      abbr: 'TAU', start:  30, end:  60, symbol: '♉', bigram: 0b11 },
  { name: 'Gemini',      abbr: 'GEM', start:  60, end:  90, symbol: '♊', bigram: 0b10 },
  { name: 'Cancer',      abbr: 'CAN', start:  90, end: 120, symbol: '♋', bigram: 0b00 },
  { name: 'Leo',         abbr: 'LEO', start: 120, end: 150, symbol: '♌', bigram: 0b01 },
  { name: 'Virgo',       abbr: 'VIR', start: 150, end: 180, symbol: '♍', bigram: 0b11 },
  { name: 'Libra',       abbr: 'LIB', start: 180, end: 210, symbol: '♎', bigram: 0b10 },
  { name: 'Scorpio',     abbr: 'SCO', start: 210, end: 240, symbol: '♏', bigram: 0b00 },
  { name: 'Sagittarius', abbr: 'SAG', start: 240, end: 270, symbol: '♐', bigram: 0b01 },
  { name: 'Capricorn',   abbr: 'CAP', start: 270, end: 300, symbol: '♑', bigram: 0b11 },
  { name: 'Aquarius',    abbr: 'AQU', start: 300, end: 330, symbol: '♒', bigram: 0b10 },
  { name: 'Pisces',      abbr: 'PIS', start: 330, end: 360, symbol: '♓', bigram: 0b00 },
];

// ── Gate → ecliptic degree (Human Design wheel, gate order starting 0° Aries) ─
// Each gate covers 5.625° (360 / 64). The HD zodiac wheel starts with Gate 41 at 0°.
const HD_GATE_ORDER = [
  41,19,13,49,30,55,37,63,22,36,25,17,21,51,42,3,
  27,24,2,23,8,20,16,35,45,12,15,52,39,53,62,56,
  31,33,7,4,29,59,40,64,47,6,46,18,48,57,32,50,
  28,44,1,43,14,34,9,5,26,11,10,58,38,54,61,60,
];

const GATE_DEG_PER_GATE = 360 / 64; // 5.625°
const DEG_PER_LINE      = GATE_DEG_PER_GATE / 6;   // 0.9375°
const DEG_PER_COLOR     = DEG_PER_LINE / 6;         // 0.15625°
const DEG_PER_TONE      = DEG_PER_COLOR / 6;        // 0.026042°
const DEG_PER_BASE      = DEG_PER_TONE / 5;         // 0.005208°

// Build gate → start degree lookup
const GATE_START_DEG = new Map();
HD_GATE_ORDER.forEach((gate, i) => {
  GATE_START_DEG.set(gate, i * GATE_DEG_PER_GATE);
});

// ── Zodiac lookup by degree ───────────────────────────────────────────────────
function zodiacAt(deg) {
  const d = ((deg % 360) + 360) % 360;
  return ZODIAC.find(z => d >= z.start && d < z.end) || ZODIAC[0];
}

// ── House (Porphyry / equal house, 30° each from 0° Aries baseline) ──────────
// In a real chart this requires ASC; here we use equal-house from ecliptic.
function houseAt(deg) {
  const d = ((deg % 360) + 360) % 360;
  return Math.floor(d / 30) + 1; // 1-12
}

// ── DMS conversion ────────────────────────────────────────────────────────────
function degToDMS(decimalDeg) {
  const d = Math.floor(decimalDeg);
  const mFull = (decimalDeg - d) * 60;
  const m = Math.floor(mFull);
  const sFull = (mFull - m) * 60;
  const s = Math.floor(sFull);
  const arc = Math.round((sFull - s) * 60); // arc-seconds
  return { DEG: d, MIN: m, SEC: s, ARC: arc };
}

// ── ARC bigram ────────────────────────────────────────────────────────────────
// ARC value (0-59) mapped to a 2-bit bigram (4 possible patterns)
function arcBigram(arcVal) {
  return arcVal % 4; // 0b00, 0b01, 0b10, 0b11
}

// ── Generative hierarchy computation ─────────────────────────────────────────
function computeHierarchy(gate, line, color, tone, base, eclipticDeg) {
  const zod = zodiacAt(eclipticDeg);
  const { ARC } = degToDMS(eclipticDeg);

  // Bigram 1 — from ARC (2-bit)
  const bigram1 = arcBigram(ARC);                  // 0-3
  // Bigram 2 — from zodiac element (2-bit)
  const bigram2 = zod.bigram;                       // 0-3
  // Combined bigrams → 4-bit → maps to color/tone position
  const bigramPair = (bigram1 << 2) | bigram2;      // 0-15

  // Trigram from Color+Tone+Base (3-bit from most significant bits)
  // Color 1-6, Tone 1-6, Base 1-5 → encode as 3 binary digits
  const trigramLower = ((color - 1) >> 1) & 0b111;   // lower trigram of current gate
  const trigramUpper = ((line - 1) >> 1) & 0b111;    // upper trigram indicator

  // Nuclear: lines 2-5 of the hexagram (computed in hexagram.js)
  // Here we encode the sub-hexagram from color+tone+base
  const subHexBits = (((color - 1) & 0b111) << 3) | ((tone - 1) & 0b111);

  return {
    bigram1,
    bigram2,
    bigramPair,
    bigramPattern: `${bigram1.toString(2).padStart(2,'0')}·${bigram2.toString(2).padStart(2,'0')}`,
    trigramLower,
    trigramUpper,
    subHexagram: subHexBits & 0b111111,
    level: computeLevel(color, tone, base),
  };
}

function computeLevel(color, tone, base) {
  // Determine which generative level this address primarily operates at
  if (base   > 0 && tone  === 1 && color === 1) return 0; // Base
  if (tone   > 1 && color === 1)                return 1; // Tone
  if (color  > 1)                               return 2; // Color
  return 3; // Monogram / line-level default
}

// ── OntologicalAddress class ──────────────────────────────────────────────────
export class OntologicalAddress {
  /**
   * @param {object} params
   *   planet  — planet name (key of PLANETS)
   *   dim     — dimension number (default = planet.DIM)
   *   gate    — gate number 1-64 (King Wen)
   *   line    — line number 1-6
   *   color   — color 1-6 (default 1)
   *   tone    — tone 1-6 (default 1)
   *   base    — base 1-5 (default 1)
   *   layerIndex — which state space layer (0-4), informs DIM if not set
   */
  constructor({ planet = 'Sun', dim = null, gate, line, color = 1, tone = 1, base = 1, layerIndex = 2 } = {}) {
    const planetData = PLANETS[planet] || PLANETS.Sun;

    this.planet    = planet;
    this.planetData = planetData;
    this.DIM       = dim ?? (planetData.DIM || (layerIndex + 1));
    this.gate      = gate;
    this.line      = line;
    this.color     = color;
    this.tone      = tone;
    this.base      = base;
    this.layerIndex = layerIndex;

    // Compute ecliptic position
    const gateStartDeg = GATE_START_DEG.get(gate) ?? 0;
    const lineOffset   = (line  - 1) * DEG_PER_LINE;
    const colorOffset  = (color - 1) * DEG_PER_COLOR;
    const toneOffset   = (tone  - 1) * DEG_PER_TONE;
    const baseOffset   = (base  - 1) * DEG_PER_BASE;

    this.eclipticDeg = gateStartDeg + lineOffset + colorOffset + toneOffset + baseOffset;

    // DMS breakdown
    const dms = degToDMS(this.eclipticDeg);
    this.DEG = dms.DEG;
    this.MIN = dms.MIN;
    this.SEC = dms.SEC;
    this.ARC = dms.ARC;

    // Zodiac + house
    this.zodiac = zodiacAt(this.eclipticDeg);
    this.ZOD    = this.zodiac.abbr;      // all caps
    this.H      = houseAt(this.eclipticDeg);

    // Generative hierarchy
    this.hierarchy = computeHierarchy(gate, line, color, tone, base, this.eclipticDeg);
  }

  /**
   * Canonical string representation.
   * Format: p.DIM{n}.G{gate}.L{line}.C{color}.T{tone}.B{base}.DEG{d}.MIN{m}.SEC{s}.ARC{a}.ZOD_{sign}.H{house}
   */
  toString() {
    return [
      this.planet,
      `DIM${this.DIM}`,
      `G${this.gate}`,
      `L${this.line}`,
      `C${this.color}`,
      `T${this.tone}`,
      `B${this.base}`,
      `DEG${this.DEG}`,
      `MIN${this.MIN}`,
      `SEC${this.SEC}`,
      `ARC${this.ARC}`,
      `ZOD_${this.ZOD}`,
      `H${this.H}`,
    ].join('.');
  }

  /**
   * Parse an address string back into components.
   */
  static parse(str) {
    const parts = str.split('.');
    if (parts.length < 13) return null;
    try {
      const [planet, dim, g, l, c, t, b, deg, min, sec, arc, zod, h] = parts;
      return new OntologicalAddress({
        planet: planet,
        dim:    parseInt(dim.replace('DIM', '')),
        gate:   parseInt(g.replace('G', '')),
        line:   parseInt(l.replace('L', '')),
        color:  parseInt(c.replace('C', '')),
        tone:   parseInt(t.replace('T', '')),
        base:   parseInt(b.replace('B', '')),
        // DEG/MIN/SEC/ARC/ZOD/H are re-computed from gate+line+color+tone+base
      });
    } catch { return null; }
  }

  /**
   * Generate the addresses for all 6 lines of a gate (one planet, default color/tone/base).
   */
  static forGate(gate, planet = 'Sun', dim = null) {
    return Array.from({ length: 6 }, (_, i) =>
      new OntologicalAddress({ planet, dim, gate, line: i + 1 })
    );
  }

  /**
   * Generate all addresses for all 64 gates × all 6 lines (384 total positions).
   */
  static fullWheel(planet = 'Sun') {
    return HD_GATE_ORDER.flatMap((gate, gi) =>
      Array.from({ length: 6 }, (_, li) =>
        new OntologicalAddress({ planet, gate, line: li + 1 })
      )
    );
  }

  /**
   * The "changing line" address — the address this node transforms TO
   * when line {lineIndex} flips. Returns the resultant gate's address.
   */
  changingLineTo(lineIndex, fuxi) {
    // Flip the line in the hexagram binary
    const newFuxi = fuxi ^ (1 << lineIndex);
    // Find the gate KW number for newFuxi
    // (import hexagrams at call time to avoid circular)
    return newFuxi; // caller resolves to KW gate
  }

  /**
   * Generative hierarchy info — what level of tool this address generates.
   */
  generativeLevel() {
    const { level } = this.hierarchy;
    const LEVEL_NAMES = [
      'Base — quantum fragment (B)',
      'Tone — monogram yin/yang (T)',
      'Color — I Ching line (C)',
      'Bigram — 2-line pattern (ARC×ZOD)',
      'Trigram — 3-line pattern (C+T+B)',
      'Hexagram — Gate (two trigrams)',
      'Channel — two gates, changing line',
      'Circuit — triple channel construct (AutoLing)',
    ];
    return { level, name: LEVEL_NAMES[level] ?? 'Unknown' };
  }

  toJSON() {
    return {
      address: this.toString(),
      planet: this.planet,
      planetSymbol: this.planetData.symbol,
      DIM: this.DIM,
      gate: this.gate,
      line: this.line,
      color: this.color,
      tone: this.tone,
      base: this.base,
      eclipticDeg: this.eclipticDeg,
      DEG: this.DEG,
      MIN: this.MIN,
      SEC: this.SEC,
      ARC: this.ARC,
      zodiac: this.zodiac,
      ZOD: this.ZOD,
      H: this.H,
      hierarchy: this.hierarchy,
      generativeLevel: this.generativeLevel(),
    };
  }
}

// ── Convenience: compute address for a hexagram node ─────────────────────────
/**
 * Given a hexagram node (with .kw gate number), compute its ontological addresses
 * for all 6 lines across all 13 planets.
 */
export function addressesForNode(kwGate, layerIndex = 2, planet = 'Sun') {
  return Array.from({ length: 6 }, (_, i) =>
    new OntologicalAddress({ planet, gate: kwGate, line: i + 1, layerIndex })
  );
}

/**
 * Primary address for a node: Sun, Line 1, Color 1, Tone 1, Base 1
 * This is the "root" address — the most stable attractor point.
 */
export function rootAddress(kwGate, layerIndex = 2) {
  return new OntologicalAddress({ planet: 'Sun', gate: kwGate, line: 1, layerIndex });
}

export { HD_GATE_ORDER, GATE_START_DEG };
