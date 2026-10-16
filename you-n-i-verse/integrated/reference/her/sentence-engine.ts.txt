/**
 * Resonance Sentence Engine - Ra Uru Hu's complete calculation framework
 * 69,120 total combinations: 5 bases × 6 tones × 6 colors × 6 lines × 64 gates
 * 
 * Birthday, time, and place determine the fractal sublayers through ephemeris calculations:
 * Longitude → Gate → Line → Color → Tone → Base
 */

// ============================================================================
// CONSTANTS FOR EPHEMERIS CALCULATIONS
// ============================================================================

// Each gate spans 5.625° (360° / 64 gates)
export const DEGREES_PER_GATE = 360 / 64;  // 5.625°

// Each line spans 0.9375° (5.625° / 6 lines)
export const DEGREES_PER_LINE = DEGREES_PER_GATE / 6;  // 0.9375°

// Each color spans 0.15625° (0.9375° / 6 colors)
export const DEGREES_PER_COLOR = DEGREES_PER_LINE / 6;  // 0.15625°

// Each tone spans 0.0260416...° (0.15625° / 6 tones)
export const DEGREES_PER_TONE = DEGREES_PER_COLOR / 6;  // ~0.026°

// Each base spans 0.00520833...° (0.026° / 5 bases)
export const DEGREES_PER_BASE = DEGREES_PER_TONE / 5;  // ~0.0052°

// Total combinations in the lattice
export const LATTICE_SIZE = 5 * 6 * 6 * 6 * 64;  // 69,120

// ============================================================================
// THE 5 BASES (DIMENSIONS)
// ============================================================================
export interface Base {
  id: number;
  name: string;
  voice: string;
  question: string;
  sense: string;
  binary: string;
  macrocosmic: string[];
  microcosmic: string[];
  keywords: string[];
  awarenessCenter: string;
  domain: string;
}

export const BASES: Base[] = [
  {
    id: 1,
    name: "Movement",
    voice: "I Define",
    question: "Where?",
    sense: "Seeing",
    binary: "Yang/Yang (Reactive)",
    macrocosmic: ["Movement is Energy", "Energy is Creation", "Creation is Seeing", "Seeing is Landscape", "Landscape is Environment"],
    microcosmic: ["Individuality", "Activity", "Reaction", "Limitation", "Perspective", "Relation"],
    keywords: ["Movement", "Energy", "Creation", "Seeing", "Landscape", "Environment"],
    awarenessCenter: "G-Center",
    domain: "Activity, uniqueness, orientation"
  },
  {
    id: 2,
    name: "Evolution",
    voice: "I Remember",
    question: "What?",
    sense: "Taste",
    binary: "Yang/Yin (Integrative)",
    macrocosmic: ["Evolution is Gravity", "Gravity is Memory", "Memory is Taste", "Taste is Love", "Love is Light"],
    microcosmic: ["The Mind", "Character", "Separation", "Nature", "Integration", "Spirit"],
    keywords: ["Evolution", "Gravity", "Memory", "Taste", "Love", "Light"],
    awarenessCenter: "Ajna",
    domain: "Character, integration, transgenerational consciousness"
  },
  {
    id: 3,
    name: "Being",
    voice: "I Am",
    question: "When?",
    sense: "Touch",
    binary: "Yin/Yin (Objective)",
    macrocosmic: ["Being is Matter", "Matter is Touch", "Touch is Sex", "Sex is Survival"],
    microcosmic: ["The Body", "Biology", "Chemistry", "Objectivity", "Geometry", "Trajectory"],
    keywords: ["Being", "Matter", "Touch", "Sex", "Survival"],
    awarenessCenter: "Splenic",
    domain: "Biology, chemistry, embodiment, genetics"
  },
  {
    id: 4,
    name: "Design",
    voice: "I Design",
    question: "Why?",
    sense: "Smell",
    binary: "Yin/Yang (Progressive)",
    macrocosmic: ["Design is Structure", "Structure is Progress", "Progress is Smell", "Smelling is Life", "Life is Art"],
    microcosmic: ["The Ego", "Homo Sapiens", "Growth", "Decay", "Continuity", "Manifestation"],
    keywords: ["Design", "Structure", "Progress", "Smell", "Life", "Art"],
    awarenessCenter: "Solar Plexus",
    domain: "Growth, continuity, decay, manifestation"
  },
  {
    id: 5,
    name: "Space",
    voice: "I Think",
    question: "Who?",
    sense: "Hearing",
    binary: "Subjective (Mutative)",
    macrocosmic: ["Space is Form", "Form is Illusion", "Illusion is Hearing", "Hearing is Music", "Music is Freedom"],
    microcosmic: ["Personality", "Type", "Fantasy", "Subjectivity", "Rhythm", "Timing"],
    keywords: ["Space", "Form", "Illusion", "Hearing", "Music", "Freedom"],
    awarenessCenter: "Throat",
    domain: "Fantasy, rhythm, subjectivity, timing"
  }
];

// ============================================================================
// THE 6 TONES (NATURE/PERCEPTION)
// ============================================================================
export interface Tone {
  id: number;
  name: string;
  theme: string;
  department: string;
  awarenessCenter: string;
  keywords: string[];
}

export const TONES: Tone[] = [
  {
    id: 1,
    name: "Security",
    theme: "Survival",
    department: "Smell",
    awarenessCenter: "Splenic",
    keywords: ["Safe", "Alert", "Instinct", "Protection", "Survival", "Fear-based awareness"]
  },
  {
    id: 2,
    name: "Uncertainty",
    theme: "Doubt/Exploration",
    department: "Taste",
    awarenessCenter: "Splenic",
    keywords: ["Questioning", "Exploring", "Tasting", "Discerning", "Curious", "Tentative"]
  },
  {
    id: 3,
    name: "Action",
    theme: "External Focus",
    department: "Outer Vision",
    awarenessCenter: "Ajna",
    keywords: ["Seeing", "External", "Objective", "Observing", "Active", "Outward"]
  },
  {
    id: 4,
    name: "Meditation",
    theme: "Internal Focus",
    department: "Inner Vision",
    awarenessCenter: "Ajna",
    keywords: ["Imagining", "Internal", "Contemplative", "Visionary", "Inward", "Reflective"]
  },
  {
    id: 5,
    name: "Judgement",
    theme: "Emotional Awareness",
    department: "Feeling",
    awarenessCenter: "Solar Plexus",
    keywords: ["Feeling", "Emotional", "Evaluating", "Judging", "Deciding", "Sensing"]
  },
  {
    id: 6,
    name: "Acceptance",
    theme: "Resonance/Touch",
    department: "Touch",
    awarenessCenter: "Solar Plexus",
    keywords: ["Listening", "Accepting", "Resonating", "Harmonizing", "Synthesizing", "All-encompassing"]
  }
];

// ============================================================================
// THE 6 COLORS (MOTIVATIONS)
// ============================================================================
export interface Color {
  id: number;
  name: string;
  modeBinary: [string, string];
  awarenessCenter: string;
  keywords: string[];
}

export const COLORS: Color[] = [
  {
    id: 1,
    name: "Fear",
    modeBinary: ["Communalist", "Separatist"],
    awarenessCenter: "Splenic",
    keywords: ["Survival", "Alert", "Cautious", "Protective", "Instinctive"]
  },
  {
    id: 2,
    name: "Hope",
    modeBinary: ["Theist", "Anti-theist"],
    awarenessCenter: "Splenic",
    keywords: ["Believing", "Trusting", "Optimistic", "Faithful", "Expecting"]
  },
  {
    id: 3,
    name: "Desire",
    modeBinary: ["Leader", "Follower"],
    awarenessCenter: "Ajna",
    keywords: ["Wanting", "Seeking", "Driving", "Pursuing", "Attracted"]
  },
  {
    id: 4,
    name: "Need",
    modeBinary: ["Master", "Novice"],
    awarenessCenter: "Solar Plexus",
    keywords: ["Requiring", "Necessary", "Essential", "Dependent", "Hungry"]
  },
  {
    id: 5,
    name: "Guilt",
    modeBinary: ["Conditioner", "Conditioned"],
    awarenessCenter: "Solar Plexus",
    keywords: ["Responsible", "Accountable", "Burdened", "Atoning", "Obligated"]
  },
  {
    id: 6,
    name: "Innocence",
    modeBinary: ["Observer", "Observed"],
    awarenessCenter: "Solar Plexus",
    keywords: ["Pure", "Witnessing", "Transparent", "Naive", "Open"]
  }
];

// ============================================================================
// THE 6 LINES (ROLES/ORBITS)
// ============================================================================
export interface Line {
  id: number;
  name: string;
  role: string;
  keywords: string[];
  theme: string;
}

export const LINES: Line[] = [
  {
    id: 1,
    name: "Foundation",
    role: "Investigator",
    theme: "Introspection",
    keywords: ["Foundation", "Study", "Research", "Authority", "Depth", "Certainty"]
  },
  {
    id: 2,
    name: "Natural",
    role: "Hermit",
    theme: "Natural Talent",
    keywords: ["Natural", "Genius", "Called", "Democratic", "Talent", "Recognition"]
  },
  {
    id: 3,
    name: "Experimentation",
    role: "Martyr",
    theme: "Trial and Error",
    keywords: ["Trial", "Error", "Adaptation", "Resilience", "Discovery", "Bonds broken"]
  },
  {
    id: 4,
    name: "Friendship",
    role: "Opportunist",
    theme: "Externalization",
    keywords: ["Network", "Influence", "Fixed", "Friendship", "Externalization", "Social"]
  },
  {
    id: 5,
    name: "Projection",
    role: "Heretic",
    theme: "Universal Solutions",
    keywords: ["Projection", "Savior", "General", "Universal", "Practical", "Paranoid"]
  },
  {
    id: 6,
    name: "Transition",
    role: "Role Model",
    theme: "Wisdom through Time",
    keywords: ["Transition", "Wisdom", "Objectivity", "Optimist", "Administrator", "Soul"]
  }
];

// ============================================================================
// ZODIAC SIGNS WITH KEYWORDS
// ============================================================================
export interface ZodiacSign {
  id: number;
  name: string;
  modality: "Cardinal" | "Fixed" | "Mutable";
  element: "Fire" | "Earth" | "Air" | "Water";
  keywords: string[];
  gateRange: [number, number];
}

export const ZODIAC_SIGNS: ZodiacSign[] = [
  { id: 1, name: "Aries", modality: "Cardinal", element: "Fire", keywords: ["Initiative", "Beginning", "Fire", "Cardinal"], gateRange: [25, 51] },
  { id: 2, name: "Taurus", modality: "Fixed", element: "Earth", keywords: ["Stability", "Form", "Earth", "Fixed"], gateRange: [2, 23] },
  { id: 3, name: "Gemini", modality: "Mutable", element: "Air", keywords: ["Communication", "Duality", "Air", "Mutable"], gateRange: [35, 45] },
  { id: 4, name: "Cancer", modality: "Cardinal", element: "Water", keywords: ["Nurturing", "Security", "Water", "Cardinal"], gateRange: [15, 52] },
  { id: 5, name: "Leo", modality: "Fixed", element: "Fire", keywords: ["Expression", "Creativity", "Fire", "Fixed"], gateRange: [7, 31] },
  { id: 6, name: "Virgo", modality: "Mutable", element: "Earth", keywords: ["Service", "Analysis", "Earth", "Mutable"], gateRange: [46, 18] },
  { id: 7, name: "Libra", modality: "Cardinal", element: "Air", keywords: ["Balance", "Harmony", "Air", "Cardinal"], gateRange: [48, 57] },
  { id: 8, name: "Scorpio", modality: "Fixed", element: "Water", keywords: ["Transformation", "Depth", "Water", "Fixed"], gateRange: [28, 44] },
  { id: 9, name: "Sagittarius", modality: "Mutable", element: "Fire", keywords: ["Expansion", "Philosophy", "Fire", "Mutable"], gateRange: [9, 5] },
  { id: 10, name: "Capricorn", modality: "Cardinal", element: "Earth", keywords: ["Structure", "Authority", "Earth", "Cardinal"], gateRange: [60, 41] },
  { id: 11, name: "Aquarius", modality: "Fixed", element: "Air", keywords: ["Innovation", "Disruption", "Air", "Fixed"], gateRange: [13, 49] },
  { id: 12, name: "Pisces", modality: "Mutable", element: "Water", keywords: ["Dissolution", "Compassion", "Water", "Mutable"], gateRange: [36, 22] }
];

// ============================================================================
// ZODIAC KEYWORDS LOOKUP
// ============================================================================
export const ZODIAC_KEYWORDS: Record<string, string[]> = {
  Aries: ["Initiative", "Beginning", "Fire", "Cardinal"],
  Taurus: ["Stability", "Form", "Earth", "Fixed"],
  Gemini: ["Communication", "Duality", "Air", "Mutable"],
  Cancer: ["Nurturing", "Security", "Water", "Cardinal"],
  Leo: ["Expression", "Creativity", "Fire", "Fixed"],
  Virgo: ["Service", "Analysis", "Earth", "Mutable"],
  Libra: ["Balance", "Harmony", "Air", "Cardinal"],
  Scorpio: ["Transformation", "Depth", "Water", "Fixed"],
  Sagittarius: ["Expansion", "Philosophy", "Fire", "Mutable"],
  Capricorn: ["Structure", "Authority", "Earth", "Cardinal"],
  Aquarius: ["Innovation", "Disruption", "Air", "Fixed"],
  Pisces: ["Dissolution", "Compassion", "Water", "Mutable"]
};

// ============================================================================
// HOUSES WITH CONTEXTS
// ============================================================================
export interface House {
  id: number;
  name: string;
  domain: string;
  keywords: string[];
  associatedSign: string;
}

export const HOUSES: House[] = [
  { id: 1, name: "First House", domain: "Identity and Self-Expression", keywords: ["Self", "Appearance", "Beginning", "Personality"], associatedSign: "Aries" },
  { id: 2, name: "Second House", domain: "Resources and Values", keywords: ["Possessions", "Values", "Money", "Self-worth"], associatedSign: "Taurus" },
  { id: 3, name: "Third House", domain: "Communication and Learning", keywords: ["Siblings", "Learning", "Short journeys", "Mind"], associatedSign: "Gemini" },
  { id: 4, name: "Fourth House", domain: "Home and Foundation", keywords: ["Home", "Family", "Roots", "Emotional security"], associatedSign: "Cancer" },
  { id: 5, name: "Fifth House", domain: "Creativity and Children", keywords: ["Creativity", "Romance", "Children", "Joy"], associatedSign: "Leo" },
  { id: 6, name: "Sixth House", domain: "Health and Service", keywords: ["Health", "Work", "Service", "Daily routine"], associatedSign: "Virgo" },
  { id: 7, name: "Seventh House", domain: "Relationships and Others", keywords: ["Partnership", "Marriage", "Others", "Balance"], associatedSign: "Libra" },
  { id: 8, name: "Eighth House", domain: "Transformation and Shared Resources", keywords: ["Transformation", "Death", "Shared resources", "Mystery"], associatedSign: "Scorpio" },
  { id: 9, name: "Ninth House", domain: "Philosophy and Higher Learning", keywords: ["Philosophy", "Travel", "Higher learning", "Truth"], associatedSign: "Sagittarius" },
  { id: 10, name: "Tenth House", domain: "Career and Public Image", keywords: ["Career", "Public image", "Achievement", "Authority"], associatedSign: "Capricorn" },
  { id: 11, name: "Eleventh House", domain: "Community and Ideals", keywords: ["Friends", "Groups", "Hopes", "Humanity"], associatedSign: "Aquarius" },
  { id: 12, name: "Twelfth House", domain: "Spirituality and Transcendence", keywords: ["Unconscious", "Secrets", "Dissolution", "Spirituality"], associatedSign: "Pisces" }
];

// ============================================================================
// HOUSE CONTEXTS LOOKUP
// ============================================================================
export const HOUSE_CONTEXTS: Record<number, string> = {
  1: "Identity and Self-Expression",
  2: "Resources and Values",
  3: "Communication and Learning",
  4: "Home and Foundation",
  5: "Creativity and Children",
  6: "Health and Service",
  7: "Relationships and Others",
  8: "Transformation and Shared Resources",
  9: "Philosophy and Higher Learning",
  10: "Career and Public Image",
  11: "Community and Ideals",
  12: "Spirituality and Transcendence"
};

// ============================================================================
// AXIS TYPES
// ============================================================================
export interface Axis {
  name: string;
  description: string;
  polarity: string;
}

export const AXES: Axis[] = [
  { name: "Vertical", description: "Heaven-Earth axis", polarity: "Above-Below" },
  { name: "Horizontal", description: "Self-Other axis", polarity: "Left-Right" }
];

// ============================================================================
// SYMBOLIC OPERATORS (from Ra's notation system)
// ============================================================================
export interface SymbolicOperator {
  symbol: string;
  name: string;
  function: string;
}

export const SYMBOLIC_OPERATORS: Record<string, SymbolicOperator> = {
  '•': { symbol: '•', name: "Singularity", function: "Pre-collapse seed, infinite potential" },
  '.': { symbol: '.', name: "Transitioner", function: "Step inward, descend to next chamber" },
  '°': { symbol: '°', name: "Collapse", function: "Anchor into coordinate (phase/degree)" },
  ':': { symbol: ':', name: "Portal", function: "Threshold, parallel chamber" },
  ';': { symbol: ';', name: "Fork", function: "Branch, divergent streams" },
  ',': { symbol: ',', name: "Breath", function: "Pause, collect fragments" },
  '–': { symbol: '–', name: "Current", function: "Span, wave flow" },
  '′': { symbol: '′', name: "Pulse", function: "Arcminute, heartbeat tick" },
  '″': { symbol: '″', name: "Flicker", function: "Arcsecond, micro-shimmer" },
  '/': { symbol: '/', name: "Blade", function: "Cut, divide, choice" },
  '\\': { symbol: '\\', name: "Escape", function: "Sideways exit" },
  '*': { symbol: '*', name: "Starburst", function: "Expansion, multiplication" },
  '=': { symbol: '=', name: "Mirror", function: "Collapse of two into one" },
  '→': { symbol: '→', name: "Vector", function: "Direction, energy flow" }
};

// ============================================================================
// GATE TO ZODIAC MAPPING (Human Design Mandala)
// ============================================================================
export const GATE_TO_ZODIAC: Record<number, string> = {
  1: "Leo", 2: "Taurus", 3: "Sagittarius", 4: "Aquarius", 5: "Sagittarius",
  6: "Pisces", 7: "Leo", 8: "Gemini", 9: "Sagittarius", 10: "Scorpio",
  11: "Aquarius", 12: "Gemini", 13: "Aquarius", 14: "Scorpio", 15: "Cancer",
  16: "Gemini", 17: "Aries", 18: "Virgo", 19: "Pisces", 20: "Gemini",
  21: "Aries", 22: "Pisces", 23: "Taurus", 24: "Aquarius", 25: "Aries",
  26: "Aries", 27: "Scorpio", 28: "Libra", 29: "Cancer", 30: "Pisces",
  31: "Leo", 32: "Virgo", 33: "Virgo", 34: "Gemini", 35: "Gemini",
  36: "Pisces", 37: "Pisces", 38: "Scorpio", 39: "Aquarius", 40: "Scorpio",
  41: "Capricorn", 42: "Aries", 43: "Taurus", 44: "Scorpio", 45: "Gemini",
  46: "Virgo", 47: "Virgo", 48: "Libra", 49: "Aquarius", 50: "Virgo",
  51: "Aries", 52: "Cancer", 53: "Cancer", 54: "Capricorn", 55: "Pisces",
  56: "Sagittarius", 57: "Libra", 58: "Capricorn", 59: "Scorpio", 60: "Capricorn",
  61: "Capricorn", 62: "Gemini", 63: "Aquarius", 64: "Aquarius"
};

// ============================================================================
// HUMAN DESIGN MANDALA: Gate order and precise tropical start degrees
// The HD wheel starts at Gate 41 at approximately 2° Capricorn (272° tropical)
// ============================================================================

// Precise tropical start degrees for each gate (from HD ephemeris tables)
// Each gate spans exactly 5.625° (360 / 64)
export const GATE_START_DEGREES: Record<number, number> = {
  41: 272.0000,  19: 277.6250,  13: 283.2500,  49: 288.8750,  30: 294.5000,  55: 300.1250,  37: 305.7500,  63: 311.3750,
  22: 317.0000,  36: 322.6250,  25: 328.2500,  17: 333.8750,  21: 339.5000,  51: 345.1250,  42: 350.7500,   3: 356.3750,
  27:   2.0000,  24:   7.6250,   2:  13.2500,  23:  18.8750,   8:  24.5000,  20:  30.1250,  16:  35.7500,  35:  41.3750,
  45:  47.0000,  12:  52.6250,  15:  58.2500,  52:  63.8750,  39:  69.5000,  53:  75.1250,  62:  80.7500,  56:  86.3750,
  31:  92.0000,  33:  97.6250,   7: 103.2500,   4: 108.8750,  29: 114.5000,  59: 120.1250,  40: 125.7500,  64: 131.3750,
  47: 137.0000,   6: 142.6250,  46: 148.2500,  18: 153.8750,  48: 159.5000,  57: 165.1250,  32: 170.7500,  50: 176.3750,
  28: 182.0000,  44: 187.6250,   1: 193.2500,  43: 198.8750,  14: 204.5000,  34: 210.1250,   9: 215.7500,   5: 221.3750,
  26: 227.0000,  11: 232.6250,  10: 238.2500,  58: 243.8750,  38: 249.5000,  54: 255.1250,  61: 260.7500,  60: 266.3750
};

// Gate order on the HD wheel (for reference and iteration)
export const HD_WHEEL_GATES: number[] = [
  41, 19, 13, 49, 30, 55, 37, 63, 22, 36, 25, 17, 21, 51, 42, 3,
  27, 24, 2, 23, 8, 20, 16, 35, 45, 12, 15, 52, 39, 53, 62, 56,
  31, 33, 7, 4, 29, 59, 40, 64, 47, 6, 46, 18, 48, 57, 32, 50,
  28, 44, 1, 43, 14, 34, 9, 5, 26, 11, 10, 58, 38, 54, 61, 60
];

// Create reverse lookup: gate number -> wheel index
export const GATE_TO_WHEEL_INDEX: Record<number, number> = {};
HD_WHEEL_GATES.forEach((gate, index) => {
  GATE_TO_WHEEL_INDEX[gate] = index;
});

// ============================================================================
// EXTENDED PLACEMENT (with all layers from ephemeris)
// ============================================================================
export interface ExtendedPlacement {
  planet: string;
  stream: "design" | "personality";
  gate: number;
  line: number;
  color: number;
  tone: number;
  base: number;
  degree: number;
  minute: number;
  second: number;
  zodiacSign: string;
  house: number;
  longitude: number;  // Original planetary longitude
  axis: string;
}

// ============================================================================
// RESONANCE SENTENCE OUTPUT
// ============================================================================
export interface ResonanceSentence {
  placement: ExtendedPlacement;
  baseVoice: string;
  toneName: string;
  colorMotivation: string;
  lineRole: string;
  centerKeywords: string[];
  zodiacFlavor: string;
  houseContext: string;
  astrologicalLens: string;
  latticePosition: number;
  completeSentence: string;
  generatedAt: string;
}

// ============================================================================
// EPHEMERIS CALIBRATION: Convert planetary longitude to all fractal sublayers
// ============================================================================

/**
 * Convert planetary longitude (0-360°) to Gate (1-64)
 * Uses precise gate start degrees from HD ephemeris tables
 */
export function longitudeToGate(longitude: number): number {
  const normalized = ((longitude % 360) + 360) % 360;
  
  // Find which gate this longitude falls into by checking each gate's range
  for (let i = 0; i < HD_WHEEL_GATES.length; i++) {
    const gate = HD_WHEEL_GATES[i];
    const gateStart = GATE_START_DEGREES[gate];
    const gateEnd = (gateStart + DEGREES_PER_GATE) % 360;
    
    // Handle wrap-around at 360/0 boundary
    if (gateStart < gateEnd) {
      if (normalized >= gateStart && normalized < gateEnd) {
        return gate;
      }
    } else {
      // Gate spans the 0° boundary (e.g., 356° to 2°)
      if (normalized >= gateStart || normalized < gateEnd) {
        return gate;
      }
    }
  }
  
  // Fallback (should not reach here with valid data)
  return 41;
}

/**
 * Get the position within a gate (0 to DEGREES_PER_GATE)
 */
function getPositionWithinGate(longitude: number, gate: number): number {
  const normalized = ((longitude % 360) + 360) % 360;
  const gateStart = GATE_START_DEGREES[gate];
  
  // Calculate offset from gate start, handling wrap-around
  let offset = normalized - gateStart;
  if (offset < 0) offset += 360;
  if (offset >= DEGREES_PER_GATE) offset -= 360;
  
  return Math.max(0, Math.min(DEGREES_PER_GATE - 0.0001, offset));
}

/**
 * Convert planetary longitude to Line (1-6)
 * The line is determined by the position within the gate
 */
export function longitudeToLine(longitude: number): number {
  const gate = longitudeToGate(longitude);
  const withinGate = getPositionWithinGate(longitude, gate);
  const lineIndex = Math.floor(withinGate / DEGREES_PER_LINE);
  return Math.min(6, Math.max(1, lineIndex + 1));
}

/**
 * Convert planetary longitude to Color (1-6)
 * The color is determined by the position within the line
 */
export function longitudeToColor(longitude: number): number {
  const gate = longitudeToGate(longitude);
  const withinGate = getPositionWithinGate(longitude, gate);
  const withinLine = withinGate % DEGREES_PER_LINE;
  const colorIndex = Math.floor(withinLine / DEGREES_PER_COLOR);
  return Math.min(6, Math.max(1, colorIndex + 1));
}

/**
 * Convert planetary longitude to Tone (1-6)
 * The tone is determined by the position within the color
 */
export function longitudeToTone(longitude: number): number {
  const gate = longitudeToGate(longitude);
  const withinGate = getPositionWithinGate(longitude, gate);
  const withinColor = withinGate % DEGREES_PER_COLOR;
  const toneIndex = Math.floor(withinColor / DEGREES_PER_TONE);
  return Math.min(6, Math.max(1, toneIndex + 1));
}

/**
 * Convert planetary longitude to Base (1-5)
 * The base is determined by the position within the tone
 */
export function longitudeToBase(longitude: number): number {
  const gate = longitudeToGate(longitude);
  const withinGate = getPositionWithinGate(longitude, gate);
  const withinTone = withinGate % DEGREES_PER_TONE;
  const baseIndex = Math.floor(withinTone / DEGREES_PER_BASE);
  return Math.min(5, Math.max(1, baseIndex + 1));
}

/**
 * Extract degree, minute, second from longitude
 */
export function longitudeToDMS(longitude: number): { degree: number; minute: number; second: number } {
  const normalized = ((longitude % 360) + 360) % 360;
  const degree = Math.floor(normalized);
  const remainderMinutes = (normalized - degree) * 60;
  const minute = Math.floor(remainderMinutes);
  const second = Math.floor((remainderMinutes - minute) * 60);
  return { degree, minute, second };
}

/**
 * Get zodiac sign from longitude (standard astrology: 0° = Aries)
 */
export function longitudeToZodiac(longitude: number): string {
  const normalized = ((longitude % 360) + 360) % 360;
  const signIndex = Math.floor(normalized / 30);
  return ZODIAC_SIGNS[signIndex]?.name || "Aries";
}

/**
 * Determine axis based on gate position in the mandala
 * Gates 1-32 on left (Yang), 33-64 on right (Yin)
 */
export function determineAxis(gate: number): string {
  return gate <= 32 ? "Vertical" : "Horizontal";
}

/**
 * Calculate house from gate position (simplified: gate mod 12)
 */
export function gateToHouse(gate: number): number {
  return ((gate - 1) % 12) + 1;
}

/**
 * Calculate unique lattice position from 69,120 combinations
 * 5 bases × 6 tones × 6 colors × 6 lines × 64 gates = 69,120
 * 
 * Lattice ordering (nested loops innermost to outermost):
 * gate (64) → line (6) → color (6) → tone (6) → base (5)
 * 
 * Formula: ((((base-1)*6 + (tone-1))*6 + (color-1))*6 + (line-1))*64 + (gate-1)
 * Which expands to:
 * (base-1)*13824 + (tone-1)*2304 + (color-1)*384 + (line-1)*64 + (gate-1)
 * 
 * This yields a 0-indexed position from 0 to 69,119
 * We return 1-indexed (1 to 69,120)
 */
export function calculateLatticePosition(gate: number, line: number, color: number, tone: number, base: number): number {
  // Validate inputs are in valid ranges
  const safeGate = Math.min(64, Math.max(1, gate));
  const safeLine = Math.min(6, Math.max(1, line));
  const safeColor = Math.min(6, Math.max(1, color));
  const safeTone = Math.min(6, Math.max(1, tone));
  const safeBase = Math.min(5, Math.max(1, base));
  
  // Calculate 0-indexed position
  const position = 
    (safeBase - 1) * 13824 +   // 6*6*6*64 = 13,824 positions per base
    (safeTone - 1) * 2304 +    // 6*6*64 = 2,304 positions per tone
    (safeColor - 1) * 384 +    // 6*64 = 384 positions per color
    (safeLine - 1) * 64 +      // 64 positions per line
    (safeGate - 1);            // Individual gate position
  
  return position + 1; // Return 1-indexed (1 to 69,120)
}

/**
 * Get line role name
 */
export function getLineRole(line: number): string {
  const roles: Record<number, string> = {
    1: "foundation",
    2: "natural",
    3: "experimental", 
    4: "opportunistic",
    5: "projective",
    6: "transcendent"
  };
  return roles[line] || "dynamic";
}

// ============================================================================
// CORE FUNCTION: Derive Extended Placement from Longitude
// ============================================================================

/**
 * Derive all fractal sublayers from planetary longitude
 * This is the core ephemeris calibration function
 */
export function deriveExtendedPlacement(
  planet: string,
  stream: "design" | "personality",
  longitude: number,
  houseOverride?: number
): ExtendedPlacement {
  const gate = longitudeToGate(longitude);
  const line = longitudeToLine(longitude);
  const color = longitudeToColor(longitude);
  const tone = longitudeToTone(longitude);
  const base = longitudeToBase(longitude);
  const { degree, minute, second } = longitudeToDMS(longitude);
  const zodiacSign = GATE_TO_ZODIAC[gate] || longitudeToZodiac(longitude);
  const axis = determineAxis(gate);
  const house = houseOverride || gateToHouse(gate);

  return {
    planet,
    stream,
    gate,
    line,
    color,
    tone,
    base,
    degree,
    minute,
    second,
    zodiacSign,
    house,
    longitude,
    axis
  };
}

/**
 * Calculate longitude from gate and line position on the HD wheel
 * Uses precise gate start degrees for accurate conversion
 * Returns the start of the line (not midpoint) for deterministic round-tripping
 */
export function gateLineToLongitude(gate: number, line: number): number {
  // Get the precise start degree for this gate
  const gateStart = GATE_START_DEGREES[gate] ?? 0;
  
  // Add the line offset (start of line, not midpoint, for deterministic round-tripping)
  const lineOffset = (line - 1) * DEGREES_PER_LINE;
  
  // Calculate tropical longitude
  const tropicalLongitude = (gateStart + lineOffset) % 360;
  
  return tropicalLongitude;
}

/**
 * Derive extended placement from basic gate/line (when longitude not available)
 * Optionally accepts pre-calculated color/tone/base values
 */
export function deriveExtendedPlacementFromGate(
  planet: string,
  stream: "design" | "personality",
  gate: number,
  line: number,
  providedColor?: number,
  providedTone?: number,
  providedBase?: number
): ExtendedPlacement {
  // Calculate synthetic longitude from gate/line position
  const longitude = gateLineToLongitude(gate, line);
  
  // Use provided values if available, otherwise derive from longitude
  const color = providedColor ?? longitudeToColor(longitude);
  const tone = providedTone ?? longitudeToTone(longitude);
  const base = providedBase ?? longitudeToBase(longitude);
  
  const { degree, minute, second } = longitudeToDMS(longitude);
  const zodiacSign = GATE_TO_ZODIAC[gate] || longitudeToZodiac(longitude);
  const axis = determineAxis(gate);
  const house = gateToHouse(gate);

  return {
    planet,
    stream,
    gate,
    line,
    color,
    tone,
    base,
    degree,
    minute,
    second,
    zodiacSign,
    house,
    longitude,
    axis
  };
}

// ============================================================================
// CORE FUNCTION: Generate Resonance Sentence
// ============================================================================

/**
 * Build the complete resonance sentence from an extended placement
 * Following Ra Uru Hu's clause structure:
 * Base→Tone→Center Keywords→Gate→Line→Color(with binary mode)→Degree→Axis→Zodiac→House
 */
export function generateResonanceSentence(placement: ExtendedPlacement): ResonanceSentence {
  const base = BASES[placement.base - 1] || BASES[0];
  const tone = TONES[placement.tone - 1] || TONES[0];
  const color = COLORS[placement.color - 1] || COLORS[0];
  const line = LINES[placement.line - 1] || LINES[0];
  const zodiac = ZODIAC_KEYWORDS[placement.zodiacSign] || ZODIAC_KEYWORDS["Aries"];
  const houseContext = HOUSE_CONTEXTS[placement.house] || HOUSE_CONTEXTS[1];
  
  const centerKeywords = base.keywords.slice(0, 3);
  const zodiacFlavor = zodiac.slice(0, 2).join(" and ");
  const lineRole = getLineRole(placement.line);
  
  // Determine color binary mode (odd lines = first mode, even lines = second mode)
  const colorMode = placement.line % 2 === 1 ? color.modeBinary[0] : color.modeBinary[1];
  
  // Build the astrological lens notation (Ra format)
  // gate.line.color.tone.base degree minute'second" axis zodiac house
  const astrologicalLens = `${placement.gate}.${placement.line}.${placement.color}.${placement.tone}.${placement.base} ${placement.degree}°${placement.minute}'${placement.second}" ${placement.axis} ${placement.zodiacSign} H${placement.house}`;
  
  // Calculate lattice position (unique identifier in 69,120 space)
  const latticePosition = calculateLatticePosition(
    placement.gate,
    placement.line,
    placement.color,
    placement.tone,
    placement.base
  );
  
  // Build the complete sentence following Ra Uru Hu clause structure:
  // 1. Base voice declaration
  // 2. Tone expression
  // 3. Base keywords (from the specific dimension)
  // 4. Gate and line role
  // 5. Color motivation with binary mode
  // 6. Degree crystallization
  // 7. Minute/second fractal refinement
  // 8. Axis alignment
  // 9. Zodiac flavor
  // 10. House contextualization
  const completeSentence = 
    `${base.voice}. ` +
    `At the Base of ${base.name}, ` +
    `expressed through the Tone of ${tone.name} (${tone.theme}), ` +
    `with keywords: ${centerKeywords.join(", ")}. ` +
    `Moving through Gate ${placement.gate} in its ${line.name} (${lineRole}) orbit, ` +
    `responding with ${color.name} as ${colorMode}. ` +
    `Crystallized at Degree ${placement.degree}, ` +
    `refined through ${placement.minute}'${placement.second}" fractal nuance, ` +
    `aligned to the ${placement.axis} Axis, ` +
    `flavored by ${placement.zodiacSign}'s ${zodiacFlavor.toLowerCase()}, ` +
    `contextualized in House ${placement.house} of ${houseContext.toLowerCase()}.`;

  return {
    placement,
    baseVoice: base.voice,
    toneName: tone.name,
    colorMotivation: `${color.name} as ${colorMode}`,
    lineRole: line.role,
    centerKeywords,
    zodiacFlavor,
    houseContext,
    astrologicalLens,
    latticePosition,
    completeSentence,
    generatedAt: new Date().toISOString()
  };
}

// ============================================================================
// HELPER: Apply symbolic operators for resonance variations
// ============================================================================
export function applySymbolicOperators(sentence: string, operatorSequence: string): string {
  let modified = sentence;
  
  for (const op of operatorSequence) {
    const operator = SYMBOLIC_OPERATORS[op];
    if (operator) {
      if (operator.function.includes("expansion")) {
        modified = modified + " *expanding outward*";
      } else if (operator.function.includes("pause")) {
        modified = modified.replace(/,/g, ", *pause*,");
      } else if (operator.function.includes("portal")) {
        modified = "*portal opens* " + modified;
      } else if (operator.function.includes("choice")) {
        modified = modified + " /at the crossroads/";
      }
    }
  }
  
  return modified;
}

// ============================================================================
// BATCH GENERATION: Generate sentences for all chart placements
// ============================================================================
export interface ChartResonance {
  placements: ResonanceSentence[];
  totalLatticeSpan: number;
  dominantBase: string;
  dominantTone: string;
  dominantColor: string;
}

export function generateChartResonance(
  placements: Array<{ planet: string; stream: "design" | "personality"; gate: number; line: number }>
): ChartResonance {
  const sentences = placements.map(p => {
    const extended = deriveExtendedPlacementFromGate(p.planet, p.stream, p.gate, p.line);
    return generateResonanceSentence(extended);
  });
  
  // Calculate dominance
  const baseCounts: Record<string, number> = {};
  const toneCounts: Record<string, number> = {};
  const colorCounts: Record<string, number> = {};
  
  sentences.forEach(s => {
    const base = BASES[s.placement.base - 1]?.name || "Unknown";
    const tone = TONES[s.placement.tone - 1]?.name || "Unknown";
    const color = COLORS[s.placement.color - 1]?.name || "Unknown";
    
    baseCounts[base] = (baseCounts[base] || 0) + 1;
    toneCounts[tone] = (toneCounts[tone] || 0) + 1;
    colorCounts[color] = (colorCounts[color] || 0) + 1;
  });
  
  const getDominant = (counts: Record<string, number>): string => {
    let max = 0;
    let dominant = "Unknown";
    for (const [key, count] of Object.entries(counts)) {
      if (count > max) {
        max = count;
        dominant = key;
      }
    }
    return dominant;
  };
  
  const latticePositions = sentences.map(s => s.latticePosition);
  const totalLatticeSpan = Math.max(...latticePositions) - Math.min(...latticePositions);
  
  return {
    placements: sentences,
    totalLatticeSpan,
    dominantBase: getDominant(baseCounts),
    dominantTone: getDominant(toneCounts),
    dominantColor: getDominant(colorCounts)
  };
}
