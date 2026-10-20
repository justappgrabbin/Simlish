// SOURCED: Lama Anagarika Govinda, The Inner Structure of the I Ching (Wheelwright Press), PDF in Alexis's Drive
// "Lama Anagarika Govinda - The Inner Structure of the I Ching - The Book of Transformations.pdf" (id 1unb8il81Anv1REcSCdrpEzMEyRvgYwGP,
// box copy /workspace/scratch/drive/govinda.pdf sha256 a2e34d4e…). Printed page n = PDF page n+20. Every value below is copied from the
// page cited; nothing is added. Where the book does not state a link, the function says INVESTIGATING.
export const GOVINDA = Object.freeze({ title: 'The Inner Structure of the I Ching', author: 'Lama Anagarika Govinda', drive: '1unb8il81Anv1REcSCdrpEzMEyRvgYwGP', pdfOffset: 20 });

// Trigrams: lines bottom->top, whole=1 (p. ix drawings); value with line 1 = bit 0 (same convention as state-space.mjs).
export const TRIGRAMS = Object.freeze(['KIAN', 'KUN', 'JEN', 'SUN', 'LI', 'KAN', 'DUI', 'GEN']); // p. ix row order
export const TRIGRAM_VALUE = Object.freeze({ KIAN: 7, KUN: 0, JEN: 1, SUN: 6, LI: 5, KAN: 2, DUI: 3, GEN: 4 });
export const trigramOfValue = (v) => TRIGRAMS.find((t) => TRIGRAM_VALUE[t] === v);

// p. ix TRANSLITERATION table. ABBREV glyphs are the closest Unicode characters to the printed signs (read from the page image).
export const TRANSLITERATION_P_IX = Object.freeze({
  KIAN: { wilhelm: 'KIĀN', baynes: "CH'IEN", wordSymbol: 'Heaven', abbrev: '○', ideogram: '乾' },
  KUN: { wilhelm: 'KUN', baynes: "K'UN", wordSymbol: 'Earth', abbrev: '□', ideogram: '坤' },
  JEN: { wilhelm: 'JEN', baynes: 'CHÊN', wordSymbol: 'Thunder', abbrev: '∪', ideogram: '震' },
  SUN: { wilhelm: 'SUN', baynes: 'SUN', wordSymbol: 'Wind', abbrev: '⌅', ideogram: '巽' },
  LI: { wilhelm: 'LI', baynes: 'LI', wordSymbol: 'Fire', abbrev: '△', ideogram: '離' },
  KAN: { wilhelm: 'KAN', baynes: "K'AN", wordSymbol: 'Water', abbrev: '▽', ideogram: '坎' },
  DUI: { wilhelm: 'DUI', baynes: 'TUI', wordSymbol: 'Lake', abbrev: '⊻', ideogram: '兌' },
  GEN: { wilhelm: 'GEN', baynes: 'KÊN', wordSymbol: 'Mountain', abbrev: '∩', ideogram: '艮' },
});
export const trigramShape = (t) => TRANSLITERATION_P_IX[t].abbrev;

// pp. 46-47 INTERPRETIVE MEANINGS OF TRIGRAMS ON DIFFERENT PLANES. One row = one plane. Columns KIAN KUN JEN SUN | LI KAN DUI GEN.
const C = (...v) => Object.freeze(Object.fromEntries(['KIAN', 'KUN', 'JEN', 'SUN', 'LI', 'KAN', 'DUI', 'GEN'].map((t, i) => [t, v[i]])));
export const PLANES_PP46_47 = Object.freeze({
  names: C('KIAN', 'KUN', 'JEN', 'SUN', 'LI', 'KAN', 'DUI', 'GEN'),
  images: C('HEAVEN', 'EARTH', 'THUNDER', 'AIR, WIND, WOOD', 'FIRE', 'WATER', 'LAKE, MIST', 'MOUNTAIN'),
  yangYin: C('YANG', 'YIN', 'YANG', 'YIN', 'YANG', 'YIN', 'YANG', 'YIN'),
  relations: C('FATHER', 'MOTHER', '1st SON', '1st DAUGHTER', '2nd DAUGHTER', '2nd SON', '3rd DAUGHTER', '3rd SON'),
  movement: C('(+A) up', '(-A) down', '(+B) up', '(-B) down', '(+C) up', '(-C) down', '(+D) up', '(-D) down'),
  generalPrinciples: C(['CREATIVE', 'ACTIVITY', 'STRENGTH'], ['RECEPTIVE', 'PASSIVITY', 'SURRENDER'], ['EXCITING', 'IMPETUS', 'STIMULATION'], ['PENETRATING', 'SENSITIVITY', 'RESPONSIVENESS'], ['BRIGHT', 'THE FORMED'], ['DARK', 'THE FORMLESS'], ['REFLECTING BUOYANT', 'LIGHTNESS'], ['STEADY', 'HEAVINESS']),
  psychologicalQualities: C(['TIME-EXPERIENCE', 'DURATION'], ['SPACE-EXPERIENCE', 'EXTENSION'], ['VOLITION', 'IMPULSE'], ['INTUITION', 'ASSIMILATION'], ['CLEARNESS', 'DISCRIMINATION', 'LOGOS'], ['UNCERTAINTY', 'EMOTION', 'EROS'], ['GAIETY', 'OBSERVATION', 'INTUITIVE VISION'], ['QUIETNESS', 'EQUANIMITY', 'CONCENTRATION']),
  generalQualities: C('IMMATERIAL', 'MATERIAL', 'VITALITY', 'PERVASIVENESS', 'SOLAR FORCES', 'LUNAR FORCES', 'VOLATILITY EVAPORATION', 'SOLIDITY INERTIA'),
  statesOfBeing: C('UNIVERSAL', 'UNIVERSAL', 'ORGANIC', 'ORGANIC', 'ELEMENTARY', 'ELEMENTARY', 'INORGANIC', 'INORGANIC'), // printed spanning two columns
  statesOfAggregation: C('FORMLESSNESS', 'FORM', 'MOBILITY', 'GASEOUSNESS', 'INCANDESCENCE', 'FLUIDITY', 'EVAPORATION', 'INERTIA'),
  qualitiesOfNature: C('ENERGY', 'MATTER', 'FERTILISATION', 'GROWTH', 'HEAT', 'COLD', 'CHANGEABILITY', 'RESISTANCE'),
  positionTimeSpace: C({ hour: 21, direction: 'NORTH-WEST' }, { hour: 15, direction: 'SOUTH-WEST' }, { hour: 6, direction: 'EAST', season: 'SPRING' }, { hour: 9, direction: 'SOUTH-EAST' }, { hour: 12, direction: 'SOUTH', season: 'SUMMER' }, { hour: 24, direction: 'NORTH', season: 'WINTER' }, { hour: 18, direction: 'WEST', season: 'AUTUMN' }, { hour: 3, direction: 'NORTH-EAST' }),
  shuoGuaSequence: C(6, 4, 1, 2, 3, 7, 5, 8), // Shuo Gua ch. II "No."
  cycleOfLife: C(['ENDEAVOUR', 'STRUGGLE'], ['DEVOTION', 'SERVING'], ['BEGINNING', 'OF MOVEMENT'], ['MATURING', 'RIPENING'], ['VISUALITY', 'COGNITION'], ['PAINS', 'EXERTION'], ['HAPPINESS, SERENITY'], ['PERFECTION', 'COMPLETION']),
  principlesAndReactions: C(['CREATIVE', 'CONQUERING'], ['RECEPTIVE', 'SURRENDERING'], ['EXCITING', 'MOTION'], ['PENETRATING', 'SOFT'], ['CLINGING', 'DEPENDENCE'], ['ABYSMAL (DEEP)', 'DANGEROUS'], ['GAYNESS', 'JOY'], ['KEEPING QUIET', 'STANDING STILL']), // Shuo Gua ch. III
  partsOfBody: C('HEAD', 'BELLY', 'FOOT', 'THIGHS', 'EYE', 'EAR', 'MOUTH', 'HAND'),
  functions: C('AWARENESS', 'DIGESTION', 'MOBILITY', 'RECEPTIVITY', 'VISIBILITY', 'SPACE', 'NOURISHMENT', 'RELIABILITY'),
});
export const PLANES = Object.freeze(Object.keys(PLANES_PP46_47));
// p. 42: "We cannot mix up the meaning that applies to the material level with a meaning of a psychological condition ...
// We must always try to remain in the same category or on the same level". One call = one plane; asking for two throws.
export const PLANE_RULE_P42 = 'We must always try to remain in the same category or on the same level, in order not to deceive ourselves with semantic speculations.';
export function trigramQuality(trigram, plane, ...more) {
  if (more.length) throw new Error('Govinda p.42: stay on one plane; got ' + [plane, ...more].join(', '));
  const row = PLANES_PP46_47[plane]; if (!row) throw new RangeError('no plane ' + plane + ' on pp.46-47');
  if (!(trigram in row)) throw new RangeError('no trigram ' + trigram); return row[trigram];
}
// A hexagram is read through its upper (lines 4-6) and lower (lines 1-3) trigrams on ONE plane.
export function hexagramQuality(macro, plane, ...more) {
  if (more.length) throw new Error('Govinda p.42: stay on one plane');
  const upper = trigramOfValue((macro >> 3) & 7), lower = trigramOfValue(macro & 7);
  return { plane, upper: { trigram: upper, value: trigramQuality(upper, plane) }, lower: { trigram: lower, value: trigramQuality(lower, plane) } };
}
// p. 43 groups: a = yang, b = yin, letters written top line first (verified: organic bba = JEN, whose bottom line is yang).
export const GROUPS_P43 = Object.freeze({ universal: ['aaa', 'bbb'], organic: ['bba', 'aab'], elemental: ['aba', 'bab'], inorganic: ['baa', 'abb'] });
export const abOf = (t) => { const v = TRIGRAM_VALUE[t]; return [2, 1, 0].map((b) => ((v >> b) & 1 ? 'a' : 'b')).join(''); };
export const groupOf = (t) => Object.keys(GROUPS_P43).find((g) => GROUPS_P43[g].includes(abOf(t)));
// p. 44: the two polarities: complement (each line changed) and reversal (lines mirrored).
export const complementTrigram = (t) => trigramOfValue((~TRIGRAM_VALUE[t]) & 7);
export const reverseTrigram = (t) => { const v = TRIGRAM_VALUE[t]; return trigramOfValue(((v & 1) << 2) | (v & 2) | ((v >> 2) & 1)); };

// p. 44 and p. 57 (ELEMENTS AND THEIR CORRESPONDENCES): five elements, as printed.
export const ELEMENTS = Object.freeze(['WOOD', 'FIRE', 'EARTH', 'IRON', 'WATER']);
export const ELEMENTS_P57 = Object.freeze({
  WOOD: { evolution: 'Producing Fire', dissolution: 'Breaks up Earth', place: 'East', season: 'Spring', time: 'Morning', condition: 'Wind', virtue: 'Love', organ: 'Stomach, Spleen', taste: 'Sour', colour: 'Green', planet: 'Jupiter' },
  FIRE: { evolution: 'Producing Earth (ashes)', dissolution: 'Melts Iron', place: 'South', season: 'Summer', time: 'Noon', condition: 'Heat', virtue: 'Virtuousness', organ: 'Lungs', taste: 'Bitter', colour: 'Red', planet: 'Mars' },
  EARTH: { evolution: 'Producing Iron', dissolution: 'Absorbs Water', place: 'Center', season: 'Last 18 days of year', time: null, condition: 'Humidity', virtue: 'Faithfulness', organ: 'Heart', taste: 'Sweet', colour: 'Yellow', planet: 'Saturn' },
  IRON: { evolution: 'Producing Liquid', dissolution: 'Cuts Wood', place: 'West', season: 'Autumn', time: 'Evening', condition: 'Dryness', virtue: 'Justice', organ: 'Liver', taste: 'Hot', colour: 'White', planet: 'Venus' },
  WATER: { evolution: 'Producing Wood', dissolution: 'Extinguishes Fire', place: 'North', season: 'Winter', time: 'Night', condition: 'Cold', virtue: 'Wisdom', organ: 'Kidneys', taste: 'Salty', colour: '(Blue) Black', planet: 'Mercury' },
});
// generating ("mother of", p. 44: Fire->Earth->Iron->Water->Wood->Fire) and destroying (p. 44: Wood breaks Earth, Earth absorbs Water, Water extinguishes Fire, Fire melts Iron, Iron cuts Wood)
export const GENERATES = Object.freeze({ FIRE: 'EARTH', EARTH: 'IRON', IRON: 'WATER', WATER: 'WOOD', WOOD: 'FIRE' });
export const DESTROYS = Object.freeze({ WOOD: 'EARTH', EARTH: 'WATER', WATER: 'FIRE', FIRE: 'IRON', IRON: 'WOOD' });
// p. 53 (Sino-Tibetan tortoise, colour print): animal panels coloured by element: Fire red, Earth yellow, Iron white, Water blue, Wood green.
// The colour NAMES are sourced (p.53 print, p.57 table). The RGB values are FILLED: a plain rendering of each name, not measured from the print.
export const ELEMENT_RGB_FILLED = Object.freeze({ FIRE: [204, 51, 51], EARTH: [236, 196, 40], IRON: [238, 238, 238], WATER: [36, 120, 196], WOOD: [70, 150, 60] });
// Five tones: p. 51 says there are five tones of classical Chinese music; NO tone names or element mapping are printed. -> INVESTIGATING.
export const FIVE_TONES_P51 = Object.freeze({ count: 5, names: null, mapping: null, status: 'INVESTIGATING', quote: 'five tones in the scale of classical Chinese music' });
// pp. 51-52: twelve animals, element, direction. Order: the p.53 ring clockwise from the top (South).
export const ANIMALS_PP51_53 = Object.freeze([
  { animal: 'Serpent', element: 'FIRE', direction: 'South' }, { animal: 'Horse', element: 'FIRE', direction: 'South' },
  { animal: 'Sheep', element: 'EARTH', direction: 'South-West' }, { animal: 'Monkey', element: 'IRON', direction: 'West' }, { animal: 'Bird', element: 'IRON', direction: 'West' },
  { animal: 'Dog', element: 'EARTH', direction: 'North-West' }, { animal: 'Boar', element: 'WATER', direction: 'North' }, { animal: 'Mouse', element: 'WATER', direction: 'North' },
  { animal: 'Bull', element: 'EARTH', direction: 'North-East' }, { animal: 'Tiger', element: 'WOOD', direction: 'East' }, { animal: 'Hare', element: 'WOOD', direction: 'East' },
  { animal: 'Dragon', element: 'EARTH', direction: 'South-East' },
]);
// p. 52: "If we combine the five elements with the twelve animals, we arrive at a cycle of sixty years (5 X 12 = 60)".
// The year-by-year pairing is on the pp. 58-59 charts, which are not machine-readable here -> order INVESTIGATING.
export const SIXTY_CYCLE_P52 = Object.freeze({ length: ELEMENTS.length * 12, yearOrder: 'INVESTIGATING (pp. 58-59 charts)' });
// Trigram -> element: the book does not state it as a table. Derived only where a trigram and an element share a printed direction
// (trigram directions pp. 46-47, element places p. 57). The four intercardinal trigrams have no element place -> INVESTIGATING.
const DIR = { EAST: 'East', SOUTH: 'South', WEST: 'West', NORTH: 'North' };
export function trigramElement(t) {
  const d = PLANES_PP46_47.positionTimeSpace[t].direction, p = DIR[d];
  const e = p && ELEMENTS.find((x) => ELEMENTS_P57[x].place === p);
  return e ? { element: e, status: 'DERIVED', via: d + ' (pp.46-47) = ' + p + ' (p.57)' } : { element: null, status: 'INVESTIGATING', via: d + ': no element is placed at an intercardinal direction on p.57' };
}

// p. 66 (ch. IX), quoted from the Book of Transformations — a phrase only, no rule is derived from it.
export const DURATION_P66 = 'DURATION is a condition whose movement is not exhausted by hindrances. It is not a state of rest (in the sense of motionlessness), because a mere standstill is regression! DURATION, therefore, is a self-renewing movement of an organized and integrated whole, which proceeds in conformity with unalterable laws.';

// p. 67 TRANSFORMATION FROM THE ABSTRACT TO THE TEMPORAL ORDER (first version). MOVEMENT dimension.
// Changes as printed: KIAN->LI, LI->JEN, JEN->GEN, GEN->KIAN (KIAN-related); KUN->KAN, KAN->DUI, DUI->SUN, SUN->KUN (KUN-related).
export const FU_HI_TO_KING_WEN_P67 = Object.freeze({ KIAN: 'LI', LI: 'JEN', JEN: 'GEN', GEN: 'KIAN', KUN: 'KAN', KAN: 'DUI', DUI: 'SUN', SUN: 'KUN' });
export const fuHiToKingWen = (t) => FU_HI_TO_KING_WEN_P67[t];
// King Wen (outer circle) positions are the pp.46-47 directions. A Fu Hi (inner) position is the place where that trigram stands before
// it "changes into" the King Wen trigram at the same place (read off the p.67 diagram; it agrees with the change list above).
export const KING_WEN_PLACE = Object.freeze(Object.fromEntries(TRIGRAMS.map((t) => [t, PLANES_PP46_47.positionTimeSpace[t].direction])));
export const FU_HI_PLACE = Object.freeze(Object.fromEntries(TRIGRAMS.map((t) => [t, KING_WEN_PLACE[fuHiToKingWen(t)]])));
// p. 67 movement statements, verbatim, with the printed pairs. "KIAN and KAN" and "LI and KAN" both name KAN: flagged, not corrected.
export const MOVEMENT_STATEMENTS_P67 = Object.freeze([
  { printed: 'KIAN and KAN move diagonally in divergent directions.', pair: ['KIAN', 'KAN'], kind: 'diagonal-divergent', flag: 'KAN is also named in the next sentence' },
  { printed: 'LI and KAN move parallel in opposite directions.', pair: ['LI', 'KAN'], kind: 'parallel-opposite' },
  { printed: 'SUN and GEN move parallel in equal directions.', pair: ['SUN', 'GEN'], kind: 'parallel-equal' },
  { printed: 'JEN and DUI move unrelated in different directions.', pair: ['JEN', 'DUI'], kind: 'unrelated' },
]);
// Vector of each trigram's move on the p.67 diagram (South at the top, West at the right, as drawn). Geometry only, for checking the statements.
const XY = { SOUTH: [0, 1], 'SOUTH-WEST': [0.7071, 0.7071], WEST: [1, 0], 'NORTH-WEST': [0.7071, -0.7071], NORTH: [0, -1], 'NORTH-EAST': [-0.7071, -0.7071], EAST: [-1, 0], 'SOUTH-EAST': [-0.7071, 0.7071] };
export function movementVector(t) { const a = XY[FU_HI_PLACE[t]], b = XY[KING_WEN_PLACE[t]]; return [b[0] - a[0], b[1] - a[1]]; }
export function classifyPair(t1, t2) {
  const [a, b] = [movementVector(t1), movementVector(t2)], cross = a[0] * b[1] - a[1] * b[0], dot = a[0] * b[0] + a[1] * b[1], n = Math.hypot(...a) * Math.hypot(...b);
  if (Math.abs(cross) / n < 1e-6) return dot > 0 ? 'parallel-equal' : 'parallel-opposite';
  const mirror = Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] + b[1]) < 1e-6; return mirror ? 'diagonal-divergent' : 'unrelated';
}

// INHERITED COLOUR RULE (Alexis 2026-10-20, Govinda pp.57-60): every member of an element group takes that element's ONE colour.
export const NEUTRAL_RGB = Object.freeze([128, 128, 128]); // 'unattributed': no element group is known
export const elementColour = (element) => (element ? ELEMENT_RGB_FILLED[element] : NEUTRAL_RGB);
// Members per element, as printed p.57 / p.60 (animals, season, direction, time, condition, virtue, organ, taste, planet).
export const ELEMENT_MEMBERS_P57_60 = Object.freeze(Object.fromEntries(ELEMENTS.map((e) => [e, Object.freeze([
  ...ANIMALS_PP51_53.filter((a) => a.element === e).map((a) => 'animal:' + a.animal),
  ...['place', 'season', 'time', 'condition', 'virtue', 'organ', 'taste', 'planet'].filter((k) => ELEMENTS_P57[e][k]).map((k) => k + ':' + ELEMENTS_P57[e][k]),
  ...TRIGRAMS.filter((t) => trigramElement(t).element === e).map((t) => 'trigram:' + t)])])));
export function memberColour(member) { const e = ELEMENTS.find((x) => ELEMENT_MEMBERS_P57_60[x].includes(member)); return { element: e || null, rgb: elementColour(e || null), status: e ? 'INHERITED' : 'unattributed' }; }
// Hexagram / glyph: inherits only when BOTH its trigrams are in the same element group (DERIVED); otherwise 'unattributed'.
export function hexagramElement(macro) { const a = trigramElement(trigramOfValue((macro >> 3) & 7)).element, b = trigramElement(trigramOfValue(macro & 7)).element; return a && a === b ? a : null; }
// p.152 PROJECTION OF ALL HEXAGRAMS ON ONE CIRCLE ACCORDING TO THE ABSTRACT ORDER (MOVEMENT): "Each line represents two movements.
// For instance: 31/41 = DUI-GEN and GEN-DUI ... The numbers outside the circle represent the reduplication of the same hexagram".
import { FUXI_DECIMAL_TO_KING_WEN as _FX2KW } from '../../vendor/kingwen.mjs';
export function abstractCircleLine(macro) {
  const u = trigramOfValue((macro >> 3) & 7), l = trigramOfValue(macro & 7), rev = (TRIGRAM_VALUE[l] << 3) | TRIGRAM_VALUE[u];
  if (u === l) return { outsideCircle: true, trigrams: [u, l], kingWen: [_FX2KW[macro]], note: 'reduplication: an inner movement that cannot be shown graphically' };
  return { outsideCircle: false, line: [u, l].sort().join('/'), movements: [u + '-' + l, l + '-' + u], kingWen: [_FX2KW[macro], _FX2KW[rev]] };
}
