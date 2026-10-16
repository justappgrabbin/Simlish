// Pure Synthia Automata — state-space constants (wheel math, scales, dimensions, trigrams, PRNG)

// Wheel precision constants (user's DMS system, spec §2)
export const WHEEL_ARCSECONDS = 1296000; // 360° in arc-seconds
export const GATE_ARCSECONDS = 20250;    // 5°37'30" per gate (64 gates)
export const LINE_ARCSECONDS = 3375;     // 0°56'15" per line (6 lines/gate)
export const COLOR_ARCSECONDS = 562.5;   // 6 colors/line
export const TONE_ARCSECONDS = 93.75;    // 6 tones/color
export const BASE_ARCSECONDS = 18.75;    // 5 bases/tone
export const ZODIAC_ARCSECONDS = 108000; // 30° per sign (12 signs)
export const HOUSE_ARCSECONDS = 162000;  // 45° per trigram house (8 houses)

// Scale ladder L0→L8+ (spec §1); index = level
export const SCALES = ['feature','phoneme','grapheme','morpheme','word','phrase','clause','sentence','discourse','automaton','mesh'];

// Five dimensional perspectives (spec §3); D3 Being is the hinge/witness
export const DIMENSIONS = ['Movement','Evolution','Being','Design','Space'];
export const DIMENSION_META = {
  Movement: { interrogative:'Where', operation:'transition',  seedGate:1,  octave:2 },
  Evolution:{ interrogative:'What',  operation:'transform',   seedGate:2,  octave:3 },
  Being:    { interrogative:'When',  operation:'instantiate', seedGate:6,  octave:4 },
  Design:   { interrogative:'Why',   operation:'structure',   seedGate:14, octave:5 },
  Space:    { interrogative:'Who',   operation:'integrate',   seedGate:20, octave:6 },
};

// Fu Xi order, bits bottom-to-top (line 1 first), value = binary (bit0 = bottom line)
export const TRIGRAMS = [
  { id:'kun',  name:'Receptive',   element:'Earth',    bits:[0,0,0], value:0 },
  { id:'zhen', name:'Arousing',    element:'Thunder',  bits:[1,0,0], value:1 },
  { id:'kan',  name:'Abysmal',     element:'Water',    bits:[0,1,0], value:2 },
  { id:'dui',  name:'Joyous',      element:'Lake',     bits:[1,1,0], value:3 },
  { id:'gen',  name:'KeepingStill',element:'Mountain', bits:[0,0,1], value:4 },
  { id:'li',   name:'Clinging',    element:'Fire',     bits:[1,0,1], value:5 },
  { id:'xun',  name:'Gentle',      element:'Wind',     bits:[0,1,1], value:6 },
  { id:'qian', name:'Creative',    element:'Heaven',   bits:[1,1,1], value:7 },
];

// Deterministic seeded PRNG factory; returns () => [0,1)
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
