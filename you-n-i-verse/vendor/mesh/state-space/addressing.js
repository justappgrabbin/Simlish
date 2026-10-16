// Pure Synthia Automata — canonical DMS address codec + Fu Xi / King Wen gate bit mapping

import {
  WHEEL_ARCSECONDS, GATE_ARCSECONDS, LINE_ARCSECONDS,
  COLOR_ARCSECONDS, TONE_ARCSECONDS, BASE_ARCSECONDS,
  ZODIAC_ARCSECONDS, HOUSE_ARCSECONDS,
} from './constants.js';

// King Wen gate number (1-64) -> Fu Xi decimal (0-63).
// Computed from the standard King Wen hexagram line patterns: gate n's six lines
// bottom-to-top become bits with line 1 (bottom) as bit 0 (yang=1, yin=0).
// Anchors: 1 ䷀ = 111111 = 63 · 2 ䷁ = 000000 = 0 · 3 ䷂ = bits[1,0,0,0,1,0] = 17.
export const KING_WEN_TO_FUXI_DECIMAL = Object.freeze({
  1:63,  2:0,   3:17,  4:34,  5:23,  6:58,  7:2,   8:16,
  9:55,  10:59, 11:7,  12:56, 13:61, 14:47, 15:4,  16:8,
  17:25, 18:38, 19:3,  20:48, 21:41, 22:37, 23:32, 24:1,
  25:57, 26:39, 27:33, 28:30, 29:18, 30:45, 31:28, 32:14,
  33:60, 34:15, 35:40, 36:5,  37:53, 38:43, 39:20, 40:10,
  41:35, 42:49, 43:31, 44:62, 45:24, 46:6,  47:26, 48:22,
  49:29, 50:46, 51:9,  52:36, 53:52, 54:11, 55:13, 56:44,
  57:54, 58:27, 59:50, 60:19, 61:51, 62:12, 63:21, 64:42,
});

const FUXI_DECIMAL_TO_KING_WEN = Object.freeze(
  Object.fromEntries(Object.entries(KING_WEN_TO_FUXI_DECIMAL).map(([g, d]) => [d, Number(g)]))
);

const mod = (n, m) => ((n % m) + m) % m;

// arc-seconds on the wheel -> canonical address (spec §2)
export function addressForArcSec(arcSec) {
  const a = mod(arcSec, WHEEL_ARCSECONDS);
  const gate = Math.floor(a / GATE_ARCSECONDS) + 1;                    // 1-64
  const withinGate = a - (gate - 1) * GATE_ARCSECONDS;
  const line = Math.floor(withinGate / LINE_ARCSECONDS) + 1;           // 1-6
  const withinLine = withinGate - (line - 1) * LINE_ARCSECONDS;
  const color = Math.floor(withinLine / COLOR_ARCSECONDS) + 1;         // 1-6
  const withinColor = withinLine - (color - 1) * COLOR_ARCSECONDS;
  const tone = Math.floor(withinColor / TONE_ARCSECONDS) + 1;          // 1-6
  const withinTone = withinColor - (tone - 1) * TONE_ARCSECONDS;
  const base = Math.floor(withinTone / BASE_ARCSECONDS) + 1;           // 1-5
  const degree = Math.floor(a / 3600);
  const minute = Math.floor((a % 3600) / 60);
  const second = a % 60;
  const zodiac = Math.floor(a / ZODIAC_ARCSECONDS) + 1;                // 1-12
  const house = Math.floor(a / HOUSE_ARCSECONDS) + 1;                  // 1-8
  return {
    gate, line, color, tone, base,
    degree, minute, second,
    arcSecond: a, zodiac, house,
  };
}

// canonical address (gate/line/color/tone/base sufficient) -> arc-seconds; inverse of above
export function arcSecForAddress(addr) {
  const { gate = 1, line = 1, color = 1, tone = 1, base = 1 } = addr;
  return mod(
    (gate - 1) * GATE_ARCSECONDS +
    (line - 1) * LINE_ARCSECONDS +
    (color - 1) * COLOR_ARCSECONDS +
    (tone - 1) * TONE_ARCSECONDS +
    (base - 1) * BASE_ARCSECONDS,
    WHEEL_ARCSECONDS
  );
}

// King Wen gate (1-64) -> 6 Fu Xi binary bits, line 1 (bottom) first
export function gateBits(gate) {
  if (!Number.isInteger(gate) || gate < 1 || gate > 64) {
    throw new RangeError(`gate must be 1-64, got ${gate}`);
  }
  const d = KING_WEN_TO_FUXI_DECIMAL[gate];
  return [0, 1, 2, 3, 4, 5].map((i) => (d >> i) & 1);
}

// 6 bits (line 1 first) -> King Wen gate number
export function gateFromBits(bits) {
  if (!Array.isArray(bits) || bits.length !== 6) {
    throw new RangeError('gateFromBits expects a 6-bit array (line 1 first)');
  }
  const d = bits.reduce((acc, b, i) => acc | ((b ? 1 : 0) << i), 0);
  return FUXI_DECIMAL_TO_KING_WEN[d];
}

// Hamming distance between two 6-bit arrays (0-6)
export function hamming(a, b) {
  if (a.length !== b.length) throw new RangeError('hamming: bit arrays must have equal length');
  return a.reduce((n, bit, i) => n + (bit !== b[i] ? 1 : 0), 0);
}

// compact canonical key, e.g. "G12.L3.C4.T2.B5"
export function addrKey(addr) {
  return `G${addr.gate}.L${addr.line}.C${addr.color}.T${addr.tone}.B${addr.base}`;
}
