/** Exact King Wen gate -> bottom-first six-bit Fu Xi decimal lookup taken from
 * user's semantic-dna-swarm (1).html, src/state-space/addressing.js.
 * Google Drive file id 1VaDdlmpLpTFMaXEdPHZrUsVnGvSHCoU_.
 * Do NOT derive gate number by treating King Wen number as binary index.
 */
import {assertSix,lines} from './zhouyi-state.mjs';
export const KING_WEN_TO_FUXI_DECIMAL = Object.freeze([
  null,
  63,0,17,34,23,58,2,16,
  55,59,7,56,61,47,4,8,
  25,38,3,48,41,37,32,1,
  57,39,33,30,18,45,28,14,
  60,15,40,5,53,43,20,10,
  35,49,31,62,24,6,26,22,
  29,46,9,36,52,11,13,44,
  54,27,50,19,51,12,21,42
]);
const reverse=Array(64);
for(let gate=1;gate<=64;gate++){
 const six=KING_WEN_TO_FUXI_DECIMAL[gate];
 if(!Number.isInteger(six)||six<0||six>63||reverse[six]!==undefined) throw Error('nonbijective King Wen/Fu Xi reference');
 reverse[six]=gate;
}
export const FUXI_DECIMAL_TO_KING_WEN=Object.freeze(reverse);
export function bitsForGate(gate){if(!Number.isInteger(gate)||gate<1||gate>64)throw RangeError('gate 1..64');return lines(KING_WEN_TO_FUXI_DECIMAL[gate]);}
export function sixForGate(gate){if(!Number.isInteger(gate)||gate<1||gate>64)throw RangeError('gate 1..64');return KING_WEN_TO_FUXI_DECIMAL[gate];}
export function gateForSix(six){return FUXI_DECIMAL_TO_KING_WEN[assertSix(six)];}
export function unicodeForGate(gate){sixForGate(gate);return String.fromCodePoint(0x4DC0+gate-1);}
