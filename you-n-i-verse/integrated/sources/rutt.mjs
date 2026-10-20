// SOURCED: Richard Rutt, Zhouyi: The Book of Changes (Curzon, 1996), PDF in Alexis's Drive (ids 1OgSb-EljAqAhM5m-uNtcP-w_V5WhgZSy,
// 1uiKe_YkJApeEo6s_hU65-0C_ep86IBYL; box copy /workspace/scratch/drive/rutt.pdf sha256 cdfae3fb…). Page numbers are the printed pages.
import { bitsOf } from '../../state-space.mjs';
import { FUXI_DECIMAL_TO_KING_WEN as FX2KW, KING_WEN_TO_FUXI_DECIMAL as KW2FX } from '../../vendor/kingwen.mjs';
import { RUTT_TABLE_10 } from '../mawangdui-rutt.mjs';
export const RUTT = Object.freeze({ title: 'Zhouyi: The Book of Changes', author: 'Richard Rutt', drive: ['1OgSb-EljAqAhM5m-uNtcP-w_V5WhgZSy', '1uiKe_YkJApeEo6s_hU65-0C_ep86IBYL'] });

// pp. 88-93, Table 4: "whole lines are represented by 1 and broken lines by 0 ... the right-hand digit representing the bottom line".
// Fu Xi order = natural binary order. table4Binary(macro) gives Rutt's 6-digit string; the repo's bitsOf (line 1 = bit 0, padded) must equal it.
export const table4Binary = (lines /* bottom->top, 1 whole */) => lines.slice().reverse().join('');
export const linesOfMacro = (m) => [0, 1, 2, 3, 4, 5].map((b) => (m >> b) & 1);
export const repoMatchesTable4 = () => Array.from({ length: 64 }, (_, m) => table4Binary(linesOfMacro(m)) === bitsOf(m)).every(Boolean);
export const fuXiOrder = () => Array.from({ length: 64 }, (_, i) => i); // natural binary: Fu Xi position i = decimal i
// pp. 88-89 yin-yang segregation: the binary halving 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64; the line added at step k is line k from the bottom.
export const segregationLevel = (m, k /* 1..6 */) => m & ((1 << k) - 1);

// pp. 99-100: bagua numerals (Diagram 6): odd numerals = whole line, even = broken; written top line first.
export function decodeBaguaNumerals(s) {
  const d = String(s).split('').map(Number); if (d.length !== 6 || d.some((x) => !Number.isInteger(x))) throw new RangeError('six numerals');
  const linesTopFirst = d.map((x) => (x % 2 ? 1 : 0)), macro = linesTopFirst.reduce((m, b, i) => m | (b << (5 - i)), 0);
  return { macro, kingWen: FX2KW[macro] };
}
export const DIAGRAM_6_P99 = Object.freeze([{ numerals: '757666', zhouyi: 12 }, { numerals: '668116', zhouyi: 46 }, { numerals: '878785', zhouyi: 63 }, { numerals: '116616', zhouyi: 59 }, { numerals: '176786', zhouyi: 53 }]);

// pp. 115-117: Mawangdui principles (Diagram 9 / Table 9). Family roles (Shuo Gua): F Qian, M Kun, S1 Zhen, S2 Kan, S3 Gen, D1 Xun, D2 Li, D3 Dui.
export const ROLE_TRIGRAM = Object.freeze({ F: 7, M: 0, S1: 1, S2: 2, S3: 4, D1: 6, D2: 5, D3: 3 });
export const UPPER_ROLES = Object.freeze(['F', 'S3', 'S2', 'S1', 'M', 'D3', 'D2', 'D1']); // yang trigrams, then yin, each from the most senior
export const LOWER_ROLES = Object.freeze(['F', 'M', 'S3', 'D3', 'S2', 'D2', 'S1', 'D1']); // paired by seniority
// GENERATED: within each upper-trigram octet the doubled trigram comes first, then the lower roles in LOWER_ROLES order.
export function mawangduiGenerated() {
  const out = [];
  for (const u of UPPER_ROLES) for (const l of [u, ...LOWER_ROLES.filter((r) => r !== u)]) out.push((ROLE_TRIGRAM[u] << 3) | ROLE_TRIGRAM[l]);
  return Object.freeze(out);
}
// Table 9 (p. 116) as printed: row = upper role (UPPER_ROLES), column = lower role (LOWER_ROLES), cell = Mawangdui sequence number.
export const TABLE_9_P116 = Object.freeze([
  [1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 9, 12, 13, 14, 15, 16], [18, 19, 20, 21, 17, 22, 23, 24], [26, 27, 28, 29, 30, 31, 25, 32],
  [34, 33, 35, 36, 37, 38, 39, 40], [42, 43, 44, 41, 45, 46, 47, 48], [50, 51, 52, 53, 54, 49, 55, 56], [58, 59, 60, 61, 62, 63, 64, 57]]);
export function table9Macros() {
  const out = new Array(64);
  TABLE_9_P116.forEach((row, r) => row.forEach((n, c) => { out[n - 1] = (ROLE_TRIGRAM[UPPER_ROLES[r]] << 3) | ROLE_TRIGRAM[LOWER_ROLES[c]]; }));
  return Object.freeze(out);
}
export const TABLE_10_P117 = RUTT_TABLE_10; // Mawangdui position -> Zhouyi (King Wen) number, checked against the p.117 image
export const mawangduiGeneratedKingWen = () => mawangduiGenerated().map((m) => FX2KW[m]);
export { KW2FX };
