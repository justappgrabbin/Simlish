// B-H-A-N-T-U-G-H: her BANTU layers (binary, ASCII, NATO, transliteration, Unicode) plus the three she named on Oct 19:
// HTML, geometry, hexadecimal. REUSED code: layer functions are her dimension-visual core (binary, asciiOf, natoOf,
// translitOf, hex, letterColor, letterHz; byte-identical copy in ./originals/dimension-visual) and her Bantu 6-bit
// tokens (../bantu.mjs encodeBytes/decodeBytes/packsOf, verbatim from her bridge.js).
// INVESTIGATING: which H is HTML and which is hexadecimal. Both are kept: H1 = html, H2 = hex is ONE reading only.
// FILLED: html = numeric character references; hex = UTF-8 bytes in hex; geometry = her 6-bit token's 3-2-1 packs
// (the only shape-like structure in her code); "on a diagonal" = layer k is read at step (i + k) mod 8 for particle i;
// Movement/Design/Space distribution = DMS[k mod 3]. All of these are builder readings of her words.
import { binary, asciiOf, natoOf, translitOf, hex, letterColor, letterHz } from './originals/dimension-visual/dimension-visual-core.mjs';
import { encodeBytes, decodeBytes, packsOf } from '../bantu.mjs';

export const DMS = ['Movement', 'Design', 'Space'];
// registry: the primitive-functions.json entry each layer spells out (null = no entry yet, gap).
export const LAYERS = Object.freeze([
  { key: 'B', name: 'binary', registry: 'layer.binary', invertible: true },
  { key: 'H', name: 'html', registry: null, invertible: true, note: 'H assignment INVESTIGATING' },
  { key: 'A', name: 'ascii', registry: 'layer.ascii', invertible: false, note: 'her asciiOf is null above 127' },
  { key: 'N', name: 'nato', registry: 'layer.nato', invertible: false, note: 'letters only, case-free' },
  { key: 'T', name: 'translit', registry: 'layer.translit', invertible: false },
  { key: 'U', name: 'unicode', registry: 'layer.unicode', invertible: true },
  { key: 'G', name: 'geometry', registry: 'layer.six', invertible: true, note: 'geometry = 3-2-1 packs, FILLED' },
  { key: 'H', name: 'hex', registry: null, invertible: true, note: 'H assignment INVESTIGATING' },
].map((l, k) => Object.freeze({ ...l, index: k, quality: DMS[k % 3] })));

const enc = new TextEncoder(), dec = new TextDecoder('utf-8', { fatal: true });
const chars = (t) => Array.from(t);
export function encodeLayer(name, text) {
  const bytes = enc.encode(text);
  switch (name) {
    case 'binary': return [...bytes].map((b) => b.toString(2).padStart(8, '0')).join(' ');
    case 'html': return chars(text).map((c) => '&#x' + c.codePointAt(0).toString(16).toUpperCase() + ';').join('');
    case 'ascii': return chars(text).map((c) => { const a = asciiOf(c); return a ? String(a.dec) : '·'; }).join(' ');
    case 'nato': return chars(text).map((c) => natoOf(c)?.word ?? c).join(' ');
    case 'translit': return chars(text).map((c) => translitOf(c).russian?.[0]?.cyr ?? c).join(''); // her Russian table; letters it does not print stay as-is
    case 'unicode': return chars(text).map((c) => hex(c.codePointAt(0))).join(' ');
    case 'geometry': return encodeBytes(bytes).map((v) => packsOf(v).join('')).join(' ');
    case 'hex': return [...bytes].map((b) => b.toString(16).toUpperCase().padStart(2, '0')).join('');
  }
  throw new Error('unknown layer ' + name);
}
export function decodeLayer(name, s) {
  switch (name) {
    case 'binary': return dec.decode(Uint8Array.from(s.split(' ').filter(Boolean), (b) => parseInt(b, 2)));
    case 'html': return s.replace(/&#x([0-9A-F]+);/g, (_, h) => String.fromCodePoint(parseInt(h, 16)));
    case 'unicode': return s.split(' ').filter(Boolean).map((u) => String.fromCodePoint(parseInt(u.slice(2), 16))).join('');
    case 'geometry': return dec.decode(decodeBytes(s.split(' ').filter(Boolean).map((p) => (+p[0] << 4) | (+p[1] << 2) | +p[2])));
    case 'hex': return dec.decode(Uint8Array.from(s.match(/../g) || [], (h) => parseInt(h, 16)));
  }
  throw new Error(name + ' is not invertible in her definition');
}
// One particle through all eight layers, read on the diagonal.
export function encodeParticle(p, i = 0) {
  const layers = LAYERS.map((l) => ({ ...l, value: encodeLayer(l.name, p.text) }));
  const diagonal = LAYERS.map((_, k) => layers[(i + k) % 8].index);
  const cp = p.text.codePointAt(0) ?? 32;
  return { id: p.id, dimension: p.dimension, layers, diagonal, note: letterColor(cp), sound: letterHz(cp), binaryFirst: binary(p.text[0] ?? ' ') };
}
