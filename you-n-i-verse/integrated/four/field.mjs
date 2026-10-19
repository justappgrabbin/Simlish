// Five-layer particle field: the play world's ground. One layer per dimension (Movement, Evolution, Being, Design, Space),
// each streaming the WORDS of the real particles the four primitives emitted for that dimension (particles.json).
// Colour per glyph = her dimension-visual letterColor(codepoint) hue; brightness from letterHz (her PROTOTYPE formulas,
// marked in her core as provisional, not from her books). FILLED: cell size, scroll speeds (layer d moves d+1 cells/frame
// along a diagonal: row r is offset by r*(d+1)), first layer paints opaque cells (that is what makes 100% coverage),
// layers 2-5 add glyphs with 'lighter' compositing (the interference). Deterministic: frame number only, no Math.random.
import { letterColor, letterHz } from './originals/dimension-visual/dimension-visual-core.mjs';
export const FIELD_DIMS = ['Movement', 'Evolution', 'Being', 'Design', 'Space'];
export const CELL = { w: 7, h: 11, font: '11px ui-monospace,monospace' };
export function streams(particles) {
  const by = Object.fromEntries(FIELD_DIMS.map((d) => [d, []]));
  for (const p of particles) (by[p.dimension] ?? by.Space).push(...(p.words?.length ? p.words : [p.text]));
  return FIELD_DIMS.map((d) => { const s = Array.from(by[d].join(' ') + ' '); return s.length > 1 ? s : Array.from(d + ' '); });
}
const hsl = (cp, sat, light) => 'hsl(' + letterColor(cp).hue + ' ' + sat + '% ' + light + '%)';
const light = (cp, base) => base + Math.round((letterHz(cp).semi / 35) * 22);
export function drawField(ctx, W, H, S, frame = 0) {
  const cols = Math.ceil(W / CELL.w), rows = Math.ceil(H / CELL.h);
  ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1; ctx.font = CELL.font; ctx.textBaseline = 'top';
  for (let d = 0; d < S.length; d++) {
    const s = S[d], v = d + 1;
    if (d > 0) { ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha = 0.55; }
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const ch = s[(((r * cols + c + r * v + frame * v) % s.length) + s.length) % s.length], cp = ch.codePointAt(0);
      const x = c * CELL.w, y = r * CELL.h;
      if (d === 0) { ctx.fillStyle = hsl(cp, 40, 9 + (r + c + d) % 4); ctx.fillRect(x, y, CELL.w, CELL.h); }
      if (ch !== ' ') { ctx.fillStyle = hsl(cp, 70, light(cp, 34 + d * 4)); ctx.fillText(ch, x, y); }
    }
  }
  ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  return { cols, rows, layers: S.length };
}
// Coverage: fraction of pixels with alpha 255 (canvas starts fully transparent, so any 'background' pixel has alpha < 255).
export function coverage(rgba) { let n = 0; for (let i = 3; i < rgba.length; i += 4) if (rgba[i] === 255) n++; return n / (rgba.length / 4); }
export function mountGround(canvas, particles, { animate = true } = {}) {
  const S = streams(particles), ctx = canvas.getContext('2d'); let frame = 0, raf = 0;
  const paint = () => { const W = canvas.width = Math.max(1, innerWidth), H = canvas.height = Math.max(1, innerHeight); return drawField(ctx, W, H, S, frame); };
  const loop = () => { frame++; paint(); raf = setTimeout(loop, 250); }; // FILLED: 4 frames/s keeps a phone cool
  const info = paint(); if (animate) raf = setTimeout(loop, 250);
  return { info, streams: S, stop: () => clearTimeout(raf), measure: () => coverage(ctx.getImageData(0, 0, canvas.width, canvas.height).data), get frame() { return frame; } };
}
