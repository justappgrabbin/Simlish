// Five-layer particle field: the play world's ground. One layer per dimension (Movement, Evolution, Being, Design, Space),
// each streaming the WORDS of the real particles her four primitives emitted for that dimension (particles.json).
// Every glyph quality (shape, texture, hue, brightness, sound...) comes from qualities.mjs; see ATTRIBUTION.md.
// Motion: the diagonal is in Movement ONLY (Alexis, 2026-10-20); other layers run straight.
// Deterministic: frame number only, no Math.random.
import { glyphQualities } from './qualities.mjs';
import { isDiagonal } from './bhantugh.mjs';
export const FIELD_DIMS = ['Movement', 'Evolution', 'Being', 'Design', 'Space'];
export const CELL = { w: 7, h: 12, font: '11px ui-monospace,monospace' }; // FILLED sizes
export function streams(particles) {
  const by = Object.fromEntries(FIELD_DIMS.map((d) => [d, []]));
  for (const p of particles) { const L = by[p.dimension] ?? by.Space; for (const w of (p.words?.length ? p.words : [p.text])) for (const ch of Array.from(w + ' ')) L.push(glyphQualities(ch, p.address)); }
  return FIELD_DIMS.map((d) => by[d].length ? by[d] : Array.from(d + ' ', (ch) => glyphQualities(ch, null)));
}
// Which stream position a cell shows. Movement: row r is shifted by r*v (a diagonal that moves); others: straight rows.
export function cellIndex(dim, r, c, cols, frame, len) { const v = FIELD_DIMS.indexOf(dim) + 1; const i = r * cols + c + (isDiagonal(dim) ? r * v : 0) + frame * v; return ((i % len) + len) % len; }
export const motionOf = (dim) => (isDiagonal(dim) ? 'diagonal' : 'straight');
// texture: Govinda p. ix trigram signs (upper over lower), coloured by element where the book places one (qualities.mjs ATTRIBUTION)
function trigramSigns(ctx, x, y, q) { ctx.font = '6px ui-monospace,monospace'; for (const [k, dy] of [['upper', 0], ['lower', 6]]) { const t = q.trigrams[k]; ctx.fillStyle = 'rgba(' + t.rgb.join(',') + ',0.45)'; ctx.fillText(t.shape, x, y + dy); } }
export function drawField(ctx, W, H, S, frame = 0) {
  const cols = Math.ceil(W / CELL.w), rows = Math.ceil(H / CELL.h);
  ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1; ctx.font = CELL.font; ctx.textBaseline = 'top';
  for (let d = 0; d < S.length; d++) {
    const s = S[d], dim = FIELD_DIMS[d];
    if (d > 0) { ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha = 0.55; }
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const q = s[cellIndex(dim, r, c, cols, frame, s.length)], x = c * CELL.w, y = r * CELL.h;
      if (d === 0) { ctx.fillStyle = '#101010'; ctx.fillRect(x, y, CELL.w, CELL.h); trigramSigns(ctx, x, y, q); ctx.font = CELL.font; }
      if (q.ch !== ' ') { ctx.fillStyle = 'rgb(' + q.rgb.join(',') + ')'; ctx.fillText(q.ch, x, y); }
    }
  }
  ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  return { cols, rows, layers: S.length };
}
// Sound: one sine voice per layer at the frequency of the glyph in that layer's top-left cell. MUTED by default.
export function soundVoices(S, frame, cols) { return S.map((s, d) => s[cellIndex(FIELD_DIMS[d], 0, 0, cols, frame, s.length)].hz); }
export function createSound(AudioCtx = globalThis.AudioContext) {
  let ac = null, osc = [], gain = null; const st = { muted: true };
  st.unmute = () => { if (!AudioCtx) return false; ac ??= new AudioCtx(); gain ??= Object.assign(ac.createGain(), {}); gain.gain.value = 0.03; gain.connect(ac.destination); st.muted = false; return true; };
  st.mute = () => { if (gain) gain.gain.value = 0; st.muted = true; };
  st.set = (hzs) => { if (!ac) return; while (osc.length < hzs.length) { const o = ac.createOscillator(); o.type = 'sine'; o.connect(gain); o.start(); osc.push(o); } hzs.forEach((h, i) => { osc[i].frequency.value = Math.min(20000, Math.max(20, h)); }); };
  return st;
}
export function coverage(rgba) { let n = 0; for (let i = 3; i < rgba.length; i += 4) if (rgba[i] === 255) n++; return n / (rgba.length / 4); }
export function mountGround(canvas, particles, { animate = true } = {}) {
  const S = streams(particles), ctx = canvas.getContext('2d'), sound = createSound(); let frame = 0, t = 0, info;
  const paint = () => { const W = canvas.width = Math.max(1, innerWidth), H = canvas.height = Math.max(1, innerHeight); info = drawField(ctx, W, H, S, frame); if (!sound.muted) sound.set(soundVoices(S, frame, info.cols)); return info; };
  const loop = () => { frame++; paint(); t = setTimeout(loop, 250); }; // FILLED: 4 frames/s
  paint(); if (animate) t = setTimeout(loop, 250);
  return { get info() { return info; }, streams: S, sound, stop: () => clearTimeout(t), measure: () => coverage(ctx.getImageData(0, 0, canvas.width, canvas.height).data), get frame() { return frame; } };
}
