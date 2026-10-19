// Video generator client — interface + offline ProceduralGenerator.
//
// API SEAM (Runway/Kling/Wan): implement the GeneratorClient interface —
//   async generate(sceneDNA) -> { kind, path, bytes, engine }
// and construct with config { endpoint, apiKey, model }. No real API keys are
// available in this repo, so the default path is ProceduralGenerator, which is
// fully functional offline: it renders an SVG keyframe storyboard plus a JSON
// "scene DNA" record under data/oracles/scenes/.

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { seededRng } from '../util/hash.js';

/** Interface: external generators (Runway/Kling/Wan) must extend this. */
export class GeneratorClient {
  constructor(config = {}) {
    this.config = config; // {endpoint, apiKey, model} for API clients
  }
  async generate(_sceneDNA) {
    throw new Error('GeneratorClient.generate: not implemented — use ProceduralGenerator or an API client');
  }
}

const ELEMENT_COLORS = {
  wood: '#3f7d3a', fire: '#c0392b', earth: '#8d6e3f', metal: '#b0b7bf', water: '#2e5f8a',
};

/** Offline default: deterministic SVG keyframe storyboard + scene DNA JSON. */
export class ProceduralGenerator extends GeneratorClient {
  constructor(config = {}) {
    super(config);
    if (!config.scenesDir) throw new Error('ProceduralGenerator: config.scenesDir required');
    this.scenesDir = config.scenesDir;
    mkdirSync(this.scenesDir, { recursive: true });
  }

  /**
   * @param {object} sceneDNA - {hash, setting, elements, lighting, mood, camera, motion, prompt, band, effectParams, leadElement}
   * @returns {{kind:'svg-storyboard', path, dnaPath, bytes, engine}}
   */
  async generate(sceneDNA) {
    if (!sceneDNA?.hash || !sceneDNA?.prompt) throw new Error('ProceduralGenerator.generate: sceneDNA with hash and prompt required');
    const rng = seededRng(`scene-render:${sceneDNA.hash}`);
    const svg = this._storyboardSvg(sceneDNA, rng);
    const path = join(this.scenesDir, `${sceneDNA.hash}.svg`);
    const dnaPath = join(this.scenesDir, `${sceneDNA.hash}.dna.json`);
    writeFileSync(path, svg);
    writeFileSync(dnaPath, JSON.stringify(sceneDNA, null, 2));
    return { kind: 'svg-storyboard', path, dnaPath, bytes: Buffer.byteLength(svg), engine: 'procedural-svg-v1' };
  }

  _storyboardSvg(dna, rng) {
    const W = 960, H = 540, panels = 3;
    const pw = W / panels;
    const fx = dna.effectParams ?? {};
    const base = ELEMENT_COLORS[dna.leadElement] ?? '#555';
    const frames = [];
    for (let p = 0; p < panels; p++) {
      const shapes = [];
      for (const el of dna.elements ?? []) {
        const cx = p * pw + 40 + rng() * (pw - 80);
        const cy = 120 + rng() * 300;
        const r = 18 + rng() * 40;
        const color = ELEMENT_COLORS[el.element] ?? base;
        const opacity = 0.5 + rng() * 0.5;
        shapes.push(`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}" fill="${color}" opacity="${opacity.toFixed(2)}"><title>${el.name}</title></circle>`);
      }
      // scan lines overlay for distortion bands (spec §4.4: count = |R|*100)
      if (fx.scanLines?.enabled && fx.scanLines.count > 0) {
        const spacing = Math.max(2, Math.round(H / fx.scanLines.count));
        for (let y = 0; y < H; y += spacing) {
          shapes.push(`<rect x="${p * pw}" y="${y}" width="${pw}" height="1" fill="#000" opacity="${fx.scanLines.opacity}"/>`);
        }
      }
      if ((fx.noiseOpacity ?? 0) > 0.05) {
        shapes.push(`<rect x="${p * pw}" y="0" width="${pw}" height="${H}" fill="#777" opacity="${fx.noiseOpacity}"/>`);
      }
      frames.push(`<g><rect x="${p * pw}" y="0" width="${pw}" height="${H}" fill="${shade(base, p * 0.08)}" stroke="#222"/>${shapes.join('')}<text x="${p * pw + 12}" y="24" fill="#eee" font-size="14">keyframe ${p + 1} — ${escapeXml(dna.camera ?? '')}</text></g>`);
    }
    const aberr = fx.chromaticAberrationPx ?? 0;
    const aberration = aberr > 0
      ? `<rect width="${W}" height="${H}" fill="none" stroke="#f00" stroke-width="${aberr}" opacity="0.25" transform="translate(${aberr},0)"/><rect width="${W}" height="${H}" fill="none" stroke="#0ff" stroke-width="${aberr}" opacity="0.25" transform="translate(${-aberr},0)"/>`
      : '';
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">` +
      `<desc>${escapeXml(dna.prompt)}</desc>${frames.join('')}${aberration}` +
      `<text x="12" y="${H - 12}" fill="#ccc" font-size="12">${escapeXml(dna.setting ?? '')} | band: ${escapeXml(dna.band ?? '')} | hash: ${dna.hash}</text></svg>`;
  }
}

function shade(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const f = (c) => Math.max(0, Math.min(255, Math.round(c * (1 - amt))));
  return `#${((f(n >> 16) << 16) | (f((n >> 8) & 255) << 8) | f(n & 255)).toString(16).padStart(6, '0')}`;
}
function escapeXml(s) {
  return String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}
