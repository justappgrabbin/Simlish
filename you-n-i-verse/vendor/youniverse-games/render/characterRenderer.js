// CharacterRenderer — renders a MorphState to an actual SVG artifact at
// data/renders/{playerId}-{embodimentId}.svg. All shapes/colors are DERIVED
// from the visual identity palette/fingerprint + morph state, so different
// embodiments visibly differ.

import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const CENTER_Y = { head: 60, ajna: 85, throat: 110, g: 145, heart: 150, sacral: 185, solarPlexus: 190, spleen: 195, root: 230 };
const CENTER_X = { head: 0, ajna: 0, throat: 0, g: 0, heart: 30, sacral: 0, solarPlexus: -32, spleen: 32, root: 0 };

export class CharacterRenderer {
  constructor(dataDir) {
    this.dir = join(dataDir, 'renders');
    mkdirSync(this.dir, { recursive: true });
  }

  render(morphState) {
    if (!morphState?.visualState || !morphState?.baseIdentity) {
      throw new Error('CharacterRenderer.render: morphState with visualState required');
    }
    const vs = morphState.visualState;
    const W = 400, H = Math.round(420 * vs.scale);
    const cx = W / 2;
    const scale = vs.scale;

    const headR = 90 * vs.proportions.headRatio * scale + 12;
    const torsoH = 160 * vs.proportions.torsoRatio * scale + 40;
    const limbL = 140 * vs.proportions.limbRatio * scale + 30;
    const bodyW = { slim: 40, human: 55, broad: 80, small: 42, bent: 60, radiant: 70, fluid: 62, shifted: 50 }[vs.bodyForm] ?? 55;

    const parts = [];
    parts.push(`<rect width="${W}" height="${H + 60}" fill="#101018"/>`);
    // Aura derived from palette + amplification
    if (vs.auraStrength > 0.1) {
      parts.push(`<ellipse cx="${cx}" cy="${H / 2}" rx="${120 * scale + 60 * vs.auraStrength}" ry="${H / 2 * 0.9}" fill="${vs.auraColor}" opacity="${(vs.auraStrength * 0.35).toFixed(2)}"/>`);
    }
    // Body silhouette
    parts.push(`<circle cx="${cx}" cy="${90 * scale + 30}" r="${headR}" fill="${vs.bodyColor}"/>`);
    parts.push(`<rect x="${cx - bodyW}" y="${90 * scale + 30 + headR}" width="${bodyW * 2}" height="${torsoH}" rx="${bodyW / 2}" fill="${vs.bodyColor}"/>`);
    parts.push(`<line x1="${cx - bodyW}" y1="${90 * scale + 40 + headR + torsoH}" x2="${cx - bodyW - 20}" y2="${90 * scale + 40 + headR + torsoH + limbL}" stroke="${vs.accentColor}" stroke-width="14" stroke-linecap="round"/>`);
    parts.push(`<line x1="${cx + bodyW}" y1="${90 * scale + 40 + headR + torsoH}" x2="${cx + bodyW + 20}" y2="${90 * scale + 40 + headR + torsoH + limbL}" stroke="${vs.accentColor}" stroke-width="14" stroke-linecap="round"/>`);
    // Glowing centers (HD amplification made visible)
    for (const gc of vs.glowingCenters) {
      const y = (CENTER_Y[gc.center] ?? 150) * scale + 30;
      const x = cx + (CENTER_X[gc.center] ?? 0);
      parts.push(`<circle cx="${x}" cy="${y}" r="${10 + 14 * gc.intensity}" fill="${gc.color}" opacity="${(0.4 + 0.6 * gc.intensity).toFixed(2)}"/>`);
      parts.push(`<text x="${x + 18}" y="${y + 4}" fill="#ddd" font-size="11">${gc.center}</text>`);
    }
    // Gate marks orbiting the figure — one per active gate, angle from fingerprint-derived state
    for (const gm of vs.gateMarks) {
      const rad = (gm.angle * Math.PI) / 180;
      const gx = cx + Math.cos(rad) * 150 * scale;
      const gy = H / 2 + Math.sin(rad) * 150 * scale;
      parts.push(`<circle cx="${gx.toFixed(1)}" cy="${gy.toFixed(1)}" r="9" fill="none" stroke="${vs.accentColor}" stroke-width="2"/>`);
      parts.push(`<text x="${gx - 8}" y="${gy + 4}" fill="${vs.accentColor}" font-size="10">${gm.gate}.${gm.line}</text>`);
    }
    parts.push(`<text x="14" y="${H + 30}" fill="#888" font-size="13">${morphState.kind} · ${morphState.narrativeRole} · ${morphState.baseIdentity.hdType} ${morphState.baseIdentity.profile} · fp ${morphState.baseIdentity.fingerprint.slice(0, 10)}</text>`);

    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H + 60}">${parts.join('\n')}</svg>`;
    const filename = `${morphState.playerId}-${morphState.embodimentId}.svg`;
    const path = join(this.dir, filename);
    writeFileSync(path, svg);
    morphState.resultingAsset = path;
    return { path, svg, bytes: Buffer.byteLength(svg) };
  }
}
