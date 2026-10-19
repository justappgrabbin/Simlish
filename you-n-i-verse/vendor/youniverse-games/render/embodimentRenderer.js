// render/embodimentRenderer.js — Embodiment renderer v2.
// Draws an EmbodimentSpec as SVG where the person's ACTUAL extracted anchors
// drive the drawing: eye spacing ratio, face proportions, skin/hair palette,
// expression signature as mouth/brow geometry, lineage marks as persistent
// symbols — mapped through species morphology. Same person across species
// shares anchor geometry (embedded in data-* metadata); different species
// visibly differ. Deterministic; errors throw visibly.

import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const hex = (rgb) => '#' + rgb.map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('');
const darken = (rgb, f = 0.55) => rgb.map((v) => Math.round(v * f));
const r2 = (v) => +v.toFixed(2);

// Species-specific skull/feature geometry drawn around the anchor-driven core.
function speciesFeatures(species, cx, cy, rx, ry, jawW, mouthYAbs, skinHex, hairHex) {
  const parts = [];
  switch (species) {
    case 'feline':
      parts.push(`<polygon points="${cx - rx * 0.7},${cy - ry * 0.75} ${cx - rx * 0.35},${cy - ry * 1.45} ${cx - rx * 0.05},${cy - ry * 0.8}" fill="${hairHex}"/>`);
      parts.push(`<polygon points="${cx + rx * 0.7},${cy - ry * 0.75} ${cx + rx * 0.35},${cy - ry * 1.45} ${cx + rx * 0.05},${cy - ry * 0.8}" fill="${hairHex}"/>`);
      for (const s of [-1, 1]) {
        for (let w = 0; w < 3; w++) {
          parts.push(`<line x1="${cx + s * jawW * 0.6}" y1="${mouthYAbs - 6 + w * 6}" x2="${cx + s * (jawW * 0.6 + rx * 0.9)}" y2="${mouthYAbs - 14 + w * 10}" stroke="#3c3c3c" stroke-width="1.5"/>`);
        }
      }
      break;
    case 'avian':
      parts.push(`<polygon points="${cx - jawW * 0.35},${mouthYAbs - 12} ${cx + jawW * 0.35},${mouthYAbs - 12} ${cx},${mouthYAbs + 22}" fill="#d8a040"/>`);
      parts.push(`<polygon points="${cx - 6},${cy - ry * 0.9} ${cx + 6},${cy - ry * 0.9} ${cx},${cy - ry * 1.5}" fill="${hairHex}"/>`);
      break;
    case 'aquatic':
      parts.push(`<polygon points="${cx},${cy - ry * 0.95} ${cx - 10},${cy - ry * 1.5} ${cx + 10},${cy - ry * 1.5}" fill="${hairHex}"/>`);
      for (const s of [-1, 1]) for (let g = 0; g < 3; g++) {
        parts.push(`<line x1="${cx + s * rx * 0.8}" y1="${cy + ry * 0.1 + g * 8}" x2="${cx + s * rx * 0.55}" y2="${cy + ry * 0.15 + g * 8}" stroke="${hairHex}" stroke-width="2"/>`);
      }
      break;
    case 'terrane':
      for (let k = 0; k < 4; k++) {
        parts.push(`<ellipse cx="${cx}" cy="${cy}" rx="${rx * (0.4 + k * 0.18)}" ry="${ry * (0.4 + k * 0.18)}" fill="none" stroke="${hairHex}" stroke-width="1" opacity="0.5"/>`);
      }
      break;
    case 'synthetic':
      for (const s of [-1, 1]) {
        parts.push(`<polyline points="${cx + s * rx * 0.3},${cy - ry * 0.6} ${cx + s * rx * 0.7},${cy - ry * 0.3} ${cx + s * rx * 0.5},${cy + ry * 0.2}" fill="none" stroke="#7fd4ff" stroke-width="1.5" opacity="0.8"/>`);
      }
      parts.push(`<line x1="${cx}" y1="${cy - ry}" x2="${cx}" y2="${cy + ry}" stroke="#7fd4ff" stroke-width="1" opacity="0.4"/>`);
      break;
    case 'hybrid':
      parts.push(`<polygon points="${cx - rx * 0.6},${cy - ry * 0.8} ${cx - rx * 0.3},${cy - ry * 1.3} ${cx - rx * 0.05},${cy - ry * 0.85}" fill="${hairHex}"/>`);
      for (const s of [1]) for (let g = 0; g < 2; g++) {
        parts.push(`<line x1="${cx + s * rx * 0.8}" y1="${cy + ry * 0.1 + g * 8}" x2="${cx + s * rx * 0.55}" y2="${cy + ry * 0.15 + g * 8}" stroke="${hairHex}" stroke-width="2"/>`);
      }
      parts.push(`<polyline points="${cx - rx * 0.3},${cy + ry * 0.4} ${cx},${cy + ry * 0.55} ${cx + rx * 0.3},${cy + ry * 0.4}" fill="none" stroke="#7fd4ff" stroke-width="1.5" opacity="0.8"/>`);
      break;
    default:
      throw new Error(`EmbodimentRenderer: no feature geometry for species "${species}"`);
  }
  return parts;
}

export class EmbodimentRenderer {
  constructor(dataDir) {
    this.dir = join(dataDir, 'renders');
    mkdirSync(this.dir, { recursive: true });
  }

  /**
   * Render an EmbodimentSpec to SVG.
   * @param {object} spec - from kleinOperators.runKleinChain
   * @param {object} [opts] - {playerId, embodimentId, persist=true}
   */
  render(spec, opts = {}) {
    if (!spec?.anchors || !spec?.anchorsAfter || !spec?.geometry) {
      throw new Error('EmbodimentRenderer.render: EmbodimentSpec with anchors/anchorsAfter/geometry required');
    }
    const A = spec.anchorsAfter;        // rendered (transformed) anchors
    const S = spec.anchors;             // preserved source anchors (invariants)
    const W = 420, H = 480;
    const cx = W / 2, cy = 200;

    // Skull geometry: species skull x face ratio (the person's own proportions).
    const skull = spec.geometry.skull;
    const baseR = 95;
    const rx = baseR * skull.rx * (0.7 + A.faceRatio * 0.3);
    const ry = baseR * skull.ry;
    const jawW = rx * skull.jawTaper;

    const skinHex = hex(A.skinTone);
    const hairHex = hex(A.hairTone);
    const eyeHex = '#1a1410';

    // Eyes: person's eye-spacing ratio + eye height, through species placement.
    const eyeDx = (A.eyeSpacingRatio * 2 * rx) / 2;
    const eyeYAbs = cy - ry + 2 * ry * A.eyeY;
    const pupilR = 7 + 6 * A.expression.browContrast;

    // Mouth from expression signature: darkness -> stroke weight, curve -> arc.
    const mouthYAbs = cy - ry + 2 * ry * A.mouthY;
    const mouthW = jawW * 1.1;
    const curvePx = -A.expression.mouthCurve * 60; // positive curve => upturned corners
    const mouthW8 = 2 + 6 * A.expression.mouthDarkness;
    const browYAbs = eyeYAbs - 16 - 10 * A.expression.browContrast;

    const parts = [];
    parts.push(`<rect width="${W}" height="${H}" fill="#0d1017"/>`);
    // biome aura tint
    parts.push(`<ellipse cx="${cx}" cy="${cy}" rx="${rx + 70}" ry="${ry + 80}" fill="${hex(spec.anchorsAfter.skinTone)}" opacity="0.08"/>`);
    // skull + jaw
    parts.push(`<ellipse cx="${cx}" cy="${cy}" rx="${r2(rx)}" ry="${r2(ry)}" fill="${skinHex}"/>`);
    parts.push(`<ellipse cx="${cx}" cy="${r2(mouthYAbs)}" rx="${r2(jawW)}" ry="${r2(ry * 0.45)}" fill="${skinHex}"/>`);
    // hair cap from the person's hair tone
    parts.push(`<path d="M ${cx - rx} ${cy - ry * 0.35} A ${r2(rx)} ${r2(ry)} 0 0 1 ${cx + rx} ${cy - ry * 0.35} L ${cx + rx * 0.8} ${cy - ry * 0.75} A ${r2(rx * 0.9)} ${r2(ry * 0.8)} 0 0 0 ${cx - rx * 0.8} ${cy - ry * 0.75} Z" fill="${hairHex}"/>`);
    // species features
    parts.push(...speciesFeatures(spec.species, cx, cy, rx, ry, jawW, mouthYAbs, skinHex, hairHex));
    // eyes (anchor geometry)
    for (const s of [-1, 1]) {
      parts.push(`<circle cx="${r2(cx + s * eyeDx)}" cy="${r2(eyeYAbs)}" r="${r2(pupilR + 4)}" fill="#f4f0ea"/>`);
      parts.push(`<circle cx="${r2(cx + s * eyeDx)}" cy="${r2(eyeYAbs)}" r="${r2(pupilR)}" fill="${eyeHex}"/>`);
    }
    // brows (expression: brow contrast -> prominence)
    for (const s of [-1, 1]) {
      parts.push(`<line x1="${r2(cx + s * (eyeDx - pupilR - 4))}" y1="${r2(browYAbs)}" x2="${r2(cx + s * (eyeDx + pupilR + 4))}" y2="${r2(browYAbs - 3)}" stroke="${hairHex}" stroke-width="${r2(2 + 5 * A.expression.browContrast)}" stroke-linecap="round"/>`);
    }
    // mouth (expression: curve + darkness)
    parts.push(`<path d="M ${r2(cx - mouthW / 2)} ${r2(mouthYAbs)} Q ${cx} ${r2(mouthYAbs + curvePx)} ${r2(cx + mouthW / 2)} ${r2(mouthYAbs)}" fill="none" stroke="${hex(darken(A.skinTone))}" stroke-width="${r2(mouthW8)}" stroke-linecap="round"/>`);
    // lineage marks as persistent symbols (orbit, deterministic position by index)
    spec.lineageMarks.forEach((m, i) => {
      const ang = (-90 + i * (360 / Math.max(3, spec.lineageMarks.length))) * Math.PI / 180;
      const lx = cx + Math.cos(ang) * (rx + 48), ly = cy + Math.sin(ang) * (ry + 55);
      parts.push(`<g class="lineage-mark"><circle cx="${r2(lx)}" cy="${r2(ly)}" r="10" fill="none" stroke="#e8c860" stroke-width="2"/><text x="${r2(lx - 9)}" y="${r2(ly + 4)}" fill="#e8c860" font-size="9">${m.symbol ?? m.type}</text></g>`);
    });
    parts.push(`<text x="14" y="${H - 14}" fill="#888" font-size="12">${spec.species} · ${spec.biome} · ${spec.material.base}/${spec.material.texture} · lineage ${spec.lineageHash}</text>`);

    // Metadata: anchor geometry embedded for continuity assertions.
    const meta = [
      `data-spec-hash="${spec.specHash.slice(0, 16)}"`,
      `data-species="${spec.species}"`, `data-biome="${spec.biome}"`,
      `data-source-eye-ratio="${S.eyeSpacingRatio}"`, `data-eye-ratio="${A.eyeSpacingRatio}"`,
      `data-source-face-ratio="${S.faceRatio}"`, `data-face-ratio="${A.faceRatio}"`,
      `data-skin-tone="${skinHex}"`, `data-hair-tone="${hairHex}"`,
      `data-silhouette="${A.silhouetteHash.slice(0, 16)}"`,
      `data-lineage-hash="${spec.lineageHash}"`,
    ].join(' ');
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" ${meta}>\n${parts.join('\n')}\n</svg>`;

    const persist = opts.persist ?? true;
    let path = null;
    if (persist) {
      const name = `${opts.playerId ?? 'spec'}-${opts.embodimentId ?? spec.specHash.slice(0, 12)}-v2.svg`;
      path = join(this.dir, name);
      writeFileSync(path, svg);
    }
    return { path, svg, bytes: Buffer.byteLength(svg) };
  }
}
