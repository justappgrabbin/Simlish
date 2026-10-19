// identity/continuity.js — identityContinuity(anchorsBefore, anchorsAfter).
// Weighted similarity over eye spacing, face ratio, palette distance,
// silhouette hash overlap, expression signature. Score 0..1; >= 0.6 means
// "recognizably the same being". Pure + deterministic.

export const CONTINUITY_THRESHOLD = 0.6;

export const CONTINUITY_WEIGHTS = {
  eyeSpacing: 0.30,
  faceRatio: 0.20,
  palette: 0.20,
  silhouette: 0.20,
  expression: 0.10,
};

const clamp01 = (v) => Math.max(0, Math.min(1, v));

function rgbDist(a, b) {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) / 441.673; // /sqrt(3*255^2)
}

function hexOverlap(h1, h2) {
  // Hamming agreement over the hex-encoded bit string.
  let same = 0, total = 0;
  for (let i = 0; i < Math.min(h1.length, h2.length); i++) {
    const d = parseInt(h1[i], 16) ^ parseInt(h2[i], 16);
    for (let b = 0; b < 4; b++) { total++; if (!((d >> b) & 1)) same++; }
  }
  return total ? same / total : 0;
}

/**
 * @param {object} a - anchors before morph (from visualDNA extraction)
 * @param {object} b - anchors after morph (from EmbodimentSpec.anchorsAfter)
 * @returns {{score:number, recognizable:boolean, components:object}}
 */
export function identityContinuity(a, b) {
  if (!a || !b) throw new Error('identityContinuity: both anchorsBefore and anchorsAfter required');
  const components = {
    eyeSpacing: clamp01(1 - Math.abs(a.eyeSpacingRatio - b.eyeSpacingRatio) / 0.25),
    faceRatio: clamp01(1 - Math.abs(a.faceRatio - b.faceRatio) / 0.5),
    palette: clamp01(1 - (rgbDist(a.skinTone, b.skinTone) * 0.6 + rgbDist(a.hairTone, b.hairTone) * 0.4) / 0.45),
    silhouette: hexOverlap(a.silhouetteHash, b.silhouetteHash),
    expression: clamp01(1 - (
      Math.abs(a.expression.mouthDarkness - b.expression.mouthDarkness) / 0.5 * 0.4 +
      Math.abs(a.expression.mouthCurve - b.expression.mouthCurve) / 0.4 * 0.3 +
      Math.abs(a.expression.browContrast - b.expression.browContrast) / 0.5 * 0.3
    )),
  };
  let score = 0;
  for (const [k, w] of Object.entries(CONTINUITY_WEIGHTS)) score += w * components[k];
  score = +score.toFixed(4);
  return { score, recognizable: score >= CONTINUITY_THRESHOLD, components };
}
