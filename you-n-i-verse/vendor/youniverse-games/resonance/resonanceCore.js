// Resonance Core — resonance function, outcome resolver, effect parameters.
//
// Resonance function (spec §4.2, exact weights):
//   R = α·cos(action,intent) + β·cos(action,destiny)
//     + γ·temporal_alignment + δ·history_influence
//   α=0.4  β=0.3  γ=0.2  δ=0.1
//
// Outcome bands (spec §4.3, exact thresholds):
//   R <  -0.5        severe-distortion -> adversity hexagram shift + trap (shadow content, never game-over)
//   -0.5..-0.3       distortion        -> obstacles
//   -0.3.. 0.3       resonance         -> transformation-free normal advance
//    0.3.. 0.5       harmonic          -> positive advance / transformation
//   R >   0.5        perfect-harmony   -> line change + major reward
//
// Effect parameters (spec §4.4, exact formulas):
//   chromatic aberration = |R|*20 px        scan lines      = |R|*100 (count)
//   stutter interval     = max(1, 30-|R|*60) noise opacity  = |R|*0.8
//   time dilation        = 1-|R|*0.5        bloom           = R*2.0
//   saturation           = R*0.5 (applied as 1+R*0.5 grade multiplier)

import { cosineSimilarity, elementInteraction } from './embeddingSpace.js';
import { SEGMENTS, cosine, updateHistory, l2normalize } from '../divination/destinyVector.js';
import { BRANCHES, BRANCH_ELEMENTS } from '../divination/bazi.js';
import { deriveFloats } from '../util/hash.js';

export const WEIGHTS = { alpha: 0.4, beta: 0.3, gamma: 0.2, delta: 0.1 };

// Spec §4.3 thresholds, ordered for evaluation.
export const BANDS = [
  { name: 'perfect-harmony', test: (s) => s > 0.5 },
  { name: 'harmonic', test: (s) => s > 0.3 },
  { name: 'resonance', test: (s) => s >= -0.3 },
  { name: 'distortion', test: (s) => s >= -0.5 },
  { name: 'severe-distortion', test: () => true },
];

export function bandForScore(score) {
  if (!Number.isFinite(score)) throw new Error(`resonanceCore.bandForScore: bad score ${score}`);
  return BANDS.find((b) => b.test(score)).name;
}

/** Branch element of the current double-hour (时辰) — the "moment's element". */
export function hourBranchElement(now = new Date()) {
  const branchIndex = Math.floor(((now.getHours() + 1) % 24) / 2);
  return { branch: BRANCHES[branchIndex], branchIndex, element: BRANCH_ELEMENTS[branchIndex] };
}

/**
 * temporal_alignment: how the action's element sits inside the current
 * double-hour's element, via the generating/controlling cycle matrix
 * (+1 generated-by-moment ... -1 controlled-by-moment).
 */
export function temporalAlignment(actionElement, now = new Date()) {
  const { element: hourElement } = hourBranchElement(now);
  return elementInteraction(hourElement, actionElement);
}

/**
 * Compute resonance for one interaction (spec §4.2).
 * @param {object} input
 *   actionEmbedding - 512-dim vector of the player's action/entity
 *   actionElement   - five-element tag of the action (for temporal alignment)
 *   intentVector    - 512-dim session intent vector
 *   destinyVector   - 512-dim player destiny vector
 *   now             - Date (for temporal alignment)
 * @returns {{score, components:{intent,destiny,temporal,history}, weights}}
 */
export function computeResonance({ actionEmbedding, actionElement, intentVector, destinyVector, now = new Date() }) {
  for (const [k, v] of Object.entries({ actionEmbedding, intentVector, destinyVector })) {
    if (!Array.isArray(v) || v.length !== 512) throw new Error(`resonanceCore.computeResonance: ${k} must be a 512-dim array`);
  }
  if (!actionElement) throw new Error('resonanceCore.computeResonance: actionElement required');

  const intent = cosineSimilarity(actionEmbedding, intentVector);
  const destiny = cosineSimilarity(actionEmbedding, destinyVector);
  const temporal = temporalAlignment(actionElement, now);

  // history_influence: cosine between the action and the destiny vector's
  // accumulated-history segment (the past pulls on the present).
  const history = cosine(
    actionEmbedding.slice(SEGMENTS.history.start, SEGMENTS.history.start + SEGMENTS.history.len),
    destinyVector.slice(SEGMENTS.history.start, SEGMENTS.history.start + SEGMENTS.history.len),
  );

  const score = WEIGHTS.alpha * intent + WEIGHTS.beta * destiny
    + WEIGHTS.gamma * temporal + WEIGHTS.delta * history;
  return {
    score: +score.toFixed(4),
    components: {
      intent: +intent.toFixed(4), destiny: +destiny.toFixed(4),
      temporal: +temporal.toFixed(4), history: +history.toFixed(4),
    },
    weights: { ...WEIGHTS },
  };
}

// Adverse hexagrams used for the adversity shift (Obstruction/Abysmal/Oppression/Darkening...).
const ADVERSITY_HEXAGRAMS = [39, 29, 47, 36, 3, 12];

/**
 * Resolve an outcome from a resonance score (spec §4.3 game effects).
 * Never returns game-over: distortion paths route to shadow/underworld content.
 */
export function resolveOutcome({ score, currentHexagram, changingLines = [], resultingHexagram = null, seedKey = '' }) {
  const band = bandForScore(score);
  const salt = Math.floor(deriveFloats('outcome:' + seedKey + ':' + band, 1)[0] * 1000);
  const effects = { band, score };

  switch (band) {
    case 'severe-distortion': {
      const adverse = ADVERSITY_HEXAGRAMS[salt % ADVERSITY_HEXAGRAMS.length];
      effects.hexagramShift = { from: currentHexagram, to: adverse, kind: 'adversity' };
      effects.obstacles = ['the path folds back on itself', 'a shadow-double mimics your movements'];
      effects.trap = { kind: 'shadow-snare', escapeHint: 'answer the echo without force' };
      effects.shadowContent = true; // underworld/shadow branch, never game-over
      effects.realmShift = 'underworld';
      break;
    }
    case 'distortion': {
      effects.obstacles = ['static obscures the scene', 'an entity resists your touch'];
      effects.shadowContent = false;
      break;
    }
    case 'resonance': {
      effects.progress = true;
      break;
    }
    case 'harmonic': {
      effects.transformation = { kind: 'elemental-attunement', magnitude: +(0.5 + score).toFixed(3) };
      effects.progress = true;
      effects.positiveAdvance = true;
      break;
    }
    case 'perfect-harmony': {
      effects.lineChange = {
        hexagram: currentHexagram,
        lines: changingLines,
        resultingHexagram,
      };
      effects.reward = { type: 'major-insight', name: 'Oracle Insight', magnitude: +(1 + score).toFixed(3) };
      effects.progress = true;
      effects.positiveAdvance = true;
      break;
    }
    default:
      throw new Error(`resonanceCore.resolveOutcome: unknown band "${band}"`);
  }
  return effects;
}

/**
 * Effect parameter generator (spec §4.4). Concrete numeric parameters per
 * band; formulas are exact per spec:
 *   distortion side (R<0): chromaticAberrationPx=|R|*20, scanLines=|R|*100,
 *     stutterIntervalMs=max(1,30-|R|*60), noiseOpacity=|R|*0.8,
 *     timeDilation=1-|R|*0.5.
 *   harmony side (R>0): bloom=R*2.0, saturation grade=1+R*0.5.
 */
export function effectParameters(band, score) {
  const s = +score;
  if (bandForScore(s) !== band) throw new Error(`resonanceCore.effectParameters: band "${band}" does not match score ${s}`);
  const a = Math.abs(s);
  switch (band) {
    case 'severe-distortion':
    case 'distortion':
      return {
        chromaticAberrationPx: +(a * 20).toFixed(2),
        scanLines: { enabled: true, count: Math.round(a * 100), opacity: +(a * 0.8).toFixed(2) },
        stutterIntervalMs: Math.max(1, Math.round(30 - a * 60)),
        noiseOpacity: +(a * 0.8).toFixed(2),
        timeDilation: +(1 - a * 0.5).toFixed(2),
        bloom: 0,
        saturation: +(1 - a * 0.5).toFixed(2),
      };
    case 'resonance':
      return {
        chromaticAberrationPx: +(a * 20).toFixed(2),
        scanLines: { enabled: false, count: 0, opacity: 0 },
        stutterIntervalMs: Math.max(1, Math.round(30 - a * 60)),
        noiseOpacity: +(a * 0.8).toFixed(2),
        timeDilation: 1,
        bloom: +Math.max(0, s * 2.0).toFixed(2),
        saturation: +(1 + s * 0.5).toFixed(2),
      };
    case 'harmonic':
    case 'perfect-harmony':
      return {
        chromaticAberrationPx: 0,
        scanLines: { enabled: false, count: 0, opacity: 0 },
        stutterIntervalMs: 0,
        noiseOpacity: 0,
        timeDilation: +(1 + s * 0.5).toFixed(2),
        bloom: +(s * 2.0).toFixed(2),
        saturation: +(1 + s * 0.5).toFixed(2),
        ...(band === 'perfect-harmony' ? { goldenPulseHz: +(0.5 + s).toFixed(2) } : {}),
      };
    default:
      throw new Error(`resonanceCore.effectParameters: unknown band "${band}"`);
  }
}

/** Embed an interaction string and blend it into the destiny history segment. */
export function recordInteraction(destinyVector, interactionEmbedding) {
  if (!Array.isArray(interactionEmbedding) || interactionEmbedding.length !== 512) {
    throw new Error('resonanceCore.recordInteraction: interactionEmbedding must be 512-dim');
  }
  const newDestiny = updateHistory(destinyVector, interactionEmbedding, 0.3);
  return { newDestiny };
}

export { l2normalize };
