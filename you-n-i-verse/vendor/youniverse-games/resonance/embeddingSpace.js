// Resonance Core — entity embedding space.
// 512-dim L2-normalized vectors, deterministic per entity string, with
// category structure and elemental tags. The five-element generating/
// controlling cycle is baked into the vector relationships: each element has
// a 16-dim signature tiled through the vector; an entity's vector also mixes
// in the signatures of the element it generates (positive) and the element
// that controls it (negative), so cosine similarity respects the cycles.

import { deriveFloats } from '../util/hash.js';
import { l2normalize } from '../divination/destinyVector.js';
import { GENERATES, CONTROLS, FIVE_ELEMENTS } from '../divination/bazi.js';

export const EMBED_DIM = 512;

// Signed (zero-centered) element signatures: entities sharing an element
// correlate; entities linked by the controlling cycle anti-correlate because
// each vector subtracts the signature of its controlling element.
export const ELEMENT_SIGNATURES = Object.fromEntries(
  FIVE_ELEMENTS.map((e) => [e, l2normalize(deriveFloats(`element-sig:${e}`, 16).map((x) => (x - 0.5) * 2))]),
);

// Element interaction matrix: +1 generates, -1 controls (from row's perspective).
export function elementInteraction(fromElement, toElement) {
  if (fromElement === toElement) return 1; // same element: full elemental alignment
  if (GENERATES[fromElement] === toElement) return 1;   // generating cycle (symmetric affinity)
  if (GENERATES[toElement] === fromElement) return 1;
  if (CONTROLS[fromElement] === toElement) return -1;   // controlling cycle (symmetric hostility)
  if (CONTROLS[toElement] === fromElement) return -1;
  return 0;
}

// --- Entity catalogue (≥64 entities across 6 categories) --------------------
// [name, category, element]
const ENTITY_DEFS = [
  // five elements
  ['wood', 'elements', 'wood'], ['fire', 'elements', 'fire'], ['earth', 'elements', 'earth'],
  ['metal', 'elements', 'metal'], ['water', 'elements', 'water'],
  // scene elements (18)
  ['river', 'scene', 'water'], ['well', 'scene', 'water'], ['rain', 'scene', 'water'], ['mist', 'scene', 'water'],
  ['mountain', 'scene', 'earth'], ['stone', 'scene', 'earth'], ['cave', 'scene', 'earth'], ['field', 'scene', 'earth'],
  ['pine-tree', 'scene', 'wood'], ['reed', 'scene', 'wood'], ['bridge', 'scene', 'wood'], ['gate', 'scene', 'wood'],
  ['lantern', 'scene', 'fire'], ['ember', 'scene', 'fire'], ['sun-disc', 'scene', 'fire'],
  ['bronze-bell', 'scene', 'metal'], ['sword', 'scene', 'metal'], ['mirror', 'scene', 'metal'],
  // actions (12)
  ['touch', 'actions', 'earth'], ['observe', 'actions', 'water'], ['speak', 'actions', 'metal'],
  ['strike', 'actions', 'fire'], ['lift', 'actions', 'wood'], ['release', 'actions', 'water'],
  ['kneel', 'actions', 'earth'], ['trace', 'actions', 'wood'], ['shatter', 'actions', 'metal'],
  ['kindle', 'actions', 'fire'], ['listen', 'actions', 'water'], ['carry', 'actions', 'earth'],
  // symbols (12)
  ['spiral', 'symbols', 'water'], ['dragon', 'symbols', 'wood'], ['phoenix', 'symbols', 'fire'],
  ['tortoise', 'symbols', 'earth'], ['tiger', 'symbols', 'metal'], ['crane', 'symbols', 'wood'],
  ['knot', 'symbols', 'earth'], ['circle', 'symbols', 'metal'], ['flame-glyph', 'symbols', 'fire'],
  ['wave-glyph', 'symbols', 'water'], ['thunder-mark', 'symbols', 'wood'], ['void-sigil', 'symbols', 'earth'],
  // hexagram archetypes (8, one per trigram)
  ['hex-creative', 'hexagrams', 'metal'], ['hex-receptive', 'hexagrams', 'earth'],
  ['hex-arousing', 'hexagrams', 'wood'], ['hex-gentle', 'hexagrams', 'wood'],
  ['hex-clinging', 'hexagrams', 'fire'], ['hex-abysmal', 'hexagrams', 'water'],
  ['hex-still', 'hexagrams', 'earth'], ['hex-joyous', 'hexagrams', 'metal'],
  // emotional states (10)
  ['calm', 'emotions', 'water'], ['grief', 'emotions', 'metal'], ['joy', 'emotions', 'fire'],
  ['anger', 'emotions', 'wood'], ['fear', 'emotions', 'water'], ['longing', 'emotions', 'earth'],
  ['resolve', 'emotions', 'metal'], ['wonder', 'emotions', 'fire'], ['sorrow', 'emotions', 'earth'],
  ['courage', 'emotions', 'wood'],
  // extra scene/symbol padding to clear 64
  ['pool', 'scene', 'water'], ['altar', 'scene', 'earth'], ['torch', 'scene', 'fire'],
  ['bell-cord', 'scene', 'wood'], ['chain', 'scene', 'metal'], ['oracle-bone', 'symbols', 'earth'],
  ['yin-yang', 'symbols', 'water'], ['five-peaks', 'symbols', 'earth'], ['stillness', 'emotions', 'earth'],
];

export const ENTITIES = ENTITY_DEFS.map(([name, category, element]) => ({ name, category, element }));

export function getEntity(name) {
  const e = ENTITIES.find((x) => x.name === name);
  if (!e) throw new Error(`embeddingSpace: unknown entity "${name}"`);
  return e;
}

// Embedding mix weights (numerically tuned so the generating/controlling
// cycle is strongly readable by cosine similarity: same-element cos ~0.8+,
// generating ~+0.3, controlling ~-0.5, while entity identity stays distinct).
export const EMBED_WEIGHTS = { base: 0.2, element: 1.0, generates: 0.25, controlledBy: 2.6 };

/**
 * Deterministic 512-dim L2-normalized embedding for an entity string.
 * Structure: base-weight seeded-random identity + element signature tiled,
 * plus +generates * generated-element signature, -controlledBy * the
 * signature of the element that controls this one (controlling cycle baked
 * in as anti-correlation).
 */
export function embed(name, category = null, element = null) {
  let ent = ENTITIES.find((x) => x.name === name);
  if (!ent) {
    if (!element) throw new Error(`embeddingSpace.embed: "${name}" not in catalogue; pass element explicitly for ad-hoc embedding`);
    ent = { name, category: category ?? 'adhoc', element };
  }
  const w = EMBED_WEIGHTS;
  const base = deriveFloats(`entity:${ent.name}:${ent.category}`, EMBED_DIM).map((x) => (x - 0.5) * 2);
  const sig = ELEMENT_SIGNATURES[ent.element];
  const genSig = ELEMENT_SIGNATURES[GENERATES[ent.element]];
  // find what controls this element
  const controller = Object.keys(CONTROLS).find((k) => CONTROLS[k] === ent.element);
  const ctlSig = ELEMENT_SIGNATURES[controller];
  const out = new Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) {
    const s = i % 16;
    out[i] = w.base * base[i] + w.element * sig[s] + w.generates * genSig[s] - w.controlledBy * ctlSig[s];
  }
  return l2normalize(out);
}

// Trigram -> five-element association (classical).
export const TRIGRAM_ELEMENTS = {
  qian: 'metal', dui: 'metal', li: 'fire', zhen: 'wood',
  xun: 'wood', kan: 'water', gen: 'earth', kun: 'earth',
};

/**
 * Trigram embedding for the gesture interaction mode: a near-pure tiled
 * element signature (plus a small deterministic per-trigram jitter), so a
 * gesture resonates almost purely by its trigram's element.
 */
export function trigramEmbedding(trigram) {
  const element = TRIGRAM_ELEMENTS[trigram];
  if (!element) throw new Error(`embeddingSpace.trigramEmbedding: unknown trigram "${trigram}"`);
  const sig = ELEMENT_SIGNATURES[element];
  const jitter = deriveFloats(`trigram-emb:${trigram}`, EMBED_DIM).map((x) => (x - 0.5) * 2);
  const out = new Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) out[i] = sig[i % 16] + 0.08 * jitter[i];
  return l2normalize(out);
}

/** Combine two entity embeddings (normalized sum); deterministic. */
export function combineEmbeddings(a, b) {
  if (a.length !== b.length) throw new Error('embeddingSpace.combineEmbeddings: dim mismatch');
  return l2normalize(a.map((x, i) => x + b[i]));
}

export function cosineSimilarity(a, b) {
  if (a.length !== b.length) throw new Error('embeddingSpace.cosineSimilarity: dim mismatch');
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Gesture symbol -> trigram mapping (for gesture interaction mode).
export const GESTURE_TRIGRAMS = {
  'sky-arc': 'qian', 'lake-crescent': 'dui', 'flame-flick': 'li', 'thunder-stomp': 'zhen',
  'wind-sweep': 'xun', 'water-trickle': 'kan', 'mountain-press': 'gen', 'earth-open': 'kun',
};
