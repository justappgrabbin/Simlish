// morph/kleinOperators.js — recorded, pure transformation operators per
// UPGRADE_PLAN §4. Each operator is a pure function; the ordered chain is
// recorded in MorphState.transformationRules.operatorChain so every morph is
// auditable and reproducible. No randomness, no clocks.

import { sha256Hex, seededRng } from '../util/hash.js';
import { makeUnit, serializeUnit, identityBitsFromHash, featureNames } from '../klein/units.js';
import { buildDefaultRegistry } from '../klein/registry.js';
import { composeUnits } from '../klein/inheritance.js';
import { solveAnalogy } from '../klein/analogy.js';

const r4 = (v) => +v.toFixed(4);
const clamp01 = (v) => Math.max(0, Math.min(1, v));

// Species morphology maps: skull shape, eye placement, proportions, material.
export const SPECIES = {
  feline:    { skull: { rx: 1.05, ry: 0.88, jawTaper: 0.55 }, eyePlacementMultiplier: 1.08, proportions: { limb: 0.95, torso: 0.9, head: 1.1 }, material: 'fur',      features: ['ears-pointed', 'whiskers', 'slit-pupils'] },
  avian:     { skull: { rx: 0.82, ry: 1.0, jawTaper: 0.8 },   eyePlacementMultiplier: 1.12, proportions: { limb: 0.8, torso: 0.85, head: 0.95 }, material: 'feathers', features: ['beak', 'crest', 'wing-lines'] },
  aquatic:   { skull: { rx: 0.95, ry: 1.12, jawTaper: 0.7 },  eyePlacementMultiplier: 0.96, proportions: { limb: 1.1, torso: 1.15, head: 0.9 }, material: 'scales',    features: ['fins', 'gill-lines', 'dorsal-ridge'] },
  terrane:   { skull: { rx: 1.0, ry: 1.05, jawTaper: 0.45 },  eyePlacementMultiplier: 1.0,  proportions: { limb: 1.15, torso: 1.1, head: 1.0 }, material: 'bark',      features: ['growth-rings', 'moss-patches', 'root-limbs'] },
  synthetic: { skull: { rx: 0.9, ry: 0.95, jawTaper: 0.6 },   eyePlacementMultiplier: 1.02, proportions: { limb: 1.0, torso: 1.0, head: 1.0 },  material: 'alloy',     features: ['circuit-traces', 'optic-lenses', 'seam-lines'] },
  hybrid:    { skull: { rx: 0.97, ry: 1.0, jawTaper: 0.62 },  eyePlacementMultiplier: 1.05, proportions: { limb: 1.0, torso: 1.0, head: 1.05 }, material: 'chimeric',  features: ['ears-pointed', 'gill-lines', 'seam-lines'] },
};

// Biome adaptation table: tint blend, texture, posture, movement.
export const BIOMES = {
  forest: { tint: [74, 124, 89],   blend: 0.12, texture: 'mottled-canopy', posture: 'alert',      movement: 'weaving' },
  desert: { tint: [214, 178, 110], blend: 0.12, texture: 'sun-bleached',   posture: 'low-slung',  movement: 'drifting' },
  ocean:  { tint: [58, 110, 165],  blend: 0.14, texture: 'ripple-sheen',   posture: 'streamlined', movement: 'undulating' },
  tundra: { tint: [205, 216, 228], blend: 0.12, texture: 'frost-rimed',    posture: 'hunched',    movement: 'plodding' },
  urban:  { tint: [140, 140, 148], blend: 0.10, texture: 'concrete-matte', posture: 'upright',    movement: 'striding' },
  cosmic: { tint: [96, 74, 160],   blend: 0.15, texture: 'starfield',      posture: 'floating',   movement: 'orbiting' },
};

export const OPERATOR_CHAIN = ['preserveAnchors', 'translateAnatomy', 'carryLineage', 'adaptMaterial', 'fuseIdentity'];

// Op 1: capture the identity invariants that must survive every transform.
export function preserveAnchors(dna) {
  if (!dna?.anchors) throw new Error('kleinOperators.preserveAnchors: visualDNA with anchors required');
  const a = dna.anchors;
  return {
    eyeSpacingRatio: a.eyeSpacingRatio,
    eyeY: a.eyeY,
    faceRatio: a.faceRatio,
    mouthY: a.mouthY,
    expression: { ...a.expression },
    silhouetteHash: a.silhouetteHash,
    skinTone: [...a.skinTone],
    hairTone: [...a.hairTone],
    anchorsHash: a.anchorsHash,
  };
}

// Op 2: map the anatomy through a species morphology map.
export function translateAnatomy(dna, species) {
  const morph = SPECIES[species];
  if (!morph) throw new Error(`kleinOperators.translateAnatomy: unknown species "${species}" (known: ${Object.keys(SPECIES).join(', ')})`);
  const a = dna.anchors;
  return {
    species,
    skull: { ...morph.skull },
    proportions: { ...morph.proportions },
    material: morph.material,
    features: [...morph.features],
    eyeSpacingRatio: r4(clamp01(a.eyeSpacingRatio * morph.eyePlacementMultiplier)),
    faceRatio: r4(a.faceRatio * (morph.skull.rx / morph.skull.ry) / 1.0 * 0.5 + a.faceRatio * 0.5),
  };
}

// Op 3: inherited lineage marks persist across forms.
export function carryLineage(dna, lineageMarkers = []) {
  const a = dna.anchors;
  const inherited = lineageMarkers.map((m) => ({ ...m, persistent: true, carriedFrom: a.anchorsHash.slice(0, 12) }));
  return {
    lineageMarks: inherited,
    lineageHash: sha256Hex({ anchorsHash: a.anchorsHash, marks: inherited.map((m) => m.symbol ?? m.gate ?? m.type) }).slice(0, 16),
  };
}

// Op 4: adapt surface to the biome (tint blend, texture, posture, movement).
export function adaptMaterial(form, biome) {
  const b = BIOMES[biome];
  if (!b) throw new Error(`kleinOperators.adaptMaterial: unknown biome "${biome}" (known: ${Object.keys(BIOMES).join(', ')})`);
  const blendRgb = (rgb) => rgb.map((v, i) => Math.round(v * (1 - b.blend) + b.tint[i] * b.blend));
  return {
    ...form,
    biome,
    texture: b.texture,
    posture: b.posture,
    movement: b.movement,
    adaptedSkinTone: blendRgb(form.skinTone),
    adaptedHairTone: blendRgb(form.hairTone),
  };
}

// Deterministic bit permutation of the 256-bit silhouette, per species.
// A position permutation preserves pairwise Hamming agreement exactly, so
// cross-person silhouette difference survives the species mapping.
function permuteSilhouette(hexHash, species) {
  const bits = hexHash.split('').map((h) => parseInt(h, 16).toString(2).padStart(4, '0')).join('').split('').map(Number);
  const rng = seededRng(`silhouette-permute:${species}`);
  const perm = [...Array(256).keys()];
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  const out = new Array(256);
  for (let i = 0; i < 256; i++) out[i] = bits[perm[i]];
  let hex = '';
  for (let i = 0; i < 256; i += 4) hex += parseInt(out.slice(i, i + 4).join(''), 2).toString(16);
  return hex;
}

// Op 5: the actual "cat-me" merge — anchors fused with species traits into
// the transformed (after) anchors used for continuity measurement.
export function fuseIdentity(preserved, anatomy, material) {
  return {
    eyeSpacingRatio: anatomy.eyeSpacingRatio,
    eyeY: preserved.eyeY,
    faceRatio: r4(preserved.faceRatio * 0.6 + anatomy.faceRatio * 0.4),
    mouthY: preserved.mouthY,
    expression: { ...preserved.expression }, // expression signature survives species change
    silhouetteHash: permuteSilhouette(preserved.silhouetteHash, anatomy.species),
    skinTone: material.adaptedSkinTone,
    hairTone: material.adaptedHairTone,
  };
}

// --- klein semantic layer ----------------------------------------------------
// Species/biome/lineage are re-expressed as klein semantic units with Boolean
// feature vectors; identity anchors map onto the reserved 10-bit identity
// segment (bits 54-63) so continuity.js's numeric metric stays the metric of
// record while the semantic core sees identity as feature overlap.

let morphRegistry = null;
/** Registry holding species/biome/role/entity units (lazy singleton). */
export function morphUnitRegistry() {
  if (!morphRegistry) morphRegistry = buildDefaultRegistry();
  return morphRegistry;
}

/** Species as a klein unit with a Boolean feature vector. */
export function speciesUnit(species) {
  const reg = morphUnitRegistry();
  if (!reg.has(species) || reg.get(species).structure?.unitType !== 'species') {
    throw new Error(`kleinOperators.speciesUnit: unknown species "${species}" (known: ${Object.keys(SPECIES).join(', ')})`);
  }
  return reg.get(species);
}

/** Biome as a klein unit. */
export function biomeUnit(biome) {
  const reg = morphUnitRegistry();
  if (!reg.has(biome) || reg.get(biome).structure?.unitType !== 'biome') {
    throw new Error(`kleinOperators.biomeUnit: unknown biome "${biome}" (known: ${Object.keys(BIOMES).join(', ')})`);
  }
  return reg.get(biome);
}

/** Identity anchors -> klein unit: reserved identity segment + human species base. */
export function identityAnchorUnit(dna) {
  if (!dna?.anchors?.anchorsHash) throw new Error('kleinOperators.identityAnchorUnit: dna.anchors.anchorsHash required');
  const u = makeUnit({
    name: `identity:${dna.anchors.anchorsHash.slice(0, 12)}`,
    kind: 'object',
    features: ['SP_HUMAN', 'ANIMATE', 'SENTIENT'],
    structure: { anchorsHash: dna.anchors.anchorsHash },
  });
  u.features |= identityBitsFromHash(dna.anchors.anchorsHash);
  return u;
}

/**
 * Morph as relational composition: identity × species × biome, with
 * bi-directional feature inheritance and explicit conflict reporting.
 * @returns {{unit, serialized, conflicts, downward, upward}}
 */
export function composeMorph(dna, { species, biome }) {
  const identity = identityAnchorUnit(dna);
  const sp = speciesUnit(species);
  const bm = biomeUnit(biome);
  // species governs the form; biome is the dependent context relation
  const composed = composeUnits(`morph:${species}:${biome}`, sp, bm);
  // identity flows down onto the composite (identity segment rides along)
  composed.unit.features |= identity.features;
  composed.unit.structure = {
    ...composed.unit.structure,
    identity: identity.name,
    identitySegment: `0x${(identity.features >> 54n).toString(16)}`,
  };
  return {
    unit: composed.unit,
    serialized: serializeUnit(composed.unit),
    conflicts: composed.conflicts,
    downward: composed.downward,
    upward: composed.upward,
    identity: serializeUnit(identity),
  };
}

/**
 * Novel form via the Boolean-group analogy engine:
 * given a known composed form A:B (e.g. human:warrior) and a new base C
 * (e.g. feline), derive D (feline-warrior) and return its registered unit.
 */
export function analogousForm(aName, bName, cName) {
  const reg = morphUnitRegistry();
  return solveAnalogy(reg.get(aName), reg.get(bName), reg.get(cName), reg);
}

/**
 * Run the full recorded operator chain.
 * @param {object} dna - seed.visualDNA ({anchors})
 * @param {object} opts - {species, biome, lineageMarkers?}
 * @returns {EmbodimentSpec}
 */
export function runKleinChain(dna, { species, biome, lineageMarkers = [] }) {
  const preserved = preserveAnchors(dna);                    // op 1
  const anatomy = translateAnatomy(dna, species);            // op 2
  const lineage = carryLineage(dna, lineageMarkers);         // op 3
  const material = adaptMaterial({ ...preserved, ...anatomy }, biome); // op 4
  const anchorsAfter = fuseIdentity(preserved, anatomy, material);     // op 5

  // klein relational composition of the same morph (semantic layer; JSON-safe)
  const klein = composeMorph(dna, { species, biome });

  const spec = {
    version: 1,
    species,
    biome,
    operatorChain: [...OPERATOR_CHAIN],
    geometry: {
      skull: anatomy.skull,
      proportions: anatomy.proportions,
      features: anatomy.features,
    },
    material: { base: anatomy.material, texture: material.texture, posture: material.posture, movement: material.movement },
    anchors: preserved,              // identity invariants (the "before")
    anchorsAfter,                    // transformed anchors (the "after")
    lineageMarks: lineage.lineageMarks,
    lineageHash: lineage.lineageHash,
    klein: {
      unit: klein.serialized,               // composed unit w/ feature vector (hex)
      featureNames: featureNames(klein.unit),
      conflicts: klein.conflicts,
      inheritance: { downward: klein.downward, upward: klein.upward },
      identity: klein.identity,
    },
    specHash: null,
  };
  spec.specHash = sha256Hex({ ...spec, specHash: undefined });
  return spec;
}
