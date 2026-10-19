// klein/registry.js — builds the default unit registry: the game's lexicon as
// klein semantic units. Entities come from the resonance embedding-space
// catalogue (adapters: the old structures stay, here they gain Boolean feature
// vectors); species/biome units feed the morph operators; role/stage units
// feed the analogy engine.

import { makeUnit, featuresFrom, UnitRegistry } from './units.js';
import { ENTITIES } from '../resonance/embeddingSpace.js';

const ELEMENT_FEATURE = { wood: 'EL_WOOD', fire: 'EL_FIRE', earth: 'EL_EARTH', metal: 'EL_METAL', water: 'EL_WATER' };
const CATEGORY_FORMCLASS = {
  elements: 'FC_OBJECT', scene: 'FC_SCENE', actions: 'FC_ACTION',
  symbols: 'FC_SYMBOL', hexagrams: 'FC_HEXAGRAM', emotions: 'FC_EMOTION',
};

// Species units — the morph system's species as Boolean feature vectors.
export const SPECIES_FEATURES = {
  human:     ['ANIMATE', 'SENTIENT', 'MOBILE', 'SP_HUMAN'],
  feline:    ['ANIMATE', 'MOBILE', 'SP_FELINE'],
  avian:     ['ANIMATE', 'MOBILE', 'SP_AVIAN'],
  aquatic:   ['ANIMATE', 'MOBILE', 'SP_AQUATIC'],
  terrane:   ['ANIMATE', 'SP_TERRANE'],
  synthetic: ['AGENT', 'ARTIFACT', 'SP_SYNTHETIC'],
  hybrid:    ['ANIMATE', 'MOBILE', 'SP_HYBRID'],
};

// Biome units — scene-class context, element per classical association.
export const BIOME_FEATURES = {
  forest: ['FC_SCENE', 'EL_WOOD', 'TRI_XUN'],
  desert: ['FC_SCENE', 'EL_EARTH', 'TRI_GEN'],
  ocean:  ['FC_SCENE', 'EL_WATER', 'TRI_KAN'],
  tundra: ['FC_SCENE', 'EL_METAL', 'TRI_QIAN'],
  urban:  ['FC_SCENE', 'EL_METAL', 'ARTIFACT'],
  cosmic: ['FC_SCENE', 'EL_FIRE', 'TRI_QIAN'],
};

/** Build the default registry with the full game lexicon. */
export function buildDefaultRegistry() {
  const reg = new UnitRegistry();

  // entity catalogue adapter
  const entForm = (e) => CATEGORY_FORMCLASS[e.category] ?? 'FC_OBJECT';
  for (const e of ENTITIES) {
    reg.register(makeUnit({
      name: e.name, kind: 'object',
      features: [entForm(e), ELEMENT_FEATURE[e.element]],
      structure: { category: e.category, element: e.element },
    }));
  }

  // species
  for (const [name, feats] of Object.entries(SPECIES_FEATURES)) {
    reg.register(makeUnit({ name, kind: 'object', features: feats, structure: { unitType: 'species' } }));
  }
  // biomes
  for (const [name, feats] of Object.entries(BIOME_FEATURES)) {
    reg.register(makeUnit({ name, kind: 'object', features: feats, structure: { unitType: 'biome' } }));
  }

  // --- role/stage units (analogy engine lexicon) --------------------------
  // Vectors are designed so Klein's Boolean-group analogy arithmetic
  // (D = C XOR (A XOR B)) resolves exactly onto registered units.
  const human = featuresFrom(SPECIES_FEATURES.human);
  reg.register(makeUnit({
    name: 'warrior', kind: 'object',
    features: human | featuresFrom(['AGENT', 'ROLE_WARRIOR', 'EL_METAL']),
    structure: { unitType: 'role', embodimentKind: 'warrior' },
  }));
  reg.register(makeUnit({
    name: 'scholar', kind: 'object',
    features: human | featuresFrom(['ROLE_SCHOLAR', 'SENTIENT']),
    structure: { unitType: 'role', embodimentKind: 'scholar' },
  }));
  reg.register(makeUnit({
    name: 'guardian', kind: 'object',
    features: human | featuresFrom(['AGENT', 'ROLE_WARRIOR', 'ROLE_GUARDIAN', 'EL_METAL', 'PROVEN']),
    structure: { unitType: 'role', embodimentKind: 'warrior', blocksUnproven: true },
  }));
  reg.register(makeUnit({
    // scholar bearing the guardian's proof: warrior:guardian :: scholar:warden
    name: 'warden', kind: 'object',
    features: human | featuresFrom(['ROLE_SCHOLAR', 'ROLE_GUARDIAN', 'SENTIENT', 'PROVEN']),
    structure: { unitType: 'role' },
  }));
  reg.register(makeUnit({ name: 'youth', kind: 'object', features: human | featuresFrom(['ST_YOUTH']), structure: { unitType: 'stage' } }));
  reg.register(makeUnit({
    name: 'apprentice', kind: 'object',
    features: human | featuresFrom(['ST_YOUTH', 'ROLE_APPRENTICE', 'AGENT']),
    structure: { unitType: 'role', embodimentKind: 'childlike' },
  }));
  reg.register(makeUnit({ name: 'elder', kind: 'object', features: human | featuresFrom(['ST_ELDER']), structure: { unitType: 'stage', embodimentKind: 'elder' } }));
  reg.register(makeUnit({
    // elder carrying the apprentice's learning: youth:apprentice :: elder:sage
    name: 'sage', kind: 'object',
    features: human | featuresFrom(['ST_ELDER', 'ROLE_APPRENTICE', 'ROLE_SAGE', 'AGENT']),
    structure: { unitType: 'role', embodimentKind: 'elder' },
  }));
  // feline-warrior embodiment: human:warrior :: feline:feline-warrior
  reg.register(makeUnit({
    name: 'feline-warrior', kind: 'object',
    features: featuresFrom(['ANIMATE', 'MOBILE', 'SP_FELINE', 'AGENT', 'ROLE_WARRIOR', 'EL_METAL']),
    structure: { unitType: 'embodiment', species: 'feline', embodimentKind: 'warrior' },
  }));

  // --- relations (relations may be logical operators) ----------------------
  reg.register(makeUnit({
    name: 'of-the-river', kind: 'relation',
    features: ['FC_RELATION', 'EL_WATER', 'TRI_KAN'],
    structure: { operator: 'contextualize', domain: 'river', element: 'water' },
    aliases: ['river-relation'],
  }));
  reg.register(makeUnit({
    name: 'blocks', kind: 'relation',
    features: ['FC_RELATION', 'VAL_NEGATIVE'],
    structure: { operator: 'logical-and-not', description: 'guardian AND NOT proven -> blocked' },
  }));

  return reg;
}

let DEFAULT = null;
/** Shared process-wide default registry (lazy). */
export function defaultRegistry() {
  if (!DEFAULT) DEFAULT = buildDefaultRegistry();
  return DEFAULT;
}
