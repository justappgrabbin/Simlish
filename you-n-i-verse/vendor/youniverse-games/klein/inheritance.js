// klein/inheritance.js — bi-directional feature inheritance (Klein: features
// percolate BOTH down from the governing unit and up from the dependent unit
// during composition, and conflicting markers are reported, never silently
// overwritten).

import { makeUnit, featureNames, FEATURE_GROUPS, featuresFrom } from './units.js';

// Default inheritance bands (which feature classes flow in which direction):
//  - DOWNWARD: animacy/agency/sentience/mobility, species, roles, proven state
//    (the governing object's nature descends onto the composite).
//  - UPWARD: elemental context, trigram membership, valence, temporality
//    (the dependent's context rises into the composite).
export const DOWNWARD_FEATURES = [
  'ANIMATE', 'AGENT', 'SENTIENT', 'MOBILE',
  'SP_HUMAN', 'SP_FELINE', 'SP_AVIAN', 'SP_AQUATIC', 'SP_TERRANE', 'SP_SYNTHETIC', 'SP_HYBRID',
  'ROLE_WARRIOR', 'ROLE_SCHOLAR', 'ROLE_SAGE', 'ROLE_APPRENTICE', 'ROLE_GUARDIAN',
  'PROVEN', 'ST_YOUTH', 'ST_ADULT', 'ST_ELDER', 'ARTIFACT',
];
export const UPWARD_FEATURES = [
  'EL_WOOD', 'EL_FIRE', 'EL_EARTH', 'EL_METAL', 'EL_WATER',
  'TRI_QIAN', 'TRI_DUI', 'TRI_LI', 'TRI_ZHEN', 'TRI_XUN', 'TRI_KAN', 'TRI_GEN', 'TRI_KUN',
  'VAL_POSITIVE', 'VAL_NEGATIVE', 'T_PAST', 'T_PRESENT', 'T_FUTURE',
];

export const DOWNWARD_MASK = featuresFrom(DOWNWARD_FEATURES);
export const UPWARD_MASK = featuresFrom(UPWARD_FEATURES);

/**
 * Compose parent + child into a new unit with bi-directional inheritance.
 * @param {string} name - name for the composed unit
 * @param {Unit} parent - governing unit (features flow DOWN)
 * @param {Unit} child  - dependent unit (context flows UP)
 * @param {object} opts - {downMask?, upMask?} BigInt overrides
 * @returns {{unit, parentAfter, childAfter, conflicts, downward, upward}}
 *   unit: the composite; parentAfter/childAfter: each side after receiving the
 *   other's inherited bits (observable in BOTH directions); conflicts: list of
 *   one-hot group collisions (resolved by union, reported explicitly).
 */
export function composeUnits(name, parent, child, { downMask = DOWNWARD_MASK, upMask = UPWARD_MASK } = {}) {
  if (!parent?.features || !child?.features) throw new Error('klein.composeUnits: parent and child units required');

  // DOWN: parent's inheritable bits descend onto the child.
  const downBits = parent.features & downMask;
  const childAfter = { ...child, features: child.features | downBits };
  // UP: child's context bits rise onto the parent.
  const upBits = child.features & upMask;
  const parentAfter = { ...parent, features: parent.features | upBits };

  // Conflict detection over one-hot groups (union resolution, but REPORTED).
  const conflicts = [];
  for (const [group, members] of Object.entries(FEATURE_GROUPS)) {
    const p = members.filter((m) => (parent.features & featuresFrom([m])) !== 0n);
    const c = members.filter((m) => (child.features & featuresFrom([m])) !== 0n);
    const clash = p.filter((m) => !c.includes(m)).length > 0 && c.filter((m) => !p.includes(m)).length > 0 && p.length + c.length > 1;
    if (clash && !(p.length === 1 && c.length === 1 && p[0] === c[0])) {
      conflicts.push({ group, parent: p, child: c, resolution: 'union', resolved: [...new Set([...p, ...c])] });
    }
  }

  const unit = makeUnit({
    name,
    kind: 'object',
    features: parentAfter.features | childAfter.features,
    structure: { relation: 'composed-of', parent: parent.name, child: child.name, conflicts: conflicts.map((c) => c.group) },
  });

  return {
    unit,
    parentAfter,
    childAfter,
    conflicts,
    downward: featureNames({ features: downBits }),
    upward: featureNames({ features: upBits }),
  };
}
