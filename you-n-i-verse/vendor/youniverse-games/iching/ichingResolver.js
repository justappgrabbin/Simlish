// Shared I Ching resolution service — reusable by any game.
// Deterministic from input state + seeded RNG. No per-game hardcoded logic:
// games call resolveIChing() and interpret the structured result.

import { seededRng, shortHash } from '../util/hash.js';
import {
  makeHexagramUnit, makeUnit, hexLinesOf,
  flipHexLine, serializeUnit, UnitRegistry,
} from '../klein/units.js';
import { composeUnits } from '../klein/inheritance.js';

// Trigram nature table: drives judgment-derived world effects.
export const TRIGRAMS = {
  qian: { symbol: '☰', element: 'heaven', behavior: 'asserts', force: 'creative' },
  dui:  { symbol: '☱', element: 'lake',    behavior: 'opens',   force: 'joyous' },
  li:   { symbol: '☲', element: 'fire',    behavior: 'reveals',  force: 'clinging' },
  zhen: { symbol: '☳', element: 'thunder', behavior: 'shakes',   force: 'arousing' },
  xun:  { symbol: '☴', element: 'wind',    behavior: 'penetrates', force: 'gentle' },
  kan:  { symbol: '☵', element: 'water',   behavior: 'floods',   force: 'abysmal' },
  gen:  { symbol: '☶', element: 'mountain', behavior: 'blocks',  force: 'still' },
  kun:  { symbol: '☷', element: 'earth',   behavior: 'yields',   force: 'receptive' },
};

// All 64 hexagrams in King Wen order: [name, upperTrigram, lowerTrigram].
export const HEXAGRAMS = [
  null, // 1-indexed
  ['The Creative','qian','qian'], ['The Receptive','kun','kun'], ['Difficulty at the Beginning','kan','zhen'],
  ['Youthful Folly','gen','kan'], ['Waiting','kan','qian'], ['Conflict','qian','kan'],
  ['The Army','kun','kan'], ['Holding Together','kan','kun'], ['The Taming Power of the Small','xun','qian'],
  ['Treading','qian','dui'], ['Peace','kun','qian'], ['Standstill','qian','kun'],
  ['Fellowship with Men','qian','li'], ['Possession in Great Measure','li','qian'], ['Modesty','kun','gen'],
  ['Enthusiasm','zhen','kun'], ['Following','dui','zhen'], ['Work on What Has Been Spoiled','gen','xun'],
  ['Approach','kun','dui'], ['Contemplation','xun','kun'], ['Biting Through','li','zhen'],
  ['Grace','gen','li'], ['Splitting Apart','gen','kun'], ['Return','kun','zhen'],
  ['Innocence','qian','zhen'], ['The Taming Power of the Great','gen','qian'], ['Nourishment','gen','zhen'],
  ['Preponderance of the Great','dui','xun'], ['The Abysmal','kan','kan'], ['The Clinging','li','li'],
  ['Influence','dui','gen'], ['Duration','zhen','xun'], ['Retreat','qian','gen'],
  ['The Power of the Great','zhen','qian'], ['Progress','li','kun'], ['Darkening of the Light','kun','li'],
  ['The Family','xun','li'], ['Opposition','li','dui'], ['Obstruction','kan','gen'],
  ['Deliverance','zhen','kan'], ['Decrease','gen','dui'], ['Increase','xun','zhen'],
  ['Breakthrough','dui','qian'], ['Coming to Meet','qian','xun'], ['Gathering Together','dui','kun'],
  ['Pushing Upward','kun','xun'], ['Oppression','dui','kan'], ['The Well','kan','xun'],
  ['Revolution','dui','li'], ['The Cauldron','li','xun'], ['The Arousing','zhen','zhen'],
  ['Keeping Still','gen','gen'], ['Development','xun','gen'], ['The Marrying Maiden','zhen','dui'],
  ['Abundance','zhen','li'], ['The Wanderer','li','gen'], ['The Gentle','xun','xun'],
  ['The Joyous','dui','dui'], ['Dispersion','xun','kan'], ['Limitation','kan','dui'],
  ['Inner Truth','xun','dui'], ['Preponderance of the Small','zhen','gen'], ['After Completion','kan','li'],
  ['Before Completion','li','kan'],
].map((entry, i) => (i === 0 ? null : { number: i, name: entry[0], upper: entry[1], lower: entry[2] }));

// Judgment-driven effect rules keyed by "upper>lower" trigram pair.
// Each rule emits concrete world-behavior effects, not text explanations.
const PAIR_RULES = {
  'kan>zhen': { environment: [{ type: 'weather', change: 'storm gathers overhead' }, { type: 'terrain', change: 'ground trembles with new springs' }], pressure: 'growth-through-difficulty', actions: ['push forward slowly', 'seek allies'] },
  'gen>kan': { environment: [{ type: 'npc', change: 'a gatekeeper doubts you' }, { type: 'object', change: 'a locked cache appears' }], pressure: 'learning-through-confusion', actions: ['ask naive questions', 'wait for a teacher'] },
  'kan>qian': { environment: [{ type: 'weather', change: 'rain hangs but does not fall' }, { type: 'terrain', change: 'river level rising' }], pressure: 'patient-waiting', actions: ['wait by the river', 'prepare provisions'] },
  'qian>kan': { environment: [{ type: 'npc', change: 'an official challenges your passage' }, { type: 'terrain', change: 'bridge barricaded' }], pressure: 'conflict', actions: ['argue the case', 'yield and reroute'] },
  'gen>dui': { environment: [{ type: 'terrain', change: 'the mountain path narrows to a single ledge' }, { type: 'object', change: 'a heavy stone seal blocks a doorway' }], pressure: 'decrease-offering', actions: ['offer something of value', 'shed baggage'] },
  'xun>zhen': { environment: [{ type: 'weather', change: 'wind rises after thunder' }, { type: 'terrain', change: 'hidden path revealed among reeds' }], pressure: 'increase', actions: ['take the new path', 'help another traveler'] },
  'li>kan': { environment: [{ type: 'weather', change: 'firelight reflects on floodwater' }, { type: 'terrain', change: 'fordable shallows shift' }], pressure: 'before-completion', actions: ['cross carefully', 'mark the far shore'] },
  'dui>gen': { environment: [{ type: 'npc', change: 'a joyful stranger blocks the mountain pass' }, { type: 'object', change: 'an invitation carved in stone' }], pressure: 'influence-wooing', actions: ['accept the invitation', 'decline politely'] },
  'zhen>qian': { environment: [{ type: 'weather', change: 'thunder rolls across clear sky' }, { type: 'npc', change: 'a rival tests your strength' }], pressure: 'great-power', actions: ['show force', 'restrain force'] },
  'kun>li': { environment: [{ type: 'weather', change: 'light dims under earth' }, { type: 'terrain', change: 'path goes underground' }], pressure: 'darkening-hiding', actions: ['travel by night', 'hide your light'] },
};

// Moving-line modifiers: line number changes the flavor of all effects.
const LINE_RULES = {
  1: { intensity: 0.6, tag: 'beginning', morphKind: 'childlike' },
  2: { intensity: 0.8, tag: 'centered', morphKind: 'normal' },
  3: { intensity: 1.0, tag: 'transition-crisis', morphKind: 'shadow' },
  4: { intensity: 0.9, tag: 'threshold', morphKind: 'wanderer' },
  5: { intensity: 1.2, tag: 'ruling', morphKind: 'royal' },
  6: { intensity: 1.1, tag: 'culmination', morphKind: 'mythic' },
};

export function getHexagram(number) {
  const h = HEXAGRAMS[number];
  if (!h) throw new Error(`ichingResolver: unknown hexagram ${number}`);
  return h;
}

/**
 * Resolve an I Ching moment into structured world/relationship/morph effects.
 * Deterministic: identical inputs -> identical outputs.
 * @param {object} input - {playerState, worldState, relationshipState,
 *   activeHexagram, activeLine, previousState, event, choice, temporalState}
 */
export function resolveIChing(input) {
  const {
    playerState = {}, worldState = {}, relationshipState = {},
    activeHexagram, activeLine = 1, previousState = null,
    event = null, choice = null, temporalState = {},
  } = input;
  if (!Number.isInteger(activeHexagram) || activeHexagram < 1 || activeHexagram > 64) {
    throw new Error('ichingResolver.resolveIChing: activeHexagram (1-64) required');
  }
  if (activeLine < 1 || activeLine > 6) throw new Error('ichingResolver: activeLine must be 1-6');

  const hex = getHexagram(activeHexagram);
  const pairKey = `${hex.upper}>${hex.lower}`;
  const rule = PAIR_RULES[pairKey] ?? genericRule(hex);
  const lineRule = LINE_RULES[activeLine];

  const rng = seededRng(JSON.stringify({
    activeHexagram, activeLine, event, choice,
    world: worldState.sceneIndex ?? worldState.tick ?? 0,
    player: playerState.playerId ?? 'anon',
  }));

  // Compute the changed (resulting) hexagram from the moving line:
  // flip the active line's trigram line to derive the next hexagram structurally.
  const nextHexNumber = deriveChangedHexagram(activeHexagram, activeLine);

  const intensity = lineRule.intensity * (0.9 + rng() * 0.2);

  const environmentalEffects = rule.environment.map((e) => ({
    ...e,
    magnitude: +intensity.toFixed(3),
    origin: `${hex.name} line ${activeLine}`,
  }));

  const morphEffects = [{
    kind: lineRule.morphKind,
    amplification: +(intensity * (1 + rng() * 0.3)).toFixed(3),
    trigramSignature: [hex.upper, hex.lower],
    reason: `hexagram ${activeHexagram} (${hex.name}) moving line ${activeLine} (${lineRule.tag})`,
  }];

  const relationshipEffects = (relationshipState.npcs ?? []).map((npc) => ({
    npc: npc.id ?? npc.name,
    dispositionShift: Math.round((rng() - 0.35) * intensity * 20) / 10,
    note: `${npc.id ?? npc.name} responds to the ${TRIGRAMS[hex.upper].force} over ${TRIGRAMS[hex.lower].force} situation`,
  }));

  const spawnedEvents = [{
    id: `ev-${shortHash({ activeHexagram, activeLine, event, choice }, 8)}`,
    kind: TRIGRAMS[hex.upper].element + '-omens',
    description: `${TRIGRAMS[hex.upper].element} ${TRIGRAMS[hex.upper].behavior} while ${TRIGRAMS[hex.lower].element} ${TRIGRAMS[hex.lower].behavior}`,
  }];

  const probabilityAdjustments = {
    danger: +(0.1 * (TRIGRAMS[hex.lower].element === 'water' ? intensity : 0.3 * intensity)).toFixed(3),
    fortune: +(0.15 * (hex.upper === 'qian' || hex.upper === 'dui' ? intensity : 0.5 * intensity)).toFixed(3),
  };

  // Klein composition: the hexagram unit (6-bit Boolean vector + trigram
  // relations) composed with a world-state unit; features unify bi-
  // directionally. Additive to the legacy result — the API above is intact.
  const worldUnit = makeUnit({
    name: `world:${worldState.sceneIndex ?? worldState.tick ?? 0}`,
    kind: 'object',
    features: [
      'FC_SCENE', 'T_PRESENT',
      ...(worldState.activeHexagram ? ['PROVEN'] : []),
    ],
    structure: { sceneIndex: worldState.sceneIndex ?? null, tick: temporalState.tick ?? null },
  });
  const composed = composeUnits(`resolution-${activeHexagram}-${activeLine}`, hexagramUnit(activeHexagram), worldUnit);

  const stateMovement = {
    from: previousState?.hexagram ?? null,
    hexagram: activeHexagram,
    line: activeLine,
    changingTo: nextHexNumber,
    direction: nextHexNumber > activeHexagram ? 'expanding' : 'contracting',
  };

  return {
    stateMovement,
    environmentalEffects,
    narrativePressure: rule.pressure,
    relationshipEffects,
    availableActions: rule.actions,
    morphEffects,
    spawnedEvents,
    probabilityAdjustments,
    kleinComposition: {
      hexagramUnit: serializeUnit(composed.parentAfter),
      worldUnit: serializeUnit(composed.childAfter),
      composed: serializeUnit(composed.unit),
      conflicts: composed.conflicts,
    },
    nextState: {
      hexagram: nextHexNumber,
      line: activeLine,
      pressure: rule.pressure,
      resolvedAt: temporalState.tick ?? 0,
    },
  };
}

function genericRule(hex) {
  const up = TRIGRAMS[hex.upper], lo = TRIGRAMS[hex.lower];
  return {
    environment: [
      { type: 'weather', change: `${up.element} ${up.behavior} above` },
      { type: 'terrain', change: `${lo.element} ${lo.behavior} below` },
    ],
    pressure: `${up.force}-over-${lo.force}`,
    actions: [`move with the ${lo.element}`, `invoke the ${up.element}`],
  };
}

// Structural line-flip via klein units: all 64 hexagrams are registered
// natively as 6-bit Boolean feature vectors (bits 44-49, bottom-up) with
// trigram membership relations; a moving line is a Boolean NOT on one bit.
const HEXAGRAM_REGISTRY = new UnitRegistry();
let hexagramUnitsReady = false;
function ensureHexagramUnits() {
  if (hexagramUnitsReady) return;
  for (const h of HEXAGRAMS) {
    if (h) HEXAGRAM_REGISTRY.register(makeHexagramUnit(h.number, h.name, h.upper, h.lower));
  }
  hexagramUnitsReady = true;
}

/** The klein unit for a hexagram (registered lazily, memoized). */
export function hexagramUnit(number) {
  ensureHexagramUnits();
  return HEXAGRAM_REGISTRY.get(`hexagram-${number}`);
}

function deriveChangedHexagram(hexNumber, movingLine) {
  ensureHexagramUnits();
  // flip the moving line bit in the hexagram's native Boolean vector, then
  // resolve the changed 6-bit pattern back to a hexagram unit.
  const flipped = flipHexLine(hexagramUnit(hexNumber).features, movingLine);
  const lines = hexLinesOf(flipped);
  const key = (ls) => ls.join('');
  const found = HEXAGRAM_REGISTRY.all().find(
    (u) => key(hexLinesOf(u.features)) === key(lines),
  );
  if (!found) throw new Error(`ichingResolver: changed line pattern ${key(lines)} matches no hexagram`);
  return found.structure.number;
}
