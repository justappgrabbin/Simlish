// Novel Writer Engine — grammar DSL + Klein-style rewrite rules.
//
// Grammar (spec §5.1):
//   SITUATION        ::= Setting Atmosphere Conflict Entities Resolution_Paths
//   Setting          ::= Element_Scene Temporal_Marker
//   Atmosphere       ::= Emotional_Tone Lighting_Condition
//   Conflict         ::= Elemental_Tension | Hexagram_Pressure | Shadow_Echo
//   Entities         ::= Entity+            (>=3 interactive scene entities)
//   Resolution_Paths ::= Path Path Path     (distortion | neutral | harmony)
//
// Klein rewrite rules (spec §5.2) implemented as evaluated rule functions:
//   Elemental_Conflict, Hexagram_Transformation, Temporal_Resonance,
//   Distortion_Recovery, Perfect_Harmony.

import { seededRng, pickSeeded } from '../util/hash.js';
import { ENTITIES, GESTURE_TRIGRAMS } from '../resonance/embeddingSpace.js';
import { GENERATES, CONTROLS } from '../divination/bazi.js';
import { RuleSet } from '../klein/rules.js';
import { defaultRegistry } from '../klein/registry.js';

const SCENE_ENTITIES = ENTITIES.filter((e) => e.category === 'scene');
const EMOTIONS = ENTITIES.filter((e) => e.category === 'emotions');

const TEMPORAL_MARKERS = ['the hour of the rat deepens', 'dawn thins the dark', 'dusk gathers its hem', 'the sun stands at zenith', 'midnight turns its face'];
const LIGHTING = ['low amber lamplight', 'cold mercury moonlight', 'flickering candle-glow', 'overcast silver diffusion', 'storm-lit green shadows'];
const EMOTIONAL_TONES = ['hushed expectancy', 'coiled tension', 'melancholy drift', 'bright unease', 'suspended grief'];

/** Parse/generate a SITUATION from the grammar. Deterministic from seedKey. */
export function generateSituation({ seedKey, hexagram, elementAnalysis, bandHint = null }) {
  if (!hexagram || !elementAnalysis) throw new Error('grammar.generateSituation: hexagram and elementAnalysis required');
  const rng = seededRng(`situation:${seedKey}`);

  const entities = [];
  const pool = SCENE_ENTITIES.slice();
  while (entities.length < 3 + Math.floor(rng() * 2)) {
    const e = pickSeeded(pool, rng);
    if (!entities.includes(e)) entities.push(e);
  }
  const dominantEntity = entities.find((e) => e.element === elementAnalysis.dominant) ?? entities[0];

  const situation = {
    nonterminal: 'SITUATION',
    setting: {
      nonterminal: 'Setting',
      elementScene: dominantEntity,
      temporalMarker: pickSeeded(TEMPORAL_MARKERS, rng),
    },
    atmosphere: {
      nonterminal: 'Atmosphere',
      emotionalTone: pickSeeded(EMOTIONAL_TONES, rng),
      lighting: pickSeeded(LIGHTING, rng),
      emotionalState: pickSeeded(EMOTIONS, rng),
    },
    conflict: {
      nonterminal: 'Conflict',
      kind: pickSeeded(['Elemental_Tension', 'Hexagram_Pressure', 'Shadow_Echo'], rng),
      description: `${elementAnalysis.dominant} dominates while ${elementAnalysis.weakest} starves`,
      dominant: elementAnalysis.dominant,
      weakest: elementAnalysis.weakest,
    },
    entities: { nonterminal: 'Entities', items: entities },
    resolutionPaths: {
      nonterminal: 'Resolution_Paths',
      paths: [
        { kind: 'distortion', hint: 'force the pattern until it breaks' },
        { kind: 'neutral', hint: 'observe what the scene is already doing' },
        { kind: 'harmony', hint: `feed ${elementAnalysis.weakest} with ${Object.keys(GENERATES).find((k) => GENERATES[k] === elementAnalysis.weakest)}` },
      ],
    },
    hexagram: { number: hexagram.number, name: hexagram.name },
    bandHint,
    seedKey,
  };
  return situation;
}

// --- Klein rewrite rules (§5.2) ---------------------------------------------
// Each rule: match(situation, context) -> transformed situation (new object).

export const kleinRules = {
  /** Elemental_Conflict: inject the controlling element of the dominant into the scene. */
  Elemental_Conflict(situation, ctx) {
    const controlled = CONTROLS[ctx.elementAnalysis.dominant];
    const injector = ENTITIES.find((e) => e.element === controlled && e.category === 'scene')
      ?? ENTITIES.find((e) => e.element === controlled);
    const s = structuredClone(situation);
    s.conflict.kind = 'Elemental_Tension';
    s.conflict.description = `${ctx.elementAnalysis.dominant} is checked by rising ${controlled}`;
    if (!s.entities.items.some((e) => e.name === injector.name)) s.entities.items.push(injector);
    s.rewrittenBy = [...(situation.rewrittenBy ?? []), 'Elemental_Conflict'];
    return s;
  },

  /** Hexagram_Transformation: rewrite resolution paths toward the resulting hexagram. */
  Hexagram_Transformation(situation, ctx) {
    const s = structuredClone(situation);
    s.hexagram = { number: ctx.resultingHexagram.number, name: ctx.resultingHexagram.name };
    s.resolutionPaths.paths = s.resolutionPaths.paths.map((p) => ({
      ...p, hint: `${p.hint} (under ${ctx.resultingHexagram.name})`,
    }));
    s.rewrittenBy = [...(situation.rewrittenBy ?? []), 'Hexagram_Transformation'];
    return s;
  },

  /** Temporal_Resonance: shift temporal marker + lighting to match the hour. */
  Temporal_Resonance(situation, ctx) {
    const s = structuredClone(situation);
    const hour = (ctx.now ?? new Date()).getHours();
    const marker = hour < 5 ? TEMPORAL_MARKERS[0] : hour < 9 ? TEMPORAL_MARKERS[1]
      : hour < 17 ? TEMPORAL_MARKERS[3] : hour < 21 ? TEMPORAL_MARKERS[2] : TEMPORAL_MARKERS[4];
    s.setting.temporalMarker = marker;
    s.atmosphere.lighting = hour < 6 || hour > 19 ? LIGHTING[1] : LIGHTING[3];
    s.rewrittenBy = [...(situation.rewrittenBy ?? []), 'Temporal_Resonance'];
    return s;
  },

  /** Distortion_Recovery: after distortion, add a shadow entity + recovery path. */
  Distortion_Recovery(situation, ctx) {
    const s = structuredClone(situation);
    const shadow = ENTITIES.find((e) => e.name === 'void-sigil');
    if (!s.entities.items.some((e) => e.name === shadow.name)) s.entities.items.push(shadow);
    s.conflict.kind = 'Shadow_Echo';
    s.conflict.description = 'the distortion leaves an echo that wants to be answered, not fought';
    s.resolutionPaths.paths[0] = { kind: 'distortion', hint: 'answer the echo without force' };
    s.rewrittenBy = [...(situation.rewrittenBy ?? []), 'Distortion_Recovery'];
    return s;
  },

  /** Perfect_Harmony: after harmony, gild the scene — add crane/phoenix + harmony path upgrade. */
  Perfect_Harmony(situation, ctx) {
    const s = structuredClone(situation);
    const grace = ENTITIES.find((e) => e.name === (ctx.elementAnalysis.dominant === 'fire' ? 'phoenix' : 'crane'));
    if (!s.entities.items.some((e) => e.name === grace.name)) s.entities.items.push(grace);
    s.atmosphere.emotionalTone = 'bright unease';
    s.resolutionPaths.paths[2] = { kind: 'harmony', hint: 'the pattern is open — walk through it' };
    s.rewrittenBy = [...(situation.rewrittenBy ?? []), 'Perfect_Harmony'];
    return s;
  },
};

/**
 * Apply rewrite rules for the current journey context.
 * @param situation - grammar SITUATION object
 * @param ctx - {elementAnalysis, resultingHexagram, now, lastBand}
 * @returns rewritten situation (rules applied in canonical order when matched)
 */
export function applyRewriteRules(situation, ctx) {
  let s = situation;
  const order = ['Elemental_Conflict', 'Hexagram_Transformation', 'Temporal_Resonance'];
  for (const name of order) s = kleinRules[name](s, ctx);
  if (ctx.lastBand === 'distortion' || ctx.lastBand === 'severe-distortion') {
    s = kleinRules.Distortion_Recovery(s, ctx);
  }
  if (ctx.lastBand === 'perfect-harmony') {
    s = kleinRules.Perfect_Harmony(s, ctx);
  }
  return s;
}

// --- SITUATION grammar as a klein RuleSet (rewire) ---------------------------
// The SITUATION grammar above is re-expressed as klein reversible production
// rules: ONE data rule set both generates scene surface text and recognizes
// player-facing text back into semantic structure. World-knowledge rules
// ("guardians block the unproven") live in the same notation (klein/rules.js).

const FIVE_ELEMENT_NAMES = ['wood', 'fire', 'earth', 'metal', 'water'];
const PATH_KINDS = ['distortion', 'neutral', 'harmony'];

export const SITUATION_GRAMMAR = new RuleSet('situation-grammar', [
  {
    id: 'situation-surface', kind: 'rewrite',
    surface: 'In the presence of {entities} | {temporal} | {tone} | {lighting} | conflict: {dominant} dominates while {weakest} starves | paths: {paths}',
    slots: {
      entities: { kind: 'list' },                 // resolved to registered scene units on recognize
      temporal: { kind: 'vocab', vocab: TEMPORAL_MARKERS },
      tone: { kind: 'vocab', vocab: EMOTIONAL_TONES },
      lighting: { kind: 'vocab', vocab: LIGHTING },
      dominant: { kind: 'vocab', vocab: FIVE_ELEMENT_NAMES },
      weakest: { kind: 'vocab', vocab: FIVE_ELEMENT_NAMES },
      paths: { kind: 'list', vocab: PATH_KINDS },
    },
  },
]);

/** GENERATE: situation semantic structure -> surface text (via klein rules). */
export function renderSituation(situation) {
  if (!situation?.nonterminal || situation.nonterminal !== 'SITUATION') {
    throw new Error('grammar.renderSituation: a SITUATION structure is required');
  }
  return SITUATION_GRAMMAR.generate('situation-surface', {
    entities: situation.entities.items.map((e) => e.name),
    temporal: situation.setting.temporalMarker,
    tone: situation.atmosphere.emotionalTone,
    lighting: situation.atmosphere.lighting,
    dominant: situation.conflict.dominant ?? situation.conflict.description.split(' ')[0],
    weakest: situation.conflict.weakest ?? situation.conflict.description.split(' ').slice(-2, -1)[0],
    paths: situation.resolutionPaths.paths.map((p) => p.kind),
  });
}

/**
 * RECOGNIZE: surface text (free-form-ish player action / scene description)
 * -> semantic structure, using the SAME rule set that generates scenes.
 * @returns {object|null} {entities:[units], temporalMarker, emotionalTone,
 *   lighting, conflict:{dominant,weakest}, paths:[kinds]} or null if no match.
 */
export function recognizeSituation(text) {
  const hit = SITUATION_GRAMMAR.recognize(text, { registry: defaultRegistry() });
  if (!hit) return null;
  const b = hit.bindings;
  return {
    nonterminal: 'SITUATION',
    ruleId: hit.ruleId,
    entities: b.entities, // klein units with Boolean feature vectors
    temporalMarker: b.temporal,
    emotionalTone: b.tone,
    lighting: b.lighting,
    conflict: { nonterminal: 'Conflict', dominant: b.dominant, weakest: b.weakest },
    resolutionPaths: { nonterminal: 'Resolution_Paths', paths: b.paths.map((kind) => ({ kind })) },
  };
}
